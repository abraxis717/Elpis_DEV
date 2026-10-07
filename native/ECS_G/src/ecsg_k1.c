#include "elpis/ecsg_k1.h"
#include "ecsg_k1_internal.h"

#include "elpis/ecsg_math.h"
#include "elpis/sha256.h"

#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * One K1 state = an authoritative image (header + W + H packed + a, native
 * binary64) plus one 64-byte-aligned arena of non-authoritative workspace:
 *
 *   cur, nxt        dim*width   the K-loop W buffers (ping-pong)
 *   grad, vjp       dim*width   G1 gradient, K1 correction J^T u
 *   pz              max_rows*width  phi'(z) of the current step
 *   error           max_rows    residuals
 *   row             ROW_BLOCK*width
 *   s3, diff, u     F each      S3(W), S3(W) - a, 1/2 H (S3(W) - a)
 *   u2              dim*dim     the symmetric second-order block of u
 *   phi             F           features of one consolidated row
 *   stage_h, stage_a            staged consolidation (direct path)
 *   cand            image bytes the open transaction's complete candidate
 *   x, y            admitted experience
 *   pairs, triples, m2, m3      packed S3 index tables (fixed per dim)
 *
 * Only the image is cognitive state. A bound state (the FMS residency adapter)
 * points `image` at FMS-resident bytes instead of owning them; the workspace is
 * the same.
 */

enum {
    ARENA_ALIGN = 64u,
    ROW_BLOCK = 8u,
    MAX_DIM = ELPIS_ECSG_K1_MAX_DIM,
    G01_HEADER = 40u
};

static const uint8_t K1_MAGIC[8] = {'E', 'L', 'P', 'I', 'S', 'G', 'K', '1'};
static const uint8_t G01_MAGIC[8] = {'E', 'L', 'P', 'I', 'S', 'G', '0', '1'};

struct elpis_ecsg_k1 {
    size_t dim;
    size_t width;
    size_t features;
    size_t pairs_n;
    size_t triples_n;
    size_t max_rows;
    size_t w_count;
    size_t h_count;
    size_t image_bytes;
    uint64_t generation;
    uint8_t *image;
    int owns_image;
    void *arena;
    size_t arena_bytes;
    double *cur;
    double *nxt;
    double *grad;
    double *vjp;
    double *pz;
    double *error;
    double *row;
    double *s3;
    double *diff;
    double *u;
    double *u2;
    double *phi;
    double *stage_h;
    double *stage_a;
    uint8_t *cand;
    double *x;
    double *y;
    uint16_t *pairs;
    uint16_t *triples;
    double *m2;
    double *m3;
    int txn_open;
    uint64_t txn_token;
    uint64_t txn_tokens_issued;
    uint64_t txn_source_generation;
    elpis_ecsg_k1_counters stats;
    atomic_uint_fast64_t busy_refusals;
    atomic_flag busy;
    /* Published by the single writer after each committed transition; read by the unguarded getters. */
    atomic_uint_fast64_t pub_epoch;
    atomic_uint_fast64_t pub_generation;
    atomic_uint_fast32_t pub_provenance;
    atomic_size_t pub_max_rows;
};

/* --- checked arithmetic and layout ------------------------------------------------------------------------- */

static int mul_ok(size_t a, size_t b, size_t *out)
{
    if (a != 0u && b > SIZE_MAX / a) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int add_ok(size_t a, size_t b, size_t *out)
{
    if (b > SIZE_MAX - a) {
        return 0;
    }
    *out = a + b;
    return 1;
}

/* A uint64 from a serialized header as size_t, refusing values the host cannot represent (no cast first). */
static int u64_size(uint64_t v, size_t *out)
{
    const size_t narrowed = (size_t)v;
    if ((uint64_t)narrowed != v) {
        return 0;
    }
    *out = narrowed;
    return 1;
}

static size_t packed_count(size_t features)
{
    return features * (features + 1u) / 2u;   /* features <= |S3(64)| = 47904: no overflow */
}

/* OK, INVALID (not a shape) or CAPACITY (an overflow, or an image beyond ELPIS_ECSG_K1_MAX_IMAGE_BYTES). Every
 * product and sum is checked; nothing is allocated here. */
static elpis_ecsg_k1_status shape_status(size_t dim, size_t width, size_t *image_bytes)
{
    size_t count;
    size_t f;
    size_t doubles;
    size_t bytes;
    if (dim < 1u || dim > MAX_DIM || width < 1u) {
        return ELPIS_ECSG_K1_INVALID;
    }
    f = elpis_ecsg_s3_size(dim);
    if (f == 0u || !mul_ok(dim, width, &count) || !add_ok(count, packed_count(f), &doubles) ||
        !add_ok(doubles, f, &doubles) || !mul_ok(doubles, sizeof(double), &bytes) ||
        !add_ok(bytes, ELPIS_ECSG_K1_HEADER_BYTES, &bytes) || bytes > (size_t)ELPIS_ECSG_K1_MAX_IMAGE_BYTES) {
        return ELPIS_ECSG_K1_CAPACITY;
    }
    if (image_bytes != NULL) {
        *image_bytes = bytes;
    }
    return ELPIS_ECSG_K1_OK;
}

size_t elpis_ecsg_k1_features(size_t dim)
{
    if (dim < 1u || dim > MAX_DIM) {
        return 0u;
    }
    return elpis_ecsg_s3_size(dim);
}

size_t elpis_ecsg_k1_image_bytes(size_t dim, size_t width)
{
    size_t image = 0u;
    return shape_status(dim, width, &image) == ELPIS_ECSG_K1_OK ? image : 0u;
}

size_t elpis_ecsg_k1_payload_bytes(size_t dim, size_t width)
{
    size_t image = elpis_ecsg_k1_image_bytes(dim, width);
    return image == 0u ? 0u : image - ELPIS_ECSG_K1_HEADER_BYTES;
}

size_t elpis_ecsg_k1_envelope_bytes(size_t dim, size_t width)
{
    size_t image = elpis_ecsg_k1_image_bytes(dim, width);
    return image == 0u ? 0u : image + ELPIS_ECSG_K1_DIGEST_BYTES;   /* image <= 64 MiB */
}

typedef struct {
    size_t cur, nxt, grad, vjp, pz, error, row, s3, diff, u, u2, phi, stage_h, stage_a, cand, x, y;
    size_t pairs, triples, m2, m3, total;
} layout;

static int segment(size_t bytes, size_t *offset, size_t *cursor)
{
    size_t rounded;
    if (!add_ok(bytes, ARENA_ALIGN - 1u, &rounded)) {
        return 0;
    }
    rounded -= rounded % ARENA_ALIGN;
    *offset = *cursor;
    return add_ok(*cursor, rounded, cursor);
}

/* Bytes of `count` elements of `size`, rounded to the arena alignment, appended at *cursor (all checked). */
static int seg(size_t count, size_t size, size_t *offset, size_t *cursor)
{
    size_t bytes;
    return mul_ok(count, size, &bytes) && segment(bytes, offset, cursor);
}

/* The arena layout: OK, INVALID (shape or max_rows = 0) or CAPACITY (overflow, or beyond the byte budgets). */
static elpis_ecsg_k1_status plan(size_t dim, size_t width, size_t max_rows, layout *l)
{
    size_t image = 0u;
    size_t f;
    size_t p;
    size_t t;
    size_t w;
    size_t xrows;
    size_t pzrows;
    size_t cursor = 0u;
    const elpis_ecsg_k1_status rc = shape_status(dim, width, &image);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    if (max_rows == 0u) {
        return ELPIS_ECSG_K1_INVALID;
    }
    f = elpis_ecsg_k1_features(dim);
    p = dim * (dim + 1u) / 2u;
    t = f - dim - p;
    w = dim * width;   /* checked by shape_status */
    if (!mul_ok(max_rows, dim, &xrows) || !mul_ok(max_rows, width, &pzrows)) {
        return ELPIS_ECSG_K1_CAPACITY;
    }
    if (!(seg(w, 8u, &l->cur, &cursor) && seg(w, 8u, &l->nxt, &cursor) && seg(w, 8u, &l->grad, &cursor) &&
          seg(w, 8u, &l->vjp, &cursor) && seg(pzrows, 8u, &l->pz, &cursor) && seg(max_rows, 8u, &l->error, &cursor) &&
          seg(width, ROW_BLOCK * 8u, &l->row, &cursor) && seg(f, 8u, &l->s3, &cursor) &&
          seg(f, 8u, &l->diff, &cursor) && seg(f, 8u, &l->u, &cursor) && seg(dim * dim, 8u, &l->u2, &cursor) &&
          seg(f, 8u, &l->phi, &cursor) && seg(packed_count(f), 8u, &l->stage_h, &cursor) &&
          seg(f, 8u, &l->stage_a, &cursor) && seg(image, 1u, &l->cand, &cursor) &&
          seg(xrows, 8u, &l->x, &cursor) && seg(max_rows, 8u, &l->y, &cursor) &&
          seg(p, 2u * sizeof(uint16_t), &l->pairs, &cursor) && seg(t, 3u * sizeof(uint16_t), &l->triples, &cursor) &&
          seg(p, 8u, &l->m2, &cursor) && seg(t, 8u, &l->m3, &cursor)) ||
        cursor > (size_t)ELPIS_ECSG_K1_MAX_WORKSPACE_BYTES) {
        return ELPIS_ECSG_K1_CAPACITY;
    }
    l->total = cursor;
    return ELPIS_ECSG_K1_OK;
}

size_t elpis_ecsg_k1_workspace_bytes(size_t dim, size_t width, size_t max_rows)
{
    layout l;
    return plan(dim, width, max_rows, &l) == ELPIS_ECSG_K1_OK ? l.total : 0u;
}

static void carve(elpis_ecsg_k1 *s, uint8_t *arena, const layout *l)
{
    s->cur = (double *)(void *)(arena + l->cur);
    s->nxt = (double *)(void *)(arena + l->nxt);
    s->grad = (double *)(void *)(arena + l->grad);
    s->vjp = (double *)(void *)(arena + l->vjp);
    s->pz = (double *)(void *)(arena + l->pz);
    s->error = (double *)(void *)(arena + l->error);
    s->row = (double *)(void *)(arena + l->row);
    s->s3 = (double *)(void *)(arena + l->s3);
    s->diff = (double *)(void *)(arena + l->diff);
    s->u = (double *)(void *)(arena + l->u);
    s->u2 = (double *)(void *)(arena + l->u2);
    s->phi = (double *)(void *)(arena + l->phi);
    s->stage_h = (double *)(void *)(arena + l->stage_h);
    s->stage_a = (double *)(void *)(arena + l->stage_a);
    s->cand = arena + l->cand;
    s->x = (double *)(void *)(arena + l->x);
    s->y = (double *)(void *)(arena + l->y);
    s->pairs = (uint16_t *)(void *)(arena + l->pairs);
    s->triples = (uint16_t *)(void *)(arena + l->triples);
    s->m2 = (double *)(void *)(arena + l->m2);
    s->m3 = (double *)(void *)(arena + l->m3);
}

static void index_tables(elpis_ecsg_k1 *s)
{
    size_t a, b, c, k = 0u, q = 0u;
    for (a = 0u; a < s->dim; ++a) {
        for (b = a; b < s->dim; ++b) {
            s->pairs[2u * k] = (uint16_t)a;
            s->pairs[2u * k + 1u] = (uint16_t)b;
            s->m2[k] = a == b ? 1.0 : 2.0;
            ++k;
        }
    }
    for (a = 0u; a < s->dim; ++a) {
        for (b = a; b < s->dim; ++b) {
            for (c = b; c < s->dim; ++c) {
                const int distinct = 1 + (b != a) + (c != b);
                s->triples[3u * q] = (uint16_t)a;
                s->triples[3u * q + 1u] = (uint16_t)b;
                s->triples[3u * q + 2u] = (uint16_t)c;
                s->m3[q] = distinct == 1 ? 1.0 : distinct == 2 ? 3.0 : 6.0;
                ++q;
            }
        }
    }
}

/* --- the image ------------------------------------------------------------------------------------------- */

static void put_u32(uint8_t *p, uint32_t v)
{
    unsigned i;
    for (i = 0u; i < 4u; ++i) {
        p[i] = (uint8_t)(v >> (8u * i));
    }
}

static void put_u64(uint8_t *p, uint64_t v)
{
    unsigned i;
    for (i = 0u; i < 8u; ++i) {
        p[i] = (uint8_t)(v >> (8u * i));
    }
}

static uint32_t get_u32(const uint8_t *p)
{
    uint32_t v = 0u;
    unsigned i;
    for (i = 0u; i < 4u; ++i) {
        v |= (uint32_t)p[i] << (8u * i);
    }
    return v;
}

static uint64_t get_u64(const uint8_t *p)
{
    uint64_t v = 0u;
    unsigned i;
    for (i = 0u; i < 8u; ++i) {
        v |= (uint64_t)p[i] << (8u * i);
    }
    return v;
}

static double *img_w(const elpis_ecsg_k1 *s, uint8_t *image)
{
    (void)s;
    return (double *)(void *)(image + ELPIS_ECSG_K1_HEADER_BYTES);
}

static double *img_h(const elpis_ecsg_k1 *s, uint8_t *image)
{
    return img_w(s, image) + s->w_count;
}

static double *img_a(const elpis_ecsg_k1 *s, uint8_t *image)
{
    return img_h(s, image) + s->h_count;
}

static void write_header(const elpis_ecsg_k1 *s, uint8_t *image, uint64_t epoch, uint32_t provenance)
{
    memcpy(image, K1_MAGIC, 8u);
    put_u32(image + 8u, ELPIS_ECSG_K1_FORMAT_VERSION);
    put_u32(image + 12u, ELPIS_ECSG_K1_MECHANISM);
    put_u64(image + 16u, (uint64_t)s->dim);
    put_u64(image + 24u, (uint64_t)s->width);
    put_u64(image + 32u, (uint64_t)s->features);
    put_u64(image + 40u, epoch);
    put_u64(image + 48u, (uint64_t)provenance);
    put_u64(image + 56u, 0u);
}

static uint64_t image_epoch(const uint8_t *image)
{
    return get_u64(image + 40u);
}

static uint32_t image_provenance(const uint8_t *image)
{
    return (uint32_t)get_u64(image + 48u);
}

static int all_finite(const double *v, size_t count)
{
    size_t i;
    double check = 0.0;
    for (i = 0u; i < count; ++i) {
        check += v[i] - v[i];
    }
    return check == 0.0;
}

/* --- entry guard ------------------------------------------------------------------------------------------ */

static int enter(elpis_ecsg_k1 *s)
{
    if (atomic_flag_test_and_set_explicit(&s->busy, memory_order_acquire)) {
        atomic_fetch_add_explicit(&s->busy_refusals, 1u, memory_order_relaxed);
        return 0;
    }
    return 1;
}

static void leave(elpis_ecsg_k1 *s)
{
    atomic_flag_clear_explicit(&s->busy, memory_order_release);
}

/* The writer publishes epoch, generation, provenance and max_rows after each committed transition. */
static void publish(elpis_ecsg_k1 *s)
{
    atomic_store_explicit(&s->pub_generation, s->generation, memory_order_release);
    atomic_store_explicit(&s->pub_max_rows, s->max_rows, memory_order_release);
    if (s->image != NULL) {
        atomic_store_explicit(&s->pub_epoch, image_epoch(s->image), memory_order_release);
        atomic_store_explicit(&s->pub_provenance, image_provenance(s->image), memory_order_release);
    }
}

/* --- kernels: G1 and forward in the reference's per-element order (as the Runtime R1 executor) ------------- */

#if defined(__SANITIZE_THREAD__)
#define ELPIS_ECSG_K1_NO_CLONES 1
#endif
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define ELPIS_ECSG_K1_NO_CLONES 1
#endif
#endif
#if defined(__x86_64__) && defined(__has_attribute) && !defined(ELPIS_ECSG_K1_NO_CLONES) && \
    !defined(ELPIS_ECSG_NO_CLONES)
#if __has_attribute(target_clones)
#define K1_HOT __attribute__((target_clones("avx2", "default")))
#endif
#endif
#ifndef K1_HOT
#define K1_HOT
#endif

static K1_HOT void
row_sums(size_t width, size_t block, const double *restrict v, double *restrict sums)
{
    size_t i;
    size_t k;
    if (block == ROW_BLOCK) {
        double acc[ROW_BLOCK];
        for (k = 0u; k < ROW_BLOCK; ++k) {
            acc[k] = 0.0;
        }
        for (i = 0u; i < width; ++i) {
            for (k = 0u; k < ROW_BLOCK; ++k) {
                acc[k] += v[k * width + i];
            }
        }
        for (k = 0u; k < ROW_BLOCK; ++k) {
            sums[k] = acc[k];
        }
        return;
    }
    for (k = 0u; k < block; ++k) {
        double acc = 0.0;
        for (i = 0u; i < width; ++i) {
            acc += v[k * width + i];
        }
        sums[k] = acc;
    }
}

/* One G1 step src -> dst, bitwise equal to the reference (same derivation as ecsg_executor.c). src != dst. */
static K1_HOT int
g1_step(size_t dim, size_t width, size_t rows, const double *restrict x, const double *restrict y,
        const double *restrict src, double *restrict dst, double *restrict pz, double *restrict error,
        double *restrict row, double *restrict grad, double learning_rate)
{
    const double scale = 2.0 / (double)rows;
    const size_t count = dim * width;
    double check = 0.0;
    size_t r;
    size_t a;
    size_t i;

    for (r = 0u; r < rows; r += ROW_BLOCK) {
        const size_t block = rows - r < ROW_BLOCK ? rows - r : ROW_BLOCK;
        double prediction[ROW_BLOCK];
        size_t k;
        for (k = 0u; k < block; ++k) {
            const double *xr = x + (r + k) * dim;
            double *restrict zk = row + k * width;
            double *restrict pr = pz + (r + k) * width;
            for (i = 0u; i < width; ++i) {
                zk[i] = 0.0;
            }
            for (a = 0u; a < dim; ++a) {
                const double xa = xr[a];
                const double *wa = src + a * width;
                for (i = 0u; i < width; ++i) {
                    zk[i] += xa * wa[i];
                }
            }
            for (i = 0u; i < width; ++i) {
                const double z = zk[i];
                const double z2 = z * z;
                pr[i] = 0.5 + z + 1.5 * z * z;
                zk[i] = 0.5 * z + 0.5 * z2 + 0.5 * z2 * z;
            }
        }
        row_sums(width, block, row, prediction);
        for (k = 0u; k < block; ++k) {
            if (!isfinite(prediction[k])) {
                return 0;
            }
            error[r + k] = prediction[k] - y[r + k];
            if (!isfinite(error[r + k])) {
                return 0;
            }
        }
    }
    for (i = 0u; i < count; ++i) {
        grad[i] = 0.0;
    }
    for (r = 0u; r < rows; ++r) {
        const double *restrict pr = pz + r * width;
        const double er = error[r];
        for (a = 0u; a < dim; ++a) {
            const double c = x[r * dim + a] * er;
            double *restrict ga = grad + a * width;
            for (i = 0u; i < width; ++i) {
                ga[i] += c * pr[i];
            }
        }
    }
    for (i = 0u; i < count; ++i) {
        const double next = src[i] - learning_rate * (scale * grad[i]);
        check += next - next;
        dst[i] = next;
    }
    return check == 0.0;
}

static K1_HOT int
forward_rows(size_t dim, size_t width, size_t rows, const double *restrict x, const double *w,
             double *restrict out, double *restrict row)
{
    size_t r;
    size_t a;
    size_t i;
    for (r = 0u; r < rows; r += ROW_BLOCK) {
        const size_t block = rows - r < ROW_BLOCK ? rows - r : ROW_BLOCK;
        double total[ROW_BLOCK];
        size_t k;
        for (k = 0u; k < block; ++k) {
            const double *xr = x + (r + k) * dim;
            double *restrict zk = row + k * width;
            for (i = 0u; i < width; ++i) {
                zk[i] = 0.0;
            }
            for (a = 0u; a < dim; ++a) {
                const double xa = xr[a];
                const double *wa = w + a * width;
                for (i = 0u; i < width; ++i) {
                    zk[i] += xa * wa[i];
                }
            }
            for (i = 0u; i < width; ++i) {
                const double z = zk[i];
                const double z2 = z * z;
                zk[i] = 0.5 * z + 0.5 * z2 + 0.5 * z2 * z;
            }
        }
        row_sums(width, block, row, total);
        for (k = 0u; k < block; ++k) {
            if (!isfinite(total[k])) {
                return 0;
            }
            out[r + k] = total[k];
        }
    }
    return 1;
}

/* --- kernels: the K1 correction and consolidation --------------------------------------------------------- */

static void s3_of(const elpis_ecsg_k1 *s, const double *w, double *out)
{
    (void)elpis_ecsg_project_s3_f64(w, s->dim, s->width, out, out + s->dim, out + s->dim + s->pairs_n);
}

static int any_nonzero(const double *v, size_t count)
{
    size_t i;
    for (i = 0u; i < count; ++i) {
        if (v[i] != 0.0) {
            return 1;
        }
    }
    return 0;
}

/* vjp = J(W)^T u with u = 1/2 H (S3(W) - a): u_1 + 2 U_2 w_i + 3 U_3(w_i, w_i, .) per entity (as the R3 laboratory). */
static K1_HOT void
k1_correction(elpis_ecsg_k1 *s, const double *restrict w, const double *restrict h, const double *restrict anchor,
              double *restrict vjp)
{
    const size_t f = s->features;
    const size_t d = s->dim;
    const size_t n = s->width;
    const double *u1 = s->u;
    const double *u2p = s->u + d;
    const double *u3 = s->u + d + s->pairs_n;
    size_t i;
    size_t j;
    size_t k = 0u;
    size_t a;
    size_t b;
    size_t q;

    s3_of(s, w, s->s3);
    for (i = 0u; i < f; ++i) {
        s->diff[i] = s->s3[i] - anchor[i];
        s->u[i] = 0.0;
    }
    for (i = 0u; i < f; ++i) {
        const double di = s->diff[i];
        double acc = s->u[i];
        for (j = i; j < f; ++j) {
            const double hij = h[k++];
            acc += hij * s->diff[j];
            if (j != i) {
                s->u[j] += hij * di;
            }
        }
        s->u[i] = acc;
    }
    for (i = 0u; i < f; ++i) {
        s->u[i] *= 0.5;
    }
    for (i = 0u; i < d * d; ++i) {
        s->u2[i] = 0.0;
    }
    for (q = 0u; q < s->pairs_n; ++q) {
        a = s->pairs[2u * q];
        b = s->pairs[2u * q + 1u];
        s->u2[a * d + b] += u2p[q];
        s->u2[b * d + a] += u2p[q];
    }
    for (a = 0u; a < d; ++a) {
        double *ga = vjp + a * n;
        for (i = 0u; i < n; ++i) {
            ga[i] = u1[a];
        }
        for (b = 0u; b < d; ++b) {
            const double c = s->u2[a * d + b];
            const double *wb = w + b * n;
            for (i = 0u; i < n; ++i) {
                ga[i] += c * wb[i];
            }
        }
    }
    for (q = 0u; q < s->triples_n; ++q) {
        const size_t ta = s->triples[3u * q];
        const size_t tb = s->triples[3u * q + 1u];
        const size_t tc = s->triples[3u * q + 2u];
        const double c = u3[q];
        const double *wa = w + ta * n;
        const double *wb = w + tb * n;
        const double *wc = w + tc * n;
        double *ga = vjp + ta * n;
        double *gb = vjp + tb * n;
        double *gc = vjp + tc * n;
        for (i = 0u; i < n; ++i) {
            ga[i] += c * wb[i] * wc[i];
            gb[i] += c * wa[i] * wc[i];
            gc[i] += c * wa[i] * wb[i];
        }
    }
}

static void features_of(const elpis_ecsg_k1 *s, const double *x, double *phi)
{
    size_t q;
    size_t a;
    for (a = 0u; a < s->dim; ++a) {
        phi[a] = x[a];
    }
    for (q = 0u; q < s->pairs_n; ++q) {
        phi[s->dim + q] = s->m2[q] * x[s->pairs[2u * q]] * x[s->pairs[2u * q + 1u]];
    }
    for (q = 0u; q < s->triples_n; ++q) {
        phi[s->dim + s->pairs_n + q] =
            s->m3[q] * x[s->triples[3u * q]] * x[s->triples[3u * q + 1u]] * x[s->triples[3u * q + 2u]];
    }
}

/* stage_h = h + (1/n) sum_r phi phi^T (packed upper), stage_a = S3(w). Returns 0 on a non-finite result. */
static int consolidate_into(elpis_ecsg_k1 *s, const double *w, const double *h, const double *x, size_t rows,
                            double *stage_h, double *stage_a)
{
    const size_t f = s->features;
    const double inv = 1.0 / (double)rows;
    size_t r;
    size_t i;
    size_t j;
    size_t k;
    for (k = 0u; k < s->h_count; ++k) {
        stage_h[k] = 0.0;
    }
    for (r = 0u; r < rows; ++r) {
        features_of(s, x + r * s->dim, s->phi);
        k = 0u;
        for (i = 0u; i < f; ++i) {
            const double pi = s->phi[i];
            for (j = i; j < f; ++j) {
                stage_h[k++] += pi * s->phi[j];
            }
        }
    }
    for (k = 0u; k < s->h_count; ++k) {
        stage_h[k] = h[k] + stage_h[k] * inv;
    }
    s3_of(s, w, stage_a);
    return all_finite(stage_h, s->h_count) && all_finite(stage_a, f);
}

/* K steps from the W of `image` with (H, a) of `image`; the result is left in s->cur. Returns the failed step
 * (1-based) or 0. Steps with H = 0 apply no correction: the reduction to G1 is exact. */
static uint64_t run_steps(elpis_ecsg_k1 *s, uint8_t *image, size_t rows, double rate, uint64_t steps)
{
    const double *h = img_h(s, image);
    const double *anchor = img_a(s, image);
    const int corrected = any_nonzero(h, s->h_count);
    uint64_t step;
    size_t i;
    memcpy(s->cur, img_w(s, image), s->w_count * sizeof(double));
    for (step = 1u; step <= steps; ++step) {
        double *tmp;
        s->stats.steps_executed += 1u;
        if (!g1_step(s->dim, s->width, rows, s->x, s->y, s->cur, s->nxt, s->pz, s->error, s->row, s->grad, rate)) {
            return step;
        }
        if (corrected) {
            double check = 0.0;
            k1_correction(s, s->cur, h, anchor, s->vjp);
            for (i = 0u; i < s->w_count; ++i) {
                const double next = s->nxt[i] - rate * s->vjp[i];
                check += next - next;
                s->nxt[i] = next;
            }
            if (check != 0.0) {
                return step;
            }
            s->stats.corrected_steps += 1u;
        }
        tmp = s->cur;
        s->cur = s->nxt;
        s->nxt = tmp;
    }
    return 0u;
}

/* --- lifecycle ----------------------------------------------------------------------------------------------- */

static elpis_ecsg_k1_status make(size_t dim, size_t width, size_t max_rows, int owns_image, elpis_ecsg_k1 **out)
{
    layout l;
    elpis_ecsg_k1 *s;
    void *arena;
    elpis_ecsg_k1_status rc;
    if (out == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    *out = NULL;
    rc = plan(dim, width, max_rows, &l);   /* every size checked and bounded before any allocation */
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    s = (elpis_ecsg_k1 *)calloc(1u, sizeof(*s));
    if (s == NULL) {
        return ELPIS_ECSG_K1_NOMEM;
    }
    arena = aligned_alloc(ARENA_ALIGN, l.total);
    if (arena == NULL) {
        free(s);
        return ELPIS_ECSG_K1_NOMEM;
    }
    memset(arena, 0, l.total);
    s->dim = dim;
    s->width = width;
    s->features = elpis_ecsg_k1_features(dim);
    s->pairs_n = dim * (dim + 1u) / 2u;
    s->triples_n = s->features - dim - s->pairs_n;
    s->max_rows = max_rows;
    s->w_count = dim * width;
    s->h_count = packed_count(s->features);
    s->image_bytes = elpis_ecsg_k1_image_bytes(dim, width);
    s->arena = arena;
    s->arena_bytes = l.total;
    s->stats.heap_allocations = 2u;
    carve(s, (uint8_t *)arena, &l);
    index_tables(s);
    if (owns_image) {
        size_t bytes = s->image_bytes + (ARENA_ALIGN - s->image_bytes % ARENA_ALIGN) % ARENA_ALIGN;
        s->image = (uint8_t *)aligned_alloc(ARENA_ALIGN, bytes);
        if (s->image == NULL) {
            free(arena);
            free(s);
            return ELPIS_ECSG_K1_NOMEM;
        }
        memset(s->image, 0, bytes);
        s->owns_image = 1;
        s->stats.heap_allocations += 1u;
    }
    atomic_init(&s->busy_refusals, 0u);
    atomic_init(&s->pub_epoch, 0u);
    atomic_init(&s->pub_generation, 0u);
    atomic_init(&s->pub_provenance, 0u);
    atomic_init(&s->pub_max_rows, max_rows);
    atomic_flag_clear(&s->busy);
    *out = s;
    return ELPIS_ECSG_K1_OK;
}

uint32_t elpis_ecsg_k1_abi_version(void)
{
    return ELPIS_ECSG_K1_ABI_V1;
}

elpis_ecsg_k1_status
elpis_ecsg_k1_create(size_t dim, size_t width, size_t max_rows, const double *initial_w, elpis_ecsg_k1 **out)
{
    elpis_ecsg_k1 *s = NULL;
    elpis_ecsg_k1_status rc;
    if (out == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    *out = NULL;
    if (initial_w == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    rc = shape_status(dim, width, NULL);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    if (!all_finite(initial_w, dim * width)) {
        return ELPIS_ECSG_K1_NONFINITE;
    }
    rc = make(dim, width, max_rows, 1, &s);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    write_header(s, s->image, 0u, ELPIS_ECSG_K1_COMPLETE);
    memcpy(img_w(s, s->image), initial_w, s->w_count * sizeof(double));
    publish(s);
    *out = s;
    return ELPIS_ECSG_K1_OK;
}

static int read_doubles(const uint8_t *src, size_t count, double *dst)
{
    size_t i;
    double check = 0.0;
    for (i = 0u; i < count; ++i) {
        const uint64_t bits = get_u64(src + 8u * i);
        double v;
        memcpy(&v, &bits, sizeof(v));
        check += v - v;
        dst[i] = v;
    }
    return check == 0.0;
}

/* Validates an envelope's header, size and checksum; returns dim and width. */
static elpis_ecsg_k1_status check_envelope(const uint8_t *e, size_t size, size_t *dim, size_t *width)
{
    uint8_t digest[32];
    size_t d = 0u;
    size_t w = 0u;
    size_t image = 0u;
    elpis_ecsg_k1_status rc;
    if (e == NULL || size < ELPIS_ECSG_K1_HEADER_BYTES + ELPIS_ECSG_K1_DIGEST_BYTES) {
        return ELPIS_ECSG_K1_CORRUPT;
    }
    if (memcmp(e, K1_MAGIC, 8u) != 0 || get_u32(e + 8u) != ELPIS_ECSG_K1_FORMAT_VERSION ||
        get_u32(e + 12u) != ELPIS_ECSG_K1_MECHANISM) {
        return ELPIS_ECSG_K1_CORRUPT;
    }
    /* Serialized dimensions are validated as uint64 before any narrowing, then bounded by the byte budget
     * before anything is sized or allocated. */
    if (!u64_size(get_u64(e + 16u), &d) || !u64_size(get_u64(e + 24u), &w)) {
        return ELPIS_ECSG_K1_CAPACITY;
    }
    rc = shape_status(d, w, &image);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc == ELPIS_ECSG_K1_INVALID ? ELPIS_ECSG_K1_CORRUPT : rc;
    }
    if (get_u64(e + 32u) != (uint64_t)elpis_ecsg_k1_features(d) ||
        get_u64(e + 48u) > ELPIS_ECSG_K1_UNCONSOLIDATED_IMPORT || get_u64(e + 56u) != 0u ||
        size != image + ELPIS_ECSG_K1_DIGEST_BYTES) {
        return ELPIS_ECSG_K1_CORRUPT;
    }
    elpis_sha256(e, size - ELPIS_ECSG_K1_DIGEST_BYTES, digest);
    if (!elpis_digest_equal(digest, e + size - ELPIS_ECSG_K1_DIGEST_BYTES)) {
        return ELPIS_ECSG_K1_CORRUPT;
    }
    *dim = d;
    *width = w;
    return ELPIS_ECSG_K1_OK;
}

/* Decodes a validated envelope's header and payload into `image` (native binary64). */
static elpis_ecsg_k1_status decode_envelope(const elpis_ecsg_k1 *s, const uint8_t *e, uint8_t *image)
{
    const size_t doubles = s->w_count + s->h_count + s->features;
    write_header(s, image, get_u64(e + 40u), (uint32_t)get_u64(e + 48u));
    return read_doubles(e + ELPIS_ECSG_K1_HEADER_BYTES, doubles, img_w(s, image)) ? ELPIS_ECSG_K1_OK
                                                                                     : ELPIS_ECSG_K1_CORRUPT;
}

elpis_ecsg_k1_status
elpis_ecsg_k1_restore(const uint8_t *envelope, size_t size, size_t max_rows, elpis_ecsg_k1 **out)
{
    size_t dim = 0u;
    size_t width = 0u;
    elpis_ecsg_k1 *s = NULL;
    elpis_ecsg_k1_status rc;
    if (out == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    *out = NULL;
    rc = check_envelope(envelope, size, &dim, &width);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    rc = make(dim, width, max_rows, 1, &s);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    rc = decode_envelope(s, envelope, s->image);
    if (rc != ELPIS_ECSG_K1_OK) {
        (void)elpis_ecsg_k1_destroy(&s);
        return rc;
    }
    publish(s);
    *out = s;
    return ELPIS_ECSG_K1_OK;
}

elpis_ecsg_k1_status
elpis_ecsg_k1_import_w_only(const uint8_t *snap, size_t size, size_t max_rows, elpis_ecsg_k1 **out)
{
    size_t d = 0u;
    size_t w = 0u;
    elpis_ecsg_k1 *s = NULL;
    elpis_ecsg_k1_status rc;
    if (out == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    *out = NULL;
    if (snap == NULL || size < G01_HEADER || memcmp(snap, G01_MAGIC, 8u) != 0 || get_u32(snap + 8u) != 1u ||
        get_u32(snap + 12u) != 0u) {
        return ELPIS_ECSG_K1_CORRUPT;
    }
    if (!u64_size(get_u64(snap + 16u), &d) || !u64_size(get_u64(snap + 24u), &w)) {
        return ELPIS_ECSG_K1_CAPACITY;
    }
    rc = shape_status(d, w, NULL);   /* bounded before the size check and before any allocation */
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc == ELPIS_ECSG_K1_INVALID ? ELPIS_ECSG_K1_CORRUPT : rc;
    }
    if (size != G01_HEADER + d * w * 8u) {   /* d * w * 8 < the image budget: no overflow */
        return ELPIS_ECSG_K1_CORRUPT;
    }
    rc = make(d, w, max_rows, 1, &s);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    write_header(s, s->image, get_u64(snap + 32u), ELPIS_ECSG_K1_UNCONSOLIDATED_IMPORT);
    if (!read_doubles(snap + G01_HEADER, s->w_count, img_w(s, s->image))) {
        (void)elpis_ecsg_k1_destroy(&s);
        return ELPIS_ECSG_K1_CORRUPT;
    }
    publish(s);
    *out = s;
    return ELPIS_ECSG_K1_OK;
}

elpis_ecsg_k1_status elpis_ecsg_k1_destroy(elpis_ecsg_k1 **state)
{
    elpis_ecsg_k1 *s;
    if (state == NULL || *state == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    s = *state;
    if (!enter(s)) {
        return ELPIS_ECSG_K1_BUSY;
    }
    if (s->owns_image) {
        free(s->image);
    }
    free(s->arena);
    free(s);
    *state = NULL;
    return ELPIS_ECSG_K1_OK;
}

elpis_ecsg_k1_status elpis_ecsg_k1_reserve(elpis_ecsg_k1 *s, size_t max_rows)
{
    layout l;
    void *arena;
    uint8_t *old;
    elpis_ecsg_k1_status rc;
    if (s == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (!enter(s)) {
        return ELPIS_ECSG_K1_BUSY;
    }
    if (s->txn_open) {
        leave(s);
        return ELPIS_ECSG_K1_BUSY;
    }
    if (max_rows <= s->max_rows) {
        leave(s);
        return ELPIS_ECSG_K1_OK;
    }
    rc = plan(s->dim, s->width, max_rows, &l);
    if (rc != ELPIS_ECSG_K1_OK) {
        leave(s);
        return rc;
    }
    arena = aligned_alloc(ARENA_ALIGN, l.total);
    if (arena == NULL) {
        leave(s);
        return ELPIS_ECSG_K1_NOMEM;
    }
    memset(arena, 0, l.total);
    old = (uint8_t *)s->arena;
    carve(s, (uint8_t *)arena, &l);
    index_tables(s);
    s->arena = arena;
    s->arena_bytes = l.total;
    s->max_rows = max_rows;
    s->stats.heap_allocations += 1u;
    free(old);
    publish(s);
    leave(s);
    return ELPIS_ECSG_K1_OK;
}

size_t elpis_ecsg_k1_dim(const elpis_ecsg_k1 *s) { return s == NULL ? 0u : s->dim; }
size_t elpis_ecsg_k1_width(const elpis_ecsg_k1 *s) { return s == NULL ? 0u : s->width; }
/* Unguarded getters: atomic loads of the values the writer last published (no data race with a writer). */
static elpis_ecsg_k1 *published(const elpis_ecsg_k1 *s)
{
    return (elpis_ecsg_k1 *)(uintptr_t)s;
}

size_t elpis_ecsg_k1_max_rows(const elpis_ecsg_k1 *s)
{
    return s == NULL ? 0u : atomic_load_explicit(&published(s)->pub_max_rows, memory_order_acquire);
}

uint64_t elpis_ecsg_k1_generation(const elpis_ecsg_k1 *s)
{
    return s == NULL ? 0u : (uint64_t)atomic_load_explicit(&published(s)->pub_generation, memory_order_acquire);
}

uint64_t elpis_ecsg_k1_epoch(const elpis_ecsg_k1 *s)
{
    return s == NULL ? 0u : (uint64_t)atomic_load_explicit(&published(s)->pub_epoch, memory_order_acquire);
}

uint32_t elpis_ecsg_k1_provenance_of(const elpis_ecsg_k1 *s)
{
    return s == NULL ? 0u : (uint32_t)atomic_load_explicit(&published(s)->pub_provenance, memory_order_acquire);
}

/* --- operations ---------------------------------------------------------------------------------------------- */

static elpis_ecsg_k1_status admit_rows(elpis_ecsg_k1 *s, const double *x, const double *y, size_t rows)
{
    if (x == NULL || rows == 0u) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (rows > s->max_rows) {
        return ELPIS_ECSG_K1_CAPACITY;
    }
    if (!all_finite(x, rows * s->dim) || (y != NULL && !all_finite(y, rows))) {
        return ELPIS_ECSG_K1_NONFINITE;
    }
    memcpy(s->x, x, rows * s->dim * sizeof(double));
    if (y != NULL) {
        memcpy(s->y, y, rows * sizeof(double));
    }
    return ELPIS_ECSG_K1_OK;
}

static void fill(elpis_ecsg_k1_transition *t, uint64_t e0, uint64_t e1, uint64_t g0, uint64_t g1, uint64_t steps,
                 uint64_t failed)
{
    if (t != NULL) {
        t->epoch_before = e0;
        t->epoch_after = e1;
        t->generation_before = g0;
        t->generation_after = g1;
        t->steps = steps;
        t->failed_step = failed;
    }
}

static elpis_ecsg_k1_status forward_on(elpis_ecsg_k1 *s, uint8_t *image, const double *x, size_t rows,
                                       double *out)
{
    if (x == NULL || out == NULL || rows == 0u) {
        return ELPIS_ECSG_K1_INVALID;
    }
    s->stats.forward_calls += 1u;
    if (!forward_rows(s->dim, s->width, rows, x, img_w(s, image), out, s->row)) {
        s->stats.refusals += 1u;
        return ELPIS_ECSG_K1_NONFINITE;
    }
    return ELPIS_ECSG_K1_OK;
}

/* K steps on the W of `image`; on success the new W is written back into `image` and its epoch advanced. */
static elpis_ecsg_k1_status learn_on(elpis_ecsg_k1 *s, uint8_t *image, const double *x, const double *y,
                                     size_t rows, double rate, uint64_t steps, uint64_t *failed)
{
    elpis_ecsg_k1_status rc;
    uint64_t epoch = image_epoch(image);
    *failed = 0u;
    if (y == NULL || steps == 0u || !isfinite(rate) || rate < 0.0 || steps > UINT64_MAX - epoch) {
        return ELPIS_ECSG_K1_INVALID;
    }
    rc = admit_rows(s, x, y, rows);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    s->stats.learn_calls += 1u;
    *failed = run_steps(s, image, rows, rate, steps);
    if (*failed != 0u) {
        s->stats.refusals += 1u;
        return ELPIS_ECSG_K1_NONFINITE;
    }
    memcpy(img_w(s, image), s->cur, s->w_count * sizeof(double));
    put_u64(image + 40u, epoch + steps);
    return ELPIS_ECSG_K1_OK;
}

static elpis_ecsg_k1_status consolidate_on(elpis_ecsg_k1 *s, uint8_t *image, const double *x, size_t rows)
{
    elpis_ecsg_k1_status rc = admit_rows(s, x, NULL, rows);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    if (!consolidate_into(s, img_w(s, image), img_h(s, image), s->x, rows, s->stage_h, s->stage_a)) {
        s->stats.refusals += 1u;
        return ELPIS_ECSG_K1_NONFINITE;
    }
    memcpy(img_h(s, image), s->stage_h, s->h_count * sizeof(double));
    memcpy(img_a(s, image), s->stage_a, s->features * sizeof(double));
    put_u64(image + 48u, ELPIS_ECSG_K1_COMPLETE);   /* a consolidated (W, epoch, H, a) is complete */
    s->stats.consolidations += 1u;
    return ELPIS_ECSG_K1_OK;
}

elpis_ecsg_k1_status elpis_ecsg_k1_forward(elpis_ecsg_k1 *s, const double *x, size_t rows, double *out)
{
    elpis_ecsg_k1_status rc;
    if (s == NULL || s->image == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (!enter(s)) {
        return ELPIS_ECSG_K1_BUSY;
    }
    rc = forward_on(s, s->image, x, rows, out);
    leave(s);
    return rc;
}

/* Direct transitions stage outside the image (cur/nxt, stage_h/stage_a) and install only on success: the
 * authoritative image never holds a partial result. They never touch an open transaction's candidate; their
 * commit advances the generation, so that transaction is refused as STALE at its next call. */
elpis_ecsg_k1_status
elpis_ecsg_k1_learn(elpis_ecsg_k1 *s, const double *x, const double *y, size_t rows, double rate,
                    uint64_t steps, elpis_ecsg_k1_transition *t)
{
    elpis_ecsg_k1_status rc;
    uint64_t failed = 0u;
    uint64_t e0;
    uint64_t g0;
    if (s == NULL || s->image == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (!enter(s)) {
        return ELPIS_ECSG_K1_BUSY;
    }
    e0 = image_epoch(s->image);
    g0 = s->generation;
    rc = learn_on(s, s->image, x, y, rows, rate, steps, &failed);   /* staged in cur/nxt; installed on success */
    if (rc == ELPIS_ECSG_K1_OK) {
        s->generation += 1u;   /* an open transaction's source is now replaced: it will be STALE */
        s->stats.commits += 1u;
        publish(s);
        fill(t, e0, image_epoch(s->image), g0, s->generation, steps, 0u);
    } else {
        fill(t, e0, e0, g0, g0, 0u, failed);
    }
    leave(s);
    return rc;
}

elpis_ecsg_k1_status
elpis_ecsg_k1_consolidate(elpis_ecsg_k1 *s, const double *x, size_t rows, elpis_ecsg_k1_transition *t)
{
    elpis_ecsg_k1_status rc;
    uint64_t e0;
    uint64_t g0;
    if (s == NULL || s->image == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (!enter(s)) {
        return ELPIS_ECSG_K1_BUSY;
    }
    e0 = image_epoch(s->image);
    g0 = s->generation;
    rc = consolidate_on(s, s->image, x, rows);   /* staged in stage_h/stage_a, installed only on success */
    if (rc == ELPIS_ECSG_K1_OK) {
        s->generation += 1u;
        s->stats.commits += 1u;
        publish(s);
    }
    fill(t, e0, e0, g0, s->generation, 0u, 0u);
    leave(s);
    return rc;
}

elpis_ecsg_k1_status elpis_ecsg_k1_reset(elpis_ecsg_k1 *s, elpis_ecsg_k1_transition *t)
{
    uint64_t e0;
    uint64_t g0;
    if (s == NULL || s->image == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (!enter(s)) {
        return ELPIS_ECSG_K1_BUSY;
    }
    e0 = image_epoch(s->image);
    g0 = s->generation;
    memset(img_h(s, s->image), 0, (s->h_count + s->features) * sizeof(double));
    put_u64(s->image + 48u, ELPIS_ECSG_K1_RESET);
    s->generation += 1u;
    s->stats.commits += 1u;
    publish(s);
    fill(t, e0, e0, g0, s->generation, 0u, 0u);
    leave(s);
    return ELPIS_ECSG_K1_OK;
}

static elpis_ecsg_k1_status copy_out(const double *src, size_t have, double *out, size_t count)
{
    if (out == NULL || count != have) {
        return ELPIS_ECSG_K1_INVALID;
    }
    memcpy(out, src, have * sizeof(double));
    return ELPIS_ECSG_K1_OK;
}

#define COPY_FN(name, ptr, n)                                                       \
    elpis_ecsg_k1_status name(elpis_ecsg_k1 *s, double *out, size_t count)         \
    {                                                                               \
        elpis_ecsg_k1_status rc;                                                    \
        if (s == NULL || s->image == NULL) {                                        \
            return ELPIS_ECSG_K1_INVALID;                                           \
        }                                                                           \
        if (!enter(s)) {                                                            \
            return ELPIS_ECSG_K1_BUSY;                                              \
        }                                                                           \
        rc = copy_out(ptr(s, s->image), s->n, out, count);                      \
        leave(s);                                                                   \
        return rc;                                                                  \
    }

COPY_FN(elpis_ecsg_k1_copy_w, img_w, w_count)
COPY_FN(elpis_ecsg_k1_copy_h_packed, img_h, h_count)
COPY_FN(elpis_ecsg_k1_copy_a, img_a, features)

size_t elpis_ecsg_k1_snapshot_size(const elpis_ecsg_k1 *s)
{
    return s == NULL ? 0u : s->image_bytes + ELPIS_ECSG_K1_DIGEST_BYTES;
}

void ecsg_k1_internal_encode(const elpis_ecsg_k1 *s, const uint8_t *image, uint8_t *out)
{
    const size_t doubles = s->w_count + s->h_count + s->features;
    const double *v = img_w(s, (uint8_t *)(uintptr_t)image);
    size_t i;
    memcpy(out, image, ELPIS_ECSG_K1_HEADER_BYTES);
    for (i = 0u; i < doubles; ++i) {
        uint64_t bits;
        memcpy(&bits, &v[i], sizeof(bits));
        put_u64(out + ELPIS_ECSG_K1_HEADER_BYTES + 8u * i, bits);
    }
    elpis_sha256(out, s->image_bytes, out + s->image_bytes);
}

static void retained_state_digest(const elpis_ecsg_k1 *s, const uint8_t *image,
                                  uint8_t out[ELPIS_ECSG_K1_DIGEST_BYTES])
{
    const size_t doubles = s->w_count + s->h_count + s->features;
    const double *v = img_w(s, (uint8_t *)(uintptr_t)image);
    elpis_sha256_ctx ctx;
    uint8_t words[1024];
    size_t used = 0u;
    size_t i;

    elpis_sha256_init(&ctx);
    elpis_sha256_update(&ctx, image, ELPIS_ECSG_K1_HEADER_BYTES);

    for (i = 0u; i < doubles; ++i) {
        uint64_t bits;
        memcpy(&bits, &v[i], sizeof(bits));
        put_u64(words + used, bits);
        used += 8u;
        if (used == sizeof(words)) {
            elpis_sha256_update(&ctx, words, used);
            used = 0u;
        }
    }

    if (used != 0u) {
        elpis_sha256_update(&ctx, words, used);
    }
    elpis_sha256_final(&ctx, out);
}

elpis_ecsg_k1_status elpis_ecsg_k1_snapshot_write(elpis_ecsg_k1 *s, uint8_t *out, size_t size)
{
    if (s == NULL || s->image == NULL || out == NULL || size < elpis_ecsg_k1_snapshot_size(s)) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (!enter(s)) {
        return ELPIS_ECSG_K1_BUSY;
    }
    ecsg_k1_internal_encode(s, s->image, out);
    leave(s);
    return ELPIS_ECSG_K1_OK;
}

elpis_ecsg_k1_status
elpis_ecsg_k1_state_digest(
    elpis_ecsg_k1 *s,
    uint8_t out[ELPIS_ECSG_K1_DIGEST_BYTES]
)
{
    if (s == NULL || s->image == NULL || out == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (!enter(s)) {
        return ELPIS_ECSG_K1_BUSY;
    }

    retained_state_digest(s, s->image, out);

    leave(s);
    return ELPIS_ECSG_K1_OK;
}

elpis_ecsg_k1_status elpis_ecsg_k1_stats(elpis_ecsg_k1 *s, elpis_ecsg_k1_counters *out)
{
    if (s == NULL || out == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (!enter(s)) {
        return ELPIS_ECSG_K1_BUSY;
    }
    *out = s->stats;
    out->workspace_bytes = s->arena_bytes;
    out->max_rows = s->max_rows;
    out->busy_refusals = atomic_load_explicit(&s->busy_refusals, memory_order_relaxed);
    leave(s);
    return ELPIS_ECSG_K1_OK;
}

/* --- transactions ------------------------------------------------------------------------------------------- */

static elpis_ecsg_k1_status txn_check(elpis_ecsg_k1 *s, uint64_t token)
{
    if (!s->txn_open || token == 0u || token != s->txn_token) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (s->generation != s->txn_source_generation) {
        s->txn_open = 0;
        s->stats.stale_refusals += 1u;
        return ELPIS_ECSG_K1_STALE;
    }
    return ELPIS_ECSG_K1_OK;
}

/* The refusal contract (ecsg_k1.h): NONFINITE from a candidate-mutating call discards the transaction;
 * INVALID and CAPACITY are refused before the candidate is touched and leave it open and unchanged. */
static void discard_on(elpis_ecsg_k1 *s, elpis_ecsg_k1_status rc)
{
    if (rc == ELPIS_ECSG_K1_NONFINITE) {
        s->txn_open = 0;
        s->stats.txn_aborts += 1u;
    }
}

elpis_ecsg_k1_status elpis_ecsg_k1_txn_begin(elpis_ecsg_k1 *s, uint64_t *token)
{
    if (s == NULL || s->image == NULL || token == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (!enter(s)) {
        return ELPIS_ECSG_K1_BUSY;
    }
    if (s->txn_open) {
        leave(s);
        return ELPIS_ECSG_K1_BUSY;
    }
    memcpy(s->cand, s->image, s->image_bytes);
    s->txn_open = 1;
    s->txn_token = ++s->txn_tokens_issued;
    s->txn_source_generation = s->generation;
    s->stats.txn_begins += 1u;
    *token = s->txn_token;
    leave(s);
    return ELPIS_ECSG_K1_OK;
}

elpis_ecsg_k1_status
elpis_ecsg_k1_txn_learn(elpis_ecsg_k1 *s, uint64_t token, const double *x, const double *y, size_t rows,
                        double rate, uint64_t steps, elpis_ecsg_k1_transition *t)
{
    elpis_ecsg_k1_status rc;
    uint64_t failed = 0u;
    uint64_t e0;
    if (s == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (!enter(s)) {
        return ELPIS_ECSG_K1_BUSY;
    }
    rc = txn_check(s, token);
    if (rc == ELPIS_ECSG_K1_OK) {
        e0 = image_epoch(s->cand);
        rc = learn_on(s, s->cand, x, y, rows, rate, steps, &failed);
        if (rc != ELPIS_ECSG_K1_OK) {
            discard_on(s, rc);   /* INVALID/CAPACITY: refused before the candidate was touched; stays open */
            fill(t, e0, e0, s->generation, s->generation, 0u, failed);
        } else {
            fill(t, e0, image_epoch(s->cand), s->generation, s->generation, steps, 0u);
        }
    }
    leave(s);
    return rc;
}

elpis_ecsg_k1_status
elpis_ecsg_k1_txn_consolidate(elpis_ecsg_k1 *s, uint64_t token, const double *x, size_t rows)
{
    elpis_ecsg_k1_status rc;
    if (s == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (!enter(s)) {
        return ELPIS_ECSG_K1_BUSY;
    }
    rc = txn_check(s, token);
    if (rc == ELPIS_ECSG_K1_OK) {
        rc = consolidate_on(s, s->cand, x, rows);
        if (rc != ELPIS_ECSG_K1_OK) {
            discard_on(s, rc);
        }
    }
    leave(s);
    return rc;
}

elpis_ecsg_k1_status
elpis_ecsg_k1_txn_forward(elpis_ecsg_k1 *s, uint64_t token, const double *x, size_t rows, double *out)
{
    elpis_ecsg_k1_status rc;
    if (s == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (!enter(s)) {
        return ELPIS_ECSG_K1_BUSY;
    }
    rc = txn_check(s, token);
    if (rc == ELPIS_ECSG_K1_OK) {
        rc = forward_on(s, s->cand, x, rows, out);
    }
    leave(s);
    return rc;
}

elpis_ecsg_k1_status elpis_ecsg_k1_txn_epoch(elpis_ecsg_k1 *s, uint64_t token, uint64_t *epoch)
{
    elpis_ecsg_k1_status rc;
    if (s == NULL || epoch == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (!enter(s)) {
        return ELPIS_ECSG_K1_BUSY;
    }
    rc = txn_check(s, token);
    if (rc == ELPIS_ECSG_K1_OK) {
        *epoch = image_epoch(s->cand);
    }
    leave(s);
    return rc;
}


static elpis_ecsg_k1_status
txn_commit_common(elpis_ecsg_k1 *s, uint64_t token, elpis_ecsg_k1_transition *t,
                  uint8_t before[ELPIS_ECSG_K1_DIGEST_BYTES],
                  uint8_t after[ELPIS_ECSG_K1_DIGEST_BYTES])
{
    elpis_ecsg_k1_status rc;
    uint64_t e0;
    uint64_t g0;

    if (s == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (!enter(s)) {
        return ELPIS_ECSG_K1_BUSY;
    }

    rc = txn_check(s, token);
    if (rc == ELPIS_ECSG_K1_OK) {
        if (before != NULL) {
            retained_state_digest(s, s->image, before);
        }

        e0 = image_epoch(s->image);
        g0 = s->generation;
        memcpy(s->image, s->cand, s->image_bytes);
        s->generation += 1u;
        s->stats.commits += 1u;
        s->txn_open = 0;
        publish(s);
        fill(t, e0, image_epoch(s->image), g0, s->generation,
             image_epoch(s->image) - e0, 0u);

        if (after != NULL) {
            retained_state_digest(s, s->image, after);
        }
    }

    leave(s);
    return rc;
}

elpis_ecsg_k1_status
elpis_ecsg_k1_txn_commit(elpis_ecsg_k1 *s, uint64_t token, elpis_ecsg_k1_transition *t)
{
    return txn_commit_common(s, token, t, NULL, NULL);
}

elpis_ecsg_k1_status
elpis_ecsg_k1_txn_commit_identity(elpis_ecsg_k1 *s, uint64_t token,
                                  elpis_ecsg_k1_commit_identity *identity)
{
    if (identity == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }

    memset(identity, 0, sizeof(*identity));
    return txn_commit_common(s, token, &identity->transition,
                             identity->state_before_digest,
                             identity->state_after_digest);
}

elpis_ecsg_k1_status elpis_ecsg_k1_txn_abort(elpis_ecsg_k1 *s, uint64_t token)
{
    elpis_ecsg_k1_status rc;
    if (s == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (!enter(s)) {
        return ELPIS_ECSG_K1_BUSY;
    }
    rc = ELPIS_ECSG_K1_OK;   /* nothing open: abort is idempotent */
    if (s->txn_open) {
        if (token == 0u || token != s->txn_token) {
            rc = ELPIS_ECSG_K1_INVALID;   /* a wrong token aborts nothing */
        } else {
            s->txn_open = 0;
            s->stats.txn_aborts += 1u;
        }
    }
    leave(s);
    return rc;
}

/* --- the experience schedule ---------------------------------------------------------------------------------- */

/* Validates a complete schedule before anything is touched: OK, INVALID, CAPACITY or NONFINITE (input). */
static elpis_ecsg_k1_status schedule_check(const elpis_ecsg_k1 *s, const double *x, const double *y,
                                           size_t total_rows, const elpis_ecsg_k1_experience *schedule,
                                           size_t experiences, double rate, const double *s3_out, size_t s3_count,
                                           uint64_t epoch)
{
    size_t rows_seen = 0u;
    size_t values;
    uint64_t steps_seen = 0u;
    size_t i;
    if (x == NULL || y == NULL || schedule == NULL || s3_out == NULL || experiences < 1u ||
        experiences > ELPIS_ECSG_K1_MAX_EXPERIENCES || s3_count != s->features || total_rows < 1u ||
        !isfinite(rate) || rate < 0.0) {
        return ELPIS_ECSG_K1_INVALID;
    }
    for (i = 0u; i < experiences; ++i) {
        size_t rows;
        if (schedule[i].rows < 1u || schedule[i].steps < 1u) {
            return ELPIS_ECSG_K1_INVALID;
        }
        if (!u64_size(schedule[i].rows, &rows) || rows > s->max_rows) {
            return ELPIS_ECSG_K1_CAPACITY;
        }
        if (!add_ok(rows_seen, rows, &rows_seen) || schedule[i].steps > UINT64_MAX - steps_seen) {
            return ELPIS_ECSG_K1_INVALID;
        }
        steps_seen += schedule[i].steps;
    }
    if (rows_seen != total_rows || steps_seen > UINT64_MAX - epoch || !mul_ok(total_rows, s->dim, &values)) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (!all_finite(x, values) || !all_finite(y, total_rows)) {
        return ELPIS_ECSG_K1_NONFINITE;
    }
    return ELPIS_ECSG_K1_OK;
}

elpis_ecsg_k1_status
elpis_ecsg_k1_txn_run_schedule(elpis_ecsg_k1 *s, uint64_t token, const double *x, const double *y,
                               size_t total_rows, const elpis_ecsg_k1_experience *schedule, size_t experiences,
                               double rate, double *s3_out, size_t s3_count, elpis_ecsg_k1_schedule_result *result)
{
    elpis_ecsg_k1_schedule_result r;
    elpis_ecsg_k1_status rc;
    if (s == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    memset(&r, 0, sizeof(r));
    if (!enter(s)) {
        return ELPIS_ECSG_K1_BUSY;
    }
    rc = txn_check(s, token);
    if (rc == ELPIS_ECSG_K1_OK) {
        r.epoch_before = r.epoch_after = image_epoch(s->cand);
        rc = schedule_check(s, x, y, total_rows, schedule, experiences, rate, s3_out, s3_count, r.epoch_before);
        if (rc == ELPIS_ECSG_K1_OK) {
            size_t offset = 0u;
            size_t i;
            for (i = 0u; i < experiences && rc == ELPIS_ECSG_K1_OK; ++i) {
                const size_t rows = (size_t)schedule[i].rows;   /* validated: fits and <= max_rows */
                const double *xt = x + offset * s->dim;
                uint64_t failed = 0u;
                rc = learn_on(s, s->cand, xt, y + offset, rows, rate, schedule[i].steps, &failed);
                if (rc == ELPIS_ECSG_K1_OK) {
                    rc = consolidate_on(s, s->cand, xt, rows);
                }
                if (rc != ELPIS_ECSG_K1_OK) {
                    r.failed_experience = (uint64_t)i + 1u;
                    r.failed_step = failed;
                } else {
                    r.experiences_applied += 1u;
                    offset += rows;
                }
            }
            if (rc != ELPIS_ECSG_K1_OK) {
                /* The candidate was mutated: it is poisoned whatever the status (after validation only
                 * NONFINITE arithmetic can occur here). */
                s->txn_open = 0;
                s->stats.txn_aborts += 1u;
            } else {
                r.epoch_after = image_epoch(s->cand);
                s3_of(s, img_w(s, s->cand), s3_out);   /* the readout: S3 of the final candidate W */
            }
        } else {
            discard_on(s, rc);   /* INVALID/CAPACITY: nothing touched, still open; NONFINITE input: discarded */
        }
    }
    if (result != NULL) {
        *result = r;
    }
    leave(s);
    return rc;
}

/* --- internal interface for the residency adapter (ecsg_k1_internal.h) ------------------------------------- */

elpis_ecsg_k1_status ecsg_k1_internal_create_bound(size_t dim, size_t width, size_t max_rows, elpis_ecsg_k1 **out)
{
    return make(dim, width, max_rows, 0, out);
}

void ecsg_k1_internal_bind(elpis_ecsg_k1 *s, uint8_t *image)
{
    s->image = image;
    if (image != NULL) {
        publish(s);   /* the resident image's epoch and provenance */
    }
}

size_t ecsg_k1_internal_image_bytes(const elpis_ecsg_k1 *s)
{
    return s->image_bytes;
}

const uint8_t *ecsg_k1_internal_image(const elpis_ecsg_k1 *s)
{
    return s->image;
}

elpis_ecsg_k1_status ecsg_k1_internal_check_envelope(const uint8_t *e, size_t size, size_t *dim, size_t *width)
{
    return check_envelope(e, size, dim, width);
}

elpis_ecsg_k1_status ecsg_k1_internal_decode(const elpis_ecsg_k1 *s, const uint8_t *envelope, uint8_t *image)
{
    return decode_envelope(s, envelope, image);
}
