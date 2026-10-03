/* DSV4.1 Native Materializer R1 (elpis/dsv41_materializer.h).
 *
 * Runtime semantics are those of the Python oracle:
 *   rows    -> elpis.inference.rows.RowEngine.lookup + decode_row
 *   expert  -> drivers.dsv41.parameters.TensorStore.stage_image_range/_copy
 *   pages   -> elpis.substrate.file_assets.FMSFileAssets (fms_file_service.c)
 * No Python, NumPy or interpreter callback is reachable from this object.
 */
#define _POSIX_C_SOURCE 200809L
#include "elpis/dsv41_materializer.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LIVE 64
#define RING 4096
#define STAGING_LIMIT (UINT64_C(512) << 20)

typedef struct { elpis_dsv41_bank_v1 b; uint64_t stride; } bank_t;
typedef struct { uint64_t row; uint32_t index; } order_t;

typedef struct {
    elpis_dsv41_materializer id;
    unsigned active;             /* pinned calls, guarded by handles_mu */
    pthread_mutex_t mu;
    elpis_dsv41_materializer_config_v1 c;
    elpis_file_service *files;
    int sealed;
    bank_t *banks;
    uint32_t bank_count;
    elpis_dsv41_expert_v1 *experts; /* [layers][expert_count+1], expert table */
    uint8_t *present;
    uint32_t *page_size;          /* per admitted asset */
    uint64_t *asset_size;
    uint32_t asset_count;
    uint8_t *raw, *rows_out, *staging;
    order_t *order;
    uint64_t live_lease, next_lease; /* one borrowed span at a time */
    uint64_t lat[RING], res[RING];
    uint32_t kind[RING];
    uint64_t ring_at;
    elpis_dsv41_materializer_stats_v1 st;
} mat_t;

static pthread_mutex_t handles_mu = PTHREAD_MUTEX_INITIALIZER;
static mat_t *handles[MAX_LIVE];
static uint64_t next_id;

static uint64_t mono(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}
static mat_t *find(elpis_dsv41_materializer id) {
    for (unsigned i = 0; id && i < MAX_LIVE; ++i) if (handles[i] && handles[i]->id == id) return handles[i];
    return NULL;
}
static mat_t *pin(elpis_dsv41_materializer id) {
    pthread_mutex_lock(&handles_mu);
    mat_t *m = find(id);
    if (m) m->active++;
    pthread_mutex_unlock(&handles_mu);
    return m;
}
static void unpin(mat_t *m) {
    pthread_mutex_lock(&handles_mu);
    m->active--;
    pthread_mutex_unlock(&handles_mu);
}
static elpis_dsv41_materializer ctx_id(void *ctx) { return (elpis_dsv41_materializer)(uintptr_t)ctx; }
static elpis_clock_code map_file(elpis_file_status rc) {
    switch (rc) {
    case ELPIS_FILE_OK: return ELPIS_CLOCK_OK;
    case ELPIS_FILE_BUSY: return ELPIS_CLOCK_DEFER;     /* interrupted; retained nothing */
    case ELPIS_FILE_MISSING: case ELPIS_FILE_UNSUPPORTED: return ELPIS_CLOCK_INVALID;
    default: return (elpis_clock_code)rc;               /* numerically aligned */
    }
}
static void sample(mat_t *m, uint64_t ns, uint32_t kind) {
    elpis_file_service_stats_v1 f;
    elpis_file_service_stats(m->files, &f);
    m->lat[m->ring_at % RING] = ns; m->res[m->ring_at % RING] = f.resident_bytes; m->kind[m->ring_at % RING] = kind;
    m->ring_at++; m->st.samples = m->ring_at;
}

/* ---- row codec ------------------------------------------------------------- */
uint64_t elpis_dsv41_row_bytes(uint32_t codec, uint32_t dimension) {
    if (!dimension) return 0;
    if (codec == ELPIS_DSV41_ROW_F32_LE) return (uint64_t)dimension * 4;
    if (codec == ELPIS_DSV41_ROW_E4M3_E8M0_BF16 && dimension % 32 == 0) return dimension + dimension / 32;
    return 0;
}
static uint32_t bits_of(float v) { uint32_t b; memcpy(&b, &v, 4); return b; }
static float float_of(uint32_t b) { float v; memcpy(&v, &b, 4); return v; }
static void store_le(uint8_t *p, uint32_t v) { for (unsigned i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i)); }

/* Writes little-endian F32 bytes. Independent of NumPy; see rows.decode_row. */
static elpis_clock_code decode(uint32_t codec, const uint8_t *raw, uint32_t dim, uint8_t *out) {
    if (codec == ELPIS_DSV41_ROW_F32_LE) {
        for (uint32_t i = 0; i < dim; ++i) {
            const uint8_t *p = raw + 4 * (size_t)i;
            uint32_t b = (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
            if ((b & 0x7f800000u) == 0x7f800000u) return ELPIS_CLOCK_ENCODING; /* non-finite row */
            store_le(out + 4 * (size_t)i, b);
        }
        return ELPIS_CLOCK_OK;
    }
    const uint8_t *scales = raw + dim;
    for (uint32_t i = 0; i < dim; ++i) if ((raw[i] & 127) == 127) return ELPIS_CLOCK_ENCODING; /* invalid FP8 code */
    for (uint32_t j = 0; j < dim / 32; ++j) if (scales[j] == 255) return ELPIS_CLOCK_ENCODING;   /* invalid E8M0 */
    for (uint32_t i = 0; i < dim; ++i) {
        uint32_t code = raw[i], exponent = (code >> 3) & 15, mantissa = code & 7;
        float base = exponent ? ldexpf((float)(mantissa + 8), (int)exponent - 10) : ldexpf((float)mantissa, -9);
        float v = ldexpf(base, (int)scales[i / 32] - 127);
        if (code & 128) v = -v;
        if (!isfinite(v)) return ELPIS_CLOCK_ENCODING;                                         /* FP8 overflow */
        uint32_t b = bits_of(v);
        b = (b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u;                                    /* BF16 RNE */
        if (!isfinite(float_of(b))) return ELPIS_CLOCK_ENCODING;                              /* non-finite row */
        store_le(out + 4 * (size_t)i, b);
    }
    return ELPIS_CLOCK_OK;
}
elpis_clock_code elpis_dsv41_row_decode(uint32_t codec, const uint8_t *raw, size_t raw_bytes,
                                        uint32_t dimension, float *out) {
    uint64_t stride = elpis_dsv41_row_bytes(codec, dimension);
    if (!raw || !out || !stride) return ELPIS_CLOCK_INVALID;
    if (raw_bytes != stride) return ELPIS_CLOCK_ENCODING; /* row length */
    return decode(codec, raw, dimension, (uint8_t *)out);
}

/* ---- runtime services --------------------------------------------------------- */
static int by_row(const void *a, const void *b) {
    const order_t *x = a, *y = b;
    return x->row < y->row ? -1 : x->row > y->row ? 1 : x->index < y->index ? -1 : x->index > y->index;
}
static void grant(mat_t *m, elpis_clock_span *span, const uint8_t *data, size_t bytes) {
    m->live_lease = ++m->next_lease;
    span->data = data; span->bytes = bytes; span->lease = (void *)(uintptr_t)m->live_lease;
    m->st.span_acquires++; m->st.live_spans = 1;
}
static elpis_clock_code rows(void *ctx, uint32_t layer, const uint8_t bank[32], const uint64_t *ids, size_t count,
                             uint32_t dimension, elpis_clock_span *span) {
    if (!span || !bank || (count && !ids)) return ELPIS_CLOCK_INVALID;
    mat_t *m = pin(ctx_id(ctx));
    if (!m) return ELPIS_CLOCK_STALE;
    uint64_t start = mono();
    pthread_mutex_lock(&m->mu);
    elpis_clock_code rc = ELPIS_CLOCK_OK;
    const bank_t *b = NULL;
    for (uint32_t i = 0; i < m->bank_count; ++i) if (m->banks[i].b.layer == layer) b = &m->banks[i];
    if (!m->sealed) rc = ELPIS_CLOCK_INVALID;
    else if (m->live_lease) rc = ELPIS_CLOCK_BUSY;
    else if (!b) rc = ELPIS_CLOCK_INTEGRITY;
    else if (count > b->b.max_rows || count > m->c.max_rows) rc = ELPIS_CLOCK_LIMIT;               /* row batch */
    else if ((uint64_t)count * b->b.dimension * 4 > b->b.max_output_bytes) rc = ELPIS_CLOCK_LIMIT;  /* output budget */
    else if (dimension != b->b.dimension || memcmp(bank, b->b.bank, 32)) rc = ELPIS_CLOCK_INTEGRITY; /* row bank */
    else for (size_t i = 0; i < count; ++i) if (ids[i] >= b->b.rows) { rc = ELPIS_CLOCK_INVALID; break; }
    m->st.row_calls++;
    if (rc == ELPIS_CLOCK_OK) {
        size_t row_bytes = (size_t)b->b.dimension * 4;
        for (size_t i = 0; i < count; ++i) m->order[i] = (order_t){ids[i], (uint32_t)i};
        qsort(m->order, count, sizeof *m->order, by_row);
        for (size_t i = 0; i < count && rc == ELPIS_CLOCK_OK;) {
            uint64_t row = m->order[i].row;
            rc = map_file(elpis_file_service_copy(m->files, b->b.asset, b->b.offset + row * b->stride, b->stride, m->raw));
            if (rc == ELPIS_CLOCK_OK) rc = decode(b->b.codec, m->raw, b->b.dimension, m->rows_out + m->order[i].index * row_bytes);
            if (rc != ELPIS_CLOCK_OK) break;
            m->st.unique_rows++;
            const uint8_t *first = m->rows_out + m->order[i].index * row_bytes;
            for (++i; i < count && m->order[i].row == row; ++i)
                memcpy(m->rows_out + m->order[i].index * row_bytes, first, row_bytes);
        }
        if (rc == ELPIS_CLOCK_OK) {
            grant(m, span, m->rows_out, count * row_bytes);
            m->st.row_requests += count; m->st.row_bytes += count * row_bytes;
        }
    }
    if (rc != ELPIS_CLOCK_OK) m->st.refusals++;
    m->st.row_ns += mono() - start;
    if (m->sealed) sample(m, mono() - start, 1);
    pthread_mutex_unlock(&m->mu);
    unpin(m);
    return rc;
}

/* TensorStore._copy: each lease spans at most half the native page budget. */
static elpis_clock_code copy_range(mat_t *m, uint32_t asset, uint64_t offset, uint64_t size, uint8_t *out) {
    uint64_t page = m->page_size[asset], pages = m->c.file.warm_bytes / (2 * page);
    if (!pages) pages = 1;
    for (uint64_t done = 0; done < size;) {
        uint64_t count = (offset / page + pages) * page - offset;
        if (count > size - done) count = size - done;
        elpis_clock_code rc = map_file(elpis_file_service_copy(m->files, asset, offset, count, out + done));
        if (rc != ELPIS_CLOCK_OK) return rc;
        m->st.expert_chunks++;
        done += count; offset += count;
    }
    return ELPIS_CLOCK_OK;
}
static elpis_clock_code expert(void *ctx, uint32_t layer, uint32_t id, uint64_t offset, size_t length,
                               elpis_clock_span *span) {
    if (!span) return ELPIS_CLOCK_INVALID;
    mat_t *m = pin(ctx_id(ctx));
    if (!m) return ELPIS_CLOCK_STALE;
    uint64_t start = mono();
    pthread_mutex_lock(&m->mu);
    elpis_clock_code rc = ELPIS_CLOCK_OK;
    size_t at = (size_t)layer * (m->c.expert_count + 1) + id;
    if (!m->sealed || layer >= m->c.layers || id > m->c.expert_count) rc = ELPIS_CLOCK_INVALID;
    else if (!m->present[at]) rc = ELPIS_CLOCK_INTEGRITY;       /* resident experts are never staged */
    else if (!length || offset > m->c.image_bytes || length > m->c.image_bytes - offset) rc = ELPIS_CLOCK_INVALID;
    else if (m->live_lease) rc = ELPIS_CLOCK_BUSY;               /* expert staging in use */
    else if (length > m->c.staging_bytes) rc = ELPIS_CLOCK_LIMIT; /* expert staging budget */
    m->st.expert_calls++;
    if (rc == ELPIS_CLOCK_OK) {
        const elpis_dsv41_expert_v1 *e = &m->experts[at];
        if (length > m->st.staging_high_water) m->st.staging_high_water = length;
        uint64_t base = 0, pos = 0;
        for (unsigned r = 0; r < 3 && rc == ELPIS_CLOCK_OK; ++r) {
            uint64_t end = offset + length, lo = offset > base ? offset : base;
            uint64_t hi = end < base + e->sizes[r] ? end : base + e->sizes[r];
            if (lo < hi) {
                rc = copy_range(m, e->assets[r], e->offsets[r] + (lo - base), hi - lo, m->staging + pos);
                pos += hi - lo;
            }
            base += e->sizes[r];
        }
        if (rc == ELPIS_CLOCK_OK) {
            grant(m, span, m->staging, length);
            memcpy(span->digests, e->digests, 96);
            m->st.expert_bytes += length;
        }
    }
    if (rc != ELPIS_CLOCK_OK) m->st.refusals++;
    m->st.expert_ns += mono() - start;
    if (m->sealed) sample(m, mono() - start, 2);
    pthread_mutex_unlock(&m->mu);
    unpin(m);
    return rc;
}
/* Exactly-once: a span whose lease is not the live one is a counted no-op. */
static void release(void *ctx, elpis_clock_span *span) {
    if (!span) return;
    mat_t *m = pin(ctx_id(ctx));
    if (m) {
        pthread_mutex_lock(&m->mu);
        if (m->live_lease && span->lease == (void *)(uintptr_t)m->live_lease) {
            m->live_lease = 0; m->st.span_releases++; m->st.live_spans = 0;
        } else m->st.stale_releases++;
        pthread_mutex_unlock(&m->mu);
        unpin(m);
    }
    memset(span, 0, sizeof *span);
}
/* Bounded, synchronous, idempotent. In-flight acquisitions observe the
 * interrupt at their next page boundary and return DEFER retaining nothing;
 * the mutex then waits for them. Any borrowed span is invalidated (its bytes
 * remain readable until destroy) and no FMS lease survives. */
static void do_quiesce(mat_t *m) {
    elpis_file_service_interrupt(m->files);
    pthread_mutex_lock(&m->mu);
    if (m->live_lease) { m->live_lease = 0; m->st.forced_releases++; m->st.live_spans = 0; }
    m->st.forced_releases += elpis_file_service_release_all(m->files);
    m->st.quiesces++;
    pthread_mutex_unlock(&m->mu);
}
static void quiesce(void *ctx) {
    mat_t *m = pin(ctx_id(ctx));
    if (m) { do_quiesce(m); unpin(m); }
}

/* ---- control plane ----------------------------------------------------------- */
static void free_mat(mat_t *m) {
    elpis_file_service_destroy(m->files);
    free(m->banks); free(m->experts); free(m->present); free(m->page_size); free(m->asset_size);
    free(m->raw); free(m->rows_out); free(m->staging); free(m->order);
    pthread_mutex_destroy(&m->mu);
    free(m);
}
uint32_t elpis_dsv41_materializer_abi_version(void) { return ELPIS_DSV41_MATERIALIZER_ABI_V1; }
elpis_clock_code elpis_dsv41_materializer_create(const elpis_dsv41_materializer_config_v1 *c,
                                                 elpis_dsv41_materializer *out) {
    if (!c || !out || *out || c->abi_version != ELPIS_DSV41_MATERIALIZER_ABI_V1 || c->reserved || c->reserved2 ||
        !c->layers || c->layers > 4096 || c->expert_count > 65535 || !c->image_bytes ||
        !c->staging_bytes || c->staging_bytes > STAGING_LIMIT || !c->max_banks || c->max_banks > c->layers ||
        !c->max_rows || c->max_rows > 65536 || !c->max_dimension || c->max_dimension > 65536)
        return ELPIS_CLOCK_INVALID;
    size_t experts = (size_t)c->layers * (c->expert_count + 1);
    mat_t *m = calloc(1, sizeof *m);
    if (!m) return ELPIS_CLOCK_LIMIT;
    if (pthread_mutex_init(&m->mu, NULL) != 0) { free(m); return ELPIS_CLOCK_LIMIT; }
    m->c = *c;
    elpis_file_status fs = elpis_file_service_create(&c->file, &m->files);
    if (fs != ELPIS_FILE_OK) { pthread_mutex_destroy(&m->mu); free(m); return map_file(fs) == ELPIS_CLOCK_DEFER ? ELPIS_CLOCK_LIMIT : map_file(fs); }
    m->banks = calloc(c->max_banks, sizeof *m->banks);
    m->experts = calloc(experts, sizeof *m->experts);
    m->present = calloc(experts, 1);
    m->page_size = calloc(c->file.max_assets, sizeof *m->page_size);
    m->asset_size = calloc(c->file.max_assets, sizeof *m->asset_size);
    m->raw = malloc((size_t)c->max_dimension * 4);
    m->rows_out = malloc((size_t)c->max_rows * c->max_dimension * 4);
    m->staging = malloc((size_t)c->staging_bytes);
    m->order = malloc((size_t)c->max_rows * sizeof *m->order);
    if (!m->banks || !m->experts || !m->present || !m->page_size || !m->asset_size || !m->raw || !m->rows_out || !m->staging || !m->order) {
        free_mat(m); return ELPIS_CLOCK_LIMIT;
    }
    m->st.staging_budget = c->staging_bytes;
    pthread_mutex_lock(&handles_mu);
    for (unsigned i = 0; i < MAX_LIVE && next_id < UINT64_MAX; ++i) if (!handles[i]) {
        m->id = ++next_id; handles[i] = m; *out = m->id;
        pthread_mutex_unlock(&handles_mu);
        return ELPIS_CLOCK_OK;
    }
    pthread_mutex_unlock(&handles_mu);
    free_mat(m);
    return ELPIS_CLOCK_LIMIT;
}

/* Cold phase helpers: pinned, locked and refused after seal. */
static mat_t *cold(elpis_dsv41_materializer id, elpis_clock_code *rc) {
    mat_t *m = pin(id);
    if (!m) { *rc = ELPIS_CLOCK_STALE; return NULL; }
    pthread_mutex_lock(&m->mu);
    if (m->sealed) { pthread_mutex_unlock(&m->mu); unpin(m); *rc = ELPIS_CLOCK_CLOSED; return NULL; }
    *rc = ELPIS_CLOCK_OK;
    return m;
}
static void cold_done(mat_t *m) { pthread_mutex_unlock(&m->mu); unpin(m); }

elpis_clock_code elpis_dsv41_materializer_admit_asset(elpis_dsv41_materializer id, const elpis_file_asset_v1 *a,
                                                      uint32_t *asset) {
    if (!a || !asset) return ELPIS_CLOCK_INVALID;
    elpis_clock_code rc; mat_t *m = cold(id, &rc);
    if (!m) return rc;
    rc = map_file(elpis_file_service_admit(m->files, a, asset));
    if (rc == ELPIS_CLOCK_OK) { m->page_size[*asset] = a->page_size; m->asset_size[*asset] = a->size; m->asset_count = *asset + 1; }
    cold_done(m);
    return rc;
}
static int asset_range(const mat_t *m, uint32_t asset, uint64_t offset, uint64_t bytes) {
    return asset < m->asset_count && offset <= m->asset_size[asset] && bytes <= m->asset_size[asset] - offset;
}
elpis_clock_code elpis_dsv41_materializer_add_bank(elpis_dsv41_materializer id, const elpis_dsv41_bank_v1 *b) {
    if (!b) return ELPIS_CLOCK_INVALID;
    elpis_clock_code rc; mat_t *m = cold(id, &rc);
    if (!m) return rc;
    uint64_t stride = elpis_dsv41_row_bytes(b->codec, b->dimension);
    if (b->reserved || !stride || !b->rows || b->dimension > m->c.max_dimension || stride > (uint64_t)m->c.max_dimension * 4 ||
        b->layer >= m->c.layers || !b->max_rows || !b->max_output_bytes || b->rows > UINT64_MAX / stride ||
        !asset_range(m, b->asset, b->offset, b->rows * stride)) rc = ELPIS_CLOCK_INVALID;
    else if (m->bank_count == m->c.max_banks) rc = ELPIS_CLOCK_LIMIT;
    for (uint32_t i = 0; rc == ELPIS_CLOCK_OK && i < m->bank_count; ++i)
        if (m->banks[i].b.layer == b->layer) rc = ELPIS_CLOCK_INVALID; /* one bank per layer */
    if (rc == ELPIS_CLOCK_OK) m->banks[m->bank_count++] = (bank_t){*b, stride};
    cold_done(m);
    return rc;
}
elpis_clock_code elpis_dsv41_materializer_add_expert(elpis_dsv41_materializer id, const elpis_dsv41_expert_v1 *e) {
    if (!e) return ELPIS_CLOCK_INVALID;
    elpis_clock_code rc; mat_t *m = cold(id, &rc);
    if (!m) return rc;
    uint64_t total = 0;
    int ok = !e->reserved && e->layer < m->c.layers && e->expert <= m->c.expert_count;
    for (unsigned r = 0; ok && r < 3; ++r) {
        ok = e->sizes[r] && e->sizes[r] <= m->c.image_bytes - total && asset_range(m, e->assets[r], e->offsets[r], e->sizes[r]);
        if (ok) total += e->sizes[r];
    }
    size_t at = ok ? (size_t)e->layer * (m->c.expert_count + 1) + e->expert : 0;
    if (!ok || total != m->c.image_bytes || m->present[at]) rc = ELPIS_CLOCK_INVALID; /* exact image geometry */
    else { m->experts[at] = *e; m->present[at] = 1; }
    cold_done(m);
    return rc;
}
elpis_clock_code elpis_dsv41_materializer_seal(elpis_dsv41_materializer id) {
    elpis_clock_code rc; mat_t *m = cold(id, &rc);
    if (!m) return rc;
    m->sealed = 1;
    cold_done(m);
    return ELPIS_CLOCK_OK;
}
void elpis_dsv41_materializer_bind(void *context, elpis_dsv41_materializer_v1 *table) {
    if (table) *table = (elpis_dsv41_materializer_v1){ELPIS_DSV41_CLOCK_ABI_V1, 0, context, rows, expert, release, quiesce};
}
elpis_clock_code elpis_dsv41_materializer_stats(elpis_dsv41_materializer id, elpis_dsv41_materializer_stats_v1 *out) {
    if (!out) return ELPIS_CLOCK_INVALID;
    mat_t *m = pin(id);
    if (!m) return ELPIS_CLOCK_STALE;
    pthread_mutex_lock(&m->mu);
    *out = m->st;
    elpis_file_service_stats(m->files, &out->file);
    pthread_mutex_unlock(&m->mu);
    unpin(m);
    return ELPIS_CLOCK_OK;
}
elpis_clock_code elpis_dsv41_materializer_pages(elpis_dsv41_materializer id, uint32_t *assets, uint64_t *pages,
                                                size_t capacity, size_t *count) {
    if (!count || (capacity && (!assets || !pages))) return ELPIS_CLOCK_INVALID;
    mat_t *m = pin(id);
    if (!m) return ELPIS_CLOCK_STALE;
    *count = elpis_file_service_pages(m->files, assets, pages, capacity);
    unpin(m);
    return ELPIS_CLOCK_OK;
}
elpis_clock_code elpis_dsv41_materializer_samples(elpis_dsv41_materializer id, uint64_t *latency_ns,
                                                  uint64_t *resident_bytes, uint32_t *kinds,
                                                  size_t capacity, size_t *count) {
    if (!count || (capacity && (!latency_ns || !resident_bytes || !kinds))) return ELPIS_CLOCK_INVALID;
    mat_t *m = pin(id);
    if (!m) return ELPIS_CLOCK_STALE;
    pthread_mutex_lock(&m->mu);
    uint64_t n = m->ring_at < RING ? m->ring_at : RING, first = m->ring_at - n;
    for (uint64_t i = 0; i < n && i < capacity; ++i) {
        uint64_t k = (first + i) % RING;
        latency_ns[i] = m->lat[k]; resident_bytes[i] = m->res[k]; kinds[i] = m->kind[k];
    }
    *count = (size_t)n;
    pthread_mutex_unlock(&m->mu);
    unpin(m);
    return ELPIS_CLOCK_OK;
}
elpis_clock_code elpis_dsv41_materializer_evict(elpis_dsv41_materializer id) {
    mat_t *m = pin(id);
    if (!m) return ELPIS_CLOCK_STALE;
    elpis_clock_code rc = map_file(elpis_file_service_evict(m->files, UINT32_MAX));
    unpin(m);
    return rc == ELPIS_CLOCK_DEFER ? ELPIS_CLOCK_BUSY : rc;
}
elpis_clock_code elpis_dsv41_materializer_quiesce(elpis_dsv41_materializer id) {
    mat_t *m = pin(id);
    if (!m) return ELPIS_CLOCK_STALE;
    do_quiesce(m);
    unpin(m);
    return ELPIS_CLOCK_OK;
}
elpis_clock_code elpis_dsv41_materializer_destroy(elpis_dsv41_materializer id) {
    pthread_mutex_lock(&handles_mu);
    mat_t *m = find(id);
    if (!m) {
        int old = id && id <= next_id;
        pthread_mutex_unlock(&handles_mu);
        return old ? ELPIS_CLOCK_OK : ELPIS_CLOCK_STALE;
    }
    /* Only pinned calls mutate live_lease; with none active it is stable here. */
    if (m->active || m->live_lease) {
        elpis_file_service_interrupt(m->files); /* lock-free; m cannot be freed while held */
        pthread_mutex_unlock(&handles_mu);
        return ELPIS_CLOCK_BUSY;
    }
    for (unsigned i = 0; i < MAX_LIVE; ++i) if (handles[i] == m) handles[i] = NULL;
    pthread_mutex_unlock(&handles_mu);
    free_mat(m);
    return ELPIS_CLOCK_OK;
}
