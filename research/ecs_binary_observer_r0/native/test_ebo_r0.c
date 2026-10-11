/* ECS Binary Observation R0: native mechanics of the observer against the real K1 library (RESEARCH_ONLY).
 *
 * Law admission and refusal, encode refusal without mutation, exact ties and signed zeros, the bounded window,
 * repeatability, non-interference with an actual K1 QUERY (answers, identity, retained state, counters), zero
 * allocation on the observation path, record admission, and the bounded handoff (DROP_NEWEST; SPSC under threads). */
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ebo_r0.h"
#include "elpis/ecsg_k1.h"

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define EBO_COUNT_ALLOCATIONS 0
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || __has_feature(memory_sanitizer)
#define EBO_COUNT_ALLOCATIONS 0
#else
#define EBO_COUNT_ALLOCATIONS 1
#endif
#else
#define EBO_COUNT_ALLOCATIONS 1
#endif

#if EBO_COUNT_ALLOCATIONS
/* Interpose the allocator for the whole process (the K1 library included) and count calls inside a window. */
extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void *__libc_memalign(size_t, size_t);
extern void __libc_free(void *);
static volatile int counting;
static volatile unsigned long allocations;
void *malloc(size_t n)
{
    if (counting) {
        allocations += 1u;
    }
    return __libc_malloc(n);
}
void *calloc(size_t a, size_t b)
{
    if (counting) {
        allocations += 1u;
    }
    return __libc_calloc(a, b);
}
void *realloc(void *p, size_t n)
{
    if (counting) {
        allocations += 1u;
    }
    return __libc_realloc(p, n);
}
void *aligned_alloc(size_t a, size_t n)
{
    if (counting) {
        allocations += 1u;
    }
    return __libc_memalign(a, n);
}
int posix_memalign(void **out, size_t a, size_t n)
{
    void *p;
    if (counting) {
        allocations += 1u;
    }
    p = __libc_memalign(a, n);
    if (p == NULL) {
        return 12;
    }
    *out = p;
    return 0;
}
void free(void *p)
{
    __libc_free(p);
}
#endif

enum { DIM = 6, WIDTH = 36 };

static void put_f64(uint8_t *p, double v)
{
    uint64_t bits;
    unsigned i;
    memcpy(&bits, &v, sizeof bits);
    for (i = 0u; i < 8u; ++i) {
        p[i] = (uint8_t)(bits >> (8u * i));
    }
}

static size_t gray4_blob(uint8_t *blob, const double *thresholds)
{
    size_t i;
    memcpy(blob, "EBOL", 4);
    blob[4] = 1u;
    blob[5] = 0u;
    blob[6] = EBO_ENCODER_GRAY4;
    blob[7] = 0u;
    for (i = 0u; i < EBO_GRAY4_THRESHOLDS; ++i) {
        put_f64(blob + 8u + 8u * i, thresholds[i]);
    }
    return ebo_law_blob_bytes(EBO_ENCODER_GRAY4);
}

static size_t sproj_blob(uint8_t *blob, double mu, double inv_scale)
{
    size_t k;
    size_t i;
    uint8_t *p = blob + 24u;
    uint64_t z = 0x1234567u;
    memcpy(blob, "EBOL", 4);
    blob[4] = 1u;
    blob[5] = 0u;
    blob[6] = EBO_ENCODER_SPROJ;
    blob[7] = 0u;
    put_f64(blob + 8u, mu);
    put_f64(blob + 16u, inv_scale);
    for (k = 0u; k < EBO_PAYLOAD_BITS; ++k) {
        for (i = 0u; i < EBO_SPROJ_TERMS; ++i) {
            z = z * 6364136223846793005ull + 1442695040888963407ull;
            p[k * EBO_SPROJ_TERMS + i] = (uint8_t)((z >> 33) % EBO_WINDOW_ROWS);
            p[EBO_PAYLOAD_BITS * EBO_SPROJ_TERMS + k * EBO_SPROJ_TERMS + i] = ((z >> 17) & 1u) ? 0x01u : 0xffu;
        }
    }
    p += 2u * EBO_PAYLOAD_BITS * EBO_SPROJ_TERMS;
    for (k = 0u; k < EBO_PAYLOAD_BITS; ++k) {
        put_f64(p + 8u * k, 0.0);
    }
    return ebo_law_blob_bytes(EBO_ENCODER_SPROJ);
}

static const double THRESHOLDS[EBO_GRAY4_THRESHOLDS] = {-1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0,
                                                         0.25,  0.5,  0.75,  1.0,  1.25,  1.5,  1.75};

static unsigned nibble(const uint8_t *record, size_t row)
{
    return (record[32u + row / 2u] >> (4u * (row & 1u))) & 0x0fu;
}

static void test_law_admission(ebo_law *gray, ebo_law *sproj)
{
    uint8_t blob[8192];
    uint8_t bad[8192];
    ebo_law probe;
    ebo_law sentinel;
    size_t n;
    double t[EBO_GRAY4_THRESHOLDS];

    assert(ebo_law_blob_bytes(EBO_ENCODER_GRAY4) == 128u);
    assert(ebo_law_blob_bytes(EBO_ENCODER_SPROJ) == 4120u);
    assert(ebo_law_blob_bytes(3u) == 0u);
    n = gray4_blob(blob, THRESHOLDS);
    assert(ebo_law_init(gray, blob, n) == EBO_OK);
    n = sproj_blob(blob, 0.0, 1.0);
    assert(ebo_law_init(sproj, blob, n) == EBO_OK);

    memset(&sentinel, 0xa5, sizeof sentinel);
    n = gray4_blob(blob, THRESHOLDS);
#define REFUSE(mutate, size, code)                                                                                   \
    do {                                                                                                             \
        memcpy(bad, blob, n);                                                                                        \
        mutate;                                                                                                      \
        memcpy(&probe, &sentinel, sizeof probe);                                                                     \
        assert(ebo_law_init(&probe, bad, (size)) == (code));                                                         \
        assert(memcmp(&probe, &sentinel, sizeof probe) == 0); /* a refused law is never installed */                 \
    } while (0)
    REFUSE(bad[0] = 'X', n, EBO_LAW);
    REFUSE(bad[4] = 2u, n, EBO_VERSION);
    REFUSE(bad[6] = 3u, n, EBO_LAW);
    REFUSE(bad[7] = 1u, n, EBO_LAW);
    REFUSE((void)0, n - 1u, EBO_INVALID);
    REFUSE((void)0, n + 1u, EBO_INVALID);
    REFUSE(put_f64(bad + 8u + 8u * 3u, -1.5), n, EBO_LAW);        /* not strictly increasing */
    REFUSE(put_f64(bad + 8u + 8u * 3u, NAN), n, EBO_LAW);
    REFUSE(put_f64(bad + 8u + 8u * 14u, INFINITY), n, EBO_LAW);
    assert(ebo_law_init(NULL, blob, n) == EBO_INVALID);
    assert(ebo_law_init(&probe, NULL, n) == EBO_INVALID);
    memcpy(t, THRESHOLDS, sizeof t);
    t[7] = t[6]; /* equal neighbours */
    n = gray4_blob(blob, t);
    REFUSE((void)0, n, EBO_LAW);

    n = sproj_blob(blob, 0.0, 1.0);
    REFUSE(put_f64(bad + 16u, 0.0), n, EBO_LAW);                 /* inv_scale must be > 0 */
    REFUSE(put_f64(bad + 16u, -2.0), n, EBO_LAW);
    REFUSE(put_f64(bad + 8u, NAN), n, EBO_LAW);
    REFUSE(bad[24u + 5u] = 64u, n, EBO_LAW);                      /* index outside the window */
    REFUSE(bad[24u + 1024u + 9u] = 0u, n, EBO_LAW);               /* sign neither +1 nor -1 */
    REFUSE(put_f64(bad + 24u + 2048u + 8u * 7u, NAN), n, EBO_LAW); /* tau */
    REFUSE((void)0, 128u, EBO_INVALID);
#undef REFUSE
}

static void test_encode_refusals(const ebo_law *gray)
{
    double x[70 * DIM];
    double v[70];
    uint8_t digest[32];
    uint8_t record[64];
    uint8_t sentinel[64];
    double x0[70 * DIM];
    double v0[70];
    ebo_law blank;
    size_t i;
    for (i = 0u; i < 70u * DIM; ++i) {
        x[i] = 0.01 * (double)i - 1.0;
    }
    for (i = 0u; i < 70u; ++i) {
        v[i] = 0.05 * (double)i - 1.5;
    }
    for (i = 0u; i < 32u; ++i) {
        digest[i] = (uint8_t)(i * 7u);
    }
    memcpy(x0, x, sizeof x);
    memcpy(v0, v, sizeof v);
    memset(sentinel, 0x5a, sizeof sentinel);
    memset(&blank, 0, sizeof blank);
#define REFUSE(call, code)                                                                                           \
    do {                                                                                                             \
        memcpy(record, sentinel, sizeof record);                                                                     \
        assert((call) == (code));                                                                                    \
        assert(memcmp(record, sentinel, sizeof record) == 0); /* the record buffer is untouched */                   \
        assert(memcmp(x, x0, sizeof x) == 0 && memcmp(v, v0, sizeof v) == 0);                                       \
    } while (0)
    REFUSE(ebo_encode(NULL, DIM, 8, x, v, digest, record), EBO_INVALID);
    REFUSE(ebo_encode(gray, DIM, 8, NULL, v, digest, record), EBO_INVALID);
    REFUSE(ebo_encode(gray, DIM, 8, x, NULL, digest, record), EBO_INVALID);
    REFUSE(ebo_encode(gray, DIM, 8, x, v, NULL, record), EBO_INVALID);
    assert(ebo_encode(gray, DIM, 8, x, v, digest, NULL) == EBO_INVALID);
    REFUSE(ebo_encode(&blank, DIM, 8, x, v, digest, record), EBO_LAW);
    REFUSE(ebo_encode(gray, 0u, 8, x, v, digest, record), EBO_SHAPE);
    REFUSE(ebo_encode(gray, EBO_MAX_DIM + 1u, 8, x, v, digest, record), EBO_SHAPE);
    REFUSE(ebo_encode(gray, DIM, 0u, x, v, digest, record), EBO_SHAPE);
    REFUSE(ebo_encode(gray, DIM, EBO_MAX_ROWS + 1u, x, v, digest, record), EBO_SHAPE);
    v[3] = NAN;
    v0[3] = NAN;
    memcpy(record, sentinel, sizeof record);
    assert(ebo_encode(gray, DIM, 8, x, v, digest, record) == EBO_NONFINITE);
    assert(memcmp(record, sentinel, sizeof record) == 0);
    v[3] = v0[3] = 0.0;
    v[5] = v0[5] = -INFINITY;
    REFUSE(ebo_encode(gray, DIM, 8, x, v, digest, record), EBO_NONFINITE);
    v[5] = v0[5] = 0.0;
    x[2 * DIM + 1] = x0[2 * DIM + 1] = INFINITY;
    REFUSE(ebo_encode(gray, DIM, 8, x, v, digest, record), EBO_NONFINITE);
    /* Outside the window nothing is read: a non-finite row 64+ is not observed and does not refuse. */
    x[2 * DIM + 1] = x0[2 * DIM + 1] = 0.0;
    v[66] = v0[66] = NAN;
    x[65 * DIM] = x0[65 * DIM] = NAN;
    assert(ebo_encode(gray, DIM, 70, x, v, digest, record) == EBO_OK);
#undef REFUSE
}

static void test_ties_window_repeatability(const ebo_law *gray, const ebo_law *sproj)
{
    double x[4096 * DIM];
    static double v[4096];
    uint8_t digest[32] = {0};
    uint8_t a[64];
    uint8_t b[64];
    size_t i;
    for (i = 0u; i < 4096u * DIM; ++i) {
        x[i] = sin((double)i);
    }
    for (i = 0u; i < 4096u; ++i) {
        v[i] = 2.0 * cos(0.37 * (double)i);
    }
    /* exact ties go to the upper level; Gray coding */
    v[0] = THRESHOLDS[0];          /* level 1 */
    v[1] = nextafter(THRESHOLDS[0], -INFINITY); /* level 0 */
    v[2] = THRESHOLDS[14];         /* level 15 */
    v[3] = 0.0;                    /* level 8 */
    v[4] = -0.0;                   /* -0.0 >= 0.0: level 8 as well */
    v[5] = 1e300;                  /* level 15 */
    v[6] = -1e300;                 /* level 0 */
    assert(ebo_encode(gray, DIM, 64, x, v, digest, a) == EBO_OK);
    assert(nibble(a, 0) == (1u ^ 0u) && nibble(a, 1) == 0u && nibble(a, 2) == (15u ^ 7u));
    assert(nibble(a, 3) == (8u ^ 4u) && nibble(a, 4) == (8u ^ 4u));
    assert(nibble(a, 5) == (15u ^ 7u) && nibble(a, 6) == 0u);
    /* header */
    assert(memcmp(a, "EBOR", 4) == 0 && a[4] == 1u && a[5] == 0u && a[6] == EBO_ENCODER_GRAY4 && a[7] == 0u);
    assert(a[8] == 64u && a[9] == 0u && a[10] == DIM && a[11] == 64u);
    assert(ebo_record_check(gray, a) == EBO_OK);
    /* repeatability */
    assert(ebo_encode(gray, DIM, 64, x, v, digest, b) == EBO_OK && memcmp(a, b, 64) == 0);
    /* bounded window: 4096 rows with the same first 64 rows -> same payload; header says truncated */
    assert(ebo_encode(gray, DIM, 4096, x, v, digest, b) == EBO_OK);
    assert(memcmp(a + 32, b + 32, 32) == 0 && b[7] == EBO_FLAG_TRUNCATED && b[8] == 0u && b[9] == 16u);
    assert(memcmp(a + 24, b + 24, 8) != 0); /* rows enter the input identity */
    assert(ebo_record_check(gray, b) == EBO_OK);
    /* padded: rows < 64, zero nibbles beyond */
    assert(ebo_encode(gray, DIM, 5, x, v, digest, b) == EBO_OK);
    assert(b[7] == EBO_FLAG_PADDED && b[11] == 5u && nibble(b, 5) == 0u && nibble(b, 63) == 0u);
    assert(ebo_record_check(gray, b) == EBO_OK);
    /* the input identity is exact over bit patterns: -0.0 and +0.0 in X differ */
    x[0] = 0.0;
    assert(ebo_encode(gray, DIM, 64, x, v, digest, a) == EBO_OK);
    x[0] = -0.0;
    assert(ebo_encode(gray, DIM, 64, x, v, digest, b) == EBO_OK);
    assert(memcmp(a + 24, b + 24, 8) != 0 && memcmp(a + 32, b + 32, 32) == 0);
    /* the state identity is the digest prefix, copied */
    digest[0] = 0xab;
    digest[7] = 0xcd;
    digest[8] = 0xef; /* beyond the prefix: not recorded */
    assert(ebo_encode(sproj, DIM, 64, x, v, digest, a) == EBO_OK);
    assert(a[16] == 0xab && a[23] == 0xcd && a[6] == EBO_ENCODER_SPROJ);
    digest[8] = 0x00;
    assert(ebo_encode(sproj, DIM, 64, x, v, digest, b) == EBO_OK && memcmp(a, b, 64) == 0);
    assert(ebo_record_check(sproj, a) == EBO_OK);
}

static void test_record_check(const ebo_law *gray, const ebo_law *sproj)
{
    double x[64 * DIM] = {0};
    double v[64] = {0};
    uint8_t digest[32] = {0};
    uint8_t r[64];
    uint8_t m[64];
    ebo_law blank;
    memset(&blank, 0, sizeof blank);
    assert(ebo_encode(gray, DIM, 10, x, v, digest, r) == EBO_OK);
    assert(ebo_record_check(gray, r) == EBO_OK);
    assert(ebo_record_check(sproj, r) == EBO_LAW);      /* bound to another law */
    assert(ebo_record_check(&blank, r) == EBO_LAW);
    assert(ebo_record_check(NULL, r) == EBO_INVALID && ebo_record_check(gray, NULL) == EBO_INVALID);
    memcpy(m, r, 64);
    m[0] = 'X';
    assert(ebo_record_check(gray, m) == EBO_INVALID);
    memcpy(m, r, 64);
    m[4] = 2u;
    assert(ebo_record_check(gray, m) == EBO_VERSION);
    memcpy(m, r, 64);
    m[13] ^= 1u;
    assert(ebo_record_check(gray, m) == EBO_LAW);
    memcpy(m, r, 64);
    m[7] = 0u;                                            /* flags inconsistent with rows */
    assert(ebo_record_check(gray, m) == EBO_INVALID);
    memcpy(m, r, 64);
    m[11] = 11u;
    assert(ebo_record_check(gray, m) == EBO_INVALID);
    memcpy(m, r, 64);
    m[32 + 31] = 0x10u;                                   /* a padded nibble is not zero */
    assert(ebo_record_check(gray, m) == EBO_INVALID);
    memcpy(m, r, 64);
    m[10] = 0u;
    assert(ebo_record_check(gray, m) == EBO_INVALID);
}

/* The observer never moves the observed computation: the same K1 QUERY sequence with and without observation gives
 * identical answers, identities, retained-state bytes and K1 counters, and the observation path allocates nothing. */
static void test_k1_non_interference(const ebo_law *gray, const ebo_law *sproj)
{
    double w[DIM * WIDTH];
    static double x[256 * DIM];
    static double out_off[256];
    static double out_on[256];
    uint8_t d_off[32];
    uint8_t d_on[32];
    uint8_t rec[64];
    static uint8_t slots[8][64];
    ebo_ring ring;
    elpis_ecsg_k1 *s = NULL;
    elpis_ecsg_k1_counters c_off;
    elpis_ecsg_k1_counters c_on;
    uint8_t *snap_before;
    uint8_t *snap_after;
    size_t snap;
    size_t i;
    int pass;
    const size_t rows_seq[5] = {1u, 7u, 64u, 65u, 256u};

    for (i = 0u; i < DIM * WIDTH; ++i) {
        w[i] = 0.18 * sin(1.7 * (double)i + 0.3);
    }
    for (i = 0u; i < 256u * DIM; ++i) {
        x[i] = 0.5 * cos(0.91 * (double)i);
    }
#if EBO_COUNT_ALLOCATIONS
    allocations = 0u;
    counting = 1;
#endif
    assert(elpis_ecsg_k1_create(DIM, WIDTH, 256u, w, &s) == ELPIS_ECSG_K1_OK);
#if EBO_COUNT_ALLOCATIONS
    counting = 0;
    assert(allocations > 0u); /* the interposed allocator is live: K1 create allocates */
#endif
    snap = elpis_ecsg_k1_snapshot_size(s);
    snap_before = malloc(snap);
    snap_after = malloc(snap);
    assert(snap_before != NULL && snap_after != NULL);
    assert(elpis_ecsg_k1_snapshot_write(s, snap_before, snap) == ELPIS_ECSG_K1_OK);
    assert(ebo_ring_init(&ring, slots, 8u) == EBO_OK);

    for (pass = 0; pass < 2; ++pass) {
        const int observe = pass == 1;
        elpis_ecsg_k1_counters before;
        elpis_ecsg_k1_counters after;
        size_t q;
        assert(elpis_ecsg_k1_stats(s, &before) == ELPIS_ECSG_K1_OK);
#if EBO_COUNT_ALLOCATIONS
        allocations = 0u;
        counting = 1;
#endif
        for (q = 0u; q < 5u; ++q) {
            const size_t rows = rows_seq[q];
            double *out = observe ? out_on : out_off;
            uint8_t *dg = observe ? d_on : d_off;
            assert(elpis_ecsg_k1_query_identity(s, DIM, x, rows, out, dg) == ELPIS_ECSG_K1_OK);
            if (observe) {
                ebo_law blank;
                double copy[256];
                uint8_t dcopy[32];
                memcpy(copy, out, rows * sizeof(double));
                memcpy(dcopy, dg, 32u);
                assert(ebo_encode(gray, DIM, rows, x, out, dg, rec) == EBO_OK);
                assert(ebo_ring_push(&ring, rec) == EBO_OK);
                assert(ebo_encode(sproj, DIM, rows, x, out, dg, rec) == EBO_OK);
                assert(ebo_ring_push(&ring, rec) == EBO_OK);
                memset(&blank, 0, sizeof blank); /* an observation failure is optional and harmless */
                assert(ebo_encode(&blank, DIM, rows, x, out, dg, rec) == EBO_LAW);
                assert(memcmp(copy, out, rows * sizeof(double)) == 0 && memcmp(dcopy, dg, 32u) == 0);
                while (ebo_ring_pop(&ring, rec) == EBO_OK) {
                    assert(ebo_record_check(rec[6] == EBO_ENCODER_GRAY4 ? gray : sproj, rec) == EBO_OK);
                }
            }
        }
#if EBO_COUNT_ALLOCATIONS
        counting = 0;
        assert(allocations == 0u); /* neither the K1 QUERY nor the observer allocates */
#endif
        assert(elpis_ecsg_k1_stats(s, &after) == ELPIS_ECSG_K1_OK);
        assert(after.forward_calls - before.forward_calls == 5u); /* the observer adds no K1 crossing */
        if (observe) {
            c_on = after;
        } else {
            c_off = after;
        }
    }
    assert(memcmp(out_off, out_on, sizeof out_off) == 0);
    assert(memcmp(d_off, d_on, 32u) == 0);
    assert(elpis_ecsg_k1_snapshot_write(s, snap_after, snap) == ELPIS_ECSG_K1_OK);
    assert(memcmp(snap_before, snap_after, snap) == 0);
    assert(c_on.learn_calls == c_off.learn_calls && c_on.commits == c_off.commits &&
           c_on.refusals == c_off.refusals && c_on.heap_allocations == c_off.heap_allocations);
    assert(elpis_ecsg_k1_epoch(s) == 0u);
    free(snap_before);
    free(snap_after);
    assert(elpis_ecsg_k1_destroy(&s) == ELPIS_ECSG_K1_OK);
}

static void test_ring_bounded(void)
{
    static uint8_t slots[4][64];
    ebo_ring ring;
    uint8_t r[64];
    uint8_t got[64];
    unsigned i;
    assert(ebo_ring_init(&ring, slots, 0u) == EBO_INVALID);
    assert(ebo_ring_init(&ring, slots, 3u) == EBO_INVALID);
    assert(ebo_ring_init(&ring, slots, 131072u) == EBO_INVALID);
    assert(ebo_ring_init(NULL, slots, 4u) == EBO_INVALID);
    assert(ebo_ring_init(&ring, NULL, 4u) == EBO_INVALID);
    assert(ebo_ring_init(&ring, slots, 4u) == EBO_OK);
    assert(ebo_ring_pop(&ring, got) == EBO_EMPTY);
    for (i = 0u; i < 6u; ++i) {
        memset(r, (int)i, sizeof r);
        assert(ebo_ring_push(&ring, r) == (i < 4u ? EBO_OK : EBO_DROPPED)); /* DROP_NEWEST, never blocks */
    }
    assert(ebo_ring_pushed(&ring) == 4u && ebo_ring_dropped(&ring) == 2u);
    for (i = 0u; i < 4u; ++i) {
        assert(ebo_ring_pop(&ring, got) == EBO_OK && got[0] == i && got[63] == i); /* FIFO; dropped never seen */
    }
    assert(ebo_ring_pop(&ring, got) == EBO_EMPTY);
    memset(r, 9, sizeof r);
    assert(ebo_ring_push(&ring, r) == EBO_OK && ebo_ring_pop(&ring, got) == EBO_OK && got[0] == 9u);
    assert(ebo_ring_push(NULL, r) == EBO_INVALID && ebo_ring_pop(NULL, got) == EBO_INVALID);
}

enum { SPSC_RECORDS = 200000 };
static ebo_ring spsc;
static uint8_t spsc_slots[64][64];
static unsigned long spsc_received;

static void *consumer(void *arg)
{
    uint8_t rec[64];
    uint64_t last = 0u;
    int done = 0;
    (void)arg;
    while (!done) {
        if (ebo_ring_pop(&spsc, rec) == EBO_OK) {
            uint64_t seq;
            memcpy(&seq, rec, sizeof seq);
            if (seq == UINT64_MAX) {
                done = 1;
            } else {
                assert(seq > last || (last == 0u && seq >= 1u)); /* strictly FIFO, no duplicate, no tear */
                assert(rec[8] == (uint8_t)seq && rec[63] == (uint8_t)(seq >> 8));
                last = seq;
                spsc_received += 1u;
            }
        }
    }
    return NULL;
}

static void test_ring_spsc_threads(void)
{
    pthread_t thread;
    uint8_t rec[64];
    uint64_t seq;
    assert(ebo_ring_init(&spsc, spsc_slots, 64u) == EBO_OK);
    assert(pthread_create(&thread, NULL, consumer, NULL) == 0);
    for (seq = 1u; seq <= SPSC_RECORDS; ++seq) {
        memset(rec, 0, sizeof rec);
        memcpy(rec, &seq, sizeof seq);
        rec[8] = (uint8_t)seq;
        rec[63] = (uint8_t)(seq >> 8);
        (void)ebo_ring_push(&spsc, rec); /* may drop under contention: the producer never waits */
    }
    seq = UINT64_MAX;
    memset(rec, 0, sizeof rec);
    memcpy(rec, &seq, sizeof seq);
    while (ebo_ring_push(&spsc, rec) != EBO_OK) {
        /* the terminator only: test-side retry */
    }
    assert(pthread_join(thread, NULL) == 0);
    assert(spsc_received + ebo_ring_dropped(&spsc) == SPSC_RECORDS);
    assert(ebo_ring_pushed(&spsc) == spsc_received + 1u);
}

int main(void)
{
    static ebo_law gray;
    static ebo_law sproj;
    assert(ebo_law_bytes() == sizeof(ebo_law) && ebo_ring_bytes() == sizeof(ebo_ring));
    test_law_admission(&gray, &sproj);
    test_encode_refusals(&gray);
    test_ties_window_repeatability(&gray, &sproj);
    test_record_check(&gray, &sproj);
    test_k1_non_interference(&gray, &sproj);
    test_ring_bounded();
    test_ring_spsc_threads();
    printf("test_ebo_r0: OK (allocation counting %s)\n", EBO_COUNT_ALLOCATIONS ? "on" : "off under a sanitizer");
    return 0;
}
