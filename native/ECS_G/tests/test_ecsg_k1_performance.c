/* Native K1 runtime performance (docs/performance/ECS_K1_RUNTIME.md). PERFORMANCE_ONLY, NO_SCIENTIFIC_CLAIM.
 *
 * Reports p50/p95/p99 latency and throughput of every K1 operation, standalone and FMS-resident, including
 * COLD->WARM and WARM->COLD. Latencies are descriptive (no latency threshold was registered). The gates here are
 * structural only: no allocation after create/reserve on any warm operation (the K1 heap counter, and the FMS
 * workspace counter, do not move), one native call per operation whatever K (each measured operation is one call),
 * and no per-query executor (the warm path never creates a state). ELPIS_K1_PERF_RUNS sets the sample count
 * (default 200); ELPIS_K1_PERF_FULL=1 adds K = 4000 (the qualified regime). */
#define _POSIX_C_SOURCE 200809L
#include "elpis/ecsg_k1_fms.h"
#include "elpis/fms_pal_posix.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { D = 6, N = 36, R = 64, WC = D * N, MAX_RUNS = 5000 };
static double W0[WC], X[R * D], Y[R], O[R];
static uint64_t samples[MAX_RUNS];
static unsigned runs = 200u;

static uint64_t ns(void)
{
    struct timespec t;
    (void)clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}

static int cmp(const void *a, const void *b)
{
    const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

static void report(const char *name, unsigned n, double steps)
{
    uint64_t total = 0u;
    unsigned i;
    for (i = 0; i < n; ++i) total += samples[i];
    qsort(samples, n, sizeof(samples[0]), cmp);
    printf("%-34s n=%-5u p50=%10.2f us p95=%10.2f us p99=%10.2f us  %12.1f ops/s", name, n,
           (double)samples[n / 2] / 1e3, (double)samples[(n * 95u) / 100u] / 1e3,
           (double)samples[(n * 99u) / 100u] / 1e3, (double)n * 1e9 / (double)total);
    if (steps > 0.0) printf("  %10.1f steps/s", steps * (double)n * 1e9 / (double)total);
    printf("\n");
}

#define TIME(name, steps, expr)                     \
    do {                                            \
        unsigned i_;                                \
        for (i_ = 0; i_ < runs; ++i_) {             \
            const uint64_t t_ = ns();               \
            assert((expr) == 0);                    \
            samples[i_] = ns() - t_;                \
        }                                           \
        report(name, runs, steps);                  \
    } while (0)

static uint64_t heap(elpis_ecsg_k1 *s)
{
    elpis_ecsg_k1_counters c;
    assert(elpis_ecsg_k1_stats(s, &c) == 0);
    return c.heap_allocations;
}

static uint64_t fms_heap(elpis_ecsg_k1_fms *r, uint64_t id)
{
    elpis_ecsg_k1_counters c;
    assert(elpis_ecsg_k1_fms_k1_stats(r, id, &c) == 0);
    return c.heap_allocations;
}

static int txn_cycle(elpis_ecsg_k1 *s)
{
    uint64_t tok = 0;
    if (elpis_ecsg_k1_txn_begin(s, &tok) != 0) return -1;
    if (elpis_ecsg_k1_txn_learn(s, tok, X, Y, R, 0.002, 10, NULL) != 0) return -1;
    if (elpis_ecsg_k1_txn_consolidate(s, tok, X, R) != 0) return -1;
    return elpis_ecsg_k1_txn_commit(s, tok, NULL);
}

static int fms_txn_cycle(elpis_ecsg_k1_fms *r, uint64_t id)
{
    uint64_t tok = 0;
    if (elpis_ecsg_k1_fms_txn_begin(r, id, &tok) != 0) return -1;
    if (elpis_ecsg_k1_fms_txn_learn(r, id, tok, X, Y, R, 0.002, 10, NULL) != 0) return -1;
    if (elpis_ecsg_k1_fms_txn_consolidate(r, id, tok, X, R) != 0) return -1;
    return elpis_ecsg_k1_fms_txn_commit(r, id, tok, NULL);
}

/* The canonical turn's ECS part: begin -> the whole experience schedule (K1 steps + consolidation per experience,
 * S3 readout of the candidate) in one call -> commit. 8 experiences of 8 rows x 10 steps. */
static const elpis_ecsg_k1_experience TURN[8] = {{8, 10}, {8, 10}, {8, 10}, {8, 10}, {8, 10}, {8, 10}, {8, 10}, {8, 10}};

static int turn_cycle(elpis_ecsg_k1 *s)
{
    uint64_t tok = 0;
    double s3[83];
    if (elpis_ecsg_k1_txn_begin(s, &tok) != 0) return -1;
    if (elpis_ecsg_k1_txn_run_schedule(s, tok, X, Y, R, TURN, 8, 0.002, s3, 83, NULL) != 0) return -1;
    return elpis_ecsg_k1_txn_commit(s, tok, NULL);
}

static int fms_turn_cycle(elpis_ecsg_k1_fms *r, uint64_t id)
{
    uint64_t tok = 0;
    double s3[83];
    if (elpis_ecsg_k1_fms_txn_begin(r, id, &tok) != 0) return -1;
    if (elpis_ecsg_k1_fms_txn_run_schedule(r, id, tok, X, Y, R, TURN, 8, 0.002, s3, 83, NULL) != 0) return -1;
    return elpis_ecsg_k1_fms_txn_commit(r, id, tok, NULL);
}

static int restore_once(const uint8_t *env, size_t n)
{
    elpis_ecsg_k1 *s = NULL;
    const int rc = elpis_ecsg_k1_restore(env, n, R, &s);
    if (rc == 0) (void)elpis_ecsg_k1_destroy(&s);
    return rc;
}

static elpis_ecsg_k1_fms *fms_runtime(const char *root, uint64_t warm)
{
    const size_t env = elpis_ecsg_k1_envelope_bytes(D, N);
    fms_config c;
    elpis_ecsg_k1_fms *r = NULL;
    fms_ctx *ctx;
    memset(&c, 0, sizeof(c));
    c.tier_budget[FMS_WARM] = c.domain_ceiling[FMS_DOM_RAM] = warm;
    c.tier_budget[FMS_COLD] = c.domain_ceiling[FMS_DOM_STORAGE] = (uint64_t)env * 16u;
    c.high_wm = 0.9f;
    c.low_wm = 0.7f;
    c.max_objects = 4;
    c.hot_absent_policy = c.cold_absent_policy = FMS_REJECT;
    ctx = fms_create(&c, fms_pal_posix_create(root));
    assert(ctx && elpis_ecsg_k1_fms_create(ctx, 4, &r) == 0);
    return r;
}

int main(void)
{
    const char *env_runs = getenv("ELPIS_K1_PERF_RUNS");
    const int full = getenv("ELPIS_K1_PERF_FULL") != NULL;
    const size_t image = elpis_ecsg_k1_image_bytes(D, N), envelope_bytes = elpis_ecsg_k1_envelope_bytes(D, N);
    static const uint64_t ks[] = {1u, 10u, 100u, 4000u};
    uint8_t *env = malloc(envelope_bytes);
    uint8_t key[32] = {1};
    elpis_ecsg_k1 *s = NULL;
    elpis_ecsg_k1_fms *r;
    uint64_t id = 0, h0;
    char name[64];
    size_t i, k;
    if (env_runs != NULL) {
        runs = (unsigned)strtoul(env_runs, NULL, 10);
        if (runs < 10u || runs > MAX_RUNS) runs = 200u;
    }
    for (i = 0; i < WC; ++i) W0[i] = 0.005 * (double)((int)(i % 17u) - 8);
    for (i = 0; i < R * D; ++i) X[i] = 0.025 * (double)((int)(i % 13u) - 6);
    for (i = 0; i < R; ++i) Y[i] = 0.02 * (double)((int)(i % 5u) - 2);
    printf("PERFORMANCE_ONLY NO_SCIENTIFIC_CLAIM d=%d N=%d rows=%d F=83 runs=%u image=%zu B envelope=%zu B "
           "workspace=%zu B\n", D, N, R, runs, image, envelope_bytes, elpis_ecsg_k1_workspace_bytes(D, N, R));

    /* standalone */
    assert(elpis_ecsg_k1_create(D, N, R, W0, &s) == 0);
    assert(elpis_ecsg_k1_consolidate(s, X, R, NULL) == 0);   /* H != 0: every step applies the K1 correction */
    h0 = heap(s);
    TIME("standalone query (64 rows)", 0.0, elpis_ecsg_k1_forward(s, X, R, O));
    for (k = 0; k < sizeof(ks) / sizeof(ks[0]); ++k) {
        const unsigned saved = runs;
        if (ks[k] == 4000u && !full) continue;
        if (ks[k] >= 100u) runs = runs < 50u ? runs : 50u;
        snprintf(name, sizeof(name), "standalone learn K=%llu", (unsigned long long)ks[k]);
        TIME(name, (double)ks[k], elpis_ecsg_k1_learn(s, X, Y, R, 0.002, ks[k], NULL));
        runs = saved;
    }
    TIME("standalone consolidate (64 rows)", 0.0, elpis_ecsg_k1_consolidate(s, X, R, NULL));
    TIME("standalone txn begin+learn10+cons+commit", 10.0, txn_cycle(s));
    TIME("standalone turn: 8 experiences x 10 steps", 80.0, turn_cycle(s));
    TIME("standalone snapshot", 0.0, elpis_ecsg_k1_snapshot_write(s, env, envelope_bytes));
    assert(heap(s) == h0 && "a warm standalone operation allocated");
    TIME("standalone restore (create+decode)", 0.0, restore_once(env, envelope_bytes));
    assert(elpis_ecsg_k1_destroy(&s) == 0);

    /* FMS-resident, WARM */
    r = fms_runtime("k1-perf-warm", (uint64_t)image * 8u);
    assert(elpis_ecsg_k1_fms_register(r, key, D, N, R, W0, &id) == 0);
    assert(elpis_ecsg_k1_fms_consolidate(r, id, X, R, NULL) == 0);
    h0 = fms_heap(r, id);
    TIME("fms warm query (64 rows)", 0.0, elpis_ecsg_k1_fms_forward(r, id, X, R, O));
    TIME("fms warm learn K=10", 10.0, elpis_ecsg_k1_fms_learn(r, id, X, Y, R, 0.002, 10, NULL));
    TIME("fms warm consolidate", 0.0, elpis_ecsg_k1_fms_consolidate(r, id, X, R, NULL));
    TIME("fms warm txn begin+learn10+cons+commit", 10.0, fms_txn_cycle(r, id));
    TIME("fms warm turn: 8 experiences x 10 steps", 80.0, fms_turn_cycle(r, id));
    TIME("fms warm snapshot", 0.0, elpis_ecsg_k1_fms_snapshot_write(r, id, env, envelope_bytes));
    assert(fms_heap(r, id) == h0 && "a warm FMS operation allocated in the K1 workspace (or rebuilt a state)");
    assert(elpis_ecsg_k1_fms_close(r, &id) == 0 && elpis_ecsg_k1_fms_destroy(&r) == 0);

    /* FMS COLD <-> WARM: the WARM budget holds less than one image, so every pump demotes */
    r = fms_runtime("k1-perf-cold", (uint64_t)image);
    assert(elpis_ecsg_k1_fms_register(r, key, D, N, R, W0, &id) == 0);
    {
        uint64_t demote[MAX_RUNS];
        unsigned j;
        elpis_ecsg_k1_fms_info info;
        for (j = 0; j < runs; ++j) {
            uint64_t t = ns();
            assert(elpis_ecsg_k1_fms_pump(r) == 0);
            demote[j] = ns() - t;
            assert(elpis_ecsg_k1_fms_inspect(r, id, &info) == 0 && info.tier == FMS_COLD);
            t = ns();
            assert(elpis_ecsg_k1_fms_forward(r, id, X, R, O) == 0);   /* COLD -> WARM, then the query */
            samples[j] = ns() - t;
        }
        report("fms COLD->WARM + query", runs, 0.0);
        memcpy(samples, demote, runs * sizeof(samples[0]));
        report("fms WARM->COLD, replica valid", runs, 0.0);
        for (j = 0; j < runs; ++j) {   /* a write invalidates the cold replica: demotion writes it back */
            uint64_t t;
            assert(elpis_ecsg_k1_fms_learn(r, id, X, Y, R, 0.002, 1, NULL) == 0);
            t = ns();
            assert(elpis_ecsg_k1_fms_pump(r) == 0);
            samples[j] = ns() - t;
            assert(elpis_ecsg_k1_fms_inspect(r, id, &info) == 0 && info.tier == FMS_COLD);
        }
        report("fms WARM->COLD, dirty (write-back)", runs, 0.0);
    }
    assert(elpis_ecsg_k1_fms_close(r, &id) == 0 && elpis_ecsg_k1_fms_destroy(&r) == 0);
    free(env);
    printf("structural: no allocation on any warm standalone or FMS operation; one native call per operation "
           "whatever K; no state rebuilt per warm query\n");
    return 0;
}
