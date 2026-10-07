/* K1 residency allocation proof: once a state is WARM, query, learn, consolidate, transactions, reset and snapshot
 * through FMS allocate nothing (pin -> native K1 over resident bytes -> unpin). Allocation failures at registration
 * refuse with no partial registration and no leaked pin. Link-time wrapping covers the adapter, K1 and FMS core. */
#include "elpis/ecsg_k1_fms.h"
#include "elpis/fms_pal_posix.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long allocations;
static unsigned fail_at, calls;
void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void *__real_aligned_alloc(size_t, size_t);
int __real_posix_memalign(void **, size_t, size_t);
static int fail_now(void) { return ++calls == fail_at; }
void *__wrap_malloc(size_t n) { ++allocations; return fail_now() ? NULL : __real_malloc(n); }
void *__wrap_calloc(size_t n, size_t z) { ++allocations; return fail_now() ? NULL : __real_calloc(n, z); }
void *__wrap_realloc(void *p, size_t n) { ++allocations; return fail_now() ? NULL : __real_realloc(p, n); }
void *__wrap_aligned_alloc(size_t a, size_t n) { ++allocations; return fail_now() ? NULL : __real_aligned_alloc(a, n); }
int __wrap_posix_memalign(void **p, size_t a, size_t n) { ++allocations; return fail_now() ? 12 : __real_posix_memalign(p, a, n); }
static void fault(unsigned n) { calls = 0; fail_at = n; }

enum { D = 6, N = 36, R = 64, WC = D * N };
static double W0[WC], X[R * D], Y[R], O[R];

static elpis_ecsg_k1_fms *runtime(void)
{
    const size_t image = elpis_ecsg_k1_image_bytes(D, N);
    fms_config c;
    elpis_ecsg_k1_fms *r = NULL;
    fms_ctx *ctx;
    memset(&c, 0, sizeof(c));
    c.tier_budget[FMS_WARM] = c.domain_ceiling[FMS_DOM_RAM] = image * 8u;
    c.high_wm = 0.9f;
    c.low_wm = 0.7f;
    c.max_objects = 4;
    c.hot_absent_policy = c.cold_absent_policy = FMS_REJECT;
    ctx = fms_create(&c, fms_pal_posix_create_ram_only());
    assert(ctx);
    assert(elpis_ecsg_k1_fms_create(ctx, 4, &r) == 0);
    return r;
}

int main(void)
{
    elpis_ecsg_k1_fms *r = runtime();
    uint8_t key[32] = {1}, other[32] = {2};
    static uint8_t env[64 + 8 * (WC + 83 * 84 / 2 + 83) + 32];
    uint64_t id = 0, tok = 0, id2 = 0;
    unsigned long before;
    unsigned n, refused = 0;
    size_t i;
    int k;
    elpis_ecsg_k1_fms_metrics m;
    for (i = 0; i < WC; ++i) W0[i] = 0.01 * (double)((i * 37u) % 23u) - 0.1;
    for (i = 0; i < R * D; ++i) X[i] = 0.02 * (double)((i * 11u) % 29u) - 0.3;
    for (i = 0; i < R; ++i) Y[i] = 0.1 * X[i * D] - 0.05 * X[i * D + 2];

    assert(elpis_ecsg_k1_fms_register(r, key, D, N, R, W0, &id) == 0);
    assert(elpis_ecsg_k1_fms_forward(r, id, X, R, O) == 0);
    before = allocations;
    for (k = 0; k < 200; ++k) {
        assert(elpis_ecsg_k1_fms_forward(r, id, X, R, O) == 0);
    }
    assert(elpis_ecsg_k1_fms_learn(r, id, X, Y, R, 0.002, 100, NULL) == 0);
    assert(elpis_ecsg_k1_fms_consolidate(r, id, X, R, NULL) == 0);
    assert(elpis_ecsg_k1_fms_txn_begin(r, id, &tok) == 0);
    assert(elpis_ecsg_k1_fms_txn_learn(r, id, tok, X, Y, R, 0.002, 50, NULL) == 0);
    assert(elpis_ecsg_k1_fms_txn_consolidate(r, id, tok, X, R) == 0);
    assert(elpis_ecsg_k1_fms_txn_forward(r, id, tok, X, R, O) == 0);
    assert(elpis_ecsg_k1_fms_txn_commit(r, id, tok, NULL) == 0);
    {   /* the canonical turn over the resident state: begin -> schedule -> commit, allocating nothing */
        const elpis_ecsg_k1_experience sched[2] = {{R / 2, 25}, {R / 2, 15}};
        double s3[83];
        assert(elpis_ecsg_k1_fms_txn_begin(r, id, &tok) == 0);
        assert(elpis_ecsg_k1_fms_txn_run_schedule(r, id, tok, X, Y, R, sched, 2, 0.002, s3, 83, NULL) == 0);
        assert(elpis_ecsg_k1_fms_txn_commit(r, id, tok, NULL) == 0);
    }
    assert(elpis_ecsg_k1_fms_reset(r, id, NULL) == 0);
    assert(elpis_ecsg_k1_fms_snapshot_write(r, id, env, sizeof(env)) == 0);
    assert(allocations == before && "a warm K1 operation through FMS allocated");

    for (n = 1; n <= 12; ++n) {
        fault(n);
        if (elpis_ecsg_k1_fms_register(r, other, D, N, R, W0, &id2) != 0) {
            assert(id2 == 0u);
            ++refused;
        } else {
            fault(0);
            assert(elpis_ecsg_k1_fms_close(r, &id2) == 0);
        }
        fault(n);
        if (elpis_ecsg_k1_fms_restore(r, other, env, sizeof(env), R, &id2) != 0) {
            assert(id2 == 0u);
            ++refused;
        } else {
            fault(0);
            assert(elpis_ecsg_k1_fms_close(r, &id2) == 0);
        }
        fault(0);
    }
    assert(refused >= 8);
    assert(elpis_ecsg_k1_fms_stats(r, &m) == 0 && m.states == 1u && m.leases == 0u && !m.residency.pinned_bytes);
    assert(elpis_ecsg_k1_fms_close(r, &id) == 0 && elpis_ecsg_k1_fms_destroy(&r) == 0);
    printf("ecsg_k1_fms allocation: 0 allocations across 200 warm queries, learn, consolidate, a transaction, reset "
           "and snapshot; %u refused registration allocation points left no partial state or pin\n", refused);
    return 0;
}
