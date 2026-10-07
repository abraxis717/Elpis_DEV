/* Native K1 allocation proof: after create (and an explicit reserve) no operation allocates; allocation failures in
 * create, restore and reserve refuse cleanly and leave the existing state intact. Link-time wrapping only. */
#include "elpis/ecsg_k1.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long allocations, frees;
static unsigned fail_at, calls;
void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void __real_free(void *);
void *__real_aligned_alloc(size_t, size_t);
int __real_posix_memalign(void **, size_t, size_t);
static int fail_now(void) { return ++calls == fail_at; }
void *__wrap_malloc(size_t n) { ++allocations; return fail_now() ? NULL : __real_malloc(n); }
void *__wrap_calloc(size_t n, size_t z) { ++allocations; return fail_now() ? NULL : __real_calloc(n, z); }
void *__wrap_realloc(void *p, size_t n) { ++allocations; return fail_now() ? NULL : __real_realloc(p, n); }
void __wrap_free(void *p) { if (p) ++frees; __real_free(p); }
void *__wrap_aligned_alloc(size_t a, size_t n) { ++allocations; return fail_now() ? NULL : __real_aligned_alloc(a, n); }
int __wrap_posix_memalign(void **p, size_t a, size_t n) { ++allocations; return fail_now() ? 12 : __real_posix_memalign(p, a, n); }
static void fault(unsigned n) { calls = 0; fail_at = n; }

enum { D = 6, N = 36, R = 64, WC = D * N };
static double W0[WC], X[R * D], Y[R], O[R];
static uint8_t ENV[64 + 8 * (WC + 83 * 84 / 2 + 83) + 32];

int main(void)
{
    elpis_ecsg_k1 *s = NULL, *r = NULL;
    uint64_t tok = 0;
    unsigned long before;
    unsigned n, refused = 0;
    size_t i;
    double w[WC];
    elpis_ecsg_k1_counters st;
    elpis_ecsg_k1_commit_identity identity;
    for (i = 0; i < WC; ++i) W0[i] = 0.01 * (double)((i * 37u) % 23u) - 0.1;
    for (i = 0; i < R * D; ++i) X[i] = 0.02 * (double)((i * 11u) % 29u) - 0.3;
    for (i = 0; i < R; ++i) Y[i] = 0.1 * X[i * D] - 0.05 * X[i * D + 2];

    assert(elpis_ecsg_k1_create(D, N, R / 2, W0, &s) == ELPIS_ECSG_K1_OK);
    assert(elpis_ecsg_k1_reserve(s, R) == ELPIS_ECSG_K1_OK);     /* the cold path */
    before = allocations;
    assert(elpis_ecsg_k1_forward(s, X, R, O) == 0);
    assert(elpis_ecsg_k1_learn(s, X, Y, R, 0.002, 200, NULL) == 0);
    assert(elpis_ecsg_k1_consolidate(s, X, R, NULL) == 0);
    assert(elpis_ecsg_k1_learn(s, X, Y, R, 0.002, 200, NULL) == 0);
    assert(elpis_ecsg_k1_txn_begin(s, &tok) == 0);
    assert(elpis_ecsg_k1_txn_learn(s, tok, X, Y, R, 0.002, 50, NULL) == 0);
    assert(elpis_ecsg_k1_txn_consolidate(s, tok, X, R) == 0);
    assert(elpis_ecsg_k1_txn_forward(s, tok, X, R, O) == 0);
    assert(elpis_ecsg_k1_txn_commit(s, tok, NULL) == 0);
    {   /* the canonical turn's experience schedule: begin -> schedule (learn + consolidate per experience,
         * S3 readout) -> commit, allocating nothing */
        const elpis_ecsg_k1_experience sched[3] = {{R / 4, 30}, {R / 4, 20}, {R / 4, 10}};   /* 48 of X's 64 rows */
        double s3[83];
        assert(elpis_ecsg_k1_txn_begin(s, &tok) == 0);
        assert(elpis_ecsg_k1_txn_run_schedule(s, tok, X, Y, 3 * (R / 4), sched, 3, 0.002, s3, 83, NULL) == 0);
        assert(elpis_ecsg_k1_txn_commit_identity(s, tok, &identity) == 0);
    }
    assert(elpis_ecsg_k1_reset(s, NULL) == 0);
    assert(elpis_ecsg_k1_snapshot_write(s, ENV, sizeof(ENV)) == 0);
    assert(elpis_ecsg_k1_copy_w(s, w, WC) == 0);
    assert(elpis_ecsg_k1_stats(s, &st) == 0);
    assert(allocations == before && "a steady-state K1 operation allocated");
    assert(st.heap_allocations == 4u);   /* object, arena, image at create; one arena at reserve */

    /* allocation failure in create/restore/reserve: refused, nothing partial, authority intact */
    for (n = 1; n <= 4; ++n) {
        fault(n);
        if (elpis_ecsg_k1_create(D, N, R, W0, &r) != 0) { assert(r == NULL); ++refused; }
        else { fault(0); elpis_ecsg_k1_destroy(&r); }
        fault(n);
        if (elpis_ecsg_k1_restore(ENV, sizeof(ENV), R, &r) != 0) { assert(r == NULL); ++refused; }
        else { fault(0); elpis_ecsg_k1_destroy(&r); }
        fault(0);
    }
    {
        uint8_t again[sizeof(ENV)];
        fault(1);
        assert(elpis_ecsg_k1_reserve(s, 4 * R) == ELPIS_ECSG_K1_NOMEM);
        fault(0);
        assert(elpis_ecsg_k1_max_rows(s) == R);
        assert(elpis_ecsg_k1_snapshot_write(s, again, sizeof(again)) == 0 && !memcmp(again, ENV, sizeof(ENV)));
        assert(elpis_ecsg_k1_learn(s, X, Y, R, 0.002, 5, NULL) == 0);
    }
    assert(refused >= 6);
    elpis_ecsg_k1_destroy(&s);
    printf("ecsg_k1 allocation: 0 allocations across query/learn/consolidate/transaction/reset/snapshot; "
           "%u refused allocation points left no partial state\n", refused);
    return 0;
}
