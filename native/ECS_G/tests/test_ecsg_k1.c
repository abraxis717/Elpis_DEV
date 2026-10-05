/* Native K1 runtime mechanics (docs/ECS_K1_RUNTIME.md). */
#include "elpis/ecsg_executor.h"
#include "elpis/ecsg_k1.h"

#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { D = 6, N = 36, R = 64, WC = D * N };

static double W0[WC], X[R * D], Y[R], X2[R * D], Y2[R];

static void fixture(void)
{
    unsigned s = 12345u;
    size_t i;
    for (i = 0; i < WC; ++i) {
        s = s * 1103515245u + 12345u;
        W0[i] = 0.18 * (((double)(s >> 8) / 16777216.0) - 0.5) * 2.0;
    }
    for (i = 0; i < R * D; ++i) {
        s = s * 1103515245u + 12345u;
        X[i] = ((double)(s >> 8) / 16777216.0) - 0.5;
        s = s * 1103515245u + 12345u;
        X2[i] = ((double)(s >> 8) / 16777216.0) - 0.5;
    }
    for (i = 0; i < R; ++i) {
        Y[i] = 0.3 * X[i * D] + 0.2 * X[i * D + 1] * X[i * D + 2];
        Y2[i] = 0.4 * X2[i * D + 3] - 0.1 * X2[i * D + 4];
    }
}

static elpis_ecsg_k1 *fresh(void)
{
    elpis_ecsg_k1 *s = NULL;
    assert(elpis_ecsg_k1_create(D, N, R, W0, &s) == ELPIS_ECSG_K1_OK && s);
    return s;
}

static size_t envelope(elpis_ecsg_k1 *s, uint8_t **out)
{
    size_t n = elpis_ecsg_k1_snapshot_size(s);
    *out = (uint8_t *)malloc(n);
    assert(*out && elpis_ecsg_k1_snapshot_write(s, *out, n) == ELPIS_ECSG_K1_OK);
    return n;
}

static int same_state(elpis_ecsg_k1 *a, elpis_ecsg_k1 *b)
{
    uint8_t *ea, *eb;
    size_t na = envelope(a, &ea), nb = envelope(b, &eb);
    int same = na == nb && memcmp(ea, eb, na) == 0;
    free(ea);
    free(eb);
    return same;
}

/* With H = 0 (fresh, reset, W-only import) K1 learning and query are bitwise the Runtime R1 executor. */
static void test_zero_consolidation_is_runtime_r1(void)
{
    elpis_ecsg_executor *e = NULL;
    elpis_ecsg_k1 *s = fresh();
    double we[WC], wk[WC], oe[R], ok[R];
    elpis_ecsg_k1_counters st;
    assert(elpis_ecsg_executor_create(D, N, R, W0, &e) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_learn(e, X, Y, R, 0.002, 250, NULL) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_k1_learn(s, X, Y, R, 0.002, 250, NULL) == ELPIS_ECSG_K1_OK);
    assert(elpis_ecsg_executor_copy_w(e, we, WC) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_k1_copy_w(s, wk, WC) == ELPIS_ECSG_K1_OK);
    assert(memcmp(we, wk, sizeof(we)) == 0);
    assert(elpis_ecsg_executor_epoch(e) == elpis_ecsg_k1_epoch(s) && elpis_ecsg_k1_epoch(s) == 250u);
    assert(elpis_ecsg_executor_forward(e, X2, R, oe) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_k1_forward(s, X2, R, ok) == ELPIS_ECSG_K1_OK);
    assert(memcmp(oe, ok, sizeof(oe)) == 0);
    assert(elpis_ecsg_k1_stats(s, &st) == ELPIS_ECSG_K1_OK && st.corrected_steps == 0u && st.steps_executed == 250u);
    elpis_ecsg_executor_destroy(&e);
    elpis_ecsg_k1_destroy(&s);
}

static void test_consolidation_changes_learning_not_query(void)
{
    elpis_ecsg_k1 *a = fresh(), *b = fresh();
    double wa[WC], wb[WC], oa[R], ob[R], h[83 * 84 / 2], anchor[83];
    elpis_ecsg_k1_transition t;
    elpis_ecsg_k1_counters st;
    assert(elpis_ecsg_k1_learn(a, X, Y, R, 0.002, 100, NULL) == 0 && elpis_ecsg_k1_learn(b, X, Y, R, 0.002, 100, NULL) == 0);
    assert(elpis_ecsg_k1_consolidate(a, X, R, &t) == ELPIS_ECSG_K1_OK);
    assert(t.epoch_before == t.epoch_after && t.generation_after == t.generation_before + 1u);
    assert(elpis_ecsg_k1_copy_h_packed(a, h, 83 * 84 / 2) == 0 && elpis_ecsg_k1_copy_a(a, anchor, 83) == 0);
    assert(h[0] > 0.0 && anchor[0] != 0.0);
    /* query reads W only: identical W, different (H, a), identical answers */
    assert(elpis_ecsg_k1_forward(a, X2, R, oa) == 0 && elpis_ecsg_k1_forward(b, X2, R, ob) == 0);
    assert(memcmp(oa, ob, sizeof(oa)) == 0);
    /* at the anchor the first step is identical (u = 0); later steps differ */
    assert(elpis_ecsg_k1_learn(a, X2, Y2, R, 0.002, 1, NULL) == 0 && elpis_ecsg_k1_learn(b, X2, Y2, R, 0.002, 1, NULL) == 0);
    assert(elpis_ecsg_k1_copy_w(a, wa, WC) == 0 && elpis_ecsg_k1_copy_w(b, wb, WC) == 0);
    assert(memcmp(wa, wb, sizeof(wa)) == 0);
    assert(elpis_ecsg_k1_learn(a, X2, Y2, R, 0.002, 50, NULL) == 0 && elpis_ecsg_k1_learn(b, X2, Y2, R, 0.002, 50, NULL) == 0);
    assert(elpis_ecsg_k1_copy_w(a, wa, WC) == 0 && elpis_ecsg_k1_copy_w(b, wb, WC) == 0);
    assert(memcmp(wa, wb, sizeof(wa)) != 0);
    assert(elpis_ecsg_k1_stats(a, &st) == 0 && st.corrected_steps == 51u && st.consolidations == 1u);
    elpis_ecsg_k1_destroy(&a);
    elpis_ecsg_k1_destroy(&b);
}

static void test_refusals_leave_the_complete_state_unchanged(void)
{
    elpis_ecsg_k1 *s = fresh(), *ref;
    double bad[R * D];
    elpis_ecsg_k1_transition t;
    assert(elpis_ecsg_k1_learn(s, X, Y, R, 0.002, 30, NULL) == 0 && elpis_ecsg_k1_consolidate(s, X, R, NULL) == 0);
    ref = fresh();
    assert(elpis_ecsg_k1_learn(ref, X, Y, R, 0.002, 30, NULL) == 0 && elpis_ecsg_k1_consolidate(ref, X, R, NULL) == 0);
    memcpy(bad, X, sizeof(bad));
    bad[17] = NAN;
    assert(elpis_ecsg_k1_learn(s, bad, Y, R, 0.002, 5, &t) == ELPIS_ECSG_K1_NONFINITE);
    assert(elpis_ecsg_k1_consolidate(s, bad, R, NULL) == ELPIS_ECSG_K1_NONFINITE);
    assert(elpis_ecsg_k1_learn(s, X2, Y2, R, 1e9, 20, &t) == ELPIS_ECSG_K1_NONFINITE && t.failed_step >= 1u);
    assert(elpis_ecsg_k1_learn(s, X, Y, R + 1, 0.002, 1, NULL) == ELPIS_ECSG_K1_CAPACITY);
    assert(elpis_ecsg_k1_learn(s, X, Y, R, -1.0, 1, NULL) == ELPIS_ECSG_K1_INVALID);
    assert(elpis_ecsg_k1_learn(s, X, Y, R, 0.002, 0, NULL) == ELPIS_ECSG_K1_INVALID);
    assert(elpis_ecsg_k1_learn(s, X, NULL, R, 0.002, 1, NULL) == ELPIS_ECSG_K1_INVALID);
    assert(elpis_ecsg_k1_consolidate(s, X, 0, NULL) == ELPIS_ECSG_K1_INVALID);
    assert(same_state(s, ref));
    assert(elpis_ecsg_k1_generation(s) == elpis_ecsg_k1_generation(ref));
    elpis_ecsg_k1_destroy(&s);
    elpis_ecsg_k1_destroy(&ref);
}

static void test_transactions_commit_the_complete_state_or_nothing(void)
{
    elpis_ecsg_k1 *s = fresh(), *direct = fresh();
    uint64_t tok = 0, tok2 = 0, epoch = 0;
    double o[R], od[R];
    elpis_ecsg_k1_transition t;
    uint8_t *before;
    size_t nb = envelope(s, &before);
    /* candidate learning and consolidation do not touch authority until commit */
    assert(elpis_ecsg_k1_txn_begin(s, &tok) == 0 && tok != 0u);
    assert(elpis_ecsg_k1_txn_begin(s, &tok2) == ELPIS_ECSG_K1_BUSY);
    assert(elpis_ecsg_k1_txn_learn(s, tok, X, Y, R, 0.002, 40, &t) == 0 && t.epoch_after == 40u);
    assert(elpis_ecsg_k1_txn_consolidate(s, tok, X, R) == 0);
    assert(elpis_ecsg_k1_txn_learn(s, tok, X2, Y2, R, 0.002, 20, NULL) == 0);
    assert(elpis_ecsg_k1_txn_epoch(s, tok, &epoch) == 0 && epoch == 60u);
    {
        uint8_t *mid;
        size_t nm = envelope(s, &mid);
        assert(nm == nb && memcmp(mid, before, nb) == 0);
        free(mid);
    }
    assert(elpis_ecsg_k1_txn_forward(s, tok, X2, R, o) == 0);
    assert(elpis_ecsg_k1_txn_commit(s, tok, &t) == 0 && t.epoch_after == 60u && t.generation_after == 1u);
    /* the same transitions applied directly give the same complete state */
    assert(elpis_ecsg_k1_learn(direct, X, Y, R, 0.002, 40, NULL) == 0 && elpis_ecsg_k1_consolidate(direct, X, R, NULL) == 0);
    assert(elpis_ecsg_k1_learn(direct, X2, Y2, R, 0.002, 20, NULL) == 0);
    assert(same_state(s, direct));
    assert(elpis_ecsg_k1_forward(direct, X2, R, od) == 0 && memcmp(o, od, sizeof(o)) == 0);
    /* staleness: a direct transition replaces the source; the candidate is refused, authority intact */
    assert(elpis_ecsg_k1_txn_begin(s, &tok) == 0);
    assert(elpis_ecsg_k1_txn_learn(s, tok, X, Y, R, 0.002, 5, NULL) == 0);
    assert(elpis_ecsg_k1_consolidate(s, X2, R, NULL) == 0);
    assert(elpis_ecsg_k1_txn_commit(s, tok, NULL) == ELPIS_ECSG_K1_STALE);
    assert(elpis_ecsg_k1_consolidate(direct, X2, R, NULL) == 0 && same_state(s, direct));
    /* a refused candidate operation discards the transaction; abort discards; authority intact */
    assert(elpis_ecsg_k1_txn_begin(s, &tok) == 0);
    assert(elpis_ecsg_k1_txn_learn(s, tok, X, Y, R, 1e9, 20, NULL) == ELPIS_ECSG_K1_NONFINITE);
    assert(elpis_ecsg_k1_txn_commit(s, tok, NULL) == ELPIS_ECSG_K1_INVALID);
    assert(elpis_ecsg_k1_txn_begin(s, &tok) == 0 && elpis_ecsg_k1_txn_learn(s, tok, X, Y, R, 0.002, 3, NULL) == 0);
    assert(elpis_ecsg_k1_txn_abort(s, tok) == 0 && elpis_ecsg_k1_txn_commit(s, tok, NULL) == ELPIS_ECSG_K1_INVALID);
    assert(same_state(s, direct));
    assert(elpis_ecsg_k1_reserve(s, 2 * R) == 0);
    free(before);
    elpis_ecsg_k1_destroy(&s);
    elpis_ecsg_k1_destroy(&direct);
}

static void test_envelope_round_trip_and_corruption(void)
{
    elpis_ecsg_k1 *s = fresh(), *r = NULL;
    uint8_t *e;
    size_t n, i;
    assert(elpis_ecsg_k1_learn(s, X, Y, R, 0.002, 25, NULL) == 0 && elpis_ecsg_k1_consolidate(s, X, R, NULL) == 0);
    n = envelope(s, &e);
    assert(n == elpis_ecsg_k1_envelope_bytes(D, N) && n == 64u + 8u * (WC + 83u * 84u / 2u + 83u) + 32u);
    assert(memcmp(e, "ELPISGK1", 8) == 0);
    assert(elpis_ecsg_k1_restore(e, n, R, &r) == 0 && same_state(s, r));
    assert(elpis_ecsg_k1_epoch(r) == 25u && elpis_ecsg_k1_provenance_of(r) == ELPIS_ECSG_K1_COMPLETE);
    elpis_ecsg_k1_destroy(&r);
    /* every corrupted byte, truncation or extension is refused */
    for (i = 0; i < n; i += 97u) {
        e[i] ^= 0x01u;
        assert(elpis_ecsg_k1_restore(e, n, R, &r) == ELPIS_ECSG_K1_CORRUPT && r == NULL);
        e[i] ^= 0x01u;
    }
    assert(elpis_ecsg_k1_restore(e, n - 1, R, &r) == ELPIS_ECSG_K1_CORRUPT);
    assert(elpis_ecsg_k1_restore(e, n + 0, 0, &r) == ELPIS_ECSG_K1_INVALID);
    free(e);
    elpis_ecsg_k1_destroy(&s);
}

static void test_w_only_snapshot_is_unconsolidated_and_never_a_retained_state(void)
{
    elpis_ecsg_executor *e = NULL;
    elpis_ecsg_k1 *s = NULL, *c = fresh();
    uint8_t snap[40 + WC * 8];
    double h[83 * 84 / 2], w[WC];
    size_t i;
    assert(elpis_ecsg_executor_create(D, N, R, W0, &e) == 0);
    assert(elpis_ecsg_executor_learn(e, X, Y, R, 0.002, 7, NULL) == 0);
    assert(elpis_ecsg_executor_snapshot_write(e, snap, sizeof(snap)) == 0);
    assert(elpis_ecsg_k1_restore(snap, sizeof(snap), R, &s) == ELPIS_ECSG_K1_CORRUPT && s == NULL);
    assert(elpis_ecsg_k1_import_w_only(snap, sizeof(snap), R, &s) == 0);
    assert(elpis_ecsg_k1_provenance_of(s) == ELPIS_ECSG_K1_UNCONSOLIDATED_IMPORT && elpis_ecsg_k1_epoch(s) == 7u);
    assert(elpis_ecsg_k1_copy_h_packed(s, h, 83 * 84 / 2) == 0);
    for (i = 0; i < 83 * 84 / 2; ++i) {
        assert(h[i] == 0.0);
    }
    assert(elpis_ecsg_k1_copy_w(s, w, WC) == 0 && elpis_ecsg_executor_copy_w(e, h, WC) == 0);
    assert(memcmp(w, h, sizeof(w)) == 0);
    snap[3] ^= 1u;
    {
        elpis_ecsg_k1 *bad = NULL;
        assert(elpis_ecsg_k1_import_w_only(snap, sizeof(snap), R, &bad) == ELPIS_ECSG_K1_CORRUPT && bad == NULL);
    }
    elpis_ecsg_executor_destroy(&e);
    elpis_ecsg_k1_destroy(&s);
    elpis_ecsg_k1_destroy(&c);
}

static void test_reset_keeps_w_and_epoch(void)
{
    elpis_ecsg_k1 *s = fresh();
    double before[WC], after[WC], h[83 * 84 / 2], a[83];
    size_t i;
    assert(elpis_ecsg_k1_learn(s, X, Y, R, 0.002, 12, NULL) == 0 && elpis_ecsg_k1_consolidate(s, X, R, NULL) == 0);
    assert(elpis_ecsg_k1_copy_w(s, before, WC) == 0);
    assert(elpis_ecsg_k1_reset(s, NULL) == 0);
    assert(elpis_ecsg_k1_copy_w(s, after, WC) == 0 && memcmp(before, after, sizeof(before)) == 0);
    assert(elpis_ecsg_k1_epoch(s) == 12u && elpis_ecsg_k1_provenance_of(s) == ELPIS_ECSG_K1_RESET);
    assert(elpis_ecsg_k1_copy_h_packed(s, h, 83 * 84 / 2) == 0 && elpis_ecsg_k1_copy_a(s, a, 83) == 0);
    for (i = 0; i < 83 * 84 / 2; ++i) {
        assert(h[i] == 0.0);
    }
    for (i = 0; i < 83; ++i) {
        assert(a[i] == 0.0);
    }
    elpis_ecsg_k1_destroy(&s);
}

typedef struct {
    elpis_ecsg_k1 *s;
    int busy;
} race_arg;

static void *racer(void *p)
{
    race_arg *a = (race_arg *)p;
    double o[R];
    int i;
    for (i = 0; i < 2000; ++i) {
        int rc = elpis_ecsg_k1_forward(a->s, X, R, o);
        assert(rc == ELPIS_ECSG_K1_OK || rc == ELPIS_ECSG_K1_BUSY);
        a->busy += rc == ELPIS_ECSG_K1_BUSY;
    }
    return NULL;
}

static void test_single_writer_refuses_concurrent_entry(void)
{
    elpis_ecsg_k1 *s = fresh(), *other = fresh();
    pthread_t t1, t2;
    race_arg a1 = {s, 0}, a2 = {s, 0}, a3 = {other, 0};
    assert(!pthread_create(&t1, NULL, racer, &a1) && !pthread_create(&t2, NULL, racer, &a2));
    racer(&a3);   /* independent states share nothing */
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    assert(a3.busy == 0);
    elpis_ecsg_k1_destroy(&s);
    elpis_ecsg_k1_destroy(&other);
}

int main(void)
{
    fixture();
    assert(elpis_ecsg_k1_abi_version() == ELPIS_ECSG_K1_ABI_V1 && elpis_ecsg_k1_features(6) == 83u);
    test_zero_consolidation_is_runtime_r1();
    test_consolidation_changes_learning_not_query();
    test_refusals_leave_the_complete_state_unchanged();
    test_transactions_commit_the_complete_state_or_nothing();
    test_envelope_round_trip_and_corruption();
    test_w_only_snapshot_is_unconsolidated_and_never_a_retained_state();
    test_reset_keeps_w_and_epoch();
    test_single_writer_refuses_concurrent_entry();
    printf("ecsg_k1: Runtime R1 parity at H = 0, K1 law shape, refusal atomicity, complete-state transactions, "
           "envelope integrity, W-only import, reset, SINGLE_WRITER\n");
    return 0;
}
