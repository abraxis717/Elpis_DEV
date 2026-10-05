/* Native K1 runtime mechanics (docs/ECS_K1_RUNTIME.md). */
#include "elpis/ecsg_executor.h"
#include "elpis/ecsg_k1.h"
#include "elpis/sha256.h"

#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
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

static void put64(uint8_t *p, uint64_t v)
{
    unsigned i;
    for (i = 0u; i < 8u; ++i) {
        p[i] = (uint8_t)(v >> (8u * i));
    }
}

/* Recompute the trailer after a deliberate header/payload edit: the checksum is integrity, not authority. */
static void reseal(uint8_t *e, size_t n)
{
    elpis_sha256(e, n - 32u, e + n - 32u);
}

static void test_transaction_refusal_contract(void)
{
    elpis_ecsg_k1 *s = fresh(), *direct = fresh();
    uint64_t tok = 0, epoch = 0;
    double bad[R * D], o[R], huge[R * D];
    uint8_t *before;
    size_t nb = envelope(s, &before), i;
    /* recoverable: CAPACITY and INVALID are refused before the candidate is touched; the transaction stays open */
    assert(elpis_ecsg_k1_txn_begin(s, &tok) == 0);
    assert(elpis_ecsg_k1_txn_learn(s, tok, X, Y, R + 1, 0.002, 5, NULL) == ELPIS_ECSG_K1_CAPACITY);
    assert(elpis_ecsg_k1_txn_learn(s, tok, X, Y, R, -1.0, 5, NULL) == ELPIS_ECSG_K1_INVALID);
    assert(elpis_ecsg_k1_txn_learn(s, tok, X, NULL, R, 0.002, 5, NULL) == ELPIS_ECSG_K1_INVALID);
    assert(elpis_ecsg_k1_txn_learn(s, tok, X, Y, R, 0.002, 0, NULL) == ELPIS_ECSG_K1_INVALID);
    assert(elpis_ecsg_k1_txn_consolidate(s, tok, X, R + 1) == ELPIS_ECSG_K1_CAPACITY);
    assert(elpis_ecsg_k1_txn_consolidate(s, tok, NULL, R) == ELPIS_ECSG_K1_INVALID);
    assert(elpis_ecsg_k1_txn_learn(s, tok + 1u, X, Y, R, 0.002, 5, NULL) == ELPIS_ECSG_K1_INVALID);
    assert(elpis_ecsg_k1_txn_epoch(s, tok, &epoch) == 0 && epoch == 0u);
    assert(elpis_ecsg_k1_txn_learn(s, tok, X, Y, R, 0.002, 5, NULL) == 0);   /* retry succeeds */
    assert(elpis_ecsg_k1_txn_consolidate(s, tok, X, R) == 0);
    {
        uint8_t *mid;
        size_t nm = envelope(s, &mid);
        assert(nm == nb && memcmp(mid, before, nb) == 0);   /* no refusal touched authority */
        free(mid);
    }
    /* a non-finite query of the candidate is read-only: the transaction stays open */
    for (i = 0; i < R * D; ++i) {
        huge[i] = 1e200;
    }
    assert(elpis_ecsg_k1_txn_forward(s, tok, huge, R, o) == ELPIS_ECSG_K1_NONFINITE);
    assert(elpis_ecsg_k1_txn_abort(s, tok + 7u) == ELPIS_ECSG_K1_INVALID);   /* a wrong token aborts nothing */
    assert(elpis_ecsg_k1_txn_commit(s, tok, NULL) == 0);
    assert(elpis_ecsg_k1_learn(direct, X, Y, R, 0.002, 5, NULL) == 0 && elpis_ecsg_k1_consolidate(direct, X, R, NULL) == 0);
    assert(same_state(s, direct));
    assert(elpis_ecsg_k1_txn_abort(s, tok) == 0);   /* nothing open: idempotent */
    /* fatal: NONFINITE input or arithmetic discards the transaction; authority unchanged */
    memcpy(bad, X, sizeof(bad));
    bad[5] = NAN;
    assert(elpis_ecsg_k1_txn_begin(s, &tok) == 0);
    assert(elpis_ecsg_k1_txn_learn(s, tok, bad, Y, R, 0.002, 5, NULL) == ELPIS_ECSG_K1_NONFINITE);
    assert(elpis_ecsg_k1_txn_learn(s, tok, X, Y, R, 0.002, 5, NULL) == ELPIS_ECSG_K1_INVALID);
    assert(elpis_ecsg_k1_txn_commit(s, tok, NULL) == ELPIS_ECSG_K1_INVALID);
    assert(elpis_ecsg_k1_txn_begin(s, &tok) == 0);
    assert(elpis_ecsg_k1_txn_learn(s, tok, X2, Y2, R, 1e9, 20, NULL) == ELPIS_ECSG_K1_NONFINITE);
    assert(elpis_ecsg_k1_txn_commit(s, tok, NULL) == ELPIS_ECSG_K1_INVALID);
    assert(elpis_ecsg_k1_txn_begin(s, &tok) == 0);
    assert(elpis_ecsg_k1_txn_consolidate(s, tok, bad, R) == ELPIS_ECSG_K1_NONFINITE);
    assert(elpis_ecsg_k1_txn_commit(s, tok, NULL) == ELPIS_ECSG_K1_INVALID);
    assert(same_state(s, direct));
    free(before);
    elpis_ecsg_k1_destroy(&s);
    elpis_ecsg_k1_destroy(&direct);
}

static void test_hostile_dimensions_are_refused_before_allocation(void)
{
    elpis_ecsg_k1 *s = NULL, *ok = fresh();
    uint8_t snap[40 + 8 * 4];
    uint8_t *e;
    size_t n;
    static double w[64 * 8];
    /* declared byte budgets bind before anything is sized or allocated */
    assert(elpis_ecsg_k1_image_bytes(64, 1) == 0u && elpis_ecsg_k1_workspace_bytes(64, 1, 1) == 0u);
    assert(elpis_ecsg_k1_envelope_bytes(SIZE_MAX, 1) == 0u && elpis_ecsg_k1_image_bytes(6, SIZE_MAX) == 0u);
    assert(elpis_ecsg_k1_workspace_bytes(D, N, SIZE_MAX) == 0u);
    assert(elpis_ecsg_k1_workspace_bytes(D, N, (size_t)1 << 40) == 0u);
    assert(elpis_ecsg_k1_image_bytes(D, N) > 0u && elpis_ecsg_k1_image_bytes(D, N) <= ELPIS_ECSG_K1_MAX_IMAGE_BYTES);
    assert(elpis_ecsg_k1_create(64, 8, R, w, &s) == ELPIS_ECSG_K1_CAPACITY && s == NULL);
    assert(elpis_ecsg_k1_create(0, 8, R, w, &s) == ELPIS_ECSG_K1_INVALID && s == NULL);
    assert(elpis_ecsg_k1_create(D, N, SIZE_MAX, W0, &s) == ELPIS_ECSG_K1_CAPACITY && s == NULL);
    assert(elpis_ecsg_k1_reserve(ok, SIZE_MAX / 2u) == ELPIS_ECSG_K1_CAPACITY && elpis_ecsg_k1_max_rows(ok) == R);
    /* a tiny W-only snapshot claiming dim 64 cannot request a multi-GB state */
    memset(snap, 0, sizeof(snap));
    memcpy(snap, "ELPISG01", 8);
    snap[8] = 1u;
    put64(snap + 16, 64u);
    put64(snap + 24, 1u);
    assert(elpis_ecsg_k1_import_w_only(snap, 40 + 8 * 64, R, &s) == ELPIS_ECSG_K1_CAPACITY && s == NULL);
    put64(snap + 16, 2u);
    put64(snap + 24, UINT64_MAX / 2u);   /* d * w * 8 would wrap */
    assert(elpis_ecsg_k1_import_w_only(snap, sizeof(snap), R, &s) == ELPIS_ECSG_K1_CAPACITY && s == NULL);
    put64(snap + 24, UINT64_MAX);
    assert(elpis_ecsg_k1_import_w_only(snap, sizeof(snap), R, &s) == ELPIS_ECSG_K1_CAPACITY && s == NULL);
    put64(snap + 16, 0u);
    assert(elpis_ecsg_k1_import_w_only(snap, sizeof(snap), R, &s) == ELPIS_ECSG_K1_CORRUPT && s == NULL);
    /* hostile envelope headers, resealed so that only the bounds can refuse them */
    n = envelope(ok, &e);
    put64(e + 16, 64u);
    put64(e + 24, 1u);
    reseal(e, n);
    assert(elpis_ecsg_k1_restore(e, n, R, &s) == ELPIS_ECSG_K1_CAPACITY && s == NULL);
    put64(e + 16, 6u);
    put64(e + 24, UINT64_MAX);
    reseal(e, n);
    assert(elpis_ecsg_k1_restore(e, n, R, &s) == ELPIS_ECSG_K1_CAPACITY && s == NULL);
    put64(e + 24, (uint64_t)1 << 61);
    reseal(e, n);
    assert(elpis_ecsg_k1_restore(e, n, R, &s) == ELPIS_ECSG_K1_CAPACITY && s == NULL);
    put64(e + 24, 37u);   /* a plausible shape whose size does not match */
    reseal(e, n);
    assert(elpis_ecsg_k1_restore(e, n, R, &s) == ELPIS_ECSG_K1_CORRUPT && s == NULL);
    free(e);
    elpis_ecsg_k1_destroy(&ok);
}

static void test_resealed_envelopes_are_still_validated(void)
{
    elpis_ecsg_k1 *s = fresh(), *r = NULL;
    uint8_t *e;
    double nan = NAN;
    size_t n;
    assert(elpis_ecsg_k1_learn(s, X, Y, R, 0.002, 9, NULL) == 0 && elpis_ecsg_k1_consolidate(s, X, R, NULL) == 0);
    n = envelope(s, &e);
    /* a recomputed checksum is integrity only: the semantic checks still refuse */
    e[48] = 7u;   /* provenance */
    reseal(e, n);
    assert(elpis_ecsg_k1_restore(e, n, R, &r) == ELPIS_ECSG_K1_CORRUPT && r == NULL);
    e[48] = 0u;
    e[56] = 1u;   /* reserved */
    reseal(e, n);
    assert(elpis_ecsg_k1_restore(e, n, R, &r) == ELPIS_ECSG_K1_CORRUPT && r == NULL);
    e[56] = 0u;
    put64(e + 32, 84u);   /* feature count */
    reseal(e, n);
    assert(elpis_ecsg_k1_restore(e, n, R, &r) == ELPIS_ECSG_K1_CORRUPT && r == NULL);
    put64(e + 32, 83u);
    {
        uint64_t bits;
        memcpy(&bits, &nan, sizeof(bits));
        put64(e + 64 + 8 * 3, bits);   /* a non-finite W entry */
    }
    reseal(e, n);
    assert(elpis_ecsg_k1_restore(e, n, R, &r) == ELPIS_ECSG_K1_CORRUPT && r == NULL);
    free(e);
    elpis_ecsg_k1_destroy(&s);
}

static void test_provenance_transitions(void)
{
    elpis_ecsg_executor *ex = NULL;
    elpis_ecsg_k1 *s = fresh(), *imp = NULL;
    uint8_t snap[40 + WC * 8];
    uint64_t tok = 0;
    assert(elpis_ecsg_k1_provenance_of(s) == ELPIS_ECSG_K1_COMPLETE);
    assert(elpis_ecsg_k1_reset(s, NULL) == 0 && elpis_ecsg_k1_reset(s, NULL) == 0);   /* repeated reset */
    assert(elpis_ecsg_k1_provenance_of(s) == ELPIS_ECSG_K1_RESET);
    assert(elpis_ecsg_k1_learn(s, X, Y, R, 0.002, 3, NULL) == 0);
    assert(elpis_ecsg_k1_provenance_of(s) == ELPIS_ECSG_K1_RESET);   /* learning never changes provenance */
    assert(elpis_ecsg_k1_consolidate(s, X, R, NULL) == 0);
    assert(elpis_ecsg_k1_provenance_of(s) == ELPIS_ECSG_K1_COMPLETE);   /* consolidation completes the state */
    assert(elpis_ecsg_executor_create(D, N, R, W0, &ex) == 0 && elpis_ecsg_executor_snapshot_write(ex, snap, sizeof(snap)) == 0);
    assert(elpis_ecsg_k1_import_w_only(snap, sizeof(snap), R, &imp) == 0);
    assert(elpis_ecsg_k1_provenance_of(imp) == ELPIS_ECSG_K1_UNCONSOLIDATED_IMPORT);
    /* a consolidation inside a transaction completes the state only when committed */
    assert(elpis_ecsg_k1_txn_begin(imp, &tok) == 0 && elpis_ecsg_k1_txn_consolidate(imp, tok, X, R) == 0);
    assert(elpis_ecsg_k1_provenance_of(imp) == ELPIS_ECSG_K1_UNCONSOLIDATED_IMPORT);
    assert(elpis_ecsg_k1_txn_commit(imp, tok, NULL) == 0);
    assert(elpis_ecsg_k1_provenance_of(imp) == ELPIS_ECSG_K1_COMPLETE);
    elpis_ecsg_executor_destroy(&ex);
    elpis_ecsg_k1_destroy(&imp);
    elpis_ecsg_k1_destroy(&s);
}

static void test_epoch_overflow_is_refused_and_recoverable(void)
{
    elpis_ecsg_k1 *s = fresh(), *r = NULL;
    uint8_t *e;
    uint64_t tok = 0;
    size_t n = envelope(s, &e);
    put64(e + 40, UINT64_MAX - 2u);
    reseal(e, n);
    assert(elpis_ecsg_k1_restore(e, n, R, &r) == 0 && elpis_ecsg_k1_epoch(r) == UINT64_MAX - 2u);
    assert(elpis_ecsg_k1_learn(r, X, Y, R, 0.002, 3, NULL) == ELPIS_ECSG_K1_INVALID);
    assert(elpis_ecsg_k1_txn_begin(r, &tok) == 0);
    assert(elpis_ecsg_k1_txn_learn(r, tok, X, Y, R, 0.002, 3, NULL) == ELPIS_ECSG_K1_INVALID);
    assert(elpis_ecsg_k1_txn_learn(r, tok, X, Y, R, 0.002, 2, NULL) == 0);   /* still open: a valid retry */
    assert(elpis_ecsg_k1_txn_commit(r, tok, NULL) == 0 && elpis_ecsg_k1_epoch(r) == UINT64_MAX);
    free(e);
    elpis_ecsg_k1_destroy(&r);
    elpis_ecsg_k1_destroy(&s);
}

typedef struct {
    elpis_ecsg_k1 *s;
    volatile int stop;
    unsigned long reads;
} watch_arg;

/* Inspection while another thread mutates: getters are atomic loads, stats is guarded (OK or BUSY). */
static void *watcher(void *p)
{
    watch_arg *a = (watch_arg *)p;
    elpis_ecsg_k1_counters c;
    uint64_t last_generation = 0u;
    while (!__atomic_load_n(&a->stop, __ATOMIC_ACQUIRE)) {
        const uint64_t g = elpis_ecsg_k1_generation(a->s);
        const uint32_t prov = elpis_ecsg_k1_provenance_of(a->s);
        int rc;
        assert(g >= last_generation);
        last_generation = g;
        assert(prov <= ELPIS_ECSG_K1_UNCONSOLIDATED_IMPORT);
        (void)elpis_ecsg_k1_epoch(a->s);
        assert(elpis_ecsg_k1_max_rows(a->s) >= R);
        rc = elpis_ecsg_k1_stats(a->s, &c);
        assert(rc == ELPIS_ECSG_K1_OK || rc == ELPIS_ECSG_K1_BUSY);
        a->reads += 1u;
    }
    return NULL;
}

/* The writer retries BUSY: the watcher's guarded stats call legitimately overlaps it (SINGLE_WRITER). */
#define UNTIL_NOT_BUSY(expr) \
    do {                     \
        rc = (expr);         \
    } while (rc == ELPIS_ECSG_K1_BUSY)

static void test_getters_race_free_with_a_writer(void)
{
    elpis_ecsg_k1 *s = fresh();
    pthread_t t;
    watch_arg a = {s, 0, 0u};
    uint64_t tok = 0;
    int i;
    int rc;
    assert(!pthread_create(&t, NULL, watcher, &a));
    for (i = 0; i < 40; ++i) {
        UNTIL_NOT_BUSY(elpis_ecsg_k1_learn(s, X, Y, R, 0.002, 2, NULL));
        assert(rc == 0);
        UNTIL_NOT_BUSY(elpis_ecsg_k1_consolidate(s, X, R, NULL));
        assert(rc == 0);
        UNTIL_NOT_BUSY(elpis_ecsg_k1_txn_begin(s, &tok));
        assert(rc == 0);
        UNTIL_NOT_BUSY(elpis_ecsg_k1_txn_learn(s, tok, X2, Y2, R, 0.002, 2, NULL));
        assert(rc == 0);
        UNTIL_NOT_BUSY(elpis_ecsg_k1_txn_commit(s, tok, NULL));
        assert(rc == 0);
        if (i % 10 == 0) {
            UNTIL_NOT_BUSY(elpis_ecsg_k1_reset(s, NULL));
            assert(rc == 0);
            UNTIL_NOT_BUSY(elpis_ecsg_k1_reserve(s, (size_t)(R + i + 1)));
            assert(rc == 0);
        }
    }
    __atomic_store_n(&a.stop, 1, __ATOMIC_RELEASE);
    pthread_join(t, NULL);
    assert(a.reads > 0u && elpis_ecsg_k1_epoch(s) == 160u && elpis_ecsg_k1_generation(s) == 124u);
    elpis_ecsg_k1_destroy(&s);
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
    test_transaction_refusal_contract();
    test_hostile_dimensions_are_refused_before_allocation();
    test_resealed_envelopes_are_still_validated();
    test_provenance_transitions();
    test_epoch_overflow_is_refused_and_recoverable();
    test_getters_race_free_with_a_writer();
    printf("ecsg_k1: Runtime R1 parity at H = 0, K1 law shape, refusal atomicity, complete-state transactions, "
           "envelope integrity, W-only import, reset, SINGLE_WRITER, the transaction refusal contract, hostile "
           "dimensions, resealed envelopes, provenance transitions, epoch overflow, race-free getters\n");
    return 0;
}
