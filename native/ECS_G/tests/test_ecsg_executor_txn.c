/*
 * ECS_G executor R1: native candidate transactions (begin, learn, readout,
 * commit, abort, generation staleness) and SINGLE_WRITER enforcement.
 */
#include "elpis/ecsg_executor.h"
#include "elpis/ecsg_math.h"
#include "elpis/ecsg_state.h"

#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { DIM = 6, WIDTH = 36, ROWS = 32, WCOUNT = DIM * WIDTH, SNAP = 40 + WCOUNT * 8 };

static uint64_t rng = UINT64_C(0x7a11ce);

static double
gauss(double scale)
{
    double u, v;
    uint64_t z;
    int k;
    double r[2];

    for (k = 0; k < 2; ++k) {
        z = (rng += UINT64_C(0x9E3779B97F4A7C15));
        z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
        z ^= z >> 31;
        r[k] = ((double)(z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    }
    u = r[0];
    v = r[1];
    return scale * sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}

static void
fill(double *v, size_t n, double scale)
{
    size_t i;
    for (i = 0u; i < n; ++i) {
        v[i] = gauss(scale);
    }
}

static double W0[WCOUNT], X[ROWS * DIM], Y[ROWS], X2[ROWS * DIM], Y2[ROWS];

static void
ref_steps(elpis_ecsg_state *s, const double *x, const double *y, size_t rows, uint64_t k)
{
    const size_t count = elpis_ecsg_state_gd_step_scratch_f64(DIM, WIDTH, rows);
    double *scratch = (double *)malloc(count * sizeof(double));
    uint64_t i;
    assert(scratch != NULL);
    for (i = 0u; i < k; ++i) {
        assert(elpis_ecsg_state_gd_step_f64(s, x, y, rows, 0.002, scratch, count) == ELPIS_ECSG_MATH_OK);
    }
    free(scratch);
}

static void
snapshot(elpis_ecsg_executor *e, uint8_t *out)
{
    assert(elpis_ecsg_executor_snapshot_write(e, out, SNAP) == ELPIS_ECSG_EXEC_OK);
}

static void
assert_matches(elpis_ecsg_executor *e, const elpis_ecsg_state *s)
{
    uint8_t a[SNAP], b[SNAP];
    snapshot(e, a);
    assert(elpis_ecsg_state_snapshot_write(s, b, SNAP) == ELPIS_ECSG_MATH_OK);
    assert(memcmp(a, b, SNAP) == 0);
}

static void
test_commit_installs_candidate_and_matches_reference(void)
{
    const elpis_ecsg_drive drives[2] = {{20u, 3u}, {12u, 2u}};
    double mu[DIM], m[21], t3[56], rmu[DIM], rm[21], rt3[56];
    double out[ROWS], rout[ROWS];
    uint8_t before[SNAP], during[SNAP];
    elpis_ecsg_executor *e = NULL;
    elpis_ecsg_state *s = NULL;
    elpis_ecsg_exec_transition t;
    elpis_ecsg_exec_stats st;
    uint64_t token = 0u, epoch = 0u;

    assert(elpis_ecsg_executor_create(DIM, WIDTH, ROWS, W0, &e) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_state_create(DIM, WIDTH, W0, &s) == ELPIS_ECSG_MATH_OK);
    snapshot(e, before);

    assert(elpis_ecsg_executor_txn_begin(e, &token) == ELPIS_ECSG_EXEC_OK && token != 0u);
    /* Before any learn the candidate is the authoritative state. */
    assert(elpis_ecsg_executor_txn_forward(e, token, X, ROWS, out) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_forward(e, X, ROWS, rout) == ELPIS_ECSG_EXEC_OK);
    assert(memcmp(out, rout, sizeof(out)) == 0);

    /* Two learns accumulate on the candidate: schedule (5 steps) then 4 steps. */
    assert(elpis_ecsg_executor_txn_learn_schedule(e, token, X, Y, drives, 2u, 0.002, &t) == ELPIS_ECSG_EXEC_OK);
    assert(t.epoch_before == 0u && t.epoch_after == 5u && t.steps == 5u && t.generation_after == 0u);
    assert(elpis_ecsg_executor_txn_learn(e, token, X2, Y2, ROWS, 0.002, 4u, &t) == ELPIS_ECSG_EXEC_OK);
    assert(t.epoch_before == 5u && t.epoch_after == 9u);
    assert(elpis_ecsg_executor_txn_epoch(e, token, &epoch) == ELPIS_ECSG_EXEC_OK && epoch == 9u);
    ref_steps(s, X, Y, 20u, 3u);
    ref_steps(s, X + 20u * DIM, Y + 20u, 12u, 2u);
    ref_steps(s, X2, Y2, ROWS, 4u);

    /* Readout comes from the candidate; the authoritative state is untouched. */
    assert(elpis_ecsg_executor_txn_project_s3(e, token, mu, m, t3) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_state_project_s3_f64(s, rmu, rm, rt3) == ELPIS_ECSG_MATH_OK);
    assert(memcmp(mu, rmu, sizeof(mu)) == 0 && memcmp(m, rm, sizeof(m)) == 0 && memcmp(t3, rt3, sizeof(t3)) == 0);
    assert(elpis_ecsg_executor_txn_forward(e, token, X, ROWS, out) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_state_forward_f64(s, X, ROWS, rout) == ELPIS_ECSG_MATH_OK);
    assert(memcmp(out, rout, sizeof(out)) == 0);
    snapshot(e, during);
    assert(memcmp(before, during, SNAP) == 0);
    assert(elpis_ecsg_executor_epoch(e) == 0u && elpis_ecsg_executor_generation(e) == 0u);

    /* Only one transaction at a time, and no growth under it. */
    {
        uint64_t other = 0u;
        assert(elpis_ecsg_executor_txn_begin(e, &other) == ELPIS_ECSG_EXEC_BUSY && other == 0u);
        assert(elpis_ecsg_executor_reserve(e, 2u * ROWS) == ELPIS_ECSG_EXEC_BUSY);
        assert(elpis_ecsg_executor_txn_commit(e, token + 1u, &t) == ELPIS_ECSG_EXEC_INVALID);
        assert(elpis_ecsg_executor_txn_commit(e, 0u, &t) == ELPIS_ECSG_EXEC_INVALID);
    }

    assert(elpis_ecsg_executor_txn_commit(e, token, &t) == ELPIS_ECSG_EXEC_OK);
    assert(t.epoch_before == 0u && t.epoch_after == 9u && t.steps == 9u);
    assert(t.generation_before == 0u && t.generation_after == 1u);
    assert(elpis_ecsg_executor_epoch(e) == 9u && elpis_ecsg_executor_generation(e) == 1u);
    assert_matches(e, s);
    /* The token is spent; the executor continues from the committed W. */
    assert(elpis_ecsg_executor_txn_forward(e, token, X, ROWS, out) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_learn(e, X, Y, ROWS, 0.002, 2u, &t) == ELPIS_ECSG_EXEC_OK);
    ref_steps(s, X, Y, ROWS, 2u);
    assert_matches(e, s);
    assert(elpis_ecsg_executor_reserve(e, 2u * ROWS) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_stats(e, &st) == ELPIS_ECSG_EXEC_OK);
    assert(st.txn_begins == 1u && st.commits == 2u && st.txn_aborts == 0u && st.stale_refusals == 0u);

    /* A transaction that learned nothing commits nothing. */
    assert(elpis_ecsg_executor_txn_begin(e, &token) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_txn_commit(e, token, &t) == ELPIS_ECSG_EXEC_OK);
    assert(t.steps == 0u && t.generation_after == t.generation_before && elpis_ecsg_executor_generation(e) == 2u);
    assert_matches(e, s);

    elpis_ecsg_executor_destroy(&e);
    elpis_ecsg_state_destroy(&s);
}

static void
test_stale_and_abort(void)
{
    double out[ROWS];
    uint8_t a[SNAP], b[SNAP];
    elpis_ecsg_executor *e = NULL;
    elpis_ecsg_state *s = NULL;
    elpis_ecsg_exec_transition t;
    elpis_ecsg_exec_stats st;
    uint64_t token = 0u, epoch = 0u;

    assert(elpis_ecsg_executor_create(DIM, WIDTH, ROWS, W0, &e) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_state_create(DIM, WIDTH, W0, &s) == ELPIS_ECSG_MATH_OK);

    /* 1. The source moved after the candidate learned: commit is refused. */
    assert(elpis_ecsg_executor_txn_begin(e, &token) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_txn_learn(e, token, X, Y, ROWS, 0.002, 3u, &t) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_learn(e, X2, Y2, ROWS, 0.002, 2u, &t) == ELPIS_ECSG_EXEC_OK);
    ref_steps(s, X2, Y2, ROWS, 2u);
    assert(elpis_ecsg_executor_txn_commit(e, token, &t) == ELPIS_ECSG_EXEC_STALE);
    assert(t.generation_after == t.generation_before && t.steps == 0u);
    assert_matches(e, s);  /* the direct learn stands; the candidate is gone */
    assert(elpis_ecsg_executor_txn_commit(e, token, &t) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_txn_abort(e, token) == ELPIS_ECSG_EXEC_OK);  /* no-op */

    /* 2. The source moved before the candidate learned: refused at once. */
    assert(elpis_ecsg_executor_txn_begin(e, &token) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_learn(e, X, Y, ROWS, 0.002, 1u, &t) == ELPIS_ECSG_EXEC_OK);
    ref_steps(s, X, Y, ROWS, 1u);
    assert(elpis_ecsg_executor_txn_learn(e, token, X, Y, ROWS, 0.002, 1u, &t) == ELPIS_ECSG_EXEC_STALE);
    assert_matches(e, s);

    /* 3. Readout of a stale (unlearned) candidate is refused too. */
    assert(elpis_ecsg_executor_txn_begin(e, &token) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_learn(e, X, Y, ROWS, 0.002, 1u, &t) == ELPIS_ECSG_EXEC_OK);
    ref_steps(s, X, Y, ROWS, 1u);
    assert(elpis_ecsg_executor_txn_forward(e, token, X, ROWS, out) == ELPIS_ECSG_EXEC_STALE);
    assert(elpis_ecsg_executor_txn_epoch(e, token, &epoch) == ELPIS_ECSG_EXEC_INVALID);

    /* 4. Abort discards the candidate. */
    snapshot(e, a);
    assert(elpis_ecsg_executor_txn_begin(e, &token) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_txn_learn(e, token, X, Y, ROWS, 0.002, 7u, &t) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_txn_abort(e, token) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_txn_commit(e, token, &t) == ELPIS_ECSG_EXEC_INVALID);
    snapshot(e, b);
    assert(memcmp(a, b, SNAP) == 0);
    assert(elpis_ecsg_executor_txn_abort(e, 0u) == ELPIS_ECSG_EXEC_INVALID);

    /* 5. A refused transaction learn (after >= 1 step) discards the transaction. */
    {
        double big_y[ROWS];
        size_t i;
        for (i = 0u; i < ROWS; ++i) {
            big_y[i] = 1e3;
        }
        assert(elpis_ecsg_executor_txn_begin(e, &token) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_txn_learn(e, token, X, Y, ROWS, 0.002, 2u, &t) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_txn_learn(e, token, X, big_y, ROWS, 50.0, 500u, &t) == ELPIS_ECSG_EXEC_NONFINITE);
        assert(t.failed_step >= 2u);
        assert(elpis_ecsg_executor_txn_commit(e, token, &t) == ELPIS_ECSG_EXEC_INVALID);
        snapshot(e, b);
        assert(memcmp(a, b, SNAP) == 0);
    }
    assert(elpis_ecsg_executor_stats(e, &st) == ELPIS_ECSG_EXEC_OK);
    assert(st.txn_begins == 5u && st.stale_refusals == 3u && st.txn_aborts == 5u);

    /* 6. A destroyed executor with an open transaction is simply gone. */
    assert(elpis_ecsg_executor_txn_begin(e, &token) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_destroy(&e) == ELPIS_ECSG_EXEC_OK && e == NULL);
    assert(elpis_ecsg_executor_txn_commit(e, token, &t) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_txn_begin(e, &token) == ELPIS_ECSG_EXEC_INVALID);
    elpis_ecsg_state_destroy(&s);
}

/* SINGLE_WRITER: a call that overlaps a running call is refused with BUSY and
 * disturbs nothing. */
typedef struct {
    elpis_ecsg_executor *e;
    const double *x, *y;
    atomic_int started;
    atomic_int finished;
    int status;
    unsigned long worker_busy;
} worker_args;

enum { BUSY_WIDTH = 288, BUSY_ROWS = 64, BUSY_STEPS = 400 };

static void *
worker(void *p)
{
    worker_args *a = (worker_args *)p;
    atomic_store(&a->started, 1);
    /* The worker is refused too whenever the main thread got in first. */
    do {
        a->status = elpis_ecsg_executor_learn(a->e, a->x, a->y, BUSY_ROWS, 0.002, BUSY_STEPS, NULL);
        a->worker_busy += a->status == ELPIS_ECSG_EXEC_BUSY;
    } while (a->status == ELPIS_ECSG_EXEC_BUSY);
    atomic_store(&a->finished, 1);
    return NULL;
}

static void
test_single_writer_refuses_concurrent_entry(void)
{
    const size_t wc = DIM * BUSY_WIDTH;
    double *w0 = (double *)malloc(wc * sizeof(double));
    double *x = (double *)malloc(BUSY_ROWS * DIM * sizeof(double));
    double *y = (double *)malloc(BUSY_ROWS * sizeof(double));
    double *wa = (double *)malloc(wc * sizeof(double));
    double *wb = (double *)malloc(wc * sizeof(double));
    double out[BUSY_ROWS];
    unsigned long busy = 0u, attempt;
    elpis_ecsg_exec_stats st;

    assert(w0 && x && y && wa && wb);
    fill(w0, wc, 0.18 * sqrt(36.0 / BUSY_WIDTH));
    fill(x, BUSY_ROWS * DIM, 0.5);
    fill(y, BUSY_ROWS, 0.5);

    for (attempt = 0u; attempt < 5u && busy == 0u; ++attempt) {
        elpis_ecsg_executor *e = NULL;
        elpis_ecsg_executor *ref = NULL;
        worker_args args;
        pthread_t thread;
        elpis_ecsg_exec_stats seen;

        assert(elpis_ecsg_executor_create(DIM, BUSY_WIDTH, BUSY_ROWS, w0, &e) == ELPIS_ECSG_EXEC_OK);
        args.e = e;
        args.x = x;
        args.y = y;
        atomic_init(&args.started, 0);
        atomic_init(&args.finished, 0);
        args.status = 99;
        args.worker_busy = 0u;
        assert(pthread_create(&thread, NULL, worker, &args) == 0);
        while (!atomic_load(&args.started)) {
        }
        busy = 0u;
        while (!atomic_load(&args.finished)) {
            int rc = elpis_ecsg_executor_forward(e, x, BUSY_ROWS, out);
            assert(rc == ELPIS_ECSG_EXEC_OK || rc == ELPIS_ECSG_EXEC_BUSY);
            busy += rc == ELPIS_ECSG_EXEC_BUSY;
            rc = elpis_ecsg_executor_stats(e, &seen);
            assert(rc == ELPIS_ECSG_EXEC_OK || rc == ELPIS_ECSG_EXEC_BUSY);
            busy += rc == ELPIS_ECSG_EXEC_BUSY;
        }
        assert(pthread_join(thread, NULL) == 0);
        assert(args.status == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_stats(e, &st) == ELPIS_ECSG_EXEC_OK);
        assert(st.busy_refusals == busy + args.worker_busy && st.commits == 1u);
        /* The overlapped learn is exactly the undisturbed one. */
        assert(elpis_ecsg_executor_create(DIM, BUSY_WIDTH, BUSY_ROWS, w0, &ref) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_learn(ref, x, y, BUSY_ROWS, 0.002, BUSY_STEPS, NULL) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_copy_w(e, wa, wc) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_copy_w(ref, wb, wc) == ELPIS_ECSG_EXEC_OK);
        assert(memcmp(wa, wb, wc * sizeof(double)) == 0);
        assert(elpis_ecsg_executor_epoch(e) == BUSY_STEPS);
        elpis_ecsg_executor_destroy(&ref);
        elpis_ecsg_executor_destroy(&e);
    }
    assert(busy > 0u);
    free(w0);
    free(x);
    free(y);
    free(wa);
    free(wb);
}

int
main(void)
{
    fill(W0, WCOUNT, 0.18);
    fill(X, ROWS * DIM, 0.5);
    fill(Y, ROWS, 0.5);
    fill(X2, ROWS * DIM, 0.5);
    fill(Y2, ROWS, 0.5);
    test_commit_installs_candidate_and_matches_reference();
    test_stale_and_abort();
    test_single_writer_refuses_concurrent_entry();
    puts("test_ecsg_executor_txn: ok");
    return 0;
}
