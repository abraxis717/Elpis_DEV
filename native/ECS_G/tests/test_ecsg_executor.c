/*
 * ECS_G executor R1: bitwise parity with the scalar reference, atomic failure,
 * epoch/generation, read-only query, snapshot/restore and bounds.
 */
#include "elpis/ecsg_executor.h"
#include "elpis/ecsg_math.h"
#include "elpis/ecsg_state.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = UINT64_C(0x5eed00e1);

static uint64_t
next_u64(void)
{
    uint64_t z = (rng += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static double
gauss(double scale)
{
    const double u = ((double)(next_u64() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    const double v = ((double)(next_u64() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    return scale * sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}

static double *
filled(size_t count, double scale)
{
    double *v = (double *)malloc((count ? count : 1u) * sizeof(double));
    size_t i;
    assert(v != NULL);
    for (i = 0u; i < count; ++i) {
        v[i] = gauss(scale);
    }
    return v;
}

/* Reference: K steps on (x, y); returns the number of steps that succeeded. */
static uint64_t
ref_steps(elpis_ecsg_state *s, const double *x, const double *y, size_t rows, double lr, uint64_t k)
{
    const size_t count = elpis_ecsg_state_gd_step_scratch_f64(elpis_ecsg_state_dim(s), elpis_ecsg_state_width(s), rows);
    double *scratch = (double *)malloc(count * sizeof(double));
    uint64_t done = 0u;
    assert(scratch != NULL);
    while (done < k && elpis_ecsg_state_gd_step_f64(s, x, y, rows, lr, scratch, count) == ELPIS_ECSG_MATH_OK) {
        ++done;
    }
    free(scratch);
    return done;
}

static void
assert_same_state(elpis_ecsg_executor *e, const elpis_ecsg_state *s)
{
    const size_t count = elpis_ecsg_state_dim(s) * elpis_ecsg_state_width(s);
    double *we = (double *)malloc(count * sizeof(double));
    double *ws = (double *)malloc(count * sizeof(double));
    const size_t bytes = elpis_ecsg_state_snapshot_size(s);
    uint8_t *se = (uint8_t *)malloc(bytes);
    uint8_t *ss = (uint8_t *)malloc(bytes);
    assert(we && ws && se && ss);
    assert(elpis_ecsg_executor_copy_w(e, we, count) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_state_copy_w(s, ws, count) == ELPIS_ECSG_MATH_OK);
    assert(memcmp(we, ws, count * sizeof(double)) == 0);
    assert(elpis_ecsg_executor_epoch(e) == elpis_ecsg_state_epoch(s));
    assert(elpis_ecsg_executor_snapshot_size(e) == bytes);
    assert(elpis_ecsg_executor_snapshot_write(e, se, bytes) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_state_snapshot_write(s, ss, bytes) == ELPIS_ECSG_MATH_OK);
    assert(memcmp(se, ss, bytes) == 0);
    free(we);
    free(ws);
    free(se);
    free(ss);
}

static void
assert_forward_parity(elpis_ecsg_executor *e, const elpis_ecsg_state *s, size_t rows)
{
    const size_t dim = elpis_ecsg_state_dim(s);
    double *x = filled(rows * dim, 0.5);
    double *oe = (double *)malloc(rows * sizeof(double));
    double *os = (double *)malloc(rows * sizeof(double));
    assert(oe && os);
    assert(elpis_ecsg_executor_forward(e, x, rows, oe) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_state_forward_f64(s, x, rows, os) == ELPIS_ECSG_MATH_OK);
    assert(memcmp(oe, os, rows * sizeof(double)) == 0);
    free(x);
    free(oe);
    free(os);
}

static void
test_fused_learn_and_forward_are_bitwise_reference(void)
{
    static const size_t dims[] = {1u, 2u, 6u, 7u};
    static const size_t widths[] = {1u, 3u, 36u, 37u, 72u};
    static const size_t rowss[] = {1u, 5u, 64u};
    static const uint64_t ks[] = {1u, 2u, 10u, 57u};
    static const double lrs[] = {0.002, 0.0, 0.01};
    size_t di, wi, ri, ki;
    unsigned cases = 0u;

    for (di = 0u; di < sizeof(dims) / sizeof(dims[0]); ++di) {
        for (wi = 0u; wi < sizeof(widths) / sizeof(widths[0]); ++wi) {
            for (ri = 0u; ri < sizeof(rowss) / sizeof(rowss[0]); ++ri) {
                for (ki = 0u; ki < sizeof(ks) / sizeof(ks[0]); ++ki) {
                    const size_t dim = dims[di], width = widths[wi], rows = rowss[ri];
                    const uint64_t k = ks[ki];
                    const double lr = lrs[(di + wi + ri + ki) % 3u];
                    double *w0 = filled(dim * width, 0.18 * sqrt(36.0 / (double)width));
                    double *x = filled(rows * dim, 0.5);
                    double *y = filled(rows, 0.5);
                    elpis_ecsg_state *s = NULL;
                    elpis_ecsg_executor *e = NULL;
                    elpis_ecsg_exec_transition t;

                    assert(elpis_ecsg_state_create(dim, width, w0, &s) == ELPIS_ECSG_MATH_OK);
                    assert(elpis_ecsg_executor_create(dim, width, rows, w0, &e) == ELPIS_ECSG_EXEC_OK);
                    assert_forward_parity(e, s, rows);
                    if (ref_steps(s, x, y, rows, lr, k) == k) {
                        assert(elpis_ecsg_executor_learn(e, x, y, rows, lr, k, &t) == ELPIS_ECSG_EXEC_OK);
                        assert(t.epoch_before == 0u && t.epoch_after == k && t.steps == k);
                        assert(t.generation_before == 0u && t.generation_after == 1u && t.failed_step == 0u);
                        assert(elpis_ecsg_executor_generation(e) == 1u);
                        assert_same_state(e, s);
                        assert_forward_parity(e, s, rows + 3u);
                        /* A second transition continues from the committed W. */
                        assert(ref_steps(s, x, y, rows, lr, 2u) == 2u);
                        assert(elpis_ecsg_executor_learn(e, x, y, rows, lr, 2u, &t) == ELPIS_ECSG_EXEC_OK);
                        assert(t.epoch_before == k && t.epoch_after == k + 2u && t.generation_after == 2u);
                        assert_same_state(e, s);
                        ++cases;
                    }
                    elpis_ecsg_executor_destroy(&e);
                    elpis_ecsg_state_destroy(&s);
                    free(w0);
                    free(x);
                    free(y);
                }
            }
        }
    }
    assert(cases >= 200u);
}

static void
test_schedule_is_drives_in_order(void)
{
    const size_t dim = 6u, width = 36u;
    const elpis_ecsg_drive drives[] = {{5u, 3u}, {2u, 1u}, {7u, 4u}};
    double *w0 = filled(dim * width, 0.18);
    double *x = filled(14u * dim, 0.5);
    double *y = filled(14u, 0.5);
    elpis_ecsg_state *s = NULL;
    elpis_ecsg_executor *e = NULL;
    elpis_ecsg_exec_transition t;

    assert(elpis_ecsg_state_create(dim, width, w0, &s) == ELPIS_ECSG_MATH_OK);
    assert(elpis_ecsg_executor_create(dim, width, 14u, w0, &e) == ELPIS_ECSG_EXEC_OK);
    assert(ref_steps(s, x, y, 5u, 0.002, 3u) == 3u);
    assert(ref_steps(s, x + 5u * dim, y + 5u, 2u, 0.002, 1u) == 1u);
    assert(ref_steps(s, x + 7u * dim, y + 7u, 7u, 0.002, 4u) == 4u);
    assert(elpis_ecsg_executor_learn_schedule(e, x, y, drives, 3u, 0.002, &t) == ELPIS_ECSG_EXEC_OK);
    assert(t.steps == 8u && t.epoch_after == 8u && t.generation_after == 1u);
    assert_same_state(e, s);
    elpis_ecsg_executor_destroy(&e);
    elpis_ecsg_state_destroy(&s);
    free(w0);
    free(x);
    free(y);
}

static void
test_failure_after_steps_leaves_state_unchanged(void)
{
    static const double lrs[] = {0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0};
    const size_t dim = 2u, width = 3u, rows = 4u;
    double *w0 = filled(dim * width, 1.0);
    double *x = filled(rows * dim, 1.0);
    double *y = filled(rows, 1.0);
    double before[6], after[6];
    elpis_ecsg_exec_stats st0, st1;
    elpis_ecsg_exec_transition t;
    elpis_ecsg_executor *e = NULL;
    elpis_ecsg_state *s = NULL;
    uint64_t fail_at = 0u;
    double lr = 0.0;
    size_t i;

    /* Find a rate whose reference trajectory refuses at step >= 2. */
    for (i = 0u; i < sizeof(lrs) / sizeof(lrs[0]) && fail_at == 0u; ++i) {
        uint64_t ok;
        assert(elpis_ecsg_state_create(dim, width, w0, &s) == ELPIS_ECSG_MATH_OK);
        ok = ref_steps(s, x, y, rows, lrs[i], 1000u);
        if (ok >= 1u && ok < 1000u) {
            fail_at = ok + 1u;
            lr = lrs[i];
        }
        elpis_ecsg_state_destroy(&s);
    }
    assert(fail_at >= 2u);

    assert(elpis_ecsg_executor_create(dim, width, rows, w0, &e) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_learn(e, x, y, rows, 0.001, 1u, NULL) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_copy_w(e, before, 6u) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_stats(e, &st0) == ELPIS_ECSG_EXEC_OK);
    /* The same refusal from the committed state: recompute where it fails. */
    assert(elpis_ecsg_state_create(dim, width, before, &s) == ELPIS_ECSG_MATH_OK);
    fail_at = ref_steps(s, x, y, rows, lr, 1000u) + 1u;
    elpis_ecsg_state_destroy(&s);
    assert(fail_at >= 2u && fail_at <= 1000u);

    assert(elpis_ecsg_executor_learn(e, x, y, rows, lr, fail_at + 5u, &t) == ELPIS_ECSG_EXEC_NONFINITE);
    assert(t.failed_step == fail_at && t.steps == 0u);
    assert(t.epoch_before == 1u && t.epoch_after == 1u && t.generation_after == t.generation_before);
    assert(elpis_ecsg_executor_copy_w(e, after, 6u) == ELPIS_ECSG_EXEC_OK);
    assert(memcmp(before, after, sizeof(before)) == 0);
    assert(elpis_ecsg_executor_epoch(e) == 1u && elpis_ecsg_executor_generation(e) == 1u);
    assert(elpis_ecsg_executor_stats(e, &st1) == ELPIS_ECSG_EXEC_OK);
    assert(st1.commits == st0.commits && st1.refusals == st0.refusals + 1u);
    assert(st1.steps_executed == st0.steps_executed + fail_at);

    /* Still usable, from the untouched authoritative W: fail_at - 1 steps match the reference. */
    assert(elpis_ecsg_state_create(dim, width, before, &s) == ELPIS_ECSG_MATH_OK);
    assert(ref_steps(s, x, y, rows, lr, fail_at - 1u) == fail_at - 1u);
    assert(elpis_ecsg_executor_learn(e, x, y, rows, lr, fail_at - 1u, &t) == ELPIS_ECSG_EXEC_OK);
    assert(t.epoch_after == fail_at && elpis_ecsg_state_epoch(s) == fail_at - 1u);
    {
        double w[6], r[6];
        assert(elpis_ecsg_executor_copy_w(e, w, 6u) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_state_copy_w(s, r, 6u) == ELPIS_ECSG_MATH_OK);
        assert(memcmp(w, r, sizeof(w)) == 0);
    }
    elpis_ecsg_state_destroy(&s);
    elpis_ecsg_executor_destroy(&e);
    free(w0);
    free(x);
    free(y);
}

static void
test_query_is_read_only(void)
{
    const size_t dim = 6u, width = 36u, rows = 64u;
    double *w0 = filled(dim * width, 0.18);
    double *x = filled(rows * dim, 0.5);
    double *y = filled(rows, 0.5);
    double out[64];
    uint8_t a[40 + 6 * 36 * 8], b[40 + 6 * 36 * 8];
    elpis_ecsg_executor *e = NULL;
    elpis_ecsg_exec_stats st;
    int i;

    assert(elpis_ecsg_executor_create(dim, width, rows, w0, &e) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_learn(e, x, y, rows, 0.002, 3u, NULL) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_snapshot_write(e, a, sizeof(a)) == ELPIS_ECSG_EXEC_OK);
    for (i = 0; i < 100; ++i) {
        assert(elpis_ecsg_executor_forward(e, x, rows, out) == ELPIS_ECSG_EXEC_OK);
    }
    assert(elpis_ecsg_executor_snapshot_write(e, b, sizeof(b)) == ELPIS_ECSG_EXEC_OK);
    assert(memcmp(a, b, sizeof(a)) == 0);
    assert(elpis_ecsg_executor_stats(e, &st) == ELPIS_ECSG_EXEC_OK);
    assert(st.forward_calls == 100u);
    /* A refused query changes nothing either. */
    x[7] = NAN;
    assert(elpis_ecsg_executor_forward(e, x, rows, out) == ELPIS_ECSG_EXEC_NONFINITE);
    assert(elpis_ecsg_executor_snapshot_write(e, b, sizeof(b)) == ELPIS_ECSG_EXEC_OK);
    assert(memcmp(a, b, sizeof(a)) == 0);
    assert(elpis_ecsg_executor_epoch(e) == 3u && elpis_ecsg_executor_generation(e) == 1u);
    elpis_ecsg_executor_destroy(&e);
    free(w0);
    free(x);
    free(y);
}

static void
test_snapshot_restore_round_trip(void)
{
    const size_t dim = 6u, width = 37u, rows = 9u;
    double *w0 = filled(dim * width, 0.18);
    double *x = filled(rows * dim, 0.5);
    double *y = filled(rows, 0.5);
    elpis_ecsg_state *s = NULL;
    elpis_ecsg_executor *e = NULL;
    size_t bytes;
    uint8_t *snap;

    assert(elpis_ecsg_state_create(dim, width, w0, &s) == ELPIS_ECSG_MATH_OK);
    assert(ref_steps(s, x, y, rows, 0.002, 11u) == 11u);
    bytes = elpis_ecsg_state_snapshot_size(s);
    snap = (uint8_t *)malloc(bytes);
    assert(snap != NULL);
    assert(elpis_ecsg_state_snapshot_write(s, snap, bytes) == ELPIS_ECSG_MATH_OK);
    assert(elpis_ecsg_executor_restore(snap, bytes, rows, &e) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_epoch(e) == 11u && elpis_ecsg_executor_generation(e) == 0u);
    assert_same_state(e, s);
    assert(ref_steps(s, x, y, rows, 0.002, 4u) == 4u);
    assert(elpis_ecsg_executor_learn(e, x, y, rows, 0.002, 4u, NULL) == ELPIS_ECSG_EXEC_OK);
    assert_same_state(e, s);
    elpis_ecsg_executor_destroy(&e);

    /* The reference restore rules apply unchanged. */
    assert(elpis_ecsg_executor_restore(snap, bytes - 1u, rows, &e) == ELPIS_ECSG_EXEC_INVALID && e == NULL);
    assert(elpis_ecsg_executor_restore(snap, bytes, 0u, &e) == ELPIS_ECSG_EXEC_INVALID && e == NULL);
    snap[0] ^= 1u;
    assert(elpis_ecsg_executor_restore(snap, bytes, rows, &e) == ELPIS_ECSG_EXEC_INVALID && e == NULL);
    snap[0] ^= 1u;
    {
        const double inf = INFINITY;
        uint64_t bits;
        unsigned b;
        memcpy(&bits, &inf, sizeof(bits));
        for (b = 0u; b < 8u; ++b) {
            snap[40u + b] = (uint8_t)(bits >> (8u * b));
        }
    }
    assert(elpis_ecsg_executor_restore(snap, bytes, rows, &e) == ELPIS_ECSG_EXEC_NONFINITE && e == NULL);
    free(snap);
    elpis_ecsg_state_destroy(&s);
    free(w0);
    free(x);
    free(y);
}

static void
test_bounds(void)
{
    const size_t dim = 6u, width = 36u;
    double *w0 = filled(dim * width, 0.18);
    double *x = filled(65u * dim, 0.5);
    double *y = filled(65u, 0.5);
    double w_before[216], w_after[216];
    elpis_ecsg_executor *e = NULL;
    elpis_ecsg_exec_transition t;
    elpis_ecsg_drive drives[2] = {{40u, 1u}, {25u, 1u}};
    elpis_ecsg_exec_stats st;
    double out[65];

    /* Shape and overflow. */
    assert(elpis_ecsg_executor_workspace_bytes(0u, 1u, 1u) == 0u);
    assert(elpis_ecsg_executor_workspace_bytes(1u, 1u, 0u) == 0u);
    assert(elpis_ecsg_executor_workspace_bytes(SIZE_MAX / 2u, 3u, 1u) == 0u);
    assert(elpis_ecsg_executor_workspace_bytes(6u, 36u, SIZE_MAX / 8u) == 0u);
    assert(elpis_ecsg_executor_create(SIZE_MAX / 2u, 3u, 1u, w0, &e) == ELPIS_ECSG_EXEC_INVALID && e == NULL);
    assert(elpis_ecsg_executor_create(6u, 36u, 0u, w0, &e) == ELPIS_ECSG_EXEC_INVALID && e == NULL);
    assert(elpis_ecsg_executor_create(6u, 36u, 1u, NULL, &e) == ELPIS_ECSG_EXEC_INVALID && e == NULL);
    assert(elpis_ecsg_executor_create(6u, 36u, 1u, w0, NULL) == ELPIS_ECSG_EXEC_INVALID);
    w0[5] = INFINITY;
    assert(elpis_ecsg_executor_create(6u, 36u, 1u, w0, &e) == ELPIS_ECSG_EXEC_NONFINITE && e == NULL);
    w0[5] = 0.1;
    assert(elpis_ecsg_executor_workspace_bytes(6u, 36u, 64u) % 64u == 0u);

    assert(elpis_ecsg_executor_create(dim, width, 64u, w0, &e) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_max_rows(e) == 64u);
    assert(elpis_ecsg_executor_stats(e, &st) == ELPIS_ECSG_EXEC_OK);
    assert(st.workspace_bytes == elpis_ecsg_executor_workspace_bytes(dim, width, 64u));
    assert(st.heap_allocations == 2u);
    assert(elpis_ecsg_executor_copy_w(e, w_before, 216u) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_copy_w(e, w_before, 215u) == ELPIS_ECSG_EXEC_INVALID);

    /* Row capacity: refused, nothing changes; queries are not capacity-bound. */
    assert(elpis_ecsg_executor_learn(e, x, y, 65u, 0.002, 1u, &t) == ELPIS_ECSG_EXEC_CAPACITY);
    assert(elpis_ecsg_executor_learn_schedule(e, x, y, drives, 2u, 0.002, &t) == ELPIS_ECSG_EXEC_CAPACITY);
    assert(elpis_ecsg_executor_forward(e, x, 65u, out) == ELPIS_ECSG_EXEC_OK);

    /* Step count, rate, rows and pointers. */
    assert(elpis_ecsg_executor_learn(e, x, y, 4u, 0.002, 0u, &t) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_learn(e, x, y, 0u, 0.002, 1u, &t) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_learn(e, x, y, 4u, -0.002, 1u, &t) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_learn(e, x, y, 4u, NAN, 1u, &t) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_learn(e, x, y, 4u, INFINITY, 1u, &t) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_learn(e, NULL, y, 4u, 0.002, 1u, &t) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_learn(e, x, NULL, 4u, 0.002, 1u, &t) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_learn_schedule(e, x, y, NULL, 1u, 0.002, &t) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_learn_schedule(e, x, y, drives, 0u, 0.002, &t) == ELPIS_ECSG_EXEC_INVALID);
    drives[0].rows = 0u;
    assert(elpis_ecsg_executor_learn_schedule(e, x, y, drives, 2u, 0.002, &t) == ELPIS_ECSG_EXEC_INVALID);
    drives[0].rows = 4u;
    drives[1].steps = UINT64_MAX;
    assert(elpis_ecsg_executor_learn_schedule(e, x, y, drives, 2u, 0.002, &t) == ELPIS_ECSG_EXEC_INVALID);
    drives[1].steps = 1u;

    /* Non-finite experience: refused at admission, before any step. */
    x[3] = NAN;
    assert(elpis_ecsg_executor_learn(e, x, y, 4u, 0.002, 5u, &t) == ELPIS_ECSG_EXEC_NONFINITE);
    assert(t.failed_step == 0u && t.steps == 0u);
    x[3] = 0.25;
    y[2] = -INFINITY;
    assert(elpis_ecsg_executor_learn(e, x, y, 4u, 0.002, 5u, &t) == ELPIS_ECSG_EXEC_NONFINITE);
    y[2] = 0.25;
    assert(elpis_ecsg_executor_copy_w(e, w_after, 216u) == ELPIS_ECSG_EXEC_OK);
    assert(memcmp(w_before, w_after, sizeof(w_before)) == 0);
    assert(elpis_ecsg_executor_epoch(e) == 0u && elpis_ecsg_executor_generation(e) == 0u);
    assert(elpis_ecsg_executor_stats(e, &st) == ELPIS_ECSG_EXEC_OK);
    assert(st.commits == 0u && st.steps_executed == 0u);

    /* Explicit cold-path growth keeps W and admits more rows. */
    assert(elpis_ecsg_executor_reserve(e, 32u) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_max_rows(e) == 64u);
    assert(elpis_ecsg_executor_reserve(e, 65u) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_reserve(e, SIZE_MAX / 4u) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_max_rows(e) == 65u);
    assert(elpis_ecsg_executor_copy_w(e, w_after, 216u) == ELPIS_ECSG_EXEC_OK);
    assert(memcmp(w_before, w_after, sizeof(w_before)) == 0);
    assert(elpis_ecsg_executor_learn(e, x, y, 65u, 0.002, 1u, &t) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_stats(e, &st) == ELPIS_ECSG_EXEC_OK);
    assert(st.heap_allocations == 3u && st.max_rows == 65u);
    assert(st.workspace_bytes == elpis_ecsg_executor_workspace_bytes(dim, width, 65u));

    /* Destroyed state: the handle is cleared and every call refuses NULL. */
    assert(elpis_ecsg_executor_destroy(&e) == ELPIS_ECSG_EXEC_OK && e == NULL);
    assert(elpis_ecsg_executor_destroy(&e) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_destroy(NULL) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_forward(e, x, 4u, out) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_learn(e, x, y, 4u, 0.002, 1u, &t) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_copy_w(e, w_after, 216u) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_snapshot_size(e) == 0u);
    assert(elpis_ecsg_executor_stats(e, &st) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_reserve(e, 8u) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_epoch(e) == 0u && elpis_ecsg_executor_dim(e) == 0u);
    free(w0);
    free(x);
    free(y);
}

static void
test_epoch_headroom_matches_reference(void)
{
    const size_t dim = 2u, width = 2u;
    const double w[4] = {0.1, -0.2, 0.3, 0.05};
    const double x[2] = {0.5, -0.5};
    const double y[1] = {0.01};
    uint8_t snap[40 + 4 * 8];
    elpis_ecsg_state *s = NULL;
    elpis_ecsg_executor *e = NULL;
    elpis_ecsg_exec_transition t;
    unsigned b;

    assert(elpis_ecsg_state_create(dim, width, w, &s) == ELPIS_ECSG_MATH_OK);
    assert(elpis_ecsg_state_snapshot_write(s, snap, sizeof(snap)) == ELPIS_ECSG_MATH_OK);
    elpis_ecsg_state_destroy(&s);
    for (b = 0u; b < 8u; ++b) {
        snap[32u + b] = (uint8_t)((UINT64_MAX - 2u) >> (8u * b));
    }
    assert(elpis_ecsg_executor_restore(snap, sizeof(snap), 1u, &e) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_learn(e, x, y, 1u, 0.002, 3u, &t) == ELPIS_ECSG_EXEC_INVALID);
    assert(elpis_ecsg_executor_epoch(e) == UINT64_MAX - 2u);
    assert(elpis_ecsg_executor_learn(e, x, y, 1u, 0.002, 2u, &t) == ELPIS_ECSG_EXEC_OK);
    assert(elpis_ecsg_executor_epoch(e) == UINT64_MAX);
    assert(elpis_ecsg_executor_learn(e, x, y, 1u, 0.002, 1u, &t) == ELPIS_ECSG_EXEC_INVALID);
    /* The reference refuses the same step. */
    assert(elpis_ecsg_state_snapshot_restore(snap, sizeof(snap), &s) == ELPIS_ECSG_MATH_OK);
    assert(ref_steps(s, x, y, 1u, 0.002, 3u) == 2u);
    assert_same_state(e, s);
    elpis_ecsg_state_destroy(&s);
    elpis_ecsg_executor_destroy(&e);
}

int
main(void)
{
    assert(elpis_ecsg_executor_abi_version() == ELPIS_ECSG_EXECUTOR_ABI_V1);
    test_fused_learn_and_forward_are_bitwise_reference();
    test_schedule_is_drives_in_order();
    test_failure_after_steps_leaves_state_unchanged();
    test_query_is_read_only();
    test_snapshot_restore_round_trip();
    test_bounds();
    test_epoch_headroom_matches_reference();
    puts("test_ecsg_executor: ok");
    return 0;
}
