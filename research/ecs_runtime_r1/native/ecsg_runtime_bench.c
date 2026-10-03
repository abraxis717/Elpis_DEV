/*
 * ECS Runtime R1 direct-native benchmark driver. PERFORMANCE_ONLY.
 *
 * Times ECS_G operations called directly from C (no Python), one sample per
 * operation, after an explicit warmup, and prints one JSON object.
 *
 *   ecsg_runtime_bench <mode> <dim> <width> <rows> <steps> <budget_seconds>
 *
 * Modes:
 *   ref_forward  reference elpis_ecsg_state_forward_f64 on <rows> rows
 *   ref_step     one reference elpis_ecsg_state_gd_step_f64 (preallocated scratch)
 *   ref_learn    <steps> reference steps in a C loop on one state (preallocated scratch)
 *   ref_cold     reference state create + destroy
 *   exec_forward executor forward (one call)
 *   exec_learn   executor learn: <steps> fused steps, one call, one commit
 *   exec_txn     executor transaction: begin + learn(<steps>) + commit
 *   exec_cold    executor create (max_rows = rows) + destroy
 *   calibrate    a dependent floating-point chain of <steps> x 1000 iterations,
 *                no memory traffic or calls: its tail is the platform's own
 *                jitter (scheduling, interrupts, frequency), for comparison
 *
 * Data: deterministic (splitmix64 + Box-Muller); W0 ~ N(0, (0.18 sqrt(36/N))^2),
 * X ~ N(0, 0.5^2), y = 0.9 f_W0(X). Values do not change the work performed.
 */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "elpis/ecsg_executor.h"
#include "elpis/ecsg_state.h"

static uint64_t rng_state = UINT64_C(20261004);

static uint64_t splitmix64(void)
{
    uint64_t z = (rng_state += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static double uniform01(void)
{
    return ((double)(splitmix64() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

static double gauss(double scale)
{
    const double u = uniform01();
    const double v = uniform01();
    return scale * sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}

static uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}

static int cmp_u64(const void *a, const void *b)
{
    const uint64_t x = *(const uint64_t *)a;
    const uint64_t y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static uint64_t pct(const uint64_t *sorted, size_t n, double q)
{
    size_t index = (size_t)ceil(q * (double)n) - 1u;
    if (index >= n) {
        index = n - 1u;
    }
    return sorted[index];
}

typedef struct {
    size_t dim, width, rows, steps;
    double *w0, *x, *y, *out, *scratch;
    size_t scratch_count;
    elpis_ecsg_state *state;
    elpis_ecsg_executor *exec;
} bench;

static void die(const char *what)
{
    fprintf(stderr, "ecsg_runtime_bench: %s\n", what);
    exit(2);
}

static void setup(bench *b)
{
    const size_t wc = b->dim * b->width;
    const double ws = 0.18 * sqrt(36.0 / (double)b->width);
    size_t i;
    b->w0 = malloc(wc * sizeof(double));
    b->x = malloc(b->rows * b->dim * sizeof(double));
    b->y = malloc(b->rows * sizeof(double));
    b->out = malloc(b->rows * sizeof(double));
    if (!b->w0 || !b->x || !b->y || !b->out) {
        die("allocation");
    }
    for (i = 0; i < wc; ++i) {
        b->w0[i] = gauss(ws);
    }
    for (i = 0; i < b->rows * b->dim; ++i) {
        b->x[i] = gauss(0.5);
    }
    if (elpis_ecsg_state_create(b->dim, b->width, b->w0, &b->state) != ELPIS_ECSG_MATH_OK ||
        elpis_ecsg_state_forward_f64(b->state, b->x, b->rows, b->y) != ELPIS_ECSG_MATH_OK) {
        die("reference setup");
    }
    for (i = 0; i < b->rows; ++i) {
        b->y[i] *= 0.9;
    }
    b->scratch_count = elpis_ecsg_state_gd_step_scratch_f64(b->dim, b->width, b->rows);
    b->scratch = malloc(b->scratch_count * sizeof(double));
    if (!b->scratch) {
        die("scratch");
    }
    if (elpis_ecsg_executor_create(b->dim, b->width, b->rows, b->w0, &b->exec) != ELPIS_ECSG_EXEC_OK) {
        die("executor setup");
    }
}

static int run_once(bench *b, const char *mode)
{
    size_t k;
    if (strcmp(mode, "ref_forward") == 0) {
        return elpis_ecsg_state_forward_f64(b->state, b->x, b->rows, b->out);
    }
    if (strcmp(mode, "ref_step") == 0) {
        return elpis_ecsg_state_gd_step_f64(b->state, b->x, b->y, b->rows, 0.002, b->scratch, b->scratch_count);
    }
    if (strcmp(mode, "ref_learn") == 0) {
        for (k = 0; k < b->steps; ++k) {
            const int rc = elpis_ecsg_state_gd_step_f64(b->state, b->x, b->y, b->rows, 0.002, b->scratch,
                                                        b->scratch_count);
            if (rc != ELPIS_ECSG_MATH_OK) {
                return rc;
            }
        }
        return 0;
    }
    if (strcmp(mode, "ref_cold") == 0) {
        elpis_ecsg_state *s = NULL;
        const int rc = elpis_ecsg_state_create(b->dim, b->width, b->w0, &s);
        elpis_ecsg_state_destroy(&s);
        return rc;
    }
    if (strcmp(mode, "exec_forward") == 0) {
        return elpis_ecsg_executor_forward(b->exec, b->x, b->rows, b->out);
    }
    if (strcmp(mode, "exec_learn") == 0) {
        return elpis_ecsg_executor_learn(b->exec, b->x, b->y, b->rows, 0.002, b->steps, NULL);
    }
    if (strcmp(mode, "exec_txn") == 0) {
        uint64_t token = 0u;
        int rc = elpis_ecsg_executor_txn_begin(b->exec, &token);
        if (rc == ELPIS_ECSG_EXEC_OK) {
            rc = elpis_ecsg_executor_txn_learn(b->exec, token, b->x, b->y, b->rows, 0.002, b->steps, NULL);
        }
        if (rc == ELPIS_ECSG_EXEC_OK) {
            rc = elpis_ecsg_executor_txn_commit(b->exec, token, NULL);
        }
        return rc;
    }
    if (strcmp(mode, "calibrate") == 0) {
        volatile double sink;
        double acc = 1.0;
        const size_t n = b->steps * 1000u;
        for (k = 0; k < n; ++k) {
            acc = acc * 1.0000001 + 1e-9;
        }
        sink = acc;
        (void)sink;
        return 0;
    }
    if (strcmp(mode, "exec_cold") == 0) {
        elpis_ecsg_executor *e = NULL;
        const int rc = elpis_ecsg_executor_create(b->dim, b->width, b->rows, b->w0, &e);
        elpis_ecsg_executor_destroy(&e);
        return rc;
    }
    die("unknown mode");
    return -1;
}

int main(int argc, char **argv)
{
    bench b;
    uint64_t *samples;
    size_t n, warmup, i;
    double budget, estimate_ns, mean = 0.0;
    uint64_t t0, cold_ns;

    if (argc != 7) {
        die("usage: <mode> <dim> <width> <rows> <steps> <budget_seconds>");
    }
    memset(&b, 0, sizeof(b));
    b.dim = (size_t)strtoull(argv[2], NULL, 10);
    b.width = (size_t)strtoull(argv[3], NULL, 10);
    b.rows = (size_t)strtoull(argv[4], NULL, 10);
    b.steps = (size_t)strtoull(argv[5], NULL, 10);
    budget = strtod(argv[6], NULL);
    setup(&b);

    t0 = now_ns();
    if (run_once(&b, argv[1]) != 0) {
        die("first operation failed");
    }
    cold_ns = now_ns() - t0;
    estimate_ns = (double)cold_ns;
    t0 = now_ns();
    for (i = 0; i < 3; ++i) {
        if (run_once(&b, argv[1]) != 0) {
            die("operation failed");
        }
    }
    estimate_ns = (double)(now_ns() - t0) / 3.0;
    if (estimate_ns < 1.0) {
        estimate_ns = 1.0;
    }
    n = (size_t)(budget * 1e9 / estimate_ns);
    n = n < 20 ? 20 : (n > 2000 ? 2000 : n);
    warmup = (size_t)(0.2 * budget * 1e9 / estimate_ns);
    warmup = warmup < 3 ? 3 : (warmup > 200 ? 200 : warmup);
    for (i = 0; i < warmup; ++i) {
        if (run_once(&b, argv[1]) != 0) {
            die("warmup failed");
        }
    }
    samples = malloc(n * sizeof(uint64_t));
    if (!samples) {
        die("samples");
    }
    for (i = 0; i < n; ++i) {
        const uint64_t s = now_ns();
        if (run_once(&b, argv[1]) != 0) {
            die("sample failed");
        }
        samples[i] = now_ns() - s;
        mean += (double)samples[i];
    }
    mean /= (double)n;
    qsort(samples, n, sizeof(uint64_t), cmp_u64);
    printf("{\"mode\":\"%s\",\"dim\":%zu,\"width\":%zu,\"rows\":%zu,\"steps\":%zu,\"warmup\":%zu,\"samples\":%zu,"
           "\"first_call_ns\":%llu,\"min_ns\":%llu,\"p50_ns\":%llu,\"p95_ns\":%llu,\"p99_ns\":%llu,\"max_ns\":%llu,"
           "\"mean_ns\":%.1f}\n",
           argv[1], b.dim, b.width, b.rows, b.steps, warmup, n, (unsigned long long)cold_ns,
           (unsigned long long)samples[0], (unsigned long long)pct(samples, n, 0.50),
           (unsigned long long)pct(samples, n, 0.95), (unsigned long long)pct(samples, n, 0.99),
           (unsigned long long)samples[n - 1], mean);
    elpis_ecsg_state_destroy(&b.state);
    elpis_ecsg_executor_destroy(&b.exec);
    free(b.w0); free(b.x); free(b.y); free(b.out); free(b.scratch); free(samples);
    return 0;
}
