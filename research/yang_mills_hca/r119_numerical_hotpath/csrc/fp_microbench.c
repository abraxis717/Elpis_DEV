/* R119 forensic microbenchmark: per-operation cost of IEEE-754 subnormal operands/results.
 *
 * RESEARCH_ONLY. Compiled at runtime into volatile scratch (never committed) with
 * -std=c11 -O2 -ffp-contract=off -fno-tree-vectorize so that every timed operation is the
 * scalar SSE2 instruction (or libm call) that the AWR kernel executes. Inputs are prepared
 * by the caller; the routine applies one operation elementwise, `reps` times.
 */
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdint.h>
#include <time.h>
#if defined(__x86_64__) || defined(__SSE2__)
#include <xmmintrin.h>
#define MB_HAVE_MXCSR 1
#else
#define MB_HAVE_MXCSR 0
#endif

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* op: 0 mul, 1 add, 2 sub, 3 div, 4 sqrt, 5 exp, 6 tanh, 7 compare, 8 fabs-compare */
double fpmb_run(int32_t op, const double *a, const double *b, double *out, int32_t n, int32_t reps,
                int32_t flush) {
#if MB_HAVE_MXCSR
    const unsigned int csr = _mm_getcsr();
    if (flush) _mm_setcsr(csr | 0x8040u);
#else
    if (flush) return -1.0;
#endif
    const double t0 = now_ns();
    for (int32_t r = 0; r < reps; r++) {
        switch (op) {
        case 0: for (int32_t i = 0; i < n; i++) out[i] = a[i] * b[i]; break;
        case 1: for (int32_t i = 0; i < n; i++) out[i] = a[i] + b[i]; break;
        case 2: for (int32_t i = 0; i < n; i++) out[i] = a[i] - b[i]; break;
        case 3: for (int32_t i = 0; i < n; i++) out[i] = a[i] / b[i]; break;
        case 4: for (int32_t i = 0; i < n; i++) out[i] = sqrt(a[i]); break;
        case 5: for (int32_t i = 0; i < n; i++) out[i] = exp(a[i]); break;
        case 6: for (int32_t i = 0; i < n; i++) out[i] = tanh(a[i]); break;
        case 7: for (int32_t i = 0; i < n; i++) out[i] = (a[i] > b[i]) ? 1.0 : 0.0; break;
        case 8: for (int32_t i = 0; i < n; i++) out[i] = (fabs(a[i]) < b[i]) ? 1.0 : 0.0; break;
        default: break;
        }
        __asm__ volatile("" ::: "memory");
    }
    const double t1 = now_ns();
#if MB_HAVE_MXCSR
    if (flush) _mm_setcsr(csr);
#endif
    return (t1 - t0) / ((double)n * (double)reps);
}
