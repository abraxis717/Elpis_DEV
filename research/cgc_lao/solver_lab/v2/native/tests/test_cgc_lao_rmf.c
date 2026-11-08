/*
 * Mechanical tests of the native CGC-LAO reliability-mixture learner (ctest label "research").
 * With CGC_LAO_RMF_WRAP_ALLOC the binary is linked with --wrap=malloc/calloc/realloc/free and proves that no
 * learner call allocates. Latency is printed, never asserted (docs/CI_POLICY.md).
 */
/* POSIX clock_gettime for the printed latency measurement only (the library itself is ISO C11). */
#define _POSIX_C_SOURCE 200809L

#include "cgc_lao_rmf.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef CGC_LAO_RMF_WRAP_ALLOC
#include <stdlib.h>
static int armed = 0;
static long allocations = 0;
void *__real_malloc(size_t n);
void *__real_calloc(size_t n, size_t s);
void *__real_realloc(void *p, size_t n);
void __real_free(void *p);
void *__wrap_malloc(size_t n) { allocations += armed; return __real_malloc(n); }
void *__wrap_calloc(size_t n, size_t s) { allocations += armed; return __real_calloc(n, s); }
void *__wrap_realloc(void *p, size_t n) { allocations += armed; return __real_realloc(p, n); }
void __wrap_free(void *p) { allocations += armed && p != NULL; __real_free(p); }
#define ARM() (armed = 1)
#define DISARM() (armed = 0)
#else
#define ARM() ((void)0)
#define DISARM() ((void)0)
#endif

/* The v2 primary hypothesis set: clean, channel 0.2, 0.4, 0.48, event-level 0.4; prior (1, 1, 1, 1, 0.25). */
static cgc_lao_rmf_config primary(void) {
    cgc_lao_rmf_config cfg;
    const uint32_t nodes[5] = {1, 1, 1, 1, 1};
    const double q[5 * CGC_LAO_RMF_MAX_NODES] = {0.0, 0, 0, 0, 0.2, 0, 0, 0, 0.4, 0, 0, 0, 0.48, 0, 0, 0, 0.4, 0, 0, 0};
    const double rho[5] = {0, 0, 0, 0, 1};
    const double prior[5] = {1, 1, 1, 1, 0.25};
    assert(cgc_lao_rmf_config_init(&cfg, 5, nodes, q, rho, prior, 1.0, 0.6, 8, CGC_LAO_RMF_PROBE_EARLY) ==
           CGC_LAO_RMF_OK);
    return cfg;
}

static uint64_t lcg_state = 88172645463325252ull;
static double unit(void) {
    lcg_state = lcg_state * 6364136223846793005ull + 1442695040888963407ull;
    return (double)(lcg_state >> 11) * (1.0 / 9007199254740992.0);
}

static void test_layout_and_config_validation(void) {
    assert(cgc_lao_rmf_state_size() == sizeof(cgc_lao_rmf_state));
    assert(sizeof(cgc_lao_rmf_state) == 480u);   /* 32 header bytes + 56 doubles; no padding */
    assert(offsetof(cgc_lao_rmf_state, log_weight) == 32u);
    assert(offsetof(cgc_lao_rmf_state, mean) == 96u && offsetof(cgc_lao_rmf_state, var) == 288u);
    assert(sizeof(cgc_lao_rmf_config) == 520u);
    cgc_lao_rmf_config cfg = primary();
    assert(cgc_lao_rmf_logical_scalars(&cfg) == 38u);
    const uint32_t nodes[2] = {1, 1};
    double q[2 * CGC_LAO_RMF_MAX_NODES] = {0};
    const double rho[2] = {0, 0};
    double prior[2] = {1, 1};
    cgc_lao_rmf_config bad;
    q[CGC_LAO_RMF_MAX_NODES] = 0.5;   /* q must be < 0.5 */
    assert(cgc_lao_rmf_config_init(&bad, 2, nodes, q, rho, prior, 1.0, 0.6, 8, 0) == CGC_LAO_RMF_E_CONFIG);
    q[CGC_LAO_RMF_MAX_NODES] = 0.2;
    prior[1] = 0.0;
    assert(cgc_lao_rmf_config_init(&bad, 2, nodes, q, rho, prior, 1.0, 0.6, 8, 0) == CGC_LAO_RMF_E_CONFIG);
    prior[1] = 1.0;
    assert(cgc_lao_rmf_config_init(&bad, 0, nodes, q, rho, prior, 1.0, 0.6, 8, 0) == CGC_LAO_RMF_E_CONFIG);
    assert(cgc_lao_rmf_config_init(&bad, 9, nodes, q, rho, prior, 1.0, 0.6, 8, 0) == CGC_LAO_RMF_E_CONFIG);
    assert(cgc_lao_rmf_config_init(&bad, 2, nodes, q, rho, prior, 1.0, 0.6, 0, 0) == CGC_LAO_RMF_E_CONFIG);
    assert(cgc_lao_rmf_config_init(&bad, 2, nodes, q, rho, prior, -1.0, 0.6, 8, 0) == CGC_LAO_RMF_E_CONFIG);
    assert(cgc_lao_rmf_config_init(&bad, 2, nodes, q, rho, prior, 1.0, 0.6, 8, 2) == CGC_LAO_RMF_E_CONFIG);
    assert(cgc_lao_rmf_config_init(NULL, 2, nodes, q, rho, prior, 1.0, 0.6, 8, 0) == CGC_LAO_RMF_E_NULL);
    cgc_lao_rmf_config tampered = cfg;
    tampered.rho[0] = 0.5;            /* a modified coefficient no longer matches the digest */
    cgc_lao_rmf_state st;
    assert(cgc_lao_rmf_reset(&st, &tampered) == CGC_LAO_RMF_E_CONFIG);
}

static void test_episode_invariants_and_fail_closed_inputs(void) {
    cgc_lao_rmf_config cfg = primary();
    cgc_lao_rmf_state st;
    assert(cgc_lao_rmf_reset(&st, &cfg) == CGC_LAO_RMF_OK);
    double w[CGC_LAO_RMF_MAX_HYP];
    const double z[3] = {0.3, -0.2, 1.1};
    uint32_t a = 99;
    for (uint32_t t = 0; t < 8; ++t) {
        assert(cgc_lao_rmf_propose(&st, &cfg, z, &a) == CGC_LAO_RMF_OK && a < 3u);
        const uint8_t reads[2] = {(uint8_t)(t & 1u), (uint8_t)(t & 1u)};
        assert(cgc_lao_rmf_observe(&st, &cfg, a, z[a], reads, t < 4 ? 2u : 1u, 1u) == CGC_LAO_RMF_OK);
        assert(cgc_lao_rmf_weights(&st, &cfg, w) == CGC_LAO_RMF_OK);
        double s = 0.0;
        for (uint32_t m = 0; m < 5; ++m) {
            assert(w[m] >= 0.0 && w[m] <= 1.0);
            s += w[m];
        }
        assert(fabs(s - 1.0) < 1e-12);
    }
    assert(st.t == 8u && st.counts[0] + st.counts[1] + st.counts[2] == 8u);
    assert(st.counts[0] > 0u && st.counts[1] > 0u && st.counts[2] > 0u);   /* coverage by the deadline */
    const uint8_t one[1] = {1};
    assert(cgc_lao_rmf_observe(&st, &cfg, 0, 0.1, one, 1, 1) == CGC_LAO_RMF_E_HORIZON);
    assert(cgc_lao_rmf_observe(&st, &cfg, 0, 0.1, one, 1, 0) == CGC_LAO_RMF_OK);   /* extra event: allowed */
    assert(cgc_lao_rmf_propose(&st, &cfg, z, &a) == CGC_LAO_RMF_E_HORIZON);
    uint32_t d = 99;
    assert(cgc_lao_rmf_decide(&st, &cfg, z, &d) == CGC_LAO_RMF_OK && d <= CGC_LAO_RMF_STOP);
    /* invalid inputs leave the state byte-for-byte unchanged */
    cgc_lao_rmf_state before = st;
    const uint8_t bad_read[1] = {2};
    assert(cgc_lao_rmf_observe(&st, &cfg, 3, 0.1, one, 1, 0) == CGC_LAO_RMF_E_INPUT);
    assert(cgc_lao_rmf_observe(&st, &cfg, 0, NAN, one, 1, 0) == CGC_LAO_RMF_E_INPUT);
    assert(cgc_lao_rmf_observe(&st, &cfg, 0, INFINITY, one, 1, 0) == CGC_LAO_RMF_E_INPUT);
    assert(cgc_lao_rmf_observe(&st, &cfg, 0, 0.1, bad_read, 1, 0) == CGC_LAO_RMF_E_INPUT);
    assert(cgc_lao_rmf_observe(&st, &cfg, 0, 0.1, one, 0, 0) == CGC_LAO_RMF_E_INPUT);
    assert(cgc_lao_rmf_observe(&st, &cfg, 0, 0.1, one, 3, 0) == CGC_LAO_RMF_E_INPUT);
    assert(cgc_lao_rmf_observe(&st, &cfg, 0, 0.1, one, 1, 2) == CGC_LAO_RMF_E_INPUT);
    const double znan[3] = {0.0, NAN, 0.0};
    assert(cgc_lao_rmf_decide(&st, &cfg, znan, &d) == CGC_LAO_RMF_E_INPUT);
    assert(memcmp(&before, &st, sizeof st) == 0);
}

static void test_contradictory_readings_eliminate_clean_and_event(void) {
    cgc_lao_rmf_config cfg = primary();
    cgc_lao_rmf_state st;
    assert(cgc_lao_rmf_reset(&st, &cfg) == CGC_LAO_RMF_OK);
    const uint8_t disagree[2] = {1, 0};
    assert(cgc_lao_rmf_observe(&st, &cfg, 1, 0.4, disagree, 2, 1) == CGC_LAO_RMF_OK);
    double w[CGC_LAO_RMF_MAX_HYP];
    assert(cgc_lao_rmf_weights(&st, &cfg, w) == CGC_LAO_RMF_OK);
    assert(w[0] == 0.0 && w[4] == 0.0);          /* clean and event-level hypotheses are impossible */
    assert(w[1] > 0.0 && w[2] > 0.0 && w[3] > 0.0);
}

static void test_extreme_logits_and_near_noise_channels(void) {
    const uint32_t nodes[3] = {1, 1, 3};
    const double q[3 * CGC_LAO_RMF_MAX_NODES] = {0.0, 0, 0, 0, 0.4999, 0, 0, 0, 0.49, 0.499, 0.4999, 0};
    const double rho[3] = {0, 0.5, 0};
    const double prior[3] = {1, 1, 1};
    cgc_lao_rmf_config cfg;
    assert(cgc_lao_rmf_config_init(&cfg, 3, nodes, q, rho, prior, 2.0, 0.6, 16, CGC_LAO_RMF_GREEDY) == CGC_LAO_RMF_OK);
    cgc_lao_rmf_state st;
    assert(cgc_lao_rmf_reset(&st, &cfg) == CGC_LAO_RMF_OK);
    const double zs[6] = {-745.0, 745.0, -1e6, 1e6, 0.0, -0.0};
    for (uint32_t i = 0; i < 6; ++i) {
        const uint8_t r[2] = {(uint8_t)(i & 1u), 1u};
        assert(cgc_lao_rmf_observe(&st, &cfg, i % 3u, zs[i], r, 1u + (i & 1u), 1u) == CGC_LAO_RMF_OK);
    }
    double p[3];
    const double zq[3] = {800.0, -800.0, 0.0};
    assert(cgc_lao_rmf_predictive(&st, &cfg, zq, p) == CGC_LAO_RMF_OK);
    assert(p[0] > 0.999 && p[1] < 0.001 && p[2] > 0.0 && p[2] < 1.0);
    for (uint32_t m = 0; m < 3; ++m) {
        for (uint32_t a = 0; a < 3; ++a) {
            assert(isfinite(st.mean[m][a]) && st.var[m][a] >= 1e-6 && isfinite(st.var[m][a]));
        }
    }
}

static void test_binding_poisoning_and_zeroisation(void) {
    cgc_lao_rmf_config cfg = primary();
    cgc_lao_rmf_config other = primary();
    const uint32_t nodes[1] = {1};
    const double q[CGC_LAO_RMF_MAX_NODES] = {0.3, 0, 0, 0};
    const double rho[1] = {0};
    const double prior[1] = {1};
    assert(cgc_lao_rmf_config_init(&other, 1, nodes, q, rho, prior, 1.0, 0.6, 8, 0) == CGC_LAO_RMF_OK);
    cgc_lao_rmf_state st;
    assert(cgc_lao_rmf_reset(&st, &cfg) == CGC_LAO_RMF_OK);
    double w[CGC_LAO_RMF_MAX_HYP];
    assert(cgc_lao_rmf_weights(&st, &other, w) == CGC_LAO_RMF_E_STATE);    /* a state is bound to its config */
    const uint8_t one[1] = {1};
    st.var[0][0] = NAN;                                                    /* simulated memory corruption */
    assert(cgc_lao_rmf_observe(&st, &cfg, 0, 0.2, one, 1, 1) == CGC_LAO_RMF_E_NUMERIC);
    assert(cgc_lao_rmf_weights(&st, &cfg, w) == CGC_LAO_RMF_E_STATE);      /* poisoned until reset */
    assert(cgc_lao_rmf_reset(&st, &cfg) == CGC_LAO_RMF_OK);
    assert(cgc_lao_rmf_weights(&st, &cfg, w) == CGC_LAO_RMF_OK);
    cgc_lao_rmf_zeroize(&st);
    static const cgc_lao_rmf_state zero;
    assert(memcmp(&st, &zero, sizeof st) == 0);
    assert(cgc_lao_rmf_weights(&st, &cfg, w) == CGC_LAO_RMF_E_STATE);
    cgc_lao_rmf_zeroize(NULL);
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void test_bounded_work_and_latency(void) {
    cgc_lao_rmf_config cfg = primary();
    cgc_lao_rmf_state st;
    const int episodes = 2000;
    double worst_update = 0.0, total_update = 0.0, worst_decide = 0.0, total_decide = 0.0;
    long updates = 0, decides = 0;
    ARM();
    for (int e = 0; e < episodes; ++e) {
        assert(cgc_lao_rmf_reset(&st, &cfg) == CGC_LAO_RMF_OK);
        for (uint32_t t = 0; t < 8; ++t) {
            const double z[3] = {4.0 * unit() - 2.0, 4.0 * unit() - 2.0, 4.0 * unit() - 2.0};
            uint32_t a = 0;
            assert(cgc_lao_rmf_propose(&st, &cfg, z, &a) == CGC_LAO_RMF_OK);
            const uint8_t r[2] = {(uint8_t)(unit() < 0.5), (uint8_t)(unit() < 0.5)};
            const double t0 = now();
            assert(cgc_lao_rmf_observe(&st, &cfg, a, z[a], r, t < 4 ? 2u : 1u, 1u) == CGC_LAO_RMF_OK);
            const double dt = now() - t0;
            total_update += dt;
            worst_update = dt > worst_update ? dt : worst_update;
            ++updates;
        }
        for (int j = 0; j < 16; ++j) {
            const double zq[3] = {4.0 * unit() - 2.0, 4.0 * unit() - 2.0, 4.0 * unit() - 2.0};
            uint32_t d = 0;
            const double t0 = now();
            assert(cgc_lao_rmf_decide(&st, &cfg, zq, &d) == CGC_LAO_RMF_OK);
            const double dt = now() - t0;
            total_decide += dt;
            worst_decide = dt > worst_decide ? dt : worst_decide;
            ++decides;
        }
        cgc_lao_rmf_zeroize(&st);
    }
    DISARM();
#ifdef CGC_LAO_RMF_WRAP_ALLOC
    assert(allocations == 0);
    printf("allocation-free: %ld learner calls, 0 allocations\n", updates + decides);
#endif
    printf("state %zu bytes (float64, capacity %u hypotheses), config %zu bytes, logical scalars %u\n",
           cgc_lao_rmf_state_size(), CGC_LAO_RMF_MAX_HYP, cgc_lao_rmf_config_size(), cgc_lao_rmf_logical_scalars(&cfg));
    printf("observe: mean %.3f us, worst %.3f us over %ld; decide: mean %.3f us, worst %.3f us over %ld\n",
           1e6 * total_update / (double)updates, 1e6 * worst_update, updates, 1e6 * total_decide / (double)decides,
           1e6 * worst_decide, decides);
}

int main(void) {
    test_layout_and_config_validation();
    test_episode_invariants_and_fail_closed_inputs();
    test_contradictory_readings_eliminate_clean_and_event();
    test_extreme_logits_and_near_noise_channels();
    test_binding_poisoning_and_zeroisation();
    test_bounded_work_and_latency();
    printf("cgc_lao_rmf: all mechanical tests passed\n");
    return 0;
}
