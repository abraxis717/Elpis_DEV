/*
 * Mechanical tests of the native relational LAO adapter (ctest label "research"). With RLAO_WRAP_ALLOC the binary
 * is linked with --wrap=malloc/calloc/realloc/free and proves that no adapter call allocates. Latency is printed,
 * never asserted (docs/CI_POLICY.md). Bitwise parity with the Python scalar reference is the Python suite's.
 */
/* POSIX clock_gettime for the printed latency measurement only (the adapter itself is ISO C11). */
#define _POSIX_C_SOURCE 200809L

#include "rlao.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef RLAO_WRAP_ALLOC
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

#define CHECK(x) assert(x)

static uint64_t lcg_state = 88172645463325252ull;
static double unit(void) {
    lcg_state = lcg_state * 6364136223846793005ull + 1442695040888963407ull;
    return (double)(lcg_state >> 11) * (1.0 / 9007199254740992.0);
}

static rlao_config h3(uint32_t readout) {
    rlao_config cfg;
    const double q[3] = {0.0, 0.4, 0.48}, rho[3] = {0, 0, 0}, prior[3] = {1, 1, 1};
    CHECK(rlao_config_init(&cfg, 3, q, rho, prior, 8, 1.0, readout) == RLAO_OK);
    return cfg;
}

static int same_belief(const rlao_state *a, const rlao_state *b) {
    return !memcmp(a->mu, b->mu, sizeof a->mu) && !memcmp(a->v, b->v, sizeof a->v) &&
           !memcmp(a->lam, b->lam, sizeof a->lam) && !memcmp(a->counts, b->counts, sizeof a->counts);
}

typedef struct {
    uint32_t slot, n;
    uint8_t reads[2];
    double z;
} ev;

static void events(ev *out, int n) {
    for (int t = 0; t < n; ++t) {
        out[t].slot = (uint32_t)(unit() * 8.0) % 8u;
        out[t].n = t < 4 ? 2u : 1u;
        out[t].reads[0] = (uint8_t)(unit() < 0.5);
        out[t].reads[1] = (uint8_t)(unit() < 0.5);
        out[t].z = 4.0 * unit() - 2.0;
    }
}

static void test_layout_and_configuration(void) {
    CHECK(rlao_state_size() == 1016u && rlao_config_size() == 128u);
    rlao_config cfg = h3(RLAO_READOUT_COUPLED);
    CHECK(rlao_logical_scalars(&cfg) == 80u);
    const double q[2] = {0.0, 0.5}, bad_rho[2] = {0, 1.5}, rho[2] = {0, 0}, prior[2] = {1, 1}, zero[2] = {1, 0};
    rlao_config c;
    CHECK(rlao_config_init(&c, 2, q, rho, prior, 8, 1.0, 0) == RLAO_E_CONFIG);          /* q must be < 0.5 */
    const double q2[2] = {0.0, 0.4};
    CHECK(rlao_config_init(&c, 2, q2, bad_rho, prior, 8, 1.0, 0) == RLAO_E_CONFIG);
    CHECK(rlao_config_init(&c, 2, q2, rho, zero, 8, 1.0, 0) == RLAO_E_CONFIG);
    CHECK(rlao_config_init(&c, 2, q2, rho, prior, 9, 1.0, 0) == RLAO_E_CONFIG);
    CHECK(rlao_config_init(&c, 5, q2, rho, prior, 8, 1.0, 0) == RLAO_E_CONFIG);
    CHECK(rlao_config_init(&c, 2, q2, rho, prior, 8, 1.0, 3) == RLAO_E_CONFIG);
    CHECK(rlao_config_init(NULL, 2, q2, rho, prior, 8, 1.0, 0) == RLAO_E_NULL);
}

static void test_capacity_is_refused_and_nothing_changes(void) {
    rlao_config cfg = h3(RLAO_READOUT_COUPLED);
    rlao_state st, before;
    CHECK(rlao_reset(&st, &cfg) == RLAO_OK);
    ev e[8];
    events(e, 8);
    for (int t = 0; t < 8; ++t) CHECK(rlao_observe(&st, &cfg, e[t].slot, e[t].z, e[t].reads, e[t].n, (uint32_t)t, 1) == 0);
    before = st;
    const uint8_t one[1] = {1};
    CHECK(rlao_observe(&st, &cfg, 0, 0.1, one, 1, 8, 1) == RLAO_E_CAPACITY);
    CHECK(!memcmp(&st, &before, sizeof st));
    const uint8_t bad[1] = {2};
    CHECK(rlao_observe(&st, &cfg, 0, 0.1, bad, 1, 8, 1) == RLAO_E_INPUT);
    CHECK(rlao_observe(&st, &cfg, 8, 0.1, one, 1, 8, 1) == RLAO_E_INPUT);
    CHECK(rlao_observe(&st, &cfg, 0, NAN, one, 1, 8, 1) == RLAO_E_INPUT);
    CHECK(!memcmp(&st, &before, sizeof st));
}

static void test_rebuild_and_slot_reset_are_exact(void) {
    rlao_config cfg = h3(RLAO_READOUT_COUPLED);
    for (int trial = 0; trial < 200; ++trial) {
        ev e[8];
        events(e, 8);
        rlao_state full, again, fresh;
        CHECK(rlao_reset(&full, &cfg) == 0);
        for (int t = 0; t < 8; ++t) CHECK(rlao_observe(&full, &cfg, e[t].slot, e[t].z, e[t].reads, e[t].n, (uint32_t)t, 7) == 0);
        again = full;
        CHECK(rlao_rebuild(&again, &cfg, 0xFFu) == 0);                 /* replaying everything changes nothing */
        CHECK(same_belief(&again, &full));
        const uint32_t retired = (uint32_t)(unit() * 8.0) % 8u;         /* B: drop one event, rebuild */
        again = full;
        CHECK(rlao_rebuild(&again, &cfg, 0xFFu & ~(1u << retired)) == 0);
        CHECK(rlao_reset(&fresh, &cfg) == 0);
        for (int t = 0; t < 8; ++t)
            if ((uint32_t)t != retired)
                CHECK(rlao_observe(&fresh, &cfg, e[t].slot, e[t].z, e[t].reads, e[t].n, (uint32_t)t, 7) == 0);
        CHECK(same_belief(&again, &fresh));
        const uint32_t victim = e[retired].slot;                         /* A: reset a slot without replay */
        again = full;
        CHECK(rlao_reset_slot(&again, &cfg, victim) == 0);
        CHECK(rlao_reset(&fresh, &cfg) == 0);
        for (int t = 0; t < 8; ++t)
            if (e[t].slot != victim)
                CHECK(rlao_observe(&fresh, &cfg, e[t].slot, e[t].z, e[t].reads, e[t].n, (uint32_t)t, 7) == 0);
        CHECK(same_belief(&again, &fresh));
        CHECK(again.buffered == fresh.buffered);
        for (uint32_t i = 0; i < again.buffered; ++i) CHECK(again.buffer[i].slot != victim);
    }
}

static void test_weights_do_not_depend_on_slot_order(void) {
    rlao_config cfg = h3(RLAO_READOUT_COUPLED);
    const uint32_t perm[8] = {3, 7, 0, 5, 1, 6, 2, 4};
    for (int trial = 0; trial < 100; ++trial) {
        ev e[8];
        events(e, 8);
        rlao_state a, b;
        CHECK(rlao_reset(&a, &cfg) == 0 && rlao_reset(&b, &cfg) == 0);
        for (int t = 0; t < 8; ++t) {
            CHECK(rlao_observe(&a, &cfg, e[t].slot, e[t].z, e[t].reads, e[t].n, (uint32_t)t, 1) == 0);
            CHECK(rlao_observe(&b, &cfg, perm[e[t].slot], e[t].z, e[t].reads, e[t].n, (uint32_t)t, 1) == 0);
        }
        double wa[4], wb[4];
        CHECK(rlao_weights(&a, &cfg, wa) == 0 && rlao_weights(&b, &cfg, wb) == 0);
        CHECK(!memcmp(wa, wb, 3 * sizeof(double)));
    }
}

static void test_contradiction_and_readouts(void) {
    rlao_config cfg = h3(RLAO_READOUT_COUPLED);
    rlao_state st;
    CHECK(rlao_reset(&st, &cfg) == 0);
    double z[8] = {0}, w[4], u_c, u_m, u_i;
    rlao_alternative alt;
    memset(&alt, 0, sizeof alt);
    alt.paths = 2;
    alt.length[0] = 2, alt.slot[0][0] = 0, alt.slot[0][1] = 3;
    alt.length[1] = 2, alt.slot[1][0] = 1, alt.slot[1][1] = 4;
    rlao_config cm = h3(RLAO_READOUT_MARGINAL), ci = h3(RLAO_READOUT_INDEPENDENT);
    rlao_state sm, si;
    CHECK(rlao_reset(&sm, &cm) == 0 && rlao_reset(&si, &ci) == 0);
    CHECK(rlao_expected_utility(&st, &cfg, &alt, z, &u_c) == 0);
    CHECK(rlao_expected_utility(&sm, &cm, &alt, z, &u_m) == 0);
    CHECK(rlao_expected_utility(&si, &ci, &alt, z, &u_i) == 0);
    CHECK(fabs(u_c - u_m) < 1e-15 && fabs(u_m - u_i) < 1e-15);       /* at the prior every readout agrees */
    const uint8_t disagree[2] = {1, 0};
    CHECK(rlao_observe(&st, &cfg, 0, 0.0, disagree, 2, 0, 1) == 0);
    CHECK(rlao_weights(&st, &cfg, w) == 0);
    CHECK(w[0] == 0.0);                                               /* q = 0 is impossible after a disagreement */
    alt.slot[1][1] = 0;                                               /* a repeated relation is refused */
    CHECK(rlao_expected_utility(&st, &cfg, &alt, z, &u_c) == RLAO_E_INPUT);
}

static void test_binding_poisoning_and_zeroisation(void) {
    rlao_config cfg = h3(RLAO_READOUT_COUPLED), other = h3(RLAO_READOUT_MARGINAL);
    rlao_state st;
    CHECK(rlao_reset(&st, &cfg) == 0);
    double w[4];
    CHECK(rlao_weights(&st, &other, w) == RLAO_E_STATE);              /* a state is bound to its configuration */
    st.v[0][0] = NAN;                                                  /* simulated memory corruption */
    const uint8_t one[1] = {1};
    CHECK(rlao_observe(&st, &cfg, 0, 0.2, one, 1, 0, 1) == RLAO_E_NUMERIC);
    CHECK(rlao_weights(&st, &cfg, w) == RLAO_E_STATE);                 /* poisoned until reset */
    CHECK(rlao_reset(&st, &cfg) == 0 && rlao_weights(&st, &cfg, w) == 0);
    rlao_zeroize(&st);
    static const rlao_state zero;
    CHECK(!memcmp(&st, &zero, sizeof st));
    CHECK(rlao_weights(&st, &cfg, w) == RLAO_E_STATE);
    rlao_zeroize(NULL);
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void test_bounded_work_and_latency(void) {
    rlao_config cfg = h3(RLAO_READOUT_COUPLED);
    rlao_state st;
    double worst = 0.0, total = 0.0, dworst = 0.0, dtotal = 0.0;
    long n = 0, nd = 0;
    rlao_alternative alt;
    memset(&alt, 0, sizeof alt);
    alt.paths = 1, alt.length[0] = 2, alt.slot[0][0] = 2, alt.slot[0][1] = 5;
    ARM();
    for (int e = 0; e < 2000; ++e) {
        CHECK(rlao_reset(&st, &cfg) == 0);
        for (uint32_t t = 0; t < 8; ++t) {
            const uint8_t r[2] = {(uint8_t)(unit() < 0.5), (uint8_t)(unit() < 0.5)};
            const double t0 = now();
            CHECK(rlao_observe(&st, &cfg, t, 4.0 * unit() - 2.0, r, t < 4 ? 2u : 1u, t, 1) == 0);
            const double dt = now() - t0;
            total += dt, worst = dt > worst ? dt : worst, ++n;
        }
        for (int j = 0; j < 16; ++j) {
            double z[8], u;
            for (int r = 0; r < 8; ++r) z[r] = 4.0 * unit() - 2.0;
            const double t0 = now();
            CHECK(rlao_expected_utility(&st, &cfg, &alt, z, &u) == 0);
            const double dt = now() - t0;
            dtotal += dt, dworst = dt > dworst ? dt : dworst, ++nd;
        }
        CHECK(rlao_rebuild(&st, &cfg, 0x7Fu) == 0);
        rlao_zeroize(&st);
    }
    DISARM();
#ifdef RLAO_WRAP_ALLOC
    CHECK(allocations == 0);
    printf("allocation-free: %ld adapter calls, 0 allocations\n", n + nd + 4000);
#endif
    printf("state %zu bytes (float64, capacity %u hypotheses x %u slots, buffer %u), config %zu bytes, "
           "logical scalars %u\n", rlao_state_size(), RLAO_MAX_HYP, RLAO_MAX_SLOTS, RLAO_BUFFER, rlao_config_size(),
           rlao_logical_scalars(&cfg));
    printf("observe: mean %.3f us, worst %.3f us over %ld; 2-hop expected utility: mean %.3f us, worst %.3f us\n",
           1e6 * total / (double)n, 1e6 * worst, n, 1e6 * dtotal / (double)nd, 1e6 * dworst);
}

int main(void) {
    test_layout_and_configuration();
    test_capacity_is_refused_and_nothing_changes();
    test_rebuild_and_slot_reset_are_exact();
    test_weights_do_not_depend_on_slot_order();
    test_contradiction_and_readouts();
    test_binding_poisoning_and_zeroisation();
    test_bounded_work_and_latency();
    printf("rlao: all mechanical tests passed\n");
    return 0;
}
