#include "elpis/ecsg_state.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int
near(double left, double right, double tolerance)
{
    return fabs(left - right) <= tolerance * (1.0 + fabs(right));
}

static void
test_known_one_dimensional_step(void)
{
    const double w0[] = {1.0};
    const double x[] = {2.0};
    const double y[] = {0.0};
    elpis_ecsg_state *state = NULL;
    size_t scratch_count;
    double *scratch;
    double w1[1];

    assert(elpis_ecsg_state_abi_version() == ELPIS_ECSG_STATE_ABI_V1);
    assert(elpis_ecsg_state_create(1u, 1u, w0, &state) ==
           ELPIS_ECSG_MATH_OK);
    assert(state != NULL);
    assert(elpis_ecsg_state_dim(state) == 1u);
    assert(elpis_ecsg_state_width(state) == 1u);
    assert(elpis_ecsg_state_epoch(state) == UINT64_C(0));

    scratch_count = elpis_ecsg_state_gd_step_scratch_f64(1u, 1u, 1u);
    assert(scratch_count == 3u);
    scratch = (double *)calloc(scratch_count, sizeof(*scratch));
    assert(scratch != NULL);

    /*
     * z=2, phi(z)=7, e=7, phi'(z)=8.5
     * grad = 2*x*e*phi'(z) = 238
     * W' = 1 - 0.002*238 = 0.524
     */
    assert(elpis_ecsg_state_gd_step_f64(
        state, x, y, 1u, ELPIS_ECSG_BRANCH36_REFERENCE_LR,
        scratch, scratch_count) == ELPIS_ECSG_MATH_OK);

    assert(elpis_ecsg_state_epoch(state) == UINT64_C(1));
    assert(elpis_ecsg_state_copy_w(state, w1, 1u) == ELPIS_ECSG_MATH_OK);
    assert(near(w1[0], 0.524, 1e-15));

    free(scratch);
    assert(elpis_ecsg_state_destroy(&state) == ELPIS_ECSG_MATH_OK);
    assert(state == NULL);
}

static void
test_atomic_rejection(void)
{
    const double w0[] = {
        0.1, 0.2,
        0.3, 0.4
    };
    const double x[] = {
        1.0, 0.0,
        0.0, 1.0
    };
    const double bad_y[] = {0.0, NAN};
    elpis_ecsg_state *state = NULL;
    size_t scratch_count;
    double *scratch;
    double before[4];
    double after[4];
    size_t i;

    assert(elpis_ecsg_state_create(2u, 2u, w0, &state) ==
           ELPIS_ECSG_MATH_OK);

    scratch_count = elpis_ecsg_state_gd_step_scratch_f64(2u, 2u, 2u);
    scratch = (double *)calloc(scratch_count, sizeof(*scratch));
    assert(scratch != NULL);

    assert(elpis_ecsg_state_copy_w(state, before, 4u) ==
           ELPIS_ECSG_MATH_OK);

    assert(elpis_ecsg_state_gd_step_f64(
        state, x, bad_y, 2u, ELPIS_ECSG_BRANCH36_REFERENCE_LR,
        scratch, scratch_count) == ELPIS_ECSG_MATH_NONFINITE);

    assert(elpis_ecsg_state_epoch(state) == UINT64_C(0));
    assert(elpis_ecsg_state_copy_w(state, after, 4u) ==
           ELPIS_ECSG_MATH_OK);

    for (i = 0u; i < 4u; ++i) {
        assert(before[i] == after[i]);
    }

    free(scratch);
    assert(elpis_ecsg_state_destroy(&state) == ELPIS_ECSG_MATH_OK);
}

int
main(void)
{
    test_known_one_dimensional_step();
    test_atomic_rejection();

    puts("PASS_ECS_G_STATEFUL_RECURRENCE_R0_C");
    return 0;
}
