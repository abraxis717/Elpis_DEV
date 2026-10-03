#include "elpis/ecsg_math.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>

static int
near(double left, double right, double tolerance)
{
    return fabs(left - right) <= tolerance * (1.0 + fabs(right));
}

static void
test_dimensions(void)
{
    assert(elpis_ecsg_math_abi_version() == ELPIS_ECSG_MATH_ABI_V1);
    assert(elpis_ecsg_symmetric2_size(6u) == 21u);
    assert(elpis_ecsg_symmetric3_size(6u) == 56u);
    assert(elpis_ecsg_s3_size(6u) == 83u);
}

static void
test_known_projection(void)
{
    const double w[] = {
        1.0, 2.0,
        3.0, 4.0
    };
    double mu[2];
    double m[3];
    double t3[4];
    const double x[] = {0.5, -0.25};
    double direct[1];
    double reduced[1];

    assert(elpis_ecsg_project_s3_f64(
        w, 2u, 2u, mu, m, t3) == ELPIS_ECSG_MATH_OK);

    assert(mu[0] == 3.0);
    assert(mu[1] == 7.0);

    assert(m[0] == 5.0);
    assert(m[1] == 11.0);
    assert(m[2] == 25.0);

    assert(t3[0] == 9.0);
    assert(t3[1] == 19.0);
    assert(t3[2] == 41.0);
    assert(t3[3] == 91.0);

    assert(elpis_ecsg_forward_f64(
        w, 2u, 2u, x, 1u, direct) == ELPIS_ECSG_MATH_OK);
    assert(elpis_ecsg_forward_s3_f64(
        mu, m, t3, 2u, x, 1u, reduced) == ELPIS_ECSG_MATH_OK);

    assert(near(direct[0], -0.1015625, 1e-15));
    assert(near(reduced[0], direct[0], 1e-14));
}

static void
test_branch36_dimensions(size_t width)
{
    enum {
        D = 6,
        ROWS = 13,
        MAX_WIDTH = 72
    };

    double w[D * MAX_WIDTH];
    double x[ROWS * D];
    double mu[D];
    double m[21];
    double t3[56];
    double direct[ROWS];
    double reduced[ROWS];
    size_t a;
    size_t i;
    size_t r;

    assert(width <= MAX_WIDTH);

    for (a = 0u; a < D; ++a) {
        for (i = 0u; i < width; ++i) {
            const double numerator =
                (double)((int)((a + 3u) * (i + 5u) % 29u) - 14);
            w[a * width + i] = numerator / 37.0;
        }
    }

    for (r = 0u; r < ROWS; ++r) {
        for (a = 0u; a < D; ++a) {
            const double numerator =
                (double)((int)((r + 2u) * (a + 7u) % 17u) - 8);
            x[r * D + a] = numerator / 11.0;
        }
    }

    assert(elpis_ecsg_project_s3_f64(
        w, D, width, mu, m, t3) == ELPIS_ECSG_MATH_OK);
    assert(elpis_ecsg_forward_f64(
        w, D, width, x, ROWS, direct) == ELPIS_ECSG_MATH_OK);
    assert(elpis_ecsg_forward_s3_f64(
        mu, m, t3, D, x, ROWS, reduced) == ELPIS_ECSG_MATH_OK);

    for (r = 0u; r < ROWS; ++r) {
        assert(near(direct[r], reduced[r], 2e-12));
    }
}

static void
test_nonfinite_rejected(void)
{
    const double w[] = {
        1.0, NAN,
        2.0, 3.0
    };
    double mu[2];
    double m[3];
    double t3[4];

    assert(elpis_ecsg_project_s3_f64(
        w, 2u, 2u, mu, m, t3) == ELPIS_ECSG_MATH_NONFINITE);
}

int
main(void)
{
    test_dimensions();
    test_known_projection();
    test_branch36_dimensions(36u);
    test_branch36_dimensions(48u);
    test_branch36_dimensions(72u);
    test_nonfinite_rejected();

    puts("PASS_ECS_G_MATH_R0_C");
    return 0;
}
