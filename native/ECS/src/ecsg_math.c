#include "elpis/ecsg_math.h"

#include <math.h>
#include <stdint.h>

static int
checked_mul_size(size_t a, size_t b, size_t *out)
{
    if (out == NULL) {
        return 0;
    }
    if (a != 0u && b > SIZE_MAX / a) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int
finite_vector(const double *x, size_t count)
{
    size_t i;

    if (x == NULL) {
        return 0;
    }
    for (i = 0u; i < count; ++i) {
        if (!isfinite(x[i])) {
            return 0;
        }
    }
    return 1;
}

static size_t
symmetric2_size(size_t dim)
{
    size_t a;
    size_t b;

    if (dim == 0u || dim == SIZE_MAX) {
        return 0u;
    }

    a = dim;
    b = dim + 1u;

    if ((a & 1u) == 0u) {
        a /= 2u;
    } else {
        b /= 2u;
    }

    if (a != 0u && b > SIZE_MAX / a) {
        return 0u;
    }
    return a * b;
}

static size_t
symmetric3_size(size_t dim)
{
    size_t a;
    size_t b;
    size_t c;
    size_t ab;

    if (dim == 0u || dim > SIZE_MAX - 2u) {
        return 0u;
    }

    a = dim;
    b = dim + 1u;
    c = dim + 2u;

    if ((a % 2u) == 0u) {
        a /= 2u;
    } else if ((b % 2u) == 0u) {
        b /= 2u;
    } else {
        c /= 2u;
    }

    if ((a % 3u) == 0u) {
        a /= 3u;
    } else if ((b % 3u) == 0u) {
        b /= 3u;
    } else {
        c /= 3u;
    }

    if (!checked_mul_size(a, b, &ab)) {
        return 0u;
    }
    if (ab != 0u && c > SIZE_MAX / ab) {
        return 0u;
    }
    return ab * c;
}

uint32_t
elpis_ecsg_math_abi_version(void)
{
    return ELPIS_ECSG_MATH_ABI_V1;
}

size_t
elpis_ecsg_symmetric2_size(size_t dim)
{
    return symmetric2_size(dim);
}

size_t
elpis_ecsg_symmetric3_size(size_t dim)
{
    return symmetric3_size(dim);
}

size_t
elpis_ecsg_s3_size(size_t dim)
{
    size_t m;
    size_t t;

    if (dim == 0u) {
        return 0u;
    }

    m = symmetric2_size(dim);
    t = symmetric3_size(dim);
    if (m == 0u || t == 0u) {
        return 0u;
    }
    if (dim > SIZE_MAX - m || dim + m > SIZE_MAX - t) {
        return 0u;
    }
    return dim + m + t;
}

elpis_ecsg_math_status
elpis_ecsg_project_s3_f64(const double *w,
                          size_t dim,
                          size_t width,
                          double *mu,
                          double *m_packed,
                          double *t3_packed)
{
    size_t w_count;
    size_t a;
    size_t b;
    size_t c;
    size_t i;
    size_t p;

    if (w == NULL || mu == NULL || m_packed == NULL || t3_packed == NULL ||
        dim == 0u || width == 0u ||
        symmetric2_size(dim) == 0u ||
        symmetric3_size(dim) == 0u ||
        !checked_mul_size(dim, width, &w_count)) {
        return ELPIS_ECSG_MATH_INVALID;
    }

    if (!finite_vector(w, w_count)) {
        return ELPIS_ECSG_MATH_NONFINITE;
    }

    for (a = 0u; a < dim; ++a) {
        double sum = 0.0;
        for (i = 0u; i < width; ++i) {
            sum += w[a * width + i];
        }
        if (!isfinite(sum)) {
            return ELPIS_ECSG_MATH_NONFINITE;
        }
        mu[a] = sum;
    }

    p = 0u;
    for (a = 0u; a < dim; ++a) {
        for (b = a; b < dim; ++b) {
            double sum = 0.0;
            for (i = 0u; i < width; ++i) {
                sum += w[a * width + i] * w[b * width + i];
            }
            if (!isfinite(sum)) {
                return ELPIS_ECSG_MATH_NONFINITE;
            }
            m_packed[p++] = sum;
        }
    }

    p = 0u;
    for (a = 0u; a < dim; ++a) {
        for (b = a; b < dim; ++b) {
            for (c = b; c < dim; ++c) {
                double sum = 0.0;
                for (i = 0u; i < width; ++i) {
                    sum += w[a * width + i] *
                           w[b * width + i] *
                           w[c * width + i];
                }
                if (!isfinite(sum)) {
                    return ELPIS_ECSG_MATH_NONFINITE;
                }
                t3_packed[p++] = sum;
            }
        }
    }

    return ELPIS_ECSG_MATH_OK;
}

elpis_ecsg_math_status
elpis_ecsg_forward_f64(const double *w,
                       size_t dim,
                       size_t width,
                       const double *x,
                       size_t rows,
                       double *out)
{
    size_t w_count;
    size_t x_count;
    size_t r;
    size_t i;
    size_t a;

    if (w == NULL || x == NULL || out == NULL ||
        dim == 0u || width == 0u || rows == 0u ||
        !checked_mul_size(dim, width, &w_count) ||
        !checked_mul_size(rows, dim, &x_count)) {
        return ELPIS_ECSG_MATH_INVALID;
    }

    if (!finite_vector(w, w_count) || !finite_vector(x, x_count)) {
        return ELPIS_ECSG_MATH_NONFINITE;
    }

    for (r = 0u; r < rows; ++r) {
        double total = 0.0;

        for (i = 0u; i < width; ++i) {
            double z = 0.0;
            double z2;
            double phi;

            for (a = 0u; a < dim; ++a) {
                z += x[r * dim + a] * w[a * width + i];
            }
            if (!isfinite(z)) {
                return ELPIS_ECSG_MATH_NONFINITE;
            }

            z2 = z * z;
            phi = 0.5 * z + 0.5 * z2 + 0.5 * z2 * z;
            total += phi;
        }

        if (!isfinite(total)) {
            return ELPIS_ECSG_MATH_NONFINITE;
        }
        out[r] = total;
    }

    return ELPIS_ECSG_MATH_OK;
}

elpis_ecsg_math_status
elpis_ecsg_forward_s3_f64(const double *mu,
                          const double *m_packed,
                          const double *t3_packed,
                          size_t dim,
                          const double *x,
                          size_t rows,
                          double *out)
{
    size_t m_count;
    size_t t_count;
    size_t x_count;
    size_t r;
    size_t a;
    size_t b;
    size_t c;

    if (mu == NULL || m_packed == NULL || t3_packed == NULL ||
        x == NULL || out == NULL || dim == 0u || rows == 0u ||
        !checked_mul_size(rows, dim, &x_count)) {
        return ELPIS_ECSG_MATH_INVALID;
    }

    m_count = symmetric2_size(dim);
    t_count = symmetric3_size(dim);
    if (m_count == 0u || t_count == 0u) {
        return ELPIS_ECSG_MATH_INVALID;
    }

    if (!finite_vector(mu, dim) ||
        !finite_vector(m_packed, m_count) ||
        !finite_vector(t3_packed, t_count) ||
        !finite_vector(x, x_count)) {
        return ELPIS_ECSG_MATH_NONFINITE;
    }

    for (r = 0u; r < rows; ++r) {
        double linear = 0.0;
        double quadratic = 0.0;
        double cubic = 0.0;
        size_t p = 0u;

        for (a = 0u; a < dim; ++a) {
            linear += mu[a] * x[r * dim + a];
        }

        p = 0u;
        for (a = 0u; a < dim; ++a) {
            for (b = a; b < dim; ++b) {
                const double multiplicity = (a == b) ? 1.0 : 2.0;
                quadratic += multiplicity *
                             m_packed[p++] *
                             x[r * dim + a] *
                             x[r * dim + b];
            }
        }

        p = 0u;
        for (a = 0u; a < dim; ++a) {
            for (b = a; b < dim; ++b) {
                for (c = b; c < dim; ++c) {
                    double multiplicity;

                    if (a == b && b == c) {
                        multiplicity = 1.0;
                    } else if (a == b || b == c || a == c) {
                        multiplicity = 3.0;
                    } else {
                        multiplicity = 6.0;
                    }

                    cubic += multiplicity *
                             t3_packed[p++] *
                             x[r * dim + a] *
                             x[r * dim + b] *
                             x[r * dim + c];
                }
            }
        }

        out[r] = 0.5 * linear + 0.5 * quadratic + 0.5 * cubic;
        if (!isfinite(out[r])) {
            return ELPIS_ECSG_MATH_NONFINITE;
        }
    }

    return ELPIS_ECSG_MATH_OK;
}
