#include "elpis/ecsg_state.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct elpis_ecsg_state {
    size_t dim;
    size_t width;
    uint64_t epoch;
    double *w;
};

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
checked_add_size(size_t a, size_t b, size_t *out)
{
    if (out == NULL || b > SIZE_MAX - a) {
        return 0;
    }
    *out = a + b;
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

static double
phi(double z)
{
    const double z2 = z * z;
    return 0.5 * z + 0.5 * z2 + 0.5 * z2 * z;
}

static double
phip(double z)
{
    return 0.5 + z + 1.5 * z * z;
}

uint32_t
elpis_ecsg_state_abi_version(void)
{
    return ELPIS_ECSG_STATE_ABI_V1;
}

elpis_ecsg_math_status
elpis_ecsg_state_create(size_t dim,
                        size_t width,
                        const double *initial_w,
                        elpis_ecsg_state **out)
{
    elpis_ecsg_state *state;
    size_t count;

    if (out == NULL) {
        return ELPIS_ECSG_MATH_INVALID;
    }
    *out = NULL;

    if (initial_w == NULL || dim == 0u || width == 0u ||
        !checked_mul_size(dim, width, &count)) {
        return ELPIS_ECSG_MATH_INVALID;
    }

    if (!finite_vector(initial_w, count)) {
        return ELPIS_ECSG_MATH_NONFINITE;
    }

    state = (elpis_ecsg_state *)calloc(1u, sizeof(*state));
    if (state == NULL) {
        return ELPIS_ECSG_MATH_INVALID;
    }

    state->w = (double *)malloc(count * sizeof(*state->w));
    if (state->w == NULL) {
        free(state);
        return ELPIS_ECSG_MATH_INVALID;
    }

    memcpy(state->w, initial_w, count * sizeof(*state->w));
    state->dim = dim;
    state->width = width;
    state->epoch = UINT64_C(0);

    *out = state;
    return ELPIS_ECSG_MATH_OK;
}

elpis_ecsg_math_status
elpis_ecsg_state_destroy(elpis_ecsg_state **state)
{
    if (state == NULL) {
        return ELPIS_ECSG_MATH_INVALID;
    }

    if (*state != NULL) {
        free((*state)->w);
        (*state)->w = NULL;
        free(*state);
        *state = NULL;
    }

    return ELPIS_ECSG_MATH_OK;
}

size_t
elpis_ecsg_state_dim(const elpis_ecsg_state *state)
{
    return state == NULL ? 0u : state->dim;
}

size_t
elpis_ecsg_state_width(const elpis_ecsg_state *state)
{
    return state == NULL ? 0u : state->width;
}

uint64_t
elpis_ecsg_state_epoch(const elpis_ecsg_state *state)
{
    return state == NULL ? UINT64_C(0) : state->epoch;
}

elpis_ecsg_math_status
elpis_ecsg_state_copy_w(const elpis_ecsg_state *state,
                        double *out,
                        size_t out_count)
{
    size_t count;

    if (state == NULL || out == NULL ||
        !checked_mul_size(state->dim, state->width, &count) ||
        out_count < count) {
        return ELPIS_ECSG_MATH_INVALID;
    }

    memcpy(out, state->w, count * sizeof(*out));
    return ELPIS_ECSG_MATH_OK;
}

elpis_ecsg_math_status
elpis_ecsg_state_project_s3_f64(const elpis_ecsg_state *state,
                                double *mu,
                                double *m_packed,
                                double *t3_packed)
{
    if (state == NULL) {
        return ELPIS_ECSG_MATH_INVALID;
    }

    return elpis_ecsg_project_s3_f64(
        state->w,
        state->dim,
        state->width,
        mu,
        m_packed,
        t3_packed);
}

elpis_ecsg_math_status
elpis_ecsg_state_forward_f64(const elpis_ecsg_state *state,
                             const double *x,
                             size_t rows,
                             double *out)
{
    if (state == NULL) {
        return ELPIS_ECSG_MATH_INVALID;
    }

    return elpis_ecsg_forward_f64(
        state->w,
        state->dim,
        state->width,
        x,
        rows,
        out);
}

size_t
elpis_ecsg_state_gd_step_scratch_f64(size_t dim,
                                     size_t width,
                                     size_t rows)
{
    size_t z_count;
    size_t w_count;
    size_t total;

    if (dim == 0u || width == 0u || rows == 0u ||
        !checked_mul_size(rows, width, &z_count) ||
        !checked_mul_size(dim, width, &w_count) ||
        !checked_add_size(z_count, rows, &total) ||
        !checked_add_size(total, w_count, &total)) {
        return 0u;
    }

    return total;
}

elpis_ecsg_math_status
elpis_ecsg_state_gd_step_f64(elpis_ecsg_state *state,
                             const double *x,
                             const double *y,
                             size_t rows,
                             double learning_rate,
                             double *scratch,
                             size_t scratch_count)
{
    size_t x_count;
    size_t w_count;
    size_t z_count;
    size_t required;
    double *z;
    double *error;
    double *candidate;
    size_t r;
    size_t i;
    size_t a;

    if (state == NULL || x == NULL || y == NULL || scratch == NULL ||
        rows == 0u || !isfinite(learning_rate) || learning_rate < 0.0 ||
        state->epoch == UINT64_MAX ||
        !checked_mul_size(rows, state->dim, &x_count) ||
        !checked_mul_size(state->dim, state->width, &w_count) ||
        !checked_mul_size(rows, state->width, &z_count)) {
        return ELPIS_ECSG_MATH_INVALID;
    }

    required = elpis_ecsg_state_gd_step_scratch_f64(
        state->dim, state->width, rows);
    if (required == 0u || scratch_count < required) {
        return ELPIS_ECSG_MATH_INVALID;
    }

    if (!finite_vector(state->w, w_count) ||
        !finite_vector(x, x_count) ||
        !finite_vector(y, rows)) {
        return ELPIS_ECSG_MATH_NONFINITE;
    }

    z = scratch;
    error = z + z_count;
    candidate = error + rows;

    for (r = 0u; r < rows; ++r) {
        double prediction = 0.0;

        for (i = 0u; i < state->width; ++i) {
            double value = 0.0;

            for (a = 0u; a < state->dim; ++a) {
                value += x[r * state->dim + a] *
                         state->w[a * state->width + i];
            }

            if (!isfinite(value)) {
                return ELPIS_ECSG_MATH_NONFINITE;
            }

            z[r * state->width + i] = value;
            prediction += phi(value);
        }

        if (!isfinite(prediction)) {
            return ELPIS_ECSG_MATH_NONFINITE;
        }

        error[r] = prediction - y[r];
        if (!isfinite(error[r])) {
            return ELPIS_ECSG_MATH_NONFINITE;
        }
    }

    for (a = 0u; a < state->dim; ++a) {
        for (i = 0u; i < state->width; ++i) {
            double sum = 0.0;
            double gradient;
            double next;
            const size_t wi = a * state->width + i;

            for (r = 0u; r < rows; ++r) {
                const double zv = z[r * state->width + i];
                sum += x[r * state->dim + a] *
                       error[r] *
                       phip(zv);
            }

            gradient = (2.0 / (double)rows) * sum;
            next = state->w[wi] - learning_rate * gradient;

            if (!isfinite(gradient) || !isfinite(next)) {
                return ELPIS_ECSG_MATH_NONFINITE;
            }

            candidate[wi] = next;
        }
    }

    memcpy(state->w, candidate, w_count * sizeof(*state->w));
    state->epoch += UINT64_C(1);

    return ELPIS_ECSG_MATH_OK;
}

/* ------------------------------------------------------------------------- */
/* Deterministic portable snapshot R0.                                       */

enum {
    ELPIS_ECSG_SNAPSHOT_HEADER_BYTES = 40u
};

static const uint8_t ELPIS_ECSG_SNAPSHOT_MAGIC[8] = {
    'E', 'L', 'P', 'I', 'S', 'G', '0', '1'
};

static void
write_u32_le(uint8_t *out, uint32_t value)
{
    out[0] = (uint8_t)(value & UINT32_C(0xff));
    out[1] = (uint8_t)((value >> 8u) & UINT32_C(0xff));
    out[2] = (uint8_t)((value >> 16u) & UINT32_C(0xff));
    out[3] = (uint8_t)((value >> 24u) & UINT32_C(0xff));
}

static uint32_t
read_u32_le(const uint8_t *in)
{
    return ((uint32_t)in[0]) |
           ((uint32_t)in[1] << 8u) |
           ((uint32_t)in[2] << 16u) |
           ((uint32_t)in[3] << 24u);
}

static void
write_u64_le(uint8_t *out, uint64_t value)
{
    unsigned shift;

    for (shift = 0u; shift < 64u; shift += 8u) {
        out[shift / 8u] = (uint8_t)((value >> shift) & UINT64_C(0xff));
    }
}

static uint64_t
read_u64_le(const uint8_t *in)
{
    uint64_t value = UINT64_C(0);
    unsigned shift;

    for (shift = 0u; shift < 64u; shift += 8u) {
        value |= ((uint64_t)in[shift / 8u]) << shift;
    }
    return value;
}

size_t
elpis_ecsg_state_snapshot_size(const elpis_ecsg_state *state)
{
    size_t count;
    size_t payload;
    size_t total;

    if (state == NULL ||
        !checked_mul_size(state->dim, state->width, &count) ||
        !checked_mul_size(count, sizeof(double), &payload) ||
        !checked_add_size((size_t)ELPIS_ECSG_SNAPSHOT_HEADER_BYTES,
                          payload,
                          &total)) {
        return 0u;
    }

    return total;
}

elpis_ecsg_math_status
elpis_ecsg_state_snapshot_write(const elpis_ecsg_state *state,
                                uint8_t *out,
                                size_t out_size)
{
    size_t required;
    size_t count;
    size_t i;

    if (state == NULL || out == NULL) {
        return ELPIS_ECSG_MATH_INVALID;
    }

    required = elpis_ecsg_state_snapshot_size(state);
    if (required == 0u || out_size < required ||
        !checked_mul_size(state->dim, state->width, &count)) {
        return ELPIS_ECSG_MATH_INVALID;
    }

    memcpy(out, ELPIS_ECSG_SNAPSHOT_MAGIC, sizeof(ELPIS_ECSG_SNAPSHOT_MAGIC));
    write_u32_le(out + 8u, UINT32_C(1));
    write_u32_le(out + 12u, UINT32_C(0));
    write_u64_le(out + 16u, (uint64_t)state->dim);
    write_u64_le(out + 24u, (uint64_t)state->width);
    write_u64_le(out + 32u, state->epoch);

    for (i = 0u; i < count; ++i) {
        uint64_t bits = UINT64_C(0);
        memcpy(&bits, &state->w[i], sizeof(bits));
        write_u64_le(out + ELPIS_ECSG_SNAPSHOT_HEADER_BYTES + i * 8u, bits);
    }

    return ELPIS_ECSG_MATH_OK;
}

elpis_ecsg_math_status
elpis_ecsg_state_snapshot_restore(const uint8_t *data,
                                  size_t data_size,
                                  elpis_ecsg_state **out)
{
    uint64_t dim64;
    uint64_t width64;
    uint64_t epoch;
    size_t dim;
    size_t width;
    size_t count;
    size_t payload;
    size_t required;
    double *w = NULL;
    elpis_ecsg_state *state = NULL;
    size_t i;
    elpis_ecsg_math_status status;

    if (out == NULL) {
        return ELPIS_ECSG_MATH_INVALID;
    }
    *out = NULL;

    if (data == NULL ||
        data_size < (size_t)ELPIS_ECSG_SNAPSHOT_HEADER_BYTES ||
        memcmp(data, ELPIS_ECSG_SNAPSHOT_MAGIC,
               sizeof(ELPIS_ECSG_SNAPSHOT_MAGIC)) != 0 ||
        read_u32_le(data + 8u) != UINT32_C(1) ||
        read_u32_le(data + 12u) != UINT32_C(0)) {
        return ELPIS_ECSG_MATH_INVALID;
    }

    dim64 = read_u64_le(data + 16u);
    width64 = read_u64_le(data + 24u);
    epoch = read_u64_le(data + 32u);

    if (dim64 == UINT64_C(0) || width64 == UINT64_C(0) ||
        dim64 > (uint64_t)SIZE_MAX || width64 > (uint64_t)SIZE_MAX) {
        return ELPIS_ECSG_MATH_INVALID;
    }

    dim = (size_t)dim64;
    width = (size_t)width64;

    if (!checked_mul_size(dim, width, &count) ||
        !checked_mul_size(count, sizeof(double), &payload) ||
        !checked_add_size((size_t)ELPIS_ECSG_SNAPSHOT_HEADER_BYTES,
                          payload,
                          &required) ||
        required != data_size) {
        return ELPIS_ECSG_MATH_INVALID;
    }

    w = (double *)malloc(payload);
    if (w == NULL) {
        return ELPIS_ECSG_MATH_INVALID;
    }

    for (i = 0u; i < count; ++i) {
        const uint64_t bits =
            read_u64_le(data + ELPIS_ECSG_SNAPSHOT_HEADER_BYTES + i * 8u);
        memcpy(&w[i], &bits, sizeof(bits));

        if (!isfinite(w[i])) {
            free(w);
            return ELPIS_ECSG_MATH_NONFINITE;
        }
    }

    status = elpis_ecsg_state_create(dim, width, w, &state);
    free(w);

    if (status != ELPIS_ECSG_MATH_OK) {
        return status;
    }

    state->epoch = epoch;
    *out = state;
    return ELPIS_ECSG_MATH_OK;
}
