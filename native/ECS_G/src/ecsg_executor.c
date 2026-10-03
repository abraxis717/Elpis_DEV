#include "elpis/ecsg_executor.h"

#include "elpis/ecsg_math.h"
#include "elpis/ecsg_state.h"

#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Arena layout (one allocation, every segment 64-byte aligned):
 *
 *   w[0], w[1], w[2]   three dim*width W buffers; `w` is authoritative,
 *                      `learn_w` stages direct learns
 *   z                  max_rows*width pre-activations of the current step
 *   error              max_rows residuals of the current step
 *   x, y               admitted experience: max_rows*dim and max_rows
 *
 * Only `w` (with epoch) is persistent learned state. Everything else is
 * scratch, overwritten by the next operation.
 */

enum {
    ARENA_ALIGN = 64u,
    ARENA_DOUBLES = ARENA_ALIGN / sizeof(double),
    SNAPSHOT_HEADER_BYTES = 40u
};

struct elpis_ecsg_executor {
    size_t dim;
    size_t width;
    size_t max_rows;
    size_t w_count;
    uint64_t epoch;
    uint64_t generation;
    double *w;
    double *learn_w;
    double *spare_w;
    double *z;
    double *error;
    double *x;
    double *y;
    void *arena;
    size_t arena_bytes;
    elpis_ecsg_exec_stats stats;
    atomic_uint_fast64_t busy_refusals;
    atomic_flag busy;
};

static int
checked_mul(size_t a, size_t b, size_t *out)
{
    if (a != 0u && b > SIZE_MAX / a) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int
checked_add(size_t a, size_t b, size_t *out)
{
    if (b > SIZE_MAX - a) {
        return 0;
    }
    *out = a + b;
    return 1;
}

/* Doubles in a segment of `count` doubles, rounded up to the arena alignment. */
static int
segment(size_t count, size_t *out)
{
    size_t padded;

    if (!checked_add(count, (size_t)ARENA_DOUBLES - 1u, &padded)) {
        return 0;
    }
    *out = padded - padded % (size_t)ARENA_DOUBLES;
    return 1;
}

typedef struct {
    size_t w, z, error, x, y, total_bytes;
} layout;

static int
plan(size_t dim, size_t width, size_t max_rows, layout *out)
{
    size_t count;
    size_t total;

    if (dim == 0u || width == 0u || max_rows == 0u ||
        !checked_mul(dim, width, &count) || !segment(count, &out->w) ||
        !checked_mul(max_rows, width, &count) || !segment(count, &out->z) ||
        !segment(max_rows, &out->error) ||
        !checked_mul(max_rows, dim, &count) || !segment(count, &out->x) ||
        !segment(max_rows, &out->y) ||
        !checked_mul(out->w, 3u, &total) ||
        !checked_add(total, out->z, &total) ||
        !checked_add(total, out->error, &total) ||
        !checked_add(total, out->x, &total) ||
        !checked_add(total, out->y, &total) ||
        !checked_mul(total, sizeof(double), &out->total_bytes)) {
        return 0;
    }
    return 1;
}

static void
carve(elpis_ecsg_executor *e, void *arena, const layout *l)
{
    double *p = (double *)arena;

    e->w = p;
    p += l->w;
    e->learn_w = p;
    p += l->w;
    e->spare_w = p;
    p += l->w;
    e->z = p;
    p += l->z;
    e->error = p;
    p += l->error;
    e->x = p;
    p += l->x;
    e->y = p;
}

static int
all_finite(const double *v, size_t count)
{
    size_t i;

    for (i = 0u; i < count; ++i) {
        if (!isfinite(v[i])) {
            return 0;
        }
    }
    return 1;
}

static int
enter(elpis_ecsg_executor *e)
{
    if (atomic_flag_test_and_set_explicit(&e->busy, memory_order_acquire)) {
        atomic_fetch_add_explicit(&e->busy_refusals, 1u, memory_order_relaxed);
        return 0;
    }
    return 1;
}

static void
leave(elpis_ecsg_executor *e)
{
    atomic_flag_clear_explicit(&e->busy, memory_order_release);
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

/*
 * One G1 step, dst = src - lr * (2/rows) X^T [e * phi'(X src)], in exactly the
 * reference's order (ecsg_state.c elpis_ecsg_state_gd_step_f64) and with its
 * finiteness rule. src and dst may be the same buffer: z is complete before
 * any write, and each weight is read once, immediately before it is written.
 * X and y were validated at admission; W is finite by construction (initial
 * and restored W are validated, every written weight is checked). Returns 0
 * if a pre-activation, prediction, residual, gradient or weight is not finite.
 */
static int
g1_step(size_t dim, size_t width, size_t rows,
        const double *x, const double *y,
        const double *src, double *dst,
        double *z, double *error, double learning_rate)
{
    const double scale = 2.0 / (double)rows;
    size_t r;
    size_t i;
    size_t a;

    for (r = 0u; r < rows; ++r) {
        double prediction = 0.0;

        for (i = 0u; i < width; ++i) {
            double value = 0.0;

            for (a = 0u; a < dim; ++a) {
                value += x[r * dim + a] * src[a * width + i];
            }
            if (!isfinite(value)) {
                return 0;
            }
            z[r * width + i] = value;
            prediction += phi(value);
        }
        if (!isfinite(prediction)) {
            return 0;
        }
        error[r] = prediction - y[r];
        if (!isfinite(error[r])) {
            return 0;
        }
    }

    for (a = 0u; a < dim; ++a) {
        for (i = 0u; i < width; ++i) {
            const size_t wi = a * width + i;
            double sum = 0.0;
            double gradient;
            double next;

            for (r = 0u; r < rows; ++r) {
                sum += x[r * dim + a] * error[r] * phip(z[r * width + i]);
            }
            gradient = scale * sum;
            next = src[wi] - learning_rate * gradient;
            if (!isfinite(gradient) || !isfinite(next)) {
                return 0;
            }
            dst[wi] = next;
        }
    }
    return 1;
}

uint32_t
elpis_ecsg_executor_abi_version(void)
{
    return ELPIS_ECSG_EXECUTOR_ABI_V1;
}

size_t
elpis_ecsg_executor_workspace_bytes(size_t dim, size_t width, size_t max_rows)
{
    layout l;

    return plan(dim, width, max_rows, &l) ? l.total_bytes : 0u;
}

/* Allocates an executor with an uninitialized authoritative W. */
static elpis_ecsg_exec_status
make(size_t dim, size_t width, size_t max_rows, elpis_ecsg_executor **out)
{
    layout l;
    elpis_ecsg_executor *e;

    if (!plan(dim, width, max_rows, &l)) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    e = (elpis_ecsg_executor *)calloc(1u, sizeof(*e));
    if (e == NULL) {
        return ELPIS_ECSG_EXEC_NOMEM;
    }
    e->arena = aligned_alloc(ARENA_ALIGN, l.total_bytes);
    if (e->arena == NULL) {
        free(e);
        return ELPIS_ECSG_EXEC_NOMEM;
    }
    memset(e->arena, 0, l.total_bytes);
    carve(e, e->arena, &l);
    e->dim = dim;
    e->width = width;
    e->max_rows = max_rows;
    e->w_count = dim * width;
    e->arena_bytes = l.total_bytes;
    e->stats.heap_allocations = 2u;
    atomic_init(&e->busy_refusals, 0u);
    atomic_flag_clear(&e->busy);
    *out = e;
    return ELPIS_ECSG_EXEC_OK;
}

elpis_ecsg_exec_status
elpis_ecsg_executor_create(size_t dim,
                           size_t width,
                           size_t max_rows,
                           const double *initial_w,
                           elpis_ecsg_executor **out)
{
    elpis_ecsg_executor *e = NULL;
    elpis_ecsg_exec_status status;
    size_t count;

    if (out == NULL) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    *out = NULL;
    if (initial_w == NULL || dim == 0u || width == 0u || max_rows == 0u ||
        !checked_mul(dim, width, &count)) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    if (!all_finite(initial_w, count)) {
        return ELPIS_ECSG_EXEC_NONFINITE;
    }
    status = make(dim, width, max_rows, &e);
    if (status != ELPIS_ECSG_EXEC_OK) {
        return status;
    }
    memcpy(e->w, initial_w, count * sizeof(double));
    *out = e;
    return ELPIS_ECSG_EXEC_OK;
}

elpis_ecsg_exec_status
elpis_ecsg_executor_restore(const uint8_t *snapshot,
                            size_t snapshot_size,
                            size_t max_rows,
                            elpis_ecsg_executor **out)
{
    elpis_ecsg_state *state = NULL;
    elpis_ecsg_executor *e = NULL;
    elpis_ecsg_exec_status status;
    elpis_ecsg_math_status math;

    if (out == NULL) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    *out = NULL;
    if (max_rows == 0u) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    /* The reference restore is the format authority (cold path). */
    math = elpis_ecsg_state_snapshot_restore(snapshot, snapshot_size, &state);
    if (math != ELPIS_ECSG_MATH_OK) {
        return math == ELPIS_ECSG_MATH_NONFINITE ? ELPIS_ECSG_EXEC_NONFINITE : ELPIS_ECSG_EXEC_INVALID;
    }
    status = make(elpis_ecsg_state_dim(state), elpis_ecsg_state_width(state), max_rows, &e);
    if (status == ELPIS_ECSG_EXEC_OK) {
        if (elpis_ecsg_state_copy_w(state, e->w, e->w_count) != ELPIS_ECSG_MATH_OK) {
            free(e->arena);
            free(e);
            e = NULL;
            status = ELPIS_ECSG_EXEC_INVALID;
        } else {
            e->epoch = elpis_ecsg_state_epoch(state);
        }
    }
    elpis_ecsg_state_destroy(&state);
    *out = e;
    return status;
}

elpis_ecsg_exec_status
elpis_ecsg_executor_destroy(elpis_ecsg_executor **exec)
{
    elpis_ecsg_executor *e;

    if (exec == NULL) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    e = *exec;
    if (e == NULL) {
        return ELPIS_ECSG_EXEC_OK;
    }
    if (!enter(e)) {
        return ELPIS_ECSG_EXEC_BUSY;
    }
    free(e->arena);
    free(e);
    *exec = NULL;
    return ELPIS_ECSG_EXEC_OK;
}

elpis_ecsg_exec_status
elpis_ecsg_executor_reserve(elpis_ecsg_executor *e, size_t max_rows)
{
    layout l;
    void *arena;
    double *w;

    if (e == NULL || max_rows == 0u) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    if (!enter(e)) {
        return ELPIS_ECSG_EXEC_BUSY;
    }
    if (max_rows <= e->max_rows) {
        leave(e);
        return ELPIS_ECSG_EXEC_OK;
    }
    if (!plan(e->dim, e->width, max_rows, &l)) {
        leave(e);
        return ELPIS_ECSG_EXEC_INVALID;
    }
    arena = aligned_alloc(ARENA_ALIGN, l.total_bytes);
    if (arena == NULL) {
        leave(e);
        return ELPIS_ECSG_EXEC_NOMEM;
    }
    memset(arena, 0, l.total_bytes);
    w = e->w;
    carve(e, arena, &l);
    memcpy(e->w, w, e->w_count * sizeof(double));
    free(e->arena);
    e->arena = arena;
    e->arena_bytes = l.total_bytes;
    e->max_rows = max_rows;
    e->stats.heap_allocations += 1u;
    leave(e);
    return ELPIS_ECSG_EXEC_OK;
}

size_t
elpis_ecsg_executor_dim(const elpis_ecsg_executor *e)
{
    return e == NULL ? 0u : e->dim;
}

size_t
elpis_ecsg_executor_width(const elpis_ecsg_executor *e)
{
    return e == NULL ? 0u : e->width;
}

size_t
elpis_ecsg_executor_max_rows(const elpis_ecsg_executor *e)
{
    return e == NULL ? 0u : e->max_rows;
}

uint64_t
elpis_ecsg_executor_epoch(const elpis_ecsg_executor *e)
{
    return e == NULL ? UINT64_C(0) : e->epoch;
}

uint64_t
elpis_ecsg_executor_generation(const elpis_ecsg_executor *e)
{
    return e == NULL ? UINT64_C(0) : e->generation;
}

elpis_ecsg_exec_status
elpis_ecsg_executor_forward(elpis_ecsg_executor *e,
                            const double *x,
                            size_t rows,
                            double *out)
{
    elpis_ecsg_math_status math;

    if (e == NULL || x == NULL || out == NULL || rows == 0u) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    if (!enter(e)) {
        return ELPIS_ECSG_EXEC_BUSY;
    }
    e->stats.forward_calls += 1u;
    math = elpis_ecsg_forward_f64(e->w, e->dim, e->width, x, rows, out);
    if (math == ELPIS_ECSG_MATH_NONFINITE) {
        e->stats.refusals += 1u;
    }
    leave(e);
    return math == ELPIS_ECSG_MATH_OK ? ELPIS_ECSG_EXEC_OK
         : math == ELPIS_ECSG_MATH_NONFINITE ? ELPIS_ECSG_EXEC_NONFINITE : ELPIS_ECSG_EXEC_INVALID;
}

/*
 * Validates a schedule against the admitted capacity and the epoch headroom
 * (entered executor). On success *total_rows and *total_steps are set.
 */
static elpis_ecsg_exec_status
check_schedule(const elpis_ecsg_executor *e,
               const elpis_ecsg_drive *drives,
               size_t drive_count,
               double learning_rate,
               uint64_t epoch,
               size_t *total_rows,
               uint64_t *total_steps)
{
    size_t rows = 0u;
    uint64_t steps = 0u;
    size_t j;

    if (drives == NULL || drive_count == 0u || !isfinite(learning_rate) || learning_rate < 0.0) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    for (j = 0u; j < drive_count; ++j) {
        if (drives[j].rows == 0u || drives[j].steps == 0u ||
            !checked_add(rows, drives[j].rows, &rows) ||
            drives[j].steps > UINT64_MAX - steps) {
            return ELPIS_ECSG_EXEC_INVALID;
        }
        steps += drives[j].steps;
    }
    if (steps > UINT64_MAX - epoch) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    if (rows > e->max_rows) {
        return ELPIS_ECSG_EXEC_CAPACITY;
    }
    *total_rows = rows;
    *total_steps = steps;
    return ELPIS_ECSG_EXEC_OK;
}

/* Copies and validates the experience into executor storage, once. */
static elpis_ecsg_exec_status
admit(elpis_ecsg_executor *e, const double *x, const double *y, size_t rows)
{
    const size_t x_count = rows * e->dim;  /* rows <= max_rows: no overflow */

    memcpy(e->x, x, x_count * sizeof(double));
    memcpy(e->y, y, rows * sizeof(double));
    if (!all_finite(e->x, x_count) || !all_finite(e->y, rows)) {
        return ELPIS_ECSG_EXEC_NONFINITE;
    }
    return ELPIS_ECSG_EXEC_OK;
}

/*
 * Runs a validated schedule on the admitted experience: step 1 reads `src`
 * and writes `dst`, later steps update `dst` in place. On a refused step,
 * returns NONFINITE and sets *failed_step.
 */
static elpis_ecsg_exec_status
run_schedule(elpis_ecsg_executor *e,
             const elpis_ecsg_drive *drives,
             size_t drive_count,
             size_t total_rows,
             uint64_t total_steps,
             double learning_rate,
             const double *src,
             double *dst,
             uint64_t *failed_step)
{
    size_t offset = 0u;
    uint64_t done = 0u;
    size_t j;

    for (j = 0u; j < drive_count; ++j) {
        /* Re-read bounds: the caller's drive array is never trusted twice. */
        const size_t rows = drives[j].rows;
        const uint64_t steps = drives[j].steps;
        uint64_t k;

        if (rows == 0u || rows > total_rows - offset || steps == 0u || steps > total_steps - done) {
            return ELPIS_ECSG_EXEC_INVALID;
        }
        for (k = 0u; k < steps; ++k) {
            const int ok = g1_step(e->dim, e->width, rows, e->x + offset * e->dim, e->y + offset,
                                   src, dst, e->z, e->error, learning_rate);
            done += 1u;
            e->stats.steps_executed += 1u;
            if (!ok) {
                *failed_step = done;
                return ELPIS_ECSG_EXEC_NONFINITE;
            }
            src = dst;
        }
        offset += rows;
    }
    return ELPIS_ECSG_EXEC_OK;
}

elpis_ecsg_exec_status
elpis_ecsg_executor_learn_schedule(elpis_ecsg_executor *e,
                                   const double *x,
                                   const double *y,
                                   const elpis_ecsg_drive *drives,
                                   size_t drive_count,
                                   double learning_rate,
                                   elpis_ecsg_exec_transition *transition)
{
    elpis_ecsg_exec_transition t;
    elpis_ecsg_exec_status status;
    size_t total_rows = 0u;
    uint64_t total_steps = 0u;
    double *swap;

    if (e == NULL || x == NULL || y == NULL) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    if (!enter(e)) {
        return ELPIS_ECSG_EXEC_BUSY;
    }
    memset(&t, 0, sizeof(t));
    t.epoch_before = t.epoch_after = e->epoch;
    t.generation_before = t.generation_after = e->generation;
    e->stats.learn_calls += 1u;
    status = check_schedule(e, drives, drive_count, learning_rate, e->epoch, &total_rows, &total_steps);
    if (status == ELPIS_ECSG_EXEC_OK) {
        status = admit(e, x, y, total_rows);
    }
    if (status == ELPIS_ECSG_EXEC_OK) {
        status = run_schedule(e, drives, drive_count, total_rows, total_steps, learning_rate, e->w, e->learn_w,
                              &t.failed_step);
    }
    if (status == ELPIS_ECSG_EXEC_OK) {
        /* Commit: exchange buffers, never copy W. */
        swap = e->w;
        e->w = e->learn_w;
        e->learn_w = swap;
        e->epoch += total_steps;
        e->generation += 1u;
        e->stats.commits += 1u;
        t.steps = total_steps;
        t.epoch_after = e->epoch;
        t.generation_after = e->generation;
    } else if (status == ELPIS_ECSG_EXEC_NONFINITE) {
        e->stats.refusals += 1u;
    }
    if (transition != NULL) {
        *transition = t;
    }
    leave(e);
    return status;
}

elpis_ecsg_exec_status
elpis_ecsg_executor_learn(elpis_ecsg_executor *e,
                          const double *x,
                          const double *y,
                          size_t rows,
                          double learning_rate,
                          uint64_t steps,
                          elpis_ecsg_exec_transition *transition)
{
    elpis_ecsg_drive drive;

    drive.rows = rows;
    drive.steps = steps;
    return elpis_ecsg_executor_learn_schedule(e, x, y, &drive, 1u, learning_rate, transition);
}

elpis_ecsg_exec_status
elpis_ecsg_executor_copy_w(elpis_ecsg_executor *e, double *out, size_t out_count)
{
    if (e == NULL || out == NULL) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    if (!enter(e)) {
        return ELPIS_ECSG_EXEC_BUSY;
    }
    if (out_count < e->w_count) {
        leave(e);
        return ELPIS_ECSG_EXEC_INVALID;
    }
    memcpy(out, e->w, e->w_count * sizeof(double));
    leave(e);
    return ELPIS_ECSG_EXEC_OK;
}

elpis_ecsg_exec_status
elpis_ecsg_executor_project_s3(elpis_ecsg_executor *e,
                               double *mu,
                               double *m_packed,
                               double *t3_packed)
{
    elpis_ecsg_math_status math;

    if (e == NULL) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    if (!enter(e)) {
        return ELPIS_ECSG_EXEC_BUSY;
    }
    math = elpis_ecsg_project_s3_f64(e->w, e->dim, e->width, mu, m_packed, t3_packed);
    leave(e);
    return math == ELPIS_ECSG_MATH_OK ? ELPIS_ECSG_EXEC_OK
         : math == ELPIS_ECSG_MATH_NONFINITE ? ELPIS_ECSG_EXEC_NONFINITE : ELPIS_ECSG_EXEC_INVALID;
}

size_t
elpis_ecsg_executor_snapshot_size(const elpis_ecsg_executor *e)
{
    size_t payload;
    size_t total;

    if (e == NULL || !checked_mul(e->w_count, sizeof(double), &payload) ||
        !checked_add((size_t)SNAPSHOT_HEADER_BYTES, payload, &total)) {
        return 0u;
    }
    return total;
}

static void
put_u64(uint8_t *out, uint64_t value)
{
    unsigned b;

    for (b = 0u; b < 8u; ++b) {
        out[b] = (uint8_t)((value >> (8u * b)) & UINT64_C(0xff));
    }
}

elpis_ecsg_exec_status
elpis_ecsg_executor_snapshot_write(elpis_ecsg_executor *e, uint8_t *out, size_t out_size)
{
    static const uint8_t magic[8] = {'E', 'L', 'P', 'I', 'S', 'G', '0', '1'};
    size_t required;
    size_t i;

    if (e == NULL || out == NULL) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    required = elpis_ecsg_executor_snapshot_size(e);
    if (required == 0u || out_size < required) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    if (!enter(e)) {
        return ELPIS_ECSG_EXEC_BUSY;
    }
    /* Byte-identical to elpis_ecsg_state_snapshot_write. */
    memcpy(out, magic, sizeof(magic));
    put_u64(out + 8u, UINT64_C(1));  /* u32 version 1, u32 reserved 0 */
    put_u64(out + 16u, (uint64_t)e->dim);
    put_u64(out + 24u, (uint64_t)e->width);
    put_u64(out + 32u, e->epoch);
    for (i = 0u; i < e->w_count; ++i) {
        uint64_t bits;
        memcpy(&bits, &e->w[i], sizeof(bits));
        put_u64(out + SNAPSHOT_HEADER_BYTES + i * 8u, bits);
    }
    leave(e);
    return ELPIS_ECSG_EXEC_OK;
}

elpis_ecsg_exec_status
elpis_ecsg_executor_stats(elpis_ecsg_executor *e, elpis_ecsg_exec_stats *out)
{
    if (e == NULL || out == NULL) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    if (!enter(e)) {
        return ELPIS_ECSG_EXEC_BUSY;
    }
    *out = e->stats;
    out->workspace_bytes = e->arena_bytes;
    out->max_rows = e->max_rows;
    out->busy_refusals = (uint64_t)atomic_load_explicit(&e->busy_refusals, memory_order_relaxed);
    leave(e);
    return ELPIS_ECSG_EXEC_OK;
}
