#ifndef ELPIS_ECSG_STATE_H
#define ELPIS_ECSG_STATE_H

#include <stddef.h>
#include <stdint.h>

#include "elpis/ecsg_math.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ELPIS_ECSG_STATE_ABI_V1 = 1u
};

#define ELPIS_ECSG_BRANCH36_REFERENCE_LR 0.002

typedef struct elpis_ecsg_state elpis_ecsg_state;

/*
 * G1 stateful microscopic recurrence.
 *
 * The authoritative world state is W[dim,width], owned by this object.
 * G1 inherits the exact cubic full-batch gradient transition used by the
 * frozen Branch36 dynamics:
 *
 *   e       = f_W(X) - y
 *   grad W  = (2/rows) X^T (e[:,None] * phi'(XW))
 *   phi'(z) = 0.5 + z + 1.5 z^2
 *   W'      = W - learning_rate * grad W
 *
 * The frozen Branch36 reference learning rate is 0.002 and had no width
 * normalization. The API keeps learning_rate explicit because that frozen
 * experimental value is not promoted into a universal production constant.
 *
 * A successful step is atomic: the entire candidate W' is computed and
 * validated in caller-owned scratch before the state's W and epoch change.
 * Any rejected/non-finite step leaves both W and epoch unchanged.
 *
 * G1 defines the microscopic recurrence only. It does not define how DSV4.1
 * latents/results become (X,y), and it grants no runtime admission.
 */

uint32_t elpis_ecsg_state_abi_version(void);

elpis_ecsg_math_status
elpis_ecsg_state_create(size_t dim,
                        size_t width,
                        const double *initial_w,
                        elpis_ecsg_state **out);

elpis_ecsg_math_status
elpis_ecsg_state_destroy(elpis_ecsg_state **state);

size_t elpis_ecsg_state_dim(const elpis_ecsg_state *state);
size_t elpis_ecsg_state_width(const elpis_ecsg_state *state);
uint64_t elpis_ecsg_state_epoch(const elpis_ecsg_state *state);

elpis_ecsg_math_status
elpis_ecsg_state_copy_w(const elpis_ecsg_state *state,
                        double *out,
                        size_t out_count);

elpis_ecsg_math_status
elpis_ecsg_state_project_s3_f64(const elpis_ecsg_state *state,
                                double *mu,
                                double *m_packed,
                                double *t3_packed);

elpis_ecsg_math_status
elpis_ecsg_state_forward_f64(const elpis_ecsg_state *state,
                             const double *x,
                             size_t rows,
                             double *out);

size_t
elpis_ecsg_state_gd_step_scratch_f64(size_t dim,
                                     size_t width,
                                     size_t rows);

elpis_ecsg_math_status
elpis_ecsg_state_gd_step_f64(elpis_ecsg_state *state,
                             const double *x,
                             const double *y,
                             size_t rows,
                             double learning_rate,
                             double *scratch,
                             size_t scratch_count);

/*
 * Deterministic portable snapshot R0.
 *
 * Snapshot bytes contain only:
 *   magic/version, dim, width, epoch, W
 *
 * Integer fields and binary64 payloads are encoded little-endian.
 * No checksum, path, timestamp, allocator state or external authority is
 * embedded. Callers may hash/store the resulting bytes as their own concern.
 *
 * Restore constructs a new independent state. Invalid/truncated/non-finite
 * snapshots fail closed and return no state.
 */
size_t
elpis_ecsg_state_snapshot_size(const elpis_ecsg_state *state);

elpis_ecsg_math_status
elpis_ecsg_state_snapshot_write(const elpis_ecsg_state *state,
                                uint8_t *out,
                                size_t out_size);

elpis_ecsg_math_status
elpis_ecsg_state_snapshot_restore(const uint8_t *data,
                                  size_t data_size,
                                  elpis_ecsg_state **out);

#ifdef __cplusplus
}
#endif

#endif
