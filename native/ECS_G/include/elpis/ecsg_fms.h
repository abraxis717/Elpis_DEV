#ifndef ELPIS_ECSG_FMS_H
#define ELPIS_ECSG_FMS_H

#include "elpis/ecsg_executor.h"
#include "elpis/fms.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mutable ECS residency adapter ABI v1.
 *
 * The authoritative mutable FMS object is the existing portable ECS
 * snapshot byte sequence (ELPISG01 header + epoch + W). No new cognitive
 * serialization exists. The adapter restores the already-qualified public
 * executor for a direct operation, publishes a completed snapshot atomically
 * under an FMS WRITE lease, then destroys the transient executor. An open
 * transaction retains one executor and one FMS lease until commit/abort.
 *
 * Existing executor/state ABIs and their source files are unchanged.
 * Return codes are executor codes -1..-6 or (-100 + fms_status).
 * Runtime takes ownership of a supplied empty FMS context on success.
 */
enum { ELPIS_ECSG_FMS_ABI_V1 = 1u, ELPIS_ECSG_FMS_ERROR_BASE = -100 };

typedef struct elpis_ecsg_fms elpis_ecsg_fms;
typedef uint64_t elpis_ecsg_fms_id;

typedef struct {
    uint8_t logical_identity[32];
    uint64_t object_handle;
    uint64_t epoch;
    uint64_t generation;
    size_t dim;
    size_t width;
    size_t max_rows;
    size_t snapshot_bytes;
    uint64_t logical_bytes;
    uint64_t resident_authoritative_bytes;
    uint64_t active_workspace_bytes;
    uint64_t acquisitions;
    uint64_t lease_failures;
    uint64_t commits;
    uint64_t aborts;
    uint64_t materialization_ns;
    uint64_t query_ns;
    uint64_t learn_ns;
    uint64_t commit_ns;
    uint32_t lease_count;
    uint8_t tier;
    uint8_t residency_state;
    uint8_t cold_replica;
    uint8_t transaction_open;
} elpis_ecsg_fms_info;

typedef struct {
    fms_stats residency;
    uint64_t logical_bytes;
    uint64_t resident_authoritative_bytes;
    uint64_t active_workspace_bytes;
    uint64_t resident_high_water;
    uint64_t leases;
    uint64_t states;
    uint64_t demotion_ns;
} elpis_ecsg_fms_metrics;

uint32_t elpis_ecsg_fms_abi_version(void);
int elpis_ecsg_fms_create(fms_ctx *owned_context, size_t max_states, elpis_ecsg_fms **out);
int elpis_ecsg_fms_destroy(elpis_ecsg_fms **runtime);

int elpis_ecsg_fms_register(elpis_ecsg_fms *, const uint8_t key[32],
    size_t dim, size_t width, size_t max_rows, const double *w, elpis_ecsg_fms_id *out);
int elpis_ecsg_fms_restore(elpis_ecsg_fms *, const uint8_t key[32],
    const uint8_t *snapshot, size_t bytes, size_t max_rows, elpis_ecsg_fms_id *out);
int elpis_ecsg_fms_close(elpis_ecsg_fms *, elpis_ecsg_fms_id *id);

int elpis_ecsg_fms_inspect(elpis_ecsg_fms *, elpis_ecsg_fms_id, elpis_ecsg_fms_info *);
int elpis_ecsg_fms_exec_stats(elpis_ecsg_fms *, elpis_ecsg_fms_id, elpis_ecsg_exec_stats *);
int elpis_ecsg_fms_stats(elpis_ecsg_fms *, elpis_ecsg_fms_metrics *);
int elpis_ecsg_fms_pump(elpis_ecsg_fms *);

int elpis_ecsg_fms_reserve(elpis_ecsg_fms *, elpis_ecsg_fms_id, size_t rows);
int elpis_ecsg_fms_forward(elpis_ecsg_fms *, elpis_ecsg_fms_id,
    const double *, size_t, double *);
int elpis_ecsg_fms_learn(elpis_ecsg_fms *, elpis_ecsg_fms_id,
    const double *, const double *, size_t, double, uint64_t, elpis_ecsg_exec_transition *);
int elpis_ecsg_fms_learn_schedule(elpis_ecsg_fms *, elpis_ecsg_fms_id,
    const double *, const double *, const elpis_ecsg_drive *, size_t,
    double, elpis_ecsg_exec_transition *);
int elpis_ecsg_fms_copy_w(elpis_ecsg_fms *, elpis_ecsg_fms_id, double *, size_t);
int elpis_ecsg_fms_project_s3(elpis_ecsg_fms *, elpis_ecsg_fms_id,
    double *, double *, double *);
int elpis_ecsg_fms_snapshot_write(elpis_ecsg_fms *, elpis_ecsg_fms_id,
    uint8_t *, size_t);

int elpis_ecsg_fms_txn_begin(elpis_ecsg_fms *, elpis_ecsg_fms_id, uint64_t *);
int elpis_ecsg_fms_txn_learn(elpis_ecsg_fms *, elpis_ecsg_fms_id, uint64_t,
    const double *, const double *, size_t, double, uint64_t, elpis_ecsg_exec_transition *);
int elpis_ecsg_fms_txn_learn_schedule(elpis_ecsg_fms *, elpis_ecsg_fms_id, uint64_t,
    const double *, const double *, const elpis_ecsg_drive *, size_t,
    double, elpis_ecsg_exec_transition *);
int elpis_ecsg_fms_txn_forward(elpis_ecsg_fms *, elpis_ecsg_fms_id, uint64_t,
    const double *, size_t, double *);
int elpis_ecsg_fms_txn_project_s3(elpis_ecsg_fms *, elpis_ecsg_fms_id, uint64_t,
    double *, double *, double *);
int elpis_ecsg_fms_txn_epoch(elpis_ecsg_fms *, elpis_ecsg_fms_id, uint64_t, uint64_t *);
int elpis_ecsg_fms_txn_commit(elpis_ecsg_fms *, elpis_ecsg_fms_id, uint64_t,
    elpis_ecsg_exec_transition *);
int elpis_ecsg_fms_txn_abort(elpis_ecsg_fms *, elpis_ecsg_fms_id, uint64_t);

#ifdef __cplusplus
}
#endif
#endif
