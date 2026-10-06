#ifndef ELPIS_ECSG_K1_FMS_H
#define ELPIS_ECSG_K1_FMS_H

#include "elpis/ecsg_k1.h"
#include "elpis/fms.h"

#ifdef __cplusplus
extern "C" {
#endif

/* K1 residency over generic FMS (docs/ECS_K1_RUNTIME.md), ABI v1.
 *
 * One FMS object per logical K1 state holds its complete resident image (header, W, H packed, a). FMS owns the
 * bytes, logical identity, tier, leases, pins, cold replicas and materialization; it never interprets them. The
 * adapter owns the cognitive interpretation through one persistent native K1 workspace per state:
 *
 *   warm path:  pin the object WARM (READ for query/copy/snapshot, WRITE for learn/consolidate/reset)
 *               -> native K1 operation directly over the resident bytes -> unpin
 *
 * The pin is FMS's unfenced acquire: a CPU operation needs no accelerator fence, and on a WARM object it allocates
 * nothing (FMS allocates only to materialize COLD -> WARM).
 *
 * There is no per-operation restore, executor construction, deserialization or serialization. A failed operation
 * changes no resident byte (K1 stages outside the image and installs only on success). A transaction holds its
 * WRITE pin from begin to commit/abort and commits W, epoch, H and a together. Transaction refusals follow the
 * contract of ecsg_k1.h exactly: INVALID/CAPACITY/BUSY keep the transaction (and its pin) open; STALE, or NONFINITE
 * from learn/consolidate, discards it and releases the pin. A failed FMS release keeps the pin it still holds
 * (consistent, retried by the next operation and by close; a held READ pin never serves a write) and never turns a
 * committed result into a refusal.
 *
 * Return codes: K1 codes -1..-7, or (-100 + fms_status). Same-state calls are SINGLE_WRITER (BUSY); distinct
 * states share no cognitive lock. The runtime takes ownership of a supplied empty FMS context on success.
 */
enum { ELPIS_ECSG_K1_FMS_ABI_V1 = 1u, ELPIS_ECSG_K1_FMS_ERROR_BASE = -100 };

typedef struct elpis_ecsg_k1_fms elpis_ecsg_k1_fms;

typedef struct {
    uint8_t logical_identity[32];
    uint64_t object_handle;
    uint64_t epoch;                 /* as of the last committed transition */
    uint64_t generation;
    size_t dim;
    size_t width;
    size_t max_rows;
    size_t image_bytes;             /* resident authoritative bytes when WARM */
    size_t envelope_bytes;          /* portable snapshot bytes */
    size_t workspace_bytes;         /* non-authoritative native workspace */
    uint64_t acquisitions;
    uint64_t lease_failures;
    uint64_t commits;
    uint64_t aborts;
    uint64_t materialization_ns;
    uint64_t query_ns;
    uint64_t learn_ns;
    uint64_t consolidate_ns;
    uint64_t commit_ns;
    uint32_t lease_count;           /* FMS pins and fenced leases currently held */
    uint32_t provenance;
    uint8_t tier;
    uint8_t residency_state;
    uint8_t cold_replica;
    uint8_t transaction_open;
} elpis_ecsg_k1_fms_info;

typedef struct {
    fms_stats residency;
    uint64_t states;
    uint64_t logical_bytes;
    uint64_t resident_authoritative_bytes;
    uint64_t workspace_bytes;
    uint64_t resident_high_water;
    uint64_t leases;
    uint64_t demotion_ns;
} elpis_ecsg_k1_fms_metrics;

uint32_t elpis_ecsg_k1_fms_abi_version(void);
int elpis_ecsg_k1_fms_create(fms_ctx *owned_context, size_t max_states, elpis_ecsg_k1_fms **out);
int elpis_ecsg_k1_fms_destroy(elpis_ecsg_k1_fms **runtime);

/* A new complete K1 state from W (H = 0, a = 0). */
int elpis_ecsg_k1_fms_register(elpis_ecsg_k1_fms *, const uint8_t key[32], size_t dim, size_t width,
                               size_t max_rows, const double *w, uint64_t *out);
/* A complete K1 state from a retained-state envelope (validated, checksum verified). */
int elpis_ecsg_k1_fms_restore(elpis_ecsg_k1_fms *, const uint8_t key[32], const uint8_t *envelope, size_t bytes,
                              size_t max_rows, uint64_t *out);
/* A W-only ELPISG01 snapshot, admitted only as UNCONSOLIDATED. */
int elpis_ecsg_k1_fms_import_w_only(elpis_ecsg_k1_fms *, const uint8_t key[32], const uint8_t *snapshot,
                                    size_t bytes, size_t max_rows, uint64_t *out);
int elpis_ecsg_k1_fms_close(elpis_ecsg_k1_fms *, uint64_t *id);

int elpis_ecsg_k1_fms_inspect(elpis_ecsg_k1_fms *, uint64_t id, elpis_ecsg_k1_fms_info *out);
int elpis_ecsg_k1_fms_k1_stats(elpis_ecsg_k1_fms *, uint64_t id, elpis_ecsg_k1_counters *out);
int elpis_ecsg_k1_fms_stats(elpis_ecsg_k1_fms *, elpis_ecsg_k1_fms_metrics *out);
int elpis_ecsg_k1_fms_pump(elpis_ecsg_k1_fms *);
int elpis_ecsg_k1_fms_reserve(elpis_ecsg_k1_fms *, uint64_t id, size_t rows);

int elpis_ecsg_k1_fms_forward(elpis_ecsg_k1_fms *, uint64_t id, const double *x, size_t rows, double *out);
int elpis_ecsg_k1_fms_learn(elpis_ecsg_k1_fms *, uint64_t id, const double *x, const double *y, size_t rows,
                            double rate, uint64_t steps, elpis_ecsg_k1_transition *transition);
int elpis_ecsg_k1_fms_consolidate(elpis_ecsg_k1_fms *, uint64_t id, const double *x, size_t rows,
                                  elpis_ecsg_k1_transition *transition);
int elpis_ecsg_k1_fms_reset(elpis_ecsg_k1_fms *, uint64_t id, elpis_ecsg_k1_transition *transition);
int elpis_ecsg_k1_fms_copy_w(elpis_ecsg_k1_fms *, uint64_t id, double *out, size_t count);
int elpis_ecsg_k1_fms_copy_h_packed(elpis_ecsg_k1_fms *, uint64_t id, double *out, size_t count);
int elpis_ecsg_k1_fms_copy_a(elpis_ecsg_k1_fms *, uint64_t id, double *out, size_t count);
int elpis_ecsg_k1_fms_snapshot_write(elpis_ecsg_k1_fms *, uint64_t id, uint8_t *out, size_t size);
int elpis_ecsg_k1_fms_state_digest(elpis_ecsg_k1_fms *, uint64_t id,
                                   uint8_t out[ELPIS_ECSG_K1_DIGEST_BYTES]);

int elpis_ecsg_k1_fms_txn_begin(elpis_ecsg_k1_fms *, uint64_t id, uint64_t *token);
int elpis_ecsg_k1_fms_txn_learn(elpis_ecsg_k1_fms *, uint64_t id, uint64_t token, const double *x, const double *y,
                                size_t rows, double rate, uint64_t steps, elpis_ecsg_k1_transition *transition);
int elpis_ecsg_k1_fms_txn_consolidate(elpis_ecsg_k1_fms *, uint64_t id, uint64_t token, const double *x,
                                      size_t rows);
int elpis_ecsg_k1_fms_txn_forward(elpis_ecsg_k1_fms *, uint64_t id, uint64_t token, const double *x, size_t rows,
                                  double *out);
int elpis_ecsg_k1_fms_txn_epoch(elpis_ecsg_k1_fms *, uint64_t id, uint64_t token, uint64_t *epoch);
int elpis_ecsg_k1_fms_txn_commit(elpis_ecsg_k1_fms *, uint64_t id, uint64_t token,
                                 elpis_ecsg_k1_transition *transition);
int elpis_ecsg_k1_fms_txn_commit_identity(elpis_ecsg_k1_fms *, uint64_t id, uint64_t token,
                                          elpis_ecsg_k1_commit_identity *identity);
int elpis_ecsg_k1_fms_txn_abort(elpis_ecsg_k1_fms *, uint64_t id, uint64_t token);

/* The experience schedule of ecsg_k1.h on the resident candidate, under the transaction's WRITE pin: one call learns
 * and consolidates every experience natively and returns S3 of the final candidate W. Same validation and refusal
 * contract as elpis_ecsg_k1_txn_run_schedule; a discarding refusal releases the pin. */
int elpis_ecsg_k1_fms_txn_run_schedule(elpis_ecsg_k1_fms *runtime, uint64_t id, uint64_t token, const double *x,
                                       const double *y, size_t total_rows, const elpis_ecsg_k1_experience *schedule,
                                       size_t experiences, double learning_rate, double *s3_out, size_t s3_count,
                                       elpis_ecsg_k1_schedule_result *result);

#ifdef __cplusplus
}
#endif
#endif
