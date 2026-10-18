#ifndef ELPIS_ECSG_EXECUTOR_H
#define ELPIS_ECSG_EXECUTOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ELPIS_ECSG_EXECUTOR_ABI_V1 = 1u
};

/*
 * ECS_G native executor R1 (docs/ECS_RUNTIME_R1.md).
 *
 * One executor owns one ECS_G state and everything its hot path touches:
 * the authoritative W, staging W, reusable scratch, the admitted-experience
 * capacity, the epoch and a commit generation. It computes exactly what the
 * scalar reference computes (ecsg_math.c forward, ecsg_state.c G1 step), in
 * the reference's per-element floating-point order, so its results are
 * bitwise equal to the reference's.
 *
 * QUERY (forward): one call, W + X -> out, read-only, no allocation.
 *
 * LEARN: one call runs K G1 steps. X and y are validated and admitted once
 * into executor-owned storage; the K loop runs natively on a staging W
 * (step 1 reads the authoritative W, later steps update the staging W in
 * place, which is exact because a step computes all of z before it writes
 * W and reads each weight once before writing it). Any refusal leaves the
 * authoritative W, the epoch and the generation unchanged. Success is one
 * commit by pointer exchange: W -> W', epoch + K, generation + 1. A schedule
 * applies several drives (row blocks of the admitted X, y) in order, each
 * for its own number of steps, as one transition.
 *
 * Memory: create allocates the executor and one 64-byte-aligned arena sized
 * for (dim, width, max_rows); no executor operation allocates afterwards
 * except an explicit reserve (cold path). Rows beyond max_rows are refused
 * with ELPIS_ECSG_EXEC_CAPACITY.
 *
 * Transactions (txn_*) stage a candidate in a third W buffer and commit it by
 * the same pointer exchange, refusing a candidate whose source state was
 * replaced since it began (generation check, no hashing).
 *
 * Concurrency: SINGLE_WRITER. The caller serializes every call on one
 * executor. Concurrent entry (including two concurrent queries) is detected
 * and refused with ELPIS_ECSG_EXEC_BUSY; no multi-writer atomicity and no
 * concurrent readers are claimed. Distinct executors are independent.
 *
 * The executor performs no I/O, logging, hashing, serialization or callback
 * inside query, learn or transaction operations, and retains no caller
 * pointer. Snapshots use the ecsg_state snapshot format byte for byte.
 */

typedef enum {
    ELPIS_ECSG_EXEC_OK = 0,
    ELPIS_ECSG_EXEC_INVALID = -1,    /* bad arguments, handle or token */
    ELPIS_ECSG_EXEC_NONFINITE = -2,  /* non-finite input or intermediate (same rule as the reference) */
    ELPIS_ECSG_EXEC_STALE = -3,      /* the transaction's source state was replaced by another commit */
    ELPIS_ECSG_EXEC_BUSY = -4,       /* concurrent entry, or a reserve while a transaction is open */
    ELPIS_ECSG_EXEC_CAPACITY = -5,   /* more rows than the admitted capacity */
    ELPIS_ECSG_EXEC_NOMEM = -6       /* allocation failed (create, restore, reserve only) */
} elpis_ecsg_exec_status;

typedef struct elpis_ecsg_executor elpis_ecsg_executor;

/* One block of the admitted experience: `rows` consecutive rows of X and y,
 * applied for `steps` G1 steps (steps >= 1). */
typedef struct {
    size_t rows;
    uint64_t steps;
} elpis_ecsg_drive;

typedef struct {
    uint64_t epoch_before;
    uint64_t epoch_after;
    uint64_t generation_before;
    uint64_t generation_after;
    uint64_t steps;        /* steps applied by this transition */
    uint64_t failed_step;  /* 1-based step that refused (nothing committed); 0 otherwise */
} elpis_ecsg_exec_transition;

typedef struct {
    size_t workspace_bytes;      /* arena bytes currently owned */
    size_t max_rows;             /* admitted-experience capacity */
    uint64_t heap_allocations;   /* allocations made by this executor (create and reserve only) */
    uint64_t forward_calls;
    uint64_t learn_calls;        /* direct and transactional learn calls */
    uint64_t steps_executed;     /* G1 steps computed, committed or not */
    uint64_t commits;
    uint64_t refusals;           /* learn/forward calls refused after admission (non-finite) */
    uint64_t txn_begins;
    uint64_t txn_aborts;
    uint64_t stale_refusals;
    uint64_t busy_refusals;
} elpis_ecsg_exec_stats;

uint32_t elpis_ecsg_executor_abi_version(void);

/* Arena bytes create would allocate (0 on invalid shape or overflow). */
size_t elpis_ecsg_executor_workspace_bytes(size_t dim, size_t width, size_t max_rows);

elpis_ecsg_exec_status
elpis_ecsg_executor_create(size_t dim,
                           size_t width,
                           size_t max_rows,
                           const double *initial_w,
                           elpis_ecsg_executor **out);

/* Restores an ecsg_state snapshot (same bytes, same validation). */
elpis_ecsg_exec_status
elpis_ecsg_executor_restore(const uint8_t *snapshot,
                            size_t snapshot_size,
                            size_t max_rows,
                            elpis_ecsg_executor **out);

/* Frees the executor and sets *exec to NULL. Refused (BUSY) while in use. */
elpis_ecsg_exec_status elpis_ecsg_executor_destroy(elpis_ecsg_executor **exec);

/* Cold path: grows the admitted capacity to at least max_rows (one new
 * arena; W is preserved). Refused while a transaction is open. */
elpis_ecsg_exec_status elpis_ecsg_executor_reserve(elpis_ecsg_executor *exec, size_t max_rows);

/* Accessors are not entry-guarded; under SINGLE_WRITER the caller serializes
 * them with every other call. They return 0 for NULL. */
size_t elpis_ecsg_executor_dim(const elpis_ecsg_executor *exec);
size_t elpis_ecsg_executor_width(const elpis_ecsg_executor *exec);
size_t elpis_ecsg_executor_max_rows(const elpis_ecsg_executor *exec);
uint64_t elpis_ecsg_executor_epoch(const elpis_ecsg_executor *exec);
uint64_t elpis_ecsg_executor_generation(const elpis_ecsg_executor *exec);

/* QUERY: out[r] = f_W(x_r) for rows x (rows x dim, row-major). Read-only. On
 * refusal the contents of out are unspecified. */
elpis_ecsg_exec_status
elpis_ecsg_executor_forward(elpis_ecsg_executor *exec,
                            const double *x,
                            size_t rows,
                            double *out);

/* LEARN: `steps` G1 steps on (x, y), committed as one transition. As in the
 * reference, a negative or non-finite learning rate and an epoch that would
 * pass UINT64_MAX are INVALID. */
elpis_ecsg_exec_status
elpis_ecsg_executor_learn(elpis_ecsg_executor *exec,
                          const double *x,
                          const double *y,
                          size_t rows,
                          double learning_rate,
                          uint64_t steps,
                          elpis_ecsg_exec_transition *transition);

/* LEARN over an ordered schedule of drives; x and y hold the drives' rows
 * back to back (total rows <= max_rows). One transition. */
elpis_ecsg_exec_status
elpis_ecsg_executor_learn_schedule(elpis_ecsg_executor *exec,
                                   const double *x,
                                   const double *y,
                                   const elpis_ecsg_drive *drives,
                                   size_t drive_count,
                                   double learning_rate,
                                   elpis_ecsg_exec_transition *transition);

elpis_ecsg_exec_status
elpis_ecsg_executor_copy_w(elpis_ecsg_executor *exec, double *out, size_t out_count);

elpis_ecsg_exec_status
elpis_ecsg_executor_project_s3(elpis_ecsg_executor *exec,
                               double *mu,
                               double *m_packed,
                               double *t3_packed);

size_t elpis_ecsg_executor_snapshot_size(const elpis_ecsg_executor *exec);

elpis_ecsg_exec_status
elpis_ecsg_executor_snapshot_write(elpis_ecsg_executor *exec, uint8_t *out, size_t out_size);

elpis_ecsg_exec_status
elpis_ecsg_executor_stats(elpis_ecsg_executor *exec, elpis_ecsg_exec_stats *out);

/*
 * Native candidate transaction: at most one open per executor.
 *
 * begin captures the source generation and returns a token. The candidate
 * starts as the authoritative W; txn_learn advances it (K steps or a
 * schedule) in the executor's candidate buffer, never touching the
 * authoritative W; txn_forward and txn_project_s3 read it. commit installs
 * the candidate by pointer exchange (epoch = candidate epoch, generation + 1)
 * if no other commit replaced the source since begin; otherwise the
 * transaction is discarded and STALE is returned. Every transaction call
 * checks staleness first. A refused txn_learn discards the transaction.
 * abort discards it; aborting a token that is no longer open is a no-op.
 * Committing a transaction that learned nothing changes nothing. A direct
 * learn while a transaction is open is allowed and makes it stale. begin
 * while a transaction is open, and reserve, are refused with BUSY.
 */
elpis_ecsg_exec_status elpis_ecsg_executor_txn_begin(elpis_ecsg_executor *exec, uint64_t *token);

elpis_ecsg_exec_status
elpis_ecsg_executor_txn_learn(elpis_ecsg_executor *exec,
                              uint64_t token,
                              const double *x,
                              const double *y,
                              size_t rows,
                              double learning_rate,
                              uint64_t steps,
                              elpis_ecsg_exec_transition *transition);

elpis_ecsg_exec_status
elpis_ecsg_executor_txn_learn_schedule(elpis_ecsg_executor *exec,
                                       uint64_t token,
                                       const double *x,
                                       const double *y,
                                       const elpis_ecsg_drive *drives,
                                       size_t drive_count,
                                       double learning_rate,
                                       elpis_ecsg_exec_transition *transition);

elpis_ecsg_exec_status
elpis_ecsg_executor_txn_forward(elpis_ecsg_executor *exec,
                                uint64_t token,
                                const double *x,
                                size_t rows,
                                double *out);

elpis_ecsg_exec_status
elpis_ecsg_executor_txn_project_s3(elpis_ecsg_executor *exec,
                                   uint64_t token,
                                   double *mu,
                                   double *m_packed,
                                   double *t3_packed);

/* Candidate epoch (source epoch + steps learned in the transaction). */
elpis_ecsg_exec_status
elpis_ecsg_executor_txn_epoch(elpis_ecsg_executor *exec, uint64_t token, uint64_t *epoch);

elpis_ecsg_exec_status
elpis_ecsg_executor_txn_commit(elpis_ecsg_executor *exec,
                               uint64_t token,
                               elpis_ecsg_exec_transition *transition);

elpis_ecsg_exec_status elpis_ecsg_executor_txn_abort(elpis_ecsg_executor *exec, uint64_t token);

#ifdef __cplusplus
}
#endif

#endif
