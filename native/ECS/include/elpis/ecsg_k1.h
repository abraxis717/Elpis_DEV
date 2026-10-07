#ifndef ELPIS_ECSG_K1_H
#define ELPIS_ECSG_K1_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ECS native consolidation runtime: K1 (docs/ECS_K1_RUNTIME.md).
 *
 * The qualified mechanism is Retention R3's K1 (OUTCOME_A,
 * docs/research/ECS_RETENTION_R3_RESULTS.md). One K1 state owns the complete
 * cognitive state
 *
 *     W[dim, width]   epoch   H (packed symmetric F x F)   a[F]
 *
 * with F = |S3| (83 at dim 6), plus non-authoritative scratch and a candidate
 * for transactions. The law, exactly as qualified:
 *
 *   CONSOLIDATE(X):  H <- H + Sigma(X),  Sigma = (1/n) sum_r phi(x_r) phi(x_r)^T
 *                    a <- S3(W)          (no target ever enters)
 *   LEARN step:      W <- G1(W; X, y) - eta J(W)^T u,  u = 1/2 H (S3(W) - a),
 *                    u evaluated at the pre-step W
 *   QUERY:           f_W(X), reads W only; H and a never touch a query
 *
 * G1 and the forward map are computed in the reference's per-element order, so
 * a state whose H is zero (fresh, reset, or a W-only import) learns and answers
 * bitwise as the Runtime R1 executor does.
 *
 * QUERY is one call (state + X -> f_W(X)), read-only, no allocation. LEARN is
 * one call for K steps (no per-step crossing): X and y are admitted once, the K
 * loop runs natively on staging buffers, any refusal leaves the complete state
 * unchanged, success commits W' and epoch + K together. CONSOLIDATE is one call
 * and commits H' and a' together. A transaction stages a complete candidate
 * (W, epoch, H, a) and commits all four at once, or nothing.
 *
 * Memory: create/restore allocate the object and one 64-byte-aligned arena;
 * no operation allocates afterwards except an explicit reserve (cold path).
 * Every size is computed with checked arithmetic and bounded before any
 * allocation: an image (header + W + H + a) above ELPIS_ECSG_K1_MAX_IMAGE_BYTES
 * or a workspace above ELPIS_ECSG_K1_MAX_WORKSPACE_BYTES is refused with
 * CAPACITY, whatever a caller or a serialized header asks for.
 *
 * Concurrency: SINGLE_WRITER per state; every operation (queries included)
 * enters one guard and a concurrent entry is refused (BUSY). The getters
 * epoch, generation, provenance_of and max_rows never enter it: they read
 * values the writer publishes atomically after each committed transition (each
 * value is current as of some committed transition; they are not a joint
 * snapshot). dim and width are immutable. Distinct states share nothing.
 *
 * Provenance: create, restore of a COMPLETE envelope and every successful
 * consolidation (direct or committed) give COMPLETE; reset gives RESET; a
 * W-only import gives UNCONSOLIDATED_IMPORT. LEARN never changes provenance.
 *
 * Persistent format (ELPIS K1 retained-state envelope v1), all integers and
 * binary64 values little-endian, explicit layout:
 *
 *   0   magic "ELPISGK1"                 8 bytes
 *   8   version (u32) = 1                12  mechanism (u32) = 1 (K1)
 *   16  dim (u64)    24 width (u64)      32  features F (u64)
 *   40  epoch (u64)  48 provenance (u64) 56  reserved (u64) = 0
 *   64  W (dim*width f64) | H packed upper, row-major (F(F+1)/2 f64) | a (F f64)
 *   end SHA-256 of every preceding byte (32 bytes)
 *
 * Restore refuses a wrong magic, version, mechanism, shape, size, provenance,
 * reserved field, checksum or any non-finite value. A canonical ELPISG01
 * W-only snapshot is not a retained state: import_w_only admits it only as an
 * UNCONSOLIDATED state (H = 0, a = 0, provenance UNCONSOLIDATED_IMPORT).
 */

enum {
    ELPIS_ECSG_K1_ABI_V1 = 1u,
    ELPIS_ECSG_K1_FORMAT_VERSION = 1u,
    ELPIS_ECSG_K1_MECHANISM = 1u,
    ELPIS_ECSG_K1_HEADER_BYTES = 64u,
    ELPIS_ECSG_K1_DIGEST_BYTES = 32u,
    ELPIS_ECSG_K1_MAX_DIM = 64u,                   /* index-table bound; the byte budgets bind first */
    ELPIS_ECSG_K1_MAX_IMAGE_BYTES = 67108864u,     /* 64 MiB: header + W + H packed + a */
    ELPIS_ECSG_K1_MAX_WORKSPACE_BYTES = 268435456u /* 256 MiB: scratch, candidate, admitted rows */
};

typedef enum {
    ELPIS_ECSG_K1_OK = 0,
    ELPIS_ECSG_K1_INVALID = -1,    /* bad arguments, handle, shape or token */
    ELPIS_ECSG_K1_NONFINITE = -2,  /* non-finite input or intermediate */
    ELPIS_ECSG_K1_STALE = -3,      /* the transaction's source was replaced */
    ELPIS_ECSG_K1_BUSY = -4,       /* concurrent entry, or reserve with an open transaction */
    ELPIS_ECSG_K1_CAPACITY = -5,   /* more rows than reserved, or a shape beyond the declared byte budgets */
    ELPIS_ECSG_K1_NOMEM = -6,      /* allocation failed (create, restore, reserve only) */
    ELPIS_ECSG_K1_CORRUPT = -7     /* a retained-state envelope failed validation */
} elpis_ecsg_k1_status;

typedef enum {
    ELPIS_ECSG_K1_COMPLETE = 0,               /* created, restored or consolidated */
    ELPIS_ECSG_K1_RESET = 1,                  /* H and a emptied by reset */
    ELPIS_ECSG_K1_UNCONSOLIDATED_IMPORT = 2   /* imported from a W-only ELPISG01 snapshot */
} elpis_ecsg_k1_provenance;

typedef struct elpis_ecsg_k1 elpis_ecsg_k1;

typedef struct {
    uint64_t epoch_before;
    uint64_t epoch_after;
    uint64_t generation_before;
    uint64_t generation_after;
    uint64_t steps;        /* steps applied by this transition */
    uint64_t failed_step;  /* 1-based step that refused (nothing committed); 0 otherwise */
} elpis_ecsg_k1_transition;


typedef struct {
    elpis_ecsg_k1_transition transition;
    uint8_t state_before_digest[ELPIS_ECSG_K1_DIGEST_BYTES];
    uint8_t state_after_digest[ELPIS_ECSG_K1_DIGEST_BYTES];
} elpis_ecsg_k1_commit_identity;

typedef struct {
    size_t workspace_bytes;      /* arena bytes currently owned */
    size_t max_rows;
    uint64_t heap_allocations;   /* create, restore and reserve only */
    uint64_t forward_calls;
    uint64_t learn_calls;
    uint64_t steps_executed;
    uint64_t corrected_steps;    /* steps that applied a non-zero K1 correction */
    uint64_t consolidations;
    uint64_t commits;
    uint64_t refusals;
    uint64_t txn_begins;
    uint64_t txn_aborts;
    uint64_t stale_refusals;
    uint64_t busy_refusals;
} elpis_ecsg_k1_counters;

uint32_t elpis_ecsg_k1_abi_version(void);

/* Sizes (0 on an invalid shape, an overflow, or a size beyond the declared budgets). */
size_t elpis_ecsg_k1_features(size_t dim);
size_t elpis_ecsg_k1_payload_bytes(size_t dim, size_t width);   /* W, H packed, a */
size_t elpis_ecsg_k1_image_bytes(size_t dim, size_t width);     /* header + payload */
size_t elpis_ecsg_k1_envelope_bytes(size_t dim, size_t width);  /* image + SHA-256 */
size_t elpis_ecsg_k1_workspace_bytes(size_t dim, size_t width, size_t max_rows);

/* A complete state from W with an empty consolidation (H = 0, a = 0). */
elpis_ecsg_k1_status
elpis_ecsg_k1_create(size_t dim, size_t width, size_t max_rows, const double *initial_w, elpis_ecsg_k1 **out);

/* A complete state from a retained-state envelope (validated). */
elpis_ecsg_k1_status
elpis_ecsg_k1_restore(const uint8_t *envelope, size_t size, size_t max_rows, elpis_ecsg_k1 **out);

/* A canonical ELPISG01 W-only snapshot, admitted only as UNCONSOLIDATED. */
elpis_ecsg_k1_status
elpis_ecsg_k1_import_w_only(const uint8_t *snapshot, size_t size, size_t max_rows, elpis_ecsg_k1 **out);

elpis_ecsg_k1_status elpis_ecsg_k1_destroy(elpis_ecsg_k1 **state);
elpis_ecsg_k1_status elpis_ecsg_k1_reserve(elpis_ecsg_k1 *state, size_t max_rows);

size_t elpis_ecsg_k1_dim(const elpis_ecsg_k1 *state);
size_t elpis_ecsg_k1_width(const elpis_ecsg_k1 *state);
size_t elpis_ecsg_k1_max_rows(const elpis_ecsg_k1 *state);
uint64_t elpis_ecsg_k1_epoch(const elpis_ecsg_k1 *state);
uint64_t elpis_ecsg_k1_generation(const elpis_ecsg_k1 *state);
uint32_t elpis_ecsg_k1_provenance_of(const elpis_ecsg_k1 *state);

/* QUERY: out[r] = f_W(x_r). Reads W only. */
elpis_ecsg_k1_status
elpis_ecsg_k1_forward(elpis_ecsg_k1 *state, const double *x, size_t rows, double *out);

/* LEARN: `steps` K1 steps on (x, y), committed as one transition (W, epoch). */
elpis_ecsg_k1_status
elpis_ecsg_k1_learn(elpis_ecsg_k1 *state, const double *x, const double *y, size_t rows,
                    double learning_rate, uint64_t steps, elpis_ecsg_k1_transition *transition);

/* CONSOLIDATE: H <- H + Sigma(x), a <- S3(W). Inputs only; committed together. */
elpis_ecsg_k1_status
elpis_ecsg_k1_consolidate(elpis_ecsg_k1 *state, const double *x, size_t rows,
                          elpis_ecsg_k1_transition *transition);

/* RESET: H <- 0, a <- 0; W and epoch kept; provenance RESET. */
elpis_ecsg_k1_status elpis_ecsg_k1_reset(elpis_ecsg_k1 *state, elpis_ecsg_k1_transition *transition);

/* Copies of the authoritative state (cold path; for qualification and export). */
elpis_ecsg_k1_status elpis_ecsg_k1_copy_w(elpis_ecsg_k1 *state, double *out, size_t count);
elpis_ecsg_k1_status elpis_ecsg_k1_copy_h_packed(elpis_ecsg_k1 *state, double *out, size_t count);
elpis_ecsg_k1_status elpis_ecsg_k1_copy_a(elpis_ecsg_k1 *state, double *out, size_t count);

size_t elpis_ecsg_k1_snapshot_size(const elpis_ecsg_k1 *state);
elpis_ecsg_k1_status elpis_ecsg_k1_snapshot_write(elpis_ecsg_k1 *state, uint8_t *out, size_t size);
elpis_ecsg_k1_status elpis_ecsg_k1_state_digest(
    elpis_ecsg_k1 *state,
    uint8_t out[ELPIS_ECSG_K1_DIGEST_BYTES]
);

elpis_ecsg_k1_status elpis_ecsg_k1_stats(elpis_ecsg_k1 *state, elpis_ecsg_k1_counters *out);

/*
 * Transactions: at most one open per state. begin stages a candidate copy of
 * the complete state; learn, consolidate and forward act on the candidate;
 * commit installs W, epoch, H and a together (generation + 1) unless another
 * commit replaced the source since begin (STALE, candidate discarded).
 *
 * Refusal contract (the status decides; identical in the FMS adapter and the
 * Python control plane):
 *   INVALID, CAPACITY, BUSY   recoverable: refused before the candidate is
 *                             touched; the transaction stays open and unchanged
 *                             and the call may be retried.
 *   STALE                     fatal: the transaction is discarded.
 *   NONFINITE                 fatal from txn_learn and txn_consolidate (non-finite
 *                             input or arithmetic): discarded. A NONFINITE query
 *                             (txn_forward) is read-only and discards nothing.
 * The authoritative state never changes on any refusal. abort with no open
 * transaction is OK; abort with a wrong token is INVALID and changes nothing.
 */
elpis_ecsg_k1_status elpis_ecsg_k1_txn_begin(elpis_ecsg_k1 *state, uint64_t *token);
elpis_ecsg_k1_status
elpis_ecsg_k1_txn_learn(elpis_ecsg_k1 *state, uint64_t token, const double *x, const double *y, size_t rows,
                        double learning_rate, uint64_t steps, elpis_ecsg_k1_transition *transition);
elpis_ecsg_k1_status
elpis_ecsg_k1_txn_consolidate(elpis_ecsg_k1 *state, uint64_t token, const double *x, size_t rows);
elpis_ecsg_k1_status
elpis_ecsg_k1_txn_forward(elpis_ecsg_k1 *state, uint64_t token, const double *x, size_t rows, double *out);
elpis_ecsg_k1_status elpis_ecsg_k1_txn_epoch(elpis_ecsg_k1 *state, uint64_t token, uint64_t *epoch);
elpis_ecsg_k1_status
elpis_ecsg_k1_txn_commit(elpis_ecsg_k1 *state, uint64_t token, elpis_ecsg_k1_transition *transition);
elpis_ecsg_k1_status
elpis_ecsg_k1_txn_commit_identity(elpis_ecsg_k1 *state, uint64_t token,
                                  elpis_ecsg_k1_commit_identity *identity);
elpis_ecsg_k1_status elpis_ecsg_k1_txn_abort(elpis_ecsg_k1 *state, uint64_t token);

/*
 * Experience schedule (the qualified K1 experience law, Retention R3): an
 * ordered sequence of experiences; for each, `steps` K1 learning steps on its
 * rows, then the consolidation of the same rows (inputs only):
 *
 *     for t in 1..n:   W, epoch <- learn(X_t, y_t, steps_t)    (K1 law)
 *                      H <- H + Sigma(X_t),  a <- S3(W)        (consolidation)
 *
 * txn_run_schedule applies the whole schedule to the open transaction's
 * candidate in one native call, then writes S3 of the final candidate W
 * (the readout) to s3_out (s3_count = F). X holds every experience's rows in
 * order (total_rows x dim, row-major), y their targets; experience t uses the
 * next schedule[t].rows rows. The authoritative state is not touched: commit
 * or abort decides.
 *
 * The complete schedule is validated before the candidate is touched
 * (pointers, 1 <= experiences <= ELPIS_ECSG_K1_MAX_EXPERIENCES, every rows and
 * steps >= 1, rows <= max_rows, the rows summing exactly to total_rows, the
 * total steps fitting the epoch, s3_count = F, a finite rate >= 0, every
 * input finite), all with checked arithmetic. INVALID and CAPACITY found there
 * are recoverable: the transaction stays open and unchanged. NONFINITE input,
 * or NONFINITE arithmetic in any experience, discards the transaction (the
 * refusal contract above); result->failed_experience names the experience
 * (1-based), failed_step the step within it (0: its consolidation).
 */
enum { ELPIS_ECSG_K1_MAX_EXPERIENCES = 64u };

typedef struct {
    uint64_t rows;    /* consecutive rows of X and y */
    uint64_t steps;   /* K1 learning steps on them, then one consolidation */
} elpis_ecsg_k1_experience;

typedef struct {
    uint64_t epoch_before;          /* candidate epoch before the schedule */
    uint64_t epoch_after;           /* candidate epoch after it (epoch_before on refusal) */
    uint64_t experiences_applied;   /* learned and consolidated */
    uint64_t failed_experience;     /* 1-based; 0 when none failed */
    uint64_t failed_step;           /* 1-based within the failed experience; 0 otherwise */
} elpis_ecsg_k1_schedule_result;

elpis_ecsg_k1_status
elpis_ecsg_k1_txn_run_schedule(elpis_ecsg_k1 *state, uint64_t token, const double *x, const double *y,
                               size_t total_rows, const elpis_ecsg_k1_experience *schedule, size_t experiences,
                               double learning_rate, double *s3_out, size_t s3_count,
                               elpis_ecsg_k1_schedule_result *result);

#ifdef __cplusplus
}
#endif

#endif
