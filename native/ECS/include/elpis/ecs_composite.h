#ifndef ELPIS_ECS_COMPOSITE_H
#define ELPIS_ECS_COMPOSITE_H

#include <stddef.h>
#include <stdint.h>

#include "elpis/ecs_memory.h"
#include "elpis/ecsg_k1.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The ECS composite retained state (docs/ECS_COMPOSITE_STATE.md): the one ECS state is the qualified K1 state
 * (W, epoch, H, a) together with the ECS memory. One object owns both, mutates both only inside one composite
 * transaction, and identifies both with one digest. Neither component is reachable from outside: the K1 state is
 * held under an internal managed lease (every unmanaged K1 mutation is refused LEASED) and the memory handle is
 * private.
 *
 * Retained-state envelope ELPISEC1 v1 (little-endian, fixed size for a configuration):
 *
 *   0    magic "ELPISEC1"
 *   8    format version (u32) = 1            12   flags (u32) = 0
 *   16   K1 envelope bytes (u64)  = elpis_ecsg_k1_envelope_bytes(dim, width)
 *   24   memory capacity (u64)    = the memory configuration's maximum ELPISEM1 size
 *   32   memory envelope bytes (u64) <= capacity
 *   40   reserved (u64) = 0
 *   48   K1 retained-state identity (32): the ELPISGK1 trailer
 *   80   memory identity (32): the ELPISEM1 trailer
 *   112  reserved (16) = 0
 *   128  the ELPISGK1 v1 envelope, unchanged
 *   ...  the ELPISEM1 v1 envelope, unchanged, then zero padding to the memory capacity
 *   end  SHA-256 of every preceding byte: the composite retained-state identity
 *
 * ELPISGK1 v1 and ELPISEM1 v1 keep their meaning and bytes; the composite identity is a new identity and never
 * a K1 identity. A bare ELPISGK1 envelope is a legacy (K1-only) retained state: it becomes composite only by the
 * explicit import below, with an empty memory.
 *
 * Transactions (one owner, one commit point; all through the leased_* entry points):
 *
 *   txn_begin            stages a K1 candidate (a leased K1 transaction) and reports the composite source identity;
 *                        the memory candidate is a copy of the committed memory, made at the first memory operation
 *   txn_run_schedule     the qualified K1 experience law on the K1 candidate (elpis_ecsg_k1_txn_run_schedule)
 *   txn_observe          OBSERVE on the memory candidate
 *   txn_memory_learn     LEARN on the memory candidate (UNQUALIFIED unless the memory admits CAP_RECURRENT)
 *   txn_memory_consolidate  CONSOLIDATE on the memory candidate (UNQUALIFIED unless it admits CAP_CONSOLIDATE)
 *   txn_snapshot_write   the candidate's complete ELPISEC1 envelope; its trailer is the identity commit reports
 *   txn_commit           commits the K1 candidate, then installs the memory candidate (a pointer exchange, which
 *                        cannot fail); reports the composite identities before and after
 *   txn_abort            discards both candidates
 *
 * A refused component operation leaves both candidates as they were (recoverable), except where K1's own
 * contract discards its transaction (STALE, NONFINITE), which closes the composite transaction too. No refusal
 * changes the authoritative state. Process-local atomicity only: durable publication is the checkpoint owner's
 * and continuity's (docs/ECS_COMPOSITE_STATE.md, "Authority").
 *
 * Concurrency: one writer XOR many readers per object (non-blocking; conflicting entry is BUSY). A transaction
 * holds the writer role from begin to commit or abort only through its calls, not between them: queries between
 * transaction calls read the committed state.
 */

enum {
    ELPIS_ECS_COMPOSITE_ABI_V1 = 1u,
    ELPIS_ECS_COMPOSITE_FORMAT_VERSION = 1u,
    ELPIS_ECS_COMPOSITE_HEADER_BYTES = 128u,
    ELPIS_ECS_COMPOSITE_DIGEST_BYTES = 32u,
    ELPIS_ECS_COMPOSITE_MAX_MEMORY_BYTES = 67108864u /* 64 MiB: the largest ELPISEM1 capacity a composite admits */
};
/* The largest ELPISEC1 envelope: header, the largest K1 envelope, the largest memory capacity, the trailer. */
#define ELPIS_ECS_COMPOSITE_MAX_ENVELOPE_BYTES                                                                 \
    ((uint64_t)ELPIS_ECS_COMPOSITE_HEADER_BYTES + (uint64_t)ELPIS_ECSG_K1_MAX_IMAGE_BYTES +                    \
     (uint64_t)ELPIS_ECSG_K1_DIGEST_BYTES + (uint64_t)ELPIS_ECS_COMPOSITE_MAX_MEMORY_BYTES +                   \
     (uint64_t)ELPIS_ECS_COMPOSITE_DIGEST_BYTES)

/* Status values extend the K1 status space with the same numbers (a RuntimeCore substrate table can drive a
 * composite exactly as it drives K1), then add the memory's own refusals. */
typedef enum {
    ELPIS_ECS_COMPOSITE_OK = 0,
    ELPIS_ECS_COMPOSITE_INVALID = -1,     /* bad argument, configuration, token or record */
    ELPIS_ECS_COMPOSITE_NONFINITE = -2,   /* non-finite K1 input or arithmetic: the transaction is discarded */
    ELPIS_ECS_COMPOSITE_STALE = -3,       /* the transaction's source was replaced: the transaction is discarded */
    ELPIS_ECS_COMPOSITE_BUSY = -4,        /* concurrent entry, or a transaction is (not) open as required */
    ELPIS_ECS_COMPOSITE_CAPACITY = -5,    /* beyond a declared bound (rows, memory capacity, envelope size) */
    ELPIS_ECS_COMPOSITE_NOMEM = -6,       /* allocation failed (create, restore, import, reserve only) */
    ELPIS_ECS_COMPOSITE_CORRUPT = -7,     /* an envelope failed validation */
    ELPIS_ECS_COMPOSITE_LEASED = -8,      /* the caller's lease is not the held lease */
    ELPIS_ECS_COMPOSITE_UNQUALIFIED = -9, /* a memory operation or content the configuration does not admit */
    ELPIS_ECS_COMPOSITE_REFUSED = -10     /* the memory refused (its status in *memory_status); candidates unchanged */
} elpis_ecs_composite_status;

typedef struct elpis_ecs_composite elpis_ecs_composite;

typedef struct {
    uint32_t abi_version;                 /* ELPIS_ECS_COMPOSITE_ABI_V1 */
    uint32_t reserved;                    /* 0 */
    uint64_t dim, width, max_rows;        /* the K1 shape and its row reservation (max_rows is not state) */
    elpis_ecs_mem_config memory;          /* the memory configuration (capabilities and byte_budget: not state) */
} elpis_ecs_composite_config;

/* A committed composite transition: the composite identities and each component's. */
typedef struct {
    elpis_ecsg_k1_transition k1;          /* the K1 transition (epochs, generations, steps) */
    uint8_t before[ELPIS_ECS_COMPOSITE_DIGEST_BYTES];
    uint8_t after[ELPIS_ECS_COMPOSITE_DIGEST_BYTES];
    uint8_t k1_before[ELPIS_ECS_COMPOSITE_DIGEST_BYTES];
    uint8_t k1_after[ELPIS_ECS_COMPOSITE_DIGEST_BYTES];
    uint8_t memory_before[ELPIS_ECS_COMPOSITE_DIGEST_BYTES];
    uint8_t memory_after[ELPIS_ECS_COMPOSITE_DIGEST_BYTES];
    uint32_t memory_changed;              /* 1 when the transaction changed the memory */
    uint32_t reserved;
} elpis_ecs_composite_commit;

typedef struct {
    uint64_t envelope_bytes;              /* the fixed ELPISEC1 size of this configuration */
    uint64_t k1_envelope_bytes;
    uint64_t memory_capacity_bytes;       /* the largest ELPISEM1 of the memory configuration */
    uint64_t memory_state_bytes;          /* the current ELPISEM1 size */
    uint64_t allocated_bytes;             /* object + K1 workspace + two memory arenas + staging buffer */
    uint64_t k1_workspace_bytes;
    uint64_t memory_arena_bytes;          /* one memory arena (two are held: committed and candidate) */
    uint64_t staging_bytes;               /* the envelope staging buffer (= envelope_bytes) */
    uint64_t k1_epoch;
    uint64_t memory_clock;
    uint32_t memory_slots_used, memory_codebook_used;
    uint32_t capabilities;                /* the memory capabilities this composite admits (policy) */
    uint32_t txn_open, lease_held, reserved;
    /* Non-authoritative telemetry. */
    uint64_t commits, aborts, refusals, memory_observes, memory_learns, memory_consolidations;
} elpis_ecs_composite_stats;

const char *elpis_ecs_composite_code_name(int status);
/* The fixed envelope size of a valid configuration (0 when invalid or beyond the declared bounds). */
size_t elpis_ecs_composite_envelope_bytes(const elpis_ecs_composite_config *config);

/* A fresh composite: K1 from initial_w (dim * width values, elpis_ecsg_k1_create) and an empty memory. */
elpis_ecs_composite_status elpis_ecs_composite_create(const elpis_ecs_composite_config *config, const double *initial_w,
                                                      elpis_ecs_composite **out);
/* Restore an ELPISEC1 envelope, validated completely before anything is returned (header, sizes, reserved fields,
 * padding, trailer, both components and their recorded identities). The caller's configuration must match the
 * envelope's K1 shape and memory configuration; memory content the configuration does not admit is UNQUALIFIED. */
elpis_ecs_composite_status elpis_ecs_composite_restore(const elpis_ecs_composite_config *config, const uint8_t *envelope,
                                                       size_t length, elpis_ecs_composite **out);
/* LEGACY: an ELPISGK1 v1 (K1-only) retained state becomes a composite with an empty memory. Explicit, never
 * implicit: restore refuses an ELPISGK1 envelope (CORRUPT). The result has a new (composite) identity. */
elpis_ecs_composite_status elpis_ecs_composite_import_k1(const elpis_ecs_composite_config *config,
                                                         const uint8_t *k1_envelope, size_t length,
                                                         elpis_ecs_composite **out);
/* Destroys the composite (aborting any open transaction). */
void elpis_ecs_composite_destroy(elpis_ecs_composite *composite);

/* ---- Reads (any caller; refused BUSY only while a writer call is inside) ---------------------------------- */

/* The composite retained-state identity: the ELPISEC1 trailer of the committed state. */
elpis_ecs_composite_status elpis_ecs_composite_state_digest(elpis_ecs_composite *composite,
                                                            uint8_t out[ELPIS_ECS_COMPOSITE_DIGEST_BYTES]);
elpis_ecs_composite_status elpis_ecs_composite_shape(elpis_ecs_composite *composite, size_t *dim, size_t *width);
/* K1 QUERY bound to the composite identity of the same committed state (one guarded call). */
elpis_ecs_composite_status elpis_ecs_composite_query_identity(elpis_ecs_composite *composite, size_t dim, const double *x,
                                                              size_t rows, double *out,
                                                              uint8_t digest[ELPIS_ECS_COMPOSITE_DIGEST_BYTES]);
/* Memory QUERY (read-only, elpis_ecs_mem_query) bound to the composite identity. A memory refusal is REFUSED with
 * the memory status in *memory_status (UNQUALIFIED is reported as itself). */
elpis_ecs_composite_status elpis_ecs_composite_memory_query(elpis_ecs_composite *composite,
                                                            const elpis_ecs_mem_request *request,
                                                            elpis_ecs_mem_answer *answer,
                                                            uint8_t digest[ELPIS_ECS_COMPOSITE_DIGEST_BYTES],
                                                            int *memory_status);
elpis_ecs_composite_status elpis_ecs_composite_memory_slot_read(elpis_ecs_composite *composite,
                                                                const elpis_ecs_mem_digest *key, elpis_ecs_mem_slot *out,
                                                                int *memory_status);
/* The committed ELPISEC1 envelope (size >= envelope_bytes). */
elpis_ecs_composite_status elpis_ecs_composite_snapshot_write(elpis_ecs_composite *composite, uint8_t *out, size_t size);
elpis_ecs_composite_status elpis_ecs_composite_inspect(elpis_ecs_composite *composite, elpis_ecs_composite_stats *out);

/* ---- Managed ownership -----------------------------------------------------------------------------------
 * Every mutating entry point takes the caller's lease and is refused LEASED unless it equals the held lease (0
 * when none is held). lease_claim (nonzero) replaces any lease; claim and release are BUSY while a transaction
 * is open. The lease is an aliasing defence between cooperating local callers, not state and not a secret. */
elpis_ecs_composite_status elpis_ecs_composite_lease_claim(elpis_ecs_composite *composite, uint64_t lease);
elpis_ecs_composite_status elpis_ecs_composite_lease_release(elpis_ecs_composite *composite, uint64_t lease);
uint64_t elpis_ecs_composite_lease_of(const elpis_ecs_composite *composite);
elpis_ecs_composite_status elpis_ecs_composite_leased_reserve(elpis_ecs_composite *composite, uint64_t lease,
                                                              size_t max_rows);

/* ---- Transactions ------------------------------------------------------------------------------------------
 * Refusal contract (as K1's): INVALID, CAPACITY, BUSY, LEASED, UNQUALIFIED and REFUSED are recoverable: refused
 * before either candidate is touched, the transaction stays open and unchanged. STALE and NONFINITE (from the K1
 * schedule) discard the whole composite transaction. No refusal changes the committed state. */
elpis_ecs_composite_status elpis_ecs_composite_leased_txn_begin(elpis_ecs_composite *composite, uint64_t lease,
                                                                uint8_t source_digest[ELPIS_ECS_COMPOSITE_DIGEST_BYTES],
                                                                uint64_t *token);
elpis_ecs_composite_status
elpis_ecs_composite_leased_txn_run_schedule(elpis_ecs_composite *composite, uint64_t lease, uint64_t token,
                                            const double *x, const double *y, size_t total_rows,
                                            const elpis_ecsg_k1_experience *schedule, size_t experiences,
                                            double learning_rate, double *s3_out, size_t s3_count,
                                            elpis_ecsg_k1_schedule_result *result);
/* OBSERVE on the memory candidate. out_key optional. */
elpis_ecs_composite_status elpis_ecs_composite_leased_txn_observe(elpis_ecs_composite *composite, uint64_t lease,
                                                                  uint64_t token,
                                                                  const elpis_ecs_mem_observation *observation,
                                                                  elpis_ecs_mem_digest *out_key, int *memory_status);
/* LEARN / CONSOLIDATE on the memory candidate (UNQUALIFIED unless the memory configuration admits them). */
elpis_ecs_composite_status elpis_ecs_composite_leased_txn_memory_learn(elpis_ecs_composite *composite, uint64_t lease,
                                                                       uint64_t token, const elpis_ecs_mem_digest *key,
                                                                       uint64_t tick, int *memory_status);
elpis_ecs_composite_status elpis_ecs_composite_leased_txn_memory_consolidate(elpis_ecs_composite *composite,
                                                                             uint64_t lease, uint64_t token,
                                                                             uint64_t tick, uint32_t max_items,
                                                                             uint32_t *replayed, int *memory_status);
/* The candidate's complete ELPISEC1 envelope (read-only; size >= envelope_bytes). Its trailer is the identity the
 * commit will report. */
elpis_ecs_composite_status elpis_ecs_composite_txn_snapshot_write(elpis_ecs_composite *composite, uint64_t token,
                                                                  uint8_t *out, size_t size);
/* Commit both candidates as one transition (the K1 commit, then the memory install, which cannot fail). */
elpis_ecs_composite_status elpis_ecs_composite_leased_txn_commit(elpis_ecs_composite *composite, uint64_t lease,
                                                                 uint64_t token, elpis_ecs_composite_commit *out);
/* The same commit, reported in the K1 commit-identity layout with composite identities as the state digests (the
 * form a RuntimeCore substrate table consumes). */
elpis_ecs_composite_status elpis_ecs_composite_leased_txn_commit_identity(elpis_ecs_composite *composite,
                                                                          uint64_t lease, uint64_t token,
                                                                          elpis_ecsg_k1_commit_identity *out);
/* Discard both candidates (any caller; installs nothing). No open transaction: OK. Wrong token: INVALID. */
elpis_ecs_composite_status elpis_ecs_composite_txn_abort(elpis_ecs_composite *composite, uint64_t token);

/* ---- Envelope inspection (pure; no handle) ------------------------------------------------------------------ */

/* Verify an ELPISEC1 envelope's structure and trailer (not its components) and report its recorded identities. */
elpis_ecs_composite_status elpis_ecs_composite_envelope_identities(const uint8_t *envelope, size_t length,
                                                                   uint8_t composite[ELPIS_ECS_COMPOSITE_DIGEST_BYTES],
                                                                   uint8_t k1[ELPIS_ECS_COMPOSITE_DIGEST_BYTES],
                                                                   uint8_t memory[ELPIS_ECS_COMPOSITE_DIGEST_BYTES]);

#ifdef __cplusplus
}
#endif
#endif
