#ifndef ELPIS_RUNTIME_H
#define ELPIS_RUNTIME_H

/* Elpis RuntimeCore C ABI v4 (docs/RUNTIME_CORE.md), implemented in Rust (native/runtime).
 *
 * One runtime handle owns one continuity directory and the mutable systems authority of an open Elpis
 * runtime: lifecycle, fail-stop, the binding of its K1 lineage to one native K1 state, the managed canonical
 * turn's native K1 transaction, the durability lane (explicit durable publication of the live state) and the
 * evolution attempt's durable reservation and finalization. It owns no semantics: callers hand it a validated
 * native-ready stimulus or an assertion digest.
 *
 * RuntimeCore links no ECS code: the native K1 state is reached through the caller's K1 function table (the
 * entries of ecsg_k1.h, ecsg_k1_fms.h or ecs_composite.h; a K1 status is an int-sized enum and a state handle an
 * opaque pointer). It computes no ECS mathematics and touches no ECS byte.
 *
 * Live and durable authority (v4). Two coordinates, never equated:
 *   live     the bound state's native generation (ecsg_k1.h, "Live coordinate") under this runtime's lease, as
 *            RuntimeCore last established it. Every HOT operation (QUERY, the managed turn, OBSERVE) is checked
 *            against it; none performs file I/O, computes a digest or waits for the durability lane.
 *   durable  continuity's record (and, when provisioned, the K1 checkpoint slot holding the same envelope): the
 *            last state a durable publication completed. Restart binds only a state whose identity is it.
 * A managed commit is VOLATILE: it installs the candidate and advances the live generation by one; nothing is
 * checkpointed or published. Durable publication is an explicit request of one of two kinds, never confused:
 * elpis_runtime_recoverable_request (the complete state: needs an attached checkpoint store; ACKNOWLEDGED only after
 * the slot is written, synced and read back verified, and continuity names it) or elpis_runtime_identity_request
 * (only the identity: no store attached; IDENTITY_PUBLISHED, never ACKNOWLEDGED; the caller keeps the bytes). Either
 * captures the live state (unsealed, under the lease: the one synchronous step) and hands it to the durability
 * lane, one worker thread that seals it, checkpoints it (recoverable only) and publishes continuity, without the
 * handle's lock. A crash loses at most the volatile work since the last completed publication; restart recovers
 * only a complete, authorized durable state, which may be a publication that completed just before the caller
 * lost its acknowledgment (an acknowledgment that never arrived is not proof of non-durability). v3 published one
 * continuity record (and wrote one checkpoint slot) per managed turn; that per-turn durability is superseded
 * (docs/RUNTIME_CORE.md, "Superseded"). A v3 caller refuses to load this library (the version check), so no caller
 * can mistake a volatile commit for a durable one.
 *
 * QUERY and LEARN. QUERY (elpis_runtime_query) is read-only: one native call answers f_W(x) from the lineage's live
 * state together with that state's generation, which must be the live one (otherwise the lineage is lost: fail-stop,
 * the answer withheld). It opens no transaction, commits nothing, publishes nothing, hashes nothing. LEARN is the
 * managed turn below: begin, the experience schedule on a candidate, the volatile commit.
 *
 * Managed ownership (v3). Binding a state (anchor, or the first QUERY, LEARN or durable request after open) checks
 * its retained-state identity against the durable one before claiming it (a mismatched state is never claimed),
 * claims it under this runtime's K1 lease, then reads its identity and generation together under the lease: from
 * then on the state refuses every unmanaged mutation (ELPIS_ECSG_K1_LEASED), and every managed transaction begin
 * reports the generation of exactly the source it copied. A bound state that is not provably the lineage (another
 * generation, or its lease taken by another owner) fail-stops with CONTINUITY_STATE_MISMATCH before anything is built
 * on it or published. elpis_runtime_release gives the state back (unbound, unleased); close does not touch the state.
 *
 * Turn lifecycle (v2). A successful turn_begin opens one native K1 transaction. RuntimeCore ends it with exactly one
 * terminal native action, the commit or an abort, before it forgets the turn; there is no third disposition. An
 * explicit turn_abort, a refused commit, close, and destroy all abort it; open on an open runtime is refused and
 * keeps it. An abort refused BUSY (a concurrent overlapping call on the state) is retried until the state admits it.
 * An implicit abort installs nothing and publishes nothing.
 *
 * Substrate lifetime (v2). To end the turn on every path, RuntimeCore retains the descriptor's handle, resident id
 * and txn_abort entry (copied; never any other entry) from a successful turn_begin until the turn ends. The native
 * state must stay live, and its library loaded, for that interval; outside it, a descriptor need only be live for
 * the call. The K1 FMS adapter enforces this natively: a resident state with an open transaction cannot be closed
 * and its runtime cannot be destroyed while a state is registered (both BUSY). A standalone K1 state's owner must
 * not destroy it while a turn on it is open (ecsg_k1.h frees a state whatever transaction it holds). The durability
 * lane never touches the state: it owns only the bytes a request captured (and calls the table's pure sealer).
 *
 * ECS composite substrates (docs/ECS_COMPOSITE_STATE.md). A substrate of kind ELPIS_RUNTIME_SUBSTRATE_COMPOSITE is an
 * ECS composite retained state (ecs_composite.h: K1 and the ECS memory under one ELPISEC1 identity). Its table is the
 * K1 table over composite identities plus leased_txn_observe; RuntimeCore binds, queries, learns, durably publishes
 * and recovers it exactly as a K1 state, and continuity records the lineage's format (a v3 record), so a K1 state
 * never stands for a composite lineage or the reverse (STATE_MISMATCH). The memory changes only through
 * elpis_runtime_turn_observe inside a managed turn. No memory LEARN or CONSOLIDATE is exposed (unqualified). A K1
 * lineage moves to a composite one only through the explicit elpis_runtime_import_composite. A composite is durably
 * published only with a checkpoint store attached (its envelope size is the store's provisioned capacity).
 *
 * The library also exports the continuity C ABI (elpis/continuity.h) of the store it embeds.
 *
 * Every function returns 0 or a positive code: a continuity code (1..17, 255) or a runtime code below.
 * Calls on one handle are serialized by the handle, except elpis_runtime_durable_wait and _durable_status, which
 * take only the durability lane's lock (they never wait behind a HOT operation, nor a HOT operation behind them). */

#include <stddef.h>
#include <stdint.h>

#include "elpis/continuity.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ELPIS_RUNTIME_ABI_V4 = 4u,
    ELPIS_RUNTIME_OK = 0,
    ELPIS_RUNTIME_INVALID = 64,                  /* malformed argument */
    ELPIS_RUNTIME_CLOSED = 65,                   /* the runtime is not open */
    ELPIS_RUNTIME_SUBSTRATE_SWITCH = 67,         /* COGNITION_SUBSTRATE_SWITCH: lineage bound to another state */
    ELPIS_RUNTIME_ECS_STATE = 68,                /* the K1 state could not be identified */
    ELPIS_RUNTIME_ECS_REFUSED = 69,              /* the K1 transaction refused; state unchanged */
    ELPIS_RUNTIME_ECS_STALE = 70,                /* the source moved meanwhile; nothing installed */
    ELPIS_RUNTIME_TURN_OPEN = 71,
    ELPIS_RUNTIME_TURN_NOT_OPEN = 72,
    ELPIS_RUNTIME_EVOLUTION_IN_FLIGHT = 73,
    ELPIS_RUNTIME_EVOLUTION_NOT_IN_FLIGHT = 74,
    ELPIS_RUNTIME_FUEL = 75,                     /* COGNITION_FUEL_EXCEEDED: outside the fuel budget (or the budget
                                                    outside the ceiling); refused before reserve or any transaction */
    ELPIS_RUNTIME_CHECKPOINT_INVALID = 76,       /* the K1 checkpoint store is not provisioned as required */
    ELPIS_RUNTIME_CHECKPOINT_IO = 77,            /* a checkpoint slot could not be written, synced or read back
                                                    verified; the authorized slot is intact */
    ELPIS_RUNTIME_CHECKPOINT_SHAPE = 78,         /* the envelope is not the provisioned capacity: never resized */
    ELPIS_RUNTIME_CHECKPOINT_UNRESOLVED = 79,    /* an unauthorized newer candidate awaits discard or adopt */
    ELPIS_RUNTIME_DURABLE_BUSY = 80,             /* a durable publication is pending or running */
    ELPIS_RUNTIME_DURABLE_PENDING = 81,          /* a durable wait timed out: the request is still pending */
    ELPIS_RUNTIME_RECOVERY_UNAVAILABLE = 82,     /* a recoverable request without a checkpoint store: refused */
    ELPIS_RUNTIME_CHECKPOINT_ATTACHED = 83,      /* an identity-only request with a checkpoint store: refused */
    ELPIS_RUNTIME_RECOVERY_NOTHING = 1,          /* continuity unanchored: no lineage; a slot never creates one */
    ELPIS_RUNTIME_RECOVERY_RESUMABLE = 2,        /* a verified slot holds continuity's identity; no candidate */
    ELPIS_RUNTIME_RECOVERY_CANDIDATE_UNRESOLVED = 3, /* a newer unauthorized slot exists: operator discard/adopt */
    ELPIS_RUNTIME_RECOVERY_CHECKPOINT_MISSING = 4,   /* no verified slot holds continuity's identity */
    ELPIS_RUNTIME_SUBSTRATE_K1 = 1,              /* standalone K1 state (ecsg_k1.h) */
    ELPIS_RUNTIME_SUBSTRATE_K1_FMS = 2,          /* FMS-resident K1 state (ecsg_k1_fms.h) */
    ELPIS_RUNTIME_SUBSTRATE_COMPOSITE = 3,       /* ECS composite retained state (ecs_composite.h; id zero) */
    /* elpis_runtime_durable_ticket.status */
    ELPIS_RUNTIME_DURABLE_NONE = 0,
    ELPIS_RUNTIME_DURABLE_REQUESTED = 1,         /* PENDING: captured; waiting for, or in, the worker */
    ELPIS_RUNTIME_DURABLE_ACKNOWLEDGED = 2,      /* RECOVERABLE only: a read-back verified slot holds the complete
                                                    state and continuity names it; recoverable is always 1 */
    ELPIS_RUNTIME_DURABLE_FAILED = 3,            /* nothing durable changed; the previous authority stands */
    ELPIS_RUNTIME_DURABLE_UNCERTAIN = 4,         /* outcome unknown (the record was being written or synced): lane
                                                    faulted until reopen; restart resolves it, possibly to this one */
    ELPIS_RUNTIME_DURABLE_SUPERSEDED = 5,        /* replaced by a newer request before the worker took it */
    ELPIS_RUNTIME_DURABLE_ABANDONED = 6,         /* not published: closed (or the process died) before continuity's
                                                    record was begun */
    ELPIS_RUNTIME_DURABLE_UNKNOWN = 7,           /* resolved too long ago to be recorded */
    ELPIS_RUNTIME_DURABLE_IDENTITY_PUBLISHED = 8 /* IDENTITY only: continuity names the identity; no store holds
                                                    the bytes; recoverable is always 0; never an acknowledgment */
};

/* What a durable request asks to make durable (elpis_runtime_durable_ticket.kind). */
enum {
    ELPIS_RUNTIME_DURABLE_KIND_RECOVERABLE = 1, /* the complete state (checkpoint slot, verified) and its identity */
    ELPIS_RUNTIME_DURABLE_KIND_IDENTITY = 2     /* only the identity (continuity) */
};

/* Layout-identical to elpis_ecsg_k1_experience, _schedule_result and _transition. */
typedef struct {
    uint64_t rows;
    uint64_t steps;
} elpis_runtime_experience;

typedef struct {
    uint64_t epoch_before;
    uint64_t epoch_after;
    uint64_t experiences_applied;
    uint64_t failed_experience;
    uint64_t failed_step;
} elpis_runtime_schedule_result;

typedef struct {
    uint64_t epoch_before;
    uint64_t epoch_after;
    uint64_t generation_before;
    uint64_t generation_after;
    uint64_t steps;
    uint64_t failed_step;
} elpis_runtime_transition;

/* The standalone K1 functions RuntimeCore calls (state = elpis_ecsg_k1 *), ABI v4. Every entry is required. Every
 * mutating entry is a leased one (ecsg_k1.h, "Managed ownership"); the live-coordinate entries (ecsg_k1.h, "Live
 * coordinate") report the generation under the same guard as their answer, begin, commit or capture. state_digest is
 * the one unleased read: a state is checked against the durable identity before it is ever claimed. */
typedef struct {
    int (*state_digest)(void *state, uint8_t out[32]);
    int (*shape)(void *state, size_t *dim, size_t *width);
    int (*lease_claim)(void *state, uint64_t lease);
    int (*lease_release)(void *state, uint64_t lease);
    int (*leased_reserve)(void *state, uint64_t lease, size_t max_rows);
    int (*leased_identity)(void *state, uint64_t lease, uint8_t digest[32], uint64_t *generation);
    int (*leased_query)(void *state, uint64_t lease, size_t dim, const double *x, size_t rows, double *out,
                        uint64_t *generation);
    int (*leased_txn_begin_at)(void *state, uint64_t lease, uint64_t *source_generation, uint64_t *token);
    int (*leased_txn_run_schedule)(void *state, uint64_t lease, uint64_t token, const double *x, const double *y,
                                   size_t total_rows, const elpis_runtime_experience *schedule, size_t experiences,
                                   double learning_rate, double *s3_out, size_t s3_count,
                                   elpis_runtime_schedule_result *result);
    int (*leased_txn_commit)(void *state, uint64_t lease, uint64_t token, elpis_runtime_transition *transition);
    int (*txn_abort)(void *state, uint64_t token);
    /* The unsealed envelope (ELPISGK1 / ELPISEC1 without its trailer) of the authoritative state, and the pure
     * sealer that completes it (called by the durability lane, on its own thread, over bytes it owns). */
    int (*leased_capture)(void *state, uint64_t lease, uint8_t *out, size_t size, uint64_t *generation);
    int (*envelope_seal)(uint8_t *envelope, size_t size);
} elpis_runtime_k1_api;

/* The K1 FMS adapter's functions (runtime = elpis_ecsg_k1_fms *, id = the resident state), ABI v4. */
typedef struct {
    int (*state_digest)(void *runtime, uint64_t id, uint8_t out[32]);
    int (*shape)(void *runtime, uint64_t id, size_t *dim, size_t *width);
    int (*lease_claim)(void *runtime, uint64_t id, uint64_t lease);
    int (*lease_release)(void *runtime, uint64_t id, uint64_t lease);
    int (*leased_reserve)(void *runtime, uint64_t id, uint64_t lease, size_t max_rows);
    int (*leased_identity)(void *runtime, uint64_t id, uint64_t lease, uint8_t digest[32], uint64_t *generation);
    int (*leased_query)(void *runtime, uint64_t id, uint64_t lease, size_t dim, const double *x, size_t rows,
                        double *out, uint64_t *generation);
    int (*leased_txn_begin_at)(void *runtime, uint64_t id, uint64_t lease, uint64_t *source_generation,
                               uint64_t *token);
    int (*leased_txn_run_schedule)(void *runtime, uint64_t id, uint64_t lease, uint64_t token, const double *x,
                                   const double *y, size_t total_rows, const elpis_runtime_experience *schedule,
                                   size_t experiences, double learning_rate, double *s3_out, size_t s3_count,
                                   elpis_runtime_schedule_result *result);
    int (*leased_txn_commit)(void *runtime, uint64_t id, uint64_t lease, uint64_t token,
                             elpis_runtime_transition *transition);
    int (*txn_abort)(void *runtime, uint64_t id, uint64_t token);
    int (*leased_capture)(void *runtime, uint64_t id, uint64_t lease, uint8_t *out, size_t size, uint64_t *generation);
    int (*envelope_seal)(uint8_t *envelope, size_t size);
} elpis_runtime_k1_fms_api;

/* The ECS composite's functions (state = elpis_ecs_composite *): the K1 table's entries over composite identities
 * (layout-identical first member; the composite status space extends K1's with the same numbers; leased_txn_commit
 * is elpis_ecs_composite_leased_txn_commit_at), then the memory OBSERVE of the open transaction. `observation` is an
 * elpis_ecs_mem_observation and `key` 32 bytes; RuntimeCore never reads the observation. */
typedef struct {
    elpis_runtime_k1_api base;
    int (*leased_txn_observe)(void *state, uint64_t lease, uint64_t token, const void *observation, uint8_t key[32],
                              int *memory_status);
} elpis_runtime_composite_api;

/* One native K1 state. `owner` is the caller's identity for the object owning `handle`; it must stay unique
 * while bound (a freed handle address reused by another state is then not mistaken for the bound one).
 * `dim` is verified natively: the readout length it implies must equal the state's own before any input byte
 * is read. */
typedef struct {
    uint32_t kind;          /* ELPIS_RUNTIME_SUBSTRATE_K1, _K1_FMS or _COMPOSITE */
    uint32_t reserved;      /* zero */
    void *handle;
    uint64_t id;            /* the resident state id (K1_FMS); zero for K1 */
    uint64_t owner;
    uint64_t dim;           /* 1..64 */
    const void *api;        /* const elpis_runtime_k1_api *, elpis_runtime_k1_fms_api * or _composite_api * */
} elpis_runtime_substrate;

typedef struct {
    elpis_runtime_schedule_result schedule;
    int32_t k1_status;      /* the K1 status behind an ECS_* refusal; 0 otherwise */
    uint32_t reserved;
} elpis_runtime_turn_begin_result;

typedef struct {
    uint64_t live_generation;   /* the live generation the answer was computed from; zero on refusal */
    int32_t k1_status;          /* the K1 status behind an ECS_* refusal; 0 otherwise */
    uint32_t reserved;
} elpis_runtime_query_result;

/* A managed commit (v4): VOLATILE. The live generation advanced by one; nothing was checkpointed or published. */
typedef struct {
    elpis_runtime_transition transition;   /* epochs, generations (before/after), steps */
    uint64_t live_generation;              /* = transition.generation_after; 0 when not committed */
    uint32_t committed;                    /* 1: the native commit happened */
    int32_t k1_status;
} elpis_runtime_volatile_commit;

/* One durable publication request. */
typedef struct {
    uint64_t sequence;          /* monotonic per handle, from 1 */
    uint64_t generation;        /* the live generation captured */
    uint32_t status;            /* ELPIS_RUNTIME_DURABLE_* */
    uint32_t recoverable;       /* 1 exactly when ACKNOWLEDGED: a read-back verified slot holds the envelope */
    int32_t code;               /* the refusal or fault behind FAILED / UNCERTAIN / ABANDONED */
    uint32_t kind;              /* ELPIS_RUNTIME_DURABLE_KIND_* (0 for no request) */
    uint8_t identity[32];       /* the captured state's retained-state identity (zero until sealed) */
} elpis_runtime_durable_ticket;

/* The durability lane as a status read sees it. */
typedef struct {
    elpis_runtime_durable_ticket last_requested;
    elpis_runtime_durable_ticket last_acknowledged;
    elpis_runtime_durable_ticket running;          /* sequence 0: none */
    uint64_t live_generation;
    uint32_t pending;                              /* 1 while a captured request waits for the worker */
    int32_t fault;                                 /* the lane's sticky fault (0: none), until reopen */
    uint32_t parked;                               /* testing builds: the step the worker is parked at */
    uint32_t durable_known;                        /* 1 when `durable` holds the current durable authority */
    uint64_t checkpoint_capacity;                  /* the attached store's envelope capacity (0: none) */
    elpis_continuity_snapshot durable;
} elpis_runtime_durability;

typedef struct {
    uint8_t relation_key[32];   /* the observation's relation key; zero on refusal */
    int32_t k1_status;          /* the composite status behind an ECS_* refusal; 0 otherwise */
    int32_t memory_status;      /* the memory's own status behind it; 0 otherwise */
} elpis_runtime_observe_result;

typedef struct {
    uint64_t k1_state_digests;
    uint64_t k1_reserves;
    uint64_t k1_txn_begins;
    uint64_t k1_run_schedules;
    uint64_t k1_commits;
    uint64_t k1_aborts;
    uint64_t publications;
    uint64_t k1_queries;
    uint64_t k1_shapes;
    uint64_t k1_lease_claims;
    uint64_t k1_identities;          /* leased_identity: binding, live_identity */
    uint64_t k1_captures;            /* leased_capture: durable requests, anchor, import */
    uint64_t durable_requests;
    uint64_t durable_superseded;
    uint64_t durable_acknowledged;
    uint64_t durable_failed;         /* FAILED and UNCERTAIN */
} elpis_runtime_counters;

/* Total cognitive fuel (docs/RUNTIME_CORE.md). One operation's totals; every field >= 1 and at most the compiled
 * ceiling (elpis_runtime_fuel_ceiling). An ECS work unit is one multiply-accumulate-class operation of the K1
 * kernels as counted by a fixed integer formula of (dim, width, rows, steps), not wall-clock:
 *   K1 learning step on r rows: 2*r*d*w + 2*d*w*F + F*F;  consolidation of r rows: r*F*F + d*w*F;
 *   experience (r, k): k*step(r) + consolidation(r);  QUERY of r rows: r*d*w;  F = elpis_runtime_features(d).
 * Checked arithmetic: a total beyond uint64 is refused, never wrapped. No wall-clock deadline is claimed: the K1
 * ABI is synchronous with no cooperative cancellation point; fuel bounds an operation before it starts. */
typedef struct {
    uint64_t max_experiences;
    uint64_t max_rows;              /* total rows of one schedule */
    uint64_t max_experience_rows;   /* rows of one experience; also the reserve (capacity) ceiling */
    uint64_t max_steps;             /* total K1 learning steps of one schedule */
    uint64_t max_work_units;
    uint64_t max_query_rows;
} elpis_runtime_budget;

/* K1 Recovery R0 (docs/K1_RECOVERY_R0.md): what restart may do with the lineage. Continuity decides; checkpoint
 * slots are evidence and never authorize themselves. */
typedef struct {
    uint32_t disposition;           /* ELPIS_RUNTIME_RECOVERY_* */
    uint32_t authorized_present;    /* 1 when a verified slot holds the authorized identity */
    uint8_t authorized[32];         /* continuity's identity (zero when unanchored) */
    uint8_t candidate[32];          /* the unresolved candidate's identity (zero when none) */
    uint64_t envelope_bytes;        /* the authorized envelope's size when RESUMABLE, else 0 */
} elpis_runtime_recovery;

typedef struct elpis_runtime elpis_runtime;

uint32_t elpis_runtime_abi_version(void);
/* The S3 readout length for an input dimension (elpis_ecsg_k1_features); 0 outside 1..64. */
size_t elpis_runtime_features(size_t dim);
/* The compiled fuel ceiling; ECS work units of a schedule or of a QUERY (RUNTIME_INVALID on an invalid shape or
 * a total beyond uint64). Pure functions. */
int elpis_runtime_fuel_ceiling(elpis_runtime_budget *out);
int elpis_runtime_work_units(size_t dim, size_t width, const elpis_runtime_experience *schedule, size_t experiences,
                             uint64_t *out);
int elpis_runtime_query_work_units(size_t dim, size_t width, size_t rows, uint64_t *out);
/* Static name of any code ("RUNTIME_OK" for 0), or NULL for an unknown value. */
const char *elpis_runtime_code_name(int code);

/* Lifecycle. create takes an absolute continuity directory path of `len` bytes. */
int elpis_runtime_create(const uint8_t *path, size_t len, elpis_runtime **out);
void elpis_runtime_destroy(elpis_runtime **runtime);   /* aborts an open turn natively, then frees */
/* Opens a closed runtime (close + open clears a fail-stop); an open one is refused (CONTINUITY_OPEN), turn kept. */
int elpis_runtime_open(elpis_runtime *runtime, elpis_continuity_snapshot *out);
void elpis_runtime_close(elpis_runtime *runtime);   /* aborts an open turn natively: nothing installed */
int elpis_runtime_fault(elpis_runtime *runtime);     /* 0, or the fail-stop disposition */
int elpis_runtime_snapshot(elpis_runtime *runtime, elpis_continuity_snapshot *out);
int elpis_runtime_read_counters(elpis_runtime *runtime, elpis_runtime_counters *out, int reset);

/* K1 lineage: explicit anchor of the first managed lineage (reads K1 only). */
int elpis_runtime_anchor(elpis_runtime *runtime, const elpis_runtime_substrate *substrate,
                         elpis_continuity_snapshot *out);

/* QUERY: the canonical read-only operation. x holds `rows` rows of the descriptor's dimension
 * (x_len = rows * dim); out receives `rows` values f_W(x_r) of the LIVE state. Admitted under `budget` (query rows,
 * work units) before the native query (RUNTIME_FUEL). The lineage must be anchored; the state is bound on first use
 * (verified against the durable identity) and its generation must be the live one (otherwise fail-stop, out zeroed).
 * Refused while a managed turn is open (RUNTIME_TURN_OPEN). Nothing is begun, committed, published, hashed or
 * written. */
int elpis_runtime_query(elpis_runtime *runtime, const elpis_runtime_substrate *substrate, const double *x,
                        size_t x_len, double *out, size_t rows, const elpis_runtime_budget *budget,
                        elpis_runtime_query_result *result);

/* LEARN: the managed canonical turn. begin admits the schedule's total fuel under `budget` (RUNTIME_FUEL) before
 * the identity check, any reserve or the transaction, then verifies the lineage (binding the state on first use; a
 * mismatch fail-stops), begins the native transaction (its source generation must be the live one) and runs the whole
 * schedule on its candidate in one native call (x: y_len rows of dim values; s3_len = elpis_runtime_features(dim)).
 * Then exactly one of commit (the native commit: VOLATILE, out->live_generation the new live generation; nothing is
 * checkpointed or published) or abort. A commit refused natively, or on a fail-stopped runtime, aborts the turn.
 * abort ends the described substrate's turn through the capability retained at begin; another substrate is refused
 * (RUNTIME_TURN_NOT_OPEN) and the turn stays open. */
int elpis_runtime_turn_begin(elpis_runtime *runtime, const elpis_runtime_substrate *substrate, const double *x,
                             size_t x_len, const double *y, size_t y_len, const elpis_runtime_experience *schedule,
                             size_t experiences, double learning_rate, const elpis_runtime_budget *budget,
                             double *s3_out, size_t s3_len, elpis_runtime_turn_begin_result *out);
int elpis_runtime_turn_commit(elpis_runtime *runtime, const elpis_runtime_substrate *substrate,
                              elpis_runtime_volatile_commit *out);
int elpis_runtime_turn_abort(elpis_runtime *runtime, const elpis_runtime_substrate *substrate);
/* OBSERVE inside the open turn of a composite substrate (observation: an elpis_ecs_mem_observation, passed through
 * unread). Authoritative only with the turn's commit; a refusal leaves the turn open and its candidate unchanged.
 * A K1 substrate is RUNTIME_INVALID. `out` is nullable. */
int elpis_runtime_turn_observe(elpis_runtime *runtime, const elpis_runtime_substrate *substrate, const void *observation,
                               elpis_runtime_observe_result *out);
/* LEGACY: move the open runtime's K1 lineage to a composite made by elpis_ecs_composite_import_k1 from exactly its
 * authorized K1 state (empty memory). Needs an attached checkpoint store provisioned for the composite envelope
 * size: the composite is claimed and captured, its envelope must name the authorized K1 identity with an empty memory
 * (else the claim is released, STATE_MISMATCH), and it is durable before continuity publishes the import (a v3
 * record). Then the composite is bound. */
int elpis_runtime_import_composite(elpis_runtime *runtime, const elpis_runtime_substrate *substrate,
                                   elpis_continuity_snapshot *out);

/* Release the bound state: its live identity is read under the lease (one hash; a control operation), then its K1
 * lease is released and the lineage unbound. The next QUERY or LEARN binds and verifies again: it accepts exactly the
 * bytes this release gave back (the live lineage, volatile work included) or the durable identity (as a restart
 * would, abandoning that work); anything else is CONTINUITY_STATE_MISMATCH. Refused while a managed turn is open;
 * another state is COGNITION_SUBSTRATE_SWITCH. */
int elpis_runtime_release(elpis_runtime *runtime, const elpis_runtime_substrate *substrate);

/* Durable publication (v4): explicit, served by the durability lane. Two kinds, never confused.
 * recoverable_request: the complete state. Refused with RUNTIME_RECOVERY_UNAVAILABLE (nothing bound or captured)
 *   unless a checkpoint store is attached. The lane seals and verifies the envelope, refuses while an unresolved
 *   candidate exists (CHECKPOINT_UNRESOLVED), writes + syncs the slot that does not hold the durable authority,
 *   reads it back and verifies it is exactly the captured envelope (CHECKPOINT_IO otherwise, the slot withdrawn),
 *   then writes + syncs the continuity record (durable identity -> captured identity). ACKNOWLEDGED (recoverable 1):
 *   restart resumes exactly this state.
 * identity_request: only the identity. Refused with RUNTIME_CHECKPOINT_ATTACHED (nothing bound or captured) while a
 *   checkpoint store is attached. The lane seals and verifies the envelope and publishes continuity alone.
 *   IDENTITY_PUBLISHED (recoverable 0): the caller keeps the bytes; never an acknowledgment of recoverable state.
 * Both bind the state if needed, capture its envelope unsealed under the lease at the live generation (the one
 *   synchronous step) and hand it to the lane; they return at once, `out` a PENDING ticket. At most one request
 *   waits: a newer request supersedes it (SUPERSEDED, never silently acknowledged). Refused while a turn is open, on
 *   an unanchored lineage (anchoring is explicit), for a composite without a checkpoint store (CHECKPOINT_INVALID)
 *   and with the lane's fault while it is faulted (an UNCERTAIN publication or a death, until reopen).
 *   FAILED: nothing durable changed (a failed write is withdrawn). ABANDONED: continuity's record was never begun.
 *   UNCERTAIN: the record's outcome is unknown; restart resolves it to exactly one complete record, which may be
 *   this one; never assumed either way.
 * wait: until `sequence` resolves or `timeout_ms` elapses (UINT64_MAX: no limit); RUNTIME_DURABLE_PENDING on timeout.
 * status: the lane's tickets, fault, the live generation and the durable authority.
 * live_identity: the bound state's retained-state identity (one hashing native call, cold path) and generation. */
int elpis_runtime_recoverable_request(elpis_runtime *runtime, const elpis_runtime_substrate *substrate,
                                      elpis_runtime_durable_ticket *out);
int elpis_runtime_identity_request(elpis_runtime *runtime, const elpis_runtime_substrate *substrate,
                                   elpis_runtime_durable_ticket *out);
int elpis_runtime_durable_wait(elpis_runtime *runtime, uint64_t sequence, uint64_t timeout_ms,
                               elpis_runtime_durable_ticket *out);
int elpis_runtime_durable_status(elpis_runtime *runtime, elpis_runtime_durability *out);
int elpis_runtime_live_identity(elpis_runtime *runtime, const elpis_runtime_substrate *substrate, uint8_t digest[32],
                                uint64_t *generation);

/* K1 Recovery R0: a bounded, crash-recoverable checkpoint owner (two fixed slot files; no history).
 * provision (operator, explicit): create the absolute directory `path` (it must not exist) holding exactly two
 * zeroed slot files sized for envelopes of `envelope_bytes` (the admitted shape's snapshot size).
 * attach: open a provisioned store for this runtime (exclusively locked; no turn open). From then on the anchor
 * persists the anchored envelope before continuity anchors it, and every durable publication persists its complete
 * envelope (in the slot that does not hold the authorized state) before continuity names it. A managed turn never
 * touches the store (v4).
 * recover: read-only; when RESUMABLE and `envelope_len` holds it, the authorized envelope is copied to `envelope`.
 * discard / adopt: explicit operator reconciliation of exactly `candidate` (adopt publishes authorized ->
 * candidate in continuity after the candidate's envelope verified; refused while a state is bound). Both refuse
 * while a durable publication is pending or running (RUNTIME_DURABLE_BUSY). */
int elpis_runtime_checkpoint_provision(const uint8_t *path, size_t len, size_t envelope_bytes);
int elpis_runtime_checkpoint_attach(elpis_runtime *runtime, const uint8_t *path, size_t len);
int elpis_runtime_checkpoint_recover(elpis_runtime *runtime, elpis_runtime_recovery *out, uint8_t *envelope,
                                     size_t envelope_len);
int elpis_runtime_checkpoint_discard(elpis_runtime *runtime, const uint8_t candidate[32]);
int elpis_runtime_checkpoint_adopt(elpis_runtime *runtime, const uint8_t candidate[32],
                                   elpis_continuity_snapshot *out);

/* Evolution: one bounded attempt at a time. authority -> (caller validates) -> reserve -> (caller executes
 * once) -> finalize, or abandon (the reservation stays pending; fail-stop CONTINUITY_EVOLUTION_PENDING).
 * reconcile is the explicit finalization of a durable pending authority after restart. Unchanged by v4: the
 * reservation is written and synced before reserve returns (on the caller's thread); volatile cognition never
 * stands in for it and never makes an executed attempt retryable. */
int elpis_runtime_evolution_authority(elpis_runtime *runtime, elpis_continuity_snapshot *out);
int elpis_runtime_evolution_reserve(elpis_runtime *runtime, const elpis_continuity_evolution *observed,
                                    const uint8_t assertion[32], elpis_continuity_snapshot *out);
int elpis_runtime_evolution_finalize(elpis_runtime *runtime, const uint8_t receipt[32],
                                     elpis_continuity_snapshot *out);
int elpis_runtime_evolution_abandon(elpis_runtime *runtime);
int elpis_runtime_evolution_reconcile(elpis_runtime *runtime, const elpis_continuity_evolution *expected,
                                      const uint8_t receipt[32], elpis_continuity_snapshot *out);

#ifdef __cplusplus
}
#endif

#endif
