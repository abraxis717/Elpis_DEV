#ifndef ELPIS_RUNTIME_H
#define ELPIS_RUNTIME_H

/* Elpis RuntimeCore C ABI v8 (docs/RUNTIME_CORE.md), implemented in Rust (native/runtime).
 *
 * Volatile only (docs/ELPIS_MISSION.md, Amendment 3). A runtime handle owns the mutable systems authority of an open
 * Elpis runtime for one session at a time: lifecycle, fail-stop, its authority (one volatile ECS_C session,
 * docs/ECS_C_SESSION.md, embedded without its C ABI), the binding of its K1 lineage to one native K1 state, the
 * managed canonical turn's native K1 transaction, and the evolution attempt's reservation and finalization. Every one
 * of them lives in memory. RuntimeCore has no persistent-state write path at all: no file, directory, checkpoint,
 * journal, publication, durability lane, recovery or resumption exists in this ABI. It owns no semantics: callers hand
 * it a validated native-ready stimulus or an assertion digest.
 *
 * RuntimeCore links no ECS code: the native K1 state is reached through the caller's K1 function table (the
 * entries of ecsg_k1.h, ecsg_k1_fms.h or ecs_composite.h; a K1 status is an int-sized enum and a state handle an
 * opaque pointer). It computes no ECS mathematics and touches no ECS byte.
 *
 * Sessions (v8). elpis_runtime_create takes a configuration (the session's entity capacity, its founding labels and an
 * optional nonce seed), never a path. Each elpis_runtime_open starts a new session at a fresh genesis (the founding
 * operator and runtime, ACTIVE; nothing anchored); elpis_runtime_close ends it, and with it every authority it held.
 * A restart is a new session: nothing is resumed, and no crash-persistent identity, recovery or exactly-once-across-
 * restart guarantee is claimed. A zero nonce seed draws each session's nonce from the kernel (getrandom(2)); a nonzero
 * seed derives each session's nonce from the seed and the session's ordinal, so runtimes configured alike open
 * identical sessions (differential testing) while one runtime's successive sessions differ.
 *
 * The live coordinate. The bound state's native generation (ecsg_k1.h, "Live coordinate") under this runtime's lease,
 * as RuntimeCore last established it. Every HOT operation (QUERY, the managed turn, OBSERVE) is checked against it;
 * none performs I/O or computes a digest. The session records the lineage's anchored identity; a managed commit
 * installs the candidate and advances the live generation by one, in memory, and the session never needs to name it.
 *
 * QUERY and LEARN. QUERY (elpis_runtime_query) is read-only: one native call answers f_W(x) from the lineage's live
 * state together with that state's generation, which must be the live one (otherwise the lineage is lost: fail-stop,
 * the answer withheld). It opens no transaction, commits nothing, hashes nothing. LEARN is the managed turn below:
 * begin, the experience schedule on a candidate, the volatile commit.
 *
 * Managed ownership (v3). Binding a state (anchor, or the first QUERY or LEARN after open) checks its retained-state
 * identity against the session's anchored one before claiming it (a mismatched state is never claimed), claims it
 * under this runtime's K1 lease, then reads its identity and generation together under the lease: from then on the
 * state refuses every unmanaged mutation (ELPIS_ECSG_K1_LEASED), and every managed transaction begin reports the
 * generation of exactly the source it copied. A bound state that is not provably the lineage (another generation, or
 * its lease taken by another owner) fail-stops with CONTINUITY_STATE_MISMATCH before anything is built on it.
 * elpis_runtime_release gives the state back (unbound, unleased); rebinding accepts exactly the bytes it gave back
 * (the live lineage) or the anchored identity (an in-memory rollback, abandoning the volatile work).
 *
 * Turn lifecycle (v2). A successful turn_begin opens one native K1 transaction. RuntimeCore ends it with exactly one
 * terminal native action, the commit or an abort, before it forgets the turn; there is no third disposition. An
 * explicit turn_abort, a refused commit, close, and destroy all abort it; open on an open runtime is refused and
 * keeps it. An abort refused BUSY (a concurrent overlapping call on the state) is retried until the state admits it.
 * An implicit abort installs nothing.
 *
 * Substrate lifetime (v2). To end the turn on every path, RuntimeCore retains the descriptor's handle, resident id
 * and txn_abort entry (copied; never any other entry) from a successful turn_begin until the turn ends. The native
 * state must stay live, and its library loaded, for that interval; outside it, a descriptor need only be live for
 * the call. The K1 FMS adapter enforces this natively: a resident state with an open transaction cannot be closed
 * and its runtime cannot be destroyed while a state is registered (both BUSY). A standalone K1 state's owner must
 * not destroy it while a turn on it is open.
 *
 * ECS composite substrates (docs/ECS_COMPOSITE_STATE.md). A substrate of kind ELPIS_RUNTIME_SUBSTRATE_COMPOSITE is an
 * ECS composite retained state (ecs_composite.h: K1 and the ECS memory under one ELPISEC1 identity). Its table is the
 * K1 table over composite identities plus leased_txn_observe; RuntimeCore binds, queries and learns it exactly as a
 * K1 state, and the session records the lineage's format, so a K1 state never stands for a composite lineage or the
 * reverse (STATE_MISMATCH). The memory changes only through elpis_runtime_turn_observe inside a managed turn (a LEARN
 * turn, or an observe-only turn below). No memory LEARN or CONSOLIDATE is exposed (unqualified).
 *
 * Observe-only turns (v5). elpis_runtime_turn_begin_observe opens a managed turn on a composite substrate that runs
 * no K1 experience schedule: the lineage is verified and bound exactly as for turn_begin and the composite
 * transaction is opened at the live generation. Typed observations then enter through elpis_runtime_turn_observe; the
 * turn ends with turn_commit (the K1 retained state stays byte-identical; a commit whose K1 transition reports a step
 * or an epoch change fail-stops STATE_MISMATCH) or turn_abort. A commit while nothing was observed is refused
 * RUNTIME_INVALID and keeps the turn open. A K1 or K1 FMS substrate is refused RUNTIME_INVALID. Every managed turn
 * records at most budget.max_observations observations; one more is refused COGNITION_FUEL_EXCEEDED before the native
 * call and leaves the turn open.
 *
 * The session authority. Each RuntimeCore event (the anchor, an evolution reservation or finalization) is one typed
 * event of the session's founding RUNTIME entity at its next sequence, compare-and-publish on the session's coordinate,
 * in memory. A typed observation is admitted only from an ACTIVE source of the session (named by the memory
 * source_authority the observation records) within that source's per-turn bound, checked before the native call
 * (RUNTIME_OBSERVATION_REFUSED; recoverable, the turn stays open). The operator's events (founding, lifecycle, source
 * versions, reclamation) reach the session only through elpis_runtime_authority_transition, built from its rows as
 * elpis_runtime_authority_entity_at and elpis_runtime_authority_next_entity_id read them; the operations of the
 * retired durable storage (legacy import, the reconciliations) are RUNTIME_RETIRED. "continuity" in this header names
 * the authority's role, the register its laws were first stated in; its view keeps continuity's layout, values and
 * names (elpis_continuity_snapshot, elpis_continuity_evolution, ELPIS_CONTINUITY_*).
 *
 * Retired by v8 (docs/history/RUNTIME_CORE_ABI_V7.md): elpis_runtime_create_ecs_c, elpis_runtime_import_composite,
 * the durable requests and the durability lane (elpis_runtime_recoverable_request, _identity_request, _durable_wait,
 * _durable_status), the K1 checkpoint store (elpis_runtime_checkpoint_provision, _attach, _recover, _discard, _adopt),
 * elpis_runtime_evolution_reconcile, the testing library and the substrate tables' leased_capture and envelope_seal;
 * the authority codes 1, 6, 7, 8, 14, 16, 17 and 255 and the runtime codes 76..83 are never returned and never reused.
 * A v7 caller refuses to load this library (the version check), so none can mistake it for a durable runtime.
 *
 * The library exports this ABI only.
 *
 * Every function returns 0 or a positive code: an authority code (ELPIS_CONTINUITY_*) or a runtime code below.
 * Calls on one handle are serialized by the handle. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The authority's codes (the names and values elpis/continuity.h gave them). */
enum {
    ELPIS_CONTINUITY_OK = 0,
    ELPIS_CONTINUITY_UNANCHORED = 2,
    ELPIS_CONTINUITY_ALREADY_ANCHORED = 3,
    ELPIS_CONTINUITY_STATE_MISMATCH = 4,
    ELPIS_CONTINUITY_CORRUPT = 5,
    ELPIS_CONTINUITY_AUTHORITY_MISMATCH = 9,
    ELPIS_CONTINUITY_EVOLUTION_PENDING = 10,
    ELPIS_CONTINUITY_EVOLUTION_NOT_PENDING = 11,
    ELPIS_CONTINUITY_EXHAUSTED = 12,
    ELPIS_CONTINUITY_INVALID = 13,
    ELPIS_CONTINUITY_OPEN = 15
};

/* Evolution authority. pending == 0: idle, assertion must be zero. pending == 1: reserved for exactly `assertion`.
 * revision 0 if and only if head is zero. reserved bytes zero. */
typedef struct {
    uint64_t revision;
    uint8_t head[32];
    uint8_t pending;
    uint8_t reserved[7];
    uint8_t assertion[32];
} elpis_continuity_evolution;

/* The session's authority view. anchored == 0: no lineage (k1_state_digest zero). state_format names what the
 * anchored digest identifies: ELPIS_CONTINUITY_FORMAT_K1 (0; the field keeps its historical name) or
 * ELPIS_CONTINUITY_FORMAT_COMPOSITE (1, anchored only). The two digests are outputs (the evolution-authority v2
 * digest and the continuity register record digest of this view); they are ignored on input. */
enum { ELPIS_CONTINUITY_FORMAT_K1 = 0u, ELPIS_CONTINUITY_FORMAT_COMPOSITE = 1u };
typedef struct {
    uint64_t generation;
    uint8_t anchored;
    uint8_t state_format;
    uint8_t reserved[6];
    uint8_t k1_state_digest[32];
    elpis_continuity_evolution evolution;
    uint8_t evolution_digest[32];
    uint8_t record_digest[32];
} elpis_continuity_snapshot;

enum {
    ELPIS_RUNTIME_ABI_V8 = 8u,
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
    ELPIS_RUNTIME_NOT_RESIDENT = 84,             /* an FMS-resident state is not WARM where a HOT operation needs
                                                    it: refused without promoting or waiting */
    ELPIS_RUNTIME_AUTHORITY = 85,                /* the session refused without an authority-code equivalent (the
                                                    exact session code in elpis_runtime_authority.ecs_c_code), or
                                                    no session nonce could be drawn at open */
    ELPIS_RUNTIME_OBSERVATION_REFUSED = 86,      /* the observation's source is not an ACTIVE source of the session,
                                                    or is past its per-turn bound: refused before the native call,
                                                    the turn stays open */
    ELPIS_RUNTIME_NOMEM = 87,                    /* v8: the session's memory could not be reserved at open; the
                                                    runtime stays closed */
    ELPIS_RUNTIME_RETIRED = 88,                  /* v8: an operation of the retired persistent runtime (a legacy
                                                    import or a reconciliation event): refused, nothing changed */
    ELPIS_RUNTIME_BACKEND_ECS_C_SESSION = 3,     /* elpis_runtime_authority.backend (v6/v7's 2 was the durable
                                                    ECS_C authority, v6's 1 standalone continuity) */
    ELPIS_RUNTIME_SUBSTRATE_K1 = 1,              /* standalone K1 state (ecsg_k1.h) */
    ELPIS_RUNTIME_SUBSTRATE_K1_FMS = 2,          /* FMS-resident K1 state (ecsg_k1_fms.h) */
    ELPIS_RUNTIME_SUBSTRATE_COMPOSITE = 3        /* ECS composite retained state (ecs_composite.h; id zero) */
};

/* What every session of a runtime is created with (elpis_runtime_create). */
typedef struct {
    uint32_t abi_version;       /* ELPIS_RUNTIME_ABI_V8 */
    uint32_t capacity;          /* the session's ECS_C entity capacity, 2..4096 (the founders occupy two rows) */
    uint8_t nonce[32];          /* zero: each session draws a fresh nonce (getrandom(2)); else each session's nonce
                                   is SHA-256("elpis.runtime.session-nonce.v1\0" || nonce || ordinal, u64 LE) */
    uint8_t operator_label[32]; /* the founding operator's label digest */
    uint8_t runtime_label[32];  /* the founding runtime's label digest */
} elpis_runtime_config;

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

/* The standalone K1 functions RuntimeCore calls (state = elpis_ecsg_k1 *), ABI v8. Every entry is required. Every
 * mutating entry is a leased one (ecsg_k1.h, "Managed ownership"); the live-coordinate entries (ecsg_k1.h, "Live
 * coordinate") report the generation under the same guard as their answer, begin or commit. state_digest is the one
 * unleased read: a state is checked against the session's identity before it is ever claimed. v8 removed
 * leased_capture and envelope_seal (they existed for durable publication). */
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
} elpis_runtime_k1_api;

/* The K1 FMS adapter's functions (runtime = elpis_ecsg_k1_fms *, id = the resident state), ABI v8. */
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
    /* HOT residency (docs/RUNTIME_CORE.md): elpis_ecsg_k1_fms_resident_hold / _resident_release. RuntimeCore holds a
     * bound state resident (taken before binding, only if it is WARM; RUNTIME_NOT_RESIDENT otherwise) and drops the
     * hold whenever it unbinds it; meanwhile every operation on it pins resident-only (never a promotion). */
    int (*resident_hold)(void *runtime, uint64_t id);
    int (*resident_release)(void *runtime, uint64_t id);
} elpis_runtime_k1_fms_api;

/* The ECS composite's functions (state = elpis_ecs_composite *): the K1 table's entries over composite identities
 * (layout-identical first member; the composite status space extends K1's with the same numbers; leased_txn_commit
 * is elpis_ecs_composite_leased_txn_commit_at), then the memory OBSERVE of the open transaction. `observation` is an
 * elpis_ecs_mem_observation and `key` 32 bytes; RuntimeCore never reads the observation beyond its source. */
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

/* A managed commit: VOLATILE. The live generation advanced by one; nothing was written anywhere. */
typedef struct {
    elpis_runtime_transition transition;   /* epochs, generations (before/after), steps */
    uint64_t live_generation;              /* = transition.generation_after; 0 when not committed */
    uint32_t committed;                    /* 1: the native commit happened */
    int32_t k1_status;
} elpis_runtime_volatile_commit;

/* The session (elpis_runtime_authority_status). While closed: backend, the sessions opened and the session code of
 * the last refused open, everything else zero. */
typedef struct {
    uint32_t backend;           /* ELPIS_RUNTIME_BACKEND_ECS_C_SESSION */
    int32_t ecs_c_code;         /* the last session refusal (ELPIS_ECS_C_*); 0 none */
    uint32_t entity_count;
    uint32_t capacity;
    uint64_t generation;
    uint64_t instance;          /* the session's process-unique instance (0: no session is open) */
    uint64_t receipt_count;
    uint64_t evidence_count;
    uint64_t volatile_bytes;    /* the session's exact, constant memory footprint (0: none open) */
    uint64_t sessions;          /* sessions this runtime has opened */
    uint64_t next_founding_index; /* with the rows: what a relational view (elpis/ecs_c_relational.h) syncs from */
    uint64_t reclaimed_count;
    uint8_t session[32];        /* the session's lineage */
    uint8_t commit[32];         /* the in-memory commit digest of the current generation */
    uint8_t state_root[32];
    uint8_t receipt_head[32];
    uint8_t evidence_head[32];
} elpis_runtime_authority;

/* One entity row of the open session (elpis_runtime_authority_entity_at), laid out as elpis_ecs_c_entity
 * (elpis/ecs_c_types.h): the operator's view for building an event. */
typedef struct {
    uint8_t id[32];
    uint8_t kind;               /* ELPIS_ECS_C_OPERATOR, _RUNTIME, _SOURCE */
    uint8_t lifecycle;          /* ELPIS_ECS_C_FOUNDED, _ACTIVE, _DORMANT, _TERMINATED */
    uint8_t reserved[2];
    uint32_t per_turn_limit;
    uint64_t founding_index;
    uint64_t founded_generation;
    uint64_t last_generation;
    uint64_t sender_watermark;  /* the entity's next event carries sequence sender_watermark + 1 */
    uint64_t source_version;
    uint8_t label[32];
    uint8_t scope[32];
    uint32_t source_authority;  /* sources: the memory source_authority its observations record */
    uint32_t reserved2;
} elpis_runtime_authority_entity;

/* An observe-only turn begun (v5). */
typedef struct {
    uint64_t live_generation;   /* the live generation the turn was opened on; zero on refusal */
    int32_t k1_status;          /* the composite status behind an ECS_* refusal; 0 otherwise */
    uint32_t reserved;
} elpis_runtime_observe_begin_result;

typedef struct {
    uint8_t relation_key[32];   /* the observation's relation key; zero on refusal */
    int32_t k1_status;          /* the composite status behind an ECS_* refusal; 0 otherwise */
    int32_t memory_status;      /* the memory's own status behind it; 0 otherwise */
} elpis_runtime_observe_result;

/* Native crossings and session transitions (v8 layout). */
typedef struct {
    uint64_t k1_state_digests;
    uint64_t k1_reserves;
    uint64_t k1_txn_begins;
    uint64_t k1_run_schedules;
    uint64_t k1_commits;
    uint64_t k1_aborts;
    uint64_t session_transitions;    /* accepted session transitions: anchor, evolution, the operator's events */
    uint64_t k1_queries;
    uint64_t k1_shapes;
    uint64_t k1_lease_claims;
    uint64_t k1_identities;          /* leased_identity: anchor, binding, release, live_identity */
    uint64_t k1_observes;            /* leased_txn_observe crossings */
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
    uint64_t max_observations;      /* typed memory observations of one managed turn (v5) */
} elpis_runtime_budget;

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
/* Static name of any code ("RUNTIME_OK" for 0), or NULL for an unknown (or retired) value. */
const char *elpis_runtime_code_name(int code);

/* Lifecycle. create: a closed runtime (RUNTIME_INVALID for a NULL argument, another abi_version or a capacity
 * outside 2..4096); no session exists until open. destroy aborts an open turn natively, ends the session and frees. */
int elpis_runtime_create(const elpis_runtime_config *config, elpis_runtime **out);
void elpis_runtime_destroy(elpis_runtime **runtime);
/* open: a new session at a fresh genesis (close + open clears a fail-stop, with the session it belonged to); an open
 * runtime is refused (CONTINUITY_OPEN), turn kept. RUNTIME_NOMEM when the session's memory cannot be reserved,
 * RUNTIME_AUTHORITY when no nonce can be drawn: the runtime stays closed. `out` (nullable): the session's view. */
int elpis_runtime_open(elpis_runtime *runtime, elpis_continuity_snapshot *out);
void elpis_runtime_close(elpis_runtime *runtime);   /* aborts an open turn natively; the session ends */
int elpis_runtime_fault(elpis_runtime *runtime);     /* 0, or the fail-stop disposition */
int elpis_runtime_snapshot(elpis_runtime *runtime, elpis_continuity_snapshot *out);
int elpis_runtime_read_counters(elpis_runtime *runtime, elpis_runtime_counters *out, int reset);

/* K1 lineage: explicit anchor of the session's lineage at the substrate's retained identity (reads K1 only). */
int elpis_runtime_anchor(elpis_runtime *runtime, const elpis_runtime_substrate *substrate,
                         elpis_continuity_snapshot *out);

/* QUERY: the canonical read-only operation. x holds `rows` rows of the descriptor's dimension
 * (x_len = rows * dim); out receives `rows` values f_W(x_r) of the LIVE state. Admitted under `budget` (query rows,
 * work units) before the native query (RUNTIME_FUEL). The lineage must be anchored; the state is bound on first use
 * (verified against the session's identity) and its generation must be the live one (otherwise fail-stop, out
 * zeroed). Refused while a managed turn is open (RUNTIME_TURN_OPEN). Nothing is begun, committed, hashed or written. */
int elpis_runtime_query(elpis_runtime *runtime, const elpis_runtime_substrate *substrate, const double *x,
                        size_t x_len, double *out, size_t rows, const elpis_runtime_budget *budget,
                        elpis_runtime_query_result *result);

/* LEARN: the managed canonical turn. begin admits the schedule's total fuel under `budget` (RUNTIME_FUEL) before
 * the identity check, any reserve or the transaction, then verifies the lineage (binding the state on first use; a
 * mismatch fail-stops), begins the native transaction (its source generation must be the live one) and runs the whole
 * schedule on its candidate in one native call (x: y_len rows of dim values; s3_len = elpis_runtime_features(dim)).
 * Then exactly one of commit (the native commit: VOLATILE, out->live_generation the new live generation) or abort. A
 * commit refused natively, or on a fail-stopped runtime, aborts the turn. abort ends the described substrate's turn
 * through the capability retained at begin; another substrate is refused (RUNTIME_TURN_NOT_OPEN) and the turn stays
 * open. */
int elpis_runtime_turn_begin(elpis_runtime *runtime, const elpis_runtime_substrate *substrate, const double *x,
                             size_t x_len, const double *y, size_t y_len, const elpis_runtime_experience *schedule,
                             size_t experiences, double learning_rate, const elpis_runtime_budget *budget,
                             double *s3_out, size_t s3_len, elpis_runtime_turn_begin_result *out);
int elpis_runtime_turn_commit(elpis_runtime *runtime, const elpis_runtime_substrate *substrate,
                              elpis_runtime_volatile_commit *out);
int elpis_runtime_turn_abort(elpis_runtime *runtime, const elpis_runtime_substrate *substrate);
/* OBSERVE inside the open turn of a composite substrate (observation: an elpis_ecs_mem_observation, passed through
 * unread beyond its source). Authoritative only with the turn's commit; a refusal leaves the turn open and its
 * candidate unchanged. A K1 substrate is RUNTIME_INVALID. `out` is nullable. */
int elpis_runtime_turn_observe(elpis_runtime *runtime, const elpis_runtime_substrate *substrate, const void *observation,
                               elpis_runtime_observe_result *out);
/* Begin an OBSERVE-ONLY managed turn on a composite substrate (v5): the lineage verified and bound as by turn_begin,
 * the composite transaction opened at the live generation, no K1 schedule run (fuel: the budget must narrow the
 * ceiling; its max_observations bounds the turn's observations). End it with turn_commit (refused RUNTIME_INVALID,
 * the turn kept open, while nothing was observed) or turn_abort. A K1 or K1 FMS substrate is RUNTIME_INVALID.
 * `out` is nullable. */
int elpis_runtime_turn_begin_observe(elpis_runtime *runtime, const elpis_runtime_substrate *substrate,
                                     const elpis_runtime_budget *budget, elpis_runtime_observe_begin_result *out);

/* Release the bound state: its live identity is read under the lease (one hash; a control operation), then its K1
 * lease is released and the lineage unbound. The next QUERY or LEARN binds and verifies again: it accepts exactly the
 * bytes this release gave back (the live lineage, volatile work included) or the session's anchored identity
 * (abandoning that work); anything else is CONTINUITY_STATE_MISMATCH. Refused while a managed turn is open; another
 * state is COGNITION_SUBSTRATE_SWITCH. */
int elpis_runtime_release(elpis_runtime *runtime, const elpis_runtime_substrate *substrate);
/* The bound state's retained-state identity (one hashing native call, cold path) and live generation. */
int elpis_runtime_live_identity(elpis_runtime *runtime, const elpis_runtime_substrate *substrate, uint8_t digest[32],
                                uint64_t *generation);

/* Evolution: one bounded attempt at a time. authority -> (caller validates) -> reserve -> (caller executes once) ->
 * finalize, or abandon (the reservation stays pending; fail-stop CONTINUITY_EVOLUTION_PENDING for the rest of the
 * session). The reservation is a session transition, in the session when reserve returns; a refused reservation
 * changes nothing (no fail-stop), a refused finalization fail-stops (the attempt executed: never re-executed). */
int elpis_runtime_evolution_authority(elpis_runtime *runtime, elpis_continuity_snapshot *out);
int elpis_runtime_evolution_reserve(elpis_runtime *runtime, const elpis_continuity_evolution *observed,
                                    const uint8_t assertion[32], elpis_continuity_snapshot *out);
int elpis_runtime_evolution_finalize(elpis_runtime *runtime, const uint8_t receipt[32],
                                     elpis_continuity_snapshot *out);
int elpis_runtime_evolution_abandon(elpis_runtime *runtime);

/* The session authority. status: readable closed (then the sessions opened and the last refused open's session
 * code). transition: one operator event (512 bytes, elpis/ecs_c_types.h: ENTITY_FOUND, ENTITY_LIFECYCLE,
 * SOURCE_VERSION or ENTITY_RECLAIM) on the open session, which then governs observation admission; a legacy import or
 * a reconciliation is RUNTIME_RETIRED, RuntimeCore's own kinds RUNTIME_INVALID; refused while a managed turn is open
 * (RUNTIME_TURN_OPEN), closed or fail-stopped. out_generation (optional) receives the session's new generation.
 * entity_at / next_entity_id: the row at `index` (below elpis_runtime_authority.entity_count) and the identity the next
 * ENTITY_FOUND of `kind` and `label` founds (the receiver that event must name); RUNTIME_INVALID past the last row and
 * for an unknown kind; refused closed or fail-stopped. Neither changes anything. */
int elpis_runtime_authority_status(elpis_runtime *runtime, elpis_runtime_authority *out);
int elpis_runtime_authority_transition(elpis_runtime *runtime, const uint8_t *event, uint64_t *out_generation);
int elpis_runtime_authority_entity_at(elpis_runtime *runtime, uint32_t index, elpis_runtime_authority_entity *out);
int elpis_runtime_authority_next_entity_id(elpis_runtime *runtime, uint8_t kind, const uint8_t label[32],
                                           uint8_t out[32]);

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(elpis_runtime_config) == 104, "elpis_runtime_config layout");
_Static_assert(sizeof(elpis_continuity_snapshot) == 192, "elpis_continuity_snapshot layout");
_Static_assert(sizeof(elpis_runtime_authority) == 240, "elpis_runtime_authority layout");
_Static_assert(sizeof(elpis_runtime_counters) == 96, "elpis_runtime_counters layout");
_Static_assert(sizeof(elpis_runtime_budget) == 56, "elpis_runtime_budget layout");
#endif

#ifdef __cplusplus
}
#endif

#endif
