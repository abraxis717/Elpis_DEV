#ifndef ELPIS_RUNTIME_H
#define ELPIS_RUNTIME_H

/* Elpis RuntimeCore C ABI v3 (docs/RUNTIME_CORE.md), implemented in Rust (native/runtime).
 *
 * One runtime handle owns one continuity directory and the mutable systems authority of an open Elpis
 * runtime: lifecycle, fail-stop, the binding of its K1 lineage to one native K1 state, the managed canonical
 * turn's native K1 transaction and its continuity publication, and the evolution attempt's durable
 * reservation and finalization. It owns no semantics: callers hand it a validated native-ready stimulus or an
 * assertion digest.
 *
 * RuntimeCore links no ECS code: the native K1 state is reached through the caller's K1 function table (the
 * entries of ecsg_k1.h or ecsg_k1_fms.h; a K1 status is an int-sized enum and a state handle an opaque
 * pointer). It computes no ECS mathematics and touches no ECS byte.
 *
 * QUERY and LEARN (v3). QUERY (elpis_runtime_query) is read-only: one native call answers f_W(x) from the lineage's
 * authoritative K1 state together with that state's retained-state identity, which must equal the durable expected
 * identity (a mismatch fail-stops and the answer is withheld). It opens no transaction, commits nothing and publishes
 * nothing. LEARN is the managed turn below: begin, the experience schedule on a candidate, commit, one publication.
 * v3 adds the query entry to both K1 function tables and the query counter; a v2 caller refuses to load it.
 *
 * Managed ownership (v3). Binding a state (anchor, or the first QUERY or LEARN after open) verifies its retained-state
 * identity against the durable one and claims it under this runtime's K1 lease: from then on the state refuses every
 * unmanaged mutation (ELPIS_ECSG_K1_LEASED), and every managed transaction begin re-establishes the state's exact
 * identity inside the same guarded K1 call. A bound state that is not provably the lineage (another identity, or
 * its lease taken by another owner) fail-stops with CONTINUITY_STATE_MISMATCH before anything is built on it or
 * published. elpis_runtime_release gives the state back (unbound, unleased); close does not touch the state.
 *
 * Turn lifecycle (v2). A successful turn_begin opens one native K1 transaction. RuntimeCore ends it with exactly one
 * terminal native action, the commit or an abort, before it forgets the turn; there is no third disposition. An
 * explicit turn_abort, a refused commit, close, and destroy all abort it; open on an open runtime is refused and
 * keeps it. An abort refused BUSY (a concurrent overlapping call on the state) is retried until the state admits it.
 * An implicit abort installs nothing and publishes nothing: W, epoch, generation, H, a, the retained-state digest
 * and continuity are unchanged, and the state takes the next transaction.
 *
 * Substrate lifetime (v2). To end the turn on every path, RuntimeCore retains the descriptor's handle, resident id
 * and txn_abort entry (copied; never any other entry) from a successful turn_begin until the turn ends. The native
 * state must stay live, and its library loaded, for that interval; outside it, a descriptor need only be live for
 * the call. The K1 FMS adapter enforces this natively: a resident state with an open transaction cannot be closed
 * and its runtime cannot be destroyed while a state is registered (both BUSY). A standalone K1 state's owner must
 * not destroy it while a turn on it is open (ecsg_k1.h frees a state whatever transaction it holds).
 *
 * The library also exports the continuity C ABI (elpis/continuity.h) of the store it embeds.
 *
 * Every function returns 0 or a positive code: a continuity code (1..17, 255) or a runtime code below.
 * Calls on one handle are serialized by the handle. */

#include <stddef.h>
#include <stdint.h>

#include "elpis/continuity.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ELPIS_RUNTIME_ABI_V3 = 3u,
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
    ELPIS_RUNTIME_SUBSTRATE_K1 = 1,              /* standalone K1 state (ecsg_k1.h) */
    ELPIS_RUNTIME_SUBSTRATE_K1_FMS = 2           /* FMS-resident K1 state (ecsg_k1_fms.h) */
};

/* Layout-identical to elpis_ecsg_k1_experience, _schedule_result, _transition and _commit_identity. */
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

typedef struct {
    elpis_runtime_transition transition;
    uint8_t state_before_digest[32];
    uint8_t state_after_digest[32];
} elpis_runtime_commit_identity;

/* The standalone K1 functions RuntimeCore calls (state = elpis_ecsg_k1 *). Every entry is required. Every mutating
 * entry is a leased one (ecsg_k1.h, "Managed ownership"): RuntimeCore claims each state it binds with its own lease,
 * so the bound state refuses every unmanaged mutation, and presents that lease on every mutation it makes. */
typedef struct {
    int (*state_digest)(void *state, uint8_t out[32]);
    int (*shape)(void *state, size_t *dim, size_t *width);
    int (*query_identity)(void *state, size_t dim, const double *x, size_t rows, double *out, uint8_t digest[32]);
    int (*lease_claim)(void *state, uint64_t lease);
    int (*lease_release)(void *state, uint64_t lease);
    int (*leased_reserve)(void *state, uint64_t lease, size_t max_rows);
    int (*leased_txn_begin)(void *state, uint64_t lease, uint8_t source_digest[32], uint64_t *token);
    int (*leased_txn_run_schedule)(void *state, uint64_t lease, uint64_t token, const double *x, const double *y,
                                   size_t total_rows, const elpis_runtime_experience *schedule, size_t experiences,
                                   double learning_rate, double *s3_out, size_t s3_count,
                                   elpis_runtime_schedule_result *result);
    int (*leased_txn_commit_identity)(void *state, uint64_t lease, uint64_t token,
                                      elpis_runtime_commit_identity *identity);
    int (*txn_abort)(void *state, uint64_t token);
} elpis_runtime_k1_api;

/* The K1 FMS adapter's functions (runtime = elpis_ecsg_k1_fms *, id = the resident state). */
typedef struct {
    int (*state_digest)(void *runtime, uint64_t id, uint8_t out[32]);
    int (*shape)(void *runtime, uint64_t id, size_t *dim, size_t *width);
    int (*query_identity)(void *runtime, uint64_t id, size_t dim, const double *x, size_t rows, double *out,
                          uint8_t digest[32]);
    int (*lease_claim)(void *runtime, uint64_t id, uint64_t lease);
    int (*lease_release)(void *runtime, uint64_t id, uint64_t lease);
    int (*leased_reserve)(void *runtime, uint64_t id, uint64_t lease, size_t max_rows);
    int (*leased_txn_begin)(void *runtime, uint64_t id, uint64_t lease, uint8_t source_digest[32], uint64_t *token);
    int (*leased_txn_run_schedule)(void *runtime, uint64_t id, uint64_t lease, uint64_t token, const double *x,
                                   const double *y, size_t total_rows, const elpis_runtime_experience *schedule,
                                   size_t experiences, double learning_rate, double *s3_out, size_t s3_count,
                                   elpis_runtime_schedule_result *result);
    int (*leased_txn_commit_identity)(void *runtime, uint64_t id, uint64_t lease, uint64_t token,
                                      elpis_runtime_commit_identity *identity);
    int (*txn_abort)(void *runtime, uint64_t id, uint64_t token);
} elpis_runtime_k1_fms_api;

/* One native K1 state. `owner` is the caller's identity for the object owning `handle`; it must stay unique
 * while bound (a freed handle address reused by another state is then not mistaken for the bound one).
 * `dim` is verified natively: the readout length it implies must equal the state's own before any input byte
 * is read. */
typedef struct {
    uint32_t kind;          /* ELPIS_RUNTIME_SUBSTRATE_K1 or _K1_FMS */
    uint32_t reserved;      /* zero */
    void *handle;
    uint64_t id;            /* the resident state id (K1_FMS); zero for K1 */
    uint64_t owner;
    uint64_t dim;           /* 1..64 */
    const void *api;        /* const elpis_runtime_k1_api * or const elpis_runtime_k1_fms_api * */
} elpis_runtime_substrate;

typedef struct {
    elpis_runtime_schedule_result schedule;
    int32_t k1_status;      /* the K1 status behind an ECS_* refusal; 0 otherwise */
    uint32_t reserved;
} elpis_runtime_turn_begin_result;

typedef struct {
    uint8_t state_digest[32];   /* the retained-state identity the answer was computed from; zero on refusal */
    int32_t k1_status;          /* the K1 status behind an ECS_* refusal; 0 otherwise */
    uint32_t reserved;
} elpis_runtime_query_result;

typedef struct {
    elpis_runtime_commit_identity identity;
    uint32_t committed;     /* 1: the native commit happened (also when its publication then failed) */
    int32_t k1_status;
} elpis_runtime_turn_commit_result;

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
 * (x_len = rows * dim); out receives `rows` values f_W(x_r). Admitted under `budget` (query rows, work units)
 * before the native query (RUNTIME_FUEL). The lineage must be anchored and the state's identity
 * must equal the durable expected identity (bound on first use; a mismatch fail-stops, out is zeroed). Refused while
 * a managed turn is open (RUNTIME_TURN_OPEN). Nothing is begun, committed, published or written. */
int elpis_runtime_query(elpis_runtime *runtime, const elpis_runtime_substrate *substrate, const double *x,
                        size_t x_len, double *out, size_t rows, const elpis_runtime_budget *budget,
                        elpis_runtime_query_result *result);

/* LEARN: the managed canonical turn. begin admits the schedule's total fuel under `budget` (RUNTIME_FUEL) before
 * the identity check, any reserve or the transaction, then verifies the lineage (binding the state on first use; a mismatch
 * fail-stops), begins the native transaction and runs the whole schedule on its candidate in one native call
 * (x: y_len rows of dim values; s3_len = elpis_runtime_features(dim)). Then exactly one of commit (native
 * commit, then one continuity publication; a failed publication fail-stops and out->committed reports the
 * standing K1 commit) or abort. A commit refused natively, or on a fail-stopped runtime, aborts the turn. abort
 * ends the described substrate's turn through the capability retained at begin; another substrate is refused
 * (RUNTIME_TURN_NOT_OPEN) and the turn stays open. */
int elpis_runtime_turn_begin(elpis_runtime *runtime, const elpis_runtime_substrate *substrate, const double *x,
                             size_t x_len, const double *y, size_t y_len, const elpis_runtime_experience *schedule,
                             size_t experiences, double learning_rate, const elpis_runtime_budget *budget,
                             double *s3_out, size_t s3_len, elpis_runtime_turn_begin_result *out);
int elpis_runtime_turn_commit(elpis_runtime *runtime, const elpis_runtime_substrate *substrate,
                              elpis_runtime_turn_commit_result *out, elpis_continuity_snapshot *snapshot);
int elpis_runtime_turn_abort(elpis_runtime *runtime, const elpis_runtime_substrate *substrate);

/* Release the bound state: its K1 lease is released and the lineage unbound (the next QUERY or LEARN binds and
 * verifies again). Refused while a managed turn is open; another state is COGNITION_SUBSTRATE_SWITCH. */
int elpis_runtime_release(elpis_runtime *runtime, const elpis_runtime_substrate *substrate);

/* Evolution: one bounded attempt at a time. authority -> (caller validates) -> reserve -> (caller executes
 * once) -> finalize, or abandon (the reservation stays pending; fail-stop CONTINUITY_EVOLUTION_PENDING).
 * reconcile is the explicit finalization of a durable pending authority after restart. */
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
