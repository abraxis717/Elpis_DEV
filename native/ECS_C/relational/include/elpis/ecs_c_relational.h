#ifndef ELPIS_ECS_C_RELATIONAL_H
#define ELPIS_ECS_C_RELATIONAL_H

#include <stddef.h>
#include <stdint.h>

#include "elpis/ecs_c_authority.h"
#include "elpis/ecs_memory.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ECS_C relational view (ECSC_DYNAMIC_RELATIONAL_DEV_R0, C ABI v1; docs/ECS_C_RELATIONAL.md).
 *
 * DEV, NOT COMPOSED. Nothing canonical links or loads it (tests/boundary/test_ecs_c_relational.py): not RuntimeCore,
 * the composite, the memory or src/. It qualifies nothing and changes no law, format or ABI of the memory or the
 * authority. It is session-local: nothing is persisted, no history is kept, no thread is started.
 *
 * The gap it closes. The ECS_C memory (ecs_memory.h) is source-blind: a slot is keyed by the relation alone, the
 * observations of every source are merged into its counts, only the last 32-bit source_authority is kept, and the
 * source version and scope an observation was admitted under are recorded nowhere. After an authority SOURCE_VERSION,
 * a lifecycle change or a reclamation, LOOKUP keeps answering from evidence the authority's own admission law
 * (elpis_ecs_c_check_evidence) now refuses, and a 32-bit source_authority reused after a reclamation attributes the
 * old evidence to a source that never asserted it.
 *
 * Objects (the real ECS_C types throughout):
 *   authority table  the synced copy of the ECS_C authority's entity rows (elpis_ecs_c_entity, read through
 *                    elpis_ecs_c_read_status and elpis_ecs_c_entity_at) and its coordinate (lineage, generation,
 *                    state root, next founding index, reclaimed count).
 *   binding          (source entity id, source version, scope): what an observation was admitted under. Entity ids
 *                    are never reused (founding indices only grow), so a binding never names two sources.
 *   contribution     the evidence of one relation under one binding: the polarity counts, ticks, admission order and
 *                    provenance of the observations admitted under it (elpis_ecs_mem_observation, exactly as
 *                    OBSERVE takes them). Authoritative input: never derived, never inferred.
 *   relation         a relation key (elpis_ecs_mem_relation_key) with its canonical participants, its contributions,
 *                    and its derived aggregate over the LIVE contributions (an elpis_ecs_mem_slot).
 *
 * The law. Under the synced table A:
 *   LIVE_A(c)       <=> elpis_ecs_c_check_evidence(A, source(c), version(c), scope(c)) == OK
 *                       (the source is a present SOURCE, ACTIVE, at exactly that version and scope)
 *   RETIRED_A(c)    <=> the source is absent (reclaimed) or TERMINATED, or its version exceeds version(c)
 *                       (versions only increase and TERMINATED is final: a retired binding is never live again)
 *   VIEW(A, E)      =   the state an ECS_C memory would hold had it OBSERVEd exactly the observations of the LIVE
 *                       contributions of E, in admission order. Queries answer from it (LOOKUP law, exactly).
 *   admit(A, b, o)  refused unless the binding passes the admission law on A, o.source_authority is the source's,
 *                   o is a valid observation and its tick is not older than the view's clock; then o is added to the
 *                   contribution (relation(o), b). Admission is compare-and-admit: the caller names the authority
 *                   coordinate it holds, which must be the synced one.
 *   sync(A -> A')   refused unless A' is a successor of A (same lineage; generation, founding index, reclaimed count
 *                   and every source version never decrease; TERMINATED stays TERMINATED; an equal generation is an
 *                   identical table); then every RETIRED_A' contribution is released, every contribution whose
 *                   binding changed is re-evaluated, and only the relations that depend on a changed source are
 *                   recomputed. Unrelated relations are not visited.
 * Consequences: a source revision invalidates the evidence admitted under the old version for good (revalidation
 * under a new version admits nothing old); DORMANT suspends evidence and ACTIVE at the same version restores it (the
 * binding is unchanged); reclaiming a source releases its evidence before any new source can reuse its 32-bit
 * source_authority; a query never answers for an authority coordinate the view has not reached (STALE_VIEW).
 *
 * Separation. The view holds admitted evidence only. No function takes a candidate, a salience or an inferred
 * relation; export is read-only, and nothing reads back from any consumer of it.
 *
 * Resources. Fixed at create: entity rows, relations, contributions; one allocation, none afterwards. Every refusal is
 * atomic (the state, its identity and every output stay unchanged). Capacity is refused, never evicted: admitted
 * evidence is never discarded to make room; only RETIRED evidence is released, at sync. Every call reports its
 * logical work. Calls on one handle must not overlap.
 *
 * Integrity. Every relation aggregate and every contribution carries a 64-bit check (accidental damage, not an
 * adversary: in-process code is outside the threat model). A read that meets a failed check refuses CORRUPT;
 * elpis_ecs_c_rel_verify recomputes every derived value and counts mismatches; elpis_ecs_c_rel_rebuild recomputes all
 * derived state exactly from the retained evidence and the synced table (refused CORRUPT when the evidence itself is
 * damaged: reset and re-admit).
 */

#define ELPIS_ECS_C_REL_ABI_VERSION 1u

#define ELPIS_ECS_C_REL_MAX_ENTITIES ELPIS_ECS_C_MAX_CAPACITY
#define ELPIS_ECS_C_REL_MAX_RELATIONS 65536u
#define ELPIS_ECS_C_REL_MAX_CONTRIBUTIONS 262144u
#define ELPIS_ECS_C_REL_MAX_ARENA_BYTES ((uint64_t)1 << 30)

/* Codes. A refusal changes nothing the caller can observe except refusal telemetry. */
#define ELPIS_ECS_C_REL_OK 0
#define ELPIS_ECS_C_REL_INVALID 1          /* malformed argument, configuration, table, observation or request */
#define ELPIS_ECS_C_REL_CAPACITY 2         /* a fixed bound (rows, relations, contributions, output) is exceeded */
#define ELPIS_ECS_C_REL_STALE_VIEW 3       /* never synced, or the caller's coordinate is not the synced one */
#define ELPIS_ECS_C_REL_UNKNOWN_SOURCE 4   /* admission: no row (ECS_C_UNKNOWN_ENTITY) */
#define ELPIS_ECS_C_REL_NOT_SOURCE 5       /* admission: the row is not a SOURCE (ECS_C_UNAUTHORIZED) */
#define ELPIS_ECS_C_REL_LIFECYCLE 6        /* admission: the source is not ACTIVE (ECS_C_LIFECYCLE) */
#define ELPIS_ECS_C_REL_SOURCE_STALE 7     /* admission: not the source's current version (ECS_C_SOURCE_STALE) */
#define ELPIS_ECS_C_REL_SCOPE_CONFLICT 8   /* admission: not the source's current scope (ECS_C_SCOPE_CONFLICT) */
#define ELPIS_ECS_C_REL_BINDING_MISMATCH 9 /* admission: the observation's source_authority is not the source's */
#define ELPIS_ECS_C_REL_CLOCK 10           /* a tick older than the view's clock, or a counter would overflow */
#define ELPIS_ECS_C_REL_LINEAGE 11         /* sync: a table of another lineage */
#define ELPIS_ECS_C_REL_REGRESSION 12      /* sync: the table is not a successor of the synced one (reset to rebuild) */
#define ELPIS_ECS_C_REL_CORRUPT 13         /* retained state failed its integrity check */
#define ELPIS_ECS_C_REL_NOT_FOUND 14       /* no live relation (in the scope) with this key; no such source */
#define ELPIS_ECS_C_REL_NOMEM 15           /* the single allocation failed (create only) */

/* Support states (elpis_ecs_c_rel_support.state). RETIRED contributions are released at sync, never reported. */
#define ELPIS_ECS_C_REL_LIVE 1u
#define ELPIS_ECS_C_REL_SUSPENDED 2u /* the source is present and not TERMINATED, but not ACTIVE (DORMANT) */

typedef struct {
    uint32_t abi_version;           /* ELPIS_ECS_C_REL_ABI_VERSION */
    uint32_t entity_capacity;       /* authority rows mirrored: 2..ELPIS_ECS_C_REL_MAX_ENTITIES */
    uint32_t relation_capacity;     /* 1..ELPIS_ECS_C_REL_MAX_RELATIONS */
    uint32_t contribution_capacity; /* 1..ELPIS_ECS_C_REL_MAX_CONTRIBUTIONS */
    uint64_t byte_budget;           /* the one allocation must fit (and ELPIS_ECS_C_REL_MAX_ARENA_BYTES) */
} elpis_ecs_c_rel_config;

/* An ECS_C authority state as sync consumes it: elpis_ecs_c_read_status (coordinate and counters) and
 * elpis_ecs_c_entity_at for every index below entity_count (the rows, in the authority's order). */
typedef struct {
    uint32_t abi_version; /* ELPIS_ECS_C_REL_ABI_VERSION */
    uint32_t entity_count;
    uint64_t generation;
    uint64_t next_founding_index;
    uint64_t reclaimed_count;
    uint8_t lineage[32];
    uint8_t state_root[32];
    const elpis_ecs_c_entity *entities;
} elpis_ecs_c_rel_authority;

/* The authority coordinate a caller holds: compare-and-admit, and a read pinned to the coordinate it is for. */
typedef struct {
    uint64_t generation;
    uint8_t state_root[32];
} elpis_ecs_c_rel_coordinate;

/* Logical work of one call (records and probes, not instructions). */
typedef struct {
    uint64_t rows;          /* authority rows read */
    uint64_t index_probes;  /* hash-index slots inspected */
    uint64_t contributions; /* contribution records visited */
    uint64_t relations;     /* relation records visited (scans, recomputes, purges) */
    uint64_t recomputes;    /* relation aggregates recomputed */
    uint64_t sorted;        /* records ordered (table validation, export, identity) */
} elpis_ecs_c_rel_work;

typedef struct {
    uint32_t sources_changed;       /* rows whose binding (lifecycle, version, scope) changed, or that vanished */
    uint32_t contributions_flipped; /* live <-> suspended */
    uint32_t contributions_retired; /* released: RETIRED under the new table */
    uint32_t relations_recomputed;
    uint32_t relations_released;    /* no contribution left */
    uint32_t noop;                  /* 1: the identical table at the synced generation (idempotent) */
    elpis_ecs_c_rel_work work;
} elpis_ecs_c_rel_sync_result;

/* One contribution of a relation: which binding supports it, and whether it counts now. In admission order. */
typedef struct {
    uint8_t source[32];
    uint8_t scope[32];
    uint64_t version;
    uint32_t source_authority; /* as the observations recorded it */
    uint32_t state;            /* ELPIS_ECS_C_REL_LIVE or ELPIS_ECS_C_REL_SUSPENDED */
    uint32_t assert_count;
    uint32_t negate_count;
    uint32_t uncertain_count;
    uint32_t reserved;
    uint64_t first_tick;
    uint64_t last_tick;
    elpis_ecs_mem_digest first_provenance;
    elpis_ecs_mem_digest last_provenance;
} elpis_ecs_c_rel_support;

typedef struct {
    elpis_ecs_c_rel_config config;
    uint32_t synced;
    uint32_t entities;
    uint32_t relations;           /* retained (with at least one contribution) */
    uint32_t live_relations;      /* with at least one LIVE contribution */
    uint32_t contributions;
    uint32_t live_contributions;
    uint64_t generation;
    uint8_t lineage[32];
    uint8_t state_root[32];
    uint64_t next_founding_index;
    uint64_t reclaimed_count;
    uint64_t clock;               /* greatest tick admitted */
    uint64_t sequence;            /* observations admitted (the admission order) */
    uint64_t revision;            /* committed mutations (admissions, effective syncs, rebuilds, resets) */
    uint64_t arena_bytes;
    /* Telemetry (not in the identity). */
    uint64_t admissions, syncs, noop_syncs, flips, retired_contributions, released_relations, rebuilds, resets;
    uint64_t refusals;
    uint64_t refusals_by_code[16];
} elpis_ecs_c_rel_status;

typedef struct elpis_ecs_c_rel elpis_ecs_c_rel;

uint32_t elpis_ecs_c_rel_abi_version(void);
const char *elpis_ecs_c_rel_code_name(int code);
void elpis_ecs_c_rel_config_default(elpis_ecs_c_rel_config *out);
/* The exact size of the one allocation of a valid configuration. */
int elpis_ecs_c_rel_required_bytes(const elpis_ecs_c_rel_config *config, uint64_t *arena_bytes);
int elpis_ecs_c_rel_create(const elpis_ecs_c_rel_config *config, elpis_ecs_c_rel **out);
void elpis_ecs_c_rel_destroy(elpis_ecs_c_rel *view);

/* Sync to an authority table (the law above). `out` is optional. */
int elpis_ecs_c_rel_sync(elpis_ecs_c_rel *view, const elpis_ecs_c_rel_authority *table,
                         elpis_ecs_c_rel_sync_result *out);
/* The admission law on the synced table (pure): OK, or the refusal admission would give. */
int elpis_ecs_c_rel_check(const elpis_ecs_c_rel *view, const uint8_t source[32], uint64_t version,
                          const uint8_t scope[32]);
/* Admit one observation under (source, version, scope) at the caller's coordinate. out_key and work are optional. */
int elpis_ecs_c_rel_admit(elpis_ecs_c_rel *view, const elpis_ecs_c_rel_coordinate *expected, const uint8_t source[32],
                          uint64_t version, const uint8_t scope[32], const elpis_ecs_mem_observation *observation,
                          elpis_ecs_mem_digest *out_key, elpis_ecs_c_rel_work *work);

/*
 * Reads. `expected` (optional) pins the read to an authority coordinate (STALE_VIEW unless it is the synced one).
 * `scope` (optional) restricts the evidence to contributions admitted under that scope; NULL is every scope.
 * query: LOOKUP only (elpis_ecs_mem_request with channel LOOKUP; any other channel is INVALID), with the memory's
 * LOOKUP law over the view; the answer's work.slot_scans counts relation scans, `work` (optional) the view's.
 */
int elpis_ecs_c_rel_query(const elpis_ecs_c_rel *view, const elpis_ecs_c_rel_coordinate *expected,
                          const uint8_t *scope, const elpis_ecs_mem_request *request, elpis_ecs_mem_answer *out,
                          elpis_ecs_c_rel_work *work);
int elpis_ecs_c_rel_relation_read(const elpis_ecs_c_rel *view, const elpis_ecs_c_rel_coordinate *expected,
                                  const uint8_t *scope, const elpis_ecs_mem_digest *key, elpis_ecs_mem_slot *out);
/* Provenance: the contributions retained for a relation (LIVE and SUSPENDED), in admission order. CAPACITY with
 * *count the number needed, and nothing written, when `capacity` is short. */
int elpis_ecs_c_rel_support_of(const elpis_ecs_c_rel *view, const elpis_ecs_mem_digest *key,
                               elpis_ecs_c_rel_support *out, size_t capacity, size_t *count);
/* Dependency: the relations holding a contribution of `source` (any version), in admission order. As above. */
int elpis_ecs_c_rel_dependents(const elpis_ecs_c_rel *view, const uint8_t source[32], elpis_ecs_mem_digest *out,
                               size_t capacity, size_t *count);
/* A complete immutable copy of the live view (in the scope), by relation key: all or nothing (CAPACITY with *count
 * the number needed when `capacity` is short). */
int elpis_ecs_c_rel_export(elpis_ecs_c_rel *view, const elpis_ecs_c_rel_coordinate *expected, const uint8_t *scope,
                           elpis_ecs_mem_slot *out, size_t capacity, size_t *count, elpis_ecs_c_rel_work *work);

int elpis_ecs_c_rel_read_status(const elpis_ecs_c_rel *view, elpis_ecs_c_rel_status *out);
/* SHA-256 of the canonical encoding of the coordinate, the table and the retained evidence (in key order; derived
 * state excluded: it is a function of them). */
int elpis_ecs_c_rel_identity(elpis_ecs_c_rel *view, uint8_t out[32]);
/* Recompute every derived value (liveness and aggregates) from the evidence and the table; count disagreements. */
int elpis_ecs_c_rel_verify(const elpis_ecs_c_rel *view, uint64_t *mismatches);
/* Exact full recomputation of all derived state (the complete-rebuild path); atomic. */
int elpis_ecs_c_rel_rebuild(elpis_ecs_c_rel *view, elpis_ecs_c_rel_work *work);
/* Discard everything: unsynced and empty (after REGRESSION, or CORRUPT evidence). */
int elpis_ecs_c_rel_reset(elpis_ecs_c_rel *view);

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(elpis_ecs_c_rel_coordinate) == 40, "elpis_ecs_c_rel_coordinate layout");
_Static_assert(sizeof(elpis_ecs_c_rel_support) == 176, "elpis_ecs_c_rel_support layout");
#endif

#ifdef __cplusplus
}
#endif

#endif
