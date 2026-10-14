#ifndef ELPIS_ECS_MEMORY_H
#define ELPIS_ECS_MEMORY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ECS memory (canonical ECS capability; docs/ECS_MEMORY.md). The bounded, model-free typed relational memory of
 * the one ECS, relocated from the research MF-R0 mechanism (research/ecs_memory_mf_r0, kept byte-frozen as the
 * independent reference): same evidence, codebook and fast-bank laws, envelope (ELPISEM1 v1) and identity, under
 * the canonical prefix. The reference's slow-bank law is not part of the canonical memory: canonical ECS code may
 * carry no unqualified mechanism of that family (the canonical-vocabulary guards of the closed R1-R3 laboratories
 * admit only the qualified K1 milestone; docs/ECS_MEMORY.md). ELPISEM1 keeps its slow banks and their cursor;
 * here they are always empty.
 *
 * It is a component of the ECS retained state, never an authority of its own: a memory handle is either owned
 * by an ECS composite state (ecs_composite.h), whose ELPISEC1 envelope binds it to the K1 state under one
 * identity and one transaction owner, or it is a free-standing handle with no durable authority at all.
 *
 * Capability status (host policy, not state; ELPIS_ECS_MEM_CAP_*):
 *
 *   ADMITTED (mechanically qualified, always on)
 *     OBSERVE        exact typed evidence slots and the entity codebook (bounded, LRU by write use)
 *     QUERY LOOKUP   exact read-only scan of resident evidence
 *     INSPECT, SNAPSHOT/RESTORE, identity
 *   UNQUALIFIED (refused ELPIS_ECS_MEM_UNQUALIFIED unless the capability is set)
 *     LEARN, QUERY FAST            CAP_RECURRENT    (MF-R0 QUAL: H1-H3 PASS in a synthetic regime; H4 NONPASS:
 *                                                    byte-matched exact storage retains more)
 *   NOT PROVIDED (always refused ELPIS_ECS_MEM_UNQUALIFIED)
 *     QUERY SLOW, and any slow-bank content or moved slow-bank cursor on restore (research reference only)
 *   A handle without RECURRENT also refuses to restore a state with nonzero fast banks: unqualified learned
 *   content can never enter through restore.
 *
 * State (fixed at create; one arena, no later allocation):
 *
 *   evidence slots   S typed relation records (resident representation of observed structure: identity,
 *                    roles, ordinals, polarity counts, ticks, provenance). Not truth, not cognition.
 *   codebook         E entity identities recently observed (candidate set for associative cleanup).
 *   fast banks       B integer D x D matrices (Q16), written only by LEARN (gated delta or Hebbian law).
 *   slow banks       B integer D x D matrices (Q16) of the ELPISEM1 layout: always zero here.
 *   clock            the greatest tick admitted; ticks are caller logical time and never decrease.
 *   slow cursor      a u32 of the ELPISEM1 layout: always zero here.
 *
 * The evidence slots and codebook are kept in LRU order of WRITE use (OBSERVE, and LEARN for slots). QUERY
 * never changes recency: it is read-only by contract, so eviction order follows writes only.
 *
 * Arithmetic: integers only. Entity, role, type and target codes are bipolar vectors derived by SHA-256 in
 * counter mode from explicit identifiers (deterministic encoding, not semantics). Banks hold int32 Q16; every
 * product is int64; every division rounds half away from zero. Results are bitwise identical on every host and
 * compiler.
 *
 * Laws (rate = learn_rate_q16 / 65536, retain = retain_q16 / 65536, all exact rationals):
 *   cue(T, target (r*,o*), known K) = majority_{p in K}(role(r_p,o_p) * entity(e_p)) * type(T) * target(r*,o*)
 *                                     (majority ties take the bit of tie(T); K empty gives the all-ones vector)
 *   bank = SHA-256("elpis.ecs-mem.bank.v1" | T | r* | o*) mod B
 *   LEARN (DELTA):   M <- retain M;  y = M c / D;  M <- M + rate (t - y) c^T    (y moves by rate (t - y))
 *   LEARN (HEBBIAN): M <- retain M;                M <- M + rate t c^T          (ablation: no residual)
 *       t = entity(e_target) (Q16 +-1) when the slot's net evidence (asserts - negates) is positive,
 *       t = 0 when it is negative (suppression), refused when it is zero.
 *   QUERY (FAST): y = M c / D; score(candidate) = sum_i y_i entity(candidate)_i. Read only.
 *   QUERY (LOOKUP):    exact scan of resident evidence slots (partial cues allowed). Read only.
 *
 * Refusal law: every non-OK status leaves the authoritative state (everything a snapshot contains) and every
 * output buffer unchanged. Only non-authoritative telemetry (elpis_ecs_mem_stats counters) records refusals.
 *
 * Bounds: every size is computed with checked arithmetic before the single allocation; a configuration whose
 * arena exceeds its declared byte_budget or ELPIS_ECS_MEM_MAX_ARENA_BYTES is refused (CAPACITY). Learning refuses
 * (NUMERIC) when a touched bank holds |w| > ELPIS_ECS_MEM_LEARN_WEIGHT_LIMIT, which proves int32 cannot overflow
 * within one call (docs/ECS_MEMORY.md, "Numerical bound").
 *
 * Concurrency: one writer XOR any number of readers per handle, enforced by a non-blocking guard: a conflicting
 * entry is refused BUSY. The guard has no fairness (readers can starve a writer). Distinct handles share nothing.
 *
 * Ownership: a managed owner may claim a nonzero lease. Every mutating call takes the caller's lease and is
 * refused LEASED unless it equals the held lease (0 when none is held). The lease is not state.
 */

enum {
    ELPIS_ECS_MEM_ABI_V1 = 1u,
    ELPIS_ECS_MEM_FORMAT_VERSION = 1u,
    ELPIS_ECS_MEM_DIGEST_BYTES = 32u,
    ELPIS_ECS_MEM_HEADER_BYTES = 128u,
    ELPIS_ECS_MEM_SLOT_RECORD_BYTES = 544u,
    ELPIS_ECS_MEM_CODEBOOK_RECORD_BYTES = 48u,
    ELPIS_ECS_MEM_MAX_PARTICIPANTS = 8u,
    ELPIS_ECS_MEM_MIN_DIM = 64u,               /* D: a power of two */
    ELPIS_ECS_MEM_MAX_DIM = 1024u,
    ELPIS_ECS_MEM_MAX_BANKS = 64u,
    ELPIS_ECS_MEM_MAX_SLOTS = 65536u,
    ELPIS_ECS_MEM_MAX_CODEBOOK = 65536u,
    ELPIS_ECS_MEM_MAX_CANDIDATES = 65536u,
    ELPIS_ECS_MEM_Q16_ONE = 65536u,
    ELPIS_ECS_MEM_LEARN_WEIGHT_LIMIT = 16777216u, /* 2^24: required of touched banks before learning */
    ELPIS_ECS_MEM_WEIGHT_LIMIT = 1073741824u      /* 2^30: any admissible stored weight */
};
#define ELPIS_ECS_MEM_MAX_ARENA_BYTES ((uint64_t)1 << 30)

typedef enum {
    ELPIS_ECS_MEM_OK = 0,
    ELPIS_ECS_MEM_INVALID = -1,         /* bad argument, handle, configuration, record or reserved field */
    ELPIS_ECS_MEM_CAPACITY = -2,        /* admission refused (REFUSE policy, or more distinct entities than capacity),
                                     or a size beyond the declared byte budget */
    ELPIS_ECS_MEM_NUMERIC = -3,         /* a touched bank is outside the proven learning bound */
    ELPIS_ECS_MEM_CLOCK = -4,           /* tick older than the state clock, or a counter would overflow */
    ELPIS_ECS_MEM_BUSY = -5,            /* concurrent entry */
    ELPIS_ECS_MEM_LEASED = -6,          /* the caller's lease is not the held lease */
    ELPIS_ECS_MEM_CORRUPT = -7,         /* a snapshot failed validation */
    ELPIS_ECS_MEM_NOMEM = -8,           /* the single allocation failed (create, restore only) */
    ELPIS_ECS_MEM_NOT_FOUND = -9,       /* no resident evidence slot has this relation key */
    ELPIS_ECS_MEM_NO_NET_EVIDENCE = -10, /* asserts == negates: nothing to learn */
    ELPIS_ECS_MEM_UNQUALIFIED = -11     /* the operation or restored content needs a capability not admitted */
} elpis_ecs_mem_status;

/* Capabilities (elpis_ecs_mem_config.capabilities). Not state: never serialized. */
enum {
    ELPIS_ECS_MEM_CAP_RECURRENT = 1u    /* LEARN, QUERY FAST: unqualified recurrent learning */
};

typedef enum { ELPIS_ECS_MEM_REFUSE = 0, ELPIS_ECS_MEM_EVICT_LRU = 1 } elpis_ecs_mem_policy;
typedef enum { ELPIS_ECS_MEM_LAW_DELTA = 0, ELPIS_ECS_MEM_LAW_HEBBIAN = 1 } elpis_ecs_mem_law;
typedef enum { ELPIS_ECS_MEM_ASSERT = 1, ELPIS_ECS_MEM_NEGATE = 2, ELPIS_ECS_MEM_UNCERTAIN = 3 } elpis_ecs_mem_polarity;
typedef enum { ELPIS_ECS_MEM_LOOKUP = 1, ELPIS_ECS_MEM_FAST = 2, ELPIS_ECS_MEM_SLOW = 3 } elpis_ecs_mem_channel;
typedef enum { ELPIS_ECS_MEM_CANDIDATES_CALLER = 0, ELPIS_ECS_MEM_CANDIDATES_CODEBOOK = 1 } elpis_ecs_mem_candidate_source;

typedef struct { uint8_t b[ELPIS_ECS_MEM_DIGEST_BYTES]; } elpis_ecs_mem_digest;

typedef struct {
    uint32_t abi_version;          /* ELPIS_ECS_MEM_ABI_V1 */
    uint32_t dim;                  /* D */
    uint32_t banks;                /* B, 1..ELPIS_ECS_MEM_MAX_BANKS */
    uint32_t slot_capacity;        /* S, 1..ELPIS_ECS_MEM_MAX_SLOTS */
    uint32_t codebook_capacity;    /* E, ELPIS_ECS_MEM_MAX_PARTICIPANTS..ELPIS_ECS_MEM_MAX_CODEBOOK */
    uint32_t slot_policy;          /* elpis_ecs_mem_policy */
    uint32_t codebook_policy;      /* elpis_ecs_mem_policy */
    uint32_t law;                  /* elpis_ecs_mem_law */
    uint32_t learn_rate_q16;       /* 1..65536 */
    uint32_t retain_q16;           /* 1..65536 (65536: no decay) */
    uint32_t slow_rate_q16;        /* 1..65536: the reference's slow-bank rate, serialized; unused here */
    uint32_t capabilities;         /* ELPIS_ECS_MEM_CAP_* (0: admitted operations only); policy, not state */
    uint64_t byte_budget;          /* the arena (all state + index + restore workspace) must fit; not state */
} elpis_ecs_mem_config;

typedef struct {
    uint32_t role;                 /* incidence-role identifier (orientation is carried by roles) */
    uint32_t ordinal;              /* position among participants of the same role (multiplicity) */
    elpis_ecs_mem_digest entity;         /* participant identity (e.g. a HACF node identity) */
} elpis_ecs_mem_participant;

typedef struct {
    uint32_t abi_version;          /* ELPIS_ECS_MEM_ABI_V1 */
    uint32_t relation_type;
    uint32_t polarity;             /* elpis_ecs_mem_polarity */
    uint32_t participant_count;    /* 1..ELPIS_ECS_MEM_MAX_PARTICIPANTS; (role, ordinal) pairs must be distinct */
    uint64_t tick;                 /* caller logical time; must be >= the state clock */
    uint32_t source_authority;     /* recorded, never interpreted */
    uint32_t reserved;             /* 0 */
    elpis_ecs_mem_digest payload;        /* part of the relation identity (zero when none) */
    elpis_ecs_mem_digest provenance;     /* recorded (first and last), not part of the identity */
    elpis_ecs_mem_digest structure_ref;  /* opaque external identity (e.g. HACF hyperedge), recorded only */
    elpis_ecs_mem_participant participants[ELPIS_ECS_MEM_MAX_PARTICIPANTS];
} elpis_ecs_mem_observation;

/* A resident evidence slot (also the exact reconstruction of what was observed). Participants are canonical:
 * sorted by (role, ordinal, entity); entries past participant_count are zero. */
typedef struct {
    elpis_ecs_mem_digest key;            /* relation key: type, payload and canonical participants */
    elpis_ecs_mem_digest payload;
    elpis_ecs_mem_digest first_provenance;
    elpis_ecs_mem_digest last_provenance;
    elpis_ecs_mem_digest structure_ref;  /* of the last observation */
    uint32_t relation_type;
    uint32_t participant_count;
    uint32_t assert_count;
    uint32_t negate_count;
    uint32_t uncertain_count;
    uint32_t last_polarity;
    uint32_t last_authority;
    uint32_t learn_count;          /* LEARN calls applied from this slot */
    uint64_t first_tick;
    uint64_t last_tick;
    uint64_t last_assert_tick;     /* 0 when assert_count == 0 */
    uint64_t last_negate_tick;     /* 0 when negate_count == 0 */
    elpis_ecs_mem_participant participants[ELPIS_ECS_MEM_MAX_PARTICIPANTS];
} elpis_ecs_mem_slot;

typedef struct {
    uint32_t abi_version;          /* ELPIS_ECS_MEM_ABI_V1 */
    uint32_t channel;              /* elpis_ecs_mem_channel */
    uint32_t relation_type;
    uint32_t target_role;
    uint32_t target_ordinal;
    uint32_t cue_count;            /* 0..ELPIS_ECS_MEM_MAX_PARTICIPANTS - 1 known participants; none is the target */
    uint32_t candidate_source;     /* FAST only; LOOKUP requires CALLER with no candidates */
    uint32_t candidate_count;      /* CALLER: 1..ELPIS_ECS_MEM_MAX_CANDIDATES */
    const elpis_ecs_mem_digest *candidates;
    elpis_ecs_mem_participant cue[ELPIS_ECS_MEM_MAX_PARTICIPANTS];
} elpis_ecs_mem_request;

typedef struct {
    uint64_t index_probes;         /* hash-index slots inspected */
    uint64_t slot_scans;           /* evidence slots examined (LOOKUP) */
    uint64_t sha_blocks;           /* SHA-256 compressions for codes and keys */
    uint64_t mac_ops;              /* integer multiply-accumulates over banks and scores */
    uint64_t candidates_scored;
} elpis_ecs_mem_work;

typedef struct {
    uint32_t found;                /* 1 if an answer is named */
    uint32_t matches;              /* LOOKUP: matching slots (any polarity); FAST: candidates scored */
    uint32_t conflicts;            /* LOOKUP: other matching slots with positive net evidence for another filler */
    uint32_t contradicted;         /* LOOKUP: matching slots with negative net evidence */
    uint32_t uncertain;            /* LOOKUP: sum of uncertain observations over matching slots */
    uint32_t reserved;
    elpis_ecs_mem_digest answer;
    int64_t score;                 /* LOOKUP: net evidence of the answer slot; FAST: raw integer score */
    int64_t runner_up;             /* second best score (INT64_MIN when none) */
    int64_t signal;                /* FAST: sum |y_i| (0: the bank holds nothing along this cue) */
    uint64_t answer_tick;          /* LOOKUP: last assert tick of the answer slot */
    elpis_ecs_mem_digest answer_key;     /* LOOKUP: relation key of the answer slot */
    elpis_ecs_mem_work work;
} elpis_ecs_mem_answer;

typedef struct {
    elpis_ecs_mem_config config;
    uint64_t clock;
    uint32_t slots_used;
    uint32_t codebook_used;
    uint32_t lease_held;           /* 0 or 1; the token is never reported */
    uint64_t state_bytes;          /* exact snapshot size of the current state */
    uint64_t arena_bytes;          /* the single allocation */
    uint64_t allocated_bytes;      /* arena + handle (allocator overhead excluded) */
    /* Non-authoritative telemetry (not in the snapshot or the identity). */
    uint64_t observes, new_slots, repeated_observations, slot_evictions;
    uint64_t codebook_admissions, codebook_evictions;
    uint64_t learns, learn_bank_updates;
    uint64_t restores, refusals;
    uint64_t refusals_by_code[12]; /* index -status */
    uint64_t index_collisions;     /* displaced insertions (hash collisions in the exact indexes) */
    elpis_ecs_mem_work write_work;       /* accumulated work of committed mutations */
} elpis_ecs_mem_stats;

typedef struct elpis_ecs_mem elpis_ecs_mem;

const char *elpis_ecs_mem_code_name(int status);
void elpis_ecs_mem_config_default(elpis_ecs_mem_config *out);
/* Exact sizes of a valid configuration: the arena and the snapshot of an empty / a full state. */
elpis_ecs_mem_status elpis_ecs_mem_required_bytes(const elpis_ecs_mem_config *config, uint64_t *arena_bytes,
                                      uint64_t *max_state_bytes);

elpis_ecs_mem_status elpis_ecs_mem_create(const elpis_ecs_mem_config *config, elpis_ecs_mem **out);
void elpis_ecs_mem_destroy(elpis_ecs_mem *memory);

elpis_ecs_mem_status elpis_ecs_mem_lease_claim(elpis_ecs_mem *memory, uint64_t lease);      /* nonzero; replaces any lease */
elpis_ecs_mem_status elpis_ecs_mem_lease_release(elpis_ecs_mem *memory, uint64_t lease);    /* must equal the held lease */

/* Pure: the relation key of an observation (validates it as OBSERVE does, except tick and polarity). */
elpis_ecs_mem_status elpis_ecs_mem_relation_key(const elpis_ecs_mem_observation *observation, elpis_ecs_mem_digest *out);

/* OBSERVE: record the observation in the resident evidence (and its entities in the codebook). Never changes
 * the banks. out_key and work are optional. */
elpis_ecs_mem_status elpis_ecs_mem_observe(elpis_ecs_mem *memory, uint64_t lease, const elpis_ecs_mem_observation *observation,
                               elpis_ecs_mem_digest *out_key, elpis_ecs_mem_work *work);
/* LEARN: one law step per participant of the resident slot `key` into the fast banks. */
elpis_ecs_mem_status elpis_ecs_mem_learn(elpis_ecs_mem *memory, uint64_t lease, const elpis_ecs_mem_digest *key, uint64_t tick,
                             elpis_ecs_mem_work *work);
/* QUERY: read only (no learning, no recency, no telemetry). */
elpis_ecs_mem_status elpis_ecs_mem_query(const elpis_ecs_mem *memory, const elpis_ecs_mem_request *query, elpis_ecs_mem_answer *out);

/* INSPECT: read only. */
elpis_ecs_mem_status elpis_ecs_mem_inspect(const elpis_ecs_mem *memory, elpis_ecs_mem_stats *out);
elpis_ecs_mem_status elpis_ecs_mem_slot_read(const elpis_ecs_mem *memory, const elpis_ecs_mem_digest *key, elpis_ecs_mem_slot *out);
elpis_ecs_mem_status elpis_ecs_mem_codebook_contains(const elpis_ecs_mem *memory, const elpis_ecs_mem_digest *entity, int *out);
/* The state identity: SHA-256 trailer of the snapshot (computed by streaming; no buffer). */
elpis_ecs_mem_status elpis_ecs_mem_identity(const elpis_ecs_mem *memory, elpis_ecs_mem_digest *out);

/*
 * SNAPSHOT / RESTORE: the ELPISEM1 v1 envelope (docs/research/ECS_MEMORY_MF_R0.md, "Envelope"), little-endian,
 * everything that determines future behaviour, nothing else, then the SHA-256 of all preceding bytes.
 * restore_into validates the whole envelope before touching the state (no allocation) and requires the
 * envelope's configuration to equal the handle's (byte_budget excepted). restore creates a handle under the
 * caller's byte budget, then restores into it.
 */
size_t elpis_ecs_mem_snapshot_size(const elpis_ecs_mem *memory);
elpis_ecs_mem_status elpis_ecs_mem_snapshot(const elpis_ecs_mem *memory, uint8_t *out, size_t capacity, size_t *written);
elpis_ecs_mem_status elpis_ecs_mem_restore_into(elpis_ecs_mem *memory, uint64_t lease, const uint8_t *bytes, size_t length);
elpis_ecs_mem_status elpis_ecs_mem_restore(const uint8_t *bytes, size_t length, uint64_t byte_budget,
                                           uint32_t capabilities, elpis_ecs_mem **out);

/* Mechanical access to the fixed encoding, for controls and differential references (no state). */
elpis_ecs_mem_status elpis_ecs_mem_entity_code(const elpis_ecs_mem_digest *entity, uint32_t dim, int8_t *out);
elpis_ecs_mem_status elpis_ecs_mem_cue(const elpis_ecs_mem_request *query, uint32_t dim, uint32_t banks, int8_t *out_cue,
                           uint32_t *out_bank);

#ifdef __cplusplus
}
#endif
#endif
