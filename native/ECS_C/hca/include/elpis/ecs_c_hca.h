#ifndef ELPIS_ECS_C_HCA_H
#define ELPIS_ECS_C_HCA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ECS_C HCA: the native research substrate of typed-hypergraph cellular memory (docs/ECS_C_HCA.md).
 *
 * RESEARCH ONLY, GATED. No canonical runtime path links or loads this library (tests/boundary/test_ecs_c_hca.py):
 * not RuntimeCore, not the composite, not the memory, not the Python runtime. It implements no admitted law, makes
 * no semantic claim and qualifies nothing. Every prediction is SALIENCE_ONLY: neither truth nor proof, even at
 * salience 1000. It is the native port of the DEV reference research/ecsc_hca_r0 (baselines.py, task.py, b2.py),
 * proven equal to it by an offline differential (tests/research/ecsc_hca_native), so the three comparators can be
 * studied under one native resource accounting:
 *
 *   B0  exact worklist closure of weighted conjunctions (the exact-storage control: not limited to B1's rounds)
 *   B1  synchronous weighted min-input / max-output rounds (a fixed, unlearned rule)
 *   B2  an unlearned asynchronous recurrence: explicit WARM ticks fire at most updates_per_tick cells in a
 *       counter-derived order; activation damage and reset are explicit; no fairness or convergence guarantee
 *
 * Graph (immutable once built; one allocation at create, none afterwards):
 *   cells    the axioms of one bounded context view: 32-byte key (strictly ascending), scope id, kind id, binding
 *            flag and source state (VALID or AUTHORITY_ONLY admit; UNKNOWN or STALE never do)
 *   schemas  relation_type -> ordered roles (distinct ids) with their kinds, and an integer weight in [1, 1000];
 *            the last role is the output
 *   factors  typed hyperedges (relation_type, status, members in schema role order), strictly ascending by
 *            (relation_type, member keys); every member in one scope, member kinds equal the schema's, at most
 *            `degree` incidences per cell. A factor's source is valid only if every member's is.
 * Message of a factor: weight * min(input scores) / 1000 (integer), 0 unless its source is valid; only its output
 * role receives it. Scores are integers in [0, 1000]; a seed is 1000.
 *
 * Task: scope, seed keys, predicates (relation_type, exact role kinds, admitted statuses), output kinds, readout
 * budget (records: binding constraints + predictions + their direct incoming evidence), work budget, steps (B1
 * rounds / B2 ticks; B0 computes the exact closure), schedule seed, updates per tick. No answer key: scoring is the
 * harness's.
 *
 * Resources: every call charges logical work (records, visits, copies, hashed bytes; not instructions) against its
 * budget and refuses ELPIS_HCA_WORK_LIMIT atomically when it would exceed it. Nothing is persisted, no history is
 * kept, no thread is started: an activation is a caller-owned fixed-size value, and a refused call writes nothing.
 */

#define ELPIS_HCA_ABI_VERSION 1u

#define ELPIS_HCA_MAX_CELLS 256u
#define ELPIS_HCA_MAX_FACTORS 512u
#define ELPIS_HCA_MAX_ARITY 16u
#define ELPIS_HCA_MAX_DEGREE 32u
#define ELPIS_HCA_MAX_ROUNDS 32u
#define ELPIS_HCA_MAX_UPDATES 256u
#define ELPIS_HCA_MAX_WORK 10000000u
#define ELPIS_HCA_MAX_STATE_BYTES 1048576u
#define ELPIS_HCA_MAX_SEEDS 256u
#define ELPIS_HCA_MAX_PREDICATES 512u
#define ELPIS_HCA_MAX_OUTPUT_KINDS 256u
#define ELPIS_HCA_MAX_READOUT 1024u

/* Status codes. A refusal changes nothing the caller can observe except `used` (work charged before it). */
#define ELPIS_HCA_OK 0
#define ELPIS_HCA_INVALID 1            /* malformed argument, limit or task */
#define ELPIS_HCA_CAPACITY 2           /* more records than the limits admit */
#define ELPIS_HCA_WORK_LIMIT 3         /* the call's work budget would be exceeded */
#define ELPIS_HCA_NOT_FOUND 4          /* a seed names no cell */
#define ELPIS_HCA_SCOPE_MISMATCH 5     /* a seed outside the task's scope, or a factor spanning scopes */
#define ELPIS_HCA_SOURCE_INVALID 6     /* a seed whose source is not valid */
#define ELPIS_HCA_ROLE_MISMATCH 7      /* a member's kind is not its schema role's */
#define ELPIS_HCA_ROLE_SCHEMA 8        /* a factor without a schema, or of another arity */
#define ELPIS_HCA_DEGREE_LIMIT 9       /* a cell with more incidences than the degree limit */
#define ELPIS_HCA_ARITY_LIMIT 10       /* a factor or schema wider than the arity limit */
#define ELPIS_HCA_DUPLICATE 11         /* cells, factors or schemas out of strict canonical order */
#define ELPIS_HCA_ROUND_LIMIT 12       /* task steps beyond the graph's rounds limit */
#define ELPIS_HCA_UPDATE_LIMIT 13      /* updates per tick beyond the graph's limit */
#define ELPIS_HCA_BINDING_READOUT 14   /* the scope's binding constraints exceed the readout budget */
#define ELPIS_HCA_STATE_LIMIT 15       /* the B2 state reservation exceeds the graph's state bytes */
#define ELPIS_HCA_INVALID_STATE 16     /* an activation not of this session (owner), malformed, or out of range */
#define ELPIS_HCA_STEP_LIMIT 17        /* the logical tick counter would overflow */
#define ELPIS_HCA_INVALID_DAMAGE 18    /* a damage list naming no, a repeated or an out-of-range cell value */

/* Source states (cells): VALID and AUTHORITY_ONLY admit; UNKNOWN and STALE do not. */
#define ELPIS_HCA_SOURCE_VALID 1u
#define ELPIS_HCA_SOURCE_AUTHORITY_ONLY 2u
#define ELPIS_HCA_SOURCE_UNKNOWN 3u
#define ELPIS_HCA_SOURCE_STALE 4u

/* Factor statuses and predicate status masks. */
#define ELPIS_HCA_CANDIDATE 1u
#define ELPIS_HCA_ADMITTED 2u
#define ELPIS_HCA_MASK_CANDIDATE 1u
#define ELPIS_HCA_MASK_ADMITTED 2u

/* Methods. */
#define ELPIS_HCA_B0 0u
#define ELPIS_HCA_B1 1u
#define ELPIS_HCA_B2 2u

/* Score sentinel for a cell outside the task's scope (elpis_hca_predict `scores`). */
#define ELPIS_HCA_UNSCOPED 0xFFFFu

typedef struct {
    uint32_t cells;            /* 1..256 */
    uint32_t factors;          /* 0..512 */
    uint32_t arity;            /* 2..16 */
    uint32_t degree;           /* 1..32 */
    uint32_t rounds;           /* 1..32 */
    uint32_t updates_per_tick; /* 1..256 */
    uint64_t work;             /* 1..10 000 000: the largest budget a call may ask for */
    uint64_t state_bytes;      /* 1..1 048 576: the B2 state reservation ceiling */
} elpis_hca_limits;

typedef struct {
    uint8_t key[32];
    uint32_t scope;
    uint32_t kind;
    uint8_t binding;      /* 1: a BINDING constraint (always read out in its scope, never a seed by itself) */
    uint8_t source_state; /* ELPIS_HCA_SOURCE_* */
    uint8_t reserved[6];  /* zero */
} elpis_hca_cell;

typedef struct {
    uint32_t relation_type; /* nonzero */
    uint32_t arity;         /* 2..16 */
    uint32_t weight;        /* 1..1000 */
    uint32_t reserved;      /* zero */
    uint32_t roles[ELPIS_HCA_MAX_ARITY]; /* distinct role ids (first `arity`; the rest zero) */
    uint32_t kinds[ELPIS_HCA_MAX_ARITY]; /* each role's member kind */
} elpis_hca_schema;

typedef struct {
    uint32_t relation_type;
    uint8_t status;      /* ELPIS_HCA_CANDIDATE or ELPIS_HCA_ADMITTED */
    uint8_t arity;       /* the members used */
    uint8_t reserved[2]; /* zero */
    uint32_t members[ELPIS_HCA_MAX_ARITY]; /* cell indices in role order; the last is the output */
} elpis_hca_factor;

/* A predicate admits the factors of exactly its schema (relation_type, roles and kinds all equal) whose status is in
 * its mask; one naming other roles or kinds admits nothing (it still counts toward the task's capacity). */
typedef struct {
    uint32_t relation_type;
    uint32_t arity;
    uint32_t status_mask; /* ELPIS_HCA_MASK_*: nonzero */
    uint32_t reserved;    /* zero */
    uint32_t roles[ELPIS_HCA_MAX_ARITY];
    uint32_t kinds[ELPIS_HCA_MAX_ARITY];
} elpis_hca_predicate;

typedef struct {
    uint32_t scope;
    uint32_t seed_count;
    uint32_t predicate_count;
    uint32_t output_kind_count;
    uint32_t readout_budget;   /* 0..1024 */
    uint32_t steps;            /* 1..32 */
    uint32_t updates_per_tick; /* 1..256 */
    uint32_t reserved;         /* zero */
    uint64_t work_budget;      /* 1..10 000 000 */
    uint64_t schedule_seed;
    const uint8_t (*seeds)[32];              /* distinct keys */
    const elpis_hca_predicate *predicates;   /* distinct */
    const uint32_t *output_kinds;            /* distinct */
} elpis_hca_task;

typedef struct {
    uint32_t cell;           /* cell index */
    uint32_t salience;       /* 1..1000 (SALIENCE_ONLY) */
    uint32_t evidence_first; /* into elpis_hca_result.evidence */
    uint32_t evidence_count; /* the cell's direct incoming factors with a positive message, in incidence order */
} elpis_hca_prediction;

typedef struct {
    uint32_t prediction_count;
    uint32_t evidence_count;
    uint32_t constraint_count;
    uint32_t omitted;  /* ranked cells not read out (the readout budget's deterministic prefix) */
    uint64_t work;     /* logical work of this call */
    elpis_hca_prediction predictions[ELPIS_HCA_MAX_READOUT];
    uint32_t evidence[ELPIS_HCA_MAX_READOUT];    /* factor indices */
    uint32_t constraints[ELPIS_HCA_MAX_READOUT]; /* binding cell indices of the scope, in cell order */
} elpis_hca_result;

/* A B2 activation: the scoped cells' values (unsigned 16-bit big-endian, in key order), the logical tick and the
 * owner (the evidence and exact task it belongs to: no cross-session transplant). A caller-owned value. */
typedef struct {
    uint64_t step;
    uint8_t owner[32];
    uint32_t cells;
    uint32_t reserved;
    uint8_t values[2 * ELPIS_HCA_MAX_CELLS];
} elpis_hca_activation;

typedef struct {
    uint32_t cell;  /* cell index (in scope) */
    uint32_t value; /* 0..1000 */
} elpis_hca_damage;

typedef struct elpis_hca_graph elpis_hca_graph;
typedef struct elpis_hca_b2 elpis_hca_b2;

uint32_t elpis_hca_abi_version(void);
const char *elpis_hca_code_name(int code);
void elpis_hca_limits_default(elpis_hca_limits *out);

/* Build an immutable graph under `budget` (<= limits->work). `used` (optional) receives the work charged. */
int elpis_hca_graph_create(const elpis_hca_limits *limits, const elpis_hca_cell *cells, size_t cell_count,
                           const elpis_hca_schema *schemas, size_t schema_count, const elpis_hca_factor *factors,
                           size_t factor_count, uint64_t budget, elpis_hca_graph **out, uint64_t *used);
void elpis_hca_graph_destroy(elpis_hca_graph *graph);
/* The evidence identity: SHA-256 of the canonical encoding of the limits, cells, schemas and factors. */
int elpis_hca_graph_evidence(const elpis_hca_graph *graph, uint8_t out[32]);

/* B0, B1 or B2 (cold start, `steps` ticks and the readout) under the task's one work budget. `scores` (optional, the
 * graph's cell count) receives each cell's final score, ELPIS_HCA_UNSCOPED outside the scope. */
int elpis_hca_predict(const elpis_hca_graph *graph, const elpis_hca_task *task, uint32_t method,
                      elpis_hca_result *out, uint16_t *scores, uint64_t *used);

/* B2 sessions. A session borrows the graph (which must outlive it) and copies the task. Every call is one bounded
 * explicit operation under `budget` (<= the task's work budget; 0 means the task's) and leaves its input untouched. */
int elpis_hca_b2_start(const elpis_hca_graph *graph, const elpis_hca_task *task, uint64_t budget,
                       elpis_hca_b2 **session, elpis_hca_activation *out, uint64_t *used);
int elpis_hca_b2_warm(const elpis_hca_b2 *session, const elpis_hca_activation *in, uint64_t budget,
                      elpis_hca_activation *out, uint64_t *used);
int elpis_hca_b2_damage(const elpis_hca_b2 *session, const elpis_hca_activation *in, const elpis_hca_damage *changes,
                        size_t change_count, uint64_t budget, elpis_hca_activation *out, uint64_t *used);
int elpis_hca_b2_reset(const elpis_hca_b2 *session, uint64_t budget, elpis_hca_activation *out, uint64_t *used);
int elpis_hca_b2_observe(const elpis_hca_b2 *session, const elpis_hca_activation *in, uint64_t budget,
                         elpis_hca_result *out, uint64_t *used);
void elpis_hca_b2_destroy(elpis_hca_b2 *session);

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(elpis_hca_cell) == 48, "elpis_hca_cell layout");
_Static_assert(sizeof(elpis_hca_schema) == 144, "elpis_hca_schema layout");
_Static_assert(sizeof(elpis_hca_factor) == 72, "elpis_hca_factor layout");
_Static_assert(sizeof(elpis_hca_predicate) == 144, "elpis_hca_predicate layout");
_Static_assert(sizeof(elpis_hca_activation) == 560, "elpis_hca_activation layout");
#endif

#ifdef __cplusplus
}
#endif

#endif
