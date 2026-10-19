/* ECS_C HCA: the native research substrate (elpis/ecs_c_hca.h, docs/ECS_C_HCA.md). Research only, gated.
 *
 * The native port of research/ecsc_hca_r0 (baselines.py `message`/`predict`, task.py `prepare`/`readout`, b2.py).
 * Same order of evaluation everywhere it matters: cells in key order, factors in canonical order, a cell's
 * incidences in factor order, B0's worklist popping the least key, B1's synchronous rounds, B2's SHA-256 schedule
 * (the reference's domain, seed and step encoding) and the readout's ranked prefix. Integer arithmetic only.
 *
 * Allocation: one block per graph and one per allocated B2 session, both sized at creation; no other call allocates
 * (elpis_hca_predict runs B2 on a stack session).
 * Every call computes into its own scratch and writes its outputs only on success. */
#include "elpis/ecs_c_hca.h"

#include <stdlib.h>
#include <string.h>

#include "elpis/sha256.h"

#define EVIDENCE_DOMAIN "elpis.ecs-c-hca.v1.evidence"
#define OWNER_DOMAIN "elpis.ecs-c-hca.v1.owner"
/* The reference's B2 schedule domain (b2.py SCHEDULE_DOMAIN), byte for byte: the schedule must be the same. */
static const uint8_t SCHEDULE_DOMAIN[24] = "ecsc-hca-r1.B2.schedule";

struct elpis_hca_graph {
    elpis_hca_limits limits;
    uint32_t cell_count, schema_count, factor_count;
    elpis_hca_cell *cells;
    elpis_hca_schema *schemas;
    elpis_hca_factor *factors;
    uint32_t *factor_schema; /* schema index of each factor */
    uint32_t *factor_scope;  /* the scope of its first member */
    uint8_t *factor_valid;   /* every member's source admits */
    uint8_t *degree;         /* incidences per cell */
    uint32_t *inc_factor;    /* cell * limits.degree + i */
    uint8_t *inc_position;
    uint8_t evidence[32];
};

/* A B2 session: fixed-size, so `elpis_hca_predict` runs one on its stack. An allocated session (b2_start) holds its
 * own copy of the task's arrays in the same block; a stack session borrows the caller's. */
struct elpis_hca_b2 {
    const elpis_hca_graph *graph;
    elpis_hca_task task;
    uint32_t cells;                                /* scoped cells */
    uint32_t factor_on_count;
    uint32_t cell_of[ELPIS_HCA_MAX_CELLS];         /* scoped position -> cell index (key order) */
    uint16_t position_of[ELPIS_HCA_MAX_CELLS];     /* cell index -> scoped position, or UNSCOPED */
    uint8_t factor_on[ELPIS_HCA_MAX_FACTORS];      /* prepared (scoped, predicate-admitted) factors */
    uint8_t seed[ELPIS_HCA_MAX_CELLS];             /* cell index -> is a task seed */
    uint8_t owner[32];
};

/* ---- metering ------------------------------------------------------------------------------------------------ */

typedef struct {
    uint64_t cap, used;
} meter;

static int charge(meter *m, uint64_t n) {
    if (n > m->cap - m->used) return ELPIS_HCA_WORK_LIMIT;
    m->used += n;
    return ELPIS_HCA_OK;
}

#define CHARGE(m, n)                                                                                                \
    do {                                                                                                            \
        const int charge_rc_ = charge((m), (n));                                                                    \
        if (charge_rc_ != ELPIS_HCA_OK) return charge_rc_;                                                          \
    } while (0)
#define TRY(expr)                                                                                                   \
    do {                                                                                                            \
        const int try_rc_ = (expr);                                                                                 \
        if (try_rc_ != ELPIS_HCA_OK) return try_rc_;                                                                \
    } while (0)
#define REQUIRE(cond, code)                                                                                         \
    do {                                                                                                            \
        if (!(cond)) return (code);                                                                                 \
    } while (0)

/* ---- canonical encodings ------------------------------------------------------------------------------------- */

static void put32(elpis_sha256_ctx *h, uint32_t v) {
    uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)};
    elpis_sha256_update(h, b, 4);
}

static void put64(elpis_sha256_ctx *h, uint64_t v) {
    put32(h, (uint32_t)v);
    put32(h, (uint32_t)(v >> 32));
}

static void put_words(elpis_sha256_ctx *h, const uint32_t *w, size_t n) {
    for (size_t i = 0; i < n; ++i) put32(h, w[i]);
}

/* Bytes the evidence encoding hashes (charged as work). */
static uint64_t evidence_bytes(uint32_t cells, uint32_t schemas, uint32_t factors) {
    return sizeof(EVIDENCE_DOMAIN) + 6 * 4 + 2 * 8 + 3 * 4 + (uint64_t)cells * 42 + (uint64_t)schemas * 144 +
           (uint64_t)factors * 72;
}

static void hash_evidence(const elpis_hca_graph *g, uint8_t out[32]) {
    elpis_sha256_ctx h;
    elpis_sha256_init(&h);
    elpis_sha256_update(&h, EVIDENCE_DOMAIN, sizeof(EVIDENCE_DOMAIN));
    const elpis_hca_limits *l = &g->limits;
    put32(&h, l->cells), put32(&h, l->factors), put32(&h, l->arity), put32(&h, l->degree), put32(&h, l->rounds);
    put32(&h, l->updates_per_tick), put64(&h, l->work), put64(&h, l->state_bytes);
    put32(&h, g->cell_count), put32(&h, g->schema_count), put32(&h, g->factor_count);
    for (uint32_t i = 0; i < g->cell_count; ++i) {
        const elpis_hca_cell *c = &g->cells[i];
        elpis_sha256_update(&h, c->key, 32);
        put32(&h, c->scope), put32(&h, c->kind);
        elpis_sha256_update(&h, &c->binding, 1);
        elpis_sha256_update(&h, &c->source_state, 1);
    }
    for (uint32_t i = 0; i < g->schema_count; ++i) {
        const elpis_hca_schema *s = &g->schemas[i];
        put32(&h, s->relation_type), put32(&h, s->arity), put32(&h, s->weight), put32(&h, 0);
        put_words(&h, s->roles, ELPIS_HCA_MAX_ARITY);
        put_words(&h, s->kinds, ELPIS_HCA_MAX_ARITY);
    }
    for (uint32_t i = 0; i < g->factor_count; ++i) {
        const elpis_hca_factor *f = &g->factors[i];
        put32(&h, f->relation_type), put32(&h, (uint32_t)f->status | (uint32_t)f->arity << 8);
        put_words(&h, f->members, ELPIS_HCA_MAX_ARITY);
    }
    elpis_sha256_final(&h, out);
}

static uint64_t owner_bytes(const elpis_hca_task *t) {
    return sizeof(OWNER_DOMAIN) + 32 + 8 * 4 + 2 * 8 + (uint64_t)t->seed_count * 32 +
           (uint64_t)t->predicate_count * 144 + (uint64_t)t->output_kind_count * 4;
}

static void hash_owner(const elpis_hca_graph *g, const elpis_hca_task *t, uint8_t out[32]) {
    elpis_sha256_ctx h;
    elpis_sha256_init(&h);
    elpis_sha256_update(&h, OWNER_DOMAIN, sizeof(OWNER_DOMAIN));
    elpis_sha256_update(&h, g->evidence, 32);
    put32(&h, t->scope), put32(&h, t->seed_count), put32(&h, t->predicate_count), put32(&h, t->output_kind_count);
    put32(&h, t->readout_budget), put32(&h, t->steps), put32(&h, t->updates_per_tick), put32(&h, 0);
    put64(&h, t->work_budget), put64(&h, t->schedule_seed);
    for (uint32_t i = 0; i < t->seed_count; ++i) elpis_sha256_update(&h, t->seeds[i], 32);
    for (uint32_t i = 0; i < t->predicate_count; ++i) {
        const elpis_hca_predicate *p = &t->predicates[i];
        put32(&h, p->relation_type), put32(&h, p->arity), put32(&h, p->status_mask), put32(&h, 0);
        put_words(&h, p->roles, ELPIS_HCA_MAX_ARITY);
        put_words(&h, p->kinds, ELPIS_HCA_MAX_ARITY);
    }
    put_words(&h, t->output_kinds, t->output_kind_count);
    elpis_sha256_final(&h, out);
}

/* ---- small helpers ------------------------------------------------------------------------------------------- */

static int all_zero(const void *p, size_t n) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; ++i)
        if (b[i]) return 0;
    return 1;
}

static int source_admits(uint8_t state) {
    return state == ELPIS_HCA_SOURCE_VALID || state == ELPIS_HCA_SOURCE_AUTHORITY_ONLY;
}

/* Distinct ids among the first `n`, the rest of the array zero. */
static int distinct_words(const uint32_t *w, uint32_t n, uint32_t width) {
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = i + 1; j < n; ++j)
            if (w[i] == w[j]) return 0;
    return all_zero(w + n, (size_t)(width - n) * sizeof(uint32_t));
}

static int64_t find_cell(const elpis_hca_graph *g, const uint8_t key[32]) {
    uint32_t lo = 0, hi = g->cell_count;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        const int c = memcmp(g->cells[mid].key, key, 32);
        if (c == 0) return mid;
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    return -1;
}

static int64_t find_schema(const elpis_hca_graph *g, uint32_t relation_type) {
    uint32_t lo = 0, hi = g->schema_count;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (g->schemas[mid].relation_type == relation_type) return mid;
        if (g->schemas[mid].relation_type < relation_type) lo = mid + 1;
        else hi = mid;
    }
    return -1;
}

/* Strict canonical order of factors: (relation_type, member keys lexicographically, shorter first on a tie). */
static int factor_cmp(const elpis_hca_graph *g, const elpis_hca_factor *a, const elpis_hca_factor *b) {
    if (a->relation_type != b->relation_type) return a->relation_type < b->relation_type ? -1 : 1;
    const uint32_t n = a->arity < b->arity ? a->arity : b->arity;
    for (uint32_t i = 0; i < n; ++i) {
        const int c = memcmp(g->cells[a->members[i]].key, g->cells[b->members[i]].key, 32);
        if (c) return c;
    }
    return a->arity == b->arity ? 0 : (a->arity < b->arity ? -1 : 1);
}

/* ---- public: constants ------------------------------------------------------------------------------------- */

uint32_t elpis_hca_abi_version(void) { return ELPIS_HCA_ABI_VERSION; }

const char *elpis_hca_code_name(int code) {
    static const char *const names[] = {
        "ELPIS_HCA_OK",           "ELPIS_HCA_INVALID",        "ELPIS_HCA_CAPACITY",     "ELPIS_HCA_WORK_LIMIT",
        "ELPIS_HCA_NOT_FOUND",    "ELPIS_HCA_SCOPE_MISMATCH", "ELPIS_HCA_SOURCE_INVALID",
        "ELPIS_HCA_ROLE_MISMATCH", "ELPIS_HCA_ROLE_SCHEMA",   "ELPIS_HCA_DEGREE_LIMIT", "ELPIS_HCA_ARITY_LIMIT",
        "ELPIS_HCA_DUPLICATE",    "ELPIS_HCA_ROUND_LIMIT",    "ELPIS_HCA_UPDATE_LIMIT",
        "ELPIS_HCA_BINDING_READOUT", "ELPIS_HCA_STATE_LIMIT", "ELPIS_HCA_INVALID_STATE", "ELPIS_HCA_STEP_LIMIT",
        "ELPIS_HCA_INVALID_DAMAGE"};
    return code >= 0 && (size_t)code < sizeof(names) / sizeof(names[0]) ? names[code] : NULL;
}

void elpis_hca_limits_default(elpis_hca_limits *out) {
    if (!out) return;
    /* baselines.py Limits defaults. */
    *out = (elpis_hca_limits){32, 32, 8, 8, 8, 256, 100000, 65536};
}

static int limits_valid(const elpis_hca_limits *l) {
    return l->cells >= 1 && l->cells <= ELPIS_HCA_MAX_CELLS && l->factors <= ELPIS_HCA_MAX_FACTORS &&
           l->arity >= 2 && l->arity <= ELPIS_HCA_MAX_ARITY && l->degree >= 1 && l->degree <= ELPIS_HCA_MAX_DEGREE &&
           l->rounds >= 1 && l->rounds <= ELPIS_HCA_MAX_ROUNDS && l->updates_per_tick >= 1 &&
           l->updates_per_tick <= ELPIS_HCA_MAX_UPDATES && l->work >= 1 && l->work <= ELPIS_HCA_MAX_WORK &&
           l->state_bytes >= 1 && l->state_bytes <= ELPIS_HCA_MAX_STATE_BYTES;
}

/* ---- the graph ------------------------------------------------------------------------------------------------ */

static size_t align8(size_t n) { return (n + 7) & ~(size_t)7; }

static int build(elpis_hca_graph *g, const elpis_hca_cell *cells, const elpis_hca_schema *schemas,
                 const elpis_hca_factor *factors, meter *m) {
    const elpis_hca_limits *l = &g->limits;
    CHARGE(m, 3 + g->cell_count + g->schema_count + g->factor_count); /* collection and record checks */
    for (uint32_t i = 0; i < g->factor_count; ++i) {
        CHARGE(m, 1);
        REQUIRE(factors[i].arity >= 2 && factors[i].arity <= ELPIS_HCA_MAX_ARITY, ELPIS_HCA_INVALID);
        REQUIRE(factors[i].arity <= l->arity, ELPIS_HCA_ARITY_LIMIT);
    }
    for (uint32_t i = 0; i < g->schema_count; ++i) {
        CHARGE(m, 1);
        REQUIRE(schemas[i].arity >= 2 && schemas[i].arity <= ELPIS_HCA_MAX_ARITY, ELPIS_HCA_INVALID);
        REQUIRE(schemas[i].arity <= l->arity, ELPIS_HCA_ARITY_LIMIT);
    }
    /* Cells: strictly ascending keys, well-formed records. */
    for (uint32_t i = 0; i < g->cell_count; ++i) {
        CHARGE(m, 1);
        const elpis_hca_cell *c = &cells[i];
        REQUIRE(c->binding <= 1 && c->source_state >= ELPIS_HCA_SOURCE_VALID &&
                    c->source_state <= ELPIS_HCA_SOURCE_STALE && all_zero(c->reserved, sizeof(c->reserved)),
                ELPIS_HCA_INVALID);
        REQUIRE(i == 0 || memcmp(cells[i - 1].key, c->key, 32) < 0, ELPIS_HCA_DUPLICATE);
        g->cells[i] = *c;
    }
    /* Schemas: strictly ascending relation types. */
    for (uint32_t i = 0; i < g->schema_count; ++i) {
        CHARGE(m, 1);
        const elpis_hca_schema *s = &schemas[i];
        REQUIRE(s->relation_type != 0 && s->weight >= 1 && s->weight <= 1000 && s->reserved == 0 &&
                    distinct_words(s->roles, s->arity, ELPIS_HCA_MAX_ARITY) &&
                    all_zero(s->kinds + s->arity, (ELPIS_HCA_MAX_ARITY - s->arity) * sizeof(uint32_t)),
                ELPIS_HCA_INVALID);
        REQUIRE(i == 0 || schemas[i - 1].relation_type < s->relation_type, ELPIS_HCA_DUPLICATE);
        g->schemas[i] = *s;
    }
    /* Factors: well-formed, strictly ascending, schema-typed, one scope, within the degree limit. */
    memset(g->degree, 0, g->cell_count);
    for (uint32_t i = 0; i < g->factor_count; ++i) {
        CHARGE(m, 1);
        const elpis_hca_factor *f = &factors[i];
        REQUIRE(f->relation_type != 0 && (f->status == ELPIS_HCA_CANDIDATE || f->status == ELPIS_HCA_ADMITTED) &&
                    all_zero(f->reserved, sizeof(f->reserved)) &&
                    all_zero(f->members + f->arity, (ELPIS_HCA_MAX_ARITY - f->arity) * sizeof(uint32_t)),
                ELPIS_HCA_INVALID);
        for (uint32_t p = 0; p < f->arity; ++p) {
            REQUIRE(f->members[p] < g->cell_count, ELPIS_HCA_INVALID);
            for (uint32_t q = 0; q < p; ++q) REQUIRE(f->members[q] != f->members[p], ELPIS_HCA_INVALID);
        }
        g->factors[i] = *f;
        REQUIRE(i == 0 || factor_cmp(g, &g->factors[i - 1], f) < 0, ELPIS_HCA_DUPLICATE);
        const int64_t s = find_schema(g, f->relation_type);
        REQUIRE(s >= 0 && g->schemas[s].arity == f->arity, ELPIS_HCA_ROLE_SCHEMA);
        const elpis_hca_schema *schema = &g->schemas[s];
        const uint32_t scope = g->cells[f->members[0]].scope;
        uint8_t valid = 1;
        for (uint32_t p = 0; p < f->arity; ++p) {
            CHARGE(m, 1);
            const uint32_t c = f->members[p];
            REQUIRE(g->cells[c].scope == scope, ELPIS_HCA_SCOPE_MISMATCH);
            REQUIRE(g->cells[c].kind == schema->kinds[p], ELPIS_HCA_ROLE_MISMATCH);
            REQUIRE(g->degree[c] < l->degree, ELPIS_HCA_DEGREE_LIMIT);
            g->inc_factor[(size_t)c * l->degree + g->degree[c]] = i;
            g->inc_position[(size_t)c * l->degree + g->degree[c]] = (uint8_t)p;
            g->degree[c]++;
            valid = valid && source_admits(g->cells[c].source_state);
        }
        g->factor_schema[i] = (uint32_t)s;
        g->factor_scope[i] = scope;
        g->factor_valid[i] = valid;
    }
    CHARGE(m, evidence_bytes(g->cell_count, g->schema_count, g->factor_count));
    hash_evidence(g, g->evidence);
    return ELPIS_HCA_OK;
}

int elpis_hca_graph_create(const elpis_hca_limits *limits, const elpis_hca_cell *cells, size_t cell_count,
                           const elpis_hca_schema *schemas, size_t schema_count, const elpis_hca_factor *factors,
                           size_t factor_count, uint64_t budget, elpis_hca_graph **out, uint64_t *used) {
    if (used) *used = 0;
    if (out) *out = NULL;
    if (!out || !limits || !limits_valid(limits)) return ELPIS_HCA_INVALID;
    if (budget < 1 || budget > limits->work) return ELPIS_HCA_INVALID;
    if ((cell_count && !cells) || (schema_count && !schemas) || (factor_count && !factors)) return ELPIS_HCA_INVALID;
    if (cell_count > limits->cells || factor_count > limits->factors || schema_count > limits->factors)
        return ELPIS_HCA_CAPACITY;
    /* One block, sized by the actual records and the degree limit. */
    const size_t degree = limits->degree;
    const size_t sizes[] = {sizeof(elpis_hca_graph),
                            cell_count * sizeof(elpis_hca_cell),
                            schema_count * sizeof(elpis_hca_schema),
                            factor_count * sizeof(elpis_hca_factor),
                            factor_count * sizeof(uint32_t),
                            factor_count * sizeof(uint32_t),
                            cell_count * degree * sizeof(uint32_t),
                            factor_count,
                            cell_count,
                            cell_count * degree};
    size_t total = 0, at[sizeof(sizes) / sizeof(sizes[0])];
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        at[i] = total;
        total += align8(sizes[i]);
    }
    uint8_t *block = calloc(1, total);
    if (!block) return ELPIS_HCA_CAPACITY;
    elpis_hca_graph *g = (elpis_hca_graph *)block;
    g->limits = *limits;
    g->cell_count = (uint32_t)cell_count;
    g->schema_count = (uint32_t)schema_count;
    g->factor_count = (uint32_t)factor_count;
    g->cells = (elpis_hca_cell *)(block + at[1]);
    g->schemas = (elpis_hca_schema *)(block + at[2]);
    g->factors = (elpis_hca_factor *)(block + at[3]);
    g->factor_schema = (uint32_t *)(block + at[4]);
    g->factor_scope = (uint32_t *)(block + at[5]);
    g->inc_factor = (uint32_t *)(block + at[6]);
    g->factor_valid = block + at[7];
    g->degree = block + at[8];
    g->inc_position = block + at[9];
    meter m = {budget, 0};
    const int rc = build(g, cells, schemas, factors, &m);
    if (used) *used = m.used;
    if (rc != ELPIS_HCA_OK) {
        free(block);
        return rc;
    }
    *out = g;
    return ELPIS_HCA_OK;
}

void elpis_hca_graph_destroy(elpis_hca_graph *graph) { free(graph); }

int elpis_hca_graph_evidence(const elpis_hca_graph *graph, uint8_t out[32]) {
    if (!graph || !out) return ELPIS_HCA_INVALID;
    memcpy(out, graph->evidence, 32);
    return ELPIS_HCA_OK;
}

/* ---- the task ---------------------------------------------------------------------------------------------- */

/* The task's own well-formedness (task.py Task / Predicate: refused before any work). */
static int task_valid(const elpis_hca_task *t) {
    if (t->reserved || t->seed_count > ELPIS_HCA_MAX_SEEDS || t->predicate_count > ELPIS_HCA_MAX_PREDICATES ||
        t->output_kind_count < 1 || t->output_kind_count > ELPIS_HCA_MAX_OUTPUT_KINDS ||
        t->readout_budget > ELPIS_HCA_MAX_READOUT || t->work_budget < 1 || t->work_budget > ELPIS_HCA_MAX_WORK ||
        t->steps < 1 || t->steps > ELPIS_HCA_MAX_ROUNDS || t->updates_per_tick < 1 ||
        t->updates_per_tick > ELPIS_HCA_MAX_UPDATES)
        return 0;
    if ((t->seed_count && !t->seeds) || (t->predicate_count && !t->predicates) || !t->output_kinds) return 0;
    for (uint32_t i = 0; i < t->seed_count; ++i)
        for (uint32_t j = i + 1; j < t->seed_count; ++j)
            if (!memcmp(t->seeds[i], t->seeds[j], 32)) return 0;
    for (uint32_t i = 0; i < t->predicate_count; ++i) {
        const elpis_hca_predicate *p = &t->predicates[i];
        if (p->relation_type == 0 || p->arity < 2 || p->arity > ELPIS_HCA_MAX_ARITY || p->reserved ||
            p->status_mask < 1 || p->status_mask > 3 || !distinct_words(p->roles, p->arity, ELPIS_HCA_MAX_ARITY) ||
            !all_zero(p->kinds + p->arity, (ELPIS_HCA_MAX_ARITY - p->arity) * sizeof(uint32_t)))
            return 0;
        for (uint32_t j = 0; j < i; ++j)
            if (!memcmp(&t->predicates[j], p, sizeof(*p))) return 0;
    }
    for (uint32_t i = 0; i < t->output_kind_count; ++i)
        for (uint32_t j = i + 1; j < t->output_kind_count; ++j)
            if (t->output_kinds[i] == t->output_kinds[j]) return 0;
    return 1;
}

/* The call's budget: 0 is the task's; otherwise 1..the task's. */
static int budget_of(const elpis_hca_task *t, uint64_t budget, meter *m) {
    if (budget > t->work_budget) return ELPIS_HCA_INVALID;
    *m = (meter){budget ? budget : t->work_budget, 0};
    return ELPIS_HCA_OK;
}

typedef struct {
    uint16_t scores[ELPIS_HCA_MAX_CELLS];
    uint8_t scoped[ELPIS_HCA_MAX_CELLS];
    uint8_t seed[ELPIS_HCA_MAX_CELLS];
    uint8_t factor_on[ELPIS_HCA_MAX_FACTORS];
    uint32_t scoped_count, factor_on_count;
} prepared;

static int predicate_matches(const elpis_hca_predicate *p, const elpis_hca_schema *s) {
    return p->relation_type == s->relation_type && p->arity == s->arity &&
           !memcmp(p->roles, s->roles, sizeof(p->roles)) && !memcmp(p->kinds, s->kinds, sizeof(p->kinds));
}

/* task.py `prepare`: the scoped cells at zero, the scoped predicate-admitted factors, the seeds at 1000. */
static int prepare(const elpis_hca_graph *g, const elpis_hca_task *t, meter *m, prepared *p) {
    REQUIRE(t->seed_count <= g->limits.cells && t->predicate_count <= g->limits.factors &&
                t->output_kind_count <= g->limits.cells,
            ELPIS_HCA_CAPACITY);
    REQUIRE(t->steps <= g->limits.rounds, ELPIS_HCA_ROUND_LIMIT);
    REQUIRE(t->updates_per_tick <= g->limits.updates_per_tick, ELPIS_HCA_UPDATE_LIMIT);
    CHARGE(m, t->predicate_count);
    uint8_t schema_mask[ELPIS_HCA_MAX_FACTORS] = {0};
    for (uint32_t i = 0; i < t->predicate_count; ++i) {
        const int64_t s = find_schema(g, t->predicates[i].relation_type);
        if (s >= 0 && predicate_matches(&t->predicates[i], &g->schemas[s]))
            schema_mask[s] |= (uint8_t)t->predicates[i].status_mask;
    }
    memset(p->scoped, 0, sizeof(p->scoped));
    memset(p->seed, 0, sizeof(p->seed));
    p->scoped_count = p->factor_on_count = 0;
    for (uint32_t c = 0; c < g->cell_count; ++c) {
        CHARGE(m, 1);
        p->scores[c] = 0;
        if (g->cells[c].scope == t->scope) p->scoped[c] = 1, p->scoped_count++;
    }
    for (uint32_t f = 0; f < g->factor_count; ++f) {
        CHARGE(m, 1);
        const uint8_t bit = g->factors[f].status == ELPIS_HCA_CANDIDATE ? ELPIS_HCA_MASK_CANDIDATE
                                                                         : ELPIS_HCA_MASK_ADMITTED;
        p->factor_on[f] = g->factor_scope[f] == t->scope && (schema_mask[g->factor_schema[f]] & bit);
        p->factor_on_count += p->factor_on[f];
    }
    for (uint32_t i = 0; i < t->seed_count; ++i) {
        CHARGE(m, 1);
        const int64_t c = find_cell(g, t->seeds[i]);
        REQUIRE(c >= 0, ELPIS_HCA_NOT_FOUND);
        REQUIRE(p->scoped[c], ELPIS_HCA_SCOPE_MISMATCH);
        REQUIRE(source_admits(g->cells[c].source_state), ELPIS_HCA_SOURCE_INVALID);
        p->scores[c] = 1000;
        p->seed[c] = 1;
    }
    return ELPIS_HCA_OK;
}

/* baselines/task.py `message`: weight * min(input scores) / 1000, 0 unless the factor's source is valid. */
static int message(const elpis_hca_graph *g, uint32_t f, const uint16_t *scores, meter *m, uint32_t *out) {
    CHARGE(m, 1);
    const elpis_hca_factor *factor = &g->factors[f];
    uint32_t value = 1000;
    for (uint32_t p = 0; p + 1 < factor->arity; ++p) {
        CHARGE(m, 1);
        if (scores[factor->members[p]] < value) value = scores[factor->members[p]];
    }
    CHARGE(m, 1);
    *out = g->factor_valid[f] ? g->schemas[g->factor_schema[f]].weight * value / 1000 : 0;
    return ELPIS_HCA_OK;
}

/* The ranked readout order: descending score, then ascending key (cell index). */
static void rank(uint32_t *cells, uint32_t n, const uint16_t *scores) {
    for (uint32_t i = 1; i < n; ++i) {
        const uint32_t c = cells[i];
        uint32_t j = i;
        while (j > 0 && (scores[cells[j - 1]] < scores[c] || (scores[cells[j - 1]] == scores[c] && cells[j - 1] > c))) {
            cells[j] = cells[j - 1];
            --j;
        }
        cells[j] = c;
    }
}

/* task.py `readout`: every binding constraint of the scope, then the ranked prefix the budget admits. */
static int readout(const elpis_hca_graph *g, const elpis_hca_task *t, const prepared *p, const uint16_t *scores,
                   meter *m, elpis_hca_result *r) {
    uint32_t ranked[ELPIS_HCA_MAX_CELLS], ranked_count = 0;
    memset(r, 0, sizeof(*r)); /* every byte of a result is defined: entries past the counts are zero */
    for (uint32_t c = 0; c < g->cell_count; ++c) {
        CHARGE(m, 1);
        if (!p->scoped[c]) continue;
        if (g->cells[c].binding) {
            if (r->constraint_count == ELPIS_HCA_MAX_READOUT) return ELPIS_HCA_BINDING_READOUT;
            r->constraints[r->constraint_count++] = c;
        }
        int kind = 0;
        for (uint32_t k = 0; k < t->output_kind_count && !kind; ++k) kind = g->cells[c].kind == t->output_kinds[k];
        if (scores[c] > 0 && source_admits(g->cells[c].source_state) && kind) ranked[ranked_count++] = c;
    }
    REQUIRE(r->constraint_count <= t->readout_budget, ELPIS_HCA_BINDING_READOUT);
    uint32_t remaining = t->readout_budget - r->constraint_count;
    CHARGE(m, r->constraint_count);
    rank(ranked, ranked_count, scores);
    for (uint32_t i = 0; i < ranked_count; ++i) {
        const uint32_t c = ranked[i];
        uint32_t evidence[ELPIS_HCA_MAX_DEGREE], n = 0;
        for (uint32_t k = 0; k < g->degree[c]; ++k) {
            CHARGE(m, 1);
            const uint32_t f = g->inc_factor[(size_t)c * g->limits.degree + k];
            if (!p->factor_on[f] || g->inc_position[(size_t)c * g->limits.degree + k] != g->factors[f].arity - 1)
                continue;
            uint32_t value;
            TRY(message(g, f, scores, m, &value));
            if (value > 0) evidence[n++] = f;
        }
        const uint32_t cost = 1 + n;
        if (cost > remaining) break; /* the deterministic ranked prefix; the rest is counted as omitted */
        CHARGE(m, cost);
        elpis_hca_prediction *out = &r->predictions[r->prediction_count++];
        *out = (elpis_hca_prediction){c, scores[c], r->evidence_count, n};
        memcpy(r->evidence + r->evidence_count, evidence, n * sizeof(uint32_t));
        r->evidence_count += n;
        remaining -= cost;
    }
    r->omitted = ranked_count - r->prediction_count;
    return ELPIS_HCA_OK;
}

/* ---- B0 and B1 ------------------------------------------------------------------------------------------------- */

/* A binary min-heap of cell indices (cell order is key order: the reference pops the least key). */
typedef struct {
    uint32_t items[ELPIS_HCA_MAX_CELLS];
    uint32_t n;
} heap;

static void heap_push(heap *h, uint32_t v) {
    uint32_t i = h->n++;
    while (i > 0 && h->items[(i - 1) / 2] > v) {
        h->items[i] = h->items[(i - 1) / 2];
        i = (i - 1) / 2;
    }
    h->items[i] = v;
}

static uint32_t heap_pop(heap *h) {
    const uint32_t top = h->items[0], last = h->items[--h->n];
    uint32_t i = 0;
    for (;;) {
        uint32_t child = 2 * i + 1;
        if (child >= h->n) break;
        if (child + 1 < h->n && h->items[child + 1] < h->items[child]) child++;
        if (h->items[child] >= last) break;
        h->items[i] = h->items[child];
        i = child;
    }
    if (h->n) h->items[i] = last;
    return top;
}

static int b0(const elpis_hca_graph *g, const elpis_hca_task *t, prepared *p, meter *m) {
    heap q = {.n = 0};
    uint8_t queued[ELPIS_HCA_MAX_CELLS] = {0};
    CHARGE(m, t->seed_count);
    for (uint32_t c = 0; c < g->cell_count; ++c)
        if (p->seed[c]) heap_push(&q, c), queued[c] = 1;
    while (q.n) {
        CHARGE(m, 1);
        const uint32_t c = heap_pop(&q);
        queued[c] = 0;
        for (uint32_t k = 0; k < g->degree[c]; ++k) {
            CHARGE(m, 1);
            const uint32_t f = g->inc_factor[(size_t)c * g->limits.degree + k];
            if (!p->factor_on[f] || g->inc_position[(size_t)c * g->limits.degree + k] == g->factors[f].arity - 1)
                continue;
            uint32_t value;
            TRY(message(g, f, p->scores, m, &value));
            const uint32_t out = g->factors[f].members[g->factors[f].arity - 1];
            CHARGE(m, 1);
            if (value > p->scores[out]) {
                p->scores[out] = (uint16_t)value;
                if (!queued[out]) {
                    CHARGE(m, 1);
                    heap_push(&q, out);
                    queued[out] = 1;
                }
            }
        }
    }
    return ELPIS_HCA_OK;
}

static int b1(const elpis_hca_graph *g, const elpis_hca_task *t, prepared *p, meter *m) {
    uint16_t next[ELPIS_HCA_MAX_CELLS];
    for (uint32_t step = 0; step < t->steps; ++step) {
        CHARGE(m, p->scoped_count);
        memcpy(next, p->scores, g->cell_count * sizeof(uint16_t));
        for (uint32_t f = 0; f < g->factor_count; ++f) {
            if (!p->factor_on[f]) continue;
            uint32_t value;
            TRY(message(g, f, p->scores, m, &value));
            const uint32_t out = g->factors[f].members[g->factors[f].arity - 1];
            CHARGE(m, 1);
            if (value > next[out]) next[out] = (uint16_t)value;
        }
        memcpy(p->scores, next, g->cell_count * sizeof(uint16_t));
    }
    return ELPIS_HCA_OK;
}

/* ---- B2 -------------------------------------------------------------------------------------------------------- */

typedef struct {
    uint8_t priority[32];
    uint32_t cell;
} scheduled;

static int schedule_before(const scheduled *a, const scheduled *b) {
    const int c = memcmp(a->priority, b->priority, 32);
    return c < 0 || (c == 0 && a->cell < b->cell);
}

/* b2.py `_schedule`: the scoped cells ordered by SHA-256(domain | seed | step | key), ties by key. */
static int schedule(const elpis_hca_b2 *s, uint64_t step, meter *m, uint32_t *order) {
    scheduled entries[ELPIS_HCA_MAX_CELLS];
    for (uint32_t i = 0; i < s->cells; ++i) {
        CHARGE(m, sizeof(SCHEDULE_DOMAIN) + 48);
        uint8_t message_bytes[sizeof(SCHEDULE_DOMAIN) + 16 + 32];
        memcpy(message_bytes, SCHEDULE_DOMAIN, sizeof(SCHEDULE_DOMAIN));
        for (size_t b = 0; b < 8; ++b) { /* big-endian, as the reference's to_bytes(8, "big") */
            message_bytes[sizeof(SCHEDULE_DOMAIN) + b] = (uint8_t)(s->task.schedule_seed >> (56 - 8 * b));
            message_bytes[sizeof(SCHEDULE_DOMAIN) + 8 + b] = (uint8_t)(step >> (56 - 8 * b));
        }
        memcpy(message_bytes + sizeof(SCHEDULE_DOMAIN) + 16, s->graph->cells[s->cell_of[i]].key, 32);
        elpis_sha256(message_bytes, sizeof(message_bytes), entries[i].priority);
        entries[i].cell = s->cell_of[i];
    }
    CHARGE(m, s->cells);
    for (uint32_t i = 1; i < s->cells; ++i) {
        const scheduled e = entries[i];
        uint32_t j = i;
        while (j > 0 && schedule_before(&e, &entries[j - 1])) {
            entries[j] = entries[j - 1];
            --j;
        }
        entries[j] = e;
    }
    for (uint32_t i = 0; i < s->cells; ++i) order[i] = entries[i].cell;
    return ELPIS_HCA_OK;
}

/* An activation of this session, in range: its values as full-graph scores (0 outside the scope). */
static int validate(const elpis_hca_b2 *s, const elpis_hca_activation *a, meter *m, uint16_t *scores) {
    REQUIRE(a && a->cells == s->cells && a->reserved == 0 && !memcmp(a->owner, s->owner, 32), ELPIS_HCA_INVALID_STATE);
    REQUIRE(all_zero(a->values + 2 * s->cells, sizeof(a->values) - 2 * (size_t)s->cells), ELPIS_HCA_INVALID_STATE);
    memset(scores, 0, s->graph->cell_count * sizeof(uint16_t));
    for (uint32_t i = 0; i < s->cells; ++i) {
        CHARGE(m, 1);
        const uint32_t v = (uint32_t)a->values[2 * i] << 8 | a->values[2 * i + 1];
        REQUIRE(v <= 1000, ELPIS_HCA_INVALID_STATE);
        scores[s->cell_of[i]] = (uint16_t)v;
    }
    return ELPIS_HCA_OK;
}

static void seal(const elpis_hca_b2 *s, const uint16_t *scores, uint64_t step, elpis_hca_activation *a) {
    memset(a, 0, sizeof(*a));
    a->step = step;
    memcpy(a->owner, s->owner, 32);
    a->cells = s->cells;
    for (uint32_t i = 0; i < s->cells; ++i) {
        a->values[2 * i] = (uint8_t)(scores[s->cell_of[i]] >> 8);
        a->values[2 * i + 1] = (uint8_t)scores[s->cell_of[i]];
    }
}

/* The prepared view of a session (scoped cells, admitted factors, seeds), for the shared message and readout. */
static void session_prepared(const elpis_hca_b2 *s, prepared *p) {
    memset(p, 0, sizeof(*p));
    for (uint32_t i = 0; i < s->cells; ++i) p->scoped[s->cell_of[i]] = 1;
    p->scoped_count = s->cells;
    memcpy(p->factor_on, s->factor_on, s->graph->factor_count);
    p->factor_on_count = s->factor_on_count;
    memcpy(p->seed, s->seed, s->graph->cell_count);
}

/* b2.py `_start` up to the initial activation: the reservation first, then prepare. */
static int start_state(const elpis_hca_graph *g, const elpis_hca_task *t, meter *m, prepared *p, uint8_t owner[32]) {
    uint64_t count = 0, factor_count = 0;
    for (uint32_t c = 0; c < g->cell_count; ++c) {
        CHARGE(m, 1);
        count += g->cells[c].scope == t->scope;
    }
    for (uint32_t f = 0; f < g->factor_count; ++f) {
        CHARGE(m, 1);
        factor_count += g->factor_scope[f] == t->scope;
    }
    const uint64_t reservation =
        8192 + 1024 * count + 256 * factor_count + 256 * (uint64_t)t->predicate_count + 256 * (uint64_t)g->limits.degree;
    REQUIRE(reservation <= g->limits.state_bytes, ELPIS_HCA_STATE_LIMIT);
    TRY(prepare(g, t, m, p));
    CHARGE(m, owner_bytes(t));
    hash_owner(g, t, owner);
    CHARGE(m, p->scoped_count);                     /* state index */
    CHARGE(m, 2 * (uint64_t)p->scoped_count);       /* state bytes */
    CHARGE(m, p->scoped_count);                     /* initial values */
    CHARGE(m, 2 * (uint64_t)p->scoped_count);       /* seal */
    CHARGE(m, p->scoped_count + p->factor_on_count + 1);
    return ELPIS_HCA_OK;
}

static void session_init(elpis_hca_b2 *s, const elpis_hca_graph *g, const elpis_hca_task *t, const prepared *p,
                         const uint8_t owner[32]) {
    memset(s, 0, sizeof(*s));
    s->graph = g;
    s->task = *t;
    for (uint32_t c = 0; c < g->cell_count; ++c) {
        s->position_of[c] = p->scoped[c] ? (uint16_t)s->cells : ELPIS_HCA_UNSCOPED;
        if (p->scoped[c]) s->cell_of[s->cells++] = c;
    }
    memcpy(s->factor_on, p->factor_on, g->factor_count);
    s->factor_on_count = p->factor_on_count;
    memcpy(s->seed, p->seed, g->cell_count);
    memcpy(s->owner, owner, 32);
}

static int warm(const elpis_hca_b2 *s, const elpis_hca_activation *in, meter *m, elpis_hca_activation *out) {
    const elpis_hca_graph *g = s->graph;
    uint16_t scores[ELPIS_HCA_MAX_CELLS];
    TRY(validate(s, in, m, scores));
    REQUIRE(in->step < UINT64_MAX, ELPIS_HCA_STEP_LIMIT);
    uint32_t order[ELPIS_HCA_MAX_CELLS];
    TRY(schedule(s, in->step, m, order));
    CHARGE(m, 2 * (uint64_t)s->cells);
    const uint32_t fire = s->task.updates_per_tick < s->cells ? s->task.updates_per_tick : s->cells;
    for (uint32_t i = 0; i < fire; ++i) {
        CHARGE(m, 1);
        const uint32_t c = order[i];
        uint32_t value = s->seed[c] ? 1000 : 0;
        for (uint32_t k = 0; k < g->degree[c]; ++k) {
            CHARGE(m, 1);
            const uint32_t f = g->inc_factor[(size_t)c * g->limits.degree + k];
            if (s->factor_on[f] && g->inc_position[(size_t)c * g->limits.degree + k] == g->factors[f].arity - 1) {
                uint32_t v;
                TRY(message(g, f, scores, m, &v));
                if (v > value) value = v;
            }
        }
        CHARGE(m, 1);
        scores[c] = source_admits(g->cells[c].source_state) ? (uint16_t)value : 0;
    }
    CHARGE(m, 2 * (uint64_t)s->cells);
    CHARGE(m, s->cells + s->factor_on_count + 1);
    seal(s, scores, in->step + 1, out);
    return ELPIS_HCA_OK;
}

/* ---- public: prediction ---------------------------------------------------------------------------------------- */

int elpis_hca_predict(const elpis_hca_graph *graph, const elpis_hca_task *task, uint32_t method,
                      elpis_hca_result *out, uint16_t *scores, uint64_t *used) {
    if (used) *used = 0;
    if (!graph || !task || !out || method > ELPIS_HCA_B2 || !task_valid(task)) return ELPIS_HCA_INVALID;
    if (task->work_budget > graph->limits.work) return ELPIS_HCA_INVALID;
    meter m = {task->work_budget, 0};
    prepared p;
    elpis_hca_result r;
    int rc;
    if (method == ELPIS_HCA_B2) {
        /* b2.py `predict_b2`: the cold start, every requested tick and the readout under one budget, on a stack
         * session borrowing the task (no allocation). */
        elpis_hca_b2 session;
        elpis_hca_activation a, next;
        uint8_t owner[32];
        rc = start_state(graph, task, &m, &p, owner);
        if (rc == ELPIS_HCA_OK) {
            session_init(&session, graph, task, &p, owner);
            seal(&session, p.scores, 0, &a);
        }
        for (uint32_t i = 0; rc == ELPIS_HCA_OK && i < task->steps; ++i) {
            rc = warm(&session, &a, &m, &next);
            a = next;
        }
        if (rc == ELPIS_HCA_OK) {
            session_prepared(&session, &p);
            rc = validate(&session, &a, &m, p.scores);
        }
        if (rc == ELPIS_HCA_OK) rc = readout(graph, task, &p, p.scores, &m, &r);
    } else {
        rc = prepare(graph, task, &m, &p);
        if (rc == ELPIS_HCA_OK) rc = method == ELPIS_HCA_B0 ? b0(graph, task, &p, &m) : b1(graph, task, &p, &m);
        if (rc == ELPIS_HCA_OK) rc = readout(graph, task, &p, p.scores, &m, &r);
    }
    if (used) *used = m.used;
    if (rc != ELPIS_HCA_OK) return rc;
    r.work = m.used;
    *out = r;
    if (scores)
        for (uint32_t c = 0; c < graph->cell_count; ++c) scores[c] = p.scoped[c] ? p.scores[c] : ELPIS_HCA_UNSCOPED;
    return ELPIS_HCA_OK;
}

/* ---- public: B2 sessions -------------------------------------------------------------------------------------- */

int elpis_hca_b2_start(const elpis_hca_graph *graph, const elpis_hca_task *task, uint64_t budget,
                       elpis_hca_b2 **session, elpis_hca_activation *out, uint64_t *used) {
    if (used) *used = 0;
    if (session) *session = NULL;
    if (!graph || !task || !session || !out || !task_valid(task) || task->work_budget > graph->limits.work)
        return ELPIS_HCA_INVALID;
    meter m;
    TRY(budget_of(task, budget, &m));
    prepared p;
    uint8_t owner[32];
    const int rc = start_state(graph, task, &m, &p, owner);
    if (used) *used = m.used;
    if (rc != ELPIS_HCA_OK) return rc;
    /* One block: the session and its copy of the task's arrays. */
    const size_t seeds = (size_t)task->seed_count * 32,
                 predicates = (size_t)task->predicate_count * sizeof(elpis_hca_predicate),
                 kinds = (size_t)task->output_kind_count * sizeof(uint32_t);
    const size_t at_seeds = align8(sizeof(elpis_hca_b2)), at_predicates = at_seeds + align8(seeds),
                 at_kinds = at_predicates + align8(predicates);
    uint8_t *block = calloc(1, at_kinds + align8(kinds));
    if (!block) return ELPIS_HCA_STATE_LIMIT;
    elpis_hca_b2 *s = (elpis_hca_b2 *)block;
    session_init(s, graph, task, &p, owner);
    if (seeds) memcpy(block + at_seeds, task->seeds, seeds);
    if (predicates) memcpy(block + at_predicates, task->predicates, predicates);
    memcpy(block + at_kinds, task->output_kinds, kinds);
    s->task.seeds = (const uint8_t(*)[32])(block + at_seeds);
    s->task.predicates = (const elpis_hca_predicate *)(block + at_predicates);
    s->task.output_kinds = (const uint32_t *)(block + at_kinds);
    seal(s, p.scores, 0, out);
    *session = s;
    return ELPIS_HCA_OK;
}

int elpis_hca_b2_warm(const elpis_hca_b2 *session, const elpis_hca_activation *in, uint64_t budget,
                      elpis_hca_activation *out, uint64_t *used) {
    if (used) *used = 0;
    if (!session || !in || !out) return ELPIS_HCA_INVALID;
    meter m;
    TRY(budget_of(&session->task, budget, &m));
    elpis_hca_activation next;
    const int rc = warm(session, in, &m, &next);
    if (used) *used = m.used;
    if (rc == ELPIS_HCA_OK) *out = next;
    return rc;
}

static int damage(const elpis_hca_b2 *s, const elpis_hca_activation *in, const elpis_hca_damage *changes, size_t n,
                  meter *m, elpis_hca_activation *out) {
    uint16_t scores[ELPIS_HCA_MAX_CELLS];
    TRY(validate(s, in, m, scores));
    REQUIRE(n <= s->cells && (n == 0 || changes), ELPIS_HCA_INVALID_DAMAGE);
    uint8_t seen[ELPIS_HCA_MAX_CELLS] = {0};
    for (size_t i = 0; i < n; ++i) {
        CHARGE(m, 1);
        const uint32_t c = changes[i].cell;
        REQUIRE(c < s->graph->cell_count && s->position_of[c] != ELPIS_HCA_UNSCOPED && !seen[c] &&
                    changes[i].value <= 1000,
                ELPIS_HCA_INVALID_DAMAGE);
        seen[c] = 1;
    }
    CHARGE(m, 2 * (uint64_t)s->cells);
    for (size_t i = 0; i < n; ++i) {
        CHARGE(m, 1);
        scores[changes[i].cell] = (uint16_t)changes[i].value;
    }
    CHARGE(m, 2 * (uint64_t)s->cells);
    CHARGE(m, s->cells + s->factor_on_count + 1);
    seal(s, scores, in->step, out);
    return ELPIS_HCA_OK;
}

int elpis_hca_b2_damage(const elpis_hca_b2 *session, const elpis_hca_activation *in, const elpis_hca_damage *changes,
                        size_t change_count, uint64_t budget, elpis_hca_activation *out, uint64_t *used) {
    if (used) *used = 0;
    if (!session || !in || !out) return ELPIS_HCA_INVALID;
    meter m;
    TRY(budget_of(&session->task, budget, &m));
    elpis_hca_activation next;
    const int rc = damage(session, in, changes, change_count, &m, &next);
    if (used) *used = m.used;
    if (rc == ELPIS_HCA_OK) *out = next;
    return rc;
}

int elpis_hca_b2_reset(const elpis_hca_b2 *session, uint64_t budget, elpis_hca_activation *out, uint64_t *used) {
    if (used) *used = 0;
    if (!session || !out) return ELPIS_HCA_INVALID;
    meter m;
    TRY(budget_of(&session->task, budget, &m));
    prepared p;
    uint8_t owner[32];
    const int rc = start_state(session->graph, &session->task, &m, &p, owner);
    if (used) *used = m.used;
    if (rc != ELPIS_HCA_OK) return rc;
    seal(session, p.scores, 0, out);
    return ELPIS_HCA_OK;
}

int elpis_hca_b2_observe(const elpis_hca_b2 *session, const elpis_hca_activation *in, uint64_t budget,
                         elpis_hca_result *out, uint64_t *used) {
    if (used) *used = 0;
    if (!session || !in || !out) return ELPIS_HCA_INVALID;
    meter m;
    TRY(budget_of(&session->task, budget, &m));
    prepared p;
    session_prepared(session, &p);
    elpis_hca_result r;
    int rc = validate(session, in, &m, p.scores);
    if (rc == ELPIS_HCA_OK) rc = readout(session->graph, &session->task, &p, p.scores, &m, &r);
    if (used) *used = m.used;
    if (rc != ELPIS_HCA_OK) return rc;
    r.work = m.used;
    *out = r;
    return ELPIS_HCA_OK;
}

void elpis_hca_b2_destroy(elpis_hca_b2 *session) { free(session); }
