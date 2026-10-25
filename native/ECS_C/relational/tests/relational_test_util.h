/* Shared helpers of the ECS_C relational view tests: a synthetic ECS_C authority table (real elpis_ecs_c_entity
 * rows; the end-to-end test uses the real authority instead), observations, and the exact-rebuild oracle: a fresh
 * ECS_C memory (the real elpis_ecs_mem) that OBSERVEs exactly the logged observations whose binding is LIVE under
 * the current table, in admission order. The view must answer exactly as that memory does. */
#ifndef ELPIS_RELATIONAL_TEST_UTIL_H
#define ELPIS_RELATIONAL_TEST_UTIL_H

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "elpis/ecs_c_relational.h"
#include "elpis/sha256.h"

#define CHECK(expr)                                                                                             \
    do {                                                                                                        \
        if (!(expr)) {                                                                                          \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #expr);                            \
            exit(1);                                                                                            \
        }                                                                                                       \
    } while (0)
#define RC(expr, code)                                                                                          \
    do {                                                                                                        \
        const int rc_ = (expr);                                                                                 \
        if (rc_ != (code)) {                                                                                    \
            fprintf(stderr, "%s:%d: %s = %d (%s), expected %d (%s)\n", __FILE__, __LINE__, #expr, rc_,         \
                    elpis_ecs_c_rel_code_name(rc_) ? elpis_ecs_c_rel_code_name(rc_) : "?", (int)(code),        \
                    elpis_ecs_c_rel_code_name(code) ? elpis_ecs_c_rel_code_name(code) : "?");                   \
            exit(1);                                                                                            \
        }                                                                                                       \
    } while (0)
#define OK(expr) RC(expr, ELPIS_ECS_C_REL_OK)

static uint64_t rng_state;
static inline uint64_t rnd(void) {
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
static inline uint32_t below(uint32_t n) { return (uint32_t)(rnd() % n); }

static inline void fill(uint8_t out[32], uint32_t tag, uint64_t value) {
    uint8_t buf[12];
    for (int i = 0; i < 4; ++i) buf[i] = (uint8_t)(tag >> (8 * i));
    for (int i = 0; i < 8; ++i) buf[4 + i] = (uint8_t)(value >> (8 * i));
    elpis_sha256(buf, sizeof(buf), out);
}

/* ---- A synthetic authority: the rows and coordinate an ECS_C authority would report. ---------------- */

#define W_MAX 64

typedef struct {
    uint8_t lineage[32];
    uint64_t generation, next_founding, reclaimed;
    uint32_t count;
    elpis_ecs_c_entity rows[W_MAX];
} world;

static inline void w_root(const world *w, uint8_t out[32]) { fill(out, 0x726f6f74u ^ (uint32_t)w->lineage[0], w->generation); }
static inline void w_bump(world *w) { w->generation++; }

static inline uint32_t w_found(world *w, uint8_t kind, uint8_t lifecycle, uint32_t authority, uint64_t version,
                        uint32_t scope) {
    CHECK(w->count < W_MAX);
    elpis_ecs_c_entity *e = &w->rows[w->count];
    memset(e, 0, sizeof(*e));
    e->founding_index = w->next_founding++;
    fill(e->id, 0x69640000u ^ w->lineage[1], e->founding_index);
    fill(e->label, 0x6c61626cu, e->founding_index);
    e->kind = kind;
    e->lifecycle = lifecycle;
    e->founded_generation = e->last_generation = w->generation;
    if (kind == ELPIS_ECS_C_SOURCE) {
        e->per_turn_limit = 16;
        e->source_version = version;
        fill(e->scope, 0x73636f70u, scope);
        e->source_authority = authority;
    }
    w_bump(w);
    return w->count++;
}
static inline void w_init(world *w, uint32_t lineage_tag) {
    memset(w, 0, sizeof(*w));
    fill(w->lineage, 0x6c696e65u, lineage_tag);
    w->generation = 1;
    (void)w_found(w, ELPIS_ECS_C_OPERATOR, ELPIS_ECS_C_ACTIVE, 0, 0, 0);
    (void)w_found(w, ELPIS_ECS_C_RUNTIME, ELPIS_ECS_C_ACTIVE, 0, 0, 0);
}
static inline void w_version(world *w, uint32_t i, uint64_t version, uint32_t scope) {
    w->rows[i].source_version = version;
    fill(w->rows[i].scope, 0x73636f70u, scope);
    w_bump(w);
}
static inline void w_lifecycle(world *w, uint32_t i, uint8_t lifecycle) {
    w->rows[i].lifecycle = lifecycle;
    w_bump(w);
}
static inline void w_reclaim(world *w, uint32_t i) {
    memmove(&w->rows[i], &w->rows[i + 1], sizeof(w->rows[0]) * (w->count - i - 1));
    w->count--;
    w->reclaimed++;
    w_bump(w);
}
static inline uint32_t w_find(const world *w, const uint8_t id[32]) {
    for (uint32_t i = 0; i < w->count; ++i)
        if (!memcmp(w->rows[i].id, id, 32)) return i;
    return UINT32_MAX;
}
static inline elpis_ecs_c_rel_authority w_table(const world *w, uint8_t root[32]) {
    elpis_ecs_c_rel_authority t;
    memset(&t, 0, sizeof(t));
    t.abi_version = ELPIS_ECS_C_REL_ABI_VERSION;
    t.entity_count = w->count;
    t.generation = w->generation;
    t.next_founding_index = w->next_founding;
    t.reclaimed_count = w->reclaimed;
    memcpy(t.lineage, w->lineage, 32);
    w_root(w, root);
    memcpy(t.state_root, root, 32);
    t.entities = w->rows;
    return t;
}
static inline elpis_ecs_c_rel_coordinate w_coord(const world *w) {
    elpis_ecs_c_rel_coordinate c;
    c.generation = w->generation;
    w_root(w, c.state_root);
    return c;
}
static inline int w_sync(elpis_ecs_c_rel *v, const world *w, elpis_ecs_c_rel_sync_result *res) {
    uint8_t root[32];
    const elpis_ecs_c_rel_authority t = w_table(w, root);
    return elpis_ecs_c_rel_sync(v, &t, res);
}

/* ---- Observations ----------------------------------------------------------------------------------- */

static inline elpis_ecs_mem_observation obs(uint32_t type, uint32_t polarity, uint64_t tick, uint32_t authority,
                                     uint32_t arity, const uint32_t *roles, const uint32_t *entities) {
    elpis_ecs_mem_observation o;
    memset(&o, 0, sizeof(o));
    o.abi_version = ELPIS_ECS_MEM_ABI_V1;
    o.relation_type = type;
    o.polarity = polarity;
    o.participant_count = arity;
    o.tick = tick;
    o.source_authority = authority;
    fill(o.provenance.b, 0x70726f76u, tick * 131u + authority);
    fill(o.structure_ref.b, 0x73747275u, tick * 17u + type);
    for (uint32_t i = 0; i < arity; ++i) {
        o.participants[i].role = roles[i];
        o.participants[i].ordinal = 0;
        fill(o.participants[i].entity.b, 0x656e7469u, entities[i]);
    }
    return o;
}
static inline elpis_ecs_mem_observation pair(uint32_t type, uint32_t polarity, uint64_t tick, uint32_t authority, uint32_t a,
                                      uint32_t b) {
    const uint32_t roles[2] = {1, 2}, ents[2] = {a, b};
    return obs(type, polarity, tick, authority, 2, roles, ents);
}
/* LOOKUP: target role `target` given the other participants named in `cue_roles`/`cue_entities`. */
static inline elpis_ecs_mem_request lookup(uint32_t type, uint32_t target, uint32_t cues, const uint32_t *cue_roles,
                                    const uint32_t *cue_entities) {
    elpis_ecs_mem_request q;
    memset(&q, 0, sizeof(q));
    q.abi_version = ELPIS_ECS_MEM_ABI_V1;
    q.channel = ELPIS_ECS_MEM_LOOKUP;
    q.relation_type = type;
    q.target_role = target;
    q.cue_count = cues;
    for (uint32_t i = 0; i < cues; ++i) {
        q.cue[i].role = cue_roles[i];
        fill(q.cue[i].entity.b, 0x656e7469u, cue_entities[i]);
    }
    return q;
}
static inline elpis_ecs_mem_request ask(uint32_t type, uint32_t a) {
    const uint32_t r = 1;
    return lookup(type, 2, 1, &r, &a);
}

static inline int admit_as(elpis_ecs_c_rel *v, const world *w, uint32_t row, const elpis_ecs_mem_observation *o,
                    elpis_ecs_mem_digest *key) {
    const elpis_ecs_c_rel_coordinate c = w_coord(w);
    const elpis_ecs_c_entity *e = &w->rows[row];
    return elpis_ecs_c_rel_admit(v, &c, e->id, e->source_version, e->scope, o, key, NULL);
}

/* ---- The exact-rebuild oracle ------------------------------------------------------------------------ */

typedef struct {
    elpis_ecs_mem_observation o;
    uint8_t source[32], scope[32];
    uint64_t version;
} logged;

#define LOG_MAX 20000
static logged evidence_log[LOG_MAX];
static uint32_t log_count;

static inline void log_add(const elpis_ecs_mem_observation *o, const elpis_ecs_c_entity *e) {
    CHECK(log_count < LOG_MAX);
    logged *l = &evidence_log[log_count++];
    l->o = *o;
    memcpy(l->source, e->id, 32);
    memcpy(l->scope, e->scope, 32);
    l->version = e->source_version;
}
/* LIVE under the world's current rows: the authority's admission law. */
static inline int log_live(const world *w, const logged *l) {
    const uint32_t i = w_find(w, l->source);
    if (i == UINT32_MAX) return 0;
    const elpis_ecs_c_entity *e = &w->rows[i];
    return e->kind == ELPIS_ECS_C_SOURCE && e->lifecycle == ELPIS_ECS_C_ACTIVE && e->source_version == l->version &&
           !memcmp(e->scope, l->scope, 32);
}
static inline elpis_ecs_mem *oracle_build(const world *w, const uint8_t *scope) {
    elpis_ecs_mem_config c;
    elpis_ecs_mem_config_default(&c);
    c.dim = 64;
    c.banks = 1;
    c.slot_capacity = 2048;
    c.codebook_capacity = 1024;
    c.slot_policy = ELPIS_ECS_MEM_REFUSE;
    c.codebook_policy = ELPIS_ECS_MEM_EVICT_LRU; /* the codebook plays no part in LOOKUP */
    c.byte_budget = (uint64_t)1 << 28;
    elpis_ecs_mem *m = NULL;
    CHECK(elpis_ecs_mem_create(&c, &m) == ELPIS_ECS_MEM_OK);
    for (uint32_t i = 0; i < log_count; ++i) {
        const logged *l = &evidence_log[i];
        if (!log_live(w, l) || (scope && memcmp(l->scope, scope, 32))) continue;
        CHECK(elpis_ecs_mem_observe(m, 0, &l->o, NULL, NULL) == ELPIS_ECS_MEM_OK);
    }
    return m;
}

static inline int answers_equal(const elpis_ecs_mem_answer *a, const elpis_ecs_mem_answer *b) {
    return a->found == b->found && a->matches == b->matches && a->conflicts == b->conflicts &&
           a->contradicted == b->contradicted && a->uncertain == b->uncertain &&
           !memcmp(a->answer.b, b->answer.b, 32) && a->score == b->score && a->runner_up == b->runner_up &&
           a->signal == b->signal && a->answer_tick == b->answer_tick && !memcmp(a->answer_key.b, b->answer_key.b, 32);
}
static inline int slots_equal(const elpis_ecs_mem_slot *a, const elpis_ecs_mem_slot *b) {
    if (memcmp(a->key.b, b->key.b, 32) || memcmp(a->payload.b, b->payload.b, 32) ||
        memcmp(a->first_provenance.b, b->first_provenance.b, 32) ||
        memcmp(a->last_provenance.b, b->last_provenance.b, 32) || memcmp(a->structure_ref.b, b->structure_ref.b, 32))
        return 0;
    if (a->relation_type != b->relation_type || a->participant_count != b->participant_count ||
        a->assert_count != b->assert_count || a->negate_count != b->negate_count ||
        a->uncertain_count != b->uncertain_count || a->last_polarity != b->last_polarity ||
        a->last_authority != b->last_authority || a->learn_count != b->learn_count)
        return 0;
    if (a->first_tick != b->first_tick || a->last_tick != b->last_tick || a->last_assert_tick != b->last_assert_tick ||
        a->last_negate_tick != b->last_negate_tick)
        return 0;
    return !memcmp(a->participants, b->participants, sizeof(a->participants));
}

/* ---- Atomicity: everything observable except refusal telemetry --------------------------------------- */

typedef struct {
    uint8_t identity[32];
    elpis_ecs_c_rel_status s;
} snapshot;

static inline snapshot snap(elpis_ecs_c_rel *v) {
    snapshot p;
    OK(elpis_ecs_c_rel_identity(v, p.identity));
    OK(elpis_ecs_c_rel_read_status(v, &p.s));
    return p;
}
/* Unchanged: identity and every state counter (refusal telemetry aside). */
static inline void same_state(elpis_ecs_c_rel *v, const snapshot *p) {
    const snapshot q = snap(v);
    CHECK(!memcmp(p->identity, q.identity, 32));
    CHECK(p->s.synced == q.s.synced && p->s.entities == q.s.entities && p->s.relations == q.s.relations);
    CHECK(p->s.live_relations == q.s.live_relations && p->s.contributions == q.s.contributions);
    CHECK(p->s.live_contributions == q.s.live_contributions && p->s.generation == q.s.generation);
    CHECK(p->s.clock == q.s.clock && p->s.sequence == q.s.sequence && p->s.revision == q.s.revision);
    CHECK(!memcmp(p->s.state_root, q.s.state_root, 32) && !memcmp(p->s.lineage, q.s.lineage, 32));
}
/* Unchanged, and still consistent. */
static inline void same(elpis_ecs_c_rel *v, const snapshot *p) {
    same_state(v, p);
    uint64_t bad = 1;
    OK(elpis_ecs_c_rel_verify(v, &bad));
    CHECK(bad == 0);
}

static inline elpis_ecs_c_rel *make_view(uint32_t entities, uint32_t relations, uint32_t contributions) {
    elpis_ecs_c_rel_config c;
    elpis_ecs_c_rel_config_default(&c);
    c.entity_capacity = entities;
    c.relation_capacity = relations;
    c.contribution_capacity = contributions;
    c.byte_budget = (uint64_t)1 << 30;
    elpis_ecs_c_rel *v = NULL;
    OK(elpis_ecs_c_rel_create(&c, &v));
    return v;
}

#endif
