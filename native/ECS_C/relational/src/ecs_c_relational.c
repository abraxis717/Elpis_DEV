/* ECS_C relational view (ECSC_DYNAMIC_RELATIONAL_DEV_R0): source-revision-aware bounded maintenance of admitted typed
 * evidence. Contract: elpis/ecs_c_relational.h and docs/ECS_C_RELATIONAL.md. DEV, not composed. Integer arithmetic
 * only; one allocation per handle; no I/O, no thread, no clock. */
#include "elpis/ecs_c_relational.h"
#include "elpis/sha256.h"

#include <stdlib.h>
#include <string.h>

#define NONE UINT32_MAX
#define MAX_PER_TURN 256u

/* ------------------------------------------------------------------------------------------------------ */
/* Records                                                                                                */
/* ------------------------------------------------------------------------------------------------------ */

/* One mirrored authority row. Key first (the index reads it at offset 0). */
typedef struct {
    uint8_t id[32];
    uint8_t scope[32];
    uint8_t label[32];
    uint64_t version;
    uint64_t founding_index;
    uint64_t stamp; /* scratch: the sync epoch that saw it */
    uint32_t source_authority;
    uint32_t per_turn_limit;
    uint32_t head, tail, count; /* its contributions, in admission order */
    uint8_t kind, lifecycle, used, reserved;
} ent_row;

/* The aggregate of a relation's LIVE contributions: the ELPISEM1 slot an exact replay would hold. */
typedef struct {
    uint32_t live_count;
    uint32_t assert_count, negate_count, uncertain_count;
    uint32_t last_polarity, last_authority;
    uint64_t first_seq, last_seq, first_tick, last_tick, last_assert_tick, last_negate_tick;
    elpis_ecs_mem_digest first_provenance, last_provenance, structure_ref;
} agg;

typedef struct {
    elpis_ecs_mem_digest key; /* first: the index reads it at offset 0 */
    elpis_ecs_mem_digest payload;
    uint32_t relation_type, participant_count;
    elpis_ecs_mem_participant participants[ELPIS_ECS_MEM_MAX_PARTICIPANTS];
    uint32_t head, tail, count;                         /* its contributions, in admission order */
    uint32_t total_assert, total_negate, total_uncertain; /* over every retained contribution */
    uint32_t prev, next;                                /* the scan list of used relations; next: the free list */
    uint32_t used, reserved;
    uint64_t stamp; /* scratch: the sync epoch that marked it dirty */
    agg live;
    uint64_t check;
} rel_row;

/* The evidence of one relation under one binding (source, version, scope). */
typedef struct {
    uint32_t relation, source;   /* record indices */
    uint32_t rprev, rnext;       /* in the relation's list; rnext: the free list */
    uint32_t sprev, snext;       /* in the source's list */
    uint64_t version;
    uint8_t scope[32];
    uint32_t observed_authority; /* the observations' source_authority (the source's, checked at admission) */
    uint32_t assert_count, negate_count, uncertain_count;
    uint32_t last_polarity;
    uint32_t live;
    uint64_t first_seq, last_seq, first_tick, last_tick, last_assert_tick, last_negate_tick;
    elpis_ecs_mem_digest first_provenance, last_provenance, structure_ref;
    uint64_t check;
} contrib;

struct elpis_ecs_c_rel {
    elpis_ecs_c_rel_config cfg;
    /* The synced coordinate. */
    uint32_t synced;
    uint64_t generation, next_founding_index, reclaimed_count;
    uint8_t lineage[32], state_root[32];
    uint64_t clock, seq, revision;
    /* Entity rows. */
    ent_row *ents;
    uint32_t *ent_index, ent_mask, ent_count, ent_hwm, *ent_free, ent_free_count;
    /* Relations. */
    rel_row *rels;
    uint32_t *rel_index, rel_mask, rel_count, rel_hwm, rel_free, rel_head, rel_tail, live_relations;
    /* Contributions. */
    contrib *cons;
    uint32_t con_count, con_hwm, con_free, live_contributions;
    /* Scratch (non-authoritative): ordering and the dirty list; the table-row map of a sync. */
    uint32_t *scratch, *map;
    uint64_t epoch;
    uint64_t arena_bytes;
    elpis_ecs_c_rel_status tel; /* telemetry fields only */
};

/* ------------------------------------------------------------------------------------------------------ */
/* Small helpers                                                                                          */
/* ------------------------------------------------------------------------------------------------------ */

static uint64_t rd64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}
static void le32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i));
}
static void le64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i));
}
static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}
static int is_zero(const void *p, size_t n) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; ++i)
        if (b[i]) return 0;
    return 1;
}
static void work_add(elpis_ecs_c_rel_work *into, const elpis_ecs_c_rel_work *w) {
    if (!into) return;
    into->rows += w->rows;
    into->index_probes += w->index_probes;
    into->contributions += w->contributions;
    into->relations += w->relations;
    into->recomputes += w->recomputes;
    into->sorted += w->sorted;
}
static int refuse(elpis_ecs_c_rel *v, int code) {
    v->tel.refusals++;
    v->tel.refusals_by_code[code & 15]++;
    return code;
}

const char *elpis_ecs_c_rel_code_name(int code) {
    switch (code) {
    case ELPIS_ECS_C_REL_OK: return "ECS_C_REL_OK";
    case ELPIS_ECS_C_REL_INVALID: return "ECS_C_REL_INVALID";
    case ELPIS_ECS_C_REL_CAPACITY: return "ECS_C_REL_CAPACITY";
    case ELPIS_ECS_C_REL_STALE_VIEW: return "ECS_C_REL_STALE_VIEW";
    case ELPIS_ECS_C_REL_UNKNOWN_SOURCE: return "ECS_C_REL_UNKNOWN_SOURCE";
    case ELPIS_ECS_C_REL_NOT_SOURCE: return "ECS_C_REL_NOT_SOURCE";
    case ELPIS_ECS_C_REL_LIFECYCLE: return "ECS_C_REL_LIFECYCLE";
    case ELPIS_ECS_C_REL_SOURCE_STALE: return "ECS_C_REL_SOURCE_STALE";
    case ELPIS_ECS_C_REL_SCOPE_CONFLICT: return "ECS_C_REL_SCOPE_CONFLICT";
    case ELPIS_ECS_C_REL_BINDING_MISMATCH: return "ECS_C_REL_BINDING_MISMATCH";
    case ELPIS_ECS_C_REL_CLOCK: return "ECS_C_REL_CLOCK";
    case ELPIS_ECS_C_REL_LINEAGE: return "ECS_C_REL_LINEAGE";
    case ELPIS_ECS_C_REL_REGRESSION: return "ECS_C_REL_REGRESSION";
    case ELPIS_ECS_C_REL_CORRUPT: return "ECS_C_REL_CORRUPT";
    case ELPIS_ECS_C_REL_NOT_FOUND: return "ECS_C_REL_NOT_FOUND";
    case ELPIS_ECS_C_REL_NOMEM: return "ECS_C_REL_NOMEM";
    default: return NULL;
    }
}

uint32_t elpis_ecs_c_rel_abi_version(void) { return ELPIS_ECS_C_REL_ABI_VERSION; }

/* ------------------------------------------------------------------------------------------------------ */
/* Exact hash index over 32-byte keys at offset 0 of fixed-stride records (linear probing, backward shift)  */
/* ------------------------------------------------------------------------------------------------------ */

typedef struct {
    const uint8_t *base;
    size_t stride;
} keyview;

static const uint8_t *key_at(keyview kv, uint32_t i) { return kv.base + (size_t)i * kv.stride; }
static uint32_t home_of(const uint8_t *key, uint32_t mask) {
    return (uint32_t)(mix64(rd64(key) ^ mix64(rd64(key + 8))) & mask);
}
static uint32_t index_find(const uint32_t *index, uint32_t mask, keyview kv, const uint8_t *key, uint64_t *probes) {
    uint32_t pos = home_of(key, mask);
    for (;;) {
        ++*probes;
        const uint32_t e = index[pos];
        if (!e) return NONE;
        if (!memcmp(key_at(kv, e - 1u), key, 32)) return e - 1u;
        pos = (pos + 1u) & mask;
    }
}
static void index_insert(uint32_t *index, uint32_t mask, keyview kv, uint32_t i, uint64_t *probes) {
    uint32_t pos = home_of(key_at(kv, i), mask);
    ++*probes;
    while (index[pos]) {
        pos = (pos + 1u) & mask;
        ++*probes;
    }
    index[pos] = i + 1u;
}
static void index_remove(uint32_t *index, uint32_t mask, keyview kv, uint32_t i, uint64_t *probes) {
    uint32_t pos = home_of(key_at(kv, i), mask);
    while (index[pos] != i + 1u) {
        pos = (pos + 1u) & mask;
        ++*probes;
    }
    uint32_t hole = pos, j = pos;
    for (;;) {
        j = (j + 1u) & mask;
        ++*probes;
        if (!index[j]) break;
        const uint32_t h = home_of(key_at(kv, index[j] - 1u), mask);
        const int stays = hole <= j ? (hole < h && h <= j) : (hole < h || h <= j);
        if (stays) continue;
        index[hole] = index[j];
        hole = j;
    }
    index[hole] = 0;
}
static keyview ent_keys(const elpis_ecs_c_rel *v) {
    keyview kv = {(const uint8_t *)v->ents, sizeof(ent_row)};
    return kv;
}
static keyview rel_keys(const elpis_ecs_c_rel *v) {
    keyview kv = {(const uint8_t *)v->rels, sizeof(rel_row)};
    return kv;
}
static uint32_t find_ent(const elpis_ecs_c_rel *v, const uint8_t id[32], uint64_t *probes) {
    return index_find(v->ent_index, v->ent_mask, ent_keys(v), id, probes);
}
static uint32_t find_rel(const elpis_ecs_c_rel *v, const uint8_t key[32], uint64_t *probes) {
    return index_find(v->rel_index, v->rel_mask, rel_keys(v), key, probes);
}

/* ------------------------------------------------------------------------------------------------------ */
/* Ordering (in-place heapsort of record indices; no allocation)                                          */
/* ------------------------------------------------------------------------------------------------------ */

typedef int (*less_fn)(const void *ctx, uint32_t a, uint32_t b);

static void sift(uint32_t *a, uint32_t start, uint32_t end, less_fn less, const void *ctx) {
    uint32_t root = start;
    while (2u * root + 1u <= end) {
        uint32_t child = 2u * root + 1u, swap = root;
        if (less(ctx, a[swap], a[child])) swap = child;
        if (child + 1u <= end && less(ctx, a[swap], a[child + 1u])) swap = child + 1u;
        if (swap == root) return;
        const uint32_t t = a[root];
        a[root] = a[swap];
        a[swap] = t;
        root = swap;
    }
}
static void heapsort_u32(uint32_t *a, uint32_t n, less_fn less, const void *ctx, uint64_t *sorted) {
    *sorted += n;
    if (n < 2) return;
    for (uint32_t s = (n - 2u) / 2u + 1u; s-- > 0;) sift(a, s, n - 1u, less, ctx);
    for (uint32_t end = n - 1u; end > 0; --end) {
        const uint32_t t = a[0];
        a[0] = a[end];
        a[end] = t;
        sift(a, 0, end - 1u, less, ctx);
    }
}
static int table_id_less(const void *ctx, uint32_t a, uint32_t b) {
    const elpis_ecs_c_entity *e = ctx;
    return memcmp(e[a].id, e[b].id, 32) < 0;
}
static int table_authority_less(const void *ctx, uint32_t a, uint32_t b) {
    const elpis_ecs_c_entity *e = ctx;
    return e[a].source_authority < e[b].source_authority;
}
static int ent_id_less(const void *ctx, uint32_t a, uint32_t b) {
    const ent_row *e = ctx;
    return memcmp(e[a].id, e[b].id, 32) < 0;
}
static int rel_key_less(const void *ctx, uint32_t a, uint32_t b) {
    const rel_row *r = ctx;
    return memcmp(r[a].key.b, r[b].key.b, 32) < 0;
}

/* ------------------------------------------------------------------------------------------------------ */
/* Integrity checks (accidental damage of retained records)                                               */
/* ------------------------------------------------------------------------------------------------------ */

static uint64_t h_u64(uint64_t h, uint64_t x) { return mix64(h ^ x) + 0x9e3779b97f4a7c15ull; }
static uint64_t h_bytes(uint64_t h, const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i += 8) h = h_u64(h, rd64(p + i));
    return h;
}
static uint64_t agg_hash(uint64_t h, const agg *a) {
    h = h_u64(h, a->live_count);
    h = h_u64(h, ((uint64_t)a->assert_count << 32) | a->negate_count);
    h = h_u64(h, ((uint64_t)a->uncertain_count << 32) | a->last_polarity);
    h = h_u64(h, a->last_authority);
    h = h_u64(h, a->first_seq);
    h = h_u64(h, a->last_seq);
    h = h_u64(h, a->first_tick);
    h = h_u64(h, a->last_tick);
    h = h_u64(h, a->last_assert_tick);
    h = h_u64(h, a->last_negate_tick);
    h = h_bytes(h, a->first_provenance.b, 32);
    h = h_bytes(h, a->last_provenance.b, 32);
    return h_bytes(h, a->structure_ref.b, 32);
}
static uint64_t rel_check(uint32_t i, const rel_row *r) {
    uint64_t h = h_u64(0x52454c0000000000ull, i);
    h = h_bytes(h, r->key.b, 32);
    h = h_bytes(h, r->payload.b, 32);
    h = h_u64(h, ((uint64_t)r->relation_type << 32) | r->participant_count);
    for (uint32_t p = 0; p < r->participant_count; ++p) {
        h = h_u64(h, ((uint64_t)r->participants[p].role << 32) | r->participants[p].ordinal);
        h = h_bytes(h, r->participants[p].entity.b, 32);
    }
    return agg_hash(h, &r->live);
}
static uint64_t con_check(uint32_t i, const contrib *c) {
    uint64_t h = h_u64(0x434f4e0000000000ull, i);
    h = h_u64(h, ((uint64_t)c->relation << 32) | c->source);
    h = h_u64(h, c->version);
    h = h_bytes(h, c->scope, 32);
    h = h_u64(h, ((uint64_t)c->observed_authority << 32) | c->live);
    h = h_u64(h, ((uint64_t)c->assert_count << 32) | c->negate_count);
    h = h_u64(h, ((uint64_t)c->uncertain_count << 32) | c->last_polarity);
    h = h_u64(h, c->first_seq);
    h = h_u64(h, c->last_seq);
    h = h_u64(h, c->first_tick);
    h = h_u64(h, c->last_tick);
    h = h_u64(h, c->last_assert_tick);
    h = h_u64(h, c->last_negate_tick);
    h = h_bytes(h, c->first_provenance.b, 32);
    h = h_bytes(h, c->last_provenance.b, 32);
    return h_bytes(h, c->structure_ref.b, 32);
}
static void seal_rel(elpis_ecs_c_rel *v, uint32_t r) { v->rels[r].check = rel_check(r, &v->rels[r]); }
static void seal_con(elpis_ecs_c_rel *v, uint32_t c) { v->cons[c].check = con_check(c, &v->cons[c]); }
static int rel_sound(const elpis_ecs_c_rel *v, uint32_t r) { return v->rels[r].check == rel_check(r, &v->rels[r]); }
static int con_sound(const elpis_ecs_c_rel *v, uint32_t c) { return v->cons[c].check == con_check(c, &v->cons[c]); }

/* ------------------------------------------------------------------------------------------------------ */
/* Configuration and layout                                                                               */
/* ------------------------------------------------------------------------------------------------------ */

void elpis_ecs_c_rel_config_default(elpis_ecs_c_rel_config *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->abi_version = ELPIS_ECS_C_REL_ABI_VERSION;
    out->entity_capacity = 64;
    out->relation_capacity = 1024;
    out->contribution_capacity = 4096;
    out->byte_budget = (uint64_t)64 << 20;
}

static int config_ok(const elpis_ecs_c_rel_config *c) {
    return c && c->abi_version == ELPIS_ECS_C_REL_ABI_VERSION && c->entity_capacity >= 2 &&
           c->entity_capacity <= ELPIS_ECS_C_REL_MAX_ENTITIES && c->relation_capacity >= 1 &&
           c->relation_capacity <= ELPIS_ECS_C_REL_MAX_RELATIONS && c->contribution_capacity >= 1 &&
           c->contribution_capacity <= ELPIS_ECS_C_REL_MAX_CONTRIBUTIONS;
}
static uint32_t index_size(uint32_t cap) {
    uint32_t n = 1;
    while (n < 2u * cap) n <<= 1;
    return n;
}

typedef struct {
    uint64_t ents, ent_index, ent_free, rels, rel_index, cons, scratch, map, total;
} layout;

static void place(uint64_t *total, uint64_t bytes, uint64_t *at) {
    *total = (*total + 63u) & ~(uint64_t)63u;
    *at = *total;
    *total += bytes;
}
static void compute_layout(const elpis_ecs_c_rel_config *c, layout *l) {
    memset(l, 0, sizeof(*l));
    uint64_t t = (sizeof(struct elpis_ecs_c_rel) + 63u) & ~(uint64_t)63u;
    const uint64_t big = c->entity_capacity > c->relation_capacity ? c->entity_capacity : c->relation_capacity;
    place(&t, (uint64_t)c->entity_capacity * sizeof(ent_row), &l->ents);
    place(&t, (uint64_t)index_size(c->entity_capacity) * 4u, &l->ent_index);
    place(&t, (uint64_t)c->entity_capacity * 4u, &l->ent_free);
    place(&t, (uint64_t)c->relation_capacity * sizeof(rel_row), &l->rels);
    place(&t, (uint64_t)index_size(c->relation_capacity) * 4u, &l->rel_index);
    place(&t, (uint64_t)c->contribution_capacity * sizeof(contrib), &l->cons);
    place(&t, big * 4u, &l->scratch);
    place(&t, (uint64_t)c->entity_capacity * 4u, &l->map);
    l->total = t; /* every term is bounded by the capacity limits: no overflow below 2^40 */
}

int elpis_ecs_c_rel_required_bytes(const elpis_ecs_c_rel_config *config, uint64_t *arena_bytes) {
    if (!config_ok(config) || !arena_bytes) return ELPIS_ECS_C_REL_INVALID;
    layout l;
    compute_layout(config, &l);
    *arena_bytes = l.total;
    return ELPIS_ECS_C_REL_OK;
}

static void clear_state(elpis_ecs_c_rel *v) {
    v->synced = 0;
    v->generation = v->next_founding_index = v->reclaimed_count = 0;
    memset(v->lineage, 0, 32);
    memset(v->state_root, 0, 32);
    v->clock = v->seq = 0;
    v->ent_count = v->ent_hwm = v->ent_free_count = 0;
    v->rel_count = v->rel_hwm = v->live_relations = 0;
    v->rel_free = v->rel_head = v->rel_tail = NONE;
    v->con_count = v->con_hwm = v->live_contributions = 0;
    v->con_free = NONE;
    memset(v->ent_index, 0, (size_t)(v->ent_mask + 1u) * 4u);
    memset(v->rel_index, 0, (size_t)(v->rel_mask + 1u) * 4u);
}

int elpis_ecs_c_rel_create(const elpis_ecs_c_rel_config *config, elpis_ecs_c_rel **out) {
    if (!out) return ELPIS_ECS_C_REL_INVALID;
    *out = NULL;
    if (!config_ok(config)) return ELPIS_ECS_C_REL_INVALID;
    layout l;
    compute_layout(config, &l);
    if (l.total > config->byte_budget || l.total > ELPIS_ECS_C_REL_MAX_ARENA_BYTES) return ELPIS_ECS_C_REL_CAPACITY;
    uint8_t *block = calloc(1, (size_t)l.total);
    if (!block) return ELPIS_ECS_C_REL_NOMEM;
    elpis_ecs_c_rel *v = (elpis_ecs_c_rel *)block;
    v->cfg = *config;
    v->arena_bytes = l.total;
    v->ents = (ent_row *)(block + l.ents);
    v->ent_index = (uint32_t *)(block + l.ent_index);
    v->ent_free = (uint32_t *)(block + l.ent_free);
    v->rels = (rel_row *)(block + l.rels);
    v->rel_index = (uint32_t *)(block + l.rel_index);
    v->cons = (contrib *)(block + l.cons);
    v->scratch = (uint32_t *)(block + l.scratch);
    v->map = (uint32_t *)(block + l.map);
    v->ent_mask = index_size(config->entity_capacity) - 1u;
    v->rel_mask = index_size(config->relation_capacity) - 1u;
    clear_state(v);
    *out = v;
    return ELPIS_ECS_C_REL_OK;
}

void elpis_ecs_c_rel_destroy(elpis_ecs_c_rel *view) { free(view); }

/* ------------------------------------------------------------------------------------------------------ */
/* The law: liveness, retirement, aggregation                                                             */
/* ------------------------------------------------------------------------------------------------------ */

/* LIVE: the binding passes the authority's admission law on the synced row. */
static int live_under(const ent_row *s, const contrib *c) {
    return s->used && s->kind == ELPIS_ECS_C_SOURCE && s->lifecycle == ELPIS_ECS_C_ACTIVE && s->version == c->version &&
           !memcmp(s->scope, c->scope, 32);
}
/* RETIRED: never live again under any successor table. */
static int retired_under(const ent_row *s, const contrib *c) {
    return !s->used || s->lifecycle == ELPIS_ECS_C_TERMINATED || s->version > c->version;
}
/* The binding invariant of every retained contribution: sync releases RETIRED evidence and admission is at the
 * current binding only, so a retained contribution names its source's present, non-TERMINATED, current version and
 * scope. (Hence liveness reduces to the source being ACTIVE; live_under still states the whole law.) */
static int binding_held(const ent_row *s, const contrib *c) {
    return !retired_under(s, c) && s->version == c->version && !memcmp(s->scope, c->scope, 32);
}

/* The aggregate of relation r's contributions (LIVE by their stored flag, or by the table when `by_table`), in
 * `scope` when given. 0 when a contribution fails its check. */
static int aggregate(const elpis_ecs_c_rel *v, uint32_t r, const uint8_t *scope, int by_table, agg *out,
                     elpis_ecs_c_rel_work *w) {
    memset(out, 0, sizeof(*out));
    ++w->recomputes;
    for (uint32_t i = v->rels[r].head; i != NONE; i = v->cons[i].rnext) {
        const contrib *c = &v->cons[i];
        ++w->contributions;
        if (!con_sound(v, i)) return 0;
        const int live = by_table ? live_under(&v->ents[c->source], c) : (int)c->live;
        if (!live || (scope && memcmp(c->scope, scope, 32))) continue;
        if (!out->live_count || c->first_seq < out->first_seq) {
            out->first_seq = c->first_seq;
            out->first_tick = c->first_tick;
            out->first_provenance = c->first_provenance;
        }
        if (!out->live_count || c->last_seq > out->last_seq) {
            out->last_seq = c->last_seq;
            out->last_tick = c->last_tick;
            out->last_polarity = c->last_polarity;
            out->last_authority = c->observed_authority;
            out->last_provenance = c->last_provenance;
            out->structure_ref = c->structure_ref;
        }
        if (c->assert_count && c->last_assert_tick >= out->last_assert_tick) out->last_assert_tick = c->last_assert_tick;
        if (c->negate_count && c->last_negate_tick >= out->last_negate_tick) out->last_negate_tick = c->last_negate_tick;
        out->assert_count += c->assert_count;
        out->negate_count += c->negate_count;
        out->uncertain_count += c->uncertain_count;
        ++out->live_count;
    }
    return 1;
}

/* Install relation r's recomputed aggregate. A damaged contribution poisons the relation's check instead: damage is
 * never sealed into derived state, and every later read of it refuses CORRUPT. */
static void refresh(elpis_ecs_c_rel *v, uint32_t r, elpis_ecs_c_rel_work *w) {
    agg a;
    if (!aggregate(v, r, NULL, 0, &a, w)) {
        v->rels[r].check = ~rel_check(r, &v->rels[r]);
        return;
    }
    const int was = v->rels[r].live.live_count > 0, now = a.live_count > 0;
    v->rels[r].live = a;
    seal_rel(v, r);
    if (was && !now) --v->live_relations;
    if (!was && now) ++v->live_relations;
}

static void slot_of(const rel_row *r, const agg *a, elpis_ecs_mem_slot *s) {
    memset(s, 0, sizeof(*s));
    s->key = r->key;
    s->payload = r->payload;
    s->first_provenance = a->first_provenance;
    s->last_provenance = a->last_provenance;
    s->structure_ref = a->structure_ref;
    s->relation_type = r->relation_type;
    s->participant_count = r->participant_count;
    s->assert_count = a->assert_count;
    s->negate_count = a->negate_count;
    s->uncertain_count = a->uncertain_count;
    s->last_polarity = a->last_polarity;
    s->last_authority = a->last_authority;
    s->first_tick = a->first_tick;
    s->last_tick = a->last_tick;
    s->last_assert_tick = a->last_assert_tick;
    s->last_negate_tick = a->last_negate_tick;
    memcpy(s->participants, r->participants, sizeof(s->participants));
}

/* ------------------------------------------------------------------------------------------------------ */
/* Record lifetimes                                                                                       */
/* ------------------------------------------------------------------------------------------------------ */

static void mark_dirty(elpis_ecs_c_rel *v, uint32_t r, uint32_t *dirty, uint32_t *nd) {
    if (v->rels[r].stamp == v->epoch) return;
    v->rels[r].stamp = v->epoch;
    dirty[(*nd)++] = r;
}

/* Release a contribution (its relation becomes dirty). */
static void retire(elpis_ecs_c_rel *v, uint32_t i, uint32_t *dirty, uint32_t *nd) {
    contrib *c = &v->cons[i];
    rel_row *r = &v->rels[c->relation];
    ent_row *s = &v->ents[c->source];
    if (c->rprev != NONE) v->cons[c->rprev].rnext = c->rnext; else r->head = c->rnext;
    if (c->rnext != NONE) v->cons[c->rnext].rprev = c->rprev; else r->tail = c->rprev;
    if (c->sprev != NONE) v->cons[c->sprev].snext = c->snext; else s->head = c->snext;
    if (c->snext != NONE) v->cons[c->snext].sprev = c->sprev; else s->tail = c->sprev;
    r->count--;
    /* Clamped: a damaged contribution cannot wrap the totals (verify reports the disagreement). */
    r->total_assert = r->total_assert >= c->assert_count ? r->total_assert - c->assert_count : 0;
    r->total_negate = r->total_negate >= c->negate_count ? r->total_negate - c->negate_count : 0;
    r->total_uncertain = r->total_uncertain >= c->uncertain_count ? r->total_uncertain - c->uncertain_count : 0;
    s->count--;
    if (c->live) v->live_contributions--;
    mark_dirty(v, c->relation, dirty, nd);
    memset(c, 0, sizeof(*c));
    c->rnext = v->con_free;
    v->con_free = i;
    v->con_count--;
}

static void release_relation(elpis_ecs_c_rel *v, uint32_t i, uint64_t *probes) {
    rel_row *r = &v->rels[i];
    index_remove(v->rel_index, v->rel_mask, rel_keys(v), i, probes);
    if (r->prev != NONE) v->rels[r->prev].next = r->next; else v->rel_head = r->next;
    if (r->next != NONE) v->rels[r->next].prev = r->prev; else v->rel_tail = r->prev;
    if (r->live.live_count) v->live_relations--;
    memset(r, 0, sizeof(*r));
    r->next = v->rel_free;
    v->rel_free = i;
    v->rel_count--;
}

/* ------------------------------------------------------------------------------------------------------ */
/* SYNC                                                                                                   */
/* ------------------------------------------------------------------------------------------------------ */

static int row_fields_ok(const elpis_ecs_c_entity *e, uint64_t next_founding) {
    if (is_zero(e->id, 32) || e->kind < ELPIS_ECS_C_OPERATOR || e->kind > ELPIS_ECS_C_SOURCE) return 0;
    if (e->lifecycle < ELPIS_ECS_C_FOUNDED || e->lifecycle > ELPIS_ECS_C_TERMINATED) return 0;
    if (e->reserved[0] || e->reserved[1] || e->reserved2 || e->founding_index >= next_founding) return 0;
    if (e->kind == ELPIS_ECS_C_SOURCE)
        return e->per_turn_limit >= 1 && e->per_turn_limit <= MAX_PER_TURN && e->source_version >= 1 &&
               !is_zero(e->scope, 32);
    return !e->per_turn_limit && !e->source_version && is_zero(e->scope, 32) && !e->source_authority;
}
/* The fields sync mirrors are those of the authority row (an equal generation must repeat them exactly). */
static int row_equal(const ent_row *s, const elpis_ecs_c_entity *e) {
    return s->kind == e->kind && s->lifecycle == e->lifecycle && s->version == e->source_version &&
           s->founding_index == e->founding_index && s->source_authority == e->source_authority &&
           s->per_turn_limit == e->per_turn_limit && !memcmp(s->scope, e->scope, 32) &&
           !memcmp(s->label, e->label, 32);
}
/* A legal successor of the synced row (transitions may be several, folded between two syncs). */
static int row_successor(const ent_row *s, const elpis_ecs_c_entity *e) {
    if (s->kind != e->kind || s->founding_index != e->founding_index || memcmp(s->label, e->label, 32) ||
        s->source_authority != e->source_authority)
        return 0;
    if (s->lifecycle == ELPIS_ECS_C_TERMINATED && e->lifecycle != ELPIS_ECS_C_TERMINATED) return 0;
    if (e->lifecycle == ELPIS_ECS_C_FOUNDED && s->lifecycle != ELPIS_ECS_C_FOUNDED) return 0;
    if (e->source_version < s->version) return 0;
    /* Scope and bound change only with a SOURCE_VERSION, which raises the version. */
    if (e->source_version == s->version && (memcmp(s->scope, e->scope, 32) || s->per_turn_limit != e->per_turn_limit))
        return 0;
    return 1;
}

static void fill_row(ent_row *s, const elpis_ecs_c_entity *e) {
    memcpy(s->id, e->id, 32);
    memcpy(s->scope, e->scope, 32);
    memcpy(s->label, e->label, 32);
    s->version = e->source_version;
    s->founding_index = e->founding_index;
    s->source_authority = e->source_authority;
    s->per_turn_limit = e->per_turn_limit;
    s->kind = e->kind;
    s->lifecycle = e->lifecycle;
}

int elpis_ecs_c_rel_sync(elpis_ecs_c_rel *v, const elpis_ecs_c_rel_authority *t, elpis_ecs_c_rel_sync_result *out) {
    if (!v) return ELPIS_ECS_C_REL_INVALID;
    elpis_ecs_c_rel_sync_result res;
    memset(&res, 0, sizeof(res));
    elpis_ecs_c_rel_work *w = &res.work;
    ++v->epoch; /* scratch only: every stamp of an earlier (possibly refused) sync is stale */

    /* 1. The table alone. */
    if (!t || t->abi_version != ELPIS_ECS_C_REL_ABI_VERSION || t->entity_count < 2 ||
        t->entity_count > ELPIS_ECS_C_REL_MAX_ENTITIES || !t->entities || !t->generation ||
        is_zero(t->lineage, 32) || is_zero(t->state_root, 32))
        return refuse(v, ELPIS_ECS_C_REL_INVALID);
    const uint32_t n = t->entity_count;
    const elpis_ecs_c_entity *rows = t->entities;
    if (n > v->cfg.entity_capacity) return refuse(v, ELPIS_ECS_C_REL_CAPACITY);
    for (uint32_t i = 0; i < n; ++i) {
        ++w->rows;
        if (!row_fields_ok(&rows[i], t->next_founding_index)) return refuse(v, ELPIS_ECS_C_REL_INVALID);
    }
    for (uint32_t i = 0; i < n; ++i) v->scratch[i] = i;
    heapsort_u32(v->scratch, n, table_id_less, rows, &w->sorted);
    for (uint32_t i = 1; i < n; ++i)
        if (!memcmp(rows[v->scratch[i]].id, rows[v->scratch[i - 1]].id, 32)) return refuse(v, ELPIS_ECS_C_REL_INVALID);
    uint32_t ns = 0;
    for (uint32_t i = 0; i < n; ++i)
        if (rows[i].kind == ELPIS_ECS_C_SOURCE) v->scratch[ns++] = i;
    heapsort_u32(v->scratch, ns, table_authority_less, rows, &w->sorted);
    for (uint32_t i = 1; i < ns; ++i)
        if (rows[v->scratch[i]].source_authority == rows[v->scratch[i - 1]].source_authority)
            return refuse(v, ELPIS_ECS_C_REL_INVALID);

    /* 2. A successor of the synced table. map[i]: the mirrored row of table row i, or NONE. */
    uint32_t inserts = 0, vanished = 0;
    for (uint32_t i = 0; i < n; ++i) {
        v->map[i] = v->synced ? find_ent(v, rows[i].id, &w->index_probes) : NONE;
        if (v->map[i] == NONE) ++inserts;
        else v->ents[v->map[i]].stamp = v->epoch;
    }
    if (v->synced) {
        if (memcmp(t->lineage, v->lineage, 32)) return refuse(v, ELPIS_ECS_C_REL_LINEAGE);
        if (t->generation < v->generation || t->next_founding_index < v->next_founding_index ||
            t->reclaimed_count < v->reclaimed_count)
            return refuse(v, ELPIS_ECS_C_REL_REGRESSION);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t m = v->map[i];
            if (m == NONE) {
                /* An entity the synced table did not hold must have been founded after it. */
                if (rows[i].founding_index < v->next_founding_index) return refuse(v, ELPIS_ECS_C_REL_REGRESSION);
            } else if (!row_successor(&v->ents[m], &rows[i])) {
                return refuse(v, ELPIS_ECS_C_REL_REGRESSION);
            }
        }
        for (uint32_t s = 0; s < v->ent_hwm; ++s) {
            if (!v->ents[s].used || v->ents[s].stamp == v->epoch) continue;
            /* Vanished: only a reclamation removes a row, and never a founder. */
            if (v->ents[s].founding_index < 2) return refuse(v, ELPIS_ECS_C_REL_REGRESSION);
            ++vanished;
        }
        if (vanished > t->reclaimed_count - v->reclaimed_count) return refuse(v, ELPIS_ECS_C_REL_REGRESSION);
        if (t->generation == v->generation) {
            /* The same coordinate: the identical table, or a forgery. */
            int same = !vanished && !inserts && !memcmp(t->state_root, v->state_root, 32) &&
                       t->next_founding_index == v->next_founding_index && t->reclaimed_count == v->reclaimed_count;
            for (uint32_t i = 0; same && i < n; ++i) same = row_equal(&v->ents[v->map[i]], &rows[i]);
            if (!same) return refuse(v, ELPIS_ECS_C_REL_REGRESSION);
            res.noop = 1;
            v->tel.syncs++;
            v->tel.noop_syncs++;
            if (out) *out = res;
            return ELPIS_ECS_C_REL_OK;
        }
    }
    if (v->ent_count - vanished + inserts > v->cfg.entity_capacity) return refuse(v, ELPIS_ECS_C_REL_CAPACITY);

    /* 3. Commit: nothing below can fail. */
    uint32_t *dirty = v->scratch, nd = 0;
    for (uint32_t s = 0; v->synced && s < v->ent_hwm; ++s) {
        ent_row *e = &v->ents[s];
        if (!e->used || e->stamp == v->epoch) continue;
        res.sources_changed++;
        for (uint32_t c = e->head; c != NONE;) { /* RETIRED: every contribution of a vanished row */
            const uint32_t next = v->cons[c].snext;
            ++w->contributions;
            retire(v, c, dirty, &nd);
            res.contributions_retired++;
            c = next;
        }
        index_remove(v->ent_index, v->ent_mask, ent_keys(v), s, &w->index_probes);
        memset(e, 0, sizeof(*e));
        v->ent_free[v->ent_free_count++] = s;
        v->ent_count--;
    }
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t s = v->map[i];
        if (s == NONE) {
            s = v->ent_free_count ? v->ent_free[--v->ent_free_count] : v->ent_hwm++;
            ent_row *e = &v->ents[s];
            memset(e, 0, sizeof(*e));
            fill_row(e, &rows[i]);
            e->used = 1;
            e->head = e->tail = NONE;
            e->stamp = v->epoch;
            index_insert(v->ent_index, v->ent_mask, ent_keys(v), s, &w->index_probes);
            v->ent_count++;
            continue;
        }
        ent_row *e = &v->ents[s];
        const int was_active = e->lifecycle == ELPIS_ECS_C_ACTIVE;
        const int changed = was_active != (rows[i].lifecycle == ELPIS_ECS_C_ACTIVE) ||
                            e->version != rows[i].source_version || memcmp(e->scope, rows[i].scope, 32) ||
                            (e->lifecycle != ELPIS_ECS_C_TERMINATED && rows[i].lifecycle == ELPIS_ECS_C_TERMINATED);
        fill_row(e, &rows[i]);
        if (!changed) continue;
        res.sources_changed++;
        /* Only this source's dependents are visited (the dependency index). */
        for (uint32_t c = e->head; c != NONE;) {
            contrib *k = &v->cons[c];
            const uint32_t next = k->snext;
            ++w->contributions;
            if (retired_under(e, k)) {
                retire(v, c, dirty, &nd);
                res.contributions_retired++;
            } else {
                const uint32_t live = (uint32_t)live_under(e, k);
                if (live != k->live) {
                    const int sound = con_sound(v, c); /* a damaged record stays detectably damaged */
                    k->live = live;
                    if (live) v->live_contributions++; else v->live_contributions--;
                    if (sound) seal_con(v, c);
                    mark_dirty(v, k->relation, dirty, &nd);
                    res.contributions_flipped++;
                }
            }
            c = next;
        }
    }
    for (uint32_t d = 0; d < nd; ++d) {
        const uint32_t r = dirty[d];
        ++w->relations;
        if (!v->rels[r].count) {
            release_relation(v, r, &w->index_probes);
            res.relations_released++;
            continue;
        }
        refresh(v, r, w);
        res.relations_recomputed++;
    }
    v->synced = 1;
    v->generation = t->generation;
    v->next_founding_index = t->next_founding_index;
    v->reclaimed_count = t->reclaimed_count;
    memcpy(v->lineage, t->lineage, 32);
    memcpy(v->state_root, t->state_root, 32);
    v->revision++;
    v->tel.syncs++;
    v->tel.flips += res.contributions_flipped;
    v->tel.retired_contributions += res.contributions_retired;
    v->tel.released_relations += res.relations_released;
    if (out) *out = res;
    return ELPIS_ECS_C_REL_OK;
}

/* ------------------------------------------------------------------------------------------------------ */
/* ADMIT                                                                                                  */
/* ------------------------------------------------------------------------------------------------------ */

static int coordinate_ok(const elpis_ecs_c_rel *v, const elpis_ecs_c_rel_coordinate *c) {
    return v->synced && c->generation == v->generation && !memcmp(c->state_root, v->state_root, 32);
}

/* The authority's admission law (elpis_ecs_c_check_evidence) on the synced table, in its order. */
static int admission(const elpis_ecs_c_rel *v, const uint8_t source[32], uint64_t version, const uint8_t scope[32],
                     uint32_t *row, uint64_t *probes) {
    const uint32_t s = find_ent(v, source, probes);
    if (s == NONE) return ELPIS_ECS_C_REL_UNKNOWN_SOURCE;
    const ent_row *e = &v->ents[s];
    if (e->kind != ELPIS_ECS_C_SOURCE) return ELPIS_ECS_C_REL_NOT_SOURCE;
    if (e->lifecycle != ELPIS_ECS_C_ACTIVE) return ELPIS_ECS_C_REL_LIFECYCLE;
    if (version != e->version) return ELPIS_ECS_C_REL_SOURCE_STALE;
    if (memcmp(scope, e->scope, 32)) return ELPIS_ECS_C_REL_SCOPE_CONFLICT;
    if (row) *row = s;
    return ELPIS_ECS_C_REL_OK;
}

int elpis_ecs_c_rel_check(const elpis_ecs_c_rel *v, const uint8_t source[32], uint64_t version,
                          const uint8_t scope[32]) {
    if (!v || !source || !scope) return ELPIS_ECS_C_REL_INVALID;
    if (!v->synced) return ELPIS_ECS_C_REL_STALE_VIEW;
    uint64_t probes = 0;
    return admission(v, source, version, scope, NULL, &probes);
}

static int participant_less(const elpis_ecs_mem_participant *a, const elpis_ecs_mem_participant *b) {
    if (a->role != b->role) return a->role < b->role;
    if (a->ordinal != b->ordinal) return a->ordinal < b->ordinal;
    return memcmp(a->entity.b, b->entity.b, 32) < 0;
}
/* The memory's canonical participant order (elpis_ecs_mem_relation_key has validated them). */
static void canonical(const elpis_ecs_mem_observation *o, elpis_ecs_mem_participant *out) {
    memset(out, 0, sizeof(elpis_ecs_mem_participant) * ELPIS_ECS_MEM_MAX_PARTICIPANTS);
    for (uint32_t i = 0; i < o->participant_count; ++i) {
        elpis_ecs_mem_participant p = o->participants[i];
        uint32_t j = i;
        while (j > 0 && participant_less(&p, &out[j - 1])) {
            out[j] = out[j - 1];
            --j;
        }
        out[j] = p;
    }
}

int elpis_ecs_c_rel_admit(elpis_ecs_c_rel *v, const elpis_ecs_c_rel_coordinate *expected, const uint8_t source[32],
                          uint64_t version, const uint8_t scope[32], const elpis_ecs_mem_observation *o,
                          elpis_ecs_mem_digest *out_key, elpis_ecs_c_rel_work *work) {
    if (!v) return ELPIS_ECS_C_REL_INVALID;
    elpis_ecs_c_rel_work w;
    memset(&w, 0, sizeof(w));
    if (!expected || !source || !scope || !o) return refuse(v, ELPIS_ECS_C_REL_INVALID);
    if (!coordinate_ok(v, expected)) return refuse(v, ELPIS_ECS_C_REL_STALE_VIEW);
    elpis_ecs_mem_digest key;
    if (elpis_ecs_mem_relation_key(o, &key) != ELPIS_ECS_MEM_OK || o->polarity < ELPIS_ECS_MEM_ASSERT ||
        o->polarity > ELPIS_ECS_MEM_UNCERTAIN)
        return refuse(v, ELPIS_ECS_C_REL_INVALID);
    uint32_t s = NONE;
    const int law = admission(v, source, version, scope, &s, &w.index_probes);
    if (law != ELPIS_ECS_C_REL_OK) return refuse(v, law);
    if (o->source_authority != v->ents[s].source_authority) return refuse(v, ELPIS_ECS_C_REL_BINDING_MISMATCH);
    if (o->tick < v->clock || v->seq == UINT64_MAX || v->revision == UINT64_MAX) return refuse(v, ELPIS_ECS_C_REL_CLOCK);

    /* Plan: the relation, then this binding's contribution in it. */
    uint32_t r = find_rel(v, key.b, &w.index_probes), c = NONE;
    if (r != NONE) {
        const rel_row *rr = &v->rels[r];
        ++w.relations;
        if (!rel_sound(v, r)) return refuse(v, ELPIS_ECS_C_REL_CORRUPT);
        const uint32_t total = o->polarity == ELPIS_ECS_MEM_ASSERT   ? rr->total_assert
                               : o->polarity == ELPIS_ECS_MEM_NEGATE ? rr->total_negate
                                                                     : rr->total_uncertain;
        if (total == UINT32_MAX) return refuse(v, ELPIS_ECS_C_REL_CLOCK);
        for (uint32_t i = rr->head; i != NONE; i = v->cons[i].rnext) {
            ++w.contributions;
            if (!con_sound(v, i)) return refuse(v, ELPIS_ECS_C_REL_CORRUPT);
            if (v->cons[i].source == s && v->cons[i].version == version) c = i;
        }
    } else if (v->rel_free == NONE && v->rel_hwm == v->cfg.relation_capacity) {
        return refuse(v, ELPIS_ECS_C_REL_CAPACITY);
    }
    if (c == NONE && v->con_free == NONE && v->con_hwm == v->cfg.contribution_capacity)
        return refuse(v, ELPIS_ECS_C_REL_CAPACITY);

    /* Commit: nothing below can fail. */
    if (r == NONE) {
        if (v->rel_free != NONE) {
            r = v->rel_free;
            v->rel_free = v->rels[r].next;
        } else {
            r = v->rel_hwm++;
        }
        rel_row *nr = &v->rels[r];
        memset(nr, 0, sizeof(*nr));
        nr->key = key;
        nr->payload = o->payload;
        nr->relation_type = o->relation_type;
        nr->participant_count = o->participant_count;
        canonical(o, nr->participants);
        nr->head = nr->tail = NONE;
        nr->used = 1;
        nr->prev = v->rel_tail;
        nr->next = NONE;
        if (v->rel_tail != NONE) v->rels[v->rel_tail].next = r; else v->rel_head = r;
        v->rel_tail = r;
        index_insert(v->rel_index, v->rel_mask, rel_keys(v), r, &w.index_probes);
        v->rel_count++;
    }
    rel_row *rr = &v->rels[r];
    if (c == NONE) {
        if (v->con_free != NONE) {
            c = v->con_free;
            v->con_free = v->cons[c].rnext;
        } else {
            c = v->con_hwm++;
        }
        contrib *k = &v->cons[c];
        ent_row *e = &v->ents[s];
        memset(k, 0, sizeof(*k));
        k->relation = r;
        k->source = s;
        k->version = version;
        memcpy(k->scope, scope, 32);
        k->observed_authority = o->source_authority;
        k->live = 1;
        k->first_seq = v->seq;
        k->first_tick = o->tick;
        k->first_provenance = o->provenance;
        k->rprev = rr->tail;
        k->rnext = NONE;
        if (rr->tail != NONE) v->cons[rr->tail].rnext = c; else rr->head = c;
        rr->tail = c;
        k->sprev = e->tail;
        k->snext = NONE;
        if (e->tail != NONE) v->cons[e->tail].snext = c; else e->head = c;
        e->tail = c;
        rr->count++;
        e->count++;
        v->con_count++;
        v->live_contributions++;
    }
    contrib *k = &v->cons[c];
    if (o->polarity == ELPIS_ECS_MEM_ASSERT) {
        k->assert_count++;
        k->last_assert_tick = o->tick;
        rr->total_assert++;
    } else if (o->polarity == ELPIS_ECS_MEM_NEGATE) {
        k->negate_count++;
        k->last_negate_tick = o->tick;
        rr->total_negate++;
    } else {
        k->uncertain_count++;
        rr->total_uncertain++;
    }
    k->last_seq = v->seq;
    k->last_tick = o->tick;
    k->last_polarity = o->polarity;
    k->last_provenance = o->provenance;
    k->structure_ref = o->structure_ref;
    seal_con(v, c);
    refresh(v, r, &w);
    v->clock = o->tick;
    v->seq++;
    v->revision++;
    v->tel.admissions++;
    if (out_key) *out_key = key;
    work_add(work, &w);
    return ELPIS_ECS_C_REL_OK;
}

/* ------------------------------------------------------------------------------------------------------ */
/* Reads                                                                                                  */
/* ------------------------------------------------------------------------------------------------------ */

static int read_ok(const elpis_ecs_c_rel *v, const elpis_ecs_c_rel_coordinate *expected) {
    if (!v->synced) return ELPIS_ECS_C_REL_STALE_VIEW;
    if (expected && !coordinate_ok(v, expected)) return ELPIS_ECS_C_REL_STALE_VIEW;
    return ELPIS_ECS_C_REL_OK;
}

/* The live aggregate of relation r in `scope` (the maintained one when unscoped). 0: a failed check. */
static int view_of(const elpis_ecs_c_rel *v, uint32_t r, const uint8_t *scope, agg *out, elpis_ecs_c_rel_work *w) {
    if (!rel_sound(v, r)) return 0;
    if (!scope) {
        *out = v->rels[r].live;
        return 1;
    }
    return aggregate(v, r, scope, 0, out, w);
}

/* The memory's LOOKUP request shape (ecs_memory.c query_ok, LOOKUP only). */
static int request_ok(const elpis_ecs_mem_request *q) {
    if (!q || q->abi_version != ELPIS_ECS_MEM_ABI_V1 || q->channel != ELPIS_ECS_MEM_LOOKUP) return 0;
    if (q->candidate_source != ELPIS_ECS_MEM_CANDIDATES_CALLER || q->candidate_count || q->candidates) return 0;
    if (q->cue_count >= ELPIS_ECS_MEM_MAX_PARTICIPANTS) return 0;
    for (uint32_t i = 0; i < q->cue_count; ++i) {
        if (q->cue[i].role == q->target_role && q->cue[i].ordinal == q->target_ordinal) return 0;
        for (uint32_t j = 0; j < i; ++j)
            if (q->cue[i].role == q->cue[j].role && q->cue[i].ordinal == q->cue[j].ordinal) return 0;
    }
    for (uint32_t i = q->cue_count; i < ELPIS_ECS_MEM_MAX_PARTICIPANTS; ++i)
        if (!is_zero(&q->cue[i], sizeof(q->cue[i]))) return 0;
    return 1;
}
static const elpis_ecs_mem_participant *filler_of(const rel_row *r, uint32_t role, uint32_t ordinal) {
    for (uint32_t i = 0; i < r->participant_count; ++i)
        if (r->participants[i].role == role && r->participants[i].ordinal == ordinal) return &r->participants[i];
    return NULL;
}
static int matches(const rel_row *r, const elpis_ecs_mem_request *q) {
    if (r->relation_type != q->relation_type) return 0;
    for (uint32_t c = 0; c < q->cue_count; ++c) {
        const elpis_ecs_mem_participant *p = filler_of(r, q->cue[c].role, q->cue[c].ordinal);
        if (!p || memcmp(p->entity.b, q->cue[c].entity.b, 32)) return 0;
    }
    return 1;
}
static int64_t net(const agg *a) { return (int64_t)a->assert_count - (int64_t)a->negate_count; }
/* The memory's LOOKUP order: latest assert, then larger net evidence, then smaller filler, then smaller key. */
static int better(const agg *a, const elpis_ecs_mem_participant *fa, const rel_row *ra, const agg *b,
                  const elpis_ecs_mem_participant *fb, const rel_row *rb) {
    if (a->last_assert_tick != b->last_assert_tick) return a->last_assert_tick > b->last_assert_tick;
    if (net(a) != net(b)) return net(a) > net(b);
    const int c = memcmp(fa->entity.b, fb->entity.b, 32);
    if (c) return c < 0;
    return memcmp(ra->key.b, rb->key.b, 32) < 0;
}

int elpis_ecs_c_rel_query(const elpis_ecs_c_rel *v, const elpis_ecs_c_rel_coordinate *expected, const uint8_t *scope,
                          const elpis_ecs_mem_request *q, elpis_ecs_mem_answer *out, elpis_ecs_c_rel_work *work) {
    if (!v || !out || !request_ok(q)) return ELPIS_ECS_C_REL_INVALID;
    const int ok = read_ok(v, expected);
    if (ok != ELPIS_ECS_C_REL_OK) return ok;
    elpis_ecs_c_rel_work w;
    memset(&w, 0, sizeof(w));
    elpis_ecs_mem_answer a;
    memset(&a, 0, sizeof(a));
    uint32_t best = NONE;
    agg ba;
    memset(&ba, 0, sizeof(ba));
    const elpis_ecs_mem_participant *bf = NULL;
    for (uint32_t r = v->rel_head; r != NONE; r = v->rels[r].next) {
        const rel_row *rr = &v->rels[r];
        ++a.work.slot_scans;
        ++w.relations;
        if (!matches(rr, q)) continue;
        const elpis_ecs_mem_participant *f = filler_of(rr, q->target_role, q->target_ordinal);
        if (!f) continue;
        agg g;
        if (!view_of(v, r, scope, &g, &w)) return ELPIS_ECS_C_REL_CORRUPT;
        if (!g.live_count) continue; /* not in the live view */
        ++a.matches;
        a.uncertain = (uint64_t)a.uncertain + g.uncertain_count > UINT32_MAX ? UINT32_MAX : a.uncertain + g.uncertain_count;
        if (net(&g) < 0) ++a.contradicted;
        if (net(&g) > 0 && (best == NONE || better(&g, f, rr, &ba, bf, &v->rels[best]))) {
            best = r;
            ba = g;
            bf = f;
        }
    }
    a.runner_up = INT64_MIN;
    if (best != NONE) {
        for (uint32_t r = v->rel_head; r != NONE; r = v->rels[r].next) {
            const rel_row *rr = &v->rels[r];
            ++a.work.slot_scans;
            ++w.relations;
            if (r == best || !matches(rr, q)) continue;
            const elpis_ecs_mem_participant *f = filler_of(rr, q->target_role, q->target_ordinal);
            if (!f || !memcmp(f->entity.b, bf->entity.b, 32)) continue;
            agg g;
            if (!view_of(v, r, scope, &g, &w)) return ELPIS_ECS_C_REL_CORRUPT;
            if (!g.live_count || net(&g) <= 0) continue;
            ++a.conflicts;
            if (net(&g) > a.runner_up) a.runner_up = net(&g);
        }
        a.found = 1;
        a.answer = bf->entity;
        a.answer_key = v->rels[best].key;
        a.score = net(&ba);
        a.answer_tick = ba.last_assert_tick;
    }
    *out = a;
    work_add(work, &w);
    return ELPIS_ECS_C_REL_OK;
}

int elpis_ecs_c_rel_relation_read(const elpis_ecs_c_rel *v, const elpis_ecs_c_rel_coordinate *expected,
                                  const uint8_t *scope, const elpis_ecs_mem_digest *key, elpis_ecs_mem_slot *out) {
    if (!v || !key || !out) return ELPIS_ECS_C_REL_INVALID;
    const int ok = read_ok(v, expected);
    if (ok != ELPIS_ECS_C_REL_OK) return ok;
    elpis_ecs_c_rel_work w;
    memset(&w, 0, sizeof(w));
    const uint32_t r = find_rel(v, key->b, &w.index_probes);
    if (r == NONE) return ELPIS_ECS_C_REL_NOT_FOUND;
    agg g;
    if (!view_of(v, r, scope, &g, &w)) return ELPIS_ECS_C_REL_CORRUPT;
    if (!g.live_count) return ELPIS_ECS_C_REL_NOT_FOUND;
    slot_of(&v->rels[r], &g, out);
    return ELPIS_ECS_C_REL_OK;
}

int elpis_ecs_c_rel_support_of(const elpis_ecs_c_rel *v, const elpis_ecs_mem_digest *key, elpis_ecs_c_rel_support *out,
                               size_t capacity, size_t *count) {
    if (!v || !key || !count || (capacity && !out)) return ELPIS_ECS_C_REL_INVALID;
    if (!v->synced) return ELPIS_ECS_C_REL_STALE_VIEW;
    uint64_t probes = 0;
    const uint32_t r = find_rel(v, key->b, &probes);
    if (r == NONE) return ELPIS_ECS_C_REL_NOT_FOUND;
    size_t n = 0;
    for (uint32_t i = v->rels[r].head; i != NONE; i = v->cons[i].rnext) {
        if (!con_sound(v, i)) return ELPIS_ECS_C_REL_CORRUPT;
        ++n;
    }
    if (n > capacity) {
        *count = n;
        return ELPIS_ECS_C_REL_CAPACITY;
    }
    size_t j = 0;
    for (uint32_t i = v->rels[r].head; i != NONE; i = v->cons[i].rnext, ++j) {
        const contrib *c = &v->cons[i];
        elpis_ecs_c_rel_support *s = &out[j];
        memset(s, 0, sizeof(*s));
        memcpy(s->source, v->ents[c->source].id, 32);
        memcpy(s->scope, c->scope, 32);
        s->version = c->version;
        s->source_authority = c->observed_authority;
        s->state = c->live ? ELPIS_ECS_C_REL_LIVE : ELPIS_ECS_C_REL_SUSPENDED;
        s->assert_count = c->assert_count;
        s->negate_count = c->negate_count;
        s->uncertain_count = c->uncertain_count;
        s->first_tick = c->first_tick;
        s->last_tick = c->last_tick;
        s->first_provenance = c->first_provenance;
        s->last_provenance = c->last_provenance;
    }
    *count = n;
    return ELPIS_ECS_C_REL_OK;
}

int elpis_ecs_c_rel_dependents(const elpis_ecs_c_rel *v, const uint8_t source[32], elpis_ecs_mem_digest *out,
                               size_t capacity, size_t *count) {
    if (!v || !source || !count || (capacity && !out)) return ELPIS_ECS_C_REL_INVALID;
    if (!v->synced) return ELPIS_ECS_C_REL_STALE_VIEW;
    uint64_t probes = 0;
    const uint32_t s = find_ent(v, source, &probes);
    if (s == NONE) return ELPIS_ECS_C_REL_NOT_FOUND;
    size_t n = 0;
    for (uint32_t i = v->ents[s].head; i != NONE; i = v->cons[i].snext) {
        if (!con_sound(v, i)) return ELPIS_ECS_C_REL_CORRUPT;
        ++n;
    }
    if (n > capacity) {
        *count = n;
        return ELPIS_ECS_C_REL_CAPACITY;
    }
    size_t j = 0;
    for (uint32_t i = v->ents[s].head; i != NONE; i = v->cons[i].snext) out[j++] = v->rels[v->cons[i].relation].key;
    *count = n;
    return ELPIS_ECS_C_REL_OK;
}

int elpis_ecs_c_rel_export(elpis_ecs_c_rel *v, const elpis_ecs_c_rel_coordinate *expected, const uint8_t *scope,
                           elpis_ecs_mem_slot *out, size_t capacity, size_t *count, elpis_ecs_c_rel_work *work) {
    if (!v || !count || (capacity && !out)) return ELPIS_ECS_C_REL_INVALID;
    const int ok = read_ok(v, expected);
    if (ok != ELPIS_ECS_C_REL_OK) return ok;
    elpis_ecs_c_rel_work w;
    memset(&w, 0, sizeof(w));
    uint32_t n = 0;
    for (uint32_t r = v->rel_head; r != NONE; r = v->rels[r].next) {
        agg g;
        ++w.relations;
        if (!view_of(v, r, scope, &g, &w)) return ELPIS_ECS_C_REL_CORRUPT;
        if (g.live_count) v->scratch[n++] = r;
    }
    if (n > capacity) {
        *count = n;
        return ELPIS_ECS_C_REL_CAPACITY;
    }
    heapsort_u32(v->scratch, n, rel_key_less, v->rels, &w.sorted);
    for (uint32_t i = 0; i < n; ++i) {
        agg g;
        (void)view_of(v, v->scratch[i], scope, &g, &w);
        slot_of(&v->rels[v->scratch[i]], &g, &out[i]);
    }
    *count = n;
    work_add(work, &w);
    return ELPIS_ECS_C_REL_OK;
}

int elpis_ecs_c_rel_read_status(const elpis_ecs_c_rel *v, elpis_ecs_c_rel_status *out) {
    if (!v || !out) return ELPIS_ECS_C_REL_INVALID;
    elpis_ecs_c_rel_status s = v->tel;
    s.config = v->cfg;
    s.synced = v->synced;
    s.entities = v->ent_count;
    s.relations = v->rel_count;
    s.live_relations = v->live_relations;
    s.contributions = v->con_count;
    s.live_contributions = v->live_contributions;
    s.generation = v->generation;
    memcpy(s.lineage, v->lineage, 32);
    memcpy(s.state_root, v->state_root, 32);
    s.next_founding_index = v->next_founding_index;
    s.reclaimed_count = v->reclaimed_count;
    s.clock = v->clock;
    s.sequence = v->seq;
    s.revision = v->revision;
    s.arena_bytes = v->arena_bytes;
    *out = s;
    return ELPIS_ECS_C_REL_OK;
}

/* ------------------------------------------------------------------------------------------------------ */
/* Identity, verification, rebuild, reset                                                                 */
/* ------------------------------------------------------------------------------------------------------ */

static void put32(elpis_sha256_ctx *h, uint32_t x) {
    uint8_t b[4];
    le32(b, x);
    elpis_sha256_update(h, b, 4);
}
static void put64(elpis_sha256_ctx *h, uint64_t x) {
    uint8_t b[8];
    le64(b, x);
    elpis_sha256_update(h, b, 8);
}

int elpis_ecs_c_rel_identity(elpis_ecs_c_rel *v, uint8_t out[32]) {
    if (!v || !out) return ELPIS_ECS_C_REL_INVALID;
    static const char domain[] = "elpis.ecs_c.relational.view.r0";
    uint64_t sorted = 0;
    elpis_sha256_ctx h;
    elpis_sha256_init(&h);
    elpis_sha256_update(&h, domain, sizeof(domain));
    put32(&h, v->synced);
    put64(&h, v->generation);
    put64(&h, v->next_founding_index);
    put64(&h, v->reclaimed_count);
    elpis_sha256_update(&h, v->lineage, 32);
    elpis_sha256_update(&h, v->state_root, 32);
    put64(&h, v->clock);
    put64(&h, v->seq);
    uint32_t n = 0;
    for (uint32_t s = 0; s < v->ent_hwm; ++s)
        if (v->ents[s].used) v->scratch[n++] = s;
    heapsort_u32(v->scratch, n, ent_id_less, v->ents, &sorted);
    put32(&h, n);
    for (uint32_t i = 0; i < n; ++i) {
        const ent_row *e = &v->ents[v->scratch[i]];
        elpis_sha256_update(&h, e->id, 32);
        put32(&h, ((uint32_t)e->kind << 8) | e->lifecycle);
        put64(&h, e->version);
        put64(&h, e->founding_index);
        put32(&h, e->source_authority);
        put32(&h, e->per_turn_limit);
        elpis_sha256_update(&h, e->scope, 32);
        elpis_sha256_update(&h, e->label, 32);
    }
    n = 0;
    for (uint32_t r = v->rel_head; r != NONE; r = v->rels[r].next) v->scratch[n++] = r;
    heapsort_u32(v->scratch, n, rel_key_less, v->rels, &sorted);
    put32(&h, n);
    for (uint32_t i = 0; i < n; ++i) {
        const rel_row *r = &v->rels[v->scratch[i]];
        elpis_sha256_update(&h, r->key.b, 32);
        elpis_sha256_update(&h, r->payload.b, 32);
        put32(&h, r->relation_type);
        put32(&h, r->participant_count);
        for (uint32_t p = 0; p < r->participant_count; ++p) {
            put32(&h, r->participants[p].role);
            put32(&h, r->participants[p].ordinal);
            elpis_sha256_update(&h, r->participants[p].entity.b, 32);
        }
        put32(&h, r->count);
        for (uint32_t c = r->head; c != NONE; c = v->cons[c].rnext) {
            const contrib *k = &v->cons[c];
            elpis_sha256_update(&h, v->ents[k->source].id, 32);
            put64(&h, k->version);
            elpis_sha256_update(&h, k->scope, 32);
            put32(&h, k->observed_authority);
            put32(&h, k->assert_count);
            put32(&h, k->negate_count);
            put32(&h, k->uncertain_count);
            put32(&h, k->last_polarity);
            put64(&h, k->first_seq);
            put64(&h, k->last_seq);
            put64(&h, k->first_tick);
            put64(&h, k->last_tick);
            put64(&h, k->last_assert_tick);
            put64(&h, k->last_negate_tick);
            elpis_sha256_update(&h, k->first_provenance.b, 32);
            elpis_sha256_update(&h, k->last_provenance.b, 32);
            elpis_sha256_update(&h, k->structure_ref.b, 32);
        }
    }
    elpis_sha256_final(&h, out);
    return ELPIS_ECS_C_REL_OK;
}

int elpis_ecs_c_rel_verify(const elpis_ecs_c_rel *v, uint64_t *mismatches) {
    if (!v || !mismatches) return ELPIS_ECS_C_REL_INVALID;
    elpis_ecs_c_rel_work w;
    memset(&w, 0, sizeof(w));
    uint64_t bad = 0;
    uint32_t live_contributions = 0, live_relations = 0, contributions = 0;
    for (uint32_t r = v->rel_head; r != NONE; r = v->rels[r].next) {
        const rel_row *rr = &v->rels[r];
        uint32_t count = 0;
        uint64_t ta = 0, tn = 0, tu = 0;
        for (uint32_t c = rr->head; c != NONE; c = v->cons[c].rnext) {
            const contrib *k = &v->cons[c];
            const ent_row *e = &v->ents[k->source];
            ++count;
            ta += k->assert_count;
            tn += k->negate_count;
            tu += k->uncertain_count;
            if (!con_sound(v, c)) ++bad;
            if (k->relation != r || (uint32_t)live_under(e, k) != k->live || !binding_held(e, k)) ++bad;
            if (k->live) ++live_contributions;
        }
        contributions += count;
        if (count != rr->count || !count || ta != rr->total_assert || tn != rr->total_negate || tu != rr->total_uncertain)
            ++bad;
        agg a;
        if (!aggregate(v, r, NULL, 1, &a, &w) || memcmp(&a, &rr->live, sizeof(a)) || !rel_sound(v, r)) ++bad;
        if (rr->live.live_count) ++live_relations;
    }
    if (contributions != v->con_count || live_contributions != v->live_contributions ||
        live_relations != v->live_relations)
        ++bad;
    *mismatches = bad;
    return ELPIS_ECS_C_REL_OK;
}

int elpis_ecs_c_rel_rebuild(elpis_ecs_c_rel *v, elpis_ecs_c_rel_work *work) {
    if (!v) return ELPIS_ECS_C_REL_INVALID;
    if (!v->synced) return refuse(v, ELPIS_ECS_C_REL_STALE_VIEW);
    if (v->revision == UINT64_MAX) return refuse(v, ELPIS_ECS_C_REL_CLOCK);
    elpis_ecs_c_rel_work w;
    memset(&w, 0, sizeof(w));
    /* The evidence must be sound: derived state is rebuilt from it, never the other way round. */
    for (uint32_t r = v->rel_head; r != NONE; r = v->rels[r].next) {
        ++w.relations;
        for (uint32_t c = v->rels[r].head; c != NONE; c = v->cons[c].rnext) {
            ++w.contributions;
            if (!con_sound(v, c) || !binding_held(&v->ents[v->cons[c].source], &v->cons[c]))
                return refuse(v, ELPIS_ECS_C_REL_CORRUPT);
        }
    }
    v->live_contributions = 0;
    v->live_relations = 0;
    for (uint32_t r = v->rel_head; r != NONE; r = v->rels[r].next) {
        for (uint32_t c = v->rels[r].head; c != NONE; c = v->cons[c].rnext) {
            contrib *k = &v->cons[c];
            ++w.contributions;
            k->live = (uint32_t)live_under(&v->ents[k->source], k);
            seal_con(v, c);
            if (k->live) v->live_contributions++;
        }
        agg a;
        (void)aggregate(v, r, NULL, 0, &a, &w);
        v->rels[r].live = a;
        seal_rel(v, r);
        if (a.live_count) v->live_relations++;
    }
    v->revision++;
    v->tel.rebuilds++;
    work_add(work, &w);
    return ELPIS_ECS_C_REL_OK;
}

int elpis_ecs_c_rel_reset(elpis_ecs_c_rel *v) {
    if (!v) return ELPIS_ECS_C_REL_INVALID;
    const uint64_t revision = v->revision;
    clear_state(v);
    v->revision = revision == UINT64_MAX ? revision : revision + 1u;
    v->tel.resets++;
    return ELPIS_ECS_C_REL_OK;
}

#ifdef ELPIS_ECS_C_REL_TESTING
/* Test-only damage of retained records (compiled into the tests, never into the library). what: 1 a relation's live
 * assert count, check not resealed (detectable on read); 2 the same, resealed (only verify detects it); 3 the first
 * contribution's assert count, not resealed; 4 the first contribution's live flag, resealed (verify detects, rebuild
 * repairs); 5 the relation's retained assert total (the overflow guard); 6 the first contribution's binding scope,
 * resealed (a binding no table can hold: verify detects, rebuild refuses). */
int ecs_c_rel_testing_damage(elpis_ecs_c_rel *v, const elpis_ecs_mem_digest *key, uint32_t what, uint64_t value) {
    uint64_t probes = 0;
    const uint32_t r = find_rel(v, key->b, &probes);
    if (r == NONE) return ELPIS_ECS_C_REL_NOT_FOUND;
    rel_row *rr = &v->rels[r];
    switch (what) {
    case 1: rr->live.assert_count = (uint32_t)value; return ELPIS_ECS_C_REL_OK;
    case 2: rr->live.assert_count = (uint32_t)value; seal_rel(v, r); return ELPIS_ECS_C_REL_OK;
    case 3: v->cons[rr->head].assert_count = (uint32_t)value; return ELPIS_ECS_C_REL_OK;
    case 4: v->cons[rr->head].live ^= 1u; seal_con(v, rr->head); return ELPIS_ECS_C_REL_OK;
    case 5: rr->total_assert = (uint32_t)value; return ELPIS_ECS_C_REL_OK;
    case 6: v->cons[rr->head].scope[0] ^= (uint8_t)value; seal_con(v, rr->head); return ELPIS_ECS_C_REL_OK;
    default: return ELPIS_ECS_C_REL_INVALID;
    }
}
#endif
