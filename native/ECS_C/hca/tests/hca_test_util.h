/* Shared fixtures of the ECS_C HCA native tests: a small world builder (cells sorted by key, factors in canonical
 * order) and the reference's coding fixture (research/ecsc_hca_r0/fixtures.py `coding`) in native form. */
#ifndef ELPIS_HCA_TEST_UTIL_H
#define ELPIS_HCA_TEST_UTIL_H

#include "elpis/ecs_c_hca.h"
#include "elpis/sha256.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OK(x) assert((x) == ELPIS_HCA_OK)

enum { KIND_ENVIRONMENT = 1, KIND_DEPENDENCY = 2, KIND_POLICY = 3, KIND_EXTRA = 4, KIND_PLAN = 20 };

typedef struct {
    elpis_hca_cell cells[ELPIS_HCA_MAX_CELLS];
    uint32_t n_cells;
    elpis_hca_schema schemas[64];
    uint32_t n_schemas;
    elpis_hca_factor factors[ELPIS_HCA_MAX_FACTORS];
    uint32_t n_factors;
} world;

static void key_of(const char *label, uint8_t out[32]) { elpis_sha256(label, strlen(label), out); }

static void add_cell(world *w, const char *label, uint32_t scope, uint32_t kind, int binding, uint8_t source) {
    elpis_hca_cell *c = &w->cells[w->n_cells++];
    memset(c, 0, sizeof(*c));
    key_of(label, c->key);
    c->scope = scope;
    c->kind = kind;
    c->binding = (uint8_t)binding;
    c->source_state = source;
}

static int cell_cmp(const void *a, const void *b) {
    return memcmp(((const elpis_hca_cell *)a)->key, ((const elpis_hca_cell *)b)->key, 32);
}

/* Sort the cells into key order (call once, after the last cell and before the first factor). */
static void seal_cells(world *w) { qsort(w->cells, w->n_cells, sizeof(elpis_hca_cell), cell_cmp); }

static uint32_t idx(const world *w, const char *label) {
    uint8_t key[32];
    key_of(label, key);
    for (uint32_t i = 0; i < w->n_cells; ++i)
        if (!memcmp(w->cells[i].key, key, 32)) return i;
    assert(!"no such cell");
    return 0;
}

/* A schema whose roles are type * 100 + position (distinct) over the given kinds. */
static void add_schema(world *w, uint32_t type, uint32_t weight, uint32_t arity, const uint32_t *kinds) {
    elpis_hca_schema *s = &w->schemas[w->n_schemas++];
    memset(s, 0, sizeof(*s));
    s->relation_type = type;
    s->arity = arity;
    s->weight = weight;
    for (uint32_t p = 0; p < arity; ++p) s->roles[p] = type * 100 + p, s->kinds[p] = kinds[p];
}

static void add_factor(world *w, uint32_t type, uint8_t status, uint32_t arity, const char *const *members) {
    elpis_hca_factor *f = &w->factors[w->n_factors++];
    memset(f, 0, sizeof(*f));
    f->relation_type = type;
    f->status = status;
    f->arity = (uint8_t)arity;
    for (uint32_t p = 0; p < arity; ++p) f->members[p] = idx(w, members[p]);
}

static const world *sorting_world;
static int factor_order(const void *a, const void *b) {
    const elpis_hca_factor *x = a, *y = b;
    if (x->relation_type != y->relation_type) return x->relation_type < y->relation_type ? -1 : 1;
    const uint32_t n = x->arity < y->arity ? x->arity : y->arity;
    for (uint32_t i = 0; i < n; ++i) {
        const int c = memcmp(sorting_world->cells[x->members[i]].key, sorting_world->cells[y->members[i]].key, 32);
        if (c) return c;
    }
    return (int)x->arity - (int)y->arity;
}
static int schema_order(const void *a, const void *b) {
    const elpis_hca_schema *x = a, *y = b;
    return x->relation_type < y->relation_type ? -1 : x->relation_type > y->relation_type;
}

/* Sort the schemas and factors into canonical order (after the last factor). */
static void seal_factors(world *w) {
    sorting_world = w;
    qsort(w->factors, w->n_factors, sizeof(elpis_hca_factor), factor_order);
    qsort(w->schemas, w->n_schemas, sizeof(elpis_hca_schema), schema_order);
}

static elpis_hca_limits big_limits(void) {
    elpis_hca_limits l;
    elpis_hca_limits_default(&l);
    l.cells = 256, l.factors = 512, l.arity = 16, l.degree = 32, l.rounds = 32;
    l.work = ELPIS_HCA_MAX_WORK, l.state_bytes = ELPIS_HCA_MAX_STATE_BYTES;
    return l;
}

static int build_world(const world *w, const elpis_hca_limits *l, elpis_hca_graph **g) {
    return elpis_hca_graph_create(l, w->cells, w->n_cells, w->schemas, w->n_schemas, w->factors, w->n_factors, l->work,
                                  g, NULL);
}

/* The coding fixture: two contradictory scopes (1 and 2), each a conjunction of `arity - 1` input kinds
 * (environment, dependency, policy (BINDING), extras) into one plan, under two schemas (71 CANDIDATE weight 750,
 * 72 ADMITTED weight 500). */
static void coding(world *w, uint32_t arity) {
    memset(w, 0, sizeof(*w));
    uint32_t kinds[ELPIS_HCA_MAX_ARITY];
    for (uint32_t p = 0; p < arity; ++p)
        kinds[p] = p == arity - 1 ? KIND_PLAN : p < 3 ? KIND_ENVIRONMENT + p : KIND_EXTRA + (p - 3);
    char label[64];
    for (uint32_t scope = 1; scope <= 2; ++scope)
        for (uint32_t p = 0; p < arity; ++p) {
            snprintf(label, sizeof label, "s%u:%u", scope, kinds[p]);
            add_cell(w, label, scope, kinds[p], kinds[p] == KIND_POLICY, ELPIS_HCA_SOURCE_VALID);
        }
    seal_cells(w);
    add_schema(w, 71, 750, arity, kinds);
    add_schema(w, 72, 500, arity, kinds);
    static char names[2][ELPIS_HCA_MAX_ARITY][64];
    const char *members[ELPIS_HCA_MAX_ARITY];
    for (uint32_t scope = 1; scope <= 2; ++scope) {
        for (uint32_t p = 0; p < arity; ++p) {
            snprintf(names[scope - 1][p], 64, "s%u:%u", scope, kinds[p]);
            members[p] = names[scope - 1][p];
        }
        add_factor(w, 71, ELPIS_HCA_CANDIDATE, arity, members);
        add_factor(w, 72, ELPIS_HCA_ADMITTED, arity, members);
    }
    seal_factors(w);
}

/* A task over one scope: seeds by label, every schema of the world admitted at both statuses, plan read out. */
typedef struct {
    elpis_hca_task task;
    uint8_t seeds[ELPIS_HCA_MAX_SEEDS][32];
    elpis_hca_predicate predicates[64];
    uint32_t output_kinds[8];
} task_buffer;

static void task_of(task_buffer *t, const world *w, uint32_t scope, const char *const *seeds, uint32_t n_seeds) {
    memset(t, 0, sizeof(*t));
    for (uint32_t i = 0; i < n_seeds; ++i) key_of(seeds[i], t->seeds[i]);
    for (uint32_t i = 0; i < w->n_schemas; ++i) {
        elpis_hca_predicate *p = &t->predicates[i];
        p->relation_type = w->schemas[i].relation_type;
        p->arity = w->schemas[i].arity;
        p->status_mask = ELPIS_HCA_MASK_CANDIDATE | ELPIS_HCA_MASK_ADMITTED;
        memcpy(p->roles, w->schemas[i].roles, sizeof(p->roles));
        memcpy(p->kinds, w->schemas[i].kinds, sizeof(p->kinds));
    }
    t->output_kinds[0] = KIND_PLAN;
    t->task = (elpis_hca_task){.scope = scope,
                               .seed_count = n_seeds,
                               .predicate_count = w->n_schemas,
                               .output_kind_count = 1,
                               .readout_budget = 32,
                               .steps = 8,
                               .updates_per_tick = 32,
                               .work_budget = 1000000,
                               .schedule_seed = 1729,
                               .seeds = (const uint8_t(*)[32])t->seeds,
                               .predicates = t->predicates,
                               .output_kinds = t->output_kinds};
}

/* The coding task of `scope`: every input kind seeded except those in `omit` (a bit per position). */
static void coding_task(task_buffer *t, const world *w, uint32_t arity, uint32_t scope, uint32_t omit) {
    static char names[ELPIS_HCA_MAX_ARITY][64];
    const char *seeds[ELPIS_HCA_MAX_ARITY];
    uint32_t n = 0;
    for (uint32_t p = 0; p + 1 < arity; ++p) {
        if (omit >> p & 1u) continue;
        const uint32_t kind = p < 3 ? KIND_ENVIRONMENT + p : KIND_EXTRA + (p - 3);
        snprintf(names[n], 64, "s%u:%u", scope, kind);
        seeds[n] = names[n];
        ++n;
    }
    task_of(t, w, scope, seeds, n);
}

#endif
