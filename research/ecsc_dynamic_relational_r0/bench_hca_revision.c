/*
 * HCA_REVISION_R0 bench (research only; PROTOCOL_HCA_REVISION_R0.md). DEV, NOT QUAL. SALIENCE_ONLY.
 *
 * ECS_C relational view (admitted evidence) -> elpis_ecs_c_rel_export -> research adapter -> native HCA -> readout.
 * One way: nothing any HCA call returns is written to the view, the authority rows or the memory.
 *
 * Methods after each event (an ECS_C source revision plus admissions): M-INC (exact incremental: view sync +
 * dependency delta + delete-and-rederive closure), M-FULL (view rebuild + export + closure from scratch), M-B0, M-B1,
 * M-B2 (export + adapter + graph + predict / ticks), M-B2-CARRY (the previous activation on the new session) and
 * M-STALE (the previous exact readout). Output: one JSON document on stdout.
 *
 * Build: research/ecsc_dynamic_relational_r0/build_bench.sh
 */
#define _POSIX_C_SOURCE 200809L
#include <time.h>

#include "elpis/ecs_c_hca.h"
#include "relational_test_util.h"

/* Allocator wrapping: the bytes of the HCA graph and session blocks (their only allocations). */
static int tracking;
static uint64_t tracked_bytes;
void *__real_calloc(size_t, size_t);
void *__wrap_calloc(size_t a, size_t b) {
    if (tracking) tracked_bytes += (uint64_t)a * b;
    return __real_calloc(a, b);
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

#define MAXE 256u
#define POOL 2048u
#define WEIGHT 900u
#define SEEDS 4u
#define SOURCES 16u
#define EVENTS 40u
#define REPLICATES 5u
#define MAX_TICKS 32u

static uint32_t decay(uint32_t x) { return WEIGHT * x / 1000u; }

/* ---- The relation pool (candidate binary relations) and its key index -------------------------------- */

typedef struct {
    uint32_t from, to;
    elpis_ecs_mem_digest key;
} edge_t;

static edge_t pool[POOL];
static uint32_t pool_n, by_key[POOL];
static uint32_t out_off[MAXE + 1], out_list[POOL], in_off[MAXE + 1], in_list[POOL];
static uint8_t entity_key[MAXE][32];

static int key_cmp(const void *a, const void *b) {
    return memcmp(pool[*(const uint32_t *)a].key.b, pool[*(const uint32_t *)b].key.b, 32);
}
static uint32_t edge_of(const elpis_ecs_mem_digest *k) {
    uint32_t lo = 0, hi = pool_n;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        const int c = memcmp(pool[by_key[mid]].key.b, k->b, 32);
        if (!c) return by_key[mid];
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    return UINT32_MAX;
}
static uint32_t entity_sorted[MAXE], entity_count;
static int entity_cmp(const void *a, const void *b) {
    return memcmp(entity_key[*(const uint32_t *)a], entity_key[*(const uint32_t *)b], 32);
}
static uint32_t entity_of(const uint8_t key[32], uint32_t e) {
    (void)e;
    uint32_t lo = 0, hi = entity_count;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        const int c = memcmp(entity_key[entity_sorted[mid]], key, 32);
        if (!c) return entity_sorted[mid];
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    return UINT32_MAX;
}
static void add_edge(uint32_t u, uint32_t v) {
    CHECK(pool_n < POOL);
    pool[pool_n].from = u;
    pool[pool_n].to = v;
    const elpis_ecs_mem_observation o = pair(5, ELPIS_ECS_MEM_ASSERT, 0, 0, u, v);
    CHECK(elpis_ecs_mem_relation_key(&o, &pool[pool_n].key) == ELPIS_ECS_MEM_OK);
    pool_n++;
}
static void index_pool(uint32_t e) {
    for (uint32_t i = 0; i < pool_n; ++i) by_key[i] = i;
    qsort(by_key, pool_n, sizeof(uint32_t), key_cmp);
    memset(out_off, 0, sizeof(out_off));
    memset(in_off, 0, sizeof(in_off));
    for (uint32_t i = 0; i < pool_n; ++i) out_off[pool[i].from + 1]++, in_off[pool[i].to + 1]++;
    for (uint32_t i = 0; i < e; ++i) out_off[i + 1] += out_off[i], in_off[i + 1] += in_off[i];
    uint32_t oc[MAXE + 1], ic[MAXE + 1];
    memcpy(oc, out_off, sizeof(oc));
    memcpy(ic, in_off, sizeof(ic));
    for (uint32_t i = 0; i < pool_n; ++i) out_list[oc[pool[i].from]++] = i, in_list[ic[pool[i].to]++] = i;
    for (uint32_t i = 0; i < e; ++i) {
        const elpis_ecs_mem_observation o = pair(5, ELPIS_ECS_MEM_ASSERT, 0, 0, i, 0);
        memcpy(entity_key[i], o.participants[0].entity.b, 32);
        entity_sorted[i] = i;
    }
    entity_count = e;
    qsort(entity_sorted, e, sizeof(uint32_t), entity_cmp);
}

/* ---- Exact closure: a max-heap Dijkstra over live edges (scores strictly decrease along a hop) ---------- */

typedef struct {
    uint32_t score, node;
} item;
static item heap[POOL * 4];
static uint32_t heap_n;
static void push(uint32_t score, uint32_t node) {
    CHECK(heap_n < POOL * 4);
    uint32_t i = heap_n++;
    heap[i].score = score, heap[i].node = node;
    while (i && heap[(i - 1) / 2].score < heap[i].score) {
        const item t = heap[i];
        heap[i] = heap[(i - 1) / 2];
        heap[(i - 1) / 2] = t;
        i = (i - 1) / 2;
    }
}
static item pop(void) {
    const item top = heap[0];
    heap[0] = heap[--heap_n];
    for (uint32_t i = 0;;) {
        uint32_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < heap_n && heap[l].score > heap[m].score) m = l;
        if (r < heap_n && heap[r].score > heap[m].score) m = r;
        if (m == i) break;
        const item t = heap[i];
        heap[i] = heap[m];
        heap[m] = t;
        i = m;
    }
    return top;
}
/* Propagate from the heap until stable (increases only). Returns the node + edge visits. */
static uint64_t propagate(uint32_t *score, const uint8_t *live) {
    uint64_t work = 0;
    while (heap_n) {
        const item it = pop();
        ++work;
        if (it.score != score[it.node]) continue;
        const uint32_t m = decay(it.score);
        if (!m) continue;
        for (uint32_t k = out_off[it.node]; k < out_off[it.node + 1]; ++k) {
            const uint32_t e = out_list[k];
            ++work;
            if (!live[e] || m <= score[pool[e].to]) continue;
            score[pool[e].to] = m;
            push(m, pool[e].to);
        }
    }
    return work;
}
static uint64_t closure(uint32_t *score, const uint8_t *live, uint32_t e) {
    memset(score, 0, sizeof(uint32_t) * e);
    heap_n = 0;
    for (uint32_t s = 0; s < SEEDS; ++s) score[s] = 1000, push(1000, s);
    return propagate(score, live) + e;
}
/* Delete-and-rederive: over-delete every score supported through a deleted edge, re-derive from intact
 * neighbours, then propagate the insertions. Exact (verified against B0 on every event). */
static uint8_t affected[MAXE];
static uint32_t stack_[MAXE];
static uint64_t dred(uint32_t *score, uint8_t *live, const uint32_t *del, uint32_t nd, const uint32_t *ins,
                     uint32_t ni) {
    uint64_t work = 0;
    uint32_t top = 0, naff = 0, list[MAXE];
    memset(affected, 0, sizeof(affected));
    for (uint32_t i = 0; i < nd; ++i) {
        const edge_t *x = &pool[del[i]];
        ++work;
        if (score[x->to] && x->to >= SEEDS && decay(score[x->from]) == score[x->to] && !affected[x->to])
            affected[x->to] = 1, stack_[top++] = x->to;
    }
    for (uint32_t i = 0; i < nd; ++i) live[del[i]] = 0;
    while (top) {
        const uint32_t v = stack_[--top];
        list[naff++] = v;
        const uint32_t m = decay(score[v]);
        for (uint32_t k = out_off[v]; k < out_off[v + 1]; ++k) {
            const uint32_t e = out_list[k];
            const uint32_t x = pool[e].to;
            ++work;
            if (live[e] && x >= SEEDS && !affected[x] && score[x] && m == score[x]) affected[x] = 1, stack_[top++] = x;
        }
    }
    heap_n = 0;
    for (uint32_t i = 0; i < naff; ++i) score[list[i]] = 0;
    for (uint32_t i = 0; i < naff; ++i) {
        const uint32_t v = list[i];
        uint32_t best = 0;
        for (uint32_t k = in_off[v]; k < in_off[v + 1]; ++k) {
            const uint32_t e = in_list[k];
            ++work;
            if (live[e] && !affected[pool[e].from] && decay(score[pool[e].from]) > best) best = decay(score[pool[e].from]);
        }
        if (best) score[v] = best, push(best, v);
    }
    work += propagate(score, live);
    for (uint32_t i = 0; i < ni; ++i) {
        const edge_t *x = &pool[ins[i]];
        live[ins[i]] = 1;
        ++work;
        const uint32_t m = decay(score[x->from]);
        if (m > score[x->to]) score[x->to] = m, push(m, x->to);
    }
    return work + propagate(score, live);
}

/* ---- The research adapter: an export -> HCA cells (canonical key order) and factors ---------------------- */

typedef struct {
    uint8_t key[32];
    uint32_t entity;
} cellrec;
static int cell_cmp(const void *a, const void *b) { return memcmp(((const cellrec *)a)->key, ((const cellrec *)b)->key, 32); }
static int factor_cmp(const void *a, const void *b) {
    const elpis_hca_factor *x = a, *y = b;
    if (x->members[0] != y->members[0]) return x->members[0] < y->members[0] ? -1 : 1;
    return x->members[1] < y->members[1] ? -1 : x->members[1] > y->members[1];
}

static elpis_ecs_mem_slot exported[POOL];
static cellrec cellrecs[MAXE];
static elpis_hca_cell cells[MAXE];
static elpis_hca_factor factors[ELPIS_HCA_MAX_FACTORS];
static uint32_t cell_entity[MAXE];

static uint32_t cell_index(uint32_t nc, const uint8_t key[32]) {
    uint32_t lo = 0, hi = nc;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        const int c = memcmp(cellrecs[mid].key, key, 32);
        if (!c) return mid;
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    return UINT32_MAX;
}

typedef struct {
    double seconds;
    uint64_t work;
} cost;

static elpis_hca_graph *build_graph(size_t n, uint32_t e, uint32_t *nc_out, cost *c) {
    const double t0 = now();
    uint8_t seen[MAXE] = {0};
    uint32_t nc = 0;
    for (uint32_t s = 0; s < SEEDS; ++s) seen[s] = 1;
    for (size_t i = 0; i < n; ++i)
        for (uint32_t p = 0; p < 2; ++p) {
            const uint32_t ent = entity_of(exported[i].participants[p].entity.b, e);
            CHECK(ent != UINT32_MAX);
            seen[ent] = 1;
        }
    for (uint32_t i = 0; i < e; ++i)
        if (seen[i]) memcpy(cellrecs[nc].key, entity_key[i], 32), cellrecs[nc++].entity = i;
    qsort(cellrecs, nc, sizeof(cellrec), cell_cmp);
    memset(cells, 0, sizeof(cells));
    for (uint32_t i = 0; i < nc; ++i) {
        memcpy(cells[i].key, cellrecs[i].key, 32);
        cells[i].scope = 0, cells[i].kind = 1, cells[i].source_state = ELPIS_HCA_SOURCE_VALID;
        cell_entity[i] = cellrecs[i].entity;
    }
    CHECK(n <= ELPIS_HCA_MAX_FACTORS);
    memset(factors, 0, sizeof(factors));
    for (size_t i = 0; i < n; ++i) {
        const elpis_ecs_mem_slot *s = &exported[i];
        factors[i].relation_type = 5;
        factors[i].status = ELPIS_HCA_CANDIDATE;
        factors[i].arity = 2;
        for (uint32_t p = 0; p < 2; ++p) {
            const uint32_t at = s->participants[p].role - 1u; /* canonical: role 1 (from) then role 2 (to) */
            factors[i].members[at] = cell_index(nc, s->participants[p].entity.b);
        }
    }
    qsort(factors, n, sizeof(elpis_hca_factor), factor_cmp);
    elpis_hca_schema schema;
    memset(&schema, 0, sizeof(schema));
    schema.relation_type = 5, schema.arity = 2, schema.weight = WEIGHT;
    schema.roles[0] = 1, schema.roles[1] = 2, schema.kinds[0] = schema.kinds[1] = 1;
    elpis_hca_limits l;
    elpis_hca_limits_default(&l);
    l.cells = ELPIS_HCA_MAX_CELLS, l.factors = ELPIS_HCA_MAX_FACTORS, l.arity = 2, l.degree = ELPIS_HCA_MAX_DEGREE;
    l.rounds = ELPIS_HCA_MAX_ROUNDS, l.updates_per_tick = ELPIS_HCA_MAX_UPDATES, l.work = ELPIS_HCA_MAX_WORK;
    l.state_bytes = ELPIS_HCA_MAX_STATE_BYTES;
    elpis_hca_graph *g = NULL;
    uint64_t used = 0;
    const int rc = elpis_hca_graph_create(&l, cells, nc, &schema, 1, factors, n, ELPIS_HCA_MAX_WORK, &g, &used);
    if (rc) fprintf(stderr, "graph_create: %s\n", elpis_hca_code_name(rc));
    CHECK(rc == ELPIS_HCA_OK);
    c->seconds += now() - t0;
    c->work += used;
    *nc_out = nc;
    return g;
}

static uint8_t seed_keys[SEEDS][32];
static elpis_hca_predicate predicate;
static const uint32_t output_kind = 1;
static elpis_hca_task make_task(void) {
    elpis_hca_task t;
    memset(&t, 0, sizeof(t));
    t.scope = 0, t.seed_count = SEEDS, t.predicate_count = 1, t.output_kind_count = 1;
    t.readout_budget = ELPIS_HCA_MAX_READOUT, t.steps = ELPIS_HCA_MAX_ROUNDS, t.updates_per_tick = ELPIS_HCA_MAX_UPDATES;
    t.work_budget = ELPIS_HCA_MAX_WORK, t.schedule_seed = 20261024;
    for (uint32_t s = 0; s < SEEDS; ++s) memcpy(seed_keys[s], entity_key[s], 32);
    memset(&predicate, 0, sizeof(predicate));
    predicate.relation_type = 5, predicate.arity = 2, predicate.status_mask = ELPIS_HCA_MASK_CANDIDATE | ELPIS_HCA_MASK_ADMITTED;
    predicate.roles[0] = 1, predicate.roles[1] = 2, predicate.kinds[0] = predicate.kinds[1] = 1;
    t.seeds = (const uint8_t(*)[32])seed_keys;
    t.predicates = &predicate;
    t.output_kinds = &output_kind;
    return t;
}

/* ---- One run of a regime ------------------------------------------------------------------------------ */

enum { M_INC, M_FULL, M_B0, M_B1, M_B2, M_SYNC, M_REBUILD, NM };
static const char *const method_name[NM] = {"M-INC", "M-FULL", "M-B0", "M-B1", "M-B2", "view_sync", "view_rebuild"};

typedef struct {
    double seconds[EVENTS][NM];
    uint64_t work[EVENTS][NM];
    uint32_t exact[NM], events;
    uint32_t b2_ticks[EVENTS], b2_converged, carry_refused, carry_attempts, b1_exact;
    double acc_stale[EVENTS], acc_b2_best_before_inc[EVENTS];
    uint32_t h4_events;
    uint64_t delta[EVENTS];
    uint64_t graph_bytes, session_bytes, inc_bytes, view_arena;
    uint32_t cells_max, factors_max;
} run_t;

static elpis_hca_result result;

static double accuracy(const uint32_t *a, const uint32_t *b, uint32_t e) {
    uint32_t same = 0;
    for (uint32_t i = 0; i < e; ++i) same += a[i] == b[i];
    return (double)same / e;
}

/* local: R-LOCAL (scale = unrelated relations); else R-RANDOM (scale = entities). */
static void run_regime(int local, uint32_t scale, uint64_t seed, run_t *out) {
    memset(out, 0, sizeof(*out));
    rng_state = seed * 7919u + scale;
    pool_n = 0;
    const uint32_t e = local ? MAXE : scale;
    uint32_t target = local ? 0 : (2 * scale < 448 ? 2 * scale : 448);
    /* Pool, with every cell's incidences over the whole pool within the HCA degree limit. */
    uint32_t inc[MAXE] = {0};
    static uint8_t used_pair[MAXE][MAXE];
    memset(used_pair, 0, sizeof(used_pair));
    if (local) {
        const uint32_t chain[8][2] = {{0, 4}, {4, 5}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10}, {10, 11}};
        for (uint32_t i = 0; i < 8; ++i) add_edge(chain[i][0], chain[i][1]), used_pair[chain[i][0]][chain[i][1]] = 1;
        for (uint32_t i = 0; i < 8; ++i) inc[chain[i][0]]++, inc[chain[i][1]]++;
        uint32_t tries = 0;
        while (pool_n < 8 + scale && tries++ < 1000000) {
            const uint32_t u = below(5) == 0 ? below(SEEDS) : 12 + below(e - 12), v = 12 + below(e - 12);
            if (u == v || used_pair[u][v] || inc[u] >= 30 || inc[v] >= 30) continue;
            used_pair[u][v] = 1, inc[u]++, inc[v]++;
            add_edge(u, v);
        }
        CHECK(pool_n == 8 + scale);
    } else {
        uint32_t tries = 0;
        while (pool_n < 3 * target && tries++ < 1000000) {
            const uint32_t u = below(e), v = below(e);
            if (u == v || v < SEEDS || used_pair[u][v] || inc[u] >= 30 || inc[v] >= 30) continue;
            used_pair[u][v] = 1, inc[u]++, inc[v]++;
            add_edge(u, v);
        }
    }
    index_pool(e);

    /* The authority: two founders and 16 sources in one scope; the view synced to it. */
    world w;
    w_init(&w, (uint32_t)(seed * 1000 + scale + (local ? 500 : 0)));
    uint32_t src[SOURCES];
    for (uint32_t s = 0; s < SOURCES; ++s) src[s] = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 100 + s, 1, 1);
    elpis_ecs_c_rel *v = make_view(32, 4096, 8192);
    OK(w_sync(v, &w, NULL));
    elpis_ecs_c_rel_status st;
    OK(elpis_ecs_c_rel_read_status(v, &st));
    out->view_arena = st.arena_bytes;
    static uint8_t live[POOL];
    memset(live, 0, sizeof(live));
    uint64_t tick = 1;
    uint32_t live_n = 0;
#define ADMIT(edge, row)                                                                                       \
    do {                                                                                                       \
        const elpis_ecs_mem_observation o_ = pair(5, ELPIS_ECS_MEM_ASSERT, tick++,                             \
                                                  w.rows[row].source_authority, pool[edge].from, pool[edge].to); \
        OK(admit_as(v, &w, row, &o_, NULL));                                                                   \
    } while (0)
    if (local) {
        for (uint32_t i = 0; i < 8; ++i) {
            ADMIT(i, src[0]);
            live[i] = 1;
        }
        for (uint32_t i = 8; i < pool_n; ++i) {
            ADMIT(i, src[1 + below(SOURCES - 1)]);
            live[i] = 1;
        }
        live_n = pool_n;
    } else {
        for (uint32_t i = 0; i < target; ++i) {
            const uint32_t s1 = below(SOURCES);
            ADMIT(i, src[s1]);
            if (below(10) < 3) ADMIT(i, src[(s1 + 1 + below(SOURCES - 1)) % SOURCES]);
            live[i] = 1;
        }
        live_n = target;
    }
    static uint32_t score[MAXE], ref[MAXE], prev[MAXE], full[MAXE], hscore[MAXE];
    (void)closure(score, live, e);
    elpis_hca_graph *prev_graph = NULL;
    elpis_hca_b2 *prev_session = NULL;
    elpis_hca_activation prev_act;
    const elpis_hca_task task = make_task();
    static elpis_ecs_mem_digest deps[POOL];
    static uint32_t del[POOL], ins[POOL];

    for (uint32_t ev = 0; ev < EVENTS; ++ev) {
        memcpy(prev, score, sizeof(uint32_t) * e);
        const uint32_t s = local ? src[0] : src[below(SOURCES)];
        const int readmit_phase = local && (ev % 2 == 1);
        size_t nd = 0;
        uint32_t ni = 0, ndel = 0;
        double t0, inc_seconds = 0;
        uint64_t inc_work = 0;
        if (!readmit_phase) {
            /* M-INC part 1: the revised source's dependents, before the event. */
            t0 = now();
            OK(elpis_ecs_c_rel_dependents(v, w.rows[s].id, deps, POOL, &nd));
            inc_seconds += now() - t0;
            inc_work += nd;
            /* The event: SOURCE_VERSION (retires the source's evidence), then the view's sync. */
            w_version(&w, s, w.rows[s].source_version + 1, 1);
            elpis_ecs_c_rel_sync_result r;
            t0 = now();
            OK(w_sync(v, &w, &r));
            out->seconds[ev][M_SYNC] = now() - t0;
            out->work[ev][M_SYNC] = r.work.rows + r.work.index_probes + r.work.contributions + r.work.relations + r.work.recomputes;
            /* Re-admission of half the source's relations under its new version (R-RANDOM). */
            if (!local)
                for (size_t i = 0; i < nd; ++i)
                    if (below(2)) ADMIT(edge_of(&deps[i]), s);
        } else {
            /* R-LOCAL even events: the designated source re-admits its 8 relations under its current version. */
            elpis_ecs_c_rel_sync_result r;
            w_bump(&w);
            t0 = now();
            OK(w_sync(v, &w, &r));
            out->seconds[ev][M_SYNC] = now() - t0;
            out->work[ev][M_SYNC] = r.work.rows + r.work.index_probes + r.work.contributions + r.work.relations + r.work.recomputes;
            for (uint32_t i = 0; i < 8; ++i) {
                ADMIT(i, s);
                ins[ni++] = i;
            }
        }
        /* M-INC part 2: which dependents are no longer live (reads of the view). */
        t0 = now();
        for (size_t i = 0; i < nd; ++i) {
            elpis_ecs_mem_slot slot;
            const int rc = elpis_ecs_c_rel_relation_read(v, NULL, NULL, &deps[i], &slot);
            CHECK(rc == ELPIS_ECS_C_REL_OK || rc == ELPIS_ECS_C_REL_NOT_FOUND);
            if (rc == ELPIS_ECS_C_REL_NOT_FOUND) del[ndel++] = edge_of(&deps[i]);
        }
        inc_seconds += now() - t0;
        inc_work += nd;
        live_n -= ndel;
        /* Fresh relations by other sources, back to the pre-event live count (R-RANDOM). */
        if (!local) {
            for (uint32_t tries = 0; live_n + ni < target && tries < 100000; ++tries) {
                const uint32_t x = below(pool_n);
                int pending = live[x];
                for (uint32_t j = 0; j < ndel && !pending; ++j) pending = del[j] == x;
                for (uint32_t j = 0; j < ni && !pending; ++j) pending = ins[j] == x;
                if (pending) continue;
                uint32_t other = src[below(SOURCES)];
                if (other == s) other = src[(below(SOURCES - 1) + 1 + (s - src[0])) % SOURCES];
                ADMIT(x, other);
                ins[ni++] = x;
            }
        }
        live_n += ni;
        out->delta[ev] = ndel + ni;
        /* M-INC part 3: delete-and-rederive over the retained adjacency. */
        t0 = now();
        inc_work += dred(score, live, del, ndel, ins, ni);
        inc_seconds += now() - t0;
        out->seconds[ev][M_INC] = inc_seconds;
        out->work[ev][M_INC] = inc_work;

        /* M-FULL: rebuild every derived value, export, closure from scratch. */
        elpis_ecs_c_rel_work rw;
        memset(&rw, 0, sizeof(rw));
        t0 = now();
        OK(elpis_ecs_c_rel_rebuild(v, &rw));
        out->seconds[ev][M_REBUILD] = now() - t0;
        out->work[ev][M_REBUILD] = rw.contributions + rw.relations + rw.recomputes;
        t0 = now();
        size_t n = 0;
        OK(elpis_ecs_c_rel_export(v, NULL, NULL, exported, POOL, &n, NULL));
        static uint8_t live_full[POOL];
        memset(live_full, 0, sizeof(live_full));
        for (size_t i = 0; i < n; ++i) live_full[edge_of(&exported[i].key)] = 1;
        const uint64_t fw = closure(full, live_full, e);
        out->seconds[ev][M_FULL] = out->seconds[ev][M_REBUILD] + (now() - t0);
        out->work[ev][M_FULL] = out->work[ev][M_REBUILD] + n + fw;
        CHECK(n == live_n);
        for (uint32_t i = 0; i < pool_n; ++i) CHECK(live_full[i] == live[i]); /* the bench's mirror is the view's */

        /* M-B0: export, adapter, graph, predict. */
        cost base = {0, 0};
        t0 = now();
        OK(elpis_ecs_c_rel_export(v, NULL, NULL, exported, POOL, &n, NULL));
        base.seconds += now() - t0;
        base.work += n;
        uint32_t nc = 0;
        tracking = 1, tracked_bytes = 0;
        elpis_hca_graph *g = build_graph(n, e, &nc, &base);
        tracking = 0;
        if (tracked_bytes > out->graph_bytes) out->graph_bytes = tracked_bytes;
        if (nc > out->cells_max) out->cells_max = nc;
        if (n > out->factors_max) out->factors_max = (uint32_t)n;
        uint16_t hs[MAXE];
        uint64_t used = 0;
        t0 = now();
        CHECK(elpis_hca_predict(g, &task, ELPIS_HCA_B0, &result, hs, &used) == ELPIS_HCA_OK);
        out->seconds[ev][M_B0] = base.seconds + (now() - t0);
        out->work[ev][M_B0] = base.work + used;
        memset(ref, 0, sizeof(uint32_t) * e);
        for (uint32_t i = 0; i < nc; ++i) ref[cell_entity[i]] = hs[i];
        /* Mechanics gates: B0 equals the independent closure; M-INC and M-FULL are exact. */
        CHECK(!memcmp(ref, full, sizeof(uint32_t) * e));
        out->exact[M_B0] += 1;
        out->exact[M_FULL] += 1;
        out->exact[M_INC] += !memcmp(ref, score, sizeof(uint32_t) * e);
        CHECK(!memcmp(ref, score, sizeof(uint32_t) * e));
        /* M-B1. */
        t0 = now();
        CHECK(elpis_hca_predict(g, &task, ELPIS_HCA_B1, &result, hs, &used) == ELPIS_HCA_OK);
        out->seconds[ev][M_B1] = base.seconds + (now() - t0);
        out->work[ev][M_B1] = base.work + used;
        memset(hscore, 0, sizeof(uint32_t) * e);
        for (uint32_t i = 0; i < nc; ++i) hscore[cell_entity[i]] = hs[i];
        const int b1 = !memcmp(ref, hscore, sizeof(uint32_t) * e);
        out->exact[M_B1] += (uint32_t)b1;
        /* M-B2: cold start, then ticks until the activation equals the reference. */
        elpis_hca_b2 *session = NULL;
        elpis_hca_activation act, next;
        double b2s = base.seconds;
        uint64_t b2w = base.work;
        tracking = 1, tracked_bytes = 0;
        t0 = now();
        CHECK(elpis_hca_b2_start(g, &task, 0, &session, &act, &used) == ELPIS_HCA_OK);
        b2s += now() - t0;
        tracking = 0;
        if (tracked_bytes > out->session_bytes) out->session_bytes = tracked_bytes;
        b2w += used;
        uint32_t k = 0;
        double best_partial = 0;
        for (;; ++k) {
            /* The instrument (not charged): this tick's activation against the reference. */
            memset(hscore, 0, sizeof(uint32_t) * e);
            for (uint32_t i = 0; i < nc; ++i) hscore[cell_entity[i]] = (uint32_t)act.values[2 * i] << 8 | act.values[2 * i + 1];
            if (!memcmp(ref, hscore, sizeof(uint32_t) * e) || k == MAX_TICKS) break;
            const double acc = accuracy(hscore, ref, e);
            if (b2s < inc_seconds + out->seconds[ev][M_SYNC] && acc > best_partial) best_partial = acc;
            t0 = now();
            CHECK(elpis_hca_b2_warm(session, &act, 0, &next, &used) == ELPIS_HCA_OK);
            b2s += now() - t0;
            b2w += used;
            act = next;
        }
        t0 = now();
        CHECK(elpis_hca_b2_observe(session, &act, 0, &result, &used) == ELPIS_HCA_OK);
        b2s += now() - t0;
        b2w += used;
        out->seconds[ev][M_B2] = b2s;
        out->work[ev][M_B2] = b2w;
        out->b2_ticks[ev] = k;
        const int b2_exact = !memcmp(ref, hscore, sizeof(uint32_t) * e);
        out->exact[M_B2] += (uint32_t)b2_exact;
        out->b2_converged += (uint32_t)b2_exact;
        /* H4: a partial B2 readout cheaper than M-INC's exact update and better than the stale readout. */
        out->acc_stale[ev] = accuracy(prev, ref, e);
        out->acc_b2_best_before_inc[ev] = best_partial;
        out->h4_events += best_partial > out->acc_stale[ev];
        /* M-B2-CARRY: the previous event's activation on this event's session. */
        if (prev_session) {
            out->carry_attempts++;
            const int rc = elpis_hca_b2_warm(session, &prev_act, 0, &next, &used);
            out->carry_refused += rc == ELPIS_HCA_INVALID_STATE;
            const int rc2 = elpis_hca_b2_observe(session, &prev_act, 0, &result, &used);
            CHECK((rc == ELPIS_HCA_INVALID_STATE) == (rc2 == ELPIS_HCA_INVALID_STATE));
            elpis_hca_b2_destroy(prev_session);
            elpis_hca_graph_destroy(prev_graph);
        }
        prev_session = session, prev_graph = g, prev_act = act;
        out->events++;
    }
    elpis_hca_b2_destroy(prev_session);
    elpis_hca_graph_destroy(prev_graph);
    /* M-INC's own retained state (beyond the view every method uses): scores, live flags, adjacency. */
    out->inc_bytes = sizeof(uint32_t) * e + pool_n + sizeof(uint32_t) * (2u * (e + 1u) + 2u * pool_n);
    elpis_ecs_c_rel_destroy(v);
#undef ADMIT
}

/* ---- Replicates, medians, JSON ------------------------------------------------------------------------ */

static int dcmp(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
static double median(double *x, uint32_t n) {
    qsort(x, n, sizeof(double), dcmp);
    return n % 2 ? x[n / 2] : 0.5 * (x[n / 2 - 1] + x[n / 2]);
}

static run_t runs[REPLICATES];

static void regime(int local, uint32_t scale, uint64_t seed, int first) {
    for (uint32_t r = 0; r < REPLICATES; ++r) run_regime(local, scale, seed, &runs[r]);
    const run_t *a = &runs[0];
    for (uint32_t r = 1; r < REPLICATES; ++r) /* determinism: every logical count repeats exactly */
        CHECK(!memcmp(runs[r].work, a->work, sizeof(a->work)) && !memcmp(runs[r].exact, a->exact, sizeof(a->exact)) &&
              !memcmp(runs[r].b2_ticks, a->b2_ticks, sizeof(a->b2_ticks)));
    double med[NM], total[NM];
    uint64_t work[NM];
    for (int m = 0; m < NM; ++m) {
        double per_event[EVENTS];
        total[m] = 0, work[m] = 0;
        for (uint32_t ev = 0; ev < a->events; ++ev) {
            double reps[REPLICATES];
            for (uint32_t r = 0; r < REPLICATES; ++r) reps[r] = runs[r].seconds[ev][m];
            per_event[ev] = median(reps, REPLICATES);
            total[m] += per_event[ev];
            work[m] += a->work[ev][m];
        }
        med[m] = median(per_event, a->events);
    }
    double acc_stale = 0, acc_b2 = 0, ticks = 0, delta = 0;
    uint32_t tick_max = 0;
    for (uint32_t ev = 0; ev < a->events; ++ev) {
        acc_stale += a->acc_stale[ev], acc_b2 += a->acc_b2_best_before_inc[ev], ticks += a->b2_ticks[ev];
        delta += (double)a->delta[ev];
        if (a->b2_ticks[ev] > tick_max) tick_max = a->b2_ticks[ev];
    }
    const double n = a->events;
    printf("%s\n  {\"regime\": \"%s\", \"scale\": %u, \"seed\": %llu, \"events\": %u, \"replicates\": %u,\n",
           first ? "" : ",", local ? "R-LOCAL" : "R-RANDOM", scale, (unsigned long long)seed, a->events, REPLICATES);
    printf("   \"cells_max\": %u, \"factors_max\": %u, \"mean_changed_relations\": %.2f,\n", a->cells_max, a->factors_max,
           delta / n);
    printf("   \"exact_events\": {");
    for (int m = 0; m < M_SYNC; ++m) printf("%s\"%s\": %u", m ? ", " : "", method_name[m], a->exact[m]);
    printf("},\n   \"median_event_us\": {");
    for (int m = 0; m < NM; ++m) printf("%s\"%s\": %.2f", m ? ", " : "", method_name[m], med[m] * 1e6);
    printf("},\n   \"total_us\": {");
    for (int m = 0; m < NM; ++m) printf("%s\"%s\": %.1f", m ? ", " : "", method_name[m], total[m] * 1e6);
    printf("},\n   \"mean_logical_work\": {");
    for (int m = 0; m < NM; ++m) printf("%s\"%s\": %.1f", m ? ", " : "", method_name[m], (double)work[m] / n);
    printf("},\n   \"b2\": {\"mean_ticks_to_exact\": %.2f, \"max_ticks\": %u, \"converged\": %u, "
           "\"carry_attempts\": %u, \"carry_refused_invalid_state\": %u},\n",
           ticks / n, tick_max, a->b2_converged, a->carry_attempts, a->carry_refused);
    printf("   \"h4\": {\"mean_accuracy_stale\": %.4f, \"mean_best_b2_partial_accuracy_cheaper_than_inc\": %.4f, "
           "\"events_where_b2_partial_beats_stale_cheaper_than_inc\": %u},\n",
           acc_stale / n, acc_b2 / n, a->h4_events);
    printf("   \"retained_bytes\": {\"view_arena_all_methods\": %llu, \"M-INC_closure\": %llu, \"hca_graph_max\": %llu, "
           "\"hca_session_max\": %llu, \"hca_activation\": %zu}}",
           (unsigned long long)a->view_arena, (unsigned long long)a->inc_bytes, (unsigned long long)a->graph_bytes,
           (unsigned long long)a->session_bytes, sizeof(elpis_hca_activation));
    fflush(stdout);
}

int main(void) {
    printf("{\"schema\": \"elpis.ecsc-dynamic-relational-r0.hca-revision.v1\", \"protocol\": "
           "\"research/ecsc_dynamic_relational_r0/PROTOCOL_HCA_REVISION_R0.md\",\n \"classification\": \"DEV, NOT QUAL; "
           "SALIENCE_ONLY; B3 NOT_AUTHORIZED\", \"weight\": %u, \"seeds\": %u, \"sources\": %u, \"events_per_run\": %u,\n"
           " \"cases\": [",
           WEIGHT, SEEDS, SOURCES, EVENTS);
    int first = 1;
    const uint32_t local_scales[5] = {32, 64, 128, 256, 440};
    for (uint32_t i = 0; i < 5; ++i, first = 0) regime(1, local_scales[i], 1, first);
    const uint32_t random_scales[3] = {64, 128, 256};
    for (uint32_t i = 0; i < 3; ++i)
        for (uint64_t seed = 1; seed <= 3; ++seed) regime(0, random_scales[i], seed, 0);
    printf("\n ]}\n");
    return 0;
}
