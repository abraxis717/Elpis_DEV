/* Differential test: ECS Memory MF-R0 against an independent reference written for clarity, not speed.
 *
 * The reference keeps residency in plain arrays (linear search, move-to-end recency, explicit victim walk), and
 * applies the learning laws with int8 vectors and its own rounding formula. It shares only the fixed encoding
 * (ecs_mem_entity_code, ecs_mem_cue) and the relation key, which are mechanical definitions, not the law.
 * Random operation sequences (seeded, deterministic) are applied to both; every outcome code, residency set,
 * LOOKUP answer and FAST/SLOW score must agree exactly. */
#include "ecs_mem_test_util.h"

#include <stdint.h>

#define RMAX 64

typedef struct {
    ecs_mem_digest key;
    ecs_mem_observation first; /* canonical participants come from the core's slot view */
    ecs_mem_slot view;         /* the reference's own bookkeeping, laid out like the public record */
} ref_slot;

typedef struct {
    ecs_mem_config cfg;
    uint64_t clock;
    uint32_t cursor;
    uint32_t nslots, nents;
    ref_slot slots[RMAX];         /* by slot index (same index discipline as the core: append, reuse victim) */
    uint32_t slot_lru[RMAX];      /* least recent first */
    ecs_mem_digest ents[RMAX];
    uint32_t ent_lru[RMAX];
    int32_t *fast, *slow;
} ref_mem;

static uint64_t rng_state;
static unsigned long outcome[3][12]; /* [observe, learn, consolidate][-status] coverage */
static uint64_t rnd(void) {
    uint64_t z = (rng_state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
static int64_t round_away(int64_t n, int64_t d) {
    const int64_t sign = n < 0 ? -1 : 1;
    const int64_t a = n < 0 ? -n : n;
    return sign * ((2 * a + d) / (2 * d));
}
static void lru_move_to_end(uint32_t *lru, uint32_t n, uint32_t v) {
    uint32_t i = 0;
    while (lru[i] != v) ++i;
    for (; i + 1 < n; ++i) lru[i] = lru[i + 1];
    lru[n - 1] = v;
}
static void lru_remove(uint32_t *lru, uint32_t n, uint32_t v) {
    uint32_t i = 0;
    while (lru[i] != v) ++i;
    for (; i + 1 < n; ++i) lru[i] = lru[i + 1];
}
static int find_slot(const ref_mem *r, const ecs_mem_digest *k) {
    for (uint32_t i = 0; i < r->nslots; ++i)
        if (tu_same(r->slots[i].key, *k)) return (int)i;
    return -1;
}
static int find_ent(const ref_mem *r, const ecs_mem_digest *e) {
    for (uint32_t i = 0; i < r->nents; ++i)
        if (tu_same(r->ents[i], *e)) return (int)i;
    return -1;
}

static ecs_mem_status ref_observe(ref_mem *r, const ecs_mem_observation *o, const ecs_mem_slot *canonical) {
    if (o->tick < r->clock) return ECS_MEM_CLOCK;
    ecs_mem_digest key;
    assert(ecs_mem_relation_key(o, &key) == ECS_MEM_OK);
    int s = find_slot(r, &key);
    int victim = -1;
    if (s < 0 && r->nslots == r->cfg.slot_capacity) {
        if (r->cfg.slot_policy == ECS_MEM_REFUSE) return ECS_MEM_CAPACITY;
        victim = (int)r->slot_lru[0];
    }
    /* distinct entities in canonical participant order */
    ecs_mem_digest uniq[8];
    int present[8];
    uint32_t nu = 0, missing = 0;
    for (uint32_t i = 0; i < canonical->participant_count; ++i) {
        int dup = 0;
        for (uint32_t j = 0; j < nu; ++j) dup |= tu_same(uniq[j], canonical->participants[i].entity);
        if (dup) continue;
        uniq[nu] = canonical->participants[i].entity;
        present[nu] = find_ent(r, &uniq[nu]);
        missing += present[nu] < 0;
        ++nu;
    }
    const uint32_t free_entries = r->cfg.codebook_capacity - r->nents;
    const uint32_t need = missing > free_entries ? missing - free_entries : 0;
    uint32_t victims[8], nv = 0;
    if (need && r->cfg.codebook_policy == ECS_MEM_REFUSE) return ECS_MEM_CAPACITY;
    for (uint32_t i = 0; i < r->nents && nv < need; ++i) {
        int is_present = 0;
        for (uint32_t j = 0; j < nu; ++j) is_present |= present[j] == (int)r->ent_lru[i];
        if (!is_present) victims[nv++] = r->ent_lru[i];
    }
    if (nv < need) return ECS_MEM_CAPACITY;
    /* commit */
    if (s < 0) {
        if (victim >= 0) {
            lru_remove(r->slot_lru, r->nslots, (uint32_t)victim);
            s = victim;
            r->slot_lru[r->nslots - 1] = (uint32_t)s;
        } else {
            s = (int)r->nslots++;
            r->slot_lru[s] = (uint32_t)s;
        }
        memset(&r->slots[s], 0, sizeof(r->slots[s]));
        r->slots[s].key = key;
        r->slots[s].view = *canonical;
        ecs_mem_slot *v = &r->slots[s].view;
        v->assert_count = v->negate_count = v->uncertain_count = 0;
    } else {
        lru_move_to_end(r->slot_lru, r->nslots, (uint32_t)s);
    }
    ecs_mem_slot *v = &r->slots[s].view;
    if (o->polarity == ECS_MEM_ASSERT) v->assert_count++;
    else if (o->polarity == ECS_MEM_NEGATE) v->negate_count++;
    else v->uncertain_count++;
    if (o->polarity == ECS_MEM_ASSERT) v->last_assert_tick = o->tick;
    uint32_t next = 0;
    for (uint32_t j = 0; j < nu; ++j) {
        if (present[j] >= 0) {
            lru_move_to_end(r->ent_lru, r->nents, (uint32_t)present[j]);
            continue;
        }
        if (r->nents < r->cfg.codebook_capacity) {
            r->ents[r->nents] = uniq[j];
            r->ent_lru[r->nents] = r->nents;
            r->nents++;
        } else {
            const uint32_t e = victims[next++];
            r->ents[e] = uniq[j];
            lru_move_to_end(r->ent_lru, r->nents, e);
        }
    }
    r->clock = o->tick;
    return ECS_MEM_OK;
}

/* One law step written directly from the definition with int8 vectors. */
static void ref_step(int32_t *M, uint32_t D, const int8_t *c, const int8_t *t, uint32_t rate, uint32_t retain,
                     uint32_t law) {
    for (uint32_t i = 0; i < D; ++i) {
        int32_t *row = M + (size_t)i * D;
        const int64_t ti = t ? 65536 * (int64_t)t[i] : 0;
        if (retain != 65536)
            for (uint32_t j = 0; j < D; ++j) row[j] = (int32_t)round_away((int64_t)retain * row[j], 65536);
        int64_t d;
        if (law == ECS_MEM_LAW_DELTA) {
            int64_t s = 0;
            for (uint32_t j = 0; j < D; ++j) s += (int64_t)row[j] * c[j];
            d = round_away((int64_t)rate * (ti - round_away(s, D)), 65536);
        } else {
            d = round_away((int64_t)rate * ti, 65536);
        }
        for (uint32_t j = 0; j < D; ++j) row[j] = (int32_t)(row[j] + d * c[j]);
    }
}
static void direction(const ecs_mem_slot *v, uint32_t t, ecs_mem_request *q) {
    memset(q, 0, sizeof(*q));
    q->abi_version = ECS_MEM_ABI_V1;
    q->relation_type = v->relation_type;
    q->target_role = v->participants[t].role;
    q->target_ordinal = v->participants[t].ordinal;
    for (uint32_t i = 0; i < v->participant_count; ++i)
        if (i != t) q->cue[q->cue_count++] = v->participants[i];
}
static int64_t maxabs(const int32_t *M, size_t n) {
    int64_t b = 0;
    for (size_t i = 0; i < n; ++i) b = M[i] < 0 ? (-(int64_t)M[i] > b ? -(int64_t)M[i] : b) : (M[i] > b ? M[i] : b);
    return b;
}
static void replay_slot(ref_mem *r, int32_t *banks, const ecs_mem_slot *v, uint32_t rate, uint32_t retain) {
    const uint32_t D = r->cfg.dim;
    const int positive = (int64_t)v->assert_count - v->negate_count > 0;
    for (uint32_t t = 0; t < v->participant_count; ++t) {
        ecs_mem_request q;
        int8_t c[ECS_MEM_MAX_DIM], code[ECS_MEM_MAX_DIM];
        uint32_t bank;
        direction(v, t, &q);
        assert(ecs_mem_cue(&q, D, r->cfg.banks, c, &bank) == ECS_MEM_OK);
        assert(ecs_mem_entity_code(&v->participants[t].entity, D, code) == ECS_MEM_OK);
        ref_step(banks + (size_t)bank * D * D, D, c, positive ? code : NULL, rate, retain, r->cfg.law);
    }
}
static ecs_mem_status ref_learn(ref_mem *r, const ecs_mem_digest *key, uint64_t tick) {
    if (tick < r->clock) return ECS_MEM_CLOCK;
    const int s = find_slot(r, key);
    if (s < 0) return ECS_MEM_NOT_FOUND;
    const ecs_mem_slot *v = &r->slots[s].view;
    if (v->assert_count == v->negate_count) return ECS_MEM_NO_NET_EVIDENCE;
    const size_t bank_n = (size_t)r->cfg.dim * r->cfg.dim;
    for (uint32_t t = 0; t < v->participant_count; ++t) {
        ecs_mem_request q;
        int8_t c[ECS_MEM_MAX_DIM];
        uint32_t bank;
        direction(v, t, &q);
        assert(ecs_mem_cue(&q, r->cfg.dim, r->cfg.banks, c, &bank) == ECS_MEM_OK);
        if (maxabs(r->fast + bank * bank_n, bank_n) > ECS_MEM_LEARN_WEIGHT_LIMIT) return ECS_MEM_NUMERIC;
    }
    replay_slot(r, r->fast, v, r->cfg.learn_rate_q16, r->cfg.retain_q16);
    lru_move_to_end(r->slot_lru, r->nslots, (uint32_t)s);
    r->clock = tick;
    return ECS_MEM_OK;
}
static ecs_mem_status ref_consolidate(ref_mem *r, uint64_t tick, uint32_t max_items, uint32_t *replayed) {
    if (tick < r->clock) return ECS_MEM_CLOCK;
    uint32_t items[16], ni = 0, cursor = r->cursor;
    for (uint32_t visited = 0; visited < r->nslots && ni < max_items; ++visited) {
        const ecs_mem_slot *v = &r->slots[cursor].view;
        if (v->assert_count != v->negate_count) items[ni++] = cursor;
        cursor = (cursor + 1u) % r->nslots;
    }
    const size_t bank_n = (size_t)r->cfg.dim * r->cfg.dim;
    for (uint32_t k = 0; k < ni; ++k)
        for (uint32_t t = 0; t < r->slots[items[k]].view.participant_count; ++t) {
            ecs_mem_request q;
            int8_t c[ECS_MEM_MAX_DIM];
            uint32_t bank;
            direction(&r->slots[items[k]].view, t, &q);
            assert(ecs_mem_cue(&q, r->cfg.dim, r->cfg.banks, c, &bank) == ECS_MEM_OK);
            if (maxabs(r->slow + bank * bank_n, bank_n) > ECS_MEM_LEARN_WEIGHT_LIMIT) return ECS_MEM_NUMERIC;
        }
    for (uint32_t k = 0; k < ni; ++k)
        replay_slot(r, r->slow, &r->slots[items[k]].view, r->cfg.consolidate_rate_q16, 65536);
    r->cursor = r->nslots ? cursor : 0;
    r->clock = tick;
    *replayed = ni;
    return ECS_MEM_OK;
}
static void ref_scores(const ref_mem *r, const ecs_mem_request *q, const ecs_mem_digest *cands, uint32_t n,
                       ecs_mem_digest *best_e, int64_t *best, int64_t *second, int64_t *signal) {
    const uint32_t D = r->cfg.dim;
    int8_t c[ECS_MEM_MAX_DIM], code[ECS_MEM_MAX_DIM];
    uint32_t bank;
    int64_t y[ECS_MEM_MAX_DIM];
    assert(ecs_mem_cue(q, D, r->cfg.banks, c, &bank) == ECS_MEM_OK);
    const int32_t *M = (q->channel == ECS_MEM_FAST ? r->fast : r->slow) + (size_t)bank * D * D;
    *signal = 0;
    for (uint32_t i = 0; i < D; ++i) {
        int64_t s = 0;
        for (uint32_t j = 0; j < D; ++j) s += (int64_t)M[(size_t)i * D + j] * c[j];
        y[i] = round_away(s, D);
        *signal += y[i] < 0 ? -y[i] : y[i];
    }
    *best = *second = INT64_MIN;
    for (uint32_t k = 0; k < n; ++k) {
        assert(ecs_mem_entity_code(&cands[k], D, code) == ECS_MEM_OK);
        int64_t s = 0;
        for (uint32_t i = 0; i < D; ++i) s += y[i] * code[i];
        if (k == 0 || s > *best || (s == *best && memcmp(cands[k].b, best_e->b, 32) < 0)) {
            if (k) *second = *best;
            *best = s;
            *best_e = cands[k];
        } else if (s > *second) {
            *second = s;
        }
    }
}

static void run(uint64_t seed, uint32_t law, uint32_t slot_policy, uint32_t ent_policy, uint32_t ops, uint32_t rate) {
    ref_mem r;
    memset(&r, 0, sizeof(r));
    r.cfg = tu_config(64, 12, 10);
    r.cfg.banks = 3;
    r.cfg.law = law;
    r.cfg.slot_policy = slot_policy;
    r.cfg.codebook_policy = ent_policy;
    r.cfg.retain_q16 = law == ECS_MEM_LAW_DELTA ? 60000u : 65536u;
    r.cfg.learn_rate_q16 = rate;
    r.cfg.consolidate_rate_q16 = 30000;
    const size_t bank_n = (size_t)r.cfg.banks * r.cfg.dim * r.cfg.dim;
    r.fast = (int32_t *)calloc(bank_n, sizeof(int32_t));
    r.slow = (int32_t *)calloc(bank_n, sizeof(int32_t));
    assert(r.fast && r.slow);
    ecs_mem *m;
    assert(ecs_mem_create(&r.cfg, &m) == ECS_MEM_OK);
    rng_state = seed;
    ecs_mem_digest seen_keys[1024];
    uint32_t nseen = 0, agree = 0;
    uint64_t tick = 0;
    for (uint32_t op = 0; op < ops; ++op) {
        const uint64_t kind = rnd() % 10;
        if (rnd() % 7) tick += rnd() % 3; /* ties and repeats of the same tick are legal */
        const uint64_t t = (rnd() % 23 == 0 && tick > 2) ? tick - 2 : tick; /* occasionally stale */
        if (kind < 6) {
            const uint32_t n = 1 + (uint32_t)(rnd() % 3);
            const uint32_t roles[3] = {1, 2, 3};
            uint64_t ents[3];
            for (uint32_t i = 0; i < n; ++i) ents[i] = rnd() % 14;
            const uint64_t p = rnd() % 10;
            ecs_mem_observation o =
                tu_obs(1 + (uint32_t)(rnd() % 2), p < 7 ? ECS_MEM_ASSERT : p < 9 ? ECS_MEM_NEGATE : ECS_MEM_UNCERTAIN,
                       t, n, roles, ents);
            ecs_mem_digest key;
            assert(ecs_mem_relation_key(&o, &key) == ECS_MEM_OK);
            /* the canonical participant order is part of the shared definition; derive it independently */
            ecs_mem_slot canon;
            memset(&canon, 0, sizeof(canon));
            canon.relation_type = o.relation_type;
            canon.participant_count = n;
            for (uint32_t i = 0; i < n; ++i) canon.participants[i] = o.participants[i]; /* roles 1..n ascending */
            const ecs_mem_status a = ecs_mem_observe(m, 0, &o, NULL, NULL);
            const ecs_mem_status b = ref_observe(&r, &o, &canon);
            assert(a == b);
            outcome[0][-(int)a]++;
            if (a == ECS_MEM_OK) seen_keys[nseen++ % 1024u] = key; /* ring of the latest admitted keys */
        } else if (kind < 8 && nseen) {
            const uint32_t held = nseen < 1024u ? nseen : 1024u;
            const ecs_mem_digest *k = &seen_keys[(nseen - 1u - (uint32_t)(rnd() % held)) % 1024u];
            const ecs_mem_status a = ecs_mem_learn(m, 0, k, t, NULL);
            const ecs_mem_status b = ref_learn(&r, k, t);
            assert(a == b);
            outcome[1][-(int)a]++;
        } else if (kind == 8) {
            const uint32_t items = 1 + (uint32_t)(rnd() % 5);
            uint32_t ra = 0, rb = 0;
            const ecs_mem_status a = ecs_mem_consolidate(m, 0, t, items, &ra, NULL);
            const ecs_mem_status b = ref_consolidate(&r, t, items, &rb);
            assert(a == b && (a != ECS_MEM_OK || ra == rb));
            outcome[2][-(int)a]++;
        }
        /* Residency agreement. */
        for (uint32_t i = 0; i < (nseen < 1024u ? nseen : 1024u); ++i) {
            ecs_mem_slot v;
            const ecs_mem_status st = ecs_mem_slot_read(m, &seen_keys[i], &v);
            const int rs = find_slot(&r, &seen_keys[i]);
            assert((st == ECS_MEM_OK) == (rs >= 0));
            if (rs >= 0) {
                assert(v.assert_count == r.slots[rs].view.assert_count);
                assert(v.negate_count == r.slots[rs].view.negate_count);
                assert(v.uncertain_count == r.slots[rs].view.uncertain_count);
                assert(!memcmp(v.participants, r.slots[rs].view.participants, sizeof(v.participants)));
            }
        }
        for (uint64_t e = 0; e < 14; ++e) {
            ecs_mem_digest d = tu_entity(e);
            int has = 0;
            assert(ecs_mem_codebook_contains(m, &d, &has) == ECS_MEM_OK);
            assert(has == (find_ent(&r, &d) >= 0));
        }
        /* Channel agreement on a random query. */
        ecs_mem_request q = tu_query(ECS_MEM_FAST + (uint32_t)(rnd() % 2), 1 + (uint32_t)(rnd() % 2),
                                     1 + (uint32_t)(rnd() % 3));
        const uint32_t cue_role = q.target_role == 1 ? 2 : 1;
        tu_cue(&q, cue_role, rnd() % 14);
        ecs_mem_digest cands[14];
        for (uint64_t e = 0; e < 14; ++e) cands[e] = tu_entity(e);
        q.candidate_source = ECS_MEM_CANDIDATES_CALLER;
        q.candidates = cands;
        q.candidate_count = 14;
        ecs_mem_answer ans;
        assert(ecs_mem_query(m, &q, &ans) == ECS_MEM_OK);
        ecs_mem_digest be;
        int64_t best, second, signal;
        ref_scores(&r, &q, cands, 14, &be, &best, &second, &signal);
        assert(ans.found && tu_same(ans.answer, be) && ans.score == best && ans.runner_up == second);
        assert(ans.signal == signal);
        /* LOOKUP agreement: brute force over the reference's resident slots. */
        ecs_mem_request lq = tu_query(ECS_MEM_LOOKUP, q.relation_type, q.target_role);
        lq.cue_count = 1;
        lq.cue[0] = q.cue[0];
        assert(ecs_mem_query(m, &lq, &ans) == ECS_MEM_OK);
        uint32_t matches = 0, contradicted = 0;
        int best_slot = -1;
        for (uint32_t i = 0; i < r.nslots; ++i) {
            const ecs_mem_slot *v = &r.slots[i].view;
            if (v->relation_type != lq.relation_type) continue;
            int cue_ok = 0, target = -1;
            for (uint32_t j = 0; j < v->participant_count; ++j) {
                if (v->participants[j].role == lq.cue[0].role && tu_same(v->participants[j].entity, lq.cue[0].entity))
                    cue_ok = 1;
                if (v->participants[j].role == lq.target_role) target = (int)j;
            }
            if (!cue_ok || target < 0) continue;
            ++matches;
            const int64_t net = (int64_t)v->assert_count - v->negate_count;
            contradicted += net < 0;
            if (net <= 0) continue;
            if (best_slot < 0) { best_slot = (int)i; continue; }
            const ecs_mem_slot *b = &r.slots[best_slot].view;
            const int64_t bnet = (int64_t)b->assert_count - b->negate_count;
            const int c = memcmp(v->participants[target].entity.b,
                                 b->participants[target].entity.b, 32); /* same target role index in both */
            if (v->last_assert_tick > b->last_assert_tick ||
                (v->last_assert_tick == b->last_assert_tick && (net > bnet || (net == bnet && (c < 0 ||
                 (c == 0 && memcmp(r.slots[i].key.b, r.slots[best_slot].key.b, 32) < 0))))))
                best_slot = (int)i;
        }
        assert(ans.matches == matches && ans.contradicted == contradicted);
        assert(ans.found == (best_slot >= 0));
        if (best_slot >= 0) assert(tu_same(ans.answer_key, r.slots[best_slot].key));
        ++agree;
    }
    if (ops > 1000) {
        /* Tail: relearn one resident slot with net evidence until the learning bound refuses (Hebbian grows
         * linearly on a repeated cue); both implementations must refuse at the same step, atomically. */
        int s = -1;
        for (uint32_t i = 0; i < r.nslots && s < 0; ++i)
            if (r.slots[i].view.assert_count > r.slots[i].view.negate_count) s = (int)i;
        assert(s >= 0);
        ecs_mem_status a = ECS_MEM_OK;
        for (uint32_t i = 0; i < 2000 && a == ECS_MEM_OK; ++i) {
            ++tick;
            a = ecs_mem_learn(m, 0, &r.slots[s].key, tick, NULL);
            assert(a == ref_learn(&r, &r.slots[s].key, tick));
            outcome[1][-(int)a]++;
        }
        assert(r.cfg.law != ECS_MEM_LAW_HEBBIAN || a == ECS_MEM_NUMERIC);
    }
    ecs_mem_destroy(m);
    free(r.fast);
    free(r.slow);
    printf("seed=%llu law=%u policies=%u/%u ops=%u agreements=%u\n", (unsigned long long)seed, law, slot_policy,
           ent_policy, ops, agree);
}

int main(void) {
    for (uint64_t seed = 1; seed <= 6; ++seed) {
        run(seed, ECS_MEM_LAW_DELTA, ECS_MEM_EVICT_LRU, ECS_MEM_EVICT_LRU, 400, 50000);
        run(seed + 100, ECS_MEM_LAW_HEBBIAN, ECS_MEM_EVICT_LRU, ECS_MEM_EVICT_LRU, 300, 50000);
        run(seed + 200, ECS_MEM_LAW_DELTA, ECS_MEM_REFUSE, ECS_MEM_REFUSE, 300, 50000);
        run(seed + 300, ECS_MEM_LAW_DELTA, ECS_MEM_EVICT_LRU, ECS_MEM_REFUSE, 300, 50000);
    }
    /* Long Hebbian traffic at full rate reaches the learning bound: NUMERIC refusals must agree too. */
    run(400, ECS_MEM_LAW_HEBBIAN, ECS_MEM_EVICT_LRU, ECS_MEM_EVICT_LRU, 4000, 65536);
    static const char *names[3] = {"observe", "learn", "consolidate"};
    for (int k = 0; k < 3; ++k) {
        printf("%s outcomes:", names[k]);
        for (int s = 0; s < 12; ++s)
            if (outcome[k][s]) printf(" %s=%lu", ecs_mem_code_name(-s), outcome[k][s]);
        printf("\n");
    }
    /* Coverage the differential must reach to mean anything. */
    fflush(stdout);
    assert(outcome[0][0] && outcome[0][-ECS_MEM_CLOCK] && outcome[0][-ECS_MEM_CAPACITY]);
    assert(outcome[1][0] && outcome[1][-ECS_MEM_NOT_FOUND] && outcome[1][-ECS_MEM_NO_NET_EVIDENCE]);
    assert(outcome[1][-ECS_MEM_CLOCK] && outcome[1][-ECS_MEM_NUMERIC] && outcome[2][0]);
    printf("PASS_ECS_MEM_REFERENCE_DIFFERENTIAL\n");
    return 0;
}
