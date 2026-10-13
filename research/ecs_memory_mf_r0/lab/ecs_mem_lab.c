/* ECS Memory MF-R0 laboratory (RESEARCH_ONLY, TRAINING=NONE, SEMANTICS=NONE).
 *
 * Implements research/ecs_memory_mf_r0/PREREGISTRATION.md: synthetic relational worlds S1-S8, nine arms, one
 * JSON document per seed on stdout with integer counts (scored by score.py). Wall-clock latency is printed
 * for description only. Parameters the preregistration leaves open are fixed here, before any run, and
 * echoed in the output ("parameters").
 *
 * usage: ecs_mem_lab --seed N        one seed, every scenario
 *        ecs_mem_lab --smoke         a reduced S1 and S2 world, checked for internal consistency (ctest) */
#define _POSIX_C_SOURCE 200809L
#include "ecs_mem.h"
#include "elpis/sha256.h"
#include "model_free_memory.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------------------------------------------------------------------------------------------------- */
/* Fixed parameters                                                                                     */
/* ---------------------------------------------------------------------------------------------------- */

enum { DIM = 128, BANKS = 4, FAST_SLOTS = 64, FAST_CODEBOOK = 64, CANDIDATES = 64 };
enum { RETAIN_S7 = 64880, CONSOLIDATE_RATE = 32768, KEYED_ENTRY_BYTES = 80, D31_RADIUS = 8 };
enum { ROLE_1 = 1, ROLE_2 = 2, ROLE_3 = 3 };
static const float D31_ALPHA[2] = {1.0f, 1.0f}, D31_BETA[2] = {1.0f, 1.0f};

/* ---------------------------------------------------------------------------------------------------- */
/* World                                                                                                */
/* ---------------------------------------------------------------------------------------------------- */

static uint64_t g_seed;
static uint64_t rng_state;
static uint64_t rnd(void) {
    uint64_t z = (rng_state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
static ecs_mem_digest ent(uint64_t k) {
    char buf[96];
    ecs_mem_digest d;
    const int n = snprintf(buf, sizeof(buf), "elpis.ecs-mem-lab.entity|%llu|%llu", (unsigned long long)g_seed,
                           (unsigned long long)k);
    elpis_sha256(buf, (size_t)n, d.b);
    return d;
}
static int same(const ecs_mem_digest *a, const ecs_mem_digest *b) { return !memcmp(a->b, b->b, 32); }

typedef struct {
    uint32_t role;
    ecs_mem_digest e;
} part;
typedef struct {
    uint32_t type, n;
    part p[3];
} rel;
static rel rel2(uint32_t type, ecs_mem_digest s, ecs_mem_digest o) {
    rel r;
    memset(&r, 0, sizeof(r));
    r.type = type;
    r.n = 2;
    r.p[0].role = ROLE_1;
    r.p[0].e = s;
    r.p[1].role = ROLE_2;
    r.p[1].e = o;
    return r;
}
static rel rel3(uint32_t type, ecs_mem_digest a, ecs_mem_digest b, ecs_mem_digest c) {
    rel r = rel2(type, a, b);
    r.n = 3;
    r.p[2].role = ROLE_3;
    r.p[2].e = c;
    return r;
}

/* ---------------------------------------------------------------------------------------------------- */
/* KEYED: byte-matched exact keyed store (independent of MF-R0), LRU eviction, last write wins           */
/* ---------------------------------------------------------------------------------------------------- */

typedef struct {
    uint32_t cap, count, mask, head, tail;
    ecs_mem_digest *key, *val;
    uint32_t *prev, *next, *index;
    uint64_t probes, writes, evictions;
} keyed;
#define KNONE UINT32_MAX
static void keyed_init(keyed *k, uint32_t cap) {
    memset(k, 0, sizeof(*k));
    k->cap = cap;
    uint32_t n = 16;
    while (n < 2u * cap) n <<= 1;
    k->mask = n - 1u;
    k->key = calloc(cap, sizeof(ecs_mem_digest));
    k->val = calloc(cap, sizeof(ecs_mem_digest));
    k->prev = calloc(cap, 4);
    k->next = calloc(cap, 4);
    k->index = calloc(n, 4);
    assert(k->key && k->val && k->prev && k->next && k->index);
    k->head = k->tail = KNONE;
}
static void keyed_free(keyed *k) {
    free(k->key); free(k->val); free(k->prev); free(k->next); free(k->index);
}
static uint32_t khome(const keyed *k, const ecs_mem_digest *d) {
    uint64_t h = 0;
    memcpy(&h, d->b, 8);
    return (uint32_t)(h & k->mask);
}
static uint32_t keyed_find(keyed *k, const ecs_mem_digest *d) {
    for (uint32_t pos = khome(k, d);; pos = (pos + 1u) & k->mask) {
        ++k->probes;
        if (!k->index[pos]) return KNONE;
        if (same(&k->key[k->index[pos] - 1u], d)) return k->index[pos] - 1u;
    }
}
static void k_unlink(keyed *k, uint32_t i) {
    if (k->prev[i] != KNONE) k->next[k->prev[i]] = k->next[i]; else k->head = k->next[i];
    if (k->next[i] != KNONE) k->prev[k->next[i]] = k->prev[i]; else k->tail = k->prev[i];
}
static void k_push(keyed *k, uint32_t i) {
    k->prev[i] = k->tail;
    k->next[i] = KNONE;
    if (k->tail != KNONE) k->next[k->tail] = i; else k->head = i;
    k->tail = i;
}
static void keyed_remove_index(keyed *k, uint32_t i) {
    uint32_t pos = khome(k, &k->key[i]);
    while (k->index[pos] != i + 1u) pos = (pos + 1u) & k->mask;
    uint32_t hole = pos;
    for (uint32_t j = (pos + 1u) & k->mask; k->index[j]; j = (j + 1u) & k->mask) {
        const uint32_t h = khome(k, &k->key[k->index[j] - 1u]);
        const int stays = hole <= j ? (hole < h && h <= j) : (hole < h || h <= j);
        if (!stays) {
            k->index[hole] = k->index[j];
            hole = j;
        }
    }
    k->index[hole] = 0;
}
static void keyed_put(keyed *k, const ecs_mem_digest *key, const ecs_mem_digest *val) {
    ++k->writes;
    uint32_t i = keyed_find(k, key);
    if (i != KNONE) {
        k->val[i] = *val;
        k_unlink(k, i);
        k_push(k, i);
        return;
    }
    if (k->count < k->cap) {
        i = k->count++;
    } else {
        i = k->head;
        k_unlink(k, i);
        keyed_remove_index(k, i);
        ++k->evictions;
    }
    k->key[i] = *key;
    k->val[i] = *val;
    uint32_t pos = khome(k, key);
    while (k->index[pos]) pos = (pos + 1u) & k->mask;
    k->index[pos] = i + 1u;
    k_push(k, i);
}

/* Canonical cue key of one direction: (type, target role, other participants by role). */
static ecs_mem_digest cue_key(uint32_t type, uint32_t trole, const part *cue, uint32_t k) {
    elpis_sha256_ctx c;
    ecs_mem_digest d;
    uint32_t hdr[3] = {type, trole, k};
    elpis_sha256_init(&c);
    elpis_sha256_update(&c, "elpis.ecs-mem-lab.cue", 21);
    elpis_sha256_update(&c, hdr, sizeof(hdr));
    for (uint32_t r = 1; r <= 3; ++r)
        for (uint32_t i = 0; i < k; ++i)
            if (cue[i].role == r) {
                elpis_sha256_update(&c, &cue[i].role, 4);
                elpis_sha256_update(&c, cue[i].e.b, 32);
            }
    elpis_sha256_final(&c, d.b);
    return d;
}
static uint64_t h64(const ecs_mem_digest *d) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | d->b[i];
    return v;
}

/* ---------------------------------------------------------------------------------------------------- */
/* Arms                                                                                                 */
/* ---------------------------------------------------------------------------------------------------- */

enum { A_FAST, A_SLOW, A_HEBB, A_LOOKUP, A_KEYED, A_D31C, A_FIXED, A_NONE, A_PERM, NARMS };
static const char *ARM[NARMS] = {"FAST", "SLOW", "HEBB", "LOOKUP", "KEYED", "D31C", "FIXED", "NONE", "PERM"};

typedef struct {
    uint64_t correct, total, stale, conflicts_seen, read_mac, read_scans, read_sha, query_ns;
} tally;

typedef struct {
    ecs_mem *fast, *hebb, *lookup, *fixed, *none;
    keyed kv;
    ecs_mf_memory *d31;
    uint64_t tick, budget, refusals[NARMS], learn_refused_numeric;
    tally t[NARMS];
} bench;

static ecs_mem_config fast_config(uint32_t retain) {
    ecs_mem_config c;
    ecs_mem_config_default(&c);
    c.dim = DIM;
    c.banks = BANKS;
    c.slot_capacity = FAST_SLOTS;
    c.codebook_capacity = FAST_CODEBOOK;
    c.learn_rate_q16 = ECS_MEM_Q16_ONE;
    c.retain_q16 = retain;
    c.consolidate_rate_q16 = CONSOLIDATE_RATE;
    c.byte_budget = (uint64_t)64 << 20;
    return c;
}
static uint64_t budget_bytes(void) {
    ecs_mem_config c = fast_config(ECS_MEM_Q16_ONE);
    uint64_t arena, state;
    assert(ecs_mem_required_bytes(&c, &arena, &state) == ECS_MEM_OK);
    return state;
}
/* Fixed random banks through the public envelope: snapshot an empty handle, fill the bank region, reseal. */
static ecs_mem *fixed_random(const ecs_mem_config *c) {
    ecs_mem *m;
    assert(ecs_mem_create(c, &m) == ECS_MEM_OK);
    const size_t n = ecs_mem_snapshot_size(m);
    uint8_t *b = malloc(n);
    size_t w;
    assert(b && ecs_mem_snapshot(m, b, n, &w) == ECS_MEM_OK);
    const size_t weights = (size_t)2 * c->banks * c->dim * c->dim;
    for (size_t i = 0; i < weights; ++i) {
        const int32_t v = (int32_t)(rnd() % 131073u) - 65536; /* uniform in [-1, 1] Q16 */
        for (int k = 0; k < 4; ++k) b[ECS_MEM_HEADER_BYTES + 4 * i + (size_t)k] = (uint8_t)((uint32_t)v >> (8 * k));
    }
    elpis_sha256(b, n - 32, b + n - 32);
    assert(ecs_mem_restore_into(m, 0, b, n) == ECS_MEM_OK);
    free(b);
    return m;
}
static void bench_init(bench *b, uint32_t retain) {
    memset(b, 0, sizeof(*b));
    b->budget = budget_bytes();
    ecs_mem_config c = fast_config(retain);
    assert(ecs_mem_create(&c, &b->fast) == ECS_MEM_OK);
    ecs_mem_config h = c;
    h.law = ECS_MEM_LAW_HEBBIAN;
    assert(ecs_mem_create(&h, &b->hebb) == ECS_MEM_OK);
    ecs_mem_config l;
    ecs_mem_config_default(&l);
    l.dim = 64;
    l.banks = 1;
    l.codebook_capacity = ECS_MEM_MAX_PARTICIPANTS;
    l.slot_capacity = 1;
    uint64_t arena, empty_state;
    assert(ecs_mem_required_bytes(&l, &arena, &empty_state) == ECS_MEM_OK);
    const uint64_t fixed_part = empty_state - (ECS_MEM_SLOT_RECORD_BYTES + 4u);
    l.slot_capacity = (uint32_t)((b->budget - fixed_part) / (ECS_MEM_SLOT_RECORD_BYTES + 4u));
    assert(ecs_mem_create(&l, &b->lookup) == ECS_MEM_OK);
    b->fixed = fixed_random(&c);
    assert(ecs_mem_create(&c, &b->none) == ECS_MEM_OK);
    keyed_init(&b->kv, (uint32_t)(b->budget / KEYED_ENTRY_BYTES));
    const uint64_t index_bytes = (uint64_t)(2 * D31_RADIUS + 3) * (2 * D31_RADIUS + 3) * 4u;
    const size_t cells = (size_t)((b->budget - index_bytes) / d31c_cell_bytes());
    b->d31 = ecs_mf_create(cells, D31C_LRU_REUSE, D31_RADIUS, D31_ALPHA, D31_BETA);
    assert(b->d31);
}
static void bench_free(bench *b) {
    ecs_mem_destroy(b->fast); ecs_mem_destroy(b->hebb); ecs_mem_destroy(b->lookup);
    ecs_mem_destroy(b->fixed); ecs_mem_destroy(b->none);
    keyed_free(&b->kv);
    ecs_mf_destroy(b->d31);
}

static ecs_mem_observation to_obs(const rel *r, uint64_t tick, uint64_t source) {
    ecs_mem_observation o;
    memset(&o, 0, sizeof(o));
    o.abi_version = ECS_MEM_ABI_V1;
    o.relation_type = r->type;
    o.polarity = ECS_MEM_ASSERT;
    o.participant_count = r->n;
    o.tick = tick;
    o.provenance.b[0] = (uint8_t)source;
    for (uint32_t i = 0; i < r->n; ++i) {
        o.participants[i].role = r->p[i].role;
        o.participants[i].entity = r->p[i].e;
    }
    return o;
}
static void mf_observe_learn(bench *b, ecs_mem *m, int arm, const ecs_mem_observation *o, int learn) {
    ecs_mem_digest key;
    ecs_mem_status s = ecs_mem_observe(m, 0, o, &key, NULL);
    assert(s == ECS_MEM_OK);
    if (!learn) return;
    s = ecs_mem_learn(m, 0, &key, o->tick, NULL);
    if (s == ECS_MEM_NUMERIC) b->learn_refused_numeric++;
    if (s != ECS_MEM_OK) b->refusals[arm]++;
    assert(s == ECS_MEM_OK || s == ECS_MEM_NUMERIC || s == ECS_MEM_NO_NET_EVIDENCE);
}
static void code64(const ecs_mem_digest *e, float out[64]) {
    int8_t c[64];
    assert(ecs_mem_entity_code(e, 64, c) == ECS_MEM_OK);
    for (int i = 0; i < 64; ++i) out[i] = 0.125f * (float)c[i];
}
/* One observation, delivered to every arm (each learns in its own way). */
static void observe(bench *b, const rel *r, uint64_t source) {
    const ecs_mem_observation o = to_obs(r, b->tick, source);
    mf_observe_learn(b, b->fast, A_FAST, &o, 1);
    mf_observe_learn(b, b->hebb, A_HEBB, &o, 1);
    mf_observe_learn(b, b->lookup, A_LOOKUP, &o, 0);
    for (uint32_t t = 0; t < r->n; ++t) {
        part cue[3];
        uint32_t k = 0;
        for (uint32_t i = 0; i < r->n; ++i)
            if (i != t) cue[k++] = r->p[i];
        const ecs_mem_digest key = cue_key(r->type, r->p[t].role, cue, k);
        keyed_put(&b->kv, &key, &r->p[t].e);
        float v[64];
        code64(&r->p[t].e, v);
        if (ecs_mf_write(b->d31, r->type, h64(&key), r->p[t].role, v) != D31C_OK) b->refusals[A_D31C]++;
    }
}
static void consolidate(bench *b) {
    assert(ecs_mem_consolidate(b->fast, 0, b->tick, 16, NULL, NULL) == ECS_MEM_OK);
}

typedef struct {
    uint32_t type, trole, k;
    part cue[2];
} query;

static uint32_t perm_role(uint32_t r) { return r == ROLE_1 ? ROLE_2 : r == ROLE_2 ? ROLE_3 : ROLE_1; }
static int64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000000000 + t.tv_nsec;
}
/* The answer of one arm: an index into cands, or -1 (no answer). */
static int answer(bench *b, int arm, const query *q, const ecs_mem_digest *cands, uint32_t n) {
    const int64_t t0 = now_ns();
    int out = -1;
    if (arm == A_KEYED) {
        const ecs_mem_digest key = cue_key(q->type, q->trole, q->cue, q->k);
        const uint32_t i = keyed_find(&b->kv, &key);
        if (i != KNONE)
            for (uint32_t c = 0; c < n; ++c)
                if (same(&cands[c], &b->kv.val[i])) out = (int)c;
    } else if (arm == A_D31C) {
        const ecs_mem_digest key = cue_key(q->type, q->trole, q->cue, q->k);
        float v[64];
        if (ecs_mf_read(b->d31, q->type, h64(&key), q->trole, v) == D31C_OK) {
            double best = 0;
            for (uint32_t c = 0; c < n; ++c) {
                float code[64];
                double s = 0;
                code64(&cands[c], code);
                for (int i = 0; i < 64; ++i) s += (double)v[i] * code[i];
                if (out < 0 || s > best || (s == best && memcmp(cands[c].b, cands[out].b, 32) < 0)) {
                    best = s;
                    out = (int)c;
                }
            }
        }
    } else {
        ecs_mem_request r;
        memset(&r, 0, sizeof(r));
        r.abi_version = ECS_MEM_ABI_V1;
        r.relation_type = q->type;
        r.target_role = arm == A_PERM ? perm_role(q->trole) : q->trole;
        r.cue_count = q->k;
        for (uint32_t i = 0; i < q->k; ++i) {
            r.cue[i].role = arm == A_PERM ? perm_role(q->cue[i].role) : q->cue[i].role;
            r.cue[i].entity = q->cue[i].e;
        }
        const ecs_mem *m = arm == A_HEBB ? b->hebb : arm == A_LOOKUP ? b->lookup : arm == A_FIXED ? b->fixed
                           : arm == A_NONE ? b->none : b->fast;
        r.channel = arm == A_LOOKUP ? ECS_MEM_LOOKUP : arm == A_SLOW ? ECS_MEM_SLOW : ECS_MEM_FAST;
        if (r.channel != ECS_MEM_LOOKUP) {
            r.candidate_source = ECS_MEM_CANDIDATES_CALLER;
            r.candidates = cands;
            r.candidate_count = n;
        }
        ecs_mem_answer a;
        assert(ecs_mem_query(m, &r, &a) == ECS_MEM_OK);
        b->t[arm].read_mac += a.work.mac_ops;
        b->t[arm].read_scans += a.work.slot_scans;
        b->t[arm].read_sha += a.work.sha_blocks;
        if (arm == A_LOOKUP && a.conflicts) b->t[arm].conflicts_seen++;
        if (a.found)
            for (uint32_t c = 0; c < n; ++c)
                if (same(&cands[c], &a.answer)) out = (int)c;
    }
    b->t[arm].query_ns += (uint64_t)(now_ns() - t0);
    return out;
}
static void score(bench *b, int arm, int got, const ecs_mem_digest *cands, const ecs_mem_digest *accept,
                  uint32_t naccept, const ecs_mem_digest *stale) {
    b->t[arm].total++;
    if (got < 0) return;
    for (uint32_t i = 0; i < naccept; ++i)
        if (same(&cands[got], &accept[i])) {
            b->t[arm].correct++;
            return;
        }
    if (stale && same(&cands[got], stale)) b->t[arm].stale++;
}

/* ---------------------------------------------------------------------------------------------------- */
/* Output                                                                                               */
/* ---------------------------------------------------------------------------------------------------- */

static int first_row = 1;
static void emit(bench *b, const char *scenario, const char *condition, const int *arms, int narms) {
    for (int k = 0; k < narms; ++k) {
        const int a = arms[k];
        ecs_mem_stats st;
        const ecs_mem *m = a == A_HEBB ? b->hebb : a == A_LOOKUP ? b->lookup : a == A_FIXED ? b->fixed
                           : a == A_NONE ? b->none : (a == A_KEYED || a == A_D31C) ? NULL : b->fast;
        uint64_t state = 0, capacity = 0, write_mac = 0;
        if (m) {
            uint64_t arena;
            assert(ecs_mem_inspect(m, &st) == ECS_MEM_OK);
            assert(ecs_mem_required_bytes(&st.config, &arena, &capacity) == ECS_MEM_OK);
            state = st.state_bytes;
            write_mac = st.write_work.mac_ops;
        } else if (a == A_KEYED) {
            state = (uint64_t)b->kv.count * KEYED_ENTRY_BYTES;
            capacity = (uint64_t)b->kv.cap * KEYED_ENTRY_BYTES;
        } else {
            state = capacity = ecs_mf_allocated_bytes(b->d31);
        }
        const tally *t = &b->t[a];
        printf("%s\n    {\"scenario\": \"%s\", \"condition\": \"%s\", \"arm\": \"%s\", \"correct\": %llu, "
               "\"total\": %llu, \"stale\": %llu, \"conflicts_seen\": %llu, \"refusals\": %llu, "
               "\"state_bytes\": %llu, \"capacity_bytes\": %llu, \"budget_bytes\": %llu, \"write_mac\": %llu, \"read_mac\": %llu, "
               "\"read_slot_scans\": %llu, \"read_sha_blocks\": %llu, \"keyed_probes\": %llu, "
               "\"query_ns\": %llu}",
               first_row ? "" : ",", scenario, condition, ARM[a], (unsigned long long)t->correct,
               (unsigned long long)t->total, (unsigned long long)t->stale, (unsigned long long)t->conflicts_seen,
               (unsigned long long)b->refusals[a], (unsigned long long)state, (unsigned long long)capacity,
               (unsigned long long)b->budget,
               (unsigned long long)write_mac, (unsigned long long)t->read_mac, (unsigned long long)t->read_scans,
               (unsigned long long)t->read_sha, (unsigned long long)(a == A_KEYED ? b->kv.probes : 0),
               (unsigned long long)t->query_ns);
        first_row = 0;
    }
}
static const int ALL[NARMS] = {A_FAST, A_SLOW, A_HEBB, A_LOOKUP, A_KEYED, A_D31C, A_FIXED, A_NONE, A_PERM};

/* ---------------------------------------------------------------------------------------------------- */
/* Scenarios                                                                                            */
/* ---------------------------------------------------------------------------------------------------- */

static void pool(ecs_mem_digest *out, uint64_t base, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) out[i] = ent(base + i);
}
static query q_of(uint32_t type, uint32_t trole, uint32_t k, uint32_t r0, ecs_mem_digest e0, uint32_t r1,
                  ecs_mem_digest e1) {
    query q;
    memset(&q, 0, sizeof(q));
    q.type = type;
    q.trole = trole;
    q.k = k;
    q.cue[0].role = r0;
    q.cue[0].e = e0;
    q.cue[1].role = r1;
    q.cue[1].e = e1;
    return q;
}

static void s1(uint32_t n) {
    bench b;
    bench_init(&b, ECS_MEM_Q16_ONE);
    ecs_mem_digest cands[CANDIDATES];
    pool(cands, 2000, CANDIDATES);
    for (uint32_t i = 0; i < n; ++i, ++b.tick) {
        const rel r = rel2(1, ent(1000 + i), cands[i % CANDIDATES]);
        observe(&b, &r, 1);
    }
    for (uint32_t i = 0; i < n; ++i) {
        const query q = q_of(1, ROLE_2, 1, ROLE_1, ent(1000 + i), 0, ent(0));
        for (int a = 0; a < NARMS; ++a) score(&b, a, answer(&b, a, &q, cands, CANDIDATES), cands, &cands[i % CANDIDATES], 1, NULL);
    }
    emit(&b, "S1", "base", ALL, NARMS);
    bench_free(&b);
}

static void s2(const char *cond, uint32_t len, uint32_t subjects) {
    bench b;
    bench_init(&b, ECS_MEM_Q16_ONE);
    ecs_mem_digest cands[CANDIDATES];
    pool(cands, 4000, CANDIDATES);
    uint32_t A[64], B[64];
    for (uint32_t i = 0; i < subjects; ++i) {
        A[i] = (uint32_t)(rnd() % CANDIDATES);
        do B[i] = (uint32_t)(rnd() % CANDIDATES); while (B[i] == A[i]);
    }
    for (uint32_t round = 0; round < len; ++round, ++b.tick)
        for (uint32_t i = 0; i < subjects; ++i) {
            const rel r = rel2(2, ent(3000 + i), cands[round % 2 ? B[i] : A[i]]);
            observe(&b, &r, 1);
        }
    for (uint32_t i = 0; i < subjects; ++i) {
        const uint32_t latest = (len - 1) % 2 ? B[i] : A[i], older = (len - 1) % 2 ? A[i] : B[i];
        const query q = q_of(2, ROLE_2, 1, ROLE_1, ent(3000 + i), 0, ent(0));
        for (int a = 0; a < NARMS; ++a)
            score(&b, a, answer(&b, a, &q, cands, CANDIDATES), cands, &cands[latest], 1, &cands[older]);
    }
    emit(&b, "S2", cond, ALL, NARMS);
    bench_free(&b);
}

static void s3(void) {
    bench b;
    bench_init(&b, ECS_MEM_Q16_ONE);
    ecs_mem_digest cands[CANDIDATES];
    pool(cands, 5000, CANDIDATES); /* 24 participants + 40 never-observed distractors */
    uint32_t tri[48][3];
    for (uint32_t j = 0; j < 48; ++j) {
        tri[j][0] = (uint32_t)(rnd() % 24);
        do tri[j][1] = (uint32_t)(rnd() % 24); while (tri[j][1] == tri[j][0]);
        do tri[j][2] = (uint32_t)(rnd() % 24); while (tri[j][2] == tri[j][0] || tri[j][2] == tri[j][1]);
        const rel r = rel3(3, cands[tri[j][0]], cands[tri[j][1]], cands[tri[j][2]]);
        observe(&b, &r, 1);
        ++b.tick;
    }
    for (uint32_t j = 0; j < 48; ++j)
        for (uint32_t t = 0; t < 3; ++t) {
            const uint32_t o1 = (t + 1) % 3, o2 = (t + 2) % 3;
            const query q = q_of(3, t + 1, 2, o1 + 1, cands[tri[j][o1]], o2 + 1, cands[tri[j][o2]]);
            ecs_mem_digest acc[48];
            uint32_t na = 0;
            for (uint32_t i = 0; i < 48; ++i)
                if (tri[i][o1] == tri[j][o1] && tri[i][o2] == tri[j][o2]) acc[na++] = cands[tri[i][t]];
            for (int a = 0; a < NARMS; ++a) score(&b, a, answer(&b, a, &q, cands, CANDIDATES), cands, acc, na, NULL);
        }
    emit(&b, "S3", "base", ALL, NARMS);
    bench_free(&b);
}

static void s4(void) {
    bench b;
    bench_init(&b, ECS_MEM_Q16_ONE);
    ecs_mem_digest cands[CANDIDATES];
    pool(cands, 7000, CANDIDATES);
    uint32_t o1[32], o2[32];
    for (uint32_t i = 0; i < 32; ++i, ++b.tick) {
        o1[i] = (uint32_t)(rnd() % CANDIDATES);
        do o2[i] = (uint32_t)(rnd() % CANDIDATES); while (o2[i] == o1[i]);
        const rel r1 = rel2(4, ent(6000 + i), cands[o1[i]]), r2 = rel2(4, ent(6000 + i), cands[o2[i]]);
        observe(&b, &r1, 1); /* two sources, the same tick */
        observe(&b, &r2, 2);
    }
    for (uint32_t i = 0; i < 32; ++i) {
        const query q = q_of(4, ROLE_2, 1, ROLE_1, ent(6000 + i), 0, ent(0));
        const ecs_mem_digest acc[2] = {cands[o1[i]], cands[o2[i]]};
        for (int a = 0; a < NARMS; ++a) score(&b, a, answer(&b, a, &q, cands, CANDIDATES), cands, acc, 2, NULL);
    }
    emit(&b, "S4", "base", ALL, NARMS);
    bench_free(&b);
}

static void s5(int shared) {
    static const char *cue_names[3] = {"agent", "recipient", "both"};
    bench b;
    bench_init(&b, ECS_MEM_Q16_ONE);
    ecs_mem_digest cands[CANDIDATES];
    pool(cands, 8200, CANDIDATES); /* 48 objects + 16 never-observed distractors */
    for (uint32_t i = 0; i < 48; ++i, ++b.tick) {
        const rel r = rel3(5, ent(8000 + (shared ? i / 2 : i)), ent(8100 + i), cands[i]);
        observe(&b, &r, 1);
    }
    for (int cue = 0; cue < 3; ++cue) {
        memset(b.t, 0, sizeof(b.t));
        for (uint32_t i = 0; i < 48; ++i) {
            const ecs_mem_digest agent = ent(8000 + (shared ? i / 2 : i)), recipient = ent(8100 + i);
            const query q = cue == 0 ? q_of(5, ROLE_3, 1, ROLE_1, agent, 0, ent(0))
                            : cue == 1 ? q_of(5, ROLE_3, 1, ROLE_2, recipient, 0, ent(0))
                                       : q_of(5, ROLE_3, 2, ROLE_1, agent, ROLE_2, recipient);
            ecs_mem_digest acc[2] = {cands[i], cands[i]};
            uint32_t na = 1;
            if (shared && cue == 0) {
                acc[1] = cands[i ^ 1u]; /* the agent's other relation is an equally valid completion */
                na = 2;
            }
            for (int a = 0; a < NARMS; ++a) score(&b, a, answer(&b, a, &q, cands, CANDIDATES), cands, acc, na, NULL);
        }
        char cond[48];
        snprintf(cond, sizeof(cond), "%s/%s", shared ? "SHARED" : "UNIQUE", cue_names[cue]);
        emit(&b, "S5", cond, ALL, NARMS);
    }
    bench_free(&b);
}

static void s6(uint32_t K) {
    bench b;
    bench_init(&b, ECS_MEM_Q16_ONE);
    ecs_mem_digest cands[CANDIDATES];
    pool(cands, 9100, CANDIDATES);
    uint32_t obj[32];
    for (uint32_t i = 0; i < 32; ++i, ++b.tick) {
        obj[i] = (uint32_t)(rnd() % CANDIDATES);
        const rel r = rel2(6, ent(9000 + i), cands[obj[i]]);
        observe(&b, &r, 1);
    }
    for (uint32_t k = 0; k < K; ++k, ++b.tick) {
        const rel r = rel2(6, ent(100000 + k), cands[rnd() % CANDIDATES]);
        observe(&b, &r, 1);
    }
    for (uint32_t i = 0; i < 32; ++i) {
        const query q = q_of(6, ROLE_2, 1, ROLE_1, ent(9000 + i), 0, ent(0));
        for (int a = 0; a < NARMS; ++a) score(&b, a, answer(&b, a, &q, cands, CANDIDATES), cands, &cands[obj[i]], 1, NULL);
    }
    char cond[16];
    snprintf(cond, sizeof(cond), "K=%u", K);
    emit(&b, "S6", cond, ALL, NARMS);
    bench_free(&b);
}

static void s7(int recur, uint32_t steps) {
    bench b;
    bench_init(&b, RETAIN_S7);
    ecs_mem_digest cands[CANDIDATES];
    pool(cands, 20100, CANDIDATES);
    uint32_t obj[16];
    for (uint32_t i = 0; i < 16; ++i) {
        obj[i] = (uint32_t)(rnd() % CANDIDATES);
        const rel r = rel2(7, ent(20000 + i), cands[obj[i]]);
        observe(&b, &r, 1);
    }
    for (uint32_t t = 1; t <= steps; ++t) {
        b.tick = t;
        const rel r = rel2(7, ent(30000 + t), cands[rnd() % CANDIDATES]);
        observe(&b, &r, 1);
        if (recur && t % 250 == 0 && t < steps)
            for (uint32_t i = 0; i < 16; ++i) {
                const rel x = rel2(7, ent(20000 + i), cands[obj[i]]);
                observe(&b, &x, 1);
            }
        if (t % 25 == 0) consolidate(&b);
    }
    for (uint32_t i = 0; i < 16; ++i) {
        const query q = q_of(7, ROLE_2, 1, ROLE_1, ent(20000 + i), 0, ent(0));
        for (int a = 0; a < NARMS; ++a) score(&b, a, answer(&b, a, &q, cands, CANDIDATES), cands, &cands[obj[i]], 1, NULL);
    }
    emit(&b, "S7", recur ? "RECUR" : "NORECUR", ALL, NARMS);
    bench_free(&b);
}

static void s8(void) {
    bench b;
    bench_init(&b, ECS_MEM_Q16_ONE);
    ecs_mem_digest bpool[CANDIDATES], cpool[CANDIDATES];
    pool(bpool, 40100, CANDIDATES); /* 32 intermediates + 32 distractors */
    pool(cpool, 40200 + 1000, CANDIDATES);
    for (uint32_t i = 0; i < 32; ++i, ++b.tick) {
        const rel r = rel2(81, ent(40000 + i), bpool[i]);
        observe(&b, &r, 1);
    }
    for (uint32_t i = 0; i < 32; ++i, ++b.tick) {
        const rel r = rel2(82, bpool[i], cpool[i]);
        observe(&b, &r, 1);
    }
    for (uint32_t i = 0; i < 32; ++i)
        for (int a = 0; a < NARMS; ++a) {
            const query q1 = q_of(81, ROLE_2, 1, ROLE_1, ent(40000 + i), 0, ent(0));
            const int mid = answer(&b, a, &q1, bpool, CANDIDATES);
            int got = -1;
            if (mid >= 0) {
                const query q2 = q_of(82, ROLE_2, 1, ROLE_1, bpool[mid], 0, ent(0));
                got = answer(&b, a, &q2, cpool, CANDIDATES);
            }
            score(&b, a, got, cpool, &cpool[i], 1, NULL);
        }
    emit(&b, "S8", "two-hop", ALL, NARMS);
    bench_free(&b);
}

static void parameters(void) {
    printf("  \"parameters\": {\"dim\": %d, \"banks\": %d, \"fast_slots\": %d, \"fast_codebook\": %d, "
           "\"candidates\": %d, \"learn_rate_q16\": 65536, \"retain_q16\": 65536, \"retain_s7_q16\": %d, "
           "\"consolidate_rate_q16\": %d, \"consolidate_every\": 25, \"consolidate_items\": 16, "
           "\"keyed_entry_bytes\": %d, \"d31_radius\": %d, \"d31_alpha\": [1, 1], \"d31_beta\": [1, 1], "
           "\"budget_bytes\": %llu},\n",
           DIM, BANKS, FAST_SLOTS, FAST_CODEBOOK, CANDIDATES, RETAIN_S7, CONSOLIDATE_RATE, KEYED_ENTRY_BYTES,
           D31_RADIUS, (unsigned long long)budget_bytes());
}

int main(int argc, char **argv) {
    int smoke = 0;
    if (argc == 2 && !strcmp(argv[1], "--smoke")) smoke = 1;
    else if (argc == 3 && !strcmp(argv[1], "--seed")) g_seed = strtoull(argv[2], NULL, 10);
    else {
        fprintf(stderr, "usage: ecs_mem_lab --seed N | --smoke\n");
        return 2;
    }
    if (smoke) g_seed = 7;
    rng_state = g_seed * UINT64_C(0x2545F4914F6CDD1D);
    printf("{\n  \"schema\": \"elpis.ecs-mem-lab.v1\",\n  \"seed\": %llu,\n  \"smoke\": %s,\n",
           (unsigned long long)g_seed, smoke ? "true" : "false");
    parameters();
    printf("  \"rows\": [");
    if (smoke) {
        s1(16);
        s2("ABAB", 4, 8);
    } else {
        s1(64);
        s2("AB", 2, 32);
        s2("ABA", 3, 32);
        s2("ABAB", 4, 32);
        s3();
        s4();
        s5(0);
        s5(1);
        s6(0);
        s6(256);
        s6(1024);
        s6(4096);
        s7(1, 4000);
        s7(0, 4000);
        s8();
    }
    printf("\n  ]\n}\n");
    return 0;
}
