/* Differential test: the canonical ECS memory (native/ECS, elpis_ecs_mem_*) against the frozen research MF-R0
 * mechanism (ecs_mem_*), its independent reference. Seeded random operation sequences (OBSERVE, LEARN, CONSOLIDATE,
 * QUERY on every channel, restore) are applied to both. With every capability admitted, every status, answer,
 * work count, identity and ELPISEM1 byte must agree exactly; envelopes cross-restore in both directions. With no
 * capability admitted, the canonical handle must agree on every admitted operation and refuse every unqualified
 * one UNQUALIFIED with its identity unchanged. RESEARCH_ONLY test; the canonical library never links research. */
#include "ecs_mem.h"
#include "elpis/ecs_memory.h"
#include "elpis/sha256.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(ecs_mem_observation) == sizeof(elpis_ecs_mem_observation), "observation layout");
_Static_assert(sizeof(ecs_mem_request) == sizeof(elpis_ecs_mem_request), "request layout");
_Static_assert(sizeof(ecs_mem_answer) == sizeof(elpis_ecs_mem_answer), "answer layout");
_Static_assert(sizeof(ecs_mem_slot) == sizeof(elpis_ecs_mem_slot), "slot layout");
_Static_assert(sizeof(ecs_mem_work) == sizeof(elpis_ecs_mem_work), "work layout");

static uint64_t rng_state;
static uint64_t rnd(void) {
    uint64_t z = (rng_state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
static void digest_of(const char *tag, uint64_t i, uint8_t out[32]) {
    char buf[64];
    const int n = snprintf(buf, sizeof(buf), "%s:%llu", tag, (unsigned long long)i);
    elpis_sha256(buf, (size_t)n, out);
}

static size_t snap_research(const ecs_mem *m, uint8_t **out) {
    const size_t n = ecs_mem_snapshot_size(m);
    size_t w = 0;
    *out = malloc(n);
    assert(*out && ecs_mem_snapshot(m, *out, n, &w) == ECS_MEM_OK && w == n);
    return n;
}
static size_t snap_canonical(const elpis_ecs_mem *m, uint8_t **out) {
    const size_t n = elpis_ecs_mem_snapshot_size(m);
    size_t w = 0;
    *out = malloc(n);
    assert(*out && elpis_ecs_mem_snapshot(m, *out, n, &w) == ELPIS_ECS_MEM_OK && w == n);
    return n;
}
static void same_identity(const ecs_mem *r, const elpis_ecs_mem *c) {
    ecs_mem_digest a;
    elpis_ecs_mem_digest b;
    assert(ecs_mem_identity(r, &a) == ECS_MEM_OK && elpis_ecs_mem_identity(c, &b) == ELPIS_ECS_MEM_OK);
    assert(!memcmp(a.b, b.b, 32));
}

typedef struct {
    unsigned long ops, observes, learns, consolidates, queries, refusals, unqualified, restores;
} tally;

/* One random world. full: the canonical handle admits every capability (exact equality required). */
static void world(uint64_t seed, int full, tally *t) {
    rng_state = seed;
    ecs_mem_config rc;
    ecs_mem_config_default(&rc);
    rc.dim = (rnd() % 4) ? 64 : 128;
    rc.banks = 1 + (uint32_t)(rnd() % 4);
    rc.slot_capacity = 4 + (uint32_t)(rnd() % 12);
    rc.codebook_capacity = 8 + (uint32_t)(rnd() % 12);
    rc.slot_policy = (uint32_t)(rnd() % 2);
    rc.codebook_policy = (uint32_t)(rnd() % 2);
    rc.law = (uint32_t)(rnd() % 2);
    rc.learn_rate_q16 = 1 + (uint32_t)(rnd() % 65536);
    rc.retain_q16 = 60000 + (uint32_t)(rnd() % 5537);
    rc.consolidate_rate_q16 = 1 + (uint32_t)(rnd() % 65536);
    elpis_ecs_mem_config cc;
    elpis_ecs_mem_config_default(&cc);
    cc.dim = rc.dim;
    cc.banks = rc.banks;
    cc.slot_capacity = rc.slot_capacity;
    cc.codebook_capacity = rc.codebook_capacity;
    cc.slot_policy = rc.slot_policy;
    cc.codebook_policy = rc.codebook_policy;
    cc.law = rc.law;
    cc.learn_rate_q16 = rc.learn_rate_q16;
    cc.retain_q16 = rc.retain_q16;
    cc.consolidate_rate_q16 = rc.consolidate_rate_q16;
    cc.byte_budget = rc.byte_budget;
    cc.capabilities = full ? (ELPIS_ECS_MEM_CAP_RECURRENT | ELPIS_ECS_MEM_CAP_CONSOLIDATE) : 0;

    ecs_mem *r = NULL;
    elpis_ecs_mem *c = NULL;
    assert(ecs_mem_create(&rc, &r) == ECS_MEM_OK && elpis_ecs_mem_create(&cc, &c) == ELPIS_ECS_MEM_OK);
    same_identity(r, c);

    ecs_mem_digest keys[64];
    unsigned nkeys = 0;
    uint64_t tick = 1;
    for (int step = 0; step < 300; ++step) {
        t->ops++;
        tick += rnd() % 3;
        const uint64_t op = rnd() % 10;
        if (op < 5) { /* OBSERVE */
            ecs_mem_observation o;
            memset(&o, 0, sizeof(o));
            o.abi_version = ECS_MEM_ABI_V1;
            o.relation_type = 1 + (uint32_t)(rnd() % 3);
            o.polarity = 1 + (uint32_t)(rnd() % 3);
            o.participant_count = 1 + (uint32_t)(rnd() % 3);
            o.tick = (rnd() % 17 == 0 && tick > 1) ? tick - 1 : tick; /* occasional stale tick */
            o.source_authority = (uint32_t)(rnd() % 5);
            digest_of("provenance", rnd() % 50, o.provenance.b);
            digest_of("structure", rnd() % 50, o.structure_ref.b);
            for (uint32_t p = 0; p < o.participant_count; ++p) {
                o.participants[p].role = 1 + p;
                o.participants[p].ordinal = 0;
                digest_of("entity", rnd() % 24, o.participants[p].entity.b);
            }
            elpis_ecs_mem_observation co;
            memcpy(&co, &o, sizeof(co));
            ecs_mem_digest k1;
            elpis_ecs_mem_digest k2;
            ecs_mem_work w1;
            elpis_ecs_mem_work w2;
            memset(&w1, 0, sizeof(w1));
            memset(&w2, 0, sizeof(w2));
            const int s1 = ecs_mem_observe(r, 0, &o, &k1, &w1);
            const int s2 = elpis_ecs_mem_observe(c, 0, &co, &k2, &w2);
            assert(s1 == s2);
            t->observes++;
            if (s1 == ECS_MEM_OK) {
                assert(!memcmp(k1.b, k2.b, 32) && !memcmp(&w1, &w2, sizeof(w1)));
                if (nkeys < 64) keys[nkeys++] = k1;
                else keys[rnd() % 64] = k1;
            } else {
                t->refusals++;
            }
        } else if (op < 7 && nkeys) { /* LEARN */
            const ecs_mem_digest k = keys[rnd() % nkeys];
            elpis_ecs_mem_digest ck;
            memcpy(ck.b, k.b, 32);
            ecs_mem_work w1;
            elpis_ecs_mem_work w2;
            memset(&w1, 0, sizeof(w1));
            memset(&w2, 0, sizeof(w2));
            const int s1 = ecs_mem_learn(r, 0, &k, tick, &w1);
            const int s2 = elpis_ecs_mem_learn(c, 0, &ck, tick, &w2);
            t->learns++;
            if (full) {
                assert(s1 == s2 && !memcmp(&w1, &w2, sizeof(w1)));
                if (s1 != ECS_MEM_OK) t->refusals++;
            } else {
                assert(s2 == ELPIS_ECS_MEM_UNQUALIFIED);
                t->unqualified++;
                if (s1 == ECS_MEM_OK) goto resync; /* the reference learned; the canonical handle may not */
            }
        } else if (op < 8) { /* CONSOLIDATE */
            const uint32_t items = 1 + (uint32_t)(rnd() % 4);
            uint32_t n1 = 0, n2 = 0;
            const int s1 = ecs_mem_consolidate(r, 0, tick, items, &n1, NULL);
            const int s2 = elpis_ecs_mem_consolidate(c, 0, tick, items, &n2, NULL);
            t->consolidates++;
            if (full) {
                assert(s1 == s2 && n1 == n2);
                if (s1 != ECS_MEM_OK) t->refusals++;
            } else {
                assert(s2 == ELPIS_ECS_MEM_UNQUALIFIED);
                t->unqualified++;
                if (s1 == ECS_MEM_OK) goto resync;
            }
        } else if (op < 9) { /* QUERY */
            ecs_mem_request q;
            memset(&q, 0, sizeof(q));
            q.abi_version = ECS_MEM_ABI_V1;
            q.channel = 1 + (uint32_t)(rnd() % 3);
            q.relation_type = 1 + (uint32_t)(rnd() % 3);
            q.target_role = 1 + (uint32_t)(rnd() % 3);
            q.candidate_source = q.channel == ECS_MEM_LOOKUP ? ECS_MEM_CANDIDATES_CALLER : ECS_MEM_CANDIDATES_CODEBOOK;
            q.cue_count = (uint32_t)(rnd() % 3);
            for (uint32_t i = 0, role = 1; i < q.cue_count; ++role) {
                if (role == q.target_role) continue;
                q.cue[i].role = role;
                digest_of("entity", rnd() % 24, q.cue[i].entity.b);
                ++i;
            }
            elpis_ecs_mem_request cq;
            memcpy(&cq, &q, sizeof(cq));
            ecs_mem_answer a1;
            elpis_ecs_mem_answer a2;
            memset(&a1, 0, sizeof(a1));
            memset(&a2, 0, sizeof(a2));
            const int s1 = ecs_mem_query(r, &q, &a1);
            const int s2 = elpis_ecs_mem_query(c, &cq, &a2);
            t->queries++;
            if (full || q.channel == ECS_MEM_LOOKUP) {
                assert(s1 == s2);
                if (s1 == ECS_MEM_OK) assert(!memcmp(&a1, &a2, sizeof(a1)));
            } else {
                assert(s2 == ELPIS_ECS_MEM_UNQUALIFIED);
                t->unqualified++;
            }
        } else { /* restore round trip through the other implementation */
            uint8_t *rb = NULL, *cb = NULL;
            const size_t rn = snap_research(r, &rb);
            const size_t cn = snap_canonical(c, &cb);
            assert(rn == cn && !memcmp(rb, cb, rn));
            assert(elpis_ecs_mem_restore_into(c, 0, rb, rn) == ELPIS_ECS_MEM_OK);
            assert(ecs_mem_restore_into(r, 0, cb, cn) == ECS_MEM_OK);
            t->restores++;
            free(rb);
            free(cb);
        }
        if (step % 8 == 7) same_identity(r, c); /* full-state hashes dominate the cost; every 8th step */
        continue;
    resync:
        /* Admitted-only arm: the reference applied a law the canonical handle refused (its state is unchanged
         * by the refusal law). Both are rebuilt from the canonical state, so they continue from equal states and
         * equal (non-authoritative) index layouts, which the work counts compared above depend on. */
        {
            uint8_t *cb = NULL;
            const size_t cn = snap_canonical(c, &cb);
            assert(ecs_mem_restore_into(r, 0, cb, cn) == ECS_MEM_OK);
            assert(elpis_ecs_mem_restore_into(c, 0, cb, cn) == ELPIS_ECS_MEM_OK);
            free(cb);
        }
        same_identity(r, c);
    }
    uint8_t *rb = NULL, *cb = NULL;
    const size_t rn = snap_research(r, &rb);
    const size_t cn = snap_canonical(c, &cb);
    assert(rn == cn && !memcmp(rb, cb, rn));
    free(rb);
    free(cb);
    ecs_mem_destroy(r);
    elpis_ecs_mem_destroy(c);
}

int main(void) {
    tally full = {0}, admitted = {0};
    for (uint64_t seed = 1; seed <= 20; ++seed) world(seed, 1, &full);
    for (uint64_t seed = 101; seed <= 120; ++seed) world(seed, 0, &admitted);
    assert(full.learns > 100 && full.consolidates > 50 && full.queries > 100 && full.restores > 100);
    assert(admitted.unqualified > 100);
    printf("canonical == research: full %lu ops (%lu observe, %lu learn, %lu consolidate, %lu query, %lu restore, "
           "%lu refusals); admitted-only %lu ops, %lu UNQUALIFIED refusals\n",
           full.ops, full.observes, full.learns, full.consolidates, full.queries, full.restores, full.refusals,
           admitted.ops, admitted.unqualified);
    return 0;
}
