/* Canonical ECS memory mechanics: contract, refusal atomicity, capacity, recency, lease, channels. */
#include "ecs_memory_test_util.h"

#include <stdint.h>

/* The identity of the script below (observe + learn, no slow-bank operation). The frozen research reference
 * (research/ecs_memory_mf_r0, ecs_mem.c) produces the same identity for the same script. */
#ifndef ELPIS_ECS_MEM_KNOWN_ANSWER
#define ELPIS_ECS_MEM_KNOWN_ANSWER "9b0e06c335467414a0afe81c19b4ed95b382d6a811622381da3790b6eb8162fc"
#endif

static void test_config(void) {
    elpis_ecs_mem_config c;
    elpis_ecs_mem *m = NULL;
    uint64_t arena = 0, state = 0;
    elpis_ecs_mem_config_default(&c);
    assert(elpis_ecs_mem_required_bytes(&c, &arena, &state) == ELPIS_ECS_MEM_OK && arena > 0 && state > 0);
    assert(elpis_ecs_mem_create(&c, &m) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_stats s;
    assert(elpis_ecs_mem_inspect(m, &s) == ELPIS_ECS_MEM_OK);
    assert(s.arena_bytes == arena && s.allocated_bytes > arena && s.slots_used == 0 && s.clock == 0);
    assert(s.state_bytes == elpis_ecs_mem_snapshot_size(m));
    elpis_ecs_mem_destroy(m);

    struct { size_t off; uint32_t bad; } cases[] = {
        {offsetof(elpis_ecs_mem_config, abi_version), 2}, {offsetof(elpis_ecs_mem_config, dim), 32},
        {offsetof(elpis_ecs_mem_config, dim), 96}, {offsetof(elpis_ecs_mem_config, dim), 2048},
        {offsetof(elpis_ecs_mem_config, banks), 0}, {offsetof(elpis_ecs_mem_config, banks), 65},
        {offsetof(elpis_ecs_mem_config, slot_capacity), 0}, {offsetof(elpis_ecs_mem_config, slot_capacity), 65537},
        {offsetof(elpis_ecs_mem_config, codebook_capacity), 7}, {offsetof(elpis_ecs_mem_config, slot_policy), 2},
        {offsetof(elpis_ecs_mem_config, codebook_policy), 9}, {offsetof(elpis_ecs_mem_config, law), 2},
        {offsetof(elpis_ecs_mem_config, learn_rate_q16), 0}, {offsetof(elpis_ecs_mem_config, learn_rate_q16), 65537},
        {offsetof(elpis_ecs_mem_config, retain_q16), 0}, {offsetof(elpis_ecs_mem_config, slow_rate_q16), 70000},
        {offsetof(elpis_ecs_mem_config, capabilities), 4},
        {offsetof(elpis_ecs_mem_config, capabilities), 2},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        elpis_ecs_mem_config b = c;
        memcpy((uint8_t *)&b + cases[i].off, &cases[i].bad, 4);
        m = (elpis_ecs_mem *)(uintptr_t)1;
        assert(elpis_ecs_mem_create(&b, &m) == ELPIS_ECS_MEM_INVALID);
        assert(m == (elpis_ecs_mem *)(uintptr_t)1); /* output untouched */
        assert(elpis_ecs_mem_required_bytes(&b, &arena, &state) == ELPIS_ECS_MEM_INVALID);
    }
    elpis_ecs_mem_config b = c;
    assert(elpis_ecs_mem_required_bytes(&b, &arena, &state) == ELPIS_ECS_MEM_OK);
    b.byte_budget = arena - 1;
    assert(elpis_ecs_mem_create(&b, &m) == ELPIS_ECS_MEM_CAPACITY);
    b.byte_budget = arena;
    assert(elpis_ecs_mem_create(&b, &m) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_destroy(m);
    /* Every admissible shape fits ELPIS_ECS_MEM_MAX_ARENA_BYTES (the cap is defensive); the budget binds first. */
    b.dim = ELPIS_ECS_MEM_MAX_DIM;
    b.banks = ELPIS_ECS_MEM_MAX_BANKS;
    b.slot_capacity = ELPIS_ECS_MEM_MAX_SLOTS;
    b.codebook_capacity = ELPIS_ECS_MEM_MAX_CODEBOOK;
    assert(elpis_ecs_mem_required_bytes(&b, &arena, &state) == ELPIS_ECS_MEM_OK && arena <= ELPIS_ECS_MEM_MAX_ARENA_BYTES);
    b.byte_budget = arena - 1;
    assert(elpis_ecs_mem_create(&b, &m) == ELPIS_ECS_MEM_CAPACITY);
    assert(elpis_ecs_mem_create(NULL, &m) == ELPIS_ECS_MEM_INVALID && elpis_ecs_mem_create(&c, NULL) == ELPIS_ECS_MEM_INVALID);
    elpis_ecs_mem_destroy(NULL);
}

static void test_observe_identity_and_reconstruction(void) {
    elpis_ecs_mem_config c = tu_config(64, 16, 16);
    elpis_ecs_mem *m;
    assert(elpis_ecs_mem_create(&c, &m) == ELPIS_ECS_MEM_OK);
    /* Ternary give(agent=1, recipient=2, object=3) presented in a non-canonical order. */
    const uint32_t roles[3] = {3, 1, 2};
    const uint64_t ents[3] = {30, 10, 20};
    elpis_ecs_mem_observation o = tu_obs(5, ELPIS_ECS_MEM_ASSERT, 10, 3, roles, ents);
    elpis_ecs_mem_digest key, key2, sentinel;
    memset(&sentinel, 0xAB, sizeof(sentinel));
    elpis_ecs_mem_work w;
    assert(elpis_ecs_mem_observe(m, 0, &o, &key, &w) == ELPIS_ECS_MEM_OK);
    assert(w.sha_blocks > 0 && w.index_probes > 0);
    assert(elpis_ecs_mem_relation_key(&o, &key2) == ELPIS_ECS_MEM_OK && tu_same(key, key2));
    /* Permuting the presentation does not change the relation identity. */
    const uint32_t roles_b[3] = {1, 2, 3};
    const uint64_t ents_b[3] = {10, 20, 30};
    elpis_ecs_mem_observation ob = tu_obs(5, ELPIS_ECS_MEM_ASSERT, 11, 3, roles_b, ents_b);
    assert(elpis_ecs_mem_relation_key(&ob, &key2) == ELPIS_ECS_MEM_OK && tu_same(key, key2));
    /* Orientation: swapping who fills which role is a different relation. */
    const uint64_t ents_swapped[3] = {20, 10, 30};
    elpis_ecs_mem_observation os = tu_obs(5, ELPIS_ECS_MEM_ASSERT, 11, 3, roles_b, ents_swapped);
    assert(elpis_ecs_mem_relation_key(&os, &key2) == ELPIS_ECS_MEM_OK && !tu_same(key, key2));
    /* Payload is part of the identity; provenance and structure_ref are not. */
    elpis_ecs_mem_observation op = ob;
    op.payload = tu_digest("payload", 1);
    assert(elpis_ecs_mem_relation_key(&op, &key2) == ELPIS_ECS_MEM_OK && !tu_same(key, key2));
    op = ob;
    op.provenance = tu_digest("other provenance", 1);
    assert(elpis_ecs_mem_relation_key(&op, &key2) == ELPIS_ECS_MEM_OK && tu_same(key, key2));

    assert(elpis_ecs_mem_observe(m, 0, &ob, NULL, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_observation neg = ob;
    neg.polarity = ELPIS_ECS_MEM_NEGATE;
    neg.tick = 12;
    neg.provenance = tu_digest("dissent", 1);
    assert(elpis_ecs_mem_observe(m, 0, &neg, NULL, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_slot s;
    assert(elpis_ecs_mem_slot_read(m, &key, &s) == ELPIS_ECS_MEM_OK);
    assert(s.participant_count == 3 && s.relation_type == 5);
    for (uint32_t i = 0; i < 3; ++i) {
        assert(s.participants[i].role == i + 1u);
        assert(tu_same(s.participants[i].entity, tu_entity(10u * (i + 1u))));
    }
    assert(s.assert_count == 2 && s.negate_count == 1 && s.uncertain_count == 0);
    assert(s.first_tick == 10 && s.last_tick == 12 && s.last_assert_tick == 11 && s.last_negate_tick == 12);
    assert(tu_same(s.first_provenance, tu_digest("provenance", 10)));
    assert(tu_same(s.last_provenance, tu_digest("dissent", 1)));
    assert(s.last_polarity == ELPIS_ECS_MEM_NEGATE && s.last_authority == 7);

    /* Multiplicity: one entity in two roles, and two ordinals of one role. */
    elpis_ecs_mem_observation mult = tu_bin(6, 40, 40, ELPIS_ECS_MEM_ASSERT, 12);
    assert(elpis_ecs_mem_observe(m, 0, &mult, &key2, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_observation ords = tu_bin(7, 41, 42, ELPIS_ECS_MEM_ASSERT, 12);
    ords.participants[1].role = 1;
    ords.participants[1].ordinal = 1;
    assert(elpis_ecs_mem_observe(m, 0, &ords, NULL, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_stats st;
    assert(elpis_ecs_mem_inspect(m, &st) == ELPIS_ECS_MEM_OK);
    assert(st.slots_used == 3 && st.codebook_used == 6 && st.observes == 5 && st.repeated_observations == 2);
    int has = 0;
    assert(elpis_ecs_mem_codebook_contains(m, &mult.participants[0].entity, &has) == ELPIS_ECS_MEM_OK && has == 1);

    /* Refusals: identity, key and work outputs unchanged. */
    const elpis_ecs_mem_digest before = tu_identity(m);
    elpis_ecs_mem_digest out = sentinel;
    elpis_ecs_mem_work wsent;
    memset(&wsent, 0x5A, sizeof(wsent));
    elpis_ecs_mem_work wout = wsent;
    elpis_ecs_mem_observation bad;
    bad = ob; bad.tick = 3;                         /* older than the clock */
    assert(elpis_ecs_mem_observe(m, 0, &bad, &out, &wout) == ELPIS_ECS_MEM_CLOCK);
    bad = ob; bad.polarity = 0;
    assert(elpis_ecs_mem_observe(m, 0, &bad, &out, &wout) == ELPIS_ECS_MEM_INVALID);
    bad = ob; bad.polarity = 4;
    assert(elpis_ecs_mem_observe(m, 0, &bad, &out, &wout) == ELPIS_ECS_MEM_INVALID);
    bad = ob; bad.participant_count = 0;
    assert(elpis_ecs_mem_observe(m, 0, &bad, &out, &wout) == ELPIS_ECS_MEM_INVALID);
    bad = ob; bad.participant_count = 9;
    assert(elpis_ecs_mem_observe(m, 0, &bad, &out, &wout) == ELPIS_ECS_MEM_INVALID);
    bad = ob; bad.participants[1].role = bad.participants[0].role; /* duplicate (role, ordinal) */
    assert(elpis_ecs_mem_observe(m, 0, &bad, &out, &wout) == ELPIS_ECS_MEM_INVALID);
    bad = ob; bad.participants[5].role = 1;          /* data past participant_count */
    assert(elpis_ecs_mem_observe(m, 0, &bad, &out, &wout) == ELPIS_ECS_MEM_INVALID);
    bad = ob; bad.reserved = 1;
    assert(elpis_ecs_mem_observe(m, 0, &bad, &out, &wout) == ELPIS_ECS_MEM_INVALID);
    bad = ob; bad.abi_version = 9;
    assert(elpis_ecs_mem_observe(m, 0, &bad, &out, &wout) == ELPIS_ECS_MEM_INVALID);
    assert(elpis_ecs_mem_observe(m, 0, NULL, &out, &wout) == ELPIS_ECS_MEM_INVALID);
    assert(elpis_ecs_mem_observe(m, 1, &ob, &out, &wout) == ELPIS_ECS_MEM_LEASED); /* no lease is held */
    assert(tu_same(out, sentinel) && !memcmp(&wout, &wsent, sizeof(wout)));
    assert(tu_same(before, tu_identity(m)));
    assert(elpis_ecs_mem_inspect(m, &st) == ELPIS_ECS_MEM_OK && st.refusals == 11);
    assert(st.refusals_by_code[1] == 9 && st.refusals_by_code[4] == 1 && st.refusals_by_code[6] == 1);
    elpis_ecs_mem_destroy(m);
}

static void test_slot_capacity_and_recency(void) {
    /* REFUSE */
    elpis_ecs_mem_config c = tu_config(64, 3, 64);
    c.slot_policy = ELPIS_ECS_MEM_REFUSE;
    elpis_ecs_mem *m;
    assert(elpis_ecs_mem_create(&c, &m) == ELPIS_ECS_MEM_OK);
    for (uint64_t i = 0; i < 3; ++i) {
        elpis_ecs_mem_observation o = tu_bin(1, i, 100 + i, ELPIS_ECS_MEM_ASSERT, i);
        assert(elpis_ecs_mem_observe(m, 0, &o, NULL, NULL) == ELPIS_ECS_MEM_OK);
    }
    elpis_ecs_mem_digest before = tu_identity(m);
    elpis_ecs_mem_observation extra = tu_bin(1, 9, 109, ELPIS_ECS_MEM_ASSERT, 5);
    assert(elpis_ecs_mem_observe(m, 0, &extra, NULL, NULL) == ELPIS_ECS_MEM_CAPACITY);
    assert(tu_same(before, tu_identity(m)));
    elpis_ecs_mem_observation again = tu_bin(1, 0, 100, ELPIS_ECS_MEM_ASSERT, 5); /* resident: admitted */
    assert(elpis_ecs_mem_observe(m, 0, &again, NULL, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_destroy(m);

    /* EVICT_LRU by write recency; reads (slot_read, QUERY) never refresh. */
    c.slot_policy = ELPIS_ECS_MEM_EVICT_LRU;
    assert(elpis_ecs_mem_create(&c, &m) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_digest k[5];
    for (uint64_t i = 0; i < 3; ++i) {
        elpis_ecs_mem_observation o = tu_bin(1, i, 100 + i, ELPIS_ECS_MEM_ASSERT, i);
        assert(elpis_ecs_mem_observe(m, 0, &o, &k[i], NULL) == ELPIS_ECS_MEM_OK);
    }
    elpis_ecs_mem_observation a0 = tu_bin(1, 0, 100, ELPIS_ECS_MEM_ASSERT, 3); /* write-touch slot 0 */
    assert(elpis_ecs_mem_observe(m, 0, &a0, NULL, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_slot s;
    for (int r = 0; r < 4; ++r) assert(elpis_ecs_mem_slot_read(m, &k[1], &s) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_request q = tu_query(ELPIS_ECS_MEM_LOOKUP, 1, 2);
    tu_cue(&q, 1, 1);
    elpis_ecs_mem_answer ans;
    assert(elpis_ecs_mem_query(m, &q, &ans) == ELPIS_ECS_MEM_OK && ans.found);
    elpis_ecs_mem_observation d = tu_bin(1, 3, 103, ELPIS_ECS_MEM_ASSERT, 4);
    assert(elpis_ecs_mem_observe(m, 0, &d, &k[3], NULL) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_slot_read(m, &k[1], &s) == ELPIS_ECS_MEM_NOT_FOUND); /* the least recently WRITTEN */
    assert(elpis_ecs_mem_slot_read(m, &k[0], &s) == ELPIS_ECS_MEM_OK && elpis_ecs_mem_slot_read(m, &k[2], &s) == ELPIS_ECS_MEM_OK);
    /* LEARN is a write use: it refreshes slot 2, so slot 0 is the next victim. */
    assert(elpis_ecs_mem_learn(m, 0, &k[2], 5, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_observation e = tu_bin(1, 4, 104, ELPIS_ECS_MEM_ASSERT, 6);
    assert(elpis_ecs_mem_observe(m, 0, &e, &k[4], NULL) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_slot_read(m, &k[0], &s) == ELPIS_ECS_MEM_NOT_FOUND);
    assert(elpis_ecs_mem_slot_read(m, &k[2], &s) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_stats st;
    assert(elpis_ecs_mem_inspect(m, &st) == ELPIS_ECS_MEM_OK && st.slot_evictions == 2 && st.slots_used == 3);
    elpis_ecs_mem_destroy(m);
}

static void test_codebook_capacity(void) {
    elpis_ecs_mem_config c = tu_config(64, 64, 8);
    c.codebook_policy = ELPIS_ECS_MEM_REFUSE;
    elpis_ecs_mem *m;
    assert(elpis_ecs_mem_create(&c, &m) == ELPIS_ECS_MEM_OK);
    const uint32_t roles[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const uint64_t ents[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    elpis_ecs_mem_observation full = tu_obs(1, ELPIS_ECS_MEM_ASSERT, 1, 8, roles, ents);
    assert(elpis_ecs_mem_observe(m, 0, &full, NULL, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_digest before = tu_identity(m);
    elpis_ecs_mem_observation more = tu_bin(2, 1, 99, ELPIS_ECS_MEM_ASSERT, 2);
    assert(elpis_ecs_mem_observe(m, 0, &more, NULL, NULL) == ELPIS_ECS_MEM_CAPACITY);
    assert(tu_same(before, tu_identity(m)));
    elpis_ecs_mem_destroy(m);

    c.codebook_policy = ELPIS_ECS_MEM_EVICT_LRU;
    assert(elpis_ecs_mem_create(&c, &m) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_observe(m, 0, &full, NULL, NULL) == ELPIS_ECS_MEM_OK);
    /* Entity 1 (least recent) is also a participant of the new observation: it must survive; the victim is 2. */
    more = tu_bin(2, 1, 99, ELPIS_ECS_MEM_ASSERT, 2);
    assert(elpis_ecs_mem_observe(m, 0, &more, NULL, NULL) == ELPIS_ECS_MEM_OK);
    int has = 0;
    elpis_ecs_mem_digest e1 = tu_entity(1), e2 = tu_entity(2), e99 = tu_entity(99);
    assert(elpis_ecs_mem_codebook_contains(m, &e1, &has) == ELPIS_ECS_MEM_OK && has);
    assert(elpis_ecs_mem_codebook_contains(m, &e2, &has) == ELPIS_ECS_MEM_OK && !has);
    assert(elpis_ecs_mem_codebook_contains(m, &e99, &has) == ELPIS_ECS_MEM_OK && has);
    /* Eight distinct new entities replace eight old ones in LRU order. */
    const uint64_t fresh[8] = {11, 12, 13, 14, 15, 16, 17, 18};
    elpis_ecs_mem_observation f = tu_obs(3, ELPIS_ECS_MEM_ASSERT, 3, 8, roles, fresh);
    assert(elpis_ecs_mem_observe(m, 0, &f, NULL, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_stats st;
    assert(elpis_ecs_mem_inspect(m, &st) == ELPIS_ECS_MEM_OK && st.codebook_used == 8 && st.codebook_evictions == 9);
    elpis_ecs_mem_destroy(m);
}

static void test_lease(void) {
    elpis_ecs_mem_config c = tu_config(64, 8, 8);
    elpis_ecs_mem *m;
    assert(elpis_ecs_mem_create(&c, &m) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_observation o = tu_bin(1, 1, 2, ELPIS_ECS_MEM_ASSERT, 1);
    assert(elpis_ecs_mem_lease_claim(m, 0) == ELPIS_ECS_MEM_INVALID);
    assert(elpis_ecs_mem_lease_claim(m, 77) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_digest before = tu_identity(m);
    assert(elpis_ecs_mem_observe(m, 0, &o, NULL, NULL) == ELPIS_ECS_MEM_LEASED);
    assert(elpis_ecs_mem_observe(m, 78, &o, NULL, NULL) == ELPIS_ECS_MEM_LEASED);
    assert(tu_same(before, tu_identity(m)));
    elpis_ecs_mem_digest key;
    assert(elpis_ecs_mem_observe(m, 77, &o, &key, NULL) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_learn(m, 0, &key, 2, NULL) == ELPIS_ECS_MEM_LEASED);
    assert(elpis_ecs_mem_learn(m, 77, &key, 2, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_stats st;
    assert(elpis_ecs_mem_inspect(m, &st) == ELPIS_ECS_MEM_OK && st.lease_held == 1);
    /* The lease is not state: the identity does not depend on it. */
    before = tu_identity(m);
    assert(elpis_ecs_mem_lease_release(m, 76) == ELPIS_ECS_MEM_LEASED);
    assert(elpis_ecs_mem_lease_release(m, 77) == ELPIS_ECS_MEM_OK);
    assert(tu_same(before, tu_identity(m)));
    assert(elpis_ecs_mem_inspect(m, &st) == ELPIS_ECS_MEM_OK && st.lease_held == 0);
    o.tick = 2;
    assert(elpis_ecs_mem_observe(m, 0, &o, NULL, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_destroy(m);
}

static void test_learn_and_query(void) {
    elpis_ecs_mem_config c = tu_config(256, 64, 64);
    elpis_ecs_mem *m;
    assert(elpis_ecs_mem_create(&c, &m) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_digest kA, kB, kU, kT, sent;
    memset(&sent, 0x11, sizeof(sent));
    elpis_ecs_mem_observation a = tu_bin(1, 7, 100, ELPIS_ECS_MEM_ASSERT, 1);
    elpis_ecs_mem_observation b = tu_bin(1, 7, 200, ELPIS_ECS_MEM_ASSERT, 2);
    assert(elpis_ecs_mem_observe(m, 0, &a, &kA, NULL) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_observe(m, 0, &b, &kB, NULL) == ELPIS_ECS_MEM_OK);

    /* OBSERVE alone never changes the banks: FAST holds no signal. */
    elpis_ecs_mem_request q = tu_query(ELPIS_ECS_MEM_FAST, 1, 2);
    tu_cue(&q, 1, 7);
    elpis_ecs_mem_answer ans;
    assert(elpis_ecs_mem_query(m, &q, &ans) == ELPIS_ECS_MEM_OK && ans.signal == 0 && ans.matches == 3);

    /* LEARN refusals. */
    elpis_ecs_mem_digest before = tu_identity(m);
    elpis_ecs_mem_digest missing = tu_digest("missing", 0);
    elpis_ecs_mem_work wsent, wout;
    memset(&wsent, 0x33, sizeof(wsent));
    wout = wsent;
    assert(elpis_ecs_mem_learn(m, 0, &missing, 3, &wout) == ELPIS_ECS_MEM_NOT_FOUND);
    assert(elpis_ecs_mem_learn(m, 0, &kA, 1, &wout) == ELPIS_ECS_MEM_CLOCK);
    assert(elpis_ecs_mem_learn(m, 0, NULL, 3, &wout) == ELPIS_ECS_MEM_INVALID);
    elpis_ecs_mem_observation u = tu_bin(1, 8, 300, ELPIS_ECS_MEM_UNCERTAIN, 3);
    assert(elpis_ecs_mem_observe(m, 0, &u, &kU, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_observation t1 = tu_bin(1, 9, 400, ELPIS_ECS_MEM_ASSERT, 3), t2 = t1;
    t2.polarity = ELPIS_ECS_MEM_NEGATE;
    assert(elpis_ecs_mem_observe(m, 0, &t1, &kT, NULL) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_observe(m, 0, &t2, NULL, NULL) == ELPIS_ECS_MEM_OK);
    before = tu_identity(m);
    assert(elpis_ecs_mem_learn(m, 0, &kU, 4, &wout) == ELPIS_ECS_MEM_NO_NET_EVIDENCE);
    assert(elpis_ecs_mem_learn(m, 0, &kT, 4, &wout) == ELPIS_ECS_MEM_NO_NET_EVIDENCE);
    assert(!memcmp(&wout, &wsent, sizeof(wout)) && tu_same(before, tu_identity(m)));

    /* A -> B -> A on the fast channel (delta law, full rate). */
    elpis_ecs_mem_digest cands[2] = {tu_entity(100), tu_entity(200)};
    q.candidate_source = ELPIS_ECS_MEM_CANDIDATES_CALLER;
    q.candidates = cands;
    q.candidate_count = 2;
    assert(elpis_ecs_mem_learn(m, 0, &kA, 4, &wout) == ELPIS_ECS_MEM_OK && wout.mac_ops > 0);
    assert(elpis_ecs_mem_query(m, &q, &ans) == ELPIS_ECS_MEM_OK && ans.found && tu_same(ans.answer, cands[0]));
    assert(ans.signal > 0 && ans.score > ans.runner_up);
    assert(elpis_ecs_mem_learn(m, 0, &kB, 5, NULL) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_query(m, &q, &ans) == ELPIS_ECS_MEM_OK && tu_same(ans.answer, cands[1]));
    assert(elpis_ecs_mem_learn(m, 0, &kA, 6, NULL) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_query(m, &q, &ans) == ELPIS_ECS_MEM_OK && tu_same(ans.answer, cands[0]));
    /* The reverse direction (object -> subject) is learned by the same LEARN. */
    elpis_ecs_mem_request r = tu_query(ELPIS_ECS_MEM_FAST, 1, 1);
    tu_cue(&r, 2, 200);
    elpis_ecs_mem_digest subj[2] = {tu_entity(7), tu_entity(8)};
    r.candidate_source = ELPIS_ECS_MEM_CANDIDATES_CALLER;
    r.candidates = subj;
    r.candidate_count = 2;
    assert(elpis_ecs_mem_query(m, &r, &ans) == ELPIS_ECS_MEM_OK && tu_same(ans.answer, subj[0]));

    /* QUERY is read only: identity and every telemetry counter unchanged. */
    elpis_ecs_mem_stats s1, s2;
    before = tu_identity(m);
    assert(elpis_ecs_mem_inspect(m, &s1) == ELPIS_ECS_MEM_OK);
    for (int i = 0; i < 5; ++i) assert(elpis_ecs_mem_query(m, &q, &ans) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_inspect(m, &s2) == ELPIS_ECS_MEM_OK && !memcmp(&s1, &s2, sizeof(s1)));
    assert(tu_same(before, tu_identity(m)));

    /* QUERY refusals leave the answer untouched. */
    elpis_ecs_mem_answer sentinel_answer;
    memset(&sentinel_answer, 0x77, sizeof(sentinel_answer));
    ans = sentinel_answer;
    elpis_ecs_mem_request bad = q;
    bad.channel = 9;
    assert(elpis_ecs_mem_query(m, &bad, &ans) == ELPIS_ECS_MEM_INVALID);
    bad = q; bad.candidate_count = 0;
    assert(elpis_ecs_mem_query(m, &bad, &ans) == ELPIS_ECS_MEM_INVALID);
    bad = q; bad.candidates = NULL;
    assert(elpis_ecs_mem_query(m, &bad, &ans) == ELPIS_ECS_MEM_INVALID);
    bad = q; bad.cue[0].role = 2; /* the cue names the target */
    assert(elpis_ecs_mem_query(m, &bad, &ans) == ELPIS_ECS_MEM_INVALID);
    bad = q; bad.cue_count = 8;
    assert(elpis_ecs_mem_query(m, &bad, &ans) == ELPIS_ECS_MEM_INVALID);
    bad = tu_query(ELPIS_ECS_MEM_LOOKUP, 1, 2); bad.candidate_count = 1; bad.candidates = cands;
    assert(elpis_ecs_mem_query(m, &bad, &ans) == ELPIS_ECS_MEM_INVALID);
    assert(!memcmp(&ans, &sentinel_answer, sizeof(ans)));

    /* Codebook candidates name the answer without caller-supplied identities. */
    q.candidate_source = ELPIS_ECS_MEM_CANDIDATES_CODEBOOK;
    q.candidates = NULL;
    q.candidate_count = 0;
    assert(elpis_ecs_mem_query(m, &q, &ans) == ELPIS_ECS_MEM_OK && tu_same(ans.answer, cands[0]));
    elpis_ecs_mem_stats st;
    assert(elpis_ecs_mem_inspect(m, &st) == ELPIS_ECS_MEM_OK && ans.matches == st.codebook_used);

    /* Negative net evidence suppresses: learning the negated relation drives its readout toward zero. */
    elpis_ecs_mem_observation nA = tu_bin(1, 7, 100, ELPIS_ECS_MEM_NEGATE, 7), nA2 = nA;
    nA2.tick = 8;
    assert(elpis_ecs_mem_observe(m, 0, &nA, NULL, NULL) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_observe(m, 0, &nA2, NULL, NULL) == ELPIS_ECS_MEM_OK); /* net: 1 - 2 < 0 */
    q.candidate_source = ELPIS_ECS_MEM_CANDIDATES_CALLER;
    q.candidates = cands;
    q.candidate_count = 2;
    assert(elpis_ecs_mem_query(m, &q, &ans) == ELPIS_ECS_MEM_OK);
    const int64_t sig = ans.signal;
    assert(elpis_ecs_mem_learn(m, 0, &kA, 9, NULL) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_query(m, &q, &ans) == ELPIS_ECS_MEM_OK && ans.signal < sig / 8);
    elpis_ecs_mem_destroy(m);
}

static void test_lookup_channel(void) {
    elpis_ecs_mem_config c = tu_config(64, 64, 64);
    elpis_ecs_mem *m;
    assert(elpis_ecs_mem_create(&c, &m) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_request q = tu_query(ELPIS_ECS_MEM_LOOKUP, 1, 2);
    tu_cue(&q, 1, 7);
    elpis_ecs_mem_answer ans;
    assert(elpis_ecs_mem_query(m, &q, &ans) == ELPIS_ECS_MEM_OK && !ans.found && ans.matches == 0);
    elpis_ecs_mem_observation o1 = tu_bin(1, 7, 100, ELPIS_ECS_MEM_ASSERT, 1);
    elpis_ecs_mem_observation o2 = tu_bin(1, 7, 200, ELPIS_ECS_MEM_ASSERT, 2);
    elpis_ecs_mem_observation o3 = tu_bin(1, 7, 300, ELPIS_ECS_MEM_NEGATE, 3);
    elpis_ecs_mem_observation o4 = tu_bin(1, 7, 400, ELPIS_ECS_MEM_UNCERTAIN, 4);
    elpis_ecs_mem_observation o5 = tu_bin(1, 8, 500, ELPIS_ECS_MEM_ASSERT, 5); /* other subject */
    assert(elpis_ecs_mem_observe(m, 0, &o1, NULL, NULL) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_observe(m, 0, &o2, NULL, NULL) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_observe(m, 0, &o3, NULL, NULL) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_observe(m, 0, &o4, NULL, NULL) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_observe(m, 0, &o5, NULL, NULL) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_query(m, &q, &ans) == ELPIS_ECS_MEM_OK && ans.found);
    assert(tu_same(ans.answer, tu_entity(200)) && ans.answer_tick == 2); /* latest assert */
    assert(ans.matches == 4 && ans.conflicts == 1 && ans.contradicted == 1 && ans.uncertain == 1);
    assert(ans.score == 1 && ans.runner_up == 1);
    /* A later explicit negation of the current answer removes it; the earlier one returns. */
    elpis_ecs_mem_observation n2 = tu_bin(1, 7, 200, ELPIS_ECS_MEM_NEGATE, 6), n2b = n2;
    n2b.tick = 7;
    assert(elpis_ecs_mem_observe(m, 0, &n2, NULL, NULL) == ELPIS_ECS_MEM_OK);
    /* Net zero (one assert, one negate) is no positive evidence: 200 is no longer named, nor contradicted. */
    assert(elpis_ecs_mem_query(m, &q, &ans) == ELPIS_ECS_MEM_OK && tu_same(ans.answer, tu_entity(100)));
    assert(ans.contradicted == 1 && ans.conflicts == 0);
    assert(elpis_ecs_mem_observe(m, 0, &n2b, NULL, NULL) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_query(m, &q, &ans) == ELPIS_ECS_MEM_OK && tu_same(ans.answer, tu_entity(100)));
    assert(ans.contradicted == 2 && ans.conflicts == 0);
    /* Partial cue over a ternary relation: the lookup scans resident evidence, it needs no full key. */
    const uint32_t roles[3] = {1, 2, 3};
    const uint64_t ents[3] = {10, 20, 30};
    elpis_ecs_mem_observation give = tu_obs(9, ELPIS_ECS_MEM_ASSERT, 8, 3, roles, ents);
    assert(elpis_ecs_mem_observe(m, 0, &give, NULL, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_request g = tu_query(ELPIS_ECS_MEM_LOOKUP, 9, 3);
    tu_cue(&g, 1, 10);
    assert(elpis_ecs_mem_query(m, &g, &ans) == ELPIS_ECS_MEM_OK && tu_same(ans.answer, tu_entity(30)));
    assert(ans.work.slot_scans >= 7);
    elpis_ecs_mem_destroy(m);
}

static void test_slow_channel_not_provided(void) {
    /* The slow-bank law is research-only: the SLOW channel is refused UNQUALIFIED whatever the capabilities, the
     * slow banks stay zero and the snapshot's slow cursor stays 0. */
    elpis_ecs_mem_config c = tu_config(64, 8, 16);
    elpis_ecs_mem *m;
    assert(elpis_ecs_mem_create(&c, &m) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_digest key;
    for (uint64_t i = 0; i < 6; ++i) {
        elpis_ecs_mem_observation o = tu_bin(1, i, 100 + i, ELPIS_ECS_MEM_ASSERT, 1 + i);
        assert(elpis_ecs_mem_observe(m, 0, &o, &key, NULL) == ELPIS_ECS_MEM_OK);
        assert(elpis_ecs_mem_learn(m, 0, &key, 1 + i, NULL) == ELPIS_ECS_MEM_OK);
    }
    elpis_ecs_mem_request sq = tu_query(ELPIS_ECS_MEM_SLOW, 1, 2);
    tu_cue(&sq, 1, 0);
    elpis_ecs_mem_answer ans;
    memset(&ans, 0x5A, sizeof(ans));
    elpis_ecs_mem_answer untouched = ans;
    assert(elpis_ecs_mem_query(m, &sq, &ans) == ELPIS_ECS_MEM_UNQUALIFIED && !memcmp(&ans, &untouched, sizeof(ans)));
    const size_t n = elpis_ecs_mem_snapshot_size(m);
    uint8_t *b = malloc(n);
    size_t w = 0;
    assert(b && elpis_ecs_mem_snapshot(m, b, n, &w) == ELPIS_ECS_MEM_OK && w == n);
    const size_t half = (size_t)c.banks * c.dim * c.dim * 4u;
    for (size_t i = 0; i < half; ++i) assert(b[128 + half + i] == 0); /* slow banks */
    assert(b[72] == 0 && b[73] == 0 && b[74] == 0 && b[75] == 0);    /* slow cursor */
    free(b);
    elpis_ecs_mem_destroy(m);
}

static void test_hebbian_numeric_bound(void) {
    /* Hebbian accumulation is unbounded; learning must refuse before int32 can overflow, atomically. */
    elpis_ecs_mem_config c = tu_config(64, 8, 8);
    c.law = ELPIS_ECS_MEM_LAW_HEBBIAN;
    c.banks = 1;
    elpis_ecs_mem *m;
    assert(elpis_ecs_mem_create(&c, &m) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_observation o = tu_bin(1, 1, 2, ELPIS_ECS_MEM_ASSERT, 0);
    elpis_ecs_mem_digest key;
    assert(elpis_ecs_mem_observe(m, 0, &o, &key, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_status s = ELPIS_ECS_MEM_OK;
    uint64_t tick = 1, learned = 0;
    while (s == ELPIS_ECS_MEM_OK && tick < 100000) {
        s = elpis_ecs_mem_learn(m, 0, &key, tick++, NULL);
        learned += s == ELPIS_ECS_MEM_OK;
    }
    assert(s == ELPIS_ECS_MEM_NUMERIC && learned > 100);
    const elpis_ecs_mem_digest before = tu_identity(m);
    assert(elpis_ecs_mem_learn(m, 0, &key, tick, NULL) == ELPIS_ECS_MEM_NUMERIC);
    assert(tu_same(before, tu_identity(m)));
    elpis_ecs_mem_destroy(m);
    /* The delta law at full rate stays bounded under the same traffic. */
    c.law = ELPIS_ECS_MEM_LAW_DELTA;
    assert(elpis_ecs_mem_create(&c, &m) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_observe(m, 0, &o, &key, NULL) == ELPIS_ECS_MEM_OK);
    for (uint64_t t = 1; t <= 2000; ++t) assert(elpis_ecs_mem_learn(m, 0, &key, t, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_destroy(m);
}

static void test_determinism_known_answer(void) {
    /* Two independent handles, one script: identical identities; and the identity is pinned so that every
     * compiler, build type and host must reproduce it bitwise (integer arithmetic only). */
    elpis_ecs_mem_digest ids[2];
    for (int h = 0; h < 2; ++h) {
        elpis_ecs_mem_config c = tu_config(128, 16, 16);
        c.retain_q16 = 61000;
        c.learn_rate_q16 = 40000;
        elpis_ecs_mem *m;
        assert(elpis_ecs_mem_create(&c, &m) == ELPIS_ECS_MEM_OK);
        for (uint64_t i = 0; i < 40; ++i) {
            elpis_ecs_mem_observation o = tu_bin(1 + (uint32_t)(i % 3), i % 7, 100 + (i * 5) % 11,
                                           i % 5 == 4 ? ELPIS_ECS_MEM_NEGATE : ELPIS_ECS_MEM_ASSERT, i);
            elpis_ecs_mem_digest key;
            assert(elpis_ecs_mem_observe(m, 0, &o, &key, NULL) == ELPIS_ECS_MEM_OK);
            const elpis_ecs_mem_status s = elpis_ecs_mem_learn(m, 0, &key, i, NULL);
            assert(s == ELPIS_ECS_MEM_OK || s == ELPIS_ECS_MEM_NO_NET_EVIDENCE);
        }
        ids[h] = tu_identity(m);
        elpis_ecs_mem_destroy(m);
    }
    assert(tu_same(ids[0], ids[1]));
    char hex[65];
    tu_hex(&ids[0], hex);
    fprintf(stderr, "known-answer identity %s\n", hex);
    assert(!strcmp(hex, ELPIS_ECS_MEM_KNOWN_ANSWER));
}

int main(void) {
    test_config();
    test_observe_identity_and_reconstruction();
    test_slot_capacity_and_recency();
    test_codebook_capacity();
    test_lease();
    test_learn_and_query();
    test_lookup_channel();
    test_slow_channel_not_provided();
    test_hebbian_numeric_bound();
    test_determinism_known_answer();
    printf("PASS_ECS_MEM_MECHANICS\n");
    return 0;
}
