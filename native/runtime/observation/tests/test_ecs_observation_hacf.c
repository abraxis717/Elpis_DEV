/* Typed HACF adapter: lossless mapping and inversion of structure's own N-ary hyperedges, every refusal class,
 * conflicting provenance, and relational reconstruction from retained memory state. Fixtures are synthetic
 * (TEST_ONLY, TRAINING=NONE, SEMANTICS=NONE). */
#include "elpis/ecs_observation_hacf.h"
#include "ecs_memory_test_util.h"

#define TYPE_GIVE (SEMANTIC_HYPEREDGE_NAMESPACE | 1u)
#define ROLE_AGENT (SEMANTIC_INCIDENCE_NAMESPACE | 1u)
#define ROLE_RECIPIENT (SEMANTIC_INCIDENCE_NAMESPACE | 2u)
#define ROLE_OBJECT (SEMANTIC_INCIDENCE_NAMESPACE | 3u)

static hacf_digest node(uint64_t i) {
    hacf_digest d;
    const elpis_ecs_mem_digest e = tu_entity(i);
    memcpy(d.bytes, e.b, 32);
    return d;
}
static elpis_semantic_hyperedge_v1 give(uint64_t agent, uint64_t recipient, uint64_t object) {
    elpis_semantic_hyperedge_v1 e;
    memset(&e, 0, sizeof(e));
    e.abi_version = SEMANTIC_ABI_VERSION;
    e.hyperedge_type = TYPE_GIVE;
    e.participant_count = 3;
    e.participants[0].incidence_role = ROLE_OBJECT; /* presented out of order; canonicalized below */
    e.participants[0].node_identity = node(object);
    e.participants[1].incidence_role = ROLE_AGENT;
    e.participants[1].node_identity = node(agent);
    e.participants[2].incidence_role = ROLE_RECIPIENT;
    e.participants[2].node_identity = node(recipient);
    assert(elpis_semantic_canonicalize_participants(e.participants, 3) == SEMANTIC_OK);
    assert(elpis_semantic_hyperedge_identity(&e, &e.hyperedge_identity) == SEMANTIC_OK);
    return e;
}
static elpis_semantic_assertion_v1 assert_of(const elpis_semantic_hyperedge_v1 *e, uint64_t source) {
    elpis_semantic_assertion_v1 a;
    memset(&a, 0, sizeof(a));
    a.abi_version = SEMANTIC_ABI_VERSION;
    a.asserted_object_kind = SEMANTIC_OBJECT_KIND_HYPEREDGE;
    a.asserted_object_digest = e->hyperedge_identity;
    const elpis_ecs_mem_digest p = tu_digest("source", source);
    memcpy(a.provenance_digest.bytes, p.b, 32);
    a.authority = 2;
    assert(elpis_semantic_assertion_identity(&a, &a.assertion_identity) == SEMANTIC_OK);
    return a;
}
static void rehash(elpis_semantic_hyperedge_v1 *e, elpis_semantic_assertion_v1 *a) {
    assert(elpis_semantic_hyperedge_identity(e, &e->hyperedge_identity) == SEMANTIC_OK);
    if (a) {
        a->asserted_object_digest = e->hyperedge_identity;
        assert(elpis_semantic_assertion_identity(a, &a->assertion_identity) == SEMANTIC_OK);
    }
}
static void expect(elpis_ecs_obs_hacf_status want, const elpis_semantic_hyperedge_v1 *e,
                   const elpis_semantic_assertion_v1 *a, uint32_t polarity) {
    elpis_ecs_mem_observation o, sentinel;
    memset(&sentinel, 0x6B, sizeof(sentinel));
    o = sentinel;
    const elpis_ecs_obs_hacf_status got = elpis_ecs_obs_hacf_observation(e, a, polarity, 1, &o);
    if (got != want) {
        fprintf(stderr, "expected %d, got %d\n", want, got);
        assert(0);
    }
    assert(!memcmp(&o, &sentinel, sizeof(o)));
}

static void test_refusals(void) {
    elpis_semantic_hyperedge_v1 e = give(1, 2, 3), b;
    elpis_semantic_assertion_v1 a = assert_of(&e, 1), c;
    expect(ELPIS_ECS_OBS_HACF_INVALID, NULL, &a, ELPIS_ECS_MEM_ASSERT);
    expect(ELPIS_ECS_OBS_HACF_INVALID, &e, NULL, ELPIS_ECS_MEM_ASSERT);
    expect(ELPIS_ECS_OBS_HACF_INVALID, &e, &a, 0);
    expect(ELPIS_ECS_OBS_HACF_INVALID, &e, &a, 4);
    b = e; b.hyperedge_type = 1; rehash(&b, NULL); c = assert_of(&b, 1);       /* outside the type namespace */
    expect(ELPIS_ECS_OBS_HACF_STRUCTURE, &b, &c, ELPIS_ECS_MEM_ASSERT);
    b = e; b.participants[0].incidence_role = 3; rehash(&b, NULL); c = assert_of(&b, 1);
    expect(ELPIS_ECS_OBS_HACF_STRUCTURE, &b, &c, ELPIS_ECS_MEM_ASSERT);
    c = a; c.authority = 4; assert(elpis_semantic_assertion_identity(&c, &c.assertion_identity) == 0);
    expect(ELPIS_ECS_OBS_HACF_STRUCTURE, &e, &c, ELPIS_ECS_MEM_ASSERT);
    b = e; b.hyperedge_identity.bytes[0] ^= 1;                                 /* stored identity is a claim */
    expect(ELPIS_ECS_OBS_HACF_IDENTITY, &b, &a, ELPIS_ECS_MEM_ASSERT);
    c = a; c.assertion_identity.bytes[5] ^= 1;
    expect(ELPIS_ECS_OBS_HACF_IDENTITY, &e, &c, ELPIS_ECS_MEM_ASSERT);
    b = give(1, 2, 4); c = assert_of(&b, 1);                                   /* asserts another relation */
    expect(ELPIS_ECS_OBS_HACF_IDENTITY, &e, &c, ELPIS_ECS_MEM_ASSERT);
    c = a; c.asserted_object_kind = SEMANTIC_OBJECT_KIND_NODE;
    assert(elpis_semantic_assertion_identity(&c, &c.assertion_identity) == 0);
    expect(ELPIS_ECS_OBS_HACF_IDENTITY, &e, &c, ELPIS_ECS_MEM_ASSERT);
    b = e; elpis_semantic_participant_descriptor t = b.participants[0];       /* non-canonical order */
    b.participants[0] = b.participants[2]; b.participants[2] = t; rehash(&b, NULL); c = assert_of(&b, 1);
    expect(ELPIS_ECS_OBS_HACF_NONCANONICAL, &b, &c, ELPIS_ECS_MEM_ASSERT);
    c = a; c.assertion_flags = 1; assert(elpis_semantic_assertion_identity(&c, &c.assertion_identity) == 0);
    expect(ELPIS_ECS_OBS_HACF_UNSUPPORTED, &e, &c, ELPIS_ECS_MEM_ASSERT);                 /* flag meaning undefined */
    b = e; b.semantic_flags = 1; rehash(&b, NULL); c = assert_of(&b, 1);
    expect(ELPIS_ECS_OBS_HACF_UNSUPPORTED, &b, &c, ELPIS_ECS_MEM_ASSERT);
    b = e; b.participants[1].participant_flags = SEMANTIC_PARTICIPANT_FLAG_QUERY; rehash(&b, NULL);
    c = assert_of(&b, 1);
    expect(ELPIS_ECS_OBS_HACF_UNSUPPORTED, &b, &c, ELPIS_ECS_MEM_ASSERT);                 /* query-local overlay */
    b = e; b.participant_count = 0; rehash(&b, NULL); c = assert_of(&b, 1);
    expect(ELPIS_ECS_OBS_HACF_UNSUPPORTED, &b, &c, ELPIS_ECS_MEM_ASSERT);
    memset(&b, 0, sizeof(b));                                                  /* nine participants */
    b.abi_version = SEMANTIC_ABI_VERSION; b.hyperedge_type = TYPE_GIVE; b.participant_count = 9;
    for (uint32_t i = 0; i < 9; ++i) { b.participants[i].incidence_role = ROLE_AGENT + i; b.participants[i].node_identity = node(i); }
    rehash(&b, NULL); c = assert_of(&b, 1);
    expect(ELPIS_ECS_OBS_HACF_UNSUPPORTED, &b, &c, ELPIS_ECS_MEM_ASSERT);
    b = e; b.participants[1].incidence_role = ROLE_AGENT; b.participants[1].ordinal = 0; /* (agent, 0) twice */
    assert(elpis_semantic_canonicalize_participants(b.participants, 3) == 0); rehash(&b, NULL); c = assert_of(&b, 1);
    expect(ELPIS_ECS_OBS_HACF_UNSUPPORTED, &b, &c, ELPIS_ECS_MEM_ASSERT);
}

static void test_lossless_round_trip_and_conflict(void) {
    elpis_ecs_mem_config cfg = tu_config(256, 64, 64);
    elpis_ecs_mem *m;
    assert(elpis_ecs_mem_create(&cfg, &m) == ELPIS_ECS_MEM_OK);
    elpis_semantic_hyperedge_v1 e = give(1, 2, 3);
    const elpis_ecs_mem_digest pay = tu_digest("payload", 9);
    memcpy(e.payload_digest.bytes, pay.b, 32);
    rehash(&e, NULL);
    elpis_semantic_assertion_v1 a1 = assert_of(&e, 1), a2 = assert_of(&e, 2);
    elpis_ecs_mem_observation o1, o2;
    assert(elpis_ecs_obs_hacf_observation(&e, &a1, ELPIS_ECS_MEM_ASSERT, 5, &o1) == ELPIS_ECS_OBS_HACF_OK);
    assert(elpis_ecs_obs_hacf_observation(&e, &a2, ELPIS_ECS_MEM_NEGATE, 6, &o2) == ELPIS_ECS_OBS_HACF_OK); /* another source */
    elpis_ecs_mem_digest key;
    assert(elpis_ecs_mem_observe(m, 0, &o1, &key, NULL) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_observe(m, 0, &o2, NULL, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_slot s;
    assert(elpis_ecs_mem_slot_read(m, &key, &s) == ELPIS_ECS_MEM_OK);
    assert(s.assert_count == 1 && s.negate_count == 1 && s.last_authority == 2);
    assert(!memcmp(s.first_provenance.b, a1.provenance_digest.bytes, 32));
    assert(!memcmp(s.last_provenance.b, a2.provenance_digest.bytes, 32));
    assert(!memcmp(s.structure_ref.b, e.hyperedge_identity.bytes, 32));
    elpis_semantic_hyperedge_v1 back;
    assert(elpis_ecs_obs_hacf_hyperedge(&s, &back) == ELPIS_ECS_OBS_HACF_OK);
    assert(!memcmp(back.hyperedge_identity.bytes, e.hyperedge_identity.bytes, 32)); /* exact structural identity */
    assert(!memcmp(&back, &e, sizeof(e)));
    /* Conflict is retained, not resolved: LEARN refuses without net evidence. */
    assert(elpis_ecs_mem_learn(m, 0, &key, 7, NULL) == ELPIS_ECS_MEM_NO_NET_EVIDENCE);
    elpis_ecs_mem_destroy(m);
}

/* N-ary relations through structure's records: reconstruct a role filler from retained state, by LOOKUP over
 * resident evidence and by the learned FAST state from a full or partial cue. */
static void test_relational_reconstruction(void) {
    elpis_ecs_mem_config cfg = tu_config(512, 64, 64);
    elpis_ecs_mem *m;
    assert(elpis_ecs_mem_create(&cfg, &m) == ELPIS_ECS_MEM_OK);
    uint64_t tick = 1;
    for (uint64_t g = 0; g < 8; ++g) { /* give(agent g, recipient 10+g, object 20+g) */
        elpis_semantic_hyperedge_v1 e = give(g, 10 + g, 20 + g);
        elpis_semantic_assertion_v1 a = assert_of(&e, g);
        elpis_ecs_mem_observation o;
        elpis_ecs_mem_digest key;
        assert(elpis_ecs_obs_hacf_observation(&e, &a, ELPIS_ECS_MEM_ASSERT, tick, &o) == ELPIS_ECS_OBS_HACF_OK);
        assert(elpis_ecs_mem_observe(m, 0, &o, &key, NULL) == ELPIS_ECS_MEM_OK);
        assert(elpis_ecs_mem_learn(m, 0, &key, tick, NULL) == ELPIS_ECS_MEM_OK);
        ++tick;
    }
    unsigned lookup_ok = 0, full_ok = 0, partial_ok = 0;
    for (uint64_t g = 0; g < 8; ++g) {
        elpis_ecs_mem_answer ans;
        elpis_ecs_mem_request q = tu_query(ELPIS_ECS_MEM_LOOKUP, TYPE_GIVE, ROLE_OBJECT);
        q.cue_count = 1;
        q.cue[0].role = ROLE_RECIPIENT;
        q.cue[0].entity = tu_entity(10 + g);
        assert(elpis_ecs_mem_query(m, &q, &ans) == ELPIS_ECS_MEM_OK);
        lookup_ok += ans.found && tu_same(ans.answer, tu_entity(20 + g));
        q.channel = ELPIS_ECS_MEM_FAST;
        q.candidate_source = ELPIS_ECS_MEM_CANDIDATES_CODEBOOK;
        q.cue_count = 2;
        q.cue[0].role = ROLE_AGENT;
        q.cue[0].entity = tu_entity(g);
        q.cue[1].role = ROLE_RECIPIENT;
        q.cue[1].entity = tu_entity(10 + g);
        assert(elpis_ecs_mem_query(m, &q, &ans) == ELPIS_ECS_MEM_OK);
        full_ok += tu_same(ans.answer, tu_entity(20 + g));
        q.cue_count = 1; /* agent only: never stored as a cue on its own */
        memset(&q.cue[1], 0, sizeof(q.cue[1]));
        assert(elpis_ecs_mem_query(m, &q, &ans) == ELPIS_ECS_MEM_OK);
        partial_ok += tu_same(ans.answer, tu_entity(20 + g));
    }
    printf("reconstruction: lookup %u/8, fast full cue %u/8, fast partial cue %u/8\n", lookup_ok, full_ok, partial_ok);
    assert(lookup_ok == 8 && full_ok == 8);
    /* The partial-cue rate is a measurement for M3, not a mechanics requirement. */
    elpis_ecs_mem_destroy(m);
}

int main(void) {
    test_refusals();
    test_lossless_round_trip_and_conflict();
    test_relational_reconstruction();
    printf("PASS_ECS_OBSERVATION_HACF\n");
    return 0;
}
