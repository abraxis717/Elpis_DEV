/*
 * ECS_C relational view over the volatile ECS_C session and the real ECS_C memory, end to end, through the production
 * libraries (libelpis_ecs_c_relational, libelpis_ecs_c_session, libelpis_ecs_memory). Nothing here opens or writes a
 * file: the session, the view and the memory are volatile, and so is everything the test checks.
 *
 * 1. The gap, demonstrated on the memory: after SOURCE_VERSION, DORMANT, and reclamation with a reused 32-bit
 *    source_authority, memory LOOKUP keeps answering from evidence the session's admission law refuses, and
 *    attributes evidence to a source that never asserted it.
 * 2. The view under the same transitions, synced from elpis_ecs_c_session_read_status / _entity_at only: it answers
 *    only from evidence the session's law still admits; revalidation under a new version admits nothing old.
 * 3. Its admission law equals elpis_ecs_c_session_check_evidence on every (row, version, scope) probe at every state.
 * 4. Contradictory evidence from two session sources, and capacity refusal of the view.
 * 5. Session lifecycle: a view synced to one session refuses another session's tables (LINEAGE) until it is reset,
 *    so no view carries one session's evidence into the next; the new session's view starts empty.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>

#include "elpis/ecs_c_relational.h"
#include "elpis/ecs_c_session.h"

#define CHECK(expr)                                                                                             \
    do {                                                                                                        \
        if (!(expr)) {                                                                                          \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #expr);                            \
            exit(1);                                                                                            \
        }                                                                                                       \
    } while (0)
#define SOK(expr) CHECK((expr) == ELPIS_ECS_C_OK)
#define ROK(expr) CHECK((expr) == ELPIS_ECS_C_REL_OK)

static void put_u64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}
static void put_u32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
}
static void event(uint8_t e[512], uint16_t kind, const uint8_t sender[32], uint64_t seq, const uint8_t *receiver) {
    memset(e, 0, 512);
    e[0] = (uint8_t)kind;
    e[1] = (uint8_t)(kind >> 8);
    memcpy(e + 8, sender, 32);
    put_u64(e + 40, seq);
    if (receiver) memcpy(e + 48, receiver, 32);
}
static int32_t send(elpis_ecs_c_session *s, const uint8_t e[512]) {
    elpis_ecs_c_session_coordinate c;
    SOK(elpis_ecs_c_session_read_coordinate(s, &c));
    return elpis_ecs_c_session_transition(s, &c, e, NULL);
}
static uint64_t next_seq(const elpis_ecs_c_session *s, const uint8_t who[32]) {
    elpis_ecs_c_entity e;
    SOK(elpis_ecs_c_session_entity_find(s, who, &e));
    return e.sender_watermark + 1;
}
static void scope_of(uint8_t out[32], uint8_t tag) { memset(out, tag, 32); }

static elpis_ecs_c_session *session_create(uint32_t capacity) {
    elpis_ecs_c_session_config c;
    memset(&c, 0, sizeof(c));
    c.abi_version = ELPIS_ECS_C_SESSION_ABI_VERSION;
    c.capacity = capacity;
    memset(c.operator_label, 1, 32);
    memset(c.runtime_label, 2, 32); /* a zero nonce: the session draws one (getrandom, no file) */
    elpis_ecs_c_session *s = NULL;
    SOK(elpis_ecs_c_session_create(&c, &s));
    return s;
}

static void found_source(elpis_ecs_c_session *s, const uint8_t op[32], uint8_t label, uint32_t authority,
                         uint8_t scope, uint8_t out[32]) {
    uint8_t l[32], e[512];
    memset(l, label, 32);
    SOK(elpis_ecs_c_session_next_entity_id(s, ELPIS_ECS_C_SOURCE, l, out));
    event(e, 2, op, next_seq(s, op), out);
    e[80] = ELPIS_ECS_C_SOURCE;
    put_u32(e + 84, 8);
    put_u64(e + 88, 1);
    memcpy(e + 96, l, 32);
    memset(e + 128, scope, 32);
    put_u32(e + 160, authority);
    SOK(send(s, e));
}
static void lifecycle(elpis_ecs_c_session *s, const uint8_t op[32], const uint8_t id[32], uint8_t from, uint8_t to) {
    uint8_t e[512];
    event(e, 3, op, next_seq(s, op), id);
    e[80] = from;
    e[81] = to;
    SOK(send(s, e));
}
static void source_version(elpis_ecs_c_session *s, const uint8_t op[32], const uint8_t id[32], uint64_t expected,
                           uint64_t version, uint8_t scope) {
    uint8_t e[512];
    event(e, 4, op, next_seq(s, op), id);
    put_u64(e + 80, expected);
    put_u64(e + 88, version);
    put_u32(e + 96, 8);
    memset(e + 104, scope, 32);
    SOK(send(s, e));
}
static void reclaim(elpis_ecs_c_session *s, const uint8_t op[32], const uint8_t id[32]) {
    uint8_t e[512];
    event(e, 14, op, next_seq(s, op), id);
    SOK(send(s, e));
}

/* The session as the view consumes it: its status (the lineage is the session's identity) and every row. */
static elpis_ecs_c_entity rows[ELPIS_ECS_C_MAX_CAPACITY];
static elpis_ecs_c_rel_authority table_of(const elpis_ecs_c_session *s, elpis_ecs_c_rel_coordinate *coord) {
    elpis_ecs_c_session_status st;
    SOK(elpis_ecs_c_session_read_status(s, &st));
    for (uint32_t i = 0; i < st.entity_count; ++i) SOK(elpis_ecs_c_session_entity_at(s, i, &rows[i]));
    elpis_ecs_c_rel_authority t;
    memset(&t, 0, sizeof(t));
    t.abi_version = ELPIS_ECS_C_REL_ABI_VERSION;
    t.entity_count = st.entity_count;
    t.generation = st.generation;
    t.next_founding_index = st.next_founding_index;
    t.reclaimed_count = st.reclaimed_count;
    memcpy(t.lineage, st.session, 32);
    memcpy(t.state_root, st.state_root, 32);
    t.entities = rows;
    if (coord) {
        coord->generation = st.generation;
        memcpy(coord->state_root, st.state_root, 32);
    }
    return t;
}
static int sync_to(elpis_ecs_c_rel *v, const elpis_ecs_c_session *s, elpis_ecs_c_rel_sync_result *r) {
    const elpis_ecs_c_rel_authority t = table_of(s, NULL);
    return elpis_ecs_c_rel_sync(v, &t, r);
}

/* The view's admission law against the session's, on every probe. */
static uint64_t law_probes, law_states;
static int mapped(int32_t code) {
    switch (code) {
    case ELPIS_ECS_C_OK: return ELPIS_ECS_C_REL_OK;
    case ELPIS_ECS_C_UNKNOWN_ENTITY: return ELPIS_ECS_C_REL_UNKNOWN_SOURCE;
    case ELPIS_ECS_C_UNAUTHORIZED: return ELPIS_ECS_C_REL_NOT_SOURCE;
    case ELPIS_ECS_C_LIFECYCLE: return ELPIS_ECS_C_REL_LIFECYCLE;
    case ELPIS_ECS_C_SOURCE_STALE: return ELPIS_ECS_C_REL_SOURCE_STALE;
    case ELPIS_ECS_C_SCOPE_CONFLICT: return ELPIS_ECS_C_REL_SCOPE_CONFLICT;
    default: return -1;
    }
}
static void same_law(const elpis_ecs_c_rel *v, const elpis_ecs_c_session *s) {
    elpis_ecs_c_session_status st;
    SOK(elpis_ecs_c_session_read_status(s, &st));
    ++law_states;
    uint8_t unknown[32], scopes[5][32];
    memset(unknown, 0xEE, 32);
    for (int i = 0; i < 4; ++i) scope_of(scopes[i], (uint8_t)(0xA0 + i));
    for (uint32_t i = 0; i <= st.entity_count; ++i) {
        elpis_ecs_c_entity e;
        const uint8_t *id = unknown;
        if (i < st.entity_count) {
            SOK(elpis_ecs_c_session_entity_at(s, i, &e));
            id = e.id;
            memcpy(scopes[4], e.scope, 32);
        }
        for (uint64_t version = 0; version <= 4; ++version)
            for (int sc = 0; sc < 5; ++sc) {
                uint32_t limit = 0;
                const int want = mapped(elpis_ecs_c_session_check_evidence(s, id, version, scopes[sc], &limit));
                CHECK(want >= 0 && elpis_ecs_c_rel_check(v, id, version, scopes[sc]) == want);
                ++law_probes;
            }
    }
}

static elpis_ecs_mem_observation pair(uint32_t polarity, uint64_t tick, uint32_t authority, uint8_t a, uint8_t b) {
    elpis_ecs_mem_observation o;
    memset(&o, 0, sizeof(o));
    o.abi_version = ELPIS_ECS_MEM_ABI_V1;
    o.relation_type = 5;
    o.polarity = polarity;
    o.participant_count = 2;
    o.tick = tick;
    o.source_authority = authority;
    memset(o.provenance.b, (int)(tick & 0xFF), 32);
    o.participants[0].role = 1;
    memset(o.participants[0].entity.b, a, 32);
    o.participants[1].role = 2;
    memset(o.participants[1].entity.b, b, 32);
    return o;
}
static elpis_ecs_mem_request ask(uint8_t a) {
    elpis_ecs_mem_request q;
    memset(&q, 0, sizeof(q));
    q.abi_version = ELPIS_ECS_MEM_ABI_V1;
    q.channel = ELPIS_ECS_MEM_LOOKUP;
    q.relation_type = 5;
    q.target_role = 2;
    q.cue_count = 1;
    q.cue[0].role = 1;
    memset(q.cue[0].entity.b, a, 32);
    return q;
}
static uint8_t answer_of(const elpis_ecs_mem_answer *ans) { return ans->found ? ans->answer.b[0] : 0; }

/* Admission as RuntimeCore's HOT lane performs it today (the observation's source_authority names an ACTIVE source),
 * into the memory; and into the view under the binding the session holds now. */
static void admit_both(elpis_ecs_mem *m, elpis_ecs_c_rel *v, const elpis_ecs_c_session *s,
                       const elpis_ecs_mem_observation *o) {
    elpis_ecs_c_entity e;
    SOK(elpis_ecs_c_session_source_for_authority(s, o->source_authority, &e));
    CHECK(e.lifecycle == ELPIS_ECS_C_ACTIVE);
    CHECK(elpis_ecs_mem_observe(m, 0, o, NULL, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_c_rel_coordinate c;
    (void)table_of(s, &c);
    ROK(elpis_ecs_c_rel_admit(v, &c, e.id, e.source_version, e.scope, o, NULL, NULL));
}

int main(void) {
    const uint32_t capacity = 16;
    elpis_ecs_c_session *a = session_create(capacity);
    elpis_ecs_c_entity op;
    SOK(elpis_ecs_c_session_entity_at(a, 0, &op));
    uint8_t s7[32], s8[32], s9[32], s10[32];
    found_source(a, op.id, 7, 7, 0xA0, s7);
    found_source(a, op.id, 8, 8, 0xA0, s8);
    found_source(a, op.id, 9, 9, 0xA1, s9);
    lifecycle(a, op.id, s7, ELPIS_ECS_C_FOUNDED, ELPIS_ECS_C_ACTIVE);
    lifecycle(a, op.id, s8, ELPIS_ECS_C_FOUNDED, ELPIS_ECS_C_ACTIVE);
    lifecycle(a, op.id, s9, ELPIS_ECS_C_FOUNDED, ELPIS_ECS_C_ACTIVE);

    elpis_ecs_c_rel_config vc;
    elpis_ecs_c_rel_config_default(&vc);
    vc.entity_capacity = capacity;
    elpis_ecs_c_rel *v = NULL;
    ROK(elpis_ecs_c_rel_create(&vc, &v));
    ROK(sync_to(v, a, NULL));
    same_law(v, a);
    elpis_ecs_mem_config mc;
    elpis_ecs_mem_config_default(&mc);
    mc.dim = 64, mc.banks = 1, mc.slot_capacity = 32, mc.codebook_capacity = 64, mc.byte_budget = 1u << 26;
    elpis_ecs_mem *m = NULL;
    CHECK(elpis_ecs_mem_create(&mc, &m) == ELPIS_ECS_MEM_OK);

    /* s7 says 1->2 once; s8 says 1->3 twice; s9 says 4->5. */
    const elpis_ecs_mem_observation o72 = pair(ELPIS_ECS_MEM_ASSERT, 1, 7, 1, 2);
    const elpis_ecs_mem_observation o83 = pair(ELPIS_ECS_MEM_ASSERT, 2, 8, 1, 3);
    const elpis_ecs_mem_observation o95 = pair(ELPIS_ECS_MEM_ASSERT, 3, 9, 4, 5);
    admit_both(m, v, a, &o72);
    admit_both(m, v, a, &o83);
    admit_both(m, v, a, &o83);
    admit_both(m, v, a, &o95);
    const elpis_ecs_mem_request q1 = ask(1), q4 = ask(4);
    elpis_ecs_mem_answer am, av;
    CHECK(elpis_ecs_mem_query(m, &q1, &am) == ELPIS_ECS_MEM_OK);
    ROK(elpis_ecs_c_rel_query(v, NULL, NULL, &q1, &av, NULL));
    CHECK(answer_of(&am) == 3 && answer_of(&av) == 3 && am.score == 2 && av.score == 2); /* agree while all valid */

    /* SOURCE_VERSION s8 v1 -> v2: VALID -> STALE for its v1 evidence, in RAM. */
    source_version(a, op.id, s8, 1, 2, 0xA0);
    uint8_t scope_a[32];
    scope_of(scope_a, 0xA0);
    uint32_t limit;
    CHECK(elpis_ecs_c_session_check_evidence(a, s8, 1, scope_a, &limit) == ELPIS_ECS_C_SOURCE_STALE);
    CHECK(elpis_ecs_mem_query(m, &q1, &am) == ELPIS_ECS_MEM_OK);
    CHECK(answer_of(&am) == 3 && am.score == 2); /* GAP: the memory answers from v1 evidence the law refuses */
    elpis_ecs_c_rel_sync_result r;
    ROK(sync_to(v, a, &r));
    CHECK(r.sources_changed == 1 && r.contributions_retired == 1 && r.relations_released == 1);
    same_law(v, a);
    ROK(elpis_ecs_c_rel_query(v, NULL, NULL, &q1, &av, NULL));
    CHECK(answer_of(&av) == 2 && av.score == 1 && av.conflicts == 0); /* only s7's admissible evidence */
    /* Revalidation under v2 admits nothing old: one new observation counts once. */
    const elpis_ecs_mem_observation o83b = pair(ELPIS_ECS_MEM_ASSERT, 4, 8, 1, 3);
    admit_both(m, v, a, &o83b);
    ROK(elpis_ecs_c_rel_query(v, NULL, NULL, &q1, &av, NULL));
    CHECK(answer_of(&av) == 3 && av.score == 1 && av.conflicts == 1); /* 1->3 (v2, once) is the latest assert */
    printf("  SOURCE_VERSION: memory LOOKUP 1->3 (score 2, from v1 evidence); view 1->2 (score 1); after "
           "revalidation 1->3 at score 1 (v2 only)\n");

    /* DORMANT s7 suspends its evidence; ACTIVE again restores it (the same binding). */
    lifecycle(a, op.id, s7, ELPIS_ECS_C_ACTIVE, ELPIS_ECS_C_DORMANT);
    ROK(sync_to(v, a, &r));
    CHECK(r.contributions_flipped == 1);
    same_law(v, a);
    ROK(elpis_ecs_c_rel_query(v, NULL, NULL, &q1, &av, NULL));
    CHECK(answer_of(&av) == 3 && av.conflicts == 0);
    lifecycle(a, op.id, s7, ELPIS_ECS_C_DORMANT, ELPIS_ECS_C_ACTIVE);
    ROK(sync_to(v, a, &r));
    CHECK(r.contributions_flipped == 1);
    same_law(v, a);

    /* TERMINATE s9, reclaim it, found s10 on the same 32-bit source_authority 9. */
    lifecycle(a, op.id, s9, ELPIS_ECS_C_ACTIVE, ELPIS_ECS_C_TERMINATED);
    reclaim(a, op.id, s9);
    found_source(a, op.id, 10, 9, 0xA1, s10);
    lifecycle(a, op.id, s10, ELPIS_ECS_C_FOUNDED, ELPIS_ECS_C_ACTIVE);
    elpis_ecs_mem_digest k45;
    CHECK(elpis_ecs_mem_relation_key(&o95, &k45) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_slot slot;
    CHECK(elpis_ecs_mem_slot_read(m, &k45, &slot) == ELPIS_ECS_MEM_OK);
    elpis_ecs_c_entity owner;
    SOK(elpis_ecs_c_session_source_for_authority(a, slot.last_authority, &owner));
    CHECK(!memcmp(owner.id, s10, 32)); /* GAP: the memory's evidence resolves to a source that never said it */
    CHECK(elpis_ecs_mem_query(m, &q4, &am) == ELPIS_ECS_MEM_OK && answer_of(&am) == 5);
    ROK(sync_to(v, a, &r));
    CHECK(r.contributions_retired == 1);
    same_law(v, a);
    ROK(elpis_ecs_c_rel_query(v, NULL, NULL, &q4, &av, NULL));
    CHECK(!av.found);
    CHECK(elpis_ecs_c_rel_relation_read(v, NULL, NULL, &k45, &slot) == ELPIS_ECS_C_REL_NOT_FOUND);
    size_t n = 7;
    ROK(elpis_ecs_c_rel_dependents(v, s10, NULL, 0, &n));
    CHECK(n == 0);
    printf("  reclaim + reuse of source_authority 9: memory attributes 4->5 to the new source; view retired it\n");

    /* Contradictory evidence: s7 asserts 6->7, s10 negates it. Net zero withholds; revoking either side decides. */
    const elpis_ecs_mem_observation pa = pair(ELPIS_ECS_MEM_ASSERT, 5, 7, 6, 7);
    const elpis_ecs_mem_observation pn = pair(ELPIS_ECS_MEM_NEGATE, 6, 9, 6, 7);
    admit_both(m, v, a, &pa);
    admit_both(m, v, a, &pn);
    const elpis_ecs_mem_request q6 = ask(6);
    ROK(elpis_ecs_c_rel_query(v, NULL, NULL, &q6, &av, NULL));
    CHECK(!av.found && av.matches == 1);
    lifecycle(a, op.id, s10, ELPIS_ECS_C_ACTIVE, ELPIS_ECS_C_DORMANT);
    ROK(sync_to(v, a, NULL));
    ROK(elpis_ecs_c_rel_query(v, NULL, NULL, &q6, &av, NULL));
    CHECK(av.found && answer_of(&av) == 7 && av.score == 1);
    lifecycle(a, op.id, s10, ELPIS_ECS_C_DORMANT, ELPIS_ECS_C_ACTIVE);
    lifecycle(a, op.id, s7, ELPIS_ECS_C_ACTIVE, ELPIS_ECS_C_DORMANT);
    ROK(sync_to(v, a, NULL));
    ROK(elpis_ecs_c_rel_query(v, NULL, NULL, &q6, &av, NULL));
    CHECK(!av.found && av.contradicted == 1);
    lifecycle(a, op.id, s7, ELPIS_ECS_C_DORMANT, ELPIS_ECS_C_ACTIVE);
    ROK(sync_to(v, a, NULL));
    same_law(v, a);

    /* Capacity: a view of two relations refuses a third, atomically, with no spill anywhere. */
    elpis_ecs_c_rel_config small = vc;
    small.relation_capacity = 2;
    small.contribution_capacity = 4;
    elpis_ecs_c_rel *tight = NULL;
    ROK(elpis_ecs_c_rel_create(&small, &tight));
    ROK(sync_to(tight, a, NULL));
    elpis_ecs_c_rel_coordinate c;
    (void)table_of(a, &c);
    elpis_ecs_c_entity e7;
    SOK(elpis_ecs_c_session_entity_find(a, s7, &e7));
    for (uint8_t i = 0; i < 2; ++i) {
        const elpis_ecs_mem_observation o = pair(ELPIS_ECS_MEM_ASSERT, 10 + i, 7, 20 + i, 30);
        ROK(elpis_ecs_c_rel_admit(tight, &c, e7.id, e7.source_version, e7.scope, &o, NULL, NULL));
    }
    uint8_t before[32], after[32];
    ROK(elpis_ecs_c_rel_identity(tight, before));
    const elpis_ecs_mem_observation third = pair(ELPIS_ECS_MEM_ASSERT, 12, 7, 22, 30);
    CHECK(elpis_ecs_c_rel_admit(tight, &c, e7.id, e7.source_version, e7.scope, &third, NULL, NULL) ==
          ELPIS_ECS_C_REL_CAPACITY);
    ROK(elpis_ecs_c_rel_identity(tight, after));
    CHECK(!memcmp(before, after, 32));
    elpis_ecs_c_rel_destroy(tight);

    /* Session lifecycle: the session ends; a new one is a fresh genesis. The view synced to the old session refuses
     * the new session's table (another lineage) and keeps nothing it could carry over once reset. */
    elpis_ecs_c_session_status last;
    SOK(elpis_ecs_c_session_read_status(a, &last));
    elpis_ecs_c_session_destroy(a);
    elpis_ecs_c_session *b = session_create(capacity);
    elpis_ecs_c_session_status fresh;
    SOK(elpis_ecs_c_session_read_status(b, &fresh));
    CHECK(fresh.generation == 1 && fresh.entity_count == 2 && memcmp(fresh.session, last.session, 32));
    CHECK(sync_to(v, b, NULL) == ELPIS_ECS_C_REL_LINEAGE);
    ROK(elpis_ecs_c_rel_reset(v));
    ROK(sync_to(v, b, NULL));
    elpis_ecs_c_rel_status vs;
    ROK(elpis_ecs_c_rel_read_status(v, &vs));
    CHECK(vs.synced == 1 && vs.relations == 0 && vs.contributions == 0 && vs.entities == 2);
    ROK(elpis_ecs_c_rel_query(v, NULL, NULL, &q1, &av, NULL));
    CHECK(!av.found && !av.matches);
    CHECK(elpis_ecs_c_session_check_evidence(b, s7, 1, scope_a, &limit) == ELPIS_ECS_C_UNKNOWN_ENTITY);
    same_law(v, b);
    printf("  session lifecycle: the next session's table is refused LINEAGE until reset; the reset view is empty\n");

    elpis_ecs_mem_destroy(m);
    elpis_ecs_c_rel_destroy(v);
    elpis_ecs_c_session_destroy(b);
    printf("  admission law == elpis_ecs_c_session_check_evidence on %llu probes over %llu session states\n",
           (unsigned long long)law_probes, (unsigned long long)law_states);
    printf("ECS_C relational view over the volatile ECS_C session: PASS\n");
    return 0;
}
