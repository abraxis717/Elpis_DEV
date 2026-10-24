/*
 * ECS_C relational view (ECSC_DYNAMIC_RELATIONAL_DEV_R0): the contract of elpis/ecs_c_relational.h, attacked.
 *
 * Built with the library source and ELPIS_ECS_C_REL_TESTING (the test-only damage hook). Every refusal is shown to
 * be atomic (identity, status and verify unchanged). The differential compares the view with the exact-rebuild
 * oracle: a fresh, real ECS_C memory replaying exactly the LIVE observations (relational_test_util.h).
 */
#include "relational_test_util.h"

int ecs_c_rel_testing_damage(elpis_ecs_c_rel *v, const elpis_ecs_mem_digest *key, uint32_t what, uint64_t value);

static elpis_ecs_mem_answer q_view(const elpis_ecs_c_rel *v, const uint8_t *scope, const elpis_ecs_mem_request *q) {
    elpis_ecs_mem_answer a;
    OK(elpis_ecs_c_rel_query(v, NULL, scope, q, &a, NULL));
    return a;
}
static uint32_t entity_index_of(const elpis_ecs_mem_digest *d) {
    for (uint32_t i = 0; i < 4096; ++i) {
        uint8_t b[32];
        fill(b, 0x656e7469u, i);
        if (!memcmp(b, d->b, 32)) return i;
    }
    return UINT32_MAX;
}

/* ---- 1. Configuration, sizes and the unsynced view ----------------------------------------------------- */

static void test_configuration(void) {
    elpis_ecs_c_rel_config c;
    elpis_ecs_c_rel_config_default(&c);
    uint64_t bytes = 0;
    OK(elpis_ecs_c_rel_required_bytes(&c, &bytes));
    CHECK(bytes > 0 && bytes <= c.byte_budget);
    elpis_ecs_c_rel *v = NULL;
    elpis_ecs_c_rel_config bad = c;
    bad.abi_version = 2;
    RC(elpis_ecs_c_rel_create(&bad, &v), ELPIS_ECS_C_REL_INVALID);
    bad = c, bad.entity_capacity = 1;
    RC(elpis_ecs_c_rel_create(&bad, &v), ELPIS_ECS_C_REL_INVALID);
    bad = c, bad.entity_capacity = ELPIS_ECS_C_REL_MAX_ENTITIES + 1;
    RC(elpis_ecs_c_rel_create(&bad, &v), ELPIS_ECS_C_REL_INVALID);
    bad = c, bad.relation_capacity = 0;
    RC(elpis_ecs_c_rel_create(&bad, &v), ELPIS_ECS_C_REL_INVALID);
    bad = c, bad.contribution_capacity = ELPIS_ECS_C_REL_MAX_CONTRIBUTIONS + 1;
    RC(elpis_ecs_c_rel_create(&bad, &v), ELPIS_ECS_C_REL_INVALID);
    bad = c, bad.byte_budget = bytes - 1;
    RC(elpis_ecs_c_rel_create(&bad, &v), ELPIS_ECS_C_REL_CAPACITY);
    CHECK(v == NULL);
    RC(elpis_ecs_c_rel_create(&c, NULL), ELPIS_ECS_C_REL_INVALID);
    bad = c, bad.byte_budget = bytes;
    OK(elpis_ecs_c_rel_create(&bad, &v));
    elpis_ecs_c_rel_status s;
    OK(elpis_ecs_c_rel_read_status(v, &s));
    CHECK(s.arena_bytes == bytes && !s.synced && !s.relations && !s.contributions && !s.revision);
    CHECK(elpis_ecs_c_rel_abi_version() == ELPIS_ECS_C_REL_ABI_VERSION);
    for (int code = 0; code <= ELPIS_ECS_C_REL_NOMEM; ++code) CHECK(elpis_ecs_c_rel_code_name(code) != NULL);
    CHECK(elpis_ecs_c_rel_code_name(16) == NULL && elpis_ecs_c_rel_code_name(-1) == NULL);

    /* Unsynced: no admission, no read, no check. */
    world w;
    w_init(&w, 1);
    const uint32_t src = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 7, 1, 1);
    const elpis_ecs_mem_observation o = pair(5, ELPIS_ECS_MEM_ASSERT, 1, 7, 1, 2);
    const snapshot before = snap(v);
    RC(admit_as(v, &w, src, &o, NULL), ELPIS_ECS_C_REL_STALE_VIEW);
    const elpis_ecs_mem_request q = ask(5, 1);
    elpis_ecs_mem_answer a;
    RC(elpis_ecs_c_rel_query(v, NULL, NULL, &q, &a, NULL), ELPIS_ECS_C_REL_STALE_VIEW);
    RC(elpis_ecs_c_rel_check(v, w.rows[src].id, 1, w.rows[src].scope), ELPIS_ECS_C_REL_STALE_VIEW);
    RC(elpis_ecs_c_rel_rebuild(v, NULL), ELPIS_ECS_C_REL_STALE_VIEW);
    same(v, &before);
    elpis_ecs_c_rel_destroy(v);
    printf("  configuration and the unsynced view: PASS\n");
}

/* ---- 2. The table: structure and successor laws ---------------------------------------------------------- */

static void test_table_validation(void) {
    elpis_ecs_c_rel *v = make_view(8, 16, 16);
    world w;
    w_init(&w, 2);
    const uint32_t a = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 7, 1, 1);
    const uint32_t b = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 8, 1, 1);
    const snapshot empty = snap(v);
    uint8_t root[32];
    elpis_ecs_c_rel_authority t = w_table(&w, root);
    /* Structure. */
    RC(elpis_ecs_c_rel_sync(v, NULL, NULL), ELPIS_ECS_C_REL_INVALID);
    elpis_ecs_c_rel_authority x = t;
    x.abi_version = 9;
    RC(elpis_ecs_c_rel_sync(v, &x, NULL), ELPIS_ECS_C_REL_INVALID);
    x = t, x.generation = 0;
    RC(elpis_ecs_c_rel_sync(v, &x, NULL), ELPIS_ECS_C_REL_INVALID);
    x = t, memset(x.lineage, 0, 32);
    RC(elpis_ecs_c_rel_sync(v, &x, NULL), ELPIS_ECS_C_REL_INVALID);
    x = t, x.entity_count = 1;
    RC(elpis_ecs_c_rel_sync(v, &x, NULL), ELPIS_ECS_C_REL_INVALID);
    world y = w;
    y.rows[b].source_authority = 7; /* two present sources on one 32-bit source_authority */
    RC(w_sync(v, &y, NULL), ELPIS_ECS_C_REL_INVALID);
    y = w, memcpy(y.rows[b].id, y.rows[a].id, 32);
    RC(w_sync(v, &y, NULL), ELPIS_ECS_C_REL_INVALID);
    y = w, y.rows[a].kind = 9;
    RC(w_sync(v, &y, NULL), ELPIS_ECS_C_REL_INVALID);
    y = w, y.rows[a].lifecycle = 0;
    RC(w_sync(v, &y, NULL), ELPIS_ECS_C_REL_INVALID);
    y = w, y.rows[a].source_version = 0;
    RC(w_sync(v, &y, NULL), ELPIS_ECS_C_REL_INVALID);
    y = w, memset(y.rows[a].scope, 0, 32);
    RC(w_sync(v, &y, NULL), ELPIS_ECS_C_REL_INVALID);
    y = w, y.rows[a].per_turn_limit = 257;
    RC(w_sync(v, &y, NULL), ELPIS_ECS_C_REL_INVALID);
    y = w, y.rows[a].reserved2 = 1;
    RC(w_sync(v, &y, NULL), ELPIS_ECS_C_REL_INVALID);
    y = w, y.rows[0].source_authority = 3; /* an operator carries no source fields */
    RC(w_sync(v, &y, NULL), ELPIS_ECS_C_REL_INVALID);
    y = w, y.rows[a].founding_index = y.next_founding;
    RC(w_sync(v, &y, NULL), ELPIS_ECS_C_REL_INVALID);
    same(v, &empty);
    /* Capacity of mirrored rows. */
    elpis_ecs_c_rel *small = make_view(3, 4, 4);
    RC(w_sync(small, &w, NULL), ELPIS_ECS_C_REL_CAPACITY);
    elpis_ecs_c_rel_destroy(small);

    /* The first table is accepted; idempotent re-sync. */
    elpis_ecs_c_rel_sync_result r;
    OK(w_sync(v, &w, &r));
    CHECK(!r.noop);
    const snapshot first = snap(v);
    OK(w_sync(v, &w, &r));
    CHECK(r.noop == 1);
    same(v, &first);

    /* Successor laws: each refused REGRESSION or LINEAGE, the view unchanged. */
    y = w, y.generation--;
    RC(w_sync(v, &y, NULL), ELPIS_ECS_C_REL_REGRESSION);
    y = w, y.lineage[0] ^= 1;
    RC(w_sync(v, &y, NULL), ELPIS_ECS_C_REL_LINEAGE);
    y = w, y.rows[a].lifecycle = ELPIS_ECS_C_DORMANT; /* same generation, other content */
    RC(w_sync(v, &y, NULL), ELPIS_ECS_C_REL_REGRESSION);
    {
        /* Same generation, another state root. */
        world z = w;
        uint8_t zr[32];
        elpis_ecs_c_rel_authority zt = w_table(&z, zr);
        uint8_t other[32];
        memcpy(other, zt.state_root, 32);
        other[0] ^= 1;
        memcpy(zt.state_root, other, 32);
        RC(elpis_ecs_c_rel_sync(v, &zt, NULL), ELPIS_ECS_C_REL_REGRESSION);
    }
    y = w, w_version(&y, a, 3, 1); /* advance to version 3 */
    OK(w_sync(v, &y, NULL));
    world z = y;
    z.rows[a].source_version = 2, w_bump(&z); /* a version that decreases */
    RC(w_sync(v, &z, NULL), ELPIS_ECS_C_REL_REGRESSION);
    z = y, fill(z.rows[a].scope, 0x73636f70u, 99), w_bump(&z); /* a scope change without a version */
    RC(w_sync(v, &z, NULL), ELPIS_ECS_C_REL_REGRESSION);
    z = y, z.rows[a].per_turn_limit = 3, w_bump(&z);
    RC(w_sync(v, &z, NULL), ELPIS_ECS_C_REL_REGRESSION);
    z = y, z.rows[a].source_authority = 70, w_bump(&z); /* identities are immutable */
    RC(w_sync(v, &z, NULL), ELPIS_ECS_C_REL_REGRESSION);
    z = y, z.rows[a].kind = ELPIS_ECS_C_OPERATOR, z.rows[a].source_authority = 0, z.rows[a].per_turn_limit = 0;
    z.rows[a].source_version = 0, memset(z.rows[a].scope, 0, 32), w_bump(&z);
    RC(w_sync(v, &z, NULL), ELPIS_ECS_C_REL_REGRESSION);
    z = y, z.rows[a].lifecycle = ELPIS_ECS_C_FOUNDED, w_bump(&z); /* FOUNDED is initial only */
    RC(w_sync(v, &z, NULL), ELPIS_ECS_C_REL_REGRESSION);
    z = y, w_reclaim(&z, b), z.next_founding = 3; /* the founding index never decreases */
    RC(w_sync(v, &z, NULL), ELPIS_ECS_C_REL_REGRESSION);
    z = y, w_reclaim(&z, b), z.reclaimed--; /* a row vanished without a reclamation */
    RC(w_sync(v, &z, NULL), ELPIS_ECS_C_REL_REGRESSION);
    z = y, w_reclaim(&z, 1); /* a founder is never reclaimed */
    RC(w_sync(v, &z, NULL), ELPIS_ECS_C_REL_REGRESSION);
    {
        /* An entity founded before the synced coordinate cannot appear later. */
        z = y;
        const uint32_t late = w_found(&z, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 9, 1, 1);
        z.rows[late].founding_index = 1, fill(z.rows[late].id, 0x1234u, 77);
        RC(w_sync(v, &z, NULL), ELPIS_ECS_C_REL_REGRESSION);
    }
    z = y, w_lifecycle(&z, a, ELPIS_ECS_C_TERMINATED);
    OK(w_sync(v, &z, NULL));
    world back = z;
    back.rows[a].lifecycle = ELPIS_ECS_C_ACTIVE, w_bump(&back); /* TERMINATED is final */
    RC(w_sync(v, &back, NULL), ELPIS_ECS_C_REL_REGRESSION);
    w_reclaim(&z, a);
    OK(w_sync(v, &z, NULL));
    back = z, back.reclaimed = 0, w_bump(&back); /* the reclaimed count never decreases */
    RC(w_sync(v, &back, NULL), ELPIS_ECS_C_REL_REGRESSION);
    /* After a REGRESSION, the view keeps its synced coordinate until it is reset. */
    elpis_ecs_c_rel_status s;
    OK(elpis_ecs_c_rel_read_status(v, &s));
    CHECK(s.generation == z.generation);
    OK(elpis_ecs_c_rel_reset(v));
    OK(elpis_ecs_c_rel_read_status(v, &s));
    CHECK(!s.synced && !s.entities && s.resets == 1);
    OK(w_sync(v, &back, NULL)); /* a reset view accepts any well-formed table */
    elpis_ecs_c_rel_destroy(v);
    printf("  authority table: structure, successor laws, idempotent re-sync: PASS\n");
}

/* ---- 3. The admission law ----------------------------------------------------------------------------- */

static void test_admission(void) {
    elpis_ecs_c_rel *v = make_view(16, 16, 16);
    world w;
    w_init(&w, 3);
    const uint32_t a = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 7, 3, 1);
    const uint32_t dormant = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_DORMANT, 8, 1, 1);
    const uint32_t founded = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_FOUNDED, 9, 1, 1);
    OK(w_sync(v, &w, NULL));
    elpis_ecs_mem_observation o = pair(5, ELPIS_ECS_MEM_ASSERT, 10, 7, 1, 2);
    elpis_ecs_mem_digest key;
    OK(admit_as(v, &w, a, &o, &key));
    elpis_ecs_mem_digest expect;
    CHECK(elpis_ecs_mem_relation_key(&o, &expect) == ELPIS_ECS_MEM_OK && !memcmp(expect.b, key.b, 32));
    const snapshot base = snap(v);
    const elpis_ecs_c_rel_coordinate c = w_coord(&w);
    const elpis_ecs_c_entity *e = &w.rows[a];
    uint8_t other_scope[32], unknown[32];
    fill(other_scope, 0x73636f70u, 2);
    fill(unknown, 0x756e6b6eu, 1);
    /* Each refusal of the authority's law, then the view's own. */
    RC(elpis_ecs_c_rel_admit(v, &c, unknown, 3, e->scope, &o, NULL, NULL), ELPIS_ECS_C_REL_UNKNOWN_SOURCE);
    RC(elpis_ecs_c_rel_admit(v, &c, w.rows[0].id, 0, w.rows[0].scope, &o, NULL, NULL), ELPIS_ECS_C_REL_NOT_SOURCE);
    o.source_authority = 8;
    RC(admit_as(v, &w, dormant, &o, NULL), ELPIS_ECS_C_REL_LIFECYCLE);
    o.source_authority = 9;
    RC(admit_as(v, &w, founded, &o, NULL), ELPIS_ECS_C_REL_LIFECYCLE);
    o.source_authority = 7;
    RC(elpis_ecs_c_rel_admit(v, &c, e->id, 2, e->scope, &o, NULL, NULL), ELPIS_ECS_C_REL_SOURCE_STALE);
    RC(elpis_ecs_c_rel_admit(v, &c, e->id, 4, e->scope, &o, NULL, NULL), ELPIS_ECS_C_REL_SOURCE_STALE);
    RC(elpis_ecs_c_rel_admit(v, &c, e->id, 3, other_scope, &o, NULL, NULL), ELPIS_ECS_C_REL_SCOPE_CONFLICT);
    o.source_authority = 8; /* another source's 32-bit identifier */
    RC(admit_as(v, &w, a, &o, NULL), ELPIS_ECS_C_REL_BINDING_MISMATCH);
    o.source_authority = 7;
    o.tick = 9; /* older than the clock */
    RC(admit_as(v, &w, a, &o, NULL), ELPIS_ECS_C_REL_CLOCK);
    o.tick = 10;
    elpis_ecs_c_rel_coordinate stale = c;
    stale.generation--;
    RC(elpis_ecs_c_rel_admit(v, &stale, e->id, 3, e->scope, &o, NULL, NULL), ELPIS_ECS_C_REL_STALE_VIEW);
    stale = c, stale.state_root[5] ^= 1;
    RC(elpis_ecs_c_rel_admit(v, &stale, e->id, 3, e->scope, &o, NULL, NULL), ELPIS_ECS_C_REL_STALE_VIEW);
    RC(elpis_ecs_c_rel_admit(v, NULL, e->id, 3, e->scope, &o, NULL, NULL), ELPIS_ECS_C_REL_INVALID);
    /* Malformed observations: the memory's own validation. */
    elpis_ecs_mem_observation m = o;
    m.polarity = 0;
    RC(admit_as(v, &w, a, &m, NULL), ELPIS_ECS_C_REL_INVALID);
    m = o, m.polarity = 4;
    RC(admit_as(v, &w, a, &m, NULL), ELPIS_ECS_C_REL_INVALID);
    m = o, m.participants[1].role = 1; /* an n-ary role collision: (role, ordinal) repeated */
    RC(admit_as(v, &w, a, &m, NULL), ELPIS_ECS_C_REL_INVALID);
    m = o, m.participant_count = 0;
    RC(admit_as(v, &w, a, &m, NULL), ELPIS_ECS_C_REL_INVALID);
    m = o, m.participant_count = 9;
    RC(admit_as(v, &w, a, &m, NULL), ELPIS_ECS_C_REL_INVALID);
    m = o, m.reserved = 1;
    RC(admit_as(v, &w, a, &m, NULL), ELPIS_ECS_C_REL_INVALID);
    m = o, m.participants[5].role = 3; /* bytes past participant_count */
    RC(admit_as(v, &w, a, &m, NULL), ELPIS_ECS_C_REL_INVALID);
    m = o, m.abi_version = 2;
    RC(admit_as(v, &w, a, &m, NULL), ELPIS_ECS_C_REL_INVALID);
    same(v, &base);
    elpis_ecs_c_rel_status s;
    OK(elpis_ecs_c_rel_read_status(v, &s));
    CHECK(s.refusals == 20 && s.refusals_by_code[ELPIS_ECS_C_REL_INVALID] == 9);
    CHECK(s.refusals_by_code[ELPIS_ECS_C_REL_SOURCE_STALE] == 2 && s.refusals_by_code[ELPIS_ECS_C_REL_LIFECYCLE] == 2);
    /* The pure law agrees with admission. */
    OK(elpis_ecs_c_rel_check(v, e->id, 3, e->scope));
    RC(elpis_ecs_c_rel_check(v, e->id, 2, e->scope), ELPIS_ECS_C_REL_SOURCE_STALE);
    RC(elpis_ecs_c_rel_check(v, w.rows[dormant].id, 1, w.rows[dormant].scope), ELPIS_ECS_C_REL_LIFECYCLE);
    RC(elpis_ecs_c_rel_check(v, unknown, 1, e->scope), ELPIS_ECS_C_REL_UNKNOWN_SOURCE);
    RC(elpis_ecs_c_rel_check(v, w.rows[1].id, 0, w.rows[1].scope), ELPIS_ECS_C_REL_NOT_SOURCE);
    elpis_ecs_c_rel_destroy(v);
    printf("  admission law (UNKNOWN, NOT_SOURCE, LIFECYCLE x2, SOURCE_STALE x2, SCOPE, BINDING, CLOCK, "
           "coordinate, 8 malformed): PASS\n");
}

/* ---- 4. Source VALID -> STALE, revalidation, suspension --------------------------------------------- */

static void test_source_revision(void) {
    elpis_ecs_c_rel *v = make_view(16, 64, 64);
    world w;
    w_init(&w, 4);
    const uint32_t a = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 7, 1, 1);
    OK(w_sync(v, &w, NULL));
    const elpis_ecs_mem_observation o1 = pair(5, ELPIS_ECS_MEM_ASSERT, 1, 7, 1, 2);
    elpis_ecs_mem_digest key;
    OK(admit_as(v, &w, a, &o1, &key));
    OK(admit_as(v, &w, a, &o1, &key)); /* duplicate evidence is counted again (the memory's law) */
    const elpis_ecs_mem_request q = ask(5, 1);
    elpis_ecs_mem_answer ans = q_view(v, NULL, &q);
    CHECK(ans.found && entity_index_of(&ans.answer) == 2 && ans.score == 2);
    elpis_ecs_mem_slot slot;
    OK(elpis_ecs_c_rel_relation_read(v, NULL, NULL, &key, &slot));
    CHECK(slot.assert_count == 2 && slot.last_authority == 7);

    /* Query pinned to the next coordinate before the view has reached it: refused, never answered. */
    world next = w;
    w_version(&next, a, 2, 3);
    const elpis_ecs_c_rel_coordinate cn = w_coord(&next), co = w_coord(&w);
    RC(elpis_ecs_c_rel_query(v, &cn, NULL, &q, &ans, NULL), ELPIS_ECS_C_REL_STALE_VIEW);
    OK(elpis_ecs_c_rel_query(v, &co, NULL, &q, &ans, NULL));
    CHECK(ans.found && ans.score == 2); /* correct for the coordinate it names */

    /* VALID -> STALE: the revision retires the v1 evidence for good. */
    elpis_ecs_c_rel_sync_result r;
    OK(w_sync(v, &next, &r));
    CHECK(r.sources_changed == 1 && r.contributions_retired == 1 && r.relations_released == 1 && !r.contributions_flipped);
    RC(elpis_ecs_c_rel_query(v, &co, NULL, &q, &ans, NULL), ELPIS_ECS_C_REL_STALE_VIEW);
    OK(elpis_ecs_c_rel_query(v, &cn, NULL, &q, &ans, NULL));
    CHECK(!ans.found && !ans.matches);
    RC(elpis_ecs_c_rel_relation_read(v, NULL, NULL, &key, &slot), ELPIS_ECS_C_REL_NOT_FOUND);
    size_t n = 9;
    RC(elpis_ecs_c_rel_support_of(v, &key, NULL, 0, &n), ELPIS_ECS_C_REL_NOT_FOUND);
    OK(elpis_ecs_c_rel_dependents(v, next.rows[a].id, NULL, 0, &n));
    CHECK(n == 0);

    /* Revalidation under the new version admits nothing old. */
    w = next;
    elpis_ecs_mem_observation o2 = o1;
    o2.tick = 2;
    const elpis_ecs_c_rel_coordinate c2 = w_coord(&w);
    RC(elpis_ecs_c_rel_admit(v, &c2, w.rows[a].id, 1, w.rows[a].scope, &o2, NULL, NULL), ELPIS_ECS_C_REL_SOURCE_STALE);
    OK(admit_as(v, &w, a, &o2, NULL));
    OK(elpis_ecs_c_rel_relation_read(v, NULL, NULL, &key, &slot));
    CHECK(slot.assert_count == 1 && slot.first_tick == 2); /* only the new version's evidence */

    /* DORMANT suspends (same binding), ACTIVE at the same version restores it exactly. */
    const elpis_ecs_mem_slot before = slot;
    w_lifecycle(&w, a, ELPIS_ECS_C_DORMANT);
    OK(w_sync(v, &w, &r));
    CHECK(r.contributions_flipped == 1 && !r.contributions_retired && r.relations_recomputed == 1);
    ans = q_view(v, NULL, &q);
    CHECK(!ans.found && !ans.matches);
    elpis_ecs_c_rel_support sup[2];
    OK(elpis_ecs_c_rel_support_of(v, &key, sup, 2, &n));
    CHECK(n == 1 && sup[0].state == ELPIS_ECS_C_REL_SUSPENDED && sup[0].version == 2 && sup[0].assert_count == 1);
    CHECK(!memcmp(sup[0].source, w.rows[a].id, 32) && sup[0].source_authority == 7);
    elpis_ecs_c_rel_status s;
    OK(elpis_ecs_c_rel_read_status(v, &s));
    CHECK(s.relations == 1 && s.live_relations == 0 && s.contributions == 1 && s.live_contributions == 0);
    w_lifecycle(&w, a, ELPIS_ECS_C_ACTIVE);
    OK(w_sync(v, &w, &r));
    CHECK(r.contributions_flipped == 1);
    OK(elpis_ecs_c_rel_relation_read(v, NULL, NULL, &key, &slot));
    CHECK(slots_equal(&slot, &before));

    /* A version raised while DORMANT retires the suspended evidence too. */
    w_lifecycle(&w, a, ELPIS_ECS_C_DORMANT);
    OK(w_sync(v, &w, NULL));
    w_version(&w, a, 3, 3);
    w_lifecycle(&w, a, ELPIS_ECS_C_ACTIVE);
    OK(w_sync(v, &w, &r));
    CHECK(r.contributions_retired == 1 && r.relations_released == 1);
    ans = q_view(v, NULL, &q);
    CHECK(!ans.found);

    /* TERMINATED retires; reclamation releases the row; a new source reusing the 32-bit identifier sees nothing. */
    o2.tick = 3;
    OK(admit_as(v, &w, a, &o2, NULL));
    w_lifecycle(&w, a, ELPIS_ECS_C_TERMINATED);
    OK(w_sync(v, &w, &r));
    CHECK(r.contributions_retired == 1);
    w_reclaim(&w, a);
    OK(w_sync(v, &w, &r));
    const uint32_t a2 = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 7, 1, 1);
    OK(w_sync(v, &w, NULL));
    ans = q_view(v, NULL, &q);
    CHECK(!ans.found);
    OK(elpis_ecs_c_rel_dependents(v, w.rows[a2].id, NULL, 0, &n));
    CHECK(n == 0);
    uint64_t bad = 1;
    OK(elpis_ecs_c_rel_verify(v, &bad));
    CHECK(bad == 0);
    elpis_ecs_c_rel_destroy(v);
    printf("  source revision: VALID->STALE, pinned reads, revalidation, DORMANT suspension, TERMINATED, reuse: PASS\n");
}

/* ---- 5. Scopes, n-ary roles, shared sources, contradictions ------------------------------------------ */

static void test_scopes_and_roles(void) {
    elpis_ecs_c_rel *v = make_view(16, 64, 64);
    world w;
    w_init(&w, 5);
    const uint32_t a = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 7, 1, 1);
    const uint32_t b = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 8, 1, 2);
    OK(w_sync(v, &w, NULL));
    /* Two scopes assert different fillers for one cue. */
    const elpis_ecs_mem_observation oa = pair(5, ELPIS_ECS_MEM_ASSERT, 1, 7, 1, 2);
    const elpis_ecs_mem_observation ob = pair(5, ELPIS_ECS_MEM_ASSERT, 2, 8, 1, 3);
    OK(admit_as(v, &w, a, &oa, NULL));
    OK(admit_as(v, &w, b, &ob, NULL));
    const elpis_ecs_mem_request q = ask(5, 1);
    const elpis_ecs_mem_answer all = q_view(v, NULL, &q);
    CHECK(all.found && entity_index_of(&all.answer) == 3 && all.conflicts == 1 && all.matches == 2);
    const elpis_ecs_mem_answer s1 = q_view(v, w.rows[a].scope, &q);
    CHECK(s1.found && entity_index_of(&s1.answer) == 2 && s1.conflicts == 0 && s1.matches == 1);
    const elpis_ecs_mem_answer s2 = q_view(v, w.rows[b].scope, &q);
    CHECK(s2.found && entity_index_of(&s2.answer) == 3 && s2.conflicts == 0 && s2.matches == 1);
    uint8_t nowhere[32];
    fill(nowhere, 0x73636f70u, 9);
    const elpis_ecs_mem_answer s3 = q_view(v, nowhere, &q);
    CHECK(!s3.found && !s3.matches);
    /* LOOKUP only: the associative channels are not the view's. */
    elpis_ecs_mem_request other = q;
    elpis_ecs_mem_answer none;
    other.channel = ELPIS_ECS_MEM_FAST;
    RC(elpis_ecs_c_rel_query(v, NULL, NULL, &other, &none, NULL), ELPIS_ECS_C_REL_INVALID);
    other.channel = ELPIS_ECS_MEM_SLOW;
    RC(elpis_ecs_c_rel_query(v, NULL, NULL, &other, &none, NULL), ELPIS_ECS_C_REL_INVALID);
    other.channel = 0;
    RC(elpis_ecs_c_rel_query(v, NULL, NULL, &other, &none, NULL), ELPIS_ECS_C_REL_INVALID);
    /* Cross-scope admission is refused: a source admits only into its own current scope. */
    const elpis_ecs_c_rel_coordinate c = w_coord(&w);
    RC(elpis_ecs_c_rel_admit(v, &c, w.rows[a].id, 1, w.rows[b].scope, &oa, NULL, NULL), ELPIS_ECS_C_REL_SCOPE_CONFLICT);
    /* A scope change (with its version) moves nothing across: the old scope's evidence is retired. */
    w_version(&w, a, 2, 2);
    OK(w_sync(v, &w, NULL));
    const elpis_ecs_mem_answer moved = q_view(v, w.rows[a].scope, &q); /* scope 2 now: only b's */
    CHECK(moved.found && entity_index_of(&moved.answer) == 3 && moved.matches == 1);
    fill(nowhere, 0x73636f70u, 1);
    CHECK(!q_view(v, nowhere, &q).matches);

    /* N-ary relations: arity 8, role identity preserved, swapped roles are another relation. */
    const uint32_t roles8[8] = {1, 2, 3, 4, 5, 6, 7, 8}, ents8[8] = {11, 12, 13, 14, 15, 16, 17, 18};
    elpis_ecs_mem_observation o8 = obs(6, ELPIS_ECS_MEM_ASSERT, 3, 8, 8, roles8, ents8);
    elpis_ecs_mem_digest k8, kswap;
    OK(admit_as(v, &w, b, &o8, &k8));
    elpis_ecs_mem_observation swapped = o8;
    swapped.participants[0].entity = o8.participants[1].entity;
    swapped.participants[1].entity = o8.participants[0].entity;
    OK(admit_as(v, &w, b, &swapped, &kswap));
    CHECK(memcmp(k8.b, kswap.b, 32));
    const uint32_t cue_roles[7] = {1, 2, 3, 4, 5, 6, 7}, cue_ents[7] = {11, 12, 13, 14, 15, 16, 17};
    const elpis_ecs_mem_request q8 = lookup(6, 8, 7, cue_roles, cue_ents);
    const elpis_ecs_mem_answer a8 = q_view(v, NULL, &q8);
    CHECK(a8.found && entity_index_of(&a8.answer) == 18 && a8.matches == 1 && !memcmp(a8.answer_key.b, k8.b, 32));
    const uint32_t r2[1] = {2}, e2[1] = {11}; /* entity 11 in role 2 exists only in the swapped relation */
    const elpis_ecs_mem_request qs = lookup(6, 1, 1, r2, e2);
    const elpis_ecs_mem_answer as = q_view(v, NULL, &qs);
    CHECK(as.found && entity_index_of(&as.answer) == 12 && !memcmp(as.answer_key.b, kswap.b, 32));
    /* Participants in any order are one relation, stored canonically: the memory's slot exactly. */
    elpis_ecs_mem_observation reversed = o8;
    for (uint32_t i = 0; i < 8; ++i) reversed.participants[i] = o8.participants[7 - i];
    reversed.tick = 3;
    elpis_ecs_mem_digest krev;
    OK(admit_as(v, &w, b, &reversed, &krev));
    CHECK(!memcmp(krev.b, k8.b, 32));
    {
        elpis_ecs_mem_config mc;
        elpis_ecs_mem_config_default(&mc);
        mc.dim = 64, mc.banks = 1, mc.slot_capacity = 4, mc.codebook_capacity = 64, mc.byte_budget = 1u << 24;
        elpis_ecs_mem *m = NULL;
        CHECK(elpis_ecs_mem_create(&mc, &m) == ELPIS_ECS_MEM_OK);
        CHECK(elpis_ecs_mem_observe(m, 0, &o8, NULL, NULL) == ELPIS_ECS_MEM_OK);
        CHECK(elpis_ecs_mem_observe(m, 0, &reversed, NULL, NULL) == ELPIS_ECS_MEM_OK);
        elpis_ecs_mem_slot want, got;
        CHECK(elpis_ecs_mem_slot_read(m, &k8, &want) == ELPIS_ECS_MEM_OK);
        OK(elpis_ecs_c_rel_relation_read(v, NULL, NULL, &k8, &got));
        CHECK(slots_equal(&want, &got) && got.participants[0].role == 1 && got.participants[7].role == 8);
        elpis_ecs_mem_destroy(m);
    }
    /* The same entity in two roles is legal (distinct (role, ordinal)). */
    const uint32_t rr[2] = {1, 2}, ee[2] = {20, 20};
    const elpis_ecs_mem_observation self = obs(7, ELPIS_ECS_MEM_ASSERT, 4, 8, 2, rr, ee);
    OK(admit_as(v, &w, b, &self, NULL));
    /* Ordinals carry multiplicity within a role. */
    elpis_ecs_mem_observation multi = self;
    multi.participants[1].role = 1;
    multi.participants[1].ordinal = 1;
    OK(admit_as(v, &w, b, &multi, NULL));
    uint64_t bad = 1;
    OK(elpis_ecs_c_rel_verify(v, &bad));
    CHECK(bad == 0);
    elpis_ecs_c_rel_destroy(v);
    printf("  scopes (filtered reads, cross-scope refusal, scope moves) and n-ary role identity: PASS\n");
}

static void test_shared_dependencies_and_contradiction(void) {
    elpis_ecs_c_rel *v = make_view(16, 256, 512);
    world w;
    w_init(&w, 6);
    const uint32_t a = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 7, 1, 1);
    const uint32_t b = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 8, 1, 1);
    const uint32_t c = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 9, 1, 1);
    OK(w_sync(v, &w, NULL));
    uint64_t tick = 1;
    /* a supports R1..R5; b supports R3; c supports 100 unrelated relations. */
    for (uint32_t i = 1; i <= 5; ++i) {
        const elpis_ecs_mem_observation o = pair(5, ELPIS_ECS_MEM_ASSERT, tick++, 7, i, 100 + i);
        OK(admit_as(v, &w, a, &o, NULL));
    }
    const elpis_ecs_mem_observation ob = pair(5, ELPIS_ECS_MEM_ASSERT, tick++, 8, 3, 103);
    OK(admit_as(v, &w, b, &ob, NULL));
    for (uint32_t i = 0; i < 100; ++i) {
        const elpis_ecs_mem_observation o = pair(9, ELPIS_ECS_MEM_ASSERT, tick++, 9, 1000 + i, 2000 + i);
        OK(admit_as(v, &w, c, &o, NULL));
    }
    elpis_ecs_mem_digest deps[8];
    size_t n = 0;
    RC(elpis_ecs_c_rel_dependents(v, w.rows[a].id, deps, 4, &n), ELPIS_ECS_C_REL_CAPACITY);
    CHECK(n == 5);
    OK(elpis_ecs_c_rel_dependents(v, w.rows[a].id, deps, 8, &n));
    CHECK(n == 5);
    /* Revoke a: exactly its five contributions are visited; c's hundred relations are not. */
    w_version(&w, a, 2, 1);
    elpis_ecs_c_rel_sync_result r;
    OK(w_sync(v, &w, &r));
    CHECK(r.sources_changed == 1 && r.contributions_retired == 5 && r.relations_released == 4);
    CHECK(r.relations_recomputed == 1 && r.work.contributions == 5 + 1 && r.work.recomputes == 1);
    CHECK(r.work.relations == 5);
    for (uint32_t i = 1; i <= 5; ++i) {
        const elpis_ecs_mem_request q = ask(5, i);
        const elpis_ecs_mem_answer ans = q_view(v, NULL, &q);
        if (i == 3) CHECK(ans.found && ans.score == 1 && entity_index_of(&ans.answer) == 103);
        else CHECK(!ans.found && !ans.matches);
    }
    elpis_ecs_c_rel_support sup[2];
    elpis_ecs_mem_digest k3;
    {
        const elpis_ecs_mem_observation o3 = pair(5, ELPIS_ECS_MEM_ASSERT, 0, 0, 3, 103);
        CHECK(elpis_ecs_mem_relation_key(&o3, &k3) == ELPIS_ECS_MEM_OK);
    }
    OK(elpis_ecs_c_rel_support_of(v, &k3, sup, 2, &n));
    CHECK(n == 1 && !memcmp(sup[0].source, w.rows[b].id, 32) && sup[0].state == ELPIS_ECS_C_REL_LIVE);

    /* Contradictory candidate relations: assert vs negate of one relation. Net 0 withholds an answer; revoking
     * either side decides it the other way. Contradiction never chooses which source is right. */
    const elpis_ecs_mem_observation pa = pair(10, ELPIS_ECS_MEM_ASSERT, tick++, 9, 50, 51);
    const elpis_ecs_mem_observation pb = pair(10, ELPIS_ECS_MEM_NEGATE, tick++, 8, 50, 51);
    OK(admit_as(v, &w, c, &pa, NULL));
    OK(admit_as(v, &w, b, &pb, NULL));
    const elpis_ecs_mem_request qc = ask(10, 50);
    elpis_ecs_mem_answer ans = q_view(v, NULL, &qc);
    CHECK(!ans.found && ans.matches == 1 && !ans.contradicted);
    world wb = w;
    w_lifecycle(&wb, b, ELPIS_ECS_C_DORMANT);
    OK(w_sync(v, &wb, NULL));
    ans = q_view(v, NULL, &qc);
    CHECK(ans.found && ans.score == 1 && entity_index_of(&ans.answer) == 51);
    w_lifecycle(&wb, b, ELPIS_ECS_C_ACTIVE);
    w_lifecycle(&wb, c, ELPIS_ECS_C_DORMANT);
    OK(w_sync(v, &wb, NULL));
    ans = q_view(v, NULL, &qc);
    CHECK(!ans.found && ans.matches == 1 && ans.contradicted == 1);
    uint64_t bad = 1;
    OK(elpis_ecs_c_rel_verify(v, &bad));
    CHECK(bad == 0);
    elpis_ecs_c_rel_destroy(v);
    printf("  shared-source dependencies (5 of 106 relations visited) and contradictory evidence: PASS\n");
}

/* ---- 6. Ordering, duplicates, determinism -------------------------------------------------------------- */

static void test_order_and_duplicates(void) {
    world w;
    w_init(&w, 7);
    const uint32_t a = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 7, 1, 1);
    const uint32_t b = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 8, 1, 1);
    /* Twelve observations at one tick, admitted in 40 random orders: every answer and count agrees. */
    elpis_ecs_mem_observation set[12];
    uint32_t who[12];
    for (uint32_t i = 0; i < 12; ++i) {
        set[i] = pair(5, i % 3 == 2 ? ELPIS_ECS_MEM_NEGATE : ELPIS_ECS_MEM_ASSERT, 5, i % 2 ? 8 : 7, i % 4, 10 + i % 3);
        who[i] = i % 2 ? b : a;
    }
    elpis_ecs_mem_answer reference[4];
    for (uint32_t trial = 0; trial < 40; ++trial) {
        uint32_t order[12];
        for (uint32_t i = 0; i < 12; ++i) order[i] = i;
        for (uint32_t i = 11; trial && i > 0; --i) {
            const uint32_t j = below(i + 1), t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
        elpis_ecs_c_rel *v = make_view(8, 32, 64);
        OK(w_sync(v, &w, NULL));
        for (uint32_t i = 0; i < 12; ++i) OK(admit_as(v, &w, who[order[i]], &set[order[i]], NULL));
        for (uint32_t k = 0; k < 4; ++k) {
            const elpis_ecs_mem_request q = ask(5, k);
            const elpis_ecs_mem_answer ans = q_view(v, NULL, &q);
            if (!trial) reference[k] = ans;
            else CHECK(answers_equal(&ans, &reference[k]));
        }
        elpis_ecs_c_rel_destroy(v);
    }
    /* Determinism: two views fed one sequence agree byte for byte; one more admission changes the identity. */
    elpis_ecs_c_rel *x = make_view(8, 32, 64), *y = make_view(8, 32, 64);
    OK(w_sync(x, &w, NULL));
    OK(w_sync(y, &w, NULL));
    for (uint32_t i = 0; i < 12; ++i) {
        OK(admit_as(x, &w, who[i], &set[i], NULL));
        OK(admit_as(y, &w, who[i], &set[i], NULL));
    }
    uint8_t ix[32], iy[32];
    OK(elpis_ecs_c_rel_identity(x, ix));
    OK(elpis_ecs_c_rel_identity(y, iy));
    CHECK(!memcmp(ix, iy, 32));
    OK(admit_as(y, &w, who[0], &set[0], NULL)); /* a duplicate is evidence again, never a no-op */
    OK(elpis_ecs_c_rel_identity(y, iy));
    CHECK(memcmp(ix, iy, 32));
    elpis_ecs_c_rel_destroy(x);
    elpis_ecs_c_rel_destroy(y);
    printf("  reordered (40 permutations) and duplicate observations, determinism: PASS\n");
}

/* ---- 7. Resources: capacity, counters, atomic outputs ------------------------------------------------- */

static void test_resources(void) {
    elpis_ecs_c_rel *v = make_view(8, 4, 6);
    world w;
    w_init(&w, 8);
    const uint32_t a = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 7, 1, 1);
    const uint32_t b = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 8, 1, 1);
    OK(w_sync(v, &w, NULL));
    elpis_ecs_mem_digest keys[4];
    for (uint32_t i = 0; i < 4; ++i) {
        const elpis_ecs_mem_observation o = pair(5, ELPIS_ECS_MEM_ASSERT, i + 1, 7, i, 50);
        OK(admit_as(v, &w, a, &o, &keys[i]));
    }
    snapshot full = snap(v);
    const elpis_ecs_mem_observation fifth = pair(5, ELPIS_ECS_MEM_ASSERT, 9, 7, 9, 50);
    RC(admit_as(v, &w, a, &fifth, NULL), ELPIS_ECS_C_REL_CAPACITY); /* relations full */
    same(v, &full);
    /* Existing relations still admit; contributions run out at six. */
    const elpis_ecs_mem_observation again = pair(5, ELPIS_ECS_MEM_ASSERT, 9, 7, 0, 50);
    OK(admit_as(v, &w, a, &again, NULL));
    const elpis_ecs_mem_observation bo0 = pair(5, ELPIS_ECS_MEM_ASSERT, 9, 8, 0, 50);
    const elpis_ecs_mem_observation bo1 = pair(5, ELPIS_ECS_MEM_ASSERT, 9, 8, 1, 50);
    const elpis_ecs_mem_observation bo2 = pair(5, ELPIS_ECS_MEM_ASSERT, 9, 8, 2, 50);
    OK(admit_as(v, &w, b, &bo0, NULL));
    OK(admit_as(v, &w, b, &bo1, NULL));
    full = snap(v);
    RC(admit_as(v, &w, b, &bo2, NULL), ELPIS_ECS_C_REL_CAPACITY);
    same(v, &full);
    /* Retirement returns capacity (never eviction of admitted evidence). */
    w_version(&w, a, 2, 1);
    elpis_ecs_c_rel_sync_result r;
    OK(w_sync(v, &w, &r));
    CHECK(r.contributions_retired == 4 && r.relations_released == 2);
    OK(admit_as(v, &w, b, &bo2, NULL));
    /* Rows: a table beyond the entity capacity is refused whole. */
    world big = w;
    for (uint32_t i = 0; i < 5; ++i) (void)w_found(&big, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 20 + i, 1, 1);
    full = snap(v);
    RC(w_sync(v, &big, NULL), ELPIS_ECS_C_REL_CAPACITY);
    same(v, &full);
    /* The polarity total's overflow guard (the memory refuses at UINT32_MAX too). */
    const elpis_ecs_mem_observation ob1 = pair(5, ELPIS_ECS_MEM_ASSERT, 10, 8, 1, 50);
    elpis_ecs_mem_digest k1;
    CHECK(elpis_ecs_mem_relation_key(&ob1, &k1) == ELPIS_ECS_MEM_OK);
    OK(ecs_c_rel_testing_damage(v, &k1, 5, UINT32_MAX));
    full = snap(v);
    RC(admit_as(v, &w, b, &ob1, NULL), ELPIS_ECS_C_REL_CLOCK);
    same_state(v, &full); /* (verify reports the poked total, by design) */
    OK(ecs_c_rel_testing_damage(v, &k1, 5, 1));
    /* All-or-nothing outputs: a short buffer gets nothing, and the count it needs. */
    elpis_ecs_mem_slot out[4];
    memset(out, 0xA5, sizeof(out));
    size_t n = 0;
    RC(elpis_ecs_c_rel_export(v, NULL, NULL, out, 1, &n, NULL), ELPIS_ECS_C_REL_CAPACITY);
    CHECK(n == 3);
    for (size_t i = 0; i < sizeof(out); ++i) CHECK(((const uint8_t *)out)[i] == 0xA5);
    OK(elpis_ecs_c_rel_export(v, NULL, NULL, out, 4, &n, NULL));
    CHECK(n == 3 && memcmp(out[0].key.b, out[1].key.b, 32) < 0 && memcmp(out[1].key.b, out[2].key.b, 32) < 0);
    RC(elpis_ecs_c_rel_dependents(v, w.rows[b].id, NULL, 0, &n), ELPIS_ECS_C_REL_CAPACITY);
    CHECK(n == 3);
    RC(elpis_ecs_c_rel_support_of(v, &keys[0], NULL, 0, &n), ELPIS_ECS_C_REL_CAPACITY);
    CHECK(n == 1);
    uint64_t bad = 1;
    OK(elpis_ecs_c_rel_verify(v, &bad));
    CHECK(bad == 0);
    elpis_ecs_c_rel_destroy(v);
    printf("  resources: relation/contribution/row capacity, overflow guard, all-or-nothing outputs: PASS\n");
}

/* ---- 8. Damaged retained state -------------------------------------------------------------------------- */

static void test_damage(void) {
    elpis_ecs_c_rel *v = make_view(8, 16, 16);
    world w;
    w_init(&w, 9);
    const uint32_t a = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 7, 1, 1);
    const uint32_t b = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 8, 1, 2);
    OK(w_sync(v, &w, NULL));
    elpis_ecs_mem_digest k, k2;
    const elpis_ecs_mem_observation o = pair(5, ELPIS_ECS_MEM_ASSERT, 1, 7, 1, 2);
    const elpis_ecs_mem_observation o2 = pair(5, ELPIS_ECS_MEM_ASSERT, 2, 8, 1, 2);
    OK(admit_as(v, &w, a, &o, &k));
    OK(admit_as(v, &w, b, &o2, &k2));
    CHECK(!memcmp(k.b, k2.b, 32));
    const elpis_ecs_mem_request q = ask(5, 1);
    const elpis_ecs_mem_answer good = q_view(v, NULL, &q);
    const snapshot clean = snap(v);
    elpis_ecs_mem_answer ans;
    elpis_ecs_mem_slot slot;
    uint64_t bad = 0;

    /* 1: a derived aggregate damaged in place: every read refuses; rebuild restores it exactly. */
    OK(ecs_c_rel_testing_damage(v, &k, 1, 77));
    RC(elpis_ecs_c_rel_query(v, NULL, NULL, &q, &ans, NULL), ELPIS_ECS_C_REL_CORRUPT);
    RC(elpis_ecs_c_rel_relation_read(v, NULL, NULL, &k, &slot), ELPIS_ECS_C_REL_CORRUPT);
    size_t n;
    RC(elpis_ecs_c_rel_export(v, NULL, NULL, NULL, 0, &n, NULL), ELPIS_ECS_C_REL_CORRUPT);
    OK(elpis_ecs_c_rel_verify(v, &bad));
    CHECK(bad >= 1);
    const elpis_ecs_c_rel_coordinate c = w_coord(&w);
    const elpis_ecs_mem_observation more = pair(5, ELPIS_ECS_MEM_ASSERT, 3, 7, 1, 2);
    RC(elpis_ecs_c_rel_admit(v, &c, w.rows[a].id, 1, w.rows[a].scope, &more, NULL, NULL), ELPIS_ECS_C_REL_CORRUPT);
    OK(elpis_ecs_c_rel_rebuild(v, NULL));
    ans = q_view(v, NULL, &q);
    CHECK(answers_equal(&ans, &good));
    OK(elpis_ecs_c_rel_verify(v, &bad));
    CHECK(bad == 0);
    {
        uint8_t id[32];
        OK(elpis_ecs_c_rel_identity(v, id));
        CHECK(!memcmp(id, clean.identity, 32)); /* rebuild changes derived state only */
    }

    /* 2: damage that also forged the check: reads cannot see it, verify does, rebuild repairs it. */
    OK(ecs_c_rel_testing_damage(v, &k, 2, 77));
    ans = q_view(v, NULL, &q);
    CHECK(ans.score == 77); /* the forged aggregate is read: this is why verify exists */
    OK(elpis_ecs_c_rel_verify(v, &bad));
    CHECK(bad >= 1);
    OK(elpis_ecs_c_rel_rebuild(v, NULL));
    ans = q_view(v, NULL, &q);
    CHECK(answers_equal(&ans, &good));

    /* 4: a stored liveness flag flipped (check resealed): verify sees it, rebuild recomputes it from the table. */
    OK(ecs_c_rel_testing_damage(v, &k, 4, 0));
    OK(elpis_ecs_c_rel_verify(v, &bad));
    CHECK(bad >= 1);
    OK(elpis_ecs_c_rel_rebuild(v, NULL));
    OK(elpis_ecs_c_rel_verify(v, &bad));
    CHECK(bad == 0);
    ans = q_view(v, NULL, &q);
    CHECK(answers_equal(&ans, &good));

    /* 6: a binding no table can hold (the scope, resealed): verify counts it, rebuild refuses it. Once on a live
     * contribution, once on a suspended one, whose liveness stays consistent so that only the binding invariant
     * can see it. */
    for (int suspended = 0; suspended < 2; ++suspended) {
        if (suspended) {
            w_lifecycle(&w, a, ELPIS_ECS_C_DORMANT);
            OK(w_sync(v, &w, NULL));
        }
        const snapshot sound = snap(v);
        OK(ecs_c_rel_testing_damage(v, &k, 6, 1));
        OK(elpis_ecs_c_rel_verify(v, &bad));
        CHECK(bad >= 1);
        RC(elpis_ecs_c_rel_rebuild(v, NULL), ELPIS_ECS_C_REL_CORRUPT);
        OK(ecs_c_rel_testing_damage(v, &k, 6, 1)); /* undone */
        same(v, &sound);
        if (suspended) {
            w_lifecycle(&w, a, ELPIS_ECS_C_ACTIVE);
            OK(w_sync(v, &w, NULL));
        }
    }
    ans = q_view(v, NULL, &q);
    CHECK(answers_equal(&ans, &good));

    /* 3: the evidence itself damaged: scoped reads, admission and rebuild refuse (derived state is never rebuilt
     * from damaged evidence), and a sync that touches the relation never seals the damage into derived state. */
    uint8_t id_sound[32], id_damaged[32];
    OK(elpis_ecs_c_rel_identity(v, id_sound));
    OK(ecs_c_rel_testing_damage(v, &k, 3, 1000));
    OK(elpis_ecs_c_rel_identity(v, id_damaged));
    CHECK(memcmp(id_sound, id_damaged, 32)); /* the identity covers the evidence counts */
    RC(elpis_ecs_c_rel_query(v, NULL, w.rows[a].scope, &q, &ans, NULL), ELPIS_ECS_C_REL_CORRUPT);
    const elpis_ecs_c_rel_coordinate c3 = w_coord(&w);
    RC(elpis_ecs_c_rel_admit(v, &c3, w.rows[a].id, 1, w.rows[a].scope, &more, NULL, NULL), ELPIS_ECS_C_REL_CORRUPT);
    OK(elpis_ecs_c_rel_verify(v, &bad));
    CHECK(bad >= 1);
    const snapshot damaged = snap(v);
    RC(elpis_ecs_c_rel_rebuild(v, NULL), ELPIS_ECS_C_REL_CORRUPT);
    same_state(v, &damaged);
    /* The damaged contribution's own source flips: the flip does not reseal it, and the recomputed relation is
     * poisoned rather than sealed. */
    w_lifecycle(&w, a, ELPIS_ECS_C_DORMANT);
    elpis_ecs_c_rel_sync_result fr;
    OK(w_sync(v, &w, &fr));
    CHECK(fr.contributions_flipped == 1);
    RC(elpis_ecs_c_rel_query(v, NULL, NULL, &q, &ans, NULL), ELPIS_ECS_C_REL_CORRUPT);
    RC(elpis_ecs_c_rel_relation_read(v, NULL, NULL, &k, &slot), ELPIS_ECS_C_REL_CORRUPT);
    w_lifecycle(&w, a, ELPIS_ECS_C_ACTIVE);
    OK(w_sync(v, &w, NULL));
    RC(elpis_ecs_c_rel_query(v, NULL, NULL, &q, &ans, NULL), ELPIS_ECS_C_REL_CORRUPT);
    w_lifecycle(&w, b, ELPIS_ECS_C_DORMANT); /* another source's flip of the same relation */
    OK(w_sync(v, &w, NULL));
    RC(elpis_ecs_c_rel_query(v, NULL, NULL, &q, &ans, NULL), ELPIS_ECS_C_REL_CORRUPT);
    RC(elpis_ecs_c_rel_relation_read(v, NULL, NULL, &k, &slot), ELPIS_ECS_C_REL_CORRUPT);
    /* The remedy is explicit: reset and re-admit. */
    OK(elpis_ecs_c_rel_reset(v));
    OK(w_sync(v, &w, NULL));
    const elpis_ecs_mem_observation o3 = pair(5, ELPIS_ECS_MEM_ASSERT, 1, 7, 1, 2);
    OK(admit_as(v, &w, a, &o3, NULL));
    ans = q_view(v, NULL, &q);
    CHECK(ans.found && ans.score == 1);
    OK(elpis_ecs_c_rel_verify(v, &bad));
    CHECK(bad == 0);
    elpis_ecs_c_rel_destroy(v);
    printf("  damaged retained state: detected on read, by verify, repaired by rebuild; damaged evidence refused: PASS\n");
}

/* ---- 9. Locality: revocation work is independent of the unrelated state ---------------------------------- */

static void test_locality(void) {
    uint64_t first = 0;
    for (uint32_t scale = 0; scale < 3; ++scale) {
        const uint32_t unrelated = scale == 0 ? 100 : scale == 1 ? 1000 : 10000;
        elpis_ecs_c_rel *v = make_view(16, 16384, 16384);
        world w;
        w_init(&w, 10);
        const uint32_t a = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 7, 1, 1);
        const uint32_t c = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 9, 1, 1);
        OK(w_sync(v, &w, NULL));
        for (uint32_t i = 0; i < 8; ++i) {
            const elpis_ecs_mem_observation o = pair(5, ELPIS_ECS_MEM_ASSERT, 1, 7, i, 100 + i);
            OK(admit_as(v, &w, a, &o, NULL));
        }
        for (uint32_t i = 0; i < unrelated; ++i) {
            const elpis_ecs_mem_observation o = pair(9, ELPIS_ECS_MEM_ASSERT, 2, 9, 1000 + i, 50000 + i);
            OK(admit_as(v, &w, c, &o, NULL));
        }
        w_lifecycle(&w, a, ELPIS_ECS_C_DORMANT);
        elpis_ecs_c_rel_sync_result r;
        OK(w_sync(v, &w, &r));
        const uint64_t visited = r.work.contributions + r.work.relations + r.work.recomputes;
        CHECK(r.contributions_flipped == 8 && r.relations_recomputed == 8 && r.work.contributions == 16);
        if (!scale) first = visited;
        else CHECK(visited == first); /* 100, 1000, 10000 unrelated relations: the same work */
        elpis_ecs_c_rel_work rw;
        memset(&rw, 0, sizeof(rw));
        OK(elpis_ecs_c_rel_rebuild(v, &rw));
        CHECK(rw.contributions >= 2u * (unrelated + 8u)); /* the full recomputation visits everything */
        elpis_ecs_c_rel_destroy(v);
    }
    printf("  locality: one source's revocation costs the same over 100, 1000 and 10000 unrelated relations: PASS\n");
}

/* ---- 10. The differential: incremental view == exact rebuild by the real memory ------------------------- */

typedef struct {
    uint32_t type, arity, roles[4], ents[4];
} template_t;

static uint64_t compared_answers, compared_slots, compared_states;
static uint64_t found_answers, conflicted_answers, contradicted_answers, uncertain_answers, present_slots;

static void compare(elpis_ecs_c_rel *v, const world *w, const template_t *T, uint32_t nt, uint8_t scopes[2][32]) {
    for (int si = -1; si < 2; ++si) {
        const uint8_t *scope = si < 0 ? NULL : scopes[si];
        elpis_ecs_mem *m = oracle_build(w, scope);
        for (uint32_t t = 0; t < nt; ++t) {
            for (uint32_t target = 0; target < T[t].arity; ++target) {
                uint32_t cr[4], ce[4], n = 0;
                for (uint32_t i = 0; i < T[t].arity; ++i)
                    if (i != target) cr[n] = T[t].roles[i], ce[n++] = T[t].ents[i];
                for (uint32_t cues = 0; cues <= n; cues += (n ? n : 1)) {
                    for (uint32_t shape = 0; shape < 2; ++shape) {
                        const uint32_t k = shape ? (cues ? 1u : 0u) : cues;
                        const elpis_ecs_mem_request q = lookup(T[t].type, T[t].roles[target], k, cr, ce);
                        elpis_ecs_mem_answer want, got;
                        CHECK(elpis_ecs_mem_query(m, &q, &want) == ELPIS_ECS_MEM_OK);
                        OK(elpis_ecs_c_rel_query(v, NULL, scope, &q, &got, NULL));
                        if (!answers_equal(&want, &got)) {
                            fprintf(stderr, "answer mismatch: template %u target %u cues %u scope %d\n", t, target, k, si);
                            exit(1);
                        }
                        ++compared_answers;
                        found_answers += want.found != 0;
                        conflicted_answers += want.conflicts != 0;
                        contradicted_answers += want.contradicted != 0;
                        uncertain_answers += want.uncertain != 0;
                    }
                }
            }
            elpis_ecs_mem_observation o = obs(T[t].type, ELPIS_ECS_MEM_ASSERT, 0, 0, T[t].arity, T[t].roles, T[t].ents);
            elpis_ecs_mem_digest key;
            CHECK(elpis_ecs_mem_relation_key(&o, &key) == ELPIS_ECS_MEM_OK);
            elpis_ecs_mem_slot want, got;
            const int present = elpis_ecs_mem_slot_read(m, &key, &want) == ELPIS_ECS_MEM_OK;
            const int rc = elpis_ecs_c_rel_relation_read(v, NULL, scope, &key, &got);
            CHECK(present ? rc == ELPIS_ECS_C_REL_OK && slots_equal(&want, &got) : rc == ELPIS_ECS_C_REL_NOT_FOUND);
            ++compared_slots;
            present_slots += (uint64_t)present;
        }
        elpis_ecs_mem_stats ms;
        CHECK(elpis_ecs_mem_inspect(m, &ms) == ELPIS_ECS_MEM_OK);
        size_t n = 0;
        RC(elpis_ecs_c_rel_export(v, NULL, scope, NULL, 0, &n, NULL), ms.slots_used ? ELPIS_ECS_C_REL_CAPACITY
                                                                                     : ELPIS_ECS_C_REL_OK);
        CHECK(n == ms.slots_used);
        elpis_ecs_mem_destroy(m);
    }
    uint64_t bad = 1;
    OK(elpis_ecs_c_rel_verify(v, &bad));
    CHECK(bad == 0);
    ++compared_states;
}

static void test_differential(uint64_t seed, uint32_t steps) {
    rng_state = seed;
    log_count = 0;
    world w;
    w_init(&w, (uint32_t)(100 + seed));
    elpis_ecs_c_rel *v = make_view(W_MAX, 512, 2048);
    uint8_t scopes[2][32];
    fill(scopes[0], 0x73636f70u, 0);
    fill(scopes[1], 0x73636f70u, 1);
    uint32_t next_authority = 1;
    for (uint32_t i = 0; i < 6; ++i) (void)w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, next_authority++, 1, i % 2);
    OK(w_sync(v, &w, NULL));
    template_t T[24];
    for (uint32_t t = 0; t < 24; ++t) {
        T[t].type = 1 + t % 2;
        T[t].arity = 2 + t % 3;
        for (uint32_t i = 0; i < T[t].arity; ++i) T[t].roles[i] = i + 1, T[t].ents[i] = below(6);
    }
    uint64_t tick = 1, admitted = 0, refused = 0, transitions = 0;
    for (uint32_t step = 0; step < steps; ++step) {
        if (below(100) < 75) {
            /* Mostly an ACTIVE source (live evidence), sometimes any source (the lifecycle refusals). */
            uint32_t sources[W_MAX], ns = 0, active[W_MAX], na = 0;
            for (uint32_t i = 0; i < w.count; ++i) {
                if (w.rows[i].kind != ELPIS_ECS_C_SOURCE) continue;
                sources[ns++] = i;
                if (w.rows[i].lifecycle == ELPIS_ECS_C_ACTIVE) active[na++] = i;
            }
            const uint32_t row = na && below(10) < 8 ? active[below(na)] : sources[below(ns)];
            const elpis_ecs_c_entity *e = &w.rows[row];
            const template_t *t = &T[below(24)];
            const uint32_t p = below(20);
            const uint32_t polarity = p < 12 ? ELPIS_ECS_MEM_ASSERT : p < 17 ? ELPIS_ECS_MEM_NEGATE : ELPIS_ECS_MEM_UNCERTAIN;
            if (!below(3)) ++tick;
            const uint32_t variant = below(20);
            const uint64_t version = variant == 0 ? e->source_version + 1 : e->source_version;
            const uint32_t authority = variant == 1 ? e->source_authority + 1000 : e->source_authority;
            elpis_ecs_mem_observation o = obs(t->type, polarity, tick, authority, t->arity, t->roles, t->ents);
            for (uint32_t i = t->arity - 1; i > 0; --i) { /* any participant order: one relation */
                const uint32_t j = below(i + 1);
                const elpis_ecs_mem_participant x = o.participants[i];
                o.participants[i] = o.participants[j];
                o.participants[j] = x;
            }
            int expect = ELPIS_ECS_C_REL_OK;
            if (e->lifecycle != ELPIS_ECS_C_ACTIVE) expect = ELPIS_ECS_C_REL_LIFECYCLE;
            else if (variant == 0) expect = ELPIS_ECS_C_REL_SOURCE_STALE;
            else if (variant == 1) expect = ELPIS_ECS_C_REL_BINDING_MISMATCH;
            const elpis_ecs_c_rel_coordinate c = w_coord(&w);
            RC(elpis_ecs_c_rel_admit(v, &c, e->id, version, e->scope, &o, NULL, NULL), expect);
            if (expect == ELPIS_ECS_C_REL_OK) log_add(&o, e), ++admitted;
            else ++refused;
        } else {
            const uint64_t generation = w.generation;
            for (uint32_t folded = 1 + below(3); folded--;) {
                uint32_t sources[W_MAX], ns = 0;
                for (uint32_t i = 0; i < w.count; ++i)
                    if (w.rows[i].kind == ELPIS_ECS_C_SOURCE) sources[ns++] = i;
                const uint32_t row = sources[below(ns)];
                elpis_ecs_c_entity *e = &w.rows[row];
                const uint32_t kind = below(100);
                ++transitions;
                if (kind < 25) {
                    if (e->lifecycle != ELPIS_ECS_C_TERMINATED) w_version(&w, row, e->source_version + 1, below(2));
                } else if (kind < 50) {
                    if (e->lifecycle == ELPIS_ECS_C_ACTIVE) w_lifecycle(&w, row, ELPIS_ECS_C_DORMANT);
                    else if (e->lifecycle != ELPIS_ECS_C_TERMINATED) w_lifecycle(&w, row, ELPIS_ECS_C_ACTIVE);
                } else if (kind < 54) {
                    if (e->lifecycle != ELPIS_ECS_C_TERMINATED) w_lifecycle(&w, row, ELPIS_ECS_C_TERMINATED);
                } else if (kind < 66) {
                    if (e->lifecycle == ELPIS_ECS_C_TERMINATED && w.count < W_MAX - 1) {
                        const uint32_t reused = e->source_authority; /* a new source on the reclaimed identifier */
                        w_reclaim(&w, row);
                        (void)w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_FOUNDED, reused, 1, below(2));
                    }
                } else if (kind < 72) {
                    if (w.count < W_MAX - 1) (void)w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_FOUNDED, next_authority++, 1, below(2));
                } else if (kind < 82) {
                    if (e->lifecycle == ELPIS_ECS_C_FOUNDED || e->lifecycle == ELPIS_ECS_C_DORMANT)
                        w_lifecycle(&w, row, ELPIS_ECS_C_ACTIVE);
                } else {
                    w_bump(&w); /* a transition that changes no entity row (a publication) */
                }
            }
            elpis_ecs_c_rel_sync_result r;
            OK(w_sync(v, &w, &r));
            CHECK(r.noop == (w.generation == generation)); /* nothing transitioned: the identical table */
            compare(v, &w, T, 24, scopes);
        }
        if (step % 40 == 39) compare(v, &w, T, 24, scopes);
        if (step % 150 == 149) {
            uint8_t before[32], after[32];
            OK(elpis_ecs_c_rel_identity(v, before));
            OK(elpis_ecs_c_rel_rebuild(v, NULL)); /* the complete recomputation agrees with the incremental state */
            OK(elpis_ecs_c_rel_identity(v, after));
            CHECK(!memcmp(before, after, 32));
            compare(v, &w, T, 24, scopes);
        }
    }
    elpis_ecs_c_rel_status s;
    OK(elpis_ecs_c_rel_read_status(v, &s));
    if (!(s.admissions == admitted && s.refusals == refused && admitted > steps / 4 && refused > 0))
        fprintf(stderr, "seed %llu: admissions %llu/%llu refusals %llu/%llu\n", (unsigned long long)seed,
                (unsigned long long)s.admissions, (unsigned long long)admitted, (unsigned long long)s.refusals,
                (unsigned long long)refused);
    CHECK(s.admissions == admitted && s.refusals == refused && admitted > steps / 4 && refused > 0);
    CHECK(s.retired_contributions > 0 && s.flips > 0 && s.released_relations > 0);
    printf("    seed %llu: %u steps, %llu admitted, %llu refused, %llu authority transitions, %llu retired, "
           "%llu flips\n",
           (unsigned long long)seed, steps, (unsigned long long)admitted, (unsigned long long)refused,
           (unsigned long long)transitions, (unsigned long long)s.retired_contributions, (unsigned long long)s.flips);
    elpis_ecs_c_rel_destroy(v);
}

int main(void) {
    rng_state = 20261024;
    printf("ECS_C relational view (ECSC_DYNAMIC_RELATIONAL_DEV_R0)\n");
    test_configuration();
    test_table_validation();
    test_admission();
    test_source_revision();
    test_scopes_and_roles();
    test_shared_dependencies_and_contradiction();
    test_order_and_duplicates();
    test_resources();
    test_damage();
    test_locality();
    for (uint64_t seed = 1; seed <= 8; ++seed) test_differential(seed, 1200);
    /* Not vacuous: the compared answers name fillers, conflicts, contradictions and uncertainty, and slots exist. */
    printf("    non-vacuity: %llu found, %llu with conflicts, %llu with contradictions, %llu with uncertainty; "
           "%llu present slots\n",
           (unsigned long long)found_answers, (unsigned long long)conflicted_answers,
           (unsigned long long)contradicted_answers, (unsigned long long)uncertain_answers,
           (unsigned long long)present_slots);
    CHECK(found_answers * 5 > compared_answers && conflicted_answers && contradicted_answers && uncertain_answers);
    CHECK(present_slots * 4 > compared_slots && present_slots < compared_slots);
    printf("  differential vs the exact rebuild (a fresh ECS_C memory replaying the LIVE evidence): %llu answers, "
           "%llu slots, %llu states, 0 mismatches: PASS\n",
           (unsigned long long)compared_answers, (unsigned long long)compared_slots,
           (unsigned long long)compared_states);
    printf("PASS\n");
    return 0;
}
