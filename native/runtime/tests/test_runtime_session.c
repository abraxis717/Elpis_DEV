/* RuntimeCore ABI v8 over its volatile ECS_C session, the real ECS composite retained state and the ECS_C relational
 * view, through the production libraries, composed in memory only (docs/RUNTIME_CORE.md, "The session authority"):
 *
 *   a runtime opens a session (no path, no provisioning); the operator founds a source through
 *   elpis_runtime_authority_transition, built from the rows the runtime reads back; the lineage is a composite;
 *   typed observations are admitted against the session's sources before the native call (an unknown source, and a
 *   source past its per-turn bound, are refused RUNTIME_OBSERVATION_REFUSED with the turn kept and nothing crossing);
 *   operator events are refused while a turn is open and for RuntimeCore's own kinds; the persistence-only kinds are
 *   RUNTIME_RETIRED; a DORMANT source is refused; a source-version change moves the scope; reclamation; capacity
 *   exhaustion refuses before anything changes; the relational view, synced from the runtime's own session rows,
 *   admits evidence and answers LOOKUP only from admissible evidence, retires it on a source version change, and
 *   refuses the next session's table (LINEAGE) until reset; close ends the session and nothing of it remains.
 *
 * Usage: test_runtime_session. Deterministic: no sleeps, no files. */
#define _POSIX_C_SOURCE 200809L
#include "elpis/ecs_c_relational.h"
#include "elpis/ecs_composite.h"
#include "elpis/ecsg_k1.h"
#include "elpis/runtime.h"
#include "k1_tables.h"
#include "session_runtime.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OK(x) assert((x) == ELPIS_RUNTIME_OK)
#define ROK(x) assert((x) == ELPIS_ECS_C_REL_OK)
enum { DIM = 6, WIDTH = 36, BOUND = 2 };
enum { SOURCE = 4242, UNKNOWN = 77 };
_Static_assert(sizeof(elpis_runtime_authority) == 240, "elpis_runtime_authority ABI size");
_Static_assert(sizeof(elpis_runtime_authority_entity) == sizeof(elpis_ecs_c_entity), "entity row layout");

/* ---- the composite's function table (typed wrappers, as test_runtime_composite.c) ------------------------- */

static int c_digest(void *s, uint8_t out[32]) { return elpis_ecs_composite_state_digest(s, out); }
static int c_shape(void *s, size_t *dim, size_t *width) { return elpis_ecs_composite_shape(s, dim, width); }
static int c_claim(void *s, uint64_t lease) { return elpis_ecs_composite_lease_claim(s, lease); }
static int c_release(void *s, uint64_t lease) { return elpis_ecs_composite_lease_release(s, lease); }
static int c_reserve(void *s, uint64_t lease, size_t rows) { return elpis_ecs_composite_leased_reserve(s, lease, rows); }
static int c_identity(void *s, uint64_t lease, uint8_t digest[32], uint64_t *generation) {
    return elpis_ecs_composite_leased_identity(s, lease, digest, generation);
}
static int c_query(void *s, uint64_t lease, size_t dim, const double *x, size_t rows, double *out, uint64_t *g) {
    return elpis_ecs_composite_leased_query(s, lease, dim, x, rows, out, g);
}
static int c_begin(void *s, uint64_t lease, uint64_t *source, uint64_t *token) {
    return elpis_ecs_composite_leased_txn_begin_at(s, lease, source, token);
}
static int c_schedule(void *s, uint64_t lease, uint64_t token, const double *x, const double *y, size_t rows,
                      const elpis_runtime_experience *schedule, size_t n, double rate, double *s3, size_t s3n,
                      elpis_runtime_schedule_result *result) {
    return elpis_ecs_composite_leased_txn_run_schedule(s, lease, token, x, y, rows,
                                                       (const elpis_ecsg_k1_experience *)schedule, n, rate, s3, s3n,
                                                       (elpis_ecsg_k1_schedule_result *)result);
}
static int c_commit(void *s, uint64_t lease, uint64_t token, elpis_runtime_transition *t) {
    return elpis_ecs_composite_leased_txn_commit_at(s, lease, token, (elpis_ecsg_k1_transition *)t);
}
static int c_abort(void *s, uint64_t token) { return elpis_ecs_composite_txn_abort(s, token); }
static int c_observe(void *s, uint64_t lease, uint64_t token, const void *observation, uint8_t key[32], int *ms) {
    return elpis_ecs_composite_leased_txn_observe(s, lease, token, observation, (elpis_ecs_mem_digest *)key, ms);
}
static const elpis_runtime_composite_api COMPOSITE_API = {
    {c_digest, c_shape, c_claim, c_release, c_reserve, c_identity, c_query, c_begin, c_schedule, c_commit, c_abort},
    c_observe};

/* ---- fixtures ------------------------------------------------------------------------------------------------ */

static elpis_runtime_budget BUDGET;
static double W0[DIM * WIDTH];

static elpis_ecs_composite_config config(void) {
    elpis_ecs_composite_config c;
    memset(&c, 0, sizeof(c));
    c.abi_version = ELPIS_ECS_COMPOSITE_ABI_V1;
    c.dim = DIM;
    c.width = WIDTH;
    c.max_rows = 8;
    elpis_ecs_mem_config_default(&c.memory);
    c.memory.dim = 64;
    c.memory.banks = 2;
    c.memory.slot_capacity = 16;
    c.memory.codebook_capacity = 32;
    return c;
}
static elpis_runtime_substrate csub(elpis_ecs_composite *x, uint64_t owner) {
    elpis_runtime_substrate d = {ELPIS_RUNTIME_SUBSTRATE_COMPOSITE, 0, x, 0, owner, DIM, &COMPOSITE_API};
    return d;
}
static elpis_ecs_mem_observation obs(uint32_t source, uint64_t tick, uint64_t subject) {
    elpis_ecs_mem_observation o;
    memset(&o, 0, sizeof(o));
    o.abi_version = ELPIS_ECS_MEM_ABI_V1;
    o.relation_type = 3;
    o.polarity = ELPIS_ECS_MEM_ASSERT;
    o.participant_count = 2;
    o.tick = tick;
    o.source_authority = source;
    o.participants[0].role = 1;
    o.participants[1].role = 2;
    for (int i = 0; i < 32; ++i) {
        o.participants[0].entity.b[i] = (uint8_t)(subject * 31 + (uint64_t)i);
        o.participants[1].entity.b[i] = (uint8_t)(7 * 17 + (uint64_t)i * 3);
        o.provenance.b[i] = (uint8_t)(tick + (uint64_t)i);
    }
    return o;
}
static elpis_runtime_authority status_of(elpis_runtime *rt) {
    elpis_runtime_authority s;
    OK(elpis_runtime_authority_status(rt, &s));
    return s;
}
/* One observe-only turn of `n` observations; the code of the first refused observation (0: all admitted). */
static int observe_turn(elpis_runtime *rt, const elpis_runtime_substrate *d, const elpis_ecs_mem_observation *o, int n,
                        int commit) {
    elpis_runtime_observe_begin_result b;
    OK(elpis_runtime_turn_begin_observe(rt, d, &BUDGET, &b));
    int refused = 0;
    for (int i = 0; i < n; ++i) {
        elpis_runtime_observe_result r;
        const int rc = elpis_runtime_turn_observe(rt, d, &o[i], &r);
        if (rc && !refused) refused = rc;
    }
    if (commit && !refused) {
        elpis_runtime_volatile_commit vc;
        OK(elpis_runtime_turn_commit(rt, d, &vc));
    } else {
        OK(elpis_runtime_turn_abort(rt, d));
    }
    return refused;
}

/* The runtime's session as the relational view consumes it: its status and every row (read back through RuntimeCore). */
static elpis_ecs_c_entity ROWS[ELPIS_ECS_C_MAX_CAPACITY];
static elpis_ecs_c_rel_authority table_of(elpis_runtime *rt, elpis_ecs_c_rel_coordinate *coord) {
    const elpis_runtime_authority st = status_of(rt);
    for (uint32_t i = 0; i < st.entity_count; ++i)
        OK(elpis_runtime_authority_entity_at(rt, i, (elpis_runtime_authority_entity *)&ROWS[i]));
    elpis_ecs_c_rel_authority t;
    memset(&t, 0, sizeof(t));
    t.abi_version = ELPIS_ECS_C_REL_ABI_VERSION;
    t.entity_count = st.entity_count;
    t.generation = st.generation;
    t.next_founding_index = st.next_founding_index;
    t.reclaimed_count = st.reclaimed_count;
    memcpy(t.lineage, st.session, 32);
    memcpy(t.state_root, st.state_root, 32);
    t.entities = ROWS;
    if (coord) {
        coord->generation = st.generation;
        memcpy(coord->state_root, st.state_root, 32);
    }
    return t;
}
static int sync_to(elpis_ecs_c_rel *v, elpis_runtime *rt) {
    const elpis_ecs_c_rel_authority t = table_of(rt, NULL);
    return elpis_ecs_c_rel_sync(v, &t, NULL);
}

int main(void) {
    for (int i = 0; i < DIM * WIDTH; ++i) W0[i] = 0.2 * (double)((i * 7) % 13 - 6) / 13.0;
    OK(elpis_runtime_fuel_ceiling(&BUDGET));
    const elpis_ecs_composite_config cfg = config();

    /* ---- a session, its operator's source, a composite lineage ------------------------------------------------ */
    elpis_runtime *rt = NULL;
    elpis_continuity_snapshot snap;
    OK(test_runtime_create(&rt));
    OK(elpis_runtime_open(rt, &snap)); /* no test sources: this test founds its own */
    elpis_runtime_authority s0 = status_of(rt);
    assert(s0.entity_count == 2 && s0.generation == 1 && s0.backend == ELPIS_RUNTIME_BACKEND_ECS_C_SESSION);
    uint8_t source[32];
    test_runtime_found_source(rt, SOURCE, BOUND, source);
    elpis_runtime_authority_entity row;
    OK(elpis_runtime_authority_entity_at(rt, 2, &row));
    assert(!memcmp(row.id, source, 32) && row.lifecycle == ELPIS_ECS_C_ACTIVE && row.source_authority == SOURCE);
    assert(row.per_turn_limit == BOUND && row.source_version == 1);
    assert(elpis_runtime_authority_entity_at(rt, 3, &row) == ELPIS_RUNTIME_INVALID);

    elpis_ecs_composite *x = NULL;
    assert(elpis_ecs_composite_create(&cfg, W0, &x) == ELPIS_ECS_COMPOSITE_OK);
    elpis_runtime_substrate d = csub(x, 1);
    OK(elpis_runtime_anchor(rt, &d, &snap));
    assert(snap.state_format == ELPIS_CONTINUITY_FORMAT_COMPOSITE);

    /* ---- admission against the session's sources ----------------------------------------------------------- */
    elpis_ecs_mem_observation two[2] = {obs(SOURCE, 10, 1), obs(SOURCE, 11, 2)};
    assert(observe_turn(rt, &d, two, 2, 1) == 0);
    elpis_runtime_counters c;
    OK(elpis_runtime_read_counters(rt, &c, 1));
    elpis_ecs_mem_observation unknown = obs(UNKNOWN, 12, 3);
    assert(observe_turn(rt, &d, &unknown, 1, 1) == ELPIS_RUNTIME_OBSERVATION_REFUSED);
    elpis_ecs_mem_observation three[3] = {obs(SOURCE, 13, 4), obs(SOURCE, 14, 5), obs(SOURCE, 15, 6)};
    assert(observe_turn(rt, &d, three, 3, 1) == ELPIS_RUNTIME_OBSERVATION_REFUSED); /* past the bound of 2 */
    OK(elpis_runtime_read_counters(rt, &c, 0));
    assert(c.k1_observes == 2); /* the two admitted of three; the refused ones never crossed */

    /* ---- operator events: not while a turn is open; not RuntimeCore's own kinds; the retired kinds ------------ */
    {
        elpis_runtime_observe_begin_result b;
        OK(elpis_runtime_turn_begin_observe(rt, &d, &BUDGET, &b));
        uint8_t e[512];
        memset(e, 0, sizeof e);
        e[80] = ELPIS_ECS_C_ACTIVE;
        e[81] = ELPIS_ECS_C_DORMANT;
        assert(test_runtime_operator(rt, 3, source, e) == ELPIS_RUNTIME_TURN_OPEN);
        OK(elpis_runtime_turn_abort(rt, &d));
        const elpis_runtime_authority before = status_of(rt);
        for (uint16_t kind = 5; kind <= 9; ++kind) { /* anchor, publish, import, evolution reserve, finalize */
            if (kind == 7) continue;                 /* 7 is IMPORT_COMPOSITE: still RuntimeCore's own */
            memset(e, 0, sizeof e);
            assert(test_runtime_operator(rt, kind, source, e) == ELPIS_RUNTIME_INVALID);
        }
        static const uint8_t none[32] = {0};
        for (uint16_t kind = 10; kind <= 13; ++kind) { /* legacy import, the three reconciliations */
            memset(e, 0, sizeof e); /* not entity-scoped: a receiver makes the event malformed */
            assert(test_runtime_operator(rt, kind, source, e) == ELPIS_RUNTIME_INVALID);
            memset(e, 0, sizeof e);
            assert(test_runtime_operator(rt, kind, none, e) == ELPIS_RUNTIME_RETIRED);
        }
        const elpis_runtime_authority after = status_of(rt);
        assert(after.generation == before.generation && !memcmp(after.state_root, before.state_root, 32));
        assert(after.ecs_c_code == ELPIS_ECS_C_PERSISTENCE_RETIRED && elpis_runtime_fault(rt) == 0);
    }

    /* ---- the relational view, synced from the runtime's own session rows ------------------------------------ */
    elpis_ecs_c_rel_config rcfg;
    elpis_ecs_c_rel_config_default(&rcfg);
    elpis_ecs_c_rel *view = NULL;
    ROK(elpis_ecs_c_rel_create(&rcfg, &view));
    ROK(sync_to(view, rt));
    {
        elpis_ecs_c_rel_coordinate at;
        (void)table_of(rt, &at);
        uint8_t scope[32];
        memset(scope, 4, 32);
        ROK(elpis_ecs_c_rel_check(view, source, 1, scope));
        elpis_ecs_mem_observation ev = obs(SOURCE, 20, 9);
        elpis_ecs_mem_digest key;
        ROK(elpis_ecs_c_rel_admit(view, &at, source, 1, scope, &ev, &key, NULL));
        /* LOOKUP answers from admissible evidence. */
        elpis_ecs_mem_request q;
        memset(&q, 0, sizeof(q));
        q.abi_version = ELPIS_ECS_MEM_ABI_V1;
        q.channel = ELPIS_ECS_MEM_LOOKUP;
        q.relation_type = 3;
        q.target_role = 2;
        q.cue_count = 1;
        q.cue[0] = ev.participants[0];
        elpis_ecs_mem_answer ans;
        ROK(elpis_ecs_c_rel_query(view, &at, NULL, &q, &ans, NULL));
        assert(ans.found && ans.matches == 1);
        /* The operator moves the source to version 2 (a new scope) through the runtime: after the next sync, the
         * version-1 evidence is no longer admissible and LOOKUP answers nothing from it. */
        uint8_t e[512];
        memset(e, 0, sizeof e);
        test_runtime_put_u64(e + 80, 1);
        test_runtime_put_u64(e + 88, 2);
        test_runtime_put_u32(e + 96, BOUND);
        memset(e + 104, 5, 32);
        OK(test_runtime_operator(rt, 4, source, e));
        assert(elpis_ecs_c_rel_query(view, &at, NULL, &q, &ans, NULL) == ELPIS_ECS_C_REL_OK); /* the old coordinate */
        ROK(sync_to(view, rt));
        elpis_ecs_c_rel_coordinate now;
        (void)table_of(rt, &now);
        assert(elpis_ecs_c_rel_query(view, &at, NULL, &q, &ans, NULL) == ELPIS_ECS_C_REL_STALE_VIEW);
        ROK(elpis_ecs_c_rel_query(view, &now, NULL, &q, &ans, NULL));
        assert(!ans.found);
        assert(elpis_ecs_c_rel_check(view, source, 1, scope) == ELPIS_ECS_C_REL_SOURCE_STALE);
        uint8_t scope2[32];
        memset(scope2, 5, 32);
        ROK(elpis_ecs_c_rel_check(view, source, 2, scope2));
    }

    /* ---- DORMANT: refused until ACTIVE again ------------------------------------------------------------------ */
    {
        uint8_t e[512];
        memset(e, 0, sizeof e);
        e[80] = ELPIS_ECS_C_ACTIVE;
        e[81] = ELPIS_ECS_C_DORMANT;
        OK(test_runtime_operator(rt, 3, source, e));
        elpis_ecs_mem_observation o = obs(SOURCE, 30, 1);
        assert(observe_turn(rt, &d, &o, 1, 1) == ELPIS_RUNTIME_OBSERVATION_REFUSED);
        memset(e, 0, sizeof e);
        e[80] = ELPIS_ECS_C_DORMANT;
        e[81] = ELPIS_ECS_C_ACTIVE;
        OK(test_runtime_operator(rt, 3, source, e));
        assert(observe_turn(rt, &d, &o, 1, 1) == 0);
    }

    /* ---- capacity: founding past the session's capacity is refused before anything changes ------------------- */
    {
        uint32_t founded = 0;
        for (uint32_t s = 1; s < 64; ++s) {
            const elpis_runtime_authority st = status_of(rt);
            if (st.entity_count == st.capacity) {
                uint8_t label[32], id[32], e[512];
                memset(label, 0x77, 32);
                OK(elpis_runtime_authority_next_entity_id(rt, ELPIS_ECS_C_SOURCE, label, id));
                memset(e, 0, sizeof e);
                e[80] = ELPIS_ECS_C_SOURCE;
                test_runtime_put_u32(e + 84, 1);
                test_runtime_put_u64(e + 88, 1);
                memcpy(e + 96, label, 32);
                memset(e + 128, 4, 32);
                test_runtime_put_u32(e + 160, 999);
                assert(test_runtime_operator(rt, 2, id, e) == ELPIS_RUNTIME_AUTHORITY);
                const elpis_runtime_authority after = status_of(rt);
                assert(after.ecs_c_code == ELPIS_ECS_C_CAPACITY && after.generation == st.generation);
                assert(after.entity_count == st.capacity && !memcmp(after.state_root, st.state_root, 32));
                break;
            }
            uint8_t id[32];
            test_runtime_found_source(rt, 1000 + s, 1, id);
            ++founded;
        }
        assert(founded == TEST_RUNTIME_CAPACITY - 3);
        assert(elpis_runtime_fault(rt) == 0);
    }

    /* ---- close ends the session: the next one has none of its sources, and the view refuses its table ---------- */
    const elpis_runtime_authority last = status_of(rt);
    elpis_runtime_close(rt);
    OK(elpis_runtime_open(rt, &snap));
    const elpis_runtime_authority next = status_of(rt);
    assert(next.entity_count == 2 && next.generation == 1 && memcmp(next.session, last.session, 32));
    assert(sync_to(view, rt) == ELPIS_ECS_C_REL_LINEAGE); /* no view outlives its session */
    ROK(elpis_ecs_c_rel_reset(view));
    ROK(sync_to(view, rt));
    OK(elpis_runtime_anchor(rt, &d, &snap)); /* the caller's composite, anchored afresh */
    elpis_ecs_mem_observation o = obs(SOURCE, 40, 1);
    assert(observe_turn(rt, &d, &o, 1, 1) == ELPIS_RUNTIME_OBSERVATION_REFUSED); /* the source is gone */
    elpis_ecs_c_rel_destroy(view);
    elpis_runtime_destroy(&rt);
    elpis_ecs_composite_destroy(x);
    puts("runtime session (ABI v8, volatile, composed): PASS");
    return 0;
}
