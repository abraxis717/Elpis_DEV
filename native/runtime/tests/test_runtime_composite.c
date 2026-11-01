/* RuntimeCore ABI v8 over the ECS composite retained state (docs/ECS_COMPOSITE_STATE.md, "Authority"), through the
 * production runtime library and the real libelpis_ecs_composite, in volatile memory only:
 *
 *   the composite anchor (the session records the composite format); managed turns that learn K1 and OBSERVE the
 *   memory, committed natively as volatile transitions (the session's anchored identity does not move); recoverable
 *   observe refusals; the composite refuses every unmanaged mutation while bound; a K1 state never stands for the
 *   composite lineage; an in-memory rollback (a snapshot buffer restored to the anchored state and rebound, the
 *   volatile work abandoned); the typed HACF path: a structure hyperedge asserted by one source and negated by another
 *   reaches the committed memory through a managed turn with both provenances, both polarities and the exact
 *   structural identity preserved; and (ABI v5) the observe-only managed turn: evidence recorded with no K1 schedule
 *   (the K1 envelope byte-identical), its refusals and observation bound. No path, file or directory is involved.
 *
 * Usage: test_runtime_composite. Deterministic: no threads, no sleeps. */
#define _POSIX_C_SOURCE 200809L
#include "elpis/ecs_composite.h"
#include "elpis/ecs_observation_hacf.h"
#include "elpis/ecsg_k1.h"
#include "elpis/runtime.h"
#include "k1_tables.h"
#include "session_runtime.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OK(x) assert((x) == ELPIS_RUNTIME_OK)
enum { DIM = 6, WIDTH = 36, ROWS = 4, F = 83 };
_Static_assert(sizeof(elpis_runtime_observe_result) == 40, "observe result ABI size");

/* ---- the composite's function table (typed wrappers, no function-pointer casts) --------------------------- */

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
static double W0[DIM * WIDTH], X[ROWS * DIM], Y[ROWS];
static const elpis_runtime_experience SCHEDULE[2] = {{2, 3}, {2, 5}};

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
    return c; /* no unqualified capability */
}
static elpis_ecs_composite *fresh(void) {
    const elpis_ecs_composite_config c = config();
    elpis_ecs_composite *x = NULL;
    assert(elpis_ecs_composite_create(&c, W0, &x) == ELPIS_ECS_COMPOSITE_OK);
    return x;
}
static elpis_runtime_substrate csub(elpis_ecs_composite *x, uint64_t owner) {
    elpis_runtime_substrate d = {ELPIS_RUNTIME_SUBSTRATE_COMPOSITE, 0, x, 0, owner, DIM, &COMPOSITE_API};
    return d;
}
static elpis_runtime_substrate ksub(elpis_ecsg_k1 *s, uint64_t owner) {
    elpis_runtime_substrate d = {ELPIS_RUNTIME_SUBSTRATE_K1, 0, s, 0, owner, DIM, &K1_API};
    return d;
}
static elpis_ecs_mem_observation obs(uint64_t subject, uint64_t object, uint64_t tick) {
    elpis_ecs_mem_observation o;
    memset(&o, 0, sizeof(o));
    o.abi_version = ELPIS_ECS_MEM_ABI_V1;
    o.relation_type = 3;
    o.polarity = ELPIS_ECS_MEM_ASSERT;
    o.participant_count = 2;
    o.tick = tick;
    o.participants[0].role = 1;
    o.participants[1].role = 2;
    for (int i = 0; i < 32; ++i) {
        o.participants[0].entity.b[i] = (uint8_t)(subject * 31 + (uint64_t)i);
        o.participants[1].entity.b[i] = (uint8_t)(object * 17 + (uint64_t)i * 3);
        o.provenance.b[i] = (uint8_t)(tick + (uint64_t)i);
    }
    return o;
}
static void identity(elpis_ecs_composite *x, uint8_t out[32]) {
    assert(elpis_ecs_composite_state_digest(x, out) == ELPIS_ECS_COMPOSITE_OK);
}
static uint32_t asserts_of(elpis_ecs_composite *x, const uint8_t key[32]) {
    elpis_ecs_mem_slot slot;
    int ms = 0;
    const int s = elpis_ecs_composite_memory_slot_read(x, (const elpis_ecs_mem_digest *)key, &slot, &ms);
    if (s != ELPIS_ECS_COMPOSITE_OK) {
        assert(ms == ELPIS_ECS_MEM_NOT_FOUND);
        return 0;
    }
    return slot.assert_count;
}

/* begin (K1 schedule) -> observe each -> commit. Returns the commit status; keys of the observations. */
static int turn(elpis_runtime *rt, const elpis_runtime_substrate *d, const elpis_ecs_mem_observation *o, int n,
                uint8_t keys[][32], elpis_runtime_volatile_commit *out) {
    double s3[F];
    elpis_runtime_turn_begin_result begun;
    int rc = elpis_runtime_turn_begin(rt, d, X, ROWS * DIM, Y, ROWS, SCHEDULE, 2, 0.002, &BUDGET, s3, F, &begun);
    if (rc != ELPIS_RUNTIME_OK) return rc;
    for (int i = 0; i < n; ++i) {
        elpis_runtime_observe_result r;
        OK(elpis_runtime_turn_observe(rt, d, &o[i], &r));
        assert(r.k1_status == 0 && r.memory_status == 0);
        if (keys) memcpy(keys[i], r.relation_key, 32);
    }
    return elpis_runtime_turn_commit(rt, d, out);
}

/* HACF hyperedge -> typed observation -> managed turn -> committed composite memory, provenance preserved. */
static void hacf_turn(elpis_runtime *rt, const elpis_runtime_substrate *d, elpis_ecs_composite *x, uint64_t tick) {
    elpis_semantic_hyperedge_v1 e;
    memset(&e, 0, sizeof(e));
    e.abi_version = SEMANTIC_ABI_VERSION;
    e.hyperedge_type = SEMANTIC_HYPEREDGE_NAMESPACE | 1u;
    e.participant_count = 3;
    for (uint32_t i = 0; i < 3; ++i) {
        e.participants[i].incidence_role = SEMANTIC_INCIDENCE_NAMESPACE | (3u - i);
        for (int k = 0; k < 32; ++k) e.participants[i].node_identity.bytes[k] = (uint8_t)(i * 41 + (uint32_t)k);
    }
    assert(elpis_semantic_canonicalize_participants(e.participants, 3) == SEMANTIC_OK);
    assert(elpis_semantic_hyperedge_identity(&e, &e.hyperedge_identity) == SEMANTIC_OK);
    elpis_semantic_assertion_v1 a[2];
    elpis_ecs_mem_observation o[2];
    for (int s = 0; s < 2; ++s) {
        memset(&a[s], 0, sizeof(a[s]));
        a[s].abi_version = SEMANTIC_ABI_VERSION;
        a[s].asserted_object_kind = SEMANTIC_OBJECT_KIND_HYPEREDGE;
        a[s].asserted_object_digest = e.hyperedge_identity;
        memset(a[s].provenance_digest.bytes, 0xA0 + s, 32);
        a[s].authority = (uint32_t)(1 + s);
        assert(elpis_semantic_assertion_identity(&a[s], &a[s].assertion_identity) == SEMANTIC_OK);
        /* Polarity is supplied by the caller (HACF does not define it): source 0 asserts, source 1 negates. */
        assert(elpis_ecs_obs_hacf_observation(&e, &a[s], s ? ELPIS_ECS_MEM_NEGATE : ELPIS_ECS_MEM_ASSERT, tick + (uint64_t)s,
                                              &o[s]) == ELPIS_ECS_OBS_HACF_OK);
    }
    uint8_t keys[2][32];
    elpis_runtime_volatile_commit committed;
    OK(turn(rt, d, o, 2, keys, &committed));
    assert(!memcmp(keys[0], keys[1], 32)); /* one relation, two observations */
    elpis_ecs_mem_slot slot;
    int ms = 0;
    assert(elpis_ecs_composite_memory_slot_read(x, (const elpis_ecs_mem_digest *)keys[0], &slot, &ms) == 0);
    assert(slot.assert_count == 1 && slot.negate_count == 1 && slot.last_polarity == ELPIS_ECS_MEM_NEGATE);
    assert(!memcmp(slot.first_provenance.b, a[0].provenance_digest.bytes, 32));
    assert(!memcmp(slot.last_provenance.b, a[1].provenance_digest.bytes, 32));
    assert(slot.last_authority == 2 && !memcmp(slot.structure_ref.b, e.hyperedge_identity.bytes, 32));
    elpis_semantic_hyperedge_v1 back;
    assert(elpis_ecs_obs_hacf_hyperedge(&slot, &back) == ELPIS_ECS_OBS_HACF_OK);
    assert(!memcmp(&back, &e, sizeof(e))); /* the exact structure, identity recomputed by structure */
    /* LOOKUP reports the conflict instead of an answer: observation is not learning or truth. */
    elpis_ecs_mem_request q;
    memset(&q, 0, sizeof(q));
    q.abi_version = ELPIS_ECS_MEM_ABI_V1;
    q.channel = ELPIS_ECS_MEM_LOOKUP;
    q.relation_type = e.hyperedge_type;
    q.target_role = slot.participants[2].role;
    q.cue_count = 2;
    q.cue[0] = slot.participants[0];
    q.cue[1] = slot.participants[1];
    elpis_ecs_mem_answer ans;
    uint8_t at[32];
    assert(elpis_ecs_composite_memory_query(x, &q, &ans, at, NULL) == 0);
    assert(ans.matches == 1 && !ans.found);
}

/* ---- ABI v5: an observe-only managed turn records typed evidence with no K1 learning ------------------------ */
static void observe_only_phase(void) {
    const elpis_ecs_composite_config cfg = config();
    const size_t envelope = elpis_ecs_composite_envelope_bytes(&cfg);
    const size_t k1n = elpis_ecsg_k1_envelope_bytes(DIM, WIDTH);
    elpis_runtime *rt = NULL;
    elpis_continuity_snapshot snap;
    OK(test_runtime_create(&rt));
    OK(test_runtime_open(rt, &snap));
    elpis_ecs_composite *x = fresh();
    elpis_runtime_substrate d = csub(x, 7);
    OK(elpis_runtime_anchor(rt, &d, &snap));
    uint8_t *before = malloc(envelope), *after = malloc(envelope);
    assert(before && after);
    assert(elpis_ecs_composite_snapshot_write(x, before, envelope) == ELPIS_ECS_COMPOSITE_OK);
    OK(elpis_runtime_read_counters(rt, &(elpis_runtime_counters){0}, 1));

    /* A K1 substrate is refused before any native call. */
    {
        elpis_ecsg_k1 *k = NULL;
        assert(elpis_ecsg_k1_create(DIM, WIDTH, 8, W0, &k) == ELPIS_ECSG_K1_OK);
        elpis_runtime_substrate dk = ksub(k, 8);
        elpis_runtime_observe_begin_result b;
        assert(elpis_runtime_turn_begin_observe(rt, &dk, &BUDGET, &b) == ELPIS_RUNTIME_INVALID && b.live_generation == 0);
        assert(elpis_ecsg_k1_destroy(&k) == ELPIS_ECSG_K1_OK);
    }
    /* One observation, no schedule: K1's envelope byte-identical, the memory and the composite identity moved. */
    elpis_runtime_observe_begin_result b;
    OK(elpis_runtime_turn_begin_observe(rt, &d, &BUDGET, &b));
    elpis_runtime_volatile_commit vc;
    assert(elpis_runtime_turn_commit(rt, &d, &vc) == ELPIS_RUNTIME_INVALID); /* nothing observed: kept open */
    elpis_ecs_mem_observation o = obs(5, 6, 20);
    elpis_runtime_observe_result orr;
    OK(elpis_runtime_turn_observe(rt, &d, &o, &orr));
    uint8_t key[32];
    memcpy(key, orr.relation_key, 32);
    OK(elpis_runtime_turn_commit(rt, &d, &vc));
    assert(vc.committed == 1 && vc.transition.steps == 0 && vc.transition.epoch_before == vc.transition.epoch_after);
    assert(vc.transition.generation_before == b.live_generation && vc.live_generation == b.live_generation + 1);
    assert(elpis_ecs_composite_snapshot_write(x, after, envelope) == ELPIS_ECS_COMPOSITE_OK);
    assert(!memcmp(before + ELPIS_ECS_COMPOSITE_HEADER_BYTES, after + ELPIS_ECS_COMPOSITE_HEADER_BYTES, k1n));
    assert(!memcmp(before + 48, after + 48, 32));                 /* the K1 identity */
    assert(memcmp(before + 80, after + 80, 32) != 0);             /* the memory identity */
    assert(memcmp(before + envelope - 32, after + envelope - 32, 32) != 0);
    assert(asserts_of(x, key) == 1);
    elpis_runtime_counters c;
    OK(elpis_runtime_read_counters(rt, &c, 0));
    assert(c.k1_txn_begins == 1 && c.k1_run_schedules == 0 && c.k1_observes == 1 && c.k1_commits == 1);
    assert(c.session_transitions == 0 && c.k1_identities == 0); /* volatile: the session did not move */
    OK(elpis_runtime_snapshot(rt, &snap));
    assert(!memcmp(snap.k1_state_digest, before + envelope - 32, 32)); /* the session's identity: the anchor */

    /* The turn's observation bound is fuel, refused before the native call; the turn stays open. */
    {
        elpis_runtime_budget one = BUDGET;
        one.max_observations = 1;
        OK(elpis_runtime_turn_begin_observe(rt, &d, &one, &b));
        elpis_ecs_mem_observation o2 = obs(9, 10, 21), o3 = obs(11, 12, 22);
        OK(elpis_runtime_turn_observe(rt, &d, &o2, &orr));
        assert(elpis_runtime_turn_observe(rt, &d, &o3, &orr) == ELPIS_RUNTIME_FUEL);
        OK(elpis_runtime_turn_abort(rt, &d));
        assert(asserts_of(x, orr.relation_key) == 0);            /* aborted: nothing installed */
        one.max_observations = 0;
        assert(elpis_runtime_turn_begin_observe(rt, &d, &one, &b) == ELPIS_RUNTIME_FUEL);
    }
    /* A malformed observation is refused by the memory and changes nothing; the turn stays open. */
    {
        uint8_t id[32], id2[32];
        identity(x, id);
        OK(elpis_runtime_turn_begin_observe(rt, &d, &BUDGET, &b));
        elpis_ecs_mem_observation bad = obs(5, 6, 23);
        bad.reserved = 1;
        assert(elpis_runtime_turn_observe(rt, &d, &bad, &orr) == ELPIS_RUNTIME_ECS_REFUSED);
        assert(orr.memory_status == ELPIS_ECS_MEM_INVALID);
        assert(elpis_runtime_turn_commit(rt, &d, &vc) == ELPIS_RUNTIME_INVALID);
        OK(elpis_runtime_turn_abort(rt, &d));
        identity(x, id2);
        assert(!memcmp(id, id2, 32));
    }
    /* The session ends; a new session anchors the evolved composite (its current identity) and continues: the
     * evidence lives on in the caller's state object, never in the session or on any storage. */
    uint8_t evolved[32];
    identity(x, evolved);
    elpis_runtime_close(rt);
    OK(test_runtime_open(rt, &snap));
    assert(!snap.anchored);
    OK(elpis_runtime_anchor(rt, &d, &snap));
    assert(!memcmp(snap.k1_state_digest, evolved, 32) && snap.state_format == ELPIS_CONTINUITY_FORMAT_COMPOSITE);
    OK(elpis_runtime_turn_begin_observe(rt, &d, &BUDGET, &b));
    elpis_ecs_mem_observation o4 = obs(5, 6, 24);
    OK(elpis_runtime_turn_observe(rt, &d, &o4, &orr));
    OK(elpis_runtime_turn_commit(rt, &d, &vc));
    assert(asserts_of(x, key) == 2); /* the same relation, observed again */
    elpis_runtime_destroy(&rt);
    elpis_ecs_composite_destroy(x);
    free(before);
    free(after);
}

int main(void) {
    for (int i = 0; i < DIM * WIDTH; ++i) W0[i] = 0.2 * (double)((i * 7) % 13 - 6) / 13.0;
    for (int i = 0; i < ROWS * DIM; ++i) X[i] = (double)((i * 5) % 11 - 5) / 16.0;
    for (int i = 0; i < ROWS; ++i) Y[i] = (double)(i % 3 - 1) / 8.0;
    OK(elpis_runtime_fuel_ceiling(&BUDGET));
    const elpis_ecs_composite_config cfg = config();
    const size_t envelope = elpis_ecs_composite_envelope_bytes(&cfg);
    assert(envelope > elpis_ecsg_k1_envelope_bytes(DIM, WIDTH));

    /* ---- anchor, managed turns, refusals ------------------------------------------------------------------ */
    elpis_runtime *rt = NULL;
    elpis_continuity_snapshot snap;
    OK(test_runtime_create(&rt));
    OK(test_runtime_open(rt, &snap));
    elpis_ecs_composite *x = fresh();
    elpis_runtime_substrate d = csub(x, 1);
    uint8_t id[32];
    identity(x, id);
    /* A bounded in-memory snapshot of the anchored state (the caller's buffer: what an in-session rollback restores). */
    uint8_t *at_anchor = malloc(envelope);
    assert(at_anchor && elpis_ecs_composite_snapshot_write(x, at_anchor, envelope) == ELPIS_ECS_COMPOSITE_OK);
    OK(elpis_runtime_anchor(rt, &d, &snap));
    assert(snap.anchored && snap.state_format == ELPIS_CONTINUITY_FORMAT_COMPOSITE && !memcmp(snap.k1_state_digest, id, 32));

    /* While bound, every unmanaged composite mutation is refused. */
    uint8_t src[32];
    uint64_t tok = 0;
    assert(elpis_ecs_composite_leased_txn_begin(x, 0, src, &tok) == ELPIS_ECS_COMPOSITE_LEASED);
    /* Observe without a turn. */
    elpis_ecs_mem_observation o1[2] = {obs(1, 2, 10), obs(3, 4, 11)};
    elpis_runtime_observe_result orr;
    assert(elpis_runtime_turn_observe(rt, &d, &o1[0], &orr) == ELPIS_RUNTIME_TURN_NOT_OPEN);
    elpis_runtime_volatile_commit committed;
    uint8_t keys[2][32], anchored[32];
    memcpy(anchored, id, 32);
    OK(turn(rt, &d, o1, 2, keys, &committed));
    identity(x, id);
    assert(committed.committed == 1 && committed.transition.generation_after == committed.transition.generation_before + 1);
    OK(elpis_runtime_snapshot(rt, &snap));
    assert(!memcmp(snap.k1_state_digest, anchored, 32));   /* volatile: the session's anchored identity did not move */
    assert(asserts_of(x, keys[0]) == 1 && asserts_of(x, keys[1]) == 1);   /* the memory transition is live */
    /* QUERY answers from the live composite, at its live generation. */
    double answer[ROWS];
    elpis_runtime_query_result qr;
    OK(elpis_runtime_query(rt, &d, X, ROWS * DIM, answer, ROWS, &BUDGET, &qr));
    assert(qr.live_generation == committed.live_generation);
    {
        uint8_t live[32];
        uint64_t generation = 0;
        OK(elpis_runtime_live_identity(rt, &d, live, &generation));
        assert(!memcmp(live, id, 32) && generation == committed.live_generation);
    }
    /* A refused observation is recoverable: the turn stays open and commits the rest. */
    {
        double s3[F];
        elpis_runtime_turn_begin_result begun;
        OK(elpis_runtime_turn_begin(rt, &d, X, ROWS * DIM, Y, ROWS, SCHEDULE, 2, 0.002, &BUDGET, s3, F, &begun));
        elpis_ecs_mem_observation bad = obs(5, 6, 12);
        bad.reserved = 1;
        assert(elpis_runtime_turn_observe(rt, &d, &bad, &orr) == ELPIS_RUNTIME_ECS_REFUSED);
        assert(orr.k1_status == ELPIS_ECS_COMPOSITE_INVALID && orr.memory_status == ELPIS_ECS_MEM_INVALID);
        elpis_ecs_mem_observation stale = obs(5, 6, 1); /* older than the memory clock */
        assert(elpis_runtime_turn_observe(rt, &d, &stale, &orr) == ELPIS_RUNTIME_ECS_REFUSED);
        assert(orr.k1_status == ELPIS_ECS_COMPOSITE_REFUSED && orr.memory_status == ELPIS_ECS_MEM_CLOCK);
        assert(elpis_runtime_turn_observe(rt, &d, NULL, &orr) == ELPIS_RUNTIME_INVALID);
        elpis_ecs_mem_observation good = obs(5, 6, 12);
        OK(elpis_runtime_turn_observe(rt, &d, &good, &orr));
        memcpy(keys[0], orr.relation_key, 32);
        OK(elpis_runtime_turn_commit(rt, &d, &committed));
        assert(asserts_of(x, keys[0]) == 1);
    }
    /* An aborted turn changes nothing (its observation included). */
    {
        identity(x, id);
        double s3[F];
        elpis_runtime_turn_begin_result begun;
        OK(elpis_runtime_turn_begin(rt, &d, X, ROWS * DIM, Y, ROWS, SCHEDULE, 2, 0.002, &BUDGET, s3, F, &begun));
        elpis_ecs_mem_observation lost = obs(7, 8, 20);
        OK(elpis_runtime_turn_observe(rt, &d, &lost, &orr));
        OK(elpis_runtime_turn_abort(rt, &d));
        uint8_t after[32];
        identity(x, after);
        assert(!memcmp(id, after, 32) && asserts_of(x, orr.relation_key) == 0);
    }
    hacf_turn(rt, &d, x, 30);

    /* ---- a K1 state never stands for the composite lineage; an in-memory rollback to the anchor ------------- */
    OK(elpis_runtime_release(rt, &d));
    {
        elpis_ecsg_k1 *k = NULL;
        assert(elpis_ecsg_k1_create(DIM, WIDTH, 8, W0, &k) == ELPIS_ECSG_K1_OK);
        elpis_runtime_substrate dk = ksub(k, 2);
        /* The K1 bytes the composite was made from: another format, never the lineage. */
        double s3[F];
        elpis_runtime_turn_begin_result begun;
        elpis_runtime *other = NULL;
        OK(test_runtime_create(&other));
        OK(test_runtime_open(other, &snap));
        elpis_ecs_composite *xa = NULL;
        assert(elpis_ecs_composite_restore(&cfg, at_anchor, envelope, &xa) == ELPIS_ECS_COMPOSITE_OK);
        elpis_runtime_substrate da = csub(xa, 3);
        OK(elpis_runtime_anchor(other, &da, &snap));
        OK(elpis_runtime_release(other, &da));
        assert(elpis_runtime_turn_begin(other, &dk, X, ROWS * DIM, Y, ROWS, SCHEDULE, 2, 0.002, &BUDGET, s3, F,
                                        &begun) == ELPIS_CONTINUITY_STATE_MISMATCH);
        assert(elpis_runtime_fault(other) == ELPIS_CONTINUITY_STATE_MISMATCH);
        elpis_runtime_destroy(&other);
        elpis_ecs_composite_destroy(xa);
        assert(elpis_ecsg_k1_destroy(&k) == ELPIS_ECSG_K1_OK);
    }
    {
        /* Rollback: a state restored from the anchor's snapshot buffer is the session's anchored identity, so it binds
         * (abandoning the volatile work, which the released state still carries); the released state then mismatches. */
        elpis_ecs_composite *rolled = NULL;
        assert(elpis_ecs_composite_restore(&cfg, at_anchor, envelope, &rolled) == ELPIS_ECS_COMPOSITE_OK);
        uint8_t back[32];
        identity(rolled, back);
        assert(!memcmp(back, anchored, 32));
        elpis_runtime_substrate dr = csub(rolled, 4);
        OK(turn(rt, &dr, NULL, 0, NULL, &committed));
        assert(asserts_of(rolled, keys[0]) == 0); /* the abandoned work is not in the rolled-back state */
        OK(elpis_runtime_release(rt, &dr));
        double s3[F];
        elpis_runtime_turn_begin_result begun;
        assert(elpis_runtime_turn_begin(rt, &d, X, ROWS * DIM, Y, ROWS, SCHEDULE, 2, 0.002, &BUDGET, s3, F, &begun) ==
               ELPIS_CONTINUITY_STATE_MISMATCH);
        elpis_ecs_composite_destroy(rolled);
    }
    elpis_runtime_destroy(&rt);
    elpis_ecs_composite_destroy(x);
    free(at_anchor);
    observe_only_phase();
    puts("runtime composite (ABI v8, volatile): PASS");
    return 0;
}
