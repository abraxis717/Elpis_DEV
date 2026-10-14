/* ECS composite retained state (ELPISEC1 v1) mechanics and recovery contract (docs/ECS_COMPOSITE_STATE.md):
 * K1-only, memory-only and joint transitions; abort and interruption; refusal atomicity; exact restore and
 * deterministic replay; corruption, mismatched identities, old and unsupported formats; capacity; capability
 * admission; leases; concurrent readers against a committing writer. */
#include "elpis/ecs_composite.h"
#include "elpis/sha256.h"

#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { D = 6, N = 36, R = 32, WC = D * N, F = 83 };
static double W0[WC], X[R * D], Y[R];

static void fixture(void) {
    unsigned s = 12345u;
    for (size_t i = 0; i < WC; ++i) {
        s = s * 1103515245u + 12345u;
        W0[i] = 0.18 * (((double)(s >> 8) / 16777216.0) - 0.5) * 2.0;
    }
    for (size_t i = 0; i < R * D; ++i) {
        s = s * 1103515245u + 12345u;
        X[i] = ((double)(s >> 8) / 16777216.0) - 0.5;
    }
    for (size_t i = 0; i < R; ++i) Y[i] = 0.3 * X[i * D] + 0.2 * X[i * D + 1] * X[i * D + 2];
}

static elpis_ecs_composite_config config(uint32_t caps) {
    elpis_ecs_composite_config c;
    memset(&c, 0, sizeof(c));
    c.abi_version = ELPIS_ECS_COMPOSITE_ABI_V1;
    c.dim = D;
    c.width = N;
    c.max_rows = R;
    elpis_ecs_mem_config_default(&c.memory);
    c.memory.dim = 64;
    c.memory.banks = 2;
    c.memory.slot_capacity = 16;
    c.memory.codebook_capacity = 32;
    c.memory.capabilities = caps;
    return c;
}

static elpis_ecs_mem_digest dg(const char *tag, uint64_t i) {
    char buf[64];
    elpis_ecs_mem_digest d;
    const int n = snprintf(buf, sizeof(buf), "%s:%llu", tag, (unsigned long long)i);
    elpis_sha256(buf, (size_t)n, d.b);
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
    o.provenance = dg("provenance", tick);
    o.structure_ref = dg("structure", tick);
    o.participants[0].role = 1;
    o.participants[0].entity = dg("entity", subject);
    o.participants[1].role = 2;
    o.participants[1].entity = dg("entity", object);
    return o;
}

static elpis_ecs_composite *fresh(uint32_t caps) {
    const elpis_ecs_composite_config c = config(caps);
    elpis_ecs_composite *x = NULL;
    assert(elpis_ecs_composite_create(&c, W0, &x) == ELPIS_ECS_COMPOSITE_OK && x);
    return x;
}
static size_t envelope(elpis_ecs_composite *x, uint8_t **out) {
    elpis_ecs_composite_stats s;
    assert(elpis_ecs_composite_inspect(x, &s) == ELPIS_ECS_COMPOSITE_OK);
    *out = malloc(s.envelope_bytes);
    assert(*out && elpis_ecs_composite_snapshot_write(x, *out, s.envelope_bytes) == ELPIS_ECS_COMPOSITE_OK);
    return s.envelope_bytes;
}
static void identity(elpis_ecs_composite *x, uint8_t out[32]) {
    assert(elpis_ecs_composite_state_digest(x, out) == ELPIS_ECS_COMPOSITE_OK);
}
static void run_schedule(elpis_ecs_composite *x, uint64_t lease, uint64_t token, uint64_t steps) {
    const elpis_ecsg_k1_experience e[1] = {{R, steps}};
    double s3[F];
    elpis_ecsg_k1_schedule_result res;
    assert(elpis_ecs_composite_leased_txn_run_schedule(x, lease, token, X, Y, R, e, 1, 0.002, s3, F, &res) ==
           ELPIS_ECS_COMPOSITE_OK);
    assert(res.experiences_applied == 1);
}
static void reseal(uint8_t *e, size_t n) { elpis_sha256(e, n - 32, e + n - 32); }

/* ---------------------------------------------------------------------------------------------------------- */

static void test_layout_and_identity(void) {
    const elpis_ecs_composite_config c = config(0);
    const size_t k1 = elpis_ecsg_k1_envelope_bytes(D, N);
    uint64_t arena = 0, cap = 0;
    assert(elpis_ecs_mem_required_bytes(&c.memory, &arena, &cap) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_composite_envelope_bytes(&c) == 128 + k1 + cap + 32);
    elpis_ecs_composite *x = fresh(0);
    uint8_t *e = NULL, id[32], cid[32], kid[32], mid[32];
    const size_t n = envelope(x, &e);
    identity(x, id);
    assert(!memcmp(e, "ELPISEC1", 8) && e[8] == 1 && !memcmp(e + n - 32, id, 32));
    assert(elpis_ecs_composite_envelope_identities(e, n, cid, kid, mid) == ELPIS_ECS_COMPOSITE_OK);
    assert(!memcmp(cid, id, 32));
    /* The K1 component is the unchanged ELPISGK1 envelope of the same state. */
    elpis_ecsg_k1 *alone = NULL;
    assert(elpis_ecsg_k1_create(D, N, R, W0, &alone) == ELPIS_ECSG_K1_OK);
    uint8_t *ke = malloc(k1);
    assert(ke && elpis_ecsg_k1_snapshot_write(alone, ke, k1) == ELPIS_ECSG_K1_OK);
    assert(!memcmp(e + 128, ke, k1) && !memcmp(kid, ke + k1 - 32, 32));
    /* The composite identity is never a component identity. */
    assert(memcmp(id, kid, 32) && memcmp(id, mid, 32));
    /* The memory component is the unchanged ELPISEM1 envelope of an empty memory. */
    elpis_ecs_mem *m = NULL;
    assert(elpis_ecs_mem_create(&c.memory, &m) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_digest md;
    assert(elpis_ecs_mem_identity(m, &md) == ELPIS_ECS_MEM_OK && !memcmp(md.b, mid, 32));
    assert(!memcmp(e + 128 + k1, "ELPISEM1", 8));
    /* Query is bound to the composite identity. */
    double out[R];
    uint8_t qd[32];
    assert(elpis_ecs_composite_query_identity(x, D, X, R, out, qd) == ELPIS_ECS_COMPOSITE_OK && !memcmp(qd, id, 32));
    double ref[R];
    assert(elpis_ecsg_k1_forward(alone, X, R, ref) == ELPIS_ECSG_K1_OK && !memcmp(out, ref, sizeof(ref)));
    assert(elpis_ecs_composite_query_identity(x, D + 1, X, R, out, qd) == ELPIS_ECS_COMPOSITE_INVALID);
    elpis_ecs_composite_stats st;
    assert(elpis_ecs_composite_inspect(x, &st) == ELPIS_ECS_COMPOSITE_OK);
    assert(st.envelope_bytes == n && st.k1_envelope_bytes == k1 && st.memory_capacity_bytes == cap);
    assert(st.allocated_bytes >= st.k1_workspace_bytes + 2 * st.memory_arena_bytes + st.staging_bytes);
    assert(st.capabilities == 0 && !st.txn_open && !st.lease_held);
    elpis_ecs_mem_destroy(m);
    elpis_ecsg_k1_destroy(&alone);
    free(ke);
    free(e);
    elpis_ecs_composite_destroy(x);
}

/* K1-only, memory-only and joint transitions; each component equals the same law applied alone. */
static void test_transitions(void) {
    const elpis_ecs_composite_config c = config(0);
    const size_t k1n = elpis_ecsg_k1_envelope_bytes(D, N);
    elpis_ecs_composite *x = fresh(0);
    elpis_ecsg_k1 *k1 = NULL;
    elpis_ecs_mem *m = NULL;
    assert(elpis_ecsg_k1_create(D, N, R, W0, &k1) == ELPIS_ECSG_K1_OK);
    assert(elpis_ecs_mem_create(&c.memory, &m) == ELPIS_ECS_MEM_OK);
    uint8_t before[32], after[32], src[32];
    uint64_t tok = 0;
    elpis_ecs_composite_commit cm;

    /* K1 only. */
    identity(x, before);
    assert(elpis_ecs_composite_leased_txn_begin(x, 0, src, &tok) == ELPIS_ECS_COMPOSITE_OK && !memcmp(src, before, 32));
    run_schedule(x, 0, tok, 5);
    uint8_t *cand = malloc(elpis_ecs_composite_envelope_bytes(&c));
    assert(elpis_ecs_composite_txn_snapshot_write(x, tok, cand, elpis_ecs_composite_envelope_bytes(&c)) == 0);
    identity(x, after);
    assert(!memcmp(after, before, 32)); /* nothing committed yet: reads see the committed state */
    assert(elpis_ecs_composite_leased_txn_commit(x, 0, tok, &cm) == ELPIS_ECS_COMPOSITE_OK);
    identity(x, after);
    assert(!memcmp(cm.before, before, 32) && !memcmp(cm.after, after, 32) && memcmp(before, after, 32));
    assert(!memcmp(cand + elpis_ecs_composite_envelope_bytes(&c) - 32, after, 32)); /* the snapshot named it */
    assert(memcmp(cm.k1_before, cm.k1_after, 32) && !memcmp(cm.memory_before, cm.memory_after, 32));
    assert(!cm.memory_changed && cm.k1.epoch_before == 0 && cm.k1.epoch_after == 5);
    {
        const elpis_ecsg_k1_experience e[1] = {{R, 5}};
        double s3[F];
        elpis_ecsg_k1_schedule_result res;
        uint64_t kt = 0;
        assert(elpis_ecsg_k1_txn_begin(k1, &kt) == 0);
        assert(elpis_ecsg_k1_txn_run_schedule(k1, kt, X, Y, R, e, 1, 0.002, s3, F, &res) == 0);
        assert(elpis_ecsg_k1_txn_commit(k1, kt, NULL) == 0);
        uint8_t kd[32];
        assert(elpis_ecsg_k1_state_digest(k1, kd) == 0 && !memcmp(kd, cm.k1_after, 32));
    }

    /* Memory only: K1 untouched (epoch, identity). */
    memcpy(before, after, 32);
    assert(elpis_ecs_composite_leased_txn_begin(x, 0, src, &tok) == ELPIS_ECS_COMPOSITE_OK);
    elpis_ecs_mem_observation o = obs(1, 2, 10);
    elpis_ecs_mem_digest key, key2;
    int ms = 99;
    assert(elpis_ecs_composite_leased_txn_observe(x, 0, tok, &o, &key, &ms) == ELPIS_ECS_COMPOSITE_OK && ms == 0);
    assert(elpis_ecs_composite_leased_txn_commit(x, 0, tok, &cm) == ELPIS_ECS_COMPOSITE_OK);
    assert(!memcmp(cm.k1_before, cm.k1_after, 32) && memcmp(cm.memory_before, cm.memory_after, 32));
    assert(cm.memory_changed && cm.k1.epoch_before == 5 && cm.k1.epoch_after == 5 && !memcmp(cm.before, before, 32));
    assert(elpis_ecs_mem_observe(m, 0, &o, &key2, NULL) == ELPIS_ECS_MEM_OK && !memcmp(key.b, key2.b, 32));
    elpis_ecs_mem_digest md;
    assert(elpis_ecs_mem_identity(m, &md) == 0 && !memcmp(md.b, cm.memory_after, 32));

    /* Both together. */
    assert(elpis_ecs_composite_leased_txn_begin(x, 0, src, &tok) == ELPIS_ECS_COMPOSITE_OK);
    o = obs(2, 3, 11);
    assert(elpis_ecs_composite_leased_txn_observe(x, 0, tok, &o, NULL, NULL) == ELPIS_ECS_COMPOSITE_OK);
    run_schedule(x, 0, tok, 3);
    elpis_ecsg_k1_commit_identity ci;
    assert(elpis_ecs_composite_leased_txn_commit_identity(x, 0, tok, &ci) == ELPIS_ECS_COMPOSITE_OK);
    identity(x, after);
    assert(!memcmp(ci.state_after_digest, after, 32) && ci.transition.epoch_after == 8);
    /* The committed envelope's components equal the components evolved alone. */
    assert(elpis_ecs_mem_observe(m, 0, &o, NULL, NULL) == ELPIS_ECS_MEM_OK);
    {
        const elpis_ecsg_k1_experience e[1] = {{R, 3}};
        double s3[F];
        elpis_ecsg_k1_schedule_result res;
        uint64_t kt = 0;
        assert(elpis_ecsg_k1_txn_begin(k1, &kt) == 0);
        assert(elpis_ecsg_k1_txn_run_schedule(k1, kt, X, Y, R, e, 1, 0.002, s3, F, &res) == 0);
        assert(elpis_ecsg_k1_txn_commit(k1, kt, NULL) == 0);
    }
    uint8_t *e = NULL;
    const size_t n = envelope(x, &e);
    uint8_t *ke = malloc(k1n);
    assert(ke && elpis_ecsg_k1_snapshot_write(k1, ke, k1n) == 0 && !memcmp(e + 128, ke, k1n));
    size_t mn = 0;
    uint8_t *me = malloc(n);
    assert(me && elpis_ecs_mem_snapshot(m, me, n, &mn) == 0 && !memcmp(e + 128 + k1n, me, mn));
    free(me);
    free(ke);
    free(e);
    free(cand);
    elpis_ecs_mem_destroy(m);
    elpis_ecsg_k1_destroy(&k1);
    elpis_ecs_composite_destroy(x);
}

/* Interrupted transitions: abort, refusals, discarded K1 transactions, destroy with a transaction open. */
static void test_abort_and_refusals(void) {
    const elpis_ecs_composite_config c = config(0);
    const size_t n = elpis_ecs_composite_envelope_bytes(&c);
    elpis_ecs_composite *x = fresh(0);
    uint8_t id0[32], id1[32], src[32];
    uint8_t *c0 = malloc(n), *c1 = malloc(n);
    assert(c0 && c1);
    identity(x, id0);
    uint64_t tok = 0, tok2 = 0;
    assert(elpis_ecs_composite_leased_txn_begin(x, 0, src, &tok) == 0);
    assert(elpis_ecs_composite_leased_txn_begin(x, 0, src, &tok2) == ELPIS_ECS_COMPOSITE_BUSY);
    elpis_ecs_mem_observation o = obs(1, 2, 5);
    assert(elpis_ecs_composite_leased_txn_observe(x, 0, tok, &o, NULL, NULL) == 0);
    run_schedule(x, 0, tok, 2);
    assert(elpis_ecs_composite_txn_snapshot_write(x, tok, c0, n) == 0);
    /* Recoverable refusals leave both candidates unchanged. */
    elpis_ecs_mem_observation bad = obs(1, 3, 6);
    bad.reserved = 1;
    int ms = 0;
    assert(elpis_ecs_composite_leased_txn_observe(x, 0, tok, &bad, NULL, &ms) == ELPIS_ECS_COMPOSITE_INVALID &&
           ms == ELPIS_ECS_MEM_INVALID);
    bad = obs(1, 3, 4); /* tick older than the candidate's clock */
    assert(elpis_ecs_composite_leased_txn_observe(x, 0, tok, &bad, NULL, &ms) == ELPIS_ECS_COMPOSITE_REFUSED &&
           ms == ELPIS_ECS_MEM_CLOCK);
    elpis_ecs_mem_digest key = dg("missing", 0);
    assert(elpis_ecs_composite_leased_txn_memory_learn(x, 0, tok, &key, 7, &ms) == ELPIS_ECS_COMPOSITE_UNQUALIFIED &&
           ms == ELPIS_ECS_MEM_UNQUALIFIED);
    const elpis_ecsg_k1_experience e[1] = {{R + 1, 1}};
    double s3[F];
    elpis_ecsg_k1_schedule_result res;
    assert(elpis_ecs_composite_leased_txn_run_schedule(x, 0, tok, X, Y, R + 1, e, 1, 0.002, s3, F, &res) ==
           ELPIS_ECS_COMPOSITE_CAPACITY);
    assert(elpis_ecs_composite_leased_txn_observe(x, 0, tok + 1, &o, NULL, NULL) == ELPIS_ECS_COMPOSITE_INVALID);
    assert(elpis_ecs_composite_txn_snapshot_write(x, tok, c1, n) == 0 && !memcmp(c0, c1, n));
    assert(elpis_ecs_composite_txn_snapshot_write(x, tok, c1, n - 1) == ELPIS_ECS_COMPOSITE_CAPACITY);
    /* Abort: wrong token refused; the right one installs nothing. */
    assert(elpis_ecs_composite_txn_abort(x, tok + 1) == ELPIS_ECS_COMPOSITE_INVALID);
    assert(elpis_ecs_composite_txn_abort(x, tok) == 0);
    assert(elpis_ecs_composite_txn_abort(x, tok) == 0); /* none open: OK */
    identity(x, id1);
    assert(!memcmp(id0, id1, 32));
    elpis_ecs_composite_commit cm;
    assert(elpis_ecs_composite_leased_txn_commit(x, 0, tok, &cm) == ELPIS_ECS_COMPOSITE_INVALID);

    /* NONFINITE K1 input discards the whole composite transaction (the memory candidate with it). */
    assert(elpis_ecs_composite_leased_txn_begin(x, 0, src, &tok) == 0);
    assert(elpis_ecs_composite_leased_txn_observe(x, 0, tok, &o, NULL, NULL) == 0);
    double xbad[R * D];
    memcpy(xbad, X, sizeof(xbad));
    xbad[3] = NAN;
    const elpis_ecsg_k1_experience ok[1] = {{R, 1}};
    assert(elpis_ecs_composite_leased_txn_run_schedule(x, 0, tok, xbad, Y, R, ok, 1, 0.002, s3, F, &res) ==
           ELPIS_ECS_COMPOSITE_NONFINITE);
    assert(elpis_ecs_composite_leased_txn_commit(x, 0, tok, &cm) == ELPIS_ECS_COMPOSITE_INVALID);
    elpis_ecs_composite_stats st;
    assert(elpis_ecs_composite_inspect(x, &st) == 0 && !st.txn_open);
    identity(x, id1);
    assert(!memcmp(id0, id1, 32));

    /* A new transaction after the discard starts from the committed state; destroy with it open is clean. */
    assert(elpis_ecs_composite_leased_txn_begin(x, 0, src, &tok) == 0 && !memcmp(src, id0, 32));
    run_schedule(x, 0, tok, 1);
    free(c0);
    free(c1);
    elpis_ecs_composite_destroy(x);
}

/* Exact restore, deterministic replay, and snapshot equivalence. */
static void test_restore_and_replay(void) {
    const elpis_ecs_composite_config c = config(0);
    elpis_ecs_composite *a = fresh(0);
    uint8_t src[32];
    uint64_t tok = 0;
    elpis_ecs_composite_commit cm;
    for (uint64_t t = 1; t <= 4; ++t) {
        assert(elpis_ecs_composite_leased_txn_begin(a, 0, src, &tok) == 0);
        elpis_ecs_mem_observation o = obs(t, t + 1, t);
        assert(elpis_ecs_composite_leased_txn_observe(a, 0, tok, &o, NULL, NULL) == 0);
        if (t % 2) run_schedule(a, 0, tok, t);
        assert(elpis_ecs_composite_leased_txn_commit(a, 0, tok, &cm) == 0);
    }
    uint8_t *e = NULL, *e2 = NULL;
    const size_t n = envelope(a, &e);
    elpis_ecs_composite *b = NULL;
    assert(elpis_ecs_composite_restore(&c, e, n, &b) == ELPIS_ECS_COMPOSITE_OK);
    uint8_t ia[32], ib[32];
    identity(a, ia);
    identity(b, ib);
    assert(!memcmp(ia, ib, 32));
    assert(envelope(b, &e2) == n && !memcmp(e, e2, n));
    free(e2);
    /* Replay: the same transitions on the original and the restored state give the same identities. */
    for (uint64_t t = 5; t <= 7; ++t) {
        elpis_ecs_composite *xs[2] = {a, b};
        uint8_t ids[2][32];
        for (int k = 0; k < 2; ++k) {
            assert(elpis_ecs_composite_leased_txn_begin(xs[k], 0, src, &tok) == 0);
            elpis_ecs_mem_observation o = obs(t, 1, t);
            assert(elpis_ecs_composite_leased_txn_observe(xs[k], 0, tok, &o, NULL, NULL) == 0);
            run_schedule(xs[k], 0, tok, 2);
            assert(elpis_ecs_composite_leased_txn_commit(xs[k], 0, tok, &cm) == 0);
            memcpy(ids[k], cm.after, 32);
        }
        assert(!memcmp(ids[0], ids[1], 32));
    }
    /* Memory reads through the composite agree on both. */
    elpis_ecs_mem_request q;
    memset(&q, 0, sizeof(q));
    q.abi_version = ELPIS_ECS_MEM_ABI_V1;
    q.channel = ELPIS_ECS_MEM_LOOKUP;
    q.relation_type = 3;
    q.target_role = 2;
    q.cue_count = 1;
    q.cue[0].role = 1;
    q.cue[0].entity = dg("entity", 2);
    elpis_ecs_mem_answer aa, ab;
    uint8_t da[32], db[32];
    assert(elpis_ecs_composite_memory_query(a, &q, &aa, da, NULL) == 0 && aa.found);
    assert(elpis_ecs_composite_memory_query(b, &q, &ab, db, NULL) == 0);
    assert(!memcmp(&aa, &ab, sizeof(aa)) && !memcmp(da, db, 32) && !memcmp(aa.answer.b, dg("entity", 3).b, 32));
    q.channel = ELPIS_ECS_MEM_FAST;
    q.candidate_source = ELPIS_ECS_MEM_CANDIDATES_CODEBOOK;
    int ms = 0;
    assert(elpis_ecs_composite_memory_query(a, &q, &aa, da, &ms) == ELPIS_ECS_COMPOSITE_UNQUALIFIED &&
           ms == ELPIS_ECS_MEM_UNQUALIFIED);
    free(e);
    elpis_ecs_composite_destroy(a);
    elpis_ecs_composite_destroy(b);
}

/* Corruption, mismatched identities, old and unsupported formats, configuration mismatch, legacy import. */
static void test_corruption_and_formats(void) {
    const elpis_ecs_composite_config c = config(0);
    const size_t k1n = elpis_ecsg_k1_envelope_bytes(D, N);
    elpis_ecs_composite *a = fresh(0);
    uint8_t src[32];
    uint64_t tok = 0;
    elpis_ecs_composite_commit cm;
    assert(elpis_ecs_composite_leased_txn_begin(a, 0, src, &tok) == 0);
    elpis_ecs_mem_observation o = obs(1, 2, 3);
    assert(elpis_ecs_composite_leased_txn_observe(a, 0, tok, &o, NULL, NULL) == 0);
    run_schedule(a, 0, tok, 2);
    assert(elpis_ecs_composite_leased_txn_commit(a, 0, tok, &cm) == 0);
    uint8_t *e = NULL;
    const size_t n = envelope(a, &e);
    uint8_t *t = malloc(n);
    assert(t);
    elpis_ecs_composite *b = (elpis_ecs_composite *)(uintptr_t)1;
    size_t mem_bytes = 0;
    for (int i = 0; i < 8; ++i) mem_bytes |= (size_t)e[32 + i] << (8 * i);

    /* Every single-bit flip anywhere is refused (stride through the whole envelope, and every header byte). */
    size_t flips = 0;
    for (size_t i = 0; i < n; i += (i < 128 ? 1 : 97)) {
        memcpy(t, e, n);
        t[i] ^= 0x10;
        assert(elpis_ecs_composite_restore(&c, t, n, &b) == ELPIS_ECS_COMPOSITE_CORRUPT);
        assert(b == (elpis_ecs_composite *)(uintptr_t)1);
        ++flips;
    }
    assert(flips > 128);
    /* Flips with a resealed trailer: each field is checked on its own, not only by the trailer. */
    const size_t at[] = {0, 8, 12, 16, 24, 32, 40, 48, 80, 112, 127,          /* header */
                         128 + 5, 128 + k1n - 1,                                /* K1 component */
                         128 + k1n + 3, 128 + k1n + mem_bytes - 1,              /* memory component */
                         128 + k1n + mem_bytes, n - 33};                        /* padding */
    for (size_t i = 0; i < sizeof(at) / sizeof(at[0]); ++i) {
        memcpy(t, e, n);
        t[at[i]] ^= 0x01;
        reseal(t, n);
        assert(elpis_ecs_composite_restore(&c, t, n, &b) == ELPIS_ECS_COMPOSITE_CORRUPT);
        assert(elpis_ecs_composite_envelope_identities(t, n, NULL, NULL, NULL) == ELPIS_ECS_COMPOSITE_CORRUPT ||
               (at[i] > 128 && at[i] < 128 + k1n + mem_bytes)); /* component bytes need component validation */
    }
    /* Mismatched component identity: a K1 component replaced by another valid K1 state, header and trailer
     * consistent with it, still restores only as that other state; the recorded identity must match. */
    {
        elpis_ecsg_k1 *k = NULL;
        assert(elpis_ecsg_k1_create(D, N, R, W0, &k) == 0);
        memcpy(t, e, n);
        assert(elpis_ecsg_k1_snapshot_write(k, t + 128, k1n) == 0); /* the pre-learning K1 state */
        reseal(t, n); /* header 48..80 still names the learned K1 state */
        assert(elpis_ecs_composite_restore(&c, t, n, &b) == ELPIS_ECS_COMPOSITE_CORRUPT);
        elpis_ecsg_k1_destroy(&k);
    }
    /* Unsupported version, flags, wrong magic, truncated and extended lengths. */
    memcpy(t, e, n);
    t[8] = 2;
    reseal(t, n);
    assert(elpis_ecs_composite_restore(&c, t, n, &b) == ELPIS_ECS_COMPOSITE_CORRUPT);
    memcpy(t, e, n);
    t[12] = 1;
    reseal(t, n);
    assert(elpis_ecs_composite_restore(&c, t, n, &b) == ELPIS_ECS_COMPOSITE_CORRUPT);
    memcpy(t, e, n);
    t[7] = '2';
    reseal(t, n);
    assert(elpis_ecs_composite_restore(&c, t, n, &b) == ELPIS_ECS_COMPOSITE_CORRUPT);
    assert(elpis_ecs_composite_restore(&c, e, n - 1, &b) == ELPIS_ECS_COMPOSITE_CORRUPT);
    assert(elpis_ecs_composite_restore(&c, e, 10, &b) == ELPIS_ECS_COMPOSITE_CORRUPT);
    assert(elpis_ecs_composite_restore(&c, NULL, n, &b) == ELPIS_ECS_COMPOSITE_INVALID);
    /* A valid envelope of another configuration. */
    elpis_ecs_composite_config other = c;
    other.memory.slot_capacity = 17;
    assert(elpis_ecs_composite_restore(&other, e, n, &b) == ELPIS_ECS_COMPOSITE_CORRUPT);
    other = c;
    other.memory.learn_rate_q16 = 1234; /* same sizes, different law parameter */
    assert(elpis_ecs_composite_restore(&other, e, n, &b) == ELPIS_ECS_COMPOSITE_CORRUPT);
    assert(b == (elpis_ecs_composite *)(uintptr_t)1);

    /* Old format: a bare ELPISGK1 is refused by restore and admitted only by the explicit import. */
    uint8_t *k1e = e + 128;
    assert(elpis_ecs_composite_restore(&c, k1e, k1n, &b) == ELPIS_ECS_COMPOSITE_CORRUPT);
    elpis_ecs_composite *imp = NULL;
    assert(elpis_ecs_composite_import_k1(&c, k1e, k1n, &imp) == ELPIS_ECS_COMPOSITE_OK);
    uint8_t ii[32], kid[32], mid[32], *ie = NULL;
    const size_t in = envelope(imp, &ie);
    assert(in == n && elpis_ecs_composite_envelope_identities(ie, in, ii, kid, mid) == 0);
    assert(!memcmp(ie + 128, k1e, k1n) && !memcmp(kid, k1e + k1n - 32, 32)); /* K1 bytes preserved */
    elpis_ecs_mem *empty = NULL;
    elpis_ecs_mem_digest md;
    assert(elpis_ecs_mem_create(&c.memory, &empty) == 0 && elpis_ecs_mem_identity(empty, &md) == 0);
    assert(!memcmp(mid, md.b, 32) && memcmp(ii, kid, 32));
    /* import_k1 refuses a composite envelope, a corrupt K1 envelope and another K1 shape. */
    assert(elpis_ecs_composite_import_k1(&c, e, n, &b) == ELPIS_ECS_COMPOSITE_CORRUPT);
    memcpy(t, k1e, k1n);
    t[100] ^= 1;
    assert(elpis_ecs_composite_import_k1(&c, t, k1n, &b) == ELPIS_ECS_COMPOSITE_CORRUPT);
    other = c;
    other.width = N + 1;
    assert(elpis_ecs_composite_import_k1(&other, k1e, k1n, &b) == ELPIS_ECS_COMPOSITE_CORRUPT);
    assert(b == (elpis_ecs_composite *)(uintptr_t)1);
    elpis_ecs_mem_destroy(empty);
    free(ie);
    elpis_ecs_composite_destroy(imp);
    free(t);
    free(e);
    elpis_ecs_composite_destroy(a);
}

/* Capacity, configuration validation and capability admission. */
static void test_capacity_and_admission(void) {
    elpis_ecs_composite_config c = config(0);
    elpis_ecs_composite *x = (elpis_ecs_composite *)(uintptr_t)1;
    elpis_ecs_composite_config bad = c;
    bad.abi_version = 2;
    assert(elpis_ecs_composite_create(&bad, W0, &x) == ELPIS_ECS_COMPOSITE_INVALID);
    bad = c;
    bad.reserved = 1;
    assert(elpis_ecs_composite_create(&bad, W0, &x) == ELPIS_ECS_COMPOSITE_INVALID);
    bad = c;
    bad.max_rows = 0;
    assert(elpis_ecs_composite_create(&bad, W0, &x) == ELPIS_ECS_COMPOSITE_INVALID);
    bad = c;
    bad.memory.capabilities = 2; /* no capability but RECURRENT exists */
    assert(elpis_ecs_composite_create(&bad, W0, &x) == ELPIS_ECS_COMPOSITE_INVALID);
    bad = c;
    bad.memory.dim = 1024;
    bad.memory.banks = 64; /* 2 * 64 * 1024^2 * 4 bytes of banks: beyond the composite memory bound */
    bad.memory.byte_budget = UINT64_C(1) << 40;
    assert(elpis_ecs_composite_envelope_bytes(&bad) == 0);
    assert(elpis_ecs_composite_create(&bad, W0, &x) == ELPIS_ECS_COMPOSITE_CAPACITY);
    bad = c;
    bad.memory.byte_budget = 1024; /* the memory arena does not fit its declared budget */
    assert(elpis_ecs_composite_create(&bad, W0, &x) == ELPIS_ECS_COMPOSITE_CAPACITY);
    assert(x == (elpis_ecs_composite *)(uintptr_t)1);
    assert(elpis_ecs_composite_create(&c, NULL, &x) == ELPIS_ECS_COMPOSITE_INVALID);

    /* Memory capacity inside a transaction: REFUSE policy, full slots. */
    c.memory.slot_capacity = 2;
    c.memory.slot_policy = ELPIS_ECS_MEM_REFUSE;
    assert(elpis_ecs_composite_create(&c, W0, &x) == 0);
    uint8_t src[32];
    uint64_t tok = 0;
    int ms = 0;
    assert(elpis_ecs_composite_leased_txn_begin(x, 0, src, &tok) == 0);
    for (uint64_t i = 0; i < 2; ++i) {
        elpis_ecs_mem_observation o = obs(i, i + 10, 1);
        assert(elpis_ecs_composite_leased_txn_observe(x, 0, tok, &o, NULL, &ms) == 0);
    }
    elpis_ecs_mem_observation o = obs(5, 6, 1);
    assert(elpis_ecs_composite_leased_txn_observe(x, 0, tok, &o, NULL, &ms) == ELPIS_ECS_COMPOSITE_CAPACITY &&
           ms == ELPIS_ECS_MEM_CAPACITY);
    elpis_ecs_composite_commit cm;
    assert(elpis_ecs_composite_leased_txn_commit(x, 0, tok, &cm) == 0 && cm.memory_changed);
    elpis_ecs_composite_destroy(x);

    /* Unqualified learned content: a composite admitting RECURRENT learns; a restore under a configuration that
     * does not admit it is UNQUALIFIED (never a silent drop); under one that does, exact. */
    elpis_ecs_composite_config full = config(ELPIS_ECS_MEM_CAP_RECURRENT);
    assert(elpis_ecs_composite_create(&full, W0, &x) == 0);
    elpis_ecs_mem_digest key;
    assert(elpis_ecs_composite_leased_txn_begin(x, 0, src, &tok) == 0);
    o = obs(1, 2, 1);
    assert(elpis_ecs_composite_leased_txn_observe(x, 0, tok, &o, &key, NULL) == 0);
    assert(elpis_ecs_composite_leased_txn_memory_learn(x, 0, tok, &key, 2, &ms) == 0);
    assert(elpis_ecs_composite_leased_txn_commit(x, 0, tok, &cm) == 0);
    uint8_t *e = NULL;
    const size_t n = envelope(x, &e);
    elpis_ecs_composite *y = (elpis_ecs_composite *)(uintptr_t)1;
    const elpis_ecs_composite_config none = config(0);
    assert(elpis_ecs_composite_restore(&none, e, n, &y) == ELPIS_ECS_COMPOSITE_UNQUALIFIED);
    assert(y == (elpis_ecs_composite *)(uintptr_t)1);
    assert(elpis_ecs_composite_restore(&full, e, n, &y) == ELPIS_ECS_COMPOSITE_OK);
    elpis_ecs_composite_destroy(y);
    free(e);
    elpis_ecs_composite_destroy(x);
}

/* Leases: the only mutation path is the held lease; abort ends any transaction; claims are BUSY mid-transaction. */
static void test_lease(void) {
    elpis_ecs_composite *x = fresh(0);
    uint8_t src[32], id0[32], id1[32];
    uint64_t tok = 0;
    identity(x, id0);
    assert(elpis_ecs_composite_lease_claim(x, 0) == ELPIS_ECS_COMPOSITE_INVALID);
    assert(elpis_ecs_composite_lease_claim(x, 41) == 0 && elpis_ecs_composite_lease_of(x) == 41);
    assert(elpis_ecs_composite_leased_txn_begin(x, 0, src, &tok) == ELPIS_ECS_COMPOSITE_LEASED);
    assert(elpis_ecs_composite_leased_txn_begin(x, 40, src, &tok) == ELPIS_ECS_COMPOSITE_LEASED);
    assert(elpis_ecs_composite_leased_reserve(x, 40, R) == ELPIS_ECS_COMPOSITE_LEASED);
    assert(elpis_ecs_composite_leased_txn_begin(x, 41, src, &tok) == 0);
    assert(elpis_ecs_composite_lease_claim(x, 42) == ELPIS_ECS_COMPOSITE_BUSY);
    assert(elpis_ecs_composite_lease_release(x, 41) == ELPIS_ECS_COMPOSITE_BUSY);
    assert(elpis_ecs_composite_leased_reserve(x, 41, R) == ELPIS_ECS_COMPOSITE_BUSY);
    elpis_ecs_mem_observation o = obs(1, 2, 1);
    assert(elpis_ecs_composite_leased_txn_observe(x, 0, tok, &o, NULL, NULL) == ELPIS_ECS_COMPOSITE_LEASED);
    assert(elpis_ecs_composite_leased_txn_observe(x, 41, tok, &o, NULL, NULL) == 0);
    elpis_ecs_composite_commit cm;
    assert(elpis_ecs_composite_leased_txn_commit(x, 7, tok, &cm) == ELPIS_ECS_COMPOSITE_LEASED);
    assert(elpis_ecs_composite_txn_abort(x, tok) == 0); /* any caller may end it; nothing is installed */
    identity(x, id1);
    assert(!memcmp(id0, id1, 32));
    assert(elpis_ecs_composite_leased_reserve(x, 41, 2 * R) == 0);
    assert(elpis_ecs_composite_lease_release(x, 40) == ELPIS_ECS_COMPOSITE_LEASED);
    assert(elpis_ecs_composite_lease_release(x, 41) == 0 && elpis_ecs_composite_lease_of(x) == 0);
    /* The reservation is not state: the identity is unchanged. */
    identity(x, id1);
    assert(!memcmp(id0, id1, 32));
    elpis_ecs_composite_destroy(x);
}

/* Concurrency: readers never observe a torn state; refusals are BUSY only. */
typedef struct {
    elpis_ecs_composite *x;
    atomic_int *stop;
    size_t n;
    unsigned long ok, busy;
} reader_arg;

static void *reader(void *p) {
    reader_arg *a = p;
    uint8_t *buf = malloc(a->n), id[32], cid[32];
    double out[R];
    assert(buf);
    while (!atomic_load(a->stop)) {
        elpis_ecs_composite_status s = elpis_ecs_composite_snapshot_write(a->x, buf, a->n);
        if (s == ELPIS_ECS_COMPOSITE_BUSY) {
            a->busy++;
            sched_yield();
            continue;
        }
        assert(s == ELPIS_ECS_COMPOSITE_OK);
        assert(elpis_ecs_composite_envelope_identities(buf, a->n, cid, NULL, NULL) == ELPIS_ECS_COMPOSITE_OK);
        s = elpis_ecs_composite_query_identity(a->x, D, X, R, out, id);
        assert(s == ELPIS_ECS_COMPOSITE_OK || s == ELPIS_ECS_COMPOSITE_BUSY);
        a->ok++;
        sched_yield(); /* the guard has no fairness: give the writer a window (progress, not ordering) */
    }
    free(buf);
    return NULL;
}

static void test_concurrency(void) {
    const elpis_ecs_composite_config c = config(0);
    elpis_ecs_composite *x = fresh(0);
    atomic_int stop;
    atomic_init(&stop, 0);
    pthread_t th[3];
    reader_arg args[3];
    for (int i = 0; i < 3; ++i) {
        args[i] = (reader_arg){x, &stop, elpis_ecs_composite_envelope_bytes(&c), 0, 0};
        assert(pthread_create(&th[i], NULL, reader, &args[i]) == 0);
    }
    uint8_t src[32];
    unsigned long commits = 0, busy = 0;
    for (uint64_t t = 1; commits < 40; ++t) {
        uint64_t tok = 0;
        elpis_ecs_composite_status s = elpis_ecs_composite_leased_txn_begin(x, 0, src, &tok);
        if (s == ELPIS_ECS_COMPOSITE_BUSY) {
            busy++;
            sched_yield();
            continue;
        }
        assert(s == 0);
        elpis_ecs_mem_observation o = obs(t % 7, t % 5 + 10, t);
        while ((s = elpis_ecs_composite_leased_txn_observe(x, 0, tok, &o, NULL, NULL)) == ELPIS_ECS_COMPOSITE_BUSY) {
            busy++;
            sched_yield();
        }
        assert(s == 0);
        const elpis_ecsg_k1_experience e[1] = {{R, 1}};
        double s3[F];
        elpis_ecsg_k1_schedule_result res;
        while ((s = elpis_ecs_composite_leased_txn_run_schedule(x, 0, tok, X, Y, R, e, 1, 0.002, s3, F, &res)) ==
               ELPIS_ECS_COMPOSITE_BUSY)
            busy++;
        assert(s == 0);
        elpis_ecs_composite_commit cm;
        while ((s = elpis_ecs_composite_leased_txn_commit(x, 0, tok, &cm)) == ELPIS_ECS_COMPOSITE_BUSY) {
            busy++;
            sched_yield();
        }
        assert(s == 0);
        commits++;
    }
    atomic_store(&stop, 1);
    unsigned long reads = 0;
    for (int i = 0; i < 3; ++i) {
        assert(pthread_join(th[i], NULL) == 0);
        reads += args[i].ok;
    }
    elpis_ecs_composite_stats st;
    assert(elpis_ecs_composite_inspect(x, &st) == 0 && st.commits == 40 && st.k1_epoch == 40);
    printf("concurrency: %lu commits, %lu consistent reads, %lu writer BUSY retries\n", commits, reads, busy);
    elpis_ecs_composite_destroy(x);
}

int main(void) {
    fixture();
    test_layout_and_identity();
    test_transitions();
    test_abort_and_refusals();
    test_restore_and_replay();
    test_corruption_and_formats();
    test_capacity_and_admission();
    test_lease();
    test_concurrency();
    assert(!strcmp(elpis_ecs_composite_code_name(ELPIS_ECS_COMPOSITE_UNQUALIFIED), "ELPIS_ECS_COMPOSITE_UNQUALIFIED"));
    puts("ecs composite: ok");
    return 0;
}
