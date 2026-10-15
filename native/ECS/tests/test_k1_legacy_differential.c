/* Legacy-surface differential guard of K1, the K1 FMS adapter and the ECS composite (the live-coordinate additions
 * of the HOT/WARM correction must change no existing behavior).
 *
 * Part A drives ONLY entry points of the frozen pre-correction surface through one deterministic battery and records
 * every status code, transition, readout, digest and envelope byte in a transcript (refusals included, each followed
 * by the complete envelope it must have left unchanged). Built against the pre-correction headers, the same source
 * is the cross-version check: its transcript over the pre-correction libraries and over the corrected ones must be
 * byte-identical (docs/RUNTIME_CORE.md, "Legacy surface"). In this tree it also proves the battery deterministic
 * (two runs, one transcript) and prints the transcript's SHA-256.
 *
 * Part B (ELPIS_K1_LIVE_COORDINATE: this tree only) proves the additive entry points against the legacy ones in the
 * same build: leased capture + envelope seal equals snapshot_write byte for byte (K1, K1 FMS, composite); a leased
 * begin_at/commit transition installs exactly the bytes the legacy begin/commit_identity installs on a twin; the
 * generation advances by exactly one per installing mutation and never on a refusal; a lease other than the
 * holder's is refused before anything is read.
 *
 * Usage: test_k1_legacy_differential <scratch-dir> [transcript-file]. */
#define _POSIX_C_SOURCE 200809L
#include "elpis/ecs_composite.h"
#include "elpis/ecsg_k1.h"
#include "elpis/ecsg_k1_fms.h"
#include "elpis/fms_pal_posix.h"
#include "elpis/sha256.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { D = 6, N = 36, R = 32, WC = D * N, F = 83, LEASE = 0x5EED };
static double W0[WC], X[R * D], Y[R], X2[R * D], Y2[R];
static const char *SCRATCH;

/* ---- the transcript ------------------------------------------------------------------------------------------- */

static uint8_t *T;
static size_t T_len, T_cap;

static void put(const void *p, size_t n) {
    if (T_len + n > T_cap) {
        T_cap = (T_len + n) * 2u + 4096u;
        T = realloc(T, T_cap);
        assert(T);
    }
    memcpy(T + T_len, p, n);
    T_len += n;
}

/* One record: tag, length, bytes. */
static void rec(const char *tag, const void *p, size_t n) {
    const uint64_t len = (uint64_t)n;
    put(tag, strlen(tag) + 1u);
    put(&len, sizeof(len));
    if (n) put(p, n);
}

static int st(const char *tag, int rc) {
    rec(tag, &rc, sizeof(rc));
    return rc;
}

static void rec_transition(const char *tag, const elpis_ecsg_k1_transition *t) {
    const uint64_t v[6] = {t->epoch_before, t->epoch_after, t->generation_before, t->generation_after, t->steps,
                           t->failed_step};
    rec(tag, v, sizeof(v));
}

static void fixture(void) {
    unsigned s = 20261015u;
    for (size_t i = 0; i < WC; ++i) {
        s = s * 1664525u + 1013904223u;
        W0[i] = ((double)(s >> 8) / 16777216.0 - 0.5) * 0.36;
    }
    for (size_t i = 0; i < R * D; ++i) {
        s = s * 1664525u + 1013904223u;
        X[i] = (double)(s >> 8) / 16777216.0 - 0.5;
        s = s * 1664525u + 1013904223u;
        X2[i] = (double)(s >> 8) / 16777216.0 - 0.5;
    }
    for (size_t i = 0; i < R; ++i) {
        Y[i] = 0.3 * X[i * D] * X[i * D + 1];
        Y2[i] = 0.2 * X2[i * D + 4];
    }
}

/* ---- Part A: the legacy surface --------------------------------------------------------------------------------- */

static void k1_envelope(const char *tag, elpis_ecsg_k1 *k) {
    const size_t n = elpis_ecsg_k1_envelope_bytes(D, N);
    uint8_t *e = calloc(1, n);   /* a refused call leaves zeros, never stale heap bytes */
    uint8_t digest[32] = {0};
    assert(e);
    st("snapshot.rc", elpis_ecsg_k1_snapshot_write(k, e, n));
    rec(tag, e, n);
    st("digest.rc", elpis_ecsg_k1_state_digest(k, digest));
    rec("digest", digest, 32);
    const uint64_t eg[2] = {elpis_ecsg_k1_epoch(k), elpis_ecsg_k1_generation(k)};
    rec("epoch.generation", eg, sizeof(eg));
    free(e);
}

static void legacy_k1(void) {
    elpis_ecsg_k1 *k = NULL, *twin = NULL;
    elpis_ecsg_k1_transition t;
    elpis_ecsg_k1_commit_identity ci;
    double out[R] = {0}, w[WC] = {0}, bad[R * D];
    uint64_t tok = 0, tok2 = 0;
    uint8_t digest[32] = {0};
    const elpis_ecsg_k1_experience schedule[2] = {{R / 2, 3}, {R / 2, 5}};
    elpis_ecsg_k1_schedule_result sr;
    double s3[F] = {0};
    memset(&t, 0, sizeof(t));
    memset(&ci, 0, sizeof(ci));
    memset(&sr, 0, sizeof(sr));

    st("k1.create", elpis_ecsg_k1_create(D, N, R, W0, &k));
    k1_envelope("k1.created", k);
    st("k1.forward", elpis_ecsg_k1_forward(k, X, R, out));
    rec("k1.forward.out", out, sizeof(out));
    memset(&t, 0, sizeof(t));
    st("k1.learn", elpis_ecsg_k1_learn(k, X, Y, R, 0.002, 40, &t));
    rec_transition("k1.learn.t", &t);
    st("k1.consolidate", elpis_ecsg_k1_consolidate(k, X, R, &t));
    rec_transition("k1.consolidate.t", &t);
    k1_envelope("k1.learned", k);
    st("k1.copy_w", elpis_ecsg_k1_copy_w(k, w, WC));
    rec("k1.w", w, sizeof(w));
    st("k1.query_identity", elpis_ecsg_k1_query_identity(k, D, X2, R, out, digest));
    rec("k1.query_identity.out", out, sizeof(out));
    rec("k1.query_identity.digest", digest, 32);

    /* transactions: commit, commit_identity, abort, schedule */
    st("k1.txn_begin", elpis_ecsg_k1_txn_begin(k, &tok));
    st("k1.txn_learn", elpis_ecsg_k1_txn_learn(k, tok, X2, Y2, R, 0.002, 7, &t));
    st("k1.txn_forward", elpis_ecsg_k1_txn_forward(k, tok, X, R, out));
    rec("k1.txn_forward.out", out, sizeof(out));
    st("k1.txn_commit", elpis_ecsg_k1_txn_commit(k, tok, &t));
    rec_transition("k1.txn_commit.t", &t);
    st("k1.txn_begin2", elpis_ecsg_k1_txn_begin(k, &tok));
    st("k1.txn_run_schedule", elpis_ecsg_k1_txn_run_schedule(k, tok, X, Y, R, schedule, 2, 0.002, s3, F, &sr));
    rec("k1.schedule.s3", s3, sizeof(s3));
    rec("k1.schedule.result", &sr, sizeof(sr));
    st("k1.txn_commit_identity", elpis_ecsg_k1_txn_commit_identity(k, tok, &ci));
    rec("k1.commit_identity", &ci, sizeof(ci));
    st("k1.txn_begin3", elpis_ecsg_k1_txn_begin(k, &tok));
    st("k1.txn_learn3", elpis_ecsg_k1_txn_learn(k, tok, X, Y, R, 0.002, 3, &t));
    st("k1.txn_abort", elpis_ecsg_k1_txn_abort(k, tok));
    k1_envelope("k1.after_txns", k);

    /* refusals: each leaves the complete state unchanged */
    memcpy(bad, X, sizeof(bad));
    bad[7] = NAN;
    st("k1.learn.nonfinite", elpis_ecsg_k1_learn(k, bad, Y, R, 0.002, 3, &t));
    st("k1.learn.diverge", elpis_ecsg_k1_learn(k, X2, Y2, R, 1e9, 20, &t));
    st("k1.learn.capacity", elpis_ecsg_k1_learn(k, X, Y, R + 1, 0.002, 1, &t));
    st("k1.copy_w.invalid", elpis_ecsg_k1_copy_w(k, w, WC + 1));
    st("k1.txn_begin.stale", elpis_ecsg_k1_txn_begin(k, &tok));
    st("k1.txn_begin.busy", elpis_ecsg_k1_txn_begin(k, &tok2));
    st("k1.learn.replaces_source", elpis_ecsg_k1_learn(k, X2, Y2, R, 0.002, 1, &t));
    st("k1.txn_commit.stale", elpis_ecsg_k1_txn_commit(k, tok, &t));
    k1_envelope("k1.after_refusals", k);

    /* managed lease */
    st("k1.lease_claim", elpis_ecsg_k1_lease_claim(k, LEASE));
    st("k1.learn.leased", elpis_ecsg_k1_learn(k, X, Y, R, 0.002, 1, &t));
    st("k1.reset.leased", elpis_ecsg_k1_reset(k, &t));
    st("k1.leased_txn_begin", elpis_ecsg_k1_leased_txn_begin(k, LEASE, digest, &tok));
    rec("k1.leased_txn_begin.source", digest, 32);
    st("k1.leased_txn_begin.other", elpis_ecsg_k1_leased_txn_begin(k, LEASE + 1, digest, &tok2));
    st("k1.leased_txn_run_schedule",
       elpis_ecsg_k1_leased_txn_run_schedule(k, LEASE, tok, X2, Y2, R, schedule, 2, 0.002, s3, F, &sr));
    rec("k1.leased.s3", s3, sizeof(s3));
    st("k1.leased_txn_commit_identity", elpis_ecsg_k1_leased_txn_commit_identity(k, LEASE, tok, &ci));
    rec("k1.leased.commit_identity", &ci, sizeof(ci));
    st("k1.lease_release.other", elpis_ecsg_k1_lease_release(k, LEASE + 1));
    st("k1.lease_release", elpis_ecsg_k1_lease_release(k, LEASE));
    k1_envelope("k1.after_lease", k);

    /* restore round trip and reset */
    {
        const size_t n = elpis_ecsg_k1_envelope_bytes(D, N);
        uint8_t *e = calloc(1, n);   /* a refused call leaves zeros, never stale heap bytes */
        assert(e);
        st("k1.snapshot", elpis_ecsg_k1_snapshot_write(k, e, n));
        st("k1.restore", elpis_ecsg_k1_restore(e, n, R, &twin));
        k1_envelope("k1.restored", twin);
        e[100] ^= 1u;
        elpis_ecsg_k1 *corrupt = NULL;
        st("k1.restore.corrupt", elpis_ecsg_k1_restore(e, n, R, &corrupt));
        free(e);
    }
    st("k1.reset", elpis_ecsg_k1_reset(k, &t));
    rec_transition("k1.reset.t", &t);
    k1_envelope("k1.reset", k);
    st("k1.destroy", elpis_ecsg_k1_destroy(&k));
    st("k1.destroy.twin", elpis_ecsg_k1_destroy(&twin));
}

static elpis_ecsg_k1_fms *fms_runtime(const char *name, uint64_t warm) {
    char root[2048];
    fms_config c;
    elpis_ecsg_k1_fms *r = NULL;
    assert(snprintf(root, sizeof root, "%s/%s", SCRATCH, name) < (int)sizeof root);
    memset(&c, 0, sizeof(c));
    c.tier_budget[FMS_WARM] = c.domain_ceiling[FMS_DOM_RAM] = warm;
    c.tier_budget[FMS_COLD] = c.domain_ceiling[FMS_DOM_STORAGE] = 64u * (uint64_t)elpis_ecsg_k1_envelope_bytes(D, N);
    c.high_wm = 0.9f;
    c.low_wm = 0.7f;
    c.max_objects = 8;
    c.hot_absent_policy = c.cold_absent_policy = FMS_REJECT;
    fms_pal *pal = fms_pal_posix_create(root);
    assert(pal);
    fms_ctx *ctx = fms_create(&c, pal);
    assert(ctx);
    assert(elpis_ecsg_k1_fms_create(ctx, 8, &r) == 0);
    return r;
}

static void fms_envelope(const char *tag, elpis_ecsg_k1_fms *r, uint64_t id) {
    const size_t n = elpis_ecsg_k1_envelope_bytes(D, N);
    uint8_t *e = calloc(1, n);   /* a refused call leaves zeros, never stale heap bytes */
    uint8_t digest[32] = {0};
    elpis_ecsg_k1_fms_info info;
    memset(&info, 0, sizeof(info));
    assert(e);
    st("fms.snapshot.rc", elpis_ecsg_k1_fms_snapshot_write(r, id, e, n));
    rec(tag, e, n);
    st("fms.digest.rc", elpis_ecsg_k1_fms_state_digest(r, id, digest));
    rec("fms.digest", digest, 32);
    st("fms.inspect.rc", elpis_ecsg_k1_fms_inspect(r, id, &info));
    /* the logical record, not the timings */
    const uint64_t v[8] = {info.epoch, info.generation, info.tier, info.cold_replica, info.lease_count,
                           info.transaction_open, info.provenance, info.max_rows};
    rec("fms.inspect", v, sizeof(v));
    rec("fms.identity", info.logical_identity, 32);
    free(e);
}

static void legacy_fms(void) {
    const uint64_t image = (uint64_t)elpis_ecsg_k1_image_bytes(D, N);
    elpis_ecsg_k1_fms *r = fms_runtime("legacy-fms", image + image / 2u);   /* one image fits */
    uint8_t ka[32] = {1}, kb[32] = {2}, digest[32] = {0};
    uint64_t a = 0, b = 0, tok = 0;
    elpis_ecsg_k1_transition t;
    elpis_ecsg_k1_commit_identity ci;
    double out[R] = {0};
    const elpis_ecsg_k1_experience schedule[1] = {{R, 4}};
    elpis_ecsg_k1_schedule_result sr;
    double s3[F] = {0};
    memset(&t, 0, sizeof(t));
    memset(&ci, 0, sizeof(ci));
    memset(&sr, 0, sizeof(sr));

    st("fms.register", elpis_ecsg_k1_fms_register(r, ka, D, N, R, W0, &a));
    fms_envelope("fms.registered", r, a);
    st("fms.forward", elpis_ecsg_k1_fms_forward(r, a, X, R, out));
    rec("fms.forward.out", out, sizeof(out));
    st("fms.learn", elpis_ecsg_k1_fms_learn(r, a, X, Y, R, 0.002, 30, &t));
    rec_transition("fms.learn.t", &t);
    st("fms.txn_begin", elpis_ecsg_k1_fms_txn_begin(r, a, &tok));
    st("fms.txn_run_schedule",
       elpis_ecsg_k1_fms_txn_run_schedule(r, a, tok, X2, Y2, R, schedule, 1, 0.002, s3, F, &sr));
    rec("fms.s3", s3, sizeof(s3));
    st("fms.txn_commit_identity", elpis_ecsg_k1_fms_txn_commit_identity(r, a, tok, &ci));
    rec("fms.commit_identity", &ci, sizeof(ci));
    fms_envelope("fms.committed", r, a);
    /* a second state demotes the first (one WARM image); an operation on it materializes it (the legacy law) */
    st("fms.register.b", elpis_ecsg_k1_fms_register(r, kb, D, N, R, W0, &b));
    st("fms.pump", elpis_ecsg_k1_fms_pump(r));
    st("fms.query_identity.demoted", elpis_ecsg_k1_fms_query_identity(r, a, D, X, R, out, digest));
    rec("fms.query_identity.out", out, sizeof(out));
    rec("fms.query_identity.digest", digest, 32);
    fms_envelope("fms.materialized", r, a);
    /* lease and refusals */
    st("fms.lease_claim", elpis_ecsg_k1_fms_lease_claim(r, a, LEASE));
    st("fms.learn.leased", elpis_ecsg_k1_fms_learn(r, a, X, Y, R, 0.002, 1, &t));
    st("fms.leased_txn_begin", elpis_ecsg_k1_fms_leased_txn_begin(r, a, LEASE, digest, &tok));
    rec("fms.leased.source", digest, 32);
    st("fms.leased_txn_run_schedule",
       elpis_ecsg_k1_fms_leased_txn_run_schedule(r, a, LEASE, tok, X, Y, R, schedule, 1, 0.002, s3, F, &sr));
    st("fms.leased_txn_commit_identity", elpis_ecsg_k1_fms_leased_txn_commit_identity(r, a, LEASE, tok, &ci));
    rec("fms.leased.commit_identity", &ci, sizeof(ci));
    st("fms.close.leased", elpis_ecsg_k1_fms_close(r, &a));
    st("fms.lease_release", elpis_ecsg_k1_fms_lease_release(r, a, LEASE));
    fms_envelope("fms.final", r, a);
    st("fms.close.a", elpis_ecsg_k1_fms_close(r, &a));
    st("fms.close.b", elpis_ecsg_k1_fms_close(r, &b));
    st("fms.destroy", elpis_ecsg_k1_fms_destroy(&r));
}

static elpis_ecs_composite_config composite_config(void) {
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
    return c;
}

static elpis_ecs_mem_digest tagged(const char *tag, uint64_t i) {
    char buf[64];
    elpis_ecs_mem_digest d;
    const int n = snprintf(buf, sizeof(buf), "%s:%llu", tag, (unsigned long long)i);
    elpis_sha256(buf, (size_t)n, d.b);
    return d;
}

static elpis_ecs_mem_observation observation(uint64_t subject, uint64_t object, uint64_t tick) {
    elpis_ecs_mem_observation o;
    memset(&o, 0, sizeof(o));
    o.abi_version = ELPIS_ECS_MEM_ABI_V1;
    o.relation_type = 3;
    o.polarity = ELPIS_ECS_MEM_ASSERT;
    o.participant_count = 2;
    o.tick = tick;
    o.provenance = tagged("provenance", tick);
    o.structure_ref = tagged("structure", tick);
    o.participants[0].role = 1;
    o.participants[0].entity = tagged("entity", subject);
    o.participants[1].role = 2;
    o.participants[1].entity = tagged("entity", object);
    return o;
}

static void composite_envelope(const char *tag, elpis_ecs_composite *x) {
    elpis_ecs_composite_stats s;
    uint8_t digest[32] = {0};
    memset(&s, 0, sizeof(s));
    st("composite.inspect", elpis_ecs_composite_inspect(x, &s));
    uint8_t *e = calloc(1, s.envelope_bytes);
    assert(e);
    st("composite.snapshot", elpis_ecs_composite_snapshot_write(x, e, s.envelope_bytes));
    rec(tag, e, s.envelope_bytes);
    st("composite.digest.rc", elpis_ecs_composite_state_digest(x, digest));
    rec("composite.digest", digest, 32);
    free(e);
}

static void legacy_composite(void) {
    const elpis_ecs_composite_config c = composite_config();
    elpis_ecs_composite *x = NULL;
    uint8_t digest[32] = {0};
    uint64_t tok = 0;
    elpis_ecs_mem_digest key;
    int memory_status = 0;
    elpis_ecs_composite_commit commit;
    elpis_ecsg_k1_commit_identity ci;
    const elpis_ecsg_k1_experience schedule[1] = {{R, 4}};
    elpis_ecsg_k1_schedule_result sr;
    double s3[F] = {0};
    memset(&key, 0, sizeof(key));
    memset(&ci, 0, sizeof(ci));
    memset(&sr, 0, sizeof(sr));

    st("composite.create", elpis_ecs_composite_create(&c, W0, &x));
    composite_envelope("composite.created", x);
    st("composite.lease_claim", elpis_ecs_composite_lease_claim(x, LEASE));
    for (uint64_t turn = 0; turn < 3; ++turn) {
        const elpis_ecs_mem_observation o = observation(turn, turn + 7u, turn + 1u);
        st("composite.begin", elpis_ecs_composite_leased_txn_begin(x, LEASE, digest, &tok));
        rec("composite.begin.source", digest, 32);
        st("composite.schedule",
           elpis_ecs_composite_leased_txn_run_schedule(x, LEASE, tok, X, Y, R, schedule, 1, 0.002, s3, F, &sr));
        rec("composite.s3", s3, sizeof(s3));
        st("composite.observe", elpis_ecs_composite_leased_txn_observe(x, LEASE, tok, &o, &key, &memory_status));
        rec("composite.observe.key", key.b, 32);
        rec("composite.observe.status", &memory_status, sizeof(memory_status));
        if (turn % 2u == 0u) {
            memset(&commit, 0, sizeof(commit));
            st("composite.commit", elpis_ecs_composite_leased_txn_commit(x, LEASE, tok, &commit));
            rec("composite.commit.out", &commit, sizeof(commit));
        } else {
            st("composite.commit_identity", elpis_ecs_composite_leased_txn_commit_identity(x, LEASE, tok, &ci));
            rec("composite.commit_identity.out", &ci, sizeof(ci));
        }
        composite_envelope("composite.turn", x);
    }
    st("composite.begin.other", elpis_ecs_composite_leased_txn_begin(x, LEASE + 1, digest, &tok));
    st("composite.begin.abort", elpis_ecs_composite_leased_txn_begin(x, LEASE, digest, &tok));
    st("composite.abort", elpis_ecs_composite_txn_abort(x, tok));
    composite_envelope("composite.final", x);
    st("composite.lease_release", elpis_ecs_composite_lease_release(x, LEASE));
    elpis_ecs_composite_destroy(x);
}

static void battery(void) {
    T_len = 0;
    legacy_k1();
    legacy_fms();
    legacy_composite();
}

/* ---- Part B: the additive live-coordinate entry points against the legacy ones (this tree only) ---------------- */

#ifdef ELPIS_K1_LIVE_COORDINATE
static void live_coordinate_laws(void) {
    const size_t n = elpis_ecsg_k1_envelope_bytes(D, N);
    uint8_t *legacy = malloc(n), *captured = malloc(n), digest[32], source[32];
    elpis_ecsg_k1 *k = NULL, *twin = NULL;
    elpis_ecsg_k1_transition t;
    elpis_ecsg_k1_commit_identity ci;
    uint64_t g = 0, g0 = 0, tok = 0, tok2 = 0;
    double out[R], out2[R];
    const elpis_ecsg_k1_experience schedule[1] = {{R, 4}};
    elpis_ecsg_k1_schedule_result sr;
    double s3[F];
    assert(legacy && captured);

    /* K1: capture + seal == snapshot_write; the new and the legacy transaction install the same bytes. */
    assert(elpis_ecsg_k1_create(D, N, R, W0, &k) == 0 && elpis_ecsg_k1_create(D, N, R, W0, &twin) == 0);
    assert(elpis_ecsg_k1_lease_claim(k, LEASE) == 0 && elpis_ecsg_k1_lease_claim(twin, LEASE) == 0);
    for (int turn = 0; turn < 4; ++turn) {
        g0 = elpis_ecsg_k1_generation(k);
        assert(elpis_ecsg_k1_leased_txn_begin_at(k, LEASE, &g, &tok) == 0 && g == g0);
        assert(elpis_ecsg_k1_leased_txn_run_schedule(k, LEASE, tok, X, Y, R, schedule, 1, 0.002, s3, F, &sr) == 0);
        assert(elpis_ecsg_k1_leased_txn_commit(k, LEASE, tok, &t) == 0);
        assert(t.generation_before == g0 && t.generation_after == g0 + 1u && elpis_ecsg_k1_generation(k) == g0 + 1u);
        assert(elpis_ecsg_k1_leased_txn_begin(twin, LEASE, source, &tok2) == 0);
        assert(elpis_ecsg_k1_leased_txn_run_schedule(twin, LEASE, tok2, X, Y, R, schedule, 1, 0.002, s3, F, &sr) ==
               0);
        assert(elpis_ecsg_k1_leased_txn_commit_identity(twin, LEASE, tok2, &ci) == 0);
        assert(elpis_ecsg_k1_snapshot_write(k, legacy, n) == 0);
        assert(elpis_ecsg_k1_leased_capture(k, LEASE, captured, n, &g) == 0 && g == g0 + 1u);
        assert(!memcmp(captured, legacy, n - 32u) && elpis_ecsg_k1_envelope_seal(captured, n) == 0);
        assert(!memcmp(captured, legacy, n));
        assert(elpis_ecsg_k1_snapshot_write(twin, captured, n) == 0 && !memcmp(captured, legacy, n));
        assert(elpis_ecsg_k1_leased_identity(k, LEASE, digest, &g) == 0 && g == g0 + 1u);
        assert(!memcmp(digest, ci.state_after_digest, 32));
        assert(elpis_ecsg_k1_leased_query(k, LEASE, D, X2, R, out, &g) == 0 && g == g0 + 1u);
        assert(elpis_ecsg_k1_query_identity(twin, D, X2, R, out2, digest) == 0 && !memcmp(out, out2, sizeof out));
    }
    /* Another lease is refused before anything is read; refusals never move the generation. */
    g0 = elpis_ecsg_k1_generation(k);
    assert(elpis_ecsg_k1_leased_query(k, LEASE + 1, D, X, R, out, &g) == ELPIS_ECSG_K1_LEASED);
    assert(elpis_ecsg_k1_leased_capture(k, LEASE + 1, captured, n, &g) == ELPIS_ECSG_K1_LEASED);
    assert(elpis_ecsg_k1_leased_txn_begin_at(k, LEASE + 1, &g, &tok) == ELPIS_ECSG_K1_LEASED);
    assert(elpis_ecsg_k1_learn(k, X, Y, R, 0.002, 1, &t) == ELPIS_ECSG_K1_LEASED);
    assert(elpis_ecsg_k1_leased_txn_begin_at(k, LEASE, &g, &tok) == 0);
    assert(elpis_ecsg_k1_leased_txn_run_schedule(k, LEASE, tok, X2, Y2, R, schedule, 1, 1e9, s3, F, &sr) ==
           ELPIS_ECSG_K1_NONFINITE);   /* diverges: the transaction is discarded, nothing installed */
    assert(elpis_ecsg_k1_generation(k) == g0);
    assert(elpis_ecsg_k1_envelope_seal(captured, n - 1u) == ELPIS_ECSG_K1_INVALID);
    assert(elpis_ecsg_k1_lease_release(k, LEASE) == 0 && elpis_ecsg_k1_lease_release(twin, LEASE) == 0);
    /* An unmanaged installing mutation advances the generation by exactly one. */
    assert(elpis_ecsg_k1_learn(k, X, Y, R, 0.002, 1, &t) == 0 && elpis_ecsg_k1_generation(k) == g0 + 1u);
    assert(elpis_ecsg_k1_reset(k, &t) == 0 && elpis_ecsg_k1_generation(k) == g0 + 2u);
    elpis_ecsg_k1_destroy(&k);
    elpis_ecsg_k1_destroy(&twin);

    /* K1 FMS: the same laws on a resident state. */
    {
        const uint64_t image = (uint64_t)elpis_ecsg_k1_image_bytes(D, N);
        elpis_ecsg_k1_fms *r = fms_runtime("live-fms", 4u * image);
        uint8_t key[32] = {7};
        uint64_t a = 0;
        assert(elpis_ecsg_k1_fms_register(r, key, D, N, R, W0, &a) == 0);
        assert(elpis_ecsg_k1_fms_lease_claim(r, a, LEASE) == 0);
        assert(elpis_ecsg_k1_fms_leased_txn_begin_at(r, a, LEASE, &g, &tok) == 0);
        assert(elpis_ecsg_k1_fms_leased_txn_run_schedule(r, a, LEASE, tok, X, Y, R, schedule, 1, 0.002, s3, F,
                                                         &sr) == 0);
        assert(elpis_ecsg_k1_fms_leased_txn_commit(r, a, LEASE, tok, &t) == 0 && t.generation_after == g + 1u);
        assert(elpis_ecsg_k1_fms_snapshot_write(r, a, legacy, n) == 0);
        assert(elpis_ecsg_k1_fms_leased_capture(r, a, LEASE, captured, n, &g) == 0 && g == t.generation_after);
        assert(elpis_ecsg_k1_fms_envelope_seal(captured, n) == 0 && !memcmp(captured, legacy, n));
        assert(elpis_ecsg_k1_fms_leased_capture(r, a, LEASE + 1, captured, n, &g) == ELPIS_ECSG_K1_LEASED);
        assert(elpis_ecsg_k1_fms_lease_release(r, a, LEASE) == 0);
        assert(elpis_ecsg_k1_fms_close(r, &a) == 0 && elpis_ecsg_k1_fms_destroy(&r) == 0);
    }

    /* Composite: capture + seal == snapshot_write; begin_at/commit_at continue the generation by one. */
    {
        const elpis_ecs_composite_config c = composite_config();
        elpis_ecs_composite *x = NULL;
        elpis_ecs_composite_stats s;
        elpis_ecs_mem_digest key;
        int memory_status = 0;
        assert(elpis_ecs_composite_create(&c, W0, &x) == 0 && elpis_ecs_composite_inspect(x, &s) == 0);
        uint8_t *cl = malloc(s.envelope_bytes), *cc = malloc(s.envelope_bytes);
        assert(cl && cc);
        assert(elpis_ecs_composite_lease_claim(x, LEASE) == 0);
        for (uint64_t turn = 0; turn < 3; ++turn) {
            const elpis_ecs_mem_observation o = observation(turn, turn + 3u, turn + 1u);
            assert(elpis_ecs_composite_leased_txn_begin_at(x, LEASE, &g0, &tok) == 0 && g0 == turn);
            assert(elpis_ecs_composite_leased_txn_run_schedule(x, LEASE, tok, X, Y, R, schedule, 1, 0.002, s3, F,
                                                               &sr) == 0);
            assert(elpis_ecs_composite_leased_txn_observe(x, LEASE, tok, &o, &key, &memory_status) == 0);
            assert(elpis_ecs_composite_leased_txn_commit_at(x, LEASE, tok, &t) == 0);
            assert(t.generation_before == turn && t.generation_after == turn + 1u);
            assert(elpis_ecs_composite_snapshot_write(x, cl, s.envelope_bytes) == 0);
            assert(elpis_ecs_composite_leased_capture(x, LEASE, cc, s.envelope_bytes, &g) == 0 && g == turn + 1u);
            assert(elpis_ecs_composite_envelope_seal(cc, s.envelope_bytes) == 0 && !memcmp(cc, cl, s.envelope_bytes));
            assert(elpis_ecs_composite_leased_identity(x, LEASE, digest, &g) == 0 && g == turn + 1u);
            assert(!memcmp(digest, cl + s.envelope_bytes - 32u, 32));
        }
        assert(elpis_ecs_composite_leased_capture(x, LEASE + 1, cc, s.envelope_bytes, &g) == ELPIS_ECS_COMPOSITE_LEASED);
        assert(elpis_ecs_composite_lease_release(x, LEASE) == 0);
        free(cl);
        free(cc);
        elpis_ecs_composite_destroy(x);
    }
    free(legacy);
    free(captured);
}
#endif

int main(int argc, char **argv) {
    assert(argc >= 2);
    SCRATCH = argv[1];
    fixture();
    battery();
    const size_t first_len = T_len;
    uint8_t *first = malloc(first_len), digest[32];
    assert(first);
    memcpy(first, T, first_len);
    battery();   /* deterministic: the same battery, the same transcript */
    assert(T_len == first_len && !memcmp(T, first, first_len));
    elpis_sha256(T, T_len, digest);
    printf("legacy transcript: %zu bytes, sha256 ", T_len);
    for (int i = 0; i < 32; ++i) printf("%02x", digest[i]);
    printf("\n");
    if (argc >= 3) {
        FILE *f = fopen(argv[2], "wb");
        assert(f && fwrite(T, 1, T_len, f) == T_len && fclose(f) == 0);
    }
#ifdef ELPIS_K1_LIVE_COORDINATE
    live_coordinate_laws();
    printf("live-coordinate laws: capture + seal == snapshot_write; begin_at/commit == begin/commit_identity; "
           "generation +1 per installing mutation, never on a refusal; other leases refused\n");
#endif
    free(first);
    free(T);
    return 0;
}
