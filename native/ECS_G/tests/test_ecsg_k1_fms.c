/* K1 residency over generic FMS (docs/ECS_K1_RUNTIME.md). */
#define _POSIX_C_SOURCE 200809L
#include "elpis/ecsg_k1_fms.h"
#include "elpis/ecsg_executor.h"
#include "elpis/fms_pal.h"
#include "elpis/fms_pal_posix.h"

#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define OK(x) assert((x) == 0)

enum { D = 6, N = 36, R = 64, WC = D * N, HP = 83 * 84 / 2 };
static double W0[WC], X[R * D], Y[R], X2[R * D], Y2[R];
static size_t ENVELOPE, IMAGE;

static void fixture(void)
{
    unsigned s = 777u;
    size_t i;
    for (i = 0; i < WC; ++i) { s = s * 1664525u + 1013904223u; W0[i] = ((double)(s >> 8) / 16777216.0 - 0.5) * 0.36; }
    for (i = 0; i < R * D; ++i) {
        s = s * 1664525u + 1013904223u; X[i] = (double)(s >> 8) / 16777216.0 - 0.5;
        s = s * 1664525u + 1013904223u; X2[i] = (double)(s >> 8) / 16777216.0 - 0.5;
    }
    for (i = 0; i < R; ++i) { Y[i] = 0.3 * X[i * D] * X[i * D + 1]; Y2[i] = 0.2 * X2[i * D + 4]; }
    ENVELOPE = elpis_ecsg_k1_envelope_bytes(D, N);
    IMAGE = elpis_ecsg_k1_image_bytes(D, N);
}

static fms_config config(uint64_t warm, unsigned objects)
{
    fms_config c;
    memset(&c, 0, sizeof(c));
    c.tier_budget[FMS_WARM] = c.domain_ceiling[FMS_DOM_RAM] = warm;
    c.tier_budget[FMS_COLD] = c.domain_ceiling[FMS_DOM_STORAGE] = (uint64_t)ENVELOPE * 100u;
    c.high_wm = 0.9f;
    c.low_wm = 0.7f;
    c.max_objects = objects;
    c.hot_absent_policy = c.cold_absent_policy = FMS_REJECT;
    return c;
}

static elpis_ecsg_k1_fms *runtime_with(fms_pal **pal_out, const char *root, uint64_t warm, unsigned objects)
{
    fms_config c = config(warm, objects);
    fms_pal *pal = fms_pal_posix_create(root);
    fms_ctx *ctx;
    elpis_ecsg_k1_fms *r = NULL;
    assert(pal);
    ctx = fms_create(&c, pal);
    assert(ctx);
    OK(elpis_ecsg_k1_fms_create(ctx, objects, &r));
    if (pal_out) *pal_out = pal;
    return r;
}

static uint64_t state(elpis_ecsg_k1_fms *r, unsigned key)
{
    uint8_t k[32] = {0};
    uint64_t id = 0;
    k[0] = (uint8_t)key;
    OK(elpis_ecsg_k1_fms_register(r, k, D, N, R, W0, &id));
    assert(id);
    return id;
}

static void envelope_of(elpis_ecsg_k1_fms *r, uint64_t id, uint8_t *out) { OK(elpis_ecsg_k1_fms_snapshot_write(r, id, out, ENVELOPE)); }

static void standalone_envelope(elpis_ecsg_k1 *s, uint8_t *out) { OK(elpis_ecsg_k1_snapshot_write(s, out, ENVELOPE)); }

/* FMS-resident K1 is the same law over the same bytes: identical envelopes after identical transitions. */
static void test_resident_k1_equals_standalone(void)
{
    elpis_ecsg_k1_fms *r = runtime_with(NULL, "k1-equal", (uint64_t)IMAGE * 8u, 4);
    elpis_ecsg_k1 *s = NULL;
    uint64_t id = state(r, 1), tok = 0, tok2 = 0;
    uint8_t *a = malloc(ENVELOPE), *b = malloc(ENVELOPE);
    double qa[R], qb[R];
    elpis_ecsg_k1_fms_info info;
    OK(elpis_ecsg_k1_create(D, N, R, W0, &s));
    OK(elpis_ecsg_k1_fms_learn(r, id, X, Y, R, 0.002, 300, NULL));
    OK(elpis_ecsg_k1_learn(s, X, Y, R, 0.002, 300, NULL));
    OK(elpis_ecsg_k1_fms_consolidate(r, id, X, R, NULL));
    OK(elpis_ecsg_k1_consolidate(s, X, R, NULL));
    OK(elpis_ecsg_k1_fms_txn_begin(r, id, &tok));
    OK(elpis_ecsg_k1_fms_txn_learn(r, id, tok, X2, Y2, R, 0.002, 200, NULL));
    OK(elpis_ecsg_k1_fms_txn_consolidate(r, id, tok, X2, R));
    OK(elpis_ecsg_k1_fms_inspect(r, id, &info));
    assert(info.transaction_open && info.lease_count == 1u && info.epoch == 300u);
    OK(elpis_ecsg_k1_fms_txn_commit(r, id, tok, NULL));
    OK(elpis_ecsg_k1_txn_begin(s, &tok2));
    OK(elpis_ecsg_k1_txn_learn(s, tok2, X2, Y2, R, 0.002, 200, NULL));
    OK(elpis_ecsg_k1_txn_consolidate(s, tok2, X2, R));
    OK(elpis_ecsg_k1_txn_commit(s, tok2, NULL));
    envelope_of(r, id, a);
    standalone_envelope(s, b);
    assert(!memcmp(a, b, ENVELOPE));
    OK(elpis_ecsg_k1_fms_forward(r, id, X2, R, qa));
    OK(elpis_ecsg_k1_forward(s, X2, R, qb));
    assert(!memcmp(qa, qb, sizeof(qa)));
    OK(elpis_ecsg_k1_fms_inspect(r, id, &info));
    assert(!info.transaction_open && info.lease_count == 0u && info.epoch == 500u && info.tier == FMS_WARM);
    assert(info.image_bytes == IMAGE && info.envelope_bytes == ENVELOPE);
    OK(elpis_ecsg_k1_fms_close(r, &id));
    OK(elpis_ecsg_k1_fms_destroy(&r));
    elpis_ecsg_k1_destroy(&s);
    free(a);
    free(b);
}


static void test_commit_identity_matches_resident_envelope(void)
{
    elpis_ecsg_k1_fms *r = runtime_with(NULL, "k1-identity", (uint64_t)IMAGE * 4u, 2);
    elpis_ecsg_k1_commit_identity identity;
    elpis_ecsg_k1_fms_info info;
    uint8_t *before = malloc(ENVELOPE), *after = malloc(ENVELOPE);
    uint64_t id = state(r, 42), tok = 0u;

    assert(before && after);
    envelope_of(r, id, before);

    OK(elpis_ecsg_k1_fms_txn_begin(r, id, &tok));
    OK(elpis_ecsg_k1_fms_txn_learn(r, id, tok, X, Y, R, 0.002, 9, NULL));
    OK(elpis_ecsg_k1_fms_txn_consolidate(r, id, tok, X, R));

    assert(elpis_ecsg_k1_fms_txn_commit_identity(r, id, tok, NULL) == ELPIS_ECSG_K1_INVALID);
    OK(elpis_ecsg_k1_fms_inspect(r, id, &info));
    assert(info.transaction_open == 1u && info.lease_count == 1u);

    OK(elpis_ecsg_k1_fms_txn_commit_identity(r, id, tok, &identity));
    envelope_of(r, id, after);

    assert(!memcmp(identity.state_before_digest,
                   before + ENVELOPE - ELPIS_ECSG_K1_DIGEST_BYTES,
                   ELPIS_ECSG_K1_DIGEST_BYTES));
    assert(!memcmp(identity.state_after_digest,
                   after + ENVELOPE - ELPIS_ECSG_K1_DIGEST_BYTES,
                   ELPIS_ECSG_K1_DIGEST_BYTES));
    assert(memcmp(identity.state_before_digest,
                  identity.state_after_digest,
                  ELPIS_ECSG_K1_DIGEST_BYTES) != 0);
    assert(identity.transition.epoch_before == 0u);
    assert(identity.transition.epoch_after == 9u);
    assert(identity.transition.generation_before == 0u);
    assert(identity.transition.generation_after == 1u);

    OK(elpis_ecsg_k1_fms_inspect(r, id, &info));
    assert(info.transaction_open == 0u && info.lease_count == 0u);

    OK(elpis_ecsg_k1_fms_close(r, &id));
    OK(elpis_ecsg_k1_fms_destroy(&r));
    free(before);
    free(after);
}

/* Warm path: no restore, no workspace growth, no FMS movement across many operations. */
static void test_warm_path_operates_over_resident_bytes(void)
{
    elpis_ecsg_k1_fms *r = runtime_with(NULL, "k1-warm", (uint64_t)IMAGE * 8u, 2);
    uint64_t id = state(r, 2);
    elpis_ecsg_k1_counters before, after;
    elpis_ecsg_k1_fms_metrics m0, m1;
    double q[R];
    int i;
    OK(elpis_ecsg_k1_fms_forward(r, id, X, R, q));
    OK(elpis_ecsg_k1_fms_k1_stats(r, id, &before));
    OK(elpis_ecsg_k1_fms_stats(r, &m0));
    for (i = 0; i < 500; ++i) {
        OK(elpis_ecsg_k1_fms_forward(r, id, X, R, q));
    }
    OK(elpis_ecsg_k1_fms_learn(r, id, X, Y, R, 0.002, 50, NULL));
    OK(elpis_ecsg_k1_fms_consolidate(r, id, X, R, NULL));
    OK(elpis_ecsg_k1_fms_k1_stats(r, id, &after));
    OK(elpis_ecsg_k1_fms_stats(r, &m1));
    assert(after.heap_allocations == before.heap_allocations && after.heap_allocations == 2u);   /* workspace only */
    assert(after.forward_calls == before.forward_calls + 500u);
    assert(m1.residency.promotions == m0.residency.promotions && m1.residency.cold_reads == m0.residency.cold_reads);
    assert(m1.leases == 0u && m1.states == 1u);
    OK(elpis_ecsg_k1_fms_close(r, &id));
    OK(elpis_ecsg_k1_fms_destroy(&r));
}

/* WARM -> COLD -> WARM through FMS preserves the complete state; an open transaction pins against demotion. */
static void test_cold_materialization_and_pinning(void)
{
    elpis_ecsg_k1_fms *r = runtime_with(NULL, "k1-cold", (uint64_t)IMAGE + IMAGE / 2u, 3);
    uint8_t *before = malloc(ENVELOPE), *after = malloc(ENVELOPE);
    elpis_ecsg_k1_fms_info info;
    uint64_t a = state(r, 3), b, tok = 0;
    double qa[R], qb[R];
    OK(elpis_ecsg_k1_fms_learn(r, a, X, Y, R, 0.002, 80, NULL));
    OK(elpis_ecsg_k1_fms_consolidate(r, a, X, R, NULL));
    envelope_of(r, a, before);
    OK(elpis_ecsg_k1_fms_forward(r, a, X2, R, qa));
    b = state(r, 4);                         /* the WARM budget holds one image: a is demoted */
    OK(elpis_ecsg_k1_fms_pump(r));
    OK(elpis_ecsg_k1_fms_inspect(r, a, &info));
    assert(info.tier == FMS_COLD && info.cold_replica);
    OK(elpis_ecsg_k1_fms_forward(r, a, X2, R, qb));   /* COLD -> WARM materialization */
    assert(!memcmp(qa, qb, sizeof(qa)));
    envelope_of(r, a, after);
    assert(!memcmp(before, after, ENVELOPE));
    /* pinned by a transaction: not demoted while open */
    OK(elpis_ecsg_k1_fms_txn_begin(r, a, &tok));
    OK(elpis_ecsg_k1_fms_txn_learn(r, a, tok, X2, Y2, R, 0.002, 10, NULL));
    OK(elpis_ecsg_k1_fms_pump(r));
    OK(elpis_ecsg_k1_fms_inspect(r, a, &info));
    assert(info.tier == FMS_WARM && info.transaction_open);
    OK(elpis_ecsg_k1_fms_txn_abort(r, a, tok));
    envelope_of(r, a, after);
    assert(!memcmp(before, after, ENVELOPE));
    OK(elpis_ecsg_k1_fms_close(r, &a));
    OK(elpis_ecsg_k1_fms_close(r, &b));
    OK(elpis_ecsg_k1_fms_destroy(&r));
    free(before);
    free(after);
}

/* Every refusal leaves the authoritative complete state unchanged. */
static int fail_ram(void *self, uint64_t bytes, void **out) { (void)self; (void)bytes; (void)out; return FMS_PAL_ENOMEM; }
static int fail_read(void *self, const fms_cold_token *t, void *out, uint64_t bytes) { (void)self; (void)t; (void)out; (void)bytes; return FMS_PAL_EIO; }

static void test_refusals_leave_the_complete_state_unchanged(void)
{
    int mode;
    for (mode = 0; mode < 4; ++mode) {
        char root[32];
        fms_pal *pal = NULL;
        elpis_ecsg_k1_fms *r;
        uint8_t *before = malloc(ENVELOPE), *after = malloc(ENVELOPE);
        uint64_t a, b = 0, tok = 0;
        double bad[R * D], q[R];
        int (*ram)(void *, uint64_t, void **);
        int (*cold_get)(void *, const fms_cold_token *, void *, uint64_t);
        snprintf(root, sizeof(root), "k1-fault-%d", mode);
        r = runtime_with(&pal, root, (uint64_t)IMAGE + IMAGE / 2u, 3);
        ram = pal->ram_alloc;
        cold_get = pal->cold_get;
        a = state(r, 5);
        OK(elpis_ecsg_k1_fms_learn(r, a, X, Y, R, 0.002, 40, NULL));
        OK(elpis_ecsg_k1_fms_consolidate(r, a, X, R, NULL));
        envelope_of(r, a, before);
        if (mode == 0) {
            /* bad dimensions, non-finite input, divergence, capacity, stale commit */
            memcpy(bad, X, sizeof(bad));
            bad[5] = NAN;
            assert(elpis_ecsg_k1_fms_learn(r, a, bad, Y, R, 0.002, 3, NULL) == ELPIS_ECSG_K1_NONFINITE);
            assert(elpis_ecsg_k1_fms_consolidate(r, a, bad, R, NULL) == ELPIS_ECSG_K1_NONFINITE);
            assert(elpis_ecsg_k1_fms_learn(r, a, X2, Y2, R, 1e9, 20, NULL) == ELPIS_ECSG_K1_NONFINITE);
            assert(elpis_ecsg_k1_fms_learn(r, a, X, Y, R + 1, 0.002, 1, NULL) == ELPIS_ECSG_K1_CAPACITY);
            assert(elpis_ecsg_k1_fms_copy_w(r, a, q, WC + 1) == ELPIS_ECSG_K1_INVALID);
            OK(elpis_ecsg_k1_fms_txn_begin(r, a, &tok));
            OK(elpis_ecsg_k1_fms_txn_learn(r, a, tok, X2, Y2, R, 0.002, 5, NULL));
            OK(elpis_ecsg_k1_fms_learn(r, a, X2, Y2, R, 0.002, 1, NULL));   /* replaces the source */
            assert(elpis_ecsg_k1_fms_txn_commit(r, a, tok, NULL) == ELPIS_ECSG_K1_STALE);
            OK(elpis_ecsg_k1_fms_reset(r, a, NULL));   /* move on: compare from here */
            envelope_of(r, a, before);
            OK(elpis_ecsg_k1_fms_txn_begin(r, a, &tok));
            assert(elpis_ecsg_k1_fms_txn_learn(r, a, tok, X2, Y2, R, 1e9, 20, NULL) == ELPIS_ECSG_K1_NONFINITE);
            assert(elpis_ecsg_k1_fms_txn_commit(r, a, tok, NULL) == ELPIS_ECSG_K1_INVALID);
        } else {
            uint8_t k[32] = {9};
            OK(elpis_ecsg_k1_fms_register(r, k, D, N, R, W0, &b));   /* demotes a */
            OK(elpis_ecsg_k1_fms_pump(r));
            if (mode == 1) {
                pal->ram_alloc = fail_ram;            /* materialization cannot allocate */
                assert(elpis_ecsg_k1_fms_learn(r, a, X, Y, R, 0.002, 3, NULL) == -102);
                assert(elpis_ecsg_k1_fms_forward(r, a, X, R, q) == -102);
                pal->ram_alloc = ram;
            } else if (mode == 2) {
                /* the cold replica cannot be read: refused; FMS marks the object FAILED (never soft-recovered,
                 * as Mutable FMS R0); no pin leaks and nothing is published */
                pal->cold_get = fail_read;
                assert(elpis_ecsg_k1_fms_learn(r, a, X, Y, R, 0.002, 3, NULL) == -106);
                pal->cold_get = cold_get;
                assert(elpis_ecsg_k1_fms_forward(r, a, X, R, q) == -108);
                {
                    elpis_ecsg_k1_fms_metrics m;
                    OK(elpis_ecsg_k1_fms_stats(r, &m));
                    assert(m.leases == 0u && m.residency.pinned_bytes == 0u);
                }
                OK(elpis_ecsg_k1_fms_close(r, &a));
                OK(elpis_ecsg_k1_fms_close(r, &b));
                OK(elpis_ecsg_k1_fms_destroy(&r));
                free(before);
                free(after);
                continue;
            } else {
                DIR *dir = opendir(root);
                struct dirent *ent;
                int changed = 0;
                assert(dir);
                while ((ent = readdir(dir))) {
                    if (strstr(ent->d_name, ".blob")) {
                        int fd = openat(dirfd(dir), ent->d_name, O_WRONLY);
                        uint8_t flip = 0xffu;
                        assert(fd >= 0 && pwrite(fd, &flip, 1, 70) == 1);
                        close(fd);
                        ++changed;
                    }
                }
                closedir(dir);
                assert(changed == 1);
                assert(elpis_ecsg_k1_fms_forward(r, a, X, R, q) == -109);   /* corrupted cold bytes refused */
                OK(elpis_ecsg_k1_fms_close(r, &a));
                OK(elpis_ecsg_k1_fms_close(r, &b));
                OK(elpis_ecsg_k1_fms_destroy(&r));
                free(before);
                free(after);
                continue;
            }
        }
        envelope_of(r, a, after);
        assert(!memcmp(before, after, ENVELOPE));
        {
            elpis_ecsg_k1_fms_metrics m;
            OK(elpis_ecsg_k1_fms_stats(r, &m));
            assert(m.leases == 0u && m.residency.pinned_bytes == 0u);
        }
        OK(elpis_ecsg_k1_fms_close(r, &a));
        if (b) OK(elpis_ecsg_k1_fms_close(r, &b));
        OK(elpis_ecsg_k1_fms_destroy(&r));
        free(before);
        free(after);
    }
}

static void test_envelopes_and_w_only_imports(void)
{
    elpis_ecsg_k1_fms *r = runtime_with(NULL, "k1-import", (uint64_t)IMAGE * 8u, 4);
    elpis_ecsg_executor *e = NULL;
    uint8_t *env = malloc(ENVELOPE), *again = malloc(ENVELOPE), snap[40 + WC * 8];
    uint8_t k1[32] = {11}, k2[32] = {12}, k3[32] = {13};
    uint64_t a = state(r, 10), b = 0, c = 0, d = 0;
    elpis_ecsg_k1_fms_info info;
    double h[HP];
    size_t i;
    OK(elpis_ecsg_k1_fms_learn(r, a, X, Y, R, 0.002, 33, NULL));
    OK(elpis_ecsg_k1_fms_consolidate(r, a, X, R, NULL));
    envelope_of(r, a, env);
    OK(elpis_ecsg_k1_fms_restore(r, k1, env, ENVELOPE, R, &b));
    envelope_of(r, b, again);
    assert(!memcmp(env, again, ENVELOPE));
    env[200] ^= 1u;
    assert(elpis_ecsg_k1_fms_restore(r, k2, env, ENVELOPE, R, &c) == ELPIS_ECSG_K1_CORRUPT && c == 0u);
    assert(elpis_ecsg_k1_fms_restore(r, k1, again, ENVELOPE, R, &c) == ELPIS_ECSG_K1_INVALID);   /* same logical id */
    OK(elpis_ecsg_executor_create(D, N, R, W0, &e));
    OK(elpis_ecsg_executor_snapshot_write(e, snap, sizeof(snap)));
    assert(elpis_ecsg_k1_fms_restore(r, k3, snap, sizeof(snap), R, &d) == ELPIS_ECSG_K1_CORRUPT);
    OK(elpis_ecsg_k1_fms_import_w_only(r, k3, snap, sizeof(snap), R, &d));
    OK(elpis_ecsg_k1_fms_inspect(r, d, &info));
    assert(info.provenance == ELPIS_ECSG_K1_UNCONSOLIDATED_IMPORT);
    OK(elpis_ecsg_k1_fms_copy_h_packed(r, d, h, HP));
    for (i = 0; i < HP; ++i) assert(h[i] == 0.0);
    elpis_ecsg_executor_destroy(&e);
    OK(elpis_ecsg_k1_fms_close(r, &a));
    OK(elpis_ecsg_k1_fms_close(r, &b));
    OK(elpis_ecsg_k1_fms_close(r, &d));
    OK(elpis_ecsg_k1_fms_destroy(&r));
    free(env);
    free(again);
}

typedef struct { elpis_ecsg_k1_fms *r; uint64_t id; int busy; } worker;

static void *work(void *p)
{
    worker *w = (worker *)p;
    double q[R];
    int i;
    for (i = 0; i < 400; ++i) {
        int rc = (i % 10 == 0) ? elpis_ecsg_k1_fms_learn(w->r, w->id, X, Y, R, 0.002, 2, NULL)
                               : elpis_ecsg_k1_fms_forward(w->r, w->id, X, R, q);
        assert(rc == 0 || rc == ELPIS_ECSG_K1_BUSY);
        w->busy += rc == ELPIS_ECSG_K1_BUSY;
    }
    return NULL;
}

static void test_independent_states_share_no_cognitive_lock(void)
{
    elpis_ecsg_k1_fms *r = runtime_with(NULL, "k1-threads", (uint64_t)IMAGE * 8u, 4);
    uint64_t a = state(r, 20), b = state(r, 21);
    worker wa = {r, a, 0}, wb = {r, b, 0}, wa2 = {r, a, 0};
    pthread_t t1, t2, t3;
    assert(!pthread_create(&t1, NULL, work, &wa) && !pthread_create(&t2, NULL, work, &wb) &&
           !pthread_create(&t3, NULL, work, &wa2));
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    pthread_join(t3, NULL);
    printf("k1-fms threads: same-state BUSY=%d, independent state BUSY=%d\n", wa.busy + wa2.busy, wb.busy);
    assert(wb.busy == 0);
    OK(elpis_ecsg_k1_fms_close(r, &a));
    OK(elpis_ecsg_k1_fms_close(r, &b));
    OK(elpis_ecsg_k1_fms_destroy(&r));
}

/* The refusal contract of ecsg_k1.h, through the adapter: recoverable refusals keep the transaction (and its WRITE
 * pin) open; fatal ones discard it and release the pin; authority never changes on a refusal. */
static void test_transaction_refusal_contract(void)
{
    elpis_ecsg_k1_fms *r = runtime_with(NULL, "k1-txn", (uint64_t)IMAGE * 4u, 2);
    elpis_ecsg_k1 *ref = NULL;
    uint8_t *got = malloc(ENVELOPE), *want = malloc(ENVELOPE);
    uint64_t a = state(r, 21), tok = 0;
    elpis_ecsg_k1_fms_info info;
    double bad[R * D];
    OK(elpis_ecsg_k1_create(D, N, R, W0, &ref));
    OK(elpis_ecsg_k1_fms_txn_begin(r, a, &tok));
    assert(elpis_ecsg_k1_fms_txn_learn(r, a, tok, X, Y, R + 1, 0.002, 5, NULL) == ELPIS_ECSG_K1_CAPACITY);
    assert(elpis_ecsg_k1_fms_txn_learn(r, a, tok, X, Y, R, -1.0, 5, NULL) == ELPIS_ECSG_K1_INVALID);
    assert(elpis_ecsg_k1_fms_txn_consolidate(r, a, tok, X, R + 1) == ELPIS_ECSG_K1_CAPACITY);
    OK(elpis_ecsg_k1_fms_inspect(r, a, &info));
    assert(info.transaction_open == 1u && info.lease_count == 1u);   /* still open, still pinned for writing */
    OK(elpis_ecsg_k1_fms_txn_learn(r, a, tok, X, Y, R, 0.002, 5, NULL));   /* retry */
    OK(elpis_ecsg_k1_fms_txn_consolidate(r, a, tok, X, R));
    assert(elpis_ecsg_k1_fms_txn_abort(r, a, tok + 1u) == ELPIS_ECSG_K1_INVALID);   /* wrong token: nothing */
    OK(elpis_ecsg_k1_fms_txn_commit(r, a, tok, NULL));
    OK(elpis_ecsg_k1_learn(ref, X, Y, R, 0.002, 5, NULL));
    OK(elpis_ecsg_k1_consolidate(ref, X, R, NULL));
    envelope_of(r, a, got);
    standalone_envelope(ref, want);
    assert(!memcmp(got, want, ENVELOPE));
    OK(elpis_ecsg_k1_fms_txn_abort(r, a, tok));   /* nothing open: OK */
    /* fatal: NONFINITE discards the transaction and releases its pin */
    memcpy(bad, X, sizeof(bad));
    bad[3] = NAN;
    OK(elpis_ecsg_k1_fms_txn_begin(r, a, &tok));
    assert(elpis_ecsg_k1_fms_txn_learn(r, a, tok, bad, Y, R, 0.002, 5, NULL) == ELPIS_ECSG_K1_NONFINITE);
    OK(elpis_ecsg_k1_fms_inspect(r, a, &info));
    assert(info.transaction_open == 0u && info.lease_count == 0u);
    assert(elpis_ecsg_k1_fms_txn_commit(r, a, tok, NULL) == ELPIS_ECSG_K1_INVALID);
    OK(elpis_ecsg_k1_fms_txn_begin(r, a, &tok));
    assert(elpis_ecsg_k1_fms_txn_learn(r, a, tok, X2, Y2, R, 1e9, 20, NULL) == ELPIS_ECSG_K1_NONFINITE);
    OK(elpis_ecsg_k1_fms_inspect(r, a, &info));
    assert(info.transaction_open == 0u && info.lease_count == 0u);
    envelope_of(r, a, got);
    assert(!memcmp(got, want, ENVELOPE));
    elpis_ecsg_k1_destroy(&ref);
    OK(elpis_ecsg_k1_fms_close(r, &a));
    OK(elpis_ecsg_k1_fms_destroy(&r));
    free(got);
    free(want);
}

/* Hostile serialized dimensions are refused before registration or allocation; an imported state that is then
 * consolidated is COMPLETE. */
static void test_hostile_imports_and_provenance(void)
{
    elpis_ecsg_k1_fms *r = runtime_with(NULL, "k1-hostile", (uint64_t)IMAGE * 4u, 2);
    elpis_ecsg_k1_fms_metrics m;
    elpis_ecsg_executor *e = NULL;
    elpis_ecsg_k1_fms_info info;
    uint8_t tiny[40 + 8 * 64], snap[40 + WC * 8], key[32] = {31};
    uint64_t id = 0;
    unsigned i;
    memset(tiny, 0, sizeof(tiny));
    memcpy(tiny, "ELPISG01", 8);
    tiny[8] = 1u;
    tiny[16] = 64u;   /* dim 64, width 1: a 552-byte request for a multi-GB K1 state */
    tiny[24] = 1u;
    assert(elpis_ecsg_k1_fms_import_w_only(r, key, tiny, sizeof(tiny), R, &id) == ELPIS_ECSG_K1_CAPACITY && id == 0u);
    for (i = 0; i < 8u; ++i) tiny[24 + i] = 0xffu;   /* width UINT64_MAX */
    tiny[16] = 2u;
    assert(elpis_ecsg_k1_fms_import_w_only(r, key, tiny, sizeof(tiny), R, &id) == ELPIS_ECSG_K1_CAPACITY && id == 0u);
    OK(elpis_ecsg_k1_fms_stats(r, &m));
    assert(m.states == 0u && m.residency.objects == 0u);
    OK(elpis_ecsg_executor_create(D, N, R, W0, &e));
    OK(elpis_ecsg_executor_snapshot_write(e, snap, sizeof(snap)));
    OK(elpis_ecsg_k1_fms_import_w_only(r, key, snap, sizeof(snap), R, &id));
    OK(elpis_ecsg_k1_fms_inspect(r, id, &info));
    assert(info.provenance == ELPIS_ECSG_K1_UNCONSOLIDATED_IMPORT);
    OK(elpis_ecsg_k1_fms_consolidate(r, id, X, R, NULL));
    OK(elpis_ecsg_k1_fms_inspect(r, id, &info));
    assert(info.provenance == ELPIS_ECSG_K1_COMPLETE);
    OK(elpis_ecsg_k1_fms_reset(r, id, NULL));
    OK(elpis_ecsg_k1_fms_inspect(r, id, &info));
    assert(info.provenance == ELPIS_ECSG_K1_RESET);
    elpis_ecsg_executor_destroy(&e);
    OK(elpis_ecsg_k1_fms_close(r, &id));
    OK(elpis_ecsg_k1_fms_destroy(&r));
}

/* The canonical turn over the resident state: the experience schedule equals the standalone one bitwise (state and
 * readout), under the transaction's WRITE pin; a discarding refusal releases the pin; authority unchanged. */
static void test_experience_schedule_resident_equals_standalone(void)
{
    static double xs[2 * R * D], ys[2 * R], bad[2 * R * D];
    const elpis_ecsg_k1_experience sched[2] = {{R, 6}, {R / 2, 11}};
    elpis_ecsg_k1_fms *r = runtime_with(NULL, "k1-schedule", (uint64_t)IMAGE * 4u, 2);
    elpis_ecsg_k1 *ref = NULL;
    elpis_ecsg_k1_schedule_result res, rres;
    elpis_ecsg_k1_fms_info info;
    uint8_t *got = malloc(ENVELOPE), *want = malloc(ENVELOPE);
    uint64_t a = state(r, 41), tok = 0, rtok = 0;
    double s3[83], rs3[83];
    size_t i;
    memcpy(xs, X, sizeof(X));
    memcpy(xs + R * D, X2, sizeof(X2));
    memcpy(ys, Y, sizeof(Y));
    memcpy(ys + R, Y2, sizeof(Y2));
    OK(elpis_ecsg_k1_create(D, N, R, W0, &ref));
    for (i = 0; i < 3; ++i) {
        OK(elpis_ecsg_k1_fms_txn_begin(r, a, &tok));
        assert(elpis_ecsg_k1_fms_txn_run_schedule(r, a, tok, xs, ys, R + R / 2, sched, 2, 0.002, s3, 83, &res) == 0);
        OK(elpis_ecsg_k1_fms_txn_commit(r, a, tok, NULL));
        OK(elpis_ecsg_k1_txn_begin(ref, &rtok));
        assert(elpis_ecsg_k1_txn_run_schedule(ref, rtok, xs, ys, R + R / 2, sched, 2, 0.002, rs3, 83, &rres) == 0);
        OK(elpis_ecsg_k1_txn_commit(ref, rtok, NULL));
        assert(!memcmp(s3, rs3, sizeof(s3)) && res.epoch_after == rres.epoch_after);
        envelope_of(r, a, got);
        standalone_envelope(ref, want);
        assert(!memcmp(got, want, ENVELOPE));
    }
    /* recoverable refusal keeps the transaction and its WRITE pin; a non-finite experience discards both */
    OK(elpis_ecsg_k1_fms_txn_begin(r, a, &tok));
    assert(elpis_ecsg_k1_fms_txn_run_schedule(r, a, tok, xs, ys, R + R / 2 + 1, sched, 2, 0.002, s3, 83, &res) ==
           ELPIS_ECSG_K1_INVALID);
    OK(elpis_ecsg_k1_fms_inspect(r, a, &info));
    assert(info.transaction_open == 1u && info.lease_count == 1u);
    memcpy(bad, xs, sizeof(bad));
    for (i = 0; i < R * D; ++i) bad[R * D + i] = 1e150;
    assert(elpis_ecsg_k1_fms_txn_run_schedule(r, a, tok, bad, ys, R + R / 2, sched, 2, 0.002, s3, 83, &res) ==
           ELPIS_ECSG_K1_NONFINITE);
    assert(res.failed_experience == 2u);
    OK(elpis_ecsg_k1_fms_inspect(r, a, &info));
    assert(info.transaction_open == 0u && info.lease_count == 0u);
    envelope_of(r, a, got);
    assert(!memcmp(got, want, ENVELOPE));
    elpis_ecsg_k1_destroy(&ref);
    OK(elpis_ecsg_k1_fms_close(r, &a));
    OK(elpis_ecsg_k1_fms_destroy(&r));
    free(got);
    free(want);
}


static void test_state_digest_matches_resident_snapshot_trailer(void)
{
    elpis_ecsg_k1_fms *r = runtime_with(NULL, "k1-state-digest", (uint64_t)IMAGE * 4u, 2);
    uint8_t digest[ELPIS_ECSG_K1_DIGEST_BYTES];
    uint8_t *snapshot = malloc(ENVELOPE);
    uint64_t id = state(r, 43);

    assert(snapshot);
    envelope_of(r, id, snapshot);
    OK(elpis_ecsg_k1_fms_state_digest(r, id, digest));

    assert(!memcmp(
        digest,
        snapshot + ENVELOPE - ELPIS_ECSG_K1_DIGEST_BYTES,
        ELPIS_ECSG_K1_DIGEST_BYTES
    ));

    OK(elpis_ecsg_k1_fms_close(r, &id));
    OK(elpis_ecsg_k1_fms_destroy(&r));
    free(snapshot);
}

int main(void)
{
    fixture();
    assert(elpis_ecsg_k1_fms_abi_version() == ELPIS_ECSG_K1_FMS_ABI_V1);
    test_resident_k1_equals_standalone();
    test_commit_identity_matches_resident_envelope();
    test_warm_path_operates_over_resident_bytes();
    test_cold_materialization_and_pinning();
    test_refusals_leave_the_complete_state_unchanged();
    test_envelopes_and_w_only_imports();
    test_independent_states_share_no_cognitive_lock();
    test_transaction_refusal_contract();
    test_hostile_imports_and_provenance();
    test_experience_schedule_resident_equals_standalone();
    printf("ecsg_k1_fms: resident K1 = standalone K1; warm path over resident bytes; COLD->WARM; pinning; "
           "refusals leave the complete state unchanged; envelopes and W-only imports; the transaction refusal "
           "contract; hostile imports; provenance; the experience schedule (resident = standalone)\n");
    return 0;
    test_state_digest_matches_resident_snapshot_trailer();
}
