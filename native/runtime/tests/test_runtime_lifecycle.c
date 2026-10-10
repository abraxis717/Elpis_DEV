/* RuntimeCore's turn lifecycle over real native K1 (production runtime library): once a managed turn has opened a
 * native transaction, exactly one terminal native action ends it (commit or abort) before RuntimeCore forgets it,
 * whatever lifecycle path the caller takes: explicit abort, commit, close, destroy, reopen, fail-stop then close.
 * Each case runs over a standalone K1 state (libelpis_ecsg_k1) and an FMS-resident one (libelpis_ecsg_k1_fms over a
 * real FMS context), and checks the native abort count, the retained-state digest, epoch and generation, the
 * durable continuity record, that the state takes another transaction, and for the resident state that the
 * transaction's WRITE pin is released and the adapter's own lifetime guard holds while the turn is open.
 *
 * Usage: test_runtime_lifecycle <scratch-dir>. Deterministic: no threads, no sleeps. */
#define _POSIX_C_SOURCE 200809L
#include "elpis/ecsg_k1.h"
#include "elpis/ecsg_k1_fms.h"
#include "elpis/fms_pal_posix.h"
#include "elpis/runtime.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OK(x) assert((x) == ELPIS_RUNTIME_OK)

enum { DIM = 6, WIDTH = 36, ROWS = 4, S3 = 83 };

/* Typed wrappers: the K1 and K1 FMS entry points as RuntimeCore's function tables (no casts of function types). */
static int k1_digest(void *s, uint8_t out[32]) { return elpis_ecsg_k1_state_digest(s, out); }
static int k1_reserve(void *s, size_t rows) { return elpis_ecsg_k1_reserve(s, rows); }
static int k1_begin(void *s, uint64_t *token) { return elpis_ecsg_k1_txn_begin(s, token); }
static int k1_schedule(void *s, uint64_t token, const double *x, const double *y, size_t rows,
                       const elpis_runtime_experience *schedule, size_t n, double rate, double *s3, size_t s3n,
                       elpis_runtime_schedule_result *result) {
    return elpis_ecsg_k1_txn_run_schedule(s, token, x, y, rows, (const elpis_ecsg_k1_experience *)schedule, n, rate,
                                          s3, s3n, (elpis_ecsg_k1_schedule_result *)result);
}
static int k1_commit(void *s, uint64_t token, elpis_runtime_commit_identity *identity) {
    return elpis_ecsg_k1_txn_commit_identity(s, token, (elpis_ecsg_k1_commit_identity *)identity);
}
static int k1_abort(void *s, uint64_t token) { return elpis_ecsg_k1_txn_abort(s, token); }
static int k1_query(void *s, size_t dim, const double *x, size_t rows, double *out, uint8_t digest[32]) {
    return elpis_ecsg_k1_query_identity(s, dim, x, rows, out, digest);
}
static const elpis_runtime_k1_api K1_API = {k1_digest, k1_reserve, k1_begin, k1_schedule, k1_commit, k1_abort,
                                          k1_query};

static int fms_digest(void *r, uint64_t id, uint8_t out[32]) { return elpis_ecsg_k1_fms_state_digest(r, id, out); }
static int fms_reserve(void *r, uint64_t id, size_t rows) { return elpis_ecsg_k1_fms_reserve(r, id, rows); }
static int fms_begin(void *r, uint64_t id, uint64_t *token) { return elpis_ecsg_k1_fms_txn_begin(r, id, token); }
static int fms_schedule(void *r, uint64_t id, uint64_t token, const double *x, const double *y, size_t rows,
                        const elpis_runtime_experience *schedule, size_t n, double rate, double *s3, size_t s3n,
                        elpis_runtime_schedule_result *result) {
    return elpis_ecsg_k1_fms_txn_run_schedule(r, id, token, x, y, rows, (const elpis_ecsg_k1_experience *)schedule,
                                              n, rate, s3, s3n, (elpis_ecsg_k1_schedule_result *)result);
}
static int fms_commit(void *r, uint64_t id, uint64_t token, elpis_runtime_commit_identity *identity) {
    return elpis_ecsg_k1_fms_txn_commit_identity(r, id, token, (elpis_ecsg_k1_commit_identity *)identity);
}
static int fms_abort(void *r, uint64_t id, uint64_t token) { return elpis_ecsg_k1_fms_txn_abort(r, id, token); }
static int fms_query_identity_entry(void *r, uint64_t id, size_t dim, const double *x, size_t rows, double *out,
                                    uint8_t digest[32]) {
    return elpis_ecsg_k1_fms_query_identity(r, id, dim, x, rows, out, digest);
}
static const elpis_runtime_k1_fms_api FMS_API = {fms_digest, fms_reserve, fms_begin, fms_schedule, fms_commit,
                                                 fms_abort, fms_query_identity_entry};

static double W0[DIM * WIDTH], X[ROWS * DIM], Y[ROWS];
static const elpis_runtime_experience SCHEDULE[2] = {{2, 3}, {2, 5}};
static char SCRATCH[3072];

/* One native K1 state of either kind, as RuntimeCore's descriptor and as the test inspects it natively. */
typedef struct {
    int resident;
    elpis_ecsg_k1 *k1;
    elpis_ecsg_k1_fms *fms;
    uint64_t id;
    elpis_runtime_substrate d;
} subject;

static subject make(int resident, unsigned n) {
    subject s;
    memset(&s, 0, sizeof(s));
    s.resident = resident;
    if (!resident) {
        assert(elpis_ecsg_k1_create(DIM, WIDTH, 8, W0, &s.k1) == ELPIS_ECSG_K1_OK);
        elpis_runtime_substrate d = {ELPIS_RUNTIME_SUBSTRATE_K1, 0, s.k1, 0, 1, DIM, &K1_API};
        s.d = d;
        return s;
    }
    char root[3200];
    assert(snprintf(root, sizeof(root), "%s/cold-%u", SCRATCH, n) < (int)sizeof(root));
    char cmd[3300];
    assert(snprintf(cmd, sizeof(cmd), "rm -rf '%s' && mkdir -p '%s'", root, root) < (int)sizeof(cmd));
    assert(system(cmd) == 0);
    size_t envelope = elpis_ecsg_k1_envelope_bytes(DIM, WIDTH);
    fms_config c;
    memset(&c, 0, sizeof(c));
    c.tier_budget[FMS_WARM] = c.domain_ceiling[FMS_DOM_RAM] = (uint64_t)envelope * 8u;
    c.tier_budget[FMS_COLD] = c.domain_ceiling[FMS_DOM_STORAGE] = (uint64_t)envelope * 100u;
    c.high_wm = 0.9f;
    c.low_wm = 0.7f;
    c.max_objects = 4;
    c.hot_absent_policy = c.cold_absent_policy = FMS_REJECT;
    fms_pal *pal = fms_pal_posix_create(root);
    assert(pal);
    fms_ctx *ctx = fms_create(&c, pal);
    assert(ctx);
    assert(elpis_ecsg_k1_fms_create(ctx, 2, &s.fms) == ELPIS_ECSG_K1_OK);
    uint8_t key[32] = {7};
    assert(elpis_ecsg_k1_fms_register(s.fms, key, DIM, WIDTH, 8, W0, &s.id) == ELPIS_ECSG_K1_OK && s.id);
    elpis_runtime_substrate d = {ELPIS_RUNTIME_SUBSTRATE_K1_FMS, 0, s.fms, s.id, 2, DIM, &FMS_API};
    s.d = d;
    return s;
}

static void destroy_subject(subject *s) {
    if (!s->resident) {
        assert(elpis_ecsg_k1_destroy(&s->k1) == ELPIS_ECSG_K1_OK);
        return;
    }
    /* Released: the resident state closes and its runtime is destroyed (both refuse BUSY while a txn is open). */
    assert(elpis_ecsg_k1_fms_close(s->fms, &s->id) == ELPIS_ECSG_K1_OK);
    assert(elpis_ecsg_k1_fms_destroy(&s->fms) == ELPIS_ECSG_K1_OK);
}

/* What an implicit abort must leave untouched. */
typedef struct {
    uint8_t digest[32];
    uint64_t epoch, generation;
} retained;

static retained observe(const subject *s) {
    retained r;
    if (!s->resident) {
        assert(elpis_ecsg_k1_state_digest(s->k1, r.digest) == ELPIS_ECSG_K1_OK);
        r.epoch = elpis_ecsg_k1_epoch(s->k1);
        r.generation = elpis_ecsg_k1_generation(s->k1);
    } else {
        elpis_ecsg_k1_fms_info info;
        assert(elpis_ecsg_k1_fms_state_digest(s->fms, s->id, r.digest) == ELPIS_ECSG_K1_OK);
        assert(elpis_ecsg_k1_fms_inspect(s->fms, s->id, &info) == ELPIS_ECSG_K1_OK);
        r.epoch = info.epoch;
        r.generation = info.generation;
    }
    return r;
}

static int same(retained a, retained b) {
    return !memcmp(a.digest, b.digest, 32) && a.epoch == b.epoch && a.generation == b.generation;
}

/* Whether the state holds an open native transaction; for a resident state also the pins it holds. */
static int txn_open(const subject *s, uint32_t *leases) {
    if (!s->resident) {
        uint64_t token = 0;
        int rc = elpis_ecsg_k1_txn_begin(s->k1, &token);   /* BUSY exactly while another transaction is open */
        if (rc == ELPIS_ECSG_K1_OK) {
            assert(elpis_ecsg_k1_txn_abort(s->k1, token) == ELPIS_ECSG_K1_OK);
            return 0;
        }
        assert(rc == ELPIS_ECSG_K1_BUSY);
        return 1;
    }
    elpis_ecsg_k1_fms_info info;
    assert(elpis_ecsg_k1_fms_inspect(s->fms, s->id, &info) == ELPIS_ECSG_K1_OK);
    if (leases) *leases = info.lease_count;
    return info.transaction_open;
}

/* A resident state's transaction and WRITE pin are gone, and another transaction begins and aborts cleanly. */
static void released(const subject *s) {
    uint32_t leases = 99;
    assert(!txn_open(s, &leases));
    if (s->resident) {
        assert(leases == 0);
        uint64_t token = 0;
        assert(elpis_ecsg_k1_fms_txn_begin(s->fms, s->id, &token) == ELPIS_ECSG_K1_OK);
        assert(txn_open(s, &leases) && leases == 1);   /* the transaction's WRITE pin */
        assert(elpis_ecsg_k1_fms_txn_abort(s->fms, s->id, token) == ELPIS_ECSG_K1_OK);
        assert(!txn_open(s, &leases) && leases == 0);
    }
}

static elpis_runtime *runtime_at(const char *dir, int fresh) {
    if (fresh) {
        char cmd[3300];
        assert(snprintf(cmd, sizeof(cmd), "rm -rf '%s'", dir) < (int)sizeof(cmd));
        assert(system(cmd) == 0);
    }
    elpis_runtime *rt = NULL;
    elpis_continuity_snapshot snap;
    OK(elpis_runtime_create((const uint8_t *)dir, strlen(dir), &rt));
    OK(elpis_runtime_open(rt, &snap));
    return rt;
}

static int begin(elpis_runtime *rt, const subject *s) {
    double s3[S3];
    elpis_runtime_turn_begin_result begun;
    return elpis_runtime_turn_begin(rt, &s->d, X, ROWS * DIM, Y, ROWS, SCHEDULE, 2, 0.002, s3, S3, &begun);
}

static void commit_turn(elpis_runtime *rt, const subject *s) {
    elpis_runtime_turn_commit_result out;
    elpis_continuity_snapshot snap;
    OK(begin(rt, s));
    OK(elpis_runtime_turn_commit(rt, &s->d, &out, &snap));
    assert(out.committed == 1);
}

static uint64_t aborts(elpis_runtime *rt) {
    elpis_runtime_counters c;
    OK(elpis_runtime_read_counters(rt, &c, 0));
    return c.k1_aborts;
}

/* The durable record restart will see (read through a fresh runtime over the same directory). */
static void durable(const char *dir, elpis_continuity_snapshot *out) {
    elpis_runtime *rt = runtime_at(dir, 0);
    OK(elpis_runtime_snapshot(rt, out));
    elpis_runtime_destroy(&rt);
}

static int same_record(const elpis_continuity_snapshot *a, const elpis_continuity_snapshot *b) {
    return a->generation == b->generation && a->anchored == b->anchored &&
           !memcmp(a->k1_state_digest, b->k1_state_digest, 32);
}

enum { CASE_ABORT, CASE_COMMIT, CASE_CLOSE, CASE_DESTROY, CASE_REOPEN, CASE_FAIL_STOP, CASES };
static const char *const NAMES[CASES] = {"abort", "commit", "close", "destroy", "reopen", "fail-stop+close"};

static void lifecycle(int resident, int which) {
    char dir[3200];
    assert(snprintf(dir, sizeof(dir), "%s/continuity-%d-%d", SCRATCH, resident, which) < (int)sizeof(dir));
    subject s = make(resident, (unsigned)(resident * 100 + which));
    elpis_continuity_snapshot snap, before_record, after_record;
    elpis_runtime *rt = runtime_at(dir, 1);
    OK(elpis_runtime_anchor(rt, &s.d, &snap));
    commit_turn(rt, &s);
    OK(elpis_runtime_snapshot(rt, &before_record));
    retained before = observe(&s);
    OK(begin(rt, &s));   /* the schedule ran on the candidate; nothing committed yet */
    uint32_t leases = 0;
    assert(txn_open(&s, &leases) && (!resident || leases == 1));
    assert(aborts(rt) == 0);
    if (resident) {
        /* The adapter's own lifetime guard: an open transaction keeps the retained handle and id live. */
        uint64_t id = s.id;
        assert(elpis_ecsg_k1_fms_close(s.fms, &id) == ELPIS_ECSG_K1_BUSY && id == s.id);
        elpis_ecsg_k1_fms *alive = s.fms;
        assert(elpis_ecsg_k1_fms_destroy(&alive) == ELPIS_ECSG_K1_BUSY && alive == s.fms);
    }

    switch (which) {
    case CASE_ABORT:
        OK(elpis_runtime_turn_abort(rt, &s.d));
        assert(aborts(rt) == 1);
        assert(elpis_runtime_turn_abort(rt, &s.d) == ELPIS_RUNTIME_TURN_NOT_OPEN);
        break;
    case CASE_COMMIT: {
        elpis_runtime_turn_commit_result out;
        OK(elpis_runtime_turn_commit(rt, &s.d, &out, &snap));
        assert(aborts(rt) == 0 && out.committed == 1);
        retained after = observe(&s);
        assert(after.generation == before.generation + 1 && after.epoch == before.epoch + 8);
        assert(!memcmp(out.identity.state_after_digest, after.digest, 32));
        assert(!memcmp(snap.k1_state_digest, after.digest, 32));
        break;
    }
    case CASE_CLOSE:
        elpis_runtime_close(rt);
        assert(aborts(rt) == 1);
        elpis_runtime_close(rt);   /* double close: nothing left to end */
        assert(aborts(rt) == 1);
        break;
    case CASE_DESTROY:
        elpis_runtime_destroy(&rt);
        assert(rt == NULL);
        break;
    case CASE_REOPEN:
        assert(elpis_runtime_open(rt, &snap) == ELPIS_CONTINUITY_OPEN);
        assert(aborts(rt) == 0 && txn_open(&s, NULL));   /* refused, and the turn is kept, not forgotten */
        elpis_runtime_close(rt);
        assert(aborts(rt) == 1);
        OK(elpis_runtime_open(rt, &snap));   /* open -> close -> open */
        break;
    case CASE_FAIL_STOP: {
        elpis_continuity_snapshot authority;
        uint8_t assertion[32], receipt[32];
        memset(assertion, 0x11, 32);
        memset(receipt, 0x22, 32);
        OK(elpis_runtime_evolution_authority(rt, &authority));
        OK(elpis_runtime_evolution_reserve(rt, &authority.evolution, assertion, &snap));
        OK(elpis_runtime_evolution_finalize(rt, receipt, &snap));
        OK(elpis_runtime_snapshot(rt, &before_record));
        /* A reservation against the replaced authority fail-stops the runtime with the turn still open. */
        assert(elpis_runtime_evolution_reserve(rt, &authority.evolution, assertion, &snap) ==
               ELPIS_CONTINUITY_AUTHORITY_MISMATCH);
        assert(elpis_runtime_fault(rt) == ELPIS_CONTINUITY_AUTHORITY_MISMATCH && txn_open(&s, NULL));
        elpis_runtime_close(rt);
        assert(aborts(rt) == 1);
        OK(elpis_runtime_open(rt, &snap));
        assert(elpis_runtime_fault(rt) == 0);
        break;
    }
    default:
        assert(0);
    }

    if (which != CASE_COMMIT) {
        /* Nothing installed, nothing published: W, epoch, H, a (the digest), generation and continuity. */
        assert(same(before, observe(&s)));
        if (rt) {
            elpis_runtime_destroy(&rt);
        }
        durable(dir, &after_record);
        assert(same_record(&before_record, &after_record));
        assert(!memcmp(after_record.k1_state_digest, before.digest, 32));
    }
    released(&s);
    /* The same live state takes the next managed turn, over a runtime resumed from the same continuity. */
    if (rt) {
        elpis_runtime_destroy(&rt);
    }
    rt = runtime_at(dir, 0);
    commit_turn(rt, &s);
    assert(observe(&s).generation == before.generation + (which == CASE_COMMIT ? 2u : 1u));
    elpis_runtime_destroy(&rt);
    released(&s);
    destroy_subject(&s);
    printf("lifecycle %-10s %-16s PASS\n", resident ? "fms-k1" : "k1", NAMES[which]);
}

/* Lifecycle calls with no turn open make no native K1 call. */
static void idle(int resident) {
    char dir[3200];
    assert(snprintf(dir, sizeof(dir), "%s/continuity-idle-%d", SCRATCH, resident) < (int)sizeof(dir));
    subject s = make(resident, (unsigned)(900 + resident));
    elpis_continuity_snapshot snap;
    elpis_runtime *rt = runtime_at(dir, 1);
    elpis_runtime_destroy(&rt);
    OK(elpis_runtime_create((const uint8_t *)dir, strlen(dir), &rt));
    elpis_runtime_close(rt);   /* never opened */
    OK(elpis_runtime_open(rt, &snap));
    OK(elpis_runtime_anchor(rt, &s.d, &snap));
    elpis_runtime_counters c;
    OK(elpis_runtime_read_counters(rt, &c, 1));
    elpis_runtime_close(rt);
    elpis_runtime_close(rt);
    OK(elpis_runtime_open(rt, &snap));
    elpis_runtime_close(rt);
    OK(elpis_runtime_open(rt, &snap));
    OK(elpis_runtime_read_counters(rt, &c, 0));
    assert(c.k1_state_digests == 0 && c.k1_txn_begins == 0 && c.k1_aborts == 0 && c.k1_commits == 0);
    elpis_runtime_destroy(&rt);   /* destroy with no turn */
    elpis_runtime_destroy(&rt);   /* NULL: no-op */
    released(&s);
    destroy_subject(&s);
    printf("lifecycle %-10s %-16s PASS\n", resident ? "fms-k1" : "k1", "no turn");
}

int main(int argc, char **argv) {
    assert(argc == 2 && strlen(argv[1]) < 2048);
    strcpy(SCRATCH, argv[1]);
    char cmd[3200];
    assert(snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", SCRATCH) < (int)sizeof(cmd));
    assert(system(cmd) == 0);
    assert(elpis_runtime_abi_version() == ELPIS_RUNTIME_ABI_V3);
    for (int i = 0; i < DIM * WIDTH; ++i) W0[i] = 0.2 * (double)((i * 7) % 13 - 6) / 13.0;
    for (int i = 0; i < ROWS * DIM; ++i) X[i] = (double)((i * 5) % 11 - 5) / 16.0;
    for (int i = 0; i < ROWS; ++i) Y[i] = (double)(i % 3 - 1) / 8.0;
    for (int resident = 0; resident <= 1; ++resident) {
        for (int which = 0; which < CASES; ++which) lifecycle(resident, which);
        idle(resident);
    }
    puts("runtime lifecycle: PASS");
    return 0;
}
