/* RuntimeCore over the ECS composite retained state (docs/ECS_COMPOSITE_STATE.md, "Authority"), through the testing
 * runtime library (crash injection) and the real libelpis_ecs_composite:
 *
 *   composite anchor (continuity v3 record, envelope durable first); managed turns that learn K1 and OBSERVE the
 *   memory, committed through checkpoint -> native commit -> publication; recoverable observe refusals; the
 *   composite refuses every unmanaged mutation while bound; a K1 state never stands for the composite lineage;
 *   process death after the checkpoint and after the native commit, then recovery from continuity + the slots with
 *   the memory transition applied exactly once; the explicit legacy import of a K1 lineage.
 *
 * Usage: test_runtime_composite <scratch-dir>. Deterministic: no threads, no sleeps. */
#define _POSIX_C_SOURCE 200809L
#include "elpis/ecs_composite.h"
#include "elpis/ecsg_k1.h"
#include "elpis/runtime.h"
#include "k1_tables.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The testing library's crash hook (not in the production header). 1: after the candidate checkpoint; 2: after the
 * native commit, before publication. */
int elpis_runtime_testing_checkpoint_crash(elpis_runtime *runtime, uint32_t point);

#define OK(x) assert((x) == ELPIS_RUNTIME_OK)
enum { DIM = 6, WIDTH = 36, ROWS = 4, F = 83 };
_Static_assert(sizeof(elpis_runtime_observe_result) == 40, "observe result ABI size");

/* ---- the composite's function table (typed wrappers, no function-pointer casts) --------------------------- */

static int c_digest(void *s, uint8_t out[32]) { return elpis_ecs_composite_state_digest(s, out); }
static int c_shape(void *s, size_t *dim, size_t *width) { return elpis_ecs_composite_shape(s, dim, width); }
static int c_query(void *s, size_t dim, const double *x, size_t rows, double *out, uint8_t digest[32]) {
    return elpis_ecs_composite_query_identity(s, dim, x, rows, out, digest);
}
static int c_claim(void *s, uint64_t lease) { return elpis_ecs_composite_lease_claim(s, lease); }
static int c_release(void *s, uint64_t lease) { return elpis_ecs_composite_lease_release(s, lease); }
static int c_reserve(void *s, uint64_t lease, size_t rows) { return elpis_ecs_composite_leased_reserve(s, lease, rows); }
static int c_begin(void *s, uint64_t lease, uint8_t source[32], uint64_t *token) {
    return elpis_ecs_composite_leased_txn_begin(s, lease, source, token);
}
static int c_schedule(void *s, uint64_t lease, uint64_t token, const double *x, const double *y, size_t rows,
                      const elpis_runtime_experience *schedule, size_t n, double rate, double *s3, size_t s3n,
                      elpis_runtime_schedule_result *result) {
    return elpis_ecs_composite_leased_txn_run_schedule(s, lease, token, x, y, rows,
                                                       (const elpis_ecsg_k1_experience *)schedule, n, rate, s3, s3n,
                                                       (elpis_ecsg_k1_schedule_result *)result);
}
static int c_commit(void *s, uint64_t lease, uint64_t token, elpis_runtime_commit_identity *identity) {
    return elpis_ecs_composite_leased_txn_commit_identity(s, lease, token, (elpis_ecsg_k1_commit_identity *)identity);
}
static int c_abort(void *s, uint64_t token) { return elpis_ecs_composite_txn_abort(s, token); }
static int c_snapshot(void *s, uint8_t *out, size_t size) { return elpis_ecs_composite_snapshot_write(s, out, size); }
static int c_txn_snapshot(void *s, uint64_t token, uint8_t *out, size_t size) {
    return elpis_ecs_composite_txn_snapshot_write(s, token, out, size);
}
static int c_observe(void *s, uint64_t lease, uint64_t token, const void *observation, uint8_t key[32], int *ms) {
    return elpis_ecs_composite_leased_txn_observe(s, lease, token, observation, (elpis_ecs_mem_digest *)key, ms);
}
static const elpis_runtime_composite_api COMPOSITE_API = {
    {c_digest, c_shape, c_query, c_claim, c_release, c_reserve, c_begin, c_schedule, c_commit, c_abort, c_snapshot,
     c_txn_snapshot},
    c_observe};

/* ---- fixtures ------------------------------------------------------------------------------------------------ */

static elpis_runtime_budget BUDGET;
static double W0[DIM * WIDTH], X[ROWS * DIM], Y[ROWS];
static const elpis_runtime_experience SCHEDULE[2] = {{2, 3}, {2, 5}};
static char cmd[9000];

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
static void reset_dir(const char *a, const char *b) {
    assert(snprintf(cmd, sizeof(cmd), "rm -rf '%s' '%s'", a, b) < (int)sizeof(cmd));
    assert(system(cmd) == 0);
}

/* begin (K1 schedule) -> observe each -> commit. Returns the commit status; keys of the observations. */
static int turn(elpis_runtime *rt, const elpis_runtime_substrate *d, const elpis_ecs_mem_observation *o, int n,
                uint8_t keys[][32], elpis_runtime_turn_commit_result *out) {
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
    elpis_continuity_snapshot snap;
    return elpis_runtime_turn_commit(rt, d, out, &snap);
}

int main(int argc, char **argv) {
    assert(argc == 2);
    for (int i = 0; i < DIM * WIDTH; ++i) W0[i] = 0.2 * (double)((i * 7) % 13 - 6) / 13.0;
    for (int i = 0; i < ROWS * DIM; ++i) X[i] = (double)((i * 5) % 11 - 5) / 16.0;
    for (int i = 0; i < ROWS; ++i) Y[i] = (double)(i % 3 - 1) / 8.0;
    OK(elpis_runtime_fuel_ceiling(&BUDGET));
    const elpis_ecs_composite_config cfg = config();
    const size_t envelope = elpis_ecs_composite_envelope_bytes(&cfg);
    assert(envelope > elpis_ecsg_k1_envelope_bytes(DIM, WIDTH));

    char dir[4096], ck[4200];
    assert(snprintf(dir, sizeof(dir), "%s/continuity", argv[1]) < (int)sizeof(dir));
    assert(snprintf(ck, sizeof(ck), "%s-composite-checkpoint", dir) < (int)sizeof(ck));
    reset_dir(dir, ck);
    OK(elpis_runtime_checkpoint_provision((const uint8_t *)ck, strlen(ck), envelope));

    /* ---- anchor, managed turns, refusals ------------------------------------------------------------------ */
    elpis_runtime *rt = NULL;
    elpis_continuity_snapshot snap;
    OK(elpis_runtime_create((const uint8_t *)dir, strlen(dir), &rt));
    OK(elpis_runtime_open(rt, &snap));
    OK(elpis_runtime_checkpoint_attach(rt, (const uint8_t *)ck, strlen(ck)));
    elpis_ecs_composite *x = fresh();
    elpis_runtime_substrate d = csub(x, 1);
    uint8_t id[32];
    identity(x, id);
    OK(elpis_runtime_anchor(rt, &d, &snap));
    assert(snap.anchored && snap.state_format == ELPIS_CONTINUITY_FORMAT_COMPOSITE && !memcmp(snap.k1_state_digest, id, 32));
    elpis_runtime_recovery rec;
    OK(elpis_runtime_checkpoint_recover(rt, &rec, NULL, 0));
    assert(rec.disposition == ELPIS_RUNTIME_RECOVERY_RESUMABLE && rec.envelope_bytes == envelope); /* durable first */

    /* While bound, every unmanaged composite mutation is refused. */
    uint8_t src[32];
    uint64_t tok = 0;
    assert(elpis_ecs_composite_leased_txn_begin(x, 0, src, &tok) == ELPIS_ECS_COMPOSITE_LEASED);
    /* Observe without a turn. */
    elpis_ecs_mem_observation o1[2] = {obs(1, 2, 10), obs(3, 4, 11)};
    elpis_runtime_observe_result orr;
    assert(elpis_runtime_turn_observe(rt, &d, &o1[0], &orr) == ELPIS_RUNTIME_TURN_NOT_OPEN);
    elpis_runtime_turn_commit_result committed;
    uint8_t keys[2][32];
    OK(turn(rt, &d, o1, 2, keys, &committed));
    identity(x, id);
    OK(elpis_runtime_snapshot(rt, &snap));
    assert(!memcmp(committed.identity.state_after_digest, id, 32) && !memcmp(snap.k1_state_digest, id, 32));
    assert(snap.state_format == ELPIS_CONTINUITY_FORMAT_COMPOSITE);
    assert(asserts_of(x, keys[0]) == 1 && asserts_of(x, keys[1]) == 1);
    /* QUERY is bound to the composite identity. */
    double answer[ROWS];
    elpis_runtime_query_result qr;
    OK(elpis_runtime_query(rt, &d, X, ROWS * DIM, answer, ROWS, &BUDGET, &qr));
    assert(!memcmp(qr.state_digest, id, 32));
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
        OK(elpis_runtime_turn_commit(rt, &d, &committed, &snap));
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
    elpis_runtime_destroy(&rt);

    /* ---- a K1 state never stands for the composite lineage ---------------------------------------------------- */
    {
        elpis_ecsg_k1 *k = NULL;
        assert(elpis_ecsg_k1_create(DIM, WIDTH, 8, W0, &k) == ELPIS_ECSG_K1_OK);
        elpis_runtime_substrate dk = ksub(k, 2);
        OK(elpis_runtime_create((const uint8_t *)dir, strlen(dir), &rt));
        OK(elpis_runtime_open(rt, &snap));
        double s3[F];
        elpis_runtime_turn_begin_result begun;
        assert(elpis_runtime_turn_begin(rt, &dk, X, ROWS * DIM, Y, ROWS, SCHEDULE, 2, 0.002, &BUDGET, s3, F, &begun) ==
               ELPIS_CONTINUITY_STATE_MISMATCH);
        assert(elpis_runtime_fault(rt) == ELPIS_CONTINUITY_STATE_MISMATCH);
        elpis_runtime_destroy(&rt);
        assert(elpis_ecsg_k1_destroy(&k) == ELPIS_ECSG_K1_OK);
    }

    /* ---- process death after the checkpoint, and after the native commit ---------------------------------------- */
    uint8_t *got = malloc(envelope);
    assert(got);
    for (uint32_t point = 1; point <= 2; ++point) {
        OK(elpis_runtime_create((const uint8_t *)dir, strlen(dir), &rt));
        OK(elpis_runtime_open(rt, &snap));
        OK(elpis_runtime_checkpoint_attach(rt, (const uint8_t *)ck, strlen(ck)));
        uint8_t authorized[32];
        memcpy(authorized, snap.k1_state_digest, 32);
        OK(elpis_runtime_testing_checkpoint_crash(rt, point));
        elpis_ecs_mem_observation oc = obs(40 + point, 50 + point, 100 + point);
        uint8_t ck_key[1][32];
        const int rc = turn(rt, &d, &oc, 1, ck_key, &committed);
        assert(rc == ELPIS_CONTINUITY_TESTING_PROCESS_DEATH);
        assert(committed.committed == (point == 2));
        elpis_runtime_destroy(&rt); /* the process ends: only continuity and the checkpoint slots remain */
        elpis_ecs_composite_destroy(x);

        OK(elpis_runtime_create((const uint8_t *)dir, strlen(dir), &rt));
        OK(elpis_runtime_open(rt, &snap));
        assert(!memcmp(snap.k1_state_digest, authorized, 32)); /* never published */
        OK(elpis_runtime_checkpoint_attach(rt, (const uint8_t *)ck, strlen(ck)));
        OK(elpis_runtime_checkpoint_recover(rt, &rec, NULL, 0));
        assert(rec.disposition == ELPIS_RUNTIME_RECOVERY_CANDIDATE_UNRESOLVED && rec.authorized_present == 1);
        if (point == 1) {
            OK(elpis_runtime_checkpoint_discard(rt, rec.candidate)); /* the commit never happened */
        } else {
            OK(elpis_runtime_checkpoint_adopt(rt, rec.candidate, &snap)); /* operator adopts the committed state */
            assert(!memcmp(snap.k1_state_digest, rec.candidate, 32));
            assert(snap.state_format == ELPIS_CONTINUITY_FORMAT_COMPOSITE);
        }
        OK(elpis_runtime_checkpoint_recover(rt, &rec, got, envelope));
        assert(rec.disposition == ELPIS_RUNTIME_RECOVERY_RESUMABLE && rec.envelope_bytes == envelope);
        x = NULL;
        assert(elpis_ecs_composite_restore(&cfg, got, envelope, &x) == ELPIS_ECS_COMPOSITE_OK);
        identity(x, id);
        OK(elpis_runtime_snapshot(rt, &snap));
        assert(!memcmp(id, snap.k1_state_digest, 32));
        /* The memory transition of the interrupted turn is present exactly when it became authoritative. */
        assert(asserts_of(x, ck_key[0]) == (point == 2 ? 1u : 0u));
        d = csub(x, 10 + point);
        uint8_t again[1][32];
        if (point == 1) {
            OK(turn(rt, &d, &oc, 1, again, &committed)); /* the retried turn applies it once */
            assert(asserts_of(x, again[0]) == 1);
        } else {
            OK(turn(rt, &d, NULL, 0, NULL, &committed)); /* the lineage continues from the adopted state */
            assert(asserts_of(x, ck_key[0]) == 1);       /* never duplicated */
        }
        elpis_runtime_destroy(&rt);
    }
    elpis_ecs_composite_destroy(x);

    /* ---- legacy import: a K1 lineage becomes a composite lineage, explicitly ---------------------------------- */
    {
        char dir2[4200], kck[4300], cck[4300];
        assert(snprintf(dir2, sizeof(dir2), "%s-legacy", dir) < (int)sizeof(dir2));
        assert(snprintf(kck, sizeof(kck), "%s-k1-checkpoint", dir2) < (int)sizeof(kck));
        assert(snprintf(cck, sizeof(cck), "%s-composite-checkpoint", dir2) < (int)sizeof(cck));
        reset_dir(dir2, kck);
        reset_dir(cck, cck);
        const size_t k1n = elpis_ecsg_k1_envelope_bytes(DIM, WIDTH);
        OK(elpis_runtime_checkpoint_provision((const uint8_t *)kck, strlen(kck), k1n));
        OK(elpis_runtime_checkpoint_provision((const uint8_t *)cck, strlen(cck), envelope));
        elpis_ecsg_k1 *k = NULL;
        assert(elpis_ecsg_k1_create(DIM, WIDTH, 8, W0, &k) == ELPIS_ECSG_K1_OK);
        elpis_runtime_substrate dk = ksub(k, 20);
        OK(elpis_runtime_create((const uint8_t *)dir2, strlen(dir2), &rt));
        OK(elpis_runtime_open(rt, &snap));
        OK(elpis_runtime_checkpoint_attach(rt, (const uint8_t *)kck, strlen(kck)));
        OK(elpis_runtime_anchor(rt, &dk, &snap));
        assert(snap.state_format == ELPIS_CONTINUITY_FORMAT_K1);
        OK(turn(rt, &dk, NULL, 0, NULL, &committed));
        uint8_t *k1e = malloc(k1n);
        assert(k1e && elpis_ecsg_k1_snapshot_write(k, k1e, k1n) == ELPIS_ECSG_K1_OK);
        OK(elpis_runtime_release(rt, &dk));
        elpis_ecs_composite *imp = NULL, *other = fresh(); /* other: the pre-learning K1 state */
        assert(elpis_ecs_composite_import_k1(&cfg, k1e, k1n, &imp) == ELPIS_ECS_COMPOSITE_OK);
        elpis_runtime_substrate di = csub(imp, 21), dother = csub(other, 22);
        /* Without a composite-sized checkpoint store attached: refused (the import must be durable first). */
        assert(elpis_runtime_import_composite(rt, &di, &snap) == ELPIS_RUNTIME_ECS_REFUSED);
        OK(elpis_runtime_checkpoint_attach(rt, (const uint8_t *)cck, strlen(cck)));
        assert(elpis_runtime_import_composite(rt, &dother, &snap) == ELPIS_CONTINUITY_STATE_MISMATCH); /* other K1 */
        /* A composite of the right K1 state but a non-empty memory is not an import either. */
        {
            elpis_ecs_composite *full = NULL;
            assert(elpis_ecs_composite_import_k1(&cfg, k1e, k1n, &full) == ELPIS_ECS_COMPOSITE_OK);
            assert(elpis_ecs_composite_leased_txn_begin(full, 0, src, &tok) == ELPIS_ECS_COMPOSITE_OK);
            elpis_ecs_mem_observation om = obs(9, 9, 1);
            assert(elpis_ecs_composite_leased_txn_observe(full, 0, tok, &om, NULL, NULL) == ELPIS_ECS_COMPOSITE_OK);
            elpis_ecs_composite_commit cm;
            assert(elpis_ecs_composite_leased_txn_commit(full, 0, tok, &cm) == ELPIS_ECS_COMPOSITE_OK);
            elpis_runtime_substrate df = csub(full, 23);
            assert(elpis_runtime_import_composite(rt, &df, &snap) == ELPIS_CONTINUITY_STATE_MISMATCH);
            elpis_ecs_composite_destroy(full);
        }
        OK(elpis_runtime_snapshot(rt, &snap));
        assert(snap.state_format == ELPIS_CONTINUITY_FORMAT_K1); /* nothing published by the refusals */
        OK(elpis_runtime_import_composite(rt, &di, &snap));
        identity(imp, id);
        assert(snap.state_format == ELPIS_CONTINUITY_FORMAT_COMPOSITE && !memcmp(snap.k1_state_digest, id, 32));
        OK(elpis_runtime_checkpoint_recover(rt, &rec, NULL, 0));
        assert(rec.disposition == ELPIS_RUNTIME_RECOVERY_RESUMABLE);
        elpis_ecs_mem_observation oi = obs(11, 12, 1);
        OK(turn(rt, &di, &oi, 1, keys, &committed)); /* the imported lineage learns and observes */
        assert(asserts_of(imp, keys[0]) == 1);
        assert(elpis_runtime_import_composite(rt, &di, &snap) == ELPIS_RUNTIME_TURN_OPEN); /* bound: refused */
        elpis_runtime_destroy(&rt);
        elpis_ecs_composite_destroy(imp);
        elpis_ecs_composite_destroy(other);
        assert(elpis_ecsg_k1_destroy(&k) == ELPIS_ECSG_K1_OK);
        free(k1e);
    }
    free(got);
    puts("runtime composite: PASS");
    return 0;
}
