/* The RuntimeCore C ABI over a real native K1 state (production runtime library, real libelpis_ecsg_k1).
 *
 * Usage: test_runtime_abi <scratch-dir>
 * Deterministic: no threads, no sleeps. The K1 function table is built from typed wrappers (no casts). */
#define _POSIX_C_SOURCE 200809L
#include "elpis/ecsg_k1.h"
#include "elpis/runtime.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OK(x) assert((x) == ELPIS_RUNTIME_OK)

enum { DIM = 6, WIDTH = 36, ROWS = 4 };

_Static_assert(sizeof(elpis_runtime_experience) == sizeof(elpis_ecsg_k1_experience), "experience layout");
_Static_assert(sizeof(elpis_runtime_schedule_result) == sizeof(elpis_ecsg_k1_schedule_result), "schedule layout");
_Static_assert(sizeof(elpis_runtime_commit_identity) == sizeof(elpis_ecsg_k1_commit_identity), "identity layout");
_Static_assert(sizeof(elpis_runtime_turn_begin_result) == 48, "begin result ABI size");
_Static_assert(sizeof(elpis_runtime_turn_commit_result) == 120, "commit result ABI size");
_Static_assert(sizeof(elpis_runtime_counters) == 64, "counters ABI size");
_Static_assert(sizeof(elpis_runtime_query_result) == 40, "query result ABI size");

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

static const elpis_runtime_k1_api API = {k1_digest, k1_reserve, k1_begin, k1_schedule, k1_commit, k1_abort,
                                          k1_query};

static elpis_ecsg_k1 *state(double seed) {
    double w[DIM * WIDTH];
    for (int i = 0; i < DIM * WIDTH; ++i) w[i] = seed * (double)((i * 7) % 13 - 6) / 13.0;
    elpis_ecsg_k1 *s = NULL;
    assert(elpis_ecsg_k1_create(DIM, WIDTH, 8, w, &s) == ELPIS_ECSG_K1_OK);
    return s;
}

static elpis_runtime_substrate sub(elpis_ecsg_k1 *s, uint64_t owner, uint64_t dim) {
    elpis_runtime_substrate d = {ELPIS_RUNTIME_SUBSTRATE_K1, 0, s, 0, owner, dim, &API};
    return d;
}

static void digest(elpis_ecsg_k1 *s, uint8_t out[32]) { assert(elpis_ecsg_k1_state_digest(s, out) == ELPIS_ECSG_K1_OK); }

static double X[ROWS * DIM], Y[ROWS];
static const elpis_runtime_experience SCHEDULE[2] = {{2, 3}, {2, 5}};

static int turn(elpis_runtime *rt, const elpis_runtime_substrate *d, elpis_runtime_turn_commit_result *out) {
    double s3[83];
    elpis_runtime_turn_begin_result begun;
    int rc = elpis_runtime_turn_begin(rt, d, X, ROWS * DIM, Y, ROWS, SCHEDULE, 2, 0.002, s3, 83, &begun);
    if (rc != ELPIS_RUNTIME_OK) return rc;
    assert(begun.schedule.experiences_applied == 2 && begun.schedule.epoch_after == begun.schedule.epoch_before + 8);
    elpis_continuity_snapshot snap;
    return elpis_runtime_turn_commit(rt, d, out, &snap);
}

int main(int argc, char **argv) {
    assert(argc == 2);
    for (int i = 0; i < ROWS * DIM; ++i) X[i] = (double)((i * 5) % 11 - 5) / 16.0;
    for (int i = 0; i < ROWS; ++i) Y[i] = (double)(i % 3 - 1) / 8.0;
    assert(elpis_runtime_abi_version() == ELPIS_RUNTIME_ABI_V3);
    assert(elpis_runtime_features(DIM) == elpis_ecsg_k1_features(DIM));
    for (size_t d = 1; d <= 64; ++d) assert(elpis_runtime_features(d) == elpis_ecsg_k1_features(d));
    assert(!strcmp(elpis_runtime_code_name(ELPIS_RUNTIME_SUBSTRATE_SWITCH), "COGNITION_SUBSTRATE_SWITCH"));
    assert(!strcmp(elpis_runtime_code_name(ELPIS_CONTINUITY_UNANCHORED), "CONTINUITY_UNANCHORED"));
    assert(elpis_runtime_code_name(1000) == NULL);

    char dir[4096];
    assert(snprintf(dir, sizeof(dir), "%s/continuity", argv[1]) < (int)sizeof(dir));
    char cmd[4200];
    assert(snprintf(cmd, sizeof(cmd), "rm -rf '%s'", dir) < (int)sizeof(cmd));
    assert(system(cmd) == 0);

    elpis_runtime *rt = NULL;
    OK(elpis_runtime_create((const uint8_t *)dir, strlen(dir), &rt));
    elpis_continuity_snapshot snap;
    assert(elpis_runtime_snapshot(rt, &snap) == ELPIS_RUNTIME_CLOSED);
    OK(elpis_runtime_open(rt, &snap));
    assert(!snap.anchored && elpis_runtime_fault(rt) == 0);

    elpis_ecsg_k1 *a = state(0.2), *b = state(0.2);
    elpis_runtime_substrate da = sub(a, 1, DIM), db = sub(b, 2, DIM);
    uint8_t before[32], now[32];
    digest(a, before);
    elpis_runtime_turn_commit_result committed;

    /* Unanchored: refused before any mutation. */
    assert(turn(rt, &da, &committed) == ELPIS_CONTINUITY_UNANCHORED);
    digest(a, now);
    assert(!memcmp(before, now, 32) && elpis_ecsg_k1_epoch(a) == 0);

    /* Explicit anchor at the retained identity; then a committed turn publishes its new identity. */
    OK(elpis_runtime_anchor(rt, &da, &snap));
    assert(snap.anchored && !memcmp(snap.k1_state_digest, before, 32));
    OK(turn(rt, &da, &committed));
    digest(a, now);
    assert(committed.committed == 1 && !memcmp(committed.identity.state_before_digest, before, 32));
    assert(!memcmp(committed.identity.state_after_digest, now, 32) && elpis_ecsg_k1_epoch(a) == 8);
    OK(elpis_runtime_snapshot(rt, &snap));
    assert(!memcmp(snap.k1_state_digest, now, 32));

    /* Another state is refused without a native call; not a fail-stop. */
    assert(turn(rt, &db, &committed) == ELPIS_RUNTIME_SUBSTRATE_SWITCH && elpis_ecsg_k1_epoch(b) == 0);
    assert(elpis_runtime_fault(rt) == 0);

    /* A declared dimension that is not the state's is refused natively before any input byte is read. */
    elpis_runtime_substrate wrong = sub(a, 1, DIM - 1);
    double s3[83];
    elpis_runtime_turn_begin_result begun;
    assert(elpis_runtime_turn_begin(rt, &wrong, X, ROWS * (DIM - 1), Y, ROWS, SCHEDULE, 2, 0.002, s3,
                                    elpis_runtime_features(DIM - 1), &begun) == ELPIS_RUNTIME_SUBSTRATE_SWITCH);

    /* Begin then abort: nothing installed. */
    digest(a, before);
    OK(elpis_runtime_turn_begin(rt, &da, X, ROWS * DIM, Y, ROWS, SCHEDULE, 2, 0.002, s3, 83, &begun));
    assert(elpis_runtime_turn_begin(rt, &da, X, ROWS * DIM, Y, ROWS, SCHEDULE, 2, 0.002, s3, 83, &begun) ==
           ELPIS_RUNTIME_TURN_OPEN);
    OK(elpis_runtime_turn_abort(rt, &da));
    digest(a, now);
    assert(!memcmp(before, now, 32) && elpis_ecsg_k1_epoch(a) == 8);

    /* Non-finite input: ECS_REFUSED with the K1 status, state unchanged. */
    X[3] = 1.0 / 0.0;
    assert(elpis_runtime_turn_begin(rt, &da, X, ROWS * DIM, Y, ROWS, SCHEDULE, 2, 0.002, s3, 83, &begun) ==
           ELPIS_RUNTIME_ECS_REFUSED);
    assert(begun.k1_status == ELPIS_ECSG_K1_NONFINITE);
    X[3] = 0.25;
    digest(a, now);
    assert(!memcmp(before, now, 32));

    /* QUERY: read-only. The answer is K1's own forward map of the authoritative state, reported against its
     * identity; W, epoch, H, a (the retained-state digest), the generation and continuity are unchanged. */
    {
        double q[2 * DIM], answer[2], direct[2];
        for (int i = 0; i < 2 * DIM; ++i) q[i] = (double)(i % 5 - 2) / 8.0;
        elpis_runtime_query_result queried;
        elpis_continuity_snapshot durable, after;
        OK(elpis_runtime_snapshot(rt, &durable));
        const uint64_t epoch = elpis_ecsg_k1_epoch(a), generation = elpis_ecsg_k1_generation(a);
        OK(elpis_runtime_read_counters(rt, &(elpis_runtime_counters){0}, 1));
        OK(elpis_runtime_query(rt, &da, q, 2 * DIM, answer, 2, &queried));
        assert(elpis_ecsg_k1_forward(a, q, 2, direct) == ELPIS_ECSG_K1_OK);
        assert(!memcmp(answer, direct, sizeof(direct)));
        digest(a, now);
        assert(!memcmp(queried.state_digest, now, 32) && !memcmp(now, before, 32));
        assert(elpis_ecsg_k1_epoch(a) == epoch && elpis_ecsg_k1_generation(a) == generation);
        OK(elpis_runtime_snapshot(rt, &after));
        assert(!memcmp(&durable, &after, sizeof(after)));
        elpis_runtime_counters c;
        OK(elpis_runtime_read_counters(rt, &c, 0));
        assert(c.k1_queries == 1 && c.k1_txn_begins == 0 && c.k1_commits == 0 && c.publications == 0);
        /* A declared dimension that is not the state's own is refused natively before any input is read. */
        elpis_runtime_substrate short_dim = sub(a, 1, DIM - 1);
        assert(elpis_runtime_query(rt, &short_dim, q, 2 * (DIM - 1), answer, 2, &queried) ==
               ELPIS_RUNTIME_SUBSTRATE_SWITCH);
        /* Another state is a switch, refused without a native call. */
        assert(elpis_runtime_query(rt, &db, q, 2 * DIM, answer, 2, &queried) == ELPIS_RUNTIME_SUBSTRATE_SWITCH);
        assert(elpis_runtime_fault(rt) == 0);
    }

    /* Evolution: reserve, finalize. */
    OK(elpis_runtime_evolution_authority(rt, &snap));
    uint8_t assertion[32], receipt[32];
    memset(assertion, 0x11, 32);
    memset(receipt, 0x22, 32);
    elpis_continuity_evolution observed = snap.evolution;
    OK(elpis_runtime_evolution_reserve(rt, &observed, assertion, &snap));
    assert(snap.evolution.pending == 1 && !memcmp(snap.evolution.assertion, assertion, 32));
    OK(elpis_runtime_evolution_finalize(rt, receipt, &snap));
    assert(snap.evolution.pending == 0 && snap.evolution.revision == observed.revision + 1);

    /* Restart: the matching state resumes; the moved state is a mismatch that fail-stops. */
    elpis_runtime_close(rt);
    OK(elpis_runtime_open(rt, &snap));
    OK(turn(rt, &da, &committed));
    elpis_runtime_close(rt);
    OK(elpis_runtime_open(rt, &snap));
    elpis_runtime_counters counters;
    OK(elpis_runtime_read_counters(rt, &counters, 1));
    assert(turn(rt, &db, &committed) == ELPIS_CONTINUITY_STATE_MISMATCH);
    assert(elpis_runtime_fault(rt) == ELPIS_CONTINUITY_STATE_MISMATCH && elpis_ecsg_k1_epoch(b) == 0);
    assert(turn(rt, &da, &committed) == ELPIS_CONTINUITY_STATE_MISMATCH);

    OK(elpis_runtime_read_counters(rt, &counters, 0));   /* the mismatch: one identity read, nothing else */
    assert(counters.k1_state_digests == 1 && counters.k1_txn_begins == 0 && counters.publications == 0);

    elpis_runtime_destroy(&rt);
    assert(rt == NULL);

    /* A declared dimension that is not the state's own is refused natively (INVALID) before any input byte is
     * read: the readout length it implies differs. */
    assert(snprintf(cmd, sizeof(cmd), "rm -rf '%s'2", dir) < (int)sizeof(cmd));
    assert(system(cmd) == 0);
    char dir2[4200];
    assert(snprintf(dir2, sizeof(dir2), "%s2", dir) < (int)sizeof(dir2));
    OK(elpis_runtime_create((const uint8_t *)dir2, strlen(dir2), &rt));
    OK(elpis_runtime_open(rt, &snap));
    elpis_runtime_substrate lying = sub(b, 3, DIM - 1);
    OK(elpis_runtime_anchor(rt, &lying, &snap));
    digest(b, before);
    assert(elpis_runtime_turn_begin(rt, &lying, X, ROWS * (DIM - 1), Y, ROWS, SCHEDULE, 2, 0.002, s3,
                                    elpis_runtime_features(DIM - 1), &begun) == ELPIS_RUNTIME_ECS_REFUSED);
    assert(begun.k1_status == ELPIS_ECSG_K1_INVALID);
    digest(b, now);
    assert(!memcmp(before, now, 32) && elpis_ecsg_k1_epoch(b) == 0);
    elpis_runtime_destroy(&rt);
    assert(elpis_ecsg_k1_destroy(&a) == ELPIS_ECSG_K1_OK && elpis_ecsg_k1_destroy(&b) == ELPIS_ECSG_K1_OK);
    puts("runtime ABI: PASS");
    return 0;
}
