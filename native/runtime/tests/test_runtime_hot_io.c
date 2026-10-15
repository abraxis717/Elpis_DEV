/* Gate A at the system-call boundary: a managed HOT turn performs no persistent I/O, and never waits for durability.
 *
 * This executable interposes the C library's persistent-I/O entry points (every write, positional write, sync,
 * rename, unlink, truncate, directory creation and every open that can create or write) and counts each call by
 * the thread that made it. RuntimeCore (the testing runtime library: its worker can be held) runs over a real
 * standalone K1 state with a provisioned K1 checkpoint store attached, so a publication on the HOT lane would have
 * real files to write:
 *
 *   1. anchor (explicit, durable: it does write) and one durable publication: the interposer sees the writes, so it
 *      is live, and they are made by the durability worker for the publication;
 *   2. the durability worker is held with a captured request pending; 120 managed turns (begin + commit) and 12
 *      queries run on this thread: zero persistent-I/O calls on ANY thread while they run (the worker cannot have
 *      written either: it is held), and none of them waited (the only durability work is stalled);
 *   3. the worker is released: the pending request resolves ACKNOWLEDGED (the capture taken when it was requested,
 *      not the live state after 120 more turns), with its I/O on the worker thread only, never this one;
 *   4. an FMS-resident K1 state with a real cold store (docs/RUNTIME_CORE.md, "HOT residency"): while bound it is
 *      held resident, so managed turns and queries under memory pressure perform no read or write at all and FMS
 *      promotes nothing; unbound and demoted, a HOT operation is refused RUNTIME_NOT_RESIDENT with no read, write or
 *      promotion; only the explicit materialization reads the cold store.
 *
 * Read-side calls (read, positional reads, read-only opens) are counted separately so a cold-store read would show.
 *
 * Usage: test_runtime_hot_io <scratch-dir>. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#define ELPIS_TEST_K1_FMS
#include "elpis/ecsg_k1.h"
#include "elpis/fms_pal_posix.h"
#include "elpis/runtime.h"
#include "k1_tables.h"

#include <assert.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

/* The testing library's durability controls (not in the production header). */
int elpis_runtime_testing_durable_hold(elpis_runtime *runtime, int hold);

#define OK(x) assert((x) == ELPIS_RUNTIME_OK)
enum { DIM = 6, WIDTH = 36, ROWS = 4, S3 = 83, TURNS = 120 };

/* ---- the interposer -------------------------------------------------------------------------------------------- */

static _Thread_local int on_main;        /* 1 on the test's own (HOT) thread */
static atomic_long main_io, other_io;    /* persistent-I/O calls by this thread / by any other thread */
static atomic_long main_reads, other_reads;   /* read-side calls by this thread / by any other thread */

static void count(void) {
    if (on_main)
        atomic_fetch_add(&main_io, 1);
    else
        atomic_fetch_add(&other_io, 1);
}

static void count_read(void) {
    if (on_main)
        atomic_fetch_add(&main_reads, 1);
    else
        atomic_fetch_add(&other_reads, 1);
}

/* The next definition of a libc entry point (the real one). dlsym returns an object pointer; the union converts it
 * to a function pointer without the conversion ISO C leaves undefined. */
typedef union {
    void *object;
    void (*function)(void);
} symbol;
static void (*next_symbol(const char *name))(void) {
    symbol s;
    s.object = dlsym(RTLD_NEXT, name);
    assert(s.object != NULL);
    return s.function;
}
#define NEXT(name) ((__typeof__(&name))next_symbol(#name))
static int writes(int flags) { return (flags & (O_WRONLY | O_RDWR | O_CREAT | O_TRUNC | O_APPEND)) != 0; }

ssize_t write(int fd, const void *buf, size_t n) { count(); return NEXT(write)(fd, buf, n); }
ssize_t pwrite(int fd, const void *buf, size_t n, off_t off) { count(); return NEXT(pwrite)(fd, buf, n, off); }
ssize_t pwrite64(int fd, const void *buf, size_t n, off64_t off) { count(); return NEXT(pwrite64)(fd, buf, n, off); }
ssize_t writev(int fd, const struct iovec *v, int n) { count(); return NEXT(writev)(fd, v, n); }
ssize_t pwritev(int fd, const struct iovec *v, int n, off_t off) { count(); return NEXT(pwritev)(fd, v, n, off); }
int fsync(int fd) { count(); return NEXT(fsync)(fd); }
int fdatasync(int fd) { count(); return NEXT(fdatasync)(fd); }
int rename(const char *a, const char *b) { count(); return NEXT(rename)(a, b); }
int renameat(int da, const char *a, int db, const char *b) { count(); return NEXT(renameat)(da, a, db, b); }
int unlink(const char *p) { count(); return NEXT(unlink)(p); }
int unlinkat(int d, const char *p, int f) { count(); return NEXT(unlinkat)(d, p, f); }
int ftruncate(int fd, off_t len) { count(); return NEXT(ftruncate)(fd, len); }
int ftruncate64(int fd, off64_t len) { count(); return NEXT(ftruncate64)(fd, len); }
int mkdir(const char *p, mode_t m) { count(); return NEXT(mkdir)(p, m); }
ssize_t read(int fd, void *buf, size_t n) { count_read(); return NEXT(read)(fd, buf, n); }
ssize_t pread(int fd, void *buf, size_t n, off_t off) { count_read(); return NEXT(pread)(fd, buf, n, off); }
ssize_t pread64(int fd, void *buf, size_t n, off64_t off) { count_read(); return NEXT(pread64)(fd, buf, n, off); }
ssize_t readv(int fd, const struct iovec *v, int n) { count_read(); return NEXT(readv)(fd, v, n); }
ssize_t preadv(int fd, const struct iovec *v, int n, off_t off) { count_read(); return NEXT(preadv)(fd, v, n, off); }

#define OPEN_WRAPPER(name, ...)                                                                      \
    int name(__VA_ARGS__, int flags, ...) {                                                          \
        mode_t mode = 0;                                                                             \
        if (flags & (O_CREAT | O_TMPFILE)) {                                                         \
            va_list ap;                                                                              \
            va_start(ap, flags);                                                                     \
            mode = (mode_t)va_arg(ap, int);                                                          \
            va_end(ap);                                                                              \
        }                                                                                            \
        if (writes(flags)) count(); else count_read();                                               \
        return OPEN_FORWARD_##name;                                                                  \
    }
#define OPEN_FORWARD_open NEXT(open)(path, flags, mode)
#define OPEN_FORWARD_open64 NEXT(open64)(path, flags, mode)
#define OPEN_FORWARD_openat NEXT(openat)(dir, path, flags, mode)
#define OPEN_FORWARD_openat64 NEXT(openat64)(dir, path, flags, mode)
OPEN_WRAPPER(open, const char *path)
OPEN_WRAPPER(open64, const char *path)
OPEN_WRAPPER(openat, int dir, const char *path)
OPEN_WRAPPER(openat64, int dir, const char *path)

/* ---- the turn -------------------------------------------------------------------------------------------------- */

static double W0[DIM * WIDTH], X[ROWS * DIM], Y[ROWS];
static const elpis_runtime_experience SCHEDULE[2] = {{2, 3}, {2, 5}};
static elpis_runtime_budget BUDGET;

static void turn(elpis_runtime *rt, const elpis_runtime_substrate *d, uint64_t *live) {
    double s3[S3];
    elpis_runtime_turn_begin_result begun;
    elpis_runtime_volatile_commit committed;
    const int rc = elpis_runtime_turn_begin(rt, d, X, ROWS * DIM, Y, ROWS, SCHEDULE, 2, 0.002, &BUDGET, s3, S3, &begun);
    if (rc != ELPIS_RUNTIME_OK) fprintf(stderr, "turn_begin: %s\n", elpis_runtime_code_name(rc));
    OK(rc);
    OK(elpis_runtime_turn_commit(rt, d, &committed));
    assert(committed.live_generation == *live + 1);
    *live = committed.live_generation;
}

static long take(atomic_long *c) { return atomic_exchange(c, 0); }

static long all_io(void) {
    return take(&main_io) + take(&other_io) + take(&main_reads) + take(&other_reads);
}

static elpis_ecsg_k1_fms_info info_of(elpis_ecsg_k1_fms *r, uint64_t id) {
    elpis_ecsg_k1_fms_info i;
    OK(elpis_ecsg_k1_fms_inspect(r, id, &i));
    return i;
}

static fms_stats residency(elpis_ecsg_k1_fms *r) {
    elpis_ecsg_k1_fms_metrics m;
    OK(elpis_ecsg_k1_fms_stats(r, &m));
    return m.residency;
}

/* 4. HOT residency over a real FMS context with a POSIX cold store. */
static void fms_resident(const char *scratch) {
    char cold[2048], continuity[2048], cmd[4300];
    assert(snprintf(cold, sizeof cold, "%s/cold", scratch) < (int)sizeof cold);
    assert(snprintf(continuity, sizeof continuity, "%s/continuity-fms", scratch) < (int)sizeof continuity);
    assert(snprintf(cmd, sizeof cmd, "rm -rf '%s' '%s' && mkdir -p '%s'", cold, continuity, cold) < (int)sizeof cmd);
    assert(system(cmd) == 0);
    const uint64_t image = elpis_ecsg_k1_image_bytes(DIM, WIDTH);
    fms_config c;
    memset(&c, 0, sizeof c);
    c.tier_budget[FMS_WARM] = c.domain_ceiling[FMS_DOM_RAM] = 2u * image + image / 2u;   /* two images fit */
    c.tier_budget[FMS_COLD] = c.domain_ceiling[FMS_DOM_STORAGE] = 64u * image;
    c.high_wm = 0.9f;
    c.low_wm = 0.7f;
    c.max_objects = 8;
    c.hot_absent_policy = c.cold_absent_policy = FMS_REJECT;
    fms_pal *pal = fms_pal_posix_create(cold);
    assert(pal);
    fms_ctx *ctx = fms_create(&c, pal);
    assert(ctx);
    elpis_ecsg_k1_fms *fr = NULL;
    assert(elpis_ecsg_k1_fms_create(ctx, 8, &fr) == ELPIS_ECSG_K1_OK);
    uint64_t a = 0, others[4] = {0, 0, 0, 0};
    uint8_t key[32] = {1};
    assert(elpis_ecsg_k1_fms_register(fr, key, DIM, WIDTH, 8, W0, &a) == ELPIS_ECSG_K1_OK);
    const elpis_runtime_substrate fd = {ELPIS_RUNTIME_SUBSTRATE_K1_FMS, 0, fr, a, 2, DIM, &FMS_API};
    elpis_runtime *rt = NULL;
    elpis_continuity_snapshot snap;
    OK(elpis_runtime_create((const uint8_t *)continuity, strlen(continuity), &rt));
    OK(elpis_runtime_open(rt, &snap));
    OK(elpis_runtime_anchor(rt, &fd, &snap));   /* binds it: held resident from here on */
    uint64_t live = 0;
    {
        uint8_t digest[32];
        OK(elpis_runtime_live_identity(rt, &fd, digest, &live));
    }
    /* Memory pressure while bound: another state arrives and the pump runs; the bound state is never the victim. */
    key[0] = 2;
    assert(elpis_ecsg_k1_fms_register(fr, key, DIM, WIDTH, 8, W0, &others[0]) == ELPIS_ECSG_K1_OK);
    assert(elpis_ecsg_k1_fms_pump(fr) == ELPIS_ECSG_K1_OK);
    assert(info_of(fr, a).tier == FMS_WARM);
    const fms_stats s0 = residency(fr);
    all_io();
    for (int n = 0; n < 40; ++n) {
        turn(rt, &fd, &live);
        if (n % 10 == 0) {
            double out[ROWS];
            elpis_runtime_query_result q;
            OK(elpis_runtime_query(rt, &fd, X, ROWS * DIM, out, ROWS, &BUDGET, &q));
            assert(q.live_generation == live);
        }
    }
    long io = all_io();
    fms_stats s1 = residency(fr);
    printf("40 held FMS-resident turns: %ld read or write calls; %llu promotions\n", io,
           (unsigned long long)(s1.promotions - s0.promotions));
    assert(io == 0 && s1.promotions == s0.promotions && s1.cold_reads == s0.cold_reads);

    /* Unbound, it may be demoted; then a HOT operation is refused before anything touches it: no I/O at all. */
    OK(elpis_runtime_release(rt, &fd));
    for (int k = 1; k < 4 && info_of(fr, a).tier != FMS_COLD; ++k) {
        key[0] = (uint8_t)(2 + k);
        assert(elpis_ecsg_k1_fms_register(fr, key, DIM, WIDTH, 8, W0, &others[k]) == ELPIS_ECSG_K1_OK);
        assert(elpis_ecsg_k1_fms_pump(fr) == ELPIS_ECSG_K1_OK);
    }
    assert(info_of(fr, a).tier == FMS_COLD);
    const fms_stats s2 = residency(fr);
    all_io();
    {
        double out[ROWS], s3[S3];
        elpis_runtime_query_result q;
        elpis_runtime_turn_begin_result begun;
        assert(elpis_runtime_query(rt, &fd, X, ROWS * DIM, out, ROWS, &BUDGET, &q) == ELPIS_RUNTIME_NOT_RESIDENT);
        assert(elpis_runtime_turn_begin(rt, &fd, X, ROWS * DIM, Y, ROWS, SCHEDULE, 2, 0.002, &BUDGET, s3, S3,
                                        &begun) == ELPIS_RUNTIME_NOT_RESIDENT);
    }
    io = all_io();
    fms_stats s3 = residency(fr);
    printf("HOT operations on a demoted state: refused NOT_RESIDENT with %ld read or write calls, %llu promotions\n",
           io, (unsigned long long)(s3.promotions - s2.promotions));
    assert(io == 0 && s3.promotions == s2.promotions && s3.cold_reads == s2.cold_reads);
    assert(elpis_runtime_fault(rt) == 0 && info_of(fr, a).tier == FMS_COLD && info_of(fr, a).lease_count == 0);

    /* The explicit materialization (a WARM operation) is the only path that reads it back; then HOT work resumes
     * on the live lineage with no I/O. */
    assert(elpis_ecsg_k1_fms_materialize(fr, a) == ELPIS_ECSG_K1_OK);
    const long reads = take(&main_reads);
    all_io();
    assert(reads > 0 && residency(fr).promotions == s2.promotions + 1u);
    turn(rt, &fd, &live);
    assert(all_io() == 0);
    printf("explicit materialization: %ld read calls, then HOT work resumed with none\n", reads);

    elpis_runtime_destroy(&rt);   /* unbinds: the residency hold is dropped */
    assert(info_of(fr, a).lease_count == 0);
    assert(elpis_ecsg_k1_fms_close(fr, &a) == ELPIS_ECSG_K1_OK);
    for (int k = 0; k < 4; ++k) assert(elpis_ecsg_k1_fms_close(fr, &others[k]) == ELPIS_ECSG_K1_OK);
    assert(elpis_ecsg_k1_fms_destroy(&fr) == ELPIS_ECSG_K1_OK);
}

int main(int argc, char **argv) {
    assert(argc == 2);
    on_main = 1;
    /* The lifecycle test's fixture (test_runtime_lifecycle.c). */
    for (int i = 0; i < DIM * WIDTH; ++i) W0[i] = 0.2 * (double)((i * 7) % 13 - 6) / 13.0;
    for (int i = 0; i < ROWS * DIM; ++i) X[i] = (double)((i * 5) % 11 - 5) / 16.0;
    for (int i = 0; i < ROWS; ++i) Y[i] = (double)(i % 3 - 1) / 8.0;
    OK(elpis_runtime_fuel_ceiling(&BUDGET));

    char continuity[2048], checkpoint[2048], cmd[4200];
    assert(snprintf(continuity, sizeof continuity, "%s/continuity", argv[1]) < (int)sizeof continuity);
    assert(snprintf(checkpoint, sizeof checkpoint, "%s/checkpoint", argv[1]) < (int)sizeof checkpoint);
    assert(snprintf(cmd, sizeof cmd, "rm -rf '%s' '%s' && mkdir -p '%s'", continuity, checkpoint, argv[1]) <
           (int)sizeof cmd);
    assert(system(cmd) == 0);

    elpis_ecsg_k1 *k1 = NULL;
    assert(elpis_ecsg_k1_create(DIM, WIDTH, 8, W0, &k1) == ELPIS_ECSG_K1_OK);
    const elpis_runtime_substrate d = {ELPIS_RUNTIME_SUBSTRATE_K1, 0, k1, 0, 1, DIM, &K1_API};
    elpis_runtime *rt = NULL;
    elpis_continuity_snapshot snap;
    OK(elpis_runtime_checkpoint_provision((const uint8_t *)checkpoint, strlen(checkpoint),
                                          elpis_ecsg_k1_envelope_bytes(DIM, WIDTH)));
    OK(elpis_runtime_create((const uint8_t *)continuity, strlen(continuity), &rt));
    OK(elpis_runtime_open(rt, &snap));
    OK(elpis_runtime_checkpoint_attach(rt, (const uint8_t *)checkpoint, strlen(checkpoint)));
    OK(elpis_runtime_anchor(rt, &d, &snap));

    /* 1. The interposer is live: a durable publication writes and syncs, on the worker thread only. */
    uint64_t live = 0;
    {
        uint8_t digest[32];
        OK(elpis_runtime_live_identity(rt, &d, digest, &live));
    }
    turn(rt, &d, &live);
    take(&main_io);
    take(&other_io);
    elpis_runtime_durable_ticket t = durable_publish(rt, &d, ELPIS_RUNTIME_DURABLE_KIND_RECOVERABLE);
    assert(t.status == ELPIS_RUNTIME_DURABLE_ACKNOWLEDGED && t.recoverable == 1);
    const long published_main = take(&main_io), published_worker = take(&other_io);
    printf("durable publication: %ld persistent-I/O calls on the worker, %ld on the caller\n", published_worker,
           published_main);
    assert(published_worker >= 4);   /* slot write + sync, record write + sync at least */
    assert(published_main == 0);

    /* 2. The worker is held with a captured request pending: HOT turns and queries do no persistent I/O. */
    assert(elpis_runtime_testing_durable_hold(rt, 1) == ELPIS_RUNTIME_OK);
    turn(rt, &d, &live);
    elpis_runtime_durable_ticket pending;
    OK(elpis_runtime_recoverable_request(rt, &d, &pending));
    assert(pending.status == ELPIS_RUNTIME_DURABLE_REQUESTED && pending.generation == live);
    uint8_t captured[32];
    {
        uint64_t g = 0;
        OK(elpis_runtime_live_identity(rt, &d, captured, &g));
        assert(g == live);
    }
    elpis_runtime_counters before;
    OK(elpis_runtime_read_counters(rt, &before, 1));
    all_io();
    for (int n = 0; n < TURNS; ++n) {
        turn(rt, &d, &live);
        if (n % 10 == 0) {
            double out[ROWS];
            elpis_runtime_query_result q;
            OK(elpis_runtime_query(rt, &d, X, ROWS * DIM, out, ROWS, &BUDGET, &q));
            assert(q.live_generation == live);
        }
    }
    const long hot_main = take(&main_io), hot_other = take(&other_io);
    elpis_runtime_counters after;
    OK(elpis_runtime_read_counters(rt, &after, 0));
    printf("%d held turns: %ld persistent-I/O calls on the HOT thread, %ld on any other thread\n", TURNS, hot_main,
           hot_other);
    assert(hot_main == 0 && hot_other == 0);
    assert(take(&main_reads) == 0 && take(&other_reads) == 0);   /* and no read either */
    assert(after.k1_commits == TURNS && after.publications == 0 && after.k1_state_digests == 0 &&
           after.k1_identities == 0 && after.k1_captures == 0 && after.durable_requests == 0);
    elpis_runtime_durability status;
    OK(elpis_runtime_durable_status(rt, &status));
    assert(status.pending == 1 && status.live_generation == live);

    /* 3. Released: the pending capture (not the live state) becomes durable, written by the worker only. */
    assert(elpis_runtime_testing_durable_hold(rt, 0) == ELPIS_RUNTIME_OK);
    elpis_runtime_durable_ticket done;
    OK(elpis_runtime_durable_wait(rt, pending.sequence, 30000u, &done));
    assert(done.status == ELPIS_RUNTIME_DURABLE_ACKNOWLEDGED && done.recoverable == 1);
    assert(!memcmp(done.identity, captured, 32));
    OK(elpis_runtime_snapshot(rt, &snap));
    assert(!memcmp(snap.k1_state_digest, captured, 32));
    const long released_main = take(&main_io), released_worker = take(&other_io);
    printf("released publication: %ld persistent-I/O calls on the worker, %ld on the caller\n", released_worker,
           released_main);
    assert(released_main == 0 && released_worker >= 4);

    elpis_runtime_destroy(&rt);
    elpis_ecsg_k1_destroy(&k1);
    fms_resident(argv[1]);
    printf("test_runtime_hot_io: OK\n");
    return 0;
}
