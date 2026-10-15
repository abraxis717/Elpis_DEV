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
 *      not the live state after 120 more turns), with its I/O on the worker thread only, never this one.
 *
 * Usage: test_runtime_hot_io <scratch-dir>. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include "elpis/ecsg_k1.h"
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

static void count(void) {
    if (on_main)
        atomic_fetch_add(&main_io, 1);
    else
        atomic_fetch_add(&other_io, 1);
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

#define OPEN_WRAPPER(name, ...)                                                                      \
    int name(__VA_ARGS__, int flags, ...) {                                                          \
        mode_t mode = 0;                                                                             \
        if (flags & (O_CREAT | O_TMPFILE)) {                                                         \
            va_list ap;                                                                              \
            va_start(ap, flags);                                                                     \
            mode = (mode_t)va_arg(ap, int);                                                          \
            va_end(ap);                                                                              \
        }                                                                                            \
        if (writes(flags)) count();                                                                  \
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
    take(&main_io);
    take(&other_io);
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
    printf("test_runtime_hot_io: OK\n");
    return 0;
}
