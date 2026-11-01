/* Zero persistent I/O at the C library's boundary, over a complete RuntimeCore ABI v8 session lifecycle.
 *
 * This executable interposes the C library's persistent-I/O entry points (every write, positional write, sync,
 * rename, unlink, truncate, directory creation and every open that can create or write) and its read-side entry
 * points, and counts each call by the thread that made it. RuntimeCore (the production library) then runs a whole
 * lifecycle over a real standalone K1 state: create, open (a session), the operator's sources and events (founding,
 * lifecycle, a version change, the retired persistence-only kinds), the anchor, 120 managed turns and 12 queries, the
 * live identity, an evolution reservation and finalization, a release and a rebind, close, a second session and
 * destroy. RuntimeCore must make zero persistent-I/O calls and zero reads on every thread: it has no persistent-state
 * path at all. A deliberate write to a pipe by the test itself is counted first, so the interposer is shown live.
 *
 * Then an FMS-resident K1 state over a real, RAM-only FMS context (no COLD tier, no filesystem): while bound it is
 * held resident, managed turns and queries under memory pressure perform no read or write, FMS promotes nothing, and
 * a registration past the WARM budget is refused (FMS_E_LIMIT) rather than spilled anywhere.
 *
 * Usage: test_runtime_hot_io. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#define ELPIS_TEST_K1_FMS
#include "elpis/ecsg_k1.h"
#include "elpis/fms_pal_posix.h"
#include "elpis/runtime.h"
#include "k1_tables.h"
#include "session_runtime.h"

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

/* The complete session lifecycle over one standalone K1 state; the test's own calls between are not RuntimeCore's,
 * so every RuntimeCore call is bracketed by the counters. */
static void lifecycle(void) {
    elpis_ecsg_k1 *k1 = NULL;
    assert(elpis_ecsg_k1_create(DIM, WIDTH, 8, W0, &k1) == ELPIS_ECSG_K1_OK);
    const elpis_runtime_substrate d = {ELPIS_RUNTIME_SUBSTRATE_K1, 0, k1, 0, 1, DIM, &K1_API};
    all_io();
    elpis_runtime *rt = NULL;
    elpis_continuity_snapshot snap;
    OK(test_runtime_create(&rt));
    OK(test_runtime_open(rt, &snap)); /* a session and the operator's three sources */
    OK(elpis_runtime_anchor(rt, &d, &snap));
    uint64_t live = 0;
    {
        uint8_t digest[32];
        OK(elpis_runtime_live_identity(rt, &d, digest, &live));
    }
    for (int n = 0; n < TURNS; ++n) {
        turn(rt, &d, &live);
        if (n % 10 == 0) {
            double out[ROWS];
            elpis_runtime_query_result q;
            OK(elpis_runtime_query(rt, &d, X, ROWS * DIM, out, ROWS, &BUDGET, &q));
            assert(q.live_generation == live);
        }
    }
    /* Operator events: a source's lifecycle and version; the retired persistence-only kinds are refused. */
    {
        elpis_runtime_authority_entity row;
        OK(elpis_runtime_authority_entity_at(rt, 2, &row));
        uint8_t e[512];
        memset(e, 0, sizeof e);
        e[80] = ELPIS_ECS_C_ACTIVE;
        e[81] = ELPIS_ECS_C_DORMANT;
        OK(test_runtime_operator(rt, 3, row.id, e));
        memset(e, 0, sizeof e);
        test_runtime_put_u64(e + 80, 1);
        test_runtime_put_u64(e + 88, 2);
        test_runtime_put_u32(e + 96, 8);
        memset(e + 104, 9, 32);
        OK(test_runtime_operator(rt, 4, row.id, e));
        static const uint8_t none[32] = {0};
        for (uint16_t kind = 10; kind <= 13; ++kind) { /* the retired persistence-only kinds */
            memset(e, 0, sizeof e);
            assert(test_runtime_operator(rt, kind, none, e) == ELPIS_RUNTIME_RETIRED);
        }
    }
    /* Evolution, in the session. */
    {
        uint8_t assertion[32], receipt[32];
        memset(assertion, 0x11, 32);
        memset(receipt, 0x22, 32);
        OK(elpis_runtime_evolution_authority(rt, &snap));
        OK(elpis_runtime_evolution_reserve(rt, &snap.evolution, assertion, &snap));
        OK(elpis_runtime_evolution_finalize(rt, receipt, &snap));
    }
    /* Release and rebind (the live lineage), status and counters, close, a second session, destroy. */
    OK(elpis_runtime_release(rt, &d));
    turn(rt, &d, &live);
    elpis_runtime_authority status;
    OK(elpis_runtime_authority_status(rt, &status));
    elpis_runtime_counters counters;
    OK(elpis_runtime_read_counters(rt, &counters, 0));
    assert(counters.k1_commits == TURNS + 1 && counters.k1_queries == 12);
    elpis_runtime_close(rt);
    OK(test_runtime_open(rt, &snap));
    OK(elpis_runtime_anchor(rt, &d, &snap));
    turn(rt, &d, &live);
    elpis_runtime_destroy(&rt);
    const long writes_main = take(&main_io), writes_other = take(&other_io);
    const long reads_main = take(&main_reads), reads_other = take(&other_reads);
    printf("a complete session lifecycle (%d turns, 12 queries, operator events, evolution, two sessions): "
           "%ld+%ld persistent-I/O calls, %ld+%ld reads (this thread + others)\n",
           TURNS + 2, writes_main, writes_other, reads_main, reads_other);
    assert(writes_main == 0 && writes_other == 0 && reads_main == 0 && reads_other == 0);
    assert(elpis_ecsg_k1_destroy(&k1) == ELPIS_ECSG_K1_OK);
}

/* An FMS-resident K1 state over a real, RAM-only FMS context. */
static void fms_resident(void) {
    const uint64_t image = elpis_ecsg_k1_image_bytes(DIM, WIDTH);
    fms_config c;
    memset(&c, 0, sizeof c);
    c.tier_budget[FMS_WARM] = c.domain_ceiling[FMS_DOM_RAM] = 2u * image + image / 2u; /* two images fit */
    c.high_wm = 0.9f;
    c.low_wm = 0.7f;
    c.max_objects = 8;
    c.hot_absent_policy = c.cold_absent_policy = FMS_REJECT;
    fms_pal *pal = fms_pal_posix_create_ram_only();
    assert(pal);
    fms_ctx *ctx = fms_create(&c, pal);
    assert(ctx);
    elpis_ecsg_k1_fms *fr = NULL;
    assert(elpis_ecsg_k1_fms_create(ctx, 8, &fr) == ELPIS_ECSG_K1_OK);
    uint64_t a = 0, b = 0, refused = 0;
    uint8_t key[32] = {1};
    assert(elpis_ecsg_k1_fms_register(fr, key, DIM, WIDTH, 8, W0, &a) == ELPIS_ECSG_K1_OK);
    const elpis_runtime_substrate fd = {ELPIS_RUNTIME_SUBSTRATE_K1_FMS, 0, fr, a, 2, DIM, &FMS_API};
    elpis_runtime *rt = NULL;
    elpis_continuity_snapshot snap;
    OK(test_runtime_create(&rt));
    OK(test_runtime_open(rt, &snap));
    OK(elpis_runtime_anchor(rt, &fd, &snap)); /* binds it: held resident from here on */
    uint64_t live = 0;
    {
        uint8_t digest[32];
        OK(elpis_runtime_live_identity(rt, &fd, digest, &live));
    }
    /* Memory pressure while bound: a second state fits; a third is refused (FMS_E_LIMIT). With no COLD tier there is
     * nothing to demote to: nothing is spilled, and the bound state stays WARM. */
    key[0] = 2;
    assert(elpis_ecsg_k1_fms_register(fr, key, DIM, WIDTH, 8, W0, &b) == ELPIS_ECSG_K1_OK);
    key[0] = 3;
    assert(elpis_ecsg_k1_fms_register(fr, key, DIM, WIDTH, 8, W0, &refused) ==
           ELPIS_ECSG_K1_FMS_ERROR_BASE + FMS_E_LIMIT);
    assert(refused == 0u && info_of(fr, a).tier == FMS_WARM && info_of(fr, b).tier == FMS_WARM);
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
    const long io = all_io();
    const fms_stats s1 = residency(fr);
    printf("40 held FMS-resident turns (RAM-only FMS): %ld read or write calls; %llu promotions, %llu demotions\n", io,
           (unsigned long long)(s1.promotions - s0.promotions), (unsigned long long)(s1.demotions - s0.demotions));
    assert(io == 0 && s1.promotions == s0.promotions && s1.demotions == s0.demotions && s1.cold_reads == 0);
    assert(s1.tier_bytes[FMS_COLD] == 0);
    elpis_runtime_destroy(&rt); /* unbinds: the residency hold is dropped */
    assert(info_of(fr, a).lease_count == 0);
    assert(elpis_ecsg_k1_fms_close(fr, &a) == ELPIS_ECSG_K1_OK);
    assert(elpis_ecsg_k1_fms_close(fr, &b) == ELPIS_ECSG_K1_OK);
    assert(elpis_ecsg_k1_fms_destroy(&fr) == ELPIS_ECSG_K1_OK);
}

int main(void) {
    on_main = 1;
    /* The lifecycle test's fixture (test_runtime_lifecycle.c). */
    for (int i = 0; i < DIM * WIDTH; ++i) W0[i] = 0.2 * (double)((i * 7) % 13 - 6) / 13.0;
    for (int i = 0; i < ROWS * DIM; ++i) X[i] = (double)((i * 5) % 11 - 5) / 16.0;
    for (int i = 0; i < ROWS; ++i) Y[i] = (double)(i % 3 - 1) / 8.0;
    OK(elpis_runtime_fuel_ceiling(&BUDGET));

    /* The interposer is live: the test's own write to a pipe (not storage) is counted on this thread. */
    int fds[2];
    assert(pipe(fds) == 0);
    all_io();
    assert(write(fds[1], "x", 1) == 1);
    char c = 0;
    assert(read(fds[0], &c, 1) == 1 && c == 'x');
    assert(take(&main_io) == 1 && take(&main_reads) == 1);
    close(fds[0]);
    close(fds[1]);

    lifecycle();
    fms_resident();
    printf("test_runtime_hot_io (ABI v8): OK\n");
    return 0;
}
