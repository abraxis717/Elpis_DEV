/* FMS ABI v3: COLD is immutable and read/run-only; WARM and HOT are volatile (docs/ELPIS_MISSION.md, Amendment 3).
 *
 * The ten cases of the volatile-residency law, each over real native components (the FMS core, the RAM-only POSIX
 * PAL, the read-only file-asset service, and the test PAL's explicitly emulated device for HOT):
 *
 *   1  frozen COLD -> WARM       a pinned immutable asset is pread through a read-only descriptor, verified page by
 *                                page, and becomes WARM-resident;
 *   2  WARM -> HOT               verified bytes registered WARM are promoted to the (emulated) device;
 *   3  HOT mutation is volatile  a write to the HOT copy changes RAM only: the COLD file is byte-identical;
 *   4  dematerialization         HOT -> WARM demotion, unregistration and page eviction write nothing anywhere;
 *   5  reload                    the asset reloads from COLD with the identical SHA-256 (the mutation is gone);
 *   6  repetition                materialize/evict cycles are stable and bounded (bytes, pages, descriptors);
 *   7  concurrent references     concurrent range leases share pages; a leased page is never evicted;
 *   8  budget exhaustion         beyond the WARM budget a page or object is refused (LIMIT), never spilled;
 *   9  read-only filesystem      (a separate executable: test_fms_volatile_cold_rofs, see main);
 *  10  legacy writers refused    a COLD budget, a storage ceiling, a COLD registration or acquisition are refused;
 *                                the retired writable constructors are not exported; a writable descriptor is
 *                                never admitted as a replica.
 *
 * Write trap: every storage-mutating libc entry point is wrapped at link time (-Wl,--wrap, test_fms_volatile_cold
 * in CMakeLists.txt) for this executable and the static FMS, PAL and file-service code linked into it. While armed
 * (after the fixture file is written, which is the test harness and not Elpis), an attempt counts unless it targets
 * standard output or standard error; the count must stay 0. Independently, the fixture's stamp (size, mtime, ctime,
 * inode) and SHA-256 and the working directory's entries must be unchanged at the end.
 *
 * Usage: test_fms_volatile_cold [--rofs]. With --rofs, case 9 alone: the working directory is bind-mounted read-only
 * in a private mount namespace (a child process; nothing global is remounted), and the frozen COLD -> WARM path must
 * work there while every write attempt fails EROFS. Exit 77 (ctest SKIP) when no private mount namespace can be made
 * on this host: the case is then reported skipped, never passed. */
#define _GNU_SOURCE 1
#include "elpis/fms.h"
#include "elpis/fms_file_service.h"
#include "elpis/fms_pal_posix.h"
#include "elpis/fms_pal_test.h"
#include "elpis/sha256.h"

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); ++fails; } } while (0)

/* ---- the write trap -------------------------------------------------------------------------------------------- */

static atomic_int armed;
static atomic_long attempts;

static void attempt(const char *what, int fd) {
    if (atomic_load(&armed) && fd != STDOUT_FILENO && fd != STDERR_FILENO) {
        atomic_fetch_add(&attempts, 1);
        fprintf(stderr, "write attempt while armed: %s (fd %d)\n", what, fd);
    }
}
static int writes_flags(int flags) {
    return (flags & O_ACCMODE) != O_RDONLY || (flags & (O_CREAT | O_TRUNC | O_APPEND | O_TMPFILE)) != 0;
}

ssize_t __real_write(int, const void *, size_t);
ssize_t __wrap_write(int fd, const void *b, size_t n) { attempt("write", fd); return __real_write(fd, b, n); }
ssize_t __real_pwrite(int, const void *, size_t, off_t);
ssize_t __wrap_pwrite(int fd, const void *b, size_t n, off_t o) { attempt("pwrite", fd); return __real_pwrite(fd, b, n, o); }
ssize_t __real_pwrite64(int, const void *, size_t, off_t);
ssize_t __wrap_pwrite64(int fd, const void *b, size_t n, off_t o) { attempt("pwrite64", fd); return __real_pwrite64(fd, b, n, o); }
ssize_t __real_writev(int, const struct iovec *, int);
ssize_t __wrap_writev(int fd, const struct iovec *v, int n) { attempt("writev", fd); return __real_writev(fd, v, n); }
ssize_t __real_pwritev(int, const struct iovec *, int, off_t);
ssize_t __wrap_pwritev(int fd, const struct iovec *v, int n, off_t o) { attempt("pwritev", fd); return __real_pwritev(fd, v, n, o); }
int __real_ftruncate(int, off_t);
int __wrap_ftruncate(int fd, off_t n) { attempt("ftruncate", fd); return __real_ftruncate(fd, n); }
int __real_truncate(const char *, off_t);
int __wrap_truncate(const char *p, off_t n) { attempt("truncate", -1); return __real_truncate(p, n); }
int __real_fallocate(int, int, off_t, off_t);
int __wrap_fallocate(int fd, int m, off_t o, off_t n) { attempt("fallocate", fd); return __real_fallocate(fd, m, o, n); }
int __real_posix_fallocate(int, off_t, off_t);
int __wrap_posix_fallocate(int fd, off_t o, off_t n) { attempt("posix_fallocate", fd); return __real_posix_fallocate(fd, o, n); }
int __real_fsync(int);
int __wrap_fsync(int fd) { attempt("fsync", fd); return __real_fsync(fd); }
int __real_fdatasync(int);
int __wrap_fdatasync(int fd) { attempt("fdatasync", fd); return __real_fdatasync(fd); }
int __real_rename(const char *, const char *);
int __wrap_rename(const char *a, const char *b) { attempt("rename", -1); return __real_rename(a, b); }
int __real_renameat(int, const char *, int, const char *);
int __wrap_renameat(int d, const char *a, int e, const char *b) { attempt("renameat", -1); return __real_renameat(d, a, e, b); }
int __real_unlink(const char *);
int __wrap_unlink(const char *p) { attempt("unlink", -1); return __real_unlink(p); }
int __real_unlinkat(int, const char *, int);
int __wrap_unlinkat(int d, const char *p, int f) { attempt("unlinkat", -1); return __real_unlinkat(d, p, f); }
int __real_mkdir(const char *, mode_t);
int __wrap_mkdir(const char *p, mode_t m) { attempt("mkdir", -1); return __real_mkdir(p, m); }
int __real_mkdirat(int, const char *, mode_t);
int __wrap_mkdirat(int d, const char *p, mode_t m) { attempt("mkdirat", -1); return __real_mkdirat(d, p, m); }
int __real_rmdir(const char *);
int __wrap_rmdir(const char *p) { attempt("rmdir", -1); return __real_rmdir(p); }
int __real_link(const char *, const char *);
int __wrap_link(const char *a, const char *b) { attempt("link", -1); return __real_link(a, b); }
int __real_symlink(const char *, const char *);
int __wrap_symlink(const char *a, const char *b) { attempt("symlink", -1); return __real_symlink(a, b); }
int __real_open(const char *, int, ...);
int __wrap_open(const char *p, int flags, ...) {
    mode_t mode = 0;
    if (flags & (O_CREAT | O_TMPFILE)) { va_list ap; va_start(ap, flags); mode = (mode_t)va_arg(ap, int); va_end(ap); }
    if (writes_flags(flags)) attempt("open for writing", -1);
    return __real_open(p, flags, mode);
}
int __real_open64(const char *, int, ...);
int __wrap_open64(const char *p, int flags, ...) {
    mode_t mode = 0;
    if (flags & (O_CREAT | O_TMPFILE)) { va_list ap; va_start(ap, flags); mode = (mode_t)va_arg(ap, int); va_end(ap); }
    if (writes_flags(flags)) attempt("open64 for writing", -1);
    return __real_open64(p, flags, mode);
}
int __real_openat(int, const char *, int, ...);
int __wrap_openat(int d, const char *p, int flags, ...) {
    mode_t mode = 0;
    if (flags & (O_CREAT | O_TMPFILE)) { va_list ap; va_start(ap, flags); mode = (mode_t)va_arg(ap, int); va_end(ap); }
    if (writes_flags(flags)) attempt("openat for writing", -1);
    return __real_openat(d, p, flags, mode);
}
int __real_creat(const char *, mode_t);
int __wrap_creat(const char *p, mode_t m) { attempt("creat", -1); return __real_creat(p, m); }
FILE *__real_fopen(const char *, const char *);
FILE *__wrap_fopen(const char *p, const char *m) {
    if (strpbrk(m, "wa+")) attempt("fopen for writing", -1);
    return __real_fopen(p, m);
}

/* ---- the fixture ----------------------------------------------------------------------------------------------- */

#define PAGE 4096u
#define PAGES 16u
#define SIZE (PAGES * PAGE - 333u)
#define ASSET "cold.asset"

static uint8_t content[SIZE];
static uint8_t page_map[PAGES * 32];
static uint8_t cold_sha[32];
static struct stat cold_stamp;

static void fixture(void) {
    for (uint32_t i = 0; i < SIZE; ++i) content[i] = (uint8_t)(i * 167u + (i >> 11) + 3u);
    unlink(ASSET);
    FILE *f = fopen(ASSET, "wb");
    if (!f || fwrite(content, 1, SIZE, f) != SIZE || fclose(f) != 0) { perror("fixture"); exit(2); }
    if (chmod(ASSET, 0444) != 0) { perror("fixture mode"); exit(2); }
    for (uint32_t p = 0; p < PAGES; ++p) {
        uint32_t n = p + 1 == PAGES ? SIZE - p * PAGE : PAGE;
        elpis_file_service_raw_digest(content + p * PAGE, n, page_map + p * 32);
    }
    elpis_sha256(content, SIZE, cold_sha);
    if (stat(ASSET, &cold_stamp) != 0) { perror("fixture stat"); exit(2); }
}

/* The COLD file now: its SHA-256 (from a fresh read-only read) and its stamp, unchanged. */
static void cold_unchanged(const char *when) {
    int fd = open(ASSET, O_RDONLY | O_CLOEXEC);
    CHECK(fd >= 0, "%s: open the COLD asset read-only", when);
    if (fd < 0) return;
    uint8_t *buf = malloc(SIZE + 1), sha[32];
    ssize_t got = pread(fd, buf, SIZE + 1, 0);
    struct stat st;
    CHECK(fstat(fd, &st) == 0, "%s: fstat", when);
    close(fd);
    CHECK(got == (ssize_t)SIZE, "%s: the COLD asset is %zd bytes, not %u", when, got, SIZE);
    elpis_sha256(buf, SIZE, sha);
    free(buf);
    CHECK(!memcmp(sha, cold_sha, 32), "%s: the COLD asset's SHA-256 changed", when);
    CHECK(st.st_ino == cold_stamp.st_ino && st.st_size == cold_stamp.st_size &&
          st.st_mtim.tv_sec == cold_stamp.st_mtim.tv_sec && st.st_mtim.tv_nsec == cold_stamp.st_mtim.tv_nsec &&
          st.st_ctim.tv_sec == cold_stamp.st_ctim.tv_sec && st.st_ctim.tv_nsec == cold_stamp.st_ctim.tv_nsec,
          "%s: the COLD asset's stamp changed", when);
}

static int entries(void) {
    DIR *d = opendir(".");
    int n = 0;
    if (!d) return -1;
    while (readdir(d)) ++n;
    closedir(d);
    return n;
}

static int open_fds(void) {
    DIR *d = opendir("/proc/self/fd");
    int n = 0;
    if (!d) return -1;
    while (readdir(d)) ++n;
    closedir(d);
    return n;
}

static elpis_file_service *service(uint64_t warm_pages, uint32_t max_pages) {
    elpis_file_service_config_v1 c = {ELPIS_FILE_SERVICE_ABI_V1, 0, warm_pages * PAGE, 4 * PAGE, 1 << 20, max_pages,
                                      4, 64, FMS_REJECT};
    elpis_file_service *s = NULL;
    CHECK(elpis_file_service_create(&c, &s) == ELPIS_FILE_OK && s, "file service");
    return s;
}

static elpis_file_status admit_fd(elpis_file_service *s, int fd, uint32_t *asset) {
    struct stat st;
    if (fstat(fd, &st) != 0) return ELPIS_FILE_IO;
    elpis_file_asset_v1 a = {ELPIS_FILE_SERVICE_ABI_V1, PAGE, fd, 0, SIZE, PAGES, {0}, page_map};
    a.stamp = (elpis_file_stamp){(uint64_t)st.st_dev, (uint64_t)st.st_ino, (uint64_t)st.st_size,
                                 (int64_t)st.st_mtim.tv_sec * 1000000000 + st.st_mtim.tv_nsec,
                                 (int64_t)st.st_ctim.tv_sec * 1000000000 + st.st_ctim.tv_nsec};
    return elpis_file_service_admit(s, &a, asset);
}

static uint32_t admit(elpis_file_service *s) {
    int fd = open(ASSET, O_RDONLY | O_CLOEXEC);
    uint32_t asset = UINT32_MAX;
    CHECK(fd >= 0 && admit_fd(s, fd, &asset) == ELPIS_FILE_OK, "admit the COLD asset read-only");
    if (fd >= 0) close(fd);   /* the service owns its own read-only duplicate */
    return asset;
}

/* COLD -> WARM: the whole asset through the service, verified against the fixture. */
static void load_all(elpis_file_service *s, uint32_t asset, uint8_t *out, const char *when) {
    CHECK(elpis_file_service_copy(s, asset, 0, SIZE, out) == ELPIS_FILE_OK, "%s: COLD -> WARM copy", when);
    CHECK(!memcmp(out, content, SIZE), "%s: WARM bytes differ from COLD", when);
}

/* A device-capable volatile context: HOT on the emulated device (discrete domain), WARM in RAM, no COLD tier. The
 * HOT watermarks make the pump demote an unpinned HOT object (HOT -> WARM). */
static fms_ctx *device_context(fms_pal **pal_out, uint64_t warm, uint64_t hot) {
    fms_pal_test_opts o;
    memset(&o, 0, sizeof o);
    o.emulate_device = 1;
    o.device_domain = FMS_DOM_DEVICE;
    o.fence_mode = FMS_FENCE_COMPLETE;
    fms_pal *pal = fms_pal_test_create(&o);
    fms_config c;
    memset(&c, 0, sizeof c);
    c.tier_budget[FMS_HOT] = hot;
    c.tier_budget[FMS_WARM] = warm;
    c.domain_ceiling[FMS_DOM_RAM] = warm;
    c.domain_ceiling[FMS_DOM_DEVICE] = hot;
    c.high_wm = 0.10f;
    c.low_wm = 0.05f;
    c.max_objects = 16;
    c.hot_absent_policy = FMS_REJECT;
    c.cold_absent_policy = FMS_REJECT;
    fms_ctx *ctx = pal ? fms_create(&c, pal) : NULL;
    CHECK(ctx != NULL, "device context");
    if (pal_out) *pal_out = pal;
    return ctx;
}

static void no_storage(fms_ctx *ctx, const char *when) {
    fms_stats st;
    fms_get_stats(ctx, &st);
    CHECK(!st.tier_bytes[FMS_COLD] && !st.domain_bytes[FMS_DOM_STORAGE] && !st.cold_reads && !st.cold_writes &&
          !st.cold_replica_reuse && !st.digest_failures, "%s: storage was charged or touched", when);
}

/* ---- cases 1-5 ------------------------------------------------------------------------------------------------- */

static void cases_1_to_5(void) {
    uint8_t *warm = malloc(SIZE), *again = malloc(SIZE), sha[32];
    elpis_file_service *s = service(PAGES, PAGES);
    uint32_t asset = admit(s);

    /* 1. frozen COLD -> WARM */
    load_all(s, asset, warm, "case 1");
    elpis_file_service_stats_v1 fs;
    elpis_file_service_stats(s, &fs);
    CHECK(fs.resident_pages == PAGES && fs.pread_bytes == SIZE && fs.integrity_failures == 0,
          "case 1: %llu pages resident, %llu bytes read", (unsigned long long)fs.resident_pages,
          (unsigned long long)fs.pread_bytes);

    /* 2. WARM -> HOT */
    fms_pal *pal = NULL;
    fms_ctx *ctx = device_context(&pal, 4ull * SIZE, 2ull * SIZE);
    fms_id id = 0;
    CHECK((int)fms_register(ctx, 0xC01Du, SIZE, FMS_WARM, 0.0f, warm, &id) == FMS_WARM, "case 2: register WARM");
    void *hot = NULL;
    CHECK((int)fms_acquire(ctx, id, FMS_HOT, FMS_WRITE, &hot) == FMS_HOT && hot, "case 2: WARM -> HOT");
    fms_object_info info;
    fms_stats st;
    fms_get_stats(ctx, &st);
    CHECK(fms_query(ctx, id, &info) == FMS_OK && info.tier == FMS_HOT, "case 2: not HOT");
    CHECK(st.promotions == 1 && fms_pal_test_device_bytes(pal) == SIZE, "case 2: %llu promotions, %llu device bytes",
          (unsigned long long)st.promotions, (unsigned long long)fms_pal_test_device_bytes(pal));
    CHECK(hot && !memcmp(hot, content, SIZE), "case 2: HOT bytes differ from COLD");

    /* 3. HOT mutation is volatile */
    if (hot) for (uint32_t i = 0; i < SIZE; i += 97) ((uint8_t *)hot)[i] ^= 0x5Au;
    CHECK(fms_release(ctx, id) == FMS_OK, "case 3: release");
    cold_unchanged("case 3");

    /* 4. dematerialization: HOT -> WARM by the pump (above the HOT watermark), then unregistered; pages evicted */
    CHECK(fms_pump(ctx) == FMS_OK, "case 4: pump");
    fms_get_stats(ctx, &st);
    CHECK(fms_query(ctx, id, &info) == FMS_OK && info.tier == FMS_WARM && st.demotions == 1,
          "case 4: the HOT copy was not dematerialized (tier %u, %llu demotions)", info.tier,
          (unsigned long long)st.demotions);
    CHECK(fms_pal_test_device_bytes(pal) == 0, "case 4: device bytes remain");
    void *w = NULL;
    CHECK((int)fms_acquire(ctx, id, FMS_WARM, FMS_READ, &w) == FMS_WARM && w, "case 4: WARM acquire");
    CHECK(w && memcmp(w, content, SIZE) != 0, "case 4: the mutation did not survive in RAM (it must, until dropped)");
    fms_release(ctx, id);
    CHECK(fms_unregister(ctx, id) == FMS_OK, "case 4: unregister");
    fms_get_stats(ctx, &st);
    CHECK(st.objects == 0 && !st.domain_bytes[FMS_DOM_RAM] && !st.domain_bytes[FMS_DOM_DEVICE],
          "case 4: residency left behind");
    no_storage(ctx, "case 4");
    CHECK(elpis_file_service_evict(s, asset) == ELPIS_FILE_OK, "case 4: evict");
    elpis_file_service_stats(s, &fs);
    CHECK(fs.resident_pages == 0 && fs.resident_bytes == 0, "case 4: pages left resident");
    cold_unchanged("case 4");

    /* 5. reload: identical SHA-256 */
    load_all(s, asset, again, "case 5");
    elpis_sha256(again, SIZE, sha);
    CHECK(!memcmp(sha, cold_sha, 32), "case 5: the reload's SHA-256 differs");
    elpis_file_service_stats(s, &fs);
    CHECK(fs.pread_bytes == 2ull * SIZE, "case 5: the reload did not read COLD again");

    fms_destroy(ctx);
    elpis_file_service_destroy(s);
    free(warm);
    free(again);
}

/* ---- case 6 ---------------------------------------------------------------------------------------------------- */

static void case_6(void) {
    uint8_t *buf = malloc(SIZE), sha[32];
    elpis_file_service *s = service(PAGES, PAGES);
    uint32_t asset = admit(s);
    fms_pal *pal = NULL;
    fms_ctx *ctx = device_context(&pal, 2ull * SIZE, 2ull * SIZE);
    const int fds = open_fds();
    for (int round = 0; round < 64; ++round) {
        load_all(s, asset, buf, "case 6");
        fms_id id = 0;
        void *hot = NULL;
        CHECK((int)fms_register(ctx, 6u, SIZE, FMS_WARM, 0.0f, buf, &id) == FMS_WARM, "case 6: register");
        CHECK((int)fms_acquire(ctx, id, FMS_HOT, FMS_WRITE, &hot) == FMS_HOT, "case 6: HOT");
        if (hot) ((uint8_t *)hot)[round] ^= 0xFFu;
        fms_release(ctx, id);
        fms_pump(ctx);
        CHECK(fms_unregister(ctx, id) == FMS_OK, "case 6: unregister");
        CHECK(elpis_file_service_evict(s, asset) == ELPIS_FILE_OK, "case 6: evict");
    }
    load_all(s, asset, buf, "case 6 (final)");
    elpis_sha256(buf, SIZE, sha);
    CHECK(!memcmp(sha, cold_sha, 32), "case 6: the SHA-256 drifted across cycles");
    elpis_file_service_stats_v1 fs;
    elpis_file_service_stats(s, &fs);
    fms_stats st;
    fms_get_stats(ctx, &st);
    CHECK(fs.resident_high_water <= (uint64_t)PAGES * PAGE && fs.evictions >= 64ull * PAGES,
          "case 6: unbounded residency (high water %llu, evictions %llu)",
          (unsigned long long)fs.resident_high_water, (unsigned long long)fs.evictions);
    CHECK(st.objects == 0 && st.promotions == 64 && st.demotions == 64 && fms_pal_test_device_bytes(pal) == 0,
          "case 6: %llu objects, %llu promotions, %llu demotions", (unsigned long long)st.objects,
          (unsigned long long)st.promotions, (unsigned long long)st.demotions);
    CHECK(open_fds() == fds, "case 6: descriptors grew");
    no_storage(ctx, "case 6");
    fms_destroy(ctx);
    elpis_file_service_destroy(s);
    free(buf);
}

/* ---- case 7 ---------------------------------------------------------------------------------------------------- */

typedef struct { elpis_file_service *s; uint32_t asset; int bad; } reader_arg;

static void *reader(void *p) {
    reader_arg *a = p;
    uint8_t buf[3 * PAGE];
    for (int i = 0; i < 200; ++i) {
        const uint64_t off = (uint64_t)((i * 7) % (PAGES - 3)) * PAGE + 11u;
        uint64_t range = 0;
        elpis_file_status rc = elpis_file_service_acquire(a->s, a->asset, off, sizeof buf, &range);
        if (rc == ELPIS_FILE_BUSY) continue;   /* only an interrupt reports BUSY; none is issued here */
        if (rc != ELPIS_FILE_OK) { a->bad++; continue; }
        if (elpis_file_service_read(a->s, range, buf, sizeof buf) != ELPIS_FILE_OK ||
            memcmp(buf, content + off, sizeof buf) != 0)
            a->bad++;
        if (elpis_file_service_release(a->s, range) != ELPIS_FILE_OK) a->bad++;
    }
    return NULL;
}

static void case_7(void) {
    elpis_file_service *s = service(PAGES, PAGES);
    uint32_t asset = admit(s);
    uint64_t r1 = 0, r2 = 0;
    CHECK(elpis_file_service_acquire(s, asset, 0, 2 * PAGE, &r1) == ELPIS_FILE_OK, "case 7: first reference");
    CHECK(elpis_file_service_acquire(s, asset, PAGE, 2 * PAGE, &r2) == ELPIS_FILE_OK, "case 7: second reference");
    elpis_file_service_stats_v1 fs;
    elpis_file_service_stats(s, &fs);
    CHECK(fs.resident_pages == 3 && fs.reads == 3, "case 7: the shared page was loaded twice (%llu reads)",
          (unsigned long long)fs.reads);
    CHECK(elpis_file_service_evict(s, asset) == ELPIS_FILE_BUSY, "case 7: a leased page was evictable");
    CHECK(elpis_file_service_release(s, r1) == ELPIS_FILE_OK, "case 7: release first");
    CHECK(elpis_file_service_evict(s, asset) == ELPIS_FILE_BUSY, "case 7: the still-referenced page was evictable");
    CHECK(elpis_file_service_release(s, r2) == ELPIS_FILE_OK, "case 7: release second");
    CHECK(elpis_file_service_evict(s, asset) == ELPIS_FILE_OK, "case 7: evict once unreferenced");

    reader_arg args[4];
    pthread_t t[4];
    for (int i = 0; i < 4; ++i) {
        args[i] = (reader_arg){s, asset, 0};
        CHECK(pthread_create(&t[i], NULL, reader, &args[i]) == 0, "case 7: thread");
    }
    for (int i = 0; i < 4; ++i) {
        pthread_join(t[i], NULL);
        CHECK(args[i].bad == 0, "case 7: reader %d saw %d bad reads", i, args[i].bad);
    }
    elpis_file_service_stats(s, &fs);
    CHECK(fs.live_ranges == 0 && fs.pinned_bytes == 0 && fs.integrity_failures == 0, "case 7: leases leaked");
    elpis_file_service_destroy(s);
}

/* ---- case 8 ---------------------------------------------------------------------------------------------------- */

static void case_8(void) {
    /* The file service: two WARM pages. Two leased pages fill it; a third page is refused, never spilled. */
    elpis_file_service *s = service(2, 2);
    uint32_t asset = admit(s);
    uint64_t r = 0, extra = 0;
    CHECK(elpis_file_service_acquire(s, asset, 0, 2 * PAGE, &r) == ELPIS_FILE_OK, "case 8: fill");
    CHECK(elpis_file_service_acquire(s, asset, 5 * PAGE, 16, &extra) == ELPIS_FILE_LIMIT,
          "case 8: a page beyond the WARM budget was not refused");
    elpis_file_service_stats_v1 fs;
    elpis_file_service_stats(s, &fs);
    CHECK(fs.resident_bytes <= 2ull * PAGE && fs.resident_pages == 2, "case 8: the budget was exceeded");
    elpis_file_service_release(s, r);
    CHECK(elpis_file_service_acquire(s, asset, 5 * PAGE, 16, &extra) == ELPIS_FILE_OK,
          "case 8: unleased pages were not reclaimed (LRU) for the next page");
    elpis_file_service_release(s, extra);
    elpis_file_service_destroy(s);

    /* A generic volatile context: one object fits WARM; the second is refused and the first stays put. */
    fms_ctx *ctx = fms_create(&(fms_config){.tier_budget = {0, 2 * PAGE, 0}, .domain_ceiling = {2 * PAGE, 1, 0},
                                            .high_wm = 0.9f, .low_wm = 0.7f, .max_objects = 4,
                                            .hot_absent_policy = FMS_REJECT, .cold_absent_policy = FMS_REJECT},
                              fms_pal_posix_create_ram_only());
    CHECK(ctx != NULL, "case 8: RAM-only context");
    if (!ctx) return;
    fms_id a = 0, b = 0;
    CHECK((int)fms_register(ctx, 8u, 3 * PAGE / 2, FMS_WARM, 0.0f, NULL, &a) == FMS_WARM, "case 8: first object");
    CHECK((int)fms_register(ctx, 8u, 3 * PAGE / 2, FMS_WARM, 0.0f, NULL, &b) == FMS_E_LIMIT && b == 0,
          "case 8: the second object was not refused");
    CHECK(fms_pump(ctx) == FMS_OK, "case 8: pump");
    fms_object_info info;
    CHECK(fms_query(ctx, a, &info) == FMS_OK && info.tier == FMS_WARM && info.state == FMS_ST_RESIDENT,
          "case 8: the first object moved");
    fms_stats st;
    fms_get_stats(ctx, &st);
    CHECK(st.demotions == 0 && st.objects == 1, "case 8: something was displaced");
    no_storage(ctx, "case 8");
    fms_destroy(ctx);
}

/* ---- case 10 --------------------------------------------------------------------------------------------------- */

static void case_10(void) {
    fms_config c;
    memset(&c, 0, sizeof c);
    c.tier_budget[FMS_WARM] = c.domain_ceiling[FMS_DOM_RAM] = 1 << 20;
    c.high_wm = 0.9f;
    c.low_wm = 0.7f;
    c.max_objects = 4;
    c.hot_absent_policy = c.cold_absent_policy = FMS_FOLD_DOWN;   /* even a folding policy cannot reach COLD */
    fms_config cold = c, storage = c;
    cold.tier_budget[FMS_COLD] = 1 << 20;
    storage.domain_ceiling[FMS_DOM_STORAGE] = 1 << 20;
    fms_pal *p1 = fms_pal_posix_create_ram_only(), *p2 = fms_pal_posix_create_ram_only();
    CHECK(fms_create(&cold, p1) == NULL, "case 10: a COLD budget was accepted");
    CHECK(fms_create(&storage, p2) == NULL, "case 10: a storage ceiling was accepted");
    if (p1) p1->destroy(p1->self);   /* fms_create takes ownership only on success */
    if (p2) p2->destroy(p2->self);
    fms_pal *pal = fms_pal_posix_create_ram_only();
    CHECK(pal && pal->abi == 3u && FMS_ABI_VERSION == 3u && !(pal->caps & ~(uint32_t)(FMS_CAP_DEVICE | FMS_CAP_ZERO_COPY)),
          "case 10: the PAL is not ABI v3 or claims a capability beyond HOT");
    fms_ctx *ctx = fms_create(&c, pal);
    CHECK(ctx != NULL, "case 10: the volatile context");
    if (ctx) {
        uint8_t bytes[64] = {1};
        fms_id id = 0, cold_id = 0;
        void *ptr = NULL;
        CHECK((int)fms_register(ctx, 10u, sizeof bytes, FMS_COLD, 0.0f, bytes, &cold_id) == FMS_E_UNSUPPORTED &&
              cold_id == 0, "case 10: a COLD registration was accepted");
        CHECK((int)fms_register(ctx, 10u, sizeof bytes, FMS_WARM, 0.0f, bytes, &id) == FMS_WARM, "case 10: WARM");
        CHECK((int)fms_acquire(ctx, id, FMS_COLD, FMS_READ, &ptr) == FMS_E_UNSUPPORTED,
              "case 10: a COLD acquisition was accepted");
        CHECK((int)fms_acquire(ctx, id, FMS_COLD, FMS_WRITE, &ptr) == FMS_E_UNSUPPORTED,
              "case 10: a COLD write-back acquisition was accepted");
        no_storage(ctx, "case 10");
        fms_destroy(ctx);
    }
    /* The retired writable constructors are not exported by the shipped FMS library; the RAM-only ones are. */
    void *lib = dlopen(ELPIS_FMS_SHARED_LIBRARY, RTLD_NOW | RTLD_LOCAL);
    CHECK(lib != NULL, "case 10: dlopen %s: %s", ELPIS_FMS_SHARED_LIBRARY, dlerror());
    if (lib) {
        static const char *retired[] = {"fms_pal_posix_create", "elpis_fms_create_posix", "fms_pal_posix_token_digest",
                                        "fms_pal_posix_token_bytes", "fms_pal_posix_token_path"};
        for (size_t i = 0; i < sizeof retired / sizeof *retired; ++i)
            CHECK(dlsym(lib, retired[i]) == NULL, "case 10: the retired writer %s is exported", retired[i]);
        CHECK(dlsym(lib, "fms_pal_posix_create_ram_only") != NULL && dlsym(lib, "fms_create") != NULL,
              "case 10: the volatile constructors are missing (the symbol check would prove nothing)");
        dlclose(lib);
    }
    /* A writable descriptor is never admitted as a COLD replica. */
    elpis_file_service *s = service(PAGES, PAGES);
    const int armed_before = atomic_exchange(&armed, 0);   /* the O_RDWR open below is the refusal's input */
    int rw = open(ASSET, O_RDWR | O_CLOEXEC);
    atomic_store(&armed, armed_before);
    uint32_t asset = 0;
    if (rw >= 0) {   /* mode 0444: refused already unless privileged; either way no writable replica */
        CHECK(admit_fd(s, rw, &asset) == ELPIS_FILE_UNSUPPORTED, "case 10: a writable descriptor was admitted");
        close(rw);
    }
    elpis_file_service_destroy(s);
}

/* ---- case 9: the read-only filesystem ---------------------------------------------------------------------------- */

static int write_file(const char *path, const char *text) {
    int fd = open(path, O_WRONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    ssize_t n = write(fd, text, strlen(text));
    close(fd);
    return n == (ssize_t)strlen(text) ? 0 : -1;
}

/* In a child: a private mount namespace (a user namespace too when unprivileged) with the working directory bound
 * read-only onto itself. Returns 77 when the namespace cannot be made. */
static int rofs_child(void) {
    const uid_t uid = getuid();
    const gid_t gid = getgid();
    if (geteuid() != 0) {
        if (unshare(CLONE_NEWUSER | CLONE_NEWNS) != 0) return 77;
        char map[64];
        snprintf(map, sizeof map, "0 %u 1", (unsigned)uid);
        if (write_file("/proc/self/setgroups", "deny") != 0 || write_file("/proc/self/uid_map", map) != 0) return 77;
        snprintf(map, sizeof map, "0 %u 1", (unsigned)gid);
        if (write_file("/proc/self/gid_map", map) != 0) return 77;
    } else if (unshare(CLONE_NEWNS) != 0) {
        return 77;
    }
    char cwd[4096];
    if (!getcwd(cwd, sizeof cwd)) return 2;
    if (mount(NULL, "/", NULL, MS_REC | MS_PRIVATE, NULL) != 0 ||
        mount(cwd, cwd, NULL, MS_BIND, NULL) != 0 ||
        mount(NULL, cwd, NULL, MS_REMOUNT | MS_BIND | MS_RDONLY, NULL) != 0)
        return 77;
    if (chdir(cwd) != 0) return 2;   /* re-enter through the read-only mount */

    /* Every write attempt fails EROFS. */
    errno = 0;
    int fd = open("probe", O_WRONLY | O_CREAT | O_CLOEXEC, 0600);
    CHECK(fd < 0 && errno == EROFS, "case 9: create on the read-only mount (errno %d)", errno);
    if (fd >= 0) close(fd);
    errno = 0;
    CHECK(mkdir("probe.d", 0700) != 0 && errno == EROFS, "case 9: mkdir (errno %d)", errno);
    errno = 0;
    CHECK(unlink(ASSET) != 0 && errno == EROFS, "case 9: unlink (errno %d)", errno);
    if (geteuid() == 0) {   /* root would otherwise bypass the 0444 mode */
        errno = 0;
        fd = open(ASSET, O_RDWR | O_CLOEXEC);
        CHECK(fd < 0 && errno == EROFS, "case 9: open the asset for writing (errno %d)", errno);
        if (fd >= 0) close(fd);
    }

    /* The frozen COLD -> WARM path works unchanged. */
    atomic_store(&armed, 1);
    uint8_t *buf = malloc(SIZE), sha[32];
    elpis_file_service *s = service(PAGES, PAGES);
    uint32_t asset = admit(s);
    load_all(s, asset, buf, "case 9");
    elpis_sha256(buf, SIZE, sha);
    CHECK(!memcmp(sha, cold_sha, 32), "case 9: SHA-256 over the read-only mount");
    CHECK(elpis_file_service_evict(s, asset) == ELPIS_FILE_OK, "case 9: evict");
    elpis_file_service_destroy(s);
    free(buf);
    atomic_store(&armed, 0);
    CHECK(atomic_load(&attempts) == 0, "case 9: %ld write attempts", atomic_load(&attempts));
    return fails ? 1 : 0;
}

static int rofs(void) {
    fixture();
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 2; }
    if (pid == 0) _exit(rofs_child());
    int status = 0;
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status)) return 2;
    const int rc = WEXITSTATUS(status);
    if (rc == 77) {
        printf("case 9 SKIPPED: no private mount namespace on this host (nothing was remounted)\n");
        return 77;
    }
    cold_unchanged("case 9 (after)");
    if (rc == 0 && !fails) printf("case 9: the frozen COLD -> WARM path runs over a read-only mount; every write "
                                  "attempt failed EROFS\n");
    return rc || fails ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--rofs")) return rofs();
    fixture();
    const int dir_entries = entries();
    atomic_store(&armed, 1);
    cases_1_to_5();
    case_6();
    case_7();
    case_8();
    case_10();
    atomic_store(&armed, 0);
    cold_unchanged("end");
    CHECK(entries() == dir_entries, "the working directory gained or lost an entry");
    CHECK(atomic_load(&attempts) == 0, "%ld write attempts while armed", atomic_load(&attempts));
    if (fails) {
        fprintf(stderr, "%d failures\n", fails);
        return 1;
    }
    printf("FMS v3 volatile residency: cases 1-8 and 10 PASS (0 write attempts; the COLD asset is byte-identical)\n");
    return 0;
}
