/* Native file-asset page service: verification, residency, leases, faults,
 * descriptor ownership and concurrency (fms_file_service.h). */
#define _POSIX_C_SOURCE 200809L
#include "elpis/fms_file_service.h"
#include "elpis/sha256.h"
#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define PAGE 4096u
#define PAGES 9u
#define SIZE (PAGES * PAGE - 100u)

static uint8_t content[SIZE];
static uint8_t digests[PAGES * 32];

static int open_fds(void) {
    DIR *d = opendir("/proc/self/fd");
    int n = 0;
    assert(d);
    while (readdir(d)) ++n;
    closedir(d);
    return n;
}
static void make_file(const char *name) {
    for (uint32_t i = 0; i < SIZE; ++i) content[i] = (uint8_t)(i * 131u + (i >> 9));
    FILE *f = fopen(name, "wb");
    assert(f && fwrite(content, 1, SIZE, f) == SIZE && fclose(f) == 0);
    for (uint32_t p = 0; p < PAGES; ++p) {
        uint32_t n = p + 1 == PAGES ? SIZE - p * PAGE : PAGE;
        elpis_file_service_raw_digest(content + p * PAGE, n, digests + p * 32);
    }
}
static elpis_file_asset_v1 describe(int fd, const uint8_t *map) {
    struct stat st;
    assert(fstat(fd, &st) == 0);
    elpis_file_asset_v1 a = {ELPIS_FILE_SERVICE_ABI_V1, PAGE, fd, 0, SIZE, PAGES, {0}, map};
    a.stamp = (elpis_file_stamp){(uint64_t)st.st_dev, (uint64_t)st.st_ino, (uint64_t)st.st_size,
        (int64_t)st.st_mtim.tv_sec * 1000000000 + st.st_mtim.tv_nsec,
        (int64_t)st.st_ctim.tv_sec * 1000000000 + st.st_ctim.tv_nsec};
    return a;
}
static elpis_file_service *service(uint64_t warm, uint32_t max_pages) {
    elpis_file_service_config_v1 c = {ELPIS_FILE_SERVICE_ABI_V1, 0, warm, 4 * PAGE, 1 << 20, max_pages, 8, 4, 0};
    elpis_file_service *s = NULL;
    assert(elpis_file_service_create(&c, &s) == ELPIS_FILE_OK && s);
    return s;
}
static uint32_t admit(elpis_file_service *s, const char *name, const uint8_t *map) {
    int fd = open(name, O_RDONLY | O_CLOEXEC);
    assert(fd >= 0);
    elpis_file_asset_v1 a = describe(fd, map);
    uint32_t asset = 99;
    assert(elpis_file_service_admit(s, &a, &asset) == ELPIS_FILE_OK);
    close(fd); /* the service owns its own duplicate */
    return asset;
}
static void check_bytes(elpis_file_service *s, uint32_t asset, uint64_t off, uint64_t len) {
    uint8_t *buf = malloc(len);
    assert(elpis_file_service_copy(s, asset, off, len, buf) == ELPIS_FILE_OK);
    assert(!memcmp(buf, content + off, len));
    free(buf);
}
static void check_no_leases(elpis_file_service *s) {
    elpis_file_service_stats_v1 st;
    elpis_file_service_stats(s, &st);
    assert(st.pinned_bytes == 0 && st.live_ranges == 0 && st.lease_acquires == st.lease_releases);
}

static void test_raw_digest_identity(void) {
    /* Values from elpis.substrate.digests.raw_digest (Python oracle). */
    static const char *abc = "7af6aa69f418423295663b23360af7196ce150d9698e21725965cd7dd9a378f6";
    static const char *ramp = "1ac631f50f69d42401af34c4468551bae50560d46c4afcce32feb98a645024ae";
    uint8_t buf[5120], d[32], want[32];
    for (unsigned i = 0; i < sizeof buf; ++i) buf[i] = (uint8_t)i;
    elpis_file_service_raw_digest("abc", 3, d); assert(elpis_unhex32(abc, want) == 0 && !memcmp(d, want, 32));
    elpis_file_service_raw_digest(buf, sizeof buf, d); assert(elpis_unhex32(ramp, want) == 0 && !memcmp(d, want, 32));
}

static void test_admission_refusals(void) {
    make_file("asset.bin");
    elpis_file_service *s = service(4 * PAGE, 16);
    int base = open_fds();
    int fd = open("asset.bin", O_RDONLY);
    elpis_file_asset_v1 a = describe(fd, digests);
    uint32_t asset;
    elpis_file_asset_v1 bad = a; bad.fd = -1;
    assert(elpis_file_service_admit(s, &bad, &asset) == ELPIS_FILE_INVALID);
    bad.fd = 1000; /* closed descriptor */
    assert(elpis_file_service_admit(s, &bad, &asset) == ELPIS_FILE_INVALID);
    bad = a; bad.page_count = PAGES + 1;
    assert(elpis_file_service_admit(s, &bad, &asset) == ELPIS_FILE_INVALID);   /* page map geometry */
    bad = a; bad.size = SIZE - 1;
    assert(elpis_file_service_admit(s, &bad, &asset) == ELPIS_FILE_INVALID);   /* stamp/size disagreement */
    bad = a; bad.stamp.mtime_ns += 1;
    assert(elpis_file_service_admit(s, &bad, &asset) == ELPIS_FILE_INTEGRITY); /* changed since intake */
    bad = a; bad.stamp.ino ^= 1;
    assert(elpis_file_service_admit(s, &bad, &asset) == ELPIS_FILE_INTEGRITY);
    int rw = open("asset.bin", O_RDWR);
    bad = describe(rw, digests);
    assert(elpis_file_service_admit(s, &bad, &asset) == ELPIS_FILE_UNSUPPORTED); /* writable descriptor */
    close(rw);
    int dir = open(".", O_RDONLY | O_DIRECTORY);
    bad = describe(dir, digests); bad.size = SIZE; bad.stamp.size = SIZE;
    assert(elpis_file_service_admit(s, &bad, &asset) == ELPIS_FILE_INTEGRITY); /* non-regular */
    close(dir);
    int pipes[2];
    assert(pipe(pipes) == 0);
    bad = a; bad.fd = pipes[0];
    assert(elpis_file_service_admit(s, &bad, &asset) == ELPIS_FILE_INTEGRITY);
    close(pipes[0]); close(pipes[1]);
    bad = a; bad.page_size = 8 * PAGE; bad.page_count = (SIZE + bad.page_size - 1) / bad.page_size;
    assert(elpis_file_service_admit(s, &bad, &asset) == ELPIS_FILE_LIMIT);     /* four-copy staging bound */
    elpis_file_service_test_hooks h = {0}; h.fail_alloc_after = 1;
    elpis_file_service_set_test_hooks(&h);
    assert(elpis_file_service_admit(s, &a, &asset) == ELPIS_FILE_LIMIT);       /* allocation failure */
    elpis_file_service_set_test_hooks(NULL);
    close(fd);
    assert(open_fds() == base); /* every refused admission closed its duplicate */
    elpis_file_service_stats_v1 st;
    elpis_file_service_stats(s, &st);
    assert(st.assets == 0 && st.map_bytes == 0);
    elpis_file_service_destroy(s);
    elpis_file_service_destroy(NULL);
}

static void test_residency_and_leases(void) {
    make_file("asset.bin");
    int base = open_fds();
    elpis_file_service *s = service(4 * PAGE, 16);
    uint32_t a = admit(s, "asset.bin", digests);
    assert(open_fds() == base + 1);
    check_bytes(s, a, 0, 10);
    check_bytes(s, a, PAGE - 5, 10);                 /* crosses a page boundary */
    check_bytes(s, a, SIZE - 7, 7);                  /* short final page */
    elpis_file_service_stats_v1 st;
    elpis_file_service_stats(s, &st);
    /* page 0 miss; pages 0 hit + 1 miss; final short page miss */
    assert(st.misses == 3 && st.hits == 1 && st.semantic_bytes == 27 && st.reads == 3);
    assert(st.pread_bytes == 2 * PAGE + (SIZE - 8 * PAGE));
    uint64_t r;
    assert(elpis_file_service_acquire(s, a, 0, 5 * PAGE, &r) == ELPIS_FILE_LIMIT);    /* exceeds native budget */
    assert(elpis_file_service_acquire(s, a, SIZE - 1, 2, &r) == ELPIS_FILE_INVALID);  /* outside asset */
    assert(elpis_file_service_acquire(s, a, 0, 0, &r) == ELPIS_FILE_INVALID);
    assert(elpis_file_service_acquire(s, 7, 0, 1, &r) == ELPIS_FILE_MISSING);
    /* LRU: resident pages oldest first; a hit moves to the end. */
    uint32_t assets[16]; uint64_t pages[16];
    size_t n = elpis_file_service_pages(s, assets, pages, 16);
    assert(n == 3 && pages[0] == 0 && pages[1] == 1 && pages[2] == 8);
    check_bytes(s, a, 0, 1);
    n = elpis_file_service_pages(s, assets, pages, 16);
    assert(n == 3 && pages[0] == 1 && pages[1] == 8 && pages[2] == 0);
    check_bytes(s, a, 2 * PAGE, 2 * PAGE);           /* evicts page 1 (oldest unleased) */
    n = elpis_file_service_pages(s, assets, pages, 16);
    assert(n == 4 && pages[0] == 8 && pages[1] == 0 && pages[2] == 2 && pages[3] == 3);
    check_bytes(s, a, 4 * PAGE, 1);                  /* evicts page 8 */
    n = elpis_file_service_pages(s, assets, pages, 16);
    assert(n == 4 && pages[0] == 0 && pages[3] == 4);
    elpis_file_service_stats(s, &st);
    assert(st.evictions == 2 && st.resident_bytes == 4 * PAGE && st.resident_high_water == 4 * PAGE);
    /* Leased pages are never evicted: hold a 3-page range, then demand 2 more. */
    uint64_t held, other;
    assert(elpis_file_service_acquire(s, a, 5 * PAGE, 3 * PAGE, &held) == ELPIS_FILE_OK);
    elpis_file_service_stats(s, &st);
    assert(st.pinned_bytes == 3 * PAGE && st.live_ranges == 1);
    assert(elpis_file_service_acquire(s, a, 0, 2 * PAGE, &other) == ELPIS_FILE_LIMIT); /* all candidates leased */
    elpis_file_service_stats(s, &st);
    assert(st.pinned_bytes == 3 * PAGE && st.live_ranges == 1);                       /* failure retained nothing */
    assert(elpis_file_service_evict(s, a) == ELPIS_FILE_BUSY);
    assert(elpis_file_service_evict(s, UINT32_MAX) == ELPIS_FILE_BUSY);
    uint8_t buf[3 * PAGE];
    assert(elpis_file_service_read(s, held, buf, sizeof buf - 1) == ELPIS_FILE_INVALID);
    assert(elpis_file_service_read(s, held, buf, sizeof buf) == ELPIS_FILE_OK && !memcmp(buf, content + 5 * PAGE, sizeof buf));
    assert(elpis_file_service_release(s, held) == ELPIS_FILE_OK);
    assert(elpis_file_service_release(s, held) == ELPIS_FILE_STALE);                  /* exactly once */
    assert(elpis_file_service_read(s, held, buf, sizeof buf) == ELPIS_FILE_STALE);
    assert(elpis_file_service_release(s, held + 4096) == ELPIS_FILE_STALE);            /* forged generation */
    assert(elpis_file_service_evict(s, a) == ELPIS_FILE_OK);
    elpis_file_service_stats(s, &st);
    assert(st.resident_bytes == 0 && st.resident_pages == 0);
    /* Range table bound and forced release. */
    uint64_t live[4];
    for (int i = 0; i < 4; ++i) assert(elpis_file_service_acquire(s, a, (uint64_t)i * PAGE, 1, &live[i]) == ELPIS_FILE_OK);
    assert(elpis_file_service_acquire(s, a, 0, 1, &other) == ELPIS_FILE_LIMIT);
    assert(elpis_file_service_release_all(s) == 4);
    for (int i = 0; i < 4; ++i) assert(elpis_file_service_release(s, live[i]) == ELPIS_FILE_STALE);
    check_no_leases(s);
    elpis_file_service_destroy(s);
    assert(open_fds() == base);
}

static int pread_mode;
static elpis_file_service *interrupt_target;
static long fault_pread(int fd, void *buf, size_t n, int64_t off) {
    static int calls;
    if (pread_mode == 1) { if (calls++ % 2 == 0) { errno = EINTR; return -1; } }        /* interrupted read */
    if (pread_mode == 2) return off == 0 ? (long)pread(fd, buf, n / 2, (off_t)off) : 0; /* short read */
    if (pread_mode == 3) { errno = EIO; return -1; }
    if (pread_mode == 4) { size_t k = n > 3 ? 3 : n; return (long)pread(fd, buf, k, (off_t)off); } /* tiny reads */
    if (pread_mode == 5 && interrupt_target) { elpis_file_service_interrupt(interrupt_target); interrupt_target = NULL; }
    return (long)pread(fd, buf, n, (off_t)off);
}

static void test_faults(void) {
    make_file("asset.bin");
    elpis_file_service_test_hooks h = {fault_pread, 0};
    elpis_file_service_set_test_hooks(&h);
    elpis_file_service_stats_v1 st;
    for (pread_mode = 1; pread_mode <= 4; ++pread_mode) {
        elpis_file_service *s = service(4 * PAGE, 16);
        uint32_t a = admit(s, "asset.bin", digests);
        uint8_t buf[PAGE];
        elpis_file_status rc = elpis_file_service_copy(s, a, 10, 100, buf);
        elpis_file_service_stats(s, &st);
        if (pread_mode == 1) assert(rc == ELPIS_FILE_OK && st.interrupted_reads >= 1 && !memcmp(buf, content + 10, 100));
        if (pread_mode == 2 || pread_mode == 3) assert(rc == ELPIS_FILE_IO && st.io_failures == 1 && st.resident_pages == 0);
        if (pread_mode == 4) assert(rc == ELPIS_FILE_OK && st.reads == (PAGE + 2) / 3 && st.pread_bytes == PAGE);
        check_no_leases(s);
        elpis_file_service_destroy(s);
    }
    /* Interrupt observed at the next page boundary: nothing retained. */
    pread_mode = 5;
    elpis_file_service *s = service(4 * PAGE, 16);
    uint32_t a = admit(s, "asset.bin", digests);
    interrupt_target = s;
    uint64_t r;
    assert(elpis_file_service_acquire(s, a, 0, 3 * PAGE, &r) == ELPIS_FILE_BUSY);
    check_no_leases(s);
    assert(elpis_file_service_acquire(s, a, 0, 3 * PAGE, &r) == ELPIS_FILE_OK); /* a new epoch proceeds */
    assert(elpis_file_service_release(s, r) == ELPIS_FILE_OK);
    elpis_file_service_destroy(s);
    pread_mode = 0;
    /* Allocation failure of the range parts. */
    s = service(4 * PAGE, 16);
    a = admit(s, "asset.bin", digests);
    h.fail_alloc_after = 1; elpis_file_service_set_test_hooks(&h);
    assert(elpis_file_service_acquire(s, a, 0, 1, &r) == ELPIS_FILE_LIMIT);
    check_no_leases(s);
    elpis_file_service_set_test_hooks(NULL);
    /* Wrong pinned page digest (corrupt page / page-map mismatch). */
    uint8_t wrong[PAGES * 32];
    memcpy(wrong, digests, sizeof wrong); wrong[3 * 32] ^= 1;
    uint32_t b = admit(s, "asset.bin", wrong);
    uint8_t buf[2 * PAGE];
    assert(elpis_file_service_copy(s, b, 2 * PAGE + 1, 2 * PAGE, buf) == ELPIS_FILE_INTEGRITY);
    elpis_file_service_stats(s, &st);
    assert(st.integrity_failures == 1);
    check_no_leases(s);
    assert(elpis_file_service_copy(s, b, 0, PAGE, buf) == ELPIS_FILE_OK); /* other pages remain usable */
    /* Changed backing object: rewrite (mtime/ctime), then truncate. */
    check_bytes(s, a, 0, 16);
    struct timespec pause = {0, 20000000};
    nanosleep(&pause, NULL); /* exceed coarse filesystem timestamp granularity */
    FILE *f = fopen("asset.bin", "r+b");
    assert(f && fputc(content[0], f) != EOF && fclose(f) == 0);
    assert(elpis_file_service_copy(s, a, 0, 16, buf) == ELPIS_FILE_INTEGRITY); /* even a resident page hit */
    assert(truncate("asset.bin", PAGE) == 0);
    assert(elpis_file_service_copy(s, a, 5 * PAGE, 16, buf) == ELPIS_FILE_INTEGRITY);
    check_no_leases(s);
    elpis_file_service_destroy(s);
}

typedef struct { elpis_file_service *s; uint32_t asset; unsigned seed; int evicts; } worker;
static void *run(void *arg) {
    worker *w = arg;
    uint8_t buf[2 * PAGE];
    unsigned x = w->seed;
    for (int i = 0; i < 400; ++i) {
        x = x * 1103515245u + 12345u;
        uint64_t off = (x >> 8) % (SIZE - 2 * PAGE), len = 1 + (x >> 3) % (2 * PAGE - 1);
        elpis_file_status rc = elpis_file_service_copy(w->s, w->asset, off, len, buf);
        assert(rc == ELPIS_FILE_OK || rc == ELPIS_FILE_LIMIT || rc == ELPIS_FILE_BUSY);
        if (rc == ELPIS_FILE_OK) assert(!memcmp(buf, content + off, len));
        if (w->evicts && i % 50 == 0) {
            rc = elpis_file_service_evict(w->s, UINT32_MAX);
            assert(rc == ELPIS_FILE_OK || rc == ELPIS_FILE_BUSY);
            elpis_file_service_interrupt(w->s);
        }
    }
    return NULL;
}
static void test_concurrency(void) {
    make_file("asset.bin");
    elpis_file_service *shared = service(8 * PAGE, 8), *alone = service(4 * PAGE, 4);
    uint32_t a = admit(shared, "asset.bin", digests), b = admit(alone, "asset.bin", digests);
    worker w[5] = {{shared, a, 1, 0}, {shared, a, 2, 1}, {shared, a, 3, 0}, {alone, b, 4, 0}, {alone, b, 5, 1}};
    pthread_t t[5];
    for (int i = 0; i < 5; ++i) assert(pthread_create(&t[i], NULL, run, &w[i]) == 0);
    for (int i = 0; i < 5; ++i) assert(pthread_join(t[i], NULL) == 0);
    check_no_leases(shared); check_no_leases(alone);
    elpis_file_service_stats_v1 s1, s2;
    elpis_file_service_stats(shared, &s1); elpis_file_service_stats(alone, &s2);
    assert(s1.resident_bytes <= 8 * PAGE && s2.resident_bytes <= 4 * PAGE && s1.resident_high_water <= 8 * PAGE);
    elpis_file_service_destroy(shared); elpis_file_service_destroy(alone);
}

int main(void) {
    assert(elpis_file_service_abi_version() == ELPIS_FILE_SERVICE_ABI_V1);
    elpis_file_service_config_v1 c = {ELPIS_FILE_SERVICE_ABI_V1, 0, 0, 1, 1, 1, 1, 1, 0};
    elpis_file_service *s = NULL;
    assert(elpis_file_service_create(&c, &s) == ELPIS_FILE_INVALID && !s);
    test_raw_digest_identity();
    test_admission_refusals();
    test_residency_and_leases();
    test_faults();
    test_concurrency();
    puts("fms file service: ok");
    return 0;
}
