/* R1 association reads over fixture files (elpis/hgram_store.h): a retained H-gram is a read-only static resource.
 *
 * Found, absent and conflicting keys; the newer of two valid copies; a torn copy leaving the other authoritative;
 * damage to both copies, a generation gap and a key mismatch between copies refused CORRUPT (never vacancy); a
 * shared lock that an exclusive holder makes BUSY; a read-only (0400) file. Under the link-time mutation trap
 * (Linux), every probe and read issues zero storage mutations: no write, resize, preallocation, creation, rename,
 * unlink or sync, and no descriptor opened for writing. The file's bytes and stamp are unchanged by the reads.
 * The retired writer (elpis_hgram_assoc_store_preapproved) is absent; fixture files come from hgram_fixture.h. */
#define _GNU_SOURCE 1
#include "elpis/hgram_store.h"
#include "hgram_fixture.h"

#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef HGRAM_WRITE_TRAP
#include <stdarg.h>
/* Test-only mutation trap (link-time --wrap, Linux only; see CMakeLists). Armed only around library calls: the
 * fixture builder (test code) writes its files with the trap disarmed. */
static struct {
    int armed;
    unsigned mutations;
} trap;
static int mutating(int flags) {
    return (flags & O_ACCMODE) != O_RDONLY || (flags & (O_CREAT | O_APPEND | O_TRUNC)) != 0 ||
           (flags & O_TMPFILE) == O_TMPFILE;
}
static int needs_mode(int flags) { return (flags & O_CREAT) != 0 || (flags & O_TMPFILE) == O_TMPFILE; }
static void hit(void) { if (trap.armed) trap.mutations++; }
ssize_t __real_pwrite(int fd, const void *b, size_t n, off_t off);
ssize_t __wrap_pwrite(int fd, const void *b, size_t n, off_t off) { hit(); return __real_pwrite(fd, b, n, off); }
ssize_t __real_write(int fd, const void *b, size_t n);
ssize_t __wrap_write(int fd, const void *b, size_t n) { if (fd > 2) hit(); return __real_write(fd, b, n); }
int __real_ftruncate(int fd, off_t len);
int __wrap_ftruncate(int fd, off_t len) { hit(); return __real_ftruncate(fd, len); }
int __real_fallocate(int fd, int mode, off_t off, off_t len);
int __wrap_fallocate(int fd, int mode, off_t off, off_t len) { hit(); return __real_fallocate(fd, mode, off, len); }
int __real_posix_fallocate(int fd, off_t off, off_t len);
int __wrap_posix_fallocate(int fd, off_t off, off_t len) { hit(); return __real_posix_fallocate(fd, off, len); }
int __real_fsync(int fd);
int __wrap_fsync(int fd) { hit(); return __real_fsync(fd); }
int __real_fdatasync(int fd);
int __wrap_fdatasync(int fd) { hit(); return __real_fdatasync(fd); }
int __real_open(const char *p, int flags, ...);
int __wrap_open(const char *p, int flags, ...) {
    mode_t m = 0;
    if (needs_mode(flags)) { va_list ap; va_start(ap, flags); m = (mode_t)va_arg(ap, int); va_end(ap); }
    if (mutating(flags)) hit();
    return __real_open(p, flags, m);
}
int __real_openat(int dfd, const char *p, int flags, ...);
int __wrap_openat(int dfd, const char *p, int flags, ...) {
    mode_t m = 0;
    if (needs_mode(flags)) { va_list ap; va_start(ap, flags); m = (mode_t)va_arg(ap, int); va_end(ap); }
    if (mutating(flags)) hit();
    return __real_openat(dfd, p, flags, m);
}
int __real___openat_2(int dfd, const char *p, int flags);
int __wrap___openat_2(int dfd, const char *p, int flags) {
    if (mutating(flags)) hit();
    return __real___openat_2(dfd, p, flags);
}
int __real_renameat(int ofd, const char *o, int nfd, const char *n);
int __wrap_renameat(int ofd, const char *o, int nfd, const char *n) { hit(); return __real_renameat(ofd, o, nfd, n); }
int __real_unlinkat(int dfd, const char *p, int flags);
int __wrap_unlinkat(int dfd, const char *p, int flags) { hit(); return __real_unlinkat(dfd, p, flags); }
#define ARM() (trap.armed = 1)
#define DISARM() (trap.armed = 0)
#else
#define ARM() ((void)0)
#define DISARM() ((void)0)
#endif

#define MIB_FILE 1u

static void join(char *out, size_t cap, const char *dir, const char *leaf) {
    int n = snprintf(out, cap, "%s/%s", dir, leaf);
    assert(n > 0 && (size_t)n < cap);
}
static unsigned file_count(const char *path) {
    DIR *d = opendir(path);
    assert(d);
    unsigned n = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) n += strcmp(e->d_name, ".") && strcmp(e->d_name, "..");
    assert(closedir(d) == 0);
    return n;
}
static uint64_t content_digest(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    assert(fd >= 0);
    static unsigned char buf[1u << 20];
    ssize_t n = pread(fd, buf, sizeof buf, 0);
    assert(n == (ssize_t)sizeof buf && close(fd) == 0);
    return hgram_fixture_fnv(buf, sizeof buf);
}

static int get(const char *bin, const uint8_t key[32], elpis_hgram_assoc_record *out) {
    ARM();
    int rc = elpis_hgram_assoc_get(bin, key, out);
    DISARM();
    return rc;
}

int main(void) {
    char cwd[PATH_MAX - 64], dir[PATH_MAX];
    assert(getcwd(cwd, sizeof cwd));
    join(dir, sizeof dir, cwd, "hgram-assoc-XXXXXX");
    assert(mkdtemp(dir) != NULL);
    char bin[PATH_MAX + 32];
    join(bin, sizeof bin, dir, "hgram.bin");
    uint8_t key[32], other[32], v1[128], v2[128], v3[128];
    memset(key, 0x17, sizeof key);
    memset(other, 0x42, sizeof other);
    memset(v1, 0x72, sizeof v1);
    memset(v2, 0x11, sizeof v2);
    memset(v3, 0x45, sizeof v3);
    elpis_hgram_assoc_record a = {0};

    assert(get(bin, key, &a) == ELPIS_HGRAM_INVALID);                   /* absent: nothing created */
    assert(file_count(dir) == 0);
    assert(hgram_fixture_create(bin, MIB_FILE, 0600) == 0);

    /* Absent key, then one valid copy, then the newer of two copies. */
    assert(get(bin, key, &a) == ELPIS_HGRAM_ASSOC_NOT_FOUND);
    assert(hgram_fixture_put(bin, MIB_FILE, key, v1, 1, 0) == 0);
    assert(get(bin, key, &a) == ELPIS_HGRAM_ASSOC_OK && a.generation == 1 && !memcmp(a.value, v1, 128));
    assert(hgram_fixture_put(bin, MIB_FILE, key, v2, 2, 1) == 0);
    assert(get(bin, key, &a) == ELPIS_HGRAM_ASSOC_OK && a.generation == 2 && !memcmp(a.value, v2, 128));

    /* A key mapping to the same bucket finds it held: CONFLICT, found through the public read alone. */
    uint8_t clash[32];
    int found = 0;
    for (unsigned n = 0; n < 200000 && !found; n++) {
        memset(clash, 0, sizeof clash);
        clash[0] = (uint8_t)n; clash[1] = (uint8_t)(n >> 8); clash[2] = (uint8_t)(n >> 16);
        if (memcmp(clash, key, 32) == 0) continue;
        found = get(bin, clash, &a) == ELPIS_HGRAM_ASSOC_CONFLICT;
    }
    assert(found);

    /* An independent key elsewhere. */
    assert(hgram_fixture_bucket(other, MIB_FILE) != hgram_fixture_bucket(key, MIB_FILE));
    assert(hgram_fixture_put(bin, MIB_FILE, other, v3, 7, 0) == 0);
    assert(get(bin, other, &a) == ELPIS_HGRAM_ASSOC_OK && a.generation == 7 && !memcmp(a.value, v3, 128));
    assert(get(bin, key, &a) == ELPIS_HGRAM_ASSOC_OK && a.generation == 2);

    /* Reads change nothing: the bytes, the stamp and the directory are as the fixture left them. */
    struct stat initial, after;
    assert(stat(bin, &initial) == 0);
    const uint64_t digest = content_digest(bin);
    for (int i = 0; i < 1000; i++) assert(get(bin, key, &a) == ELPIS_HGRAM_ASSOC_OK);
    ARM();
    elpis_hgram_info info;
    assert(elpis_hgram_probe(bin, &info) == ELPIS_HGRAM_EXISTS);
    DISARM();
    assert(stat(bin, &after) == 0 && after.st_size == initial.st_size && after.st_ino == initial.st_ino &&
           after.st_mtim.tv_sec == initial.st_mtim.tv_sec && after.st_mtim.tv_nsec == initial.st_mtim.tv_nsec &&
           after.st_ctim.tv_sec == initial.st_ctim.tv_sec && after.st_ctim.tv_nsec == initial.st_ctim.tv_nsec);
    assert(content_digest(bin) == digest && file_count(dir) == 1);

    /* An exclusive holder (another tool) makes the shared read BUSY, in another process. */
    int locked = open(bin, O_RDONLY | O_CLOEXEC);
    assert(locked >= 0 && flock(locked, LOCK_EX | LOCK_NB) == 0);
    pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        elpis_hgram_assoc_record tmp = {0};
        _exit(elpis_hgram_assoc_get(bin, key, &tmp) == ELPIS_HGRAM_ASSOC_BUSY ? 0 : 1);
    }
    int status = 0;
    assert(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    assert(flock(locked, LOCK_UN) == 0 && close(locked) == 0);

    /* Damage: the newer copy torn leaves the older authoritative; both damaged is CORRUPT, never vacancy. */
    const uint64_t first = hgram_fixture_offset(key, MIB_FILE, 0), second = hgram_fixture_offset(key, MIB_FILE, 1);
    assert(hgram_fixture_flip(bin, second + 120) == 0);
    assert(get(bin, key, &a) == ELPIS_HGRAM_ASSOC_OK && a.generation == 1 && !memcmp(a.value, v1, 128));
    assert(hgram_fixture_flip(bin, first + 119) == 0);
    assert(get(bin, key, &a) == ELPIS_HGRAM_ASSOC_CORRUPT);
    /* Two valid copies with a generation gap, or with different keys: CORRUPT. */
    assert(hgram_fixture_put(bin, MIB_FILE, key, v1, 1, 0) == 0 && hgram_fixture_put(bin, MIB_FILE, key, v2, 3, 1) == 0);
    assert(get(bin, key, &a) == ELPIS_HGRAM_ASSOC_CORRUPT);
    assert(hgram_fixture_put(bin, MIB_FILE, key, v2, 2, 1) == 0);
    assert(get(bin, key, &a) == ELPIS_HGRAM_ASSOC_OK && a.generation == 2);

    /* A read-only static resource (mode 0400) reads the same. */
    assert(chmod(bin, 0400) == 0);
    assert(get(bin, key, &a) == ELPIS_HGRAM_ASSOC_OK && a.generation == 2 && !memcmp(a.value, v2, 128));
    assert(get(bin, other, &a) == ELPIS_HGRAM_ASSOC_OK && a.generation == 7);

#ifdef HGRAM_WRITE_TRAP
    assert(trap.mutations == 0);
    printf("ELPIS_HGRAM_ASSOC_READS mutations=%u (every probe and read under the trap)\n", trap.mutations);
#endif
    assert(file_count(dir) == 1);
    assert(unlink(bin) == 0 && rmdir(dir) == 0);
    puts("ELPIS_HGRAM_ASSOC_READ_ONLY_PASS");
    return 0;
}
