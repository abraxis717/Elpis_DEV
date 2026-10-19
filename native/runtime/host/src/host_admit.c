/* Native admission (host_admit.h). Linux only: openat2, sealed memfd, dlopen of /proc/self/fd/N. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include "host_admit.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/openat2.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "elpis/sha256.h"

#define CHUNK (1u << 20)

static int fail(char *err, size_t errlen, const char *what, const char *name) {
    if (err && errlen) snprintf(err, errlen, "%s: %s%s%s", what, name, errno ? ": " : "", errno ? strerror(errno) : "");
    return -1;
}

int host_open_beneath(int dirfd, const char *name) {
    struct open_how how;
    memset(&how, 0, sizeof how);
    how.flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW;
    how.resolve = RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS;
    return (int)syscall(SYS_openat2, dirfd, name, &how, sizeof how);
}

static int read_exact(int fd, uint8_t *buf, size_t n, off_t at) {
    size_t done = 0;
    while (done < n) {
        const ssize_t r = pread(fd, buf + done, n - done, at + (off_t)done);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;
        done += (size_t)r;
    }
    return 0;
}

static int digest_fd(int fd, uint64_t size, uint8_t out[32]) {
    static uint8_t buf[CHUNK];
    elpis_sha256_ctx h;
    elpis_sha256_init(&h);
    for (uint64_t at = 0; at < size; at += CHUNK) {
        const size_t n = (size_t)(size - at < CHUNK ? size - at : CHUNK);
        if (read_exact(fd, buf, n, (off_t)at)) return -1;
        elpis_sha256_update(&h, buf, n);
    }
    elpis_sha256_final(&h, out);
    return 0;
}

int host_digest_file(int dirfd, const char *name, uint64_t *size, uint8_t out[32], char *err, size_t errlen) {
    errno = 0;
    const int fd = host_open_beneath(dirfd, name);
    if (fd < 0) return fail(err, errlen, "cannot open", name);
    struct stat st;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode)) {
        close(fd);
        return fail(err, errlen, "not a regular file", name);
    }
    *size = (uint64_t)st.st_size;
    const int rc = digest_fd(fd, *size, out);
    close(fd);
    return rc ? fail(err, errlen, "cannot read", name) : 0;
}

int host_digest_self(uint8_t out[32], char *err, size_t errlen) {
    errno = 0;
    const int fd = open("/proc/self/exe", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return fail(err, errlen, "cannot open", "/proc/self/exe");
    struct stat st;
    int rc = fstat(fd, &st) ? -1 : digest_fd(fd, (uint64_t)st.st_size, out);
    close(fd);
    return rc ? fail(err, errlen, "cannot read", "/proc/self/exe") : 0;
}

/* Copy, seal, hash, then load from the seal: the verified bytes are the bytes that run. The sealed descriptor stays
 * open (*kept): the loader knows a loaded object by its path, and a closed descriptor's number, reused by the next
 * memfd, would name the next library exactly as this one. */
static void *load_sealed(int dirfd, const host_pin *pin, int *kept, char *err, size_t errlen) {
    static uint8_t buf[CHUNK];
    errno = 0;
    const int src = host_open_beneath(dirfd, pin->file);
    if (src < 0) return fail(err, errlen, "cannot open beneath lib/", pin->file), NULL;
    struct stat before, after;
    if (fstat(src, &before) || !S_ISREG(before.st_mode) || (uint64_t)before.st_size != pin->size) {
        close(src);
        errno = 0;
        return fail(err, errlen, "size differs from its pin", pin->file), NULL;
    }
    const int sealed = memfd_create("elpis-admitted", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (sealed < 0) {
        close(src);
        return fail(err, errlen, "memfd_create", pin->file), NULL;
    }
    void *handle = NULL;
    for (uint64_t at = 0; at < pin->size; at += CHUNK) {
        const size_t n = (size_t)(pin->size - at < CHUNK ? pin->size - at : CHUNK);
        if (read_exact(src, buf, n, (off_t)at)) goto out;
        for (size_t done = 0; done < n;) {
            const ssize_t w = write(sealed, buf + done, n - done);
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) goto out;
            done += (size_t)w;
        }
    }
    if (fstat(src, &after) || after.st_size != before.st_size || after.st_mtim.tv_sec != before.st_mtim.tv_sec ||
        after.st_mtim.tv_nsec != before.st_mtim.tv_nsec || after.st_ino != before.st_ino) {
        errno = 0;
        fail(err, errlen, "changed while being copied", pin->file);
        goto out;
    }
    const int seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
    if (fcntl(sealed, F_ADD_SEALS, seals) || (fcntl(sealed, F_GET_SEALS) & seals) != seals) {
        fail(err, errlen, "cannot seal", pin->file);
        goto out;
    }
    uint8_t digest[32];
    if (digest_fd(sealed, pin->size, digest) || memcmp(digest, pin->sha256, 32)) {
        errno = 0;
        fail(err, errlen, "SHA-256 differs from its pin", pin->file);
        goto out;
    }
    char path[64];
    snprintf(path, sizeof path, "/proc/self/fd/%d", sealed);
    handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        errno = 0;
        snprintf(err, errlen, "dlopen %s: %s", pin->file, dlerror());
    }
out:
    if (!handle && err && !err[0]) fail(err, errlen, "cannot copy", pin->file);
    close(src);
    if (handle) *kept = sealed;
    else close(sealed);
    return handle;
}

int host_admit(int dirfd, const host_pin *pins, size_t n, void **handles, int *sealed, char *err, size_t errlen) {
    if (err && errlen) err[0] = 0;
    for (size_t i = 0; i < n; ++i) {
        sealed[i] = -1;
        handles[i] = load_sealed(dirfd, &pins[i], &sealed[i], err, errlen);
        if (!handles[i]) return -1;
        for (size_t j = 0; j < i; ++j)
            if (handles[j] == handles[i]) { /* the loader handed back an object it already had: never this one */
                errno = 0;
                return fail(err, errlen, "the loader did not load a new object for", pins[i].file);
            }
    }
    /* Every SONAME a later library may bind must resolve to exactly the object admitted under it. */
    for (size_t i = 0; i < n; ++i) {
        void *bound = dlopen(pins[i].soname, RTLD_NOW | RTLD_NOLOAD);
        const int same = bound == handles[i];
        if (bound) dlclose(bound); /* drops the reference RTLD_NOLOAD took */
        if (!same) {
            errno = 0;
            return fail(err, errlen, "the loader binds another object to", pins[i].soname);
        }
    }
    return 0;
}

void (*host_symbol(void *handle, const char *name))(void) {
    union {
        void *object;
        void (*function)(void);
    } u;
    u.object = dlsym(handle, name);
    return u.function;
}
