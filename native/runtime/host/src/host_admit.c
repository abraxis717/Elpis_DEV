/* Native admission (host_admit.h). Linux only: openat2 (read-only), sealed memfd, dlopen of /proc/self/fd/N. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include "host_admit.h"

#include <dlfcn.h>
#include <elf.h>
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

#define CHUNK (1u << 16)

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

int host_digest_self(uint8_t out[32], char *err, size_t errlen) {
    errno = 0;
    const int fd = open("/proc/self/exe", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return fail(err, errlen, "cannot open", "/proc/self/exe");
    struct stat st;
    int rc = fstat(fd, &st) ? -1 : digest_fd(fd, (uint64_t)st.st_size, out);
    close(fd);
    return rc ? fail(err, errlen, "cannot read", "/proc/self/exe") : 0;
}

/* ---- the dependencies a library names (its ELF dynamic section), checked before it is loaded ----------------- */

#if defined(__x86_64__)
#define HOST_MACHINE EM_X86_64
#elif defined(__aarch64__)
#define HOST_MACHINE EM_AARCH64
#else
#error "the native host admits 64-bit x86 or AArch64 ELF objects only"
#endif

/* The platform the loader, the kernel and the C runtime provide (trusted, docs/NATIVE_HOST.md), sanitizer runtimes
 * included. Every other dependency must be a library admitted before this one. */
static int platform_library(const char *name) {
    static const char *const exact[] = {"libc.so.6", "libm.so.6", "libgcc_s.so.1", "libdl.so.2", "libpthread.so.0",
                                        "librt.so.1", "ld-linux-x86-64.so.2", "ld-linux-aarch64.so.1"};
    static const char *const prefix[] = {"libasan.so.", "libubsan.so.", "libtsan.so."};
    for (size_t i = 0; i < sizeof exact / sizeof *exact; ++i)
        if (!strcmp(name, exact[i])) return 1;
    for (size_t i = 0; i < sizeof prefix / sizeof *prefix; ++i)
        if (!strncmp(name, prefix[i], strlen(prefix[i]))) return 1;
    return 0;
}

/* The file offset of virtual address `va` (inside a PT_LOAD segment's file image), or (uint64_t)-1. */
static uint64_t file_offset(const Elf64_Phdr *ph, size_t n, uint64_t va, uint64_t size) {
    for (size_t i = 0; i < n; ++i)
        if (ph[i].p_type == PT_LOAD && va >= ph[i].p_vaddr && va - ph[i].p_vaddr < ph[i].p_filesz) {
            const uint64_t off = ph[i].p_offset + (va - ph[i].p_vaddr);
            return off < size ? off : (uint64_t)-1;
        }
    return (uint64_t)-1;
}

/* Refuse unless every DT_NEEDED of the sealed image names an admitted SONAME (pins[0..i)) or a platform library. */
static int check_needed(int sealed, const host_pin *pins, size_t i, char *err, size_t errlen) {
    const host_pin *pin = &pins[i];
    void *map = mmap(NULL, pin->size, PROT_READ, MAP_PRIVATE, sealed, 0);
    if (map == MAP_FAILED) return fail(err, errlen, "cannot map", pin->file);
    const uint8_t *img = map;
    const uint64_t size = pin->size;
    int rc = -1;
    const char *why = "not a well-formed 64-bit ELF shared object for this machine";
    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)img;
    if (size < sizeof *eh || memcmp(eh->e_ident, ELFMAG, SELFMAG) || eh->e_ident[EI_CLASS] != ELFCLASS64 ||
        eh->e_ident[EI_DATA] != ELFDATA2LSB || eh->e_type != ET_DYN || eh->e_machine != HOST_MACHINE ||
        eh->e_phentsize != sizeof(Elf64_Phdr) || eh->e_phoff > size || eh->e_phnum > (size - eh->e_phoff) / sizeof(Elf64_Phdr))
        goto out;
    const Elf64_Phdr *ph = (const Elf64_Phdr *)(img + eh->e_phoff);
    const Elf64_Dyn *dyn = NULL;
    size_t ndyn = 0;
    for (size_t k = 0; k < eh->e_phnum; ++k)
        if (ph[k].p_type == PT_DYNAMIC) {
            if (ph[k].p_offset > size || ph[k].p_filesz > size - ph[k].p_offset) goto out;
            dyn = (const Elf64_Dyn *)(img + ph[k].p_offset), ndyn = ph[k].p_filesz / sizeof(Elf64_Dyn);
        }
    uint64_t strtab = (uint64_t)-1, strsz = 0;
    for (size_t k = 0; dyn && k < ndyn && dyn[k].d_tag != DT_NULL; ++k) {
        if (dyn[k].d_tag == DT_STRTAB) strtab = file_offset(ph, eh->e_phnum, dyn[k].d_un.d_ptr, size);
        if (dyn[k].d_tag == DT_STRSZ) strsz = dyn[k].d_un.d_val;
    }
    if (!dyn || strtab == (uint64_t)-1 || strsz > size - strtab) goto out;
    for (size_t k = 0; k < ndyn && dyn[k].d_tag != DT_NULL; ++k) {
        if (dyn[k].d_tag != DT_NEEDED) continue;
        const uint64_t at = dyn[k].d_un.d_val;
        if (at >= strsz || !memchr(img + strtab + at, 0, strsz - at)) goto out;
        const char *name = (const char *)img + strtab + at;
        int admitted = platform_library(name);
        for (size_t j = 0; j < i && !admitted; ++j) admitted = !strcmp(name, pins[j].soname);
        if (!admitted) {
            errno = 0;
            snprintf(err, errlen, "%s depends on %s, which is neither admitted before it nor a platform library",
                     pin->file, name);
            rc = -2;
            goto out;
        }
    }
    rc = 0;
out:
    munmap(map, pin->size);
    if (rc == -1) {
        errno = 0;
        snprintf(err, errlen, "%s: %s", pin->file, why);
    }
    return rc ? -1 : 0;
}

int host_seal(int dirfd, const char *path, host_pin *pin, int *sealed, char *err, size_t errlen) {
    static uint8_t buf[CHUNK];
    if (err && errlen) err[0] = 0;
    *sealed = -1;
    errno = 0;
    const int src = host_open_beneath(dirfd, path);
    if (src < 0) return fail(err, errlen, "cannot open read-only", path);
    struct stat before, after;
    if (fstat(src, &before) || !S_ISREG(before.st_mode) || (uint64_t)before.st_size > ((uint64_t)1 << 28)) {
        close(src);
        errno = 0;
        return fail(err, errlen, "not a regular library file of at most 256 MiB", path);
    }
    const uint64_t size = (uint64_t)before.st_size;
    const int fd = memfd_create("elpis-admitted", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) {
        close(src);
        return fail(err, errlen, "memfd_create", path);
    }
    int rc = -1;
    for (uint64_t at = 0; at < size; at += CHUNK) {
        const size_t n = (size_t)(size - at < CHUNK ? size - at : CHUNK);
        if (read_exact(src, buf, n, (off_t)at)) goto out;
        for (size_t done = 0; done < n;) {
            const ssize_t w = write(fd, buf + done, n - done); /* the memfd: anonymous memory, not a file */
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) goto out;
            done += (size_t)w;
        }
    }
    if (fstat(src, &after) || after.st_size != before.st_size || after.st_mtim.tv_sec != before.st_mtim.tv_sec ||
        after.st_mtim.tv_nsec != before.st_mtim.tv_nsec || after.st_ino != before.st_ino) {
        errno = 0;
        fail(err, errlen, "changed while being copied", path);
        goto out;
    }
    const int seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
    if (fcntl(fd, F_ADD_SEALS, seals) || (fcntl(fd, F_GET_SEALS) & seals) != seals) {
        fail(err, errlen, "cannot seal", path);
        goto out;
    }
    if (digest_fd(fd, size, pin->sha256)) goto out;
    pin->size = size;
    rc = 0;
out:
    if (rc && err && !err[0]) fail(err, errlen, "cannot copy", path);
    close(src);
    if (rc) close(fd);
    else *sealed = fd;
    return rc;
}

int host_load(const host_pin *pins, const int *sealed, size_t n, void **handles, char *err, size_t errlen) {
    if (err && errlen) err[0] = 0;
    for (size_t i = 0; i < n; ++i) {
        handles[i] = NULL;
        if (check_needed(sealed[i], pins, i, err, errlen)) return -1; /* before the loader sees it */
        char path[64];
        snprintf(path, sizeof path, "/proc/self/fd/%d", sealed[i]);
        handles[i] = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handles[i]) {
            errno = 0;
            snprintf(err, errlen, "dlopen %s: %s", pins[i].file, dlerror());
            return -1;
        }
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
