/* Native admission (host_admit.h). Linux only: openat2, sealed memfd, dlopen of /proc/self/fd/N. */
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

/* Copy, seal, hash, then load from the seal: the verified bytes are the bytes that run. The sealed descriptor stays
 * open (*kept): the loader knows a loaded object by its path, and a closed descriptor's number, reused by the next
 * memfd, would name the next library exactly as this one. */
static void *load_sealed(int dirfd, const host_pin *pins, size_t index, int *kept, char *err, size_t errlen) {
    const host_pin *pin = &pins[index];
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
    if (check_needed(sealed, pins, index, err, errlen)) goto out; /* before the loader sees it */
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
        handles[i] = load_sealed(dirfd, pins, i, &sealed[i], err, errlen);
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
