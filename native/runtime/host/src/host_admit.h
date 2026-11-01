/* Native admission of the host's libraries (docs/NATIVE_HOST.md, "Admission"): the native counterpart of
 * elpis.substrate.native_admission. Every library is opened read-only beneath the library directory with
 * openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS), copied into a sealed memfd and measured there (size and SHA-256):
 * the immutable library file is read, never written, and the bytes that run are the sealed in-memory copy. Only after
 * the whole measured set matches the operator's catalog pin are the seals loaded, in dependency order. Before the
 * loader sees a library, every DT_NEEDED it names must be the SONAME of a library admitted before it or a platform
 * library (the C runtime, the loader, the sanitizer runtimes); afterwards every SONAME is checked to bind exactly the
 * object admitted under it. The pins authenticate bytes, not a publisher; the loader, the kernel and the platform
 * libraries are trusted. */
#ifndef ELPIS_HOST_ADMIT_H
#define ELPIS_HOST_ADMIT_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const char *id;   /* catalog identifier, e.g. elpis_ecsg_k1 */
    const char *file; /* file name in an installed library directory */
    const char *soname;
    uint64_t size;
    uint8_t sha256[32];
} host_pin;

/* Open `name` beneath `dirfd` for reading: openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS),
 * O_NOFOLLOW, close-on-exec. A descriptor, or -1 with errno. */
int host_open_beneath(int dirfd, const char *name);
/* SHA-256 and size of the running executable (/proc/self/exe). */
int host_digest_self(uint8_t out[32], char *err, size_t errlen);
/* Seal one library: open `path` read-only beneath `dirfd`, copy it into a sealed memfd (write, grow, shrink and seal
 * sealed) and measure the seal: its size and SHA-256 into `pin`. The sealed descriptor into *sealed; the caller keeps
 * it open for the life of the process (the loader names an object by its /proc/self/fd path). Nothing is written to
 * any file. */
int host_seal(int dirfd, const char *path, host_pin *pin, int *sealed, char *err, size_t errlen);
/* Load `n` sealed, measured libraries in order (dependency check, then dlopen of the seal); `handles` receives each
 * handle. Refuses (-1) on any mismatch, leaving earlier libraries loaded (sealed objects are never unloaded). */
int host_load(const host_pin *pins, const int *sealed, size_t n, void **handles, char *err, size_t errlen);
/* A function pointer from an admitted handle, or NULL. */
void (*host_symbol(void *handle, const char *name))(void);

#endif
