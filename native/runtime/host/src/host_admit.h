/* Native admission of the host's libraries (docs/NATIVE_HOST.md, "Admission"): the native counterpart of
 * elpis.substrate.native_admission. Every library is opened beneath the deployment's lib directory with
 * openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS), copied into a sealed memfd, hashed there against its catalog pin and
 * loaded from the seal, in dependency order. Before the loader sees a library, every DT_NEEDED it names must be the
 * SONAME of a library admitted before it or a platform library (the C runtime, the loader, the sanitizer runtimes);
 * afterwards every SONAME is checked to bind exactly the object admitted under it. The pins authenticate bytes, not a
 * publisher; the loader, the kernel and the platform libraries are trusted. */
#ifndef ELPIS_HOST_ADMIT_H
#define ELPIS_HOST_ADMIT_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const char *id;   /* catalog identifier, e.g. elpis_ecsg_k1 */
    const char *file; /* file name beneath lib/ */
    const char *soname;
    uint64_t size;
    uint8_t sha256[32];
} host_pin;

/* Open `name` beneath `dirfd` for reading: openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS),
 * O_NOFOLLOW, close-on-exec. A descriptor, or -1 with errno. */
int host_open_beneath(int dirfd, const char *name);
/* SHA-256 and size of a regular file opened beneath `dirfd` (no symlinks). 0 or -1 (errno-style message in err). */
int host_digest_file(int dirfd, const char *name, uint64_t *size, uint8_t out[32], char *err, size_t errlen);
/* SHA-256 of the running executable (/proc/self/exe). */
int host_digest_self(uint8_t out[32], char *err, size_t errlen);
/* Admit `n` pinned libraries from beneath `dirfd`, in order; `handles` receives each dlopen handle and `sealed` each
 * sealed descriptor, which the caller keeps open for the life of the process (the loader names an object by its
 * /proc/self/fd path). Refuses (-1) on any mismatch, leaving earlier libraries loaded (sealed objects are never
 * unloaded). */
int host_admit(int dirfd, const host_pin *pins, size_t n, void **handles, int *sealed, char *err, size_t errlen);
/* A function pointer from an admitted handle, or NULL. */
void (*host_symbol(void *handle, const char *name))(void);

#endif
