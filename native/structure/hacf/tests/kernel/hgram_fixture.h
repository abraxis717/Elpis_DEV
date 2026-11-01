/* hgram_fixture.h - TEST FIXTURE ONLY: builds H-gram files for the read-only tests.
 *
 * Elpis never writes an H-gram (elpis/hgram_store.h: its initializer and association writer are retired). The
 * read-only probe and association reads still need well-formed and deliberately damaged files to read, so the tests
 * build them here, in their own scratch directory, from the documented format (an ABI-1 4096-byte header, then
 * 256-byte slots; an association record pair per bucket). This header is included only by test sources: no library
 * or executable target compiles it, and nothing in it is reachable from a runtime. */
#ifndef ELPIS_HGRAM_FIXTURE_H
#define ELPIS_HGRAM_FIXTURE_H

#define _GNU_SOURCE 1
#include "elpis/hgram_store.h"

#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define HGRAM_FIXTURE_MIB 1048576ULL

static inline uint64_t hgram_fixture_fnv(const unsigned char *p, size_t n) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= UINT64_C(1099511628211); }
    return h;
}
static inline void hgram_fixture_put32(unsigned char *p, uint32_t v) {
    for (unsigned i = 0; i < 4; i++) p[i] = (unsigned char)(v >> (8 * i));
}
static inline void hgram_fixture_put64(unsigned char *p, uint64_t v) {
    for (unsigned i = 0; i < 8; i++) p[i] = (unsigned char)(v >> (8 * i));
}

/* A complete, fully allocated H-gram file of `mib` MiB at `path` (which must not exist), then chmod `mode`.
 * Returns 0 on success. */
static inline int hgram_fixture_create(const char *path, uint64_t mib, mode_t mode) {
    const uint64_t bytes = mib * HGRAM_FIXTURE_MIB;
    unsigned char h[ELPIS_HGRAM_HEADER_BYTES];
    memset(h, 0, sizeof h);
    memcpy(h, "ELPHGRA1", 8);
    hgram_fixture_put32(h + 8, ELPIS_HGRAM_ABI);
    hgram_fixture_put32(h + 12, ELPIS_HGRAM_HEADER_BYTES);
    hgram_fixture_put64(h + 16, bytes);
    hgram_fixture_put64(h + 24, bytes - ELPIS_HGRAM_HEADER_BYTES);
    hgram_fixture_put32(h + 32, ELPIS_HGRAM_SLOT_BYTES);
    hgram_fixture_put64(h + 40, (bytes - ELPIS_HGRAM_HEADER_BYTES) / ELPIS_HGRAM_SLOT_BYTES);
    hgram_fixture_put64(h + 48, hgram_fixture_fnv(h, 48));
    int fd = open(path, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) return -1;
    int rc = posix_fallocate(fd, 0, (off_t)bytes) != 0 || pwrite(fd, h, sizeof h, 0) != (ssize_t)sizeof h ||
             fsync(fd) != 0;
    rc |= close(fd) != 0;
    rc |= chmod(path, mode) != 0;
    return rc ? -1 : 0;
}

static inline uint64_t hgram_fixture_bucket(const uint8_t key[ELPIS_HGRAM_ASSOC_KEY_BYTES], uint64_t mib) {
    const uint64_t slots = (mib * HGRAM_FIXTURE_MIB - ELPIS_HGRAM_HEADER_BYTES) / ELPIS_HGRAM_SLOT_BYTES;
    return hgram_fixture_fnv(key, ELPIS_HGRAM_ASSOC_KEY_BYTES) % (slots / 2);
}

/* The file offset of copy `copy` (0 or 1) of `key`'s bucket. */
static inline uint64_t hgram_fixture_offset(const uint8_t key[ELPIS_HGRAM_ASSOC_KEY_BYTES], uint64_t mib, int copy) {
    return ELPIS_HGRAM_HEADER_BYTES + hgram_fixture_bucket(key, mib) * 2u * ELPIS_HGRAM_SLOT_BYTES +
           (uint64_t)copy * ELPIS_HGRAM_SLOT_BYTES;
}

/* Write one valid association record (key, value, generation) into copy `copy` of `key`'s bucket, as the retired
 * writer did. The file is opened for writing by the test, which must make it writable first (chmod) if needed. */
static inline int hgram_fixture_put(const char *path, uint64_t mib, const uint8_t key[ELPIS_HGRAM_ASSOC_KEY_BYTES],
                             const uint8_t value[ELPIS_HGRAM_ASSOC_VALUE_BYTES], uint64_t generation, int copy) {
    unsigned char p[ELPIS_HGRAM_SLOT_BYTES];
    memset(p, 0, sizeof p);
    memcpy(p, "ELPHAS01", 8);
    hgram_fixture_put32(p + 8, ELPIS_HGRAM_ASSOC_RECORD_ABI);
    hgram_fixture_put32(p + 12, 1u);
    hgram_fixture_put64(p + 16, hgram_fixture_bucket(key, mib));
    hgram_fixture_put64(p + 24, generation);
    memcpy(p + 32, key, ELPIS_HGRAM_ASSOC_KEY_BYTES);
    memcpy(p + 64, value, ELPIS_HGRAM_ASSOC_VALUE_BYTES);
    hgram_fixture_put64(p + 248, hgram_fixture_fnv(p, 248));
    int fd = open(path, O_WRONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    int rc = pwrite(fd, p, sizeof p, (off_t)hgram_fixture_offset(key, mib, copy)) != (ssize_t)sizeof p;
    rc |= close(fd) != 0;
    return rc ? -1 : 0;
}

/* Flip one byte at `offset` (damage: a torn or rotted record). */
static inline int hgram_fixture_flip(const char *path, uint64_t offset) {
    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) return -1;
    unsigned char b = 0;
    int rc = pread(fd, &b, 1, (off_t)offset) != 1;
    b ^= 0x88u;
    rc |= pwrite(fd, &b, 1, (off_t)offset) != 1;
    rc |= close(fd) != 0;
    return rc ? -1 : 0;
}

#endif
