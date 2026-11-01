/* A retained H-gram: a fixed-capacity file of association cells, READ ONLY (docs/ELPIS_MISSION.md, Amendment 3).
 *
 * Elpis reads a retained H-gram as a static resource (probe, assoc_get) and never writes, creates, initializes,
 * preallocates or resizes one: there is no persistent H-gram learning. The initializer (elpis_hgram_init_once, the
 * elpis_hgram_init executable) and the in-place association writer (elpis_hgram_assoc_store_preapproved) are retired
 * by the volatile-native migration; they are in git history only, and producing such a file is offline tooling
 * that needs separate human authorization (it is not part of this build). The file format below is unchanged.
 */
#ifndef ELPIS_HGRAM_STORE_H
#define ELPIS_HGRAM_STORE_H
#include <stdint.h>
#define ELPIS_HGRAM_HEADER_BYTES 4096u
#define ELPIS_HGRAM_SLOT_BYTES 256u
#define ELPIS_HGRAM_ABI 1u

typedef struct elpis_hgram_info {
    uint64_t total_bytes;
    uint64_t slots;
    uint64_t capacity_mib;
} elpis_hgram_info;

enum elpis_hgram_status {
    ELPIS_HGRAM_CREATED = 0,   /* the retired initializer's; never returned */
    ELPIS_HGRAM_EXISTS = 1,
    ELPIS_HGRAM_INVALID = -1,
    ELPIS_HGRAM_PATH = -2,
    ELPIS_HGRAM_SPACE = -3,    /* the retired initializer's; never returned */
    ELPIS_HGRAM_IO = -4,
    ELPIS_HGRAM_RACE = -5      /* the retired initializer's; never returned */
};
/* Strictly read-only: absent, incomplete, modified or wrong-version file refused. The file must be a single-link
 * regular file of this user, private to it (mode 0400 or 0600), fully allocated, at least 1 MiB.
 * `out` is written only on valid existing file. */
int elpis_hgram_probe(const char *absolute_file, elpis_hgram_info *out);
/* R1 fixed-layout association cells, read only. Two 256-byte physical slots per deterministic bucket. */
#define ELPIS_HGRAM_ASSOC_KEY_BYTES 32u
#define ELPIS_HGRAM_ASSOC_VALUE_BYTES 128u
#define ELPIS_HGRAM_ASSOC_RECORD_ABI 1u

typedef struct elpis_hgram_assoc_record {
    uint8_t value[ELPIS_HGRAM_ASSOC_VALUE_BYTES];
    uint64_t generation;
} elpis_hgram_assoc_record;

enum elpis_hgram_assoc_status {
    ELPIS_HGRAM_ASSOC_OK = 0,
    ELPIS_HGRAM_ASSOC_NOT_FOUND = 1,
    ELPIS_HGRAM_ASSOC_BUSY = -6,
    ELPIS_HGRAM_ASSOC_CONFLICT = -7,
    ELPIS_HGRAM_ASSOC_STALE = -8,      /* the retired writer's; never returned */
    ELPIS_HGRAM_ASSOC_CORRUPT = -9
};
/* Key is a 32-byte identity, not semantic text. A bucket held by another key is CONFLICT; one torn copy leaves the
 * other authoritative; two damaged copies are CORRUPT, never vacancy. Reads never create files or change contents;
 * the descriptor is O_RDONLY under a shared advisory lock (BUSY while another holder has it exclusively). */
int elpis_hgram_assoc_get(const char *absolute_file,
                           const uint8_t key[ELPIS_HGRAM_ASSOC_KEY_BYTES],
                           elpis_hgram_assoc_record *out);
#endif
