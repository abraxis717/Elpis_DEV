/* Fixed-capacity adaptive-state backing file, NOT the learning policy.
 * Elpis may attach bounded associative state to this file only after its
 * allocator/update format is separately qualified. Never create on read.
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
    ELPIS_HGRAM_CREATED = 0,
    ELPIS_HGRAM_EXISTS = 1,
    ELPIS_HGRAM_INVALID = -1,
    ELPIS_HGRAM_PATH = -2,
    ELPIS_HGRAM_SPACE = -3,
    ELPIS_HGRAM_IO = -4,
    ELPIS_HGRAM_RACE = -5
};
/* Strictly read-only: absent, incomplete, modified or wrong-version file refused.
 * `out` is written only on valid existing file. */
int elpis_hgram_probe(const char *absolute_file, elpis_hgram_info *out);
/* Explicit initialization. Never resizes or replaces an existing file.
 * `size_mib` is exact 1,048,576-byte MiB. Checks free space and reserves at
 * least 1GiB OR 10% of current available filesystem bytes, whichever larger.
 * Preallocates full physical extent and only then commits its header.
 * Any unexpected existing file is a hard refusal, not something to overwrite.
 */
int elpis_hgram_init_once(const char *absolute_file, uint64_t size_mib,
                          elpis_hgram_info *out);
/* R1 fixed-layout association cells. These functions are native storage
 * mechanics, NOT a credit policy or authority to call them from inference.
 * A trusted, separately qualified broker must gate write access. No caller
 * may treat an arbitrary model-produced key/value as preapproved.
 * Two 256-byte physical slots per deterministic bucket, no eviction.
 */
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
    ELPIS_HGRAM_ASSOC_STALE = -8,
    ELPIS_HGRAM_ASSOC_CORRUPT = -9
};
/* Key is an already-authorized 32-byte identity, not semantic text.
 * Collision means no mutation. Reads never create files or change contents.
 * The 4096-byte R0 header remains unchanged. */
int elpis_hgram_assoc_get(const char *absolute_file,
                           const uint8_t key[ELPIS_HGRAM_ASSOC_KEY_BYTES],
                           elpis_hgram_assoc_record *out);
/* Authorized caller only (not an inference-time API). `expected_generation`
 * is 0 for new keys or the last observed generation for updates. An exact
 * retry of the immediately preceding committed update is idempotent. No
 * destructive eviction, no variable-length data, no sidecar, no resize.
 * Preserves one valid previous version across detected torn inactive writes,
 * contingent on ordinary filesystem sync semantics (NOT a power-loss proof).
 */
int elpis_hgram_assoc_store_preapproved(const char *absolute_file,
             const uint8_t key[ELPIS_HGRAM_ASSOC_KEY_BYTES],
             const uint8_t value[ELPIS_HGRAM_ASSOC_VALUE_BYTES],
             uint64_t expected_generation,
             elpis_hgram_assoc_record *out);

#endif
