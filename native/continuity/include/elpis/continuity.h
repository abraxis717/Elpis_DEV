#ifndef ELPIS_CONTINUITY_H
#define ELPIS_CONTINUITY_H

/* Elpis continuity ABI v1: the minimal durable runtime authority (docs/CONTINUITY.md).
 *
 * Implemented in Rust (native/continuity). One fixed-size current-authority record in a
 * two-slot crash-safe register v2 (ELPCONT\x02, 176-byte records, 352 durable bytes):
 * the expected K1 retained-state digest of the ECS lineage and the evolution authority
 * (idle at a head, or reserved for one exact assertion). No history, no events, no replay.
 *
 * Every function returns ELPIS_CONTINUITY_OK (0) or a positive stable code below.
 * A store handle is used by one caller at a time; calls on one handle must not overlap.
 * Pointer arguments are NULL (refused, or the documented meaning) or valid for their size.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ELPIS_CONTINUITY_ABI_V1 = 1u,
    ELPIS_CONTINUITY_RECORD_SIZE = 176u,
};

enum {
    ELPIS_CONTINUITY_OK = 0,
    ELPIS_CONTINUITY_UNINITIALIZED = 1,
    ELPIS_CONTINUITY_UNANCHORED = 2,
    ELPIS_CONTINUITY_ALREADY_ANCHORED = 3,
    ELPIS_CONTINUITY_STATE_MISMATCH = 4,
    ELPIS_CONTINUITY_CORRUPT = 5,
    ELPIS_CONTINUITY_PUBLICATION_REFUSED = 6,
    ELPIS_CONTINUITY_PUBLICATION_UNCERTAIN = 7,
    ELPIS_CONTINUITY_LOCKED = 8,
    ELPIS_CONTINUITY_AUTHORITY_MISMATCH = 9,
    ELPIS_CONTINUITY_EVOLUTION_PENDING = 10,
    ELPIS_CONTINUITY_EVOLUTION_NOT_PENDING = 11,
    ELPIS_CONTINUITY_EXHAUSTED = 12,
    ELPIS_CONTINUITY_INVALID = 13,
    ELPIS_CONTINUITY_PATH = 14,
    ELPIS_CONTINUITY_OPEN = 15,
    ELPIS_CONTINUITY_LEGACY_STORAGE = 16,
    ELPIS_CONTINUITY_IO = 17,
    /* Produced only by the testing library (simulated process death). */
    ELPIS_CONTINUITY_TESTING_PROCESS_DEATH = 255,
};

/* Evolution authority. pending == 0: idle, assertion must be zero. pending == 1: reserved
 * for exactly `assertion`. revision 0 if and only if head is zero. reserved bytes zero. */
typedef struct {
    uint64_t revision;
    uint8_t head[32];
    uint8_t pending;
    uint8_t reserved[7];
    uint8_t assertion[32];
} elpis_continuity_evolution;

/* One verified authority. anchored == 0: no K1 lineage (k1_state_digest zero). The two
 * digests are outputs (evolution-authority v2 digest and the record checksum); they are
 * ignored on input. */
typedef struct {
    uint64_t generation;
    uint8_t anchored;
    uint8_t reserved[7];
    uint8_t k1_state_digest[32];
    elpis_continuity_evolution evolution;
    uint8_t evolution_digest[32];
    uint8_t record_digest[32];
} elpis_continuity_snapshot;

typedef struct elpis_continuity_store elpis_continuity_store;

uint32_t elpis_continuity_abi_version(void);
size_t elpis_continuity_record_size(void);
/* Static name of a code ("CONTINUITY_OK" for 0), or NULL for an unknown value. */
const char *elpis_continuity_code_name(int code);

/* Pure record functions. */
int elpis_continuity_evolution_digest(const elpis_continuity_evolution *, uint8_t out[32]);
int elpis_continuity_record_encode(const elpis_continuity_snapshot *, uint8_t out[176]);
/* *empty = 1 for an all-zero slot (out untouched); CORRUPT for anything not a valid v2 record. */
int elpis_continuity_record_decode(const uint8_t *raw, size_t len, elpis_continuity_snapshot *out, int *empty);

/* The store. `path` is an absolute directory path of `len` bytes. */
int elpis_continuity_store_create(const uint8_t *path, size_t len, elpis_continuity_store **out);
void elpis_continuity_store_destroy(elpis_continuity_store **store);
int elpis_continuity_store_open(elpis_continuity_store *, elpis_continuity_snapshot *out /* nullable */);
void elpis_continuity_store_close(elpis_continuity_store *);
int elpis_continuity_store_snapshot(elpis_continuity_store *, elpis_continuity_snapshot *out);

/* Transitions. Each publishes one complete record (one write, one fdatasync) or refuses
 * without change. `out` is nullable. A NULL digest argument is ELPIS_CONTINUITY_INVALID;
 * a NULL or malformed `expected` is ELPIS_CONTINUITY_AUTHORITY_MISMATCH. */
int elpis_continuity_anchor_cognition(elpis_continuity_store *, const uint8_t k1[32],
                                      elpis_continuity_snapshot *out);
int elpis_continuity_commit_cognition(elpis_continuity_store *, const uint8_t before[32],
                                      const uint8_t after[32], elpis_continuity_snapshot *out);
/* idle -> pending: durably reserve `assertion` before the caller may execute it. */
int elpis_continuity_reserve_evolution(elpis_continuity_store *, const elpis_continuity_evolution *expected,
                                       const uint8_t assertion[32], elpis_continuity_snapshot *out);
/* pending -> next idle with the established receipt digest; also explicit reconciliation. */
int elpis_continuity_finalize_evolution(elpis_continuity_store *, const elpis_continuity_evolution *expected,
                                        const uint8_t receipt[32], elpis_continuity_snapshot *out);

#ifdef __cplusplus
}
#endif

#endif
