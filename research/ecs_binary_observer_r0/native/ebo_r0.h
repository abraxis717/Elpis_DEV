/*
 * ECS Binary Observation R0: a deterministic fixed-width binary record of one accepted K1 QUERY.
 * RESEARCH_ONLY, NO_RUNTIME_AUTHORITY, UNQUALIFIED. Nothing in src/ or native/runtime calls this.
 *
 * The observer reads what one completed canonical QUERY (elpis_ecsg_k1_query_identity) already left in caller memory:
 * the query rows X, the answers f_W(X) and the 32-byte retained-state digest (research/ecs_binary_observer_r0/ADR.md).
 * It links no ECS symbol, makes no ECS call, allocates nothing, performs no I/O and never writes its inputs.
 * A refusal leaves the caller's record buffer untouched.
 *
 * Record EBO1, 64 bytes, little-endian:
 *   0   4  magic "EBOR"
 *   4   2  format version (1)
 *   6   1  encoder id (1 = GRAY4, 2 = SPROJ)
 *   7   1  flags (bit 0: rows > window, values beyond the window not observed; bit 1: rows < window, padded)
 *   8   2  rows of the query
 *   10  1  dim
 *   11  1  window rows = min(rows, 64)
 *   12  4  law tag = SHA-256(law blob)[0..4]
 *   16  8  state identity = digest[0..8] (exact identity bytes, truncated; no geometry)
 *   24  8  input identity = H64(dim, rows, bit patterns of the window rows of X) (exact, non-cryptographic)
 *   32 32  payload: 256 bits encoding the window answers under the law (the only learned-value field)
 */
#ifndef ELPIS_EBO_R0_H
#define ELPIS_EBO_R0_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    EBO_RECORD_BYTES = 64u,
    EBO_FORMAT_VERSION = 1u,
    EBO_LAW_VERSION = 1u,
    EBO_WINDOW_ROWS = 64u,
    EBO_PAYLOAD_BITS = 256u,
    EBO_MAX_ROWS = 4096u,
    EBO_MAX_DIM = 64u,
    EBO_GRAY4_THRESHOLDS = 15u,
    EBO_SPROJ_TERMS = 4u,
    EBO_DIGEST_BYTES = 32u,
    EBO_FLAG_TRUNCATED = 1u,
    EBO_FLAG_PADDED = 2u
};

enum { EBO_ENCODER_GRAY4 = 1, EBO_ENCODER_SPROJ = 2 };

typedef enum {
    EBO_OK = 0,
    EBO_INVALID = -1,    /* NULL argument or malformed buffer */
    EBO_NONFINITE = -2,  /* a non-finite value in the observed window (answers or query rows) */
    EBO_SHAPE = -3,      /* dim or rows outside the admitted shape */
    EBO_VERSION = -4,    /* law or record of another format/law version */
    EBO_LAW = -5,        /* law not admitted, or a record bound to another law */
    EBO_DROPPED = -6,    /* handoff ring full: the record was dropped (DROP_NEWEST) */
    EBO_EMPTY = -7       /* handoff ring empty */
} ebo_status;

/* An admitted law. Built only by ebo_law_init from a canonical blob; never mutated afterwards. */
typedef struct {
    uint32_t ready;                                  /* EBO_LAW_READY after a successful init */
    uint32_t encoder;
    uint8_t tag[4];
    double thresholds[EBO_GRAY4_THRESHOLDS];         /* GRAY4: strictly increasing, finite */
    double mu;                                       /* SPROJ: u = (v - mu) * inv_scale */
    double inv_scale;
    uint8_t index[EBO_PAYLOAD_BITS][EBO_SPROJ_TERMS]; /* SPROJ: window rows, < 64 */
    int8_t sign[EBO_PAYLOAD_BITS][EBO_SPROJ_TERMS];  /* SPROJ: +1 / -1 */
    double tau[EBO_PAYLOAD_BITS];                    /* SPROJ: bit k = (s_k >= tau_k) */
} ebo_law;

/*
 * Canonical law blob v1, little-endian:
 *   0  4  magic "EBOL"
 *   4  2  law version (1)
 *   6  1  encoder id
 *   7  1  zero
 *   GRAY4: 15 binary64 thresholds                                   (total 8 + 120  = 128 bytes)
 *   SPROJ: mu, inv_scale (binary64); 256 x 4 index bytes; 256 x 4 sign bytes (0x01 = +1, 0xFF = -1);
 *          256 binary64 tau                                          (total 8 + 16 + 1024 + 1024 + 2048 = 4120 bytes)
 */
size_t ebo_law_blob_bytes(uint32_t encoder);
ebo_status ebo_law_init(ebo_law *law, const uint8_t *blob, size_t size);

/* Encode one accepted QUERY. x: rows*dim values (row-major), values: rows answers, digest: 32 bytes.
 * Work is bounded by the window (min(rows, 64) rows), never by rows. */
ebo_status ebo_encode(const ebo_law *law, size_t dim, size_t rows, const double *x, const double *values,
                      const uint8_t digest[EBO_DIGEST_BYTES], uint8_t record[EBO_RECORD_BYTES]);

/* Consumer-side admission of a record against the law it claims. */
ebo_status ebo_record_check(const ebo_law *law, const uint8_t record[EBO_RECORD_BYTES]);

/*
 * Bounded nonblocking handoff: one producer, one consumer, caller-owned storage of `capacity` records
 * (a power of two, 1..65536). push never blocks and never allocates: when the ring is full the new record is
 * dropped and counted (DROP_NEWEST); a failed or dropped push never affects the observed computation.
 */
typedef struct {
    uint8_t (*slots)[EBO_RECORD_BYTES];
    uint64_t mask;
    uint64_t producer_pad[6];
    uint64_t tail;       /* written by the producer only (atomic) */
    uint64_t pushed;     /* producer-owned counters (atomic) */
    uint64_t dropped;
    uint64_t pad2[5];
    uint64_t head;       /* written by the consumer only (atomic) */
    uint64_t pad3[7];
} ebo_ring;

ebo_status ebo_ring_init(ebo_ring *ring, uint8_t (*slots)[EBO_RECORD_BYTES], uint64_t capacity);
ebo_status ebo_ring_push(ebo_ring *ring, const uint8_t record[EBO_RECORD_BYTES]);
ebo_status ebo_ring_pop(ebo_ring *ring, uint8_t record[EBO_RECORD_BYTES]);
uint64_t ebo_ring_pushed(const ebo_ring *ring);
uint64_t ebo_ring_dropped(const ebo_ring *ring);
size_t ebo_ring_bytes(void);
size_t ebo_law_bytes(void);

#ifdef __cplusplus
}
#endif

#endif
