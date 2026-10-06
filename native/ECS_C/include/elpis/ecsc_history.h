#ifndef ELPIS_ECSC_HISTORY_H
#define ELPIS_ECSC_HISTORY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ELPIS_ECSC_OK = 0,
    ELPIS_ECSC_INVALID = -1,
    ELPIS_ECSC_CAPACITY = -2,
    ELPIS_ECSC_UTF8 = -3
};

enum {
    ELPIS_ECSC_MAX_BINDINGS = 16,
    ELPIS_ECSC_MAX_NAME_BYTES = 128,
    ELPIS_ECSC_MAX_VALUE_BYTES = 256
};

typedef struct elpis_ecsc_binding_view {
    const char *name;
    size_t name_len;
    const char *value;
    size_t value_len;
} elpis_ecsc_binding_view;

/* Native runtime-history ABI, additive v1 foundation. */
uint32_t elpis_ecsc_history_abi_version(void);

/*
 * Exact native encoding of elpis.runtime.history.ReceiptRecord.payload().
 *
 * Canonical byte form:
 *   json.dumps(..., sort_keys=True, separators=(",", ":"),
 *              ensure_ascii=True, allow_nan=False).encode("ascii")
 *
 * Binding names must already be strictly sorted and unique, matching the
 * existing Python ReceiptRecord invariant.
 */
int elpis_ecsc_receipt_payload_size(
    const char *subsystem,
    size_t subsystem_len,
    const char *kind,
    size_t kind_len,
    const char *digest_hex,
    size_t digest_len,
    const elpis_ecsc_binding_view *bindings,
    size_t binding_count,
    size_t *out_size
);

int elpis_ecsc_receipt_payload_write(
    const char *subsystem,
    size_t subsystem_len,
    const char *kind,
    size_t kind_len,
    const char *digest_hex,
    size_t digest_len,
    const elpis_ecsc_binding_view *bindings,
    size_t binding_count,
    uint8_t *out,
    size_t out_capacity,
    size_t *out_written
);

/*
 * Exact native equivalent of elpis.ECS_C.canonical.digest_bytes(raw):
 *
 *   sha256(canonical_bytes({"bytes": raw.hex()}))
 *
 * Output is lowercase hex plus terminating NUL.
 */
int elpis_ecsc_digest_bytes(
    const void *data,
    size_t size,
    char out_hex[65]
);


/*
 * Native equivalent of ECS_C.bus.message_id() for canonical ECS entity IDs.
 *
 * The runtime history path deals in actual ECS entity identities, which are
 * lowercase 64-hex content IDs. Sequence is the committed per-sender sequence.
 */
int elpis_ecsc_message_id(
    const char sender_entity_id[64],
    const char receiver_entity_id[64],
    uint64_t sequence,
    const void *payload,
    size_t payload_size,
    char out_message_id[65],
    char out_payload_digest[65]
);

/*
 * Exact canonical bytes of ECS_C.bus.Envelope.to_dict().
 *
 * The envelope is sealed from trusted sender identity + receiver + sequence +
 * payload + commit logical clock. No caller-supplied message/payload digest is
 * accepted.
 */
int elpis_ecsc_envelope_size(
    const char sender_entity_id[64],
    const char receiver_entity_id[64],
    uint64_t sequence,
    const void *payload,
    size_t payload_size,
    uint64_t logical_clock,
    size_t *out_size
);

int elpis_ecsc_envelope_write(
    const char sender_entity_id[64],
    const char receiver_entity_id[64],
    uint64_t sequence,
    const void *payload,
    size_t payload_size,
    uint64_t logical_clock,
    uint8_t *out,
    size_t out_capacity,
    size_t *out_written,
    char out_message_id[65],
    char out_payload_digest[65]
);

/*
 * ECS_C persistence frame:
 *
 *     uint64 big-endian canonical-event-byte-length
 *     canonical event bytes
 *
 * This primitive frames already-canonical event bytes. Event semantic
 * construction/verification is a later layer.
 */
int elpis_ecsc_event_frame_size(
    size_t canonical_event_size,
    size_t *out_size
);

int elpis_ecsc_event_frame_write(
    const void *canonical_event,
    size_t canonical_event_size,
    uint8_t *out,
    size_t out_capacity,
    size_t *out_written
);


/*
 * Exact committed MESSAGE_ENQUEUED event construction for the live ECS_C
 * protocol.
 *
 * logical_clock is derived as event_index + 1. transaction_id is derived as
 * ENQ:<message_id>. The envelope and event payload are constructed natively.
 *
 * before/after/prev are lowercase 64-hex canonical ECS_C digests.
 */
int elpis_ecsc_enqueue_event_size(
    const char sender_entity_id[64],
    const char receiver_entity_id[64],
    uint64_t sequence,
    const void *payload,
    size_t payload_size,
    uint64_t event_index,
    const char before_state_root[64],
    const char after_state_root[64],
    const char prev_event_digest[64],
    size_t *out_size
);

int elpis_ecsc_enqueue_event_write(
    const char sender_entity_id[64],
    const char receiver_entity_id[64],
    uint64_t sequence,
    const void *payload,
    size_t payload_size,
    uint64_t event_index,
    const char before_state_root[64],
    const char after_state_root[64],
    const char prev_event_digest[64],
    uint8_t *out,
    size_t out_capacity,
    size_t *out_written,
    char out_event_digest[65],
    char out_intent_digest[65],
    char out_message_id[65]
);

/*
 * Exact committed MESSAGE_PROCESSED event construction.
 *
 * transaction_id is PROC:<message_id>. entity_id and
 * receiver_entity_id are the receiver.
 */
int elpis_ecsc_processed_event_size(
    const char receiver_entity_id[64],
    const char message_id[64],
    uint64_t event_index,
    const char before_state_root[64],
    const char after_state_root[64],
    const char prev_event_digest[64],
    size_t *out_size
);

int elpis_ecsc_processed_event_write(
    const char receiver_entity_id[64],
    const char message_id[64],
    uint64_t event_index,
    const char before_state_root[64],
    const char after_state_root[64],
    const char prev_event_digest[64],
    uint8_t *out,
    size_t out_capacity,
    size_t *out_written,
    char out_event_digest[65],
    char out_intent_digest[65]
);

#ifdef __cplusplus
}
#endif

#endif
