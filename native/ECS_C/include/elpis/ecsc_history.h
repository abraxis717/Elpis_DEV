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


enum {
    ELPIS_ECSC_SCHEDULER_V1 = 1,
    ELPIS_ECSC_SCHEDULER_V2 = 2
};

/*
 * Exact ECS_C genesis descriptor identity.
 *
 * Equivalent to:
 *
 *   persistence.genesis_descriptor_digest(
 *       genesis_label,
 *       scheduler_protocol=<selected protocol>
 *   )
 *
 * Current protocol constants are bound exactly. This is content identity,
 * not authentication.
 */
int elpis_ecsc_genesis_digest(
    const char *genesis_label,
    size_t genesis_label_len,
    uint32_t scheduler_protocol,
    char out_digest[65]
);

/*
 * Exact identity of one founded ECS_C entity:
 *
 *   entity_id_from_founding(
 *       founding_record(index, label, genesis_digest)
 *   )
 */
int elpis_ecsc_entity_id(
    uint64_t founding_index,
    const char *label,
    size_t label_len,
    const char genesis_digest[64],
    char out_entity_id[65]
);

/*
 * Exact canonical version-0 entity-state identity:
 *
 *   initial_state_digest(entity_id)
 */
int elpis_ecsc_initial_state_digest(
    const char entity_id[64],
    char out_state_digest[65]
);


enum {
    ELPIS_ECSC_LIFECYCLE_FOUNDED = 1,
    ELPIS_ECSC_LIFECYCLE_ACTIVE = 2,
    ELPIS_ECSC_LIFECYCLE_DORMANT = 3,
    ELPIS_ECSC_LIFECYCLE_TERMINATED = 4
};

typedef struct elpis_ecsc_state_entity_view {
    const char *registry_key;
    const char *entity_id;

    const char *label;
    size_t label_len;

    uint64_t founding_index;
    const char *founding_digest;

    const char *state_entity_id;
    const char *prev_state_digest;

    uint32_t lifecycle;
    uint64_t state_version;
    const char *state_digest;

    /* Runtime-history entity payload is either {} or {"delivered":N}. */
    uint32_t has_delivered;
    uint64_t delivered;
} elpis_ecsc_state_entity_view;

typedef struct elpis_ecsc_state_envelope_view {
    uint64_t logical_clock;

    const char *message_id;
    const char *payload_digest;

    const void *payload;
    size_t payload_size;

    const char *receiver_entity_id;
    const char *sender_entity_id;

    uint64_t sequence;
} elpis_ecsc_state_envelope_view;

typedef struct elpis_ecsc_state_mailbox_view {
    const char *mailbox_key;
    const char *receiver_entity_id;

    uint64_t capacity;

    const elpis_ecsc_state_envelope_view *contents;
    size_t content_count;
} elpis_ecsc_state_mailbox_view;

typedef struct elpis_ecsc_state_watermark_view {
    const char *sender_entity_id;
    uint64_t sequence;
} elpis_ecsc_state_watermark_view;

typedef struct elpis_ecsc_state_root_view {
    const char *genesis_digest;
    const char *history_digest;

    uint64_t logical_clock;
    uint64_t mailbox_capacity;
    uint64_t mailbox_default_capacity;
    uint64_t next_founding_index;

    const elpis_ecsc_state_entity_view *entities;
    size_t entity_count;

    const elpis_ecsc_state_mailbox_view *mailboxes;
    size_t mailbox_count;

    const elpis_ecsc_state_watermark_view *watermarks;
    size_t watermark_count;
} elpis_ecsc_state_root_view;

/*
 * Exact native equivalent of:
 *
 *   state_root_digest(build_state_root(...))
 *
 * for the bounded runtime-history projection.
 *
 * The caller supplies already-materialized ECS_C projection fields. This
 * function is serialization/content-identity authority only: it does not
 * mutate history and does not infer missing state.
 */
int elpis_ecsc_state_root_digest(
    const elpis_ecsc_state_root_view *view,
    char out_digest[65]
);


/* Native durable-log status codes. */
enum {
    ELPIS_ECSC_IO = -4,
    ELPIS_ECSC_LOCKED = -5,
    ELPIS_ECSC_CORRUPT = -6,
    ELPIS_ECSC_APPEND_ROLLED_BACK = -7,
    ELPIS_ECSC_APPEND_UNCERTAIN = -8,
    ELPIS_ECSC_NOT_READY = -9,
    ELPIS_ECSC_PARTIAL_COMMIT = -10,
    /*
     * Bounded-segment dispositions (additive). SEGMENT_FULL is returned
     * BEFORE any byte is appended: the session stays usable and its state is
     * unchanged; the owner must compact and reopen. SEGMENT_MISMATCH means
     * the supplied global/base counts do not describe the segment file.
     */
    ELPIS_ECSC_SEGMENT_FULL = -11,
    ELPIS_ECSC_SEGMENT_MISMATCH = -12
};

typedef struct elpis_ecsc_log elpis_ecsc_log;

typedef struct elpis_ecsc_log_scan {
    uint64_t total_size;
    uint64_t complete_prefix;
    uint64_t frame_count;
    uint32_t has_incomplete_tail;
} elpis_ecsc_log_scan;

/*
 * Open one ECS_C event log as its exclusive local owner.
 *
 * The parent directory must already exist. The file is created mode 0600 if
 * absent. A non-blocking exclusive flock prevents a second owner.
 *
 * Open starts in recovery mode. No append is permitted until:
 *
 *   recover_scan
 *   semantic validation by the owner
 *   finish_recovery(validated_complete_prefix)
 */
int elpis_ecsc_log_open(
    const char *path,
    size_t path_len,
    elpis_ecsc_log **out_log
);

void elpis_ecsc_log_close(
    elpis_ecsc_log *log
);

/*
 * Structural bounded scan only.
 *
 * It recognizes complete ECS_C 8-byte-big-endian frames and an incomplete
 * crash tail. It intentionally does NOT declare complete frame payloads
 * semantically valid.
 */
int elpis_ecsc_log_recover_scan(
    elpis_ecsc_log *log,
    elpis_ecsc_log_scan *out_scan
);

/*
 * Authorize recovery only after the caller has semantically validated every
 * complete frame discovered by recover_scan().
 *
 * validated_complete_prefix MUST equal the structural complete prefix. A
 * smaller value is corruption, not permission to erase a complete frame.
 *
 * If an incomplete trailing frame exists, it is truncated and fsynced here.
 */
int elpis_ecsc_log_finish_recovery(
    elpis_ecsc_log *log,
    uint64_t validated_complete_prefix
);

/*
 * Durable append of one already-canonical, already-semantically-validated
 * ECS_C event payload. This function owns framing/write/fsync.
 *
 * Ordinary I/O failure:
 *   restore previous length + fsync -> ELPIS_ECSC_APPEND_ROLLED_BACK
 *
 * Failed rollback or failed rollback fsync:
 *   close/poison owner -> ELPIS_ECSC_APPEND_UNCERTAIN
 */
int elpis_ecsc_log_append_event_bytes(
    elpis_ecsc_log *log,
    const void *canonical_event,
    size_t canonical_event_size
);

uint64_t elpis_ecsc_log_size(
    const elpis_ecsc_log *log
);

uint64_t elpis_ecsc_log_frame_count(
    const elpis_ecsc_log *log
);


/*
 * Exact state digest for the runtime history receiver after one or more
 * MESSAGE_PROCESSED transitions:
 *
 *   state_digest(entity_id, version, {"delivered": delivered})
 */
int elpis_ecsc_delivered_state_digest(
    const char entity_id[64],
    uint64_t version,
    uint64_t delivered,
    char out_state_digest[65]
);

typedef struct elpis_ecsc_runtime_record_plan_result {
    uint64_t sequence;

    uint64_t enqueue_event_index;
    uint64_t processed_event_index;
    uint64_t final_logical_clock;

    uint64_t final_history_state_version;
    uint64_t final_delivered;

    size_t enqueue_event_size;
    size_t processed_event_size;

    char message_id[65];

    char enqueue_state_root[65];
    char final_state_root[65];
    char final_history_digest[65];

    char enqueue_event_digest[65];
    char processed_event_digest[65];

    char final_history_state_digest[65];
} elpis_ecsc_runtime_record_plan_result;

/*
 * Pure transition planner for one quiescent runtime-history receipt.
 *
 * Input authority:
 *   - a fully validated current ecs.state_root.v3 projection;
 *   - the current committed event digest;
 *   - one ACTIVE recorder identity;
 *   - the ACTIVE history receiver identity;
 *   - canonical receipt payload bytes.
 *
 * Output:
 *   MESSAGE_ENQUEUED and MESSAGE_PROCESSED canonical event bytes plus the
 *   exact resulting state identities.
 *
 * This function performs NO file I/O and mutates no caller-owned projection.
 * It is the state-transition layer later composed with elpis_ecsc_log_*.
 */
int elpis_ecsc_runtime_record_plan(
    const elpis_ecsc_state_root_view *current,
    const char current_event_digest[64],
    const char sender_entity_id[64],
    const char history_entity_id[64],
    const void *receipt_payload,
    size_t receipt_payload_size,

    uint8_t *enqueue_event,
    size_t enqueue_capacity,

    uint8_t *processed_event,
    size_t processed_capacity,

    elpis_ecsc_runtime_record_plan_result *out
);


/*
 * Native durable runtime-history session.
 *
 * The seed state must already have been fully semantically validated by the
 * canonical ECS_C replay authority and must be quiescent: every materialized
 * mailbox is empty. The native owner then acquires events.log, structurally
 * verifies the validated event count, and owns all later appends.
 *
 * No sidecar history authority is created.
 */
typedef struct elpis_ecsc_runtime_session
    elpis_ecsc_runtime_session;

int elpis_ecsc_runtime_session_open(
    const char *log_path,
    size_t log_path_len,
    const elpis_ecsc_state_root_view *validated_state,
    const char current_event_digest[64],
    uint64_t validated_event_count,
    elpis_ecsc_runtime_session **out_session
);

void elpis_ecsc_runtime_session_close(
    elpis_ecsc_runtime_session *session
);

/*
 * One native receipt operation.
 *
 * Success durably appends exactly:
 *   MESSAGE_ENQUEUED
 *   MESSAGE_PROCESSED
 *
 * APPEND_ROLLED_BACK on the first append leaves the session usable and the
 * pre-call state authoritative.
 *
 * If the enqueue is durably committed but the processed append definitely
 * rolls back, PARTIAL_COMMIT is returned and the session is poisoned. Reopen
 * and canonical replay are then required.
 *
 * Any indeterminate append also poisons the session.
 */
int elpis_ecsc_runtime_session_record(
    elpis_ecsc_runtime_session *session,
    const char sender_entity_id[64],
    const char history_entity_id[64],
    const void *receipt_payload,
    size_t receipt_payload_size,
    elpis_ecsc_runtime_record_plan_result *out
);

int elpis_ecsc_runtime_session_state_root_digest(
    const elpis_ecsc_runtime_session *session,
    char out_digest[65]
);

uint64_t elpis_ecsc_runtime_session_event_count(
    const elpis_ecsc_runtime_session *session
);

/*
 * Bounded active-segment ownership (additive ABI).
 *
 * A compacted runtime history keeps ONE global logical history: the segment
 * file holds only the events at global indices >= segment_base, with their
 * original global indices; the retired prefix is represented by a verified
 * compaction checkpoint whose state is the supplied validated_state.
 *
 * Open requires:
 *   global_event_count == validated_state->logical_clock      (else INVALID)
 *   segment_base <= global_event_count                         (else SEGMENT_MISMATCH)
 *   segment frame count == global_event_count - segment_base   (else SEGMENT_MISMATCH)
 *   policy finite and non-zero                                 (else INVALID)
 *   segment bytes/frames within policy                         (else SEGMENT_FULL)
 *
 * After open, every record checks the PLANNED frame sizes against the policy
 * before mutation: if the two framed events would not fit, SEGMENT_FULL is
 * returned with no append, no state change and no poisoning. All other
 * failure classes and poisoning rules are exactly those of
 * elpis_ecsc_runtime_session_record. event_count() reports the GLOBAL count.
 */
typedef struct elpis_ecsc_segment_policy {
    uint64_t max_segment_bytes;
    uint64_t max_segment_frames;
} elpis_ecsc_segment_policy;

int elpis_ecsc_runtime_session_open_segment(
    const char *segment_path,
    size_t segment_path_len,
    const elpis_ecsc_state_root_view *validated_state,
    const char current_event_digest[64],
    uint64_t global_event_count,
    uint64_t segment_base,
    const elpis_ecsc_segment_policy *policy,
    elpis_ecsc_runtime_session **out_session
);

/* UINT64_MAX when the session is NULL. */
uint64_t elpis_ecsc_runtime_session_segment_base(
    const elpis_ecsc_runtime_session *session
);

/* UINT64_MAX when the session is NULL, poisoned or has no open segment. */
uint64_t elpis_ecsc_runtime_session_segment_bytes(
    const elpis_ecsc_runtime_session *session
);

uint64_t elpis_ecsc_runtime_session_segment_frames(
    const elpis_ecsc_runtime_session *session
);

#ifdef __cplusplus
}
#endif

#endif
