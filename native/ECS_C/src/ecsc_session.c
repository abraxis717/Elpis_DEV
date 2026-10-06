#include "elpis/ecsc_history.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ELPIS_ECSC_SESSION_EVENT_CAPACITY 262144u
#define ELPIS_ECSC_RUNTIME_MAX_ENTITIES (64u)
#define ELPIS_ECSC_RUNTIME_MAX_MAILBOXES (64u)
#define ELPIS_ECSC_RUNTIME_MAX_WATERMARKS (64u)

struct elpis_ecsc_runtime_session {
    elpis_ecsc_log *log;
    int poisoned;

    char genesis_digest[65];
    char history_digest[65];
    char event_digest[65];

    uint64_t logical_clock;
    uint64_t next_founding_index;
    uint64_t mailbox_capacity;
    uint64_t mailbox_default_capacity;
    uint64_t event_count;

    elpis_ecsc_state_entity_view
        entities[ELPIS_ECSC_RUNTIME_MAX_ENTITIES];

    char entity_registry_key
        [ELPIS_ECSC_RUNTIME_MAX_ENTITIES][65];
    char entity_id
        [ELPIS_ECSC_RUNTIME_MAX_ENTITIES][65];
    char entity_founding_digest
        [ELPIS_ECSC_RUNTIME_MAX_ENTITIES][65];
    char entity_state_entity_id
        [ELPIS_ECSC_RUNTIME_MAX_ENTITIES][65];
    char entity_prev_state_digest
        [ELPIS_ECSC_RUNTIME_MAX_ENTITIES][65];
    char entity_state_digest
        [ELPIS_ECSC_RUNTIME_MAX_ENTITIES][65];

    char *entity_label[
        ELPIS_ECSC_RUNTIME_MAX_ENTITIES
    ];

    size_t entity_count;

    elpis_ecsc_state_mailbox_view
        mailboxes[ELPIS_ECSC_RUNTIME_MAX_MAILBOXES];

    char mailbox_key
        [ELPIS_ECSC_RUNTIME_MAX_MAILBOXES][65];
    char mailbox_receiver
        [ELPIS_ECSC_RUNTIME_MAX_MAILBOXES][65];
    uint64_t mailbox_slot_capacity[
        ELPIS_ECSC_RUNTIME_MAX_MAILBOXES
    ];

    size_t mailbox_count;

    elpis_ecsc_state_watermark_view
        watermarks[ELPIS_ECSC_RUNTIME_MAX_WATERMARKS];

    char watermark_sender
        [ELPIS_ECSC_RUNTIME_MAX_WATERMARKS][65];
    uint64_t watermark_sequence[
        ELPIS_ECSC_RUNTIME_MAX_WATERMARKS
    ];

    size_t watermark_count;

    elpis_ecsc_state_root_view root;

    uint8_t *enqueue_event;
    uint8_t *processed_event;
};

static void copy_digest(
    char dst[65],
    const char *src
)
{
    memcpy(dst, src, 64u);
    dst[64] = '\0';
}

static void session_refresh_root(
    elpis_ecsc_runtime_session *session
)
{
    size_t i;

    for (i = 0u; i < session->entity_count; ++i) {
        session->entities[i].registry_key =
            session->entity_registry_key[i];

        session->entities[i].entity_id =
            session->entity_id[i];

        session->entities[i].label =
            session->entity_label[i];

        session->entities[i].founding_digest =
            session->entity_founding_digest[i];

        session->entities[i].state_entity_id =
            session->entity_state_entity_id[i];

        session->entities[i].prev_state_digest =
            session->entity_prev_state_digest[i];

        session->entities[i].state_digest =
            session->entity_state_digest[i];
    }

    for (i = 0u; i < session->mailbox_count; ++i) {
        session->mailboxes[i].mailbox_key =
            session->mailbox_key[i];

        session->mailboxes[i].receiver_entity_id =
            session->mailbox_receiver[i];

        session->mailboxes[i].capacity =
            session->mailbox_slot_capacity[i];

        session->mailboxes[i].contents = NULL;
        session->mailboxes[i].content_count = 0u;
    }

    for (i = 0u; i < session->watermark_count; ++i) {
        session->watermarks[i].sender_entity_id =
            session->watermark_sender[i];

        session->watermarks[i].sequence =
            session->watermark_sequence[i];
    }

    memset(
        &session->root,
        0,
        sizeof session->root
    );

    session->root.genesis_digest =
        session->genesis_digest;

    session->root.history_digest =
        session->history_digest;

    session->root.logical_clock =
        session->logical_clock;

    session->root.mailbox_capacity =
        session->mailbox_capacity;

    session->root.mailbox_default_capacity =
        session->mailbox_default_capacity;

    session->root.next_founding_index =
        session->next_founding_index;

    session->root.entities =
        session->entity_count != 0u
            ? session->entities
            : NULL;

    session->root.entity_count =
        session->entity_count;

    session->root.mailboxes =
        session->mailbox_count != 0u
            ? session->mailboxes
            : NULL;

    session->root.mailbox_count =
        session->mailbox_count;

    session->root.watermarks =
        session->watermark_count != 0u
            ? session->watermarks
            : NULL;

    session->root.watermark_count =
        session->watermark_count;
}

static void session_free(
    elpis_ecsc_runtime_session *session
)
{
    size_t i;

    if (session == NULL) {
        return;
    }

    if (session->log != NULL) {
        elpis_ecsc_log_close(session->log);
        session->log = NULL;
    }

    for (i = 0u; i < session->entity_count; ++i) {
        free(session->entity_label[i]);
        session->entity_label[i] = NULL;
    }

    free(session->enqueue_event);
    free(session->processed_event);

    free(session);
}

static void session_poison(
    elpis_ecsc_runtime_session *session
)
{
    if (session == NULL) {
        return;
    }

    session->poisoned = 1;

    if (session->log != NULL) {
        elpis_ecsc_log_close(session->log);
        session->log = NULL;
    }
}

static int copy_seed_state(
    elpis_ecsc_runtime_session *session,
    const elpis_ecsc_state_root_view *state
)
{
    size_t i;

    if (session == NULL || state == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    if (state->entity_count >
            ELPIS_ECSC_RUNTIME_MAX_ENTITIES ||
        state->mailbox_count >
            ELPIS_ECSC_RUNTIME_MAX_MAILBOXES ||
        state->watermark_count >
            ELPIS_ECSC_RUNTIME_MAX_WATERMARKS) {
        return ELPIS_ECSC_INVALID;
    }

    /*
     * The durable runtime-session handoff is defined only at a quiescent
     * transition boundary. Materialized empty mailboxes are allowed.
     */
    for (i = 0u; i < state->mailbox_count; ++i) {
        if (state->mailboxes[i].content_count != 0u) {
            return ELPIS_ECSC_NOT_READY;
        }
    }

    copy_digest(
        session->genesis_digest,
        state->genesis_digest
    );

    copy_digest(
        session->history_digest,
        state->history_digest
    );

    session->logical_clock =
        state->logical_clock;

    session->next_founding_index =
        state->next_founding_index;

    session->mailbox_capacity =
        state->mailbox_capacity;

    session->mailbox_default_capacity =
        state->mailbox_default_capacity;

    session->entity_count =
        state->entity_count;

    for (i = 0u; i < state->entity_count; ++i) {
        const elpis_ecsc_state_entity_view *src =
            &state->entities[i];

        elpis_ecsc_state_entity_view *dst =
            &session->entities[i];

        size_t label_size =
            src->label_len;

        session->entity_label[i] =
            (char *)malloc(label_size + 1u);

        if (session->entity_label[i] == NULL) {
            return ELPIS_ECSC_IO;
        }

        memcpy(
            session->entity_label[i],
            src->label,
            label_size
        );

        session->entity_label[i][label_size] =
            '\0';

        copy_digest(
            session->entity_registry_key[i],
            src->registry_key
        );

        copy_digest(
            session->entity_id[i],
            src->entity_id
        );

        copy_digest(
            session->entity_founding_digest[i],
            src->founding_digest
        );

        copy_digest(
            session->entity_state_entity_id[i],
            src->state_entity_id
        );

        copy_digest(
            session->entity_prev_state_digest[i],
            src->prev_state_digest
        );

        copy_digest(
            session->entity_state_digest[i],
            src->state_digest
        );

        *dst = *src;

        dst->label_len = label_size;
    }

    session->mailbox_count =
        state->mailbox_count;

    for (i = 0u; i < state->mailbox_count; ++i) {
        copy_digest(
            session->mailbox_key[i],
            state->mailboxes[i].mailbox_key
        );

        copy_digest(
            session->mailbox_receiver[i],
            state->mailboxes[i].receiver_entity_id
        );

        session->mailbox_slot_capacity[i] =
            state->mailboxes[i].capacity;
    }

    session->watermark_count =
        state->watermark_count;

    for (i = 0u; i < state->watermark_count; ++i) {
        copy_digest(
            session->watermark_sender[i],
            state->watermarks[i].sender_entity_id
        );

        session->watermark_sequence[i] =
            state->watermarks[i].sequence;
    }

    session_refresh_root(session);

    return ELPIS_ECSC_OK;
}

static int session_find_entity(
    const elpis_ecsc_runtime_session *session,
    const char entity_id[64],
    size_t *out_index
)
{
    size_t i;

    if (session == NULL ||
        entity_id == NULL ||
        out_index == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    for (i = 0u; i < session->entity_count; ++i) {
        if (memcmp(
                session->entity_id[i],
                entity_id,
                64u
            ) == 0) {
            *out_index = i;
            return ELPIS_ECSC_OK;
        }
    }

    return ELPIS_ECSC_INVALID;
}

static int session_commit_watermark(
    elpis_ecsc_runtime_session *session,
    const char sender_entity_id[64],
    uint64_t sequence
)
{
    size_t i;
    size_t at = 0u;

    if (session == NULL ||
        sender_entity_id == NULL ||
        sequence == 0u) {
        return ELPIS_ECSC_INVALID;
    }

    for (i = 0u; i < session->watermark_count; ++i) {
        int cmp = memcmp(
            session->watermark_sender[i],
            sender_entity_id,
            64u
        );

        if (cmp == 0) {
            session->watermark_sequence[i] =
                sequence;

            return ELPIS_ECSC_OK;
        }

        if (cmp < 0) {
            at = i + 1u;
        } else {
            break;
        }
    }

    if (session->watermark_count >=
            ELPIS_ECSC_RUNTIME_MAX_WATERMARKS) {
        return ELPIS_ECSC_CAPACITY;
    }

    if (at < session->watermark_count) {
        memmove(
            &session->watermark_sender[at + 1u],
            &session->watermark_sender[at],
            (session->watermark_count - at) *
                sizeof session->watermark_sender[0]
        );

        memmove(
            &session->watermark_sequence[at + 1u],
            &session->watermark_sequence[at],
            (session->watermark_count - at) *
                sizeof session->watermark_sequence[0]
        );
    }

    copy_digest(
        session->watermark_sender[at],
        sender_entity_id
    );

    session->watermark_sequence[at] =
        sequence;

    session->watermark_count += 1u;

    return ELPIS_ECSC_OK;
}

static int session_commit_empty_history_mailbox(
    elpis_ecsc_runtime_session *session,
    const char history_entity_id[64]
)
{
    if (session == NULL ||
        history_entity_id == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    if (session->mailbox_count == 0u) {
        copy_digest(
            session->mailbox_key[0],
            history_entity_id
        );

        copy_digest(
            session->mailbox_receiver[0],
            history_entity_id
        );

        session->mailbox_slot_capacity[0] =
            session->mailbox_capacity;

        session->mailbox_count = 1u;

        return ELPIS_ECSC_OK;
    }

    /*
     * runtime_record_plan has the same runtime-history restriction:
     * at most the single history mailbox may be materialized.
     */
    if (session->mailbox_count != 1u ||
        memcmp(
            session->mailbox_key[0],
            history_entity_id,
            64u
        ) != 0 ||
        memcmp(
            session->mailbox_receiver[0],
            history_entity_id,
            64u
        ) != 0) {
        return ELPIS_ECSC_INVALID;
    }

    return ELPIS_ECSC_OK;
}

static int session_install_success(
    elpis_ecsc_runtime_session *session,
    const char sender_entity_id[64],
    const char history_entity_id[64],
    const elpis_ecsc_runtime_record_plan_result *result
)
{
    size_t history_index;
    char computed_root[65];
    int rc;

    rc = session_find_entity(
        session,
        history_entity_id,
        &history_index
    );

    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = session_commit_watermark(
        session,
        sender_entity_id,
        result->sequence
    );

    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = session_commit_empty_history_mailbox(
        session,
        history_entity_id
    );

    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    copy_digest(
        session->entity_prev_state_digest[
            history_index
        ],
        session->entity_state_digest[
            history_index
        ]
    );

    copy_digest(
        session->entity_state_digest[
            history_index
        ],
        result->final_history_state_digest
    );

    session->entities[
        history_index
    ].state_version =
        result->final_history_state_version;

    session->entities[
        history_index
    ].has_delivered = 1u;

    session->entities[
        history_index
    ].delivered =
        result->final_delivered;

    session->logical_clock =
        result->final_logical_clock;

    copy_digest(
        session->history_digest,
        result->final_history_digest
    );

    copy_digest(
        session->event_digest,
        result->processed_event_digest
    );

    session->event_count += 2u;

    session_refresh_root(session);

    rc = elpis_ecsc_state_root_digest(
        &session->root,
        computed_root
    );

    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    if (memcmp(
            computed_root,
            result->final_state_root,
            64u
        ) != 0) {
        return ELPIS_ECSC_CORRUPT;
    }

    if (session->log == NULL ||
        elpis_ecsc_log_frame_count(
            session->log
        ) != session->event_count) {
        return ELPIS_ECSC_CORRUPT;
    }

    return ELPIS_ECSC_OK;
}

int elpis_ecsc_runtime_session_open(
    const char *log_path,
    size_t log_path_len,
    const elpis_ecsc_state_root_view *validated_state,
    const char current_event_digest[64],
    uint64_t validated_event_count,
    elpis_ecsc_runtime_session **out_session
)
{
    elpis_ecsc_runtime_session *session = NULL;
    elpis_ecsc_log_scan scan;
    char source_root[65];
    char copied_root[65];
    int rc;

    if (out_session == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    *out_session = NULL;

    if (log_path == NULL ||
        log_path_len == 0u ||
        validated_state == NULL ||
        current_event_digest == NULL ||
        validated_event_count !=
            validated_state->logical_clock) {
        return ELPIS_ECSC_INVALID;
    }

    rc = elpis_ecsc_state_root_digest(
        validated_state,
        source_root
    );

    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    session = (elpis_ecsc_runtime_session *)
        calloc(1u, sizeof *session);

    if (session == NULL) {
        return ELPIS_ECSC_IO;
    }

    session->enqueue_event =
        (uint8_t *)malloc(
            ELPIS_ECSC_SESSION_EVENT_CAPACITY
        );

    session->processed_event =
        (uint8_t *)malloc(
            ELPIS_ECSC_SESSION_EVENT_CAPACITY
        );

    if (session->enqueue_event == NULL ||
        session->processed_event == NULL) {
        session_free(session);
        return ELPIS_ECSC_IO;
    }

    rc = copy_seed_state(
        session,
        validated_state
    );

    if (rc != ELPIS_ECSC_OK) {
        session_free(session);
        return rc;
    }

    copy_digest(
        session->event_digest,
        current_event_digest
    );

    session->event_count =
        validated_event_count;

    rc = elpis_ecsc_state_root_digest(
        &session->root,
        copied_root
    );

    if (rc != ELPIS_ECSC_OK ||
        memcmp(
            source_root,
            copied_root,
            64u
        ) != 0) {
        session_free(session);
        return ELPIS_ECSC_CORRUPT;
    }

    rc = elpis_ecsc_log_open(
        log_path,
        log_path_len,
        &session->log
    );

    if (rc != ELPIS_ECSC_OK) {
        session_free(session);
        return rc;
    }

    memset(&scan, 0, sizeof scan);

    rc = elpis_ecsc_log_recover_scan(
        session->log,
        &scan
    );

    if (rc != ELPIS_ECSC_OK) {
        session_free(session);
        return rc;
    }

    /*
     * G2B1 accepts only a cold handoff after canonical replay has already
     * recovered any crash residue. Native crash-tail reconciliation is a
     * later restart/recovery phase.
     */
    if (scan.has_incomplete_tail != 0u ||
        scan.total_size != scan.complete_prefix ||
        scan.frame_count != validated_event_count) {
        session_free(session);
        return ELPIS_ECSC_CORRUPT;
    }

    rc = elpis_ecsc_log_finish_recovery(
        session->log,
        scan.complete_prefix
    );

    if (rc != ELPIS_ECSC_OK) {
        session_free(session);
        return rc;
    }

    session_refresh_root(session);

    *out_session = session;

    return ELPIS_ECSC_OK;
}

void elpis_ecsc_runtime_session_close(
    elpis_ecsc_runtime_session *session
)
{
    session_free(session);
}

int elpis_ecsc_runtime_session_record(
    elpis_ecsc_runtime_session *session,
    const char sender_entity_id[64],
    const char history_entity_id[64],
    const void *receipt_payload,
    size_t receipt_payload_size,
    elpis_ecsc_runtime_record_plan_result *out
)
{
    elpis_ecsc_runtime_record_plan_result plan;
    int rc;

    if (session == NULL ||
        sender_entity_id == NULL ||
        history_entity_id == NULL ||
        receipt_payload == NULL ||
        out == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    memset(out, 0, sizeof *out);

    if (session->poisoned != 0 ||
        session->log == NULL) {
        return ELPIS_ECSC_NOT_READY;
    }

    memset(&plan, 0, sizeof plan);

    rc = elpis_ecsc_runtime_record_plan(
        &session->root,
        session->event_digest,
        sender_entity_id,
        history_entity_id,
        receipt_payload,
        receipt_payload_size,
        session->enqueue_event,
        ELPIS_ECSC_SESSION_EVENT_CAPACITY,
        session->processed_event,
        ELPIS_ECSC_SESSION_EVENT_CAPACITY,
        &plan
    );

    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = elpis_ecsc_log_append_event_bytes(
        session->log,
        session->enqueue_event,
        plan.enqueue_event_size
    );

    if (rc == ELPIS_ECSC_APPEND_ROLLED_BACK) {
        return rc;
    }

    if (rc != ELPIS_ECSC_OK) {
        session_poison(session);
        return rc;
    }

    rc = elpis_ecsc_log_append_event_bytes(
        session->log,
        session->processed_event,
        plan.processed_event_size
    );

    if (rc == ELPIS_ECSC_APPEND_ROLLED_BACK) {
        /*
         * The enqueue is known durable and the processed event is known
         * absent. The durable history is therefore a valid but non-quiescent
         * prefix. Reopen/replay must reconcile it.
         */
        session_poison(session);
        return ELPIS_ECSC_PARTIAL_COMMIT;
    }

    if (rc != ELPIS_ECSC_OK) {
        session_poison(session);
        return rc;
    }

    rc = session_install_success(
        session,
        sender_entity_id,
        history_entity_id,
        &plan
    );

    if (rc != ELPIS_ECSC_OK) {
        /*
         * Both frames are already durable. Any in-memory projection mismatch
         * is fail-stop: never continue from a state that may diverge from
         * durable authority.
         */
        session_poison(session);
        return rc;
    }

    *out = plan;

    return ELPIS_ECSC_OK;
}

int elpis_ecsc_runtime_session_state_root_digest(
    const elpis_ecsc_runtime_session *session,
    char out_digest[65]
)
{
    if (session == NULL ||
        out_digest == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    if (session->poisoned != 0 ||
        session->log == NULL) {
        return ELPIS_ECSC_NOT_READY;
    }

    return elpis_ecsc_state_root_digest(
        &session->root,
        out_digest
    );
}

uint64_t elpis_ecsc_runtime_session_event_count(
    const elpis_ecsc_runtime_session *session
)
{
    if (session == NULL) {
        return 0u;
    }

    return session->event_count;
}
