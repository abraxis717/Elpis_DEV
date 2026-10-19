/*
 * ECS_C authority: the bounded, authenticated continuity and entity authority (C ABI v1).
 *
 * Normative description: docs/ECS_C_AUTHORITY.md. Implementation: native/ECS_C/authority (Rust).
 *
 * One deployment is two directories:
 *   state directory      authority.a, authority.b   fixed-size records, R(capacity) bytes each
 *   authority directory  deployment.key (0400), witness.a, witness.b (240 bytes each)
 *                        owner-only, never inside the state directory (and vice versa)
 * Nothing is created, renamed or removed after provisioning; every transition rewrites one record slot and one
 * witness slot in place (R + 240 bytes, two fdatasync calls), whatever the length of the history.
 *
 * Every function returns ELPIS_ECS_C_OK (0) or a positive stable code (below). A handle is owned by one caller at
 * a time; calls on one handle must not overlap. Digests are 32 bytes; an event is 512 bytes; integers inside
 * records and events are little-endian.
 */
#ifndef ELPIS_ECS_C_AUTHORITY_H
#define ELPIS_ECS_C_AUTHORITY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ELPIS_ECS_C_ABI_VERSION 1u

#define ELPIS_ECS_C_MIN_CAPACITY 2u
#define ELPIS_ECS_C_MAX_CAPACITY 4096u
#define ELPIS_ECS_C_MAX_PER_TURN 256u
#define ELPIS_ECS_C_EVENT_BYTES 512u
#define ELPIS_ECS_C_PAYLOAD_AT 80u
#define ELPIS_ECS_C_LEGACY_RECORD_BYTES 176u
#define ELPIS_ECS_C_WITNESS_BYTES 240u

/* Codes. */
#define ELPIS_ECS_C_OK 0
#define ELPIS_ECS_C_INVALID 1
#define ELPIS_ECS_C_STALE 2
#define ELPIS_ECS_C_UNAUTHORIZED 3
#define ELPIS_ECS_C_UNKNOWN_ENTITY 4
#define ELPIS_ECS_C_LIFECYCLE 5
#define ELPIS_ECS_C_DUPLICATE 6
#define ELPIS_ECS_C_SEQUENCE_GAP 7
#define ELPIS_ECS_C_CAPACITY 8
#define ELPIS_ECS_C_EXHAUSTED 9
#define ELPIS_ECS_C_SOURCE_STALE 10
#define ELPIS_ECS_C_SCOPE_CONFLICT 11
#define ELPIS_ECS_C_UNANCHORED 12
#define ELPIS_ECS_C_ALREADY_ANCHORED 13
#define ELPIS_ECS_C_STATE_MISMATCH 14
#define ELPIS_ECS_C_EVOLUTION_MISMATCH 15
#define ELPIS_ECS_C_EVOLUTION_PENDING 16
#define ELPIS_ECS_C_EVOLUTION_NOT_PENDING 17
#define ELPIS_ECS_C_LEGACY_REFUSED 18
#define ELPIS_ECS_C_PUBLICATION_REFUSED 19   /* nothing changed */
#define ELPIS_ECS_C_PUBLICATION_UNCERTAIN 20 /* handle closed; reopen resolves to exactly one authority */
#define ELPIS_ECS_C_LOCKED 21
#define ELPIS_ECS_C_IO 22
#define ELPIS_ECS_C_PATH 23
#define ELPIS_ECS_C_UNPROTECTED 24
#define ELPIS_ECS_C_NOT_PROVISIONED 25
#define ELPIS_ECS_C_ALREADY_PROVISIONED 26
#define ELPIS_ECS_C_PROVISION_INCOMPLETE 27
#define ELPIS_ECS_C_CORRUPT 28
#define ELPIS_ECS_C_RECOVERY_REFUSED 29 /* the recovery report names the disposition */
#define ELPIS_ECS_C_RECONCILE_REQUIRED 30
#define ELPIS_ECS_C_NOT_RECONCILABLE 31
#define ELPIS_ECS_C_OPEN 32
#define ELPIS_ECS_C_CLOSED 33
#define ELPIS_ECS_C_TESTING_PROCESS_DEATH 255 /* testing library only */

/* Recovery dispositions: 1..5 open (4 restricted), 10..13 open reconcile-only, 20..23 refuse to open. */
#define ELPIS_ECS_C_CLEAN 1u
#define ELPIS_ECS_C_TRAILING_INCOMPLETE 2u
#define ELPIS_ECS_C_INTERRUPTED_COMPLETED 3u
#define ELPIS_ECS_C_RESTRICTED_UNAUTHENTICATED 4u
#define ELPIS_ECS_C_PREDECESSOR_UNAVAILABLE 5u
#define ELPIS_ECS_C_ROLLBACK 10u
#define ELPIS_ECS_C_CORRUPT_ACKNOWLEDGED 11u
#define ELPIS_ECS_C_UNAUTHENTICATED_ACKNOWLEDGED 12u
#define ELPIS_ECS_C_WITNESS_MISSING 13u
#define ELPIS_ECS_C_CONFLICT 20u
#define ELPIS_ECS_C_WITNESS_INCONSISTENT 21u
#define ELPIS_ECS_C_REPLAY_MISMATCH 22u
#define ELPIS_ECS_C_UNRECOVERABLE 23u

/* Modes: what an open handle accepts. */
#define ELPIS_ECS_C_MODE_NORMAL 0u
#define ELPIS_ECS_C_MODE_RESTRICTED 1u      /* reads; only RECONCILE_UNAUTHENTICATED for the observed slot */
#define ELPIS_ECS_C_MODE_ROLLBACK 2u        /* a candidate; only RECONCILE_ROLLBACK of the observed pin */
#define ELPIS_ECS_C_MODE_WITNESS_MISSING 3u /* a candidate; only RECONCILE_REPIN of the observed witness classes */

/* Slot classes (records and witnesses). */
#define ELPIS_ECS_C_SLOT_EMPTY 0u
#define ELPIS_ECS_C_SLOT_TORN 1u
#define ELPIS_ECS_C_SLOT_UNAUTHENTICATED 2u
#define ELPIS_ECS_C_SLOT_INVALID 3u
#define ELPIS_ECS_C_SLOT_VALID 4u

/* Entity kinds and lifecycles. FOUNDED -> ACTIVE <-> DORMANT; any non-terminal -> TERMINATED (final). */
#define ELPIS_ECS_C_OPERATOR 1u
#define ELPIS_ECS_C_RUNTIME 2u
#define ELPIS_ECS_C_SOURCE 3u
#define ELPIS_ECS_C_FOUNDED 1u
#define ELPIS_ECS_C_ACTIVE 2u
#define ELPIS_ECS_C_DORMANT 3u
#define ELPIS_ECS_C_TERMINATED 4u

/*
 * Events (512 bytes): kind u16 @0, reserved zero @2..8, sender id @8, sender sequence u64 @40 (the sender's
 * watermark + 1), receiver id @48 (entity-scoped kinds only, else zero), payload @80 (unused bytes zero).
 * Offsets below are relative to the payload. Sender column: who may send it.
 *
 *  kind                          sender            receiver      payload
 *  2  ENTITY_FOUND               operator          the new id    kind u8 @0, per-turn bound u32 @4,
 *                                                                source version u64 @8, label @16, scope @48,
 *                                                                source authority u32 @80 (sources: unique among
 *                                                                present source rows; others 0)
 *  3  ENTITY_LIFECYCLE           operator          the entity    from u8 @0, to u8 @1
 *  4  SOURCE_VERSION             operator          the source    expected version u64 @0, new version u64 @8,
 *                                                                per-turn bound u32 @16, scope @24
 *  5  ANCHOR                     runtime           -             format u8 @0 (0 ELPISGK1, 1 ELPISEC1), identity @8
 *  6  PUBLISH                    runtime           -             format u8 @0, before @8, after @40,
 *                                                                evidence count u64 @72, evidence digest @80
 *  7  IMPORT_COMPOSITE           runtime           -             K1 identity @0, composite identity @32
 *  8  EVOLUTION_RESERVE          runtime/operator  -             expected evolution @0 (revision u64, head @8,
 *                                                                pending u8 @40, assertion @48), assertion @80
 *  9  EVOLUTION_FINALIZE         runtime/operator  -             expected evolution @0, result @80
 *  10 IMPORT_LEGACY              operator          -             a continuity register record (176 bytes) @0
 *  11 RECONCILE_ROLLBACK         operator          -             abandoned generation u64 @0, class u8 @8,
 *                                                                abandoned record digest @16
 *  12 RECONCILE_UNAUTHENTICATED  operator          -             slot u8 @0, class u8 @1 (2), observed digest @8
 *  13 RECONCILE_REPIN            operator          -             witness classes u8 @0, @1, pinned digest @8
 *  14 ENTITY_RECLAIM             operator          the entity    (none; the entity must be TERMINATED)
 * GENESIS (kind 1) is written by provisioning only.
 */

typedef struct elpis_ecs_c_authority elpis_ecs_c_authority;

typedef struct {
    uint32_t capacity;
    uint32_t reserved;
    uint64_t record_bytes;
    uint8_t lineage[32];
    uint8_t deployment[32];
    uint8_t record_digest[32];
    uint8_t state_root[32];
} elpis_ecs_c_provisioned;

typedef struct {
    uint32_t disposition;
    uint32_t mode;
    uint8_t slot_class[2];
    uint8_t witness_class[2];
    uint8_t replay_verified;
    uint8_t witness_completed;
    uint8_t evolution_pending; /* an outstanding cross-domain operation: never cleared automatically */
    uint8_t reserved;
    uint64_t slot_generation[2]; /* untrusted header claims, 0 when unreadable */
    uint64_t witness_generation;
    uint64_t current_generation;
    uint64_t reads; /* recovery work: reads, bytes read, transitions replayed */
    uint64_t read_bytes;
    uint64_t replays;
    uint8_t observed[2][32]; /* SHA-256 of each slot's raw bytes */
} elpis_ecs_c_report;

/* The evolution authority, laid out as continuity's (include/elpis/continuity.h). */
typedef struct {
    uint64_t revision;
    uint8_t head[32];
    uint8_t pending;
    uint8_t reserved[7];
    uint8_t assertion[32];
} elpis_ecs_c_evolution;

typedef struct {
    uint64_t generation;
    uint64_t record_bytes;
    uint32_t capacity;
    uint32_t entity_count;
    uint32_t mode;
    uint8_t anchored;
    uint8_t state_format;
    uint8_t legacy_imported;
    uint8_t legacy_format;
    uint8_t lineage[32];
    uint8_t deployment[32];
    uint8_t record_digest[32];
    uint8_t state_root[32];
    uint8_t identity[32];
    uint64_t publications;
    uint64_t last_publication_generation;
    elpis_ecs_c_evolution evolution;
    uint8_t evolution_digest[32];
    uint8_t receipt_head[32];
    uint64_t receipt_count;
    uint8_t evidence_head[32];
    uint64_t evidence_count;
    uint8_t reconcile_head[32];
    uint64_t reconcile_count;
    uint8_t reclaimed_head[32];
    uint64_t reclaimed_count;
    uint64_t next_founding_index;
    uint64_t legacy_generation;
    uint8_t legacy_digest[32];
} elpis_ecs_c_status;

typedef struct {
    uint8_t id[32];
    uint8_t kind;
    uint8_t lifecycle;
    uint8_t reserved[2];
    uint32_t per_turn_limit;
    uint64_t founding_index;
    uint64_t founded_generation;
    uint64_t last_generation;
    uint64_t sender_watermark;
    uint64_t source_version;
    uint8_t label[32];
    uint8_t scope[32];
    uint32_t source_authority; /* sources: the 32-bit source_authority its typed memory observations record */
    uint32_t reserved2;
} elpis_ecs_c_entity;

typedef struct {
    uint64_t generation;
    uint64_t predecessor_generation;
    uint64_t receipt_count;
    uint8_t founded_present;
    uint8_t reserved[7];
    uint8_t receipt[32];
    uint8_t record_digest[32];
    uint8_t state_root[32];
    uint8_t receipt_head[32];
    uint8_t founded[32];
} elpis_ecs_c_receipt;

/* The one reconciliation the open authority accepts in its mode (elpis_ecs_c_read_reconciliation): the event kind
 * and the payload fields it must name exactly. In the normal mode kind is 0 (none is accepted). */
typedef struct {
    uint32_t mode;                 /* ELPIS_ECS_C_MODE_* */
    uint16_t kind;                 /* 11 RECONCILE_ROLLBACK, 12 RECONCILE_UNAUTHENTICATED, 13 RECONCILE_REPIN; or 0 */
    uint8_t slot;                  /* RECONCILE_UNAUTHENTICATED: payload @0 */
    uint8_t class_code;            /* RECONCILE_UNAUTHENTICATED: payload @1 (2); RECONCILE_ROLLBACK: payload @8 */
    uint8_t witness_classes[2];    /* RECONCILE_REPIN: payload @0, @1 */
    uint8_t reserved[6];
    uint64_t abandoned_generation; /* RECONCILE_ROLLBACK: payload @0 */
    uint8_t digest[32];            /* RECONCILE_ROLLBACK: the abandoned record @16; RECONCILE_UNAUTHENTICATED: the
                                      observed slot bytes @8; RECONCILE_REPIN: the current record @8 */
} elpis_ecs_c_reconciliation;

typedef struct {
    uint64_t opens, creates, preads, pread_bytes, pwrites, pwrite_bytes, data_syncs, dir_syncs, renames, unlinks,
        replays, key_reads;
} elpis_ecs_c_counters;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(elpis_ecs_c_provisioned) == 144, "elpis_ecs_c_provisioned layout");
_Static_assert(sizeof(elpis_ecs_c_report) == 136, "elpis_ecs_c_report layout");
_Static_assert(sizeof(elpis_ecs_c_status) == 528, "elpis_ecs_c_status layout");
_Static_assert(sizeof(elpis_ecs_c_entity) == 152, "elpis_ecs_c_entity layout");
_Static_assert(sizeof(elpis_ecs_c_receipt) == 192, "elpis_ecs_c_receipt layout");
_Static_assert(sizeof(elpis_ecs_c_counters) == 96, "elpis_ecs_c_counters layout");
_Static_assert(sizeof(elpis_ecs_c_reconciliation) == 56, "elpis_ecs_c_reconciliation layout");
#endif

uint32_t elpis_ecs_c_abi_version(void);
/* Bytes of one record slot for `capacity`, or 0 outside the provisionable range. */
uint64_t elpis_ecs_c_record_bytes(uint32_t capacity);
/* The stable name of a code ("ECS_C_OK" for 0), or NULL. Static storage. */
const char *elpis_ecs_c_code_name(int32_t code);

/* Provision a new deployment (fresh deployment key, lineage, genesis record and witness). `out` is optional. */
int32_t elpis_ecs_c_provision(const char *state_dir, size_t state_dir_len, const char *authority_dir,
                              size_t authority_dir_len, uint32_t capacity, const uint8_t operator_label[32],
                              const uint8_t runtime_label[32], elpis_ecs_c_provisioned *out);
/* Open and recover. On success *out is a handle (possibly restricted or reconcile-only: see report->mode). On
 * ELPIS_ECS_C_RECOVERY_REFUSED no handle is returned and the report names the disposition. `report` is optional. */
int32_t elpis_ecs_c_open(const char *state_dir, size_t state_dir_len, const char *authority_dir,
                         size_t authority_dir_len, elpis_ecs_c_authority **out, elpis_ecs_c_report *report);
void elpis_ecs_c_close(elpis_ecs_c_authority *authority);

int32_t elpis_ecs_c_read_status(const elpis_ecs_c_authority *authority, elpis_ecs_c_status *out);
int32_t elpis_ecs_c_read_report(const elpis_ecs_c_authority *authority, elpis_ecs_c_report *out);
int32_t elpis_ecs_c_read_reconciliation(const elpis_ecs_c_authority *authority, elpis_ecs_c_reconciliation *out);
int32_t elpis_ecs_c_entity_at(const elpis_ecs_c_authority *authority, uint32_t index, elpis_ecs_c_entity *out);
int32_t elpis_ecs_c_entity_find(const elpis_ecs_c_authority *authority, const uint8_t id[32],
                                elpis_ecs_c_entity *out);
/* The present source row owning a memory source-authority identifier, or UNKNOWN_ENTITY. */
int32_t elpis_ecs_c_source_for_authority(const elpis_ecs_c_authority *authority, uint32_t source_authority,
                                         elpis_ecs_c_entity *out);
/* The identity the next ENTITY_FOUND of `kind` and `label` founds (the receiver that event must name). */
int32_t elpis_ecs_c_next_entity_id(const elpis_ecs_c_authority *authority, uint8_t kind, const uint8_t label[32],
                                   uint8_t out[32]);
/* Typed-evidence admission (pure): OK with the per-turn bound, or UNKNOWN_ENTITY, UNAUTHORIZED (not a source),
 * LIFECYCLE (not ACTIVE), SOURCE_STALE, SCOPE_CONFLICT; RECONCILE_REQUIRED for a candidate. */
int32_t elpis_ecs_c_check_evidence(const elpis_ecs_c_authority *authority, const uint8_t source[32],
                                   uint64_t version, const uint8_t scope[32], uint32_t *out_per_turn_limit);
/* Compare-and-publish one event: STALE unless (expected_generation, expected_state_root) are current; validated
 * by the transition law; durable and witnessed before OK. `out` is optional. */
int32_t elpis_ecs_c_transition(elpis_ecs_c_authority *authority, uint64_t expected_generation,
                               const uint8_t expected_state_root[32], const uint8_t event[512],
                               elpis_ecs_c_receipt *out);

/* Read the current record of a standalone continuity directory for IMPORT_LEGACY (never written). */
int32_t elpis_ecs_c_legacy_record(const char *continuity_dir, size_t len, uint8_t out[176]);
/* The next receipt chain head (pure): verify kept receipts against receipt_head. */
int32_t elpis_ecs_c_receipt_chain(const uint8_t head[32], const uint8_t receipt[32], uint8_t out[32]);
/* elpis.continuity.evolution-authority.v2 of an evolution authority (pure; byte-identical to continuity's). */
int32_t elpis_ecs_c_evolution_digest(const elpis_ecs_c_evolution *evolution, uint8_t out[32]);

/* libelpis_ecs_c_authority_testing.so only. Actions: 0 clear; 1 die at step `arg`; record slot 2 write fails,
 * 3 torn write of `arg` bytes then refusal, 4 torn write then death, 5 sync fails with the bytes lost, 6 sync fails
 * but durable; 7..11 the same for the witness. `publication` n >= 1 is the n-th transition from now. */
int32_t elpis_ecs_c_testing_arm(elpis_ecs_c_authority *authority, uint64_t publication, uint32_t action,
                                uint64_t arg);
int32_t elpis_ecs_c_testing_counters(elpis_ecs_c_authority *authority, int32_t reset, elpis_ecs_c_counters *out);

#ifdef __cplusplus
}
#endif

#endif
