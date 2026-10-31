/*
 * ECS_C volatile session authority (ECSC_VOLATILE_SESSION_R0), C ABI v1. Normative description:
 * docs/ECS_C_SESSION.md. Implementation: native/ECS_C/session (Rust, no_std), libelpis_ecs_c_session.so.
 *
 * The ECS_C authority's logic as bounded state in volatile memory only. A session starts at a fresh genesis (the
 * founding operator and runtime, ACTIVE), evolves only by compare-and-publish transitions of the one transition law,
 * and is gone when it is destroyed or its process ends. Nothing is written, recovered, reconciled or resumed: there
 * is no path, file, directory or descriptor anywhere in this ABI, and the library imports no file or I/O function
 * (only malloc, free, abort, the mem* functions and getrandom). A new session is a new genesis.
 *
 *   Staged transitions. A transition is validated, applied to a staged copy and its state root computed before
 *   anything is published; a refusal leaves the session exactly as it was (only its refusal counter moves).
 *   Bounded memory. A session reserves elpis_ecs_c_session_bytes(capacity) bytes at creation (refused NOMEM, never an
 *   abort) and allocates nothing afterwards.
 *   Identity. A coordinate names the session (its lineage, from its nonce and capacity), the instance (unique in the
 *   process), the generation and the state root: another session's coordinate is SESSION_MISMATCH, an old one STALE.
 *   Retired. Events 10 IMPORT_LEGACY and 11..13 RECONCILE_* (operations of durable storage) are PERSISTENCE_RETIRED.
 *
 * Every function returns ELPIS_ECS_C_OK (0) or a positive code (elpis/ecs_c_types.h) and writes its outputs only on
 * success (elpis_ecs_c_session_create excepted: it clears *out first, so a refused creation leaves it NULL). A handle
 * is owned by one caller at a time; calls on one handle must not overlap. Events are the ECS_C authority's 512-byte
 * events (docs/ECS_C_AUTHORITY.md, "Entities, events and the transition law"), GENESIS aside.
 */
#ifndef ELPIS_ECS_C_SESSION_H
#define ELPIS_ECS_C_SESSION_H

#include <stddef.h>
#include <stdint.h>

#include "elpis/ecs_c_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ELPIS_ECS_C_SESSION_ABI_VERSION 1u

/* A zero nonce asks the session to draw one from the kernel's random source (getrandom(2): a system call, no file). */
typedef struct {
    uint32_t abi_version; /* ELPIS_ECS_C_SESSION_ABI_VERSION */
    uint32_t capacity;    /* entity rows, 2..4096 (the founding operator and runtime occupy two) */
    uint8_t nonce[32];
    uint8_t operator_label[32];
    uint8_t runtime_label[32];
} elpis_ecs_c_session_config;

typedef struct {
    uint8_t session[32];  /* the session's lineage */
    uint64_t instance;    /* unique among the process's sessions, never 0 */
    uint64_t generation;
    uint8_t state_root[32];
} elpis_ecs_c_session_coordinate;

typedef struct {
    uint64_t generation;
    uint64_t predecessor_generation;
    uint64_t receipt_count;
    uint8_t founded_present;
    uint8_t reserved[7];
    uint8_t receipt[32];
    uint8_t commit[32];       /* the in-memory digest of the new generation */
    uint8_t state_root[32];
    uint8_t receipt_head[32];
    uint8_t founded[32];      /* the identity an ENTITY_FOUND founded (founded_present) */
} elpis_ecs_c_session_receipt;

typedef struct {
    uint32_t capacity;
    uint32_t entity_count;
    uint64_t instance;
    uint64_t generation;
    uint8_t session[32];
    uint8_t state_root[32];
    uint8_t commit[32];
    uint8_t anchored;     /* the in-session cognition lineage */
    uint8_t state_format; /* 0 ELPISGK1, 1 ELPISEC1 */
    uint8_t reserved[6];
    uint8_t identity[32];
    uint64_t publications;
    uint64_t last_publication_generation;
    elpis_ecs_c_evolution evolution;
    uint8_t evolution_digest[32];
    uint8_t receipt_head[32];
    uint64_t receipt_count;
    uint8_t evidence_head[32];
    uint64_t evidence_count;
    uint8_t reclaimed_head[32];
    uint64_t reclaimed_count;
    uint64_t next_founding_index;
    uint64_t volatile_bytes; /* constant for the session's life */
    uint64_t transitions;    /* accepted (telemetry) */
    uint64_t refusals;       /* refused (telemetry) */
} elpis_ecs_c_session_status;

typedef struct elpis_ecs_c_session elpis_ecs_c_session;

uint32_t elpis_ecs_c_session_abi_version(void);
/* The stable name of a code ("ECS_C_OK" for 0), or NULL (storage codes a session never returns included). */
const char *elpis_ecs_c_session_code_name(int32_t code);
/* Exact bytes a session of `capacity` reserves for its life, or 0 outside 2..4096. */
uint64_t elpis_ecs_c_session_bytes(uint32_t capacity);

/* *out is set to NULL first; on OK it holds the new session. INVALID (a NULL argument, a wrong abi_version, a
 * capacity outside 2..4096), NOMEM or ENTROPY_UNAVAILABLE create nothing. */
int32_t elpis_ecs_c_session_create(const elpis_ecs_c_session_config *config, elpis_ecs_c_session **out);
/* Frees the session (its last state image cleared first). NULL is ignored. */
void elpis_ecs_c_session_destroy(elpis_ecs_c_session *session);

int32_t elpis_ecs_c_session_read_coordinate(const elpis_ecs_c_session *session, elpis_ecs_c_session_coordinate *out);
int32_t elpis_ecs_c_session_read_status(const elpis_ecs_c_session *session, elpis_ecs_c_session_status *out);
int32_t elpis_ecs_c_session_entity_at(const elpis_ecs_c_session *session, uint32_t index, elpis_ecs_c_entity *out);
int32_t elpis_ecs_c_session_entity_find(const elpis_ecs_c_session *session, const uint8_t id[32],
                                        elpis_ecs_c_entity *out);
/* The present source row owning a memory source-authority identifier, or UNKNOWN_ENTITY. */
int32_t elpis_ecs_c_session_source_for_authority(const elpis_ecs_c_session *session, uint32_t source_authority,
                                                 elpis_ecs_c_entity *out);
/* The identity the next ENTITY_FOUND of `kind` and `label` founds (the receiver that event must name). */
int32_t elpis_ecs_c_session_next_entity_id(const elpis_ecs_c_session *session, uint8_t kind, const uint8_t label[32],
                                           uint8_t out[32]);
/* Typed-evidence admission (pure): OK with the per-turn bound, or UNKNOWN_ENTITY, UNAUTHORIZED (not a source),
 * LIFECYCLE (not ACTIVE), SOURCE_STALE, SCOPE_CONFLICT. */
int32_t elpis_ecs_c_session_check_evidence(const elpis_ecs_c_session *session, const uint8_t source[32],
                                           uint64_t version, const uint8_t scope[32], uint32_t *out_per_turn_limit);
/* Compare-and-publish one event, all or nothing: SESSION_MISMATCH or STALE unless `expected` is the current
 * coordinate; then the transition law. `out` is optional. */
int32_t elpis_ecs_c_session_transition(elpis_ecs_c_session *session, const elpis_ecs_c_session_coordinate *expected,
                                       const uint8_t event[512], elpis_ecs_c_session_receipt *out);

/* The next receipt chain head (pure): verify kept receipts against a session's receipt_head. */
int32_t elpis_ecs_c_session_receipt_chain(const uint8_t head[32], const uint8_t receipt[32], uint8_t out[32]);
/* elpis.continuity.evolution-authority.v2 of an evolution authority (pure). */
int32_t elpis_ecs_c_session_evolution_digest(const elpis_ecs_c_evolution *evolution, uint8_t out[32]);

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(elpis_ecs_c_session_config) == 104, "elpis_ecs_c_session_config layout");
_Static_assert(sizeof(elpis_ecs_c_session_coordinate) == 80, "elpis_ecs_c_session_coordinate layout");
_Static_assert(sizeof(elpis_ecs_c_session_receipt) == 192, "elpis_ecs_c_session_receipt layout");
_Static_assert(sizeof(elpis_ecs_c_session_status) == 440, "elpis_ecs_c_session_status layout");
#endif

#ifdef __cplusplus
}
#endif

#endif
