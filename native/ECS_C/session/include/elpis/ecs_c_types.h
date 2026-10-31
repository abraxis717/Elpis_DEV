/*
 * ECS_C shared types: the entity row, the evolution authority, entity kinds and lifecycles, capacity bounds and the
 * refusal codes of the ECS_C transition law. Owned by the volatile session (include/elpis/ecs_c_session.h,
 * docs/ECS_C_SESSION.md); included by the retiring durable authority's header and by the relational view, so none of
 * them depends on another's ABI. Integers are little-endian; digests are 32 bytes.
 */
#ifndef ELPIS_ECS_C_TYPES_H
#define ELPIS_ECS_C_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ELPIS_ECS_C_MIN_CAPACITY 2u
#define ELPIS_ECS_C_MAX_CAPACITY 4096u
#define ELPIS_ECS_C_MAX_PER_TURN 256u
#define ELPIS_ECS_C_EVENT_BYTES 512u
#define ELPIS_ECS_C_PAYLOAD_AT 80u

/* Codes of the transition law (every ECS_C authority, durable or volatile, returns them). */
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
#define ELPIS_ECS_C_CORRUPT 28
/* Codes of the volatile session only. */
#define ELPIS_ECS_C_SESSION_MISMATCH 34    /* a coordinate of another session, or another instance of one */
#define ELPIS_ECS_C_PERSISTENCE_RETIRED 35 /* legacy import or a reconciliation: operations of storage, retired */
#define ELPIS_ECS_C_NOMEM 36               /* session memory could not be reserved; nothing was created */
#define ELPIS_ECS_C_ENTROPY_UNAVAILABLE 37 /* no nonce given and the kernel refused one; nothing was created */

/* Entity kinds and lifecycles. FOUNDED -> ACTIVE <-> DORMANT; any non-terminal -> TERMINATED (final). */
#define ELPIS_ECS_C_OPERATOR 1u
#define ELPIS_ECS_C_RUNTIME 2u
#define ELPIS_ECS_C_SOURCE 3u
#define ELPIS_ECS_C_FOUNDED 1u
#define ELPIS_ECS_C_ACTIVE 2u
#define ELPIS_ECS_C_DORMANT 3u
#define ELPIS_ECS_C_TERMINATED 4u

/* The evolution authority (the continuity register's layout). */
typedef struct {
    uint64_t revision;
    uint8_t head[32];
    uint8_t pending;
    uint8_t reserved[7];
    uint8_t assertion[32];
} elpis_ecs_c_evolution;

/* One entity row. */
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

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(elpis_ecs_c_evolution) == 80, "elpis_ecs_c_evolution layout");
_Static_assert(sizeof(elpis_ecs_c_entity) == 152, "elpis_ecs_c_entity layout");
#endif

#ifdef __cplusplus
}
#endif

#endif
