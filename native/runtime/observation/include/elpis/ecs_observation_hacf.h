#ifndef ELPIS_ECS_OBSERVATION_HACF_H
#define ELPIS_ECS_OBSERVATION_HACF_H

#include "elpis/ecs_memory.h"
#include "elpis_semantic/identity.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Typed HACF -> ECS memory observation path (runtime composition layer; docs/ECS_MEMORY.md, "HACF observations").
 * It lives in the runtime subsystem because it joins two subsystems that may not depend on each other: structure
 * (HACF, the owner of hyperedge and assertion identity and validation) and ECS (the memory, which depends on the
 * substrate only). It produces an observation; it records nothing, learns nothing and holds no state. Its output
 * reaches the ECS only through a managed turn (elpis_runtime_turn_observe on a composite substrate). It is the
 * research MF-R0 adapter (research/ecs_memory_mf_r0, ecs_mem_hacf.c) under the canonical prefix, unchanged in law.
 *
 * Structure owns identity and validation: this adapter calls structure's own validators and identity functions
 * and refuses anything they refuse or anything whose stored identity differs from the recomputed one. It maps a
 * validated hyperedge and an assertion that names it to one typed observation, losslessly for the fields it
 * accepts: relation type = hyperedge type, payload = payload digest, participants = (incidence role, ordinal,
 * node identity), provenance = assertion provenance digest, source authority = assertion authority, and
 * structure_ref = the hyperedge identity. elpis_ecs_obs_hacf_hyperedge inverts the map from a resident slot and
 * recomputes the identity, so retention of the exact structural identity is testable.
 *
 * UNDEFINED in the repository, therefore refused or supplied explicitly (never inferred):
 *   - observation polarity: HACF assertion flags (mask 0x03) have no named meaning; the caller supplies
 *     polarity, and assertion flags must be zero;
 *   - hyperedge semantic flag 0x01 and participant flag QUERY: refused (QUERY marks query-local overlay
 *     participants, which are not persistent evidence);
 *   - participant counts of 0 or above ELPIS_ECS_MEM_MAX_PARTICIPANTS, and two participants sharing (role, ordinal):
 *     not representable as one ECS memory observation, refused.
 * A mapped observation is evidence to be recorded, not a truth and not an authorization to learn.
 */

typedef enum {
    ELPIS_ECS_OBS_HACF_OK = 0,
    ELPIS_ECS_OBS_HACF_INVALID = -1,       /* NULL argument or polarity outside elpis_ecs_mem_polarity */
    ELPIS_ECS_OBS_HACF_STRUCTURE = -2,     /* structure's validator refused the hyperedge or the assertion */
    ELPIS_ECS_OBS_HACF_IDENTITY = -3,      /* a stored identity differs from the recomputed one, or the assertion does
                                        not assert this hyperedge */
    ELPIS_ECS_OBS_HACF_NONCANONICAL = -4,  /* participants are not in structure's canonical order */
    ELPIS_ECS_OBS_HACF_UNSUPPORTED = -5    /* flags with undefined meaning, query-local participants, a participant
                                        count or (role, ordinal) multiplicity the memory cannot represent */
} elpis_ecs_obs_hacf_status;

elpis_ecs_obs_hacf_status elpis_ecs_obs_hacf_observation(const elpis_semantic_hyperedge_v1 *edge,
                                             const elpis_semantic_assertion_v1 *assertion, uint32_t polarity,
                                             uint64_t tick, elpis_ecs_mem_observation *out);

/* Rebuild the hyperedge a resident slot was observed from (identity recomputed by structure). The rebuilt
 * identity equals the slot's structure_ref exactly when the slot was observed through this adapter. */
elpis_ecs_obs_hacf_status elpis_ecs_obs_hacf_hyperedge(const elpis_ecs_mem_slot *slot, elpis_semantic_hyperedge_v1 *out);

#ifdef __cplusplus
}
#endif
#endif
