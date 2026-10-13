#ifndef ELPIS_ECS_MEM_HACF_H
#define ELPIS_ECS_MEM_HACF_H

#include "ecs_mem.h"
#include "elpis_semantic/identity.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Typed HACF -> ECS Memory MF-R0 observation adapter (RESEARCH_ONLY, NO_RUNTIME_AUTHORITY).
 *
 * Structure owns identity and validation: this adapter calls structure's own validators and identity functions
 * and refuses anything they refuse or anything whose stored identity differs from the recomputed one. It maps a
 * validated hyperedge and an assertion that names it to one typed observation, losslessly for the fields it
 * accepts: relation type = hyperedge type, payload = payload digest, participants = (incidence role, ordinal,
 * node identity), provenance = assertion provenance digest, source authority = assertion authority, and
 * structure_ref = the hyperedge identity. ecs_mem_hacf_hyperedge inverts the map from a resident slot and
 * recomputes the identity, so retention of the exact structural identity is testable.
 *
 * UNDEFINED in the repository, therefore refused or supplied explicitly (never inferred):
 *   - observation polarity: HACF assertion flags (mask 0x03) have no named meaning; the caller supplies
 *     polarity, and assertion flags must be zero;
 *   - hyperedge semantic flag 0x01 and participant flag QUERY: refused (QUERY marks query-local overlay
 *     participants, which are not persistent evidence);
 *   - participant counts of 0 or above ECS_MEM_MAX_PARTICIPANTS, and two participants sharing (role, ordinal):
 *     not representable as one MF-R0 observation, refused.
 * A mapped observation is evidence to be recorded, not a truth and not an authorization to learn.
 */

typedef enum {
    ECS_MEM_HACF_OK = 0,
    ECS_MEM_HACF_INVALID = -1,       /* NULL argument or polarity outside ecs_mem_polarity */
    ECS_MEM_HACF_STRUCTURE = -2,     /* structure's validator refused the hyperedge or the assertion */
    ECS_MEM_HACF_IDENTITY = -3,      /* a stored identity differs from the recomputed one, or the assertion does
                                        not assert this hyperedge */
    ECS_MEM_HACF_NONCANONICAL = -4,  /* participants are not in structure's canonical order */
    ECS_MEM_HACF_UNSUPPORTED = -5    /* flags with undefined meaning, query-local participants, a participant
                                        count or (role, ordinal) multiplicity MF-R0 cannot represent */
} ecs_mem_hacf_status;

ecs_mem_hacf_status ecs_mem_hacf_observation(const elpis_semantic_hyperedge_v1 *edge,
                                             const elpis_semantic_assertion_v1 *assertion, uint32_t polarity,
                                             uint64_t tick, ecs_mem_observation *out);

/* Rebuild the hyperedge a resident slot was observed from (identity recomputed by structure). The rebuilt
 * identity equals the slot's structure_ref exactly when the slot was observed through this adapter. */
ecs_mem_hacf_status ecs_mem_hacf_hyperedge(const ecs_mem_slot *slot, elpis_semantic_hyperedge_v1 *out);

#ifdef __cplusplus
}
#endif
#endif
