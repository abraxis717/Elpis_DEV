/* Typed HACF -> ECS memory observation path (runtime composition layer). Contract: elpis/ecs_observation_hacf.h. */
#include "elpis/ecs_observation_hacf.h"

#include <string.h>

static int digest_equal(const hacf_digest *a, const hacf_digest *b) {
    return !memcmp(a->bytes, b->bytes, HACF_DIGEST_BYTES);
}

elpis_ecs_obs_hacf_status elpis_ecs_obs_hacf_observation(const elpis_semantic_hyperedge_v1 *edge,
                                             const elpis_semantic_assertion_v1 *assertion, uint32_t polarity,
                                             uint64_t tick, elpis_ecs_mem_observation *out) {
    if (!edge || !assertion || !out) return ELPIS_ECS_OBS_HACF_INVALID;
    if (polarity < ELPIS_ECS_MEM_ASSERT || polarity > ELPIS_ECS_MEM_UNCERTAIN) return ELPIS_ECS_OBS_HACF_INVALID;
    if (elpis_semantic_hyperedge_validate(edge) != SEMANTIC_OK) return ELPIS_ECS_OBS_HACF_STRUCTURE;
    if (elpis_semantic_assertion_validate(assertion) != SEMANTIC_OK) return ELPIS_ECS_OBS_HACF_STRUCTURE;
    hacf_digest id;
    if (elpis_semantic_hyperedge_identity(edge, &id) != SEMANTIC_OK || !digest_equal(&id, &edge->hyperedge_identity))
        return ELPIS_ECS_OBS_HACF_IDENTITY;
    if (elpis_semantic_assertion_identity(assertion, &id) != SEMANTIC_OK ||
        !digest_equal(&id, &assertion->assertion_identity))
        return ELPIS_ECS_OBS_HACF_IDENTITY;
    if (assertion->asserted_object_kind != SEMANTIC_OBJECT_KIND_HYPEREDGE ||
        !digest_equal(&assertion->asserted_object_digest, &edge->hyperedge_identity))
        return ELPIS_ECS_OBS_HACF_IDENTITY;
    const uint32_t n = edge->participant_count;
    if (n < 1 || n > ELPIS_ECS_MEM_MAX_PARTICIPANTS) return ELPIS_ECS_OBS_HACF_UNSUPPORTED;
    if (edge->semantic_flags || assertion->assertion_flags) return ELPIS_ECS_OBS_HACF_UNSUPPORTED;
    elpis_semantic_participant_descriptor canon[ELPIS_ECS_MEM_MAX_PARTICIPANTS];
    memcpy(canon, edge->participants, sizeof(canon[0]) * n);
    if (elpis_semantic_canonicalize_participants(canon, n) != SEMANTIC_OK ||
        memcmp(canon, edge->participants, sizeof(canon[0]) * n))
        return ELPIS_ECS_OBS_HACF_NONCANONICAL;
    for (uint32_t i = 0; i < n; ++i) {
        if (edge->participants[i].participant_flags) return ELPIS_ECS_OBS_HACF_UNSUPPORTED;
        if (i && edge->participants[i].incidence_role == edge->participants[i - 1].incidence_role &&
            edge->participants[i].ordinal == edge->participants[i - 1].ordinal)
            return ELPIS_ECS_OBS_HACF_UNSUPPORTED;
    }
    elpis_ecs_mem_observation o;
    memset(&o, 0, sizeof(o));
    o.abi_version = ELPIS_ECS_MEM_ABI_V1;
    o.relation_type = edge->hyperedge_type;
    o.polarity = polarity;
    o.participant_count = n;
    o.tick = tick;
    o.source_authority = assertion->authority;
    memcpy(o.payload.b, edge->payload_digest.bytes, 32);
    memcpy(o.provenance.b, assertion->provenance_digest.bytes, 32);
    memcpy(o.structure_ref.b, edge->hyperedge_identity.bytes, 32);
    for (uint32_t i = 0; i < n; ++i) {
        o.participants[i].role = edge->participants[i].incidence_role;
        o.participants[i].ordinal = edge->participants[i].ordinal;
        memcpy(o.participants[i].entity.b, edge->participants[i].node_identity.bytes, 32);
    }
    *out = o;
    return ELPIS_ECS_OBS_HACF_OK;
}

elpis_ecs_obs_hacf_status elpis_ecs_obs_hacf_hyperedge(const elpis_ecs_mem_slot *slot, elpis_semantic_hyperedge_v1 *out) {
    if (!slot || !out || slot->participant_count < 1 || slot->participant_count > ELPIS_ECS_MEM_MAX_PARTICIPANTS)
        return ELPIS_ECS_OBS_HACF_INVALID;
    elpis_semantic_hyperedge_v1 e;
    memset(&e, 0, sizeof(e));
    e.abi_version = SEMANTIC_ABI_VERSION;
    e.hyperedge_type = slot->relation_type;
    memcpy(e.payload_digest.bytes, slot->payload.b, 32);
    e.participant_count = slot->participant_count;
    for (uint32_t i = 0; i < slot->participant_count; ++i) {
        e.participants[i].incidence_role = slot->participants[i].role;
        e.participants[i].ordinal = slot->participants[i].ordinal;
        memcpy(e.participants[i].node_identity.bytes, slot->participants[i].entity.b, 32);
    }
    if (elpis_semantic_hyperedge_validate(&e) != SEMANTIC_OK) return ELPIS_ECS_OBS_HACF_STRUCTURE;
    if (elpis_semantic_hyperedge_identity(&e, &e.hyperedge_identity) != SEMANTIC_OK) return ELPIS_ECS_OBS_HACF_IDENTITY;
    *out = e;
    return ELPIS_ECS_OBS_HACF_OK;
}
