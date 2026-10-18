/* Shared fixtures for the canonical ECS memory tests (TEST_ONLY, TRAINING=NONE, SEMANTICS=NONE). */
#ifndef ELPIS_ECS_MEMORY_TEST_UTIL_H
#define ELPIS_ECS_MEMORY_TEST_UTIL_H

#include "elpis/ecs_memory.h"
#include "elpis/sha256.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Every capability: the mechanics tests exercise LEARN, so they admit the unqualified recurrent capability. */
#define TU_CAPS ELPIS_ECS_MEM_CAP_RECURRENT

static inline elpis_ecs_mem_digest tu_digest(const char *tag, uint64_t i) {
    char buf[96];
    elpis_ecs_mem_digest d;
    const int n = snprintf(buf, sizeof(buf), "%s:%llu", tag, (unsigned long long)i);
    elpis_sha256(buf, (size_t)n, d.b);
    return d;
}
static inline elpis_ecs_mem_digest tu_entity(uint64_t i) { return tu_digest("entity", i); }

static inline elpis_ecs_mem_observation tu_obs(uint32_t type, uint32_t polarity, uint64_t tick, uint32_t n,
                                         const uint32_t *roles, const uint64_t *entities) {
    elpis_ecs_mem_observation o;
    memset(&o, 0, sizeof(o));
    o.abi_version = ELPIS_ECS_MEM_ABI_V1;
    o.relation_type = type;
    o.polarity = polarity;
    o.participant_count = n;
    o.tick = tick;
    o.source_authority = 7;
    o.provenance = tu_digest("provenance", tick);
    o.structure_ref = tu_digest("structure", tick);
    for (uint32_t i = 0; i < n; ++i) {
        o.participants[i].role = roles[i];
        o.participants[i].ordinal = 0;
        o.participants[i].entity = tu_entity(entities[i]);
    }
    return o;
}
/* Binary relation type(subject -> object) with roles 1 (subject) and 2 (object). */
static inline elpis_ecs_mem_observation tu_bin(uint32_t type, uint64_t subject, uint64_t object, uint32_t polarity,
                                         uint64_t tick) {
    const uint32_t roles[2] = {1, 2};
    const uint64_t ents[2] = {subject, object};
    return tu_obs(type, polarity, tick, 2, roles, ents);
}
static inline elpis_ecs_mem_request tu_query(uint32_t channel, uint32_t type, uint32_t target_role) {
    elpis_ecs_mem_request q;
    memset(&q, 0, sizeof(q));
    q.abi_version = ELPIS_ECS_MEM_ABI_V1;
    q.channel = channel;
    q.relation_type = type;
    q.target_role = target_role;
    q.candidate_source = channel == ELPIS_ECS_MEM_LOOKUP ? ELPIS_ECS_MEM_CANDIDATES_CALLER : ELPIS_ECS_MEM_CANDIDATES_CODEBOOK;
    return q;
}
static inline void tu_cue(elpis_ecs_mem_request *q, uint32_t role, uint64_t entity) {
    q->cue[q->cue_count].role = role;
    q->cue[q->cue_count].ordinal = 0;
    q->cue[q->cue_count].entity = tu_entity(entity);
    q->cue_count++;
}
static inline elpis_ecs_mem_digest tu_identity(const elpis_ecs_mem *m) {
    elpis_ecs_mem_digest d;
    assert(elpis_ecs_mem_identity(m, &d) == ELPIS_ECS_MEM_OK);
    return d;
}
static inline int tu_same(elpis_ecs_mem_digest a, elpis_ecs_mem_digest b) { return !memcmp(a.b, b.b, 32); }
static inline elpis_ecs_mem_config tu_config(uint32_t dim, uint32_t slots, uint32_t ents) {
    elpis_ecs_mem_config c;
    elpis_ecs_mem_config_default(&c);
    c.dim = dim;
    c.banks = 2;
    c.slot_capacity = slots;
    c.codebook_capacity = ents;
    c.capabilities = TU_CAPS;
    return c;
}
static inline void tu_hex(const elpis_ecs_mem_digest *d, char out[65]) {
    elpis_hex32(d->b, out);
}

#endif
