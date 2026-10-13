#ifndef ELPIS_ECS_MEM_COMPOSITE_H
#define ELPIS_ECS_MEM_COMPOSITE_H

#include "ecs_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Composite identity of one K1 retained state and one MF-R0 state (RESEARCH ONLY, NOT BOUND BY ANY AUTHORITY):
 *     SHA-256("elpis.ecs-composite.mf-r0.v0" NUL | k1_state_digest | memory_identity)
 * Continuity binds the K1 state digest alone (docs/CONTINUITY.md), so nothing durable binds this value. It is
 * the identity a future composite retained-state contract would need, defined here so that its absence from the
 * current authority is testable (docs/research/ECS_MEMORY_MF_R0.md, "M4").
 */
ecs_mem_status ecs_mem_composite_identity(const uint8_t k1_state_digest[ECS_MEM_DIGEST_BYTES],
                                          const ecs_mem_digest *memory_identity, ecs_mem_digest *out);

#ifdef __cplusplus
}
#endif
#endif
