#ifndef ELPIS_EXPERIMENTAL_MODEL_FREE_MEMORY_H
#define ELPIS_EXPERIMENTAL_MODEL_FREE_MEMORY_H
#include "indexed_lattice.h"
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Purely algorithmic, untrained feature/address derivation.
 * Not a semantic encoder, not a language model and not a cognitive writer.
 * Callers must supply explicit typed identifiers and a 64-float payload.
 * No filesystem access, no hidden state, no allocation during operations.
 */
typedef struct ecs_mf_memory ecs_mf_memory;
ecs_mf_memory *ecs_mf_create(size_t cells, d31c_policy policy, int radius,
                             const float alpha[2], const float beta[2]);
void ecs_mf_destroy(ecs_mf_memory *m);
void ecs_mf_address(uint64_t context, uint32_t role, int radius, double out_xy[2]);
void ecs_mf_features(uint64_t entity, uint32_t role, float out_key[64]);
d31c_status ecs_mf_write(ecs_mf_memory *m, uint64_t context, uint64_t entity,
                         uint32_t role, const float value[64]);
d31c_status ecs_mf_read(const ecs_mf_memory *m, uint64_t context, uint64_t entity,
                        uint32_t role, float out_value[64]);
size_t ecs_mf_allocated_bytes(const ecs_mf_memory *m);
size_t ecs_mf_occupied_cells(const ecs_mf_memory *m);
uint64_t ecs_mf_evictions(const ecs_mf_memory *m);
uint64_t ecs_mf_commits(const ecs_mf_memory *m);
#ifdef __cplusplus
}
#endif
#endif
