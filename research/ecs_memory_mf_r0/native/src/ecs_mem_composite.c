/* Composite K1 + MF-R0 identity (RESEARCH_ONLY). Contract: ecs_mem_composite.h. */
#include "ecs_mem_composite.h"
#include "elpis/sha256.h"

ecs_mem_status ecs_mem_composite_identity(const uint8_t k1_state_digest[ECS_MEM_DIGEST_BYTES],
                                          const ecs_mem_digest *memory_identity, ecs_mem_digest *out) {
    static const char domain[] = "elpis.ecs-composite.mf-r0.v0";
    if (!k1_state_digest || !memory_identity || !out) return ECS_MEM_INVALID;
    elpis_sha256_ctx ctx;
    elpis_sha256_init(&ctx);
    elpis_sha256_update(&ctx, domain, sizeof(domain));
    elpis_sha256_update(&ctx, k1_state_digest, ECS_MEM_DIGEST_BYTES);
    elpis_sha256_update(&ctx, memory_identity->b, ECS_MEM_DIGEST_BYTES);
    elpis_sha256_final(&ctx, out->b);
    return ECS_MEM_OK;
}
