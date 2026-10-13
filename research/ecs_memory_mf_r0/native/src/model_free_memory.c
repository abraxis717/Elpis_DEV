#include "model_free_memory.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

struct ecs_mf_memory {
    d31c_arena *arena;
    int radius;
    float alpha[2];
    float beta[2];
};

/* Fixed, portable SplitMix64 permutation: deterministic only, not cryptographic. */
static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

void ecs_mf_address(uint64_t context, uint32_t role, int radius, double out_xy[2]) {
    if (!out_xy) return;
    if (radius < 2 || radius > 256) { out_xy[0] = NAN; out_xy[1] = NAN; return; }
    const uint64_t seed = mix64(context ^ ((uint64_t)role << 32) ^ UINT64_C(0x3141592653589793));
    const uint64_t a = mix64(seed + UINT64_C(0x9e3779b97f4a7c15));
    const uint64_t b = mix64(seed + UINT64_C(0xd1b54a32d192ed03));
    const uint64_t span = (uint64_t)(2 * radius - 1);
    out_xy[0] = (double)((int64_t)(a % span) - radius + 1) + 0.17;
    out_xy[1] = (double)((int64_t)(b % span) - radius + 1) - 0.27;
}

void ecs_mf_features(uint64_t entity, uint32_t role, float out_key[64]) {
    if (!out_key) return;
    uint64_t s = mix64(entity ^ ((uint64_t)role << 32) ^ UINT64_C(0xc2b2ae3d27d4eb4f));
    /* Deterministic ±1/sqrt(32) per dimension and head; no trained embedding. */
    for (size_t i = 0; i < 64; ++i) {
        s = mix64(s + UINT64_C(0x9e3779b97f4a7c15) + (uint64_t)i);
        out_key[i] = (s & 1u) ? 0.176776692f : -0.176776692f;
    }
}

ecs_mf_memory *ecs_mf_create(size_t cells, d31c_policy policy, int radius,
                             const float alpha[2], const float beta[2]) {
    if (!alpha || !beta) return NULL;
    for (size_t i = 0; i < 2; ++i)
        if (!isfinite(alpha[i]) || !isfinite(beta[i]) || alpha[i] < 0.0f ||
            alpha[i] > 1.0f || beta[i] < 0.0f || beta[i] > 1.0f) return NULL;
    ecs_mf_memory *m = (ecs_mf_memory *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->arena = d31c_create(cells, policy, radius);
    if (!m->arena) { free(m); return NULL; }
    m->radius = radius;
    for (size_t i = 0; i < 2; ++i) { m->alpha[i] = alpha[i]; m->beta[i] = beta[i]; }
    return m;
}
void ecs_mf_destroy(ecs_mf_memory *m) { if (m) { d31c_destroy(m->arena); free(m); } }

d31c_status ecs_mf_write(ecs_mf_memory *m, uint64_t context, uint64_t entity,
                         uint32_t role, const float value[64]) {
    if (!m || !value) return D31C_ERR_ARGUMENT;
    double xy[2]; float key[64];
    ecs_mf_address(context, role, m->radius, xy);
    ecs_mf_features(entity, role, key);
    return d31c_write(m->arena, xy, key, value, m->alpha, m->beta);
}

d31c_status ecs_mf_read(const ecs_mf_memory *m, uint64_t context, uint64_t entity,
                        uint32_t role, float out_value[64]) {
    if (!m || !out_value) return D31C_ERR_ARGUMENT;
    double xy[2]; float key[64];
    ecs_mf_address(context, role, m->radius, xy);
    ecs_mf_features(entity, role, key);
    return d31c_read(m->arena, xy, key, out_value);
}
size_t ecs_mf_allocated_bytes(const ecs_mf_memory *m) {
    return m ? sizeof(*m) + d31c_allocated_bytes(m->arena) : 0;
}
size_t ecs_mf_occupied_cells(const ecs_mf_memory *m) {
    return m ? d31c_occupied(m->arena) : 0;
}
uint64_t ecs_mf_evictions(const ecs_mf_memory *m) {
    return m ? d31c_evictions(m->arena) : 0;
}
uint64_t ecs_mf_commits(const ecs_mf_memory *m) {
    return m ? d31c_commits(m->arena) : 0;
}
