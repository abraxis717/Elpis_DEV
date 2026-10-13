/* M4: MF-R0 next to the one ECS (native K1) and the generic substrate (FMS ABI v2), without new authority.
 *
 * 1. Authority gap, demonstrated: the K1 retained-state digest (the only cognitive identity continuity binds)
 *    is unchanged by any MF-R0 transition, so continuity cannot detect, order or restore memory state. The
 *    composite identity changes with either component; nothing durable holds it.
 * 2. Residency: the ELPISEM1 envelope is an opaque generic FMS object. Demoted to COLD under WARM pressure and
 *    rematerialized (FMS verifies its own cold replica), it restores the identical memory state. FMS
 *    interprets nothing; this is placement, not a checkpoint authority (docs/K1_RECOVERY_R0.md). */
#define _POSIX_C_SOURCE 200809L
#include "ecs_mem_composite.h"
#include "ecs_mem_test_util.h"
#include "elpis/ecsg_k1.h"
#include "elpis/fms.h"
#include "elpis/fms_pal_posix.h"

#include <sys/stat.h>

enum { DIM = 6, WIDTH = 36 };

static void k1_digest(elpis_ecsg_k1 *k1, uint8_t out[32]) {
    assert(elpis_ecsg_k1_state_digest(k1, out) == ELPIS_ECSG_K1_OK);
}

static void test_authority_gap(void) {
    double w[DIM * WIDTH];
    for (size_t i = 0; i < DIM * WIDTH; ++i) w[i] = 0.01 * (double)((i * 37u) % 23u) - 0.1;
    elpis_ecsg_k1 *k1 = NULL;
    assert(elpis_ecsg_k1_create(DIM, WIDTH, 8, w, &k1) == ELPIS_ECSG_K1_OK);
    ecs_mem_config c = tu_config(64, 8, 8);
    ecs_mem *m;
    assert(ecs_mem_create(&c, &m) == ECS_MEM_OK);

    uint8_t k1_before[32], k1_after[32];
    ecs_mem_digest comp0, comp1, comp2;
    k1_digest(k1, k1_before);
    ecs_mem_digest mem0 = tu_identity(m);
    assert(ecs_mem_composite_identity(k1_before, &mem0, &comp0) == ECS_MEM_OK);

    /* Memory transitions: the K1 digest, hence what continuity would compare on restart, does not move. */
    ecs_mem_observation o = tu_bin(1, 1, 2, ECS_MEM_ASSERT, 1);
    ecs_mem_digest key;
    assert(ecs_mem_observe(m, 0, &o, &key, NULL) == ECS_MEM_OK);
    assert(ecs_mem_learn(m, 0, &key, 1, NULL) == ECS_MEM_OK);
    k1_digest(k1, k1_after);
    assert(!memcmp(k1_before, k1_after, 32));
    ecs_mem_digest mem1 = tu_identity(m);
    assert(!tu_same(mem0, mem1));
    assert(ecs_mem_composite_identity(k1_after, &mem1, &comp1) == ECS_MEM_OK);
    assert(!tu_same(comp0, comp1));

    /* A K1 transition moves the K1 digest and the composite, never the memory identity. */
    double x[DIM] = {0.1, -0.2, 0.3, 0.05, -0.1, 0.2};
    elpis_ecsg_k1_transition tr;
    assert(elpis_ecsg_k1_consolidate(k1, x, 1, &tr) == ELPIS_ECSG_K1_OK);
    k1_digest(k1, k1_after);
    assert(memcmp(k1_before, k1_after, 32));
    assert(tu_same(mem1, tu_identity(m)));
    assert(ecs_mem_composite_identity(k1_after, &mem1, &comp2) == ECS_MEM_OK);
    assert(!tu_same(comp1, comp2));
    /* Domain separation: the composite is never either component's identity. */
    assert(memcmp(comp2.b, k1_after, 32) && !tu_same(comp2, mem1));
    assert(ecs_mem_composite_identity(NULL, &mem1, &comp2) == ECS_MEM_INVALID);
    ecs_mem_destroy(m);
    assert(elpis_ecsg_k1_destroy(&k1) == ELPIS_ECSG_K1_OK);
}

static void test_generic_fms_residency(void) {
    ecs_mem_config c = tu_config(64, 8, 8);
    ecs_mem *m;
    assert(ecs_mem_create(&c, &m) == ECS_MEM_OK);
    for (uint64_t i = 0; i < 6; ++i) {
        ecs_mem_observation o = tu_bin(1, i, 100 + i, ECS_MEM_ASSERT, i);
        ecs_mem_digest key;
        assert(ecs_mem_observe(m, 0, &o, &key, NULL) == ECS_MEM_OK);
        assert(ecs_mem_learn(m, 0, &key, i, NULL) == ECS_MEM_OK);
    }
    const ecs_mem_digest id = tu_identity(m);
    const size_t n = ecs_mem_snapshot_size(m);
    uint8_t *env = malloc(n), *filler = calloc(1, n);
    size_t written;
    assert(env && filler && ecs_mem_snapshot(m, env, n, &written) == ECS_MEM_OK && written == n);

    mkdir("ecs_mem_cold", 0700);
    fms_config fc;
    memset(&fc, 0, sizeof(fc));
    fc.tier_budget[FMS_WARM] = fc.domain_ceiling[FMS_DOM_RAM] = (uint64_t)n + n / 2; /* room for one object */
    fc.tier_budget[FMS_COLD] = fc.domain_ceiling[FMS_DOM_STORAGE] = (uint64_t)n * 8;
    fc.high_wm = 0.9f;
    fc.low_wm = 0.5f;
    fc.max_objects = 4;
    fc.hot_absent_policy = fc.cold_absent_policy = FMS_REJECT;
    fms_pal *pal = fms_pal_posix_create("ecs_mem_cold");
    assert(pal);
    fms_ctx *ctx = fms_create(&fc, pal);
    assert(ctx);
    fms_id obj, other;
    assert(fms_register(ctx, 7, n, FMS_WARM, 0.0f, env, &obj) >= 0);
    assert(fms_register(ctx, 7, n, FMS_WARM, 0.0f, filler, &other) >= 0); /* pressure: the envelope is demoted */
    fms_object_info info;
    assert(fms_query(ctx, obj, &info) == FMS_OK && info.tier == FMS_COLD && info.cold_replica);
    fms_stats before, after;
    fms_get_stats(ctx, &before);
    void *p = NULL;
    assert(fms_acquire(ctx, obj, FMS_WARM, FMS_READ, &p) >= 0 && p);
    fms_get_stats(ctx, &after);
    assert(after.cold_reads > before.cold_reads);
    ecs_mem *back;
    assert(ecs_mem_restore((const uint8_t *)p, n, c.byte_budget, &back) == ECS_MEM_OK);
    assert(fms_release(ctx, obj) == FMS_OK);
    assert(tu_same(id, tu_identity(back)));
    ecs_mem_destroy(back);
    fms_destroy(ctx);
    free(env);
    free(filler);
    ecs_mem_destroy(m);
}

int main(void) {
    test_authority_gap();
    test_generic_fms_residency();
    printf("PASS_ECS_MEM_COMPOSITE_M4\n");
    return 0;
}
