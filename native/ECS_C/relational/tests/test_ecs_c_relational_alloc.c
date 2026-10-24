/* ECS_C relational view: one heap block per view, none per call (link-time allocator wrapping, GNU/Linux,
 * uninstrumented builds). Nothing grows however many calls run: every sync, admission, read, export, identity,
 * verification, rebuild, reset and refusal computes in the view's fixed arena. */
#include "relational_test_util.h"

static int tracking;
static unsigned long heap_ops;
void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void *__real_aligned_alloc(size_t, size_t);
void __real_free(void *);
void *__wrap_malloc(size_t n) { heap_ops += (unsigned long)tracking; return __real_malloc(n); }
void *__wrap_calloc(size_t a, size_t b) { heap_ops += (unsigned long)tracking; return __real_calloc(a, b); }
void *__wrap_realloc(void *p, size_t n) { heap_ops += (unsigned long)tracking; return __real_realloc(p, n); }
void *__wrap_aligned_alloc(size_t a, size_t n) { heap_ops += (unsigned long)tracking; return __real_aligned_alloc(a, n); }
void __wrap_free(void *p) { heap_ops += (unsigned long)tracking; __real_free(p); }

static elpis_ecs_mem_slot exported[64];

int main(void) {
    world w;
    w_init(&w, 11);
    const uint32_t a = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 7, 1, 1);
    const uint32_t b = w_found(&w, ELPIS_ECS_C_SOURCE, ELPIS_ECS_C_ACTIVE, 8, 1, 2);
    elpis_ecs_c_rel_config c;
    elpis_ecs_c_rel_config_default(&c);
    c.relation_capacity = 64;
    c.contribution_capacity = 128;
    elpis_ecs_c_rel *v = NULL;
    heap_ops = 0, tracking = 1;
    OK(elpis_ecs_c_rel_create(&c, &v));
    tracking = 0;
    CHECK(heap_ops == 1); /* the view's one block */
    OK(w_sync(v, &w, NULL));
    heap_ops = 0, tracking = 1;
    uint64_t tick = 1;
    for (uint32_t i = 0; i < 2000; ++i) {
        const elpis_ecs_mem_observation oa = pair(5, ELPIS_ECS_MEM_ASSERT, tick, 7, i % 16, 100 + i % 4);
        const elpis_ecs_mem_observation ob = pair(5, ELPIS_ECS_MEM_NEGATE, tick++, 8, i % 16, 100 + i % 4);
        elpis_ecs_mem_digest key;
        OK(admit_as(v, &w, a, &oa, &key));
        OK(admit_as(v, &w, b, &ob, NULL));
        const elpis_ecs_mem_request q = ask(5, i % 16);
        elpis_ecs_mem_answer ans;
        OK(elpis_ecs_c_rel_query(v, NULL, NULL, &q, &ans, NULL));
        OK(elpis_ecs_c_rel_query(v, NULL, w.rows[a].scope, &q, &ans, NULL));
        elpis_ecs_mem_slot slot;
        OK(elpis_ecs_c_rel_relation_read(v, NULL, NULL, &key, &slot));
        size_t n;
        OK(elpis_ecs_c_rel_export(v, NULL, NULL, exported, 64, &n, NULL));
        elpis_ecs_c_rel_support sup[4];
        OK(elpis_ecs_c_rel_support_of(v, &key, sup, 4, &n));
        elpis_ecs_mem_digest deps[64];
        OK(elpis_ecs_c_rel_dependents(v, w.rows[a].id, deps, 64, &n));
        uint8_t id[32];
        OK(elpis_ecs_c_rel_identity(v, id));
        uint64_t bad;
        OK(elpis_ecs_c_rel_verify(v, &bad));
        CHECK(bad == 0);
        /* Authority transitions: suspension, restoration, and every 200 rounds a revision (retirement). */
        w_lifecycle(&w, b, ELPIS_ECS_C_DORMANT);
        OK(w_sync(v, &w, NULL));
        w_lifecycle(&w, b, ELPIS_ECS_C_ACTIVE);
        OK(w_sync(v, &w, NULL));
        if (i % 200 == 199) {
            w_version(&w, a, w.rows[a].source_version + 1, 1);
            OK(w_sync(v, &w, NULL));
            OK(elpis_ecs_c_rel_rebuild(v, NULL));
        }
        /* Refusals allocate nothing either. */
        RC(elpis_ecs_c_rel_export(v, NULL, NULL, exported, 0, &n, NULL), ELPIS_ECS_C_REL_CAPACITY);
        elpis_ecs_mem_observation stale = oa;
        stale.source_authority = 8;
        RC(admit_as(v, &w, a, &stale, NULL), ELPIS_ECS_C_REL_BINDING_MISMATCH);
        world back = w;
        back.generation--;
        RC(w_sync(v, &back, NULL), ELPIS_ECS_C_REL_REGRESSION);
    }
    OK(elpis_ecs_c_rel_reset(v));
    tracking = 0;
    CHECK(heap_ops == 0);
    heap_ops = 0, tracking = 1;
    elpis_ecs_c_rel_destroy(v);
    tracking = 0;
    CHECK(heap_ops == 1);
    printf("PASS_ECS_C_RELATIONAL_ONE_BLOCK_PER_VIEW heap_ops_per_call=0\n");
    return 0;
}
