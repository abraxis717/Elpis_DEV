/* ECS Memory MF-R0: no heap operation after create (link-time allocator wrapping, GNU/Linux, uninstrumented). */
#include "ecs_memory_test_util.h"

static int tracking;
static unsigned long heap_ops;
void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void *__real_aligned_alloc(size_t, size_t);
void __real_free(void *);
void *__wrap_malloc(size_t n) { heap_ops += tracking; return __real_malloc(n); }
void *__wrap_calloc(size_t a, size_t b) { heap_ops += tracking; return __real_calloc(a, b); }
void *__wrap_realloc(void *p, size_t n) { heap_ops += tracking; return __real_realloc(p, n); }
void *__wrap_aligned_alloc(size_t a, size_t n) { heap_ops += tracking; return __real_aligned_alloc(a, n); }
void __wrap_free(void *p) { heap_ops += tracking; __real_free(p); }

int main(void) {
    elpis_ecs_mem_config c = tu_config(128, 8, 8);
    elpis_ecs_mem *m;
    heap_ops = 0;
    tracking = 1;
    assert(elpis_ecs_mem_create(&c, &m) == ELPIS_ECS_MEM_OK);
    tracking = 0;
    assert(heap_ops == 2); /* the handle and its one arena */
    static uint8_t snap[1u << 22];
    size_t written = 0;
    heap_ops = 0;
    tracking = 1;
    for (uint64_t i = 0; i < 200; ++i) {
        elpis_ecs_mem_observation o = tu_bin(1 + (uint32_t)(i % 2), i % 11, 50 + i % 13, ELPIS_ECS_MEM_ASSERT, i);
        elpis_ecs_mem_digest key;
        assert(elpis_ecs_mem_observe(m, 0, &o, &key, NULL) == ELPIS_ECS_MEM_OK);
        assert(elpis_ecs_mem_learn(m, 0, &key, i, NULL) == ELPIS_ECS_MEM_OK);
        elpis_ecs_mem_request q = tu_query(ELPIS_ECS_MEM_FAST, 1, 2);
        tu_cue(&q, 1, i % 11);
        elpis_ecs_mem_answer a;
        assert(elpis_ecs_mem_query(m, &q, &a) == ELPIS_ECS_MEM_OK);
        q.channel = ELPIS_ECS_MEM_LOOKUP;
        q.candidate_source = ELPIS_ECS_MEM_CANDIDATES_CALLER;
        assert(elpis_ecs_mem_query(m, &q, &a) == ELPIS_ECS_MEM_OK);
        elpis_ecs_mem_stats s;
        assert(elpis_ecs_mem_inspect(m, &s) == ELPIS_ECS_MEM_OK);
        (void)tu_identity(m);
        assert(elpis_ecs_mem_snapshot(m, snap, sizeof(snap), &written) == ELPIS_ECS_MEM_OK);
        assert(elpis_ecs_mem_restore_into(m, 0, snap, written) == ELPIS_ECS_MEM_OK);
    }
    tracking = 0;
    assert(heap_ops == 0);
    elpis_ecs_mem_destroy(m);
    printf("PASS_ECS_MEM_NO_ALLOCATION_AFTER_CREATE heap_ops=%lu\n", heap_ops);
    return 0;
}
