/* ECS Memory MF-R0: no heap operation after create (link-time allocator wrapping, GNU/Linux, uninstrumented). */
#include "ecs_mem_test_util.h"

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
    ecs_mem_config c = tu_config(128, 8, 8);
    ecs_mem *m;
    heap_ops = 0;
    tracking = 1;
    assert(ecs_mem_create(&c, &m) == ECS_MEM_OK);
    tracking = 0;
    assert(heap_ops == 2); /* the handle and its one arena */
    static uint8_t snap[1u << 22];
    size_t written = 0;
    heap_ops = 0;
    tracking = 1;
    for (uint64_t i = 0; i < 200; ++i) {
        ecs_mem_observation o = tu_bin(1 + (uint32_t)(i % 2), i % 11, 50 + i % 13, ECS_MEM_ASSERT, i);
        ecs_mem_digest key;
        assert(ecs_mem_observe(m, 0, &o, &key, NULL) == ECS_MEM_OK);
        assert(ecs_mem_learn(m, 0, &key, i, NULL) == ECS_MEM_OK);
        if (i % 7 == 0) assert(ecs_mem_consolidate(m, 0, i, 4, NULL, NULL) == ECS_MEM_OK);
        ecs_mem_request q = tu_query(ECS_MEM_FAST, 1, 2);
        tu_cue(&q, 1, i % 11);
        ecs_mem_answer a;
        assert(ecs_mem_query(m, &q, &a) == ECS_MEM_OK);
        q.channel = ECS_MEM_LOOKUP;
        q.candidate_source = ECS_MEM_CANDIDATES_CALLER;
        assert(ecs_mem_query(m, &q, &a) == ECS_MEM_OK);
        ecs_mem_stats s;
        assert(ecs_mem_inspect(m, &s) == ECS_MEM_OK);
        (void)tu_identity(m);
        assert(ecs_mem_snapshot(m, snap, sizeof(snap), &written) == ECS_MEM_OK);
        assert(ecs_mem_restore_into(m, 0, snap, written) == ECS_MEM_OK);
    }
    tracking = 0;
    assert(heap_ops == 0);
    ecs_mem_destroy(m);
    printf("PASS_ECS_MEM_NO_ALLOCATION_AFTER_CREATE heap_ops=%lu\n", heap_ops);
    return 0;
}
