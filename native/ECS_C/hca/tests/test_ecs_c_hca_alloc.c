/* ECS_C HCA: one heap block per graph and per allocated B2 session, none per call (link-time allocator wrapping,
 * GNU/Linux, uninstrumented builds). Nothing grows however many calls run: every prediction, tick, damage, reset
 * and observation computes in fixed scratch. */
#include "hca_test_util.h"

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

static elpis_hca_result result;

int main(void) {
    world w;
    coding(&w, 8);
    const elpis_hca_limits l = big_limits();
    elpis_hca_graph *g;
    heap_ops = 0, tracking = 1;
    OK(build_world(&w, &l, &g));
    tracking = 0;
    assert(heap_ops == 1); /* the graph's one block */
    task_buffer t;
    coding_task(&t, &w, 8, 1, 0);
    elpis_hca_b2 *s;
    elpis_hca_activation a;
    heap_ops = 0, tracking = 1;
    OK(elpis_hca_b2_start(g, &t.task, 0, &s, &a, NULL));
    tracking = 0;
    assert(heap_ops == 1); /* the session's one block */
    heap_ops = 0, tracking = 1;
    for (uint32_t i = 0; i < 2000; ++i) {
        for (uint32_t method = ELPIS_HCA_B0; method <= ELPIS_HCA_B2; ++method)
            OK(elpis_hca_predict(g, &t.task, method, &result, NULL, NULL));
        elpis_hca_activation next;
        OK(elpis_hca_b2_warm(s, &a, 0, &next, NULL));
        a = next;
        const elpis_hca_damage d = {idx(&w, "s1:20"), i % 1001};
        OK(elpis_hca_b2_damage(s, &a, &d, 1, 0, &next, NULL));
        OK(elpis_hca_b2_observe(s, &next, 0, &result, NULL));
        if (i % 100 == 0) OK(elpis_hca_b2_reset(s, 0, &a, NULL));
        uint8_t evidence[32];
        OK(elpis_hca_graph_evidence(g, evidence));
        /* Refusals allocate nothing either. */
        t.task.work_budget = 1;
        assert(elpis_hca_predict(g, &t.task, ELPIS_HCA_B2, &result, NULL, NULL) == ELPIS_HCA_WORK_LIMIT);
        t.task.work_budget = 1000000;
    }
    tracking = 0;
    assert(heap_ops == 0);
    heap_ops = 0, tracking = 1;
    elpis_hca_b2_destroy(s);
    elpis_hca_graph_destroy(g);
    tracking = 0;
    assert(heap_ops == 2);
    printf("PASS_ECS_C_HCA_ONE_BLOCK_PER_HANDLE heap_ops_per_call=0\n");
    return 0;
}
