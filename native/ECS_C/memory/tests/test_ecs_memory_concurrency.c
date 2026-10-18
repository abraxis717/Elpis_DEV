/* ECS Memory MF-R0: the single-writer / many-reader guard under real threads (meaningful under TSan).
 * Concurrent entries are either admitted or refused BUSY; admitted queries always see a committed state.
 * The guard is non-blocking and has no fairness: continuously overlapping readers can starve a writer
 * (observed: 400 of 400 writes refused BUSY with three busy readers). Readers yield here so that writes and
 * reads interleave; the writer retries BUSY, which is the caller's policy, not the memory's. */
#define _POSIX_C_SOURCE 200809L
#include "ecs_memory_test_util.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>

static elpis_ecs_mem *shared;
static atomic_int stop;
static atomic_ulong busy_reads, ok_reads, busy_writes, ok_writes;

static void *reader(void *arg) {
    (void)arg;
    elpis_ecs_mem_digest cands[4] = {tu_entity(100), tu_entity(101), tu_entity(102), tu_entity(103)};
    while (!atomic_load(&stop)) {
        elpis_ecs_mem_request q = tu_query(ELPIS_ECS_MEM_FAST, 1, 2);
        tu_cue(&q, 1, 7);
        q.candidate_source = ELPIS_ECS_MEM_CANDIDATES_CALLER;
        q.candidates = cands;
        q.candidate_count = 4;
        elpis_ecs_mem_answer a;
        const elpis_ecs_mem_status s = elpis_ecs_mem_query(shared, &q, &a);
        if (s == ELPIS_ECS_MEM_BUSY) atomic_fetch_add(&busy_reads, 1);
        else {
            assert(s == ELPIS_ECS_MEM_OK);
            /* Every committed state names one of the learned objects (never a torn mixture). */
            assert(a.found && (tu_same(a.answer, cands[0]) || tu_same(a.answer, cands[1])));
            atomic_fetch_add(&ok_reads, 1);
        }
        elpis_ecs_mem_stats st;
        const elpis_ecs_mem_status si = elpis_ecs_mem_inspect(shared, &st);
        assert(si == ELPIS_ECS_MEM_OK || si == ELPIS_ECS_MEM_BUSY);
        sched_yield();
    }
    return NULL;
}
static void *writer(void *arg) {
    (void)arg;
    uint64_t tick = 10;
    for (int i = 0; i < 200; ++i) {
        elpis_ecs_mem_observation o = tu_bin(1, 7, 100 + (uint64_t)(i % 2), ELPIS_ECS_MEM_ASSERT, tick);
        elpis_ecs_mem_digest key;
        elpis_ecs_mem_status s;
        while ((s = elpis_ecs_mem_observe(shared, 0, &o, &key, NULL)) == ELPIS_ECS_MEM_BUSY) {
            atomic_fetch_add(&busy_writes, 1);
            sched_yield();
        }
        assert(s == ELPIS_ECS_MEM_OK);
        while ((s = elpis_ecs_mem_learn(shared, 0, &key, tick, NULL)) == ELPIS_ECS_MEM_BUSY) {
            atomic_fetch_add(&busy_writes, 1);
            sched_yield();
        }
        assert(s == ELPIS_ECS_MEM_OK);
        atomic_fetch_add(&ok_writes, 1);
        ++tick;
    }
    atomic_store(&stop, 1);
    return NULL;
}

int main(void) {
    elpis_ecs_mem_config c = tu_config(64, 8, 8);
    assert(elpis_ecs_mem_create(&c, &shared) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_observation o = tu_bin(1, 7, 100, ELPIS_ECS_MEM_ASSERT, 1);
    elpis_ecs_mem_digest key;
    assert(elpis_ecs_mem_observe(shared, 0, &o, &key, NULL) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_learn(shared, 0, &key, 1, NULL) == ELPIS_ECS_MEM_OK);
    pthread_t r[3], w;
    for (int i = 0; i < 3; ++i) assert(pthread_create(&r[i], NULL, reader, NULL) == 0);
    assert(pthread_create(&w, NULL, writer, NULL) == 0);
    assert(pthread_join(w, NULL) == 0);
    for (int i = 0; i < 3; ++i) assert(pthread_join(r[i], NULL) == 0);
    assert(atomic_load(&ok_writes) == 200 && atomic_load(&ok_reads) > 0);
    elpis_ecs_mem_stats st;
    assert(elpis_ecs_mem_inspect(shared, &st) == ELPIS_ECS_MEM_OK && st.learns == 201);
    elpis_ecs_mem_destroy(shared);
    printf("PASS_ECS_MEM_CONCURRENCY ok_reads=%lu busy_reads=%lu ok_writes=%lu busy_writes=%lu\n",
           atomic_load(&ok_reads), atomic_load(&busy_reads), atomic_load(&ok_writes), atomic_load(&busy_writes));
    return 0;
}
