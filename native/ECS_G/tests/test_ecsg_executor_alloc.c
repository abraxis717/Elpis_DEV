/*
 * ECS_G executor R1: no heap allocation after creation.
 *
 * The executor and kernel sources are linked into this test directly and
 * every allocator entry point they call is redirected with -Wl,--wrap, so
 * each allocation they make is counted. Positive controls (create, reserve)
 * prove the interposition is live.
 */
#include "elpis/ecsg_executor.h"
#include "elpis/ecsg_math.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long allocations;
static unsigned long releases;
static int counting;

void *__real_malloc(size_t size);
void *__real_calloc(size_t count, size_t size);
void *__real_realloc(void *ptr, size_t size);
void __real_free(void *ptr);
void *__real_aligned_alloc(size_t alignment, size_t size);
int __real_posix_memalign(void **out, size_t alignment, size_t size);

void *__wrap_malloc(size_t size);
void *__wrap_calloc(size_t count, size_t size);
void *__wrap_realloc(void *ptr, size_t size);
void __wrap_free(void *ptr);
void *__wrap_aligned_alloc(size_t alignment, size_t size);
int __wrap_posix_memalign(void **out, size_t alignment, size_t size);

void *__wrap_malloc(size_t size)
{
    allocations += (unsigned long)counting;
    return __real_malloc(size);
}

void *__wrap_calloc(size_t count, size_t size)
{
    allocations += (unsigned long)counting;
    return __real_calloc(count, size);
}

void *__wrap_realloc(void *ptr, size_t size)
{
    allocations += (unsigned long)counting;
    return __real_realloc(ptr, size);
}

void __wrap_free(void *ptr)
{
    releases += (unsigned long)(counting && ptr != NULL);
    __real_free(ptr);
}

void *__wrap_aligned_alloc(size_t alignment, size_t size)
{
    allocations += (unsigned long)counting;
    return __real_aligned_alloc(alignment, size);
}

int __wrap_posix_memalign(void **out, size_t alignment, size_t size)
{
    allocations += (unsigned long)counting;
    return __real_posix_memalign(out, alignment, size);
}

enum { DIM = 6, WIDTH = 36, ROWS = 64 };

int
main(void)
{
    static double w0[DIM * WIDTH], x[ROWS * DIM], y[ROWS], out[ROWS], w[DIM * WIDTH];
    static double mu[DIM], m[21], t3[56];
    static uint8_t snap[40 + DIM * WIDTH * 8];
    const elpis_ecsg_drive drives[2] = {{40u, 3u}, {24u, 2u}};
    elpis_ecsg_executor *e = NULL;
    elpis_ecsg_exec_transition t;
    elpis_ecsg_exec_stats st;
    uint64_t token = 0u, epoch = 0u;
    size_t i;
    int round;

    for (i = 0u; i < DIM * WIDTH; ++i) {
        w0[i] = 0.01 * (double)((int)(i % 17u) - 8);
    }
    for (i = 0u; i < ROWS * DIM; ++i) {
        x[i] = 0.05 * (double)((int)(i % 11u) - 5);
    }
    for (i = 0u; i < ROWS; ++i) {
        y[i] = 0.02 * (double)((int)(i % 7u) - 3);
    }

    /* Positive control: creation allocates (executor + arena). */
    counting = 1;
    assert(elpis_ecsg_executor_create(DIM, WIDTH, ROWS, w0, &e) == ELPIS_ECSG_EXEC_OK);
    counting = 0;
    assert(allocations == 2u);

    /* Steady state: query, learn (1..1000 steps), schedules, refusals, reads. */
    allocations = releases = 0u;
    counting = 1;
    for (round = 0; round < 3; ++round) {
        assert(elpis_ecsg_executor_forward(e, x, ROWS, out) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_learn(e, x, y, ROWS, 0.002, 1u, &t) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_learn(e, x, y, ROWS, 0.002, 1000u, &t) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_learn_schedule(e, x, y, drives, 2u, 0.002, &t) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_learn(e, x, y, ROWS + 1u, 0.002, 1u, &t) == ELPIS_ECSG_EXEC_CAPACITY);
        assert(elpis_ecsg_executor_learn(e, x, y, ROWS, 1e6, 50u, &t) == ELPIS_ECSG_EXEC_NONFINITE);
        assert(elpis_ecsg_executor_copy_w(e, w, DIM * WIDTH) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_project_s3(e, mu, m, t3) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_snapshot_write(e, snap, sizeof(snap)) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_stats(e, &st) == ELPIS_ECSG_EXEC_OK);
        /* Transactions: begin, learn, readout, commit; stale; abort; refused learn. */
        assert(elpis_ecsg_executor_txn_begin(e, &token) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_txn_learn(e, token, x, y, ROWS, 0.002, 100u, &t) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_txn_learn_schedule(e, token, x, y, drives, 2u, 0.002, &t) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_txn_forward(e, token, x, ROWS, out) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_txn_project_s3(e, token, mu, m, t3) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_txn_epoch(e, token, &epoch) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_txn_commit(e, token, &t) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_txn_begin(e, &token) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_learn(e, x, y, ROWS, 0.002, 1u, &t) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_txn_commit(e, token, &t) == ELPIS_ECSG_EXEC_STALE);
        assert(elpis_ecsg_executor_txn_begin(e, &token) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_txn_learn(e, token, x, y, ROWS, 0.002, 5u, &t) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_txn_abort(e, token) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_txn_begin(e, &token) == ELPIS_ECSG_EXEC_OK);
        assert(elpis_ecsg_executor_txn_learn(e, token, x, y, ROWS, 1e6, 50u, &t) == ELPIS_ECSG_EXEC_NONFINITE);
    }
    counting = 0;
    assert(allocations == 0u && releases == 0u);
    assert(st.heap_allocations == 2u);

    /* Positive control: explicit growth is the one cold-path allocation. */
    counting = 1;
    assert(elpis_ecsg_executor_reserve(e, 2u * ROWS) == ELPIS_ECSG_EXEC_OK);
    counting = 0;
    assert(allocations == 1u && releases == 1u);

    allocations = releases = 0u;
    counting = 1;
    assert(elpis_ecsg_executor_learn(e, x, y, ROWS, 0.002, 10u, &t) == ELPIS_ECSG_EXEC_OK);
    counting = 0;
    assert(allocations == 0u && releases == 0u);

    assert(elpis_ecsg_executor_destroy(&e) == ELPIS_ECSG_EXEC_OK);
    puts("test_ecsg_executor_alloc: ok");
    return 0;
}
