/* elpis/vector_execution.h - bounded parallel exact search over immutable HACF shards.
 *
 * This is an opt-in adapter. The existing elpis_vector_index_search() remains
 * unchanged and is the serial correctness oracle. One executor owns one native
 * execution runtime and borrows one vector index; destroy the executor before
 * the index. Searches on one executor are serialized because the underlying
 * execution runtime has one ordered consumer. */
#ifndef ELPIS_VECTOR_EXECUTION_H
#define ELPIS_VECTOR_EXECUTION_H

#include "elpis/execution.h"
#include "elpis/vector_index.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ELPIS_VECTOR_EXEC_ABI_VERSION 1u
#define ELPIS_VECTOR_EXEC_MAX_K 4096u
enum { ELPIS_EXEC_VECTOR_SHARD = 2u };

typedef struct elpis_vector_executor elpis_vector_executor;

/* workers: 0 = executor default, or explicit 1..4: this executor's concurrency
 * cap on the one module-wide execution pool (at most four threads) shared by all
 * native execution adapters. Creation does not compete for private threads. */
int elpis_vector_executor_create(elpis_vector_index *index, unsigned workers,
                                 elpis_vector_executor **out);
void elpis_vector_executor_destroy(elpis_vector_executor *executor);

/* Exact search with deterministic parity to elpis_vector_index_search for all
 * accepted requests. Independent immutable shards score on executor workers;
 * each worker emits at most local top-k. Ordered retirement merges and globally
 * sorts by the existing canonical score-key/digest comparator. No shard bytes
 * cross the execution queue. k is bounded to ELPIS_VECTOR_EXEC_MAX_K. */
int elpis_vector_executor_search(elpis_vector_executor *executor,
                                 const elpis_vector_query *query,
                                 elpis_vector_hit *hits, uint32_t *n_out);

/* Cumulative metrics of the owned execution runtime. */
void elpis_vector_executor_metrics(elpis_vector_executor *executor,
                                   elpis_exec_metrics *out);

#ifdef __cplusplus
}
#endif
#endif
