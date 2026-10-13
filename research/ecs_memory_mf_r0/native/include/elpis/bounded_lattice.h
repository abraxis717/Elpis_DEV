#ifndef ELPIS_D31B_BOUNDED_LATTICE_H
#define ELPIS_D31B_BOUNDED_LATTICE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define D31B_HEADS 2u
#define D31B_DK 32u
#define D31B_DV 32u
#define D31B_FANOUT 9u
#define D31B_TENSOR_FLOATS (D31B_HEADS*D31B_DK*D31B_DV)
#define D31B_INPUT_FLOATS (D31B_HEADS*D31B_DK)
#define D31B_OUTPUT_FLOATS (D31B_HEADS*D31B_DV)
#define D31B_CAP_MAX 8192u
typedef struct d31b_arena d31b_arena;
typedef enum { D31B_REFUSE=0, D31B_LRU_REUSE=1 } d31b_policy;
typedef enum { D31B_OK=0, D31B_ERR_ARGUMENT=-1, D31B_ERR_CAPACITY=-2,
               D31B_ERR_NUMERICAL=-3, D31B_ERR_CLOCK=-4 } d31b_status;
/* The ONLY heap allocation is creation; no heap calls in read/write. */
d31b_arena *d31b_create(size_t cells, d31b_policy policy);
void d31b_destroy(d31b_arena *a);
/* No output mutation on non-OK. No cell allocation on read. */
d31b_status d31b_read(const d31b_arena *a, const double xy[2],
                      const float query[D31B_INPUT_FLOATS],
                      float output[D31B_OUTPUT_FLOATS]);
/* Atomic nine-neighborhood write: all capacity/numerics validated before commit. */
d31b_status d31b_write(d31b_arena *a, const double xy[2],
                       const float key[D31B_INPUT_FLOATS],
                       const float value[D31B_OUTPUT_FLOATS],
                       const float alpha[D31B_HEADS],
                       const float beta[D31B_HEADS]);
size_t d31b_capacity(const d31b_arena *a);
size_t d31b_occupied(const d31b_arena *a);
size_t d31b_allocated_bytes(const d31b_arena *a);
size_t d31b_cell_bytes(void);
uint64_t d31b_evictions(const d31b_arena *a);
uint64_t d31b_commits(const d31b_arena *a);
uint64_t d31b_cell_updates(const d31b_arena *a);
/* Counts exclude read computation and initial creation; fixed 9 updates per commit. */
#ifdef __cplusplus
}
#endif
#endif
