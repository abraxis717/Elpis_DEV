#ifndef ELPIS_D31C_INDEXED_LATTICE_H
#define ELPIS_D31C_INDEXED_LATTICE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define D31C_HEADS 2u
#define D31C_DK 32u
#define D31C_DV 32u
#define D31C_FANOUT 9u
#define D31C_TENSOR_FLOATS (D31C_HEADS*D31C_DK*D31C_DV)
#define D31C_INPUT_FLOATS (D31C_HEADS*D31C_DK)
#define D31C_OUTPUT_FLOATS (D31C_HEADS*D31C_DV)
#define D31C_CAP_MAX 8192u
typedef struct d31c_arena d31c_arena;
typedef enum { D31C_REFUSE=0, D31C_LRU_REUSE=1 } d31c_policy;
typedef enum { D31C_OK=0, D31C_ERR_ARGUMENT=-1, D31C_ERR_CAPACITY=-2,
               D31C_ERR_NUMERICAL=-3, D31C_ERR_CLOCK=-4, D31C_ERR_RANGE=-5 } d31c_status;
/* Three heap allocations during creation (arena,cells,direct index); no heap calls in read/write. */
d31c_arena *d31c_create(size_t cells, d31c_policy policy, int radius);
void d31c_destroy(d31c_arena *a);
/* No output mutation on non-OK. No cell allocation on read. */
d31c_status d31c_read(const d31c_arena *a, const double xy[2],
                      const float query[D31C_INPUT_FLOATS],
                      float output[D31C_OUTPUT_FLOATS]);
/* Atomic nine-neighborhood write: all capacity/numerics validated before commit. */
d31c_status d31c_write(d31c_arena *a, const double xy[2],
                       const float key[D31C_INPUT_FLOATS],
                       const float value[D31C_OUTPUT_FLOATS],
                       const float alpha[D31C_HEADS],
                       const float beta[D31C_HEADS]);
size_t d31c_capacity(const d31c_arena *a);
size_t d31c_occupied(const d31c_arena *a);
size_t d31c_allocated_bytes(const d31c_arena *a);
size_t d31c_cell_bytes(void);
/* Byte accounting includes a fixed direct coordinate map.  */
size_t d31c_index_bytes(const d31c_arena *a);
/* Counts 9 address-map probes per SUCCESSFUL write; each read makes 9 probes. */
uint64_t d31c_index_lookups(const d31c_arena *a);
uint64_t d31c_admission_scan_steps(const d31c_arena *a);
uint64_t d31c_evictions(const d31c_arena *a);
uint64_t d31c_commits(const d31c_arena *a);
uint64_t d31c_cell_updates(const d31c_arena *a);
/* Counts exclude read computation and initial creation; fixed 9 updates per commit. */
#ifdef __cplusplus
}
#endif
#endif
