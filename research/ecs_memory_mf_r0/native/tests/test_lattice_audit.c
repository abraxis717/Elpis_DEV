/* Characterization of the PR #47 D31-C lattice as it is (docs/research/ECS_MEMORY_MF_R0.md, "Audit").
 * These are not desired properties; they pin the audited behaviour so a later change to the frozen donor
 * sources cannot pass silently. */
#include "indexed_lattice.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
    d31c_arena *a = d31c_create(18, D31C_LRU_REUSE, 96);
    assert(a);
    float k[64], v[64], q[64], o[64];
    const float alpha[2] = {1.0f, 1.0f}, beta[2] = {1.0f, 1.0f};
    for (size_t i = 0; i < 64; ++i) { k[i] = 0.125f; v[i] = 1.0f; q[i] = 0.125f; }
    const double A[2] = {0.1, 0.1}, B[2] = {20.1, 20.1}, C[2] = {40.1, 40.1};
    assert(d31c_write(a, A, k, v, alpha, beta) == D31C_OK);
    assert(d31c_write(a, B, k, v, alpha, beta) == D31C_OK);
    const uint64_t lookups = d31c_index_lookups(a);
    for (int r = 0; r < 5; ++r) assert(d31c_read(a, A, q, o) == D31C_OK);
    /* A1: reads are not counted by any work counter (the header's "each read makes 9 probes" is uncounted). */
    assert(d31c_index_lookups(a) == lookups);
    /* A2: "LRU" is write recency: A, read most recently, is still the victim of the next admission. */
    assert(d31c_write(a, C, k, v, alpha, beta) == D31C_OK);
    assert(d31c_evictions(a) == 9);
    assert(d31c_read(a, A, q, o) == D31C_OK && o[0] == 0.0f);
    assert(d31c_read(a, B, q, o) == D31C_OK && o[0] != 0.0f);
    /* A3: admission planning scans O(capacity) per missing cell (no free list). */
    assert(d31c_admission_scan_steps(a) > 9u * 18u);
    d31c_destroy(a);
    printf("PASS_D31C_AUDIT_CHARACTERIZATION\n");
    return 0;
}
