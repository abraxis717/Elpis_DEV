#include "elpis/ecsg_state.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
test_roundtrip(void)
{
    enum {
        D = 6,
        N = 36
    };

    double w0[D * N];
    double x[2 * D];
    double y[2];
    double original_w[D * N];
    double restored_w[D * N];
    elpis_ecsg_state *state = NULL;
    elpis_ecsg_state *restored = NULL;
    double *scratch;
    uint8_t *snapshot;
    size_t scratch_count;
    size_t snapshot_size;
    size_t i;

    for (i = 0u; i < D * N; ++i) {
        w0[i] = ((double)((int)(i % 19u) - 9)) / 41.0;
    }

    for (i = 0u; i < 2u * D; ++i) {
        x[i] = ((double)((int)(i % 7u) - 3)) / 5.0;
    }

    y[0] = 0.25;
    y[1] = -0.5;

    assert(elpis_ecsg_state_create(D, N, w0, &state) ==
           ELPIS_ECSG_MATH_OK);

    scratch_count = elpis_ecsg_state_gd_step_scratch_f64(D, N, 2u);
    scratch = (double *)calloc(scratch_count, sizeof(*scratch));
    assert(scratch != NULL);

    assert(elpis_ecsg_state_gd_step_f64(
        state, x, y, 2u, ELPIS_ECSG_BRANCH36_REFERENCE_LR,
        scratch, scratch_count) == ELPIS_ECSG_MATH_OK);

    assert(elpis_ecsg_state_epoch(state) == UINT64_C(1));
    assert(elpis_ecsg_state_copy_w(state, original_w, D * N) ==
           ELPIS_ECSG_MATH_OK);

    snapshot_size = elpis_ecsg_state_snapshot_size(state);
    assert(snapshot_size == 40u + D * N * sizeof(double));

    snapshot = (uint8_t *)malloc(snapshot_size);
    assert(snapshot != NULL);

    assert(elpis_ecsg_state_snapshot_write(
        state, snapshot, snapshot_size) == ELPIS_ECSG_MATH_OK);

    assert(elpis_ecsg_state_snapshot_restore(
        snapshot, snapshot_size, &restored) == ELPIS_ECSG_MATH_OK);

    assert(restored != NULL);
    assert(elpis_ecsg_state_dim(restored) == D);
    assert(elpis_ecsg_state_width(restored) == N);
    assert(elpis_ecsg_state_epoch(restored) == UINT64_C(1));

    assert(elpis_ecsg_state_copy_w(
        restored, restored_w, D * N) == ELPIS_ECSG_MATH_OK);
    assert(memcmp(original_w, restored_w, sizeof(original_w)) == 0);

    free(snapshot);
    free(scratch);
    assert(elpis_ecsg_state_destroy(&state) == ELPIS_ECSG_MATH_OK);
    assert(elpis_ecsg_state_destroy(&restored) == ELPIS_ECSG_MATH_OK);
}

static void
test_corruption_and_truncation_rejected(void)
{
    const double w0[] = {0.25, -0.5};
    elpis_ecsg_state *state = NULL;
    elpis_ecsg_state *restored = NULL;
    uint8_t *snapshot;
    size_t snapshot_size;

    assert(elpis_ecsg_state_create(1u, 2u, w0, &state) ==
           ELPIS_ECSG_MATH_OK);

    snapshot_size = elpis_ecsg_state_snapshot_size(state);
    snapshot = (uint8_t *)malloc(snapshot_size);
    assert(snapshot != NULL);

    assert(elpis_ecsg_state_snapshot_write(
        state, snapshot, snapshot_size) == ELPIS_ECSG_MATH_OK);

    assert(elpis_ecsg_state_snapshot_restore(
        snapshot, snapshot_size - 1u, &restored) ==
        ELPIS_ECSG_MATH_INVALID);
    assert(restored == NULL);

    snapshot[0] ^= UINT8_C(1);
    assert(elpis_ecsg_state_snapshot_restore(
        snapshot, snapshot_size, &restored) ==
        ELPIS_ECSG_MATH_INVALID);
    assert(restored == NULL);

    free(snapshot);
    assert(elpis_ecsg_state_destroy(&state) == ELPIS_ECSG_MATH_OK);
}

int
main(void)
{
    test_roundtrip();
    test_corruption_and_truncation_rejected();

    puts("PASS_ECS_G_SNAPSHOT_R0_C");
    return 0;
}
