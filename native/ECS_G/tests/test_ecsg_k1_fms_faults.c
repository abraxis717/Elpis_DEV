/* K1 residency: a failed FMS release is reconciled, never forgotten (docs/ECS_K1_RUNTIME.md). Link-time wrapping of
 * fms_release injects the failure. The adapter keeps the pin it still holds (FMS has not dropped it, so the image
 * cannot move), never serves a write through a held READ pin, retries the release on the next operation and on
 * close, and never reports a committed operation as refused. */
#include "elpis/ecsg_k1_fms.h"
#include "elpis/fms_pal_posix.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail_release;
fms_status __real_fms_release(fms_ctx *c, fms_id id);
fms_status __wrap_fms_release(fms_ctx *c, fms_id id)
{
    if (fail_release > 0) {
        --fail_release;
        return FMS_E_BUSY;
    }
    return __real_fms_release(c, id);
}

#define OK(x) assert((x) == 0)

enum { D = 6, N = 36, R = 64, WC = D * N };
static double W0[WC], X[R * D], Y[R], O[R];

static uint32_t pins(elpis_ecsg_k1_fms *r, uint64_t id, uint32_t *open_txn)
{
    elpis_ecsg_k1_fms_info info;
    OK(elpis_ecsg_k1_fms_inspect(r, id, &info));
    if (open_txn != NULL) {
        *open_txn = info.transaction_open;
    }
    return info.lease_count;
}

int main(void)
{
    const size_t image = elpis_ecsg_k1_image_bytes(D, N), env = elpis_ecsg_k1_envelope_bytes(D, N);
    const int busy = ELPIS_ECSG_K1_FMS_ERROR_BASE + FMS_E_BUSY;
    uint8_t key[32] = {7};
    uint8_t *got = malloc(env), *want = malloc(env);
    uint64_t id = 0, tok = 0;
    uint32_t open_txn = 0;
    elpis_ecsg_k1 *ref = NULL;
    elpis_ecsg_k1_fms *r = NULL;
    fms_config c;
    fms_ctx *ctx;
    size_t i;
    for (i = 0; i < WC; ++i) W0[i] = 0.01 * (double)((i * 37u) % 23u) - 0.1;
    for (i = 0; i < R * D; ++i) X[i] = 0.02 * (double)((i * 11u) % 29u) - 0.3;
    for (i = 0; i < R; ++i) Y[i] = 0.1 * X[i * D] - 0.05 * X[i * D + 2];
    memset(&c, 0, sizeof(c));
    c.tier_budget[FMS_WARM] = c.domain_ceiling[FMS_DOM_RAM] = image * 8u;
    c.high_wm = 0.9f;
    c.low_wm = 0.7f;
    c.max_objects = 4;
    c.hot_absent_policy = c.cold_absent_policy = FMS_REJECT;
    ctx = fms_create(&c, fms_pal_posix_create_ram_only());
    assert(ctx && elpis_ecsg_k1_fms_create(ctx, 4, &r) == 0);
    OK(elpis_ecsg_k1_fms_register(r, key, D, N, R, W0, &id));
    OK(elpis_ecsg_k1_create(D, N, R, W0, &ref));

    /* a READ operation whose release fails: the result stands, the pin is kept and retried */
    fail_release = 1;
    OK(elpis_ecsg_k1_fms_forward(r, id, X, R, O));
    assert(pins(r, id, NULL) == 1u);
    OK(elpis_ecsg_k1_fms_forward(r, id, X, R, O));   /* reuses the held READ pin, then releases it */
    assert(pins(r, id, NULL) == 0u);

    /* a held READ pin never serves a write: the write must drop it first, and refuses if it cannot */
    fail_release = 1;
    OK(elpis_ecsg_k1_fms_forward(r, id, X, R, O));
    assert(pins(r, id, NULL) == 1u);
    fail_release = 1;
    assert(elpis_ecsg_k1_fms_learn(r, id, X, Y, R, 0.002, 7, NULL) == busy);   /* refused, nothing changed */
    OK(elpis_ecsg_k1_fms_snapshot_write(r, id, got, env));
    OK(elpis_ecsg_k1_snapshot_write(ref, want, env));
    assert(!memcmp(got, want, env));
    assert(pins(r, id, NULL) == 0u);   /* the snapshot's operation retried and released the leftover pin */

    /* a committed write whose release fails is still reported committed */
    fail_release = 1;
    OK(elpis_ecsg_k1_fms_learn(r, id, X, Y, R, 0.002, 7, NULL));
    OK(elpis_ecsg_k1_learn(ref, X, Y, R, 0.002, 7, NULL));
    assert(pins(r, id, NULL) == 1u);
    OK(elpis_ecsg_k1_fms_consolidate(r, id, X, R, NULL));   /* reuses the held WRITE pin */
    OK(elpis_ecsg_k1_consolidate(ref, X, R, NULL));
    assert(pins(r, id, NULL) == 0u);

    /* a transaction commit whose release fails: committed, closed, pin kept until the next release */
    OK(elpis_ecsg_k1_fms_txn_begin(r, id, &tok));
    OK(elpis_ecsg_k1_fms_txn_learn(r, id, tok, X, Y, R, 0.002, 3, NULL));
    fail_release = 1;
    OK(elpis_ecsg_k1_fms_txn_commit(r, id, tok, NULL));
    OK(elpis_ecsg_k1_learn(ref, X, Y, R, 0.002, 3, NULL));
    assert(pins(r, id, &open_txn) == 1u && open_txn == 0u);
    OK(elpis_ecsg_k1_fms_snapshot_write(r, id, got, env));
    OK(elpis_ecsg_k1_snapshot_write(ref, want, env));
    assert(!memcmp(got, want, env));
    assert(pins(r, id, NULL) == 0u);

    /* close retries a held pin and refuses (registered, consistent) while it cannot release */
    fail_release = 1;
    OK(elpis_ecsg_k1_fms_forward(r, id, X, R, O));
    fail_release = 1;
    assert(elpis_ecsg_k1_fms_close(r, &id) == busy && id != 0u);
    assert(pins(r, id, NULL) == 1u);
    OK(elpis_ecsg_k1_fms_close(r, &id));
    assert(id == 0u);
    OK(elpis_ecsg_k1_fms_destroy(&r));
    elpis_ecsg_k1_destroy(&ref);
    free(got);
    free(want);
    printf("ecsg_k1_fms faults: failed releases keep a consistent, retried pin; a READ pin never serves a write; "
           "committed results stand; close refuses until the pin is released\n");
    return 0;
}
