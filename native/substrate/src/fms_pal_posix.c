/* fms_pal_posix.c - the CPU PAL: RAM only. Per-instance state; no globals.
 *
 * It performs no filesystem operation: no open, no write, no rename, no fsync, no unlink. ABI v2's durable cold
 * store (a content-addressed file per demoted object under a locked cold root) is retired with FMS ABI v3
 * (docs/ELPIS_MISSION.md, Amendment 3); immutable COLD backing is the read-only file-asset service.
 */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE 1

#include "elpis/fms_pal_posix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- PAL entry points ------------------------------------------------------ */

static uint64_t p_now(void *self) {
    (void)self;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int p_ram_alloc(void *self, uint64_t n, void **out) {
    (void)self;
    *out = malloc((size_t)n);
    return *out ? 0 : FMS_PAL_ENOMEM;
}

static void p_ram_free(void *self, void *p, uint64_t n) { (void)self; (void)n; free(p); }

static int p_hot_profile(void *self, fms_hot_profile *out) {
    (void)self;
    memset(out, 0, sizeof *out);
    out->available = 0;
    out->domain = FMS_DOM_DEVICE;
    snprintf(out->backend, sizeof out->backend, "none");
    snprintf(out->device, sizeof out->device, "no accelerator probed");
    return 0;
}

static uint64_t p_bandwidth(void *self, int from, int to) {
    (void)self;
    int lo = from < to ? from : to, hi = from < to ? to : from;
    if (lo == FMS_HOT && hi == FMS_WARM) return 12ull << 30;
    return 1ull << 30;
}

static void p_destroy(void *self) { free(self); }

fms_pal *fms_pal_posix_create_ram_only(void) {
    fms_pal *p = (fms_pal *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->self = p;
    p->abi = FMS_ABI_VERSION;
    p->caps = 0;
    p->destroy = p_destroy;
    p->now_ns = p_now;
    p->hot_profile = p_hot_profile;
    p->ram_alloc = p_ram_alloc;
    p->ram_free = p_ram_free;
    p->bandwidth = p_bandwidth;
    return p;
}

fms_ctx *elpis_fms_create_ram_only(const fms_config *cfg) {
    fms_pal *pal = fms_pal_posix_create_ram_only();
    fms_ctx *ctx;
    if (!pal) return NULL;
    ctx = fms_create(cfg, pal);
    if (!ctx) pal->destroy(pal->self);
    return ctx;
}
