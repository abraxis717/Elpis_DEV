/* File-asset page-object bridge. FMS ABI v2 remains residency authority.
 * Immutable external replicas are catalogued by the file-asset extension;
 * this context has no writable cold tier and cannot duplicate a giant asset.
 */
#include "elpis/fms.h"
#include "elpis/fms_pal_posix.h"
#include <string.h>

static int create_pages(fms_pal *pal, uint64_t warm_bytes,
                        uint32_t max_pages, int absent_policy, fms_ctx **out) {
    if (!pal) return FMS_E_IO;
    fms_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.tier_budget[FMS_WARM] = warm_bytes;
    cfg.domain_ceiling[FMS_DOM_RAM] = warm_bytes;
    cfg.high_wm = 0.95f;
    cfg.low_wm = 0.75f;
    cfg.max_objects = max_pages;
    cfg.hot_absent_policy = (uint8_t)absent_policy;
    cfg.cold_absent_policy = FMS_REJECT;
    *out = fms_create(&cfg, pal);
    if (!*out) { pal->destroy(pal->self); return FMS_E_NOMEM; }
    return FMS_OK;
}

int elpis_fms_file_create_memory(uint64_t warm_bytes, uint32_t max_pages,
                                int absent_policy, fms_ctx **out) {
    if (!out || !warm_bytes || !max_pages ||
        (absent_policy != FMS_FOLD_DOWN && absent_policy != FMS_REJECT))
        return FMS_E_INVAL;
    *out = NULL;
    return create_pages(fms_pal_posix_create_ram_only(), warm_bytes, max_pages, absent_policy, out);
}

int elpis_fms_file_stats(fms_ctx *ctx, uint64_t *out) {
    if (!ctx || !out) return FMS_E_INVAL;
    fms_stats s;
    fms_get_stats(ctx, &s);
    for (int i = 0; i < FMS_NTIERS; ++i) out[i] = s.tier_bytes[i];
    for (int i = 0; i < FMS_NDOMAINS; ++i) out[3+i] = s.domain_bytes[i];
    out[6] = s.objects;
    out[7] = s.pinned_bytes;
    return FMS_OK;
}
