/* elpis/fms_pal_posix.h - the CPU PAL: RAM only. No device tier and no COLD tier, so a context built on it is WARM
 * only and any HOT request follows cfg.hot_absent_policy. It performs no filesystem operation of any kind.
 *
 * ABI v3 retired the durable POSIX cold store (fms_pal_posix_create(cold_root), elpis_fms_create_posix and the token
 * accessors: temp file, write, fsync, rename, directory fsync and unlink of demoted objects). It exists in no form,
 * behind no flag (docs/ELPIS_MISSION.md, Amendment 3). */
#ifndef ELPIS_FMS_PAL_POSIX_H
#define ELPIS_FMS_PAL_POSIX_H

#include "elpis/fms_pal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* RAM-only PAL. Ownership passes to fms_create(); if the context is never created, call pal->destroy(pal->self). */
fms_pal *fms_pal_posix_create_ram_only(void);

/* Convenience lifecycle constructor over the RAM-only PAL; disposes the PAL if FMS rejects cfg. */
fms_ctx *elpis_fms_create_ram_only(const fms_config *cfg);

#ifdef __cplusplus
}
#endif
#endif
