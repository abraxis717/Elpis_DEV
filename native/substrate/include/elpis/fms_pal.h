/* elpis/fms_pal.h - Platform Abstraction Layer, ABI v3.
 * Every PAL instance owns its state. No globals. All entry points must be
 * callable concurrently from multiple threads on distinct objects.
 *
 * ABI v3 (docs/ELPIS_MISSION.md, Amendment 3): no PAL writes storage. ABI v2's COLD entries (cold_put: a durable,
 * content-addressed file per demoted object; cold_get; cold_drop: unlink; cold_digest and cold_bytes over an opaque
 * token) and the FMS_CAP_COLD capability are retired with their return codes: there is no WARM -> COLD operation.
 * Immutable COLD backing (read-only, verified file assets loaded into WARM) is the file-asset service
 * (elpis/fms_file_service.h), which reads and never writes. */
#ifndef ELPIS_FMS_PAL_H
#define ELPIS_FMS_PAL_H

#include "elpis/fms.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Capability bits reported by the PAL. The core honours these literally. 0x2u (ABI v2's FMS_CAP_COLD) is retired and
 * never reused. */
enum { FMS_CAP_DEVICE = 0x1u, FMS_CAP_ZERO_COPY = 0x4u };

/* PAL return codes (negative). Mapped by the core onto fms_status. -2 and -3 (ABI v2's cold digest and size
 * failures) are retired and never reused. */
enum { FMS_PAL_OK = 0, FMS_PAL_EIO = -1, FMS_PAL_ENOMEM = -4, FMS_PAL_EDEVICE = -5 };

/* How a HOT allocation consumes physical resources. */
typedef struct fms_hot_profile {
    uint8_t  available;        /* 0 => the core must apply hot_absent_policy */
    uint8_t  domain;           /* FMS_DOM_* charged by a HOT allocation */
    uint8_t  zero_copy;        /* 1 => hot_alloc aliases the host allocation; no duplicate bytes */
    uint8_t  reserved;
    uint64_t stage_bytes_max;  /* transient staging buffer charged to FMS_DOM_RAM during a hop */
    char     backend[32];      /* "cpu-reference", "vulkan", ... */
    char     device[96];
} fms_hot_profile;

/* Fence query result. */
enum { FMS_FENCE_COMPLETE = 0, FMS_FENCE_PENDING = 1, FMS_FENCE_LOST = -1 };

struct fms_pal {
    void    *self;
    uint32_t abi;              /* must equal FMS_ABI_VERSION */
    uint32_t caps;

    void     (*destroy)(void *self);
    uint64_t (*now_ns)(void *self);

    /* --- HOT (accelerator-addressable residency) --------------------------- */
    int  (*hot_profile)(void *self, fms_hot_profile *out);
    /* host_alias is non-NULL only when the profile advertises zero_copy. */
    int  (*hot_alloc)(void *self, uint64_t bytes, void *host_alias, void **out);
    void (*hot_free)(void *self, void *p, uint64_t bytes);
    int  (*hot_upload)(void *self, void *dev_dst, const void *host_src, uint64_t bytes);
    int  (*hot_download)(void *self, void *host_dst, const void *dev_src, uint64_t bytes);
    int  (*fence_query)(void *self, void *fence);   /* FMS_FENCE_* */

    /* --- WARM (CPU-addressable residency) ---------------------------------- */
    int  (*ram_alloc)(void *self, uint64_t bytes, void **out);
    void (*ram_free)(void *self, void *p, uint64_t bytes);

    uint64_t (*bandwidth)(void *self, int from_tier, int to_tier);
};

#ifdef __cplusplus
}
#endif
#endif /* ELPIS_FMS_PAL_H */
