/* elpis/fms_file_service.h - native verified file-asset page service, ABI v1.
 *
 * Substrate-generic runtime half of elpis.substrate.file_assets. It contains no
 * model semantics and creates no authority:
 *
 *   - It never opens, resolves or discovers a path. Admission receives an
 *     already-open, already-authorized READ-ONLY regular-file descriptor plus
 *     the immutable pinned geometry, page map and identity stamp recorded by
 *     cold authority (FMSFileAssets.register). The descriptor is duplicated
 *     (F_DUPFD_CLOEXEC); the duplicate is owned by the service and closed on
 *     destroy. The caller's descriptor is never retained.
 *   - Every page load re-checks the descriptor stamp, reads with bounded pread
 *     (EINTR retried; zero-byte return is IO), verifies the exact
 *     elpis.inference.raw-bytes.r0 page digest, re-checks the stamp, and only
 *     then registers the verified bytes as one FMS ABI v2 WARM object.
 *   - FMS ABI v2 is the residency authority: one private FMS context per
 *     service. Pages are evicted least-recently-used, only when unleased, and
 *     only when FMS refuses a registration. Leased pages cannot be evicted.
 *   - Buffered pread: the Linux page cache is neither charged nor bounded.
 *
 * Thread safety: every entry point may be called concurrently; one mutex per
 * service serializes I/O (the same R0 policy as the Python provider). No
 * global mutable state is shared between services.
 */
#ifndef ELPIS_FMS_FILE_SERVICE_H
#define ELPIS_FMS_FILE_SERVICE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { ELPIS_FILE_SERVICE_ABI_V1 = 1 };

/* Numerically aligned with elpis_clock_code where the meanings coincide. */
typedef enum {
    ELPIS_FILE_OK = 0, ELPIS_FILE_INVALID = 1, ELPIS_FILE_STALE = 2, ELPIS_FILE_BUSY = 3,
    ELPIS_FILE_LIMIT = 4, ELPIS_FILE_DEVICE = 5, ELPIS_FILE_INTEGRITY = 6, ELPIS_FILE_IO = 8,
    ELPIS_FILE_CLOSED = 9, ELPIS_FILE_MISSING = 11, ELPIS_FILE_UNSUPPORTED = 12
} elpis_file_status;

typedef struct {
    uint32_t abi_version, reserved;
    uint64_t warm_bytes;    /* FMS WARM tier and RAM domain ceiling */
    uint64_t staging_bytes; /* page_size <= min(staging_bytes / 4, warm_bytes) */
    uint64_t map_bytes;     /* ceiling on retained page-map digest storage */
    uint32_t max_pages;     /* FMS object ceiling, <= 1 << 24 */
    uint32_t max_assets;    /* <= 65536 */
    uint32_t max_ranges;    /* concurrently live range leases, <= 4096 */
    uint32_t absent_policy; /* FMS_FOLD_DOWN or FMS_REJECT for HOT */
} elpis_file_service_config_v1;

/* (st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns) as recorded at intake. */
typedef struct { uint64_t dev, ino, size; int64_t mtime_ns, ctime_ns; } elpis_file_stamp;

typedef struct {
    uint32_t abi_version, page_size;
    int32_t fd;                  /* borrowed; duplicated by admit */
    uint32_t reserved;
    uint64_t size, page_count;
    elpis_file_stamp stamp;
    const uint8_t *page_digests; /* [page_count][32] raw-bytes.r0 digests */
} elpis_file_asset_v1;

typedef struct {
    uint64_t warm_budget, staging_budget, max_pages, map_bytes, assets;
    uint64_t resident_bytes, resident_high_water, resident_pages, pinned_bytes;
    uint64_t hits, misses, reads, pread_bytes, semantic_bytes, evictions, interrupted_reads;
    uint64_t lease_acquires, lease_releases, range_acquires, range_releases, live_ranges, forced_releases;
    uint64_t integrity_failures, io_failures, staging_high_water;
    uint64_t read_ns, integrity_ns, staging_ns;
} elpis_file_service_stats_v1;

typedef struct elpis_file_service elpis_file_service;

uint32_t elpis_file_service_abi_version(void);
elpis_file_status elpis_file_service_create(const elpis_file_service_config_v1 *, elpis_file_service **out);
/* Refuses non-regular, writable, changed (stamp), mis-sized or mis-mapped objects. */
elpis_file_status elpis_file_service_admit(elpis_file_service *, const elpis_file_asset_v1 *, uint32_t *asset);
/* Range lease of [offset, offset+length). Every covered page is loaded,
 * verified and FMS-leased; on failure nothing acquired by this call remains
 * leased (loaded pages may stay resident, unleased). Handles are
 * (generation, slot) identities; stale handles return STALE. */
elpis_file_status elpis_file_service_acquire(elpis_file_service *, uint32_t asset, uint64_t offset,
                                             uint64_t length, uint64_t *range);
elpis_file_status elpis_file_service_read(elpis_file_service *, uint64_t range, void *dst, uint64_t bytes);
elpis_file_status elpis_file_service_release(elpis_file_service *, uint64_t range);
/* acquire + read + release. */
elpis_file_status elpis_file_service_copy(elpis_file_service *, uint32_t asset, uint64_t offset,
                                          uint64_t length, void *dst);
/* Evict every unleased page of one asset (UINT32_MAX: all); BUSY if any is leased. */
elpis_file_status elpis_file_service_evict(elpis_file_service *, uint32_t asset);
/* Asynchronously abandons in-flight acquisitions at their next page boundary
 * (they return BUSY retaining nothing). Lock-free, idempotent. */
void elpis_file_service_interrupt(elpis_file_service *);
/* Synchronously releases every live range lease (forced); returns the count. */
uint64_t elpis_file_service_release_all(elpis_file_service *);
void elpis_file_service_stats(elpis_file_service *, elpis_file_service_stats_v1 *);
/* Resident pages oldest-first (LRU order). Returns the total resident count. */
size_t elpis_file_service_pages(elpis_file_service *, uint32_t *assets, uint64_t *pages, size_t capacity);
/* Releases every lease, evicts, closes owned descriptors, frees. NULL-safe. */
void elpis_file_service_destroy(elpis_file_service *);
/* Exact elpis.inference.raw-bytes.r0 identity of a byte string. */
void elpis_file_service_raw_digest(const void *data, size_t bytes, uint8_t out[32]);

#ifdef ELPIS_FILE_SERVICE_TESTING
/* Test-build-only fault injection; absent from production objects. */
typedef struct {
    long (*pread)(int fd, void *buf, size_t count, int64_t offset);
    uint64_t fail_alloc_after; /* nonzero: the Nth subsequent allocation fails */
} elpis_file_service_test_hooks;
void elpis_file_service_set_test_hooks(const elpis_file_service_test_hooks *);
#endif

#ifdef __cplusplus
}
#endif
#endif
