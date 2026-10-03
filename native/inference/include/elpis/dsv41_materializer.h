#ifndef ELPIS_DSV41_MATERIALIZER_H
#define ELPIS_DSV41_MATERIALIZER_H
#include "elpis/dsv41_clock.h"
#include "elpis/fms_file_service.h"
#ifdef __cplusplus
extern "C" {
#endif

/* DSV4.1 Native Materializer R1: the production Elpis host service behind the
 * unchanged elpis_dsv41_materializer_v1 clock table. Never accelerator-provider
 * authority: the provider receives decoded rows and expert bytes only.
 *
 * Two phases:
 *   1. Cold admission (control plane): create, admit_asset (already-authorized
 *      read-only descriptors + pinned page maps/stamps), add_bank, add_expert,
 *      seal. Admission creates no authority; it consumes FMSFileAssets'.
 *   2. Runtime (sealed): rows/expert/release/quiesce through bind(). Bounded,
 *      native, no interpreter callback. Admission is refused after seal.
 *
 * Handles are non-reused identities (64 live maximum); the clock context is
 * the identity itself, so a destroyed service yields STALE without any
 * dereference. One mutex per instance; independent instances share no
 * mutable state except the identity table (held only for lookup).
 * destroy returns BUSY while a call is in flight or a span is borrowed;
 * destroy of a previously issued identity is idempotent. */
enum { ELPIS_DSV41_MATERIALIZER_ABI_V1 = 1 };
enum { ELPIS_DSV41_ROW_F32_LE = 1, ELPIS_DSV41_ROW_E4M3_E8M0_BF16 = 2 };
typedef uint64_t elpis_dsv41_materializer;

typedef struct {
    uint32_t abi_version, reserved;
    elpis_file_service_config_v1 file; /* FMS residency and page staging budgets */
    uint32_t layers, expert_count;     /* expert table [layers][expert_count + 1] */
    uint64_t image_bytes;              /* canonical w1 || w3 || w2 image */
    uint64_t staging_bytes;            /* TensorStore staging budget: bounds one part */
    uint32_t max_banks, max_rows, max_dimension, reserved2;
} elpis_dsv41_materializer_config_v1;

typedef struct {
    uint32_t layer, dimension, codec, asset;
    uint64_t rows, offset;             /* row-major table at offset of asset */
    uint8_t bank[32];                  /* pinned bank identity */
    uint32_t max_rows, reserved;       /* RowEngine bounds */
    uint64_t max_output_bytes;
} elpis_dsv41_bank_v1;

typedef struct {
    uint32_t layer, expert;            /* expert == expert_count: shared */
    uint32_t assets[3], reserved;      /* w1, w3, w2 */
    uint64_t offsets[3], sizes[3];
    uint8_t digests[96];               /* admitted binding digests, role order */
} elpis_dsv41_expert_v1;

typedef struct {
    elpis_file_service_stats_v1 file;
    uint64_t row_calls, row_requests, unique_rows, row_bytes, row_ns;
    uint64_t expert_calls, expert_bytes, expert_chunks, expert_ns;
    uint64_t span_acquires, span_releases, live_spans, stale_releases, forced_releases;
    uint64_t quiesces, refusals, staging_budget, staging_high_water, samples;
} elpis_dsv41_materializer_stats_v1;

uint32_t elpis_dsv41_materializer_abi_version(void);
elpis_clock_code elpis_dsv41_materializer_create(const elpis_dsv41_materializer_config_v1 *,
                                                 elpis_dsv41_materializer *out);
elpis_clock_code elpis_dsv41_materializer_admit_asset(elpis_dsv41_materializer, const elpis_file_asset_v1 *,
                                                      uint32_t *asset);
elpis_clock_code elpis_dsv41_materializer_add_bank(elpis_dsv41_materializer, const elpis_dsv41_bank_v1 *);
elpis_clock_code elpis_dsv41_materializer_add_expert(elpis_dsv41_materializer, const elpis_dsv41_expert_v1 *);
elpis_clock_code elpis_dsv41_materializer_seal(elpis_dsv41_materializer);
/* NativeMaterializer factory: context is (void *)(uintptr_t)identity. */
void elpis_dsv41_materializer_bind(void *context, elpis_dsv41_materializer_v1 *table);
elpis_clock_code elpis_dsv41_materializer_stats(elpis_dsv41_materializer, elpis_dsv41_materializer_stats_v1 *);
/* Resident pages, oldest first. *count receives the total. */
elpis_clock_code elpis_dsv41_materializer_pages(elpis_dsv41_materializer, uint32_t *assets, uint64_t *pages,
                                                size_t capacity, size_t *count);
/* Most recent acquisitions (ring of 4096): latency, resident bytes, kind 1 rows / 2 expert. */
elpis_clock_code elpis_dsv41_materializer_samples(elpis_dsv41_materializer, uint64_t *latency_ns,
                                                  uint64_t *resident_bytes, uint32_t *kinds,
                                                  size_t capacity, size_t *count);
elpis_clock_code elpis_dsv41_materializer_evict(elpis_dsv41_materializer);
elpis_clock_code elpis_dsv41_materializer_quiesce(elpis_dsv41_materializer);
elpis_clock_code elpis_dsv41_materializer_destroy(elpis_dsv41_materializer);
/* Pure row codec (rows.decode_row): exact F32 LE out, ENCODING on rejection. */
elpis_clock_code elpis_dsv41_row_decode(uint32_t codec, const uint8_t *raw, size_t raw_bytes,
                                        uint32_t dimension, float *out);
uint64_t elpis_dsv41_row_bytes(uint32_t codec, uint32_t dimension);
#ifdef __cplusplus
}
#endif
#endif
