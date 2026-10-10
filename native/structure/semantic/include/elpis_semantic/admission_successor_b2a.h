/* B2a: prepare a complete immutable successor in memory after fresh B1p audit.
 * No filesystem writes, active-head mutation, or admission capability.
 * Admission authority catalog pin and base-manifest pin MUST be trusted inputs.
 */
#ifndef ELPIS_SEMANTIC_ADMISSION_SUCCESSOR_B2A_H
#define ELPIS_SEMANTIC_ADMISSION_SUCCESSOR_B2A_H
#include "elpis_semantic/admission_populated_b1o.h"
#include "elpis_semantic/snapshot.h"
#include "elpis_semantic/type_registry.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct semantic_b2a_witness {
    const uint8_t *catalog;
    size_t catalog_bytes;
    const hacf_digest *trusted_catalog_sha256;
    const semantic_snapshot_manifest *base;
    const hacf_digest *trusted_base_manifest_digest;
    const char *const *base_segment_paths;
    uint32_t base_segment_count;
    const semantic_type_registry *registry;
    const semantic_query_overlay *overlay;
    const elpis_evidence_typing_bundle_v1 *typing_bundle;
    const elpis_evidence_admission_policy_v1 *policy;
    const elpis_evidence_admission_v1 *layer;
    const elpis_evidence_admission_decision_v1 *decision;
    const elpis_evidence_admission_receipt_v1 *receipt;
    const semantic_b1b_claim_source *claim_source;
    const uint8_t *raw_bundle_json;
    size_t raw_bundle_bytes;
    const uint8_t *canonical_payload;
    size_t canonical_payload_bytes;
} semantic_b2a_witness;

/* On SEMANTIC_OK, both outputs are fully built. On failure they are untouched.
 * NOT a write permit. Callers must independently authorize and serialize any
 * artifacts; active-head CAS/recovery is outside this phase. All inputs must
 * remain immutable for the duration of the call. */
int semantic_b2a_prepare_successor(const semantic_b2a_witness *w,
                                   semantic_segment_record *segment_out,
                                   semantic_snapshot_manifest *successor_out);
#ifdef __cplusplus
}
#endif
#endif
