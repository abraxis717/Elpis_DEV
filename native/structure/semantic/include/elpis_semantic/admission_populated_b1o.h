/* B1o: verified populated-genesis node NONMEMBERSHIP plus exact claim witness.
 * Descriptive only. No admission capability or publication authorization.
 */
#ifndef ELPIS_SEMANTIC_ADMISSION_POPULATED_B1O_H
#define ELPIS_SEMANTIC_ADMISSION_POPULATED_B1O_H
#include "elpis_semantic/admission_materialization_b1i.h"
#include "elpis_semantic/admission_populated_b1l.h"
#ifdef __cplusplus
extern "C" {
#endif
int semantic_b1o_populated_claim_novelty_audit(
    const uint8_t *catalog, size_t catalog_bytes,
    const hacf_digest *trusted_catalog_sha256,
    const semantic_snapshot_manifest *base,
    const hacf_digest *trusted_base_manifest_digest,
    const char *const *base_segment_paths, uint32_t base_segment_count,
    const semantic_query_overlay *overlay,
    const elpis_evidence_typing_bundle_v1 *typing_bundle,
    const elpis_evidence_admission_policy_v1 *policy,
    const elpis_evidence_admission_v1 *layer,
    const elpis_evidence_admission_decision_v1 *decision,
    const elpis_evidence_admission_receipt_v1 *receipt,
    const semantic_b1b_claim_source *claim_source,
    const uint8_t *raw_bundle_json, size_t raw_bundle_bytes,
    const uint8_t *canonical_payload, size_t canonical_payload_bytes,
    hacf_digest *descriptive_report_out);
#ifdef __cplusplus
}
#endif
#endif
