/* B1k: descriptive-only composition of B1i identity-projection with B1j
 * independently pinned GENESIS-ONLY empty serialized segment chain.
 * A successful result is not durable publication permission. */
#ifndef ELPIS_SEMANTIC_ADMISSION_GENESIS_COMPOSITION_B1K_H
#define ELPIS_SEMANTIC_ADMISSION_GENESIS_COMPOSITION_B1K_H
#include "elpis_semantic/admission_materialization_b1i.h"
#include "elpis_semantic/admission_base_b1j.h"
#ifdef __cplusplus
extern "C" {
#endif
int semantic_b1k_genesis_claim_audit(
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
