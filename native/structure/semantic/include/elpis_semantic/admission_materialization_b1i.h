/* B1i: restricted canonical identity-projection payload-byte audit.
 * No semantic translation, graph nonmembership claim, readjudication authority,
 * storage operation or publication permission. Canonical payload bytes here
 * must be the IDENTICAL byte sequence underlying both claim digest fields and
 * the query-local node's payload digest; other encodings are not qualified.
 */
#ifndef ELPIS_SEMANTIC_ADMISSION_MATERIALIZATION_B1I_H
#define ELPIS_SEMANTIC_ADMISSION_MATERIALIZATION_B1I_H
#include "elpis_semantic/admission_decision_b1h.h"
#ifdef __cplusplus
extern "C" {
#endif
#define SEMANTIC_B1I_MAX_CANONICAL_BYTES (1u << 20)
int semantic_b1i_identity_projection_audit(
    const uint8_t *catalog, size_t catalog_bytes,
    const hacf_digest *trusted_catalog_sha256,
    const semantic_snapshot_manifest *base,
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
