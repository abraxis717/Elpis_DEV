/* B1h: restricted single-claim decision-field replay over B1g's exact witness.
 * Descriptive only. Not full readjudication, admission, or publication.
 * The caller's catalog SHA MUST be anchored by independent deployment authority.
 */
#ifndef ELPIS_SEMANTIC_ADMISSION_DECISION_B1H_H
#define ELPIS_SEMANTIC_ADMISSION_DECISION_B1H_H
#include "elpis_semantic/admission_composition_b1g.h"
#ifdef __cplusplus
extern "C" {
#endif
int semantic_b1h_claim_decision_replay_audit(
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
    hacf_digest *descriptive_report_out);
#ifdef __cplusplus
}
#endif
#endif
