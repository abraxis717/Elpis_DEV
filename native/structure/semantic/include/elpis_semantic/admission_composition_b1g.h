/* B1g: read-only *composition* of already defined B1f and hardened B1b audits.
 * ONE primary retrieval item and ONE new admitted claim ONLY.
 * This descriptive result is NOT readjudication, authorization, publication,
 * persistence, or a portable admission token. Externally supplied catalog pin
 * MUST originate outside the untrusted candidate. No filesystem or Python.
 */
#ifndef ELPIS_SEMANTIC_ADMISSION_COMPOSITION_B1G_H
#define ELPIS_SEMANTIC_ADMISSION_COMPOSITION_B1G_H
#include "elpis_semantic/admission_item_b1f.h"
#include "elpis_semantic/admission_source_b1b.h"
#ifdef __cplusplus
extern "C" {
#endif
int semantic_b1g_single_claim_composition_audit(
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
