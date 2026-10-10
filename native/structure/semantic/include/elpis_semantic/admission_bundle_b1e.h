/* B1e: recompute HACF package identity from exact retrieval JSON bytes.
 * This is a READ-ONLY descriptive gate, not an admission/commit/promotion token.
 * Caller-supplied trusted_catalog_sha256 MUST be pinned independently of the
 * untrusted catalog and retrieval payload. This API cannot establish the pin.
 * It verifies canonical outer serialization and JSON grammar, but deliberately
 * does NOT adjudicate the inner retrieval items, their provenance, or policy.
 */
#ifndef ELPIS_SEMANTIC_ADMISSION_BUNDLE_B1E_H
#define ELPIS_SEMANTIC_ADMISSION_BUNDLE_B1E_H
#include "elpis_semantic/admission_authority_b1d.h"
#ifdef __cplusplus
extern "C" {
#endif
int semantic_b1e_retrieval_package_audit(
    const uint8_t *catalog, size_t catalog_bytes,
    const hacf_digest *trusted_catalog_sha256,
    const semantic_snapshot_manifest *base,
    const elpis_evidence_admission_policy_v1 *policy,
    const uint8_t *raw_bundle_json, size_t raw_bundle_bytes,
    hacf_digest *verified_package_out,
    hacf_digest *descriptive_audit_out);
#ifdef __cplusplus
}
#endif
#endif
