/* B1a: read-only native claim-admission CORRESPONDENCE AUDIT, not admission authority.
 * No filesystem I/O, no persistence, no activation, no promotion, no Python.
 * Strict claim-only subset. Untrusted caller can fabricate coherent records;
 * B1b MUST independently re-adjudicate actual bytes under deployed policy.
 * Not suitable as a publication authorization token.
 */
#ifndef ELPIS_SEMANTIC_ADMISSION_PREFLIGHT_B1A_H
#define ELPIS_SEMANTIC_ADMISSION_PREFLIGHT_B1A_H
#include "elpis_semantic/query_overlay.h"
#include "elpis_semantic/evidence_admission.h"
#include "elpis_semantic/evidence_admission_decision.h"
#include "elpis_semantic/evidence_admission_receipt.h"
#include "elpis_semantic/evidence_admission_policy.h"
#include "elpis_semantic/evidence_typing_bundle.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#define SEMANTIC_B1A_MAX_CLAIMS 64u
/* All inputs read-only. audit_digest_out is descriptive; never a promotion permit.
 * No output is written on failure. No partial admission.
 * Fail closed on mixed/rejected evidence, extra overlay records, missing source
 * spans/attachments, incorrect decision/receipt hashes or inconsistent ordering.
 * This is a separate preflight stage, NOT proof of independent policy provenance.
 */
int semantic_b1a_claim_correspondence_audit(
    const semantic_snapshot_manifest *base,
    const semantic_query_overlay *overlay,
    const elpis_evidence_typing_bundle_v1 *bundle,
    const elpis_evidence_admission_policy_v1 *policy,
    const hacf_digest *independent_policy_pin,
    const elpis_evidence_admission_v1 *layer,
    const elpis_evidence_admission_decision_v1 *decisions,
    const elpis_evidence_admission_receipt_v1 *receipts,
    size_t count,
    hacf_digest *audit_digest_out);
#ifdef __cplusplus
}
#endif
#endif
