/* B1b: exact source-byte cross-binding of B1a claim admission.
 * Audit-only; no writes, privilege, promotion, or durable snapshot publication.
 * Caller-provided policy/package pins are comparisons, NOT authenticated authority.
 * A separately trusted deployment boundary must supply and protect those pins.
 * Narrow v1: one PRIMARY attachment and one exact span per new claim.
 */
#ifndef ELPIS_SEMANTIC_ADMISSION_SOURCE_B1B_H
#define ELPIS_SEMANTIC_ADMISSION_SOURCE_B1B_H
#include "elpis_semantic/admission_preflight_b1a.h"
#include "elpis_semantic/evidence_claim_candidate.h"
#include "elpis_semantic/evidence_span.h"
#include "elpis_semantic/retrieval_item_attachment.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#define SEMANTIC_B1B_MAX_RAW_ITEM_BYTES (1u << 20)
typedef struct semantic_b1b_claim_source {
    const elpis_evidence_claim_candidate_v1 *candidate;
    const elpis_evidence_span_v1 *span;
    const elpis_retrieval_item_attachment_v1 *attachment;
    const uint8_t *item_text;
    size_t item_text_bytes;
} semantic_b1b_claim_source;
/* On any failure, report_out remains untouched. No partial claim acceptance.
 * Every indexed decision and receipt must correspond to the same indexed
 * candidate/source tuple; extraneous bundle spans and claims are rejected.
 * Accepted report is descriptive, not an admission or publication token. */
int semantic_b1b_claim_source_audit(
    const semantic_snapshot_manifest *base,
    const semantic_query_overlay *overlay,
    const elpis_evidence_typing_bundle_v1 *bundle,
    const elpis_evidence_admission_policy_v1 *policy,
    const hacf_digest *expected_policy_pin,
    const hacf_digest *expected_retrieval_bundle_package_pin,
    const elpis_evidence_admission_v1 *layer,
    const elpis_evidence_admission_decision_v1 *decisions,
    const elpis_evidence_admission_receipt_v1 *receipts,
    const semantic_b1b_claim_source *sources,
    size_t count,
    hacf_digest *report_out);
#ifdef __cplusplus
}
#endif
#endif
