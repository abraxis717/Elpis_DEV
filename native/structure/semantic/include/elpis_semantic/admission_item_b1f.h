/* B1f: read-only, independently pinned exact one-primary-item correspondence.
 * This is a descriptive audit, NEVER a publication or admission permit.
 * Only canonical native HACF retrieval-bundle JSON is accepted; exactly one
 * primary item, authority literal "reference", and no graph item expansion.
 * No Python, filesystem, clock, network, or mutable global state.
 */
#ifndef ELPIS_SEMANTIC_ADMISSION_ITEM_B1F_H
#define ELPIS_SEMANTIC_ADMISSION_ITEM_B1F_H
#include "elpis_semantic/admission_bundle_b1e.h"
#include "elpis_semantic/retrieval_item_attachment.h"
#ifdef __cplusplus
extern "C" {
#endif
int semantic_b1f_primary_item_audit(
    const uint8_t *catalog, size_t catalog_bytes,
    const hacf_digest *trusted_catalog_sha256,
    const semantic_snapshot_manifest *base,
    const elpis_evidence_admission_policy_v1 *policy,
    const uint8_t *raw_bundle_json, size_t raw_bundle_bytes,
    const elpis_retrieval_item_attachment_v1 *attachment,
    const uint8_t *exact_item_text, size_t exact_item_text_bytes,
    hacf_digest *descriptive_report_out);
#ifdef __cplusplus
}
#endif
#endif
