/* B1g native single-claim correspondence composition. NO WRITE PRIVILEGES. */
#include "elpis_semantic/admission_composition_b1g.h"
#include "elpis/sha256.h"
#include <stdint.h>
#include <string.h>

/* Do not let an output overwrite any part of untrusted input/authority records. */
static int overlap(const void *out, const void *p, size_t len) {
    const uintptr_t a = (uintptr_t)out, b = (uintptr_t)p;
    if (!out || !p || len == 0 || a > UINTPTR_MAX - sizeof(hacf_digest) ||
        b > UINTPTR_MAX - len) return 1;
    return a < b + len && b < a + sizeof(hacf_digest);
}
#define CONFLICT(x) overlap(descriptive_report_out,(x),sizeof(*(x)))
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
    hacf_digest *descriptive_report_out) {
    if (!catalog || catalog_bytes != SEMANTIC_B1D_CATALOG_BYTES ||
        !trusted_catalog_sha256 || !base || !overlay || !typing_bundle ||
        !policy || !layer || !decision || !receipt || !claim_source ||
        !raw_bundle_json || !raw_bundle_bytes || !descriptive_report_out ||
        !claim_source->candidate || !claim_source->span ||
        !claim_source->attachment || !claim_source->item_text ||
        !claim_source->item_text_bytes ||
        claim_source->item_text_bytes > SEMANTIC_B1B_MAX_RAW_ITEM_BYTES ||
        raw_bundle_bytes > SEMANTIC_B1D_MAX_ARTIFACT_BYTES ||
        overlap(descriptive_report_out,catalog,catalog_bytes) ||
        overlap(descriptive_report_out,raw_bundle_json,raw_bundle_bytes) ||
        overlap(descriptive_report_out,claim_source->item_text,claim_source->item_text_bytes) ||
        CONFLICT(trusted_catalog_sha256) || CONFLICT(base) || CONFLICT(overlay) ||
        CONFLICT(typing_bundle) || CONFLICT(policy) || CONFLICT(layer) ||
        CONFLICT(decision) || CONFLICT(receipt) || CONFLICT(claim_source) ||
        CONFLICT(claim_source->candidate) || CONFLICT(claim_source->span) ||
        CONFLICT(claim_source->attachment)) return SEMANTIC_E_INVAL;

    /* The full, identical witness is used by both audits: no caller-supplied
     * intermediary report is trusted. All outputs are local until success. */
    hacf_digest policy_pin, package_pin, catalog_report, item_report, claim_report;
    if (semantic_b1d_authority_bytes_audit(catalog,catalog_bytes,
            trusted_catalog_sha256,base,policy,raw_bundle_json,raw_bundle_bytes,
            &policy_pin,&package_pin,&catalog_report) != SEMANTIC_OK)
        return SEMANTIC_E_AUTHORITY;
    if (semantic_b1f_primary_item_audit(catalog,catalog_bytes,
            trusted_catalog_sha256,base,policy,raw_bundle_json,raw_bundle_bytes,
            claim_source->attachment,claim_source->item_text,
            claim_source->item_text_bytes,&item_report) != SEMANTIC_OK)
        return SEMANTIC_E_AUTHORITY;
    if (semantic_b1b_claim_source_audit(base,overlay,typing_bundle,policy,
            &policy_pin,&package_pin,layer,decision,receipt,claim_source,1,
            &claim_report) != SEMANTIC_OK)
        return SEMANTIC_E_AUTHORITY;

    /* A descriptive hash cannot convey external policy deployment authority. */
    static const char domain[] = "elpis.semantic.b1g.claim-item-composition.v1";
    elpis_sha256_ctx h;
    elpis_sha256_init(&h);
    elpis_sha256_update(&h,domain,sizeof(domain)-1);
    elpis_sha256_update(&h,trusted_catalog_sha256->bytes,32);
    elpis_sha256_update(&h,catalog_report.bytes,32);
    elpis_sha256_update(&h,item_report.bytes,32);
    elpis_sha256_update(&h,claim_report.bytes,32);
    elpis_sha256_update(&h,base->manifest_digest.bytes,32);
    elpis_sha256_update(&h,overlay->overlay_identity.bytes,32);
    elpis_sha256_update(&h,layer->admission_layer_digest.bytes,32);
    hacf_digest report;
    elpis_sha256_final(&h,report.bytes);
    *descriptive_report_out = report;
    return SEMANTIC_OK;
}
#undef CONFLICT
