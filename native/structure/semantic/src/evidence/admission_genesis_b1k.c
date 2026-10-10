/* B1k is a read-only composition of two prior independent audits.
 * Neither prior descriptive digest is used as an authorization capability.
 * No filesystem writes, storage admission, or policy mutation occur here.
 */
#include "elpis_semantic/admission_genesis_b1k.h"
#include "elpis/sha256.h"
#include "elpis_semantic/identity.h"
#include <stdint.h>
#include <string.h>
static int aliases(const void *a,size_t na,const void *b,size_t nb) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if (!a || !b || !na || !nb || x>UINTPTR_MAX-na || y>UINTPTR_MAX-nb)
        return 1;
    return x<y+nb && y<x+na;
}
#define BAD_ALIAS(p,n) aliases(descriptive_report_out,sizeof(*descriptive_report_out),(p),(n))
#define BAD_OBJECT(p) BAD_ALIAS((p),sizeof(*(p)))
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
    hacf_digest *descriptive_report_out) {
    if (!catalog || !catalog_bytes || !trusted_catalog_sha256 || !base ||
        !trusted_base_manifest_digest || !base_segment_paths || !base_segment_count ||
        base_segment_count>SEMANTIC_MAX_SEGMENTS || !overlay || !typing_bundle ||
        !policy || !layer || !decision || !receipt || !claim_source ||
        !claim_source->candidate || !claim_source->span || !claim_source->attachment ||
        !claim_source->item_text || !claim_source->item_text_bytes || !raw_bundle_json ||
        !raw_bundle_bytes || !canonical_payload || !canonical_payload_bytes ||
        canonical_payload_bytes>SEMANTIC_B1I_MAX_CANONICAL_BYTES ||
        !descriptive_report_out || BAD_ALIAS(catalog,catalog_bytes) ||
        BAD_ALIAS(raw_bundle_json,raw_bundle_bytes) ||
        BAD_ALIAS(canonical_payload,canonical_payload_bytes) ||
        BAD_ALIAS(claim_source->item_text,claim_source->item_text_bytes) ||
        BAD_ALIAS(base_segment_paths,(size_t)base_segment_count*sizeof(*base_segment_paths)) ||
        BAD_OBJECT(trusted_catalog_sha256) || BAD_OBJECT(trusted_base_manifest_digest) ||
        BAD_OBJECT(base) || BAD_OBJECT(overlay) || BAD_OBJECT(typing_bundle) ||
        BAD_OBJECT(policy) || BAD_OBJECT(layer) || BAD_OBJECT(decision) ||
        BAD_OBJECT(receipt) || BAD_OBJECT(claim_source) ||
        BAD_OBJECT(claim_source->candidate) || BAD_OBJECT(claim_source->span) ||
        BAD_OBJECT(claim_source->attachment))
        return SEMANTIC_E_INVAL;
    for (uint32_t i=0;i<base_segment_count;++i) {
        if (!base_segment_paths[i]) return SEMANTIC_E_INVAL;
        size_t n=0;
        while (n<=4000u && base_segment_paths[i][n]) ++n;
        if (!n || n>4000u || BAD_ALIAS(base_segment_paths[i],n+1u))
            return SEMANTIC_E_INVAL;
    }
    hacf_digest base_report,claim_report;
    if (semantic_b1j_empty_base_chain_audit(base,trusted_base_manifest_digest,
            base_segment_paths,base_segment_count,&base_report)!=SEMANTIC_OK)
        return SEMANTIC_E_AUTHORITY;
    /* Run the entire B1i chain fresh, with the IDENTICAL base pointer and
     * independently pinned catalog, rather than accepting supplied reports. */
    if (semantic_b1i_identity_projection_audit(catalog,catalog_bytes,
            trusted_catalog_sha256,base,overlay,typing_bundle,policy,layer,
            decision,receipt,claim_source,raw_bundle_json,raw_bundle_bytes,
            canonical_payload,canonical_payload_bytes,&claim_report)!=SEMANTIC_OK)
        return SEMANTIC_E_AUTHORITY;
    static const char tag[]="elpis.semantic.b1k.genesis-claim-composition.v1";
    elpis_sha256_ctx h;hacf_digest result;
    elpis_sha256_init(&h);
    elpis_sha256_update(&h,tag,sizeof(tag)-1);
    elpis_sha256_update(&h,trusted_base_manifest_digest->bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&h,base_report.bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&h,claim_report.bytes,HACF_DIGEST_BYTES);
    elpis_sha256_final(&h,result.bytes);
    *descriptive_report_out=result;  /* All-or-nothing descriptive output. */
    return SEMANTIC_OK;
}
#undef BAD_ALIAS
#undef BAD_OBJECT
