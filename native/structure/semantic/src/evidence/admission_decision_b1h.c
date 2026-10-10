/* Native C11 B1h restricted policy/decision replay; no write privileges.
 * Does NOT establish target resolution, base-graph newness, duplicate/conflict
 * outcomes, or externally authorized admission. Those remain closed gates.
 */
#include "elpis_semantic/admission_decision_b1h.h"
#include "elpis/sha256.h"
#include <stdint.h>
#include <string.h>

static int zero(const hacf_digest *d) {
    static const uint8_t zeros[HACF_DIGEST_BYTES]={0};
    return memcmp(d->bytes,zeros,HACF_DIGEST_BYTES)==0;
}
static int overlaps(const void *a,size_t na,const void *b,size_t nb){
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(!a||!b||!na||!nb||x>UINTPTR_MAX-na||y>UINTPTR_MAX-nb)return 1;
    return x<y+nb && y<x+na;
}
#define REFUSE_ALIAS(p,n) overlaps(descriptive_report_out,sizeof(*descriptive_report_out),(p),(n))
#define REFUSE_OBJ(p) REFUSE_ALIAS((p),sizeof(*(p)))
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
    hacf_digest *descriptive_report_out) {
    if (!catalog || !trusted_catalog_sha256 || !base || !overlay ||
        !typing_bundle || !policy || !layer || !decision || !receipt ||
        !claim_source || !raw_bundle_json || !descriptive_report_out ||
        !claim_source->candidate || !claim_source->span ||
        !claim_source->attachment || !claim_source->item_text ||
        !catalog_bytes || !raw_bundle_bytes ||
        REFUSE_ALIAS(catalog,catalog_bytes) ||
        REFUSE_ALIAS(raw_bundle_json,raw_bundle_bytes) ||
        REFUSE_OBJ(trusted_catalog_sha256) || REFUSE_OBJ(base) ||
        REFUSE_OBJ(overlay) || REFUSE_OBJ(typing_bundle) ||
        REFUSE_OBJ(policy) || REFUSE_OBJ(layer) ||
        REFUSE_OBJ(decision) || REFUSE_OBJ(receipt) ||
        REFUSE_OBJ(claim_source) || REFUSE_OBJ(claim_source->candidate) ||
        REFUSE_OBJ(claim_source->span) || REFUSE_OBJ(claim_source->attachment) ||
        REFUSE_ALIAS(claim_source->item_text,claim_source->item_text_bytes))
        return SEMANTIC_E_INVAL;

    /* Re-run the native end-to-end correspondence chain; never trust reports. */
    hacf_digest prior;
    if (semantic_b1g_single_claim_composition_audit(catalog,catalog_bytes,
        trusted_catalog_sha256,base,overlay,typing_bundle,policy,layer,
        decision,receipt,claim_source,raw_bundle_json,raw_bundle_bytes,
        &prior) != SEMANTIC_OK) return SEMANTIC_E_AUTHORITY;

    /* Restricted deterministic replay: no unresolved candidate references;
     * one source item with no independent authority attenuation witness.
     * Missing evidence of any other ceiling always fails closed here. */
    const elpis_evidence_claim_candidate_v1 *c=claim_source->candidate;
    const elpis_retrieval_item_attachment_v1 *a=claim_source->attachment;
    const uint32_t replay_authority = a->item_authority < policy->maximum_claim_authority
        ? a->item_authority : policy->maximum_claim_authority;
    if (decision->candidate_kind!=CANDIDATE_KIND_CLAIM ||
        decision->decision_disposition!=DISPOSITION_ADMITTED_NEW_OBJECT ||
        decision->validation_stage_reached!=VALIDATION_STAGE_COMPLETE ||
        decision->decision_reason!=REASON_NONE ||
        !zero(&decision->decision_diagnostic_digest) ||
        replay_authority==0 ||
        decision->effective_authority!=replay_authority ||
        a->item_authority<policy->minimum_source_authority ||
        c->confidence_key<policy->minimum_claim_confidence_key ||
        c->subject_object_kind!=SUBJECT_KIND_NONE ||
        !zero(&c->subject_object_digest) ||
        !zero(&c->claim_scope_digest) ||
        !zero(&c->claim_qualifier_digest) ||
        c->candidate_flags!=0 ||
        !elpis_policy_allows_claim_type(policy,c->claim_type) ||
        !elpis_policy_allows_typer(policy,&c->typer_profile_digest))
        return SEMANTIC_E_AUTHORITY;

    static const char domain[]="elpis.semantic.b1h.decision-replay-restricted.v1";
    elpis_sha256_ctx h;
    elpis_sha256_init(&h);
    elpis_sha256_update(&h,domain,sizeof(domain)-1);
    elpis_sha256_update(&h,prior.bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&h,decision->decision_identity.bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&h,c->candidate_identity.bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&h,trusted_catalog_sha256->bytes,HACF_DIGEST_BYTES);
    hacf_digest out;
    elpis_sha256_final(&h,out.bytes);
    *descriptive_report_out=out;
    return SEMANTIC_OK;
}
#undef REFUSE_ALIAS
#undef REFUSE_OBJ
