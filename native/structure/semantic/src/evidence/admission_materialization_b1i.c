/* B1i exact byte-identity projection, read-only native C11.
 * Validates the narrowly defined identity projection into one overlay node.
 * It does NOT prove semantic interpretation of payload or base-graph novelty.
 */
#include "elpis_semantic/admission_materialization_b1i.h"
#include "elpis/sha256.h"
#include <stdint.h>
#include <string.h>
static int overlap(const void *a,size_t na,const void *b,size_t nb){
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(!a||!b||!na||!nb||x>UINTPTR_MAX-na||y>UINTPTR_MAX-nb)return 1;
    return x<y+nb && y<x+na;
}
static int eq(const hacf_digest *a,const hacf_digest *b){
    return memcmp(a->bytes,b->bytes,HACF_DIGEST_BYTES)==0;
}
#define ALIAS(p,n) overlap(descriptive_report_out,sizeof(*descriptive_report_out),(p),(n))
#define ALIAS_OBJ(p) ALIAS((p),sizeof(*(p)))
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
    hacf_digest *descriptive_report_out) {
    if (!catalog || !catalog_bytes || !trusted_catalog_sha256 || !base ||
        !overlay || !typing_bundle || !policy || !layer || !decision ||
        !receipt || !claim_source || !claim_source->candidate ||
        !claim_source->span || !claim_source->attachment ||
        !claim_source->item_text || !claim_source->item_text_bytes ||
        !raw_bundle_json || !raw_bundle_bytes || !canonical_payload ||
        !canonical_payload_bytes || canonical_payload_bytes>SEMANTIC_B1I_MAX_CANONICAL_BYTES ||
        !descriptive_report_out || ALIAS(catalog,catalog_bytes) ||
        ALIAS(raw_bundle_json,raw_bundle_bytes) ||
        ALIAS(canonical_payload,canonical_payload_bytes) ||
        ALIAS(claim_source->item_text,claim_source->item_text_bytes) ||
        ALIAS_OBJ(trusted_catalog_sha256) || ALIAS_OBJ(base) ||
        ALIAS_OBJ(overlay) || ALIAS_OBJ(typing_bundle) || ALIAS_OBJ(policy) ||
        ALIAS_OBJ(layer) || ALIAS_OBJ(decision) || ALIAS_OBJ(receipt) ||
        ALIAS_OBJ(claim_source) || ALIAS_OBJ(claim_source->candidate) ||
        ALIAS_OBJ(claim_source->span) || ALIAS_OBJ(claim_source->attachment))
        return SEMANTIC_E_INVAL;

    /* B1h is executed afresh over this same complete witness. */
    hacf_digest prior;
    if (semantic_b1h_claim_decision_replay_audit(
        catalog,catalog_bytes,trusted_catalog_sha256,base,overlay,
        typing_bundle,policy,layer,decision,receipt,claim_source,
        raw_bundle_json,raw_bundle_bytes,&prior)!=SEMANTIC_OK)
        return SEMANTIC_E_AUTHORITY;

    const elpis_evidence_claim_candidate_v1 *c=claim_source->candidate;
    if(c->claim_type==0 || c->claim_type>=SEMANTIC_NODE_NAMESPACE ||
       !overlay->local_builder ||
       semantic_builder_node_count(overlay->local_builder)!=1 ||
       semantic_builder_assertion_count(overlay->local_builder)!=1)
        return SEMANTIC_E_AUTHORITY;
    const elpis_semantic_node_v1 *node=semantic_builder_get_node(overlay->local_builder,0);
    const elpis_semantic_assertion_v1 *assertion=semantic_builder_get_assertion(overlay->local_builder,0);
    if(!node || !assertion ||
       node->node_type!=(SEMANTIC_NODE_NAMESPACE|c->claim_type) ||
       node->semantic_flags!=SEMANTIC_NODE_FLAG_NONE ||
       assertion->asserted_object_kind!=SEMANTIC_OBJECT_KIND_NODE ||
       !eq(&assertion->asserted_object_digest,&node->node_identity) ||
       !eq(&decision->semantic_object_digest,&node->node_identity))
        return SEMANTIC_E_AUTHORITY;

    hacf_digest computed,identity;
    elpis_sha256(canonical_payload,canonical_payload_bytes,computed.bytes);
    if(!eq(&computed,&c->claim_payload_digest) ||
       !eq(&computed,&c->claim_payload_object_digest) ||
       !eq(&computed,&node->payload_digest) ||
       elpis_semantic_node_validate(node)!=SEMANTIC_OK ||
       elpis_semantic_node_identity(node,&identity)!=SEMANTIC_OK ||
       !eq(&identity,&node->node_identity))
        return SEMANTIC_E_AUTHORITY;

    static const char domain[]="elpis.semantic.b1i.identity-projection-bytes.v1";
    elpis_sha256_ctx h;
    hacf_digest result;
    elpis_sha256_init(&h);
    elpis_sha256_update(&h,domain,sizeof(domain)-1);
    elpis_sha256_update(&h,prior.bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&h,computed.bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&h,identity.bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&h,c->candidate_identity.bytes,HACF_DIGEST_BYTES);
    elpis_sha256_final(&h,result.bytes);
    *descriptive_report_out=result;
    return SEMANTIC_OK;
}
#undef ALIAS
#undef ALIAS_OBJ
