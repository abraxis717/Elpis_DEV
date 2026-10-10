/* B1o: execute both prior verifiers afresh against one pinned base.
 * A claimed ADMITTED_NEW_OBJECT disposition is never a novelty oracle.
 * Only a complete, populated GENESIS chain with 2..64 segments is supported.
 */
#define _POSIX_C_SOURCE 200809L
#include "elpis_semantic/admission_populated_b1o.h"
#include "elpis_semantic/hypergraph.h"
#include "elpis_semantic/identity.h"
#include "elpis/sha256.h"
#include <stdint.h>
#include <string.h>
static int overlaps(const void *a,size_t na,const void *b,size_t nb) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(!a||!b||!na||!nb||x>UINTPTR_MAX-na||y>UINTPTR_MAX-nb)return 1;
    return x<y+nb&&y<x+na;
}
#define ALIAS(p,n) overlaps(descriptive_report_out,sizeof(*descriptive_report_out),(p),(n))
#define ALIAS_OBJ(p) ALIAS((p),sizeof(*(p)))
int semantic_b1o_populated_claim_novelty_audit(
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
    if(!catalog||!catalog_bytes||!trusted_catalog_sha256||!base||
       !trusted_base_manifest_digest||!base_segment_paths||
       base_segment_count<2||base_segment_count>SEMANTIC_B1M_MAX_SEGMENTS||
       !overlay||!overlay->local_builder||!typing_bundle||!policy||!layer||
       !decision||!receipt||!claim_source||!claim_source->candidate||
       !claim_source->span||!claim_source->attachment||!claim_source->item_text||
       !claim_source->item_text_bytes||!raw_bundle_json||!raw_bundle_bytes||
       !canonical_payload||!canonical_payload_bytes||
       canonical_payload_bytes>SEMANTIC_B1I_MAX_CANONICAL_BYTES||
       !descriptive_report_out||ALIAS(catalog,catalog_bytes)||
       ALIAS(raw_bundle_json,raw_bundle_bytes)||
       ALIAS(canonical_payload,canonical_payload_bytes)||
       ALIAS(claim_source->item_text,claim_source->item_text_bytes)||
       ALIAS(base_segment_paths,(size_t)base_segment_count*sizeof(*base_segment_paths))||
       ALIAS_OBJ(trusted_catalog_sha256)||ALIAS_OBJ(trusted_base_manifest_digest)||
       ALIAS_OBJ(base)||ALIAS_OBJ(overlay)||ALIAS_OBJ(typing_bundle)||
       ALIAS_OBJ(policy)||ALIAS_OBJ(layer)||ALIAS_OBJ(decision)||
       ALIAS_OBJ(receipt)||ALIAS_OBJ(claim_source)||
       ALIAS_OBJ(claim_source->candidate)||ALIAS_OBJ(claim_source->span)||
       ALIAS_OBJ(claim_source->attachment))return SEMANTIC_E_INVAL;
    for(uint32_t i=0;i<base_segment_count;++i){
        const char *p=base_segment_paths[i];
        if(!p)return SEMANTIC_E_INVAL;
        size_t n=strnlen(p,4001u);
        if(!n||n>4000u||ALIAS(p,n+1))return SEMANTIC_E_INVAL;
    }
    /* The target is obtained from the actual overlay, not caller-provided
       trust metadata. B1i later recomputes this node's identity and payload. */
    if(semantic_builder_node_count(overlay->local_builder)!=1u||
       semantic_builder_assertion_count(overlay->local_builder)!=1u)
        return SEMANTIC_E_AUTHORITY;
    const elpis_semantic_node_v1 *node=semantic_builder_get_node(overlay->local_builder,0);
    if(!node)return SEMANTIC_E_AUTHORITY;
    hacf_digest base_report,claim_report;
    uint32_t present=2;
    if(semantic_b1n_global_unique_node_audit(base,trusted_base_manifest_digest,
            base_segment_paths,base_segment_count,&node->node_identity,
            &present,&base_report)!=SEMANTIC_OK||present!=0)
        return SEMANTIC_E_AUTHORITY;
    /* B1p: node identity includes semantic_flags. The restricted canonical
     * payload policy must also refuse an existing node with the same type and
     * payload digest but different flags. Every scan is a complete validated
     * descriptor read, re-bound to the same pinned manifest/graph chain. */
    hacf_digest previous=base->genesis_identity;
    for(uint32_t i=0;i<base_segment_count;++i) {
        semantic_segment_record segment;
        hacf_digest segment_digest;
        uint32_t collisions=UINT32_MAX;
        if(semantic_segment_read_typed_payload_occurrences(
               base_segment_paths[i],node->node_type,&node->payload_digest,
               &segment,&segment_digest,&collisions)!=SEMANTIC_OK ||
           collisions!=0 ||
           memcmp(segment_digest.bytes,base->segment_digests[i].bytes,HACF_DIGEST_BYTES)!=0 ||
           memcmp(segment.prior_snapshot_digest.bytes,previous.bytes,HACF_DIGEST_BYTES)!=0 ||
           memcmp(segment.type_registry_digest.bytes,base->type_registry_digest.bytes,HACF_DIGEST_BYTES)!=0)
            return SEMANTIC_E_AUTHORITY;
        previous=segment.hacf_next_snapshot;
    }
    if(memcmp(previous.bytes,base->hacf_graph_snapshot_digest.bytes,HACF_DIGEST_BYTES)!=0)
        return SEMANTIC_E_AUTHORITY;
    /* Same base pointer, same pinned manifest and same exact overlay witness. */
    if(semantic_b1i_identity_projection_audit(catalog,catalog_bytes,
            trusted_catalog_sha256,base,overlay,typing_bundle,policy,layer,
            decision,receipt,claim_source,raw_bundle_json,raw_bundle_bytes,
            canonical_payload,canonical_payload_bytes,&claim_report)!=SEMANTIC_OK)
        return SEMANTIC_E_AUTHORITY;
    /* Manifest pin is an external input, not an inferred trust decision.
       Verify directly in case future B1i changes its manifest handling. */
    if(memcmp(trusted_base_manifest_digest->bytes,base->manifest_digest.bytes,
              HACF_DIGEST_BYTES)!=0)return SEMANTIC_E_AUTHORITY;
    static const char tag[]="elpis.semantic.b1o.populated-claim-nonmembership.v1";
    elpis_sha256_ctx h;hacf_digest result;
    elpis_sha256_init(&h);
    elpis_sha256_update(&h,tag,sizeof(tag)-1);
    elpis_sha256_update(&h,trusted_base_manifest_digest->bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&h,node->node_identity.bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&h,base_report.bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&h,claim_report.bytes,HACF_DIGEST_BYTES);
    elpis_sha256_final(&h,result.bytes);
    *descriptive_report_out=result; /* no partial output; never an admission permit */
    return SEMANTIC_OK;
}
#undef ALIAS
#undef ALIAS_OBJ
