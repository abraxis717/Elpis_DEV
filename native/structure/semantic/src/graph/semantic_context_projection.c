#include "elpis_semantic/semantic_context_projection.h"
#include "elpis_semantic/identity.h"
#include "elpis/sha256.h"
#include <string.h>
#include <stdlib.h>

static int equal(const hacf_digest *a,const hacf_digest *b) {
    return memcmp(a->bytes,b->bytes,HACF_DIGEST_BYTES)==0;
}
static int zero(const hacf_digest *d) {
    static const uint8_t z[HACF_DIGEST_BYTES]={0};
    return memcmp(d->bytes,z,sizeof(z))==0;
}
static int member(const hacf_digest *d,const hacf_digest *v,uint32_t n){
    for(uint32_t i=0;i<n;i++) if(equal(d,&v[i])) return 1;
    return 0;
}
static void hash_raw(const void *bytes,size_t n,hacf_digest *out){
    elpis_sha256_ctx ctx;
    elpis_sha256_init(&ctx); elpis_sha256_update(&ctx,bytes,n);
    elpis_sha256_final(&ctx,out->bytes);
}
static int valid_layer(const elpis_evidence_admission_v1 *l,const hacf_digest *snapshot){
    hacf_digest v;
    return l&&snapshot&&!zero(snapshot)&&
        elpis_evidence_admission_validate(l)==SEMANTIC_OK&&
        equal(&l->base_snapshot_digest,snapshot)&&
        elpis_evidence_admission_identity(l,&v)==SEMANTIC_OK&&
        equal(&l->admission_layer_digest,&v);
}
static int accepted(const elpis_evidence_admission_v1 *l,
                    const elpis_evidence_admission_decision_v1 *d,
                    const elpis_evidence_admission_receipt_v1 *r,
                    evidence_candidate_kind kind){
    hacf_digest v;
    if(!l||!d||!r||
       d->source_span_count>EVIDENCE_DECISION_MAX_SPANS||
       d->source_attachment_count>EVIDENCE_DECISION_MAX_ITEMS||
       d->candidate_kind!=kind||
       d->semantic_object_kind!=(kind==CANDIDATE_KIND_RELATION?
                             SEMANTIC_OBJECT_KIND_RELATION:SEMANTIC_OBJECT_KIND_CLAIM)||
       !elpis_disposition_is_admitted(d->decision_disposition)||
       d->validation_stage_reached!=VALIDATION_STAGE_COMPLETE||
       d->effective_authority>3u||zero(&d->semantic_object_digest)||
       elpis_admission_decision_validate(d)!=SEMANTIC_OK||
       elpis_admission_decision_identity(d,&v)!=SEMANTIC_OK||
       !equal(&d->decision_identity,&v)||
       !member(&v,l->admission_decision_digests,l->admission_decision_count)||
       elpis_admission_receipt_validate(r)!=SEMANTIC_OK||
       !equal(&r->admission_decision_digest,&v)||
       !equal(&r->base_snapshot_digest,&l->base_snapshot_digest)||
       !equal(&r->query_overlay_digest,&l->query_overlay_digest)||
       !equal(&r->retrieval_expansion_digest,&l->retrieval_expansion_digest)||
       !equal(&r->typing_bundle_digest,&l->typing_bundle_digest)||
       !equal(&r->admission_policy_digest,&l->admission_policy_digest)||
       !equal(&r->HACF_package_digest,&l->HACF_package_digest)||
       !equal(&r->candidate_digest,&d->candidate_digest)||
       !equal(&r->semantic_object_digest,&d->semantic_object_digest)||
       r->source_span_count!=d->source_span_count||
       r->retrieval_item_attachment_count!=d->source_attachment_count||
       elpis_admission_receipt_identity(r,&v)!=SEMANTIC_OK||
       !equal(&r->receipt_digest,&v)||
       !member(&v,l->admission_receipt_digests,l->admission_receipt_count)) return 0;
    for(uint32_t i=0;i<d->source_span_count;i++)
        if(!member(&d->source_span_digests[i],r->source_span_digests,r->source_span_count)) return 0;
    for(uint32_t i=0;i<d->source_attachment_count;i++)
        if(!member(&d->source_attachment_digests[i],r->retrieval_item_attachment_digests,
                   r->retrieval_item_attachment_count)) return 0;
    return 1;
}
/* Resolve the actual chunk in this volatile corpus, and verify that provided
 * primary-item bytes are precisely those the corpus owns, not just a matching
 * claimed digest. This v1 text-only projection refuses embedded NUL bytes. */
static int endpoint(const elpis_semantic_cgraph_projection_v1 *p,
                    const elpis_semantic_cgraph_endpoint_v1 *e){
    hacf_digest v;
    elpis_chunk_ref ref;
    char hex[65];char *actual=NULL;
    if(!e->decision||!e->receipt||!e->span||!e->attachment||!e->item_text||
       !e->item_text_bytes||e->item_text_bytes>65535u||
       !accepted(p->admission_layer,e->decision,e->receipt,CANDIDATE_KIND_CLAIM)||
       elpis_attachment_validate(e->attachment)!=SEMANTIC_OK||
       e->attachment->item_kind!=1u||e->attachment->graph_hop!=0u||
       e->attachment->graph_edge_provenance_status!=GRAPH_PROVENANCE_NOT_APPLICABLE||
       (e->attachment->source_mask&3u)==0u||(e->attachment->source_mask&~3u)!=0u||
       e->span->span_flags!=EVIDENCE_SPAN_FLAG_PRIMARY||
       e->receipt->graph_edge_provenance_status!=GRAPH_PROVENANCE_NOT_APPLICABLE||
       elpis_attachment_digest(e->attachment,&v)!=SEMANTIC_OK||
       !equal(&v,&e->attachment->attachment_digest)||
       !member(&v,e->decision->source_attachment_digests,e->decision->source_attachment_count)||
       !member(&v,e->receipt->retrieval_item_attachment_digests,
                 e->receipt->retrieval_item_attachment_count)||
       elpis_evidence_span_validate(e->span,e->item_text,e->item_text_bytes)!=SEMANTIC_OK||
       elpis_evidence_span_identity(e->span,&v)!=SEMANTIC_OK||
       !equal(&v,&e->span->span_identity)||
       !member(&v,e->decision->source_span_digests,e->decision->source_span_count)||
       !member(&v,e->receipt->source_span_digests,e->receipt->source_span_count)||
       !equal(&e->span->retrieval_item_attachment_digest,&e->attachment->attachment_digest)||
       !equal(&e->span->evidence_node_digest,&e->attachment->evidence_node_digest)||
       !equal(&e->span->retrieval_bundle_digest,&e->attachment->retrieval_bundle_digest)||
       !equal(&e->span->retrieval_bundle_package_digest,
              &e->attachment->retrieval_bundle_package_digest)||
       !equal(&e->span->retrieval_expansion_digest,
              &p->admission_layer->retrieval_expansion_digest)||
       !equal(&e->span->chunk_digest,&e->attachment->chunk_digest)||
       !equal(&e->span->item_text_digest,&e->attachment->text_digest)||
       !member(&e->attachment->retrieval_bundle_package_digest,
               e->receipt->retrieval_bundle_package_digests,
               e->receipt->retrieval_bundle_count)||
       zero(&e->attachment->chunk_digest)||
       memchr(e->item_text,'\0',e->item_text_bytes)) return 0;
    hash_raw(e->item_text,e->item_text_bytes,&v);
    if(!equal(&v,&e->attachment->text_digest))return 0;
    elpis_hex32(e->attachment->chunk_digest.bytes,hex);
    if(elpis_corpus_chunk_lookup(p->corpus,hex,&ref)!=0)return 0;
    if(elpis_corpus_chunk_text(p->corpus,hex,&actual)!=0||!actual)return 0;
    size_t n=0;
    while(n<=(size_t)e->item_text_bytes && actual[n])n++;
    int good=n==e->item_text_bytes&&
        memcmp(actual,e->item_text,e->item_text_bytes)==0;
    elpis_free(actual);
    return good;
}
static uint32_t map_type(evidence_relation_type type){
    switch(type){
        case RELATION_TYPE_MENTIONS:return ELPIS_CGRAPH_SEMANTIC_MENTIONS;
        case RELATION_TYPE_DEFINES:return ELPIS_CGRAPH_SEMANTIC_DEFINES;
        case RELATION_TYPE_SUPPORTS:return ELPIS_CGRAPH_SEMANTIC_SUPPORTS;
        case RELATION_TYPE_CONTRADICTS:return ELPIS_CGRAPH_SEMANTIC_CONTRADICTS;
        case RELATION_TYPE_QUALIFIES:return ELPIS_CGRAPH_SEMANTIC_QUALIFIES;
        case RELATION_TYPE_LIMITS_SCOPE_OF:return ELPIS_CGRAPH_SEMANTIC_LIMITS_SCOPE;
        case RELATION_TYPE_PROVIDES_CONTEXT_FOR:return ELPIS_CGRAPH_SEMANTIC_PROVIDES_CONTEXT;
        default:return 0;
    }
}
int elpis_semantic_cgraph_project_v1(const elpis_semantic_cgraph_projection_v1 *p,
                                     elpis_context_edge_input *out,
                                     elpis_semantic_cgraph_audit_v1 *audit_out){
    hacf_digest v;
    if(!p||!out||!audit_out||!p->corpus||!p->relation||!p->source.decision||!p->target.decision||
       !valid_layer(p->admission_layer,p->trusted_base_snapshot_digest)||
       !accepted(p->admission_layer,p->relation_decision,p->relation_receipt,CANDIDATE_KIND_RELATION)||
       p->relation->source_span_count>EVIDENCE_MAX_RELATION_SOURCE_SPANS||
       p->relation->additional_participant_count!=0||
       p->relation->evidence_object_kind!=OBJECT_KIND_CLAIM_NODE||
       p->relation->target_object_kind!=OBJECT_KIND_CLAIM_NODE||
       p->relation->evidence_role!=RELATION_ROLE_EVIDENCE||
       p->relation->target_role!=RELATION_ROLE_TARGET||
       elpis_relation_candidate_validate(p->relation)!=SEMANTIC_OK||
       elpis_relation_candidate_identity(p->relation,&v)!=SEMANTIC_OK||
       !equal(&p->relation->candidate_identity,&v)||
       !equal(&v,&p->relation_decision->candidate_digest)||
       !equal(&p->relation->evidence_claim_candidate_digest,
              &p->source.decision->candidate_digest)||
       !equal(&p->relation->evidence_object_digest,&p->source.decision->semantic_object_digest)||
       !equal(&p->relation->target_object_digest,&p->target.decision->semantic_object_digest)||
       !equal(&p->relation->typer_profile_digest,&p->relation_receipt->typer_profile_digest)||
       p->relation_receipt->graph_edge_provenance_status!=GRAPH_PROVENANCE_NOT_APPLICABLE||
       !endpoint(p,&p->source)||!endpoint(p,&p->target)||
       !member(&p->source.span->span_identity,p->relation->source_span_digests,
               p->relation->source_span_count)||
       !member(&p->source.span->span_identity,p->relation_decision->source_span_digests,
               p->relation_decision->source_span_count)||
       !member(&p->source.span->span_identity,p->relation_receipt->source_span_digests,
               p->relation_receipt->source_span_count)||
       !member(&p->source.attachment->attachment_digest,
               p->relation_decision->source_attachment_digests,
               p->relation_decision->source_attachment_count)||
       equal(&p->source.attachment->chunk_digest,&p->target.attachment->chunk_digest)||
       !map_type(p->relation->relation_type))return SEMANTIC_E_INVAL;
    uint32_t authority=p->relation_decision->effective_authority;
    const uint32_t a[]={p->source.decision->effective_authority,
                         p->target.decision->effective_authority,
                         p->source.attachment->item_authority,
                         p->target.attachment->item_authority};
    for(size_t i=0;i<sizeof(a)/sizeof(a[0]);i++) if(a[i]<authority)authority=a[i];
    if(authority==0||authority>3)return SEMANTIC_E_AUTHORITY;
    elpis_semantic_cgraph_audit_v1 audit={0};
    audit.abi_version=ELPIS_SEMANTIC_CGRAPH_PROJECTION_ABI_VERSION;
    audit.admission_layer_digest=p->admission_layer->admission_layer_digest;
    audit.snapshot_digest=*p->trusted_base_snapshot_digest;
    audit.relation_candidate_digest=p->relation->candidate_identity;
    audit.relation_decision_digest=p->relation_decision->decision_identity;
    audit.relation_receipt_digest=p->relation_receipt->receipt_digest;
    audit.source_decision_digest=p->source.decision->decision_identity;
    audit.source_receipt_digest=p->source.receipt->receipt_digest;
    audit.source_span_digest=p->source.span->span_identity;
    audit.source_attachment_digest=p->source.attachment->attachment_digest;
    audit.target_decision_digest=p->target.decision->decision_identity;
    audit.target_receipt_digest=p->target.receipt->receipt_digest;
    audit.target_span_digest=p->target.span->span_identity;
    audit.target_attachment_digest=p->target.attachment->attachment_digest;
    /* Domain-separated provenance of the *audited witness chain*, not a
     * synthetic claim that retrieval metadata itself proves a relation. */
    static const char domain[]="elpis.semantic.chunk_context_provenance.v1";
    elpis_sha256_ctx ctx;
    elpis_sha256_init(&ctx);
    elpis_sha256_update(&ctx,(const uint8_t *)domain,sizeof(domain)-1u);
    const hacf_digest *parts[]={&audit.admission_layer_digest,&audit.snapshot_digest,
        &audit.relation_candidate_digest,&audit.relation_decision_digest,&audit.relation_receipt_digest,
        &audit.source_decision_digest,&audit.source_receipt_digest,&audit.source_span_digest,
        &audit.source_attachment_digest,&audit.target_decision_digest,&audit.target_receipt_digest,
        &audit.target_span_digest,&audit.target_attachment_digest,
        &p->source.attachment->chunk_digest,&p->target.attachment->chunk_digest};
    for(size_t i=0;i<sizeof(parts)/sizeof(parts[0]);i++)
        elpis_sha256_update(&ctx,parts[i]->bytes,HACF_DIGEST_BYTES);
    elpis_sha256_final(&ctx,audit.provenance_digest.bytes);
    elpis_context_edge_input edge={0};
    elpis_hex32(p->source.attachment->chunk_digest.bytes,edge.subject_chunk_digest);
    elpis_hex32(p->target.attachment->chunk_digest.bytes,edge.object_chunk_digest);
    elpis_hex32(audit.provenance_digest.bytes,edge.provenance_digest);
    edge.edge_type=map_type(p->relation->relation_type);
    edge.authority=authority;
    *out=edge;*audit_out=audit;
    return SEMANTIC_OK;
}
