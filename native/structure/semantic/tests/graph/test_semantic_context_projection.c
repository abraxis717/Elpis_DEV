#include "elpis_semantic/semantic_context_projection.h"
#include "elpis_semantic/identity.h"
#include "elpis/sha256.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct fixture {
    elpis_corpus *corpus;
    elpis_evidence_admission_v1 layer;
    elpis_evidence_admission_decision_v1 decisions[3];
    elpis_evidence_admission_receipt_v1 receipts[3];
    elpis_evidence_relation_candidate_v1 relation;
    elpis_evidence_span_v1 spans[2];
    elpis_retrieval_item_attachment_v1 attachments[2];
    char texts[2][128];
    elpis_semantic_cgraph_projection_v1 p;
} fixture;
static int eq(const hacf_digest *a,const hacf_digest *b){return memcmp(a,b,32)==0;}
static void hash(const void *p,size_t n,hacf_digest *out){elpis_sha256(p,n,out->bytes);}
static hacf_digest h(const char *s){hacf_digest d;hash(s,strlen(s),&d);return d;}
static void endpoint(fixture *f,unsigned i){
    elpis_chunk_ref c={0};
    uint32_t count=0;
    assert(elpis_corpus_list_chunks(f->corpus,NULL,NULL,i,1,&c,&count)==0 && count==1);
    char *txt=NULL;
    assert(elpis_corpus_chunk_text(f->corpus,c.chunk_digest,&txt)==0 && txt);
    size_t len=strlen(txt);assert(len>0&&len<sizeof(f->texts[i]));
    memcpy(f->texts[i],txt,len+1);elpis_free(txt);
    elpis_retrieval_item_attachment_v1 *a=&f->attachments[i];
    elpis_attachment_init(a);
    a->evidence_node_digest=h(i?"node-b":"node-a");
    a->retrieval_bundle_digest=h("bundle");
    a->retrieval_bundle_package_digest=h("bundle-package");
    a->retrieval_requirement_digest=h("requirement");
    assert(hacf_digest_from_hex(c.chunk_digest,&a->chunk_digest)==0);
    assert(hacf_digest_from_hex(c.doc_digest,&a->document_digest)==0);
    hash(f->texts[i],len,&a->text_digest);
    a->namespace_digest=h("namespace");
    a->item_authority=(i==0)?2u:1u;
    a->item_kind=1;a->source_mask=1;a->final_rank=i+1;
    a->graph_edge_provenance_status=GRAPH_PROVENANCE_NOT_APPLICABLE;
    assert(elpis_attachment_digest(a,&a->attachment_digest)==SEMANTIC_OK);
    elpis_evidence_span_v1 *s=&f->spans[i];
    elpis_evidence_span_init(s);
    s->retrieval_expansion_digest=f->layer.retrieval_expansion_digest;
    s->retrieval_bundle_digest=a->retrieval_bundle_digest;
    s->retrieval_bundle_package_digest=a->retrieval_bundle_package_digest;
    s->retrieval_item_attachment_digest=a->attachment_digest;
    s->evidence_node_digest=a->evidence_node_digest;
    s->chunk_digest=a->chunk_digest;
    s->item_text_digest=a->text_digest;
    s->byte_end_exclusive=(uint32_t)len;
    hash(f->texts[i],len,&s->span_bytes_digest);
    s->span_flags=EVIDENCE_SPAN_FLAG_PRIMARY;
    assert(elpis_evidence_span_identity(s,&s->span_identity)==SEMANTIC_OK);
    assert(elpis_evidence_span_validate(s,(const uint8_t *)f->texts[i],(uint32_t)len)==SEMANTIC_OK);
}
static void decision(fixture *f,unsigned i,int relation){
    elpis_evidence_admission_decision_v1 *d=&f->decisions[i];
    elpis_admission_decision_init(d);
    d->candidate_kind=relation?CANDIDATE_KIND_RELATION:CANDIDATE_KIND_CLAIM;
    d->candidate_digest=relation?f->relation.candidate_identity:h(i?"cand-b":"cand-a");
    d->typing_bundle_digest=f->layer.typing_bundle_digest;
    d->admission_policy_digest=f->layer.admission_policy_digest;
    d->validation_stage_reached=VALIDATION_STAGE_COMPLETE;
    d->decision_disposition=DISPOSITION_ADMITTED_NEW_OBJECT;
    d->semantic_object_kind=relation?SEMANTIC_OBJECT_KIND_RELATION:SEMANTIC_OBJECT_KIND_CLAIM;
    d->semantic_object_digest=h(relation?"relation-object":i?"claim-b":"claim-a");
    d->effective_authority=relation?2u:i?1u:2u;
    unsigned which=relation?0:i;
    d->source_span_count=1;d->source_span_digests[0]=f->spans[which].span_identity;
    d->source_attachment_count=1;
    d->source_attachment_digests[0]=f->attachments[which].attachment_digest;
    d->decision_diagnostic_digest=h("diagnostic");
    assert(elpis_admission_decision_identity(d,&d->decision_identity)==SEMANTIC_OK);
    elpis_evidence_admission_receipt_v1 *r=&f->receipts[i];
    elpis_admission_receipt_init(r);
    r->base_snapshot_digest=f->layer.base_snapshot_digest;
    r->query_overlay_digest=f->layer.query_overlay_digest;
    r->retrieval_expansion_digest=f->layer.retrieval_expansion_digest;
    r->retrieval_expanded_view_digest=h("expanded");
    r->typing_bundle_digest=f->layer.typing_bundle_digest;
    r->typer_profile_digest=h("typer");
    r->candidate_digest=d->candidate_digest;
    r->admission_policy_digest=f->layer.admission_policy_digest;
    r->admission_decision_digest=d->decision_identity;
    r->semantic_object_digest=d->semantic_object_digest;
    r->source_span_count=1;r->source_span_digests[0]=d->source_span_digests[0];
    r->retrieval_bundle_count=1;r->retrieval_bundle_package_digests[0]=f->attachments[which].retrieval_bundle_package_digest;
    r->retrieval_item_attachment_count=1;r->retrieval_item_attachment_digests[0]=d->source_attachment_digests[0];
    r->graph_edge_provenance_status=GRAPH_PROVENANCE_NOT_APPLICABLE;
    r->HACF_package_digest=f->layer.HACF_package_digest;
    assert(elpis_admission_receipt_identity(r,&r->receipt_digest)==SEMANTIC_OK);
}
static void finalize(fixture *f){
    for(unsigned i=0;i<3;i++){
       f->layer.admission_decision_digests[i]=f->decisions[i].decision_identity;
       f->layer.admission_receipt_digests[i]=f->receipts[i].receipt_digest;
    }
    f->layer.admission_decision_count=f->layer.admission_receipt_count=3;
    f->layer.admitted_claim_count=2;f->layer.admitted_relation_count=1;
    assert(elpis_evidence_admission_identity(&f->layer,&f->layer.admission_layer_digest)==SEMANTIC_OK);
    f->p.admission_layer=&f->layer;
    f->p.trusted_base_snapshot_digest=&f->layer.base_snapshot_digest;
    f->p.relation=&f->relation;
    f->p.relation_decision=&f->decisions[2];f->p.relation_receipt=&f->receipts[2];
    f->p.source=(elpis_semantic_cgraph_endpoint_v1){&f->decisions[0],&f->receipts[0],
       &f->spans[0],&f->attachments[0],(const uint8_t *)f->texts[0],(uint32_t)strlen(f->texts[0])};
    f->p.target=(elpis_semantic_cgraph_endpoint_v1){&f->decisions[1],&f->receipts[1],
       &f->spans[1],&f->attachments[1],(const uint8_t *)f->texts[1],(uint32_t)strlen(f->texts[1])};
    f->p.corpus=f->corpus;
}
static void set_relation(fixture *f,evidence_relation_type type){
    elpis_relation_candidate_init(&f->relation);
    f->relation.typer_profile_digest=h("typer");
    f->relation.relation_type=type;
    f->relation.evidence_claim_candidate_digest=f->decisions[0].candidate_digest;
    f->relation.evidence_object_kind=OBJECT_KIND_CLAIM_NODE;
    f->relation.evidence_object_digest=f->decisions[0].semantic_object_digest;
    f->relation.target_object_kind=OBJECT_KIND_CLAIM_NODE;
    f->relation.target_object_digest=f->decisions[1].semantic_object_digest;
    f->relation.evidence_role=RELATION_ROLE_EVIDENCE;
    f->relation.target_role=RELATION_ROLE_TARGET;
    f->relation.relation_polarity=RELATION_POLARITY_AFFIRMATIVE;
    f->relation.source_span_count=1;
    f->relation.source_span_digests[0]=f->spans[0].span_identity;
    f->relation.confidence_key=100;
    assert(elpis_relation_candidate_identity(&f->relation,&f->relation.candidate_identity)==SEMANTIC_OK);
    decision(f,2,1);finalize(f);
}
static fixture *make_fixture(void){
    fixture *f=calloc(1,sizeof(*f));assert(f);
    assert(elpis_corpus_open_ephemeral(&f->corpus)==0);
    const char *data[]={"Alpha original evidence chunk.","Beta target evidence chunk."};
    for(unsigned i=0;i<2;i++){
        elpis_ingest_meta m={"elpis.docs","reference","text/plain",i?"beta":"alpha"};
        elpis_ingest_result r={0};
        assert(elpis_corpus_ingest_bytes(f->corpus,data[i],strlen(data[i]),&m,&r)==0);
    }
    elpis_evidence_admission_init(&f->layer);
    f->layer.base_snapshot_digest=h("snapshot");f->layer.query_overlay_digest=h("overlay");
    f->layer.retrieval_expansion_digest=h("expansion");
    f->layer.retrieval_expanded_view_digest=h("expanded");
    f->layer.typing_bundle_digest=h("typing");
    f->layer.admission_policy_digest=h("policy");
    f->layer.admission_segment_digest=h("admission-segment");
    f->layer.HACF_package_digest=h("hacf-package");
    endpoint(f,0);endpoint(f,1);
    decision(f,0,0);decision(f,1,0);
    set_relation(f,RELATION_TYPE_SUPPORTS);
    return f;
}
static void insist_refusal(const fixture *f){
    elpis_context_edge_input edge;elpis_semantic_cgraph_audit_v1 audit;
    memset(&edge,0xA5,sizeof(edge));memset(&audit,0x5A,sizeof(audit));
    assert(elpis_semantic_cgraph_project_v1(&f->p,&edge,&audit)!=SEMANTIC_OK);
    const uint8_t *e=(const uint8_t *)&edge,*a=(const uint8_t *)&audit;
    for(size_t i=0;i<sizeof(edge);i++)assert(e[i]==0xA5);
    for(size_t i=0;i<sizeof(audit);i++)assert(a[i]==0x5A);
}
int main(void){
    fixture *f=make_fixture();
    elpis_context_edge_input a,b,edges[3];
    elpis_semantic_cgraph_audit_v1 audit,other;
    assert(elpis_semantic_cgraph_project_v1(&f->p,&a,&audit)==SEMANTIC_OK);
    assert(a.edge_type==ELPIS_CGRAPH_SEMANTIC_SUPPORTS && a.authority==1);
    assert(elpis_semantic_cgraph_project_v1(&f->p,&b,&other)==SEMANTIC_OK);
    assert(memcmp(&a,&b,sizeof(a))==0 && eq(&audit.provenance_digest,&other.provenance_digest));
    assert(strcmp(a.subject_chunk_digest,a.object_chunk_digest)!=0);
    assert(strcmp(a.provenance_digest,"0000000000000000000000000000000000000000000000000000000000000000")!=0);
    /* Existing native graph identity and duplicate law remain authority. */
    elpis_context_graph *g=NULL,*g2=NULL;
    assert(elpis_context_graph_create(&a,1,&g)==0);
    assert(elpis_context_graph_create((elpis_context_edge_input[]){a,a},2,&g2)==0);
    char digest1[65],digest2[65];
    assert(elpis_context_graph_digest(g,digest1)==0&&elpis_context_graph_digest(g2,digest2)==0);
    assert(strcmp(digest1,digest2)==0);
    elpis_context_graph_destroy(g);elpis_context_graph_destroy(g2);
    set_relation(f,RELATION_TYPE_CONTRADICTS);
    assert(elpis_semantic_cgraph_project_v1(&f->p,&b,&other)==SEMANTIC_OK);
    assert(b.edge_type==ELPIS_CGRAPH_SEMANTIC_CONTRADICTS);
    edges[0]=a;edges[1]=b;edges[2]=a;
    assert(elpis_context_graph_create(edges,3,&g)==0);
    edges[0]=b;edges[1]=a;edges[2]=a;
    assert(elpis_context_graph_create(edges,3,&g2)==0);
    assert(elpis_context_graph_digest(g,digest1)==0&&elpis_context_graph_digest(g2,digest2)==0);
    assert(strcmp(digest1,digest2)==0);
    elpis_context_graph_destroy(g);elpis_context_graph_destroy(g2);
    /* Adversarial rejections must not write partial edge/audit output. */
    f->decisions[2].decision_disposition=DISPOSITION_REJECTED_POLICY;insist_refusal(f);
    f->decisions[2].decision_disposition=DISPOSITION_ADMITTED_NEW_OBJECT;
    f->relation.target_object_digest=h("wrong-target");insist_refusal(f);
    f->relation.target_object_digest=f->decisions[1].semantic_object_digest;
    f->spans[0].byte_end_exclusive++;insist_refusal(f);f->spans[0].byte_end_exclusive--;
    char corrupt[128]={0}; memcpy(corrupt,f->texts[1],strlen(f->texts[1])+1);corrupt[0]^=1;
    f->p.target.item_text=(const uint8_t *)corrupt;insist_refusal(f);
    f->p.target.item_text=(const uint8_t *)f->texts[1];
    f->p.target.attachment=&f->attachments[0];insist_refusal(f);
    f->p.target.attachment=&f->attachments[1];
    f->attachments[1].item_authority=3;insist_refusal(f);
    f->attachments[1].item_authority=1;
    f->p.trusted_base_snapshot_digest=&f->decisions[0].candidate_digest;insist_refusal(f);
    f->p.trusted_base_snapshot_digest=&f->layer.base_snapshot_digest;
    f->p.corpus=NULL;insist_refusal(f);
    elpis_corpus_close(f->corpus);free(f);
    puts("PASS_SEMANTIC_CONTEXT_PROJECTION_V1_ADVERSARIAL");
    return 0;
}
