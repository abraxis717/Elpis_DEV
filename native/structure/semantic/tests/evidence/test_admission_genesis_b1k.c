#define _POSIX_C_SOURCE 200809L
/* B1i byte-identity materialization witness built on B1h end-to-end claim/actual retrieval package composition fixture.
 * The complete chain is materialized; downstream audits are NOT mocked.
 */
#include "elpis_semantic/admission_genesis_b1k.h"
#include "elpis_semantic/type_registry.h"
#include "elpis_semantic/hypergraph.h"
#include <unistd.h>
#include <sys/stat.h>
#include "elpis/sha256.h"
#include "elpis/retrieval_bundle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"B1K_FAIL line=%d check=%s\n",__LINE__,#x);return 1;} } while(0)
static void fill(hacf_digest *d,uint8_t c){memset(d->bytes,c,32);}
static const uint8_t canonical_payload[]="single canonical claim payload bytes";
static void node(elpis_semantic_node_v1 *n){
 memset(n,0,sizeof(*n));n->abi_version=SEMANTIC_ABI_VERSION;
 n->node_type=SEMANTIC_NODE_NAMESPACE|1;elpis_sha256(canonical_payload,sizeof(canonical_payload)-1,n->payload_digest.bytes);
 (void)elpis_semantic_node_identity(n,&n->node_identity);
}
static void hex(const hacf_digest *d,char s[65]){elpis_hex32(d->bytes,s);}
static void hex_bytes(const uint8_t *p,size_t n,char *s){
 static const char h[]="0123456789abcdef";
 for(size_t i=0;i<n;i++){s[i*2]=h[p[i]>>4];s[i*2+1]=h[p[i]&15];}
 s[n*2]=0;
}
static int make_json(char *out,size_t cap,
                     const elpis_retrieval_item_attachment_v1 *a,
                     const uint8_t *text,size_t text_n){
 char chunk[65],doc[65],parent[65],td[65],corp[65],policy[65],graph[65],query[65],index[65];
 char texthex[512];
 if(text_n>255)return -1;
 hex_bytes(text,text_n,texthex);
 hex(&a->chunk_digest,chunk);hex(&a->document_digest,doc);
 hex(&a->graph_parent_digest,parent);hex(&a->text_digest,td);
 hacf_digest d;fill(&d,0x11);hex(&d,corp);fill(&d,0x17);hex(&d,policy);
 fill(&d,0);hex(&d,graph);fill(&d,0x13);hex(&d,query);fill(&d,0x19);hex(&d,index);
 return snprintf(out,cap,
  "{\"abi_version\":1,\"authority_filter\":null,\"corpus_manifest_digest\":\"%s\",\"fusion_policy_digest\":\"%s\",\"graph_snapshot_digest\":\"%s\",\"items\":[{\"authority\":\"reference\",\"chunk_digest\":\"%s\",\"dense_rank\":2,\"dense_score_key\":-12,\"doc_digest\":\"%s\",\"edge_authority\":0,\"edge_type\":0,\"final_rank\":0,\"fusion_score_key\":500,\"graph_hop\":0,\"graph_parent_digest\":\"%s\",\"item_kind\":1,\"lexical_rank\":1,\"namespace_hex\":\"7265706f\",\"source_mask\":3,\"text_hex\":\"%s\",\"text_bytes\":%zu,\"text_digest\":\"%s\"}],\"namespace_filter_hex\":null,\"query_digest\":\"%s\",\"query_text_hex\":\"6173\",\"schema\":\"%s\",\"vector_index_manifest_digest\":\"%s\"}",
  corp,policy,graph,chunk,doc,parent,texthex,text_n,td,query,ELPIS_RETRIEVAL_BUNDLE_SCHEMA,index);
}
static hacf_digest package(const uint8_t *raw,size_t len){
 hacf_digest p={0},deps[2],schema,query,pol;
 fill(&query,0x13);fill(&pol,0x17);fill(&deps[0],0x11);fill(&deps[1],0x19);
 elpis_sha256(ELPIS_RETRIEVAL_BUNDLE_SCHEMA,strlen(ELPIS_RETRIEVAL_BUNDLE_SCHEMA),schema.bytes);
 hacf_package_spec spec={0};
 spec.abi_version=1;spec.object_type=HACF_OBJ_RETRIEVAL_BUNDLE;
 spec.schema_version=1;spec.authority=HACF_AUTH_REFERENCE;
 spec.schema_digest=schema;spec.policy_digest=pol;
 spec.parents=&query;spec.parent_count=1;spec.dependencies=deps;spec.dependency_count=2;
 spec.payload=raw;spec.payload_bytes=len;
 if(hacf_digest_package(&spec,&p))memset(&p,0,sizeof(p));
 return p;
}
static void catalog_build(uint8_t catalog[SEMANTIC_B1D_CATALOG_BYTES],hacf_digest *pin,
                          const semantic_snapshot_manifest *base,
                          const elpis_evidence_admission_policy_v1 *policy,
                          const uint8_t *raw,size_t raw_n){
 hacf_digest policy_digest,package_digest=package(raw,raw_n);
 (void)elpis_admission_policy_identity(policy,&policy_digest);
 memset(catalog,0,SEMANTIC_B1D_CATALOG_BYTES);
 memcpy(catalog,"ELPIS-B1D-AUTH01",16);catalog[19]=1;
 memcpy(catalog+20,policy_digest.bytes,32);
 memcpy(catalog+52,package_digest.bytes,32);
 elpis_sha256(raw,raw_n,catalog+84);
 memcpy(catalog+116,base->manifest_digest.bytes,32);
 elpis_sha256(catalog,SEMANTIC_B1D_CATALOG_BYTES,pin->bytes);
}
static int run(void){
 static const uint8_t text[]="source byte anchor raw text";
 const size_t text_n=sizeof(text)-1;
 semantic_snapshot_manifest *base=semantic_snapshot_create();
 elpis_evidence_typing_bundle_v1 *bundle=calloc(1,sizeof(*bundle));
 elpis_evidence_admission_v1 *layer=calloc(1,sizeof(*layer));
 elpis_evidence_admission_decision_v1 *d=calloc(1,sizeof(*d));
 elpis_evidence_admission_receipt_v1 *r=calloc(1,sizeof(*r));
 elpis_evidence_admission_policy_v1 *policy=calloc(1,sizeof(*policy));
 CHECK(base && bundle && layer && d && r && policy);
 elpis_admission_policy_init_default(policy);
 semantic_type_registry *reg=semantic_type_registry_create();CHECK(reg);
 semantic_node_type_entry nt={0};nt.node_type=SEMANTIC_NODE_NAMESPACE|1;
 nt.semantic_flag_mask=SEMANTIC_NODE_FLAG_MASK;nt.min_authority=0;nt.max_authority=3;
 CHECK(semantic_type_registry_add_node_type(reg,&nt)==SEMANTIC_OK);
 CHECK(semantic_type_registry_seal(reg,NULL)==SEMANTIC_OK);
 /* Build REAL serialized zero-record genesis segment and finalized manifest. */
 char temporary_dir[]="/tmp/elpis-b1k-XXXXXX";
 CHECK(mkdtemp(temporary_dir)!=NULL);
 char segment_path[512];
 CHECK(snprintf(segment_path,sizeof(segment_path),"%s/genesis.segment",temporary_dir)>0);
 semantic_hypergraph_builder *empty_builder=semantic_builder_create(reg);
 CHECK(empty_builder);
 CHECK(semantic_type_registry_digest(reg,&base->type_registry_digest)==SEMANTIC_OK);
 CHECK(semantic_genesis_identity(&base->type_registry_digest,&base->genesis_identity)==SEMANTIC_OK);
 semantic_segment_record empty_segment;
 CHECK(semantic_segment_build(empty_builder,reg,&base->genesis_identity,&empty_segment)==SEMANTIC_OK);
 CHECK(semantic_segment_write(&empty_segment,empty_builder,segment_path,NULL)==SEMANTIC_OK);
 CHECK(semantic_snapshot_add_segment(base,&empty_segment)==SEMANTIC_OK);
 CHECK(semantic_snapshot_finalize(base)==SEMANTIC_OK);
 CHECK(semantic_snapshot_validate(base)==SEMANTIC_OK);
 semantic_builder_destroy(empty_builder);
 const char *segment_paths[]={segment_path};
 hacf_digest base_pin=base->manifest_digest, base_report;
 CHECK(semantic_b1j_empty_base_chain_audit(base,&base_pin,segment_paths,1,&base_report)==SEMANTIC_OK);
 hacf_digest query;fill(&query,11);
 semantic_query_overlay *ov=semantic_overlay_create(base,reg,&query);CHECK(ov);
 elpis_semantic_node_v1 nd;node(&nd);
 CHECK(semantic_overlay_add_node(ov,&nd)==SEMANTIC_OK);
 elpis_retrieval_item_attachment_v1 att={0};
 att.abi_version=RETRIEVAL_ITEM_ATTACHMENT_ABI_VERSION;
 fill(&att.evidence_node_digest,23);
 fill(&att.retrieval_requirement_digest,25);
 fill(&att.chunk_digest,22);fill(&att.document_digest,24);
 elpis_sha256("repo",4,att.namespace_digest.bytes);
 elpis_sha256(text,text_n,att.text_digest.bytes);
 att.item_authority=1;att.item_kind=1;
 att.source_mask=3;att.lexical_rank=1;att.dense_rank=2;
 att.dense_score_key=-12;att.fusion_score_key=500;
 att.graph_edge_provenance_status=GRAPH_PROVENANCE_NOT_APPLICABLE;
 char raw[8192];int n=make_json(raw,sizeof(raw),&att,text,text_n);
 CHECK(n>0&&(size_t)n<sizeof(raw));
 att.retrieval_bundle_package_digest=package((const uint8_t*)raw,(size_t)n);
 elpis_sha256(raw,(size_t)n,att.retrieval_bundle_digest.bytes);
 CHECK(elpis_attachment_digest(&att,&att.attachment_digest)==SEMANTIC_OK);
 hacf_digest retrieval,expanded,typer;fill(&retrieval,4);fill(&expanded,5);fill(&typer,6);
 elpis_evidence_span_v1 sp={0};sp.abi_version=EVIDENCE_SPAN_ABI_VERSION;
 sp.retrieval_expansion_digest=retrieval;
 sp.retrieval_bundle_digest=att.retrieval_bundle_digest;
 sp.retrieval_bundle_package_digest=att.retrieval_bundle_package_digest;
 sp.retrieval_item_attachment_digest=att.attachment_digest;
 sp.evidence_node_digest=att.evidence_node_digest;
 sp.chunk_digest=att.chunk_digest;sp.item_text_digest=att.text_digest;
 sp.byte_start=1;sp.byte_end_exclusive=(uint32_t)text_n-1;
 sp.span_flags=EVIDENCE_SPAN_FLAG_PRIMARY;
 elpis_sha256(text+1,text_n-2,sp.span_bytes_digest.bytes);
 CHECK(elpis_evidence_span_identity(&sp,&sp.span_identity)==SEMANTIC_OK);
 elpis_semantic_assertion_v1 assertion={0};
 assertion.abi_version=SEMANTIC_ABI_VERSION;
 assertion.asserted_object_kind=SEMANTIC_OBJECT_KIND_NODE;
 assertion.asserted_object_digest=nd.node_identity;
 assertion.provenance_digest=sp.span_identity;assertion.authority=1;
 CHECK(elpis_semantic_assertion_identity(&assertion,&assertion.assertion_identity)==SEMANTIC_OK);
 CHECK(semantic_overlay_add_assertion(ov,&assertion)==SEMANTIC_OK);
 CHECK(semantic_overlay_finalize(ov)==SEMANTIC_OK);
 elpis_evidence_claim_candidate_v1 claim={0};
 claim.abi_version=EVIDENCE_CLAIM_CANDIDATE_ABI_VERSION;
 claim.typer_profile_digest=typer;claim.claim_type=1;
 elpis_sha256(canonical_payload,sizeof(canonical_payload)-1,claim.claim_payload_digest.bytes);
 claim.claim_payload_object_digest=claim.claim_payload_digest;
 claim.source_span_count=1;claim.source_span_digests[0]=sp.span_identity;
 claim.claim_polarity=CLAIM_POLARITY_AFFIRMATIVE;
 claim.claim_modality=CLAIM_MODALITY_ASSERTED;claim.confidence_key=9000;
 CHECK(elpis_claim_candidate_identity(&claim,&claim.candidate_identity)==SEMANTIC_OK);
 hacf_digest policy_pin;CHECK(elpis_admission_policy_identity(policy,&policy_pin)==SEMANTIC_OK);
 elpis_typing_bundle_init(bundle);
 bundle->base_snapshot_digest=base->manifest_digest;
 bundle->query_overlay_digest=ov->overlay_identity;
 bundle->retrieval_expansion_digest=retrieval;
 bundle->retrieval_expanded_view_digest=expanded;
 bundle->typer_profile_digest=typer;
 bundle->evidence_span_count=1;bundle->evidence_span_digests[0]=sp.span_identity;
 bundle->claim_candidate_count=1;bundle->claim_candidate_digests[0]=claim.candidate_identity;
 fill(&bundle->typing_bundle_policy_digest,7);
 CHECK(elpis_typing_bundle_identity(bundle,&bundle->typing_bundle_digest)==SEMANTIC_OK);
 CHECK(elpis_typing_bundle_validate(bundle)==SEMANTIC_OK);
 elpis_admission_decision_init(d);
 d->candidate_kind=CANDIDATE_KIND_CLAIM;d->candidate_digest=claim.candidate_identity;
 d->typing_bundle_digest=bundle->typing_bundle_digest;
 d->admission_policy_digest=policy_pin;
 d->validation_stage_reached=VALIDATION_STAGE_COMPLETE;
 d->decision_disposition=DISPOSITION_ADMITTED_NEW_OBJECT;
 d->semantic_object_kind=SEMANTIC_OBJECT_KIND_CLAIM;
 d->semantic_object_digest=nd.node_identity;d->effective_authority=1;
 d->source_span_count=1;d->source_span_digests[0]=sp.span_identity;
 d->source_attachment_count=1;d->source_attachment_digests[0]=att.attachment_digest;
 CHECK(elpis_admission_decision_identity(d,&d->decision_identity)==SEMANTIC_OK);
 elpis_admission_receipt_init(r);
 r->base_snapshot_digest=base->manifest_digest;r->query_overlay_digest=ov->overlay_identity;
 r->retrieval_expansion_digest=retrieval;r->retrieval_expanded_view_digest=expanded;
 r->typing_bundle_digest=bundle->typing_bundle_digest;r->typer_profile_digest=typer;
 r->candidate_digest=claim.candidate_identity;r->admission_policy_digest=policy_pin;
 r->admission_decision_digest=d->decision_identity;r->semantic_object_digest=nd.node_identity;
 r->source_span_count=1;r->source_span_digests[0]=sp.span_identity;
 r->retrieval_bundle_count=1;
 r->retrieval_bundle_package_digests[0]=att.retrieval_bundle_package_digest;
 r->retrieval_item_attachment_count=1;
 r->retrieval_item_attachment_digests[0]=att.attachment_digest;
 CHECK(elpis_admission_receipt_identity(r,&r->receipt_digest)==SEMANTIC_OK);
 elpis_evidence_admission_init(layer);
 layer->base_snapshot_digest=base->manifest_digest;
 layer->query_overlay_digest=ov->overlay_identity;
 layer->retrieval_expansion_digest=retrieval;
 layer->retrieval_expanded_view_digest=expanded;
 layer->typing_bundle_digest=bundle->typing_bundle_digest;
 layer->admission_policy_digest=policy_pin;
 layer->admission_decision_count=1;
 layer->admission_decision_digests[0]=d->decision_identity;
 layer->admission_receipt_count=1;
 layer->admission_receipt_digests[0]=r->receipt_digest;
 layer->admitted_claim_count=1;
 CHECK(elpis_evidence_admission_identity(layer,&layer->admission_layer_digest)==SEMANTIC_OK);
 semantic_b1b_claim_source source={&claim,&sp,&att,text,text_n};
 uint8_t catalog[SEMANTIC_B1D_CATALOG_BYTES];hacf_digest pin;
 catalog_build(catalog,&pin,base,policy,(const uint8_t*)raw,(size_t)n);
 hacf_digest report,sentinel,good,item_report,source_report;
 fill(&sentinel,0xa5);
#define AUDIT() semantic_b1h_claim_decision_replay_audit(catalog,sizeof(catalog),&pin,base,ov,bundle,policy,layer,d,r,&source,(const uint8_t*)raw,(size_t)n,&report)
#define REFUSE() do {report=sentinel;CHECK(AUDIT()!=SEMANTIC_OK);CHECK(memcmp(&report,&sentinel,32)==0);}while(0)
 CHECK(semantic_b1f_primary_item_audit(catalog,sizeof(catalog),&pin,base,policy,
    (const uint8_t*)raw,(size_t)n,&att,text,text_n,&item_report)==SEMANTIC_OK);
 CHECK(semantic_b1b_claim_source_audit(base,ov,bundle,policy,&policy_pin,
    &att.retrieval_bundle_package_digest,layer,d,r,&source,1,&source_report)==SEMANTIC_OK);
 report=sentinel;CHECK(AUDIT()==SEMANTIC_OK);good=report;
 report=sentinel;CHECK(AUDIT()==SEMANTIC_OK && !memcmp(&good,&report,32));
 /* A malicious decision can be self-consistently rehashed and pass B1g
  * while claiming admission with an error reason. B1h must reject it. */
 d->decision_reason=REASON_POLICY;
 CHECK(elpis_admission_decision_identity(d,&d->decision_identity)==SEMANTIC_OK);
 r->admission_decision_digest=d->decision_identity;
 layer->admission_decision_digests[0]=d->decision_identity;
 CHECK(elpis_admission_receipt_identity(r,&r->receipt_digest)==SEMANTIC_OK);
 layer->admission_receipt_digests[0]=r->receipt_digest;
 CHECK(elpis_evidence_admission_identity(layer,&layer->admission_layer_digest)==SEMANTIC_OK);
 CHECK(semantic_b1g_single_claim_composition_audit(catalog,sizeof(catalog),&pin,
     base,ov,bundle,policy,layer,d,r,&source,(const uint8_t*)raw,(size_t)n,
     &source_report)==SEMANTIC_OK);
 REFUSE();
 d->decision_reason=REASON_NONE;
 CHECK(elpis_admission_decision_identity(d,&d->decision_identity)==SEMANTIC_OK);
 r->admission_decision_digest=d->decision_identity;
 layer->admission_decision_digests[0]=d->decision_identity;
 CHECK(elpis_admission_receipt_identity(r,&r->receipt_digest)==SEMANTIC_OK);
 layer->admission_receipt_digests[0]=r->receipt_digest;
 CHECK(elpis_evidence_admission_identity(layer,&layer->admission_layer_digest)==SEMANTIC_OK);
 report=sentinel;CHECK(AUDIT()==SEMANTIC_OK && !memcmp(&report,&good,32));
 /* Another self-consistent decision diagnostic cannot be silently ignored. */
 fill(&d->decision_diagnostic_digest,0x49);
 CHECK(elpis_admission_decision_identity(d,&d->decision_identity)==SEMANTIC_OK);
 r->admission_decision_digest=d->decision_identity;
 layer->admission_decision_digests[0]=d->decision_identity;
 CHECK(elpis_admission_receipt_identity(r,&r->receipt_digest)==SEMANTIC_OK);
 layer->admission_receipt_digests[0]=r->receipt_digest;
 CHECK(elpis_evidence_admission_identity(layer,&layer->admission_layer_digest)==SEMANTIC_OK);
 CHECK(semantic_b1g_single_claim_composition_audit(catalog,sizeof(catalog),&pin,
     base,ov,bundle,policy,layer,d,r,&source,(const uint8_t*)raw,(size_t)n,
     &source_report)==SEMANTIC_OK);
 REFUSE();
 memset(&d->decision_diagnostic_digest,0,sizeof(d->decision_diagnostic_digest));
 CHECK(elpis_admission_decision_identity(d,&d->decision_identity)==SEMANTIC_OK);
 r->admission_decision_digest=d->decision_identity;
 layer->admission_decision_digests[0]=d->decision_identity;
 CHECK(elpis_admission_receipt_identity(r,&r->receipt_digest)==SEMANTIC_OK);
 layer->admission_receipt_digests[0]=r->receipt_digest;
 CHECK(elpis_evidence_admission_identity(layer,&layer->admission_layer_digest)==SEMANTIC_OK);
 report=sentinel;CHECK(AUDIT()==SEMANTIC_OK && !memcmp(&report,&good,32));
 /* Byte-level identity projection requires an explicitly materialized payload.
  * Unlike B1h, B1i must reject altered bytes even with valid source/receipt. */
 hacf_digest b1i_report,b1i_baseline;
#define B1I(bytes,sz) semantic_b1i_identity_projection_audit(catalog,sizeof(catalog),&pin,base,ov,bundle,policy,layer,d,r,&source,(const uint8_t*)raw,(size_t)n,(bytes),(sz),&b1i_report)
 b1i_report=sentinel;CHECK(B1I(canonical_payload,sizeof(canonical_payload)-1)==SEMANTIC_OK);
 b1i_baseline=b1i_report;
 b1i_report=sentinel;CHECK(B1I(canonical_payload,sizeof(canonical_payload)-1)==SEMANTIC_OK && !memcmp(&b1i_baseline,&b1i_report,32));
 /* B1k positive: BOTH audits recomputed using this same base manifest. */
 hacf_digest b1k_result,b1k_baseline,b1k_sentinel=sentinel;
#define B1K() semantic_b1k_genesis_claim_audit(catalog,sizeof(catalog),&pin,base,&base_pin,segment_paths,1,ov,bundle,policy,layer,d,r,&source,(const uint8_t*)raw,(size_t)n,canonical_payload,sizeof(canonical_payload)-1,&b1k_result)
 b1k_result=sentinel; CHECK(B1K()==SEMANTIC_OK);
 b1k_baseline=b1k_result;
 b1k_result=sentinel; CHECK(B1K()==SEMANTIC_OK && !memcmp(&b1k_result,&b1k_baseline,32));
 /* Wrong externally pinned base cannot be rescued by a valid B1i claim. */
 base_pin.bytes[0]^=1;b1k_result=sentinel;
 CHECK(B1K()!=SEMANTIC_OK && !memcmp(&b1k_result,&b1k_sentinel,32));
 base_pin.bytes[0]^=1;
 /* Corrupt serialized segment while keeping catalog, policy, and claim valid. */
 FILE *seg_file=fopen(segment_path,"r+b");CHECK(seg_file);
 unsigned char byte;CHECK(fread(&byte,1,1,seg_file)==1);
 CHECK(fseek(seg_file,0,SEEK_SET)==0);unsigned char corrupted_byte=(unsigned char)(byte^0x80);
 CHECK(fwrite(&corrupted_byte,1,1,seg_file)==1);CHECK(fflush(seg_file)==0);CHECK(fclose(seg_file)==0);
 b1k_result=sentinel;CHECK(B1K()!=SEMANTIC_OK && !memcmp(&b1k_result,&b1k_sentinel,32));
 seg_file=fopen(segment_path,"r+b");CHECK(seg_file);
 CHECK(fwrite(&byte,1,1,seg_file)==1);CHECK(fflush(seg_file)==0);CHECK(fclose(seg_file)==0);
 b1k_result=sentinel;CHECK(B1K()==SEMANTIC_OK && !memcmp(&b1k_result,&b1k_baseline,32));
 /* A forged empty-count claim over non-genesis history must be rejected even
  * if all claim receipts remain internally self-consistent. */
 hacf_digest old_prior=base->prior_manifest_digest;
 fill(&base->prior_manifest_digest,0x44);
 CHECK(semantic_snapshot_finalize(base)==SEMANTIC_OK);
 base_pin=base->manifest_digest;
 b1k_result=sentinel;CHECK(B1K()!=SEMANTIC_OK && !memcmp(&b1k_result,&b1k_sentinel,32));
 base->prior_manifest_digest=old_prior;
 CHECK(semantic_snapshot_finalize(base)==SEMANTIC_OK);
 base_pin=base->manifest_digest;
 b1k_result=sentinel;CHECK(B1K()==SEMANTIC_OK && !memcmp(&b1k_result,&b1k_baseline,32));
 /* Transactional output alias protection; do not overwrite trusted pin. */
 hacf_digest before=base_pin;
 CHECK(semantic_b1k_genesis_claim_audit(catalog,sizeof(catalog),&pin,base,&base_pin,segment_paths,1,ov,bundle,policy,layer,d,r,&source,(const uint8_t*)raw,(size_t)n,canonical_payload,sizeof(canonical_payload)-1,&base_pin)!=SEMANTIC_OK);
 CHECK(!memcmp(&before,&base_pin,32));
#undef B1K
 uint8_t corrupted[sizeof(canonical_payload)-1];memcpy(corrupted,canonical_payload,sizeof(corrupted));corrupted[0]^=1;
 b1i_report=sentinel;CHECK(B1I(corrupted,sizeof(corrupted))!=SEMANTIC_OK && !memcmp(&b1i_report,&sentinel,32));
 b1i_report=sentinel;CHECK(B1I(canonical_payload,sizeof(canonical_payload)-2)!=SEMANTIC_OK && !memcmp(&b1i_report,&sentinel,32));
 /* Rewrite every identity affected by the candidate's altered payload digest.
  * The predecessor B1h must accept the self-consistent rehash, while B1i
  * rejects the unsupported payload projection. */
 fill(&claim.claim_payload_digest,0x77);
 CHECK(elpis_claim_candidate_identity(&claim,&claim.candidate_identity)==SEMANTIC_OK);
 bundle->claim_candidate_digests[0]=claim.candidate_identity;
 CHECK(elpis_typing_bundle_identity(bundle,&bundle->typing_bundle_digest)==SEMANTIC_OK);
 d->candidate_digest=claim.candidate_identity;d->typing_bundle_digest=bundle->typing_bundle_digest;
 CHECK(elpis_admission_decision_identity(d,&d->decision_identity)==SEMANTIC_OK);
 r->candidate_digest=claim.candidate_identity;r->typing_bundle_digest=bundle->typing_bundle_digest;
 r->admission_decision_digest=d->decision_identity;
 CHECK(elpis_admission_receipt_identity(r,&r->receipt_digest)==SEMANTIC_OK);
 layer->typing_bundle_digest=bundle->typing_bundle_digest;
 layer->admission_decision_digests[0]=d->decision_identity;layer->admission_receipt_digests[0]=r->receipt_digest;
 CHECK(elpis_evidence_admission_identity(layer,&layer->admission_layer_digest)==SEMANTIC_OK);
 CHECK(AUDIT()==SEMANTIC_OK);
 b1i_report=sentinel;CHECK(B1I(canonical_payload,sizeof(canonical_payload)-1)!=SEMANTIC_OK && !memcmp(&b1i_report,&sentinel,32));
 claim.claim_payload_object_digest=claim.claim_payload_digest;
 CHECK(elpis_claim_candidate_identity(&claim,&claim.candidate_identity)==SEMANTIC_OK);
 bundle->claim_candidate_digests[0]=claim.candidate_identity;
 CHECK(elpis_typing_bundle_identity(bundle,&bundle->typing_bundle_digest)==SEMANTIC_OK);
 d->candidate_digest=claim.candidate_identity;d->typing_bundle_digest=bundle->typing_bundle_digest;
 CHECK(elpis_admission_decision_identity(d,&d->decision_identity)==SEMANTIC_OK);
 r->candidate_digest=claim.candidate_identity;r->typing_bundle_digest=bundle->typing_bundle_digest;
 r->admission_decision_digest=d->decision_identity;
 CHECK(elpis_admission_receipt_identity(r,&r->receipt_digest)==SEMANTIC_OK);
 layer->typing_bundle_digest=bundle->typing_bundle_digest;
 layer->admission_decision_digests[0]=d->decision_identity;layer->admission_receipt_digests[0]=r->receipt_digest;
 CHECK(elpis_evidence_admission_identity(layer,&layer->admission_layer_digest)==SEMANTIC_OK);
 CHECK(AUDIT()==SEMANTIC_OK);
 b1i_report=sentinel;CHECK(B1I(canonical_payload,sizeof(canonical_payload)-1)!=SEMANTIC_OK && !memcmp(&b1i_report,&sentinel,32));
 elpis_sha256(canonical_payload,sizeof(canonical_payload)-1,claim.claim_payload_digest.bytes);
 claim.claim_payload_object_digest=claim.claim_payload_digest;
 CHECK(elpis_claim_candidate_identity(&claim,&claim.candidate_identity)==SEMANTIC_OK);
 bundle->claim_candidate_digests[0]=claim.candidate_identity;
 CHECK(elpis_typing_bundle_identity(bundle,&bundle->typing_bundle_digest)==SEMANTIC_OK);
 d->candidate_digest=claim.candidate_identity;d->typing_bundle_digest=bundle->typing_bundle_digest;
 CHECK(elpis_admission_decision_identity(d,&d->decision_identity)==SEMANTIC_OK);
 r->candidate_digest=claim.candidate_identity;r->typing_bundle_digest=bundle->typing_bundle_digest;
 r->admission_decision_digest=d->decision_identity;
 CHECK(elpis_admission_receipt_identity(r,&r->receipt_digest)==SEMANTIC_OK);
 layer->typing_bundle_digest=bundle->typing_bundle_digest;
 layer->admission_decision_digests[0]=d->decision_identity;layer->admission_receipt_digests[0]=r->receipt_digest;
 CHECK(elpis_evidence_admission_identity(layer,&layer->admission_layer_digest)==SEMANTIC_OK);
 b1i_report=sentinel;CHECK(B1I(canonical_payload,sizeof(canonical_payload)-1)==SEMANTIC_OK && !memcmp(&b1i_baseline,&b1i_report,32));
#undef B1I
 /* Source/receipt corruption must fail without writing the report. */
 r->source_span_digests[0].bytes[0]^=1;REFUSE();r->source_span_digests[0].bytes[0]^=1;
 source.item_text_bytes--;REFUSE();source.item_text_bytes++;
 pin.bytes[0]^=1;REFUSE();pin.bytes[0]^=1;
 /* A different valid catalog/raw bundle is not allowed to borrow the old
  * admission records: package identity, attachment and receipt must agree. */
 char alternative[8192];memcpy(alternative,raw,(size_t)n+1);
 char *q=strstr(alternative,"\"query_text_hex\":\"6173\"");CHECK(q);
 q+=strlen("\"query_text_hex\":\"");*q='b';
 uint8_t other_catalog[SEMANTIC_B1D_CATALOG_BYTES];hacf_digest other_pin;
 catalog_build(other_catalog,&other_pin,base,policy,(const uint8_t*)alternative,(size_t)n);
 hacf_digest alternative_package,outer_report;
 CHECK(semantic_b1e_retrieval_package_audit(other_catalog,sizeof(other_catalog),
     &other_pin,base,policy,(const uint8_t*)alternative,(size_t)n,
     &alternative_package,&outer_report)==SEMANTIC_OK);
 report=sentinel;
 CHECK(semantic_b1g_single_claim_composition_audit(other_catalog,sizeof(other_catalog),
     &other_pin,base,ov,bundle,policy,layer,d,r,&source,
     (const uint8_t*)alternative,(size_t)n,&report)!=SEMANTIC_OK);
 CHECK(!memcmp(&report,&sentinel,32));
 /* Policy with stronger source diversity cannot pass the B1b/B1c witness,
  * despite having a correctly re-pinned B1f package. */
 policy->minimum_distinct_documents=2;
 catalog_build(other_catalog,&other_pin,base,policy,(const uint8_t*)raw,(size_t)n);
 CHECK(semantic_b1f_primary_item_audit(other_catalog,sizeof(other_catalog),
     &other_pin,base,policy,(const uint8_t*)raw,(size_t)n,&att,
     text,text_n,&item_report)==SEMANTIC_OK);
 report=sentinel;
 CHECK(semantic_b1g_single_claim_composition_audit(other_catalog,sizeof(other_catalog),
     &other_pin,base,ov,bundle,policy,layer,d,r,&source,
     (const uint8_t*)raw,(size_t)n,&report)!=SEMANTIC_OK);
 CHECK(!memcmp(&report,&sentinel,32));
 policy->minimum_distinct_documents=1;
 report=sentinel;CHECK(AUDIT()==SEMANTIC_OK && !memcmp(&report,&good,32));
 /* Output aliases must be refused before any write. */
 CHECK(semantic_b1g_single_claim_composition_audit(catalog,sizeof(catalog),&pin,
     base,ov,bundle,policy,layer,d,r,&source,(const uint8_t*)raw,(size_t)n,
     (hacf_digest*)(void*)catalog)!=SEMANTIC_OK);
 CHECK(!memcmp(catalog,"ELPIS-B1D-AUTH01",16));
 semantic_overlay_destroy(ov);semantic_type_registry_destroy(reg);
 semantic_snapshot_destroy(base);free(bundle);free(layer);free(d);free(r);free(policy);
 CHECK(unlink(segment_path)==0);CHECK(rmdir(temporary_dir)==0);
 return 0;
}
int main(void){int rc=run();puts(rc?"B1I_FAIL":"PASS_B1K_GENESIS_CLAIM_COMPOSITION");return rc;}
