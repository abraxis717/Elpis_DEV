/* B1f native read-only item audit: positive and re-pinned adversarial cases. */
#include "elpis_semantic/admission_item_b1f.h"
#include "elpis/retrieval_bundle.h"
#include "elpis/sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL B1f line %d: %s\n",__LINE__,#x);return 1;}}while(0)
static const uint8_t sample[]={'A','\0','z',0xff};
static void hex(const hacf_digest *d,char out[65]){elpis_hex32(d->bytes,out);}
static void fill(hacf_digest *d,uint8_t n){memset(d->bytes,n,32);}
static void setup(elpis_retrieval_item_attachment_v1 *a){
 memset(a,0,sizeof(*a));
 a->abi_version=RETRIEVAL_ITEM_ATTACHMENT_ABI_VERSION;
 fill(&a->evidence_node_digest,0x25);
 fill(&a->retrieval_requirement_digest,0x26);
 fill(&a->chunk_digest,0x27);
 fill(&a->document_digest,0x28);
 a->item_authority=1;a->source_mask=3;a->lexical_rank=1;a->dense_rank=2;
 a->dense_score_key=-12;a->fusion_score_key=500;
 a->item_kind=1;a->graph_edge_provenance_status=GRAPH_PROVENANCE_NOT_APPLICABLE;
 elpis_sha256("repo",4,a->namespace_digest.bytes);
 elpis_sha256(sample,sizeof(sample),a->text_digest.bytes);
}
static int make_json(char *out,size_t cap,const elpis_retrieval_item_attachment_v1 *a){
 char chunk[65],doc[65],parent[65],text_digest[65];
 char corpus[65],fusion[65],graph[65],query[65],index[65];
 hacf_digest d;
 hex(&a->chunk_digest,chunk);hex(&a->document_digest,doc);
 hex(&a->graph_parent_digest,parent);hex(&a->text_digest,text_digest);
 fill(&d,0x11);hex(&d,corpus);
 fill(&d,0x17);hex(&d,fusion);
 fill(&d,0);hex(&d,graph);
 fill(&d,0x13);hex(&d,query);
 fill(&d,0x19);hex(&d,index);
 return snprintf(out,cap,
  "{\"abi_version\":1,\"authority_filter\":null,\"corpus_manifest_digest\":\"%s\",\"fusion_policy_digest\":\"%s\",\"graph_snapshot_digest\":\"%s\",\"items\":[{\"authority\":\"reference\",\"chunk_digest\":\"%s\",\"dense_rank\":2,\"dense_score_key\":-12,\"doc_digest\":\"%s\",\"edge_authority\":0,\"edge_type\":0,\"final_rank\":0,\"fusion_score_key\":500,\"graph_hop\":0,\"graph_parent_digest\":\"%s\",\"item_kind\":1,\"lexical_rank\":1,\"namespace_hex\":\"7265706f\",\"source_mask\":3,\"text_hex\":\"41007aff\",\"text_bytes\":4,\"text_digest\":\"%s\"}],\"namespace_filter_hex\":null,\"query_digest\":\"%s\",\"query_text_hex\":\"6173\",\"schema\":\"%s\",\"vector_index_manifest_digest\":\"%s\"}",
  corpus,fusion,graph,chunk,doc,parent,text_digest,query,ELPIS_RETRIEVAL_BUNDLE_SCHEMA,index);
}
static hacf_digest package(const uint8_t *raw,size_t len){
 hacf_digest r={0},deps[2],schema,q,policy;
 fill(&q,0x13);fill(&policy,0x17);fill(&deps[0],0x11);fill(&deps[1],0x19);
 elpis_sha256(ELPIS_RETRIEVAL_BUNDLE_SCHEMA,strlen(ELPIS_RETRIEVAL_BUNDLE_SCHEMA),schema.bytes);
 hacf_package_spec spec={0};
 spec.abi_version=1;spec.object_type=HACF_OBJ_RETRIEVAL_BUNDLE;
 spec.schema_version=1;spec.authority=HACF_AUTH_REFERENCE;
 spec.schema_digest=schema;spec.policy_digest=policy;
 spec.parents=&q;spec.parent_count=1;spec.dependencies=deps;spec.dependency_count=2;
 spec.payload=raw;spec.payload_bytes=len;
 if(hacf_digest_package(&spec,&r)!=0)memset(&r,0,sizeof(r));
 return r;
}
static void rebind(uint8_t catalog[SEMANTIC_B1D_CATALOG_BYTES],hacf_digest *pin,
    const semantic_snapshot_manifest *base,const elpis_evidence_admission_policy_v1 *policy,
    elpis_retrieval_item_attachment_v1 *a,const uint8_t *raw,size_t bytes){
 hacf_digest p,policy_digest;
 p=package(raw,bytes);
 a->retrieval_bundle_package_digest=p;
 elpis_sha256(raw,bytes,a->retrieval_bundle_digest.bytes);
 (void)elpis_attachment_digest(a,&a->attachment_digest);
 (void)elpis_admission_policy_identity(policy,&policy_digest);
 memset(catalog,0,SEMANTIC_B1D_CATALOG_BYTES);
 memcpy(catalog,"ELPIS-B1D-AUTH01",16);catalog[19]=1;
 memcpy(catalog+20,policy_digest.bytes,32);
 memcpy(catalog+52,p.bytes,32);
 elpis_sha256(raw,bytes,catalog+84);
 memcpy(catalog+116,base->manifest_digest.bytes,32);
 elpis_sha256(catalog,SEMANTIC_B1D_CATALOG_BYTES,pin->bytes);
}
static int run(void){
 semantic_snapshot_manifest *base=semantic_snapshot_create();
 elpis_evidence_admission_policy_v1 *policy=calloc(1,sizeof(*policy));
 CHECK(base && policy);
 base->segment_count=1;CHECK(semantic_snapshot_finalize(base)==SEMANTIC_OK);
 elpis_admission_policy_init_default(policy);
 elpis_retrieval_item_attachment_v1 a;setup(&a);
 char raw[8192],saved[8192];int n=make_json(raw,sizeof(raw),&a);
 CHECK(n>0 && (size_t)n<sizeof(raw));
 uint8_t catalog[SEMANTIC_B1D_CATALOG_BYTES];hacf_digest pin,report,sentinel,good;
 memset(sentinel.bytes,0xa5,32);
 rebind(catalog,&pin,base,policy,&a,(const uint8_t*)raw,(size_t)n);
#define AUDIT() semantic_b1f_primary_item_audit(catalog,sizeof(catalog),&pin,base,policy,(const uint8_t*)raw,(size_t)n,&a,sample,sizeof(sample),&report)
#define RESET() (report=sentinel)
#define REFUSE() do {RESET();CHECK(AUDIT()!=SEMANTIC_OK);CHECK(!memcmp(&report,&sentinel,32));}while(0)
 RESET();CHECK(AUDIT()==SEMANTIC_OK);good=report;
 RESET();CHECK(AUDIT()==SEMANTIC_OK && !memcmp(&report,&good,32));
 memcpy(saved,raw,(size_t)n+1);
 /* Mutations are rebound to a newly valid catalog, matching the raw package.
  * A failure therefore tests item correspondence rather than an old digest. */
 struct scenario {const char *needle;size_t delta;char replace;} tests[]={
  {"\"authority\":\"reference\"",14,'X'},
  {"\"text_hex\":\"41007aff\"",13,'b'},
  {"\"source_mask\":3",14,'7'},
  {"\"final_rank\":0",13,'1'},
  {"\"dense_score_key\":-12",19,'9'},
  {"\"namespace_hex\":\"7265706f\"",18,'9'},
  {"\"item_kind\":1",12,'2'},
  {"\"text_bytes\":4",13,'9'},
 };
 for(size_t i=0;i<sizeof(tests)/sizeof(tests[0]);i++){
  memcpy(raw,saved,(size_t)n+1);
  char *q=strstr(raw,tests[i].needle);CHECK(q);
  CHECK(tests[i].delta<strlen(tests[i].needle));
  q[tests[i].delta]=tests[i].replace;
  rebind(catalog,&pin,base,policy,&a,(const uint8_t*)raw,(size_t)n);
  hacf_digest item_pkg,inner_outer_report;
  CHECK(semantic_b1e_retrieval_package_audit(catalog,sizeof(catalog),&pin,
     base,policy,(const uint8_t*)raw,(size_t)n,&item_pkg,&inner_outer_report)==SEMANTIC_OK);
  REFUSE();
 }
 memcpy(raw,saved,(size_t)n+1);
 rebind(catalog,&pin,base,policy,&a,(const uint8_t*)raw,(size_t)n);
 a.text_digest.bytes[0]^=1;
 CHECK(elpis_attachment_digest(&a,&a.attachment_digest)==SEMANTIC_OK);
 REFUSE();a.text_digest.bytes[0]^=1;
 CHECK(elpis_attachment_digest(&a,&a.attachment_digest)==SEMANTIC_OK);
 a.retrieval_bundle_package_digest.bytes[0]^=1;REFUSE();a.retrieval_bundle_package_digest.bytes[0]^=1;
 a.source_mask=7;
 CHECK(elpis_attachment_digest(&a,&a.attachment_digest)==SEMANTIC_OK);
 REFUSE();a.source_mask=3;
 CHECK(elpis_attachment_digest(&a,&a.attachment_digest)==SEMANTIC_OK);
 a.graph_hop=1;
 CHECK(elpis_attachment_digest(&a,&a.attachment_digest)==SEMANTIC_OK);
 REFUSE();a.graph_hop=0;
 CHECK(elpis_attachment_digest(&a,&a.attachment_digest)==SEMANTIC_OK);
 a.item_authority=2;
 CHECK(elpis_attachment_digest(&a,&a.attachment_digest)==SEMANTIC_OK);
 REFUSE();a.item_authority=1;
 CHECK(elpis_attachment_digest(&a,&a.attachment_digest)==SEMANTIC_OK);
 RESET();CHECK(AUDIT()==SEMANTIC_OK && !memcmp(&report,&good,32));
 /* Catalog pin is always external; altered pin refuses. */
 pin.bytes[0]^=1;REFUSE();pin.bytes[0]^=1;
 /* Output aliases must not overwrite the catalog or raw retrieval bytes. */
 CHECK(semantic_b1f_primary_item_audit(catalog,sizeof(catalog),&pin,base,policy,
    (const uint8_t*)raw,(size_t)n,&a,sample,sizeof(sample),
    (hacf_digest*)(void*)catalog)!=SEMANTIC_OK);
 CHECK(!memcmp(catalog,"ELPIS-B1D-AUTH01",16));
 CHECK(semantic_b1f_primary_item_audit(catalog,sizeof(catalog),&pin,base,policy,
    (const uint8_t*)raw,(size_t)n,&a,sample,sizeof(sample),
    (hacf_digest*)(void*)raw)!=SEMANTIC_OK);
 CHECK(!memcmp(raw,saved,(size_t)n+1));
 /* An undersized bound never accepts a truncated bundle. */
 n--;REFUSE();n++;
 RESET();CHECK(AUDIT()==SEMANTIC_OK && !memcmp(&report,&good,32));
 semantic_snapshot_destroy(base);free(policy);
 return 0;
}
int main(void){int r=run();puts(r?"B1F_FAIL":"PASS_B1F_NATIVE_ONE_PRIMARY_ITEM");return r;}
