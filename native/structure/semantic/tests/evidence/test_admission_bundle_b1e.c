/* B1e native package verifier: actual HACF package digest versus pinned B1d. */
#include "elpis_semantic/admission_bundle_b1e.h"
#include "elpis/retrieval_bundle.h"
#include "elpis/sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(e) do { if(!(e)) {fprintf(stderr,"FAIL B1e line %d: %s\n",__LINE__,#e);return 1;} } while(0)
static void hex(const hacf_digest *d,char out[65]) {elpis_hex32(d->bytes,out);}
static void field(char out[65],uint8_t x){hacf_digest d;memset(d.bytes,x,32);hex(&d,out);}
static hacf_digest package(const uint8_t *raw,size_t len, uint8_t graph) {
    hacf_digest r={0},deps[3],schema,query,policy;
    memset(query.bytes,0x13,32);
    memset(policy.bytes,0x17,32);
    memset(deps[0].bytes,0x11,32);
    memset(deps[1].bytes,0x19,32);
    memset(deps[2].bytes,graph,32);
    elpis_sha256(ELPIS_RETRIEVAL_BUNDLE_SCHEMA,strlen(ELPIS_RETRIEVAL_BUNDLE_SCHEMA),schema.bytes);
    hacf_package_spec spec={0};
    spec.abi_version=1;spec.object_type=HACF_OBJ_RETRIEVAL_BUNDLE;
    spec.schema_version=1;spec.authority=HACF_AUTH_REFERENCE;
    spec.schema_digest=schema;spec.policy_digest=policy;
    spec.parents=&query;spec.parent_count=1;
    spec.dependencies=deps;spec.dependency_count=graph?3:2;
    spec.payload=raw;spec.payload_bytes=len;
    if(hacf_digest_package(&spec,&r)!=0)memset(&r,0,sizeof(r));
    return r;
}
static int fixture(char *dst,size_t cap,uint8_t graph) {
    char corpus[65],fusion[65],gr[65],query[65],vindex[65];
    field(corpus,0x11);field(fusion,0x17);field(gr,graph);field(query,0x13);field(vindex,0x19);
    return snprintf(dst,cap,"{\"abi_version\":1,\"authority_filter\":null,\"corpus_manifest_digest\":\"%s\",\"fusion_policy_digest\":\"%s\",\"graph_snapshot_digest\":\"%s\",\"items\":[{\"authority\":\"reference\",\"chunk_digest\":\"01\",\"dense_rank\":0,\"final_rank\":0,\"text_hex\":\"616263\"}],\"namespace_filter_hex\":null,\"query_digest\":\"%s\",\"query_text_hex\":\"6173\",\"schema\":\"%s\",\"vector_index_manifest_digest\":\"%s\"}",corpus,fusion,gr,query,ELPIS_RETRIEVAL_BUNDLE_SCHEMA,vindex);
}
static void catalog_init(uint8_t catalog[SEMANTIC_B1D_CATALOG_BYTES],
                         const hacf_digest *pol,const hacf_digest *pkg,
                         const hacf_digest *base,const uint8_t *raw,size_t len,
                         hacf_digest *pin) {
    memset(catalog,0,SEMANTIC_B1D_CATALOG_BYTES);
    memcpy(catalog,"ELPIS-B1D-AUTH01",16);
    catalog[19]=1;
    memcpy(catalog+20,pol->bytes,32);
    memcpy(catalog+52,pkg->bytes,32);
    elpis_sha256(raw,len,catalog+84);
    memcpy(catalog+116,base->bytes,32);
    elpis_sha256(catalog,SEMANTIC_B1D_CATALOG_BYTES,pin->bytes);
}
static int run(void) {
    semantic_snapshot_manifest *base=semantic_snapshot_create();
    elpis_evidence_admission_policy_v1 *policy=calloc(1,sizeof(*policy));
    CHECK(base && policy);
    base->segment_count=1;
    CHECK(semantic_snapshot_finalize(base)==SEMANTIC_OK);
    elpis_admission_policy_init_default(policy);
    hacf_digest pol;
    CHECK(elpis_admission_policy_identity(policy,&pol)==SEMANTIC_OK);
    char raw[4096];
    int n=fixture(raw,sizeof(raw),0);
    CHECK(n>0 && (size_t)n<sizeof(raw));
    hacf_digest pkg=package((const uint8_t *)raw,(size_t)n,0),pin,got,report,sentinel;
    uint8_t catalog[SEMANTIC_B1D_CATALOG_BYTES];
    catalog_init(catalog,&pol,&pkg,&base->manifest_digest,(const uint8_t *)raw,(size_t)n,&pin);
    memset(sentinel.bytes,0xa5,32);
#define AUDIT() semantic_b1e_retrieval_package_audit(catalog,sizeof(catalog),&pin,base,policy,(const uint8_t *)raw,(size_t)n,&got,&report)
#define RESET() do {got=report=sentinel;} while(0)
#define REFUSE() do {CHECK(AUDIT()!=SEMANTIC_OK);CHECK(!memcmp(&got,&sentinel,32));CHECK(!memcmp(&report,&sentinel,32));} while(0)
    RESET();CHECK(AUDIT()==SEMANTIC_OK);
    CHECK(memcmp(&got,&pkg,32)==0);
    hacf_digest good=report;
    RESET();CHECK(AUDIT()==SEMANTIC_OK && !memcmp(&report,&good,32));
    /* Rebound catalog/raw bytes each time, so refusal proves package parsing. */
    char saved[4096];memcpy(saved,raw,(size_t)n+1);
    char *s=strstr(raw,"\"schema\":\"elpis.retrieval_bundle.v1\"");CHECK(s);
    s[13]='X'; /* mutate inside canonical schema value */
    pkg=package((const uint8_t *)raw,(size_t)n,0);
    catalog_init(catalog,&pol,&pkg,&base->manifest_digest,(const uint8_t *)raw,(size_t)n,&pin);
    RESET();REFUSE();memcpy(raw,saved,(size_t)n+1);
    s=strstr(raw,"\"query_digest\":\"");CHECK(s);
    s[1]='Q';
    pkg=package((const uint8_t *)raw,(size_t)n,0);
    catalog_init(catalog,&pol,&pkg,&base->manifest_digest,(const uint8_t *)raw,(size_t)n,&pin);
    RESET();REFUSE();memcpy(raw,saved,(size_t)n+1);
    s=strstr(raw,"\"items\":[{");CHECK(s);
    s[8]='!';
    pkg=package((const uint8_t *)raw,(size_t)n,0);
    catalog_init(catalog,&pol,&pkg,&base->manifest_digest,(const uint8_t *)raw,(size_t)n,&pin);
    RESET();REFUSE();memcpy(raw,saved,(size_t)n+1);
    /* Catalog retains a wrong package hash even with independently pinned bytes. */
    pkg=package((const uint8_t *)raw,(size_t)n,0);
    pkg.bytes[0]^=1;
    catalog_init(catalog,&pol,&pkg,&base->manifest_digest,(const uint8_t *)raw,(size_t)n,&pin);
    RESET();REFUSE();pkg.bytes[0]^=1;
    catalog_init(catalog,&pol,&pkg,&base->manifest_digest,(const uint8_t *)raw,(size_t)n,&pin);
    RESET();CHECK(AUDIT()==SEMANTIC_OK && !memcmp(&report,&good,32));
    /* Optional graph dependency changes package identity with same byte layout. */
    n=fixture(raw,sizeof(raw),0x23); CHECK(n>0 && (size_t)n<sizeof(raw));
    pkg=package((const uint8_t *)raw,(size_t)n,0x23);
    catalog_init(catalog,&pol,&pkg,&base->manifest_digest,(const uint8_t *)raw,(size_t)n,&pin);
    RESET();CHECK(AUDIT()==SEMANTIC_OK && !memcmp(&got,&pkg,32));
    /* Outputs alias must reject transactionally. */
    RESET();CHECK(semantic_b1e_retrieval_package_audit(catalog,sizeof(catalog),&pin,base,policy,(const uint8_t *)raw,(size_t)n,&got,&got)!=SEMANTIC_OK);
    CHECK(!memcmp(&got,&sentinel,32));
    semantic_snapshot_destroy(base);free(policy);
    return 0;
}
int main(void){int r=run();puts(r?"B1E_FAIL":"PASS_B1E_NATIVE_READ_ONLY_PACKAGE_RECOMPUTE");return r;}
