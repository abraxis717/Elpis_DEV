/* B1f restricted native HACF item correspondence. No writes or authority. */
#include "elpis_semantic/admission_item_b1f.h"
#include "elpis/sha256.h"
#include <string.h>
#include <limits.h>
#include <stdint.h>

typedef struct {const uint8_t *p,*end;} cur;
static int literal(cur *c,const char *s){
    size_t n=strlen(s);
    if((size_t)(c->end-c->p)<n || memcmp(c->p,s,n))return 0;
    c->p+=n;return 1;
}
static int hval(uint8_t a){
    if(a>='0'&&a<='9')return a-'0';
    if(a>='a'&&a<='f')return a-'a'+10;
    return -1;
}
static int digest(cur *c,const hacf_digest *expected){
    if((size_t)(c->end-c->p)<64)return 0;
    uint8_t raw[32];
    for(size_t k=0;k<32;k++){
        int hi=hval(c->p[k*2]),lo=hval(c->p[k*2+1]);
        if(hi<0||lo<0)return 0;
        raw[k]=(uint8_t)((hi<<4)|lo);
    }
    if(memcmp(raw,expected->bytes,32))return 0;
    c->p+=64;return 1;
}
static int number(cur *c,uint64_t *out){
    if(c->p==c->end||*c->p<'0'||*c->p>'9')return 0;
    if(*c->p=='0' && c->p+1<c->end && c->p[1]>='0' && c->p[1]<='9')return 0;
    uint64_t n=0;
    while(c->p<c->end && *c->p>='0' && *c->p<='9'){
        uint8_t v=(uint8_t)(*c->p-'0');
        if(n>(UINT64_MAX-v)/10u)return 0;
        n=n*10+v;++c->p;
    }
    *out=n;return 1;
}
static int u32(cur *c,uint32_t n){uint64_t v;return number(c,&v)&&v==n;}
static int u64(cur *c,uint64_t n){uint64_t v;return number(c,&v)&&v==n;}
static int s64(cur *c,int64_t n){
    int minus=0;
    if(c->p<c->end && *c->p=='-'){minus=1;++c->p;}
    uint64_t v;if(!number(c,&v))return 0;
    if(minus){
        if(v==0||v>((uint64_t)INT64_MAX)+1)return 0;
        return (v==((uint64_t)INT64_MAX)+1 ? n==INT64_MIN : n==-(int64_t)v);
    }
    return v<=(uint64_t)INT64_MAX&&n==(int64_t)v;
}
/* Interpret serialized namespace_hex as actual namespace bytes. */
static int namespace_hex(cur *c,const hacf_digest *expected){
    if(!literal(c,"\""))return 0;
    elpis_sha256_ctx hash;elpis_sha256_init(&hash);
    size_t n=0;
    while(c->p<c->end && *c->p!='"'){
        if(c->end-c->p<2 || n>=95)return 0;
        int hi=hval(c->p[0]),lo=hval(c->p[1]);
        if(hi<0||lo<0)return 0;
        uint8_t v=(uint8_t)((hi<<4)|lo);
        elpis_sha256_update(&hash,&v,1);
        c->p+=2;n++;
    }
    if(!literal(c,"\""))return 0;
    hacf_digest d;elpis_sha256_final(&hash,d.bytes);
    return !memcmp(d.bytes,expected->bytes,32);
}
/* Compare text_hex bytes without allocating or NUL-terminating untrusted bytes. */
static int text_hex(cur *c,const uint8_t *expected,size_t n){
    if(!literal(c,"\""))return 0;
    if(n>(size_t)(c->end-c->p)/2)return 0;
    for(size_t i=0;i<n;i++){
        int hi=hval(c->p[2*i]),lo=hval(c->p[2*i+1]);
        if(hi<0||lo<0 || (uint8_t)((hi<<4)|lo)!=expected[i])return 0;
    }
    c->p+=2*n;
    return literal(c,"\"");
}
static int primary_item(cur *c,const elpis_retrieval_item_attachment_v1 *a,
                        const uint8_t *text,size_t text_bytes){
    /* Reject any unknown, duplicate, reordered, or noncanonical item field. */
    if(!literal(c,"{\"authority\":\"reference\",\"chunk_digest\":\""))return 0;
    if(!digest(c,&a->chunk_digest))return 0;
    if(!literal(c,"\",\"dense_rank\":" )||!u32(c,a->dense_rank))return 0;
    if(!literal(c,",\"dense_score_key\":")||!s64(c,a->dense_score_key))return 0;
    if(!literal(c,",\"doc_digest\":\"")||!digest(c,&a->document_digest))return 0;
    if(!literal(c,"\",\"edge_authority\":")||!u32(c,0))return 0;
    if(!literal(c,",\"edge_type\":")||!u32(c,0))return 0;
    if(!literal(c,",\"final_rank\":")||!u32(c,0))return 0;
    if(!literal(c,",\"fusion_score_key\":")||!u64(c,a->fusion_score_key))return 0;
    if(!literal(c,",\"graph_hop\":")||!u32(c,0))return 0;
    if(!literal(c,",\"graph_parent_digest\":\"")||!digest(c,&a->graph_parent_digest))return 0;
    if(!literal(c,"\",\"item_kind\":")||!u32(c,1))return 0;
    if(!literal(c,",\"lexical_rank\":")||!u32(c,a->lexical_rank))return 0;
    if(!literal(c,",\"namespace_hex\":")||!namespace_hex(c,&a->namespace_digest))return 0;
    if(!literal(c,",\"source_mask\":")||!u32(c,a->source_mask))return 0;
    if(!literal(c,",\"text_hex\":")||!text_hex(c,text,text_bytes))return 0;
    if(!literal(c,",\"text_bytes\":")||!u64(c,text_bytes))return 0;
    if(!literal(c,",\"text_digest\":\"")||!digest(c,&a->text_digest))return 0;
    return literal(c,"\"}");
}
/* Do not overwrite any read-only authority or witness input through output aliasing. */
static int overlaps(const void *a,size_t na,const void *b,size_t nb){
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(!a||!b||x>UINTPTR_MAX-na||y>UINTPTR_MAX-nb)return 1;
    return x<y+nb && y<x+na;
}
int semantic_b1f_primary_item_audit(
    const uint8_t *catalog,size_t catalog_bytes,
    const hacf_digest *trusted_catalog_sha256,
    const semantic_snapshot_manifest *base,
    const elpis_evidence_admission_policy_v1 *policy,
    const uint8_t *raw,size_t raw_bytes,
    const elpis_retrieval_item_attachment_v1 *a,
    const uint8_t *text,size_t text_bytes,
    hacf_digest *report_out){
    static const uint8_t zero[32]={0};
    if(!catalog||!trusted_catalog_sha256||!base||!policy||!raw||!a||!text||!report_out||
       !text_bytes||text_bytes> (1u<<20) || !raw_bytes||
       raw_bytes>SEMANTIC_B1D_MAX_ARTIFACT_BYTES ||
       overlaps(report_out,sizeof(*report_out),catalog,catalog_bytes) ||
       overlaps(report_out,sizeof(*report_out),trusted_catalog_sha256,sizeof(*trusted_catalog_sha256)) ||
       overlaps(report_out,sizeof(*report_out),base,sizeof(*base)) ||
       overlaps(report_out,sizeof(*report_out),policy,sizeof(*policy)) ||
       overlaps(report_out,sizeof(*report_out),raw,raw_bytes) ||
       overlaps(report_out,sizeof(*report_out),a,sizeof(*a)) ||
       overlaps(report_out,sizeof(*report_out),text,text_bytes))return SEMANTIC_E_INVAL;
    hacf_digest package, b1e_report;
    if(semantic_b1e_retrieval_package_audit(catalog,catalog_bytes,
         trusted_catalog_sha256,base,policy,raw,raw_bytes,&package,
         &b1e_report)!=SEMANTIC_OK)return SEMANTIC_E_AUTHORITY;
    if(elpis_attachment_validate(a)!=SEMANTIC_OK ||
       a->item_kind!=1 || a->item_authority!=1 ||
       (a->source_mask&~3u) || a->source_mask==0 ||
       a->graph_hop!=0 || a->graph_edge_type!=0 || a->graph_edge_authority!=0 ||
       memcmp(a->graph_parent_digest.bytes,zero,32) ||
       a->graph_edge_provenance_status!=GRAPH_PROVENANCE_NOT_APPLICABLE ||
       (a->lexical_rank==0)!=((a->source_mask&1u)==0) ||
       (a->dense_rank==0)!=((a->source_mask&2u)==0) ||
       a->final_rank!=0 ||
       memcmp(a->retrieval_bundle_package_digest.bytes,package.bytes,32))
        return SEMANTIC_E_AUTHORITY;
    hacf_digest bundle_sha, text_sha, att_digest;
    elpis_sha256(raw,raw_bytes,bundle_sha.bytes);
    elpis_sha256(text,text_bytes,text_sha.bytes);
    if(memcmp(a->retrieval_bundle_digest.bytes,bundle_sha.bytes,32)||
       memcmp(a->text_digest.bytes,text_sha.bytes,32)||
       elpis_attachment_digest(a,&att_digest)!=SEMANTIC_OK ||
       memcmp(att_digest.bytes,a->attachment_digest.bytes,32))return SEMANTIC_E_AUTHORITY;
    static const char marker[]=",\"items\":[";
    const size_t marker_len=sizeof(marker)-1;
    const uint8_t *start=NULL;
    for(size_t i=0;i+marker_len<=raw_bytes;i++){
        if(!memcmp(raw+i,marker,marker_len)){
            if(start)return SEMANTIC_E_INVAL;
            start=raw+i+marker_len;
        }
    }
    if(!start)return SEMANTIC_E_INVAL;
    cur c={start,raw+raw_bytes};
    if(!primary_item(&c,a,text,text_bytes) || !literal(&c,"]"))return SEMANTIC_E_AUTHORITY;
    /* All fields beyond ] were verified as canonical outer data by B1e. */
    static const char domain[]="elpis.semantic.b1f.primary-item-audit.v1";
    elpis_sha256_ctx h;elpis_sha256_init(&h);
    elpis_sha256_update(&h,domain,sizeof(domain)-1);
    elpis_sha256_update(&h,b1e_report.bytes,32);
    elpis_sha256_update(&h,att_digest.bytes,32);
    elpis_sha256_update(&h,text_sha.bytes,32);
    hacf_digest report;elpis_sha256_final(&h,report.bytes);
    *report_out=report;
    return SEMANTIC_OK;
}
