/* B1e native retrieval-bundle package recomputation. No I/O or persistence. */
#include "elpis_semantic/admission_bundle_b1e.h"
#include "elpis/retrieval_bundle.h"
#include "elpis/sha256.h"
#include <string.h>

/* The on-wire JSON is produced by retrieval_bundle.cpp:build_json_and_identities.
 * No whitespace or field reordering is emitted. We validate that exact outer
 * order, lengths, schema and overall JSON grammar; the items array remains
 * structurally parsed but SEMANTICALLY UNTRUSTED until a later verifier.
 * Bounded reads use [cur,end) throughout, never NUL termination.
 */
typedef struct {const uint8_t *cur, *end;} json_cursor;
static int byte(json_cursor *j, uint8_t b) {
    if (j->cur==j->end || *j->cur!=b) return 0;
    ++j->cur; return 1;
}
static int lit(json_cursor *j, const char *s) {
    size_t n=strlen(s);
    if ((size_t)(j->end-j->cur)<n || memcmp(j->cur,s,n)) return 0;
    j->cur+=n; return 1;
}
static int hexc(uint8_t x) {return (x>='0'&&x<='9')||(x>='a'&&x<='f');}
static int hexval(uint8_t x) {return x<='9'?x-'0':x-'a'+10;}
static int hex_digest(json_cursor *j, hacf_digest *d) {
    if ((size_t)(j->end-j->cur)<64) return 0;
    for (size_t i=0;i<64;i++) if (!hexc(j->cur[i])) return 0;
    for (size_t i=0;i<32;i++)d->bytes[i]=(uint8_t)((hexval(j->cur[2*i])<<4)|hexval(j->cur[2*i+1]));
    j->cur+=64;return 1;
}
static int hex_string(json_cursor *j) {
    if (!byte(j,'"')) return 0;
    size_t n=0;
    while (j->cur!=j->end && *j->cur!='"') {
        if (!hexc(*j->cur)) return 0;
        ++j->cur; ++n;
    }
    return !(n&1) && byte(j,'"');
}
static int json_string(json_cursor *j) {
    if (!byte(j,'"'))return 0;
    while (j->cur!=j->end) {
        uint8_t b=*j->cur++;
        if (b=='"')return 1;
        if (b<32)return 0;
        if (b=='\\') {
            if (j->cur==j->end)return 0;
            b=*j->cur++;
            if (b=='u') {
                for (int i=0;i<4;i++) {
                    if (j->cur==j->end)return 0;
                    uint8_t c=*j->cur++;
                    if (!((c>='0'&&c<='9')||(c>='a'&&c<='f')||(c>='A'&&c<='F')))return 0;
                }
            } else if (b==0 || !strchr("\"\\/bfnrt",(int)b))return 0;
        }
    }
    return 0;
}
static int json_number(json_cursor *j) {
    const uint8_t *p=j->cur;
    if (!byte(j,'-')) j->cur=p;
    if (byte(j,'0')) {
        if (j->cur!=j->end && *j->cur>='0' && *j->cur<='9')return 0;
    } else {
        if (j->cur==j->end || *j->cur<'1' || *j->cur>'9')return 0;
        do {++j->cur;} while (j->cur!=j->end && *j->cur>='0' && *j->cur<='9');
    }
    /* HACF writer only emits integers, never floating point or exponents. */
    return j->cur!=p && (j->cur==j->end || (*j->cur!='.' && *j->cur!='e' && *j->cur!='E'));
}
static int json_value(json_cursor *j, unsigned depth) {
    if (depth>12 || j->cur==j->end)return 0;
    if (*j->cur=='"')return json_string(j);
    if (*j->cur=='{') {
        ++j->cur;
        if (byte(j,'}'))return 1;
        do {
            if (!json_string(j)||!byte(j,':')||!json_value(j,depth+1))return 0;
            if (byte(j,'}'))return 1;
        } while(byte(j,','));
        return 0;
    }
    if (*j->cur=='[') {
        ++j->cur;
        if (byte(j,']'))return 1;
        do {
            if (!json_value(j,depth+1))return 0;
            if (byte(j,']'))return 1;
        } while(byte(j,','));
        return 0;
    }
    if (*j->cur=='t')return lit(j,"true");
    if (*j->cur=='f')return lit(j,"false");
    if (*j->cur=='n')return lit(j,"null");
    return json_number(j);
}
static int items_array(json_cursor *j) {
    if (!byte(j,'['))return 0;
    if (byte(j,']'))return 1;
    for (unsigned i=0;i<128;i++) {
        if (j->cur==j->end || *j->cur!='{' || !json_value(j,0))return 0;
        if (byte(j,']'))return 1;
        if (!byte(j,','))return 0;
    }
    return 0;
}
static int outer(json_cursor *j, hacf_digest *corpus,hacf_digest *fusion,
                 hacf_digest *graph,hacf_digest *query,hacf_digest *vindex) {
    /* These keys and order match the frozen native serializer exactly. */
    if(!lit(j,"{\"abi_version\":1,\"authority_filter\":"))return 0;
    if(!lit(j,"null") && !json_string(j))return 0;
    if(!lit(j,",\"corpus_manifest_digest\":\"")||!hex_digest(j,corpus)||!byte(j,'"'))return 0;
    if(!lit(j,",\"fusion_policy_digest\":\"")||!hex_digest(j,fusion)||!byte(j,'"'))return 0;
    if(!lit(j,",\"graph_snapshot_digest\":\"")||!hex_digest(j,graph)||!byte(j,'"'))return 0;
    if(!lit(j,",\"items\":"))return 0;
    /* B1e intentionally requires an array, not arbitrary JSON. */
    if(!items_array(j))return 0;
    if(!lit(j,",\"namespace_filter_hex\":"))return 0;
    if(!lit(j,"null") && !hex_string(j))return 0;
    if(!lit(j,",\"query_digest\":\"")||!hex_digest(j,query)||!byte(j,'"'))return 0;
    if(!lit(j,",\"query_text_hex\":")||!hex_string(j))return 0;
    if(!lit(j,",\"schema\":\"" ELPIS_RETRIEVAL_BUNDLE_SCHEMA "\""))return 0;
    if(!lit(j,",\"vector_index_manifest_digest\":\"")||!hex_digest(j,vindex)||!lit(j,"\"}"))return 0;
    return j->cur==j->end;
}
int semantic_b1e_retrieval_package_audit(
    const uint8_t *catalog,size_t catalog_bytes,
    const hacf_digest *trusted_catalog_sha256,
    const semantic_snapshot_manifest *base,
    const elpis_evidence_admission_policy_v1 *policy,
    const uint8_t *raw_bundle_json,size_t raw_bundle_bytes,
    hacf_digest *verified_package_out,hacf_digest *descriptive_audit_out) {
    if (!catalog || !trusted_catalog_sha256 || !base || !policy ||
        !raw_bundle_json || !raw_bundle_bytes ||
        raw_bundle_bytes>SEMANTIC_B1D_MAX_ARTIFACT_BYTES ||
        !verified_package_out || !descriptive_audit_out ||
        verified_package_out==descriptive_audit_out)return SEMANTIC_E_INVAL;
    hacf_digest pinned_policy,pinned_package,b1d_report;
    int rc=semantic_b1d_authority_bytes_audit(catalog,catalog_bytes,
        trusted_catalog_sha256,base,policy,raw_bundle_json,raw_bundle_bytes,
        &pinned_policy,&pinned_package,&b1d_report);
    if(rc!=SEMANTIC_OK)return rc;
    hacf_digest corpus,fusion,graph,query,vindex;
    json_cursor j={raw_bundle_json,raw_bundle_json+raw_bundle_bytes};
    if(!outer(&j,&corpus,&fusion,&graph,&query,&vindex))return SEMANTIC_E_INVAL;
    static const uint8_t zero[32]={0};
    if(!memcmp(corpus.bytes,zero,32)||!memcmp(query.bytes,zero,32)||
       !memcmp(vindex.bytes,zero,32)||!memcmp(fusion.bytes,zero,32))return SEMANTIC_E_INVAL;
    hacf_digest schema,deps[3],got;
    elpis_sha256(ELPIS_RETRIEVAL_BUNDLE_SCHEMA,
                 strlen(ELPIS_RETRIEVAL_BUNDLE_SCHEMA),schema.bytes);
    deps[0]=corpus;deps[1]=vindex;
    uint32_t dep_count=2;
    if(memcmp(graph.bytes,zero,32)) {deps[2]=graph;dep_count=3;}
    hacf_package_spec pkg={0};
    pkg.abi_version=1;
    pkg.object_type=HACF_OBJ_RETRIEVAL_BUNDLE;
    pkg.schema_version=1;
    pkg.authority=HACF_AUTH_REFERENCE;
    pkg.schema_digest=schema;
    pkg.policy_digest=fusion;
    pkg.parents=&query;
    pkg.parent_count=1;
    pkg.dependencies=deps;
    pkg.dependency_count=dep_count;
    pkg.payload=raw_bundle_json;
    pkg.payload_bytes=raw_bundle_bytes;
    if(hacf_digest_package(&pkg,&got)!=0)return SEMANTIC_E_INVAL;
    if(memcmp(got.bytes,pinned_package.bytes,32))return SEMANTIC_E_AUTHORITY;
    hacf_digest report;
    static const char tag[]="elpis.semantic.b1e.retrieval-package-audit.v1";
    elpis_sha256_ctx h;
    elpis_sha256_init(&h);
    elpis_sha256_update(&h,tag,sizeof(tag)-1);
    elpis_sha256_update(&h,b1d_report.bytes,32);
    elpis_sha256_update(&h,got.bytes,32);
    elpis_sha256_final(&h,report.bytes);
    *verified_package_out=got;
    *descriptive_audit_out=report;
    return SEMANTIC_OK;
}
