/* B1j: verify ALL serialized segments of a genesis-only empty snapshot.
 * Uses the existing native segment reader, which verifies complete serialized
 * records and HACF projections. Never infer absence from an injected view or
 * an unbound manifest count. Restricted genesis-only all-zero-record chain.
 */
#define _POSIX_C_SOURCE 200809L
#include "elpis_semantic/admission_base_b1j.h"
#include "elpis/sha256.h"
#include "elpis_semantic/identity.h"
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>

static int equal_digest(const hacf_digest *a,const hacf_digest *b) {
    return memcmp(a->bytes,b->bytes,HACF_DIGEST_BYTES)==0;
}
static int zero_digest(const hacf_digest *a) {
    const hacf_digest z={{0}};
    return equal_digest(a,&z);
}
static int disjoint(const void *a,size_t na,const void *b,size_t nb) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(!a||!b||!na||!nb||x>UINTPTR_MAX-na||y>UINTPTR_MAX-nb) return 0;
    return x+na<=y || y+nb<=x;
}
int semantic_b1j_empty_base_chain_audit(
    const semantic_snapshot_manifest *manifest,
    const hacf_digest *trusted_manifest_digest,
    const char *const *segment_paths,
    uint32_t path_count,
    hacf_digest *descriptive_report_out) {
    if(!manifest || !trusted_manifest_digest || !segment_paths || !descriptive_report_out ||
       path_count==0 || path_count>SEMANTIC_MAX_SEGMENTS ||
       !disjoint(descriptive_report_out,sizeof(*descriptive_report_out),manifest,sizeof(*manifest)) ||
       !disjoint(descriptive_report_out,sizeof(*descriptive_report_out),trusted_manifest_digest,sizeof(*trusted_manifest_digest)) ||
       !disjoint(descriptive_report_out,sizeof(*descriptive_report_out),segment_paths,(size_t)path_count*sizeof(*segment_paths)))
        return SEMANTIC_E_INVAL;
    if(semantic_snapshot_validate(manifest)!=SEMANTIC_OK ||
       !equal_digest(&manifest->manifest_digest,trusted_manifest_digest) ||
       zero_digest(trusted_manifest_digest) ||
       zero_digest(&manifest->type_registry_digest) ||
       !zero_digest(&manifest->prior_manifest_digest) ||
       manifest->segment_count!=path_count ||
       manifest->unique_node_count || manifest->unique_hyperedge_count ||
       manifest->assertion_count || manifest->incidence_count)
        return SEMANTIC_E_AUTHORITY;
    hacf_digest genesis;
    if(semantic_genesis_identity(&manifest->type_registry_digest,&genesis)!=SEMANTIC_OK ||
       !equal_digest(&manifest->genesis_identity,&genesis))
        return SEMANTIC_E_AUTHORITY;
    hacf_digest predecessor=genesis;
    for(uint32_t i=0;i<path_count;++i) {
        const char *path=segment_paths[i];
        /* A verified immutable pathname is required. The returned chain hash
         * binds the contents read, not any future replacement of the path. */
        if(!path || !*path || strlen(path)>4000) return SEMANTIC_E_AUTHORITY;
        struct stat st;
        if(lstat(path,&st)!=0 || !S_ISREG(st.st_mode)) return SEMANTIC_E_AUTHORITY;
        semantic_segment_record seg;
        hacf_digest verified_digest;
        if(semantic_segment_read(path,&seg,&verified_digest)!=SEMANTIC_OK ||
           !equal_digest(&verified_digest,&manifest->segment_digests[i]) ||
           !equal_digest(&seg.segment_identity,&verified_digest) ||
           !equal_digest(&seg.type_registry_digest,&manifest->type_registry_digest) ||
           !equal_digest(&seg.prior_snapshot_digest,&predecessor) ||
           seg.node_count || seg.assertion_count || seg.hyperedge_count ||
           seg.incidence_count || seg.hacf_op_count)
            return SEMANTIC_E_AUTHORITY;
        predecessor=seg.hacf_next_snapshot;
    }
    if(!equal_digest(&predecessor,&manifest->hacf_graph_snapshot_digest))
        return SEMANTIC_E_AUTHORITY;
    static const char domain[]="elpis.semantic.b1j.genesis-empty-chain.v1";
    elpis_sha256_ctx hash;
    hacf_digest result;
    elpis_sha256_init(&hash);
    elpis_sha256_update(&hash,domain,sizeof(domain)-1);
    elpis_sha256_update(&hash,trusted_manifest_digest->bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&hash,genesis.bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&hash,predecessor.bytes,HACF_DIGEST_BYTES);
    for(uint32_t i=0;i<path_count;++i)
        elpis_sha256_update(&hash,manifest->segment_digests[i].bytes,HACF_DIGEST_BYTES);
    elpis_sha256_final(&hash,result.bytes);
    *descriptive_report_out=result; /* output only after EVERY gate succeeded */
    return SEMANTIC_OK;
}
