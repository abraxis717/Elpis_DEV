/* B1l: verify exact node membership over a populated genesis segment.
 * Unlike injected snapshot_view lookups, this scans the records actually
 * decoded and validated by the native reader. No publication authority.
 */
#define _POSIX_C_SOURCE 200809L
#include "elpis_semantic/admission_populated_b1l.h"
#include "elpis_semantic/identity.h"
#include "elpis/sha256.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int eq(const hacf_digest *a,const hacf_digest *b) {
    return memcmp(a->bytes,b->bytes,HACF_DIGEST_BYTES)==0;
}
static int disjoint(const void *a,size_t na,const void *b,size_t nb) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(!a||!b||!na||!nb||x>UINTPTR_MAX-na||y>UINTPTR_MAX-nb)return 0;
    return x+na<=y || y+nb<=x;
}
#define SEP(a,na,b,nb) disjoint((a),(na),(b),(nb))
int semantic_b1l_populated_genesis_node_audit(
    const semantic_snapshot_manifest *manifest,
    const hacf_digest *trusted_manifest_digest,
    const char *segment_path,
    const hacf_digest *target_node_identity,
    uint32_t *node_present_out,
    hacf_digest *descriptive_report_out) {
    static const hacf_digest zero={{0}};
    if(!manifest||!trusted_manifest_digest||!segment_path||!target_node_identity||
       !node_present_out||!descriptive_report_out ||
       !SEP(node_present_out,sizeof(*node_present_out),descriptive_report_out,sizeof(*descriptive_report_out))||
       !SEP(node_present_out,sizeof(*node_present_out),manifest,sizeof(*manifest))||
       !SEP(descriptive_report_out,sizeof(*descriptive_report_out),manifest,sizeof(*manifest))||
       !SEP(node_present_out,sizeof(*node_present_out),trusted_manifest_digest,sizeof(*trusted_manifest_digest))||
       !SEP(descriptive_report_out,sizeof(*descriptive_report_out),trusted_manifest_digest,sizeof(*trusted_manifest_digest))||
       !SEP(node_present_out,sizeof(*node_present_out),target_node_identity,sizeof(*target_node_identity))||
       !SEP(descriptive_report_out,sizeof(*descriptive_report_out),target_node_identity,sizeof(*target_node_identity)))
        return SEMANTIC_E_INVAL;
    size_t path_len=strnlen(segment_path,4001u);
    if(path_len==0 || path_len>4000 ||
       !SEP(node_present_out,sizeof(*node_present_out),segment_path,path_len+1)||
       !SEP(descriptive_report_out,sizeof(*descriptive_report_out),segment_path,path_len+1))
        return SEMANTIC_E_INVAL;
    /* One genesis segment: no incomplete historical count inference. */
    if(semantic_snapshot_validate(manifest)!=SEMANTIC_OK ||
       !eq(trusted_manifest_digest,&manifest->manifest_digest) ||
       eq(trusted_manifest_digest,&zero) ||
       eq(target_node_identity,&zero) ||
       eq(&manifest->type_registry_digest,&zero) ||
       !eq(&manifest->prior_manifest_digest,&zero) ||
       manifest->segment_count!=1 || manifest->unique_node_count==0)
        return SEMANTIC_E_AUTHORITY;
    hacf_digest genesis;
    if(semantic_genesis_identity(&manifest->type_registry_digest,&genesis)!=SEMANTIC_OK ||
       !eq(&genesis,&manifest->genesis_identity))return SEMANTIC_E_AUTHORITY;
    struct stat st;
    if(lstat(segment_path,&st)!=0 || !S_ISREG(st.st_mode))return SEMANTIC_E_AUTHORITY;
    semantic_segment_record seg;
    hacf_digest digest;
    uint32_t occurrences=0;
    if(semantic_segment_read_node_presence(segment_path,target_node_identity,&seg,
           &digest,&occurrences)!=SEMANTIC_OK ||
       occurrences>1 || !eq(&digest,&manifest->segment_digests[0]) ||
       !eq(&seg.segment_identity,&digest) ||
       !eq(&seg.type_registry_digest,&manifest->type_registry_digest) ||
       !eq(&seg.prior_snapshot_digest,&genesis) ||
       !eq(&seg.hacf_next_snapshot,&manifest->hacf_graph_snapshot_digest) ||
       seg.node_count!=manifest->unique_node_count ||
       seg.hyperedge_count!=manifest->unique_hyperedge_count ||
       seg.assertion_count!=manifest->assertion_count ||
       seg.incidence_count!=manifest->incidence_count)
        return SEMANTIC_E_AUTHORITY;
    static const char domain[]="elpis.semantic.b1l.verified-populated-node.v1";
    elpis_sha256_ctx hash;hacf_digest result;
    const uint8_t present=(uint8_t)(occurrences==1);
    elpis_sha256_init(&hash);
    elpis_sha256_update(&hash,domain,sizeof(domain)-1);
    elpis_sha256_update(&hash,trusted_manifest_digest->bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&hash,digest.bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&hash,target_node_identity->bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&hash,&present,1);
    elpis_sha256_final(&hash,result.bytes);
    *node_present_out=(uint32_t)present;
    *descriptive_report_out=result;
    return SEMANTIC_OK;
}

/* B1m: verify every segment in a bounded populated genesis chain before
 * deriving the target's membership. Reuses the B1l native reader, including
 * its complete identity/reference/HACF projection verification. The manifest
 * must be independently pinned. No publication or mutation is performed. */
int semantic_b1m_populated_chain_node_audit(
    const semantic_snapshot_manifest *manifest,
    const hacf_digest *trusted_manifest_digest,
    const char *const *segment_paths,
    uint32_t path_count,
    const hacf_digest *target_node_identity,
    uint32_t *node_present_out,
    hacf_digest *descriptive_report_out) {
    static const hacf_digest zero = {{0}};
    if (!manifest || !trusted_manifest_digest || !segment_paths ||
        !target_node_identity || !node_present_out || !descriptive_report_out ||
        path_count < 2 || path_count > SEMANTIC_B1M_MAX_SEGMENTS ||
        path_count != manifest->segment_count ||
        !SEP(node_present_out,sizeof(*node_present_out),descriptive_report_out,sizeof(*descriptive_report_out)) ||
        !SEP(node_present_out,sizeof(*node_present_out),manifest,sizeof(*manifest)) ||
        !SEP(descriptive_report_out,sizeof(*descriptive_report_out),manifest,sizeof(*manifest)) ||
        !SEP(node_present_out,sizeof(*node_present_out),trusted_manifest_digest,sizeof(*trusted_manifest_digest)) ||
        !SEP(descriptive_report_out,sizeof(*descriptive_report_out),trusted_manifest_digest,sizeof(*trusted_manifest_digest)) ||
        !SEP(node_present_out,sizeof(*node_present_out),target_node_identity,sizeof(*target_node_identity)) ||
        !SEP(descriptive_report_out,sizeof(*descriptive_report_out),target_node_identity,sizeof(*target_node_identity)) ||
        !SEP(node_present_out,sizeof(*node_present_out),segment_paths,(size_t)path_count*sizeof(*segment_paths)) ||
        !SEP(descriptive_report_out,sizeof(*descriptive_report_out),segment_paths,(size_t)path_count*sizeof(*segment_paths)))
        return SEMANTIC_E_INVAL;
    if (semantic_snapshot_validate(manifest)!=SEMANTIC_OK ||
        eq(trusted_manifest_digest,&zero) ||
        !eq(&manifest->manifest_digest,trusted_manifest_digest) ||
        eq(target_node_identity,&zero) ||
        eq(&manifest->type_registry_digest,&zero) ||
        !eq(&manifest->prior_manifest_digest,&zero) ||
        manifest->unique_node_count==0)
        return SEMANTIC_E_AUTHORITY;
    hacf_digest genesis;
    if (semantic_genesis_identity(&manifest->type_registry_digest,&genesis)!=SEMANTIC_OK ||
        !eq(&manifest->genesis_identity,&genesis)) return SEMANTIC_E_AUTHORITY;

    hacf_digest previous=genesis;
    uint64_t nodes=0, assertions=0, hyperedges=0, incidences=0;
    uint32_t total_occurrences=0;
    for(uint32_t i=0;i<path_count;++i) {
        const char *path=segment_paths[i];
        if(!path) return SEMANTIC_E_INVAL;
        size_t n=strnlen(path,4001u);
        if(n==0 || n>4000u ||
           !SEP(node_present_out,sizeof(*node_present_out),path,n+1) ||
           !SEP(descriptive_report_out,sizeof(*descriptive_report_out),path,n+1))
            return SEMANTIC_E_INVAL;
        for(uint32_t j=0;j<i;++j) {
            if(eq(&manifest->segment_digests[i],&manifest->segment_digests[j]) ||
               strcmp(path,segment_paths[j])==0) return SEMANTIC_E_AUTHORITY;
        }
        semantic_segment_record seg;
        hacf_digest file_id;
        uint32_t occurrences=0;
        if(semantic_segment_read_node_presence(path,target_node_identity,
                        &seg,&file_id,&occurrences)!=SEMANTIC_OK ||
           occurrences>1 || total_occurrences > 1u-occurrences ||
           !eq(&file_id,&manifest->segment_digests[i]) ||
           !eq(&seg.segment_identity,&file_id) ||
           !eq(&seg.type_registry_digest,&manifest->type_registry_digest) ||
           !eq(&seg.prior_snapshot_digest,&previous))
            return SEMANTIC_E_AUTHORITY;
        nodes+=seg.node_count;
        assertions+=seg.assertion_count;
        hyperedges+=seg.hyperedge_count;
        incidences+=seg.incidence_count;
        total_occurrences+=occurrences;
        previous=seg.hacf_next_snapshot;
    }
    if(nodes!=manifest->unique_node_count ||
       assertions!=manifest->assertion_count ||
       hyperedges!=manifest->unique_hyperedge_count ||
       incidences!=manifest->incidence_count ||
       !eq(&previous,&manifest->hacf_graph_snapshot_digest))
        return SEMANTIC_E_AUTHORITY;
    static const char domain[]="elpis.semantic.b1m.verified-multisegment-node.v1";
    const uint8_t present=(uint8_t)(total_occurrences==1);
    elpis_sha256_ctx hash;
    hacf_digest result;
    elpis_sha256_init(&hash);
    elpis_sha256_update(&hash,domain,sizeof(domain)-1);
    elpis_sha256_update(&hash,trusted_manifest_digest->bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&hash,genesis.bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&hash,target_node_identity->bytes,HACF_DIGEST_BYTES);
    for(uint32_t i=0;i<path_count;++i)
        elpis_sha256_update(&hash,manifest->segment_digests[i].bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&hash,previous.bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&hash,&present,1);
    elpis_sha256_final(&hash,result.bytes);
    *node_present_out=(uint32_t)present;
    *descriptive_report_out=result;
    return SEMANTIC_OK;
}


/* B1n: identical to B1m's pinned genesis-chain model, but verifies every
 * node identity in every segment, not just the target. One native reader pass
 * per file produces both the checked segment and a complete identity list.
 * Fail-closed for larger/unproven histories. */
static int b1n_node_cmp(const void *a,const void *b) {
    return memcmp(((const hacf_digest *)a)->bytes,
                  ((const hacf_digest *)b)->bytes,HACF_DIGEST_BYTES);
}
int semantic_b1n_global_unique_node_audit(
    const semantic_snapshot_manifest *manifest,
    const hacf_digest *trusted_manifest_digest,
    const char *const *segment_paths, uint32_t path_count,
    const hacf_digest *target_node_identity,
    uint32_t *node_present_out, hacf_digest *descriptive_report_out) {
    static const hacf_digest zero={{0}};
    if (!manifest||!trusted_manifest_digest||!segment_paths||
        !target_node_identity||!node_present_out||!descriptive_report_out||
        path_count<2||path_count>SEMANTIC_B1M_MAX_SEGMENTS||
        path_count!=manifest->segment_count||
        !SEP(node_present_out,sizeof(*node_present_out),descriptive_report_out,sizeof(*descriptive_report_out))||
        !SEP(node_present_out,sizeof(*node_present_out),manifest,sizeof(*manifest))||
        !SEP(descriptive_report_out,sizeof(*descriptive_report_out),manifest,sizeof(*manifest))||
        !SEP(node_present_out,sizeof(*node_present_out),trusted_manifest_digest,sizeof(*trusted_manifest_digest))||
        !SEP(descriptive_report_out,sizeof(*descriptive_report_out),trusted_manifest_digest,sizeof(*trusted_manifest_digest))||
        !SEP(node_present_out,sizeof(*node_present_out),target_node_identity,sizeof(*target_node_identity))||
        !SEP(descriptive_report_out,sizeof(*descriptive_report_out),target_node_identity,sizeof(*target_node_identity))||
        !SEP(node_present_out,sizeof(*node_present_out),segment_paths,(size_t)path_count*sizeof(*segment_paths))||
        !SEP(descriptive_report_out,sizeof(*descriptive_report_out),segment_paths,(size_t)path_count*sizeof(*segment_paths)))
        return SEMANTIC_E_INVAL;
    if (semantic_snapshot_validate(manifest)!=SEMANTIC_OK||
        eq(&manifest->manifest_digest,&zero)||
        !eq(&manifest->manifest_digest,trusted_manifest_digest)||
        eq(target_node_identity,&zero)||
        eq(&manifest->type_registry_digest,&zero)||
        !eq(&manifest->prior_manifest_digest,&zero)||
        manifest->unique_node_count==0||
        manifest->unique_node_count>SEMANTIC_B1N_MAX_TOTAL_NODES)
        return SEMANTIC_E_AUTHORITY;
    hacf_digest genesis;
    if(semantic_genesis_identity(&manifest->type_registry_digest,&genesis)!=SEMANTIC_OK||
       !eq(&manifest->genesis_identity,&genesis)) return SEMANTIC_E_AUTHORITY;
    hacf_digest *all=malloc((size_t)manifest->unique_node_count*sizeof(*all));
    if (!all) return SEMANTIC_E_NOMEM;
    uint32_t total=0,occurrences=0;
    uint64_t nodes=0,assertions=0,hyperedges=0,incidences=0;
    hacf_digest previous=genesis;
    int rc=SEMANTIC_E_AUTHORITY;
    for(uint32_t i=0;i<path_count;++i) {
        const char *path=segment_paths[i];
        if(!path){rc=SEMANTIC_E_INVAL;goto done;}
        size_t n=strnlen(path,4001u);
        if(n==0||n>4000u||
           !SEP(node_present_out,sizeof(*node_present_out),path,n+1)||
           !SEP(descriptive_report_out,sizeof(*descriptive_report_out),path,n+1)){
            rc=SEMANTIC_E_INVAL;goto done;
        }
        for(uint32_t j=0;j<i;++j)
            if(eq(&manifest->segment_digests[i],&manifest->segment_digests[j])||
               strcmp(path,segment_paths[j])==0) goto done;
        semantic_segment_record seg;
        hacf_digest file_digest;
        hacf_digest *inventory=NULL;
        uint32_t count=0;
        int read_rc=semantic_segment_read_node_inventory(path,&seg,&file_digest,&inventory,&count);
        if(read_rc!=SEMANTIC_OK) {free(inventory);goto done;}
        if(count!=seg.node_count||count>manifest->unique_node_count-total||
           !eq(&file_digest,&manifest->segment_digests[i])||
           !eq(&seg.segment_identity,&file_digest)||
           !eq(&seg.type_registry_digest,&manifest->type_registry_digest)||
           !eq(&seg.prior_snapshot_digest,&previous)){
            free(inventory);goto done;
        }
        for(uint32_t j=0;j<count;++j){
            if(eq(&inventory[j],target_node_identity))++occurrences;
            all[total+j]=inventory[j];
        }
        free(inventory);
        total+=count;
        nodes+=seg.node_count;assertions+=seg.assertion_count;
        hyperedges+=seg.hyperedge_count;incidences+=seg.incidence_count;
        previous=seg.hacf_next_snapshot;
    }
    if(nodes!=manifest->unique_node_count||total!=manifest->unique_node_count||
       assertions!=manifest->assertion_count||
       hyperedges!=manifest->unique_hyperedge_count||
       incidences!=manifest->incidence_count||
       !eq(&previous,&manifest->hacf_graph_snapshot_digest)||occurrences>1)
        goto done;
    qsort(all,total,sizeof(*all),b1n_node_cmp);
    for(uint32_t i=1;i<total;++i)
        if(eq(&all[i-1],&all[i]))goto done;
    static const char domain[]="elpis.semantic.b1n.global-node-unique.v1";
    elpis_sha256_ctx h;hacf_digest report;
    const uint8_t present=(uint8_t)(occurrences==1);
    elpis_sha256_init(&h);
    elpis_sha256_update(&h,domain,sizeof(domain)-1);
    elpis_sha256_update(&h,trusted_manifest_digest->bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&h,genesis.bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&h,target_node_identity->bytes,HACF_DIGEST_BYTES);
    for(uint32_t i=0;i<total;++i)elpis_sha256_update(&h,all[i].bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&h,previous.bytes,HACF_DIGEST_BYTES);
    elpis_sha256_update(&h,&present,1);
    elpis_sha256_final(&h,report.bytes);
    *node_present_out=(uint32_t)present;
    *descriptive_report_out=report;
    rc=SEMANTIC_OK;
done:
    free(all);
    return rc;
}

#undef SEP
