/* B2a: native, in-memory successor materialization from a freshly verified
 * one-claim, populated-genesis B1p admission witness. No writes or head CAS. */
#include "elpis_semantic/admission_successor_b2a.h"
#include "elpis_semantic/hypergraph.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int overlap(const void *a, size_t na, const void *b, size_t nb) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(!a||!b||!na||!nb||x>UINTPTR_MAX-na||y>UINTPTR_MAX-nb)return 1;
    return x<y+nb && y<x+na;
}
#define BAD(p) overlap(segment_out,sizeof(*segment_out),(p),sizeof(*(p))) || \
               overlap(successor_out,sizeof(*successor_out),(p),sizeof(*(p)))
#define BAD_BYTES(p,n) overlap(segment_out,sizeof(*segment_out),(p),(n)) || \
                       overlap(successor_out,sizeof(*successor_out),(p),(n))
int semantic_b2a_prepare_successor(const semantic_b2a_witness *w,
                                   semantic_segment_record *segment_out,
                                   semantic_snapshot_manifest *successor_out) {
    if(!w||!segment_out||!successor_out||
       overlap(segment_out,sizeof(*segment_out),successor_out,sizeof(*successor_out))||
       BAD(w)||!w->base||BAD(w->base)||!w->registry||
       !w->trusted_base_manifest_digest||BAD(w->trusted_base_manifest_digest)||
       !w->trusted_catalog_sha256||BAD(w->trusted_catalog_sha256)||
       !w->overlay||BAD(w->overlay)||!w->overlay->local_builder||
       !w->typing_bundle||BAD(w->typing_bundle)||!w->policy||BAD(w->policy)||
       !w->layer||BAD(w->layer)||!w->decision||BAD(w->decision)||
       !w->receipt||BAD(w->receipt)||!w->claim_source||BAD(w->claim_source)||
       !w->catalog||!w->catalog_bytes||BAD_BYTES(w->catalog,w->catalog_bytes)||
       !w->raw_bundle_json||!w->raw_bundle_bytes||BAD_BYTES(w->raw_bundle_json,w->raw_bundle_bytes)||
       !w->canonical_payload||!w->canonical_payload_bytes||
       BAD_BYTES(w->canonical_payload,w->canonical_payload_bytes)||
       !w->base_segment_paths||!w->base_segment_count||
       BAD_BYTES(w->base_segment_paths,(size_t)w->base_segment_count*sizeof(char*)))
        return SEMANTIC_E_INVAL;
    if(w->base->prior_manifest_digest.bytes[0] != 0) return SEMANTIC_E_AUTHORITY;
    /* B1p audits require a genesis-only populated chain; do not add a new
     * history interpretation to the existing snapshot ABI. */
    if(w->base->segment_count<2 || w->base->segment_count>64 ||
       w->base->segment_count!=w->base_segment_count ||
       w->base->segment_count>=SEMANTIC_MAX_SEGMENTS ||
       semantic_snapshot_validate(w->base)!=SEMANTIC_OK)
        return SEMANTIC_E_AUTHORITY;
    for(size_t i=0;i<HACF_DIGEST_BYTES;i++)
        if(w->base->prior_manifest_digest.bytes[i]!=0)return SEMANTIC_E_AUTHORITY;
    hacf_digest registry_digest={0};
    if(semantic_type_registry_digest(w->registry,&registry_digest)!=SEMANTIC_OK ||
       memcmp(&registry_digest,&w->base->type_registry_digest,sizeof(registry_digest)))
        return SEMANTIC_E_AUTHORITY;
    /* Only the exact one-node/one-assertion object certified by B1p may be
     * materialized. Never copy unrelated overlay records into persistence. */
    const semantic_hypergraph_builder *b=w->overlay->local_builder;
    if(semantic_builder_node_count(b)!=1 || semantic_builder_assertion_count(b)!=1 ||
       semantic_builder_hyperedge_count(b)!=0 || semantic_builder_incidence_count(b)!=0)
        return SEMANTIC_E_AUTHORITY;
    hacf_digest verified={0};
    if(semantic_b1o_populated_claim_novelty_audit(
       w->catalog,w->catalog_bytes,w->trusted_catalog_sha256,
       w->base,w->trusted_base_manifest_digest,w->base_segment_paths,
       w->base_segment_count,w->overlay,w->typing_bundle,w->policy,w->layer,
       w->decision,w->receipt,w->claim_source,w->raw_bundle_json,
       w->raw_bundle_bytes,w->canonical_payload,w->canonical_payload_bytes,
       &verified)!=SEMANTIC_OK)return SEMANTIC_E_AUTHORITY;
    (void)verified; /* descriptive result is not itself a publication permit */
    semantic_segment_record seg={0};
    if(semantic_segment_build(b,w->registry,&w->base->hacf_graph_snapshot_digest,
                              &seg)!=SEMANTIC_OK ||
       seg.node_count!=1 || seg.assertion_count!=1 ||
       seg.hyperedge_count!=0 || seg.incidence_count!=0 ||
       !semantic_segment_validate(&seg,&seg.segment_identity))
        return SEMANTIC_E_AUTHORITY;
    semantic_snapshot_manifest *next=malloc(sizeof(*next));
    if(!next)return SEMANTIC_E_NOMEM;
    memcpy(next,w->base,sizeof(*next));
    next->prior_manifest_digest=w->base->manifest_digest;
    int rc=semantic_snapshot_add_segment(next,&seg);
    if(rc==SEMANTIC_OK)rc=semantic_snapshot_finalize(next);
    if(rc==SEMANTIC_OK)rc=semantic_snapshot_validate(next);
    if(rc==SEMANTIC_OK &&
       (next->segment_count!=w->base->segment_count+1 ||
        memcmp(&next->segment_digests[w->base->segment_count],
               &seg.segment_identity,sizeof(hacf_digest)) ||
        memcmp(&next->hacf_graph_snapshot_digest,&seg.hacf_next_snapshot,sizeof(hacf_digest)) ||
        memcmp(&next->prior_manifest_digest,&w->base->manifest_digest,sizeof(hacf_digest)) ||
        !memcmp(&next->manifest_digest,&w->base->manifest_digest,sizeof(hacf_digest))))
        rc=SEMANTIC_E_AUTHORITY;
    if(rc==SEMANTIC_OK) { *segment_out=seg; *successor_out=*next; }
    free(next);
    return rc;
}
#undef BAD
#undef BAD_BYTES
