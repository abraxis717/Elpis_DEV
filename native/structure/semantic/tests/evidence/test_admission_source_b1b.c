/* B1a audited correspondence, positive and adversarial in-memory fixtures.
 * This test never publishes a segment or snapshot. */
#include "elpis_semantic/admission_source_b1b.h"
#include "elpis/sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) {fprintf(stderr,"FAIL B1a line %d: %s\n",__LINE__,#c); return 1;} } while(0)
static void fill(hacf_digest *d, unsigned char b) { memset(d->bytes,b,32); }
static void setup_node(elpis_semantic_node_v1 *n) {
    memset(n,0,sizeof(*n)); n->abi_version=SEMANTIC_ABI_VERSION;
    n->node_type=SEMANTIC_NODE_NAMESPACE|1; fill(&n->payload_digest,9);
    elpis_semantic_node_identity(n,&n->node_identity);
}
static int run(void) {
    semantic_snapshot_manifest *base=semantic_snapshot_create();
    elpis_evidence_typing_bundle_v1 *bundle=calloc(1,sizeof(*bundle));
    elpis_evidence_admission_v1 *layer=calloc(1,sizeof(*layer));
    elpis_evidence_admission_decision_v1 *d=calloc(1,sizeof(*d));
    elpis_evidence_admission_receipt_v1 *r=calloc(1,sizeof(*r));
    elpis_evidence_admission_policy_v1 *p=calloc(1,sizeof(*p));
    CHECK(base&&bundle&&layer&&d&&r&&p);
    base->segment_count=1;
    fill(&base->hacf_graph_snapshot_digest,21);
    CHECK(semantic_snapshot_finalize(base)==SEMANTIC_OK);
    semantic_type_registry *reg=semantic_type_registry_create(); CHECK(reg);
    semantic_node_type_entry t={0}; t.node_type=SEMANTIC_NODE_NAMESPACE|1;
    t.semantic_flag_mask=SEMANTIC_NODE_FLAG_MASK; t.min_authority=0; t.max_authority=3;
    CHECK(semantic_type_registry_add_node_type(reg,&t)==SEMANTIC_OK);
    CHECK(semantic_type_registry_seal(reg,NULL)==SEMANTIC_OK);
    hacf_digest query;fill(&query,11);
    semantic_query_overlay *ov=semantic_overlay_create(base,reg,&query);CHECK(ov);
    elpis_semantic_node_v1 n; setup_node(&n);
    CHECK(semantic_overlay_add_node(ov,&n)==SEMANTIC_OK);
    hacf_digest span,attachment,candidate,retrieval,expanded,typer;
    fill(&span,1);fill(&attachment,2);fill(&candidate,3);
    fill(&retrieval,4);fill(&expanded,5);fill(&typer,6);
    elpis_semantic_assertion_v1 a={0}; a.abi_version=SEMANTIC_ABI_VERSION;
    a.asserted_object_kind=SEMANTIC_OBJECT_KIND_NODE;
    a.asserted_object_digest=n.node_identity; a.authority=1;
    /* B1b real source-byte fixture: replacement of earlier fake digest fixtures. */
    const uint8_t raw_text[] = "source byte anchor raw text";
    const size_t raw_bytes = sizeof(raw_text)-1;
    elpis_retrieval_item_attachment_v1 att={0};
    att.abi_version=RETRIEVAL_ITEM_ATTACHMENT_ABI_VERSION;
    fill(&att.evidence_node_digest,23);
    fill(&att.retrieval_bundle_digest,18);
    fill(&att.retrieval_bundle_package_digest,20);
    fill(&att.retrieval_requirement_digest,25);
    fill(&att.chunk_digest,22);
    fill(&att.document_digest,24);
    fill(&att.namespace_digest,26);
    att.item_authority=1;
    att.item_kind=1;
    att.graph_hop=0;
    att.graph_edge_provenance_status=GRAPH_PROVENANCE_NOT_APPLICABLE;
    elpis_sha256(raw_text,raw_bytes,att.text_digest.bytes);
    CHECK(elpis_attachment_digest(&att,&att.attachment_digest)==SEMANTIC_OK);
    attachment=att.attachment_digest;
    elpis_evidence_span_v1 sp={0};
    sp.abi_version=EVIDENCE_SPAN_ABI_VERSION;
    sp.retrieval_expansion_digest=retrieval;
    sp.retrieval_bundle_digest=att.retrieval_bundle_digest;
    sp.retrieval_bundle_package_digest=att.retrieval_bundle_package_digest;
    sp.retrieval_item_attachment_digest=attachment;
    sp.evidence_node_digest=att.evidence_node_digest;
    sp.chunk_digest=att.chunk_digest;
    sp.item_text_digest=att.text_digest;
    sp.byte_start=1;sp.byte_end_exclusive=(uint32_t)raw_bytes-1;
    sp.span_flags=EVIDENCE_SPAN_FLAG_PRIMARY;
    elpis_sha256(raw_text+1,raw_bytes-2,sp.span_bytes_digest.bytes);
    CHECK(elpis_evidence_span_identity(&sp,&sp.span_identity)==SEMANTIC_OK);
    span=sp.span_identity;
    /* The assertion must bind the verified span, never a fixture placeholder. */
    a.provenance_digest=span;
    CHECK(elpis_semantic_assertion_identity(&a,&a.assertion_identity)==SEMANTIC_OK);
    CHECK(semantic_overlay_add_assertion(ov,&a)==SEMANTIC_OK);
    CHECK(semantic_overlay_finalize(ov)==SEMANTIC_OK);
    elpis_evidence_claim_candidate_v1 claim={0};
    claim.abi_version=EVIDENCE_CLAIM_CANDIDATE_ABI_VERSION;
    claim.typer_profile_digest=typer;
    claim.claim_type=1;
    fill(&claim.claim_payload_digest,31);
    fill(&claim.claim_payload_object_digest,32);
    claim.source_span_count=1;claim.source_span_digests[0]=span;
    claim.claim_polarity=CLAIM_POLARITY_AFFIRMATIVE;
    claim.claim_modality=CLAIM_MODALITY_ASSERTED;
    claim.confidence_key=9000;
    CHECK(elpis_claim_candidate_identity(&claim,&claim.candidate_identity)==SEMANTIC_OK);
    candidate=claim.candidate_identity;
    elpis_admission_policy_init_default(p);
    hacf_digest policy_pin;
    CHECK(elpis_admission_policy_identity(p,&policy_pin)==SEMANTIC_OK);
    elpis_typing_bundle_init(bundle);
    bundle->base_snapshot_digest=base->manifest_digest;
    bundle->query_overlay_digest=ov->overlay_identity;
    bundle->retrieval_expansion_digest=retrieval;
    bundle->retrieval_expanded_view_digest=expanded;
    bundle->typer_profile_digest=typer;
    bundle->evidence_span_count=1;bundle->evidence_span_digests[0]=span;
    bundle->claim_candidate_count=1;bundle->claim_candidate_digests[0]=candidate;
    fill(&bundle->typing_bundle_policy_digest,7);
    CHECK(elpis_typing_bundle_identity(bundle,&bundle->typing_bundle_digest)==SEMANTIC_OK);
    CHECK(elpis_typing_bundle_validate(bundle)==SEMANTIC_OK);
    elpis_admission_decision_init(d);
    d->candidate_kind=CANDIDATE_KIND_CLAIM;d->candidate_digest=candidate;
    d->typing_bundle_digest=bundle->typing_bundle_digest;
    d->admission_policy_digest=policy_pin;
    d->validation_stage_reached=VALIDATION_STAGE_COMPLETE;
    d->decision_disposition=DISPOSITION_ADMITTED_NEW_OBJECT;
    d->semantic_object_kind=SEMANTIC_OBJECT_KIND_CLAIM;
    d->semantic_object_digest=n.node_identity;d->effective_authority=1;
    d->source_span_count=1;d->source_span_digests[0]=span;
    d->source_attachment_count=1;d->source_attachment_digests[0]=attachment;
    CHECK(elpis_admission_decision_identity(d,&d->decision_identity)==SEMANTIC_OK);
    elpis_admission_receipt_init(r);
    r->base_snapshot_digest=base->manifest_digest;r->query_overlay_digest=ov->overlay_identity;
    r->retrieval_expansion_digest=retrieval;r->retrieval_expanded_view_digest=expanded;
    r->typing_bundle_digest=bundle->typing_bundle_digest;r->typer_profile_digest=typer;
    r->candidate_digest=candidate;r->admission_policy_digest=policy_pin;
    r->admission_decision_digest=d->decision_identity;r->semantic_object_digest=n.node_identity;
    r->source_span_count=1;r->source_span_digests[0]=span;
    r->retrieval_bundle_count=1;fill(&r->retrieval_bundle_package_digests[0],20);
    r->retrieval_item_attachment_count=1;r->retrieval_item_attachment_digests[0]=attachment;
    CHECK(elpis_admission_receipt_identity(r,&r->receipt_digest)==SEMANTIC_OK);
    elpis_evidence_admission_init(layer);
    layer->base_snapshot_digest=base->manifest_digest;
    layer->query_overlay_digest=ov->overlay_identity;
    layer->retrieval_expansion_digest=retrieval;
    layer->retrieval_expanded_view_digest=expanded;
    layer->typing_bundle_digest=bundle->typing_bundle_digest;
    layer->admission_policy_digest=policy_pin;
    layer->admission_decision_count=1;layer->admission_decision_digests[0]=d->decision_identity;
    layer->admission_receipt_count=1;layer->admission_receipt_digests[0]=r->receipt_digest;
    layer->admitted_claim_count=1;
    CHECK(elpis_evidence_admission_identity(layer,&layer->admission_layer_digest)==SEMANTIC_OK);
    hacf_digest report,sentinel;fill(&sentinel,0xa5);report=sentinel;
#define AUDIT() semantic_b1a_claim_correspondence_audit(base,ov,bundle,p,&policy_pin,layer,d,r,1,&report)
    CHECK(AUDIT()==SEMANTIC_OK);
    CHECK(memcmp(&report,&sentinel,sizeof(report))!=0);
    /* Source attestation is still descriptive, never publication authority. */
    semantic_b1b_claim_source src={&claim,&sp,&att,raw_text,raw_bytes};
    hacf_digest package_pin=att.retrieval_bundle_package_digest;
    hacf_digest source_report, source_sentinel;fill(&source_sentinel,0xa5);
#define SOURCE_AUDIT() semantic_b1b_claim_source_audit(base,ov,bundle,p,&policy_pin,&package_pin,layer,d,r,&src,1,&source_report)
    source_report=source_sentinel;
    CHECK(SOURCE_AUDIT()==SEMANTIC_OK);
    CHECK(memcmp(&source_report,&source_sentinel,sizeof(source_report))!=0);
    hacf_digest source_good=source_report;
    hacf_digest b1a_original=report;
    uint8_t bad_text[sizeof(raw_text)];memcpy(bad_text,raw_text,sizeof(raw_text));
    bad_text[5]^=1;src.item_text=bad_text;source_report=source_sentinel;
    CHECK(SOURCE_AUDIT()!=SEMANTIC_OK);
    CHECK(memcmp(&source_report,&source_sentinel,sizeof(source_report))==0);
    src.item_text=raw_text;
    sp.span_bytes_digest.bytes[0]^=1;
    CHECK(SOURCE_AUDIT()!=SEMANTIC_OK);
    sp.span_bytes_digest.bytes[0]^=1;
    att.text_digest.bytes[0]^=1;
    CHECK(SOURCE_AUDIT()!=SEMANTIC_OK);
    att.text_digest.bytes[0]^=1;
    claim.confidence_key=1;p->minimum_claim_confidence_key=2;
    CHECK(SOURCE_AUDIT()!=SEMANTIC_OK);
    claim.confidence_key=9000;p->minimum_claim_confidence_key=0;
    package_pin.bytes[0]^=1;
    CHECK(SOURCE_AUDIT()!=SEMANTIC_OK);
    package_pin.bytes[0]^=1;
    src.item_text_bytes=0;
    CHECK(SOURCE_AUDIT()!=SEMANTIC_OK);
    src.item_text_bytes=raw_bytes;
    source_report=source_sentinel;
    CHECK(SOURCE_AUDIT()==SEMANTIC_OK);
    CHECK(memcmp(&source_report,&source_good,sizeof(source_report))==0);
    /* B1c: policy-completeness negatives MUST keep all identity records
     * consistent, otherwise a rejection could merely be stale digest data.
     * The predecessor B1a audit is required to ACCEPT each profile while
     * B1c rejects policies whose obligations this narrow witness cannot prove. */
#define REBIND_POLICY() do { \
    CHECK(elpis_admission_policy_identity(p,&policy_pin)==SEMANTIC_OK); \
    d->admission_policy_digest=policy_pin; \
    CHECK(elpis_admission_decision_identity(d,&d->decision_identity)==SEMANTIC_OK); \
    r->admission_policy_digest=policy_pin; \
    r->admission_decision_digest=d->decision_identity; \
    CHECK(elpis_admission_receipt_identity(r,&r->receipt_digest)==SEMANTIC_OK); \
    layer->admission_policy_digest=policy_pin; \
    layer->admission_decision_digests[0]=d->decision_identity; \
    layer->admission_receipt_digests[0]=r->receipt_digest; \
    CHECK(elpis_evidence_admission_identity(layer,&layer->admission_layer_digest)==SEMANTIC_OK); \
} while (0)
    for (unsigned profile=0; profile<7; ++profile) {
        if (profile==0) p->minimum_distinct_source_spans=2;
        if (profile==1) p->minimum_distinct_retrieval_items=2;
        if (profile==2) p->minimum_distinct_documents=2;
        if (profile==3) p->minimum_distinct_bundles=2;
        if (profile==4) p->require_subject_resolution=1;
        if (profile==5) p->require_scope_resolution_when_present=1;
        if (profile==6) p->require_qualifier_resolution_when_present=1;
        REBIND_POLICY();
        CHECK(elpis_admission_policy_validate(p)==SEMANTIC_OK);
        CHECK(AUDIT()==SEMANTIC_OK);
        source_report=source_sentinel;
        CHECK(SOURCE_AUDIT()==SEMANTIC_E_AUTHORITY);
        CHECK(memcmp(&source_report,&source_sentinel,sizeof(source_report))==0);
        if (profile==0) p->minimum_distinct_source_spans=1;
        if (profile==1) p->minimum_distinct_retrieval_items=1;
        if (profile==2) p->minimum_distinct_documents=1;
        if (profile==3) p->minimum_distinct_bundles=1;
        if (profile==4) p->require_subject_resolution=0;
        if (profile==5) p->require_scope_resolution_when_present=0;
        if (profile==6) p->require_qualifier_resolution_when_present=0;
        REBIND_POLICY();
        source_report=source_sentinel;
        CHECK(SOURCE_AUDIT()==SEMANTIC_OK);
        CHECK(memcmp(&source_report,&source_good,sizeof(source_report))==0);
    }
#undef REBIND_POLICY
    /* Refresh B1a after the seven policy profiles; do not reuse a digest
     * produced under an adversarial profile. Require exact baseline replay. */
    CHECK(AUDIT()==SEMANTIC_OK);
    CHECK(memcmp(&report,&b1a_original,sizeof(report))==0);
    hacf_digest good=report;
    report=sentinel;r->semantic_object_digest.bytes[0]^=1;
    CHECK(AUDIT()!=SEMANTIC_OK);
    CHECK(memcmp(&report,&sentinel,sizeof(report))==0);
    r->semantic_object_digest.bytes[0]^=1;
    d->decision_identity.bytes[0]^=1;
    CHECK(AUDIT()!=SEMANTIC_OK);
    d->decision_identity.bytes[0]^=1;
    policy_pin.bytes[0]^=1;CHECK(AUDIT()!=SEMANTIC_OK);policy_pin.bytes[0]^=1;
    ov->overlay_identity.bytes[0]^=1;CHECK(AUDIT()!=SEMANTIC_OK);
    ov->overlay_identity.bytes[0]^=1;
    r->source_span_digests[0].bytes[0]^=1;CHECK(AUDIT()!=SEMANTIC_OK);
    r->source_span_digests[0].bytes[0]^=1;
    CHECK(AUDIT()==SEMANTIC_OK && memcmp(&good,&report,sizeof(report))==0);
    CHECK(semantic_b1a_claim_correspondence_audit(base,ov,bundle,p,&policy_pin,layer,d,r,0,&report)!=SEMANTIC_OK);
    semantic_overlay_destroy(ov); semantic_type_registry_destroy(reg);
    semantic_snapshot_destroy(base);free(bundle);free(layer);free(d);free(r);free(p);
    return 0;
}
int main(void) { int rc=run();puts(rc?"B1B_FAIL":"B1B_SOURCE_AUDIT_ONLY_PASS");return rc; }
