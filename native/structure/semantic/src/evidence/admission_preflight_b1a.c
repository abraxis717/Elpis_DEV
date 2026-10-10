/* B1a read-only audit. No publication API called and no write capability held. */
#include "elpis_semantic/admission_preflight_b1a.h"
#include "elpis/sha256.h"
#include <string.h>
#include <arpa/inet.h>

static int eq(const hacf_digest *a, const hacf_digest *b) {
    return memcmp(a->bytes, b->bytes, HACF_DIGEST_BYTES) == 0;
}
static int present(const hacf_digest *d) {
    static const uint8_t zero[HACF_DIGEST_BYTES] = {0};
    return d && memcmp(d->bytes, zero, HACF_DIGEST_BYTES) != 0;
}
static int digest_in(const hacf_digest *needle, const hacf_digest *list, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) if (eq(needle, &list[i])) return 1;
    return 0;
}
static int record_has_span(const elpis_evidence_admission_receipt_v1 *receipt,
                           const hacf_digest *span) {
    return digest_in(span, receipt->source_span_digests, receipt->source_span_count);
}

int semantic_b1a_claim_correspondence_audit(
    const semantic_snapshot_manifest *base,
    const semantic_query_overlay *overlay,
    const elpis_evidence_typing_bundle_v1 *bundle,
    const elpis_evidence_admission_policy_v1 *policy,
    const hacf_digest *independent_policy_pin,
    const elpis_evidence_admission_v1 *layer,
    const elpis_evidence_admission_decision_v1 *decisions,
    const elpis_evidence_admission_receipt_v1 *receipts,
    size_t count, hacf_digest *audit_digest_out) {
    if (!base || !overlay || !bundle || !policy || !independent_policy_pin ||
        !layer || !decisions || !receipts || !audit_digest_out ||
        count == 0 || count > SEMANTIC_B1A_MAX_CLAIMS ||
        !present(independent_policy_pin)) return SEMANTIC_E_INVAL;
    if (semantic_snapshot_validate(base) != SEMANTIC_OK ||
        elpis_admission_policy_validate(policy) != SEMANTIC_OK ||
        elpis_typing_bundle_validate(bundle) != SEMANTIC_OK ||
        elpis_evidence_admission_validate(layer) != SEMANTIC_OK)
        return SEMANTIC_E_AUTHORITY;
    if (overlay->abi_version != SEMANTIC_OVERLAY_ABI_VERSION ||
        !overlay->local_builder || overlay->external_dependency_count > SEMANTIC_MAX_EXTERNAL_DEPS)
        return SEMANTIC_E_AUTHORITY;
    /* Recompute overlay without changing the caller's identity or builder. */
    semantic_query_overlay check_overlay = *overlay;
    if (semantic_overlay_finalize(&check_overlay) != SEMANTIC_OK ||
        !eq(&check_overlay.overlay_identity, &overlay->overlay_identity) ||
        !eq(&check_overlay.query_local_segment_digest, &overlay->query_local_segment_digest))
        return SEMANTIC_E_AUTHORITY;

    hacf_digest policy_id, bundle_id, layer_id;
    if (elpis_admission_policy_identity(policy, &policy_id) != SEMANTIC_OK ||
        !eq(&policy_id, independent_policy_pin) ||
        elpis_typing_bundle_identity(bundle, &bundle_id) != SEMANTIC_OK ||
        !eq(&bundle_id, &bundle->typing_bundle_digest) ||
        elpis_evidence_admission_identity(layer, &layer_id) != SEMANTIC_OK ||
        !eq(&layer_id, &layer->admission_layer_digest)) return SEMANTIC_E_AUTHORITY;
    if (!eq(&base->manifest_digest, &overlay->base_snapshot_manifest_digest) ||
        !eq(&base->hacf_graph_snapshot_digest, &overlay->base_hacf_graph_snapshot_digest) ||
        !eq(&bundle->base_snapshot_digest, &base->manifest_digest) ||
        !eq(&bundle->query_overlay_digest, &overlay->overlay_identity) ||
        !eq(&layer->base_snapshot_digest, &base->manifest_digest) ||
        !eq(&layer->query_overlay_digest, &overlay->overlay_identity) ||
        !eq(&layer->retrieval_expansion_digest, &bundle->retrieval_expansion_digest) ||
        !eq(&layer->retrieval_expanded_view_digest, &bundle->retrieval_expanded_view_digest) ||
        !eq(&layer->typing_bundle_digest, &bundle->typing_bundle_digest) ||
        !eq(&layer->admission_policy_digest, independent_policy_pin))
        return SEMANTIC_E_AUTHORITY;

    if (layer->admission_decision_count != count || layer->admission_receipt_count != count ||
        layer->admitted_claim_count != count || layer->admitted_relation_count ||
        layer->rejected_claim_count || layer->rejected_relation_count ||
        semantic_builder_node_count(overlay->local_builder) != count ||
        semantic_builder_assertion_count(overlay->local_builder) != count ||
        semantic_builder_hyperedge_count(overlay->local_builder) != 0 ||
        semantic_builder_incidence_count(overlay->local_builder) != 0)
        return SEMANTIC_E_AUTHORITY;

    for (size_t i = 0; i < count; ++i) {
        const elpis_evidence_admission_decision_v1 *d = &decisions[i];
        const elpis_evidence_admission_receipt_v1 *r = &receipts[i];
        hacf_digest did, rid;
        if (elpis_admission_decision_validate(d) != SEMANTIC_OK ||
            elpis_admission_receipt_validate(r) != SEMANTIC_OK ||
            d->candidate_kind != CANDIDATE_KIND_CLAIM ||
            d->decision_disposition != DISPOSITION_ADMITTED_NEW_OBJECT ||
            d->validation_stage_reached != VALIDATION_STAGE_COMPLETE ||
            d->semantic_object_kind != SEMANTIC_OBJECT_KIND_CLAIM ||
            d->effective_authority < 1 || d->effective_authority > 3 ||
            d->effective_authority > policy->maximum_claim_authority ||
            d->source_span_count == 0 || d->source_attachment_count == 0 ||
            r->source_span_count != d->source_span_count ||
            r->retrieval_item_attachment_count != d->source_attachment_count ||
            r->retrieval_bundle_count == 0 ||
            !present(&d->candidate_digest) || !present(&d->semantic_object_digest) ||
            !digest_in(&d->candidate_digest, bundle->claim_candidate_digests, bundle->claim_candidate_count))
            return SEMANTIC_E_AUTHORITY;
        for (uint32_t k = 0; k < d->source_span_count; ++k) {
            if (!present(&d->source_span_digests[k]) ||
                !eq(&d->source_span_digests[k], &r->source_span_digests[k]) ||
                !digest_in(&d->source_span_digests[k], bundle->evidence_span_digests, bundle->evidence_span_count))
                return SEMANTIC_E_AUTHORITY;
        }
        for (uint32_t k = 0; k < d->source_attachment_count; ++k)
            if (!present(&d->source_attachment_digests[k]) ||
                !eq(&d->source_attachment_digests[k], &r->retrieval_item_attachment_digests[k]))
                return SEMANTIC_E_AUTHORITY;
        for (uint32_t k = 0; k < r->retrieval_bundle_count; ++k)
            if (!present(&r->retrieval_bundle_package_digests[k])) return SEMANTIC_E_AUTHORITY;
        if (elpis_admission_decision_identity(d, &did) != SEMANTIC_OK ||
            !eq(&did, &d->decision_identity) ||
            !eq(&did, &layer->admission_decision_digests[i]) ||
            elpis_admission_receipt_identity(r, &rid) != SEMANTIC_OK ||
            !eq(&rid, &r->receipt_digest) ||
            !eq(&rid, &layer->admission_receipt_digests[i]) ||
            !eq(&r->admission_decision_digest, &did) ||
            !eq(&r->base_snapshot_digest, &base->manifest_digest) ||
            !eq(&r->query_overlay_digest, &overlay->overlay_identity) ||
            !eq(&r->retrieval_expansion_digest, &bundle->retrieval_expansion_digest) ||
            !eq(&r->retrieval_expanded_view_digest, &bundle->retrieval_expanded_view_digest) ||
            !eq(&r->typing_bundle_digest, &bundle->typing_bundle_digest) ||
            !eq(&r->typer_profile_digest, &bundle->typer_profile_digest) ||
            !eq(&r->candidate_digest, &d->candidate_digest) ||
            !eq(&r->admission_policy_digest, independent_policy_pin) ||
            !eq(&r->semantic_object_digest, &d->semantic_object_digest) ||
            r->graph_edge_provenance_status != GRAPH_PROVENANCE_UNAVAILABLE)
            return SEMANTIC_E_AUTHORITY;
        for (size_t j = 0; j < i; ++j)
            if (eq(&d->candidate_digest, &decisions[j].candidate_digest) ||
                eq(&d->semantic_object_digest, &decisions[j].semantic_object_digest))
                return SEMANTIC_E_DUPLICATE;

        /* Exactly one asserted overlay node; no unadmitted surplus allowed. */
        const elpis_semantic_node_v1 *n = NULL;
        const elpis_semantic_assertion_v1 *a = NULL;
        for (uint32_t k = 0; k < count; ++k) {
            const elpis_semantic_node_v1 *candidate = semantic_builder_get_node(overlay->local_builder, k);
            const elpis_semantic_assertion_v1 *assertion = semantic_builder_get_assertion(overlay->local_builder, k);
            if (candidate && eq(&candidate->node_identity, &d->semantic_object_digest)) n = candidate;
            if (assertion && eq(&assertion->asserted_object_digest, &d->semantic_object_digest)) a = assertion;
        }
        hacf_digest nid, aid;
        if (!n || !a ||
            elpis_semantic_node_validate(n) != SEMANTIC_OK ||
            elpis_semantic_node_identity(n, &nid) != SEMANTIC_OK ||
            !eq(&nid, &n->node_identity) ||
            elpis_semantic_assertion_validate(a) != SEMANTIC_OK ||
            elpis_semantic_assertion_identity(a, &aid) != SEMANTIC_OK ||
            !eq(&aid, &a->assertion_identity) ||
            a->asserted_object_kind != SEMANTIC_OBJECT_KIND_NODE ||
            a->authority != d->effective_authority ||
            !record_has_span(r, &a->provenance_digest)) return SEMANTIC_E_AUTHORITY;
    }
    /* Audit digest is descriptive only. It is not a signature/capability. */
    elpis_sha256_ctx ctx;
    elpis_sha256_init(&ctx);
    const char domain[] = "elpis.semantic.b1a.correspondence-audit.v1";
    elpis_sha256_update(&ctx, domain, sizeof(domain) - 1);
    elpis_sha256_update(&ctx, base->manifest_digest.bytes, HACF_DIGEST_BYTES);
    elpis_sha256_update(&ctx, overlay->overlay_identity.bytes, HACF_DIGEST_BYTES);
    elpis_sha256_update(&ctx, layer->admission_layer_digest.bytes, HACF_DIGEST_BYTES);
    uint32_t be = htonl((uint32_t)count);
    elpis_sha256_update(&ctx, &be, 4);
    elpis_sha256_final(&ctx, audit_digest_out->bytes);
    return SEMANTIC_OK;
}
