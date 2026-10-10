/* B1b read-only exact-byte verifier. Never call persistence APIs here. */
#include "elpis_semantic/admission_source_b1b.h"
#include "elpis/sha256.h"
#include <string.h>
#include <arpa/inet.h>

static int equal(const hacf_digest *a, const hacf_digest *b) {
    return memcmp(a->bytes, b->bytes, HACF_DIGEST_BYTES) == 0;
}
static int nonzero(const hacf_digest *d) {
    static const uint8_t zero[HACF_DIGEST_BYTES] = {0};
    return d && memcmp(d->bytes, zero, sizeof(zero)) != 0;
}
int semantic_b1b_claim_source_audit(
    const semantic_snapshot_manifest *base,
    const semantic_query_overlay *overlay,
    const elpis_evidence_typing_bundle_v1 *bundle,
    const elpis_evidence_admission_policy_v1 *policy,
    const hacf_digest *expected_policy_pin,
    const hacf_digest *expected_retrieval_bundle_package_pin,
    const elpis_evidence_admission_v1 *layer,
    const elpis_evidence_admission_decision_v1 *decisions,
    const elpis_evidence_admission_receipt_v1 *receipts,
    const semantic_b1b_claim_source *sources,
    size_t count,
    hacf_digest *report_out) {
    if (!base || !overlay || !bundle || !policy || !expected_policy_pin ||
        !expected_retrieval_bundle_package_pin || !layer || !decisions ||
        !receipts || !sources || !report_out || count == 0 ||
        count > SEMANTIC_B1A_MAX_CLAIMS ||
        !nonzero(expected_policy_pin) ||
        !nonzero(expected_retrieval_bundle_package_pin)) return SEMANTIC_E_INVAL;
    /* The upstream B1a audit is necessary but is not an admission permit. */
    hacf_digest b1a;
    if (semantic_b1a_claim_correspondence_audit(base, overlay, bundle,
            policy, expected_policy_pin, layer, decisions, receipts,
            count, &b1a) != SEMANTIC_OK) return SEMANTIC_E_AUTHORITY;
    if (policy->require_exact_span_validation != 1 ||
        policy->allow_primary_items != 1 ||
        (policy->admission_limit && count > policy->admission_limit) ||
        bundle->claim_candidate_count != count ||
        bundle->evidence_span_count != count ||
        bundle->relation_candidate_count != 0) return SEMANTIC_E_AUTHORITY;
    /* B1c: this audit witnesses exactly one PRIMARY item/span per claim.
     * A policy requiring corroboration or resolution outside that witness
     * cannot be satisfied here. Refuse instead of treating absent proof as a
     * successful policy check. Later profiles must supply real witnesses. */
    if (policy->minimum_distinct_source_spans > 1 ||
        policy->minimum_distinct_retrieval_items > 1 ||
        policy->minimum_distinct_documents > 1 ||
        policy->minimum_distinct_bundles > 1 ||
        policy->require_subject_resolution != 0 ||
        policy->require_scope_resolution_when_present != 0 ||
        policy->require_qualifier_resolution_when_present != 0)
        return SEMANTIC_E_AUTHORITY;
    if (!elpis_policy_allows_typer(policy, &bundle->typer_profile_digest))
        return SEMANTIC_E_AUTHORITY;
    elpis_sha256_ctx ctx;
    elpis_sha256_init(&ctx);
    static const char domain[] = "elpis.semantic.b1b.claim-source-audit.v1";
    elpis_sha256_update(&ctx, domain, sizeof(domain)-1);
    elpis_sha256_update(&ctx, b1a.bytes, HACF_DIGEST_BYTES);
    uint32_t be = htonl((uint32_t)count);
    elpis_sha256_update(&ctx, &be, sizeof(be));
    for (size_t i = 0; i < count; ++i) {
        const semantic_b1b_claim_source *s = &sources[i];
        const elpis_evidence_admission_decision_v1 *d = &decisions[i];
        const elpis_evidence_admission_receipt_v1 *r = &receipts[i];
        if (!s->candidate || !s->span || !s->attachment || !s->item_text ||
            s->item_text_bytes == 0 ||
            s->item_text_bytes > SEMANTIC_B1B_MAX_RAW_ITEM_BYTES ||
            d->source_span_count != 1 || d->source_attachment_count != 1 ||
            r->source_span_count != 1 || r->retrieval_item_attachment_count != 1 ||
            r->retrieval_bundle_count != 1 ||
            s->candidate->source_span_count != 1)
            return SEMANTIC_E_AUTHORITY;
        const elpis_evidence_claim_candidate_v1 *c = s->candidate;
        const elpis_evidence_span_v1 *sp = s->span;
        const elpis_retrieval_item_attachment_v1 *att = s->attachment;
        hacf_digest cid, sid, aid, full_text_digest;
        if (elpis_claim_candidate_validate(c) != SEMANTIC_OK ||
            elpis_claim_candidate_identity(c, &cid) != SEMANTIC_OK ||
            !equal(&cid, &c->candidate_identity) ||
            !equal(&cid, &d->candidate_digest) ||
            !equal(&cid, &bundle->claim_candidate_digests[i]) ||
            !equal(&c->typer_profile_digest, &bundle->typer_profile_digest) ||
            c->confidence_key < policy->minimum_claim_confidence_key ||
            !elpis_policy_allows_claim_type(policy, c->claim_type))
            return SEMANTIC_E_AUTHORITY;
        if (elpis_attachment_validate(att) != SEMANTIC_OK ||
            elpis_attachment_digest(att, &aid) != SEMANTIC_OK ||
            !equal(&aid, &att->attachment_digest) ||
            att->item_kind != 1 || att->graph_hop != 0 ||
            att->graph_edge_provenance_status != GRAPH_PROVENANCE_NOT_APPLICABLE ||
            att->item_authority < policy->minimum_source_authority ||
            att->item_authority < d->effective_authority ||
            !equal(&att->retrieval_bundle_package_digest,
                   expected_retrieval_bundle_package_pin) ||
            !equal(&att->retrieval_bundle_package_digest,
                   &r->retrieval_bundle_package_digests[0]) ||
            !equal(&aid, &d->source_attachment_digests[0]) ||
            !equal(&aid, &r->retrieval_item_attachment_digests[0]))
            return SEMANTIC_E_AUTHORITY;
        elpis_sha256(s->item_text, s->item_text_bytes,
                     full_text_digest.bytes);
        if (!equal(&full_text_digest, &att->text_digest) ||
            !equal(&full_text_digest, &sp->item_text_digest) ||
            elpis_evidence_span_validate(sp, s->item_text,
                   (uint32_t)s->item_text_bytes) != SEMANTIC_OK ||
            elpis_evidence_span_identity(sp, &sid) != SEMANTIC_OK ||
            !equal(&sid, &sp->span_identity) ||
            !equal(&sid, &c->source_span_digests[0]) ||
            !equal(&sid, &bundle->evidence_span_digests[i]) ||
            !equal(&sid, &d->source_span_digests[0]) ||
            !equal(&sid, &r->source_span_digests[0]) ||
            !equal(&sp->retrieval_expansion_digest,
                   &bundle->retrieval_expansion_digest) ||
            !equal(&sp->retrieval_item_attachment_digest, &aid) ||
            !equal(&sp->retrieval_bundle_digest, &att->retrieval_bundle_digest) ||
            !equal(&sp->retrieval_bundle_package_digest,
                   &att->retrieval_bundle_package_digest) ||
            !equal(&sp->evidence_node_digest, &att->evidence_node_digest) ||
            !equal(&sp->chunk_digest, &att->chunk_digest) ||
            sp->span_flags != EVIDENCE_SPAN_FLAG_PRIMARY)
            return SEMANTIC_E_AUTHORITY;
        /* Prevent exact-reuse of a source record between different claims. */
        for (size_t j = 0; j < i; ++j) {
            if (equal(&sid, &sources[j].span->span_identity) ||
                equal(&aid, &sources[j].attachment->attachment_digest))
                return SEMANTIC_E_DUPLICATE;
        }
        elpis_sha256_update(&ctx, cid.bytes, HACF_DIGEST_BYTES);
        elpis_sha256_update(&ctx, sid.bytes, HACF_DIGEST_BYTES);
        elpis_sha256_update(&ctx, aid.bytes, HACF_DIGEST_BYTES);
        elpis_sha256_update(&ctx, full_text_digest.bytes, HACF_DIGEST_BYTES);
    }
    hacf_digest result;
    elpis_sha256_final(&ctx, result.bytes);
    *report_out = result;
    return SEMANTIC_OK;
}
