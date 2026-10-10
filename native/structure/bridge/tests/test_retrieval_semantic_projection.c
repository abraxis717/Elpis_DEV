/* Track C v1 through the retrieval bridge: explicit admitted claim -> claim relation proofs (exact ABI v1 record
 * images) projected against a volatile epoch's own corpus into context edges, then an epoch over the same
 * documents built with those edges. All-or-nothing refusal leaves the caller's edges untouched.
 *
 *   test_retrieval_semantic_projection                 run the adversarial suite
 *   test_retrieval_semantic_projection --emit PATH     write two valid proofs (SUPPORTS, CONTRADICTS) over
 *                                                      DOCS for the Python adapter test (test-only fixture;
 *                                                      its numbers mean nothing beyond the admission law)
 */
#include "retrieval_bridge.h"
#include "elpis/corpus.h"
#include "elpis/sha256.h"
#include "elpis_semantic/identity.h"
#include "elpis_semantic/semantic_context_projection.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL_RETRIEVAL_SEMANTIC_PROJECTION line=%d %s\n", __LINE__, #c); exit(1); } } while (0)

/* Must equal DOCS in tests/structure/test_semantic_context_adapter.py. */
static const char *LABELS[] = {"alpha", "beta"};
static const char *TEXTS[] = {"Alpha original evidence chunk.", "Beta target evidence chunk."};
static const char *NAMESPACES[] = {"elpis.docs", "elpis.docs"};
static const char *AUTHORITIES[] = {"reference", "reference"};

typedef struct fixture {
    elpis_corpus *corpus;
    elpis_evidence_admission_v1 layer;
    elpis_evidence_admission_decision_v1 decisions[3];
    elpis_evidence_admission_receipt_v1 receipts[3];
    elpis_evidence_relation_candidate_v1 relation;
    elpis_evidence_span_v1 spans[2];
    elpis_retrieval_item_attachment_v1 attachments[2];
    char texts[2][128];
} fixture;

static void hash(const void *p, size_t n, hacf_digest *out) { elpis_sha256(p, n, out->bytes); }
static hacf_digest h(const char *s) { hacf_digest d; hash(s, strlen(s), &d); return d; }

static void endpoint(fixture *f, unsigned i) {
    elpis_chunk_ref c = {0};
    uint32_t count = 0;
    CHECK(elpis_corpus_list_chunks(f->corpus, NULL, NULL, i, 1, &c, &count) == 0 && count == 1);
    char *txt = NULL;
    CHECK(elpis_corpus_chunk_text(f->corpus, c.chunk_digest, &txt) == 0 && txt);
    size_t len = strlen(txt);
    CHECK(len > 0 && len < sizeof f->texts[i]);
    memcpy(f->texts[i], txt, len + 1);
    elpis_free(txt);
    elpis_retrieval_item_attachment_v1 *a = &f->attachments[i];
    elpis_attachment_init(a);
    a->evidence_node_digest = h(i ? "node-b" : "node-a");
    a->retrieval_bundle_digest = h("bundle");
    a->retrieval_bundle_package_digest = h("bundle-package");
    a->retrieval_requirement_digest = h("requirement");
    CHECK(hacf_digest_from_hex(c.chunk_digest, &a->chunk_digest) == 0);
    CHECK(hacf_digest_from_hex(c.doc_digest, &a->document_digest) == 0);
    hash(f->texts[i], len, &a->text_digest);
    a->namespace_digest = h("namespace");
    a->item_authority = i == 0 ? 2u : 1u;
    a->item_kind = 1;
    a->source_mask = 1;
    a->final_rank = i + 1;
    a->graph_edge_provenance_status = GRAPH_PROVENANCE_NOT_APPLICABLE;
    CHECK(elpis_attachment_digest(a, &a->attachment_digest) == SEMANTIC_OK);
    elpis_evidence_span_v1 *s = &f->spans[i];
    elpis_evidence_span_init(s);
    s->retrieval_expansion_digest = f->layer.retrieval_expansion_digest;
    s->retrieval_bundle_digest = a->retrieval_bundle_digest;
    s->retrieval_bundle_package_digest = a->retrieval_bundle_package_digest;
    s->retrieval_item_attachment_digest = a->attachment_digest;
    s->evidence_node_digest = a->evidence_node_digest;
    s->chunk_digest = a->chunk_digest;
    s->item_text_digest = a->text_digest;
    s->byte_end_exclusive = (uint32_t)len;
    hash(f->texts[i], len, &s->span_bytes_digest);
    s->span_flags = EVIDENCE_SPAN_FLAG_PRIMARY;
    CHECK(elpis_evidence_span_identity(s, &s->span_identity) == SEMANTIC_OK);
}

static void decision(fixture *f, unsigned i, int relation) {
    elpis_evidence_admission_decision_v1 *d = &f->decisions[i];
    elpis_admission_decision_init(d);
    d->candidate_kind = relation ? CANDIDATE_KIND_RELATION : CANDIDATE_KIND_CLAIM;
    d->candidate_digest = relation ? f->relation.candidate_identity : h(i ? "cand-b" : "cand-a");
    d->typing_bundle_digest = f->layer.typing_bundle_digest;
    d->admission_policy_digest = f->layer.admission_policy_digest;
    d->validation_stage_reached = VALIDATION_STAGE_COMPLETE;
    d->decision_disposition = DISPOSITION_ADMITTED_NEW_OBJECT;
    d->semantic_object_kind = relation ? SEMANTIC_OBJECT_KIND_RELATION : SEMANTIC_OBJECT_KIND_CLAIM;
    d->semantic_object_digest = h(relation ? "relation-object" : i ? "claim-b" : "claim-a");
    d->effective_authority = relation ? 2u : i ? 1u : 2u;
    unsigned which = relation ? 0 : i;
    d->source_span_count = 1;
    d->source_span_digests[0] = f->spans[which].span_identity;
    d->source_attachment_count = 1;
    d->source_attachment_digests[0] = f->attachments[which].attachment_digest;
    d->decision_diagnostic_digest = h("diagnostic");
    CHECK(elpis_admission_decision_identity(d, &d->decision_identity) == SEMANTIC_OK);
    elpis_evidence_admission_receipt_v1 *r = &f->receipts[i];
    elpis_admission_receipt_init(r);
    r->base_snapshot_digest = f->layer.base_snapshot_digest;
    r->query_overlay_digest = f->layer.query_overlay_digest;
    r->retrieval_expansion_digest = f->layer.retrieval_expansion_digest;
    r->retrieval_expanded_view_digest = h("expanded");
    r->typing_bundle_digest = f->layer.typing_bundle_digest;
    r->typer_profile_digest = h("typer");
    r->candidate_digest = d->candidate_digest;
    r->admission_policy_digest = f->layer.admission_policy_digest;
    r->admission_decision_digest = d->decision_identity;
    r->semantic_object_digest = d->semantic_object_digest;
    r->source_span_count = 1;
    r->source_span_digests[0] = d->source_span_digests[0];
    r->retrieval_bundle_count = 1;
    r->retrieval_bundle_package_digests[0] = f->attachments[which].retrieval_bundle_package_digest;
    r->retrieval_item_attachment_count = 1;
    r->retrieval_item_attachment_digests[0] = d->source_attachment_digests[0];
    r->graph_edge_provenance_status = GRAPH_PROVENANCE_NOT_APPLICABLE;
    r->HACF_package_digest = f->layer.HACF_package_digest;
    CHECK(elpis_admission_receipt_identity(r, &r->receipt_digest) == SEMANTIC_OK);
}

static void make_fixture(fixture *f, elpis_corpus *corpus, evidence_relation_type type) {
    memset(f, 0, sizeof *f);
    f->corpus = corpus;
    elpis_evidence_admission_init(&f->layer);
    f->layer.base_snapshot_digest = h("snapshot");
    f->layer.query_overlay_digest = h("overlay");
    f->layer.retrieval_expansion_digest = h("expansion");
    f->layer.retrieval_expanded_view_digest = h("expanded");
    f->layer.typing_bundle_digest = h("typing");
    f->layer.admission_policy_digest = h("policy");
    f->layer.admission_segment_digest = h("admission-segment");
    f->layer.HACF_package_digest = h("hacf-package");
    endpoint(f, 0);
    endpoint(f, 1);
    decision(f, 0, 0);
    decision(f, 1, 0);
    elpis_relation_candidate_init(&f->relation);
    f->relation.typer_profile_digest = h("typer");
    f->relation.relation_type = type;
    f->relation.evidence_claim_candidate_digest = f->decisions[0].candidate_digest;
    f->relation.evidence_object_kind = OBJECT_KIND_CLAIM_NODE;
    f->relation.evidence_object_digest = f->decisions[0].semantic_object_digest;
    f->relation.target_object_kind = OBJECT_KIND_CLAIM_NODE;
    f->relation.target_object_digest = f->decisions[1].semantic_object_digest;
    f->relation.evidence_role = RELATION_ROLE_EVIDENCE;
    f->relation.target_role = RELATION_ROLE_TARGET;
    f->relation.relation_polarity = RELATION_POLARITY_AFFIRMATIVE;
    f->relation.source_span_count = 1;
    f->relation.source_span_digests[0] = f->spans[0].span_identity;
    f->relation.confidence_key = 100;
    CHECK(elpis_relation_candidate_identity(&f->relation, &f->relation.candidate_identity) == SEMANTIC_OK);
    decision(f, 2, 1);
    for (unsigned i = 0; i < 3; i++) {
        f->layer.admission_decision_digests[i] = f->decisions[i].decision_identity;
        f->layer.admission_receipt_digests[i] = f->receipts[i].receipt_digest;
    }
    f->layer.admission_decision_count = f->layer.admission_receipt_count = 3;
    f->layer.admitted_claim_count = 2;
    f->layer.admitted_relation_count = 1;
    CHECK(elpis_evidence_admission_identity(&f->layer, &f->layer.admission_layer_digest) == SEMANTIC_OK);
}

static elpis_retrieval_blob blob(const void *p, size_t n) { return (elpis_retrieval_blob){p, n}; }

static elpis_retrieval_semantic_proof_v1 images(const fixture *f) {
    elpis_retrieval_semantic_proof_v1 p;
    memset(&p, 0, sizeof p);
    p.admission_layer = blob(&f->layer, sizeof f->layer);
    p.trusted_base_snapshot_digest = blob(&f->layer.base_snapshot_digest, sizeof f->layer.base_snapshot_digest);
    p.relation = blob(&f->relation, sizeof f->relation);
    p.relation_decision = blob(&f->decisions[2], sizeof f->decisions[2]);
    p.relation_receipt = blob(&f->receipts[2], sizeof f->receipts[2]);
    for (unsigned i = 0; i < 2; i++) {
        elpis_retrieval_semantic_endpoint_v1 *e = i ? &p.target : &p.source;
        e->decision = blob(&f->decisions[i], sizeof f->decisions[i]);
        e->receipt = blob(&f->receipts[i], sizeof f->receipts[i]);
        e->span = blob(&f->spans[i], sizeof f->spans[i]);
        e->attachment = blob(&f->attachments[i], sizeof f->attachments[i]);
        e->item_text = blob(f->texts[i], strlen(f->texts[i]));
    }
    return p;
}

static elpis_retrieval_env_t *epoch(int with_graph, const elpis_context_edge_input *edges, uint32_t n) {
    char error[256] = {0};
    elpis_retrieval_env_t *env = elpis_retrieval_env_create("/nonexistent/volatile", LABELS, TEXTS, NAMESPACES,
                                                            AUTHORITIES, 2, with_graph, edges, n, error);
    if (!env) fprintf(stderr, "epoch error: %s\n", error);
    CHECK(env != NULL);
    return env;
}

static void refused(elpis_retrieval_env_t *env, const elpis_retrieval_semantic_proof_v1 *p, uint32_t n) {
    elpis_context_edge_input out[4];
    memset(out, 0xA5, sizeof out);
    char error[256] = {0};
    CHECK(elpis_retrieval_env_project_semantic_edges(env, p, n, out, error) != 0 && error[0]);
    const uint8_t *b = (const uint8_t *)out;
    for (size_t i = 0; i < sizeof out; i++) CHECK(b[i] == 0xA5);
}

static void put_blob(FILE *o, elpis_retrieval_blob b) {
    uint32_t n = (uint32_t)b.len;
    CHECK(fwrite(&n, sizeof n, 1, o) == 1 && fwrite(b.bytes, 1, b.len, o) == b.len);
}

static int emit(const char *path) {
    elpis_retrieval_env_t *env = epoch(0, NULL, 0);
    elpis_corpus *corpus = (elpis_corpus *)elpis_retrieval_env_borrow_corpus(env);
    static fixture f[2];
    make_fixture(&f[0], corpus, RELATION_TYPE_SUPPORTS);
    make_fixture(&f[1], corpus, RELATION_TYPE_CONTRADICTS);
    FILE *o = fopen(path, "wb");
    CHECK(o != NULL);
    uint32_t count = 2;
    CHECK(fwrite("ELPTC1\0\0", 1, 8, o) == 8 && fwrite(&count, sizeof count, 1, o) == 1);
    for (unsigned k = 0; k < 2; k++) {
        elpis_retrieval_semantic_proof_v1 p = images(&f[k]);
        put_blob(o, p.admission_layer); put_blob(o, p.trusted_base_snapshot_digest); put_blob(o, p.relation);
        put_blob(o, p.relation_decision); put_blob(o, p.relation_receipt);
        const elpis_retrieval_semantic_endpoint_v1 *ends[2] = {&p.source, &p.target};
        for (unsigned i = 0; i < 2; i++) {
            put_blob(o, ends[i]->decision); put_blob(o, ends[i]->receipt); put_blob(o, ends[i]->span);
            put_blob(o, ends[i]->attachment); put_blob(o, ends[i]->item_text);
        }
    }
    CHECK(fclose(o) == 0);
    elpis_retrieval_env_destroy(env);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "--emit") == 0) return emit(argv[2]);
    CHECK(argc == 1 || argc == 2);
    CHECK(elpis_retrieval_semantic_record_bytes(ELPIS_RETRIEVAL_RECORD_ADMISSION_LAYER) ==
          sizeof(elpis_evidence_admission_v1));
    CHECK(elpis_retrieval_semantic_record_bytes(ELPIS_RETRIEVAL_RECORD_RELATION) ==
          sizeof(elpis_evidence_relation_candidate_v1));
    CHECK(elpis_retrieval_semantic_record_bytes(0) == 0 && elpis_retrieval_semantic_record_bytes(7) == 0);

    elpis_retrieval_env_t *env = epoch(0, NULL, 0);
    elpis_corpus *corpus = (elpis_corpus *)elpis_retrieval_env_borrow_corpus(env);
    static fixture f[2];
    make_fixture(&f[0], corpus, RELATION_TYPE_SUPPORTS);
    make_fixture(&f[1], corpus, RELATION_TYPE_CONTRADICTS);
    elpis_retrieval_semantic_proof_v1 proofs[2] = {images(&f[0]), images(&f[1])};

    /* Positive: two verified edges, deterministic, between the two distinct chunks. */
    elpis_context_edge_input edges[2], again[2];
    char error[256] = {0};
    CHECK(elpis_retrieval_env_project_semantic_edges(env, proofs, 2, edges, error) == 0);
    CHECK(elpis_retrieval_env_project_semantic_edges(env, proofs, 2, again, error) == 0);
    CHECK(memcmp(edges, again, sizeof edges) == 0);
    CHECK(edges[0].edge_type == ELPIS_CGRAPH_SEMANTIC_SUPPORTS && edges[1].edge_type == ELPIS_CGRAPH_SEMANTIC_CONTRADICTS);
    CHECK(edges[0].authority == 1 && strcmp(edges[0].subject_chunk_digest, edges[0].object_chunk_digest) != 0);
    CHECK(elpis_retrieval_env_project_semantic_edges(env, proofs, 0, again, error) == 0);

    /* The edges build an immutable graph in a new epoch over the same documents. */
    elpis_retrieval_env_t *graph = epoch(1, edges, 2);
    CHECK(elpis_retrieval_env_graph_edge_count(graph) == 2);
    CHECK(strcmp(elpis_retrieval_env_graph_digest(graph),
                 "0000000000000000000000000000000000000000000000000000000000000000") != 0);
    elpis_retrieval_env_destroy(graph);

    /* All-or-nothing refusals; the caller's edges stay untouched. */
    elpis_retrieval_semantic_proof_v1 bad = proofs[1];
    bad.relation.len -= 1;                                   /* not an exact record image */
    refused(env, (elpis_retrieval_semantic_proof_v1[]){proofs[0], bad}, 2);
    bad = proofs[1];
    bad.target.item_text.len = 0;
    refused(env, &bad, 1);
    f[1].decisions[2].decision_disposition = DISPOSITION_REJECTED_POLICY;   /* not admitted */
    refused(env, (elpis_retrieval_semantic_proof_v1[]){proofs[0], proofs[1]}, 2);
    f[1].decisions[2].decision_disposition = DISPOSITION_ADMITTED_NEW_OBJECT;
    f[1].layer.base_snapshot_digest.bytes[0] ^= 1;          /* layer no longer matches the trusted snapshot */
    refused(env, &proofs[1], 1);
    f[1].layer.base_snapshot_digest.bytes[0] ^= 1;
    char corrupt[128];
    memcpy(corrupt, f[0].texts[1], strlen(f[0].texts[1]) + 1);
    corrupt[0] ^= 1;
    bad = proofs[0];
    bad.target.item_text.bytes = corrupt;                    /* bytes the corpus does not own */
    refused(env, &bad, 1);
    CHECK(elpis_retrieval_env_project_semantic_edges(env, proofs, ELPIS_CGRAPH_MAX_EDGES + 1u, edges, error) != 0);
    CHECK(elpis_retrieval_env_project_semantic_edges(NULL, proofs, 1, edges, error) != 0);

    /* A proof bound to another epoch's chunks is refused here. */
    const char *other_texts[] = {"Gamma unrelated chunk.", "Delta unrelated chunk."};
    char oerr[256] = {0};
    elpis_retrieval_env_t *other = elpis_retrieval_env_create("/nonexistent/volatile", LABELS, other_texts,
                                                              NAMESPACES, AUTHORITIES, 2, 0, NULL, 0, oerr);
    CHECK(other != NULL);
    refused(other, &proofs[0], 1);
    elpis_retrieval_env_destroy(other);

    CHECK(elpis_retrieval_env_project_semantic_edges(env, proofs, 2, again, error) == 0);
    CHECK(memcmp(edges, again, sizeof edges) == 0);
    elpis_retrieval_env_destroy(env);
    puts("PASS_RETRIEVAL_SEMANTIC_PROJECTION_V1");
    return 0;
}
