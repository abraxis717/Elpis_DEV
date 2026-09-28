#include "regex_hacf_query_ingress.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static void req(bool ok, const char *msg) {
    if (!ok) {
        std::cerr << "FAIL " << msg << "\n";
        std::exit(1);
    }
}

struct CorpusGuard {
    elpis_corpus *p = nullptr;
    ~CorpusGuard() {
        if (p)
            elpis_corpus_close(p);
    }
};

struct GraphGuard {
    elpis_context_graph *p = nullptr;
    ~GraphGuard() {
        if (p)
            elpis_context_graph_destroy(p);
    }
};

struct ResultGuard {
    elpis_regex_hacf_query_ingress_result_v1 *p = nullptr;
    ~ResultGuard() {
        if (p)
            elpis_regex_hacf_query_ingress_result_destroy_v1(p);
    }
};

static std::string checked(const char *p) {
    req(p != nullptr, "unexpected null string accessor");
    return std::string(p);
}

static void append_field(
    std::string &out,
    const char *name,
    const char *value)
{
    req(value != nullptr, "null identity field");
    out += name;
    out.push_back('=');
    out += value;
    out.push_back('\n');
}

static void append_candidates(
    std::string &out,
    const elpis_regex_hacf_query_ingress_result_v1 *r,
    bool require_nonempty)
{
    const uint32_t n =
        elpis_regex_hacf_query_ingress_result_candidate_count_v1(r);

    if (require_nonempty)
        req(n > 0u, "positive Regex result produced zero candidates");

    out += "candidate_count=";
    out += std::to_string(n);
    out.push_back('\n');

    for (uint32_t i = 0; i < n; ++i) {
        elpis_regex_hacf_query_ingress_candidate_view_v1 row{};

        req(
            elpis_regex_hacf_query_ingress_result_candidate_at_v1(
                r, i, &row) ==
                ELPIS_REGEX_HACF_QUERY_INGRESS_OK,
            "candidate accessor failed");

        out += "candidate_";
        out += std::to_string(i);
        out.push_back('=');
        out += row.candidate_id;
        out.push_back('\n');
    }
}

static std::string positive_identity(
    const std::string &task,
    size_t chunk,
    elpis_corpus *corpus,
    elpis_context_graph *graph)
{
    ResultGuard r;

    const int rc =
        elpis_regex_hacf_query_ingress_run_v1(
            reinterpret_cast<const uint8_t *>(task.data()),
            task.size(),
            chunk,
            corpus,
            graph,
            &r.p);

    if (rc != ELPIS_REGEX_HACF_QUERY_INGRESS_OK) {
        std::cerr
            << "FAIL positive composite rc=" << rc
            << " chunk=" << chunk
            << " error="
            << elpis_regex_hacf_query_ingress_last_error_v1()
            << "\n";
        std::exit(1);
    }

    req(r.p != nullptr, "positive composite returned null");

    req(
        elpis_regex_hacf_query_ingress_result_fail_closed_v1(r.p) == 0,
        "positive composite unexpectedly fail-closed");

    req(
        elpis_regex_hacf_query_ingress_result_batch_published_v1(r.p) == 1,
        "positive composite did not publish atomic batch");

    req(
        checked(
            elpis_regex_hacf_query_ingress_result_status_v1(r.p)) ==
            "PUBLISHED_ATOMIC_QUERY_OVERLAY",
        "unexpected positive composite status");

    req(
        elpis_regex_hacf_query_ingress_result_semantic_authority_v1(r.p) == 0,
        "semantic authority became nonzero");

    req(
        elpis_regex_hacf_query_ingress_result_admission_authority_v1(r.p) == 0,
        "admission authority became nonzero");

    req(
        elpis_regex_hacf_query_ingress_result_execution_authority_v1(r.p) == 0,
        "execution authority became nonzero");

    req(
        elpis_regex_hacf_query_ingress_result_runtime_admission_v1(r.p) == 0,
        "runtime admission became nonzero");

    std::string out;

    append_field(
        out,
        "source_sha256",
        elpis_regex_hacf_query_ingress_result_source_sha256_v1(r.p));

    append_field(
        out,
        "proposal_digest",
        elpis_regex_hacf_query_ingress_result_proposal_digest_v1(r.p));

    append_field(
        out,
        "corpus_manifest_digest",
        elpis_regex_hacf_query_ingress_result_corpus_manifest_digest_v1(r.p));

    append_field(
        out,
        "context_graph_manifest_digest",
        elpis_regex_hacf_query_ingress_result_context_graph_manifest_digest_v1(r.p));

    append_field(
        out,
        "proposal_set_digest",
        elpis_regex_hacf_query_ingress_result_proposal_set_digest_v1(r.p));

    append_field(
        out,
        "query_local_segment_digest",
        elpis_regex_hacf_query_ingress_result_query_local_segment_digest_v1(r.p));

    append_field(
        out,
        "overlay_identity",
        elpis_regex_hacf_query_ingress_result_overlay_identity_v1(r.p));

    append_field(
        out,
        "batch_receipt_identity",
        elpis_regex_hacf_query_ingress_result_batch_receipt_identity_v1(r.p));

    append_candidates(out, r.p, true);

    return out;
}

static std::string contradiction_identity(
    const std::string &task,
    size_t chunk,
    elpis_corpus *corpus,
    elpis_context_graph *graph)
{
    ResultGuard r;

    const int rc =
        elpis_regex_hacf_query_ingress_run_v1(
            reinterpret_cast<const uint8_t *>(task.data()),
            task.size(),
            chunk,
            corpus,
            graph,
            &r.p);

    if (rc != ELPIS_REGEX_HACF_QUERY_INGRESS_OK) {
        std::cerr
            << "FAIL contradiction composite rc=" << rc
            << " chunk=" << chunk
            << " error="
            << elpis_regex_hacf_query_ingress_last_error_v1()
            << "\n";
        std::exit(1);
    }

    req(r.p != nullptr, "contradiction returned null result");

    req(
        elpis_regex_hacf_query_ingress_result_fail_closed_v1(r.p) == 1,
        "contradiction did not fail closed");

    req(
        elpis_regex_hacf_query_ingress_result_batch_published_v1(r.p) == 0,
        "contradiction published a batch");

    req(
        checked(
            elpis_regex_hacf_query_ingress_result_status_v1(r.p)) ==
            "REJECTED_PRE_BATCH_AMBIGUITY",
        "unexpected contradiction status");

    std::string out;

    append_field(
        out,
        "source_sha256",
        elpis_regex_hacf_query_ingress_result_source_sha256_v1(r.p));

    append_field(
        out,
        "proposal_digest",
        elpis_regex_hacf_query_ingress_result_proposal_digest_v1(r.p));

    append_field(
        out,
        "corpus_manifest_digest",
        elpis_regex_hacf_query_ingress_result_corpus_manifest_digest_v1(r.p));

    append_field(
        out,
        "context_graph_manifest_digest",
        elpis_regex_hacf_query_ingress_result_context_graph_manifest_digest_v1(r.p));

    append_candidates(out, r.p, false);

    return out;
}

static void oversized_rejection(
    const std::string &task,
    size_t chunk,
    elpis_corpus *corpus,
    elpis_context_graph *graph)
{
    elpis_regex_hacf_query_ingress_result_v1 *r =
        reinterpret_cast<elpis_regex_hacf_query_ingress_result_v1 *>(0x1);

    const int rc =
        elpis_regex_hacf_query_ingress_run_v1(
            reinterpret_cast<const uint8_t *>(task.data()),
            task.size(),
            chunk,
            corpus,
            graph,
            &r);

    if (rc != ELPIS_REGEX_HACF_QUERY_INGRESS_E_REGEX) {
        std::cerr
            << "FAIL oversized composite rc=" << rc
            << " chunk=" << chunk
            << " error="
            << elpis_regex_hacf_query_ingress_last_error_v1()
            << "\n";
        std::exit(1);
    }

    req(r == nullptr, "oversized composite published a result");

    const char *err =
        elpis_regex_hacf_query_ingress_last_error_v1();

    req(err != nullptr, "oversized composite has null error");

    req(
        std::strcmp(
            err,
            "REGEX_ABI:-4:INPUT_EXCEEDS_CARRY") == 0,
        "unexpected oversized composite error");
}

int main(int argc, char **argv) {
    req(
        argc == 2,
        "usage: test_query_ingress_bounded STATE_ROOT");

    const std::vector<size_t> chunks =
        {1, 2, 3, 7, 13, 64, 4096};

    CorpusGuard corpus;

    req(
        elpis_corpus_open(argv[1], &corpus.p) == 0,
        "failed to open qualification corpus");

    req(corpus.p != nullptr, "corpus returned null");

    GraphGuard graph;

    req(
        elpis_context_graph_create(
            nullptr, 0u, &graph.p) == 0,
        "failed to create empty context graph");

    req(graph.p != nullptr, "context graph returned null");

    uint64_t docs_before = 999u;
    uint64_t corpus_chunks_before = 999u;

    req(
        elpis_corpus_counts(
            corpus.p,
            &docs_before,
            &corpus_chunks_before) == 0,
        "initial corpus counts failed");

    req(
        docs_before == 0u &&
        corpus_chunks_before == 0u,
        "qualification corpus unexpectedly nonempty");

    req(
        elpis_context_graph_edge_count(graph.p) == 0u,
        "qualification graph unexpectedly nonempty");

    /* Mechanical fixtures of the bounded grammar. */
    const std::string positive =
        "touching endpoints may merge; maximum end.";

    const std::string contradictory =
        "touching endpoints do not merge; "
        "touching endpoints may merge; maximum end.";

    const std::string oversized_counterexample =
        std::string("touching endpoints do not merge.") +
        std::string(1100, ' ') +
        "touching endpoints may merge; maximum end.";

    req(oversized_counterexample.size() == 1174u, "oversized fixture length changed");

    std::string positive_baseline;
    std::string contradiction_baseline;

    for (size_t chunk : chunks) {
        const std::string p =
            positive_identity(
                positive,
                chunk,
                corpus.p,
                graph.p);

        if (positive_baseline.empty())
            positive_baseline = p;
        else
            req(
                p == positive_baseline,
                "positive full-chain identity is chunk-sensitive");

        const std::string c =
            contradiction_identity(
                contradictory,
                chunk,
                corpus.p,
                graph.p);

        if (contradiction_baseline.empty())
            contradiction_baseline = c;
        else
            req(
                c == contradiction_baseline,
                "contradiction full-chain identity is chunk-sensitive");

        oversized_rejection(
            oversized_counterexample,
            chunk,
            corpus.p,
            graph.p);
    }

    /*
     * Pinned identities: byte-identical to the donor implementation at the
     * migration basis commit over an empty corpus and empty context graph.
     * Any drift is an identity change, not a refactor.
     */
    static const char *const pinned_positive =
        "source_sha256=3497e9b3d1a1b6fd66e26b1cda20581dd7b098a58aa6f44ec90cf353c8a53fc6\n"
        "proposal_digest=ad1f6d771a8312c87f679bb665bed7d738cf07c387bb3c99518da6032f529c74\n"
        "corpus_manifest_digest=270bf8d9f6bc321a67a0bebbc3a0a08719bf90124edfc349c64e49d8b7995bf5\n"
        "context_graph_manifest_digest=cf703476481da9196b9b43bcaa050d40a9b05cc25e52c0bd1069b8209d1c3e15\n"
        "proposal_set_digest=8a6bb34300d30bfa7a6e74808b1d3bafb73f886c1313475fb3b8f18f8b02f905\n"
        "query_local_segment_digest=59ddbd0f90ccc67b8f7b257b0a5707fd8fde8f0f60ebdd49e908bd8c41cbde09\n"
        "overlay_identity=35952c41e0927cccfd59dec25400d217849b5be6d95af60b9d486e132487c98b\n"
        "batch_receipt_identity=05ec95801dae76c18dcc7f8cd8a236dea9ab8b324385fdb95712b458a95de6dc\n"
        "candidate_count=1\n"
        "candidate_0=3b46de55e010865d591ccdc1dd730783e905008e179d7f0f2180729753fff028\n";
    static const char *const pinned_contradiction =
        "source_sha256=b3edc1ee2940c86f72e91fc84c88d6a42ae9fa5357c652505d376f46b7cc6ec9\n"
        "proposal_digest=35da57406ab61c2c3a1fee8ad5c3da573c0e71dfffdd0212e4fa6896259a61a3\n"
        "corpus_manifest_digest=270bf8d9f6bc321a67a0bebbc3a0a08719bf90124edfc349c64e49d8b7995bf5\n"
        "context_graph_manifest_digest=cf703476481da9196b9b43bcaa050d40a9b05cc25e52c0bd1069b8209d1c3e15\n"
        "candidate_count=0\n";
    req(positive_baseline == pinned_positive,
        "positive identity drifted from the pinned identity");
    req(contradiction_baseline == pinned_contradiction,
        "contradiction identity drifted from the pinned identity");

    uint64_t docs_after = 999u;
    uint64_t corpus_chunks_after = 999u;

    req(
        elpis_corpus_counts(
            corpus.p,
            &docs_after,
            &corpus_chunks_after) == 0,
        "final corpus counts failed");

    req(
        docs_after == docs_before &&
        corpus_chunks_after == corpus_chunks_before,
        "composite mutated HACF corpus");

    req(
        elpis_context_graph_edge_count(graph.p) == 0u,
        "composite mutated context graph");

    std::cout
        << "positive_identity_begin\n"
        << positive_baseline
        << "positive_identity_end\n"
        << "contradiction_identity_begin\n"
        << contradiction_baseline
        << "contradiction_identity_end\n"
        << "oversized_default_profile=REJECTED_PRE_HACF\n"
        << "oversized_result_published=false\n"
        << "oversized_batch_published=false\n"
        << "semantic_authority=false\n"
        << "admission_authority=false\n"
        << "execution_authority=false\n"
        << "runtime_admission=false\n"
        << "PASS_QUERY_INGRESS_BOUNDED\n";

    return 0;
}
