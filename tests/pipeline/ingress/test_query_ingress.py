"""Regex -> HACF -> query-local proposal batch through the Python binding.

The corpus is built by the structure subsystem's retrieval bridge; the ingress
only opens it. Every result is a proposal with zero authority.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from elpis.pipeline.ingress import (
    PROPOSAL_SCHEMA,
    ContextEdge,
    IngressError,
    IngressLibrary,
    QueryIngress,
    lex,
    lex_stream,
)
from elpis.structure.retrieval.hacf import RetrievalLibrary, build_corpus_and_index

from ...conftest import require_native_library

DOCS = [
    ("spec", "Intervals whose touching endpoints may merge keep the maximum end.", "elpis.docs", "canonical"),
    ("note", "Keep at least one merged range for every overlap.", "elpis.docs", "reference"),
]
POSITIVE = b"touching endpoints may merge; maximum end."
CONTRADICTION = b"touching endpoints do not merge; touching endpoints may merge; maximum end."
OVERSIZED = b"touching endpoints do not merge." + b" " * 1100 + b"touching endpoints may merge; maximum end."
ZERO_AUTHORITY = ("semantic_authority", "admission_authority", "execution_authority", "runtime_admission")


@pytest.fixture(scope="module")
def library():
    return IngressLibrary(require_native_library("elpis_ingress_bridge"))


@pytest.fixture(scope="module")
def corpus_root(tmp_path_factory):
    retrieval = RetrievalLibrary(require_native_library("elpis_retrieval_bridge"))
    state = tmp_path_factory.mktemp("ingress-state")
    build_corpus_and_index(retrieval, state, DOCS).destroy()
    return state / "corpus"


@pytest.fixture
def ingress(library, corpus_root):
    with QueryIngress(library, corpus_root) as handle:
        yield handle


def _hit_chunks(result, pattern_id):
    proposal = json.loads(result.proposal_json)
    return [hit["chunk_digest"] for row in proposal["hacf"]["retrieval"] if row["pattern_id"] == pattern_id
            for hit in row["hacf_primary_hits"]]


def test_positive_task_publishes_atomic_overlay(ingress):
    result = ingress.run(POSITIVE)
    assert result.status == "PUBLISHED_ATOMIC_QUERY_OVERLAY"
    assert result.batch_published and not result.fail_closed
    assert result.source_sha256 == hashlib.sha256(POSITIVE).hexdigest()
    assert all((result.proposal_set_digest, result.query_local_segment_digest,
                result.overlay_identity, result.batch_receipt_identity))
    assert len(result.candidate_ids) == 1
    assert _hit_chunks(result, "coal.touching.allowed.v1")


def test_proposal_export_is_zero_authority_and_self_identifying(ingress):
    result = ingress.run(POSITIVE)
    proposal = json.loads(result.proposal_json)
    assert proposal["schema"] == PROPOSAL_SCHEMA
    assert proposal["candidate_status"] == "PROPOSED_UNADMITTED"
    assert all(proposal[flag] is False for flag in ZERO_AUTHORITY)
    assert proposal["source_sha256"] == result.source_sha256
    assert proposal["hacf"]["corpus_manifest_digest"] == result.corpus_manifest_digest
    assert proposal["hacf"]["dense_vector_used"] is False
    body = dict(proposal)
    digest = body.pop("proposal_digest")
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert digest == result.proposal_digest == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize("chunk", [1, 7, 64, 4096])
def test_identity_is_independent_of_transport_chunking(ingress, chunk):
    reference = ingress.run(POSITIVE, chunk_size=4096)
    assert ingress.run(POSITIVE, chunk_size=chunk) == reference


def test_ambiguity_is_rejected_before_any_batch(ingress):
    result = ingress.run(CONTRADICTION)
    assert result.status == "REJECTED_PRE_BATCH_AMBIGUITY"
    assert result.fail_closed and not result.batch_published
    assert result.candidate_ids == ()
    assert (result.proposal_set_digest, result.query_local_segment_digest,
            result.overlay_identity, result.batch_receipt_identity) == (None, None, None, None)


def test_input_beyond_the_bounded_profile_publishes_nothing(ingress):
    with pytest.raises(IngressError) as info:
        ingress.run(OVERSIZED)
    assert info.value.code == "REGEX"
    assert "INPUT_EXCEEDS_CARRY" in info.value.detail


def test_ingress_never_mutates_the_corpus(library, corpus_root):
    def snapshot():
        return {p.name: p.read_bytes() for p in sorted((corpus_root / "corpus").rglob("*")) if p.is_file()}

    before_blobs = snapshot()
    with QueryIngress(library, corpus_root) as handle:
        counts = handle.corpus_counts()
        manifest = handle.run(POSITIVE).corpus_manifest_digest
        handle.run(CONTRADICTION)
        assert handle.corpus_counts() == counts == (2, 2)
        assert handle.run(POSITIVE).corpus_manifest_digest == manifest
    assert snapshot() == before_blobs


def test_context_edges_reach_the_proposal(library, corpus_root, ingress):
    spec = _hit_chunks(ingress.run(POSITIVE), "coal.touching.allowed.v1")[0]
    note = _hit_chunks(ingress.run(b"at least 3"), "cmp.gte.at_least.v1")[0]
    assert spec != note
    edge = ContextEdge.of(spec, note, "a" * 64, 1, 1)
    with QueryIngress(library, corpus_root, (edge,)) as linked:
        assert linked.edge_count == 1
        result = linked.run(POSITIVE)
    proposal = json.loads(result.proposal_json)
    hit = proposal["hacf"]["retrieval"][0]["hacf_primary_hits"][0]
    assert [n["chunk_digest"] for n in hit["context_neighbors"]] == [note]
    assert hit["context_neighbors"][0]["text"] == DOCS[1][1]
    assert result.context_graph_manifest_digest != ingress.run(POSITIVE).context_graph_manifest_digest


def test_invalid_context_edges_are_refused(library, corpus_root):
    with pytest.raises(IngressError):
        ContextEdge.of("A" * 64, "b" * 64, "c" * 64, 1, 1)
    with pytest.raises(IngressError):
        ContextEdge.of("a" * 64, "b" * 64, "c" * 64, 0, 1)
    with pytest.raises(IngressError) as info:
        QueryIngress(library, corpus_root, (ContextEdge.of("a" * 64, "a" * 64, "c" * 64, 1, 1),))
    assert info.value.code == "GRAPH"


def test_no_corpus_is_ever_created(library, tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(IngressError) as info:
        QueryIngress(library, missing)
    assert info.value.code == "CORPUS" and not missing.exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(IngressError):
        QueryIngress(library, empty)
    assert list(empty.iterdir()) == []


def test_symlinked_or_relative_corpus_root_is_refused(library, corpus_root, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(corpus_root, target_is_directory=True)
    with pytest.raises(IngressError):
        QueryIngress(library, link)
    with pytest.raises(IngressError):
        QueryIngress(library, Path(os.path.relpath(corpus_root)))


def test_closed_handle_refuses_work(library, corpus_root):
    handle = QueryIngress(library, corpus_root)
    handle.close()
    handle.close()
    with pytest.raises(IngressError):
        handle.run(POSITIVE)


def test_library_must_be_an_explicit_existing_path(tmp_path):
    with pytest.raises(IngressError):
        IngressLibrary("libelpis_ingress_bridge.so")
    with pytest.raises(IngressError):
        IngressLibrary(tmp_path / "absent.so")


def test_lexical_results_expose_bounded_evidence(library):
    result = lex(library, POSITIVE)
    assert result.source_sha256 == hashlib.sha256(POSITIVE).hexdigest()
    assert result.source_bytes == len(POSITIVE)
    assert {e.pattern_id for e in result.evidence} == {"coal.touching.allowed.v1", "coal.reducer.max.v1"}
    assert not result.fail_closed and result.ambiguity_count == 0
    assert lex_stream(library, [POSITIVE[:5], POSITIVE[5:]]) == result
    ambiguous = lex(library, CONTRADICTION)
    assert ambiguous.fail_closed and ambiguous.ambiguity_count > 0
    with pytest.raises(IngressError) as info:
        lex(library, OVERSIZED)
    assert info.value.code == "RANGE"
    with pytest.raises(IngressError):
        lex(library, b"\xff invalid utf-8")
