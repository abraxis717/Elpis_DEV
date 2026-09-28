"""Retrieval-bundle boundary: validated retrieval bundle -> AddressProposal.

``structural.from_retrieval_bundle`` delegates bundle validation to
``elpis.structure.retrieval`` and adds an inference-local export pin. It must:
- accept a valid bundle and produce a correctly bound AddressProposal;
- fail closed (typed ContractError) when the export pin is tampered;
- fail closed when the bundle is invalid (validation stays in structure).

The retrieval stage is a read-only dependency.
"""
from __future__ import annotations

import pytest

from elpis.inference.structural import from_retrieval_bundle
from elpis.inference.contracts import ContractError, identity
from elpis.structure.retrieval.contracts import RetrievalBundle, RetrievalItem, _sha256_hex


def _item(text: str, **kw) -> RetrievalItem:
    td = _sha256_hex(text.encode("utf-8")) if text else ""
    defaults = dict(
        chunk_digest="a" * 64, doc_digest="b" * 64, namespace="elpis.docs",
        authority="canonical", graph_parent_digest="0" * 64, text_digest=td,
        fusion_score_key=100, dense_score_key=50, lexical_rank=1, dense_rank=1,
        final_rank=0, source_mask=3, item_kind=1, graph_hop=0, edge_type=0,
        edge_authority=0, text=text, text_bytes=len(text.encode("utf-8")) if text else 0,
    )
    defaults.update(kw)
    return RetrievalItem(**defaults)


def _bundle(items):
    return RetrievalBundle(
        schema="elpis.retrieval_bundle.v1",
        query_digest="1" * 64,
        corpus_manifest_digest="2" * 64,
        vector_index_manifest_digest="v" * 64,
        graph_snapshot_digest="g" * 64,
        fusion_policy_digest="f" * 64,
        bundle_digest="b" * 64,
        hacf_package_digest="h" * 64,
        corpus_epoch=0,
        vector_index_epoch=0,
        items=tuple(items),
    )


def test_valid_bundle_produces_bound_proposal():
    text = "alpha engine exact retrieval anchor"
    b = _bundle([_item(text)])
    expected_query = "1" * 64
    expected_corpus = "2" * 64
    expected_bundle = identity("r1-bundle-export", b.to_canonical_dict())
    proposal = from_retrieval_bundle(
        b, expected_query=expected_query, expected_corpus=expected_corpus,
        expected_bundle=expected_bundle, context_snapshot="3" * 64,
        query_overlay="4" * 64,
    )
    assert proposal.source == expected_query
    assert proposal.corpus == expected_corpus
    assert proposal.context_snapshot == "3" * 64
    assert proposal.query_overlay == "4" * 64
    assert proposal.route_key == "HACF_R1"
    assert proposal.objects == ("a" * 64,)
    assert proposal.provenance == (expected_bundle, "g" * 64, "h" * 64)


def test_tampered_export_pin_fails_closed():
    text = "alpha engine exact retrieval anchor"
    b = _bundle([_item(text)])
    with pytest.raises(ContractError) as exc:
        from_retrieval_bundle(
            b, expected_query="1" * 64, expected_corpus="2" * 64,
            expected_bundle="0" * 64, context_snapshot="3" * 64,
            query_overlay="4" * 64,
        )
    assert exc.value.code.value == "IDENTITY"


def test_wrong_query_digest_fails_closed():
    text = "alpha engine exact retrieval anchor"
    b = _bundle([_item(text)])
    expected_bundle = identity("r1-bundle-export", b.to_canonical_dict())
    with pytest.raises(Exception):
        from_retrieval_bundle(
            b, expected_query="9" * 64, expected_corpus="2" * 64,
            expected_bundle=expected_bundle, context_snapshot="3" * 64,
            query_overlay="4" * 64,
        )


def test_invalid_bundle_fails_closed():
    # Empty text -> bundle validation fails (MISSING_FROZEN_TEXT).
    b = _bundle([_item("")])
    with pytest.raises(Exception):
        from_retrieval_bundle(
            b, expected_query="1" * 64, expected_corpus="2" * 64,
            expected_bundle=identity("r1-bundle-export", b.to_canonical_dict()),
            context_snapshot="3" * 64, query_overlay="4" * 64,
        )
