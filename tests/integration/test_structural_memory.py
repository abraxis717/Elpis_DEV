"""Structural memory end to end: HACF corpus -> hybrid retrieval -> validated bundle -> evidence."""
from __future__ import annotations

import hashlib
import json

import pytest

from elpis.structure.retrieval.errors import BundleValidationError
from elpis.structure.retrieval.evidence import build_evidence_envelope
from elpis.structure.retrieval.hacf import bundle_from_json, hybrid_retrieve
from elpis.structure.retrieval.query import derive_query

from .conftest import POSITIVE

REQUEST = {"request_id": "integration", "prompt": "touching endpoints merge maximum end",
           "domain": "elpis.docs", "entrypoint": "retrieve"}


def _retrieve(handle):
    query = derive_query(REQUEST)
    result = hybrid_retrieve(handle, query.query_text)
    return query, result, bundle_from_json(result["bundle_json"], result)


def test_validated_bundle_feeds_evidence_without_a_continuity_write(runtime, corpus):
    handle, _ = corpus
    query, result, bundle = _retrieve(handle)
    assert bundle.items, "structural memory returned nothing for an on-topic query"
    before = runtime.continuity.snapshot()
    decision = runtime.admit_retrieval(bundle, expected_query=result["query_digest"],
                                       expected_corpus=result["corpus_manifest_digest"])
    assert decision is not None and not decision.exceeded
    assert runtime.continuity.snapshot() == before
    envelope = build_evidence_envelope(
        hashlib.sha256(json.dumps(REQUEST, sort_keys=True).encode()).hexdigest(),
        query.query_digest, bundle,
        hashlib.sha256(json.dumps(decision.to_canonical_dict(), sort_keys=True).encode()).hexdigest())
    assert envelope.evidence_references == tuple(item.chunk_digest for item in bundle.items)


@pytest.mark.parametrize("field", ["expected_query", "expected_corpus"])
def test_bundle_bound_to_other_query_or_corpus_is_refused(runtime, corpus, field):
    _, result, bundle = _retrieve(corpus[0])
    kwargs = dict(expected_query=result["query_digest"], expected_corpus=result["corpus_manifest_digest"])
    kwargs[field] = "0" * 64
    before = runtime.continuity.snapshot()
    with pytest.raises(BundleValidationError):
        runtime.admit_retrieval(bundle, **kwargs)
    assert runtime.continuity.snapshot() == before


def test_ingress_and_retrieval_read_the_same_structural_memory(runtime, corpus, ingress):
    _, result, bundle = _retrieve(corpus[0])
    ingress_result = runtime.run_ingress(ingress, POSITIVE)
    runtime.admit_retrieval(bundle, expected_query=result["query_digest"],
                            expected_corpus=result["corpus_manifest_digest"])
    retrieved = {item.chunk_digest for item in bundle.items}
    proposal = json.loads(ingress_result.proposal_json)
    proposed = {hit["chunk_digest"] for row in proposal["hacf"]["retrieval"] for hit in row["hacf_primary_hits"]}
    assert proposed & retrieved, "ingress and retrieval must reach the same stored chunks"
