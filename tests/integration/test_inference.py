"""Inference end to end on the synthetic DSV4 fixture.

Real ingress export -> structural address proposals -> committed decode
transaction -> recorded receipt; model output never becomes authority.
"""
from __future__ import annotations

import pytest

from elpis.inference.context import initial_snapshot
from elpis.inference.contracts import ContractError
from elpis.inference.drivers.dsv4.fixtures import make_fixture
from elpis.inference.structural import RouteRule, from_regex_hacf
from elpis.inference.transaction import InferenceEngine, InferenceRequest
from elpis.substrate.digests import raw_digest
from elpis.substrate.synthetic import SyntheticFileAssets

from .conftest import POSITIVE


@pytest.fixture
def engine(native_workspace, fms_file_library):
    provider = SyntheticFileAssets(root=native_workspace, library=fms_file_library,
                                   warm_bytes=64, staging_bytes=128)
    target, _, meta = make_fixture(provider, native_workspace / "dsv4")
    assert meta["training"] == "NONE"
    yield InferenceEngine(target), target, provider
    provider.close()


def _proposals(result, snapshot):
    payload = result.proposal_json
    return from_regex_hacf(payload, expected_payload=raw_digest(payload), expected_source=result.source_sha256,
                           expected_corpus=result.corpus_manifest_digest, context_snapshot=snapshot,
                           query_overlay=result.overlay_identity,
                           rules=(RouteRule("touching-merge", ("fixture-bank",), ()),))


def test_ingress_proposals_drive_a_committed_recorded_decode(runtime, ingress, engine):
    rt, target, _ = engine
    ingress_result, ingress_record = runtime.run_ingress(ingress, POSITIVE)
    state = rt.initial(initial_snapshot())
    proposals = _proposals(ingress_result, state.context.digest)
    assert proposals and all(not p.semantic_authority and not p.execution_authority for p in proposals)
    request = InferenceRequest("integration-prefill", state.context.digest, "PREFILL", (1, 2, 3),
                               proposals=proposals)
    result, recorded = runtime.decode(rt, state, request, expected_state=state.digest)
    assert result.receipt.terminal == "COMMITTED" and result.state.structural == proposals
    record = recorded.record
    assert (record.subsystem, record.kind, record.digest) == ("inference", "inference.decode", result.receipt.digest)
    assert dict(record.bindings)["model"] == target.model_identity
    assert dict(record.bindings)["output_state"] == result.receipt.output_state

    # A fresh engine revalidates the committed history by replay before continuing.
    fresh = InferenceEngine(target)
    follow = InferenceRequest("integration-greedy", result.state.context.digest, "GREEDY", count=2)
    continued, recorded2 = runtime.decode(fresh, result.state, follow, expected_state=result.state.digest)
    assert continued.receipt.terminal == "COMMITTED" and recorded2.event_index > recorded.event_index
    assert [r.record.subsystem for r in runtime.history.records()] == ["pipeline", "inference", "inference"]
    assert ingress_record in runtime.history.records()


def test_stale_or_tampered_inputs_commit_nothing_and_record_nothing(runtime, ingress, engine):
    rt, _, _ = engine
    ingress_result, _ = runtime.run_ingress(ingress, POSITIVE)
    state = rt.initial(initial_snapshot())
    recorded_before = runtime.history.records()

    # Proposals bound to another context snapshot are stale: typed failure, state unchanged.
    foreign = _proposals(ingress_result, "0" * 64)
    request = InferenceRequest("stale", state.context.digest, "PREFILL", (1, 2), proposals=foreign)
    result, recorded = runtime.decode(rt, state, request, expected_state=state.digest)
    assert result.receipt.terminal == "FAILED" and result.state == state and recorded is None

    # A caller-selected expected state that is not the committed base is refused before any step.
    request = InferenceRequest("stale-base", state.context.digest, "PREFILL", (1,))
    result, recorded = runtime.decode(rt, state, request, expected_state="0" * 64)
    assert result.receipt.terminal == "FAILED" and recorded is None

    # A tampered ingress export never becomes a proposal.
    payload = ingress_result.proposal_json.replace(b'"semantic_authority":false', b'"semantic_authority":true')
    assert payload != ingress_result.proposal_json
    with pytest.raises(ContractError):
        from_regex_hacf(payload, expected_payload=raw_digest(payload), expected_source=ingress_result.source_sha256,
                        expected_corpus=ingress_result.corpus_manifest_digest,
                        context_snapshot=state.context.digest, query_overlay=ingress_result.overlay_identity,
                        rules=())
    assert runtime.history.records() == recorded_before
