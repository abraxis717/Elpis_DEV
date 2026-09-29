"""Sequence transactions: streamed tokens, a zero-identity token loop, legacy-identical commits."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from elpis.inference.context import initial_snapshot
from elpis.inference.contracts import ContractError
from elpis.inference.drivers.dsv4.fixtures import make_fixture
from elpis.inference.sequence import Sequence
from elpis.inference.structural import AddressProposal
from elpis.inference.target import LatentInput
from elpis.inference.transaction import InferenceEngine, InferenceRequest
from elpis.substrate.synthetic import SyntheticFileAssets

from .test_token_stream_kernel import Recorder


@pytest.fixture
def warm_provider(native_workspace, fms_file_library):
    provider = SyntheticFileAssets(root=native_workspace, library=fms_file_library,
                                   warm_bytes=1 << 20, staging_bytes=1 << 16)
    yield provider
    provider.close()


def _latents(target, context):
    return tuple(
        LatentInput(ch, target.projections[ch].source_schema, "a" * 64, context, target.projections[ch].digest,
                    tuple(float((k % 5) - 2) / 8 for k in range(target.projections[ch].weights.shape[0])))
        for ch in ("G", "X", "R"))


def _proposal(context, route="route"):
    return AddressProposal("1" * 64, "2" * 64, "3" * 64, context, "4" * 64, route, (), (), (), 0.5, ("5" * 64,))


def _turns(context, target):
    latents, proposals = _latents(target, context), (_proposal(context),)
    return [
        InferenceRequest("prefill", context, "PREFILL", (1, 4, 9, 2, 7, 11, 3, 5, 8, 13, 6, 0),
                         proposals=proposals, latents=latents),
        InferenceRequest("greedy-1", context, "GREEDY", count=30, latents=latents),
        InferenceRequest("greedy-2", context, "GREEDY", count=5, proposals=proposals),
    ]


def _run(engine, state, request, **kwargs):
    sequence = engine.begin(state, request, expected_state=state.digest, **kwargs)
    tokens = list(sequence)
    return tokens, engine.finalize(sequence)


@pytest.mark.parametrize("scheme", ["DSV41_ENGRAM", "QWEN38_PLE"])
@pytest.mark.parametrize("resident", [False, True], ids=["streamed-experts", "resident-experts"])
def test_finalized_sequences_equal_the_legacy_transaction(warm_provider, native_workspace, scheme, resident):
    target, resident_bytes, _ = make_fixture(warm_provider, native_workspace / scheme, scheme_name=scheme)
    legacy_engine, stream_engine = InferenceEngine(target), InferenceEngine(target)
    state = legacy_engine.initial(initial_snapshot())
    stream_engine.initial(initial_snapshot())
    for request in _turns(state.context.digest, target):
        legacy = legacy_engine.execute(state, request, expected_state=state.digest)
        tokens, result = _run(stream_engine, state, request,
                              resident_experts=resident_bytes if resident else None)
        assert legacy.receipt.terminal == result.receipt.terminal == "COMMITTED"
        assert result.state == legacy.state and result.receipt == legacy.receipt
        assert tuple(tokens) == legacy.receipt.accepted_prefix
        state = result.state


def test_finalized_history_replays_on_a_fresh_engine(warm_provider, native_workspace):
    target, _, _ = make_fixture(warm_provider, native_workspace / "t")
    engine = InferenceEngine(target)
    state = engine.initial(initial_snapshot())
    for request in _turns(state.context.digest, target)[:2]:
        base = state
        _, result = _run(engine, base, request)
        state = result.state
    assert InferenceEngine(target).replay(base, request, result.receipt).state == state
    fresh = InferenceEngine(target)  # validates the whole finalized lineage by replay
    follow = InferenceRequest("follow", state.context.digest, "GREEDY", count=3)
    assert fresh.execute(state, follow, expected_state=state.digest).receipt.terminal == "COMMITTED"


def test_token_loop_computes_no_identity_and_hashes_nothing_resident(warm_provider, native_workspace):
    target, resident_bytes, _ = make_fixture(warm_provider, native_workspace / "t")
    engine = InferenceEngine(target)
    state = engine.initial(initial_snapshot())
    prefill, greedy, _ = _turns(state.context.digest, target)
    committed = _run(engine, state, prefill, resident_experts=resident_bytes)[1].state

    def token_loop():
        sequence = engine.begin(committed, greedy, expected_state=committed.digest,
                                resident_experts=resident_bytes)
        recorder, before = Recorder(), warm_provider.stats()["misses"]
        tokens = recorder.run(lambda: list(sequence))
        assert len(tokens) == greedy.count
        return sequence, recorder.calls, warm_provider.stats()["misses"] - before

    _, calls, misses = token_loop()  # first pass: only cold pages it faults in are verified
    assert calls == Counter({"py:digests.py:raw_digest": misses, "py:digests.py:raw_sha256": misses,
                             "c:_hashlib.openssl_sha256": misses} if misses else {}), calls
    sequence, calls, misses = token_loop()  # same deterministic path, every page resident
    assert misses == 0 and calls == Counter(), calls

    boundary = Recorder()
    boundary.run(lambda: engine.finalize(sequence))
    assert boundary.calls["py:identity.py:content_digest"] > 0  # provenance happens here, not above


def test_stop_and_stop_tokens_commit_the_effective_request(warm_provider, native_workspace):
    target, _, _ = make_fixture(warm_provider, native_workspace / "t")
    engine = InferenceEngine(target)
    state = engine.initial(initial_snapshot())
    prefill, _, _ = _turns(state.context.digest, target)
    state = _run(engine, state, prefill)[1].state
    request = InferenceRequest("long", state.context.digest, "GREEDY", count=20)
    reference = InferenceEngine(target).execute(state, request, expected_state=state.digest)

    sequence = engine.begin(state, request, expected_state=state.digest)
    produced = [sequence.next() for _ in range(7)]
    sequence.stop()
    assert sequence.next() is None and sequence.stop_reason == "YIELD"
    result = engine.finalize(sequence)
    cut = replace(request, count=7)
    assert sequence.effective_request() == cut
    assert result.receipt == InferenceEngine(target).execute(state, cut, expected_state=state.digest).receipt
    assert tuple(produced) == reference.receipt.accepted_prefix[:7]

    stop_at = reference.receipt.accepted_prefix.index(reference.receipt.accepted_prefix[3])
    sequence = engine.begin(state, request, expected_state=state.digest,
                            stop_tokens=(reference.receipt.accepted_prefix[3],))
    tokens = list(sequence)
    assert sequence.stop_reason == "STOP_TOKEN" and len(tokens) == stop_at + 1
    result = engine.finalize(sequence)
    expected = InferenceEngine(target).execute(state, replace(request, count=stop_at + 1),
                                               expected_state=state.digest)
    assert result.state == expected.state and result.receipt == expected.receipt


def test_failures_before_finalization_leave_the_committed_state_untouched(warm_provider, native_workspace):
    target, _, _ = make_fixture(warm_provider, native_workspace / "t")
    engine, legacy = InferenceEngine(target), InferenceEngine(target)
    state = engine.initial(initial_snapshot())
    legacy.initial(initial_snapshot())

    stale = InferenceRequest("stale", state.context.digest, "PREFILL", (1, 2))
    sequence = engine.begin(state, stale, expected_state="0" * 64)
    assert list(sequence) == [] and sequence.stop_reason == "FAILED"
    result = engine.finalize(sequence)
    reference = legacy.execute(state, stale, expected_state="0" * 64)
    assert result.state is state and result.receipt == reference.receipt and result.receipt.failure == "STALE"

    # Tokens already streamed are visible but uncommitted when a later step fails.
    bad = InferenceRequest("bad-token", state.context.digest, "PREFILL", (1, 2, 3, 99))
    sequence = engine.begin(state, bad, expected_state=state.digest)
    assert list(sequence) == [1, 2, 3] and sequence.stop_reason == "FAILED"
    result = engine.finalize(sequence)
    assert result.state is state and result.receipt == legacy.execute(state, bad, expected_state=state.digest).receipt

    wrong_context = _latents(target, "9" * 64)
    request = InferenceRequest("latent", state.context.digest, "PREFILL", (1, 2), latents=wrong_context)
    sequence = engine.begin(state, request, expected_state=state.digest)
    assert list(sequence) == [] and engine.finalize(sequence).receipt == \
        legacy.execute(state, request, expected_state=state.digest).receipt


def test_finalize_requires_a_finished_sequence_of_the_same_engine(warm_provider, native_workspace):
    target, _, _ = make_fixture(warm_provider, native_workspace / "t")
    engine = InferenceEngine(target)
    state = engine.initial(initial_snapshot())
    sequence = engine.begin(state, InferenceRequest("p", state.context.digest, "PREFILL", (1, 2, 3)),
                            expected_state=state.digest)
    sequence.next()
    with pytest.raises(ContractError, match="sequence still active"):
        engine.finalize(sequence)
    with pytest.raises(ContractError, match="sequence engine"):
        InferenceEngine(target).finalize(sequence)
    sequence.stop()
    assert engine.finalize(sequence).state.neural.tokens == (1,)


def test_sequence_inputs_are_frozen_at_begin(warm_provider, native_workspace):
    target, _, _ = make_fixture(warm_provider, native_workspace / "t")
    engine = InferenceEngine(target)
    state = engine.initial(initial_snapshot())
    sequence = engine.begin(state, InferenceRequest("p", state.context.digest, "PREFILL", (1, 2)),
                            expected_state=state.digest)
    assert type(sequence) is Sequence and not hasattr(sequence, "digest")
    for name in ("latents", "proposals", "inject", "steer"):
        assert not hasattr(sequence, name)
        with pytest.raises(AttributeError):
            setattr(sequence, name, ())
