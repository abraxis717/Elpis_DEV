from __future__ import annotations
from dataclasses import replace
import numpy as np
import pytest
from elpis.inference.context import initial_snapshot
from elpis.inference.contracts import ContractError
from elpis.inference.structural import AddressProposal
from elpis.inference.transaction import InferenceRequest, InferenceEngine
from elpis.inference.speculative import Draft, run_speculative, verify_draft


def proposal(state, route_key="route"):
    return AddressProposal(
        "1" * 64,
        "2" * 64,
        "3" * 64,
        state.context.digest,
        "4" * 64,
        route_key,
        (),
        (),
        (),
        0.5,
        ("5" * 64,),
    )


def prefilled(target):
    runtime = InferenceEngine(target)
    initial = runtime.initial(initial_snapshot())
    request = InferenceRequest(
        "prefill", initial.context.digest, "PREFILL", (1, 2, 3)
    )
    result = runtime.execute(initial, request, expected_state=initial.digest)
    assert result.receipt.terminal == "COMMITTED"
    return runtime, initial, result.state


def test_inference_request_rejects_malformed_proposal_elements_at_construction(target):
    runtime = InferenceEngine(target[0])
    state = runtime.initial(initial_snapshot())
    with pytest.raises(ContractError, match="INVALID:proposal packet type"):
        InferenceRequest(
            "bad",
            state.context.digest,
            "PREFILL",
            (1,),
            proposals=(object(),),
        )


def test_tampered_malformed_proposals_return_typed_atomic_failure_receipt(target):
    runtime = InferenceEngine(target[0])
    state = runtime.initial(initial_snapshot())

    def malformed():
        request = InferenceRequest(
            "malformed", state.context.digest, "PREFILL", (1,)
        )
        object.__setattr__(request, "proposals", (object(),))
        return request

    a = runtime.execute(state, malformed(), expected_state=state.digest)
    b = runtime.execute(state, malformed(), expected_state=state.digest)
    assert a.state == b.state == state
    assert a.receipt.terminal == b.receipt.terminal == "FAILED"
    assert a.receipt.failure == b.receipt.failure == "INVALID"
    assert a.receipt.target_steps == b.receipt.target_steps == ()
    assert len(a.receipt.structural) == 1
    assert a.receipt.request == b.receipt.request
    assert a.receipt.structural == b.receipt.structural
    assert a.receipt.digest == b.receipt.digest


def test_speculative_malformed_proposals_cannot_escape_failure_receipt(target):
    runtime, _, state = prefilled(target[0])
    request = InferenceRequest(
        "bad-spec", state.context.digest, "GREEDY", count=1
    )
    object.__setattr__(request, "proposals", (object(),))
    result = run_speculative(
        runtime,
        state,
        request,
        object(),
        expected_state=state.digest,
        block_size=1,
    )
    assert result.state == state
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "INVALID"
    assert result.receipt.target_steps == ()
    assert len(result.receipt.structural) == 1


def test_fresh_runtime_replays_structural_and_prefetch_history(target):
    runtime = InferenceEngine(target[0])
    initial = runtime.initial(initial_snapshot())
    p = proposal(initial)
    request = InferenceRequest(
        "prefill-proposals",
        initial.context.digest,
        "PREFILL",
        (1, 2),
        proposals=(p,),
    )
    committed = runtime.execute(initial, request, expected_state=initial.digest)
    assert committed.receipt.terminal == "COMMITTED"
    assert committed.state.structural == (p,)
    assert committed.state.step_proposals == ((p,), (p,))
    assert len(committed.state.prefetch) == 2

    restarted = InferenceEngine(target[0])
    continuation = InferenceRequest(
        "continue", committed.state.context.digest, "GREEDY", count=1
    )
    result = restarted.execute(
        committed.state,
        continuation,
        expected_state=committed.state.digest,
    )
    assert result.receipt.terminal == "COMMITTED"
    assert committed.state.digest in restarted._validated_states


def test_forged_structural_prefetch_planes_are_rejected_and_not_cached(target):
    runtime = InferenceEngine(target[0])
    initial = runtime.initial(initial_snapshot())
    p = proposal(initial)
    request = InferenceRequest(
        "prefill-forgery",
        initial.context.digest,
        "PREFILL",
        (1, 2),
        proposals=(p,),
    )
    committed = runtime.execute(initial, request, expected_state=initial.digest).state

    forged_proposal = replace(p, route_key="forged-route")
    forged_prefetch = tuple(
        replace(plan, proposals=(forged_proposal.digest,))
        for plan in committed.prefetch
    )
    forged = replace(
        committed,
        structural=(forged_proposal,),
        prefetch=forged_prefetch,
    )

    restarted = InferenceEngine(target[0])
    continuation = InferenceRequest(
        "continue-forged", forged.context.digest, "GREEDY", count=1
    )
    result = restarted.execute(
        forged,
        continuation,
        expected_state=forged.digest,
    )
    assert result.state == forged
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "IDENTITY"
    assert forged.digest not in restarted._validated_states


def test_failed_transaction_does_not_cache_transient_overlay_states(target):
    runtime, _, state = prefilled(target[0])
    before = set(runtime._validated_states)
    result = runtime.execute(
        state,
        InferenceRequest(
            "rollback-cache", state.context.digest, "PREFILL", (1, 2, 999)
        ),
        expected_state=state.digest,
    )
    assert result.state == state
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "INVALID"
    assert runtime._validated_states == before


def test_rejected_speculative_suffix_does_not_enter_validation_cache(target):
    runtime, _, state = prefilled(target[0])
    before = set(runtime._validated_states)
    greedy = int(np.argmax(state.neural.logits))
    wrong = (greedy + 1) % runtime.target.config.vocab
    draft = Draft(
        state.digest,
        runtime.target.model_identity,
        "6" * 64,
        (wrong, (wrong + 1) % runtime.target.config.vocab),
        (0.9, 0.9),
    )
    request = InferenceRequest(
        "reject-cache", state.context.digest, "GREEDY", count=2
    )
    verification, _ = verify_draft(runtime, state, request, draft)
    assert verification.accepted_count == 0
    assert verification.accepted_state == state
    assert runtime._validated_states == before


def test_validation_cache_is_bounded(target):
    runtime = InferenceEngine(target[0])
    runtime.initial(initial_snapshot())
    for i in range(1100):
        runtime._remember_validated(f"{i:064x}")
    assert len(runtime._validated_states) <= 1024
    assert len(runtime._validated_state_order) <= 1024
