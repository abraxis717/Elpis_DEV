"""Typed-failure contract regressions.

The engine contract: "Typed failure returns the original state with a failure
receipt." A tampered state/request may carry non-canonical values (e.g. NaN
floats) whose digest raises CanonicalIdentityError. Before the fix, such inputs
leaked an untyped CanonicalIdentityError out of execute/verify_draft/
run_speculative, violating the contract and the fail-closed boundary.

These are permanent regressions for the engine-local defect: non-canonical
identities must surface as typed ContractError (IDENTITY), and stale bases
must return a FAILED receipt rather than raise.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from elpis.inference.transaction import InferenceRequest
from elpis.inference.speculative import Draft, run_speculative, verify_draft
from elpis.inference.contracts import ContractError
from elpis.inference.context import initial_snapshot
from elpis.identity import CanonicalIdentityError

from ._helpers import build_runtime, greedy, make_drafter, nan_state, prefill


def _assert_typed(exc):
    assert isinstance(exc, ContractError), f"expected ContractError, got {type(exc).__name__}"
    assert not isinstance(exc, CanonicalIdentityError)


def test_execute_tampered_state_is_typed(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    bad = nan_state(state)
    # Pass the pre-tamper digest so the NaN digest is computed INSIDE execute.
    with pytest.raises(ContractError) as exc:
        rt.execute(bad, InferenceRequest("g", state.context.digest, "GREEDY", count=1),
                   expected_state=state.digest)
    _assert_typed(exc.value)
    assert exc.value.code.value == "IDENTITY"


def test_execute_tampered_request_is_typed(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    bad_req = InferenceRequest("bad", state.context.digest, "PREFILL", (1.0, 2.0, float("nan")))
    with pytest.raises(ContractError) as exc:
        rt.execute(state, bad_req, expected_state=state.digest)
    _assert_typed(exc.value)
    assert exc.value.code.value == "IDENTITY"


def test_verify_draft_tampered_state_is_typed(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    bad = nan_state(state)
    req = InferenceRequest("g", state.context.digest, "GREEDY", count=2)
    # Draft carries the valid pre-tamper base; the NaN state.digest is computed
    # INSIDE verify_draft when comparing draft.committed_base == state.digest.
    draft = Draft(state.digest, rt.target.model_identity, "1" * 64, (4, 5), (0.9, 0.9))
    with pytest.raises(ContractError) as exc:
        verify_draft(rt, bad, req, draft)
    _assert_typed(exc.value)
    assert exc.value.code.value == "IDENTITY"


def test_run_speculative_tampered_state_is_typed(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    bad = nan_state(state)
    drafter = make_drafter(rt, state)
    req = InferenceRequest("g", state.context.digest, "GREEDY", count=2)
    # Pass the valid pre-tamper base so the NaN digest is computed INSIDE.
    with pytest.raises(ContractError) as exc:
        run_speculative(rt, bad, req, drafter, expected_state=state.digest)
    _assert_typed(exc.value)
    assert exc.value.code.value == "IDENTITY"


def test_initial_non_snapshot_is_typed(target):
    rt = build_runtime(target[0])
    with pytest.raises(ContractError) as exc:
        rt.initial("not-a-snapshot")
    _assert_typed(exc.value)
    assert exc.value.code.value == "INVALID"


def test_stale_base_returns_failed_receipt_not_raise(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    result = rt.execute(state, InferenceRequest("g", state.context.digest, "GREEDY", count=1),
                       expected_state="0" * 64)
    assert result.state == state
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "STALE"
    assert result.receipt.target_steps == ()


def test_run_speculative_stale_base_returns_failed_receipt(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    drafter = make_drafter(rt, state)
    req = InferenceRequest("g", state.context.digest, "GREEDY", count=2)
    result = run_speculative(rt, state, req, drafter, expected_state="0" * 64)
    assert result.state == state
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "STALE"


def test_replay_tampered_receipt_is_typed(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    committed = greedy(rt, state, count=2)
    bad_receipt = replace(committed.receipt, accepted_prefix=(99,))
    with pytest.raises(ContractError) as exc:
        rt.replay(state, InferenceRequest("g", state.context.digest, "GREEDY", count=2), bad_receipt)
    _assert_typed(exc.value)
    assert exc.value.code.value == "IDENTITY"
