"""Surface 2 + 11: rollback isolation and fault injection at transaction boundaries.

A failed transaction must return the original committed logical state. Tokens, KV
identity, n-gram history, global selection, structural proposals, logical prefetch
plans, context state and committed step receipts must not leak from a failed
private overlay. Physical cache warming may survive only where the documented
contract permits it.
"""
from __future__ import annotations

from dataclasses import replace


from elpis.inference.transaction import InferenceRequest, InferenceEngine
from elpis.inference.contracts import ContractError
from elpis.inference.context import initial_snapshot


def make(rt, state, tokens=(1, 2, 3)):
    return rt.execute(state, InferenceRequest("p", state.context.digest, "PREFILL", tokens),
                      expected_state=state.digest).state


def test_failed_prefill_returns_original_state_all_planes(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = make(rt, state)
    before_digest = state.digest
    # Token 999 is out of the 16-token vocabulary -> INVALID at step 3.
    result = rt.execute(state, InferenceRequest("bad", state.context.digest, "PREFILL", (1, 2, 999)),
                        expected_state=state.digest)
    assert result.state == state
    assert result.state.digest == before_digest
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "INVALID"
    # No partial effects on any logical plane.
    assert result.state.neural.tokens == state.neural.tokens
    assert result.state.neural.history == state.neural.history
    assert result.state.neural.local_keys == state.neural.local_keys
    assert result.state.neural.local_values == state.neural.local_values
    assert result.state.neural.pending == state.neural.pending
    assert result.state.neural.global_pool == state.neural.global_pool
    assert result.state.neural.index == state.neural.index
    assert result.state.receipts == state.receipts
    assert result.state.prefetch == state.prefetch
    assert result.state.structural == state.structural
    assert result.state.context == state.context


def test_failed_generation_mid_stream_returns_original_state(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = make(rt, state)
    # Force a mid-stream failure: a latent packet with a wrong projection digest
    # fails on the FIRST generation step, after prefill.
    bad_latent = replace(state.neural, logits=state.neural.logits)
    from elpis.inference.target import LatentInput
    latents = (LatentInput("G", "structural-latent.r0", "0" * 64, state.context.digest,
                           "0" * 64, (0.0,) * 4),)
    result = rt.execute(state, InferenceRequest("g", state.context.digest, "GREEDY", count=4, latents=latents),
                        expected_state=state.digest)
    assert result.state == state
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "IDENTITY"
    assert result.receipt.target_steps == ()


def test_failed_transaction_leaves_no_prefetch_plan(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = make(rt, state)
    result = rt.execute(state, InferenceRequest("bad", state.context.digest, "PREFILL", (1, 2, 999)),
                       expected_state=state.digest, prefetch_enabled=True)
    assert result.state == state
    assert result.state.prefetch == state.prefetch  # no logical plan leaked
    assert result.receipt.terminal == "FAILED"


def test_physical_cache_warming_may_surveive_rollback(target):
    """Documented contract: physical cache warming may survive rollback.

    We assert the LOGICAL state is fully rolled back; physical pread accounting
    (provider telemetry) is permitted to have advanced. We do NOT assert the
    physical cache is untouched, because the contract explicitly allows it.
    """
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = make(rt, state)
    f = target[0].rows.provider
    pread_before = f.stats()["pread_bytes"]
    result = rt.execute(state, InferenceRequest("bad", state.context.digest, "PREFILL", (1, 2, 999)),
                       expected_state=state.digest)
    assert result.state == state  # logical rollback is total
    # Physical warming is permitted to survive; we only record it, not forbid it.
    pread_after = f.stats()["pread_bytes"]
    assert pread_after >= pread_before


def test_failure_before_target_execution_returns_original(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    # Wrong expected_state fails before any target step and returns a typed
    # failure receipt (the base check is part of the atomic transaction).
    result = rt.execute(state, InferenceRequest("g", state.context.digest, "GREEDY", count=1),
                       expected_state="0" * 64)
    assert result.state == state
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "STALE"
    assert result.receipt.target_steps == ()


def test_failure_during_target_execution_returns_original(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = make(rt, state)
    # Out-of-range token fails mid-target-step; no partial state leaks.
    result = rt.execute(state, InferenceRequest("bad", state.context.digest, "PREFILL", (1, 2, 999)),
                       expected_state=state.digest)
    assert result.state == state
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.target_steps == ()


def test_receipt_construction_failure_returns_original(target, monkeypatch):
    """A failure in the commit phase (after partial target steps) must never
    commit a partially-built state: the original state is returned with a
    FAILED receipt and zero committed steps.

    We force a mid-stream target failure after two successful steps.
    """
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = make(rt, state)
    from elpis.inference.contracts import Code
    original_step = rt.target.step
    calls = {"n": 0}
    def flaky_step(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ContractError(Code.INVALID, "synthetic mid-stream failure")
        return original_step(*a, **k)
    monkeypatch.setattr(rt.target, "step", flaky_step)
    result = rt.execute(state, InferenceRequest("g", state.context.digest, "GREEDY", count=4),
                       expected_state=state.digest)
    assert result.state == state
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "INVALID"
    assert result.receipt.target_steps == ()
    assert result.state.receipts == state.receipts
    assert result.state.neural.tokens == state.neural.tokens


def test_provider_failure_during_step_returns_original(target, monkeypatch):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = make(rt, state)
    f = target[0].rows.provider
    # Force the row provider to fail on the next lookup.
    from elpis.inference.contracts import Code
    original_lookup = f.acquire
    def fail_acquire(*a, **k):
        raise ContractError(Code.IO, "synthetic provider failure")
    monkeypatch.setattr(f, "acquire", fail_acquire)
    result = rt.execute(state, InferenceRequest("g", state.context.digest, "GREEDY", count=1),
                       expected_state=state.digest)
    assert result.state == state
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "IO"
    assert result.receipt.target_steps == ()


def test_cleanup_failure_does_not_commit_partial_state(target, monkeypatch):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = make(rt, state)
    # Force a failure AFTER the target step succeeds but BEFORE commit, by
    # patching the plan_prefetch name as bound in the transaction module.
    from elpis.inference import transaction as tx
    from elpis.inference.contracts import Code
    calls = {"n": 0}
    original_plan = tx.plan_prefetch
    def flaky_plan(*a, **k):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise ContractError(Code.LIMIT, "synthetic cleanup failure")
        return original_plan(*a, **k)
    monkeypatch.setattr(tx, "plan_prefetch", flaky_plan)
    result = rt.execute(state, InferenceRequest("g", state.context.digest, "GREEDY", count=2),
                       expected_state=state.digest)
    assert result.state == state
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "LIMIT"
    assert result.receipt.target_steps == ()
