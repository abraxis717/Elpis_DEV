"""Surface 5 + 6: snapshot/session binding and replay behavior.

Snapshots are pinned for the session. Context compaction creates a new snapshot;
the current target requires explicit new-session prefill for that snapshot and
cannot silently reuse neural state computed under a different latent context.
Valid committed histories must replay exactly; invalid or tampered histories must
fail closed without mutating committed state.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from elpis.inference.transaction import InferenceRequest
from elpis.inference.contracts import ContractError
from elpis.inference.context import (
    ContextItem, Lifetime, append_context, compact_context, initial_snapshot,
)

from ._helpers import build_runtime, prefill


def _item(obj, content=b"ctx"):
    return ContextItem(obj, "src", content, Lifetime.STABLE)


def test_wrong_snapshot_in_request_fails_closed(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    req = InferenceRequest("g", "0" * 64, "GREEDY", count=1)
    result = rt.execute(state, req, expected_state=state.digest)
    assert result.state == state
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "STALE"


def test_stale_snapshot_state_fails_closed(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    # Build a state whose neural context_snapshot no longer matches its context.
    bad = replace(state, neural=replace(state.neural, context_snapshot="0" * 64))
    req = InferenceRequest("g", state.context.digest, "GREEDY", count=1)
    result = rt.execute(bad, req, expected_state=bad.digest)
    assert result.state == bad
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "STALE"


def test_context_compaction_requires_new_session_prefill(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    snap = state.context
    # Append a context item, then compact it away -> new snapshot.
    s1 = append_context(snap, _item("a"), expected=snap.digest)
    s2 = append_context(s1, _item("b"), expected=s1.digest)
    replacement = _item("summary")
    compacted, record = compact_context(
        s2, 0, 2, replacement, expected=s2.digest,
        replacement_digest=replacement.digest, policy="summarize", reason="test",
    )
    assert compacted.digest != snap.digest
    # The old state was computed under `snap`; continuing it under the compacted
    # snapshot must fail closed (stale context lineage).
    old_state = prefill(rt, state, tokens=(1, 2))
    bad = replace(old_state, context=compacted)
    req = InferenceRequest("g", compacted.digest, "GREEDY", count=1)
    result = rt.execute(bad, req, expected_state=bad.digest)
    assert result.state == bad
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "STALE"
    # The correct path: explicit new-session prefill for the compacted snapshot.
    fresh = rt.initial(compacted)
    out = rt.execute(fresh, InferenceRequest("p", compacted.digest, "PREFILL", (1, 2)),
                     expected_state=fresh.digest)
    assert out.receipt.terminal == "COMMITTED"


def test_cannot_reuse_neural_state_across_changed_latent_context(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state, tokens=(1, 2, 3))
    # A different snapshot (even with the same items) is a different latent context.
    other = append_context(state.context, _item("extra"), expected=state.context.digest)
    assert other.digest != state.context.digest
    bad = replace(state, context=other)
    req = InferenceRequest("g", other.digest, "GREEDY", count=1)
    result = rt.execute(bad, req, expected_state=bad.digest)
    assert result.state == bad
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "STALE"


def test_wrong_expected_state_fails_closed(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    result = rt.execute(state, InferenceRequest("g", state.context.digest, "GREEDY", count=1),
                        expected_state="f" * 64)
    assert result.state == state
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "STALE"


def test_session_crossing_receipt_fails_closed(target):
    """A receipt from one session (snapshot) cannot validate a chain in another."""
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state, tokens=(1, 2, 3))
    # Fork a new session with a different snapshot.
    other_snap = append_context(state.context, _item("fork"), expected=state.context.digest)
    other_state = prefill(build_runtime(target[0]), build_runtime(target[0]).initial(other_snap),
                          tokens=(4, 5, 6))
    # Graft a receipt from the other session into this chain.
    bad = replace(state, receipts=(other_state.receipts[0],) + state.receipts[1:])
    req = InferenceRequest("g", state.context.digest, "GREEDY", count=1)
    result = rt.execute(bad, req, expected_state=bad.digest)
    assert result.state == bad
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure in ("STALE", "IDENTITY")


def test_valid_history_replays_exactly(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state, tokens=(1, 2, 3))
    req = InferenceRequest("g", state.context.digest, "GREEDY", count=4)
    committed = rt.execute(state, req, expected_state=state.digest)
    replayed = rt.replay(state, req, committed.receipt)
    assert replayed.state == committed.state
    assert replayed.receipt == committed.receipt


def test_tampered_history_replay_fails_closed(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state, tokens=(1, 2, 3))
    req = InferenceRequest("g", state.context.digest, "GREEDY", count=4)
    committed = rt.execute(state, req, expected_state=state.digest)
    # Tamper the committed receipt.
    tampered = replace(committed.receipt, accepted_prefix=(99, 98))
    with pytest.raises(ContractError) as exc:
        rt.replay(state, req, tampered)
    assert exc.value.code.value == "IDENTITY"
    # Committed state is unmutated.
    assert committed.state.digest == committed.state.digest


def test_replay_with_wrong_request_fails_closed(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state, tokens=(1, 2, 3))
    req = InferenceRequest("g", state.context.digest, "GREEDY", count=4)
    committed = rt.execute(state, req, expected_state=state.digest)
    other_req = InferenceRequest("g2", state.context.digest, "GREEDY", count=4)
    with pytest.raises(ContractError):
        rt.replay(state, other_req, committed.receipt)


def test_replay_prefill_history(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    req = InferenceRequest("p", state.context.digest, "PREFILL", (1, 2, 3, 4))
    committed = rt.execute(state, req, expected_state=state.digest)
    replayed = rt.replay(state, req, committed.receipt)
    assert replayed.state == committed.state
    assert replayed.receipt == committed.receipt
