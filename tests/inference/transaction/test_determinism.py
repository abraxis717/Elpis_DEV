"""Surface 3 + 8: deterministic prefill/greedy equivalence and long-sequence determinism.

Repeated runs over identical immutable snapshots/requests must produce identical
target-authoritative outputs and receipts. Long generated histories must preserve
receipt cardinality, chain integrity, state growth, and show no hidden dependence
on process-global mutable state.
"""
from __future__ import annotations



from elpis.inference.transaction import InferenceRequest
from elpis.inference.context import initial_snapshot

from ._helpers import build_runtime, greedy, prefill


def test_repeated_prefill_identical(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    req = InferenceRequest("p", state.context.digest, "PREFILL", (1, 2, 3, 4, 5))
    a = rt.execute(state, req, expected_state=state.digest)
    b = rt.execute(state, req, expected_state=state.digest)
    assert a.state == b.state
    assert a.receipt == b.receipt
    assert a.state.digest == b.state.digest


def test_repeated_greedy_identical(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    req = InferenceRequest("g", state.context.digest, "GREEDY", count=6)
    a = rt.execute(state, req, expected_state=state.digest)
    b = rt.execute(state, req, expected_state=state.digest)
    assert a.state == b.state
    assert a.receipt == b.receipt
    assert a.state.neural.tokens == b.state.neural.tokens


def test_independent_runtimes_same_snapshot_identical(target):
    """Two independent InferenceEngine instances over the same immutable snapshot and
    request must produce identical committed state and receipts."""
    snap = initial_snapshot()
    rt1, rt2 = build_runtime(target[0]), build_runtime(target[0])
    s1, s2 = rt1.initial(snap), rt2.initial(snap)
    assert s1 == s2
    req = InferenceRequest("p", snap.digest, "PREFILL", (1, 2, 3))
    a = rt1.execute(s1, req, expected_state=s1.digest)
    b = rt2.execute(s2, req, expected_state=s2.digest)
    assert a.state == b.state
    assert a.receipt == b.receipt
    g = InferenceRequest("g", snap.digest, "GREEDY", count=4)
    a2 = rt1.execute(a.state, g, expected_state=a.state.digest)
    b2 = rt2.execute(b.state, g, expected_state=b.state.digest)
    assert a2.state == b2.state
    assert a2.receipt == b2.receipt


def test_long_history_receipt_cardinality_and_chain(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state, tokens=(1, 2, 3))
    # Grow to a long history in chunks.
    for _ in range(6):
        state = greedy(rt, state, count=4).state
    n = len(state.neural.tokens)
    assert n == 3 + 24
    # Receipt cardinality matches token count.
    assert len(state.receipts) == n
    # Chain integrity: revalidate the full chain via a further transaction.
    again = greedy(rt, state, count=1)
    assert again.receipt.terminal == "COMMITTED"
    assert len(again.state.receipts) == n + 1
    assert len(again.state.neural.tokens) == n + 1


def test_long_history_state_growth_monotonic(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state, tokens=(1, 2, 3))
    sizes = [len(state.neural.tokens)]
    for _ in range(8):
        state = greedy(rt, state, count=3).state
        sizes.append(len(state.neural.tokens))
    assert sizes == sorted(sizes)
    assert sizes[-1] == 3 + 24
    # KV window is bounded by local_window; global pool grows with compression.
    assert len(state.neural.local_keys) <= rt.target.config.local_window


def test_no_hidden_process_global_state(target):
    """Repeated identical transactions must not drift due to process-global
    mutable state (e.g. target.last_metrics, provider telemetry)."""
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    req = InferenceRequest("g", state.context.digest, "GREEDY", count=3)
    first = rt.execute(state, req, expected_state=state.digest)
    # Run several intervening unrelated transactions to perturb global state.
    for _ in range(3):
        rt.execute(state, InferenceRequest("x", state.context.digest, "PREFILL", (7, 8)),
                  expected_state=state.digest)
    second = rt.execute(state, req, expected_state=state.digest)
    assert first.state == second.state
    assert first.receipt == second.receipt


def test_greedy_count_zero_is_noop_commit(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    result = rt.execute(state, InferenceRequest("g0", state.context.digest, "GREEDY", count=0),
                        expected_state=state.digest)
    assert result.state == state
    assert result.receipt.terminal == "COMMITTED"
    assert result.receipt.accepted_prefix == ()
    assert result.receipt.target_steps == ()
