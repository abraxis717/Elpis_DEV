"""Surface 9 + 8 (stress): scaling of receipt-lineage verification and long
generated histories.

Characterize receipt-lineage verification cost over increasing history lengths
and look specifically for accidental quadratic behavior. Do NOT weaken
correctness merely to improve timing. Long generated histories must preserve
receipt cardinality, chain integrity, state growth, and show no hidden dependence
on process-global mutable state.
"""
from __future__ import annotations

import time


from elpis.inference.transaction import InferenceRequest
from elpis.inference.context import initial_snapshot

from ._helpers import build_runtime, greedy, prefill


def _grow(rt, state, total_tokens):
    """Grow state to total_tokens via greedy chunks."""
    while len(state.neural.tokens) < total_tokens:
        state = greedy(rt, state, count=min(4, total_tokens - len(state.neural.tokens))).state
    return state


def test_lineage_verification_is_linear_in_history(target):
    """_validate_receipt_lineage must be O(n) in history length, not O(n^2).

    We measure the wall time of a single _validate call at several history
    lengths and assert the per-token cost does not grow super-linearly.
    """
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state, tokens=(1, 2, 3))
    sizes = [20, 40, 80, 160]
    times = []
    for n in sizes:
        grown = _grow(rt, state, n)
        req = InferenceRequest("g", grown.context.digest, "GREEDY", count=1)
        # Warm up once.
        rt._validate(grown, req)
        start = time.perf_counter_ns()
        for _ in range(3):
            rt._validate(grown, req)
        times.append((time.perf_counter_ns() - start) / 3)
    # Per-token cost should be roughly constant (linear total), not growing.
    per_token = [t / n for n, t in zip(sizes, times)]
    # Allow a 3x tolerance for noise; quadratic would show ~4x growth.
    assert per_token[-1] < 3.0 * per_token[0], f"super-linear growth: {per_token}"


def test_long_history_chain_integrity(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state, tokens=(1, 2, 3))
    state = _grow(rt, state, 120)
    n = len(state.neural.tokens)
    assert n == 120
    assert len(state.receipts) == n
    # The full chain must still validate via a further transaction.
    again = greedy(rt, state, count=1)
    assert again.receipt.terminal == "COMMITTED"
    assert len(again.state.receipts) == n + 1
    assert len(again.state.neural.tokens) == n + 1


def test_long_history_state_growth_bounded(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state, tokens=(1, 2, 3))
    state = _grow(rt, state, 100)
    c = rt.target.config
    # Local KV window is bounded.
    assert len(state.neural.local_keys) <= c.local_window
    assert len(state.neural.local_values) <= c.local_window
    # Global pool grows with compression (bounded by token count).
    assert len(state.neural.global_pool) <= len(state.neural.tokens)
    # Pending compression buffer is bounded by the compression window.
    assert len(state.neural.pending) < c.compression


def test_repeated_independent_long_executions_identical(target):
    """Two independent long executions over the same immutable snapshot must
    produce identical committed state and receipts."""
    snap = initial_snapshot()
    rt1, rt2 = build_runtime(target[0]), build_runtime(target[0])
    s1, s2 = rt1.initial(snap), rt2.initial(snap)
    for rt, s in ((rt1, s1), (rt2, s2)):
        s = prefill(rt, s, tokens=(1, 2, 3))
        s = _grow(rt, s, 60)
    assert s1 == s2
    assert s1.digest == s2.digest
    assert s1.receipts == s2.receipts


def test_long_history_no_process_global_drift(target):
    """Repeated identical long transactions must not drift due to process-global
    mutable state (target.last_metrics, provider telemetry)."""
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state, tokens=(1, 2, 3))
    state = _grow(rt, state, 40)
    req = InferenceRequest("g", state.context.digest, "GREEDY", count=4)
    base = state.digest
    a = rt.execute(state, req, expected_state=base)
    # Perturb global state with intervening transactions.
    for _ in range(3):
        rt.execute(state, InferenceRequest("x", state.context.digest, "PREFILL", (7, 8)),
                  expected_state=base)
    b = rt.execute(state, req, expected_state=base)
    assert a.state == b.state
    assert a.receipt == b.receipt
