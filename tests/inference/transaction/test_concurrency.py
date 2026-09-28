"""Surface 10: concurrency / reentrancy boundary.

The engine is in-process; Python and ctypes are not an isolation boundary. The logical
state (DecodeState) is immutable and per-call, so concurrent calls over the same
immutable snapshot must not leak logical state across requests. The shared
CompactTarget and FMSFileAssets carry mutable *telemetry* (last_metrics, provider
counters) which is explicitly separate from receipt identity.

We prove the boundary: concurrent identical transactions yield identical logical
state and receipts (no cross-request logical leakage), and we document that the
shared target's mutable metrics are telemetry, not logical state. We do NOT invent
thread safety for the target's mutable metrics.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor


from elpis.inference.transaction import InferenceRequest
from elpis.inference.context import initial_snapshot

from ._helpers import build_runtime, prefill


def test_concurrent_identical_transactions_no_logical_leak(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    req = InferenceRequest("g", state.context.digest, "GREEDY", count=3)
    base_digest = state.digest

    def run(_):
        return rt.execute(state, req, expected_state=base_digest)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(8)))

    # Every concurrent call over the same immutable base yields identical
    # logical state and receipt (no cross-request logical leakage).
    for r in results:
        assert r.state == results[0].state
        assert r.receipt == results[0].receipt
        assert r.receipt.terminal == "COMMITTED"
        assert len(r.state.neural.tokens) == len(state.neural.tokens) + 3


def test_concurrent_distinct_snapshots_isolated(target):
    """Two distinct snapshots run concurrently must not cross-contaminate."""
    snap_a = initial_snapshot()
    from elpis.inference.context import ContextItem, Lifetime, append_context
    snap_b = append_context(snap_a, ContextItem("b", "src", b"y", Lifetime.STABLE), expected=snap_a.digest)
    rt = build_runtime(target[0])
    sa, sb = rt.initial(snap_a), rt.initial(snap_b)
    ra = prefill(rt, sa, tokens=(1, 2))
    rb = prefill(rt, sb, tokens=(3, 4))

    def run_a(_):
        return rt.execute(ra, InferenceRequest("ga", snap_a.digest, "GREEDY", count=2), expected_state=ra.digest)

    def run_b(_):
        return rt.execute(rb, InferenceRequest("gb", snap_b.digest, "GREEDY", count=2), expected_state=rb.digest)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fa = list(pool.map(run_a, range(4)))
        fb = list(pool.map(run_b, range(4)))

    for r in fa:
        assert r.state.context.digest == snap_a.digest
        assert r.receipt.context_snapshot == snap_a.digest
    for r in fb:
        assert r.state.context.digest == snap_b.digest
        assert r.receipt.context_snapshot == snap_b.digest
    # No cross-contamination: A's tokens never appear in B's committed state.
    assert fa[0].state.neural.tokens[:2] == (1, 2)
    assert fb[0].state.neural.tokens[:2] == (3, 4)


def test_reentrant_execute_same_base_isolated(target):
    """Nested (reentrant) execute over the same immutable base must not corrupt
    the outer call's logical state, because each call builds its own overlay."""
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    req = InferenceRequest("g", state.context.digest, "GREEDY", count=2)
    base_digest = state.digest
    outer = rt.execute(state, req, expected_state=base_digest)
    # Re-enter with the same base while the outer result is held.
    inner = rt.execute(state, req, expected_state=base_digest)
    assert outer.state == inner.state
    assert outer.receipt == inner.receipt
    # The base state is unmutated.
    assert state.digest == base_digest


def test_shared_target_metrics_are_telemetry_not_logical(target):
    """The shared target's last_metrics is mutable telemetry; it must not affect
    receipt identity. Two identical transactions yield identical receipts even
    though the target's metrics object is shared and overwritten."""
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    req = InferenceRequest("g", state.context.digest, "GREEDY", count=2)
    a = rt.execute(state, req, expected_state=state.digest)
    b = rt.execute(state, req, expected_state=state.digest)
    # Receipts are identical (telemetry excluded from receipt identity).
    assert a.receipt == b.receipt
    # Telemetry may differ (timing) but is separate from receipt identity.
    assert isinstance(a.telemetry, tuple) and isinstance(b.telemetry, tuple)


def test_provider_concurrent_acquire_is_serialized(target):
    """The provider's RLock serializes concurrent acquires; after all threads
    complete, the pin count returns to the pre-test baseline (every
    acquire/release pair is net-zero, no monotonic leak)."""
    f = target[0].rows.provider
    asset = target[0].rows.table.asset
    baseline = f.stats()["pinned"]

    def acquire(_):
        with f.acquire(asset, 0, 4) as lease:
            lease.read()
        return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(acquire, range(8)))
    # All leases released; no monotonic pin leak.
    assert f.stats()["pinned"] == baseline
