"""Surface 7 + 12: FMS/file-provider failure behavior and resource cleanup.

Where the engine touches the provider (prefetch path and row lookup), short reads,
integrity mismatch, missing assets, lease failures, HOT/WARM/COLD transitions,
immutable COLD assets, and resource cleanup must behave correctly. Repeated
success/failure/speculative cycles must not monotonically leak file descriptors,
leases, temporary files, or logical cache ownership.

We do NOT pretend buffered Linux page cache is measured device I/O, and we do
not claim accelerator support (the CPU POSIX profile has none).

Invariant note: the provider's pin/page counters are cumulative across the
fixture's lifetime (prefill already warmed pages). The correct no-leak invariant
is "no net change around a single acquire/release pair", not "returns to a
pre-prefill baseline".
"""
from __future__ import annotations

import pytest

from elpis.inference.transaction import InferenceRequest, InferenceEngine
from elpis.inference.speculative import run_speculative
from elpis.inference.contracts import Code, ContractError
from elpis.inference.context import initial_snapshot
from elpis.substrate.file_assets import inspect_asset
from elpis.substrate.synthetic import SyntheticFileAssets as FMSFileAssets

from ._helpers import build_runtime, greedy, make_drafter, prefill


def test_provider_failure_during_demand_fails_closed(target, monkeypatch):
    """A provider failure in the demand path (row lookup) must fail the
    transaction closed and return the original state (no partial commit)."""
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    f = target[0].rows.provider
    original = f.acquire
    calls = {"n": 0}
    def fail_once(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ContractError(Code.IO, "synthetic demand provider failure")
        return original(*a, **k)
    monkeypatch.setattr(f, "acquire", fail_once)
    result = rt.execute(state, InferenceRequest("g", state.context.digest, "GREEDY", count=1),
                        expected_state=state.digest)
    # The demand-path acquire fails -> transaction fails closed.
    assert result.state == state
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "IO"
    assert result.receipt.target_steps == ()


def test_prefetch_with_catalog_is_physical_warming_only(target):
    """A real prefetch catalog exercises execute_prefetch. Prefetch is physical
    warming only: it must not change the committed logical state (neural,
    receipts, or the logical prefetch plan) versus a run with prefetch disabled.
    Only the telemetry (prefetch_results) differs."""
    from elpis.inference.prefetch import RangeHint
    from elpis.inference.structural import AddressProposal
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    f = target[0].rows.provider
    asset = target[0].rows.table.asset
    bank = target[0].rows.table.bank.digest
    hint = RangeHint(asset, 0, 4, 1, 10**9, asset)
    proposal = AddressProposal("0" * 64, "0" * 64, "0" * 64, state.context.digest, "0" * 64,
                               "route", (bank,), (), (), 0.5, ())
    # Same catalog runtime for both runs so the logical prefetch plan is identical.
    rt2 = InferenceEngine(target[0], prefetch_catalog={bank: (hint,)})
    state2 = rt2.initial(state.context)
    state2 = rt2.execute(state2, InferenceRequest("p", state.context.digest, "PREFILL", (1, 2, 3)),
                         expected_state=state2.digest).state
    req = InferenceRequest("g", state.context.digest, "GREEDY", count=2, proposals=(proposal,))
    off = rt2.execute(state2, req, expected_state=state2.digest, prefetch_enabled=False)
    on = rt2.execute(state2, req, expected_state=state2.digest, prefetch_enabled=True)
    # Prefetch is physical warming only; the full logical state is identical.
    assert off.state == on.state
    assert off.state.neural == on.state.neural
    assert off.state.receipts == on.state.receipts
    assert off.state.prefetch == on.state.prefetch
    # The prefetch telemetry recorded the hint execution in the enabled run.
    assert any(m.get("prefetch") for m in on.telemetry)
    assert not any(m.get("prefetch") for m in off.telemetry)


def test_missing_asset_in_demand_fails_closed(target, monkeypatch):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    f = target[0].rows.provider
    def missing(*a, **k):
        raise ContractError(Code.MISSING, "synthetic missing asset")
    monkeypatch.setattr(f, "acquire", missing)
    result = rt.execute(state, InferenceRequest("g", state.context.digest, "GREEDY", count=1),
                        expected_state=state.digest)
    assert result.state == state
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "MISSING"
    assert result.receipt.target_steps == ()


def test_cold_tier_is_not_addressable(target):
    """COLD is external storage, not an addressable lease."""
    f = target[0].rows.provider
    with pytest.raises(ContractError) as exc:
        f.acquire("0" * 64, 0, 1, tier="COLD")
    assert exc.value.code.value in ("MISSING", "UNSUPPORTED")


def test_hot_warm_tier_transitions(target):
    """HOT/WARM transitions: a HOT request on a FOLD_DOWN provider folds to WARM."""
    f = target[0].rows.provider
    asset = target[0].rows.table.asset
    with f.acquire(asset, 0, 4, tier="HOT") as lease:
        assert lease.actual_tier in ("HOT", "WARM")
        assert lease.read() is not None


def test_lease_release_no_net_pin_change(target):
    """A single acquire/release pair must leave the pinned byte count unchanged
    and allow evict() to succeed (no active lease retained)."""
    f = target[0].rows.provider
    asset = target[0].rows.table.asset
    before = f.stats()["pinned"]
    with f.acquire(asset, 0, 4) as lease:
        assert f.stats()["pinned"] > before  # page pinned while held
        lease.read()
    assert f.stats()["pinned"] == before  # released
    f.evict()  # succeeds only if no active lease remains
    assert f.stats()["pages"] == 0


def test_repeated_cycles_no_monotonic_pin_leak(target):
    """Repeated success/failure/speculative cycles must not monotonically leak
    pinned leases. Pages may warm (permitted) but pins must not grow."""
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    f = target[0].rows.provider
    drafter = make_drafter(rt, state)
    baseline_pins = f.stats()["pinned"]
    for _ in range(6):
        state = greedy(rt, state, count=2).state
        rt.execute(state, InferenceRequest("bad", state.context.digest, "PREFILL", (1, 999)),
                   expected_state=state.digest)
        run_speculative(rt, state, InferenceRequest("g", state.context.digest, "GREEDY", count=2),
                        drafter, expected_state=state.digest, block_size=2)
    # All leases released across the cycles; no monotonic pin leak.
    assert f.stats()["pinned"] == baseline_pins
    # Pages bounded by max_pages.
    assert f.stats()["pages"] <= 1024


def test_provider_close_rejects_further_use(target, tmp_path, fms_file_library, native_workspace):
    """A closed provider rejects further use with CLOSED."""
    f = FMSFileAssets(root=native_workspace, library=fms_file_library,
                      warm_bytes=64, staging_bytes=128)
    path = tmp_path / "asset2.dat"
    path.write_bytes(bytes(range(128)))
    m = inspect_asset(native_workspace, path, 16)
    asset = f.register(path, m, expected_manifest=m.digest)
    f.close()
    with pytest.raises(ContractError) as exc:
        f.acquire(asset, 0, 4)
    assert exc.value.code.value == "CLOSED"


def test_no_accelerator_fence(target):
    """The CPU POSIX profile has no accelerator fences."""
    f = target[0].rows.provider
    asset = target[0].rows.table.asset
    with f.acquire(asset, 0, 4) as lease:
        with pytest.raises(ContractError) as exc:
            lease.bind_fence(object())
        assert exc.value.code.value == "UNSUPPORTED"


def test_buffered_page_cache_not_accounted(target):
    """Reported read bytes are pread-returned bytes, not measured device I/O."""
    f = target[0].rows.provider
    stats = f.stats()
    assert stats["buffered_page_cache_accounted"] is False
