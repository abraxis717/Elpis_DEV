"""YTS-R0 failure semantics: every message boundary, FMS failure, cancellation, release paths.

Three frozen classes: a failure before TOKEN_BEGIN releases the stream normally; a
host-originated failure after TOKEN_BEGIN discards the stream but keeps the model; a
provider-originated failure quarantines the runtime. There is never CPU arithmetic for
a provider sequence, and every terminal path releases leases, buffers and provider
resources.
"""
from __future__ import annotations

import threading

import pytest

from elpis.inference.contracts import Code, ContractError

from . import provider_harness as H
from .test_provider_stream import assert_same_run, balanced, forbid_host_arithmetic, run


@pytest.fixture
def rig(native_workspace, v41):
    rig = H.make_rig(native_workspace, v41)
    yield rig
    rig.close()


def no_pins(fms):
    return all(count == 0 for count in fms._pins.values())


def message_map(rig, tokens):
    """Request indices per token of a clean run (admission, then OPEN, then tokens)."""
    provider = rig.provider()
    target = rig.provider_target(rig.fms(), "map", provider)
    first = provider.totals["admission_submissions"] + 1
    state, snaps = run(target, tokens)
    target.release_window(state)
    provider.close()
    spans, at = [], first
    for snap in snaps:
        spans.append(range(at, at + snap["metrics"]["submissions"]))
        at += snap["metrics"]["submissions"]
    return snaps, spans, at  # `at` is the STREAM_RELEASE index


def faulted_run(rig, monkeypatch, tokens, kind, index, *, poll_limit=1000, mode=H.MODE_SYNC):
    rig.configure(mode=mode, notify=1, fault_kind=kind, fault_message=index)
    fms = rig.fms()
    provider = rig.provider(poll_limit=poll_limit)
    target = rig.provider_target(fms, f"fault-{kind}-{index}-{mode}", provider)
    rig.configure()
    with monkeypatch.context() as patch:
        forbid_host_arithmetic(patch)
        state = target.window_initial()
        experts = target.admit_stream()
        error = None
        try:
            for token in tokens:
                target.window_step(state, token, experts=experts)
        except ContractError as exc:
            error = exc
        target.release_window(state)
    return error, provider, target, fms


# ---------------------------------------------------------------------------
# Q5 + Q13: a provider fault at every message boundary of a token
# ---------------------------------------------------------------------------

def test_q5_q13_provider_fault_at_every_message_of_a_token(rig, monkeypatch):
    tokens = rig.tokens(4)
    _, spans, _ = message_map(rig, tokens)
    span = spans[2]  # third token: TOKEN_BEGIN, 21 supplies; the last supply's reply is COMPLETE
    assert len(span) == 22
    begin, first_part, middle, last = span[0], span[1], span[11], span[-1]
    cases = [(H.FAULT_POLL_FAIL, i, Code.DEVICE) for i in span]
    for i in (begin, first_part, middle, last):
        cases += [(H.FAULT_SUBMIT_REJECT, i, Code.DEVICE), (H.FAULT_BAD_ECHO, i, Code.INTEGRITY)]
    cases += [(H.FAULT_POLL_TIMEOUT, i, Code.DEVICE) for i in (begin, middle, last)]
    cases += [(H.FAULT_BAD_NEED, begin, Code.INTEGRITY), (H.FAULT_BAD_NEED, middle, Code.INTEGRITY),
              (H.FAULT_BAD_CURSOR, first_part, Code.INTEGRITY), (H.FAULT_NONFINITE, last, Code.ENCODING)]
    outcomes = []
    for kind, index, code in cases:
        aborts = rig.counters()["aborts"]
        error, provider, target, fms = faulted_run(rig, monkeypatch, tokens, kind, index,
                                                   poll_limit=3 if kind == H.FAULT_POLL_TIMEOUT else 1000)
        assert error is not None and error.code == code, (kind, index, error)
        assert provider.state == "QUARANTINED" and provider.failure[0] == code
        assert provider.final_metrics["backend_fallback"] == 0
        assert all(v == 0 for v in rig.live().values()), (kind, index, rig.live())
        assert balanced(provider) and no_pins(fms) and target.store.staged_bytes == 0
        if kind == H.FAULT_POLL_TIMEOUT:
            assert rig.counters()["aborts"] == aborts + 1
        with pytest.raises(ContractError):  # no CPU continuation, no silent fallback
            target.window_initial()
        outcomes.append((kind, index - begin, code.value))
    print(f"PASS_YTS_Q5 {len(outcomes)} provider faults across token messages 0..21: quarantined, released")


def test_q5_provider_faults_in_async_device_mode(rig, monkeypatch):
    tokens = rig.tokens(3)
    _, spans, _ = message_map(rig, tokens)
    span = spans[1]
    for kind, index, code in ((H.FAULT_POLL_FAIL, span[0], Code.DEVICE), (H.FAULT_POLL_FAIL, span[-1], Code.DEVICE),
                              (H.FAULT_SUBMIT_REJECT, span[5], Code.DEVICE),
                              (H.FAULT_POLL_TIMEOUT, span[7], Code.DEVICE)):
        # The timed-out request never completes; 20 polls keep every other async request inside budget.
        error, provider, target, fms = faulted_run(rig, monkeypatch, tokens, kind, index, mode=H.MODE_DEVICE,
                                                   poll_limit=20 if kind == H.FAULT_POLL_TIMEOUT else 1000)
        assert error is not None and error.code == code
        assert provider.state == "QUARANTINED" and provider.final_metrics["backend_fallback"] == 0
        assert all(v == 0 for v in rig.live().values()) and balanced(provider) and no_pins(fms)


def test_q14_device_step_beyond_poll_budget_aborts_and_quarantines(rig, monkeypatch):
    """A 2 s device step against a 20-poll budget: timeout, synchronous abort, quarantine.

    Only the faulted request is slow; the budget is wide enough that every other request
    (admission parts included) completes well inside it even on a loaded host.
    """
    import time
    tokens = rig.tokens(2)
    _, spans, _ = message_map(rig, tokens)
    rig.configure(mode=H.MODE_DEVICE, notify=1, delay_us=2_000_000, fault_kind=H.FAULT_SLOW,
                  fault_message=spans[1][0])
    fms = rig.fms()
    provider = rig.provider(poll_limit=20)
    target = rig.provider_target(fms, "slow", provider)
    rig.configure()
    aborts = rig.counters()["aborts"]
    with monkeypatch.context() as patch:
        forbid_host_arithmetic(patch)
        state = target.window_initial()
        experts = target.admit_stream()
        target.window_step(state, tokens[0], experts=experts)
        begin = time.monotonic()
        with pytest.raises(ContractError) as info:
            target.window_step(state, tokens[1], experts=experts)
        elapsed = time.monotonic() - begin
    assert info.value.code == Code.DEVICE and "BACKEND_UNAVAILABLE" in str(info.value)
    assert rig.counters()["aborts"] == aborts + 1
    assert elapsed < 1.0  # abort quiesced the 2 s device step instead of waiting it out
    assert provider.state == "QUARANTINED" and provider.final_metrics["backend_fallback"] == 0
    target.release_window(state)
    assert all(v == 0 for v in rig.live().values()) and balanced(provider) and no_pins(fms)


# ---------------------------------------------------------------------------
# Release paths
# ---------------------------------------------------------------------------

def test_release_path_faults_quarantine_without_raising(rig, monkeypatch):
    tokens = rig.tokens(2)
    _, _, release_index = message_map(rig, tokens)
    # STREAM_RELEASE fails: release_window must not raise; the runtime is quarantined.
    rig.configure(fault_kind=H.FAULT_POLL_FAIL, fault_message=release_index)
    provider = rig.provider()
    rig.configure()
    fms = rig.fms()
    target = rig.provider_target(fms, "release", provider)
    state, _ = run(target, tokens)
    target.release_window(state)
    assert provider.state == "QUARANTINED" and provider.failure[0] == Code.DEVICE
    assert all(v == 0 for v in rig.live().values()) and balanced(provider) and no_pins(fms)
    # MODEL_RELEASE fails: close() quarantines and never raises.
    rig.configure(fault_kind=H.FAULT_POLL_FAIL, fault_message=release_index + 1)
    provider = rig.provider()
    rig.configure()
    target = rig.provider_target(rig.fms(), "model-release", provider)
    state, _ = run(target, tokens)
    target.release_window(state)
    assert provider.state == "READY" and rig.live()["streams"] == 0
    provider.close()
    assert provider.state == "QUARANTINED"
    assert all(v == 0 for v in rig.live().values()) and balanced(provider)


# ---------------------------------------------------------------------------
# Q4: FMS integrity failure while parked: discard the stream, keep the model
# ---------------------------------------------------------------------------

def test_q4_fms_integrity_failure_after_begin_discards_stream_and_keeps_model(rig, monkeypatch):
    fms = rig.fms()
    provider = rig.provider()
    target = rig.provider_target(fms, "integrity", provider)
    state = target.window_initial()
    experts = target.admit_stream()
    tokens = rig.tokens(5)
    for token in tokens[:3]:
        target.window_step(state, token, experts=experts)
    asset = next(b.source.asset for n, b in target.store.bindings.items() if ".ffn.experts." in n)
    fms.evict(asset)
    H.flip_expert_bytes(rig.workspace / "integrity" / "fixture-experts.dat", 0, 64)
    before = rig.counters()
    with monkeypatch.context() as patch:
        forbid_host_arithmetic(patch)
        with pytest.raises(ContractError) as info:
            target.window_step(state, tokens[3], experts=experts)
    assert info.value.code == Code.INTEGRITY
    after = rig.counters()
    # TOKEN_BEGIN advanced the provider, the first staging failed, the stream was discarded.
    assert after["messages"] - before["messages"] == 2
    assert state.provider.state == "DISCARDED" and provider.state == "READY"
    assert provider.totals["discards"] == 1 and provider.totals["quarantines"] == 0
    live = rig.live()
    assert live["models"] == 1 and live["streams"] == 0 and live["slots"] == 1
    assert no_pins(fms) and target.store.staged_bytes == 0 and balanced(provider)
    with pytest.raises(ContractError) as info:  # the discarded stream never continues
        target.window_step(state, tokens[3], experts=experts)
    assert info.value.code == Code.STALE
    target.release_window(state)
    provider.close()
    assert all(v == 0 for v in rig.live().values())


# ---------------------------------------------------------------------------
# Q3: transient FMS pressure retries between submissions; persistent pressure discards
# ---------------------------------------------------------------------------

def _pressure(fms, monkeypatch, failures):
    real = fms.acquire
    state = dict(active=False, left=failures, raised=0)

    def acquire(*args, **kwargs):
        if state["active"] and state["left"] != 0:
            state["left"] -= 1
            state["raised"] += 1
            raise ContractError(Code.LIMIT, "injected FMS headroom pressure")
        return real(*args, **kwargs)

    monkeypatch.setattr(fms, "acquire", acquire)
    return state


def test_q3_transient_fms_limit_retries_and_persistent_limit_discards(rig, monkeypatch):
    tokens = rig.tokens(6)
    provider = rig.provider()
    target = rig.provider_target(rig.fms(), "baseline", provider)
    state, baseline = run(target, tokens)
    target.release_window(state)
    provider.close()

    fms = rig.fms()
    provider = rig.provider(poll_limit=3)
    target = rig.provider_target(fms, "transient", provider)
    pressure = _pressure(fms, monkeypatch, failures=-1)
    state = target.window_initial()

    def observer(event, info):
        if event == "STAGING":
            pressure["active"], pressure["left"] = True, 3  # three LIMITs before every supply succeeds
        elif event in ("PARKED", "COMPLETE"):
            pressure["active"] = False

    state.provider.observer = observer
    _, ours = run(target, tokens, state)
    assert_same_run(baseline, ours)
    assert pressure["raised"] == 3 * 21 * len(tokens)
    assert all(s["metrics"]["polls"] == s["metrics"]["submissions"] for s in ours)
    target.release_window(state)

    # Persistent pressure: the host deadline expires; discard, keep the model, start again.
    provider.staging_deadline_s = 0.05
    state = target.window_initial()
    experts = target.admit_stream()
    target.window_step(state, tokens[0], experts=experts)

    def persistent(event, info):
        if event == "STAGING":  # Engram rows (before TOKEN_BEGIN) are unaffected
            pressure["active"], pressure["left"] = True, -1

    state.provider.observer = persistent
    with monkeypatch.context() as patch:
        forbid_host_arithmetic(patch)
        with pytest.raises(ContractError) as info:
            target.window_step(state, tokens[1], experts=experts)
    assert info.value.code == Code.LIMIT
    assert state.provider.state == "DISCARDED" and provider.state == "READY"
    assert no_pins(fms) and target.store.staged_bytes == 0
    target.release_window(state)
    pressure["active"] = False
    state, again = run(target, tokens)  # the retained admission serves a fresh stream
    assert_same_run(baseline, again)
    target.release_window(state)
    provider.close()
    assert balanced(provider) and all(v == 0 for v in rig.live().values())


# ---------------------------------------------------------------------------
# Q6: cancellation in every state
# ---------------------------------------------------------------------------

def _fresh_first_token_matches(rig, target, baseline_first):
    state, snaps = run(target, rig.tokens(1))
    assert H.bitwise_equal(snaps[0]["logits"], baseline_first["logits"])
    target.release_window(state)


def test_q6_cancellation_in_every_state(rig, monkeypatch):
    tokens = rig.tokens(4)
    provider = rig.provider()
    target = rig.provider_target(rig.fms(), "baseline", provider)
    state, baseline = run(target, tokens)
    target.release_window(state)
    provider.close()

    # BOUNDARY: a stop between tokens is a normal end; finalize-style release.
    fms = rig.fms()
    provider = rig.provider()
    target = rig.provider_target(fms, "cancel", provider)
    state, snaps = run(target, tokens[:2])
    state.provider.cancel()
    target.release_window(state)
    assert state.provider is None and provider.totals["releases"] == 1 and provider.totals["discards"] == 0

    # PREPARING: nothing was sent; the stream is released normally.
    state = target.window_initial()
    experts = target.admit_stream()
    stream = state.provider
    stream.cancel()
    before = rig.counters()["messages"]
    with pytest.raises(ContractError) as info:
        target.window_step(state, tokens[0], experts=experts)
    assert info.value.code == Code.CLOSED and stream.state == "RELEASED"
    assert rig.counters()["messages"] - before == 1  # STREAM_RELEASE only, no TOKEN_BEGIN
    target.release_window(state)

    # PARKED and STAGING (between submissions, after TOKEN_BEGIN): discard.
    for event in ("PARKED", "STAGING"):
        state = target.window_initial()
        experts = target.admit_stream()
        target.window_step(state, tokens[0], experts=experts)
        stream = state.provider

        def observer(name, info, stream=stream, event=event):
            if name == event:
                stream.cancel()

        stream.observer = observer
        with monkeypatch.context() as patch:
            forbid_host_arithmetic(patch)
            with pytest.raises(ContractError) as info:
                target.window_step(state, tokens[1], experts=experts)
        assert info.value.code == Code.CLOSED and stream.state == "DISCARDED", event
        assert provider.state == "READY" and rig.live()["streams"] == 0 and rig.live()["models"] == 1
        assert no_pins(fms) and target.store.staged_bytes == 0
        target.release_window(state)
        _fresh_first_token_matches(rig, target, baseline[0])
    provider.close()
    assert balanced(provider) and all(v == 0 for v in rig.live().values())

    # IN_FLIGHT: a parked port token cannot be cancelled; the request takes effect at retirement.
    rig.configure(mode=H.MODE_DEVICE, notify=1, delay_us=30_000)
    provider = rig.provider()
    target = rig.provider_target(rig.fms(), "inflight", provider)
    state = target.window_initial()
    experts = target.admit_stream()
    stream = state.provider
    timer = threading.Timer(0.01, stream.cancel)

    def start_timer(name, info):
        if name == "IN_FLIGHT":
            timer.start()

    stream.observer = start_timer
    with pytest.raises(ContractError) as info:
        target.window_step(state, tokens[0], experts=experts)
    timer.join()
    rig.configure()
    assert info.value.code == Code.CLOSED and stream.state == "DISCARDED" and provider.state == "READY"
    target.release_window(state)
    provider.close()
    assert balanced(provider) and all(v == 0 for v in rig.live().values())


# ---------------------------------------------------------------------------
# Q9: an Engram row failure happens before TOKEN_BEGIN: normal release
# ---------------------------------------------------------------------------

def test_q9_engram_row_failure_before_token_begin_releases_normally(rig, monkeypatch):
    provider = rig.provider()
    target = rig.provider_target(rig.fms(), "rows", provider)
    state = target.window_initial()
    experts = target.admit_stream()
    tokens = rig.tokens(3)
    target.window_step(state, tokens[0], experts=experts)
    engine = target.rows[5]

    def broken(requests):
        raise ContractError(Code.IO, "injected Engram row I/O failure")

    monkeypatch.setattr(engine, "lookup", broken)
    before = rig.counters()["messages"]
    with pytest.raises(ContractError) as info:
        target.window_step(state, tokens[1], experts=experts)
    assert info.value.code == Code.IO
    assert rig.counters()["messages"] - before == 1  # STREAM_RELEASE; TOKEN_BEGIN never sent
    assert state.provider.state == "RELEASED" and provider.totals["discards"] == 0
    assert provider.state == "READY" and rig.live()["streams"] == 0
    target.release_window(state)
    provider.close()
    assert balanced(provider) and all(v == 0 for v in rig.live().values())
