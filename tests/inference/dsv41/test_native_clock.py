"""Native Clock R0 qualification against the unchanged Python YTS/Principal."""
import ctypes as C
from dataclasses import replace
import itertools
import sys
import time

import numpy as np
import pytest

from elpis.inference.associative import AddressScheme, DSV41Parameters, History
from elpis.inference.contracts import Code, ContractError, RowIdentity
from elpis.inference.drivers.dsv41.native_clock import NativeClock, NativeMaterializer, Metrics, U32, U64, VP, run_principal
from elpis.inference.drivers.dsv41 import stream_protocol as P
from elpis.inference.principal import PrincipalEngine, PrincipalRequest
from . import provider_harness as H
from .test_provider_stream import run, forbid_host_arithmetic
from .test_tower import CONTEXT, admission


class Bank(C.Structure):
    _fields_ = [("layer", U32), ("dimension", U32), ("rows", U64), ("bank", C.c_uint8 * 32), ("values", VP)]


class Expert(C.Structure):
    _fields_ = [("layer", U32), ("expert", U32), ("bytes", U64), ("digests", C.c_uint8 * 96), ("image", VP)]


class MaterializerContext(C.Structure):
    _fields_ = [("banks", C.POINTER(Bank)), ("experts", C.POINTER(Expert)),
                ("bank_count", U32), ("expert_count", U32), ("scratch", VP), ("scratch_bytes", C.c_size_t)] + [
        (n, U64) for n in ("calls", "acquires", "releases", "live", "quiesces", "fault_call")] + [
        ("fault_code", U32), ("fault_repeat", U32), ("cancel_clock", U64), ("cancel", VP)]


@pytest.fixture(params=("synthetic", "production"))
def rig(native_workspace, request, monkeypatch):
    monkeypatch.setattr(H, "LIBRARIES", H.LIBRARIES + ("elpis_dsv41_clock", "elpis_dsv41_clock_test_materializer"))
    if request.param == "synthetic":
        from .clock_harness import make_rig
        r = make_rig(native_workspace)
    else:
        r = H.make_rig(native_workspace, request.getfixturevalue("v41"))
    yield r
    r.close()
    assert all(v == 0 for v in r.live().values())


def materializer(rig, target):
    """Cold test-only admission: FMS verifies/decodes before clock entry."""
    keep, banks, experts = [], [], []

    def data(raw):
        a = C.create_string_buffer(raw)
        keep.append(a)
        return C.cast(a, VP)

    for layer, engine in target.rows.items():
        b = engine.table.bank
        chunks = []
        for start in range(0, b.rows, engine.max_rows):
            rows = engine.lookup(tuple(RowIdentity(b.digest, i) for i in range(start, min(b.rows, start + engine.max_rows))))
            chunks.append(rows.tobytes())
        bank = Bank(layer=layer, dimension=b.dimension, rows=b.rows, values=data(b"".join(chunks)))
        bank.bank[:] = bytes.fromhex(b.digest)
        banks.append(bank)
    for layer in range(target.config.layers):
        for expert in range(target.config.expert_count + 1):
            name = "shared" if expert == target.config.expert_count else str(expert)
            roles = tuple(f"layers.{layer}.ffn.experts.{name}.{w}" for w in ("w1", "w3", "w2"))
            raw = b"".join(target.store._read(target.store.bindings[r]) for r in roles)
            e = Expert(layer=layer, expert=expert, bytes=len(raw), image=data(raw))
            e.digests[:] = b"".join(bytes.fromhex(target.store.bindings[r].digest) for r in roles)
            experts.append(e)
    bank_array, expert_array = (Bank * len(banks))(*banks), (Expert * len(experts))(*experts)
    scratch = C.create_string_buffer(target.config.hash_columns * target.config.engram_dim * 4)
    ctx = MaterializerContext(bank_array, expert_array, len(banks), len(experts), C.cast(scratch, VP), len(scratch))
    keep.extend((bank_array, expert_array, scratch, ctx))
    return NativeMaterializer(rig.workspace, rig.paths["elpis_dsv41_clock_test_materializer"],
        authority=rig.authority, library_id="elpis_dsv41_clock_test_materializer",
        bind_symbol="elpis_dsv41_clock_test_materializer", context=C.byref(ctx), owner=keep), ctx


def clock(rig, target, mat, prefill, count=0, stops=()):
    return NativeClock(target, mat, rig.workspace, rig.paths["elpis_dsv41_clock"], authority=rig.authority,
                       library_id="elpis_dsv41_clock", prefill=prefill, max_new_tokens=count, stop_tokens=stops)


@pytest.mark.parametrize("part,cache", [(128, 0), (1152, 0), (1152, 7 * 5 * 1152)])
def test_clock_bitwise_yts_native_and_python_free(rig, monkeypatch, part, cache):
    tokens = rig.tokens(24)
    native_target = rig.native_target(rig.fms(), "native")
    native_state, native = run(native_target, tokens)
    native_target.release_window(native_state)
    p = rig.provider(part_bytes=part, expert_cache_bytes=cache)
    target = rig.provider_target(rig.fms(), "provider", p)
    bodies = []
    original_exchange = p.exchange
    def capture(*args, **kwargs):
        reply, body, sent, received = original_exchange(*args, **kwargs)
        if reply.kind == P.COMPLETE:
            bodies.append(bytes(body))
        return reply, body, sent, received
    with monkeypatch.context() as patch:
        patch.setattr(p, "exchange", capture)
        state, oracle = run(target, tokens)
    target.release_window(state)
    # Reset admission/cache so both clocks see identical initial provider state.
    p.release_model()
    from elpis.inference.drivers.dsv41.numerics import rope_frequencies
    p.admit(target, {b: rope_frequencies(target.config, b) for b in (False, True)})
    mat, ctx = materializer(rig, target)
    with clock(rig, target, mat, tokens) as cl:
        before = rig.counters()
        with monkeypatch.context() as patch:
            forbid_host_arithmetic(patch)
            def trap(*args, **kwargs):
                raise AssertionError("Python recurrence service executed")
            patch.setattr(target.scheme, "stream_hash", trap)
            patch.setattr(target.store, "stage_image_range", trap)
            for engine in target.rows.values():
                patch.setattr(engine, "lookup", trap)
            patch.setattr(p, "exchange", trap)
            patch.setattr(p.runtime, "take", trap) if hasattr(p.runtime, "__dict__") else None
            # Profile actual Python function entry events during the DIRECT C call.
            # A CFUNCTYPE trampoline invoking Python would emit a call event here.
            fn, handle, result = cl._lib.elpis_dsv41_clock_advance, cl._handle, Metrics()
            result_pointer = C.byref(result)
            entered, calls = False, []
            def profile(frame, event, arg):
                if entered and event == "call":
                    calls.append((frame.f_code.co_filename, frame.f_code.co_name))
            previous = sys.getprofile()
            sys.setprofile(profile)
            try:
                def probe():
                    return None
                entered = True
                probe()
                entered = False
                assert calls and calls[-1][1] == "probe"  # instrument positive control
                calls.clear()
                entered = True
                rc = fn(handle, 32, result_pointer)
                entered = False
            finally:
                sys.setprofile(previous)
            assert rc == 0 and calls == [], calls
            assert result.position == 24 and result.outcome == 2
        after = rig.counters()
        assert result.submissions - 1 == after["messages"] - before["messages"]
        assert result.bytes_h2p - 136 == after["bytes_in"] - before["bytes_in"]
        assert result.bytes_p2h - 136 == after["bytes_out"] - before["bytes_out"]
        assert result.sequence == cl._config.sequence + result.submissions
        assert result.submissions == 2 + sum(r["metrics"]["submissions"] for r in oracle)
        assert result.bytes_h2p == 264 + sum(r["metrics"]["bytes_h2p"] for r in oracle)
        assert result.bytes_p2h == 264 + sum(r["metrics"]["bytes_p2h"] for r in oracle)
        traces = cl.traces()
        for i, t in enumerate(traces):
            complete = t["complete"]
            assert t["body"] == bodies[i]  # Includes route/elision/count/position and cache telemetry bytes.
            for ref in (native[i], oracle[i]):
                assert H.bitwise_equal(complete.logits, ref["logits"])
                assert H.bitwise_equal(complete.layer_streams, ref["streams"])
                assert t["rows"] == ref["step"].rows
                assert complete.positions == ref["selected"] and complete.counts == ref["counts"]
                route = tuple(l * 5 + e for l, selected in enumerate(complete.selected) for e in (*selected, 4))
                assert route == ref["step"].route
                assert int(np.argmax(complete.logits)) == int(np.argmax(ref["logits"]))
        assert result.acquires == result.releases == ctx.acquires == ctx.releases and ctx.live == 0
        assert result.allocations == result.consumed == result.outputs_released
        assert p.runtime.metrics()["backend_fallback"] == 0
        print(f"PYTHON_CALLBACKS_INSIDE_NATIVE_CLOCK=0 tokens={result.position} submissions={result.submissions} "
              f"polls={result.polls} h2p={result.bytes_h2p} p2h={result.bytes_p2h}")
    assert p.state == "READY" and ctx.quiesces == 1


def test_native_principal_commit_replay_stop_and_zero(rig):
    p = rig.provider()
    target = rig.provider_target(rig.fms(), "provider", p)
    engine = PrincipalEngine(target)
    state = engine.initial(CONTEXT)
    from elpis.inference.admission import ContextAdmission, ContextBudget
    from elpis.inference.text import DSV41_RENDERER
    context = ContextAdmission(CONTEXT, target.model_identity, target.config.tokenizer, DSV41_RENDERER,
                               "d" * 64, (), (), 0, ContextBudget(4, 4096, 2048))
    for maximum, stops in ((8, ()), (0, ()), (8, tuple(range(target.config.vocab)))):
        request = PrincipalRequest("native-clock", rig.tokens(2), maximum, stops)
        seq = engine.begin(state, request, context, expected_state=state.digest)
        list(seq)
        expected = engine.finalize(seq)
        mat, ctx = materializer(rig, target)
        factory = lambda prefill, count, stop: clock(rig, target, mat, prefill, count, stop)
        actual = run_principal(engine, state, request, context, expected_state=state.digest, clock_factory=factory)
        assert actual == expected
        assert engine.replay(state, request, context, actual.commit) == expected
        assert ctx.live == 0 and ctx.acquires == ctx.releases
    request = PrincipalRequest("native-yield", rig.tokens(2), 8)
    seq = engine.begin(state, request, context, expected_state=state.digest)
    for _ in range(3):
        seq.next()
    seq.stop("YIELD")
    expected = engine.finalize(seq)
    mat, _ = materializer(rig, target)
    actual = run_principal(engine, state, request, context, expected_state=state.digest,
        clock_factory=lambda prefill, count, stop: clock(rig, target, mat, prefill, count, stop), stop_after=3)
    assert actual == expected
    assert engine.replay(state, request, context, actual.commit) == expected


@pytest.mark.parametrize("call,code,repeat", [(1, 8, 0), (3, 8, 0), (3, 3, 0), (3, 10, 0), (3, 4, 0)])
def test_materialization_faults_pressure(rig, call, code, repeat):
    p = rig.provider(part_bytes=128)
    t = rig.provider_target(rig.fms(), "provider", p)
    mat, ctx = materializer(rig, t)
    ctx.fault_call, ctx.fault_code, ctx.fault_repeat = call, code, repeat
    with clock(rig, t, mat, rig.tokens(2)) as cl:
        m = cl.advance()
        if code in (3, 4, 10):
            assert m["outcome"] == "MATERIALIZATION_NEEDED"
            assert p.runtime.metrics()["outstanding"] == 0 and ctx.live == 0
            m = cl.advance()
            assert m["outcome"] == "COMPLETE"
        else:
            assert m["outcome"] == "FAILED" and m["code"] == code
            assert m["state"] == (5 if call == 1 else 6)
        assert ctx.live == 0 and ctx.acquires == ctx.releases
    assert p.state == "READY" and rig.live()["models"] == 1 and rig.live()["streams"] == 0


def test_hash_exhaustive_small_history_and_extreme_products(rig):
    p = DSV41Parameters("test", (0, 1, 2), 3, 0, (1,), 4, 2, 32,
        ((1, 3, 5, 7),), ((2, 3, 5, 7, 11, 13),), ((0, 2, 5, 10, 17, 28),), (41,))
    provider = rig.provider()
    target = rig.provider_target(rig.fms(), "provider", provider)
    mat, _ = materializer(rig, target)
    with clock(rig, target, mat, rig.tokens(1)) as cl:
        fn = cl._lib.elpis_dsv41_clock_hash
        fn.argtypes = [U32, C.POINTER(C.c_int64), U32, U32, U32, C.POINTER(U64), C.POINTER(U64), C.POINTER(U64), C.POINTER(U64)]
        fn.restype = C.c_int
        cases = 0
        for multipliers in ((1, 3, 5, 7), (3074457345618258601,) * 4):
            pp = replace(p, multipliers=(multipliers,))
            scheme = AddressScheme(pp, expected_digest=pp.digest, tokenizer=pp.tokenizer, scheme=pp.schema)
            for tail in itertools.product(range(-1, 3), repeat=3):
                history = replace(scheme.initial(), tail=tail)
                for token in range(3):
                    actual = (U64 * 6)()
                    assert fn(token, (C.c_int64 * 3)(*tail), 4, 2, 0, (U64 * 4)(*multipliers),
                              (U64 * 6)(*p.primes[0]), (U64 * 6)(*p.offsets[0]), actual) == 0
                    assert tuple(actual) == scheme.stream_hash(history, (token,)).rows[0][0]
                    cases += 1
        print(f"native address exhaustive cases={cases}")


def test_clock_cancel_pressure_and_wrapper_ownership(rig):
    p = rig.provider(part_bytes=128)
    t = rig.provider_target(rig.fms(), "provider", p)
    mat, ctx = materializer(rig, t)
    ctx.fault_call, ctx.fault_code, ctx.fault_repeat = 3, 10, 1
    with clock(rig, t, mat, rig.tokens(2)) as cl:
        with pytest.raises(ContractError) as info:
            p.exchange(P.STREAM_OPEN)
        assert info.value.code == Code.BUSY
        assert cl.advance()["outcome"] == "MATERIALIZATION_NEEDED"
        cl.cancel()
        m = cl.advance()
        assert m["outcome"] == "CANCELLED" and m["state"] == 6 and m["position"] == 0
        assert ctx.acquires == ctx.releases and ctx.live == 0
    cl.close()
    with pytest.raises(ContractError):
        cl.advance()
    assert p.state == "READY" and rig.live()["models"] == 1


def test_clock_provider_faults_quarantine_wrapper(rig):
    # Cold count is deterministic; each configured provider snapshots its faults.
    p = rig.provider()
    t = rig.provider_target(rig.fms(), "map", p)
    index = p.seq + 1  # zero-based message index of first TOKEN_BEGIN
    p.close()
    for kind, code in ((H.FAULT_SUBMIT_REJECT, 5), (H.FAULT_POLL_FAIL, 5),
                       (H.FAULT_BAD_ECHO, 6), (H.FAULT_BAD_NEED, 6), (H.FAULT_POLL_TIMEOUT, 5)):
        rig.configure(fault_kind=kind, fault_message=index)
        p = rig.provider(poll_limit=3 if kind == H.FAULT_POLL_TIMEOUT else 1000)
        t = rig.provider_target(rig.fms(), f"fault-{kind}", p)
        mat, ctx = materializer(rig, t)
        with clock(rig, t, mat, rig.tokens(2)) as cl:
            m = cl.advance()
            assert m["outcome"] == "FAILED" and m["code"] == code and m["state"] == 7
        assert p.state == "QUARANTINED" and p.runtime.closed
        assert p.final_metrics["backend_fallback"] == 0
        assert ctx.acquires == ctx.releases and ctx.live == 0
        assert all(v == 0 for v in rig.live().values())


@pytest.mark.parametrize("host_failure", (False, True))
def test_principal_result_survives_release_failure(rig, monkeypatch, host_failure):
    from elpis.inference.admission import ContextAdmission, ContextBudget
    from elpis.inference.text import DSV41_RENDERER
    p = rig.provider()
    rig.provider_target(rig.fms(), "map", p)
    # One prefill token: normal release after 22 token messages, or host IO
    # after TOKEN_BEGIN before the first supply. Provider faults on release.
    release_index = p.seq + (2 if host_failure else 23)
    p.close()
    results = []
    for use_native in (False, True):
        rig.configure(fault_kind=H.FAULT_POLL_FAIL, fault_message=release_index)
        p = rig.provider()
        t = rig.provider_target(rig.fms(), f"release-{use_native}", p)
        engine = PrincipalEngine(t)
        state = engine.initial(CONTEXT)
        request = PrincipalRequest("release-failure", rig.tokens(1), 0)
        context = ContextAdmission(CONTEXT, t.model_identity, t.config.tokenizer, DSV41_RENDERER,
                                   "d" * 64, (), (), 0, ContextBudget(4, 4096, 2048))
        if use_native:
            mat, ctx = materializer(rig, t)
            if host_failure:
                ctx.fault_call, ctx.fault_code = 3, 8
            result = run_principal(engine, state, request, context, expected_state=state.digest,
                clock_factory=lambda prefill, count, stop: clock(rig, t, mat, prefill, count, stop))
        else:
            with monkeypatch.context() as patch:
                if host_failure:
                    def broken(*args, **kwargs):
                        raise ContractError(Code.IO, "injected host IO")
                    patch.setattr(t.store, "stage_image_range", broken)
                seq = engine.begin(state, request, context, expected_state=state.digest)
                list(seq)
                result = engine.finalize(seq)
        results.append(result)
        assert p.state == "QUARANTINED" and p.failure[0] == Code.DEVICE and p.runtime.closed
        assert not any(rig.live().values())
    assert results[0] == results[1]
    assert results[1].failure == (Code.IO.value if host_failure else None)
    assert (results[1].commit is not None) == (not host_failure)


def test_clock_runtime_characterization(rig, monkeypatch):
    """A characterization, never an accelerator-speedup gate. Setup excluded."""
    import json
    c = rig.config
    p = rig.provider(part_bytes=1152)
    t = rig.provider_target(rig.fms(), "provider", p)
    mat, ctx = materializer(rig, t)
    # Match host-service backing for the timing comparison. Both sides use the
    # very same cold FMS-verified, decoded bank and expert bytes. This avoids
    # attributing removal of file/page/FP8 work to the recurrence controller.
    for i in range(ctx.bank_count):
        bank = ctx.banks[i]
        values = np.ctypeslib.as_array((C.c_float * (bank.rows * bank.dimension)).from_address(bank.values))
        values = values.reshape(bank.rows, bank.dimension)
        digest = bytes(bank.bank).hex()
        def lookup(requests, values=values, digest=digest):
            assert all(r.bank == digest and r.row < len(values) for r in requests)
            return values[[r.row for r in requests]]
        monkeypatch.setattr(t.rows[bank.layer], "lookup", lookup)
    images = {}
    for i in range(ctx.expert_count):
        e = ctx.experts[i]
        name = "shared" if e.expert == c.expert_count else str(e.expert)
        roles = tuple(f"layers.{e.layer}.ffn.experts.{name}.{w}" for w in ("w1", "w3", "w2"))
        images[roles] = memoryview((C.c_ubyte * e.bytes).from_address(e.image)).cast("B")
    def stage(roles, offset, length, out):
        assert offset + length <= len(images[roles]) and len(out) == length
        out[:] = images[roles][offset:offset + length]
        return length
    monkeypatch.setattr(t.store, "stage_image_range", stage)
    prefill = rig.tokens(2)
    generated = 16
    py_ns, native_ns, py_provider, native_provider = [], [], [], []
    py_submits = py_polls = py_h2p = py_p2h = 0
    for repeat in range(5):
        expected_tokens = []
        work = t.window_initial()
        experts = t.admit_stream()
        for token in prefill:
            t.window_step(work, token, experts=experts)
        before = p.runtime.metrics()
        for _ in range(generated):
            begin = time.perf_counter_ns()
            token = int(np.argmax(work.logits))
            expected_tokens.append(token)
            t.window_step(work, token, experts=experts)
            py_ns.append(time.perf_counter_ns() - begin)
            py_submits += t.last_metrics["submissions"]
            py_polls += t.last_metrics["polls"]
            py_h2p += t.last_metrics["bytes_h2p"]
            py_p2h += t.last_metrics["bytes_p2h"]
        after = p.runtime.metrics()
        py_provider.append(after["compute_ns"] - before["compute_ns"])
        t.release_window(work)
        with clock(rig, t, mat, prefill, generated) as cl:
            assert cl.advance(len(prefill))["outcome"] == "PROGRESS"
            before = p.runtime.metrics()
            m = cl.advance()
            after = p.runtime.metrics()
            native_provider.append(after["compute_ns"] - before["compute_ns"])
            assert m["outcome"] == "COMPLETE"
            traces = cl.traces()[len(prefill):]
            assert [t["token"] for t in traces] == expected_tokens
            native_ns.extend(t["elapsed_ns"] for t in traces)
    # Count actual Python-originated runtime ABI invocations separately so the
    # counter itself does not distort the latency samples above.
    work = t.window_initial()
    experts = t.admit_stream()
    count = [0]
    with monkeypatch.context() as patch:
        for name in ("buffer_alloc", "buffer_mutable_data", "buffer_data", "buffer_size", "buffer_release",
                     "submit", "take", "get_metrics"):
            name = "elpis_exec_" + name
            original = getattr(p.exec._lib, name)
            def counted(*args, original=original):
                count[0] += 1
                return original(*args)
            patch.setattr(p.exec._lib, name, counted)
        for token in rig.tokens(4):
            t.window_step(work, token, experts=experts)
    t.release_window(work)
    with clock(rig, t, mat, prefill, 0) as cl:
        cl.advance()
        samples = []
        fn, handle, result = cl._lib.elpis_dsv41_clock_advance, cl._handle, Metrics()
        ptr = C.byref(result)
        for _ in range(200):
            start = time.perf_counter_ns()
            assert fn(handle, 1, ptr) == 0
            samples.append(time.perf_counter_ns() - start)

    def quantiles(values):
        return dict(zip(("median", "p95", "p99"), map(float, np.percentile(values, (50, 95, 99)))))
    report = dict(classification="CPU_REFERENCE_PROVIDER_ONLY", materializers="identical cold verified memory bytes",
                  transition_scope="explicit execution-runtime/clock ABI; final trace reads excluded", samples=len(py_ns),
                  python_ns=quantiles(py_ns), native_ns=quantiles(native_ns), call_overhead_ns=quantiles(samples),
                  python_native_calls_per_token=count[0] / 4, native_calls_per_generated_token=1 / generated,
                  native_python_callbacks=0, submissions_per_token=py_submits / len(py_ns),
                  polls_per_submission=py_polls / py_submits, h2p_per_token=py_h2p / len(py_ns),
                  p2h_per_token=py_p2h / len(py_ns), native_provider_compute_ns_per_token=sum(native_provider) / len(native_ns),
                  python_provider_compute_ns_per_token=sum(py_provider) / len(py_ns),
                  native_request_allocations_per_token=21 + 1, materialization_yields=0,
                  python_allocations_per_token="not measured", native_clock_storage_bytes=m["storage_bytes"])
    print("CLOCK_CHARACTERIZATION=" + json.dumps(report, sort_keys=True))


def test_clock_numpy_tolerance_unchanged(rig):
    tokens = rig.tokens(24)
    reference = rig.numpy_target(rig.fms(), "numpy")
    state, snapshots = run(reference, tokens)
    reference.release_window(state)
    p = rig.provider()
    t = rig.provider_target(rig.fms(), "provider", p)
    mat, _ = materializer(rig, t)
    worst_logits = worst_streams = 0.0
    with clock(rig, t, mat, tokens) as cl:
        assert cl.advance()["outcome"] == "COMPLETE"
        for mine, ref in zip(cl.traces(), snapshots):
            a = mine["complete"]
            np.testing.assert_allclose(a.logits, ref["logits"], rtol=H.RTOL, atol=H.ATOL)
            np.testing.assert_allclose(a.layer_streams, ref["streams"], rtol=H.RTOL, atol=H.ATOL)
            assert np.argmax(a.logits) == np.argmax(ref["logits"])
            worst_logits = max(worst_logits, float(np.max(np.abs(a.logits - ref["logits"]))))
            worst_streams = max(worst_streams, float(np.max(np.abs(a.layer_streams - ref["streams"]))))
    print(f"CLOCK_NUMPY rtol={H.RTOL} atol={H.ATOL} worst_logits={worst_logits} worst_streams={worst_streams}")


@pytest.mark.parametrize("name", (
    "test_q5_q13_provider_fault_at_every_message_of_a_token",
    "test_q5_provider_faults_in_async_device_mode",
    "test_q14_device_step_beyond_poll_budget_aborts_and_quarantines",
    "test_release_path_faults_quarantine_without_raising",
    "test_q4_fms_integrity_failure_after_begin_discards_stream_and_keeps_model",
    "test_q3_transient_fms_limit_retries_and_persistent_limit_discards",
    "test_q6_cancellation_in_every_state",
    "test_q9_engram_row_failure_before_token_begin_releases_normally"))
def test_existing_yts_fault_oracle_on_clock_fixture(rig, monkeypatch, name):
    # Execute the unchanged existing regression assertions on the explicit raw
    # token fixture as additional coverage; not a production-tokenizer rerun.
    from . import test_provider_stream_faults as original
    getattr(original, name)(rig, monkeypatch)


@pytest.mark.parametrize("name", (
    "test_q1_q10_provider_bitwise_native_and_within_numpy_tolerance",
    "test_q2_delayed_fms_fulfilment_consumes_no_port_polls",
    "test_q14_notify_avoids_the_poll_floor_and_changes_nothing"))
def test_existing_yts_numeric_oracle_on_clock_fixture(rig, name):
    from . import test_provider_stream as original
    getattr(original, name)(rig)


@pytest.mark.parametrize("name", ("test_q12_no_host_arithmetic_and_exact_yts_traffic",
                                  "test_q9_engram_rows_travel_decoded_with_token_begin"))
def test_existing_yts_boundary_oracle_on_clock_fixture(rig, monkeypatch, name):
    from . import test_provider_stream as original
    getattr(original, name)(rig, monkeypatch)
