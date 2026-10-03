"""DSV4.1 Native Materializer R1 qualification against the unchanged Python oracle.

The oracle is the checked-in production Python path: FMSFileAssets (page
verification, FMS residency/leases/eviction), RowEngine.lookup/decode_row and
TensorStore.stage_image_range, driven by the Python YTS. The native service is
the sealed elpis_dsv41_materializer behind the unchanged Native Clock table.

Test code calls native table entries directly through foreign-function
prototypes (Python -> C only). No Python callable is ever installed into a
native table; the clock advance runs with every Python recurrence and
materialization service trapped and a profiler proving zero Python frames.
"""
import ctypes as C
import json
import os
import sys
import threading
import time

import numpy as np
import pytest

from elpis.inference.associative import AddressScheme
from elpis.inference.contracts import Code, ContractError, RowIdentity
from elpis.inference.drivers.dsv41 import stream_protocol as P
from elpis.inference.drivers.dsv41.native_clock import CODES, Metrics, NativeClock, run_principal
from elpis.inference.drivers.dsv41.native_materializer import Config, FileAsset, FileMaterializer, Stamp
from elpis.inference.drivers.dsv41.parameters import TensorStore
from elpis.inference.drivers.dsv41.provider_stream import DSV41StreamProvider
from elpis.inference.principal import PrincipalEngine, PrincipalRequest
from elpis.inference.rows import RowEngine, decode_row
from elpis.substrate import file_assets as FA
from . import provider_harness as H
from .test_provider_stream import forbid_host_arithmetic, run
from .test_tower import CONTEXT

U32, U64, VP = C.c_uint32, C.c_uint64, C.c_void_p
EXTRA = ("elpis_dsv41_clock", "elpis_dsv41_clock_test_materializer", "elpis_dsv41_materializer")


class Span(C.Structure):
    _fields_ = [("data", VP), ("bytes", C.c_size_t), ("lease", VP), ("digests", C.c_uint8 * 96)]


ROWS = C.CFUNCTYPE(C.c_int, VP, U32, VP, C.POINTER(U64), C.c_size_t, U32, C.POINTER(Span))
EXPERT = C.CFUNCTYPE(C.c_int, VP, U32, U32, U64, C.c_size_t, C.POINTER(Span))
RELEASE = C.CFUNCTYPE(None, VP, C.POINTER(Span))
QUIESCE = C.CFUNCTYPE(None, VP)


class Table:
    """Direct Python -> C calls of a sealed table (foreign pointers, never callbacks)."""
    def __init__(self, fm):
        t = fm.materializer._table
        self.ctx = t.context
        self._rows, self._expert = ROWS(t.rows), EXPERT(t.expert)
        self._release, self._quiesce = RELEASE(t.release), QUIESCE(t.quiesce)

    def rows(self, layer, bank_hex, ids, dimension, *, hold=False):
        span = Span()
        rc = self._rows(self.ctx, layer, bytes.fromhex(bank_hex), (U64 * max(1, len(ids)))(*ids), len(ids),
                        dimension, C.byref(span))
        return self._take(rc, span, hold)

    def expert(self, layer, expert, offset, length, *, hold=False):
        span = Span()
        return self._take(self._expert(self.ctx, layer, expert, offset, length, C.byref(span)), span, hold)

    def release(self, span):
        self._release(self.ctx, C.byref(span))

    def _take(self, rc, span, hold):
        if rc:
            assert not span.lease and not span.data
            return rc, None
        if hold:
            return rc, span
        data = (C.string_at(span.data, span.bytes), bytes(span.digests))
        self.release(span)
        assert not span.lease
        return rc, data


# Python Code -> native clock code. IDENTITY has no clock code and is INTEGRITY.
NATIVE = {Code.INVALID: 1, Code.STALE: 2, Code.BUSY: 3, Code.LIMIT: 4, Code.DEVICE: 5, Code.INTEGRITY: 6,
          Code.IDENTITY: 6, Code.ENCODING: 7, Code.IO: 8, Code.CLOSED: 9}


@pytest.fixture(params=("synthetic", "production"))
def rig(native_workspace, request, monkeypatch):
    monkeypatch.setattr(H, "LIBRARIES", H.LIBRARIES + EXTRA)
    if request.param == "synthetic":
        from .clock_harness import make_rig
        r = make_rig(native_workspace)
    else:
        r = H.make_rig(native_workspace, request.getfixturevalue("v41"))
    yield r
    r.close()
    assert all(v == 0 for v in r.live().values())


def materializer(rig, target, fms, **kwargs):
    return FileMaterializer(target, fms, rig.workspace, rig.paths["elpis_dsv41_materializer"],
                            authority=rig.authority, library_id="elpis_dsv41_materializer", **kwargs)


def clock(rig, target, fm, prefill, count=0, stops=()):
    return NativeClock(target, fm.materializer, rig.workspace, rig.paths["elpis_dsv41_clock"],
                       authority=rig.authority, library_id="elpis_dsv41_clock", prefill=prefill,
                       max_new_tokens=count, stop_tokens=stops)


def py_pages(fms):
    return list(fms._pages)


TELEMETRY = ("hits", "misses", "pread_bytes", "reads", "semantic_bytes")


def py_snapshot(fms, leases):
    s = fms.stats()
    return dict({k: s[k] for k in TELEMETRY}, resident=s["hot"] + s["warm"], pages=s["pages"],
                pinned=s["pinned"], leases=leases[0])


def native_snapshot(fm):
    s = fm.stats()
    return dict({k: s["file_" + k] for k in TELEMETRY}, resident=s["file_resident_bytes"],
                pages=s["file_resident_pages"], pinned=s["file_pinned_bytes"], leases=s["file_lease_acquires"])


def count_leases(fms, monkeypatch):
    counter = [0]
    original = fms._native.acquire

    def counted(*args):
        counter[0] += 1
        return original(*args)
    monkeypatch.setattr(fms._native, "acquire", counted)
    return counter


def assert_parity(fms, fm, base, leases):
    """Equal deltas of every comparable counter, equal resident set in LRU order."""
    py, nat = py_snapshot(fms, leases), native_snapshot(fm)
    for key in TELEMETRY + ("leases",):
        assert py[key] - base[0][key] == nat[key] - base[1][key], key
    assert (py["resident"], py["pages"], py["pinned"]) == (nat["resident"], nat["pages"], nat["pinned"])
    assert py_pages(fms) == fm.pages()
    assert nat["pinned"] == 0


# --------------------------------------------------------------------------- codec


def test_row_codec_exhaustive_differential(rig):
    from elpis.substrate.boundary import RootCapability, load_native
    with RootCapability(rig.workspace) as root:
        lib = load_native(root, rig.paths["elpis_dsv41_materializer"], rig.authority.libraries["elpis_dsv41_materializer"])
    fn = lib.elpis_dsv41_row_decode
    fn.argtypes, fn.restype = [U32, C.c_char_p, C.c_size_t, U32, VP], C.c_int

    def native(codec, raw, dim):
        out = np.empty(dim, dtype="<f4")
        rc = fn(codec, raw, len(raw), dim, out.ctypes.data)
        return rc, out

    def oracle(raw, dim, codec):
        try:
            return 0, decode_row(raw, dim, codec)
        except ContractError as exc:
            return NATIVE[exc.code], None

    cases = accepted = 0
    # Every (code, scale) pair, including invalid codes/scales and overflow.
    for scale in range(256):
        for code in range(256):
            raw = bytes(7) + bytes([code]) + bytes(24) + bytes([scale])
            want_rc, want = oracle(raw, 32, "DS4_E4M3_E8M0_BF16")
            rc, got = native(2, raw, 32)
            assert rc == want_rc, (code, scale)
            if not rc:
                assert np.array_equal(got.view("<u4"), want.view("<u4")), (code, scale)
                accepted += 1
            cases += 1
    rng = np.random.default_rng(41)
    for trial in range(3000):  # adversarial multi-block rows
        dim = 32 * int(rng.integers(1, 9))
        codes = rng.integers(0, 256, dim, dtype=np.uint8)
        if trial % 3:
            codes[(codes & 127) == 127] = 0
        scales = rng.integers(0 if trial % 5 else 200, 256 if trial % 7 == 0 else 255, dim // 32, dtype=np.uint8)
        raw = codes.tobytes() + scales.tobytes()
        want_rc, want = oracle(raw, dim, "DS4_E4M3_E8M0_BF16")
        rc, got = native(2, raw, dim)
        assert rc == want_rc
        if not rc:
            assert np.array_equal(got.view("<u4"), want.view("<u4"))
        cases += 1
    for bits in (0x00000001, 0x80000000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00001, 0x3F800001):
        raw = np.array([np.float32(1.5).view("<u4"), bits, np.float32(-2).view("<u4")], dtype="<u4")
        want_rc, want = oracle(raw.tobytes(), 3, "F32_LE")
        rc, got = native(1, raw.tobytes(), 3)
        assert rc == want_rc and (rc or np.array_equal(got.view("<u4"), want.view("<u4")))
    assert native(2, bytes(32), 32)[0] == 7 and oracle(bytes(32), 32, "DS4_E4M3_E8M0_BF16")[0] == 7  # row length
    print(f"ROW_CODEC_DIFFERENTIAL cases={cases} accepted_single_values={accepted}")


# ---------------------------------------------------------------- rows and experts


@pytest.mark.parametrize("warm,pages", [(1 << 20, 1024), (16384, 6)])
def test_rows_and_file_service_parity(rig, monkeypatch, warm, pages):
    fms = rig.fms(warm_bytes=warm, max_pages=pages)
    p = rig.provider()
    t = rig.provider_target(fms, "provider", p)
    fm = materializer(rig, t, fms)
    table = Table(fm)
    fms.evict()
    leases = count_leases(fms, monkeypatch)
    base = py_snapshot(fms, leases), native_snapshot(fm)
    rng = np.random.default_rng(7)
    c = t.config
    for trial in range(60):
        for layer in c.engram_layers:
            engine = t.rows[layer]
            bank = engine.table.bank
            ids = [int(x) for x in rng.integers(0, bank.rows, c.hash_columns)]
            if trial % 4 == 0:
                ids[-1] = ids[0]  # duplicates keep request order
            want = engine.lookup(tuple(RowIdentity(bank.digest, r) for r in ids))
            rc, got = table.rows(layer, bank.digest, ids, bank.dimension)
            assert rc == 0 and got[0] == want.astype("<f4").tobytes()
            assert_parity(fms, fm, base, leases)
    engine = t.rows[c.engram_layers[0]]
    bank = engine.table.bank
    for ids, digest in (([0, bank.rows], bank.digest), ([0], "0" * 64)):
        with pytest.raises(ContractError) as info:
            engine.lookup(tuple(RowIdentity(digest, r) for r in ids))
        assert table.rows(c.engram_layers[0], digest, ids, bank.dimension)[0] == NATIVE[info.value.code]
    with pytest.raises(ContractError) as info:
        engine.lookup(tuple(RowIdentity(bank.digest, 0) for _ in range(engine.max_rows + 1)))
    assert table.rows(c.engram_layers[0], bank.digest, [0] * (engine.max_rows + 1), bank.dimension)[0] == NATIVE[info.value.code]
    assert_parity(fms, fm, base, leases)
    s = fm.stats()
    assert s["live_spans"] == 0 and s["span_acquires"] == s["span_releases"]
    print(f"ROWS_PARITY warm={warm} pages={pages} " + json.dumps({k: s["file_" + k] for k in (
        "hits", "misses", "pread_bytes", "semantic_bytes", "evictions", "lease_acquires", "resident_high_water")}))
    fm.close()


def expert_ranges(image, part, page=4096):
    w1, w3 = image.sizes[0], image.sizes[0] + image.sizes[1]
    total = image.image_bytes
    ranges = {(0, 1), (total - 1, 1), (w1 - 1, 2), (w3 - 1, 2), (w1 - 17, 40), (w3 - 17, 40),
              (0, min(total, part)), (w1 - 5, min(part, total - w1 + 5)), (page - 3, 6)}
    ranges |= {(o, min(part, total - o)) for o in range(0, total, part)}
    return sorted((o, n) for o, n in ranges if 0 <= o and n >= 1 and o + n <= total)


@pytest.mark.parametrize("warm,pages", [(1 << 20, 1024), (16384, 6)])
def test_expert_ranges_parity(rig, monkeypatch, warm, pages):
    fms = rig.fms(warm_bytes=warm, max_pages=pages)
    p = rig.provider()
    t = rig.provider_target(fms, "provider", p)
    store, c = t.store, t.config
    fm = materializer(rig, t, fms)
    table = Table(fm)
    leases = count_leases(fms, monkeypatch)
    staged = 0
    for layer in range(c.layers):
        for e in range(c.expert_count + 1):
            name = "shared" if e == c.expert_count else str(e)
            roles = tuple(f"layers.{layer}.ffn.experts.{name}.{w}" for w in ("w1", "w3", "w2"))
            image = store.expert_image(roles)
            if image.resident:
                assert table.expert(layer, e, 0, 8)[0] == 6
                continue
            canonical = b"".join(store._read(store.bindings[r]) for r in roles)
            fms.evict()  # isolate admission-style reads from the parity window
            fm.evict()
            base = py_snapshot(fms, leases), native_snapshot(fm)
            for offset, length in expert_ranges(image, p.part_bytes) + ([(0, image.image_bytes)]
                                                                       if image.image_bytes <= store.staging_budget else []):
                out = bytearray(length)
                store.stage_image_range(roles, offset, length, out)
                rc, got = table.expert(layer, e, offset, length)
                assert rc == 0 and got[0] == bytes(out) == canonical[offset:offset + length]
                assert got[1] == b"".join(image.digests)
                staged += 1
                assert_parity(fms, fm, base, leases)
            for offset, length in ((image.image_bytes - 4, 8), (0, 0)):
                with pytest.raises(ContractError) as info:
                    store.stage_image_range(roles, offset, length, bytearray(max(length, 1)))
                assert table.expert(layer, e, offset, length)[0] == NATIVE[info.value.code]
    assert table.expert(c.layers, 0, 0, 1)[0] == 1 and table.expert(0, c.expert_count + 1, 0, 1)[0] == 1
    s = fm.stats()
    assert s["live_spans"] == 0 and s["staging_high_water"] <= store.staging_budget
    print(f"EXPERT_PARITY warm={warm} ranges={staged} chunks={s['expert_chunks']} leases={s['file_lease_acquires']} "
          f"evictions={s['file_evictions']}")
    fm.close()


# ----------------------------------------------------------------- native clock


def trap_python_services(patch):
    def trap(*args, **kwargs):
        raise AssertionError("Python recurrence/materialization service executed")
    forbid_host_arithmetic(patch)
    patch.setattr(AddressScheme, "stream_hash", trap)
    patch.setattr(RowEngine, "lookup", trap)
    patch.setattr(FA.FMSFileAssets, "acquire", trap)
    patch.setattr(FA.FMSFileAssets, "_load", trap)
    patch.setattr(FA.RangeLease, "read", trap)
    patch.setattr(FA.RangeLease, "readinto", trap)
    patch.setattr(TensorStore, "stage_image_range", trap)
    patch.setattr(TensorStore, "_copy", trap)
    patch.setattr(DSV41StreamProvider, "exchange", trap)
    patch.setattr(FA.os, "pread", trap)


@pytest.mark.parametrize("part,cache,warm", [(128, 0, 1 << 20), (1152, 0, 1 << 20), (1152, 7 * 5 * 1152, 1 << 20),
                                            (1152, 0, 16384)])
def test_clock_production_materializer_bitwise_and_python_free(rig, monkeypatch, part, cache, warm):
    tokens = rig.tokens(24)
    native_target = rig.native_target(rig.fms(), "native")
    native_state, native = run(native_target, tokens)
    native_target.release_window(native_state)
    fms = rig.fms(warm_bytes=warm, max_pages=1024 if warm > 16384 else 6)
    p = rig.provider(part_bytes=part, expert_cache_bytes=cache)
    target = rig.provider_target(fms, "provider", p)
    bodies, requests = [], []
    original_exchange = p.exchange

    def capture(kind, **kwargs):
        reply, body, sent, received = original_exchange(kind, **kwargs)
        requests.append((kind, kwargs.get("layer", P.NONE), kwargs.get("position", P.NONE), sent, reply.kind))
        if reply.kind == P.COMPLETE:
            bodies.append(bytes(body))
        return reply, body, sent, received
    fms.evict()
    leases = count_leases(fms, monkeypatch)
    base = py_snapshot(fms, leases)
    oracle_counters = rig.counters()
    with monkeypatch.context() as patch:
        patch.setattr(p, "exchange", capture)
        state, oracle = run(target, tokens)
    target.release_window(state)
    oracle_counters = {k: rig.counters()[k] - oracle_counters[k] for k in ("messages", "bytes_in", "bytes_out",
                                                                            "supplied_experts", "cache_hits")}
    py_after, py_resident = py_snapshot(fms, leases), py_pages(fms)
    p.release_model()
    from elpis.inference.drivers.dsv41.numerics import rope_frequencies
    p.admit(target, {b: rope_frequencies(target.config, b) for b in (False, True)})
    fm = materializer(rig, target, fms)
    with clock(rig, target, fm, tokens) as cl:
        before = rig.counters()
        with monkeypatch.context() as patch:
            trap_python_services(patch)
            fn, handle, result = cl._lib.elpis_dsv41_clock_advance, cl._handle, Metrics()
            result_pointer = C.byref(result)
            entered, calls = False, []

            def profile(frame, event, arg):
                if entered and event in ("call", "c_call"):
                    calls.append((frame.f_code.co_filename, frame.f_code.co_name, event))
            previous = sys.getprofile()
            sys.setprofile(profile)
            try:
                def probe():
                    return None
                entered = True
                probe()
                entered = False
                assert calls and calls[-1][1] == "probe"  # positive control: the instrument sees Python entry
                calls.clear()
                entered = True
                rc = fn(handle, 32, result_pointer)
                entered = False
            finally:
                sys.setprofile(previous)
            python_frames = [c for c in calls if c[2] == "call"]
            assert rc == 0 and python_frames == [], python_frames
            assert result.position == 24 and result.outcome == 2
        after = rig.counters()
        native_counters = {k: after[k] - before[k] for k in oracle_counters}
        # YTS message sequence: identical traffic. The clock's STREAM_OPEN (136 B each
        # way) preceded `before`; both windows contain their STREAM_RELEASE.
        assert native_counters["messages"] + 1 == oracle_counters["messages"] == len(requests) + 1  # + release_window
        assert native_counters["bytes_in"] + 136 == oracle_counters["bytes_in"]
        assert native_counters["bytes_out"] + 136 == oracle_counters["bytes_out"]
        assert native_counters["supplied_experts"] == oracle_counters["supplied_experts"]
        assert native_counters["cache_hits"] == oracle_counters["cache_hits"]
        assert result.sequence == cl._config.sequence + result.submissions
        assert result.submissions == 2 + sum(r["metrics"]["submissions"] for r in oracle)
        assert result.bytes_h2p == 264 + sum(r["metrics"]["bytes_h2p"] for r in oracle)
        assert result.bytes_p2h == 264 + sum(r["metrics"]["bytes_p2h"] for r in oracle)
        for i, t in enumerate(cl.traces()):
            complete = t["complete"]
            assert t["body"] == bodies[i]
            for ref in (native[i], oracle[i]):
                assert H.bitwise_equal(complete.logits, ref["logits"])
                assert H.bitwise_equal(complete.layer_streams, ref["streams"])
                assert t["rows"] == ref["step"].rows
                assert complete.positions == ref["selected"] and complete.counts == ref["counts"]
                route = tuple(l * 5 + e for l, selected in enumerate(complete.selected) for e in (*selected, 4))
                assert route == ref["step"].route
                assert int(np.argmax(complete.logits)) == int(np.argmax(ref["logits"]))
        # File service parity with the Python oracle over the whole recurrence.
        nat = native_snapshot(fm)
        for key in TELEMETRY + ("leases",):
            assert py_after[key] - base[key] == nat[key], key
        assert fm.pages() == py_resident and nat["pinned"] == 0
        s = fm.stats()
        assert result.acquires == result.releases == s["span_acquires"] == s["span_releases"] and s["live_spans"] == 0
        assert result.allocations == result.consumed == result.outputs_released
        assert s["file_lease_acquires"] == s["file_lease_releases"] and s["refusals"] == 0
        print(f"PYTHON_CALLBACKS_INSIDE_NATIVE_CLOCK=0 PRODUCTION_MATERIALIZER=1 tokens={result.position} "
              f"submissions={result.submissions} pread={nat['pread_bytes']} hits={nat['hits']} misses={nat['misses']} "
              f"leases={nat['leases']} evictions={s['file_evictions']}")
    assert p.state == "READY" and fm.stats()["quiesces"] == 1
    fm.close()
    fm.close()  # repeated close is safe
    with pytest.raises(ContractError) as info:
        fm.stats()
    assert info.value.code == Code.CLOSED


def principal_context(t):
    from elpis.inference.admission import ContextAdmission, ContextBudget
    from elpis.inference.text import DSV41_RENDERER
    return ContextAdmission(CONTEXT, t.model_identity, t.config.tokenizer, DSV41_RENDERER,
                            "d" * 64, (), (), 0, ContextBudget(4, 4096, 2048))


def both_principal(rig, t, fm, request, *, stop_after=None):
    """Python oracle Principal run, then the native clock with the production materializer."""
    engine = PrincipalEngine(t)
    state = engine.initial(CONTEXT)
    context = principal_context(t)
    seq = engine.begin(state, request, context, expected_state=state.digest)
    if stop_after is None:
        list(seq)
    else:
        for _ in range(stop_after):
            seq.next()
        seq.stop("YIELD")
    expected = engine.finalize(seq)
    actual = run_principal(engine, state, request, context, expected_state=state.digest, stop_after=stop_after,
                           clock_factory=lambda prefill, count, stop: clock(rig, t, fm, prefill, count, stop))
    return engine, state, context, expected, actual


def test_principal_commit_replay_stop_yield(rig):
    fms = rig.fms()
    p = rig.provider()
    t = rig.provider_target(fms, "provider", p)
    fm = materializer(rig, t, fms)
    for maximum, stops, stop_after in ((8, (), None), (0, (), None), (8, tuple(range(t.config.vocab)), None),
                                       (8, (), 3)):
        request = PrincipalRequest("native-materializer", rig.tokens(2), maximum, stops)
        engine, state, context, expected, actual = both_principal(rig, t, fm, request, stop_after=stop_after)
        assert actual == expected and actual.failure is None
        assert engine.replay(state, request, context, actual.commit) == expected
        s = fm.stats()
        assert s["live_spans"] == 0 and s["file_pinned_bytes"] == 0 and s["span_acquires"] == s["span_releases"]
    assert p.state == "READY"
    fm.close()


def rewrite(path, *, truncate=False):
    time.sleep(0.02)  # beyond coarse filesystem timestamp granularity
    data = path.read_bytes()
    with open(path, "r+b") as f:
        if truncate:
            f.truncate(len(data) // 2)
        else:
            f.write(data[:1])


def row_paths(rig, name):
    return sorted((rig.workspace / name).glob("rows-*.dat")) or sorted((rig.workspace / name).glob("fixture-engram-*.dat"))


@pytest.mark.parametrize("scenario", ("rows_changed", "rows_truncated", "experts_changed", "experts_truncated",
                                      "warm_pressure", "invalid_code", "invalid_scale", "overflow", "bf16_flush"))
def test_fault_dispositions_match_oracle(rig, scenario):
    mutators = {
        "invalid_code": lambda codes, scales: (np.where(np.arange(codes.shape[1]) == 5, 0x7F, codes).astype(np.uint8), scales),
        "invalid_scale": lambda codes, scales: (codes, np.full_like(scales, 255)),
        "overflow": lambda codes, scales: (np.full_like(codes, 0x7E), np.full_like(scales, 254)),
        "bf16_flush": lambda codes, scales: (codes, np.zeros_like(scales)),
    }
    if scenario in mutators and not hasattr(rig, "file_assets"):
        pytest.skip("stored-code mutation uses the synthetic raw-token fixture builder")
    # One-page WARM budget: every page load evicts. TensorStore._copy then never
    # requests a multi-page range, so the oracle completes under pressure (hard
    # LIMIT / all-pages-leased are qualified in the native service tests).
    warm = dict(warm_bytes=4096, max_pages=1) if scenario == "warm_pressure" else {}
    fms = rig.fms(**warm)
    p = rig.provider(staging_deadline_s=0.05)
    kwargs = {"row_mutator": mutators[scenario]} if scenario in mutators else {}
    t = rig.provider_target(fms, "provider", p, **kwargs)
    fm = materializer(rig, t, fms)
    if scenario.startswith("rows"):
        for path in row_paths(rig, "provider"):
            rewrite(path, truncate=scenario.endswith("truncated"))
    if scenario.startswith("experts"):
        rewrite(next((rig.workspace / "provider").glob("fixture-experts.dat")), truncate=scenario.endswith("truncated"))
    request = PrincipalRequest("native-materializer-fault", rig.tokens(2), 8)
    _, _, _, expected, actual = both_principal(rig, t, fm, request)
    assert actual == expected
    want = {"warm_pressure": None, "bf16_flush": None}.get(
        scenario, Code.ENCODING.value if scenario in mutators else Code.INTEGRITY.value)
    assert actual.failure == want, (scenario, actual.failure)
    s = fm.stats()
    assert s["live_spans"] == 0 and s["file_pinned_bytes"] == 0 and s["file_live_ranges"] == 0
    assert s["span_acquires"] == s["span_releases"] and s["quiesces"] == 1
    assert scenario != "warm_pressure" or (s["file_evictions"] > 0 and s["file_resident_high_water"] <= 4096)
    assert p.state == "READY" and rig.live()["models"] == 1 and rig.live()["streams"] == 0
    print(f"FAULT {scenario}: failure={actual.failure} refusals={s['refusals']} "
          f"integrity={s['file_integrity_failures']}")
    fm.close()


def test_provider_and_release_faults_with_production_materializer(rig):
    p = rig.provider()
    rig.provider_target(rig.fms(), "map", p)
    index = p.seq + 1
    release_index = p.seq + 23
    p.close()
    for kind, code in ((H.FAULT_SUBMIT_REJECT, 5), (H.FAULT_POLL_FAIL, 5), (H.FAULT_BAD_ECHO, 6), (H.FAULT_BAD_NEED, 6)):
        rig.configure(fault_kind=kind, fault_message=index)
        p = rig.provider()
        fms = rig.fms()
        t = rig.provider_target(fms, f"fault-{kind}", p)
        fm = materializer(rig, t, fms)
        with clock(rig, t, fm, rig.tokens(2)) as cl:
            m = cl.advance()
            assert m["outcome"] == "FAILED" and m["code"] == code and m["state"] == 7
        assert p.state == "QUARANTINED" and p.runtime.closed
        s = fm.stats()
        assert s["live_spans"] == 0 and s["file_pinned_bytes"] == 0 and s["quiesces"] == 1
        assert all(v == 0 for v in rig.live().values())
        fm.close()
    # STREAM_RELEASE failure preserves the completed Principal result.
    results = []
    for native in (False, True):
        rig.configure(fault_kind=H.FAULT_POLL_FAIL, fault_message=release_index)
        p = rig.provider()
        fms = rig.fms()
        t = rig.provider_target(fms, f"release-{native}", p)
        engine = PrincipalEngine(t)
        state = engine.initial(CONTEXT)
        request = PrincipalRequest("release-failure", rig.tokens(1), 0)
        context = principal_context(t)
        if native:
            fm = materializer(rig, t, fms)
            result = run_principal(engine, state, request, context, expected_state=state.digest,
                                   clock_factory=lambda prefill, count, stop: clock(rig, t, fm, prefill, count, stop))
            assert fm.stats()["live_spans"] == 0
            fm.close()
        else:
            seq = engine.begin(state, request, context, expected_state=state.digest)
            list(seq)
            result = engine.finalize(seq)
        results.append(result)
        assert p.state == "QUARANTINED" and not any(rig.live().values())
    assert results[0] == results[1] and results[1].failure is None and results[1].commit is not None
    rig.configure()


def test_lifecycle_cancel_stale_busy_and_cold_refusals(rig):
    fms = rig.fms()
    p = rig.provider()
    t = rig.provider_target(fms, "provider", p)
    fm = materializer(rig, t, fms)
    # Cancellation before acquisition.
    with clock(rig, t, fm, rig.tokens(2)) as cl:
        cl.cancel()
        m = cl.advance()
        assert m["outcome"] == "CANCELLED" and m["position"] == 0 and m["acquires"] == 0
    # Outstanding borrowed span: destroy refused, quiesce forces release, stale release is safe.
    table = Table(fm)
    layer = t.config.engram_layers[0]
    bank = t.rows[layer].table.bank
    rc, span = table.rows(layer, bank.digest, [1, 2], bank.dimension, hold=True)
    assert rc == 0 and span.lease
    with pytest.raises(ContractError) as info:
        fm.close()
    assert info.value.code == Code.BUSY and not fm.closed
    stale = Span.from_buffer_copy(span)
    fm.quiesce()
    fm.quiesce()
    s = fm.stats()
    assert s["forced_releases"] == 1 and s["live_spans"] == 0 and s["file_pinned_bytes"] == 0
    table.release(stale)
    assert fm.stats()["stale_releases"] == 1
    # Cold refusals on a fresh handle of the same sealed object.
    lib = fm._lib
    handle = U64()
    assert lib.elpis_dsv41_materializer_create(C.byref(fm.config), C.byref(handle)) == 0
    index = U32()
    digests = C.create_string_buffer(32)
    record = FileAsset(1, 4096, -1, 0, 1, 1, Stamp(0, 0, 1, 0, 0), C.cast(digests, VP))
    assert lib.elpis_dsv41_materializer_admit_asset(handle, C.byref(record), C.byref(index)) == 1  # bad descriptor
    read, write = os.pipe()
    record.fd = read
    assert lib.elpis_dsv41_materializer_admit_asset(handle, C.byref(record), C.byref(index)) == 6  # non-regular
    os.close(read), os.close(write)
    path = next((rig.workspace / "provider").glob("*.dat"))
    fd = os.open(path, os.O_RDWR)
    st = os.fstat(fd)
    record = FileAsset(1, 4096, fd, 0, st.st_size, (st.st_size + 4095) // 4096,
                       Stamp(st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns),
                       C.cast(C.create_string_buffer(32 * ((st.st_size + 4095) // 4096)), VP))
    assert lib.elpis_dsv41_materializer_admit_asset(handle, C.byref(record), C.byref(index)) == 1  # writable
    os.close(fd)
    assert lib.elpis_dsv41_materializer_seal(handle) == 0
    assert lib.elpis_dsv41_materializer_admit_asset(handle, C.byref(record), C.byref(index)) == 9  # sealed
    assert lib.elpis_dsv41_materializer_destroy(handle) == 0 and lib.elpis_dsv41_materializer_destroy(handle) == 0
    with pytest.raises(ContractError) as info:
        fms.transfer_asset("0" * 64, lambda *args: None)
    assert info.value.code == Code.MISSING
    # A clock bound to a destroyed service fails STALE without dereferencing it.
    with clock(rig, t, fm, rig.tokens(2)) as cl:
        fm.close()
        fm.close()
        m = cl.advance()
        assert m["outcome"] == "FAILED" and m["code"] == 2 and m["state"] == 5
    assert p.state == "READY" and rig.live()["streams"] == 0
    table.release(stale)  # stale context: no dereference


def test_independent_instances_concurrently(rig):
    fa, fb = rig.fms(), rig.fms()
    ta = rig.provider_target(fa, "a", rig.provider())
    tb = rig.provider_target(fb, "b", rig.provider())
    ma, mb = materializer(rig, ta, fa), materializer(rig, tb, fb)
    c = ta.config
    rng = np.random.default_rng(3)
    work = []
    for i in range(120):
        layer = c.engram_layers[i % len(c.engram_layers)]
        bank = ta.rows[layer].table.bank
        ids = [int(x) for x in rng.integers(0, bank.rows, c.hash_columns)]
        want = ta.rows[layer].lookup(tuple(RowIdentity(bank.digest, r) for r in ids)).astype("<f4").tobytes()
        e = i % c.expert_count
        roles = tuple(f"layers.{i % c.layers}.ffn.experts.{e}.{w}" for w in ("w1", "w3", "w2"))
        offset = int(rng.integers(0, ta.store.expert_image(roles).image_bytes - 64))
        out = bytearray(64)
        ta.store.stage_image_range(roles, offset, 64, out)
        work.append((layer, bank.digest, ids, want, i % c.layers, e, offset, bytes(out)))
    errors = []

    def worker(fm, quiesce=False):
        table = Table(fm)
        try:
            for layer, digest, ids, want, el, e, offset, expected in work:
                for _ in range(1000):
                    rc, got = table.rows(layer, digest, ids, c.engram_dim)
                    if rc != 10:
                        break
                assert rc == 0 and got[0] == want
                for _ in range(1000):
                    rc, got = table.expert(el, e, offset, 64)
                    if rc != 10:
                        break
                assert rc == 0 and got[0] == expected
        except BaseException as exc:  # surfaced on the main thread
            errors.append(exc)

    def quiescer(fm, stop):
        while not stop.is_set():
            fm.quiesce()
    stop = threading.Event()
    threads = [threading.Thread(target=worker, args=(ma,)), threading.Thread(target=worker, args=(mb,)),
               threading.Thread(target=quiescer, args=(ma, stop))]
    for th in threads:
        th.start()
    for th in threads[:2]:
        th.join()
    stop.set()
    threads[2].join()
    assert not errors, errors
    for fm in (ma, mb):
        s = fm.stats()
        assert s["live_spans"] == 0 and s["file_pinned_bytes"] == 0
    assert ma.stats()["span_acquires"] == mb.stats()["span_acquires"] == 240
    ma.close(), mb.close()


def test_materializer_runtime_characterization(rig, monkeypatch):
    """Characterization only (CPU reference provider); never an accelerator speed claim."""
    c = rig.config
    fms = rig.fms()
    p = rig.provider(part_bytes=1152)
    t = rig.provider_target(fms, "provider", p)
    fm = materializer(rig, t, fms)
    table = Table(fm)

    def q(values):
        return dict(zip(("median", "p95", "p99"), map(float, np.percentile(values, (50, 95, 99))))) if values else {}
    rng = np.random.default_rng(11)
    py_rows, nat_rows, py_expert, nat_expert = [], [], [], []
    for i in range(400):
        layer = c.engram_layers[i % len(c.engram_layers)]
        engine = t.rows[layer]
        bank = engine.table.bank
        ids = [int(x) for x in rng.integers(0, bank.rows, c.hash_columns)]
        requests = tuple(RowIdentity(bank.digest, r) for r in ids)
        start = time.perf_counter_ns(); engine.lookup(requests); py_rows.append(time.perf_counter_ns() - start)
        start = time.perf_counter_ns(); assert table.rows(layer, bank.digest, ids, bank.dimension)[0] == 0
        nat_rows.append(time.perf_counter_ns() - start)
        e = i % (c.expert_count + 1)
        roles = tuple(f"layers.{i % c.layers}.ffn.experts.{'shared' if e == c.expert_count else e}.{w}"
                      for w in ("w1", "w3", "w2"))
        out = bytearray(1152)
        start = time.perf_counter_ns(); t.store.stage_image_range(roles, 0, 1152, out)
        py_expert.append(time.perf_counter_ns() - start)
        start = time.perf_counter_ns(); assert table.expert(i % c.layers, e, 0, 1152)[0] == 0
        nat_expert.append(time.perf_counter_ns() - start)
    # Python YTS with the Python file materializer: per-token host materialization and recurrence.
    prefill, generated = rig.tokens(2), 16
    py_host, py_token, py_calls = [], [], [0]
    names = ("register", "acquire", "release", "evict", "stats")
    for name in names:
        original = getattr(fms._native, name)

        def counted(*args, original=original):
            py_calls[0] += 1
            return original(*args)
        monkeypatch.setattr(fms._native, name, counted)
    for name in ("buffer_alloc", "buffer_mutable_data", "buffer_data", "buffer_size", "buffer_release",
                 "submit", "take", "get_metrics"):
        original = getattr(p.exec._lib, "elpis_exec_" + name)

        def counted(*args, original=original):
            py_calls[0] += 1
            return original(*args)
        monkeypatch.setattr(p.exec._lib, "elpis_exec_" + name, counted)
    work = t.window_initial()
    experts = t.admit_stream()
    for token in prefill:
        t.window_step(work, token, experts=experts)
    expected, submissions = [], 0
    py_calls[0] = 0
    for _ in range(generated):
        token = int(np.argmax(work.logits))
        expected.append(token)
        start = time.perf_counter_ns()
        t.window_step(work, token, experts=experts)
        py_token.append(time.perf_counter_ns() - start)
        py_host.append(t.last_metrics["engram_ns"] + t.last_metrics["expert_materialize_ns"])
        submissions += t.last_metrics["submissions"]
    python_native_calls_per_token = py_calls[0] / generated
    t.release_window(work)
    monkeypatch.undo()
    stats_before = fm.stats()
    with clock(rig, t, fm, prefill, generated) as cl:
        assert cl.advance(len(prefill))["outcome"] == "PROGRESS"
        before = fm.stats()
        m = cl.advance()
        after = fm.stats()
        assert m["outcome"] == "COMPLETE"
        traces = cl.traces()[len(prefill):]
        assert [x["token"] for x in traces] == expected
        native_token = [x["elapsed_ns"] for x in traces]
    native_host_per_token = (after["row_ns"] + after["expert_ns"] - before["row_ns"] - before["expert_ns"]) / generated
    samples = fm.samples()
    resident = [r for _, _, r in samples]
    report = dict(
        classification="CPU_REFERENCE_PROVIDER_ONLY; Linux page cache not accounted",
        budgets=dict(warm_bytes=fm.config.file.warm_bytes, max_pages=fm.config.file.max_pages,
                     page_staging_bytes=fm.config.file.staging_bytes, expert_staging_bytes=fm.config.staging_bytes),
        row_lookup_ns=dict(python=q(py_rows), native=q(nat_rows)),
        expert_part_1152_ns=dict(python=q(py_expert), native=q(nat_expert)),
        host_materialization_per_token_ns=dict(python=q(py_host), native_mean=native_host_per_token),
        recurrence_per_token_ns=dict(python_yts=q(py_token), native_clock=q(native_token)),
        python_native_calls_per_token=dict(python_yts=python_native_calls_per_token,
                                           native_clock=1 / (generated + len(prefill))),
        python_callbacks_per_clock_advance=0,
        yts_submissions_per_token=submissions / generated,
        resident_bytes=dict(current=after["file_resident_bytes"], p95=float(np.percentile(resident, 95)),
                            high_water=after["file_resident_high_water"], pinned=after["file_pinned_bytes"]),
        native_clock_window=dict(pread_bytes=after["file_pread_bytes"] - before["file_pread_bytes"],
                                 semantic_bytes=after["file_semantic_bytes"] - before["file_semantic_bytes"],
                                 hits=after["file_hits"] - before["file_hits"],
                                 misses=after["file_misses"] - before["file_misses"],
                                 leases=after["file_lease_acquires"] - before["file_lease_acquires"],
                                 row_bytes=after["row_bytes"] - before["row_bytes"],
                                 expert_bytes=after["expert_bytes"] - before["expert_bytes"],
                                 spans=after["span_acquires"] - before["span_acquires"]),
        materializer_total=dict(pread_bytes=after["file_pread_bytes"], hits=after["file_hits"],
                                misses=after["file_misses"], evictions=after["file_evictions"],
                                staging_high_water=after["staging_high_water"],
                                page_staging_high_water_analytical=after["file_staging_high_water"],
                                calls_before_clock=stats_before["span_acquires"]),
        buffered_page_cache_accounted=False)
    print("MATERIALIZER_CHARACTERIZATION=" + json.dumps(report, sort_keys=True))
    fm.close()
