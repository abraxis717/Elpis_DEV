"""Native Clock R0 control plane. No Python callback is installed.

Admission stays with DSV41StreamProvider. NativeMaterializer binds a service
table from an independently pinned native object. NativeClock borrows exclusive
runtime use until close; one advance can consume prefill and all generation.
Trace decoding and Principal commits happen only after the coarse call returns.
The production file-backed host service is Native Materializer R1
(native_materializer.FileMaterializer -> NativeMaterializer); R0 tests keep the
test-only materializer.
"""
from __future__ import annotations

import ctypes as C
import secrets

from elpis.substrate.authority import PinnedAuthority
from elpis.substrate.boundary import RootCapability, load_native
from elpis.inference.contracts import Code, ContractError, require
from . import stream_protocol as P

U32, U64, VP = C.c_uint32, C.c_uint64, C.c_void_p
U32P, U64P = C.POINTER(U32), C.POINTER(U64)


class Host(C.Structure):
    _fields_ = [("abi_version", U32), ("reserved", U32)] + [(n, VP) for n in (
        "buffer_alloc", "buffer_mutable_data", "buffer_data", "buffer_size", "buffer_release",
        "submit", "take", "metrics", "shutdown")]


class Materializer(C.Structure):
    _fields_ = [("abi_version", U32), ("reserved", U32)] + [(n, VP) for n in (
        "context", "rows", "expert", "release", "quiesce")]


class Config(C.Structure):
    _fields_ = [("abi_version", U32), ("reserved", U32), ("runtime", VP), ("host", Host),
                ("materializer", Materializer)] + [(n, U64) for n in ("sequence", "stream_id", "epoch")] + [
        ("manifest", C.c_uint8 * 32)] + [(n, U32) for n in (
            "vocab", "layers", "active_experts", "expert_count", "index_topk", "dimension", "hc_mult",
            "max_tokens", "features", "engram_count", "order", "heads", "row_dimension", "pad")] + [
        ("ratios", U32P), ("layer_flags", U32P), ("resident", VP), ("engram_layers", U32P), ("banks", VP),
        ("token_map", U32P), ("multipliers", U64P), ("primes", U64P), ("offsets", U64P)] + [
        (n, U64) for n in ("image_bytes", "part_bytes", "memory_budget")] + [(n, U32) for n in (
            "max_input_bytes", "max_output_bytes", "materialization_timeout_ms", "exchange_timeout_ms")] + [
        ("prefill", U32P), ("stop_tokens", U32P)] + [(n, U32) for n in ("prefill_count", "stop_count", "max_new_tokens")]


class Metrics(C.Structure):
    _fields_ = [(n, U32) for n in ("outcome", "code", "state", "position", "generated", "token", "argmax", "provider_code")] + [
        (n, U64) for n in ("sequence", "submissions", "bytes_h2p", "bytes_p2h", "polls", "provider_ns", "advance_ns",
                          "materialization_yields", "acquires", "releases", "allocations", "consumed",
                          "outputs_released", "discards", "normal_releases", "quarantines", "storage_bytes", "requests_released")]


class Trace(C.Structure):
    _fields_ = [("token", U32), ("position", U32), ("rows", U64P), ("complete", VP),
                ("complete_bytes", C.c_size_t), ("elapsed_ns", U64)]


CODES = {0: None, 1: Code.INVALID, 2: Code.STALE, 3: Code.BUSY, 4: Code.LIMIT, 5: Code.DEVICE,
         6: Code.INTEGRITY, 7: Code.ENCODING, 8: Code.IO, 9: Code.CLOSED, 10: Code.BUSY}
OUTCOMES = ("PROGRESS", "MATERIALIZATION_NEEDED", "COMPLETE", "STOP_TOKEN", "CANCELLED", "FAILED", "STOPPED")


def _check(rc):
    require(rc == 0, CODES.get(rc, Code.INVALID), "native clock status " + str(rc))


def _load(root, library, authority, library_id):
    require(type(authority) is PinnedAuthority and authority.provenance in ("deployment", "synthetic-test"),
            Code.IDENTITY, "native clock authority")
    require(library_id in authority.libraries, Code.IDENTITY, "native clock library pin")
    with RootCapability(root) as boundary:
        return load_native(boundary, library, authority.libraries[library_id])


class NativeMaterializer:
    """Cold binding of an Elpis native host service from a pinned object.

    The factory fills the table in native code; no caller-provided function
    pointers are accepted. context is a native configuration borrow kept alive
    by its owner, which is retained for the clock's lifetime.
    """
    def __init__(self, root, library, *, authority, library_id, bind_symbol, context, owner):
        self._lib = _load(root, library, authority, library_id)
        self._owner, self._context = owner, context
        self._table = Materializer()
        bind = getattr(self._lib, bind_symbol)
        bind.argtypes, bind.restype = [VP, C.POINTER(Materializer)], None
        bind(context, C.byref(self._table))
        require(self._table.abi_version == 1 and not self._table.reserved and
                all(getattr(self._table, n) for n in ("rows", "expert", "release", "quiesce")),
                Code.UNSUPPORTED, "native materializer ABI")


class NativeClock:
    """Explicit opt-in coarse recurrence beside the unchanged Python oracle."""
    def __init__(self, target, materializer, root, library, *, authority, library_id,
                 prefill, max_new_tokens, stop_tokens=(), memory_budget=128 << 20):
        provider, c, p = target.provider_stream, target.config, target.scheme.parameters
        require(provider is not None and provider.state == "READY" and provider.stream is None and provider.target is target,
                Code.STALE, "idle admitted YTS runtime")
        require(type(materializer) is NativeMaterializer, Code.IDENTITY, "native host materializer")
        require(type(prefill) is tuple and bool(prefill) and type(stop_tokens) is tuple and
                all(type(t) is int and 0 <= t < c.vocab for t in prefill + stop_tokens), detail="clock tokens")
        require(type(max_new_tokens) is int and max_new_tokens >= 0 and
                len(prefill) + max_new_tokens <= c.max_tokens, Code.LIMIT, "clock token budget")
        self.target, self.provider, self.materializer = target, provider, materializer
        self._lib = _load(root, library, authority, library_id)
        self._keep = []
        sig = {
            "abi_version": ([], U32), "create": ([C.POINTER(Config), C.POINTER(U64)], C.c_int),
            "open": ([U64], C.c_int), "advance": ([U64, U32, C.POINTER(Metrics)], C.c_int),
            "metrics": ([U64, C.POINTER(Metrics)], C.c_int),
            "trace": ([U64, U32, C.POINTER(Trace)], C.c_int),
            "cancel": ([U64], C.c_int), "stop": ([U64], C.c_int),
            "close": ([U64], C.c_int), "destroy": ([U64], C.c_int)}
        for name, (args, result) in sig.items():
            fn = getattr(self._lib, "elpis_dsv41_clock_" + name)
            fn.argtypes, fn.restype = args, result
        require(self._lib.elpis_dsv41_clock_abi_version() == 1, Code.UNSUPPORTED, "native clock ABI")

        def array(kind, values):
            a = (kind * len(values))(*values)
            self._keep.append(a)
            return a

        x = Config(abi_version=1, runtime=provider.runtime.handle, materializer=materializer._table)
        entries = ("buffer_alloc", "buffer_mutable_data", "buffer_data", "buffer_size", "buffer_release",
                   "submit", "take", "get_metrics", "shutdown")
        x.host = Host(1, 0, *(C.cast(getattr(provider.exec._lib, "elpis_exec_" + n), VP).value for n in entries))
        x.sequence, x.stream_id, x.epoch = provider.seq, provider._stream_ids + 1, secrets.randbits(64) | 1
        x.manifest[:] = provider.manifest
        for name in ("vocab", "layers", "active_experts", "expert_count", "index_topk", "dimension", "hc_mult", "max_tokens"):
            setattr(x, name, getattr(c, name))
        x.features, x.engram_count, x.order, x.heads, x.row_dimension, x.pad = (
            provider.features, len(c.engram_layers), p.order, p.heads, p.head_dimension, p.pad)
        x.ratios, x.layer_flags = array(U32, c.compress_ratios), array(U32, P.layer_flags(c))
        x.resident = C.cast(array(C.c_uint8, [int((l, e) in provider.resident) for l in range(c.layers)
                                           for e in range(c.expert_count + 1)]), VP)
        x.engram_layers = array(U32, c.engram_layers)
        x.banks = C.cast(array(C.c_uint8, b"".join(bytes.fromhex(target.rows[l].table.bank.digest)
                                               for l in c.engram_layers)), VP)
        x.token_map = array(U32, p.token_map)
        for name in ("multipliers", "primes", "offsets"):
            setattr(x, name, array(U64, [v for group in getattr(p, name) for v in group]))
        x.image_bytes, x.part_bytes = provider.image_bytes, min(provider.part_bytes, provider.image_bytes)
        x.memory_budget = memory_budget
        x.max_input_bytes, x.max_output_bytes = provider.max_input_bytes, provider.max_output_bytes
        x.materialization_timeout_ms = max(1, int(provider.staging_deadline_s * 1000))
        x.exchange_timeout_ms = 60000
        x.prefill, x.stop_tokens = array(U32, prefill), array(U32, stop_tokens)
        x.prefill_count, x.stop_count, x.max_new_tokens = len(prefill), len(stop_tokens), max_new_tokens
        self._config, self._handle, self.closed = x, U64(), False
        _check(self._lib.elpis_dsv41_clock_create(C.byref(x), C.byref(self._handle)))
        provider._stream_ids = x.stream_id
        provider._native_clock, provider.state = self, "CLOCK_ACTIVE"
        try:
            _check(self._lib.elpis_dsv41_clock_open(self._handle))
        except BaseException:
            self.close()
            raise

    def advance(self, token_budget=None):
        require(not self.closed, Code.CLOSED, "native clock closed")
        budget = self.target.config.max_tokens if token_budget is None else token_budget
        require(type(budget) is int and 1 <= budget <= 0xffffffff, detail="advance token budget")
        m = Metrics()
        _check(self._lib.elpis_dsv41_clock_advance(self._handle, budget, C.byref(m)))
        if m.state == 7:
            self.provider.seq = m.sequence
            self.provider._quarantine(CODES[m.provider_code], "native clock provider failure")
        return {**{name: getattr(m, name) for name, _ in Metrics._fields_}, "outcome": OUTCOMES[m.outcome]}

    def traces(self):
        require(not self.closed, Code.CLOSED, "native clock closed")
        m = Metrics()
        _check(self._lib.elpis_dsv41_clock_metrics(self._handle, C.byref(m)))
        c = self.target.config
        traces = []
        for position in range(m.position):
            t = Trace()
            _check(self._lib.elpis_dsv41_clock_trace(self._handle, position, C.byref(t)))
            rows = tuple(tuple(t.rows[i * c.hash_columns + j] for j in range(c.hash_columns))
                         for i in range(len(c.engram_layers)))
            body = C.string_at(t.complete, t.complete_bytes)
            complete = P.decode_complete(body, c, bool(self.provider.features & P.FEATURE_OBSERVE_LAYER_STREAMS))
            traces.append(dict(token=t.token, position=t.position, rows=rows, complete=complete,
                               body=body, elapsed_ns=t.elapsed_ns))
        return traces

    def cancel(self):
        require(not self.closed, Code.CLOSED, "native clock closed")
        _check(self._lib.elpis_dsv41_clock_cancel(self._handle))

    def stop(self):
        require(not self.closed, Code.CLOSED, "native clock closed")
        _check(self._lib.elpis_dsv41_clock_stop(self._handle))

    def close(self):
        if self.closed:
            return
        _check(self._lib.elpis_dsv41_clock_close(self._handle))
        m = Metrics()
        _check(self._lib.elpis_dsv41_clock_metrics(self._handle, C.byref(m)))
        _check(self._lib.elpis_dsv41_clock_destroy(self._handle))
        self.closed = True
        p = self.provider
        p.seq = m.sequence
        p._native_clock = None
        if m.state == 7:
            p._quarantine(CODES[m.provider_code], "native clock provider failure")
        else:
            p.state = "READY"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run_principal(engine, state, request, admission, *, expected_state, clock_factory, stop_after=None):
    """Coarse opt-in using the existing Principal finalizer (no provider commits).

    clock_factory(prefill, max_new_tokens, stop_tokens) is called ONLY during cold
    setup. It returns a NativeClock with native services. A materialization yield
    is returned to this control loop (R1: native staging pressure or a quiesce interrupt).
    """
    from elpis.inference.admission import ContextAdmission
    from elpis.inference.principal import PrincipalSequence, PrincipalState, PrincipalRequest
    from elpis.inference.target import WindowStep
    sequence = PrincipalSequence(engine, state, request, admission)
    target, c = engine.target, engine.target.config
    try:
        require(type(state) is PrincipalState and type(request) is PrincipalRequest and
                type(admission) is ContextAdmission, detail="principal inputs")
        require(state.digest == expected_state, Code.STALE, "committed principal state")
        require(state.model == target.model_identity, Code.IDENTITY, "principal model")
        require(state.numerical_profile == target.numerical_profile, Code.UNSUPPORTED, "numerical profile")
        require(admission.model == target.model_identity and admission.tokenizer == c.tokenizer,
                Code.IDENTITY, "admission model/tokenizer")
        require(admission.context_snapshot == state.context_snapshot, Code.STALE, "admission snapshot")
        prefill = admission.tokens + request.prompt
        require(bool(prefill) and all(t < c.vocab for t in prefill + request.stop_tokens), detail="principal tokens")
        require(len(prefill) + request.max_new_tokens <= c.max_tokens, Code.LIMIT, "sequence token budget")
        require(stop_after is None or (type(stop_after) is int and 0 <= stop_after <= request.max_new_tokens),
                detail="principal yield boundary")
        with clock_factory(prefill, request.max_new_tokens, request.stop_tokens) as clock:
            require(type(clock) is NativeClock and clock.target is target and
                    tuple(clock._config.prefill[i] for i in range(clock._config.prefill_count)) == prefill and
                    tuple(clock._config.stop_tokens[i] for i in range(clock._config.stop_count)) == request.stop_tokens and
                    clock._config.max_new_tokens == request.max_new_tokens,
                    Code.IDENTITY, "native principal clock binding")
            position = 0
            while True:
                budget = None if stop_after is None else max(1, len(prefill) + stop_after - position)
                m = clock.advance(budget)
                position = m["position"]
                if stop_after is not None and position == len(prefill) + stop_after and m["outcome"] == "PROGRESS":
                    clock.stop()
                    m = clock.advance()
                if m["outcome"] not in ("PROGRESS", "MATERIALIZATION_NEEDED"):
                    break
                if m["outcome"] == "MATERIALIZATION_NEEDED":
                    import time
                    time.sleep(.001)  # control-plane pressure only; never an in-clock callback
            if m["outcome"] in ("FAILED", "CANCELLED"):
                raise ContractError(CODES[m["code"]], "native principal " + m["outcome"])
            traces = clock.traces()
        sequence._steps = [WindowStep(t["token"], t["rows"], tuple(
            l * (c.expert_count + 1) + e for l, selected in enumerate(t["complete"].selected)
            for e in (*selected, c.expert_count))) for t in traces]
        sequence._outputs = [t["token"] for t in traces[len(prefill):]]
        sequence._prefill = len(prefill)
        sequence._done, sequence._stop_reason = True, "YIELD" if m["outcome"] == "STOPPED" else m["outcome"]
    except ContractError as exc:
        sequence._fail(exc)
    except BaseException as exc:
        from elpis.inference.transaction import typed_failure
        sequence._abort()
        raise typed_failure(exc) from exc
    return engine.finalize(sequence)
