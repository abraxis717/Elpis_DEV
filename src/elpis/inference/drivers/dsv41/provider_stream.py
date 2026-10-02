"""YTS-R0 host adapter: a DSV4.1 tower executed by a provider stream.

Implements the Yielded Token Stream (docs/inference/DSV41_PROVIDER_STREAM.md)
entirely above the existing generic execution port: one adapter-defined
``BACKEND_ONLY`` operation, ``workers=1``, ``capacity=1``, strict alternation
(submit -> take -> validate -> next submit). A materialization yield is the
completed result of a submission; FMS work happens between submissions on the
principal thread, never inside a provider callback and never on a pool worker.

Failure classes (frozen):

* before ``TOKEN_BEGIN``: the provider stream is untouched; release it normally;
* host-originated after ``TOKEN_BEGIN`` (FMS integrity/IO/staging, cancellation):
  discard the stream (``STREAM_RELEASE``); model admission and cache survive;
* provider-originated (port rejection/failure/timeout, a malformed or non-finite
  successful reply, a failed release): quarantine the runtime (shutdown with
  cancellation, destroy -> ``backend.shutdown``, detach). Nothing survives.

There is never a CPU continuation of a provider sequence.
"""
from __future__ import annotations

import ctypes as C
import secrets
import time
from time import perf_counter_ns

import numpy as np

from elpis.identity import content_digest
from elpis.substrate import execution as X
from elpis.substrate.authority import PinnedAuthority
from elpis.substrate.boundary import RootCapability, load_native
from elpis.substrate.digests import raw_digest
from ...contracts import Code, ContractError, integer, require
from . import stream_protocol as P

PROTOCOL = "elpis.inference.dsv41.yielded-token-stream.r0"


class _HostV1(C.Structure):
    _fields_ = [("abi_version", C.c_uint32), ("reserved", C.c_uint32), ("buffer_alloc", C.c_void_p),
                ("buffer_mutable_data", C.c_void_p), ("buffer_data", C.c_void_p), ("buffer_size", C.c_void_p),
                ("buffer_release", C.c_void_p), ("notify", C.c_void_p)]


class ProviderAttention:
    """Host view of one provider-owned attention state: its compressed count only."""

    __slots__ = ("count",)
    nbytes = 0

    def __init__(self):
        self.count = 0


class DSV41StreamProvider:
    """One attached YTS-R0 provider runtime. Single principal thread; one active stream."""

    def __init__(self, root, *, execution_library, provider_library, authority, execution_id, provider_id,
                 max_input_bytes=1 << 20, max_output_bytes=8 << 20, poll_limit=1000, part_bytes=None,
                 expert_cache_bytes=0, observe_layer_streams=False, staging_deadline_s=5.0, take_wait_ms=1000):
        require(type(authority) is PinnedAuthority and authority.provenance in ("deployment", "synthetic-test"),
                Code.IDENTITY, "YTS-R0 provider authority")
        require(type(provider_id) is str and provider_id in authority.libraries, Code.IDENTITY,
                "YTS-R0 provider library identifier")
        for value, low in ((max_input_bytes, P.HEADER_BYTES + P.SUPPLY_PREFIX_BYTES + 1),
                           (max_output_bytes, P.HEADER_BYTES), (poll_limit, 1), (take_wait_ms, 1)):
            integer(value, low)
        require(max_input_bytes <= X.BUFFER_LIMIT and max_output_bytes <= X.BUFFER_LIMIT and poll_limit <= 1000,
                Code.LIMIT, "execution port limits")
        integer(expert_cache_bytes)
        require(part_bytes is None or (type(part_bytes) is int and 1 <= part_bytes <=
                                       max_input_bytes - P.HEADER_BYTES - P.SUPPLY_PREFIX_BYTES),
                Code.LIMIT, "expert part size")
        self.exec = X.ExecutionLibrary(root, execution_library, authority=authority, library_id=execution_id)
        self.identity = authority.libraries[provider_id]
        with RootCapability(root) as boundary:
            lib = load_native(boundary, provider_library, self.identity)
        try:
            abi = lib.elpis_dsv41_stream_provider_abi_version
            attach = lib.elpis_dsv41_stream_provider_attach
            bind = lib.elpis_dsv41_stream_provider_bind_runtime
            detach = lib.elpis_dsv41_stream_provider_detach
        except AttributeError as exc:
            raise ContractError(Code.UNSUPPORTED, "YTS-R0 provider attachment symbols") from exc
        abi.argtypes, abi.restype = [], C.c_uint32
        attach.argtypes, attach.restype = [C.POINTER(_HostV1), C.POINTER(X.Backend)], C.c_int
        bind.argtypes, bind.restype = [C.c_void_p, C.c_void_p], C.c_int
        detach.argtypes, detach.restype = [C.c_void_p], None
        require(abi() == 1, Code.UNSUPPORTED, "YTS-R0 provider ABI version")
        self._host = _HostV1(1, 0, *(self.exec.entry_point(n) for n in (
            "elpis_exec_buffer_alloc", "elpis_exec_buffer_mutable_data", "elpis_exec_buffer_data",
            "elpis_exec_buffer_size", "elpis_exec_buffer_release", "elpis_exec_notify")))
        self._backend = X.Backend()
        require(attach(C.byref(self._host), C.byref(self._backend)) == 0 and bool(self._backend.context),
                Code.UNSUPPORTED, "YTS-R0 provider attach")
        self._lib, self._detach = lib, detach
        self.runtime = None
        try:
            self.runtime = X.Runtime(self.exec, self._backend, workers=1, capacity=1,
                                     max_input_bytes=max_input_bytes, max_output_bytes=max_output_bytes,
                                     poll_limit=poll_limit)
            require(bind(self._backend.context, self.runtime.handle) == 0, Code.UNSUPPORTED,
                    "YTS-R0 runtime binding")
        except BaseException:
            # Binding may already have retained the runtime for notifications.
            # Quiesce callbacks/device work before freeing the provider context.
            if self.runtime is not None:
                self.runtime.destroy()
            detach(self._backend.context)
            raise
        self.max_input_bytes, self.max_output_bytes = max_input_bytes, max_output_bytes
        self.requested_part_bytes, self.cache_bytes = part_bytes, expert_cache_bytes
        self.observe_layer_streams = bool(observe_layer_streams)
        self.staging_deadline_s, self.take_wait_ms = staging_deadline_s, take_wait_ms
        self.state = "ATTACHED"
        self.failure = None
        self.seq = 0
        self._sent_any = False
        self._stream_ids = 0
        self.stream = None
        self.target = self.manifest = self.profile = None
        self.features = 0
        self.resident = frozenset()
        self.part_bytes = self.image_bytes = self.reserved_bytes = 0
        self.final_metrics = None
        self.last_exchange = {}
        self.totals = dict(submissions=0, bytes_h2p=0, bytes_p2h=0, request_high_water=0, quarantines=0,
                           discards=0, releases=0, admission_submissions=0)

    # ------------------------------------------------------------------ port

    def _quarantine(self, code, detail):
        """Provider-originated failure: shut the runtime down and drop every provider resource."""
        if self.state != "QUARANTINED":
            self.state, self.failure = "QUARANTINED", (code, detail)
            self.totals["quarantines"] += 1
            if self.stream is not None:
                self.stream.state = "QUARANTINED"
                self.stream = None
            self._teardown()
        return ContractError(code, detail)

    def _teardown(self):
        if not self.runtime.closed:
            self.final_metrics = self.runtime.metrics()
            self.runtime.destroy()  # shutdown(cancel) -> backend.shutdown releases every provider resource
            self._detach(self._backend.context)

    def _usable(self):
        require(self.state not in ("QUARANTINED", "CLOSED"), Code.DEVICE if self.state == "QUARANTINED"
                else Code.CLOSED, "YTS-R0 provider " + self.state.lower())

    def exchange(self, kind, *, stream=None, position=P.NONE, layer=P.NONE, body=b"", tail=0, fill=None):
        """One strictly alternating BACKEND_ONLY exchange. ``fill(view)`` writes ``tail`` bytes after the body."""
        self._usable()
        seq = self.seq + 1  # advanced only by an accepted submission: a failed staging sends nothing
        header = P.Header(kind, stream.id if stream else 0, stream.epoch if stream else 0, seq, position,
                          layer, len(body) + tail, self.manifest)
        size = P.HEADER_BYTES + len(body) + tail
        require(size <= self.max_input_bytes, Code.LIMIT, "YTS-R0 request exceeds the port input limit")
        buffer = self.exec.buffer(size)
        try:
            view = buffer.view()
            view[:P.HEADER_BYTES] = header.pack()
            view[P.HEADER_BYTES:P.HEADER_BYTES + len(body)] = body
            if tail:
                fill(view[P.HEADER_BYTES + len(body):])
            del view
            polls = self.runtime.metrics()["backend_polls"]  # before submit: the first poll follows it at once
            for _ in range(100):  # capacity=1 and strict alternation leave no backpressure
                rc, _ = self.runtime.submit_backend_only(P.OPERATION, kind, seq, buffer)
                if rc != X.WOULD_BLOCK:
                    break
                time.sleep(0.001)
        finally:
            buffer.release()  # no-op once submitted (ownership moved to the runtime)
        if rc == X.BACKEND_UNAVAILABLE and kind == P.MODEL_ADMIT_BEGIN and not self._sent_any:
            # Refused before admission: nothing was consumed and no provider state exists.
            self.state = "CLOSED"
            self._teardown()
            raise ContractError(Code.UNSUPPORTED, "provider does not advertise YTS-R0")
        if rc != X.OK:
            raise self._quarantine(Code.CLOSED if rc == X.CLOSED else Code.DEVICE,
                                   "YTS-R0 submit: " + X.STATUS_NAMES.get(rc, str(rc)))
        self.seq, self._sent_any = seq, True
        self.totals["submissions"] += 1
        self.totals["bytes_h2p"] += size
        self.totals["request_high_water"] = max(self.totals["request_high_water"], size)
        while True:
            rc, result = self.runtime.take(self.take_wait_ms)
            if rc == X.OK:
                break
            if rc != X.WOULD_BLOCK:
                raise self._quarantine(Code.DEVICE, "YTS-R0 take: " + X.STATUS_NAMES.get(rc, str(rc)))
        self.last_exchange = dict(kind=kind, polls=self.runtime.metrics()["backend_polls"] - polls,
                                  reply_bytes=len(result.output or b""))
        if result.status != X.OK or result.output is None:
            raise self._quarantine(Code.DEVICE, "YTS-R0 provider " + P.KIND_NAMES[kind] + ": " +
                                   X.STATUS_NAMES.get(result.status, str(result.status)))
        self.totals["bytes_p2h"] += len(result.output)
        try:
            reply = P.unpack_header(result.output)
            P.check_reply(reply, header)
        except ContractError as exc:
            raise self._quarantine(exc.code, str(exc)) from exc
        return reply, memoryview(result.output)[P.HEADER_BYTES:], size, len(result.output)

    # ------------------------------------------------------------- admission

    def admit(self, target, frequencies):
        """Admit the target's resident tensors and derived tables. Called by DSV41Target."""
        require(self.state == "ATTACHED" and self.stream is None, Code.STALE, "YTS-R0 admission state")
        store, c = target.store, target.config
        experts = {}
        for name in store.bindings:
            if ".ffn.experts." in name:
                experts.setdefault(name.rsplit(".", 1)[0], []).append(name in store.dense)
        require(all(all(v) or not any(v) for v in experts.values()), Code.UNSUPPORTED,
                "YTS-R0 expert residency is whole")
        resident = set()
        for key, flags in experts.items():
            if all(flags):
                _, layer, expert = P.role_code(key + ".w1", c.expert_count)
                resident.add((layer, expert))
        tensors = []
        for name in sorted(store.dense):
            binding = store.bindings[name]
            tensors.append((name, binding.shape, P.digest_bytes(binding.digest), store.dense[name]))
        for name, compressed in (("rope.local", False), ("rope.compressed", True)):
            table = np.ascontiguousarray(frequencies[compressed], dtype="<f4")
            digest = content_digest("elpis.inference.dsv41.derived-table.v1", dict(
                name=name, shape=table.shape, content=raw_digest(table.tobytes())))
            tensors.append((name, table.shape, P.digest_bytes(digest), table))
        limit = self.max_input_bytes - P.HEADER_BYTES - P.SUPPLY_PREFIX_BYTES
        part = self.requested_part_bytes or min(store.staging_budget, limit)
        require(part <= store.staging_budget and part <= limit, Code.LIMIT, "expert part exceeds host staging budget")
        features = (P.FEATURE_CACHE if self.cache_bytes else 0) | (
            P.FEATURE_OBSERVE_LAYER_STREAMS if self.observe_layer_streams else 0)
        self.manifest = P.digest_bytes(store.manifest.digest)
        before = self.totals["submissions"]
        self.exchange(P.MODEL_ADMIT_BEGIN, body=P.encode_admit_begin(
            c, features=features, part_bytes=part, cache_bytes=self.cache_bytes, config_digest=c.digest))
        chunk = self.max_input_bytes - P.HEADER_BYTES - P.ADMIT_TENSOR_PREFIX_BYTES
        for name, shape, digest, array in tensors:
            data = np.ascontiguousarray(array, dtype="<f4").tobytes()
            kind, layer, expert = P.role_code(name, c.expert_count)
            for offset in range(0, len(data), chunk):
                piece = data[offset:offset + chunk]
                self.exchange(P.MODEL_ADMIT_TENSOR, body=P.encode_admit_tensor_prefix(
                    kind, layer, expert, shape, digest, len(data), offset, len(piece)) + piece)
        _, body, _, _ = self.exchange(P.MODEL_ADMIT_END, body=P.encode_admit_end(len(tensors)))
        try:
            ack = P.decode_admit_ack(body, features)
        except ContractError as exc:
            raise self._quarantine(exc.code, str(exc)) from exc
        self.totals["admission_submissions"] = self.totals["submissions"] - before
        self.target, self.features, self.resident = target, ack.features, frozenset(resident)
        self.part_bytes, self.image_bytes = part, 3 * c.expert_dim * c.dimension * 4
        self.profile = dict(protocol=PROTOCOL, provider_library=self.identity.sha256,
                            kernel_profile=ack.profile.hex())
        self.reserved_bytes = ack.reserved_bytes
        self.state = "READY"
        return ack

    def release_model(self):
        """MODEL_RELEASE: clears every provider tensor and cache entry; the runtime may admit again."""
        require(self.state == "READY" and self.stream is None, Code.STALE, "YTS-R0 model release state")
        self.exchange(P.MODEL_RELEASE)
        self.state, self.target, self.manifest = "ATTACHED", None, None

    def close(self):
        """Release everything the provider holds. Idempotent; never raises on provider failure."""
        if self.state in ("QUARANTINED", "CLOSED"):
            return
        try:
            if self.stream is not None:
                self.stream.release()
            if self.state == "READY" and self.stream is None:
                self.release_model()
        except ContractError:
            pass  # already quarantined
        if self.state != "QUARANTINED":
            self.state = "CLOSED"
            self._teardown()

    # ---------------------------------------------------------------- streams

    def open_stream(self):
        require(self.state == "READY", Code.STALE if self.state != "QUARANTINED" else Code.DEVICE,
                "YTS-R0 provider is not ready")
        require(self.stream is None, Code.UNSUPPORTED, "YTS-R0 R0 admits one active stream per runtime")
        self._stream_ids += 1
        stream = ProviderStream(self, self._stream_ids, secrets.randbits(64) | 1)
        observe = self.features & P.FEATURE_OBSERVE_LAYER_STREAMS
        _, body, _, _ = self.exchange(P.STREAM_OPEN, stream=stream,
                                      body=P.encode_stream_open(self.target.config.max_tokens, observe))
        try:
            stream.state_bytes = P.decode_opened(body)
        except ContractError as exc:
            raise self._quarantine(exc.code, str(exc)) from exc
        stream.observe = bool(observe)
        self.stream = stream
        return stream

    def counters(self):
        """Python-side buffer accounting (allocated == released + consumed) and port metrics."""
        out = dict(self.exec.counts)
        if not self.runtime.closed:
            out.update({"metrics": self.runtime.metrics()})
        return out


class ProviderStream:
    """One provider stream for one principal sequence (YTS-R0 state machine)."""

    def __init__(self, provider, stream_id, epoch):
        self.provider, self.id, self.epoch = provider, stream_id, epoch
        self.position = 0
        self.state = "BOUNDARY"
        self.observe = False
        self.state_bytes = 0
        self.cancelled = False
        self.observer = None  # test instrumentation: observer(event, info)
        self.last = {}

    def _event(self, name, info=None):
        if self.observer is not None:
            self.observer(name, info)

    def cancel(self):
        """Request cancellation; honoured at the next submission boundary."""
        self.cancelled = True

    def release(self):
        """Normal release at a token boundary (or no-op once gone). Never raises."""
        if self.state != "BOUNDARY":
            return
        self._release("RELEASED")

    def _release(self, final):
        provider = self.provider
        try:
            provider.exchange(P.STREAM_RELEASE, stream=self)
        except ContractError:
            return  # provider-originated: already quarantined
        self.state = final
        provider.stream = None
        provider.totals["releases" if final == "RELEASED" else "discards"] += 1

    def _host_failure(self, exc):
        """Host-originated failure after TOKEN_BEGIN: discard the stream, keep the model."""
        if self.state == "IN_TOKEN":
            self._release("DISCARDED")
            if self.state == "IN_TOKEN":
                self.state = "QUARANTINED"
        return exc

    def _note_polls(self, stats, reply):
        polls = self.provider.last_exchange["polls"]
        if reply.kind == P.NEED:
            stats["polls_need"].append(polls)
        else:
            stats["polls_complete"] = polls

    def _validated(self, fn, *args):
        try:
            return fn(*args)
        except ContractError as exc:
            raise self.provider._quarantine(exc.code, str(exc)) from exc

    def token(self, position, token, segments, store):
        """Execute one token. Returns the validated COMPLETE."""
        provider = self.provider
        require(self.state == "BOUNDARY" and provider.stream is self, Code.STALE, "YTS-R0 stream state")
        require(position == self.position, Code.STALE, "YTS-R0 stream position")
        if self.cancelled:
            self.release()
            raise ContractError(Code.CLOSED, "YTS-R0 token cancelled before TOKEN_BEGIN")
        c = provider.target.config
        plan = P.TokenPlan(c, resident=provider.resident, cache=provider.features & P.FEATURE_CACHE,
                           image_bytes=provider.image_bytes, part_bytes=provider.part_bytes)
        stats = dict(submissions=0, bytes_h2p=0, bytes_p2h=0, needs=0, supplies=0, staging_high_water=0,
                     staging_ns=0, polls=0, polls_need=[], polls_complete=0)
        polls_before = provider.runtime.metrics()["backend_polls"]
        start = perf_counter_ns()
        self.state = "IN_TOKEN"
        try:
            self._event("IN_FLIGHT", "TOKEN_BEGIN")
            reply, body, sent, received = provider.exchange(
                P.TOKEN_BEGIN, stream=self, position=position, body=P.encode_token_begin(token, segments))
            stats["submissions"] += 1
            stats["bytes_h2p"] += sent
            stats["bytes_p2h"] += received
            self._note_polls(stats, reply)
            while reply.kind == P.NEED:
                need = self._validated(P.decode_need, body, c.active_experts)
                if need.layer != reply.layer:
                    raise provider._quarantine(Code.INTEGRITY, "YTS-R0 protocol violation: NEED header layer")
                expert, offset, length = self._validated(plan.on_need, need)
                stats["needs"] += 1
                self._event("PARKED", need)
                if self.cancelled:
                    raise ContractError(Code.CLOSED, "YTS-R0 token cancelled while parked")
                roles = tuple(f"layers.{need.layer}.ffn.experts."
                              f"{'shared' if expert == c.expert_count else expert}.{r}" for r in ("w1", "w3", "w2"))
                image = store.expert_image(roles)
                if image.resident or image.image_bytes != provider.image_bytes:
                    raise provider._quarantine(Code.INTEGRITY, "YTS-R0 protocol violation: need of a resident expert")
                prefix = P.encode_supply_prefix(need.layer, expert, need.cursor_index, image.image_bytes, offset,
                                                length, image.digests)
                staged = []

                def fill(view, roles=roles, offset=offset, length=length):
                    begin = perf_counter_ns()
                    deadline = time.monotonic() + provider.staging_deadline_s
                    while True:
                        try:
                            store.stage_image_range(roles, offset, length, view)
                            break
                        except ContractError as exc:
                            # Transient FMS pressure: nothing is in flight and the provider is
                            # unchanged by waiting, so retry until the host's own deadline.
                            if exc.code not in (Code.LIMIT, Code.BUSY) or time.monotonic() >= deadline:
                                raise
                            time.sleep(0.001)
                    staged.append(perf_counter_ns() - begin)

                self._event("STAGING", (need.layer, expert, offset, length))
                reply, body, sent, received = provider.exchange(
                    P.EXPERT_SUPPLY, stream=self, position=position, layer=need.layer, body=prefix,
                    tail=length, fill=fill)
                stats["submissions"] += 1
                stats["supplies"] += 1
                stats["bytes_h2p"] += sent
                stats["bytes_p2h"] += received
                stats["staging_high_water"] = max(stats["staging_high_water"], sent)
                stats["staging_ns"] += staged[0]
                self._note_polls(stats, reply)
                plan.on_supplied(length)
                if self.cancelled:
                    raise ContractError(Code.CLOSED, "YTS-R0 token cancelled between submissions")
            complete = self._validated(P.decode_complete, body, c, self.observe)
            self._validated(plan.on_complete, complete)
        except BaseException as exc:
            if provider.state == "QUARANTINED":
                self.state = "QUARANTINED"
                raise
            raise self._host_failure(exc)
        self.state = "BOUNDARY"
        self.position += 1
        stats["polls"] = provider.runtime.metrics()["backend_polls"] - polls_before
        stats["provider_ns"] = perf_counter_ns() - start
        stats["elided_layers"] = sum(complete.elided)
        stats.update({"provider_" + k: v for k, v in complete.telemetry.items()})
        self.last = stats
        self._event("COMPLETE", stats)
        return complete
