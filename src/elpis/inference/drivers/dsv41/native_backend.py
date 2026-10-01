"""Sealed, capability-gated DSV4.1 native backend and execution bridge.

Admission remains explicit: callers provide an independently pinned
PinnedAuthority and library identifier. The candidate ELF is opened only through
RootCapability + load_native(), so the bytes executed are the sealed bytes whose
SHA-256 was independently authorized.

R10B adds the Python orchestration bridge over the already-qualified R0-R9 ABI.
Row lookup and TensorStore/FMS expert materialization remain Python-owned.
"""
from __future__ import annotations

import ctypes as C
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from elpis.substrate.authority import PinnedAuthority
from elpis.substrate.boundary import RootCapability, load_native
from ...contracts import Code, ContractError, RowIdentity, require


ABI_VERSION = 1
REQUIRED_CAPABILITIES = (1 << 23) - 1

CAPABILITY_NAMES = (
    "LINEAR_F32",
    "RMS_F32",
    "HC_PRE_F32",
    "HC_POST_F32",
    "QUANT_DEQUANT_F32",
    "ROPE_F32",
    "CANDIDATE_MASK_F32",
    "SELECT_POSITIONS_F32",
    "SPARSE_ATTN_F32",
    "HC_MIXES_F32",
    "ROUTE_F32",
    "EXPERT_F32",
    "ATTN_STATE_V1",
    "ENGRAM_GATED_WRITE_F32",
    "INDEX_SCORES_F32",
    "GROUPED_OUTPUT_F32",
    "LOCAL_ATTN_APPLY_F32",
    "COMPRESSED_ATTN_APPLY_F32",
    "ROUTE_ORDER_U32",
    "EXPERT_ACCUMULATE_F32",
    "LAYER_BEGIN_F32",
    "LAYER_AFTER_ATTN_ROUTE_F32",
    "LAYER_FINISH_F32",
)

_REQUIRED_SYMBOLS = (
    "elpis_dsv41_native_abi_version",
    "elpis_dsv41_native_capabilities",
    "elpis_dsv41_linear_f32",
    "elpis_dsv41_rms_f32",
    "elpis_dsv41_hc_pre_f32",
    "elpis_dsv41_hc_post_f32",
    "elpis_dsv41_quant_dequant_f32",
    "elpis_dsv41_rope_f32",
    "elpis_dsv41_candidate_mask_f32",
    "elpis_dsv41_select_positions_f32",
    "elpis_dsv41_sparse_attention_f32",
    "elpis_dsv41_hc_mixes_f32",
    "elpis_dsv41_route_f32",
    "elpis_dsv41_expert_f32",
    "elpis_dsv41_attention_state_create",
    "elpis_dsv41_attention_state_destroy",
    "elpis_dsv41_attention_state_count",
    "elpis_dsv41_attention_local_store_f32",
    "elpis_dsv41_attention_local_copy_f32",
    "elpis_dsv41_attention_compress_push_f32",
    "elpis_dsv41_attention_publish_group_f32",
    "elpis_dsv41_attention_index_copy_f32",
    "elpis_dsv41_attention_gather_compressed_f32",
    "elpis_dsv41_engram_gated_write_f32",
    "elpis_dsv41_index_scores_f32",
    "elpis_dsv41_grouped_output_f32",
    "elpis_dsv41_local_attention_scratch_floats",
    "elpis_dsv41_local_attention_apply_f32",
    "elpis_dsv41_compressed_attention_scratch_floats",
    "elpis_dsv41_compressed_attention_apply_f32",
    "elpis_dsv41_route_order_u32",
    "elpis_dsv41_expert_accumulate_f32",
    "elpis_dsv41_layer_frame_scratch_floats",
    "elpis_dsv41_layer_begin_f32",
    "elpis_dsv41_layer_after_attention_route_f32",
    "elpis_dsv41_layer_finish_f32",
)

P = C.POINTER(C.c_float)
U32P = C.POINTER(C.c_uint32)
U8P = C.POINTER(C.c_uint8)
StateP = C.c_void_p

_SCORE_MODES = {
    "softmax": 0,
    "sigmoid": 1,
    "sqrtsoftplus": 2,
}


def _symbol(lib, name):
    try:
        return getattr(lib, name)
    except AttributeError as exc:
        raise ContractError(Code.UNSUPPORTED, "DSV4.1 native symbol: " + name) from exc


def _f32(array):
    value = np.asarray(array)
    require(value.dtype == np.dtype("<f4") and value.flags.c_contiguous,
            Code.ENCODING, "native contiguous F32")
    return value


def _fptr(array):
    return _f32(array).ctypes.data_as(P)


def _fptr_or_none(array):
    return None if array is None else _fptr(array)


def _u32ptr(array):
    value = np.asarray(array)
    require(value.dtype == np.dtype("uint32") and value.flags.c_contiguous,
            Code.ENCODING, "native contiguous U32")
    return value.ctypes.data_as(U32P)


def _u8ptr(array):
    value = np.asarray(array)
    require(value.dtype == np.dtype("uint8") and value.flags.c_contiguous,
            Code.ENCODING, "native contiguous U8")
    return value.ctypes.data_as(U8P)


def _check(rc, detail):
    require(int(rc) == 0, Code.ENCODING, "native DSV4.1 " + detail)


class NativeAttentionState:
    """Sequence-local native attention state owned by one TowerState."""

    __slots__ = ("backend", "handle", "_nbytes", "closed")

    def __init__(self, backend, config, layer):
        self.backend = backend
        ratio = config.compress_ratios[layer]
        owner = layer in config.kv_sources
        handle = StateP()
        _check(
            backend._lib.elpis_dsv41_attention_state_create(
                config.local_window,
                config.head_dim,
                config.index_dim,
                config.max_tokens,
                ratio,
                int(owner),
                C.byref(handle),
            ),
            "attention state create",
        )
        require(bool(handle.value), Code.ENCODING, "native attention state handle")
        capacity = config.max_tokens // ratio if owner else 0
        floats = config.local_window * config.head_dim
        if owner:
            floats += capacity * (config.head_dim + config.index_dim)
            if ratio > 1:
                floats += 2 * ratio * config.head_dim
        self.handle = handle
        self._nbytes = 4 * floats
        self.closed = False

    @property
    def nbytes(self):
        return 0 if self.closed else self._nbytes

    @property
    def count(self):
        require(not self.closed, Code.CLOSED, "native attention state")
        return int(self.backend._lib.elpis_dsv41_attention_state_count(self.handle))

    def close(self):
        if not self.closed:
            _check(
                self.backend._lib.elpis_dsv41_attention_state_destroy(
                    C.byref(self.handle)
                ),
                "attention state destroy",
            )
            self.closed = True
            self._nbytes = 0


class DSV41NativeBackend:
    """Explicitly admitted cumulative R0-R9 native DSV4.1 execution backend."""

    __slots__ = (
        "_lib",
        "library_id",
        "identity",
        "abi_version",
        "capabilities",
    )

    def __init__(self, root, library, *, authority, library_id):
        require(type(authority) is PinnedAuthority,
                Code.IDENTITY, "DSV4.1 native authority")
        require(type(library_id) is str and library_id in authority.libraries,
                Code.IDENTITY, "DSV4.1 native library identifier")
        require(authority.provenance in ("deployment", "synthetic-test"),
                Code.IDENTITY, "DSV4.1 native authority provenance")

        root = Path(root)
        library = Path(library)
        require(root.is_absolute(), detail="DSV4.1 native root")
        require(".." not in library.parts and "\x00" not in str(library),
                detail="DSV4.1 native path")

        identity = authority.libraries[library_id]
        with RootCapability(root) as boundary:
            lib = load_native(boundary, library, identity)

        abi = _symbol(lib, "elpis_dsv41_native_abi_version")
        abi.argtypes = []
        abi.restype = C.c_uint32

        capabilities = _symbol(lib, "elpis_dsv41_native_capabilities")
        capabilities.argtypes = []
        capabilities.restype = C.c_uint64

        version = int(abi())
        bitmap = int(capabilities())
        require(version == ABI_VERSION, Code.UNSUPPORTED,
                "DSV4.1 native ABI version")
        require(bitmap & REQUIRED_CAPABILITIES == REQUIRED_CAPABILITIES,
                Code.UNSUPPORTED, "DSV4.1 cumulative native capabilities")

        for name in _REQUIRED_SYMBOLS:
            _symbol(lib, name)

        self._bind_execution_abi(lib)

        self._lib = lib
        self.library_id = library_id
        self.identity = identity
        self.abi_version = version
        self.capabilities = bitmap

    @staticmethod
    def _bind_execution_abi(lib):
        lib.elpis_dsv41_attention_state_create.argtypes = [
            C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t,
            C.c_size_t, C.c_int, C.POINTER(StateP),
        ]
        lib.elpis_dsv41_attention_state_create.restype = C.c_int
        lib.elpis_dsv41_attention_state_destroy.argtypes = [C.POINTER(StateP)]
        lib.elpis_dsv41_attention_state_destroy.restype = C.c_int
        lib.elpis_dsv41_attention_state_count.argtypes = [StateP]
        lib.elpis_dsv41_attention_state_count.restype = C.c_size_t

        lib.elpis_dsv41_local_attention_scratch_floats.argtypes = [
            C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t,
            C.c_size_t, C.c_size_t, C.c_size_t,
        ]
        lib.elpis_dsv41_local_attention_scratch_floats.restype = C.c_size_t
        lib.elpis_dsv41_local_attention_apply_f32.argtypes = [
            StateP, C.c_size_t,
            P, P, P, P, P, P, P, P, P, P,
            C.c_float, P, C.c_size_t, P,
            C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t,
            C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t,
        ]
        lib.elpis_dsv41_local_attention_apply_f32.restype = C.c_int

        lib.elpis_dsv41_compressed_attention_scratch_floats.argtypes = [
            C.c_size_t, C.c_size_t, C.c_size_t,
            C.c_size_t, C.c_size_t,
            C.c_size_t, C.c_size_t,
            C.c_size_t, C.c_size_t, C.c_size_t,
        ]
        lib.elpis_dsv41_compressed_attention_scratch_floats.restype = C.c_size_t
        lib.elpis_dsv41_compressed_attention_apply_f32.argtypes = [
            StateP, StateP, C.c_size_t,
            P, P, P, P, P, P, P, P, P,
            P, P, P, P, P, P, P,
            P, P, C.c_float,
            U32P, C.c_size_t, C.POINTER(C.c_size_t),
            U8P, C.c_size_t, C.POINTER(C.c_size_t),
            P, C.c_size_t, P,
            C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t,
            C.c_size_t, C.c_size_t, C.c_size_t,
            C.c_size_t, C.c_size_t, C.c_size_t,
            C.c_size_t, C.c_size_t, C.c_size_t,
            C.c_int, C.c_int, C.c_int,
            C.c_size_t, C.c_size_t,
        ]
        lib.elpis_dsv41_compressed_attention_apply_f32.restype = C.c_int

        lib.elpis_dsv41_layer_frame_scratch_floats.argtypes = [
            C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t, C.c_int,
        ]
        lib.elpis_dsv41_layer_frame_scratch_floats.restype = C.c_size_t
        lib.elpis_dsv41_layer_begin_f32.argtypes = [
            P, P,
            P, P, P, P,
            C.c_int, C.c_size_t,
            P, P, P, P,
            C.c_float, C.c_float, C.c_size_t,
            P, C.c_size_t,
            P, P, P, P, P,
            C.c_size_t, C.c_size_t,
        ]
        lib.elpis_dsv41_layer_begin_f32.restype = C.c_int
        lib.elpis_dsv41_layer_after_attention_route_f32.argtypes = [
            P, P, P, P, P,
            P, P, P, P,
            P, P,
            C.c_float, C.c_float, C.c_size_t,
            C.c_uint32, C.c_float, C.c_int, C.c_float,
            P, C.c_size_t,
            P, P, P, P, P,
            U32P, P, U32P,
            C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t,
        ]
        lib.elpis_dsv41_layer_after_attention_route_f32.restype = C.c_int
        lib.elpis_dsv41_layer_finish_f32.argtypes = [
            P, P, P, P, P, C.c_size_t, C.c_size_t,
        ]
        lib.elpis_dsv41_layer_finish_f32.restype = C.c_int

        lib.elpis_dsv41_expert_accumulate_f32.argtypes = [
            P, P, P, P,
            C.c_float, C.c_int, C.c_float,
            P, P, P, P,
            C.c_size_t, C.c_size_t,
        ]
        lib.elpis_dsv41_expert_accumulate_f32.restype = C.c_int

        lib.elpis_dsv41_hc_pre_f32.argtypes = [
            P, P, P, C.c_size_t, C.c_size_t,
        ]
        lib.elpis_dsv41_hc_pre_f32.restype = C.c_int
        lib.elpis_dsv41_rms_f32.argtypes = [
            P, P, C.c_float, P, C.c_size_t,
        ]
        lib.elpis_dsv41_rms_f32.restype = C.c_int
        lib.elpis_dsv41_linear_f32.argtypes = [
            P, P, P, C.c_size_t, C.c_size_t,
        ]
        lib.elpis_dsv41_linear_f32.restype = C.c_int

    @property
    def capability_names(self):
        return tuple(
            name for bit, name in enumerate(CAPABILITY_NAMES)
            if self.capabilities & (1 << bit)
        )

    def local_attention_scratch_floats(
        self, dimension, q_rank, heads, head_dim,
        o_groups, o_rank, local_window,
    ):
        value = int(self._lib.elpis_dsv41_local_attention_scratch_floats(
            dimension, q_rank, heads, head_dim,
            o_groups, o_rank, local_window,
        ))
        require(value > 0, Code.INVALID, "native local-attention scratch geometry")
        return value

    def compressed_attention_scratch_floats(
        self, q_rank, heads, head_dim, index_heads, index_dim,
        o_groups, o_rank, local_window, max_groups, index_topk,
    ):
        value = int(self._lib.elpis_dsv41_compressed_attention_scratch_floats(
            q_rank, heads, head_dim, index_heads, index_dim,
            o_groups, o_rank, local_window, max_groups, index_topk,
        ))
        require(value > 0, Code.INVALID,
                "native compressed-attention scratch geometry")
        return value

    def layer_frame_scratch_floats(
        self, copies, dim, row_values, expert_count, engram_enabled,
    ):
        value = int(self._lib.elpis_dsv41_layer_frame_scratch_floats(
            copies, dim, row_values, expert_count, int(bool(engram_enabled)),
        ))
        require(value > 0, Code.INVALID, "native layer-frame scratch geometry")
        return value

    def create_attention_state(self, config, layer):
        return NativeAttentionState(self, config, layer)

    def _local_attention(self, layer, x, state, position):
        c, attention, w = layer.config, layer.attention, layer.attention.w
        freq = attention.freqs[position]
        rope_pairs = freq.shape[0]
        scratch_n = self.local_attention_scratch_floats(
            c.dimension, c.q_rank, c.heads, c.head_dim,
            c.o_groups, c.o_rank, c.local_window,
        )
        scratch = np.empty(scratch_n, dtype="<f4")
        out = np.empty(c.dimension, dtype="<f4")
        _check(
            self._lib.elpis_dsv41_local_attention_apply_f32(
                state.handle,
                position,
                _fptr(x),
                _fptr(w["wq_a"]),
                _fptr(w["q_norm"]),
                _fptr(w["wq_b"]),
                _fptr(w["wkv"]),
                _fptr(w["kv_norm"]),
                _fptr(w["attn_sink"]),
                _fptr(w["wo_a"]),
                _fptr(w["wo_b"]),
                _fptr(freq),
                c.norm_eps,
                _fptr(scratch),
                scratch_n,
                _fptr(out),
                c.dimension,
                c.q_rank,
                c.heads,
                c.head_dim,
                rope_pairs,
                c.o_groups,
                c.o_rank,
                c.local_window,
            ),
            "local attention apply",
        )
        return out, ()

    def _compressed_attention(self, layer, x, state, shared, position):
        c, attention, w = layer.config, layer.attention, layer.attention.w
        ratio = attention.ratio
        require(ratio > 0, detail="compressed attention ratio")
        if attention.kv_owner:
            shared.owner = state
        owner = shared.owner
        require(type(owner) is NativeAttentionState and not owner.closed,
                Code.STALE, "native shared-attention owner")

        max_groups = c.max_tokens // ratio
        selected_buf = np.zeros(c.index_topk, dtype=np.uint32)
        selected_count = C.c_size_t(0)
        if shared.selected is not None:
            previous = np.asarray(shared.selected, dtype=np.uint32)
            require(previous.size <= c.index_topk,
                    Code.ENCODING, "native shared selection capacity")
            selected_buf[:previous.size] = previous
            selected_count.value = previous.size

        candidates_buf = np.zeros(max_groups, dtype=np.uint8)
        candidate_count = C.c_size_t(0)
        if shared.candidates is not None:
            previous = np.asarray(shared.candidates, dtype=np.uint8)
            require(previous.size <= max_groups,
                    Code.ENCODING, "native shared candidate capacity")
            candidates_buf[:previous.size] = previous
            candidate_count.value = previous.size

        if attention.index_source and layer.layer == c.candidate_source:
            candidate_mode = 1
        elif attention.index_source and 0 <= c.candidate_source < layer.layer:
            candidate_mode = 2
        else:
            candidate_mode = 0

        freq = attention.freqs[position]
        rope_pairs = freq.shape[0]
        group_freq = None
        if attention.kv_owner and (position + 1) % ratio == 0:
            group_freq = attention.freqs[position + 1 - ratio]

        scratch_n = self.compressed_attention_scratch_floats(
            c.q_rank, c.heads, c.head_dim,
            c.index_heads, c.index_dim,
            c.o_groups, c.o_rank,
            c.local_window, max_groups, c.index_topk,
        )
        scratch = np.empty(scratch_n, dtype="<f4")
        out = np.empty(c.dimension, dtype="<f4")

        _check(
            self._lib.elpis_dsv41_compressed_attention_apply_f32(
                state.handle,
                owner.handle,
                position,
                _fptr(x),
                _fptr(w["wq_a"]),
                _fptr(w["q_norm"]),
                _fptr(w["wq_b"]),
                _fptr(w["wkv"]),
                _fptr(w["kv_norm"]),
                _fptr(w["attn_sink"]),
                _fptr(w["wo_a"]),
                _fptr(w["wo_b"]),
                _fptr_or_none(w.get("compressor.wkv")),
                _fptr_or_none(w.get("compressor.wgate")),
                _fptr_or_none(w.get("compressor.norm")),
                _fptr_or_none(w.get("indexer.wk")),
                _fptr_or_none(w.get("indexer.k_norm")),
                _fptr_or_none(w.get("indexer.wq_b")),
                _fptr_or_none(w.get("indexer.weights_proj")),
                _fptr(freq),
                _fptr_or_none(group_freq),
                c.norm_eps,
                _u32ptr(selected_buf),
                c.index_topk,
                C.byref(selected_count),
                _u8ptr(candidates_buf),
                max_groups,
                C.byref(candidate_count),
                _fptr(scratch),
                scratch_n,
                _fptr(out),
                c.dimension,
                c.q_rank,
                c.heads,
                c.head_dim,
                rope_pairs,
                c.index_heads,
                c.index_dim,
                c.o_groups,
                c.o_rank,
                c.local_window,
                ratio,
                max_groups,
                c.index_topk,
                int(attention.kv_owner),
                int(attention.index_source),
                candidate_mode,
                c.candidate_blocks,
                c.candidate_block_size,
            ),
            "compressed attention apply",
        )

        selected = selected_buf[:selected_count.value].copy()
        if attention.index_source:
            shared.selected = selected
        if candidate_mode == 1:
            shared.candidates = candidates_buf[:candidate_count.value].astype(
                bool, copy=True
            )
        return out, tuple(int(i) for i in selected)

    def apply_layer(
        self, layer, stream, pre_mix, attention_state,
        shared, position, row_ids, metrics,
    ):
        c, w = layer.config, layer.w
        require(type(attention_state) is NativeAttentionState and
                attention_state.backend is self and not attention_state.closed,
                Code.IDENTITY, "native layer attention state")

        engram_enabled = layer.engram is not None
        row_values = 0
        rows = None
        lookup_start = perf_counter_ns()
        if engram_enabled:
            rows = layer.engram.rows.lookup(tuple(
                RowIdentity(layer.engram.bank_id, r) for r in row_ids
            ))
            rows = np.ascontiguousarray(rows, dtype="<f4")
            row_values = rows.size
            metrics["engram_ns"] += perf_counter_ns() - lookup_start

        scratch_n = self.layer_frame_scratch_floats(
            c.hc_mult, c.dimension, row_values,
            c.expert_count, engram_enabled,
        )
        scratch = np.empty(scratch_n, dtype="<f4")
        stream_work = np.empty_like(stream)
        ap = np.empty(c.hc_mult, dtype="<f4")
        apost = np.empty(c.hc_mult, dtype="<f4")
        ac = np.empty((c.hc_mult, c.hc_mult), dtype="<f4")
        attention_x = np.empty(c.dimension, dtype="<f4")

        begin = perf_counter_ns()
        _check(
            self._lib.elpis_dsv41_layer_begin_f32(
                _fptr(stream),
                _fptr(pre_mix),
                _fptr_or_none(rows),
                _fptr_or_none(
                    None if not engram_enabled else layer.engram.weights["wkv"]
                ),
                _fptr_or_none(
                    None if not engram_enabled else layer.engram.weights["q_weight"]
                ),
                _fptr_or_none(
                    None if not engram_enabled else layer.engram.weights["k_weight"]
                ),
                int(engram_enabled),
                row_values,
                _fptr(w["hc_attn_fn"]),
                _fptr(w["hc_attn_scale"]),
                _fptr(w["hc_attn_base"]),
                _fptr(w["attn_norm"]),
                c.norm_eps,
                c.hc_eps,
                c.hc_sinkhorn_iters,
                _fptr(scratch),
                scratch_n,
                _fptr(stream_work),
                _fptr(ap),
                _fptr(apost),
                _fptr(ac),
                _fptr(attention_x),
                c.hc_mult,
                c.dimension,
            ),
            "layer begin",
        )
        metrics["mhc_norm_ns"] += perf_counter_ns() - begin

        start = perf_counter_ns()
        if layer.attention.ratio:
            attention_out, selected = self._compressed_attention(
                layer, attention_x, attention_state, shared, position
            )
        else:
            attention_out, selected = self._local_attention(
                layer, attention_x, attention_state, position
            )
        metrics["attention_ns"] += perf_counter_ns() - start

        stream_after = np.empty_like(stream)
        fp = np.empty(c.hc_mult, dtype="<f4")
        fpost = np.empty(c.hc_mult, dtype="<f4")
        fc = np.empty((c.hc_mult, c.hc_mult), dtype="<f4")
        moe_x = np.empty(c.dimension, dtype="<f4")
        chosen = np.empty(c.active_experts, dtype=np.uint32)
        route_values = np.empty(c.active_experts, dtype="<f4")
        route_order = np.empty(c.active_experts, dtype=np.uint32)

        start = perf_counter_ns()
        _check(
            self._lib.elpis_dsv41_layer_after_attention_route_f32(
                _fptr(stream_work),
                _fptr(attention_out),
                _fptr(ap),
                _fptr(apost),
                _fptr(ac),
                _fptr(w["hc_ffn_fn"]),
                _fptr(w["hc_ffn_scale"]),
                _fptr(w["hc_ffn_base"]),
                _fptr(w["ffn_norm"]),
                _fptr(layer.moe.router),
                _fptr(layer.moe.bias),
                c.norm_eps,
                c.hc_eps,
                c.hc_sinkhorn_iters,
                _SCORE_MODES[c.score_func],
                c.gate_temp,
                int(c.norm_topk_prob),
                c.route_scale,
                _fptr(scratch),
                scratch_n,
                _fptr(stream_after),
                _fptr(fp),
                _fptr(fpost),
                _fptr(fc),
                _fptr(moe_x),
                _u32ptr(chosen),
                _fptr(route_values),
                _u32ptr(route_order),
                c.hc_mult,
                c.dimension,
                c.expert_count,
                c.active_experts,
            ),
            "layer after attention/route",
        )
        metrics["mhc_norm_ns"] += perf_counter_ns() - start

        gate = np.empty(c.expert_dim, dtype="<f4")
        up = np.empty(c.expert_dim, dtype="<f4")
        expert_out = np.empty(c.dimension, dtype="<f4")
        moe_out = np.zeros(c.dimension, dtype="<f4")

        start = perf_counter_ns()
        for route_index in route_order:
            ri = int(route_index)
            expert_id = int(chosen[ri])
            with layer.moe.store.expert(layer.moe.roles[expert_id]) as tensors:
                w1, w3, w2 = tensors
                _check(
                    self._lib.elpis_dsv41_expert_accumulate_f32(
                        _fptr(moe_x),
                        _fptr(w1),
                        _fptr(w3),
                        _fptr(w2),
                        float(route_values[ri]),
                        1,
                        c.swiglu_limit,
                        _fptr(gate),
                        _fptr(up),
                        _fptr(expert_out),
                        _fptr(moe_out),
                        c.dimension,
                        c.expert_dim,
                    ),
                    "selected expert accumulate",
                )

        with layer.moe.store.expert(layer.moe.roles[-1]) as tensors:
            w1, w3, w2 = tensors
            _check(
                self._lib.elpis_dsv41_expert_accumulate_f32(
                    _fptr(moe_x),
                    _fptr(w1),
                    _fptr(w3),
                    _fptr(w2),
                    0.0,
                    0,
                    c.swiglu_limit,
                    _fptr(gate),
                    _fptr(up),
                    _fptr(expert_out),
                    _fptr(moe_out),
                    c.dimension,
                    c.expert_dim,
                ),
                "shared expert accumulate",
            )
        metrics["moe_ns"] += perf_counter_ns() - start

        stream_out = np.empty_like(stream)
        start = perf_counter_ns()
        _check(
            self._lib.elpis_dsv41_layer_finish_f32(
                _fptr(moe_out),
                _fptr(stream_after),
                _fptr(fpost),
                _fptr(fc),
                _fptr(stream_out),
                c.hc_mult,
                c.dimension,
            ),
            "layer finish",
        )
        metrics["mhc_norm_ns"] += perf_counter_ns() - start

        route = tuple(int(i) for i in chosen)
        layer.moe.last_route = route
        return stream_out, fp, route, selected

    def final_head(self, stream, pre_mix, norm, head, logits, config):
        pre = np.empty(config.dimension, dtype="<f4")
        hidden = np.empty(config.dimension, dtype="<f4")
        _check(
            self._lib.elpis_dsv41_hc_pre_f32(
                _fptr(stream),
                _fptr(pre_mix),
                _fptr(pre),
                config.hc_mult,
                config.dimension,
            ),
            "final hc pre",
        )
        _check(
            self._lib.elpis_dsv41_rms_f32(
                _fptr(pre),
                _fptr(norm),
                config.norm_eps,
                _fptr(hidden),
                config.dimension,
            ),
            "final rms",
        )
        _check(
            self._lib.elpis_dsv41_linear_f32(
                _fptr(hidden),
                _fptr(head),
                _fptr(logits),
                config.dimension,
                config.vocab,
            ),
            "final head",
        )
