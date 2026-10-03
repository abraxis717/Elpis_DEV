"""YTS-R0 wire codec and validation (pure: no I/O, no ctypes, no provider access).

Mirrors ``research/dsv41_tower/native/include/elpis/dsv41_stream.h``. A provider reply is
a claim, never semantic authority: every reply is checked here against what the
host itself expects (stream, epoch, sequence number, position, layer order,
expert bounds and uniqueness, the selected/needed relationship, cursor and part
order, representation, sizes, per-token bounds) before the host acts on it.
A violation is ``ContractError(INTEGRITY)``; non-finite provider output is
``ContractError(ENCODING)``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import numpy as np

from elpis.inference.contracts import Code, ContractError, require

VERSION = 1
OPERATION = 41
HEADER_BYTES = 128
MAGIC = 0x31535459
NONE = 0xFFFFFFFF

MODEL_ADMIT_BEGIN, MODEL_ADMIT_TENSOR, MODEL_ADMIT_END = 0x01, 0x02, 0x03
STREAM_OPEN, TOKEN_BEGIN, EXPERT_SUPPLY, STREAM_RELEASE, MODEL_RELEASE = 0x04, 0x05, 0x06, 0x07, 0x08
ADMIT_PART_ACK, ADMIT_ACK, OPENED, NEED, COMPLETE, RELEASED, MODEL_RELEASED = (
    0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87)
REPLY_FOR = {MODEL_ADMIT_BEGIN: (ADMIT_PART_ACK,), MODEL_ADMIT_TENSOR: (ADMIT_PART_ACK,),
             MODEL_ADMIT_END: (ADMIT_ACK,), STREAM_OPEN: (OPENED,), TOKEN_BEGIN: (NEED, COMPLETE),
             EXPERT_SUPPLY: (NEED, COMPLETE), STREAM_RELEASE: (RELEASED,), MODEL_RELEASE: (MODEL_RELEASED,)}
KIND_NAMES = {MODEL_ADMIT_BEGIN: "MODEL_ADMIT_BEGIN", MODEL_ADMIT_TENSOR: "MODEL_ADMIT_TENSOR",
              MODEL_ADMIT_END: "MODEL_ADMIT_END", STREAM_OPEN: "STREAM_OPEN", TOKEN_BEGIN: "TOKEN_BEGIN",
              EXPERT_SUPPLY: "EXPERT_SUPPLY", STREAM_RELEASE: "STREAM_RELEASE", MODEL_RELEASE: "MODEL_RELEASE"}

FEATURE_CACHE = 1 << 0
FEATURE_OBSERVE_LAYER_STREAMS = 1 << 1
REPR_F32_LE_ROW_MAJOR_OUT_IN = 1
LAYER_KV_OWNER, LAYER_INDEX_SOURCE, LAYER_ENGRAM = 1, 2, 4
ADMIT_BEGIN_FIXED_BYTES = 27 * 4 + 4 + 5 * 4 + 2 * 4 + 2 * 8 + 32
ADMIT_TENSOR_PREFIX_BYTES = 9 * 4 + 32 + 3 * 8
SUPPLY_PREFIX_BYTES = 4 * 4 + 4 * 8 + 3 * 32
SCORE_MODES = {"softmax": 0, "sigmoid": 1, "sqrtsoftplus": 2}

GLOBAL_ROLES = {"embed": 1, "head": 2, "norm": 3, "rope.local": 4, "rope.compressed": 5}
LAYER_ROLES = {
    "attn_norm": 16, "ffn_norm": 17, "attn.wq_a": 18, "attn.q_norm": 19, "attn.wq_b": 20, "attn.wkv": 21,
    "attn.kv_norm": 22, "attn.attn_sink": 23, "attn.wo_a": 24, "attn.wo_b": 25, "ffn.gate.weight": 26,
    "ffn.gate.bias": 27, "hc_attn_fn": 28, "hc_attn_base": 29, "hc_attn_scale": 30, "hc_ffn_fn": 31,
    "hc_ffn_base": 32, "hc_ffn_scale": 33, "attn.compressor.wkv": 34, "attn.compressor.norm": 35,
    "attn.compressor.wgate": 36, "attn.indexer.wk": 37, "attn.indexer.k_norm": 38, "attn.indexer.wq_b": 39,
    "attn.indexer.weights_proj": 40, "engram.wkv": 41, "engram.q_weight": 42, "engram.k_weight": 43}
EXPERT_ROLES = {"w1": 64, "w3": 65, "w2": 66}

_HEADER = struct.Struct("<IHHIIQQQIIQ32s40s")
assert _HEADER.size == HEADER_BYTES


def violation(detail):
    return ContractError(Code.INTEGRITY, "YTS-R0 protocol violation: " + detail)


def _need(condition, detail):
    if not condition:
        raise violation(detail)


def digest_bytes(digest):
    require(type(digest) is str and len(digest) == 64, detail="digest")
    return bytes.fromhex(digest)


@dataclass(frozen=True)
class Header:
    kind: int
    stream_id: int
    epoch: int
    seq: int
    position: int
    layer: int
    body_bytes: int
    manifest: bytes
    flags: int = 0

    def pack(self):
        return _HEADER.pack(MAGIC, VERSION, self.kind, HEADER_BYTES, self.flags, self.stream_id, self.epoch,
                            self.seq, self.position, self.layer, self.body_bytes, self.manifest, bytes(40))


def unpack_header(data):
    _need(len(data) >= HEADER_BYTES, "short message")
    (magic, version, kind, header_bytes, flags, stream_id, epoch, seq, position, layer, body_bytes, manifest,
     reserved) = _HEADER.unpack_from(data, 0)
    _need(magic == MAGIC and version == VERSION and header_bytes == HEADER_BYTES, "header identity")
    _need(flags == 0 and reserved == bytes(40), "reserved header fields")
    _need(body_bytes == len(data) - HEADER_BYTES, "body length")
    return Header(kind, stream_id, epoch, seq, position, layer, body_bytes, manifest)


def check_reply(reply, request):
    """Echo and kind checks of one reply against the request that produced it."""
    _need(reply.kind in REPLY_FOR[request.kind], f"reply kind {reply.kind:#x} to {KIND_NAMES[request.kind]}")
    _need((reply.stream_id, reply.epoch, reply.seq) == (request.stream_id, request.epoch, request.seq),
          "stream/epoch/sequence echo")
    _need(reply.manifest == request.manifest, "manifest echo")
    if request.kind in (TOKEN_BEGIN, EXPERT_SUPPLY):
        _need(reply.position == request.position, "position echo")
        _need(reply.layer == NONE if reply.kind == COMPLETE else reply.layer < NONE, "layer field")
    else:
        _need(reply.position == NONE and reply.layer == NONE, "model-level position/layer")
    _need(reply.kind != ADMIT_PART_ACK and reply.kind != RELEASED and reply.kind != MODEL_RELEASED
          or reply.body_bytes == 0, "empty reply body")


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

def role_code(name, expert_count):
    """(role_kind, layer, expert) for a tensor role name; derived tables included."""
    if name in GLOBAL_ROLES:
        return GLOBAL_ROLES[name], NONE, NONE
    parts = name.split(".")
    require(len(parts) >= 3 and parts[0] == "layers" and parts[1].isdigit(), Code.UNSUPPORTED, "tensor role " + name)
    layer, rest = int(parts[1]), ".".join(parts[2:])
    if rest in LAYER_ROLES:
        return LAYER_ROLES[rest], layer, NONE
    sub = rest.split(".")
    require(len(sub) == 4 and sub[0] == "ffn" and sub[1] == "experts" and sub[3] in EXPERT_ROLES,
            Code.UNSUPPORTED, "tensor role " + name)
    expert = expert_count if sub[2] == "shared" else int(sub[2])
    require(0 <= expert <= expert_count, Code.UNSUPPORTED, "expert role " + name)
    return EXPERT_ROLES[sub[3]], layer, expert


def _f32(value):
    return struct.pack("<f", float(value))


def layer_flags(config):
    flags = []
    for i in range(config.layers):
        f = (LAYER_KV_OWNER if i in config.kv_sources else 0) | (LAYER_INDEX_SOURCE if i in config.index_sources else 0)
        flags.append(f | (LAYER_ENGRAM if i in config.engram_layers else 0))
    return flags


def encode_admit_begin(config, *, features, part_bytes, cache_bytes, config_digest):
    c = config
    words = (c.vocab, c.dimension, c.layers, c.heads, c.head_dim, c.rope_dim, c.q_rank, c.o_groups, c.o_rank,
             c.local_window, c.max_tokens, c.index_heads, c.index_dim, c.index_topk, c.expert_count,
             c.active_experts, c.expert_dim, c.engram_order, c.engram_heads, c.engram_dim, c.hc_mult,
             c.hc_sinkhorn_iters, SCORE_MODES[c.score_func], int(c.norm_topk_prob), c.candidate_blocks,
             c.candidate_block_size, c.hash_columns)
    body = struct.pack("<27Ii", *words, c.candidate_source)
    # Exactly the binary32 values the native ABI receives through ctypes c_float.
    body += b"".join(_f32(x) for x in (c.norm_eps, c.hc_eps, c.gate_temp, c.route_scale, c.swiglu_limit))
    body += struct.pack("<IIQQ", features, 1, part_bytes, cache_bytes) + digest_bytes(config_digest)
    body += struct.pack(f"<{c.layers}I", *c.compress_ratios) + struct.pack(f"<{c.layers}I", *layer_flags(c))
    assert len(body) == ADMIT_BEGIN_FIXED_BYTES + 8 * c.layers
    return body


def encode_admit_tensor_prefix(kind, layer, expert, shape, digest, tensor_bytes, part_offset, part_len):
    require(1 <= len(shape) <= 4 and len(digest) == 32, detail="tensor descriptor")
    dims = tuple(shape) + (0,) * (4 - len(shape))
    return (struct.pack("<9I", kind, layer, expert, REPR_F32_LE_ROW_MAJOR_OUT_IN, len(shape), *dims) + digest +
            struct.pack("<QQQ", tensor_bytes, part_offset, part_len))


def encode_admit_end(count):
    return struct.pack("<II", count, 0)


def encode_stream_open(max_tokens, flags):
    return struct.pack("<II", max_tokens, flags)


def encode_token_begin(token, segments):
    """segments: ascending (layer, rows[n, dim] float32) for exactly the Engram layers."""
    out = [struct.pack("<II", token, len(segments))]
    for layer, rows in segments:
        rows = np.ascontiguousarray(rows, dtype="<f4")
        require(rows.ndim == 2, detail="Engram row block")
        out.append(struct.pack("<IIII", layer, rows.shape[0], rows.shape[1], 0))
        out.append(rows.tobytes())
    return b"".join(out)


def encode_supply_prefix(layer, expert, cursor, image_bytes, part_offset, part_len, digests):
    require(len(digests) == 3 and all(len(d) == 32 for d in digests), detail="expert binding digests")
    return (struct.pack("<IIIIQQQQ", layer, expert, cursor, REPR_F32_LE_ROW_MAJOR_OUT_IN, image_bytes, part_offset,
                        part_len, 0) + b"".join(digests))


# ---------------------------------------------------------------------------
# Reply bodies
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdmitAck:
    profile: bytes
    features: int
    reserved_bytes: int


def decode_admit_ack(body, requested):
    _need(len(body) == 48, "ADMIT_ACK size")
    profile = bytes(body[:32])
    features, reserved, reserved_bytes = struct.unpack_from("<IIQ", body, 32)
    _need(profile != bytes(32) and reserved == 0, "provider profile identity")
    _need(features & ~requested == 0, "granted features exceed the request")
    return AdmitAck(profile, features, reserved_bytes)


def decode_opened(body):
    _need(len(body) == 8, "OPENED size")
    return struct.unpack("<Q", body)[0]


@dataclass(frozen=True)
class Need:
    layer: int
    selected: tuple
    need: tuple
    cursor_index: int
    cursor_offset: int
    cache_hits: int


def decode_need(body, active):
    _need(len(body) >= 8, "NEED size")
    layer, k = struct.unpack_from("<II", body, 0)
    _need(k == active and len(body) >= 8 + 4 * k + 4, "NEED selected count")
    selected = struct.unpack_from(f"<{k}I", body, 8)
    (count,) = struct.unpack_from("<I", body, 8 + 4 * k)
    _need(count <= active + 1 and len(body) == 8 + 4 * k + 4 + 4 * count + 8 + 16, "NEED size")
    need = struct.unpack_from(f"<{count}I", body, 12 + 4 * k)
    cursor, reserved, offset, hits = struct.unpack_from("<IIQQ", body, 12 + 4 * k + 4 * count)
    _need(reserved == 0, "NEED reserved")
    return Need(layer, tuple(selected), tuple(need), cursor, offset, hits)


@dataclass(frozen=True)
class Complete:
    logits: np.ndarray
    selected: tuple          # per layer, router-score order
    elided: tuple
    counts: tuple
    positions: tuple         # per layer, tuple of selected compressed positions
    layer_streams: np.ndarray | None
    telemetry: dict


def decode_complete(body, config, observe):
    c = config
    fixed = 24 + 48
    per_layer = 4 * c.active_experts + 12 + 4 * c.index_topk
    streams = 4 * c.layers * c.hc_mult * c.dimension if observe else 0
    _need(len(body) == fixed + 4 * c.vocab + c.layers * per_layer + streams, "COMPLETE size")
    vocab, layers, k, topk, flags, reserved = struct.unpack_from("<6I", body, 0)
    _need((vocab, layers, k, topk, reserved) == (c.vocab, c.layers, c.active_experts, c.index_topk, 0),
          "COMPLETE geometry")
    _need(flags == (FEATURE_OBSERVE_LAYER_STREAMS if observe else 0), "COMPLETE observation flags")
    names = ("cache_bytes_in_use", "cache_entries", "cache_evictions", "cache_hits", "expert_slot_high_water",
             "supplied_experts")
    telemetry = dict(zip(names, struct.unpack_from("<6Q", body, 24)))
    logits = np.frombuffer(body, dtype="<f4", count=c.vocab, offset=fixed).copy()
    at = fixed + 4 * c.vocab
    selected, elided, counts, positions = [], [], [], []
    for _ in range(c.layers):
        selected.append(struct.unpack_from(f"<{k}I", body, at))
        e, count, npos = struct.unpack_from("<III", body, at + 4 * k)
        pos = struct.unpack_from(f"<{topk}I", body, at + 4 * k + 12)
        _need(e in (0, 1) and npos <= topk and all(p == 0 for p in pos[npos:]), "COMPLETE layer record")
        elided.append(bool(e))
        counts.append(count)
        positions.append(tuple(pos[:npos]))
        at += per_layer
    layer_streams = None
    if observe:
        layer_streams = np.frombuffer(body, dtype="<f4", count=c.layers * c.hc_mult * c.dimension,
                                      offset=at).reshape(c.layers, c.hc_mult, c.dimension).copy()
    return Complete(logits, tuple(selected), tuple(elided), tuple(counts), tuple(positions), layer_streams, telemetry)


# ---------------------------------------------------------------------------
# Per-token reply validation and supply planning (pure state machine)
# ---------------------------------------------------------------------------

class TokenPlan:
    """Validates one token's NEED/COMPLETE replies and plans every supply part.

    ``resident`` is the set of (layer, expert) admitted whole to the provider;
    ``cache`` says whether the provider was granted the optional cache. Experts
    are indexed 0..expert_count-1, with expert_count denoting the shared expert.
    """

    def __init__(self, config, *, resident, cache, image_bytes, part_bytes):
        require(part_bytes >= 1 and image_bytes >= 1, detail="supply geometry")
        self.c, self.resident, self.cache = config, frozenset(resident), bool(cache)
        self.image_bytes, self.part_bytes = image_bytes, part_bytes
        parts = math.ceil(image_bytes / part_bytes)
        self.max_supplies = config.layers * (config.active_experts + 1) * parts
        self.max_replies = 1 + self.max_supplies
        self.layer = -1
        self.record = None
        self.cursor = self.offset = 0
        self.needed = {}
        self.replies = self.supplies = 0
        self.done = False

    def _ordered(self, selected):
        return tuple(sorted(selected)) + (self.c.expert_count,)

    def _check_selected(self, selected):
        E = self.c.expert_count
        _need(len(selected) == self.c.active_experts and len(set(selected)) == len(selected), "selected experts")
        _need(all(0 <= e < E for e in selected), "selected expert out of range")

    def on_need(self, need):
        """Validate a NEED and return the next part to supply: (expert, offset, length)."""
        _need(not self.done, "NEED after COMPLETE")
        self.replies += 1
        _need(self.replies <= self.max_replies, "per-token reply bound")
        _need(0 <= need.layer < self.c.layers, "NEED layer")
        self._check_selected(need.selected)
        ordered = self._ordered(need.selected)
        nonresident = tuple(e for e in ordered if (need.layer, e) not in self.resident)
        _need(len(need.need) >= 1 and len(set(need.need)) == len(need.need), "need list")
        _need(list(need.need) == sorted(need.need) and set(need.need) <= set(nonresident), "need ⊆ selected ∪ shared")
        if self.cache:
            served = {j for j, e in enumerate(ordered) if e in nonresident and e not in need.need}
            _need(need.cache_hits == sum(1 << j for j in served), "cache-hit map")
        else:
            _need(need.need == nonresident and need.cache_hits == 0, "need without a granted cache")
        if need.layer == self.layer:
            _need((need.selected, need.need, need.cache_hits) == self.record, "NEED changed within a layer")
        else:
            _need(need.layer > self.layer, "layer order")
            if self.record is not None:
                _need(self.cursor == len(self.record[1]), "previous layer incompletely supplied")
            self.layer, self.record = need.layer, (need.selected, need.need, need.cache_hits)
            self.needed[need.layer] = need.selected
            self.cursor = self.offset = 0
        _need(need.cursor_index == self.cursor and need.cursor_offset == self.offset, "supply cursor")
        _need(self.cursor < len(need.need), "cursor beyond need list")
        length = min(self.part_bytes, self.image_bytes - self.offset)
        return need.need[self.cursor], self.offset, length

    def on_supplied(self, length):
        self.supplies += 1
        _need(self.supplies <= self.max_supplies, "per-token supply bound")
        self.offset += length
        if self.offset == self.image_bytes:
            self.cursor += 1
            self.offset = 0

    def on_complete(self, complete):
        _need(not self.done, "duplicate COMPLETE")
        self.replies += 1
        _need(self.replies <= self.max_replies, "per-token reply bound")
        if self.record is not None:
            _need(self.cursor == len(self.record[1]) and self.offset == 0, "COMPLETE before supply finished")
        c = self.c
        owner = None
        for layer in range(c.layers):
            selected = complete.selected[layer]
            self._check_selected(selected)
            if layer in self.needed:
                _need(selected == self.needed[layer] and not complete.elided[layer], "COMPLETE trace vs NEED")
            else:
                all_resident = all((layer, e) in self.resident for e in self._ordered(selected))
                _need(complete.elided[layer] and (self.cache or all_resident), "unannounced layer elision")
            if layer in c.kv_sources:
                owner = layer
            positions = complete.positions[layer]
            if not c.compress_ratios[layer]:
                _need(not positions, "positions on a local layer")
            else:
                _need(owner is not None and list(positions) == sorted(set(positions)), "selected positions order")
                _need(all(p < complete.counts[owner] for p in positions), "selected position beyond owner count")
        if not np.all(np.isfinite(complete.logits)):
            raise ContractError(Code.ENCODING, "YTS-R0 provider published non-finite logits")
        if complete.layer_streams is not None and not np.all(np.isfinite(complete.layer_streams)):
            raise ContractError(Code.ENCODING, "YTS-R0 provider published non-finite layer streams")
        self.done = True
