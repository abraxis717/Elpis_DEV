"""ELPIS_TORCH_R9 P3 — whole-model memory and bandwidth accounting (Boardy Gate B). RESEARCH_ONLY.

EXACT (formulas, and the sizes of state tensors as allocated), MEASURED (timed or counted on this machine) and
PROJECTED (EXACT quantities under the frozen assumptions of ``results/r9/preregistration.json``, key ``p3``) are kept
in separate fields; nothing projected is reported as measured.

Commands
    account    components by context length (1K-1M, history capacity 1M, plus the steady state after the R5 ring
               has wrapped) for the R9 study arm, the R8 timing geometry and r6_500m / r6_1b (meta device): weights
               resident and read per token, CGC, ERSA state, window KV, the R5 token ring, descriptors, R8 fine keys,
               PLE, routing and decode/prefill scratch; per decoded token on the active path: the UNIQUE persistent
               bytes a token must bring in (the compulsory traffic, nothing reused across tokens) and FLOPs
    trace      EXECUTED logical bytes of single decode steps (every aten op intercepted; operands labelled by the
               storage they live in: a weight, a state tier, a lookup row, a buffer or a step-local intermediate),
               by data class and pipeline stage, at each length, after a synthetic history fill that is validated
               against a real prefill. Logical operand bytes, not DRAM transfers (this VM exposes no PMU)
    bandwidth  MEASURED memory-copy bandwidth of this machine (large tensor copies, best of several)
    rss        MEASURED process RSS / USS growth when one session state at capacity 1M is created, filled and one
               token decoded (enc256, PLE on; one fine-key mode per process), against its EXACT bytes
    roofline   PROJECTED DDR3-scenario and GPU-scenario ceilings from ``account`` and the frozen assumptions
    measured   the R8 admission records measured for R9, summarized by length (decode p50/p95, stages, RSS)

Nothing here changes a model: ``trace`` observes ops through ``TorchDispatchMode`` and module hooks that return
nothing. Records go only where ``--out`` says.
"""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import inspect
import json
import resource
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from ..context_runtime import context_interface
from ..ersa_model import ErsaLM
from ..evidence_model import EvidenceLM
from ..model import Rotary
from ..sparse_channel import active_parameters
from . import run_r6, run_r8, run_r9

PHASE = "ELPIS_TORCH_R9_BOARDY_QUALIFICATION_P3"
RUNNER = Path(__file__).resolve()
PACKAGE = str(RUNNER.parents[1]) + "/"
GiB, MiB = 1 << 30, 1 << 20


def _evidence_model(name: str, fine_keys: str, *, device="meta"):
    """A model with an R5 head (on the meta device unless asked: never allocated)."""
    capacity = run_r9.frozen()["p3"]["capacity"]
    if name in run_r6.SCALES:
        model = run_r6.scale_model(name, capacity=capacity, device=device)
    else:
        with torch.device(device):
            model = run_r8.timing_model("enc256", capacity=capacity)
    model.evidence = model.evidence.replace(fine_keys=fine_keys)
    return model


def _gather_tables(model) -> dict:
    """``nn.Embedding`` tables read by row: every one but the tied input/output table ``embed`` (its full matrix is
    the output head's, read for every token) and the PLE table (accounted on its own): name -> (rows, width)."""
    return {name: tuple(module.weight.shape) for name, module in model.named_modules()
            if isinstance(module, torch.nn.Embedding) and name not in ("embed", "ple.table")}


def _encoder_matmul_parameters(model) -> int:
    """The evidence encoder's matrix weights: its parameters minus its row-read tables and position table."""
    gathers = sum(r * w for n, (r, w) in _gather_tables(model).items() if n.startswith("encoder."))
    position = sum(p.numel() for n, p in model.named_parameters() if n.startswith("encoder.") and "position" in n)
    return sum(p.numel() for n, p in model.named_parameters() if n.startswith("encoder.")) - gathers - position


def _ple_parameters(model) -> dict:
    ple = getattr(model, "ple", None)
    if ple is None:
        return {"table": 0, "other": 0, "rows_read_per_token": 0, "head_dim": 0}
    table = ple.table.weight.numel()
    return {"table": table, "other": sum(p.numel() for p in ple.parameters()) - table,
            "rows_read_per_token": ple.config.lookups, "head_dim": ple.config.head_dim}


def _cgc(model) -> dict:
    """CGC parameters, and the CGC intermediates of one token on the active path (bottleneck, cells, channels)."""
    params, per_token = 0, 0
    for block in model.blocks:
        channel = getattr(block, "sparse", None)
        experts = list(channel.experts) + ([channel.shared] if channel is not None and channel.shared else []) \
            if channel is not None else []
        dense = [] if channel is not None else [block]
        for e in experts:
            if e.cgc is not None:
                params += sum(p.numel() for p in e.cgc.parameters()) + 1
        if channel is not None and experts and experts[0].cgc is not None:
            core = experts[0].cgc.core
            width = core.in_dim + core.channels * core.units + core.channels
            per_token += (channel.config.top_k + (1 if channel.config.shared else 0)) * width
        for b in dense:
            cgc = getattr(b, "cgc", None)
            if cgc is not None:
                params += sum(p.numel() for p in cgc.parameters())
                per_token += cgc.core.in_dim + cgc.core.channels * cgc.core.units + cgc.core.channels
    return {"parameters": params, "intermediate_values_per_token": per_token}


def _routing(model) -> dict:
    blocks = [b.sparse for b in model.blocks if getattr(b, "sparse", None) is not None]
    return {"routed_blocks": len(blocks),
            "router_logits_per_token": sum(c.config.experts for c in blocks),
            "routing_index_per_token": sum(c.config.top_k for c in blocks)}


def _window(model) -> dict:
    """The local attention window: layers, query and KV heads, head width, window length (zeros without one)."""
    ersa = model.ersa
    layers = len(ersa.window_layers) if getattr(ersa, "attention", None) == "window" else 0
    if not layers:
        return {"layers": 0, "heads": 0, "kv_heads": 0, "head_dim": 0, "window": 0}
    return {"layers": layers, "heads": ersa.attn_heads, "kv_heads": ersa.attn_kv_heads,
            "head_dim": ersa.model.d_model // ersa.attn_heads, "window": ersa.attn_window}


def account_configuration(label: str, model, *, ple: bool, lengths, scalar_bytes: int) -> dict:
    """EXACT components and PROJECTED per-token traffic and FLOPs of one configuration at every length.

    The traffic is the COMPULSORY (unique) traffic of a decoded token: every persistent byte the token needs, once
    (weights, state tiers, gathered rows), nothing reused across tokens, step-local intermediates excluded. The
    executed operand traffic of the same step (``trace``) is larger: intermediates and repeated reads."""
    counts = active_parameters(model)
    ple_p = _ple_parameters(model)
    if not ple:                                   # the PLE removed from the architecture (not merely disabled)
        counts = {"total": counts["total"] - ple_p["table"] - ple_p["other"],
                  "active_per_token": counts["active_per_token"] - ple_p["rows_read_per_token"] * ple_p["head_dim"]
                  - ple_p["other"]}
        ple_p = {"table": 0, "other": 0, "rows_read_per_token": 0, "head_dim": 0}
    face = context_interface(model)
    fixed = face.fixed_bytes(1)
    evidence = getattr(model, "evidence", None)
    window = _window(model)
    cgc = _cgc(model)
    encoder = _encoder_matmul_parameters(model)
    gathers = _gather_tables(model)
    gather_values = sum(r * w for r, w in gathers.values())
    encoder_row_width = sum(w for n, (r, w) in gathers.items() if n.startswith("encoder."))
    d_model = model.config.d_model
    # matrix weights on the active path: active parameters without row-read tables and without the PLE rows
    matmul_active = counts["active_per_token"] - gather_values - ple_p["rows_read_per_token"] * ple_p["head_dim"]
    core_matmul = matmul_active - encoder
    points = [(length, False) for length in lengths]
    if evidence is not None:                      # an endless stream lives here: the ring has wrapped
        points.append((evidence.capacity + 4096, True))
    rows = []
    for length, wrapped in points:
        live = length if evidence is None else min(length, evidence.capacity)
        row = {"length": length, "steady_state_wrapped": wrapped}
        exact = {"weights_resident_parameters": counts["total"],
                 "weights_active_parameters_per_token": counts["active_per_token"],
                 "cgc_parameters": cgc["parameters"],
                 "ple_table_parameters": ple_p["table"], "ple_other_parameters": ple_p["other"],
                 "ersa_state_bytes": fixed.get("ersa_cells", 0), "conv_tail_bytes": fixed.get("conv_tail", 0),
                 "window_kv_bytes_allocated": fixed.get("attention_window", 0),
                 "window_kv_bytes_live": (fixed.get("attention_window", 0) * min(length, window["window"])
                                          // window["window"]) if window["window"] else 0}
        blocks = 0
        if evidence is not None:
            blocks = live // evidence.block - (1 if wrapped else 0)   # wrapped: the edge block lost its left context
            exact.update({
                "r5_token_ring_bytes_allocated": fixed["history_tokens"],
                "r5_token_ring_bytes_live": live * 4,
                "descriptor_bytes_allocated": fixed["index_keys"] + fixed["index_ids"],
                "descriptor_bytes_live": blocks * evidence.per_block * evidence.key_dim * 4 + blocks * 8,
                "fine_key_bytes_allocated": fixed.get("index_fine_keys", 0),
                "fine_key_bytes_live": (blocks * evidence.block * evidence.fine_dim * 4
                                        if evidence.fine_keys == "stored" else 0)})
        exact["fixed_state_bytes_allocated_total"] = fixed["total"]
        exact["row_read_tables"] = {n: {"rows": r, "width": w} for n, (r, w) in gathers.items()}
        exact["encoder_matrix_parameters"] = encoder
        row["exact"] = exact
        scratch = face.scratch_bytes(batch=1, chunk=1, options={}, queries=1)["total"] if evidence is not None \
            else None
        row["projected_scratch"] = {
            "decode_step_bytes": scratch,
            "prefill_512_every_position_bytes": face.scratch_bytes(batch=1, chunk=512, options={}, queries=512)["total"]
            if evidence is not None else None,
            "routing_values_per_token": _routing(model),
            "cgc_intermediate_values_per_token": cgc["intermediate_values_per_token"],
            "note": "PROJECTED working memory: the largest intermediates as if live at once; allocator reserve and "
                    "fragmentation are not modelled"}
        traffic = {"weights_bytes_per_parameter": scalar_bytes,
                   "weights": matmul_active * scalar_bytes,
                   "ersa_state_read_write": 2 * (exact["ersa_state_bytes"] + exact["conv_tail_bytes"]),
                   "window_kv_read": exact["window_kv_bytes_live"] * scalar_bytes // 4 if window["window"] else 0,
                   "ple_rows": ple_p["rows_read_per_token"] * ple_p["head_dim"] * scalar_bytes}
        flops = {"weights": 2 * core_matmul}
        if window["window"]:
            seen = min(length, window["window"])
            flops["window_attention"] = 2 * 2 * window["layers"] * seen * window["heads"] * window["head_dim"]
        if evidence is not None:
            scan = blocks * evidence.per_block * evidence.key_dim
            traffic["r5_descriptor_scan"] = scan * 4
            flops["r5_descriptor_scan"] = 2 * scan
            if wrapped:                           # the live index is two views joined: a full copy every read
                traffic["r5_wrapped_index_copy"] = 2 * scan * 4
            candidates = min(evidence.coarse_top, blocks)
            reencoded = candidates if evidence.fine_keys == "reread" else min(evidence.evidence, blocks)
            span = evidence.span                  # tokens per encoder window: overlap + block
            windows = reencoded + 1               # re-encoded blocks + the open local block, every decoded token
            traffic["r5_token_reread"] = reencoded * span * 4
            traffic["r5_fine_keys_gathered"] = (candidates * evidence.block * evidence.fine_dim * 4
                                                if evidence.fine_keys == "stored" else 0)
            traffic["encoder_rows_gathered"] = windows * span * encoder_row_width * scalar_bytes
            traffic["encoder_core_embedding_rows"] = windows * span * d_model * scalar_bytes
            flops["local_block_encoding"] = 2 * encoder * span
            flops["r5_reencoding"] = reencoded * span * 2 * encoder
            flops["r5_rerank"] = 2 * candidates * evidence.block * evidence.fine_dim
        traffic["total"] = sum(v for k, v in traffic.items() if k != "weights_bytes_per_parameter")
        flops["total"] = sum(flops.values())
        row["projected_per_decoded_token"] = {
            "bytes": traffic, "flops": flops,
            "labels": "PROJECTED compulsory traffic from EXACT counts: active path, each persistent byte once, nothing "
                      "reused across tokens, intermediates excluded (the executed operand traffic is in trace)"}
        rows.append(row)
    return {"configuration": label, "ple": ple, "fine_keys": None if evidence is None else evidence.fine_keys,
            "history_capacity": None if evidence is None else evidence.capacity, "rows": rows}


def command_account(args) -> dict:
    plan = run_r9.frozen()
    lengths = plan["p3"]["context_lengths"]
    out = []
    with torch.device("meta"):
        arm = run_r9.build(plan["arms"]["A"], 0)
    out.append(account_configuration("r9_arm_A_d64", arm, ple=True, lengths=lengths, scalar_bytes=4))
    for fine in ("reread", "stored"):
        for ple in (True, False):
            out.append(account_configuration(f"r8_enc256_{fine}", _evidence_model("enc256", fine), ple=ple,
                                             lengths=lengths, scalar_bytes=4))
        for name in run_r6.SCALES:
            out.append(account_configuration(f"{name}_{fine}", _evidence_model(name, fine), ple=False,
                                             lengths=lengths, scalar_bytes=2))
    return {"phase": PHASE, "command": "account", "provenance": _provenance(), "configurations": out,
            "labels": {"exact": "EXACT", "projected_scratch": "PROJECTED", "projected_per_decoded_token": "PROJECTED"},
            "notes": ["fixed state is EXACT and allocated at capacity (1,048,576 tokens of history), whatever the live "
                      "length; *_live fields give what live history occupies inside that allocation",
                      "the last row of every configuration with an R5 head is the steady state of an endless stream "
                      "(capacity + 4,096 tokens): the ring has wrapped, one edge block is not live, and the live index "
                      "is copied before every scan",
                      "r6_500m / r6_1b weights and window KV at 2 bytes (bfloat16) are a PROJECTION: this "
                      "implementation does not decode in bfloat16 on CPU (the chunked ERSA scan's triangular solve has "
                      "no bfloat16 CPU kernel); descriptors and ERSA state are counted at their float32 CPU widths",
                      "r6_500m / r6_1b carry no PLE; the R7 accounting (results/r7/accounting.json) gives PLE tables at "
                      "that scale",
                      "the 28,552-byte figure of K1 retention describes none of these configurations"]}


# -- trace: executed logical bytes of single decode steps -----------------------------------------------------------

GATHER = {"embedding": "weight", "index": "self", "_unsafe_index": "self", "index_select": "self", "gather": "self",
          "take": "self"}
INDEXED_WRITE = {"index_copy_": "source", "index_put_": "values", "_index_put_impl_": "values", "scatter_": "src",
                 "scatter_add_": "src", "index_add_": "source", "masked_scatter_": "source"}
WRITE_ONLY = {"copy_", "zero_", "fill_", "normal_", "uniform_", "random_"}
MATMUL = {"mm", "addmm", "bmm", "baddbmm"}


def _line_of(function, needle: str) -> int:
    lines, start = inspect.getsourcelines(function)
    for i, line in enumerate(lines):
        if needle in line:
            return start + i
    raise LookupError(f"{needle!r} not in {function.__qualname__}")


def _span_of(function) -> tuple:
    lines, start = inspect.getsourcelines(function)
    return start, start + len(lines) - 1


class _Lines:
    """Source lines that separate stages inside one function, found in the source (robust to code motion)."""

    def __init__(self):
        self.coarse_end = _line_of(EvidenceLM._retrieve_tile, 'clock.lap("coarse_scan")')
        self.setup_end = _line_of(EvidenceLM._retrieve, "ends = (table + 1)")
        self.commit = (_line_of(ErsaLM.forward, "state.cells[index].copy_"),
                       _line_of(ErsaLM.forward, "state.tails[index] = tails[index] if graph"))
        self.rotary = _span_of(Rotary.forward)


def _stage(chain: list, module: str, lines: _Lines) -> str:
    """The pipeline stage of an op from the package call chain (innermost first) and the innermost module."""
    names = [n for _, n, _ in chain]
    if "_reread_encoded" in names:
        return "r5_reencode_candidates"
    if "_chunk_context" in names:
        return "r5_local_block_encode"
    if "_retrieve_tile" in names or "_retrieve" in names:
        inner = names[0]
        if inner == "match":
            return "r5_coarse_scan"
        for f, n, line in chain:
            if n == "_retrieve_tile":
                if line <= lines.coarse_end:
                    return "r5_coarse_bookkeeping"
                break
            if n == "_retrieve":
                if line <= lines.setup_end:
                    return "r5_coarse_bookkeeping"
                break
        if inner in ("_check", "read_fine", "reread", "live"):
            return "r5_validate_and_fetch"
        if inner == "read":
            return "r5_readout"
        if inner in ("fine_keys", "read_keys"):
            return "r5_key_projections"
        return "r5_rerank_and_select"
    if "_extra_mix" in names:
        return "r5_describe_inject"
    if "_commit_context" in names:
        return "r5_commit"
    if any(f == "scroll.py" and n == "commit" for f, n, _ in chain):
        return "window_attention_commit"
    if module == "rotary" or any(f == "model.py" and n == "forward" and lines.rotary[0] <= line <= lines.rotary[1]
                                 for f, n, line in chain):
        return "window_attention"
    if any(f == "ersa_model.py" and n == "forward" and line in lines.commit for f, n, line in chain[:1]):
        return "ersa_state_commit"
    if ".ersa" in module or "chunked_scan" in names or "signals" in names:
        return "ersa_mixer"
    if ".attn" in module:
        return "window_attention"
    if ".sparse" in module:
        return "routed_channel"
    if module.startswith("ple"):
        return "ple"
    if module.endswith((".ffn", ".cgc")) or ".ffn." in module or ".cgc." in module:
        return "dense_channel"
    if "norm" in module:
        return "norms"
    return "core_" + (names[0] if names else "unknown")


class _Labels:
    """storage pointer -> label, for parameters, buffers and the session state's tensors."""

    def __init__(self, model, state):
        self.map = {}
        for name, p in model.named_parameters():
            self.map[p.untyped_storage().data_ptr()] = "param:" + name
        for name, b in model.named_buffers():
            self.map.setdefault(b.untyped_storage().data_ptr(), "buffer:" + name)

        def put(t, label):
            if isinstance(t, torch.Tensor):
                self.map[t.untyped_storage().data_ptr()] = label
        for c in state.cells:
            put(c, "state:ersa_state")
        for t in state.tails:
            put(t, "state:conv_tail")
        if state.window is not None:
            put(state.window.tokens, "state:window_tokens")
            for layer in state.window.layers:
                for v in layer.values():
                    put(v, "state:window_kv")
        put(state.ple_history, "state:ple_history")
        ring = getattr(state, "ring", None)
        if ring is not None:
            put(ring.tokens, "state:ring_tokens")
            put(ring.keys, "state:descriptors")
            put(ring.ids, "state:descriptor_ids")
            put(ring.fine, "state:fine_keys")

    def of(self, t) -> str:
        try:
            return self.map.get(t.untyped_storage().data_ptr(), "scratch")
        except (RuntimeError, NotImplementedError):
            return "scratch"


class _Recorder(TorchDispatchMode):
    """Operand bytes of every aten op of one call, keyed (stage, label, kind, direction); matmul FLOPs by stage."""

    def __init__(self, labels: _Labels, modules: list, lines: _Lines):
        super().__init__()
        self.labels, self.modules, self.lines = labels, modules, lines
        self.bytes = collections.Counter()
        self.flops = collections.Counter()
        self.largest = []

    def _chain(self) -> list:
        chain, frame = [], sys._getframe(2)
        while frame is not None and len(chain) < 12:
            path = frame.f_code.co_filename
            if path.startswith(PACKAGE) and "experiments" not in path:
                chain.append((path[len(PACKAGE):], frame.f_code.co_name, frame.f_lineno))
            frame = frame.f_back
        return chain

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if func.is_view:
            return out
        name, schema = func.overloadpacket.__name__, func._schema
        stage = _stage(self._chain(), self.modules[-1] if self.modules else "", self.lines)
        results = [o for o in (out if isinstance(out, (tuple, list)) else (out,)) if isinstance(o, torch.Tensor)]
        out_bytes = sum(o.numel() * o.element_size() for o in results)
        values = {a.name: (args[i] if i < len(args) else kwargs.get(a.name)) for i, a in enumerate(schema.arguments)}
        total = 0
        for a in schema.arguments:
            v = values[a.name]
            tensors = [v] if isinstance(v, torch.Tensor) else \
                [t for t in v if isinstance(t, torch.Tensor)] if isinstance(v, (list, tuple)) else []
            for t in tensors:
                size = t.numel() * t.element_size()
                written = a.alias_info is not None and a.alias_info.is_write
                if name in GATHER and a.name == GATHER[name]:
                    entries = [("rows", "read", out_bytes)]
                elif written and name in INDEXED_WRITE:
                    source = values.get(INDEXED_WRITE[name])
                    entries = [("indexed", "write", source.numel() * t.element_size()
                                if isinstance(source, torch.Tensor) else 0)]
                elif written and name in WRITE_ONLY:
                    entries = [("dense", "write", size)]
                elif written:
                    entries = [("dense", "read", size), ("dense", "write", size)]
                else:
                    entries = [("dense", "read", size)]
                for kind, direction, n in entries:
                    self.bytes[(stage, self.labels.of(t), kind, direction)] += n
                    total += n
        aliased = [r.alias_info is not None for r in schema.returns]
        for j, o in enumerate(results):
            if not (j < len(aliased) and aliased[j]):
                self.bytes[(stage, "scratch", "dense", "write")] += o.numel() * o.element_size()
                total += o.numel() * o.element_size()
        if name in MATMUL:
            x, y = (args[1], args[2]) if name in ("addmm", "baddbmm") else (args[0], args[1])
            self.flops[stage] += 2 * (x.shape[0] if x.ndim == 3 else 1) * x.shape[-2] * x.shape[-1] * y.shape[-1]
        if total >= MiB:
            self.largest.append((total, str(func), stage, [self.labels.of(t) for t in args
                                                           if isinstance(t, torch.Tensor)][:3]))
        return out


def _classify(stage: str, label: str, kind: str, direction: str) -> tuple:
    """(class, item) of one byte chunk. Classes are disjoint and sum to the step's total."""
    if stage in ("r5_local_block_encode", "r5_reencode_candidates"):
        if label == "state:ring_tokens":
            return "kv_history_reads", "r5_ring_tokens"
        if label.startswith("param:") and kind == "rows":
            return "evidence_reencoding", "core_embedding_rows" if label.startswith("param:embed") else \
                "encoder_delta_rows"
        if label.startswith("param:") or label.startswith("buffer:"):
            return "evidence_reencoding", "encoder_and_key_projection_weights"
        return "evidence_reencoding", "encoder_intermediates"
    if label.startswith("param:"):
        n = label[len("param:"):]
        if kind == "rows":
            return "embedding_lookups", "ple_rows" if n.startswith("ple.") else "input_token_embedding_row"
        if n.startswith("embed."):
            return "weight_traffic", "tied_lm_head"
        if n.startswith("lookup."):
            return "weight_traffic", "r5_head_projections"
        if n.startswith("ple."):
            return "weight_traffic", "ple_fusion"
        if ".sparse.experts." in n:
            return "weight_traffic", "routed_experts_top_k"
        if ".sparse.router." in n:
            return "weight_traffic", "routing_matrices"
        if ".ersa." in n:
            return "weight_traffic", "ersa_projections"
        if ".attn." in n:
            return "weight_traffic", "window_attention_projections"
        if ".ffn." in n or ".cgc" in n:
            return "weight_traffic", "dense_channels"
        return "weight_traffic", "norms"
    if label in ("state:ersa_state", "state:conv_tail"):
        return "ersa_recurrent_state", f"{label[6:]}_{direction}"
    if stage in ("ersa_mixer", "ersa_state_commit"):
        return "ersa_recurrent_state", "state_sized_intermediates"
    if label.startswith("state:"):
        tier = label[6:]
        if direction == "write":
            return "other", "state_commits"
        return "kv_history_reads", {"window_kv": "window_kv", "window_tokens": "window_tokens",
                                    "ring_tokens": "r5_ring_tokens", "descriptors": "r5_descriptor_scan",
                                    "descriptor_ids": "r5_descriptor_ids", "fine_keys": "r8_fine_keys",
                                    "ple_history": "ple_history"}[tier]
    if label.startswith("buffer:rotary") or stage.startswith("window_attention"):
        return "other", "window_attention_recompute"
    if stage == "r5_coarse_scan":
        return "other", "r5_coarse_scan_intermediates"
    if stage == "r5_coarse_bookkeeping":
        return "other", "r5_per_block_bookkeeping_and_wrapped_index_copy"
    if stage.startswith("r5_"):
        return "other", "r5_rerank_readout_inject_intermediates"
    if stage in ("dense_channel", "routed_channel", "norms", "ple") or stage.startswith("core_channel"):
        return "other", "channel_ple_norm_intermediates"
    return "other", "embedding_output_and_logits"


def _module_stack(model):
    names = {m: n for n, m in model.named_modules()}
    stack = []

    def enter(module, inputs):
        stack.append(names.get(module, "?"))      # returns None: inputs untouched

    def leave(module, inputs, output):
        stack.pop()                                # returns None: the output untouched

    hooks = (torch.nn.modules.module.register_module_forward_pre_hook(enter),
             torch.nn.modules.module.register_module_forward_hook(leave))
    return stack, hooks


@torch.no_grad()
def synthetic_fill(model, state, length: int, seed: int) -> None:
    """MEASUREMENT ONLY: put ``state`` where a stream of ``length`` tokens leaves it (counters, live range, window
    position), with random contents. A decode step's operand bytes depend on these shapes and counters, not on
    the values (validated against a real prefill in ``trace``)."""
    g = torch.Generator().manual_seed(seed)
    e, ring = model.evidence, state.ring
    vocab = model.config.vocab_size
    ring.tokens.copy_(torch.randint(0, vocab, ring.tokens.shape, generator=g, dtype=torch.int32))
    ring.position = length
    blocks = length // e.block
    count = min(blocks, ring.slots)
    ring.ids.fill_(-1)
    live = torch.arange(blocks - count, blocks)
    ring.ids[live % ring.slots] = live
    ring.keys.copy_(F.normalize(torch.randn(ring.keys.shape, generator=g), dim=-1).to(ring.keys.dtype))
    if ring.fine is not None:
        ring.fine.copy_(torch.randn(ring.fine.shape, generator=g).to(ring.fine.dtype))
    ring.next_block, ring.count = blocks, count
    for c in state.cells:
        c.copy_(torch.randn(c.shape, generator=g) * 0.01)
    for t in state.tails:
        if t is not None:
            t.copy_(torch.randn(t.shape, generator=g) * 0.1)
    if state.window is not None:
        state.window.position = length
        state.window.tokens.copy_(torch.randint(0, vocab, state.window.tokens.shape, generator=g, dtype=torch.int32))
        for layer in state.window.layers:
            for v in layer.values():
                v.copy_(torch.randn(v.shape, generator=g) * 0.1)
    if state.ple_history is not None:
        state.ple_history.copy_(torch.randint(0, vocab, state.ple_history.shape, generator=g))
    state.position = length


@torch.no_grad()
def _real_fill(model, state, length: int, seed: int, chunk: int = 512) -> None:
    stream = torch.randint(0, 256, (1, length), generator=torch.Generator().manual_seed(seed))
    for at in range(0, length, chunk):
        model(stream[:, at:at + chunk], state, logit_positions=torch.zeros(0, dtype=torch.long))


@torch.no_grad()
def traced_step(model, state, lines: _Lines, token: int = 123) -> _Recorder:
    stack, hooks = _module_stack(model)
    recorder = _Recorder(_Labels(model, state), stack, lines)
    try:
        with recorder:
            model(torch.tensor([[token]]), state, logit_positions=torch.tensor([0]))
    finally:
        for hook in hooks:
            hook.remove()
    return recorder


def _step_record(model, recorder: _Recorder, sizes: dict) -> dict:
    classes = collections.defaultdict(collections.Counter)
    stages = collections.Counter()
    weights, rows, state = collections.Counter(), 0, collections.Counter()
    for (stage, label, kind, direction), n in recorder.bytes.items():
        cls, item = _classify(stage, label, kind, direction)
        classes[cls][item] += n
        stages[stage] += n
        if label.startswith("param:"):
            if kind == "rows":
                rows += n
            else:
                weights[label[6:]] += n
        elif label.startswith("state:"):
            state[f"{label[6:]}_{direction}"] += n
    if "ersa_state_read" in state:           # read three times per layer and step (S k, S q, retain S): once is
        state["ersa_state_read"] //= 3        # compulsory
    unique = {"weights": sum(min(n, sizes[name]) for name, n in weights.items()), "gathered_rows": rows,
              "state_tiers": dict(state)}
    unique["total"] = unique["weights"] + rows + sum(state.values())
    largest = sorted(recorder.largest, key=lambda r: -r[0])[:8]
    return {"total": sum(recorder.bytes.values()), "classes": {c: dict(v) for c, v in classes.items()},
            "stages": dict(stages), "unique_persistent": unique, "matmul_flops_by_stage": dict(recorder.flops),
            "largest_ops": [{"bytes": b, "op": op, "stage": s, "operands": o} for b, op, s, o in largest]}


def _pmu() -> dict:
    """Whether this machine exposes hardware performance counters (it decides whether DRAM traffic is measurable)."""
    devices = sorted(p.name for p in Path("/sys/bus/event_source/devices").glob("*")) \
        if Path("/sys/bus/event_source/devices").exists() else []
    paranoid = Path("/proc/sys/kernel/perf_event_paranoid")
    return {"event_sources": devices, "hardware_pmu": "cpu" in devices or any(d.startswith("uncore") for d in devices),
            "perf_event_paranoid": paranoid.read_text().strip() if paranoid.exists() else None}


@torch.no_grad()
def command_trace(args) -> dict:
    """EXECUTED logical bytes of decode steps of r6_500m (float32 CPU, the R6 scale model with its R5 head) and of
    the R8 timing geometry enc256 (PLE on and off), by data class and stage, at every preregistered length and in
    the wrapped steady state; each after a synthetic fill, which is validated against a real prefill."""
    torch.set_num_threads(args.threads)
    plan = run_r9.frozen()
    lengths = plan["p3"]["context_lengths"] + [plan["p3"]["capacity"] + 4096]
    lines = _Lines()
    record = {"phase": PHASE, "command": "trace", "provenance": _provenance(), "threads": args.threads,
              "pmu": _pmu(), "lengths": lengths, "configurations": [],
              "labels": {"all": "EXECUTED LOGICAL operand bytes per decoded token (a dense op reads every operand "
                                "and writes every result in full; a gather reads the gathered elements; views move "
                                "nothing). Not DRAM transfers: this VM exposes no hardware counters (pmu). "
                                "float32 as executed; CUDA NOT_RUN",
                         "unique_persistent": "per-token unique bytes of persistent tensors touched (each weight "
                                              "once, gathered rows, each state tier's reads once and writes once): "
                                              "the compulsory traffic when nothing survives in cache between tokens"}}
    specs = [("r6_500m", None, (16, 31)), ("enc256", True, (16,)), ("enc256", False, (16,))]
    for name, ple, offsets in specs:
        model = _evidence_model(name, "stored", device="cpu").eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if ple is not None:
            model.ple.enabled = ple
        sizes = {n: p.numel() * p.element_size() for n, p in model.named_parameters()}
        config = {"configuration": name, "ple": ple, "dtype": str(model.embed.weight.dtype),
                  "parameters": sum(p.numel() for p in model.parameters()), "modes": {}}
        for mode in ("stored", "reread"):
            model.evidence = model.evidence.replace(fine_keys=mode)
            state = model.new_state(1)
            steps = []
            for length in lengths:
                for offset in offsets:
                    synthetic_fill(model, state, length + offset, seed=length + offset)
                    recorder = traced_step(model, state, lines)
                    step = {"length": length, "position": length + offset,
                            "completes_block": (length + offset + 1) % model.evidence.block == 0,
                            "counters": dict(state.counters)}
                    step.update(_step_record(model, recorder, sizes))
                    if not (name == "r6_500m" and length >= plan["p3"]["capacity"] and offset == offsets[0]):
                        step.pop("largest_ops")
                    steps.append(step)
                    state.counters.clear()
                    print(f"[trace] {name} ple={ple} {mode} {length}+{offset}: {step['total'] / MiB:.1f} MiB",
                          flush=True)
            validation = {}
            if name == "r6_500m" or ple:
                totals = {}
                for fill in ("real", "synthetic"):
                    fresh = model.new_state(1)
                    (_real_fill if fill == "real" else synthetic_fill)(model, fresh, args.validate, 7)
                    recorder = traced_step(model, fresh, lines)
                    totals[fill] = (sum(recorder.bytes.values()), dict(recorder.bytes), dict(fresh.counters))
                    del fresh
                real, synthetic = totals["real"], totals["synthetic"]
                differing = [k for k in set(real[1]) | set(synthetic[1]) if real[1].get(k, 0) != synthetic[1].get(k, 0)]
                validation = {"length": args.validate, "real_total": real[0], "synthetic_total": synthetic[0],
                              "equal_totals": real[0] == synthetic[0], "counters_equal": real[2] == synthetic[2],
                              "differing_keys": len(differing),
                              "differing_keys_not_routed_expert_weights": sorted(
                                  "|".join(k) for k in differing if not (k[0] == "routed_channel" and
                                                                         ".sparse.experts." in k[1]))}
            config["modes"][mode] = {"steps": steps, "validation": validation,
                                     "state_nbytes": state.nbytes(),
                                     "fixed_bytes_exact": model.context_interface().fixed_bytes(1)}
            del state
            gc.collect()
        record["configurations"].append(config)
        del model
        gc.collect()
    return record


# -- measurements ---------------------------------------------------------------------------------------------------

def command_bandwidth(args) -> dict:
    """MEASURED: large float32 tensor copies (read + write) on this machine; best of several repetitions."""
    torch.set_num_threads(args.threads)
    sizes, out = (256 * MiB, 1 * GiB), []
    for size in sizes:
        n = size // 4
        src, dst = torch.ones(n), torch.empty(n)
        dst.copy_(src)
        best = None
        for _ in range(5):
            started = time.perf_counter()
            dst.copy_(src)
            elapsed = time.perf_counter() - started
            best = elapsed if best is None else min(best, elapsed)
        out.append({"bytes": size, "best_seconds": best, "copy_gb_per_s_read_plus_write": 2 * size / best / 1e9})
        del src, dst
    return {"phase": PHASE, "command": "bandwidth", "provenance": _provenance(), "threads": args.threads,
            "copies": out, "pmu": _pmu(),
            "label": "MEASURED on this execution environment (a VM); not the DDR3 scenario"}


def _proc_memory() -> dict:
    """This process's Rss, Pss and USS (private clean + dirty) in bytes, from /proc/self/smaps_rollup."""
    values = {}
    for line in Path("/proc/self/smaps_rollup").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].rstrip(":") in ("Rss", "Pss", "Private_Clean", "Private_Dirty"):
            values[parts[0].rstrip(":")] = int(parts[1]) * 1024
    values["Uss"] = values.get("Private_Clean", 0) + values.get("Private_Dirty", 0)
    return values


@torch.no_grad()
def command_rss(args) -> dict:
    """MEASURED allocation of one session's state (and of its first decode step) against the EXACT fixed bytes."""
    torch.set_num_threads(args.threads)
    capacity = run_r9.frozen()["p3"]["capacity"]
    out = {}
    for fine in (args.fine_keys,):                # one mode per process: freed pages of another mode would be reused
        gc.collect()
        model = run_r8.timing_model("enc256", capacity=capacity, fine_keys=fine)
        exact = context_interface(model).fixed_bytes(1)
        before = _proc_memory()
        state = model.new_state(1)
        created = _proc_memory()
        tokens = torch.randint(0, 259, (1, args.fill), generator=torch.Generator().manual_seed(9))
        for at in range(0, args.fill, 512):                      # live history before the measured decode step
            model(tokens[:, at:at + 512], state)
        filled = _proc_memory()
        peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        model(tokens[:, -1:], state)
        stepped = _proc_memory()
        out[fine] = {"exact_fixed_bytes": exact,
                     "measured_state_rss_bytes": created["Rss"] - before["Rss"],
                     "measured_state_uss_bytes": created["Uss"] - before["Uss"],
                     "state_rss_over_exact": (created["Rss"] - before["Rss"]) / exact["total"],
                     "live_history_tokens": args.fill,
                     "rss_retained_after_filling_bytes": filled["Rss"] - created["Rss"],
                     "rss_retained_after_one_decode_step_bytes": stepped["Rss"] - filled["Rss"],
                     "process_peak_rss_before_step_bytes": peak_before,
                     "process_peak_rss_after_step_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                     "process_rss_after_bytes": stepped["Rss"], "process_uss_after_bytes": stepped["Uss"]}
        del state, model
    return {"phase": PHASE, "command": "rss", "provenance": _provenance(), "threads": args.threads,
            "capacity": capacity, "states": out,
            "label": "MEASURED (Linux /proc/self/smaps_rollup) in a process holding one model; allocator effects "
                     "included, which is the point"}


def command_roofline(args) -> dict:
    """PROJECTED ceilings per decoded token. For every configuration, length, weight precision, batch and
    cache-resident fraction: bytes and FLOPs per token, the memory-bound ceiling at every (bandwidth, achieved
    fraction) and the compute-bound ceiling at every compute rate. The ceiling is the smaller of the two."""
    plan = run_r9.frozen()["p3"]
    account = json.loads(Path(args.account).read_text(encoding="utf-8"))
    ddr3, gpu = plan["ddr3_scenario"], plan["gpu_scenario"]
    rows = []
    for config in account["configurations"]:
        if config["configuration"].startswith("r9_arm"):
            continue
        for row in config["rows"]:
            per = row["projected_per_decoded_token"]
            base = per["bytes"]
            matrix = base["weights"] / base["weights_bytes_per_parameter"]       # matrix weights per token
            other = base["total"] - base["weights"]
            flops = per["flops"]["total"]
            for precision, width in ddr3["weight_bytes_per_parameter"].items():
                for batch in ddr3["batch"]:
                    for cached in ddr3["cache_resident_weight_fraction"]:
                        traffic = matrix * width * (1 - cached) / batch + other   # weights amortized over a batch
                        rows.append({"scenario": "ddr3", "configuration": config["configuration"],
                                     "ple": config["ple"], "length": row["length"],
                                     "steady_state_wrapped": row["steady_state_wrapped"], "precision": precision,
                                     "batch": batch, "cache_resident": cached, "bytes_per_token": round(traffic),
                                     "flops_per_token": flops,
                                     "memory_bound_tokens_per_s": {
                                         f"{bw}GBps_x{a}": round(bw * 1e9 * a / traffic, 2)
                                         for bw in ddr3["bandwidth_gb_per_s"] for a in ddr3["achieved_fraction"]},
                                     "compute_bound_tokens_per_s": {
                                         f"{c}GFLOPs": round(c * 1e9 / flops, 2) for c in ddr3["compute_gflop_per_s"]}})
            traffic = matrix * 2 + other                                          # bfloat16 weights, batch 1
            rows.append({"scenario": "gpu", "configuration": config["configuration"], "ple": config["ple"],
                         "length": row["length"], "steady_state_wrapped": row["steady_state_wrapped"],
                         "precision": "bfloat16", "batch": 1, "cache_resident": 0.0,
                         "bytes_per_token": round(traffic), "flops_per_token": flops,
                         "memory_bound_tokens_per_s": {f"{bw}GBps_x0.8": round(bw * 1e9 * 0.8 / traffic, 2)
                                                       for bw in gpu["bandwidth_gb_per_s"]},
                         "compute_bound_tokens_per_s": {f"{tf}TFLOPs": round(tf * 1e12 / flops, 2)
                                                        for tf in gpu["compute_tflop_per_s"]}})
    return {"phase": PHASE, "command": "roofline", "provenance": _provenance(), "rows": rows,
            "account_sha256": hashlib.sha256(Path(args.account).read_bytes()).hexdigest(),
            "assumptions": {"ddr3": ddr3, "gpu": gpu}, "target_tokens_per_second": plan["target_tokens_per_second"],
            "labels": {"all": "PROJECTED ceilings (upper bounds on tokens/s) from EXACT counts under the frozen ASSUMED "
                              "platforms; the ceiling is min(memory bound, compute bound); bytes are the compulsory "
                              "traffic (account), so these are optimistic; CUDA NOT_RUN; not measurements"}}


def command_measured(args) -> dict:
    """Summary of R8-runner admission records (MEASURED here): decode p50/p95 by configuration and length."""
    out = {}
    for path in args.records.split(","):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = []
        for r in record["rows"]:
            configs = {}
            for name, c in r["configs"].items():
                natural = c["decode_natural"]["all"]
                configs[name] = {"decode_ms_p50": natural["step_ms"]["p50"], "decode_ms_p95": natural["step_ms"]["p95"],
                                 "read_ms_p50": natural.get("minus_off_ms", {}).get("p50"),
                                 "stages_ms_mean": natural.get("stage_ms_mean"),
                                 "state_bytes": c.get("state_bytes", {}).get("total"),
                                 "prefill_last_ms_p50": c["prefill_last"][0]["step_ms"]["p50"]
                                 if c.get("prefill_last") else None}
            rows.append({"length": r["occupancy"], "ple": r["ple"], "configs": configs,
                         "peak_rss_mib": r.get("peak_rss_mib")})
        out[Path(path).name] = {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                "source_revision": record.get("environment", {}).get("source_revision"),
                                "threads": record.get("threads"), "rows": rows}
    return {"phase": PHASE, "command": "measured", "provenance": _provenance(), "records": out,
            "label": "MEASURED on this CPU (4 threads unless the record says otherwise), untrained timing models"}


def _provenance() -> dict:
    out = run_r9.provenance(run_r9.frozen())
    out["p3_runner_sha256"] = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    return out


COMMANDS = {"account": command_account, "trace": command_trace, "bandwidth": command_bandwidth, "rss": command_rss,
            "roofline": command_roofline, "measured": command_measured}


def _write(path: str, record: dict) -> None:
    Path(path).write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--account", help="roofline: the account record")
    parser.add_argument("--records", help="measured: comma-separated R8 admission records")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fill", type=int, default=65536, help="rss: live history before the measured decode step")
    parser.add_argument("--fine-keys", default="reread", choices=("reread", "stored"), help="rss: the mode measured")
    parser.add_argument("--validate", type=int, default=2064, help="trace: the length of the real-prefill check")
    parser.add_argument("--out", help="the JSON record to write (nothing is written without it)")
    args = parser.parse_args(argv)
    record = COMMANDS[args.command](args)
    if args.out:
        _write(args.out, record)
    else:
        json.dump({k: v for k, v in record.items() if k in ("phase", "command")}, sys.stdout)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
