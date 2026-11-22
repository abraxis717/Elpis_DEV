"""ELPIS_TORCH_R13: the Geodome cognitive substrate, admitted and qualified (P0-P4). RESEARCH_ONLY.

Mechanical and gradient qualification only. No competence experiment runs here (P5 is a protocol for review).

Commands (each writes one JSON record with ``--out``):
    identity    P1, gate 1: the vendored sources against their pins and the upstream manifest, the frequency-120
                identity invariants, Hexaflag and arc round trips, non-edge refusal, logical prefixes that are not
                connected regions, import isolation, the Elpis base sources unchanged
    audit       P0: the EvidenceLM seam measured (return_pre, the captured channel rows, the tail recomputation and
                its batch variance, diagnostics, chunk limit, refusals leave the state unchanged, evidence modes)
    parity      gates 3, 4, 7: the tensor executor against Geodome's own scheduler (float64, every status), dense
                against sparse residency, the synchronous schedule against the scheduler on the class where they
                coincide, a divergence witness outside it, hop causality
    gradients   gates 5, 6, 7: gradients through a multi-hop path on the real bridge, the zero-gate property, the
                hop bound, float64 gradcheck (executor; and end to end through the actual final channel), the
                bounded graph, no host extraction on the cognitive path
    bridge      gates 2, 4, 8-12 on actual EvidenceLM variants (evidence off/learned, fine keys reread/stored, PLE
                off/on, nonzero origin, window attention; every stream wraps the ring and evicts): disabled-path
                parity, zero-gate parity, the context clock, admission and its refusals, atomicity, isolation,
                immutability, reset
    resources   parameters, state, topology and plan bytes, gradient scratch, visits, active vertices, latency, RSS
    verdict     the R13 dispositions, computed from the six records above
    manifest    every R13 record with its digest
"""
from __future__ import annotations

import argparse
import ast
import copy
import ctypes
import hashlib
import json
import math
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.func import functional_call
from torch.overrides import TorchFunctionMode

from .. import evidence_tasks as et
from ..config import ModelConfig
from ..context_runtime import ModelRevisionError, SessionClosedError
from ..ersa_model import ErsaConfig
from ..evidence_model import EvidenceConfig, EvidenceLM
from ..evidence_ring import BlockAddress, StaleAddressError
from ..lastblock import capture_pre
from ..model import parameter_digest
from ..ple import PLEConfig
from ..scroll import ChunkError
from ..sparse_channel import SparseConfig
from ..geodome_cognitive import source as gsource
from ..geodome_cognitive.activation import run_changed_destination, run_synchronous
from ..geodome_cognitive.binding import CognitiveError, CognitiveSession
from ..geodome_cognitive.kernels import CognitiveConfig, CognitiveKernels
from ..geodome_cognitive.plan import ExecutionPlan, PlanError, compile_plan
from ..geodome_cognitive.reference import scheduler_reference
from ..geodome_cognitive.state import TensorWorkingState, WorkingStateError
from ..geodome_cognitive.topology import substrate

PHASE = "ELPIS_TORCH_R13_GEODOME_SUBSTRATE"
ELPIS_BASE = "d73cdde3a5050de50cce098472ee966eec103000"        # main after R12 (the base this phase builds on)
REPO = Path(__file__).resolve().parents[3]
TORCH = REPO / "research" / "elpis_torch"
RESULTS = TORCH / "results" / "r13"
COGNITIVE = ("source", "topology", "plan", "state", "kernels", "activation", "binding", "reference")
SOURCES = {"runner": Path(__file__).resolve(),
           **{name: TORCH / "geodome_cognitive" / f"{name}.py" for name in COGNITIVE},
           **{name: TORCH / f"{name}.py" for name in ("evidence_model", "evidence_ring", "ersa_model", "lastblock",
                                                     "sparse_channel", "context_runtime", "model", "config", "scroll",
                                                     "ple")}}
PARITY = 1e-12                                     # float64 agreement where reduction shapes differ
BLOCK, CAPACITY = 16, 256
EXPERTS = SparseConfig(experts=4, top_k=2, layers=(0, 1), shared=True)
PLE = PLEConfig(orders=(2, 3), heads=2, rows=67, head_dim=4, layer=1)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args) -> str:
    run = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return run.stdout.strip() if run.returncode == 0 else ""


def provenance() -> dict:
    dirty = bool(_git("status", "--porcelain", "--", ".", ":(exclude)research/elpis_torch/results"))
    return {**{f"{k}_sha256": sha(v) for k, v in SOURCES.items()},
            "geodome_files_sha256": gsource.verify_sources(), "geodome_commit": gsource.GEODOME_COMMIT,
            "elpis_base": ELPIS_BASE, "git_revision": _git("rev-parse", "HEAD") + ("+dirty" if dirty else ""),
            "torch": torch.__version__, "python": platform.python_version(), "threads": torch.get_num_threads()}


# -- fixtures (shared with the tests) -------------------------------------------------------------------------------

def build_backbone(*, fine_keys: str = "reread", ple: bool = False, window: bool = False, dtype: str = "float64",
                   seed: int = 5) -> EvidenceLM:
    """A frozen two-block EvidenceLM: routed experts (shared + 4, top 2) in both blocks, a 256-token ring of 16-token
    blocks (every stream below wraps it), optional PLE before the last block and optional window attention."""
    chunk = 32 if window else 70
    widths = ModelConfig(vocab_size=et.VOCAB, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=64,
                         channel="cgc", cgc_bottleneck=8, cgc_channels=24, cgc_units=4, seed=seed, dtype=dtype)
    extra = dict(attention="window", attn_heads=2, attn_kv_heads=1, attn_window=40) if window else {}
    ersa = ErsaConfig(widths, ersa_heads=2, key_dim=8, value_dim=8, ersa_kernel="chunked", scan_block=16,
                      conv_kernel=4, chunk=chunk, sparse=EXPERTS, ple=PLE if ple else None, **extra)
    evidence = EvidenceConfig(block=BLOCK, capacity=CAPACITY, per_block=2, key_dim=16, coarse_top=6, evidence=3,
                              fine_top=4, overlap=4, encoder_dim=32, fine_dim=8, read_heads=2, read_head_dim=8,
                              query_tile=7, fine_keys=fine_keys)
    model = EvidenceLM(ersa, evidence)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def stream_tokens(length: int, seed: int = 3) -> torch.Tensor:
    return torch.randint(0, et.VOCAB, (1, length), generator=torch.Generator().manual_seed(seed))


def kernels_for(model, *, width: int = 16, hidden: int = 32, capacity: int = 64, gate: float = 0.0,
                seed: int = 13, residency: str = "sparse") -> CognitiveKernels:
    config = CognitiveConfig(width=width, hidden=hidden, capacity=capacity, gate_init=gate, seed=seed,
                             residency=residency)
    return CognitiveKernels(config, model.config.d_model, model.encoder.width, dtype=model.embed.weight.dtype)


def tensor_bytes(t: torch.Tensor) -> bytes:
    """A tensor's dtype, shape and raw bytes (no NumPy: CI's PyTorch environment has none, as R10 documents)."""
    data = t.detach().contiguous().cpu()
    head = f"{data.dtype}|{tuple(data.shape)}|".encode()
    return head + (ctypes.string_at(data.data_ptr(), data.numel() * data.element_size()) if data.numel() else b"")


def state_digest(state) -> str:
    """SHA-256 of every semantic tensor of an EvidenceState and its clocks: ERSA cells and tails, the window cache,
    the PLE history, ring tokens, index keys, ids and stored fine keys, positions. Identity (session, epoch) and
    accounting counters are excluded: two sessions fed the same stream have equal digests."""
    h = hashlib.sha256()
    tensors = [*state.cells, *[t for t in state.tails if t is not None], state.ring.tokens, state.ring.keys,
               state.ring.ids]
    if state.ring.fine is not None:
        tensors.append(state.ring.fine)
    if state.ple_history is not None:
        tensors.append(state.ple_history)
    if state.window is not None:
        tensors.append(state.window.tokens)
        tensors.extend(t for layer in state.window.layers for _, t in sorted(layer.items()))
    for t in tensors:
        h.update(tensor_bytes(t))
    clocks = (state.position, state.ring.position, state.ring.next_block, state.ring.count,
              None if state.window is None else state.window.position)
    h.update(repr(clocks).encode())
    return h.hexdigest()


def counters(state) -> dict:
    return {**{f"ring_{k}": v for k, v in state.ring.stats.items()}, **dict(state.counters)}


def chain(sub, length: int, start: int = 0) -> tuple:
    """An induced physical path: each vertex is adjacent to its predecessor and to no earlier member."""
    walk, seen = [start], {start}
    while len(walk) < length:
        options = [v for v in sub.neighbors(walk[-1])
                   if v not in seen and not any(v in sub.neighbors(w) for w in walk[:-1])]
        if not options:
            raise RuntimeError("the walk is stuck")
        walk.append(options[0])
        seen.add(options[0])
    return tuple(walk)


def ball(sub, center: int, size: int) -> tuple:
    """The first ``size`` vertices of a breadth-first search from ``center`` (neighbours in index order)."""
    order, seen = [center], {center}
    at = 0
    while len(order) < size:
        for v in sub.neighbors(order[at]):
            if v not in seen and len(order) < size:
                seen.add(v)
                order.append(v)
        at += 1
    return tuple(sorted(order))


def seeded_rows(n: int, width: int, seed: int, dtype=torch.float64) -> torch.Tensor:
    return (torch.randn(n, width, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) * 0.5).to(dtype)


def observe_stream(session, tokens, chunks: int | None = None) -> list:
    """Feed ``tokens`` through ``session.observe`` in chunks of the backbone's chunk length; returns the logits."""
    chunk = session.model.ersa.chunk
    stops = range(0, tokens.shape[1] - tokens.shape[1] % chunk, chunk)
    return [session.observe(tokens[:, at:at + chunk]) for at in list(stops)[:chunks]]


def query_block(session) -> int:
    return session._observation.position // session.model.evidence.block


def _refuses(fn, kinds, code: str | None = None) -> dict:
    try:
        fn()
    except kinds as error:
        got = getattr(error, "code", None) or (str(error.args[0]) if error.args else None)
        out = {"refused": True, "error": type(error).__name__, "code": got}
        if code is not None:
            out["ok_code"] = got == code
        return out
    except Exception as error:                     # the wrong refusal is a failure, recorded
        return {"refused": False, "error": f"unexpected {type(error).__name__}: {error}"}
    return {"refused": False, "error": None}


def _gate(checks) -> str:
    """PASS iff every boolean under a key named refused / pass or starting with equal / exact / ok / within is true."""
    flat = []

    def walk(x):
        if isinstance(x, dict):
            for key, value in x.items():
                if isinstance(value, bool) and (key in ("refused", "pass") or key.startswith(("equal", "exact", "ok",
                                                                                            "within"))):
                    flat.append(value)
                elif isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(x, list):
            for value in x:
                walk(value)
    walk(checks)
    return "PASS" if flat and all(flat) else "FAIL"


# -- P1: identity -----------------------------------------------------------------------------------------------------

def command_identity(args) -> dict:
    record = gsource.source_record()
    sub = substrate()
    hxf = sys.modules["Hexaflag"]
    pentagon = next(v for v in range(sub.vertices) if len(sub.neighbors(v)) == 5)
    sample = (0, 1, pentagon, 71999, 144001)
    roundtrip = {str(v): {"address": [list(d) for d in sub.address(v)[1:]],
                          "ok_address": sub.resolve(sub.address(v)) == v,
                          "ok_coordinate": sub.resolve_coordinate(sub.coordinate(v)) == v} for v in sample}
    arcs = {}
    for v in sample:
        for w in sub.neighbors(v):
            a = sub.arc(v, w)
            arcs[f"{v}->{w}"] = {"arc": a, "ok_endpoints": sub.endpoints(a) == (v, w),
                                 "ok_reverse": sub.endpoints(a ^ 1) == (w, v),
                                 "ok_port": sub.neighbors(v)[sub.port(v, w)] == w}
    two_hop = next(w for w in sub.neighbors(sub.neighbors(0)[0]) if w != 0 and w not in sub.neighbors(0))
    groups = []                         # depth-3 logical prefixes (about 420 vertices each, one page)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                digits = (hxf.DIGITS[i], hxf.DIGITS[j], hxf.DIGITS[k])
                count = sub.prefix_count((3, *digits))
                if 0 < count <= 1024:
                    members = sub.prefix_page((3, *digits), 0, 1024)
                    groups.append({"prefix": [list(d) for d in digits], "vertices": count,
                                   "components": len(sub.components(members))})
    loaded = {name: getattr(sys.modules.get(name), "__file__", None) for name in gsource.MODULES}
    inside = {name: path is not None and Path(path).resolve().is_relative_to(gsource.VENDORED)
              for name, path in loaded.items() if name != "GEO144K"}
    changed = _git("diff", "--name-only", "--diff-filter=MDR", ELPIS_BASE, "--",
                   ":(glob)research/elpis_torch/**/*.py").split()
    records = _git("diff", "--name-only", "--diff-filter=MDR", ELPIS_BASE, "--", "research/elpis_torch/results").split()
    checks = {"sources": {"ok_verified": bool(record["files"])},
              "identity": {"equal_invariants": all(sub.identity_record()[k] == gsource.IDENTITY[k]
                                                   for k in ("vertices", "edges", "arcs", "capacity", "depth",
                                                             "mapping_sha256", "edge_sha256",
                                                             "index_payload_bytes"))},
              "roundtrips": roundtrip, "arcs": arcs,
              "nonedge": _refuses(lambda: sub.arc(0, two_hop), KeyError),
              "out_of_range_arc": _refuses(lambda: sub.endpoints(sub.arcs), ValueError),
              "modules": {"ok_inside_vendored": all(inside.values()) and len(inside) == len(gsource.MODULES) - 1},
              "built_once": {"ok_singleton": substrate() is sub},
              "elpis_base_sources": {"modified_or_removed": changed, "ok_additive_only": changed == [],
                                     "prior_records_modified_or_removed": records,
                                     "ok_prior_records_untouched": records == []}}
    disconnected = [g for g in groups if g["components"] > 1]
    return {"phase": PHASE, "command": "identity", "source": record, "identity": sub.identity_record(),
            "pentagon_sample": pentagon, "prefix_groups": groups,
            "logical_prefixes_disconnected": len(disconnected), "logical_prefixes_examined": len(groups),
            "modules_loaded_from": {k: str(Path(v).resolve().relative_to(REPO)) if v else None
                                    for k, v in loaded.items()},
            "checks": checks, "gate_1": _gate(checks), "provenance": provenance()}


# -- P0: the seam ----------------------------------------------------------------------------------------------------

def tail_logits(model, rows: torch.Tensor) -> torch.Tensor:
    """The backbone's own tail on pre-channel rows (what the bridge's ``final_logits`` computes), last position."""
    block = model.blocks[-1]
    x = rows + block.channel(block.channel_norm(rows))
    x = model.final_norm(x[:, -1:])
    return (torch.nn.functional.linear(x, model.embed.weight) if model.head is None else model.head(x))[:, 0]


@torch.no_grad()
def command_audit(args) -> dict:
    model = build_backbone()
    tokens = stream_tokens(420)
    last = torch.tensor([69])
    out = {}
    state = model.new_state(1)
    for at in range(0, 280, 70):
        model(tokens[:, at:at + 70], state, logit_positions=last)
    before, position = state_digest(state), state.position
    full_state = model.new_state(1)
    for at in range(0, 280, 70):
        model(tokens[:, at:at + 70], full_state, logit_positions=last)
    with capture_pre(model) as captured:
        logits, pre = model(tokens[:, 280:350], state, return_pre=True, logit_positions=last)
    rows = captured.pre
    full = model(tokens[:, 280:350], full_state)
    out["return_pre"] = {"logits_shape": list(logits.shape), "pre_shape": list(pre.shape),
                         "captured_rows_shape": list(rows.shape), "equal_final_row": torch.equal(rows[:, -1:], pre),
                         "ok_captured_every_row": rows.shape[1] == 70,
                         "ok_position_advanced_by_chunk": state.position == position + 70,
                         "ok_state_changed": state_digest(state) != before}
    tail = tail_logits(model, rows)
    single = tail_logits(model, rows[:, -1:])
    out["tail"] = {"equal_T_row_recomputation": torch.equal(tail, logits[:, 0]),
                   "single_row_max_abs_diff": float((single - logits[:, 0]).abs().max()),
                   "full_forward_last_position_max_abs_diff": float((full[:, -1] - logits[:, 0]).abs().max()),
                   "note": "the routed channel's grouped dispatch is batch-variant (R6): the bridge recomputes the "
                           "final channel over the same T rows as the forward, never one row"}
    probe = model.new_state(1)
    for at in range(0, 350, 70):
        result = model(tokens[:, at:at + 70], probe, return_pre=True, logit_positions=last, diagnostics=True)
    out["diagnostics"] = {"ok_triple": len(result) == 3, "keys": sorted(result[2])}
    digest = state_digest(state)
    out["chunk_limit"] = _refuses(lambda: model(tokens[:, :71], state), ChunkError)
    out["chunk_limit"]["equal_state_unchanged"] = state_digest(state) == digest
    bad = tokens[:, :8].clone()
    bad[0, 3] = et.VOCAB
    out["invalid_tokens"] = _refuses(lambda: model(bad, state), ValueError)
    out["invalid_tokens"]["equal_state_unchanged"] = state_digest(state) == digest
    out["oracle_without_labels"] = _refuses(lambda: model(tokens[:, :8], state, logit_positions=torch.tensor([7]),
                                                          mode="oracle"), ValueError)
    out["oracle_without_labels"]["equal_state_unchanged"] = state_digest(state) == digest
    modes, states = {}, {}
    for mode in ("off", "learned"):
        s = model.new_state(1)
        with capture_pre(model) as cap:
            for at in range(0, 350, 70):
                model(tokens[:, at:at + 70], s, logit_positions=last, mode=mode)
        modes[mode], states[mode] = cap.pre, state_digest(s)
    out["evidence_modes"] = {"equal_rows_before_final": torch.equal(modes["off"][:, :-1], modes["learned"][:, :-1]),
                             "equal_state": states["off"] == states["learned"],
                             "final_row_read_max_abs": float((modes["off"][:, -1] - modes["learned"][:, -1]).abs().max()),
                             "ok_read_present": bool((modes["off"][:, -1] != modes["learned"][:, -1]).any())}
    split = model.new_state(1)
    for at in range(0, 280, 70):
        model(tokens[:, at:at + 70], split, logit_positions=last)
    model(tokens[:, 280:315], split, logit_positions=torch.tensor([34]))
    halves = model(tokens[:, 315:350], split, logit_positions=torch.tensor([34]))
    out["chunk_partition"] = {"final_logits_max_abs_diff_70_vs_35_35": float((halves[:, 0] - logits[:, 0]).abs().max()),
                              "note": "informational: the routed channel is batch-variant across partitions (R6)"}
    out["placement"] = {"read_layer": model.read_layer, "ok_last_block": model.read_layer == len(model.blocks) - 1}
    return {"phase": PHASE, "command": "audit", "checks": out, "gate_seam": _gate(out), "provenance": provenance()}


# -- gates 3, 4, 7: parity -------------------------------------------------------------------------------------------

def reference_kernels(width: int = 8, hidden: int = 16, seed: int = 13) -> CognitiveKernels:
    config = CognitiveConfig(width=width, hidden=hidden, capacity=256, max_rounds=64, max_arcs=4096, seed=seed)
    return CognitiveKernels(config, d_model=32, encoder_width=32, dtype=torch.float64)


def tensor_run(sub, kernels, vertices, arcs, init, *, schedule, rounds, start=(), max_active=None, max_visits=None,
               residency="sparse"):
    working = TensorWorkingState(sub, capacity=max(len(vertices), 1), width=init.shape[1], dtype=torch.float64,
                                 residency=residency)
    working.acquire(vertices)
    plan = ExecutionPlan(vertices=tuple(vertices), arcs=tuple(arcs), rounds=rounds, schedule=schedule,
                         start=tuple(start), readout=(vertices[0],), max_active=max_active, max_visits=max_visits)
    compiled = compile_plan(plan, working, max_rounds=256, max_arcs=24576, max_visits=6291456, max_evidence=8)
    with torch.no_grad():
        run = run_changed_destination(init, compiled, kernels) if schedule == "changed_destination" \
            else run_synchronous(init, compiled, kernels)
    return run, compiled


@torch.no_grad()
def command_parity(args) -> dict:
    sub = substrate()
    kernels = reference_kernels()
    width = kernels.config.width
    pentagon = next(v for v in range(sub.vertices) if len(sub.neighbors(v)) == 5)
    region, wide, five = ball(sub, 0, 40), ball(sub, 5000, 60), ball(sub, pentagon, 30)
    path = chain(sub, 8, start=777)
    forward = tuple(sub.arc(path[i], path[i + 1]) for i in range(len(path) - 1))
    workloads = {
        "spread_single_source": dict(vertices=region, arcs=sub.physical_arcs(region), start=(0,), rounds=12),
        "multi_source": dict(vertices=wide, arcs=sub.physical_arcs(wide), start=wide[:3], rounds=10),
        "degree_five_centre": dict(vertices=five, arcs=sub.physical_arcs(five), start=(pentagon,), rounds=8),
        "chain_forward_arcs": dict(vertices=path, arcs=forward, start=(path[0],), rounds=20),
        "quiescent_no_outgoing": dict(vertices=path[:2], arcs=forward[:1], start=(path[1],), rounds=4),
        "active_budget": dict(vertices=region, arcs=sub.physical_arcs(region), start=region[:8], rounds=5,
                              max_active=12),
        "arc_budget": dict(vertices=region, arcs=sub.physical_arcs(region), start=(0,), rounds=8, max_visits=40),
        "round_budget": dict(vertices=region, arcs=sub.physical_arcs(region), start=(0,), rounds=2),
    }
    results = {}
    for index, (name, w) in enumerate(workloads.items()):
        init = seeded_rows(len(w["vertices"]), width, seed=index + 1)
        active = w.get("max_active") or min(1024, len(w["vertices"]))
        visits = w.get("max_visits") or w["rounds"] * len(w["arcs"])
        entry, tensors = {"enrolled": len(w["vertices"]), "admitted_arcs": len(w["arcs"])}, {}
        for residency in ("sparse", "dense"):
            mine, _ = tensor_run(sub, kernels, w["vertices"], w["arcs"], init, schedule="changed_destination",
                                 rounds=w["rounds"], start=w["start"], max_active=active, max_visits=visits,
                                 residency=residency)
            ref = scheduler_reference(sub, kernels, w["vertices"], w["arcs"], init, w["start"], rounds=w["rounds"],
                                      max_active=active, max_visits=visits, mode=residency)
            records = [(r.status, r.sources, r.arcs, r.destinations, r.changed, r.touched) for r in mine.records]
            diff = float((mine.rows - ref["rows"]).abs().max())
            entry[residency] = {"status": mine.status, "rounds": mine.rounds, "visits": mine.visits,
                                "equal_status": mine.status == ref["status"], "equal_rounds": mine.rounds == ref["rounds"],
                                "equal_visits": mine.visits == ref["visits"], "equal_records": records == ref["records"],
                                "rows_max_abs_diff": diff, "within_tolerance": diff <= PARITY}
            tensors[residency] = mine.rows
        entry["equal_dense_sparse_tensor_rows"] = torch.equal(tensors["sparse"], tensors["dense"])
        results[name] = entry
    statuses = sorted({results[n]["sparse"]["status"] for n in results})
    # synchronous, on the class where it coincides with Geodome's scheduler: every enrolled vertex receives, the
    # frontier starts full, every destination changes every round (read from the reference's own round records)
    matched = ball(sub, 777, 50)
    arcs = sub.physical_arcs(matched)
    init = seeded_rows(len(matched), width, seed=99)
    sync, compiled = tensor_run(sub, kernels, matched, arcs, init, schedule="synchronous", rounds=6)
    ref = scheduler_reference(sub, kernels, matched, arcs, init, matched, rounds=6, max_active=len(matched),
                              max_visits=6 * len(arcs))
    diff = float((sync.rows - ref["rows"]).abs().max())
    synchronous = {"ok_in_matched_class": bool(compiled.incoming.all()) and len(ref["records"]) >= 6
                   and all(r[4] == r[3] == len(matched) for r in ref["records"][:6]),
                   "rows_max_abs_diff": diff, "within_tolerance": diff <= PARITY,
                   "equal_visits": sync.visits == sum(r[2] for r in ref["records"][:6])}
    # outside that class (a one-directional chain): the schedules differ, by design; recorded, never claimed equal
    init = seeded_rows(len(path), width, seed=7)
    sync, _ = tensor_run(sub, kernels, path, forward, init, schedule="synchronous", rounds=6)
    event, _ = tensor_run(sub, kernels, path, forward, init, schedule="changed_destination", rounds=6,
                          start=(path[0],), max_active=len(path), max_visits=6 * len(forward))
    witness = {"workload": "chain of 8, forward arcs only, frontier = the first vertex, 6 rounds",
               "max_abs_difference": float((sync.rows - event.rows).abs().max()), "claimed_equal": False}
    # hop causality: perturb the first vertex; after r synchronous rounds only vertices within r hops may change
    hops = {}
    base = seeded_rows(len(path), width, seed=5)
    bumped = base.clone()
    bumped[0] += 0.25
    for rounds in (1, 3, 6):
        a, _ = tensor_run(sub, kernels, path, forward, base, schedule="synchronous", rounds=rounds)
        b, _ = tensor_run(sub, kernels, path, forward, bumped, schedule="synchronous", rounds=rounds)
        moved = [i for i in range(1, len(path)) if not torch.equal(a.rows[i], b.rows[i])]
        hops[str(rounds)] = {"moved_hops": moved, "ok_within_hops": all(i <= rounds for i in moved),
                             "ok_reaches_hop": rounds in moved}
    expected = {"QUIESCENT", "ACTIVE_BUDGET", "ARC_BUDGET", "ROUND_BUDGET"}
    gate3 = {"w": {n: {"s": v["sparse"], "d": v["dense"]} for n, v in results.items()}, "sync": synchronous,
             "statuses": {"ok_every_status": expected <= set(statuses)}}
    gate4 = {n: {"equal_dense_sparse_tensor_rows": v["equal_dense_sparse_tensor_rows"],
                 "equal_dense_records": v["dense"]["equal_records"]} for n, v in results.items()}
    return {"phase": PHASE, "command": "parity", "statuses_exercised": statuses, "divergence_witness": witness,
            "workloads": results, "synchronous_matched": synchronous, "hop_causality": hops,
            "gate_3": _gate(gate3), "gate_4_executor": _gate(gate4), "gate_7_executor": _gate(hops),
            "tolerance": PARITY, "pentagon": pentagon, "provenance": provenance()}


# -- gates 5, 6, 7: gradients ----------------------------------------------------------------------------------------

class HostExtraction(TorchFunctionMode):
    """Records every tensor-to-host conversion requested inside the context (item, tolist, numpy, bool/int/float)."""
    NAMES = {"item", "tolist", "numpy", "__bool__", "__float__", "__int__", "__index__", "__array__"}

    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", "")
        if name in self.NAMES:
            self.calls.append(name)
        return func(*args, **(kwargs or {}))


class _Substituted(torch.nn.Module):
    """``fn`` evaluated with some of ``kernels``' parameters substituted by ``torch.func.functional_call``."""

    def __init__(self, kernels, fn):
        super().__init__()
        self.kernels = kernels
        self.fn = fn

    def forward(self, *inputs):
        return self.fn(*inputs)


def substituted(kernels, names, fn):
    wrapper = _Substituted(kernels, fn)

    def call(*args):
        split = len(args) - len(names)
        return functional_call(wrapper, {f"kernels.{n}": w for n, w in zip(names, args[split:])}, args[:split])
    return call


def graph_size(tensor) -> int:
    """Autograd nodes reachable from ``tensor`` (0: no graph)."""
    seen, todo = set(), [tensor.grad_fn]
    while todo:
        node = todo.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        todo.extend(n for n, _ in node.next_functions)
    return len(seen)


def _prepared(model, kernels, tokens, path, *, chunks: int = 8, evidence: int = 1):
    session = CognitiveSession(model, kernels)
    observe_stream(session, tokens, chunks)
    qb = query_block(session)
    session.enrol(path)
    session.admit_evidence([session.state.ring.address(qb - 1 - k) for k in range(evidence)])
    return session


def command_gradients(args) -> dict:
    sub = substrate()
    model = build_backbone()
    tokens = stream_tokens(600)
    target = torch.tensor([3])
    out = {}
    path = chain(sub, 6)
    arcs = sub.physical_arcs(path)
    plan = ExecutionPlan(vertices=path, arcs=arcs, rounds=len(path), evidence=(path[0],), query=(path[2],),
                         readout=(path[-1],))
    # gate 5: every kernel parameter receives a finite nonzero gradient through the 5-hop path
    kernels = kernels_for(model, gate=0.5)
    session = _prepared(model, kernels, tokens, path)
    with torch.enable_grad():
        session.reason(plan)
        loss = torch.nn.functional.cross_entropy(session.readout().logits, target)
        loss.backward()
    grads = {n: (None if p.grad is None else float(p.grad.abs().sum())) for n, p in kernels.named_parameters()}
    out["multi_hop"] = {"hops_evidence_to_readout": len(path) - 1, "rounds": plan.rounds, "grad_l1": grads,
                        "ok_all_finite_nonzero": all(g is not None and math.isfinite(g) and g > 0
                                                     for g in grads.values())}
    # bounded graph: repeated reason() calls on one observation build graphs of the same size; observe drops them
    sizes = []
    with torch.enable_grad():
        for _ in range(3):
            session.reason(plan)
            sizes.append(graph_size(session._reasoning.latent))
    out["bounded_graph"] = {"graph_nodes_per_call": sizes, "equal_sizes": len(set(sizes)) == 1,
                            "ok_working_state_holds_no_graph": session.working.latent.grad_fn is None}
    session.observe(tokens[:, 560:600])
    out["bounded_graph"]["ok_observation_drops_results"] = session._reasoning is None
    session.close()
    # the zero gate: the refined logits start as the backbone's; only the gate receives a gradient
    kernels0 = kernels_for(model, gate=0.0)
    s0 = _prepared(model, kernels0, tokens, path)
    with torch.enable_grad():
        s0.reason(plan)
        torch.nn.functional.cross_entropy(s0.readout().logits, target).backward()
    nonzero = sorted(n for n, p in kernels0.named_parameters() if p.grad is not None and bool(p.grad.abs().sum() > 0))
    out["zero_gate"] = {"nonzero_gradients": nonzero, "ok_only_gate": nonzero == ["gate"]}
    s0.close()
    # gate 7 on the bridge: evidence at one end, readout at the other; fewer rounds than hops -> exactly zero
    hop = {}
    for rounds in (len(path) - 2, len(path) - 1):
        k = kernels_for(model, gate=0.5)
        s = _prepared(model, k, tokens, path)
        short = ExecutionPlan(vertices=path, arcs=arcs, rounds=rounds, evidence=(path[0],), readout=(path[-1],))
        with torch.enable_grad():
            s.reason(short)
            torch.nn.functional.cross_entropy(s.readout().logits, target).backward()
        g = 0.0 if k.evidence_in.weight.grad is None else float(k.evidence_in.weight.grad.abs().sum())
        hop[str(rounds)] = {"evidence_ingress_grad_l1": g, "distance": len(path) - 1,
                            "ok_bound": (g == 0.0) if rounds < len(path) - 1 else (g > 0.0)}
        s.close()
    out["hop_bound"] = hop
    # gate 6a: float64 gradcheck of the executor and readout (initial rows and kernel weights)
    small = reference_kernels(width=3, hidden=4)
    tiny = chain(sub, 4, start=31)
    working = TensorWorkingState(sub, capacity=4, width=3, dtype=torch.float64)
    working.acquire(tiny)
    compiled = compile_plan(ExecutionPlan(vertices=tiny, arcs=sub.physical_arcs(tiny), rounds=3, readout=(tiny[-1],)),
                            working, max_rounds=8, max_arcs=64, max_visits=1024, max_evidence=1)
    names = ("message_in.weight", "port_out", "port_in", "message_out.weight", "update_gate.weight",
             "update_cand.weight", "readout_proj.weight")
    params = dict(small.named_parameters())
    fn = substituted(small, names, lambda rows: small.readout(
        run_synchronous(rows, compiled, small).rows.index_select(0, compiled.readout)))
    inputs = (seeded_rows(len(tiny), 3, seed=4).requires_grad_(True),
              *(params[n].detach().clone().requires_grad_(True) for n in names))
    ok = torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-8, rtol=1e-6)
    out["gradcheck_executor_float64"] = {"inputs": ["initial_rows", *names], "rounds": 3, "hops": 3,
                                         "ok_gradcheck": bool(ok)}
    # gate 6b: float64 gradcheck end to end on the actual bridge: ingress, rounds, readout, injection and the
    # final block's actual routed channel, final norm and head (projected to 4 outputs to bound the cost)
    short = chain(sub, 4, start=2024)
    k = kernels_for(model, width=4, hidden=6, gate=0.5)
    s = _prepared(model, k, tokens, short)
    compiled = compile_plan(ExecutionPlan(vertices=short, arcs=sub.physical_arcs(short), rounds=3,
                                          evidence=(short[0],), query=(short[1],), readout=(short[-1],)),
                            s.working, max_rounds=16, max_arcs=1024, max_visits=16384, max_evidence=8,
                            evidence_available=1)
    rows0 = s.working.gather(compiled.slots)
    projection = torch.randn(model.config.vocab_size, 4, generator=torch.Generator().manual_seed(8),
                             dtype=torch.float64)
    names = ("query_in.weight", "evidence_query", "evidence_in.weight", "message_in.weight", "port_out",
             "update_cand.weight", "readout_proj.weight", "inject.weight", "gate")

    def end_to_end():
        _, latent = s.compute(compiled, rows0, s._observation, s._evidence)
        return s.refine(latent, s._observation) @ projection
    params = dict(k.named_parameters())
    inputs = tuple(params[n].detach().clone().requires_grad_(True) for n in names)
    ok = torch.autograd.gradcheck(substituted(k, names, end_to_end), inputs, eps=1e-6, atol=1e-8, rtol=1e-6)
    out["gradcheck_bridge_float64"] = {"inputs": list(names), "outputs": "refined logits @ a fixed [vocab, 4] map",
                                       "ok_gradcheck": bool(ok)}
    # no host extraction on the cognitive path (ingress, synchronous rounds, readout, injection)
    with HostExtraction() as control:
        t = torch.ones(3, dtype=torch.float64)
        t.sum().item(), t.tolist(), bool(t.any()), float(t[0]), int(t[1])
        try:
            t.numpy()
        except RuntimeError:                       # NumPy absent (as in CI): the request is still recorded
            pass
    with torch.enable_grad(), HostExtraction() as watch:
        _, latent = s.compute(compiled, rows0, s._observation, s._evidence)
        k.injection(latent)
    imports = {}
    for name in COGNITIVE:
        tree = ast.parse((TORCH / "geodome_cognitive" / f"{name}.py").read_text(encoding="utf-8"))
        imports[name] = sorted({a.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                                for a in node.names} | {(node.module or "").split(".")[0] for node in ast.walk(tree)
                                                        if isinstance(node, ast.ImportFrom) and node.level == 0})
    path_modules = ("kernels", "activation", "state", "plan", "binding")
    out["host_extraction"] = {"detector_positive_control": control.calls,
                              "ok_detector_sees_extraction": {"item", "tolist", "__bool__", "__float__", "__int__",
                                                              "numpy"} <= set(control.calls),
                              "calls_on_cognitive_path": watch.calls, "ok_none_on_cognitive_path": watch.calls == [],
                              "imports": imports,
                              "ok_no_numpy_or_ctypes_imports": not any({"numpy", "ctypes"} & set(imports[m])
                                                                       for m in path_modules),
                              "scope": "ingress, synchronous rounds, readout and injection. The backbone's routed "
                                       "channel keeps its own integer routing bookkeeping (R6) and the reference "
                                       "path (reference.py, Geodome's scheduler) converts rows by design."}
    s.close()
    differentiable = {k2: out[k2] for k2 in ("multi_hop", "bounded_graph", "zero_gate", "host_extraction")}
    return {"phase": PHASE, "command": "gradients", "checks": out,
            "gate_5": _gate({"m": out["multi_hop"], "z": out["zero_gate"]}),
            "gate_6": _gate({"a": out["gradcheck_executor_float64"], "b": out["gradcheck_bridge_float64"]}),
            "gate_7": _gate(hop), "gate_differentiable_path": _gate(differentiable), "provenance": provenance()}


# -- gates 2, 4, 8-12: the bridge on actual EvidenceLM ---------------------------------------------------------------

VARIANTS = [dict(mode=m, fine_keys=f, ple=p, origin=o, window=False)
            for m in ("off", "learned") for f in ("reread", "stored") for p in (False, True) for o in (0, 64)] + \
           [dict(mode="learned", fine_keys="reread", ple=False, origin=0, window=True),
            dict(mode="learned", fine_keys="stored", ple=True, origin=64, window=True)]


def _cognitive_round(session, path, sub, rounds: int = 4):
    """Enrol ``path``, admit two blocks before the query's own block, reason and read out (no_grad)."""
    qb = query_block(session)
    session.enrol(path)
    session.admit_evidence([session.state.ring.address(qb - 3), session.state.ring.address(qb - 1)])
    plan = ExecutionPlan(vertices=path, arcs=sub.physical_arcs(path), rounds=rounds, query=(path[2],),
                         evidence=(path[0], path[4]), readout=(path[1], path[3]))
    with torch.no_grad():
        session.reason(plan)
        return session.readout()


def run_variant(v: dict, *, length: int = 600) -> dict:
    """Disabled-path parity, zero-gate parity, the context clock, admission and its refusals, reset; one variant."""
    sub = substrate()
    model = build_backbone(fine_keys=v["fine_keys"], ple=v["ple"], window=v["window"])
    chunk = model.ersa.chunk
    tokens = stream_tokens(length, seed=11)
    options = {"mode": v["mode"]}
    session = CognitiveSession(model, kernels_for(model, gate=0.0), origin=v["origin"], evidence_options=options)
    reference = model.new_state(1, origin=v["origin"])
    out = {"variant": v}
    parity, digests, early = [], [], None
    with torch.no_grad():
        for at in range(0, length - length % chunk, chunk):
            piece = tokens[:, at:at + chunk]
            got = session.observe(piece)
            want = model(piece, reference, logit_positions=torch.tensor([piece.shape[1] - 1]), **options)[:, 0]
            parity.append(torch.equal(got, want))
            digests.append(state_digest(session.state) == state_digest(reference))
            lo, hi = session.state.ring.live()
            if early is None and hi - lo >= 2:
                early = session.state.ring.address(lo)
    out["disabled_path"] = {"equal_logits_every_chunk": all(parity), "equal_state_every_chunk": all(digests),
                            "chunks": len(parity)}
    ring = session.state.ring
    lo, hi = ring.live()
    observation = session._observation
    qb = query_block(session)
    out["ring"] = {"live": [lo, hi], "query_position": observation.position, "query_block": qb,
                   "ok_wrapped": observation.position + 1 - v["origin"] > CAPACITY,
                   "ok_evicted": early is not None and early.block < lo}
    path = chain(sub, 5, start=2024)
    clock = state_digest(session.state)
    before = counters(session.state)
    read = _cognitive_round(session, path, sub)
    after = counters(session.state)
    out["zero_gate"] = {"equal_refined_logits": torch.equal(read.logits, observation.logits),
                        "ok_latent_finite": bool(torch.isfinite(read.latent).all())}
    delta = {key: after[key] - before.get(key, 0) for key in after if after[key] != before.get(key, 0)}
    out["context_clock"] = {"equal_state_after_admit_reason_readout": state_digest(session.state) == clock,
                            "counter_deltas": delta,
                            "ok_exact_reread_only": delta.get("blocks_reencoded", 0) == 2
                            and delta.get("fine_tier_reads", 0) == 0}

    def canonical(b, **change):
        fields = dict(session=ring.session, epoch=ring.epoch, block=b, slot=b % ring.slots,
                      generation=b // ring.slots, start=b * BLOCK, end=(b + 1) * BLOCK)
        fields.update(change)
        return BlockAddress(**fields)
    out["admission"] = {
        "admitted": 2,
        "evicted": _refuses(lambda: session.admit_evidence([early]), StaleAddressError),
        "own_block": _refuses(lambda: session.admit_evidence([canonical(qb)]), CognitiveError, "EVIDENCE_CAUSALITY"),
        "future_block": _refuses(lambda: session.admit_evidence([canonical(hi + 2)]), CognitiveError,
                                 "EVIDENCE_CAUSALITY"),
        "tampered_generation": _refuses(lambda: session.admit_evidence(
            [canonical(qb - 2, generation=(qb - 2) // ring.slots + 1)]), StaleAddressError),
        "tampered_span": _refuses(lambda: session.admit_evidence([canonical(qb - 2, end=(qb - 2) * BLOCK + 8)]),
                                  StaleAddressError),
        "other_epoch": _refuses(lambda: session.admit_evidence([canonical(qb - 2, epoch=ring.epoch + 1)]),
                                StaleAddressError),
        "duplicate": _refuses(lambda: session.admit_evidence([canonical(qb - 2), canonical(qb - 2)]), CognitiveError,
                              "EVIDENCE_DUPLICATE"),
        "not_an_address": _refuses(lambda: session.admit_evidence([qb - 2]), CognitiveError, "EVIDENCE_TYPE")}
    out["context_clock"]["equal_state_after_refusals"] = state_digest(session.state) == clock
    # inference and training: observe() runs the frozen backbone without gradients even inside enable_grad (so the
    # stored fine keys of R8 stay an inference tier); the admitted evidence is always the exact reread
    stale_address = ring.address(qb - 2)
    with torch.enable_grad():
        logits = session.observe(tokens[:, length - chunk:])
    out["training_mode"] = {"ok_backbone_without_grad": not logits.requires_grad
                            and not session._observation.rows.requires_grad,
                            "fine_tier_reads_total": counters(session.state).get("fine_tier_reads", 0),
                            "stored_fine_keys": v["fine_keys"] == "stored"}
    # reset: a clean session; the same stream gives what a brand-new session gives; old identities are refused
    old_handle = session.working.handle(path[0])
    session.reset()
    fresh = CognitiveSession(model, kernels_for(model, gate=0.0), origin=v["origin"], evidence_options=options)
    equal = [torch.equal(a, b) for a, b in zip(observe_stream(session, tokens), observe_stream(fresh, tokens))]
    mine, theirs = _cognitive_round(session, path, sub), _cognitive_round(fresh, path, sub)
    out["reset"] = {"equal_fresh_session_logits": all(equal) and len(equal) == out["disabled_path"]["chunks"],
                    "equal_fresh_session_state": state_digest(session.state) == state_digest(fresh.state),
                    "equal_fresh_cognitive_latent": torch.equal(mine.latent, theirs.latent),
                    "equal_fresh_cognitive_logits": torch.equal(mine.logits, theirs.logits),
                    "stale_handle": _refuses(lambda: session.working.validate([old_handle]), ValueError),
                    "stale_address_of_the_previous_epoch": _refuses(lambda: session.admit_evidence([stale_address]),
                                                                    StaleAddressError)}
    session.reset()
    out["reset"]["ok_working_empty"] = session.working.resident == 0
    fresh.close()
    session.close()
    out["pass"] = _gate({key: value for key, value in out.items() if key != "variant"}) == "PASS"
    return out


def refusal_battery() -> dict:
    """Gates 8 and 9: every refusal, each followed by a check that nothing changed (atomicity)."""
    sub = substrate()
    model = build_backbone()
    tokens = stream_tokens(560, seed=21)
    kernels = kernels_for(model, gate=0.5, capacity=8)
    session = CognitiveSession(model, kernels)
    observe_stream(session, tokens)
    ring = session.state.ring
    qb = query_block(session)
    path = chain(sub, 5, start=4040)
    session.enrol(path)
    session.admit_evidence([ring.address(qb - 2)])
    good = ExecutionPlan(vertices=path, arcs=sub.physical_arcs(path), rounds=3, evidence=(path[0],),
                         readout=(path[-1],))
    results = {}
    with torch.no_grad():
        session.reason(good)
        results["before_readout"] = {"ok_bypass": session.prediction()[1] == "backbone"}
        committed = session.readout()
    results["after_readout"] = {"ok_cognitive": session.prediction()[1] == "cognitive",
                                "equal_committed": torch.equal(session.prediction()[0], committed.logits)}

    def snapshot():
        w = session.working
        return {"latent": w.latent.clone(), "resident": w.resident, "epoch": w.epoch, "version": w.version,
                "state": state_digest(session.state), "position": session.state.position,
                "objects": (session._observation, session._evidence, session._reasoning, session._readout)}

    def unchanged(a, b) -> bool:
        return (torch.equal(a["latent"], b["latent"])
                and all(a[k] == b[k] for k in ("resident", "epoch", "version", "state", "position"))
                and all(x is y for x, y in zip(a["objects"], b["objects"])))
    base = snapshot()
    far = chain(sub, 3, start=90000)
    outsider = sub.arc(far[0], far[1])
    two_hop = next(w for w in sub.neighbors(sub.neighbors(path[0])[0])
                   if w != path[0] and w not in sub.neighbors(path[0]))
    other = CognitiveSession(model, kernels_for(model, capacity=8))
    observe_stream(other, tokens)
    other.enrol((path[0],))
    foreign_handle = other.working.handle(path[0])
    foreign_address = other.state.ring.address(qb - 2)          # the same block, slot and generation; another ring
    compiled = compile_plan(good, session.working, max_rounds=16, max_arcs=1024, max_visits=16384, max_evidence=8,
                            evidence_available=1)
    rows = session.working.gather(compiled.slots)

    def plan(**changes):
        fields = dict(vertices=path, arcs=sub.physical_arcs(path), rounds=3, evidence=(path[0],), readout=(path[-1],))
        fields.update(changes)
        return ExecutionPlan(**fields)
    cases = {
        "unauthorized_arc": (lambda: session.reason(plan(arcs=(*sub.physical_arcs(path), outsider))), "ARC_AUTHORITY"),
        "fabricated_non_edge": (lambda: sub.arc(path[0], two_hop), None),
        "arc_out_of_range": (lambda: session.reason(plan(arcs=(sub.arcs,))), "ARC_RANGE"),
        "duplicate_arc": (lambda: session.reason(plan(arcs=(sub.physical_arcs(path)[0],) * 2)), "DUPLICATE_ARC"),
        "nonresident_vertex": (lambda: session.reason(plan(vertices=(*path, far[0]))), "NONRESIDENT"),
        "rounds_over_cap": (lambda: session.reason(plan(rounds=17)), "ROUND_RANGE"),
        "visit_budget": (lambda: session.reason(plan(max_visits=3)), "VISIT_BUDGET"),
        "evidence_binding_without_evidence": (lambda: session.reason(plan(evidence=(path[0], path[1]))),
                                              "EVIDENCE_BINDING"),
        "binding_overlap": (lambda: session.reason(plan(query=(path[0],))), "BINDING_OVERLAP"),
        "readout_missing": (lambda: session.reason(plan(readout=())), "READOUT_BINDING"),
        "frontier_on_synchronous": (lambda: session.reason(plan(start=(path[0],))), "START"),
        "unknown_schedule": (lambda: session.reason(plan(schedule="eager")), "SCHEDULE"),
        "capacity_overflow": (lambda: session.enrol(chain(sub, 9, start=60000)), "STATE_CAPACITY"),
        "already_resident": (lambda: session.enrol((path[0],)), "ALREADY_RESIDENT"),
        "foreign_handle": (lambda: session.working.validate([foreign_handle]), "FOREIGN_OR_STALE_HANDLE"),
        "foreign_address": (lambda: session.admit_evidence([foreign_address]), None),
        "too_much_evidence": (lambda: session.admit_evidence([ring.address(b) for b in range(qb - 9, qb)]),
                              "EVIDENCE_COUNT"),
        "publish_nonfinite_rows": (lambda: session.working.publish(
            compiled.handles, compiled.slots, rows.index_fill(0, compiled.readout, float("nan"))), None),
        "publish_wrong_shape": (lambda: session.working.publish(compiled.handles, compiled.slots, rows[:-1]), None),
        "publish_wrong_dtype": (lambda: session.working.publish(compiled.handles, compiled.slots, rows.float()), None),
        "publish_foreign_slots": (lambda: session.working.publish(compiled.handles, compiled.slots.flip(0), rows),
                                  None),
        "wrong_token_dtype": (lambda: session.observe(tokens[:, :8].float()), None),
        "too_many_tokens": (lambda: session.observe(tokens[:, :71]), None),
        "batch_two": (lambda: session.observe(tokens[:, :8].expand(2, -1)), None),
    }
    kinds = (PlanError, KeyError, ValueError, OverflowError, StaleAddressError, CognitiveError, WorkingStateError)
    for name, (fn, code) in cases.items():
        r = _refuses(fn, kinds, code)
        r["equal_unchanged"] = unchanged(snapshot(), base)
        results[name] = r
    # a non-finite computation: refused, nothing published, the committed readout stays the answer
    with torch.no_grad():
        saved = kernels.update_cand.weight.clone()
        kernels.update_cand.weight[0, 0] = float("nan")
    r = _refuses(lambda: session.reason(good), CognitiveError, "NONFINITE")
    r["equal_unchanged"] = unchanged(snapshot(), base)
    r["equal_committed_answer"] = torch.equal(session.prediction()[0], committed.logits)
    results["nonfinite_computation"] = r
    with torch.no_grad():
        kernels.update_cand.weight.copy_(saved)
    # lifetime refusals
    other.close()
    results["closed_session"] = _refuses(lambda: other.observe(tokens[:, :8]), SessionClosedError)
    results["closed_session_reason"] = _refuses(lambda: other.reason(good), SessionClosedError)
    results["no_serialization"] = _refuses(lambda: session.__reduce_ex__(2), TypeError)
    results["no_copy"] = _refuses(lambda: copy.deepcopy(session), TypeError)
    with torch.no_grad():
        model.blocks[0].mix_norm.weight.mul_(1.0)          # an in-place write: the backbone's revision advances
    results["backbone_revision"] = _refuses(lambda: session.reason(good), ModelRevisionError)
    results["backbone_revision_sticky"] = _refuses(lambda: session.prediction(), ModelRevisionError)
    session.close()
    return results


def residency_bridge() -> dict:
    """Gate 4 on the bridge: the same cognitive computation in sparse and dense residency, bit for bit."""
    sub = substrate()
    model = build_backbone()
    tokens = stream_tokens(560, seed=31)
    out = {}
    for residency in ("sparse", "dense"):
        session = CognitiveSession(model, kernels_for(model, gate=0.5), residency=residency)
        observe_stream(session, tokens)
        out[residency] = _cognitive_round(session, chain(sub, 5, start=5151), sub)
        session.close()
    return {"equal_latent": torch.equal(out["sparse"].latent, out["dense"].latent),
            "equal_logits": torch.equal(out["sparse"].logits, out["dense"].logits)}


def command_bridge(args) -> dict:
    variants = [run_variant(v) for v in VARIANTS]
    battery = refusal_battery()
    residency = residency_bridge()
    # gate 10: training the cognitive kernels never writes the backbone
    sub = substrate()
    model = build_backbone()
    digest = parameter_digest(model)
    kernels = kernels_for(model, gate=0.5)
    path = chain(sub, 5)
    session = _prepared(model, kernels, stream_tokens(600), path, evidence=1)
    optimizer = torch.optim.SGD(kernels.parameters(), lr=0.1)
    for step in range(3):
        with torch.enable_grad():
            session.reason(ExecutionPlan(vertices=path, arcs=sub.physical_arcs(path), rounds=4, evidence=(path[0],),
                                         query=(path[2],), readout=(path[-1],)))
            loss = torch.nn.functional.cross_entropy(session.readout().logits, torch.tensor([step + 1]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    immutable = {"equal_parameter_digest": parameter_digest(model) == digest,
                 "ok_no_backbone_grad": all(p.grad is None for p in model.parameters()),
                 "ok_kernels_trained": any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                           for p in kernels.parameters())}
    session.close()
    gates = {"gate_2": all(v["disabled_path"]["equal_logits_every_chunk"] and v["disabled_path"]["equal_state_every_chunk"]
                           and v["zero_gate"]["equal_refined_logits"] for v in variants),
             "gate_4_bridge": all(residency.values()),
             "gate_8": all(r.get("refused", True) and r.get("ok_code", True) for r in battery.values())
             and all(x.get("refused", True) and x.get("ok_code", True)
                     for v in variants for x in v["admission"].values() if isinstance(x, dict)),
             "gate_9": all(r.get("equal_unchanged", True) for r in battery.values())
             and all(v["context_clock"]["equal_state_after_refusals"] for v in variants),
             "gate_10": all(immutable.values()),
             "gate_11": all(v["context_clock"]["equal_state_after_admit_reason_readout"] for v in variants),
             "gate_12": all(_gate(v["reset"]) == "PASS" for v in variants),
             "evidence_admission": all(_gate(v["admission"]) == "PASS" and v["context_clock"]["ok_exact_reread_only"]
                                       and v["ring"]["ok_wrapped"] and v["ring"]["ok_evicted"] for v in variants),
             "training_mode": all(v["training_mode"]["ok_backbone_without_grad"] for v in variants),
             "variants": all(v["pass"] for v in variants)}
    return {"phase": PHASE, "command": "bridge", "variants": variants, "refusals": battery,
            "residency_bridge": residency, "immutability": immutable,
            "gates": {k: "PASS" if ok else "FAIL" for k, ok in gates.items()}, "provenance": provenance()}


# -- resources --------------------------------------------------------------------------------------------------------

def command_resources(args) -> dict:
    sub = substrate()
    model = build_backbone(dtype="float32")
    tokens = stream_tokens(600)
    kernels = kernels_for(model, width=32, hidden=64, capacity=64, gate=0.5)
    session = CognitiveSession(model, kernels)
    timings = {"observe": [], "admit_evidence_4_blocks": [], "reason_inference": [], "readout_inference": [],
               "reason_readout_backward_training": []}
    chunk = model.ersa.chunk
    for at in range(0, 560, chunk):
        t = time.perf_counter()
        session.observe(tokens[:, at:at + chunk])
        timings["observe"].append(time.perf_counter() - t)
    region = ball(sub, 3000, 48)
    arcs = sub.physical_arcs(region)
    session.enrol(region)
    qb = query_block(session)
    plan = ExecutionPlan(vertices=region, arcs=arcs, rounds=8, query=(region[0],), evidence=region[1:5],
                         readout=region[-4:])
    for _ in range(5):
        t = time.perf_counter()
        session.admit_evidence([session.state.ring.address(b) for b in range(qb - 4, qb)])
        timings["admit_evidence_4_blocks"].append(time.perf_counter() - t)
        with torch.no_grad():
            t = time.perf_counter()
            session.reason(plan)
            timings["reason_inference"].append(time.perf_counter() - t)
            t = time.perf_counter()
            session.readout()
            timings["readout_inference"].append(time.perf_counter() - t)
    saved, scratch, nodes = [], [], []

    def pack(t):
        saved.append(t.numel() * t.element_size())
        return t
    for _ in range(3):
        saved.clear()
        t = time.perf_counter()
        with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            session.reason(plan)
            loss = session.readout().logits.logsumexp(-1).sum()
        nodes.append(graph_size(loss))
        loss.backward()
        timings["reason_readout_backward_training"].append(time.perf_counter() - t)
        scratch.append(sum(saved))
        kernels.zero_grad()
    compiled = compile_plan(plan, session.working, max_rounds=16, max_arcs=1024, max_visits=16384, max_evidence=8,
                            evidence_available=4)
    plan_bytes = sum(t.numel() * t.element_size() for t in (compiled.slots, compiled.src, compiled.dst, compiled.ports,
                                                             compiled.back_ports, compiled.arc_ids, compiled.incoming,
                                                             compiled.start, compiled.query, compiled.evidence,
                                                             compiled.readout))
    record = {"phase": PHASE, "command": "resources",
              "label": "MEASURED on this CPU container; float32 backbone and kernels; latencies are medians",
              "parameters": {"kernels": sum(p.numel() for p in kernels.parameters()),
                             "by_tensor": {n: p.numel() for n, p in kernels.named_parameters()},
                             "backbone": sum(p.numel() for p in model.parameters())},
              "bytes": {"session_projection": session.projected_bytes(), "working_state": session.working.nbytes(),
                        "topology_index_payload_shared": sub.index.payload_bytes(), "compiled_plan": plan_bytes,
                        "gradient_saved_tensors_per_training_step": scratch,
                        "ok_bounded_scratch": len(set(scratch)) == 1},
              "graph_nodes_per_training_step": nodes,
              "plan": {"enrolled": len(region), "admitted_arcs": len(arcs), "rounds": plan.rounds,
                       "message_visits_per_reason": plan.rounds * len(arcs),
                       "active_vertices_per_round": int(compiled.incoming.sum()),
                       "evidence_items": 4, "readout_vertices": 4},
              "latency_ms": {k: round(1000 * statistics.median(v), 3) for k, v in timings.items()},
              "index_build_seconds": round(sub.build_seconds, 3),
              "peak_rss_bytes_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "threads": torch.get_num_threads(), "provenance": provenance()}
    session.close()
    return record


RECORDS = ("identity", "audit", "parity", "gradients", "bridge", "resources")
REASONS = {
    "DISTRIBUTED_COMPETENCE": "P5 was not run: the owner asked for the P0-P4 results before any competence study. The "
                              "protocol is a DRAFT for review (PREREGISTRATION_R13_P5.md); nothing was trained for "
                              "competence and no data was collected.",
    "LEARNED_ADDRESSING": "P6 is admitted only after P1-P5; plans here are host-supplied and validated.",
    "CGC_ADMISSION": "P7 (optional) was not started; no CGC volatile weight adaptation enters this first integration.",
    "HCA88_ADMISSION": "HCA88 is qualified only for C4 two-arc relation windows; it is not used here.",
    "COGNITIVE_ADVANTAGE": "Mechanical and gradient qualification supports no cognitive or geometric advantage claim.",
}


def command_verdict(args) -> dict:
    """The R13 dispositions, computed from the raw gate records (the records test recomputes them independently)."""
    r = {name: json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8")) for name in RECORDS}
    bridge, refusals = r["bridge"]["gates"], r["bridge"]["refusals"]

    def passed(*values) -> str:
        return "PASS" if values and all(v == "PASS" for v in values) else "FAIL"
    caps = ("rounds_over_cap", "visit_budget", "capacity_overflow", "too_much_evidence", "too_many_tokens")
    bounded = (all(refusals[k]["refused"] and refusals[k]["equal_unchanged"] for k in caps)
               and r["gradients"]["checks"]["bounded_graph"]["equal_sizes"]
               and r["gradients"]["checks"]["bounded_graph"]["ok_working_state_holds_no_graph"]
               and r["resources"]["bytes"]["ok_bounded_scratch"])
    dispositions = {
        "GEODOME_SOURCE_PIN": f"{gsource.GEODOME_REPOSITORY} @ {gsource.GEODOME_COMMIT}, Geodome/ (Reference API "
                              f"v{gsource.GEODOME_API_VERSION}, source pin {gsource.GEODOME_SOURCE_PIN}, "
                              f"PUBLIC.sha256 {gsource.MANIFEST_SHA256})",
        "ELPIS_BASE_PIN": ELPIS_BASE,
        "SUBSTRATE_IDENTITY_PARITY": passed(r["identity"]["gate_1"], r["parity"]["gate_3"],
                                            r["parity"]["gate_4_executor"], bridge["gate_4_bridge"]),
        "DIFFERENTIABLE_EXECUTION": passed(r["gradients"]["gate_5"], r["gradients"]["gate_6"], r["gradients"]["gate_7"],
                                           r["gradients"]["gate_differentiable_path"], r["parity"]["gate_7_executor"]),
        "DISABLED_PATH_PARITY": passed(bridge["gate_2"]),
        "CONTEXT_STATE_COHERENCE": passed(r["audit"]["gate_seam"], bridge["gate_11"], bridge["gate_12"],
                                          bridge["training_mode"]),
        "EVIDENCE_ADMISSION": passed(bridge["evidence_admission"]),
        "SESSION_ISOLATION": passed(bridge["gate_8"], bridge["gate_9"], bridge["gate_10"], bridge["variants"]),
        "BOUNDED_COMPUTATION": "PASS" if bounded else "FAIL",
        "DISTRIBUTED_COMPETENCE": "HOLD",
        "LEARNED_ADDRESSING": "NOT_STARTED",
        "CGC_ADMISSION": "NOT_STARTED",
        "HCA88_ADMISSION": "NOT_ADMITTED",
        "COGNITIVE_ADVANTAGE": "NOT_CLAIMED",
    }
    return {"phase": PHASE, "command": "verdict", "inputs": {f"{n}.json": sha(RESULTS / f"{n}.json") for n in RECORDS},
            "dispositions": dispositions, "reasons": REASONS,
            "scope": "Mechanical and gradient qualification of P0-P4 on CPU (float64 reference checks, float32 "
                     "resources). A mechanical PASS and a learned-competence HOLD are independent outcomes.",
            "ci_status": "not a property of this record: the exact-head CI result is reported in EVIDENCE_R13.md "
                         "and the pull request",
            "provenance": provenance()}


def command_manifest(args) -> dict:
    files = sorted(p for p in RESULTS.glob("*.json") if p.name != "MANIFEST.json")
    return {"phase": PHASE, "command": "manifest", "files": {p.name: sha(p) for p in files},
            "geodome": gsource.source_record(), "sources": {k: sha(v) for k, v in SOURCES.items()}}


COMMANDS = {"identity": command_identity, "audit": command_audit, "parity": command_parity,
            "gradients": command_gradients, "bridge": command_bridge, "resources": command_resources,
            "verdict": command_verdict, "manifest": command_manifest}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", help="the JSON record to write")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    record = COMMANDS[args.command](args)
    text = json.dumps(record, indent=1, sort_keys=True, default=str) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
