#!/usr/bin/env python3
"""Elpis Torch R6: measurements for the expertized backbone. Records go only where --out says.

    scale      the 500M / 1B sparse configurations: total, resident and active parameters, weight memory, per-token
               execution cost, context memory at 8K .. 1M (EXACT state, PROJECTED scratch), and a BOUNDED real
               forward of each instantiated model on this CPU (weights are untrained: no capability is implied)
    dispatch   grouped (selected experts only) against the masked reference: time and agreement
    controls   small training comparison on this repository's own source bytes: dense FFN+CGC, routed FFN, routed
               FFN+CGC, routed FFN+CGC + shared, and routed FFN+CGC with a FROZEN random router (does routing help?)
    cache      the R5 reread cache, cold against warm, on an untrained evidence model (mechanics, not quality)
    smoke      every command once, tiny

Wall-clock numbers are PERFORMANCE_ONLY (one CPU, shared).
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import time

import torch

from .. import evidence_tasks as et
from .. import lm_training as lmt
from ..config import ModelConfig
from ..ersa_model import ErsaConfig, ErsaLM
from ..evidence_model import EvidenceConfig, EvidenceLM
from ..sparse_channel import SparseChannel, SparseConfig, active_parameters, channel_reports
from ..tokenizer import ByteTokenizer
from ..training import TrainConfig

PHASE = "ELPIS_TORCH_R6_EXPERTIZED_BACKBONE"
VOCAB = 65_536
SCALES = {   # name: (d_model, layers, experts, top_k, shared expert)
    "r6_500m": (768, 20, 8, 2, False),
    "r6_1b": (1024, 18, 16, 2, True),
}
CONTEXTS = (8192, 32768, 131072, 262144, 1 << 20)


def scale_model(name: str, *, capacity: int = 1 << 20, device="meta"):
    d, layers, experts, top_k, shared = SCALES[name]
    heads = d // 128
    kv = 2 if heads % 2 == 0 else 1
    m = ModelConfig(vocab_size=VOCAB, d_model=d, n_layers=layers, n_heads=heads, n_kv_heads=kv, max_seq_len=64,
                    channel="ffn+cgc", ffn_hidden=round(8 * d / 3), cgc_bottleneck=d // 2, cgc_channels=2 * d,
                    cgc_units=4, cgc_gate_init=0.1)
    sparse = SparseConfig(experts=experts, top_k=top_k, layers=tuple(range(1, layers, 2)), shared=shared,
                          ffn_hidden=round(4 * d / 3 / (2 if shared else 1)), cgc_channels=d, cgc_bottleneck=d // 4)
    ersa = ErsaConfig(m, ersa_heads=heads, key_dim=128, value_dim=128, conv_kernel=4, attention="window",
                      attn_heads=heads, attn_kv_heads=kv, attn_window=4096, attn_layers=tuple(range(3, layers, 4)),
                      chunk=512, ersa_kernel="chunked", scan_block=64, sparse=sparse)
    # A narrow evidence encoder, as in R3's scaling. 32-query tiles bound a read at every position (R5's default
    # 128-query tile can re-encode 8,192 distinct candidates at once: over the 2 GiB ceiling at 262K); the tile is an
    # execution option and changes no logit (tested).
    evidence = EvidenceConfig(capacity=capacity, encoder_dim=256, encoder_heads=4, read_heads=8, read_head_dim=128,
                              fine_dim=64, query_tile=32)
    with torch.device(device):
        return EvidenceLM(ersa, evidence)


def _group_counts(model) -> dict:
    groups = {}
    for name, p in model.named_parameters():
        key = ("embeddings_and_head" if name.startswith("embed.") else
               "evidence_path" if name.startswith(("encoder.", "lookup.")) else
               "routers" if ".router." in name else
               "routed_experts" if ".experts." in name else
               "shared_experts" if ".shared." in name else
               "attention" if ".attn." in name else
               "ersa" if ".ersa." in name else
               "dense_channels" if (".ffn." in name or ".cgc." in name or name.endswith("cgc_gate")) else "norms_other")
        groups[key] = groups.get(key, 0) + p.numel()
    return groups


def per_token_macs(model, context: int) -> dict:
    """Multiply-accumulates per decoded token (weights; elementwise work ignored). PROJECTED."""
    e = model.evidence
    counts = active_parameters(model)
    weights_active = counts["active_per_token"] - model.embed.weight.numel()            # lookup, not a product
    head = model.embed.weight.numel()                                                   # tied output head
    live = min(context, e.capacity) // e.block
    encoder_per_token = sum(p.numel() for n, p in model.named_parameters()
                            if n.startswith("encoder.") and p.ndim >= 2 and "delta" not in n)
    return {"active_weights": weights_active, "output_head": head,
            "coarse_scan": live * e.per_block * e.key_dim,
            "reread_reencode": e.coarse_top * e.span * encoder_per_token,
            "note": "per query of the evidence read; the core terms are per token"}


def command_scale(args) -> dict:
    out = {"phase": PHASE, "configurations": {}}
    for name in SCALES:
        model = scale_model(name)
        counts = active_parameters(model)
        groups = _group_counts(model)
        interface = model.context_interface()
        contexts = {}
        for length in CONTEXTS:
            sized = scale_model(name, capacity=length)
            fixed = sized.context_interface().fixed_bytes(1, {"history_device": "cuda"})      # bfloat16 index
            fixed_cpu = sized.context_interface().fixed_bytes(1)
            decode = sized.context_interface().scratch_bytes(batch=1, chunk=1, options={}, queries=1)
            prefill = sized.context_interface().scratch_bytes(batch=1, chunk=512, options={}, queries=512)
            contexts[length] = {"state_exact_bf16_index": fixed, "state_exact_fp32_index": fixed_cpu["total"],
                                "scratch_decode_projected_fp32": decode["total"],
                                "scratch_prefill512_projected_fp32": prefill["total"],
                                "scratch_prefill512_by_component": prefill,
                                "scratch_prefill512_by_query_tile": _prefill_by_query_tile(sized),
                                "coarse_scan_macs_per_query": (length // sized.evidence.block) *
                                sized.evidence.per_block * sized.evidence.key_dim}
        out["configurations"][name] = {
            "geometry": dict(zip(("d_model", "layers", "experts", "top_k", "shared"), SCALES[name])),
            "parameters": counts, "by_group": groups,
            "weight_bytes": {"fp32": 4 * counts["resident"], "bf16": 2 * counts["resident"],
                             "note": "all experts resident: sparse activation saves compute, not weight memory"},
            "macs_per_token_at_1M": per_token_macs(model, 1 << 20),
            "context": contexts, "interface": interface.kind}
        if not args.smoke and not args.no_forward:
            out["configurations"][name]["bounded_forward"] = _bounded_forward(name)
    return out


def _prefill_by_query_tile(model, tiles=(128, 64, 32, 16)) -> dict:
    """PROJECTED scratch of a 512-token prefill reading at every position, per query tile (R5's default is 128)."""
    kept, out = model.evidence, {}
    try:
        for tile in tiles:
            model.evidence = kept.replace(query_tile=tile)
            out[str(tile)] = model.context_interface().scratch_bytes(batch=1, chunk=512, options={},
                                                                      queries=512)["total"]
    finally:
        model.evidence = kept
    return out


@torch.no_grad()
def _bounded_forward(name: str) -> dict:
    """A real forward of the instantiated model on this CPU: 2 x 64 tokens, then 4 decode steps. Untrained."""
    started = time.perf_counter()
    model = scale_model(name, capacity=1 << 14, device="cpu").eval()
    built = time.perf_counter() - started
    state = model.new_state(1)
    tokens = torch.randint(0, VOCAB, (1, 128), generator=torch.Generator().manual_seed(0))
    started = time.perf_counter()
    logits = model(tokens[:, :64], state, logit_positions=torch.tensor([63]))
    logits = model(tokens[:, 64:], state, logit_positions=torch.tensor([63]))
    prefill = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(4):
        logits = model(logits[:, -1].argmax(-1)[:, None], state)
    decode = (time.perf_counter() - started) / 4
    reports = channel_reports(model)
    return {"finite": bool(torch.isfinite(logits).all()), "build_seconds": round(built, 1),
            "prefill_128_seconds": round(prefill, 2), "decode_seconds_per_token": round(decode, 3),
            "state_bytes": state.nbytes()["total"], "routed_blocks": len(reports),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "note": "capacity 16,384 for the forward (the 1M ring is accounted above); untrained weights"}


def command_dispatch(args) -> dict:
    m = ModelConfig(vocab_size=259, d_model=256, n_layers=2, n_heads=2, max_seq_len=64, channel="ffn+cgc",
                    ffn_hidden=341, cgc_bottleneck=64, cgc_channels=256, cgc_units=4)
    out = {"phase": PHASE, "rows": []}
    shapes = ((8, 2, 512),) if args.smoke else ((8, 2, 512), (8, 2, 4096), (16, 2, 4096), (64, 2, 4096))
    for experts, top_k, tokens in shapes:
        grouped = SparseChannel(m, SparseConfig(experts=experts, top_k=top_k, layers=(0,)), torch.float32)
        reference = SparseChannel(m, SparseConfig(experts=experts, top_k=top_k, layers=(0,), dispatch="reference"),
                                  torch.float32)
        with torch.no_grad():
            for p in grouped.parameters():
                p.normal_(0, 0.05)
        reference.load_state_dict(grouped.state_dict())
        u = torch.randn(tokens, 256)
        row = {"experts": experts, "top_k": top_k, "tokens": tokens}
        for label, channel in (("grouped", grouped), ("reference", reference)):
            times = []
            for _ in range(3):
                x = u.clone().requires_grad_(True)
                started = time.perf_counter()
                channel(x).square().mean().backward()
                times.append(time.perf_counter() - started)
            row[f"{label}_seconds_median"] = round(sorted(times)[1], 4)
        with torch.no_grad():
            row["max_abs_difference"] = float((grouped(u) - reference(u)).abs().max())
        row["speedup"] = round(row["reference_seconds_median"] / row["grouped_seconds_median"], 2)
        row["expert_evaluations_grouped"] = tokens * top_k
        row["expert_evaluations_reference"] = tokens * experts
        out["rows"].append(row)
    return out


CONTROL_VARIANTS = {   # name: (dense channel, sparse config or None, freeze the router)
    "dense_ffn_cgc": ("ffn+cgc", None, False),
    "routed_ffn": ("ffn+cgc", SparseConfig(experts=8, top_k=2, layers=(0, 1), kind="ffn", ffn_hidden=86), False),
    "routed_ffn_cgc": ("ffn+cgc", SparseConfig(experts=8, top_k=2, layers=(0, 1), ffn_hidden=86, cgc_channels=48,
                                               cgc_bottleneck=16), False),
    "routed_ffn_cgc_shared": ("ffn+cgc", SparseConfig(experts=8, top_k=2, layers=(0, 1), shared=True, ffn_hidden=43,
                                                      cgc_channels=48, cgc_bottleneck=16), False),
    "routed_ffn_cgc_frozen_router": ("ffn+cgc", SparseConfig(experts=8, top_k=2, layers=(0, 1), ffn_hidden=86,
                                                             cgc_channels=48, cgc_bottleneck=16), True),
}


def control_model(name: str, seed: int):
    channel, sparse, _ = CONTROL_VARIANTS[name]
    m = ModelConfig(vocab_size=259, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64, channel=channel,
                    ffn_hidden=171, cgc_bottleneck=32, cgc_channels=96, cgc_units=4, cgc_gate_init=0.1, seed=seed)
    return ErsaLM(ErsaConfig(m, conv_kernel=4, chunk=64, ersa_kernel="chunked", scan_block=64, sparse=sparse))


def command_controls(args) -> dict:
    import pathlib
    tokenizer = ByteTokenizer()
    root = pathlib.Path(lmt.__file__).parent
    docs = list(lmt.iter_documents([str(root)], suffixes=(".py",)))
    val_docs, train_docs = docs[:4], docs[4:]
    lm = lmt.LMConfig(seq=128, chunk=64)
    steps = 6 if args.smoke else args.steps
    seeds = (0,) if args.smoke else tuple(range(args.seeds))
    val = list(lmt.document_batches(list(lmt.document_rows(val_docs, tokenizer, lm.seq)), 16, seed=99,
                                    ignore_pad=tokenizer.PAD))[:2 if args.smoke else 8]
    out = {"phase": PHASE, "corpus": {"train_documents": len(train_docs), "val_documents": len(val_docs),
                                      "train_bytes": sum(map(len, train_docs)), "val_bytes": sum(map(len, val_docs)),
                                      "source": "research/elpis_torch/*.py of this repository (project-owned)"},
           "steps": steps, "rows": []}
    for name in CONTROL_VARIANTS:
        for seed in seeds:
            model = control_model(name, seed)
            if CONTROL_VARIANTS[name][2]:
                for block in model.blocks:
                    block.sparse.router.weight.requires_grad_(False)
            rows = list(lmt.document_rows(train_docs, tokenizer, lm.seq))
            started = time.perf_counter()
            result = lmt.train_lm(model, lmt.document_batches(rows * 4, 16, seed=seed, ignore_pad=tokenizer.PAD),
                                  TrainConfig(steps=steps, batch_size=16, lr=3e-3, warmup_steps=min(50, steps),
                                              seed=seed), lm)
            evaluation = lmt.evaluate_lm(model, val, lm)
            counts = active_parameters(model)
            routing = [r["stats"] for r in channel_reports(model)]
            out["rows"].append({"variant": name, "seed": seed, "status": result["status"],
                                "train_seconds": round(time.perf_counter() - started, 1),
                                "val_bits_per_byte": round(evaluation["bits_per_byte"], 4),
                                "parameters": counts,
                                "routing": [{k: r[k] for k in ("load_imbalance_max_over_mean", "router_entropy",
                                                               "top1_probability", "inactive_fraction",
                                                               "active_macs_per_token")} for r in routing]})
            print(f"[r6 controls] {name} s{seed}: {evaluation['bits_per_byte']:.3f} bits/byte "
                  f"(total {counts['total']}, active {counts['active_per_token']})", flush=True)
            if args.out:
                _write(args.out, out)
    return out


@torch.no_grad()
def command_cache(args) -> dict:
    from ..context_runtime import ContextSession
    out = {"phase": PHASE, "rows": [], "model": "R5 default geometry, UNTRAINED (mechanics only)"}
    lengths = (4096,) if args.smoke else (65536, 262144)
    for length in lengths:
        stream = et.generate_records(et.RecordTask(queries=48, background=8.0), 1, length, seed=17)
        probes = sorted(set((stream.probes["answer"] - 1).tolist()))
        for cache in (0, 16 << 20, 64 << 20):
            m = ModelConfig(vocab_size=et.VOCAB, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64,
                            channel="cgc", cgc_bottleneck=32, cgc_channels=96, cgc_units=4, seed=0)
            model = EvidenceLM(ErsaConfig(m, conv_kernel=4, chunk=4096, ersa_kernel="chunked", scan_block=64),
                               EvidenceConfig()).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            with ContextSession(model, batch=1, chunk=4096, cache_bytes=cache) as session:
                session.state.timing = {}
                started = time.perf_counter()
                result = session.prefill(stream.tokens, probes=probes, keep_last=False)
                seconds = time.perf_counter() - started
                counters = session.counters()
                timing = session.state.timing
            queries = max(1, timing.get("queries", 1))
            out["rows"].append({"tokens": length, "cache_bytes": cache, "seconds": round(seconds, 2),
                                "reread_ms_per_query": round(1000 * timing.get("exact_reread", 0) / queries, 2),
                                "blocks_reencoded": counters.get("blocks_reencoded", 0),
                                "cache_hits": counters.get("cache_hits"), "cache_misses": counters.get("cache_misses"),
                                "queries": queries, "probes": len(result.probes)})
            print(f"[r6 cache] {length} cache={cache >> 20} MiB: {out['rows'][-1]}", flush=True)
    return out


CAUSAL_CUT = 301


def _causality_model(chunk: int = 1024, dispatch: str = "grouped"):
    """The model of test_experts_with_r5_are_causal_and_chunk_invariant (float64, untrained, frozen)."""
    widths = ModelConfig(vocab_size=et.VOCAB, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=64,
                         channel="cgc", cgc_bottleneck=8, cgc_channels=24, cgc_units=4, seed=5, dtype="float64")
    sparse = SparseConfig(experts=4, top_k=2, layers=(0, 1), shared=True, dispatch=dispatch)
    ersa = ErsaConfig(widths, ersa_heads=2, key_dim=8, value_dim=8, ersa_kernel="chunked", scan_block=16,
                      conv_kernel=4, chunk=chunk, sparse=sparse)
    model = EvidenceLM(ersa, EvidenceConfig(block=16, capacity=256, per_block=2, key_dim=16, coarse_top=6,
                                            evidence=3, fine_top=4, overlap=4, encoder_dim=32, fine_dim=8,
                                            read_heads=2, read_head_dim=8, query_tile=7)).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _ordered_bits(x: torch.Tensor) -> torch.Tensor:
    bits = x.contiguous().view(torch.int64)
    magnitude = bits & 0x7FFFFFFFFFFFFFFF
    return torch.where(bits < 0, -magnitude, magnitude)       # monotone in the value: differences count ULPs


def _prefix_difference(a: torch.Tensor, b: torch.Tensor, cut: int = CAUSAL_CUT) -> dict:
    a, b = a[:, :cut], b[:, :cut]
    gap = (a - b).abs()
    differing = gap != 0
    row = {"bitwise_equal": bool(torch.equal(a, b)), "max_abs": float(gap.max()),
           "differing_elements": int(differing.sum()), "elements": a.numel()}
    if bool(differing.any()):
        scale = torch.maximum(a.abs(), b.abs())
        row.update(max_abs_over_largest_value=float(gap.max() / scale.max()),
                   max_rel=float((gap / scale.clamp_min(torch.finfo(a.dtype).tiny)).max()),
                   earliest_position=int(differing.any(-1).nonzero()[:, 1].min()),
                   positions_affected=int(differing.any(-1).any(0).sum()),
                   max_ulp=int((_ordered_bits(a) - _ordered_bits(b)).abs().max()))
    return row


def _grouped_on_all_rows(self, flat, index, weight, out):
    """DIAGNOSIS ONLY: the grouped selection and scatter, with every expert's GEMMs run on all rows (a fixed row
    count) and the routed rows selected afterwards."""
    k = index.shape[1]
    assignment = index.reshape(-1)
    order = torch.argsort(assignment, stable=True)
    counts = torch.bincount(assignment, minlength=len(self.experts)).tolist()
    token, scale, start = order // k, weight.reshape(-1)[order], 0
    for expert, count in zip(self.experts, counts):
        if count:
            rows = token[start:start + count]
            out = out.index_add(0, rows, expert(flat)[rows] * scale[start:start + count, None])
        start += count
    return out


@torch.no_grad()
def command_causality(args) -> dict:
    """Diagnosis of the exact-head CI failure at eba8646: two streams that differ only from position 301 on, through
    the R6 test model, compared bit for bit per module and dispatch, against a token-serial causal control, plus the
    row-count dependence of one routed expert's GEMMs and the size of planted leaks. MEASURED in this process's BLAS
    configuration (repeat with MKL_ENABLE_INSTRUCTIONS=AVX2 for the kernel path of CI's runners)."""
    import os
    import types
    stream = et.generate_records(et.RecordTask(queries=6, background=20.0), 2, 600, seed=7).tokens
    changed = stream.clone()
    changed[:, CAUSAL_CUT:] = torch.randint(1, 30, changed[:, CAUSAL_CUT:].shape,
                                            generator=torch.Generator().manual_seed(0))
    model = _causality_model()
    reference = _causality_model(dispatch="reference")
    reference.load_state_dict(model.state_dict())
    out = {"phase": PHASE, "cut": CAUSAL_CUT, "blas": {"MKL_ENABLE_INSTRUCTIONS": os.environ.get(
        "MKL_ENABLE_INSTRUCTIONS"), "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "threads": torch.get_num_threads()}, "test_contract_tolerance": 1e-12}
    captured, routes = {}, {}
    hooks = []
    for i, block in enumerate(model.blocks):
        for name, module in (("ersa", block.ersa), ("routed_channel", block.sparse)):
            hooks.append(module.register_forward_hook(
                lambda m, inputs, output, key=f"block{i}.{name}": captured.setdefault(key, []).append(
                    (output[0] if isinstance(output, tuple) else output).clone())))   # the mixer returns (y, state, ...)
        def route(module, inputs, output, i=i):             # returns None: a hook's return value replaces outputs
            captured.setdefault(f"block{i}.router_weights", []).append(output[3])
            routes.setdefault(i, []).append(output[2])
        hooks.append(block.sparse.router.register_forward_hook(route))
    whole, other = model(stream, model.new_state(2)), model(changed, model.new_state(2))
    for hook in hooks:
        hook.remove()
    batch, length = stream.shape
    out["grouped_logits"] = _prefix_difference(whole, other)
    out["per_module"] = {key: _prefix_difference(*(v.view(batch, length, -1) for v in values))
                         for key, values in captured.items()}
    out["past_routes_identical"] = {f"block{i}": bool(torch.equal(*(v.view(batch, length, -1)[:, :CAUSAL_CUT]
                                                                    for v in values))) for i, values in routes.items()}
    out["reference_logits"] = _prefix_difference(reference(stream, reference.new_state(2)),
                                                 reference(changed, reference.new_state(2)))
    out["grouped_vs_reference_whole_sequence_max_abs"] = float((whole - reference(stream, reference.new_state(2)))
                                                               .abs().max())
    for channel in (b.sparse for b in model.blocks):
        channel._grouped = types.MethodType(_grouped_on_all_rows, channel)
    try:
        out["grouped_gemms_on_all_rows_logits"] = _prefix_difference(model(stream, model.new_state(2)),
                                                                     model(changed, model.new_state(2)))
    finally:
        for channel in (b.sparse for b in model.blocks):
            del channel._grouped
    out["r5_off_logits"] = _prefix_difference(model(stream, model.new_state(2), mode="off"),
                                              model(changed, model.new_state(2), mode="off"))
    model(stream, model.new_state(2))
    sizes = [b.sparse.report["stats"]["counts"] for b in model.blocks]
    model(changed, model.new_state(2))
    out["expert_row_counts"] = {"original": sizes, "changed": [b.sparse.report["stats"]["counts"]
                                                               for b in model.blocks]}
    serial = _causality_model(chunk=1)
    serial.load_state_dict(model.state_dict())
    state = serial.new_state(2)
    steps = torch.cat([serial(stream[:, t:t + 1], state) for t in range(CAUSAL_CUT)], dim=1)
    out["token_serial_control"] = {
        "note": "positions < cut computed one token at a time, before any changed token exists",
        "max_abs_vs_original_whole": float((steps - whole[:, :CAUSAL_CUT]).abs().max()),
        "max_abs_vs_changed_whole": float((steps - other[:, :CAUSAL_CUT]).abs().max())}
    expert = model.blocks[1].sparse.experts[0]
    x = torch.randn(800, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    probes = {"expert": expert, "expert.ffn": expert.ffn, "expert.cgc": expert.cgc,
              "expert.cgc.proj_out": lambda v: expert.cgc.proj_out(v[:, :expert.cgc.proj_out.in_features])}
    out["row_count_dependence"] = {"note": "row counts M in [500, 800] for which rows [0, 500) differ from M = 700"}
    for name, fn in probes.items():
        base = fn(x[:700])[:500]
        changed_rows = [m for m in range(500, 801) if not torch.equal(fn(x[:m])[:500], base)]
        out["row_count_dependence"][name] = {"row_counts_changing_earlier_rows": len(changed_rows), "of": 301,
                                             "max_abs": max([float((fn(x[:m])[:500] - base).abs().max())
                                                             for m in changed_rows] or [0.0])}
    leaks = {}
    for scale in (1e-6, 1e-8, 1e-10):
        for name, net in (("grouped", model), ("reference", reference)):
            for channel in (b.sparse for b in net.blocks):
                plain = type(channel).forward
                channel.forward = types.MethodType(
                    lambda self, u, plain=plain, scale=scale: plain(self, u) + scale * u.mean(1, keepdim=True), channel)
            try:
                leaks[f"{name}_scale_{scale:g}"] = _prefix_difference(net(stream, net.new_state(2)),
                                                                      net(changed, net.new_state(2)))["max_abs"]
            finally:
                for channel in (b.sparse for b in net.blocks):
                    del channel.forward
    out["planted_leak_max_abs"] = leaks
    return out


def command_smoke(args) -> dict:
    ns = argparse.Namespace(**{**vars(args), "smoke": True, "out": None, "no_forward": True})
    return {"scale": sorted(command_scale(ns)["configurations"]), "dispatch": len(command_dispatch(ns)["rows"]),
            "controls": len(command_controls(ns)["rows"]), "cache": len(command_cache(ns)["rows"])}


def _finite(value):
    if isinstance(value, float):      # six significant digits: rounding to decimal places erased small differences
        return float(f"{value:.6g}") if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _write(path: str, record: dict) -> None:
    environment = {"torch": torch.__version__, "python": platform.python_version(), "threads": torch.get_num_threads(),
                   "cuda_available": torch.cuda.is_available()}
    text = json.dumps(_finite(dict(record, environment=environment)), indent=1, sort_keys=True, allow_nan=False)
    with open(path, "w", encoding="utf-8") as handle:           # explicit offline artifact, requested by --out
        handle.write(text + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["scale", "dispatch", "controls", "cache", "causality", "smoke"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-forward", action="store_true")
    args = parser.parse_args(argv)
    args.smoke = args.command == "smoke"
    torch.set_num_threads(args.threads)
    result = {"scale": command_scale, "dispatch": command_dispatch, "controls": command_controls,
              "cache": command_cache, "causality": command_causality, "smoke": command_smoke}[args.command](args)
    if args.out and args.command != "controls":
        _write(args.out, result)
    elif not args.out:
        print(json.dumps(_finite(result), indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
