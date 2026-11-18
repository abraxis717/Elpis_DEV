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


def command_smoke(args) -> dict:
    ns = argparse.Namespace(**{**vars(args), "smoke": True, "out": None, "no_forward": True})
    return {"scale": sorted(command_scale(ns)["configurations"]), "dispatch": len(command_dispatch(ns)["rows"]),
            "controls": len(command_controls(ns)["rows"]), "cache": len(command_cache(ns)["rows"])}


def _finite(value):
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
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
    parser.add_argument("command", choices=["scale", "dispatch", "controls", "cache", "smoke"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-forward", action="store_true")
    args = parser.parse_args(argv)
    args.smoke = args.command == "smoke"
    torch.set_num_threads(args.threads)
    result = {"scale": command_scale, "dispatch": command_dispatch, "controls": command_controls,
              "cache": command_cache, "smoke": command_smoke}[args.command](args)
    if args.out and args.command != "controls":
        _write(args.out, result)
    elif not args.out:
        print(json.dumps(_finite(result), indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
