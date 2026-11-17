#!/usr/bin/env python3
"""Elpis Torch R4: context-runtime engineering benchmarks. Each sub-command is bounded; files only where --out says.

    bench    paths A/B/C at 4K .. 1M (+ continuation past the 1M horizon): prefill and decode throughput, work
             counters, state bytes and their stability, peak RSS. Each (length, path) runs in its own process.
               A  the R3 reference execution (full score table, re-encoding every read, full per-chunk logits)
               B  the R4 runtime: tiled exact scoring, bounded logits, no cache
               C  the R4 runtime with a bounded encoded-block cache
    parity   A, B and C on one stream: selected block ids and logits at probe positions, compared
    spark    D: a minimal Spark-style 3:1 sliding/full hybrid (untrained, published geometry scaled down), measured
             where feasible, analytical beyond; plus Spark-4B's own KV bytes from its published configuration
    scale    exact tensor accounting of the 500M / 1B configurations' 1M-token context (state INSTANTIATED on
             `meta`, cache configured, scratch PROJECTED) for GPU and CPU placements
    gpu      LIGHTWEIGHT CUDA qualification (R4.1): small model only, FP32 parity (GPU and CPU-resident history, cache
             cold/warm) and BF16 autocast; needs an explicit --device cuda:N; admission-checked; NOT_RUN without CUDA
    gpu-horizon  EXPENSIVE, explicit (--device cuda:N --confirm-full-horizon): small-model phases first, then the
             500M configuration (seed weights) streamed past 2^20 tokens, peak memory above a baseline
    cache-patterns / cache-workloads  R4.1 adversarial block-cache measurements
    smoke    every path once, tiny

Random token streams: these measure the runtime, not retrieval. Nothing here is evidence of 1M recall.
Wall-clock numbers are PERFORMANCE_ONLY (one shared CPU).
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import subprocess
import sys
import time

import torch

from ..context_runtime import BlockCache, ContextSession, RuntimeOptions

PHASE = "ELPIS_TORCH_R4_CONTEXT_RUNTIME"
LENGTHS = (4096, 32768, 131072, 262144, 1 << 20)
PAST_HORIZON = 1 << 16          # extra tokens streamed after 2^20 at the 1M stage: rollover and eviction
CACHE_BYTES = 32 << 20
DECODE_TOKENS = 256


def _model(ckpt: str | None, chunk: int, horizon: int):
    from .run_r3 import build, load
    if ckpt:
        return load(ckpt, chunk=chunk, horizon=horizon)[0]
    model = build("history", "facts", 0, chunk=chunk, horizon=horizon).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _environment() -> dict:
    return {"torch": torch.__version__, "python": platform.python_version(), "machine": platform.machine(),
            "threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available()}


def bench_one(args) -> dict:
    """One (length, path) in this process."""
    torch.manual_seed(0)
    chunk, length, path = args.chunk, args.length, args.path
    model = _model(args.ckpt, chunk, 1 << 20)
    total = length + (PAST_HORIZON if length >= 1 << 20 else 0)
    decode_tokens = DECODE_TOKENS
    tokens = torch.randint(1, model.config.vocab_size, (1, total + decode_tokens),
                           generator=torch.Generator().manual_seed(1))
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = {"path": path, "tokens": total, "chunk": chunk}
    if path == "A":
        state = model.new_state(1)
        state.runtime = RuntimeOptions(scorer="reference")       # counters only; the R3 execution itself
        before, pointers = state.nbytes(), state.tokens.tokens.data_ptr()
        started = time.perf_counter()
        with torch.no_grad():
            for at in range(0, total, chunk):
                logits = model(tokens[:, at:at + chunk], state)  # R3: [B, chunk, vocab] every chunk
                logits.argmax(-1)
        prefill = time.perf_counter() - started
        times = []
        with torch.no_grad():
            for i in range(decode_tokens):
                started = time.perf_counter()
                model(tokens[:, total + i:total + i + 1], state)
                times.append(time.perf_counter() - started)
        counters, after = dict(state.runtime.counters), state.nbytes()
        stable = state.tokens.tokens.data_ptr() == pointers
        position, oldest = state.position, state.tokens.oldest
        out["logits_bytes_per_chunk"] = chunk * model.config.vocab_size * 4
    else:
        session = ContextSession(model, batch=1, chunk=chunk, scorer="tiled", tile=args.tile,
                                 cache_bytes=CACHE_BYTES if path == "C" else 0)
        before, pointers = session.nbytes(), session.state.tokens.tokens.data_ptr()
        started = time.perf_counter()
        session.prefill(tokens[:, :total])
        prefill = time.perf_counter() - started
        mark = dict(session.counters())
        times = []
        for i in range(decode_tokens):
            started = time.perf_counter()
            session.step(tokens[:, total + i])
            times.append(time.perf_counter() - started)
        counters = session.counters()
        counters = {k: v for k, v in counters.items()}
        out["decode_counters"] = {k: counters.get(k, 0) - mark.get(k, 0) for k in counters
                                  if isinstance(counters.get(k), int)}
        after = session.nbytes()
        stable = session.state.tokens.tokens.data_ptr() == pointers
        position, oldest = session.position, session.state.tokens.oldest
        out["logits_bytes_per_chunk"] = model.config.vocab_size * 4
        out["projected_bytes"] = session.projected_bytes()
        session.close()
    times.sort()
    decode = sum(times) / len(times)
    out.update({"prefill_seconds": round(prefill, 2), "prefill_tokens_per_second": round(total / prefill),
                "decode_ms_per_token": round(decode * 1000, 3), "decode_tokens_per_second": round(1 / decode, 1),
                "decode_ms_median": round(times[len(times) // 2] * 1000, 3),
                "decode_ms_p90": round(times[int(len(times) * 0.9)] * 1000, 3), "decode_tokens": decode_tokens,
                "counters": counters, "state_bytes_before": before, "state_bytes_after": after,
                "state_bytes_constant": before["total"] == after["total"] if path != "C" else
                {k: v for k, v in after.items() if k not in ("block_cache", "total")} ==
                {k: v for k, v in before.items() if k not in ("block_cache", "total")},
                "buffers_not_reallocated": stable, "final_position": position, "oldest_token_held": oldest,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "rss_growth_bytes": (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0) * 1024})
    if counters.get("records_scored"):
        out["records_scored_per_prefill_token"] = round(counters["records_scored"] / max(1, total), 1)
    return out


def command_bench(args) -> dict:
    out = {"phase": PHASE, "runs": [], "environment": _environment(), "note": "random tokens: runtime only"}
    for length in args.lengths:
        for path in args.paths:
            cmd = [sys.executable, "-B", "-m", "research.elpis_torch.experiments.run_r4", "bench-one",
                   "--length", str(length), "--path", path, "--chunk", str(args.chunk), "--tile", str(args.tile),
                   "--threads", str(args.threads)] + (["--ckpt", args.ckpt] if args.ckpt else [])
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            if run.returncode:
                record = {"path": path, "tokens": length, "status": "FAILED", "stderr": run.stderr[-2000:]}
            else:
                record = json.loads(run.stdout)
                record["status"] = "MEASURED"
            out["runs"].append(record)
            print(f"[{path} {length}] {record.get('prefill_tokens_per_second')} tok/s prefill, "
                  f"{record.get('decode_ms_median')} ms/token decode (median), peak RSS "
                  f"{record.get('peak_rss_bytes', 0) >> 20} MiB {record['status']}", file=sys.stderr, flush=True)
            if args.out:
                _write(args.out, out)
    return out


@torch.no_grad()
def command_parity(args) -> dict:
    """A, B, C on one stream: identical selections and logits (float32 here, the checkpoint dtype)."""
    model = _model(args.ckpt, args.chunk, 1 << 20)
    length = args.length
    tokens = torch.randint(1, model.config.vocab_size, (1, length), generator=torch.Generator().manual_seed(2))
    probes = sorted({0, length // 3, length // 2, length - 1} | set(range(length - 8, length)))
    results = {}
    for path in ("A", "B", "C"):
        state = model.new_state(1)
        state.runtime = None if path == "A" else RuntimeOptions(
            scorer="tiled", tile=args.tile, cache=BlockCache(CACHE_BYTES) if path == "C" else None)
        logits, picks = {}, []
        for at in range(0, length, args.chunk):
            out, diag = model(tokens[:, at:at + args.chunk], state, diagnostics=True)
            picks.append(diag[model.read_layers[-1]]["selected"])
            for p in probes:
                if at <= p < at + args.chunk:
                    logits[p] = out[:, p - at]
        results[path] = (logits, torch.cat(picks, 1))
    ref_logits, ref_picks = results["A"]
    out = {"phase": PHASE, "tokens": length, "chunk": args.chunk, "tile": args.tile}
    for path in ("B", "C"):
        logits, picks = results[path]
        out[path] = {"selected_blocks_identical": bool(torch.equal(picks, ref_picks)),
                     "selections_compared": int(picks.numel()),
                     "max_abs_logit_difference": max(float((logits[p] - ref_logits[p]).abs().max()) for p in probes),
                     "probes": len(probes)}
    return out


def command_spark(args) -> dict:
    """D: the Spark-style hybrid, measured at feasible lengths; Spark-4B's KV from its published geometry."""
    from ..spark_reference import SPARK_4B, SparkStyleConfig, SparkStyleLM, spark_kv_bytes
    out = {"phase": PHASE, "spark_4b_published_geometry": SPARK_4B, "spark_4b_kv": {}, "measured": [],
           "source": "https://huggingface.co/XHToken/Spark-X2.5-4B (config.json); X2.6-4B: same architecture"}
    for tokens in LENGTHS:
        out["spark_4b_kv"][tokens] = spark_kv_bytes(tokens=tokens)
    for length in args.spark_lengths:
        c = SparkStyleConfig(capacity=length + 64)
        model = SparkStyleLM(c).eval()
        tokens = torch.randint(0, c.vocab, (1, length + 64), generator=torch.Generator().manual_seed(3))
        cache = model.new_cache(1)
        started = time.perf_counter()
        for at in range(0, length, 512):
            model(tokens[:, at:at + 512], cache, logit_positions=torch.tensor([0]))
        prefill = time.perf_counter() - started
        started = time.perf_counter()
        for i in range(32):
            model(tokens[:, length + i:length + i + 1], cache)
        decode = (time.perf_counter() - started) / 32
        kv = cache.nbytes()
        held = spark_kv_bytes(tokens=length + 32, layers_full=c.layer_types.count("full"),
                              layers_sliding=c.layer_types.count("sliding"), kv_heads=c.kv_heads,
                              head_dim=c.head_dim, window=c.window, scalar_bytes=4)
        out["measured"].append({"tokens": length, "geometry": {"layers": c.layers, "hidden": c.hidden,
                                                               "query_heads": c.query_heads, "kv_heads": c.kv_heads,
                                                               "head_dim": c.head_dim, "window": c.window},
                                "prefill_tokens_per_second": round(length / prefill),
                                "decode_ms_per_token": round(decode * 1000, 3),
                                "kv_allocated_bytes": kv, "kv_live_bytes_formula": held,
                                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                "status": "MEASURED"})
        print(f"[spark-style {length}] {length / prefill:.0f} tok/s prefill {decode * 1000:.2f} ms/token",
              file=sys.stderr, flush=True)
    for length in LENGTHS:
        if length not in args.spark_lengths:
            out["measured"].append({"tokens": length, "status": "NOT_RUN (analytical only)"})
    return out


def command_gradients(args) -> dict:
    """P4: chunked training with and without key replay against the unchunked gradient (float64, small model).
    Reports the largest parameter-gradient difference per replay horizon and the replay work and activations."""
    from ..config import ModelConfig
    from ..ersa_model import ErsaConfig
    from ..history_model import ErsaHistoryLM, HistoryConfig
    widths = dict(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64, channel="cgc",
                  cgc_bottleneck=8, cgc_channels=16, dtype="float64", seed=1)
    config = ErsaConfig(ModelConfig(**widths), ersa_heads=2, key_dim=8, value_dim=8, conv_kernel=3, chunk=512,
                        ersa_kernel="chunked")
    model = ErsaHistoryLM(config, HistoryConfig(block=8, horizon=512, topk=2, index_dim=8, read_head_dim=8,
                                                overlap=3)).train()
    tokens = torch.randint(0, 64, (2, 480), generator=torch.Generator().manual_seed(0))
    weights = torch.randn(480, 64, generator=torch.Generator().manual_seed(5), dtype=torch.float64)

    def gradients(step, replay):
        model.zero_grad(set_to_none=True)
        state = model.new_state(2)
        state.runtime = RuntimeOptions(key_grad_blocks=replay)
        loss = 0.0
        for at in range(0, tokens.shape[1], step):
            logits = model(tokens[:, at:at + step], state)
            loss = loss + (torch.log_softmax(logits, -1) * weights[at:at + step]).sum()
        loss.backward()
        return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}, state.runtime.counters

    whole, _ = gradients(480, 0)
    scale = max(float(g.abs().max()) for g in whole.values())
    out = {"phase": PHASE, "tokens": 480, "chunk": 60, "block": 8, "reference": "one 480-token chunk",
           "largest_gradient_entry": scale, "runs": []}
    for replay in (0, 2, 8, 32, -1):
        grads, counters = gradients(60, replay)
        encoder = max(float((grads[n] - whole[n]).abs().max()) for n in whole if n.startswith(("encoder.", "readers.")))
        everything = max(float((grads[n] - whole[n]).abs().max()) for n in whole)
        replayed = counters.get("replayed_key_blocks", 0)
        out["runs"].append({"key_grad_blocks": replay, "max_gradient_difference_all": everything,
                            "max_gradient_difference_encoder_and_readers": encoder,
                            "replayed_key_blocks": replayed,
                            "replay_activation_scalars_projected": replayed * model.history.span
                            * model.encoder.width * 8})
    return out


def command_cache_patterns(args) -> dict:
    """P3 (R4.1): the cache's own behaviour under synthetic access sequences over a 1M horizon (32,768 blocks),
    with entries the size of the 500M configuration's encoded block (65 x 256 bf16 = 33,280 bytes)."""
    import random
    entry = torch.zeros(1, 65, 256, dtype=torch.bfloat16)
    size = entry.numel() * entry.element_size()
    blocks, accesses = 1 << 15, 20_000
    rng = random.Random(0)
    patterns = {
        "uniform_random": [rng.randrange(blocks) for _ in range(accesses)],
        "alternating_oldest_newest": [0 if i % 2 else blocks - 1 for i in range(accesses)],
        "cyclic_scan_2x_capacity": [i % (2 * (CACHE_BYTES // size)) for i in range(accesses)],
        "hot_set_10_at_90pct": [rng.randrange(10) if rng.random() < 0.9 else rng.randrange(blocks)
                                for _ in range(accesses)],
        "cold_never_repeated": list(range(accesses)),
    }
    out = {"phase": PHASE, "entry_bytes": size, "horizon_blocks": blocks, "accesses": accesses, "runs": []}
    for capacity in (CACHE_BYTES, 64 * size, size - 1):
        for name, sequence in patterns.items():
            cache = BlockCache(capacity)
            started = time.perf_counter_ns()
            for block in sequence:
                if cache.get(block) is None:
                    cache.put(block, entry)
            elapsed = time.perf_counter_ns() - started
            stats = cache.stats
            out["runs"].append({"pattern": name, "capacity_bytes": capacity, "entries_capacity": capacity // size,
                                "hit_ratio": stats["hits"] / accesses, "misses": stats["misses"],
                                "evictions": stats["evictions"], "rejected": stats["rejected_too_large"],
                                "bytes_held": cache.bytes, "within_bound": cache.bytes <= capacity,
                                "management_ns_per_access": stats["management_ns"] / accesses,
                                "total_ns_per_access": elapsed / accesses})
    return out


def _crafted_stream(pattern: str, history: int, questions: int, seed: int = 0):
    """A token stream of the R3 facts vocabulary whose DECODE part steers the model's own selections: 32 keys bound
    once each, evenly over the history; then question/answer pairs chosen by ``pattern``. Teacher-forced."""
    from .. import longrange_tasks as F
    g = torch.Generator().manual_seed(seed)
    if pattern == "identical_content":
        unit = torch.randint(F.FILLER0, F.FILLER0 + F.FILLERS, (32,), generator=g)
        tokens = unit.repeat(history // 32 + 1)[:history].clone()       # every block: the same 32 tokens
    else:
        tokens = torch.randint(F.FILLER0, F.FILLER0 + F.FILLERS, (history,), generator=g)
    values = torch.randint(0, F.VALUES, (32,), generator=g)
    for key in range(32):
        at = 64 + key * (history - 128) // 32
        tokens[at], tokens[at + 1] = F.KEY0 + key, F.VAL0 + int(values[key])
    order = {"filler": None, "uniform": torch.randint(0, 32, (questions,), generator=g).tolist(),
             "alternating": [0 if i % 2 == 0 else 31 for i in range(questions)],
             "hot": [5] * questions, "cyclic_scan": [i % 32 for i in range(questions)],
             "identical_content": torch.randint(0, 32, (questions,), generator=g).tolist()}[pattern]
    if order is None:
        tail = torch.randint(F.FILLER0, F.FILLER0 + F.FILLERS, (2 * questions,), generator=g)
    else:
        tail = torch.tensor([t for k in order for t in (F.ASK0 + k, F.VAL0 + int(values[k]))])
    return torch.cat((tokens, tail))[None], history


def command_cache_workloads(args) -> dict:
    """P3 (R4.1): decode latency, encodings and cache behaviour of the trained model on crafted workloads, without
    a cache (B) and with caches of 32 MiB, 7 entries and less than one entry (C). Logits of C are compared with B
    at every decoded position."""
    model = _model(args.ckpt, args.chunk, 1 << 20)
    entry = (model.history.block + 1) * model.encoder.width * 4
    capacities = tuple(args.capacities) if args.capacities else (0, CACHE_BYTES, 7 * entry, entry - 1)
    patterns = args.patterns or ("filler", "uniform", "alternating", "hot", "cyclic_scan", "identical_content")
    out = {"phase": PHASE, "history_tokens": args.length, "decoded_tokens": 256, "entry_bytes": entry, "runs": []}
    for pattern in patterns:
        tokens, history = _crafted_stream(pattern, args.length, 128)
        reference = None
        for capacity in capacities:
            session = ContextSession(model, chunk=args.chunk, cache_bytes=capacity)
            started = time.perf_counter()
            session.prefill(tokens[:, :history])
            prefill = time.perf_counter() - started
            mark = dict(session.counters())
            times, logits = [], []
            for position in range(history, tokens.shape[1]):
                started = time.perf_counter()
                logits.append(session.step(tokens[:, position]))
                times.append(time.perf_counter() - started)
            decoded = torch.cat(logits)
            counters = session.counters()
            delta = {k: counters.get(k, 0) - mark.get(k, 0) for k in counters if isinstance(counters.get(k), int)}
            times.sort()
            record = {"pattern": pattern, "cache_bytes_capacity": capacity,
                      "prefill_tokens_per_second": round(history / prefill),
                      "decode_ms_median": round(times[len(times) // 2] * 1000, 3),
                      "decode_ms_p90": round(times[int(len(times) * 0.9)] * 1000, 3),
                      "decode_ms_max": round(times[-1] * 1000, 3),
                      "decode_blocks_encoded": delta.get("blocks_reencoded", 0),
                      "decode_cache_hits": delta.get("cache_hits", 0), "decode_cache_misses": delta.get("cache_misses", 0),
                      "decode_cache_evictions": delta.get("cache_evictions", 0),
                      "decode_cache_management_ms": round(delta.get("cache_management_ns", 0) / 1e6, 3),
                      "cache_bytes_held": counters.get("cache_bytes", 0)}
            accesses = record["decode_cache_hits"] + record["decode_cache_misses"]
            record["decode_hit_ratio"] = record["decode_cache_hits"] / accesses if accesses else None
            if reference is None:
                reference = decoded
            else:
                record["max_abs_logit_difference_vs_uncached"] = float((decoded - reference).abs().max())
            out["runs"].append(record)
            print(f"[{pattern} cache {capacity}] decode median {record['decode_ms_median']} ms p90 "
                  f"{record['decode_ms_p90']} encoded {record['decode_blocks_encoded']} hit "
                  f"{record['decode_hit_ratio']}", file=sys.stderr, flush=True)
            session.close()
    return out


def command_scale(args) -> dict:
    from .. import scaling
    out = {"phase": PHASE, "configurations": {}}
    for name in scaling.CONFIGURATIONS:
        out["configurations"][name] = scaling.r4_context_budget(name, chunk=args.chunk, tile=args.tile,
                                                                cache_bytes=CACHE_BYTES)
    return out


def _cuda_device(args):
    """Explicit device selection only: --device cuda:N is required; nothing is chosen implicitly."""
    if not torch.cuda.is_available():
        return None, {"phase": PHASE, "status": "NOT_RUN", "reason": "torch.cuda.is_available() is False",
                      "environment": _environment()}
    if not args.device or not args.device.startswith("cuda"):
        raise SystemExit("GPU commands need an explicit --device cuda:N (no default device is used)")
    device = torch.device(args.device)
    if device.index is None or device.index >= torch.cuda.device_count():
        raise SystemExit(f"--device {args.device}: no such CUDA device")
    return device, None


def _admit(device, need_bytes: int, label: str) -> dict:
    """Resource admission BEFORE allocating: free device memory (as the driver reports it, other processes
    included) must exceed ``need_bytes`` plus a 512 MiB margin. Nothing else on the device is touched."""
    free, total = torch.cuda.mem_get_info(device)
    record = {"phase_label": label, "free_bytes": free, "total_bytes": total, "required_bytes": need_bytes,
              "margin_bytes": 512 << 20}
    record["admitted"] = free >= need_bytes + (512 << 20)
    return record


def _release() -> None:
    """After the caller dropped its references: collect and return cached blocks to the driver."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _small_parity(args, device) -> dict:
    """Lightweight CUDA phases on the small experiment model: FP32 parity against CPU (GPU-resident and CPU-resident
    token history, cache cold and warm), then BF16 autocast finiteness and agreement. Admission-checked; cleans up."""
    out = {"model": "R3 experiment model (history, ~173k parameters)", "phases": []}
    small = _model(args.ckpt, 256, 4096)
    tokens = torch.randint(1, small.config.vocab_size, (1, 8192), generator=torch.Generator().manual_seed(4))
    probes = range(8000, 8192)
    reference = ContextSession(small, chunk=256).prefill(tokens, probes=probes)
    admission = _admit(device, 256 << 20, "small model")
    out["admission"] = admission
    if not admission["admitted"]:
        out["status"] = "REFUSED_ADMISSION"
        return out
    gpu = None
    try:
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        gpu = _model(args.ckpt, 256, 4096).to(device)
        for label, kwargs in (("FP32 gpu-history", {}), ("FP32 cpu-history", {"history_device": "cpu"}),
                              ("FP32 cache", {"cache_bytes": CACHE_BYTES})):
            torch.cuda.reset_peak_memory_stats(device)
            with ContextSession(gpu, chunk=256, **kwargs) as session:
                got = session.prefill(tokens.to(device), probes=probes)
                if "cache" in label:                       # warm pass over the same stream
                    session.reset()
                    got = session.prefill(tokens.to(device), probes=probes)
                torch.cuda.synchronize(device)
                worst = max(float((got.probes[p].float().cpu() - reference.probes[p]).abs().max()) for p in probes)
                out["phases"].append({"phase": label, "max_abs_logit_difference": worst,
                                      "status": "PASS" if worst <= 1e-3 else "NUMERICAL_FAILURE",
                                      "peak_allocated_above_baseline": torch.cuda.max_memory_allocated(device)
                                      - baseline, "counters": session.counters()})
        with torch.autocast("cuda", dtype=torch.bfloat16), ContextSession(gpu, chunk=256) as session:
            low = session.prefill(tokens.to(device)).last_logits.float().cpu()
            agree = float((low.argmax(-1) == reference.last_logits.argmax(-1)).double().mean())
            out["phases"].append({"phase": "BF16 autocast", "finite": bool(torch.isfinite(low).all()),
                                  "ersa_state_dtype": str(session.state.cells[0].dtype),
                                  "argmax_agreement_with_fp32_cpu": agree,
                                  "status": "PASS" if bool(torch.isfinite(low).all()) else "NUMERICAL_FAILURE"})
        out["status"] = "PASS" if all(p["status"] == "PASS" for p in out["phases"]) else "NUMERICAL_FAILURE"
    except torch.cuda.OutOfMemoryError as error:
        out["status"], out["error"] = "ALLOCATION_FAILURE", str(error)[:500]
    finally:
        gpu = None                                  # noqa: F841  (drop the reference before releasing)
        _release()
    return out


def command_gpu(args) -> dict:
    """LIGHTWEIGHT CUDA qualification (small model only; seconds; < 256 MiB). NOT_RUN without CUDA."""
    device, refused = _cuda_device(args)
    if refused:
        return refused
    return {"phase": PHASE, "command": "gpu", "device": str(device), "name": torch.cuda.get_device_name(device),
            "environment": _environment(), "small_model": _small_parity(args, device)}


def command_gpu_horizon(args) -> dict:
    """EXPENSIVE, explicit only (--confirm-full-horizon): the 500M configuration (seed weights: memory and numerics,
    not a trained model) streamed past 2^20 tokens with measured peak memory relative to a baseline, decode
    latency, budget refusal. Runs the small-model phases first and stops if they fail."""
    device, refused = _cuda_device(args)
    if refused:
        return refused
    if not args.confirm_full_horizon:
        raise SystemExit("gpu-horizon streams a 500M model through more than 2^20 tokens; rerun with "
                         "--confirm-full-horizon to proceed")
    from .. import scaling
    from ..context_runtime import BudgetError
    from ..history_model import ErsaHistoryLM
    out = {"phase": PHASE, "command": "gpu-horizon", "device": str(device), "name": torch.cuda.get_device_name(device),
           "environment": _environment()}
    out["small_model"] = _small_parity(args, device)
    if out["small_model"].get("status") != "PASS":
        out["status"] = "STOPPED_AFTER_SMALL_MODEL"
        return out
    name, tokens_total = args.config, args.tokens
    budget = scaling.r4_context_budget(name, chunk=512, cache_bytes=CACHE_BYTES)
    weights = budget["weights_bytes"]["fp32"]
    need = weights + budget["placement_all_gpu"]["gpu"]["total"] + (weights // 2)     # + autocast weight copies
    admission = _admit(device, need, name)
    out["admission"] = admission
    if not admission["admitted"]:
        out["status"] = "REFUSED_ADMISSION"
        return out
    big = session = None
    try:
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        ersa, history = scaling.configuration(name)
        big = ErsaHistoryLM(ersa, history).to(device).eval()
        for p in big.parameters():
            p.requires_grad_(False)
        torch.cuda.synchronize(device)
        weights_bytes = torch.cuda.memory_allocated(device) - baseline
        try:
            ContextSession(big, chunk=512, cache_bytes=CACHE_BYTES, budget_bytes=1 << 20)
            out["budget_refusal"] = "FAIL (not refused)"
        except BudgetError:
            out["budget_refusal"] = "PASS"
        torch.cuda.reset_peak_memory_stats(device)
        after_weights = torch.cuda.memory_allocated(device)
        peaks, piece = {}, torch.randint(0, 1000, (1, 512), device=device)
        marks = {1 << 16, 1 << 18, 1 << 19, 1 << 20, tokens_total}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            session = ContextSession(big, chunk=512, cache_bytes=CACHE_BYTES, budget_bytes=2 << 30)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            for step in range(tokens_total // 512):
                session.prefill(piece, keep_last=(step + 1) * 512 == tokens_total)
                if (step + 1) * 512 in marks:
                    torch.cuda.synchronize(device)
                    peaks[(step + 1) * 512] = {"allocated": torch.cuda.max_memory_allocated(device) - after_weights,
                                               "reserved": torch.cuda.max_memory_reserved(device) - after_weights}
            torch.cuda.synchronize(device)
            prefill = time.perf_counter() - started
            times = []
            for _ in range(64):
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                session.step(piece[:, 0])
                torch.cuda.synchronize(device)
                times.append(time.perf_counter() - started)
        times.sort()
        last = peaks[max(peaks)]["allocated"]
        out[name] = {"weights_bytes_measured": weights_bytes, "context_peak_above_weights": peaks,
                     "state": session.nbytes(), "prefill_tokens_per_second": tokens_total / prefill,
                     "decode_ms_median": times[32] * 1000, "decode_ms_p90": times[57] * 1000,
                     "flat_after_2_pow_18": max(v["allocated"] for k, v in peaks.items() if k >= 1 << 18)
                     <= 1.05 * peaks[1 << 18]["allocated"],
                     "within_1GiB": last <= 1 << 30, "within_2GiB": last <= 2 << 30,
                     "note": "seed weights; peak includes autocast weight casts (not context)"}
        out["status"] = "MEASURED"
    except torch.cuda.OutOfMemoryError as error:
        out["status"], out["error"] = "ALLOCATION_FAILURE", str(error)[:500]
    finally:
        if session is not None:
            session.close()
        session = big = None                        # noqa: F841
        _release()
    return out


def command_smoke(args) -> dict:
    args.lengths, args.paths, args.length = [2048], ["A", "B", "C"], 2048
    bench = command_bench(args)
    parity = command_parity(args)
    args.spark_lengths = [1024]
    spark = command_spark(args)
    return {"bench": [r["status"] for r in bench["runs"]], "parity": {k: parity[k]["selected_blocks_identical"]
                                                                       for k in ("B", "C")},
            "spark": spark["measured"][0]["status"], "scale": sorted(command_scale(args)["configurations"]),
            "gpu": (command_gpu(args)["status"] if args.device or not torch.cuda.is_available()
                    else "SKIPPED (smoke never starts CUDA work without an explicit --device)")}


def _finite(value):
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _write(path: str, record: dict) -> None:
    text = json.dumps(_finite(record), sort_keys=True, separators=(",", ":"), allow_nan=False)
    with open(path, "w", encoding="utf-8") as handle:           # explicit offline artifact, requested by --out
        handle.write(text + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["bench", "bench-one", "parity", "spark", "scale", "gradients", "gpu",
                                            "gpu-horizon", "cache-patterns", "cache-workloads", "smoke"])
    parser.add_argument("--device", default=None, help="GPU commands: an explicit cuda:N")
    parser.add_argument("--capacities", type=int, nargs="+", default=None, help="cache-workloads: cache bytes")
    parser.add_argument("--patterns", nargs="+", default=None, help="cache-workloads: workload names")
    parser.add_argument("--confirm-full-horizon", action="store_true", help="gpu-horizon: run the 500M stream")
    parser.add_argument("--config", default="r3_500m", choices=["r3_500m", "r3_1b"])
    parser.add_argument("--tokens", type=int, default=(1 << 20) + PAST_HORIZON)
    parser.add_argument("--ckpt", default=None, help="an R3 history checkpoint (default: seed weights)")
    parser.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    parser.add_argument("--spark-lengths", type=int, nargs="+", default=[4096, 32768, 131072])
    parser.add_argument("--paths", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--path", default="B", choices=["A", "B", "C"])
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--tile", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.command == "bench-one":
        print(json.dumps(_finite(bench_one(args))))
        return 0
    if args.command == "smoke":
        args.timeout, args.out = 600, None
    result = {"bench": command_bench, "parity": command_parity, "spark": command_spark, "scale": command_scale,
              "gradients": command_gradients, "gpu": command_gpu, "gpu-horizon": command_gpu_horizon,
              "cache-patterns": command_cache_patterns,
              "cache-workloads": command_cache_workloads, "smoke": command_smoke}[args.command](args)
    if args.out and args.command != "bench":
        _write(args.out, result)
    elif not args.out:
        print(json.dumps(_finite(result), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
