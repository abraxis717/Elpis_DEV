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
    gpu      the CUDA qualification (NOT_RUN without CUDA)
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


def command_scale(args) -> dict:
    from .. import scaling
    out = {"phase": PHASE, "configurations": {}}
    for name in scaling.CONFIGURATIONS:
        out["configurations"][name] = scaling.r4_context_budget(name, chunk=args.chunk, tile=args.tile,
                                                                cache_bytes=CACHE_BYTES)
    return out


def command_gpu(args) -> dict:
    """CUDA qualification: parity (fp32), bf16 autocast, history on CPU, cache cold/warm, measured peak memory of
    the 500M configuration (seed weights) streamed past 2^20 tokens, and budget refusal. NOT_RUN without CUDA."""
    if not torch.cuda.is_available():
        return {"phase": PHASE, "status": "NOT_RUN", "reason": "torch.cuda.is_available() is False",
                "environment": _environment()}
    from .. import scaling
    from ..context_runtime import BudgetError
    from ..history_model import ErsaHistoryLM
    out = {"phase": PHASE, "status": "MEASURED", "device": torch.cuda.get_device_name(0), "environment": _environment()}
    small = _model(args.ckpt, 256, 4096)
    tokens = torch.randint(1, small.config.vocab_size, (1, 8192), generator=torch.Generator().manual_seed(4))
    reference = ContextSession(small, chunk=256).prefill(tokens, probes=range(8000, 8192))
    gpu = _model(args.ckpt, 256, 4096).cuda()
    for label, kwargs in (("fp32", {}), ("fp32_history_cpu", {"history_device": "cpu"}),
                          ("fp32_cache", {"cache_bytes": CACHE_BYTES})):
        with ContextSession(gpu, chunk=256, **kwargs) as session:
            got = session.prefill(tokens.cuda(), probes=range(8000, 8192))
            out[label] = {"max_abs_logit_difference": max(float((got.probes[p].cpu() - reference.probes[p]).abs().max())
                                                          for p in range(8000, 8192)),
                          "counters": session.counters()}
    with torch.autocast("cuda", dtype=torch.bfloat16), ContextSession(gpu, chunk=256) as session:
        low = session.prefill(tokens.cuda()).last_logits.float().cpu()
        out["bf16_autocast"] = {"finite": bool(torch.isfinite(low).all()),
                                "ersa_state_dtype": str(session.state.cells[0].dtype)}
    del gpu
    torch.cuda.empty_cache()
    ersa, history = scaling.configuration("r3_500m")
    big = ErsaHistoryLM(ersa, history).cuda().eval()
    for p in big.parameters():
        p.requires_grad_(False)
    weights = torch.cuda.memory_allocated()
    try:
        ContextSession(big, chunk=512, cache_bytes=CACHE_BYTES, budget_bytes=1 << 20)
        out["budget_refusal"] = "FAIL (not refused)"
    except BudgetError:
        out["budget_refusal"] = "PASS"
    torch.cuda.reset_peak_memory_stats()
    peaks, piece = {}, torch.randint(0, 1000, (1, 512), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16), ContextSession(big, chunk=512, cache_bytes=CACHE_BYTES,
                                                                      budget_bytes=2 << 30) as session:
        started = time.perf_counter()
        for step in range(((1 << 20) + PAST_HORIZON) // 512):
            session.prefill(piece)
            if (step + 1) * 512 in (1 << 16, 1 << 18, 1 << 19, 1 << 20, (1 << 20) + PAST_HORIZON):
                torch.cuda.synchronize()
                peaks[(step + 1) * 512] = torch.cuda.max_memory_allocated() - weights
        prefill = time.perf_counter() - started
        started = time.perf_counter()
        for _ in range(64):
            session.step(piece[:, 0])
        torch.cuda.synchronize()
        out["r3_500m"] = {"weights_bytes": weights, "context_peak_bytes": peaks, "state": session.nbytes(),
                          "prefill_tokens_per_second": ((1 << 20) + PAST_HORIZON) / prefill,
                          "decode_ms_per_token": (time.perf_counter() - started) / 64 * 1000,
                          "flat_after_2_pow_18": max(v for k, v in peaks.items() if k >= 1 << 18) <= 1.05 * peaks[1 << 18],
                          "within_1GiB": peaks[max(peaks)] <= 1 << 30, "within_2GiB": peaks[max(peaks)] <= 2 << 30,
                          "note": "seed weights: memory and numerics qualification only"}
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
            "gpu": command_gpu(args)["status"]}


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
                                            "smoke"])
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
              "gradients": command_gradients, "gpu": command_gpu, "smoke": command_smoke}[args.command](args)
    if args.out and args.command != "bench":
        _write(args.out, result)
    elif not args.out:
        print(json.dumps(_finite(result), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
