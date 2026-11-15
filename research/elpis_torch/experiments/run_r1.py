#!/usr/bin/env python3
"""Elpis Torch R1: scrolling-context experiments. Sub-commands, each resumable and each bounded.

    train    train one topology (sparse | local | dense) on the long-stream task and write a checkpoint
    ladder   stream held-out data of 8K -> 32K -> 128K -> 256K -> 1M tokens (and beyond the horizon) through a
             trained checkpoint in fixed-size chunks; report accuracy by probe kind and distance, measured
             memory and measured throughput
    adapt    R0 full-stack fast weights against last-block fast weights on the R0 transduction task, and the
             cost of a last-block update on a long scrolling stream
    capacity print the projected 24 GB placement (formulas only)

    python -m research.elpis_torch.experiments.run_r1 train --topology sparse --steps 4000 --out /tmp/sparse.ckpt
    python -m research.elpis_torch.experiments.run_r1 ladder --ckpt /tmp/sparse.ckpt --out /tmp/sparse.json
    python -m research.elpis_torch.experiments.run_r1 adapt --ckpt /tmp/sparse.ckpt --out /tmp/adapt.json
    python -m research.elpis_torch.experiments.run_r1 smoke        # seconds: every path once, nothing written

Checkpoints and records are EXPLICIT OFFLINE ARTIFACTS, written only to the paths given. No stage builds a
dense attention matrix over a long stream: the dense reference is run only at lengths where it is cheap, through
R0's incremental cache. Wall-clock numbers are PERFORMANCE_ONLY (one machine) and never a pass criterion.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time

import torch

from .. import accounting
from ..config import ModelConfig
from ..data import SINGLE, build_splits, generate as generate_r0, make_world, rewire
from ..model import ElpisTorchLM, parameter_digest
from ..scroll import ScrollConfig, ScrollLM
from ..scroll_data import KINDS, generate, probe_loss, score
from ..scroll_session import ScrollSession
from ..session import SessionConfig, SupportSet, VolatileSession
from ..training import TrainConfig, freeze, train as train_r0
from ..upstream import UPSTREAM_COMMIT

PHASE = "ELPIS_TORCH_SCROLLING_CONTEXT_R1"
MODEL = dict(vocab_size=259, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, channel="cgc", cgc_bottleneck=32,
             cgc_channels=96, cgc_units=4)
SCROLL = dict(window=64, block=16, chunk=48, global_topk=6, fine_blocks=2, sinks=4)
HORIZON = 1 << 20
TRAINING = dict(length=512, events=60, batch=24, lr=3e-3, warmup=100, weight_decay=0.01, grad_clip=1.0)
EDGES = (64, 512, 8192, 32768, 131072, 262144, 1048576)
LADDER = ((8192, 4), (32768, 4), (131072, 4), (262144, 2), (1048576, 2), (1310720, 1))     # (tokens, streams)
EVAL_SEED = 9_000_000          # training draws seeds below 5,000,000


def build(topology: str, *, horizon: int, seed: int, dense_context: int = 512):
    model = ModelConfig(**MODEL, max_seq_len=dense_context, seed=seed)
    if topology == "dense":
        return ElpisTorchLM(model)
    layers = (model.n_layers - 1,) if topology == "sparse" else ()
    return ScrollLM(ScrollConfig(model, horizon=horizon, global_layers=layers, **SCROLL))


def _forward(model, tokens):
    return model.run(tokens)[0] if isinstance(model, ScrollLM) else model(tokens)


def _quick_eval(model, seed: int, length: int, events: int, count: int = 32) -> dict:
    held = generate(count, length, events=events, seed=seed)
    with torch.no_grad():
        logits = _forward(model, held.tokens[:, :-1])
    nll = -torch.log_softmax(logits.double(), -1).gather(-1, held.tokens[:, 1:, None])[..., 0]
    pad = lambda t: torch.cat((t, t[:, :1]), dim=1)   # noqa: E731
    return score(pad(logits.argmax(-1)), pad(nll), held, edges=(SCROLL["window"],))


def command_train(args) -> dict:
    t = dict(TRAINING, **({"length": 256, "events": 28, "batch": 4} if args.smoke else {}))
    horizon = max(1024, t["length"] * 2)
    model = build(args.topology, horizon=horizon, seed=args.seed, dense_context=t["length"])
    named = dict(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in named.values() if p.ndim >= 2], "weight_decay": t["weight_decay"]},
         {"params": [p for p in named.values() if p.ndim < 2], "weight_decay": 0.0}], lr=t["lr"], betas=(0.9, 0.95))
    schedule = TrainConfig(steps=args.steps, lr=t["lr"], warmup_steps=min(t["warmup"], args.steps))
    history, started, status = [], time.perf_counter(), "COMPLETED"
    for step in range(args.steps):
        model.train()
        batch = generate(t["batch"], t["length"], events=t["events"], seed=1000 + args.seed * 1_000_000 + step)
        loss = probe_loss(_forward(model, batch.tokens[:, :-1]), batch.tokens, batch.kind)
        if not torch.isfinite(loss):
            status = "DIVERGED"
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(named.values(), t["grad_clip"])
        for group in optimizer.param_groups:
            group["lr"] = schedule.lr_at(step)
        optimizer.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            model.eval()
            held = _quick_eval(model, 5_000_000 + step, t["length"], t["events"], 8 if args.smoke else 32)
            history.append({"step": step + 1, "seconds": time.perf_counter() - started, "loss": float(loss.detach()),
                            "held_out": {k: {"accuracy": v["accuracy"], "by_distance": v["by_distance"]}
                                         for k, v in held.items()}})
            print(f"[{args.topology}] step {step + 1} loss {float(loss.detach()):.3f} " + " ".join(
                f"{k}={v['accuracy']:.2f}" for k, v in held.items()), file=sys.stderr, flush=True)
            if args.out:
                _save(model, args, history, status, t, started)
    record = _save(model, args, history, status, t, started) if args.out else None
    return {"topology": args.topology, "status": status, "history": history, "checkpoint": record}


def _save(model, args, history, status, training, started) -> dict:
    """EXPLICIT OFFLINE ARTIFACT: the checkpoint requested with --out."""
    record = {"phase": PHASE, "topology": args.topology, "seed": args.seed, "steps": args.steps, "status": status,
              "training": training, "history": history, "train_seconds": time.perf_counter() - started,
              "parameters": sum(p.numel() for p in model.parameters()), "parameter_digest": parameter_digest(model)}
    torch.save({"record": record, "state_dict": {k: v.detach() for k, v in model.state_dict().items()}}, args.out)
    return record


def load(path: str, *, horizon: int, dense_context: int = 512):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    record = saved["record"]
    model = build(record["topology"], horizon=horizon, seed=record["seed"], dense_context=dense_context)
    model.load_state_dict(saved["state_dict"], strict=True)
    return freeze(model), record


@torch.no_grad()
def stream_through(model, streams, *, chunk: int | None = None, diagnose: bool = True) -> dict:
    """Feed whole streams in bounded chunks; keep only argmax, loss and (for the sparse topology) whether the
    block holding each probe's fact was among the blocks the query selected."""
    tokens = streams.tokens
    count, length = tokens.shape
    predicted = torch.zeros(count, length, dtype=torch.long)
    nll = torch.zeros(count, length, dtype=torch.float32)
    scroll = isinstance(model, ScrollLM)
    step = chunk or (model.scroll.chunk if scroll else 256)
    state = model.new_state(count) if scroll else model.new_cache(count)
    sizes = state.nbytes() if scroll else None
    hits = {"selected": 0, "fine": 0, "eligible": 0}
    layer = model.scroll.global_layers[-1] if scroll and model.scroll.global_layers else None
    started = time.perf_counter()
    for at in range(0, length, step):
        piece = tokens[:, at:at + step]
        if scroll:
            logits, chosen = model(piece, state, return_selected=True)
        else:
            logits, chosen = model(piece, state), {}
        predicted[:, at:at + piece.shape[1]] = logits.argmax(-1)
        following = tokens[:, at + 1:at + piece.shape[1] + 1]
        nll[:, at:at + following.shape[1]] = -torch.log_softmax(logits[:, :following.shape[1]], -1).gather(
            -1, following[..., None])[..., 0]
        if diagnose and layer in chosen:
            # A probe at token t is predicted by the query at t - 1; its fact sits `distance` tokens before t.
            kinds = streams.kind[:, at + 1:at + piece.shape[1] + 1]
            rows, cols = torch.nonzero(kinds != 0, as_tuple=True)
            for row, col in zip(rows.tolist(), cols.tolist()):
                target = at + 1 + col
                source = target - int(streams.distance[row, target])
                if source > at + col - model.scroll.window:        # still in the window: not a memory read
                    continue
                picks = chosen[layer][row, col].tolist()
                block = source // model.scroll.block
                hits["eligible"] += 1
                hits["selected"] += block in picks
                hits["fine"] += block in picks[:model.scroll.fine_blocks]
    seconds = time.perf_counter() - started
    out = {"tokens": length, "streams": count, "seconds": seconds, "tokens_per_second": count * length / seconds,
           "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
           "probes": score(predicted, nll, streams, edges=EDGES)}
    if scroll:
        out["state_bytes"] = state.nbytes()
        out["state_bytes_constant"] = state.nbytes() == sizes
        out["state_bytes_formula"] = accounting.scroll_state_bytes(model.scroll, batch=count)
        out["coordinates"] = state.coordinates()
        if hits["eligible"]:
            out["memory_reads"] = {"probes_beyond_window": hits["eligible"],
                                   "fact_block_selected": hits["selected"] / hits["eligible"],
                                   "fact_block_read_as_raw_tokens": hits["fine"] / hits["eligible"]}
    else:
        out["cache_bytes"] = state.nbytes()
    return out


def command_ladder(args) -> dict:
    stages = ((2048, 2), (4096, 1)) if args.smoke else LADDER
    horizon = 2048 if args.smoke else HORIZON
    limit = args.max_tokens or (1 << 62)
    out = {"phase": PHASE, "horizon": horizon, "stages": []}
    for length, count in stages:
        if length > limit:
            continue
        model, record = load(args.ckpt, horizon=horizon, dense_context=length)
        if record["topology"] == "dense" and length > args.dense_limit:
            out["stages"].append({"tokens": length, "status": "NOT_RUN_DENSE_REFERENCE_INFEASIBLE"})
            continue
        streams = generate(count, length, events=128 if not args.smoke else 48, seed=EVAL_SEED + length, span=True)
        result = stream_through(model, streams)
        result["status"] = "MEASURED"
        result["beyond_horizon"] = length > horizon
        out["stages"].append(result)
        summary = " ".join(f"{k}={v['accuracy']:.2f}({v['n']})" for k, v in result["probes"].items())
        print(f"[{record['topology']}] {length} tokens x {count}: {summary} "
              f"{result['tokens_per_second']:.0f} tok/s", file=sys.stderr, flush=True)
        out.update(topology=record["topology"], checkpoint=record)
        if args.out:
            _write(args.out, out)
    return out


def _metrics(logits, tokens, mask) -> dict:
    nll = -torch.log_softmax(logits.double(), -1).gather(-1, tokens[:, 1:, None])[..., 0]
    right = logits.argmax(-1) == tokens[:, 1:]
    return {"loss": float(nll[mask].mean()), "accuracy": float(right[mask].double().mean())}


def command_adapt(args) -> dict:
    """Full-stack (R0) against last-block fast weights: gain, interference, cost, memory, cache validity."""
    from .run_r0 import BASE, VARIANTS
    seeds = [0] if args.smoke else [0, 1, 2]
    steps = 200 if args.smoke else 1500
    world = make_world(10, 5, seed=0)
    splits = build_splits(world, train=4096, val=512, comp=512, length=BASE["max_seq_len"], seed=0)
    out = {"phase": PHASE, "task": "R0 transduction, last operation rewired after training", "runs": []}
    for name in ("B_cgc_bottleneck", "B_cgc_dense"):
        for seed in seeds:
            model = ElpisTorchLM(ModelConfig(**BASE, **VARIANTS[name][0], seed=seed))
            train_r0(model, splits.train, TrainConfig(steps=steps, batch_size=64, lr=3e-3, warmup_steps=50, seed=seed))
            freeze(model)
            shifted = rewire(world, op=world.ops - 1, seed=seed)
            changed = shifted.changed_from(world)
            evidence = generate_r0(shifted, 16, BASE["max_seq_len"], seed=7000 + seed, p_double=0.0)
            held = generate_r0(shifted, 256, BASE["max_seq_len"], seed=8000 + seed, p_double=0.0)
            hit = torch.zeros_like(held.kind, dtype=torch.bool)
            for state, op in changed:
                hit |= (held.state == state) & (held.op == op)
            hit &= held.kind == SINGLE
            masks = {"changed": hit[:, 1:], "unchanged": ((held.kind == SINGLE) & ~hit)[:, 1:]}
            support = SupportSet.observed(evidence.tokens, mask=evidence.target_mask(), source="rewired world")
            stream = evidence.tokens[0].tolist()
            digest = parameter_digest(model)
            for scope in ("all", "last"):
                for lr in (0.05, 0.2, 1.0):
                    config = SessionConfig(lr=lr, max_steps=16, fast_scope=scope)
                    with VolatileSession(model, config) as session:
                        before = {k: _metrics(session.forward(held.tokens[:, :-1]), held.tokens, m)
                                  for k, m in masks.items()}
                        started = time.perf_counter()
                        results = session.adapt(support, steps=16)
                        seconds = time.perf_counter() - started
                        after = {k: _metrics(session.forward(held.tokens[:, :-1]), held.tokens, m)
                                 for k, m in masks.items()}
                        fast = session.volatile_bytes()["fast_weights"]
                        norm = session.delta_norm()
                    with VolatileSession(model, config) as session:        # the stream form: is the cache kept?
                        session.observe(stream[:30])
                        cache = session._cache
                        session.adapt_on_stream(steps=4)
                        logits = session.next_logits()
                        kept = session._cache is cache
                        fresh = session.forward(torch.tensor([stream[:30]]))[0, -1]
                        rebuilds = session.cache_rebuilds
                    out["runs"].append({
                        "variant": name, "seed": seed, "scope": scope, "lr": lr,
                        "applied": sum(r.applied for r in results), "before": before, "after": after,
                        "seconds_per_update": seconds / max(1, len(results)), "fast_weight_bytes": fast,
                        "delta_norm": norm, "stream_cache_kept": kept, "stream_cache_rebuilds": rebuilds,
                        "stream_logits_error_vs_recompute": float((logits - fresh).abs().max()),
                        "base_unchanged": parameter_digest(model) == digest})
            print(f"[adapt] {name} seed {seed} done", file=sys.stderr, flush=True)
    if args.ckpt:
        out["scrolling_stream"] = _scroll_update_cost(args.ckpt, 4096 if args.smoke else 65536)
    return out


def _scroll_update_cost(ckpt: str, length: int) -> dict:
    """A last-block update on a long scrolling stream: what it costs, and that it leaves the memory untouched."""
    model, record = load(ckpt, horizon=max(2048, length))
    if not isinstance(model, ScrollLM):
        return {"status": "NOT_RUN_NOT_A_SCROLLING_CHECKPOINT"}
    stream = generate(1, length, events=128, seed=EVAL_SEED + 77).tokens[0].tolist()
    with ScrollSession(model, SessionConfig(lr=0.1, max_steps=8, fast_scope="last"), learn_window=256) as session:
        started = time.perf_counter()
        session.observe(stream)
        streamed = time.perf_counter() - started
        before = [{k: v.clone() for k, v in layer.items()} for layer in session.state.layers]
        started = time.perf_counter()
        results = session.adapt_on_stream(steps=8)
        updating = time.perf_counter() - started
        same = all(torch.equal(layer[k], kept[k]) for layer, kept in zip(session.state.layers, before) for k in kept)
        bytes_held = session.volatile_bytes()
    return {"status": "MEASURED", "stream_tokens": length, "applied": sum(r.applied for r in results),
            "support_loss": [results[0].loss_before, results[-1].loss_after],
            "seconds_per_update": updating / max(1, len(results)), "seconds_to_stream_once": streamed,
            "full_recompute_would_cost_updates": streamed / (updating / max(1, len(results))),
            "stream_state_bitwise_unchanged": same, "volatile_bytes": bytes_held}


def command_capacity(args) -> dict:
    small = ScrollConfig(ModelConfig(**MODEL, max_seq_len=64), horizon=HORIZON, global_layers=(1,), **SCROLL)
    return {"experiment_model_per_stream": accounting.scroll_state_bytes(small),
            "experiment_model_compute_per_token": accounting.scroll_compute_per_token(small),
            "deployment_fine_tier_on_cpu": accounting.deployment_projection(),
            "deployment_fine_tier_on_gpu": accounting.deployment_projection(fine_on_gpu=True)}


def command_smoke(args) -> dict:
    """Every path once, small, in memory only (a temporary checkpoint is written and removed)."""
    import os
    import tempfile
    out = {}
    with tempfile.TemporaryDirectory() as folder:
        for topology in ("sparse", "local", "dense"):
            path = os.path.join(folder, topology)
            namespace = argparse.Namespace(topology=topology, steps=12, seed=0, eval_every=12, out=path, smoke=True,
                                           ckpt=path, max_tokens=None, dense_limit=4096)
            trained = command_train(namespace)
            ladder = command_ladder(argparse.Namespace(**{**vars(namespace), "out": None}))
            out[topology] = {"train_status": trained["status"], "stages": [s["status"] for s in ladder["stages"]]}
        out["adapt"] = len(command_adapt(argparse.Namespace(smoke=True, ckpt=os.path.join(folder, "sparse")))["runs"])
    out["capacity"] = command_capacity(args)["deployment_fine_tier_on_cpu"]["fits_gpu"]
    return out


def _finite(value):
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _write(path: str, record: dict) -> None:
    record = dict(record, environment={"torch": torch.__version__, "python": platform.python_version(),
                                       "machine": platform.machine(), "threads": torch.get_num_threads(),
                                       "cgc_upstream_commit": UPSTREAM_COMMIT})
    with open(path, "w", encoding="utf-8") as handle:           # explicit offline artifact, requested by --out
        handle.write(json.dumps(_finite(record), indent=1, sort_keys=True, allow_nan=False) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["train", "ladder", "adapt", "capacity", "smoke"])
    parser.add_argument("--topology", choices=["sparse", "local", "dense"], default="sparse")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--dense-limit", type=int, default=32768)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    args.smoke = args.command == "smoke"
    torch.set_num_threads(args.threads)
    result = {"train": command_train, "ladder": command_ladder, "adapt": command_adapt,
              "capacity": command_capacity, "smoke": command_smoke}[args.command](args)
    if args.command in ("adapt",) and args.out:
        _write(args.out, result)
    if args.command in ("capacity", "smoke") or not args.out:
        print(json.dumps(_finite(result if args.command != "train" else {k: v for k, v in result.items()
                                                                          if k != "checkpoint"}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
