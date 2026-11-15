#!/usr/bin/env python3
"""Elpis Torch R2: ERSA experiments. Each sub-command is bounded and writes only where ``--out`` says.

    transduction  R0's sequential-dependency task: the attention-based CGC baseline against ERSA with no
                  attention, with a short convolution, with a 16-token window, and with two grouped-query heads
    exact         the exact policy at length: a cell written once, 1,048,576 further instructions through a
                  compiled mixer's forward pass, then read back; plus the recall-capacity curve of the
                  learning policy
    latency       incremental decoding rate and context-state bytes after a long prefix, per architecture
    projection    the projected deployment context budget (formulas only)
    smoke         every path once, small

The long-range retrieval comparison uses R1's driver unchanged, with the ERSA variants added to it:
``run_r1 train --topology ersa_conv ...`` and ``run_r1 ladder --ckpt ...``.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time

import torch

from .. import accounting
from ..config import ModelConfig
from ..data import DOUBLE, HELDOUT, KIND_NAMES, SINGLE, build_splits, make_world
from ..ersa import cell_code, compile_cell_mixer, recall_capacity
from ..ersa_model import ErsaConfig, ErsaLM
from ..model import ElpisTorchLM, next_token_loss
from ..scroll import ScrollConfig, ScrollLM
from ..training import TrainConfig, evaluate, freeze, rollout, train
from .run_r0 import BASE, VARIANTS

PHASE = "ELPIS_TORCH_ERSA_R2"
CHANNEL = VARIANTS["B_cgc_bottleneck"][0]            # R0's exactly parameter-matched CGC channel
ERSA = dict(ersa_heads=4, key_dim=16, value_dim=16)
ARCHITECTURES = {
    "attention_baseline": None,                                                  # R0 B: 4 heads, 2 kv, dense
    "ersa": dict(),
    "ersa_conv": dict(conv_kernel=4),
    "ersa_conv_window": dict(conv_kernel=4, attention="window", attn_window=16, chunk=15),
    "ersa_conv_gqa": dict(conv_kernel=4, attention="window", attn_window=64, chunk=48),      # the whole 48-token context
}


def build(name: str, seed: int):
    widths = ModelConfig(**BASE, **CHANNEL, seed=seed)
    if ARCHITECTURES[name] is None:
        return ElpisTorchLM(widths)
    return ErsaLM(ErsaConfig(widths, **ERSA, **ARCHITECTURES[name]))


@torch.no_grad()
def evaluate_stream_model(model: ErsaLM, batch) -> dict:
    """R0's ``evaluate`` for a model that runs through a state."""
    tokens, kind = batch.tokens, batch.kind[:, 1:]
    logits = model.run(tokens[:, :-1])[0]
    nll = -torch.log_softmax(logits.double(), -1).gather(-1, tokens[:, 1:, None])[..., 0]
    right = logits.argmax(-1) == tokens[:, 1:]
    out = {"loss": float(nll.mean())}
    for code in (SINGLE, DOUBLE, HELDOUT):
        mask = kind == code
        if int(mask.sum()):
            out[KIND_NAMES[code]] = {"loss": float(nll[mask].mean()), "accuracy": float(right[mask].double().mean()),
                                     "targets": int(mask.sum())}
    return out


@torch.no_grad()
def rollout_stream_model(model: ErsaLM, batch) -> dict:
    """R0's free-running rollout: every state after '=' is the model's own token, fed back through the state."""
    tokens, is_state = batch.tokens, batch.kind != 0
    state = model.new_state(tokens.shape[0])
    current, right = tokens[:, :1], torch.zeros(tokens.shape, dtype=torch.bool)
    for position in range(tokens.shape[1] - 1):
        predicted = model(current, state)[:, -1].argmax(-1)
        target = is_state[:, position + 1]
        right[:, position + 1] = (predicted == tokens[:, position + 1]) & target
        current = torch.where(target, predicted, tokens[:, position + 1])[:, None]
    return {"state_accuracy": int(right.sum()) / int(is_state.sum()),
            "exact_trajectories": float((right | ~is_state).all(dim=1).double().mean())}


def train_stream_model(model: ErsaLM, data, config: TrainConfig, val) -> dict:
    """R0's training loop (same optimizer, schedule, clipping, batches) for a model that runs through a state."""
    named = dict(model.named_parameters())
    optimizer = torch.optim.AdamW([{"params": [p for p in named.values() if p.ndim >= 2],
                                    "weight_decay": config.weight_decay},
                                   {"params": [p for p in named.values() if p.ndim < 2], "weight_decay": 0.0}],
                                  lr=config.lr, betas=(0.9, 0.95))
    sampler, history, status, started = torch.Generator().manual_seed(config.seed), [], "COMPLETED", time.perf_counter()
    first = last = float("nan")
    for step in range(config.steps):
        model.train()
        tokens = data.tokens[torch.randint(len(data), (config.batch_size,), generator=sampler)]
        loss = next_token_loss(model.run(tokens[:, :-1])[0], tokens)
        if not torch.isfinite(loss):
            status = "DIVERGED"
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(bool(torch.isfinite(p.grad).all()) for p in named.values()):
            status = "DIVERGED"
            break
        torch.nn.utils.clip_grad_norm_(named.values(), config.grad_clip)
        for group in optimizer.param_groups:
            group["lr"] = config.lr_at(step)
        optimizer.step()
        last = float(loss.detach())
        first = last if step == 0 else first
        if (step + 1) % config.eval_every == 0:
            model.eval()
            history.append({"step": step + 1, "val": evaluate_stream_model(model, val)})
    model.eval()
    return {"status": status, "first_loss": first, "final_loss": last, "history": history,
            "seconds": time.perf_counter() - started}


def command_transduction(args) -> dict:
    seeds, steps = ([0], 150) if args.smoke else (args.seeds, args.steps)
    world = make_world(10, 5, seed=0)
    splits = build_splits(world, train=4096, val=512, comp=512, length=BASE["max_seq_len"], seed=0)
    runs = []
    for name in args.architectures:
        for seed in seeds:
            model = build(name, seed)
            config = TrainConfig(steps=steps, batch_size=64, lr=3e-3, warmup_steps=50, weight_decay=0.01,
                                 grad_clip=1.0, seed=seed, eval_every=100)
            if isinstance(model, ElpisTorchLM):
                report = train(model, splits.train, config, val=splits.val)
                result = {"status": report.status, "first_loss": report.first_loss, "final_loss": report.final_loss,
                          "history": report.history, "seconds": report.seconds}
                freeze(model)
                val, comp, free = evaluate(model, splits.val), evaluate(model, splits.comp), rollout(
                    model, splits.val.select(slice(0, 256)))
                context = {"kv_cache_at_full_context": accounting.memory_estimate(
                    model.config, batch=1, length=BASE["max_seq_len"])["kv_cache_bytes_at_full_context"]}
            else:
                result = train_stream_model(model, splits.train, config, splits.val)
                freeze(model)
                val, comp, free = evaluate_stream_model(model, splits.val), evaluate_stream_model(
                    model, splits.comp), rollout_stream_model(model, splits.val.select(slice(0, 256)))
                context = model.new_state(1).nbytes()
            reached = next((h["step"] for h in result["history"]
                            if h["val"].get("double", {}).get("accuracy", 0) >= 0.99), None)
            runs.append({"architecture": name, "seed": seed, "status": result["status"],
                         "parameters": sum(p.numel() for p in model.parameters()),
                         "train_loss": [result["first_loss"], result["final_loss"]], "val": val,
                         "compositional_heldout_accuracy": comp["heldout"]["accuracy"], "rollout": free,
                         "steps_to_val_double_accuracy_0.99": reached, "train_seconds": result["seconds"],
                         "train_steps_per_second": steps / result["seconds"], "context_state_bytes": context,
                         "history": [[h["step"], round(h["val"]["loss"], 4), round(h["val"]["single"]["accuracy"], 3),
                                      round(h["val"]["double"]["accuracy"], 3)] for h in result["history"]]})
            print(f"[{name} seed {seed}] {result['status']} val {val['loss']:.3f} single "
                  f"{val['single']['accuracy']:.2f} double {val['double']['accuracy']:.2f} "
                  f"({result['seconds']:.0f}s)", file=sys.stderr, flush=True)
    return {"phase": PHASE, "task": "R0 finite-state transduction", "steps": steps, "seeds": seeds, "runs": runs}


def command_exact(args) -> dict:
    """The exact policy over a long stream, and the learning policy's recall capacity."""
    total = 4096 if args.smoke else 1 << 20
    cells, width = 64, 1
    mixer = compile_cell_mixer(cells, width, dtype=torch.float32)
    state = mixer.new_state(1)
    marker = cell_code(cells, width, address=7, value=[0.123456789], store=True, dtype=torch.float32)
    _, state, _ = mixer(marker[None, None], state)
    kept = state[0, 0, :, 7].clone()
    generator, started, size = torch.Generator().manual_seed(0), time.perf_counter(), mixer.state_bytes(1)
    block = 4096
    for at in range(0, total, block):
        # Traffic on every other cell: stores and adds with random values, and reads.
        address = torch.randint(cells - 1, (block,), generator=generator)
        address = address + (address >= 7).long()
        code = torch.zeros(block, cells + width + 4)
        code[torch.arange(block), address] = 1.0
        code[:, cells] = torch.randint(-8, 9, (block,), generator=generator).float()
        kind = torch.randint(3, (block,), generator=generator)
        code[kind == 0, cells + width + 0] = 1.0            # STORE
        code[kind == 1, cells + width + 1] = 1.0            # ADD
        _, state, _ = mixer(code[None], state)
    seconds = time.perf_counter() - started
    read, _, _ = mixer(cell_code(cells, width, address=7, dtype=torch.float32)[None, None], state)
    return {"phase": PHASE,
            "exact_policy": {"instructions_after_the_write": total, "cells": cells,
                             "marker_recovered_bit_for_bit": bool(torch.equal(read[0, 0, cells:cells + width], kept)),
                             "state_finite": bool(torch.isfinite(state).all()), "state_bytes": size,
                             "state_bytes_after": mixer.state_bytes(1), "seconds": seconds,
                             "instructions_per_second": total / seconds},
            "learning_policy_recall_capacity": [recall_capacity(k, 16, int(load * k), seed=0)
                                                for k in (16, 32, 64) for load in (0.25, 0.5, 1.0, 2.0, 4.0)]}


@torch.no_grad()
def command_latency(args) -> dict:
    """Incremental decoding after a prefix: tokens per second and bytes of context state, per architecture."""
    prefix, measured = (256, 64) if args.smoke else (8192, 512)
    widths = ModelConfig(**{**BASE, "max_seq_len": prefix + measured + 1}, **CHANNEL, seed=0)
    tokens = torch.randint(0, 259, (1, prefix + measured), generator=torch.Generator().manual_seed(0))
    models = {"attention_baseline (dense KV cache)": ElpisTorchLM(widths),
              "r1_local_window_64": ScrollLM(ScrollConfig(widths, window=64, block=16, chunk=48, horizon=1 << 20)),
              **{name: ErsaLM(ErsaConfig(widths, **ERSA, **options)) for name, options in ARCHITECTURES.items()
                 if options is not None}}
    out = {}
    for name, model in models.items():
        model.eval()
        if isinstance(model, ElpisTorchLM):
            state = model.new_cache(1)
            for at in range(0, prefix, 256):
                model(tokens[:, at:at + 256], state)
            step = lambda t: model(t, state)       # noqa: E731
            size = lambda: state.nbytes()          # noqa: E731
        else:
            state = model.new_state(1)
            model.run(tokens[:, :prefix], state)
            step = lambda t: model(t, state)       # noqa: E731
            size = lambda: state.nbytes()["total"]  # noqa: E731
        before = size()
        started = time.perf_counter()
        for at in range(prefix, prefix + measured):
            step(tokens[:, at:at + 1])
        seconds = time.perf_counter() - started
        out[name] = {"prefix_tokens": prefix, "tokens_per_second": measured / seconds,
                     "context_bytes_before": before, "context_bytes_after": size(),
                     "context_grows_with_stream": size() > before}
    return {"phase": PHASE, "architectures": out}


def command_projection(args) -> dict:
    return {"ersa_only": accounting.ersa_context_projection(attn_window=0),
            "ersa_with_4096_token_window": accounting.ersa_context_projection(),
            "ersa_with_window_and_compressed_index": accounting.ersa_context_projection(index_layers=6),
            "r1_hybrid_for_comparison": {k: accounting.deployment_projection(fine_on_gpu=True)["gpu_resident"][k]
                                         for k in ("window", "compressed", "fine")}}


def command_smoke(args) -> dict:
    args.architectures = list(ARCHITECTURES)
    transduction = command_transduction(args)
    return {"transduction": [(r["architecture"], r["status"]) for r in transduction["runs"]],
            "exact": command_exact(args)["exact_policy"]["marker_recovered_bit_for_bit"],
            "latency": sorted(command_latency(args)["architectures"]),
            "projection_within_target": command_projection(args)["ersa_with_4096_token_window"]["within_target"]}


def _finite(value):
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["transduction", "exact", "latency", "projection", "smoke"])
    parser.add_argument("--architectures", nargs="+", default=list(ARCHITECTURES), choices=list(ARCHITECTURES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    args.smoke = args.command == "smoke"
    torch.set_num_threads(args.threads)
    result = {"transduction": command_transduction, "exact": command_exact, "latency": command_latency,
              "projection": command_projection, "smoke": command_smoke}[args.command](args)
    result = dict(result, environment={"torch": torch.__version__, "python": platform.python_version(),
                                       "machine": platform.machine(), "threads": torch.get_num_threads()})
    text = json.dumps(_finite(result), indent=1, sort_keys=True, allow_nan=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:       # explicit offline artifact, requested by --out
            handle.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
