#!/usr/bin/env python3
"""Elpis Torch R3: long-context addressing. Sub-commands, each bounded; files are written only where --out says.

    train     train one architecture on one task (facts | program) for a fixed budget; write a checkpoint
    facts     evaluate a facts checkpoint: length ladder, fact-density sweep, planted begin/middle/end facts, in
              every mode the checkpoint supports (recurrent-only; learned; oracle-address diagnosis; history off),
              with addressing (recall@K) and readout (evidence-token hit) measured separately
    program   evaluate a program checkpoint: teacher-forced determined-token accuracy and FREE-RUNNING generation
              of thousands of tokens, checked by the exact program checker
    scale     P4: instantiated parameter counts of the 500M / 1B configurations (meta device: no memory), and
              the context-memory accounting at 1,048,576 tokens (projected) beside measured small-scale bytes
    smoke     every path once, tiny, nothing kept

    python -m research.elpis_torch.experiments.run_r3 train --arch history --task facts --seed 0 --out /tmp/h0.pt
    python -m research.elpis_torch.experiments.run_r3 facts --ckpt /tmp/h0.pt --out /tmp/h0_facts.json

Labels from the task generator are used for scoring, for the TRAINING-ONLY addressing loss of ``history`` and
for the oracle-address DIAGNOSIS modes. In ``learned`` evaluation no label reaches the model.
Checkpoints and records are EXPLICIT OFFLINE ARTIFACTS. Wall-clock numbers are PERFORMANCE_ONLY.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import sys
import time

import torch

from .. import longrange_tasks as facts_task
from .. import program_tasks as program_task
from ..config import ModelConfig
from ..ersa_model import ErsaConfig, ErsaLM
from ..history_model import ErsaHistoryLM, HistoryConfig, addressing_loss
from ..model import parameter_digest, sample_next
from ..training import TrainConfig

PHASE = "ELPIS_TORCH_R3_LONG_CONTEXT_ADDRESSING"
CGC = dict(channel="cgc", cgc_bottleneck=32, cgc_channels=96, cgc_units=4)
FFN = dict(channel="ffn", ffn_hidden=107)              # 20,544 channel parameters against CGC's 20,480
WIDTHS = dict(d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)
ERSA = dict(ersa_heads=4, key_dim=16, value_dim=16, ersa_kernel="chunked", scan_block=64)
HISTORY = dict(block=32, topk=2, index_dim=32, per_block=2, read_heads=2, read_kv_heads=1, read_head_dim=32,
               overlap=8, layers=(-1,))
LOCAL = dict(attention="window", attn_heads=2, attn_kv_heads=1, attn_layers=(1,))
ARCHS = {   # name: (channel, ERSA options, history?, how the history path is trained)
    # "learned": learned selection + the training-only addressing loss; "learned-noaux": no addressing loss;
    # "oracle": the fixture's source block is read (diagnosis), no addressing loss; "oracle+aux": both
    "ersa": (CGC, dict(), False, None),
    "ersa_conv": (CGC, dict(conv_kernel=4), False, None),
    "ersa_conv_ffn": (FFN, dict(conv_kernel=4), False, None),
    "ersa_conv_local128": (CGC, dict(conv_kernel=4, attn_window=128, **LOCAL), False, None),
    "ersa_conv_local512": (CGC, dict(conv_kernel=4, attn_window=512, **LOCAL), False, None),
    "ersa_conv_local1025": (CGC, dict(conv_kernel=4, attn_window=1025, **LOCAL), False, None),   # covers training
    "history": (CGC, dict(conv_kernel=4), True, "learned"),
    "history_oracle": (CGC, dict(conv_kernel=4), True, "oracle"),
    "history_oracle_aux": (CGC, dict(conv_kernel=4), True, "oracle+aux"),      # oracle reads + addressing loss
    "history_noaux": (CGC, dict(conv_kernel=4), True, "learned-noaux"),         # end-to-end gradient only
    "history_ffn": (FFN, dict(conv_kernel=4), True, "learned"),
    "history_local128": (CGC, dict(conv_kernel=4, attn_window=128, **LOCAL), True, "learned"),
}
TASKS = {
    "facts": dict(length=1024, batch=16, task=facts_task.FactTask(keys=32, events=112)),
    "program": dict(length=1792, batch=12, task=program_task.ProgramTask()),
}
OPTIM = dict(lr=3e-3, warmup=100, weight_decay=0.01, grad_clip=1.0, aux_weight=1.0)
TRAIN_SEED, VAL_SEED, TEST_SEED = 0, 600_000_000, 700_000_000      # disjoint seed domains
LADDER = ((1024, 16), (4096, 8), (16384, 4), (65536, 2), (262144, 2), (1048576, 1))
DENSITY = (8, 16, 32, 48, 96)
PLANTED = (0.05, 0.10, 0.45, 0.50, 0.85, 0.90)
EVAL_EVENTS = 112


def vocab(task: str) -> int:
    return facts_task.VOCAB if task == "facts" else program_task.VOCAB


def build(arch: str, task: str, seed: int, *, chunk: int | None = None, horizon: int | None = None,
          length: int | None = None):
    channel, ersa, history, _ = ARCHS[arch]
    length = length or TASKS[task]["length"]
    window = ersa.get("attn_window")
    chunk = chunk or (window - 1 if window else length)
    if window:
        chunk = min(chunk, window - 1)
    widths = ModelConfig(vocab_size=vocab(task), **WIDTHS, **channel, seed=seed)
    config = ErsaConfig(widths, **ERSA, **ersa, chunk=chunk)
    if not history:
        return ErsaLM(config)
    return ErsaHistoryLM(config, HistoryConfig(horizon=horizon or 4 * length, **HISTORY))


def eligible_address(blocks: torch.Tensor, model) -> torch.Tensor:
    """Oracle/label address of each query: its source block if that block is complete at the query and inside
    the horizon, else -1. ``blocks [B, T]`` from ``query_block``; positions are 0-based in the stream."""
    h = model.history
    position = torch.arange(blocks.shape[1])[None]
    complete = (blocks + 1) * h.block - 1 <= position
    inside = blocks >= (position + 1) // h.block - h.slots
    return torch.where((blocks >= 0) & complete & inside, blocks, torch.full_like(blocks, -1))


def _batch(task: str, seed: int, count: int, length: int, keys: int | None = None):
    spec = TASKS[task]
    if task == "facts":
        fact = facts_task.FactTask(keys=keys or spec["task"].keys, events=length * 7 // 64)   # 112 per 1,024 tokens
        return facts_task.generate(fact, count, length, seed=seed)
    return program_task.generate(spec["task"], count, seed=seed, length=length)


def _loss(arch, task, model, batch, *, diagnostics_needed: bool):
    tokens = batch.tokens[:, :-1]
    state = model.new_state(tokens.shape[0])
    mode = ARCHS[arch][3]
    extra = {}
    if mode in ("oracle", "oracle+aux"):
        extra = dict(mode="oracle", address=eligible_address(batch.query_block(model.history.block)[:, :-1], model))
    if mode is not None:
        extra["diagnostics"] = True
    supervise = mode in ("learned", "oracle+aux") and task == "facts"
    gold = eligible_address(batch.query_block(model.history.block)[:, :-1], model) if supervise else None
    logits, aux = [], []
    step = model.ersa.chunk
    for at in range(0, tokens.shape[1], step):
        piece = {k: (v[:, at:at + step] if k == "address" else v) for k, v in extra.items()}
        out = model(tokens[:, at:at + step], state, **piece)
        if isinstance(out, tuple):
            out, diagnostics = out
            if supervise:             # TRAINING-ONLY labels: the generator's source block of each probe query
                aux.append(addressing_loss(diagnostics, gold[:, at:at + step]))
        logits.append(out)
    logits = torch.cat(logits, dim=1)
    if task == "facts":
        loss = facts_task.probe_loss(logits, batch.tokens, batch.kind)
    else:
        loss = program_task.program_loss(logits, batch.tokens, batch.role)
    aux = torch.stack(aux).mean() if aux else None
    if aux is not None:
        loss = loss + OPTIM["aux_weight"] * aux
    return loss, aux


def command_train(args) -> dict:
    spec = TASKS[args.task]
    length, batch_size = (256, 4) if args.smoke else (args.length or spec["length"], args.batch or spec["batch"])
    model = build(args.arch, args.task, args.seed, length=length)
    named = dict(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in named.values() if p.ndim >= 2], "weight_decay": OPTIM["weight_decay"]},
         {"params": [p for p in named.values() if p.ndim < 2], "weight_decay": 0.0}], lr=OPTIM["lr"], betas=(0.9, 0.95))
    schedule = TrainConfig(steps=args.steps, lr=OPTIM["lr"], warmup_steps=min(OPTIM["warmup"], args.steps))
    history, started, status = [], time.perf_counter(), "COMPLETED"
    for step in range(args.steps):
        model.train()
        data = _batch(args.task, TRAIN_SEED + args.seed * 10_000_000 + step, batch_size, length, args.keys)
        loss, aux = _loss(args.arch, args.task, model, data, diagnostics_needed=True)
        if not torch.isfinite(loss):
            status = "DIVERGED"
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(named.values(), OPTIM["grad_clip"])
        for group in optimizer.param_groups:
            group["lr"] = schedule.lr_at(step)
        optimizer.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            model.eval()
            val = _validate(args.arch, args.task, model, length, 4 if args.smoke else 16, args.keys)
            history.append({"step": step + 1, "seconds": round(time.perf_counter() - started, 1),
                            "loss": float(loss.detach()), "aux": None if aux is None else float(aux.detach()),
                            "val": val})
            print(f"[{args.arch}/{args.task} s{args.seed}] step {step + 1} loss {float(loss.detach()):.3f} "
                  + json.dumps(val), file=sys.stderr, flush=True)
            if args.out:
                _save(model, args, history, status, started, length, batch_size)
    record = _save(model, args, history, status, started, length, batch_size) if args.out else None
    return {"arch": args.arch, "task": args.task, "status": status, "history": history, "checkpoint": record}


@torch.no_grad()
def _validate(arch, task, model, length, count, keys=None) -> dict:
    data = _batch(task, VAL_SEED, count, length, keys)
    kind = ARCHS[arch][3]
    modes = [None] if kind is None else ["oracle"] if kind.startswith("oracle") else ["learned"]
    out = {}
    for mode in modes:
        run = stream(model, data, mode=mode, chunk=model.ersa.chunk)
        if task == "facts":
            out[mode or "recurrent"] = {k: round(v["accuracy"], 3) for k, v in run["probes"].items()}
            if "recall_at_k" in run:
                out["recall_at_k"] = run["recall_at_k"]
        else:
            out[mode or "recurrent"] = {k: round(v["accuracy"], 3) for k, v in run["teacher_forced"].items()}
    return out


def _save(model, args, history, status, started, length, batch_size) -> dict:
    """EXPLICIT OFFLINE ARTIFACT: the checkpoint requested with --out."""
    record = {"phase": PHASE, "arch": args.arch, "task": args.task, "seed": args.seed, "steps": args.steps,
              "status": status, "history": history, "train_seconds": round(time.perf_counter() - started, 1),
              "optim": OPTIM, "train_length": length, "train_batch": batch_size,
              "train_keys": args.keys or (TASKS[args.task]["task"].keys if args.task == "facts" else None),
              "parameters": sum(p.numel() for p in model.parameters()), "parameter_digest": parameter_digest(model)}
    torch.save({"record": record, "state_dict": {k: v.detach() for k, v in model.state_dict().items()}}, args.out)
    return record


def load(path: str, *, chunk: int | None = None, horizon: int | None = None):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    record = saved["record"]
    model = build(record["arch"], record["task"], record["seed"], chunk=chunk, horizon=horizon,
                  length=record.get("train_length"))
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, record


@torch.no_grad()
def stream(model, data, *, mode=None, chunk=256) -> dict:
    """Feed whole streams chunk by chunk through the model's bounded state; score probes; keep diagnostics."""
    tokens = data.tokens
    count, length = tokens.shape
    history = isinstance(model, ErsaHistoryLM)
    predicted = torch.zeros(count, length, dtype=torch.long)
    selected_hit = torch.zeros(count, length, dtype=torch.bool)
    read_hit = torch.zeros(count, length, dtype=torch.bool)
    null_weight = torch.zeros(count, length)
    state = model.new_state(count)
    sizes = state.nbytes()
    address = None
    if history and data.__class__.__name__ == "FactStreams":
        address = eligible_address(data.query_block(model.history.block), model)
        source = data.query_source()
    started = time.perf_counter()
    for at in range(0, length, chunk):
        piece = tokens[:, at:at + chunk]
        if history:
            extra = {"mode": mode or "learned", "diagnostics": address is not None}
            if mode in ("oracle", "oracle_override"):
                extra["address"] = address[:, at:at + chunk]
            out = model(piece, state, **extra)
            logits, diag = out if isinstance(out, tuple) else (out, None)
            if diag:
                record = diag[model.read_layers[-1]]
                sel, gold = record["selected"], address[:, at:at + chunk]
                selected_hit[:, at:at + piece.shape[1]] = (sel == gold[..., None]).any(-1) & (gold >= 0)
                read_hit[:, at:at + piece.shape[1]] = record["read_position"] == source[:, at:at + chunk]
                null_weight[:, at:at + piece.shape[1]] = record["null_weight"].float()
        else:
            logits = model(piece, state)
        predicted[:, at:at + piece.shape[1]] = logits.argmax(-1)
    seconds = time.perf_counter() - started
    out = {"tokens": length, "streams": count, "seconds": round(seconds, 2),
           "tokens_per_second": round(count * length / seconds), "state_bytes": state.nbytes(),
           "state_bytes_constant": state.nbytes() == sizes,
           "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if data.__class__.__name__ == "FactStreams":
        extra = None
        if address is not None:
            has_gold = address >= 0
            extra = {"source_block_eligible": has_gold}
            if mode != "off":
                extra.update(source_block_selected=selected_hit, evidence_token_read=read_hit & has_gold)
        out["probes"] = facts_task.score(predicted, data, extra=extra)
        if address is not None and mode != "off":
            probe_query = torch.zeros_like(has_gold)
            probe_query[:, :-1] = data.kind[:, 1:] != 0
            usable = probe_query & has_gold
            if int(usable.sum()):
                out["recall_at_k"] = round(float(selected_hit[usable].double().mean()), 4)
                out["evidence_token_read"] = round(float(read_hit[usable].double().mean()), 4)
                out["null_weight_at_probes"] = round(float(null_weight[usable].double().mean()), 4)
                out["null_weight_elsewhere"] = round(float(null_weight[~probe_query].double().mean()), 4)
    else:
        out["teacher_forced"] = program_task.teacher_forced(predicted, data)
    return out


def _modes(record) -> list:
    if not ARCHS[record["arch"]][2]:
        return [None]
    if ARCHS[record["arch"]][3].startswith("oracle"):
        return ["oracle", "learned", "off"]
    return ["learned", "oracle_override", "off"]


def command_facts(args) -> dict:
    model, record = load(args.ckpt, chunk=args.chunk, horizon=1 << 20)
    out = {"phase": PHASE, "checkpoint": record, "ladder": [], "density": [], "chunk": model.ersa.chunk}
    spec = TASKS["facts"]["task"]
    ladder = ((512, 2), (1024, 2)) if args.smoke else LADDER
    for length, count in ladder:
        if length > args.max_tokens:
            continue
        task = facts_task.FactTask(keys=spec.keys, events=min(EVAL_EVENTS, length * 7 // 64), planted=PLANTED)
        data = facts_task.generate(task, count if not args.streams else min(count, args.streams), length,
                                   seed=TEST_SEED + length)
        for mode in _modes(record):
            if mode in ("oracle_override", "off") and length > args.diagnosis_max:
                continue
            result = stream(model, data, mode=mode, chunk=model.ersa.chunk)
            result["mode"] = mode or "recurrent"
            out["ladder"].append(result)
            print(f"[{record['arch']} s{record['seed']}] {length} {result['mode']}: " + " ".join(
                f"{k}={v['accuracy']:.2f}" for k, v in result["probes"].items())
                + f" recall@K={result.get('recall_at_k')} read={result.get('evidence_token_read')} "
                  f"{result['tokens_per_second']} tok/s", file=sys.stderr, flush=True)
            if args.out:
                _write(args.out, out)
    out["distractors"] = []
    for length in (() if args.smoke else (4096, 16384)):
        task = facts_task.FactTask(keys=spec.keys, events=min(EVAL_EVENTS * 2, length * 7 // 64), p_bind=0.25,
                                   p_distract=0.30, distractor_keys=32)
        data = facts_task.generate(task, 8 if length == 4096 else 4, length, seed=TEST_SEED + 11 + length)
        for mode in _modes(record)[:1]:
            result = stream(model, data, mode=mode, chunk=model.ersa.chunk)
            out["distractors"].append({"tokens": length, "mode": mode or "recurrent", "p_distract": 0.30,
                                       "probes": {k: {"n": v["n"], "accuracy": v["accuracy"]}
                                                  for k, v in result["probes"].items()},
                                       "recall_at_k": result.get("recall_at_k")})
    for keys in (() if args.smoke else DENSITY):
        task = facts_task.FactTask(keys=keys, events=EVAL_EVENTS)
        data = facts_task.generate(task, 16, 1024, seed=TEST_SEED + 7 + keys)
        for mode in _modes(record)[:1]:
            result = stream(model, data, mode=mode, chunk=model.ersa.chunk)
            out["density"].append({"keys": keys, "mode": mode or "recurrent",
                                   "probes": {k: {"n": v["n"], "accuracy": v["accuracy"]}
                                              for k, v in result["probes"].items()},
                                   "recall_at_k": result.get("recall_at_k")})
        if args.out:
            _write(args.out, out)
    return out


@torch.no_grad()
def free_run(model, prompt: torch.Tensor, new_tokens: int, *, temperature: float, seed: int, chunk: int):
    """Feed the prompt, then generate ``new_tokens`` one at a time through the bounded state (learned mode)."""
    state = model.new_state(prompt.shape[0])
    generator = torch.Generator().manual_seed(seed)
    logits = None
    for at in range(0, prompt.shape[1], chunk):
        logits = model(prompt[:, at:at + chunk], state)
    out = []
    for _ in range(new_tokens):
        token = sample_next(logits[:, -1], temperature=temperature, top_k=None, generator=generator)
        out.append(token)
        logits = model(token[:, None], state)
    return torch.stack(out, dim=1), state


def command_program(args) -> dict:
    model, record = load(args.ckpt, chunk=args.chunk, horizon=1 << 20)
    spec = TASKS["program"]
    out = {"phase": PHASE, "checkpoint": record, "teacher_forced": [], "free_running": []}
    count = 2 if args.smoke else 8
    reference = program_task.generate(spec["task"], count * 2, seed=TEST_SEED + 1, length=spec["length"])
    out["teacher_forced"].append({"header_tokens": spec["task"].header_tokens,
                                  **stream(model, reference, chunk=model.ersa.chunk)})
    headers = (1024,) if args.smoke else (1024, 4096)
    generated = 128 if args.smoke else args.generate
    for header in headers:
        task = program_task.ProgramTask(header_tokens=header)
        progs = program_task.generate(task, count, seed=TEST_SEED + 2 + header, length=None)
        cut = min(progs.header)
        prompt = progs.tokens[:, :cut]
        for temperature in (0.0, 0.7):
            started = time.perf_counter()
            body, state = free_run(model, prompt, generated, temperature=temperature, seed=TEST_SEED,
                                   chunk=model.ersa.chunk)
            checks = [program_task.check(prompt[row].tolist(), body[row].tolist()) for row in range(count)]
            summary = {key: (sum(c[key] for c in checks) / count if checks[0][key] is not None and
                             not isinstance(checks[0][key], bool) else None)
                       for key in checks[0] if key != "first_error_token"}
            summary["first_error_token_median"] = sorted(
                (c["first_error_token"] if c["first_error_token"] is not None else generated) for c in checks)[count // 2]
            entry = {"header_tokens": header, "prompt_tokens": cut, "generated_tokens": generated,
                     "temperature": temperature, "streams": count, "mean": summary,
                     "seconds": round(time.perf_counter() - started, 1), "state_bytes": state.nbytes()}
            out["free_running"].append(entry)
            print(f"[{record['arch']} s{record['seed']}] header {header} T={temperature}: valid "
                  f"{summary['valid_statement_fraction']:.3f} arity {summary['arity_accuracy']:.3f} print "
                  f"{summary['print_accuracy']:.3f} viol/1k {summary['violations_per_1k_tokens']:.1f}",
                  file=sys.stderr, flush=True)
            if args.out:
                _write(args.out, out)
    return out


def command_gpu(args) -> dict:
    """Workstation qualification (RTX PRO 4000 Blackwell, 24 GB). Without CUDA everything is NOT_RUN.

    1. CPU/CUDA parity of the R3 model (float32; trained checkpoint if --ckpt, else seed weights): logits and
       block selections over a stream longer than the history horizon, chunked.
    2. bfloat16 autocast on CUDA: finite logits, ERSA state kept in float32, agreement with float32.
    3. Memory at length: the 500M configuration (seed weights: NOT a trained model) streamed to 1,048,576 tokens
       with torch.cuda.max_memory_allocated recorded at 2^16 .. 2^20 tokens. Pass: flat from 2^18 to 2^20, and the
       context part within the 1 GiB target (2 GiB ceiling).
    """
    if not torch.cuda.is_available():
        return {"phase": PHASE, "status": "NOT_RUN", "reason": "torch.cuda.is_available() is False on this host",
                "environment": _environment()}
    from .. import scaling
    out = {"phase": PHASE, "status": "MEASURED", "device": torch.cuda.get_device_name(0),
           "capability": list(torch.cuda.get_device_capability(0)), "environment": _environment()}
    if args.ckpt:
        model, _ = load(args.ckpt, chunk=256, horizon=4096)
    else:
        model = build("history", "facts", 0, chunk=256, horizon=4096).eval()
    data = facts_task.generate(facts_task.FactTask(keys=32, events=200), 2, 8192, seed=TEST_SEED + 99)

    def run(device, autocast=False):
        model.to(device)
        state, logits, picks = model.new_state(2), [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            for at in range(0, 8192, 256):
                out, diag = model(data.tokens[:, at:at + 256].to(device), state, diagnostics=True)
                logits.append(out.float().cpu())
                picks.append(diag[model.read_layers[-1]]["selected"].cpu())
        return torch.cat(logits, 1), torch.cat(picks, 1), state

    reference, chosen, _ = run("cpu")
    on_gpu, chosen_gpu, _ = run("cuda")
    low, _, state = run("cuda", autocast=True)
    out["parity_float32"] = {"max_abs_logit_difference": float((on_gpu - reference).abs().max()),
                             "selection_agreement": float((chosen_gpu == chosen).double().mean())}
    out["bf16_autocast"] = {"finite": bool(torch.isfinite(low).all()), "ersa_state_dtype": str(state.cells[0].dtype),
                            "argmax_agreement_with_float32": float((low.argmax(-1) == reference.argmax(-1))
                                                                   .double().mean())}
    model.to("cpu")
    torch.cuda.empty_cache()
    ersa, history = scaling.configuration("r3_500m")
    big = ErsaHistoryLM(ersa, history).to("cuda").eval()
    weights = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    state, peaks, chunk = big.new_state(1), {}, 512
    piece = torch.randint(0, 1000, (1, chunk), device="cuda")
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for step in range((1 << 20) // chunk):
            big(piece, state)
            if (step + 1) * chunk in (1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20):
                torch.cuda.synchronize()
                peaks[(step + 1) * chunk] = torch.cuda.max_memory_allocated() - weights
    out["r3_500m_stream"] = {"weights_bytes": weights, "context_peak_bytes_by_tokens": peaks,
                             "state_bytes": state.nbytes(), "seconds": time.perf_counter() - started,
                             "flat_from_256k": max(peaks[k] for k in peaks if k >= 1 << 18)
                             <= 1.05 * peaks[1 << 18],
                             "within_1GiB_target": peaks[1 << 20] <= 1 << 30,
                             "within_2GiB_ceiling": peaks[1 << 20] <= 2 << 30,
                             "note": "seed weights: a memory and numerics qualification, not a trained model"}
    return out


def command_scale(args) -> dict:
    from .. import scaling
    return scaling.report(measure=not args.smoke)


def command_smoke(args) -> dict:
    import os
    import tempfile
    out = {}
    with tempfile.TemporaryDirectory() as folder:
        for arch, task in (("ersa_conv", "facts"), ("history", "facts"), ("history_oracle", "facts"),
                           ("history_local128", "facts"), ("history", "program")):
            path = os.path.join(folder, f"{arch}_{task}.pt")
            ns = argparse.Namespace(**{**vars(args), "arch": arch, "task": task, "steps": 3, "seed": 0,
                                       "eval_every": 3, "out": path, "smoke": True})
            trained = command_train(ns)
            evaluate = command_facts if task == "facts" else command_program
            result = evaluate(argparse.Namespace(**{**vars(ns), "ckpt": path, "out": None}))
            out[f"{arch}/{task}"] = trained["status"], len(result.get("ladder", result.get("free_running", [])))
    out["scale"] = sorted(command_scale(args)["configurations"])
    return out


def _finite(value):
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _environment() -> dict:
    return {"torch": torch.__version__, "python": platform.python_version(), "machine": platform.machine(),
            "threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available(),
            "processor": platform.processor() or None}


def _write(path: str, record: dict) -> None:
    text = json.dumps(_finite(dict(record, environment=_environment())), indent=1, sort_keys=True, allow_nan=False)
    with open(path, "w", encoding="utf-8") as handle:           # explicit offline artifact, requested by --out
        handle.write(text + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["train", "facts", "program", "scale", "gpu", "smoke"])
    parser.add_argument("--arch", choices=list(ARCHS), default="history")
    parser.add_argument("--task", choices=list(TASKS), default="facts")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--chunk", type=int, default=None, help="evaluation chunk (default: the training chunk)")
    parser.add_argument("--max-tokens", type=int, default=1 << 21)
    parser.add_argument("--diagnosis-max", type=int, default=1 << 18)
    parser.add_argument("--streams", type=int, default=None)
    parser.add_argument("--generate", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--keys", type=int, default=None, help="facts: training fact density (default 32)")
    parser.add_argument("--length", type=int, default=None, help="training length override")
    parser.add_argument("--batch", type=int, default=None, help="training batch override")
    args = parser.parse_args(argv)
    args.smoke = args.command == "smoke"
    torch.set_num_threads(args.threads)
    result = {"train": command_train, "facts": command_facts, "program": command_program, "scale": command_scale,
              "gpu": command_gpu, "smoke": command_smoke}[args.command](args)
    if args.command in ("scale", "gpu") and args.out:
        _write(args.out, result)
    elif not args.out or args.command == "smoke":
        print(json.dumps(_finite(result if args.command != "train" else {k: v for k, v in result.items()
                                                                          if k != "checkpoint"}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
