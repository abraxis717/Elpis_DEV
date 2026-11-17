#!/usr/bin/env python3
"""Elpis Torch R5: learned coarse-to-fine evidence retrieval over a 1,048,576-token exact history.

    train     Phase A: 1,024-token mixture, every parameter. Phase B: the curriculum 4K -> 16K -> 64K -> 262K -> 1M
              with the core FROZEN (only the evidence path learns) and a 1,024-token rehearsal batch per step.
              Writes a checkpoint (--out).
    eval      a checkpoint on the evaluation sets over the length ladder: coarse / fine / readout / end-to-end
              metrics with Wilson 95 % intervals, revision and program metrics, system measurements
    ablate    evaluation-time controls at one length (oracle source, coarse only, no reranker, coarse top-K,
              untrained index, shuffled index identity)
    r3base    the R3 architecture (ErsaHistoryLM, top-2, addressing loss) trained on the same mixture (control)
    memory    memory by category: exact formulas, measured tensors, measured peak RSS
    smoke     every path once, tiny, nothing kept

Labels from the generators are used for scoring, for the TRAINING-ONLY coarse / fine losses and teacher insertion,
and for the oracle DIAGNOSIS mode. In learned evaluation no label reaches the model (tested). Checkpoints and
records are EXPLICIT OFFLINE ARTIFACTS written only where --out says. Wall-clock numbers are PERFORMANCE_ONLY.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import resource
import sys
import time

import torch
from torch.nn import functional as F

from .. import evidence_tasks as et
from ..config import ModelConfig
from ..ersa_model import ErsaConfig
from ..evidence_model import EvidenceConfig, EvidenceLM
from ..model import parameter_digest
from ..training import TrainConfig

PHASE = "ELPIS_TORCH_R5_LEARNED_EVIDENCE_RETRIEVAL"
CORE = dict(d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64, channel="cgc", cgc_bottleneck=32,
            cgc_channels=96, cgc_units=4)                         # the R3 core, unchanged
ERSA = dict(ersa_heads=4, key_dim=16, value_dim=16, ersa_kernel="chunked", scan_block=64, conv_kernel=4)
OPTIM = dict(lr=3e-3, lr_b=1e-3, warmup=100, weight_decay=0.01, grad_clip=1.0, background=0.05)
TRAIN_SEED, VAL_SEED, TEST_SEED = 0, 600_000_000, 700_000_000     # the R3 seed domains (disjoint)
R5_TEST = TEST_SEED + 50_000_000
LADDER = (1024, 4096, 16384, 65536, 262144, 1048576)
CURRICULUM = ((4096, 4, 120), (16384, 2, 80), (65536, 1, 60), (262144, 1, 24), (1048576, 1, 12))
SETS = ("records", "revision", "scoped", "lexical", "low_overlap", "r3facts")
R3_PLANTED = (0.05, 0.10, 0.45, 0.50, 0.85, 0.90)


def build(seed: int, *, chunk: int = 1024, **evidence) -> EvidenceLM:
    widths = ModelConfig(vocab_size=et.VOCAB, **CORE, seed=seed)
    return EvidenceLM(ErsaConfig(widths, **ERSA, chunk=chunk), EvidenceConfig(**evidence))


# -- data -----------------------------------------------------------------------------------------------------

def queries_for(length: int) -> int:
    return max(8, min(96, int(12 * math.sqrt(length / 1024))))


def mixture(seed: int, length: int, batch: int, *, r3: bool = True) -> et.EvidenceStreams:
    """One training batch: records, scoped symbols and (at 1K) R3 facts, with the density, revision and
    hard-negative rates drawn independently per batch."""
    rng = random.Random(f"elpis-torch-r5-mixture:{seed}")
    n_r3 = batch // 3 if r3 else 0
    n_sc = (batch - n_r3) // 2
    n_rec = batch - n_r3 - n_sc
    if batch == 1 and rng.random() < 0.5:                # one long stream: records or scoped, alternately
        n_rec, n_sc = 0, 1
    q = queries_for(length)
    density = rng.choice((1.0, 4.0, 8.0, 16.0))
    rev = rng.choice(((0, 1), (0, 2), (1, 4), (2, 6)))
    neg = rng.choice(((0, 0), (0, 1), (1, 2)))
    rec = et.RecordTask(keys=128, queries=q, revisions=rev, negated=neg, reference=neg, overlap=neg,
                        background=density, far=0.3)
    sc = et.ScopeTask(queries=q, revisions=rev, wrong_scope=(1, 2), reference=neg, same_value=neg,
                      background=0.75 * density, far=0.3)
    parts = []
    if n_rec:
        parts.append(et.generate_records(rec, n_rec, length, seed=seed))
    if n_sc:
        parts.append(et.generate_scoped(sc, n_sc, length, seed=seed))
    if n_r3:
        parts.append(et.generate_r3(n_r3, length, seed=seed))
    return et.concat(parts)


def eval_set(name: str, length: int, *, streams: int | None = None, seed_offset: int = 0) -> et.EvidenceStreams:
    count = streams or {1024: 16, 4096: 8, 16384: 4, 65536: 2, 262144: 2, 1048576: 2}.get(length, 1)
    q = queries_for(length)
    seed = R5_TEST + SETS.index(name) * 1_000_000 + length + seed_offset if name != "r3facts" else TEST_SEED + length
    if name == "records":
        task = et.RecordTask(keys=128, queries=q, revisions=(0, 2), negated=(0, 1), reference=(0, 1), overlap=(0, 1),
                             background=8.0, far=0.5)
        return et.with_family(et.generate_records(task, count, length, seed=seed))
    if name == "revision":
        task = et.RecordTask(keys=128, queries=q, revisions=(4, 8), negated=(0, 1), reference=(0, 1), overlap=(0, 1),
                             p_stale_question=0.6, background=8.0, far=0.5)
        return et.with_family(et.generate_records(task, count, length, seed=seed))
    if name == "lexical":
        task = et.RecordTask(keys=128, queries=q, revisions=(1, 2), negated=(2, 3), reference=(2, 3), overlap=(2, 3),
                             background=8.0, far=0.5)
        return et.with_family(et.generate_records(task, count, length, seed=seed))
    if name == "low_overlap":
        task = et.RecordTask(keys=128, queries=q, revisions=(0, 0), negated=(0, 0), reference=(0, 0), overlap=(0, 0),
                             p_stale_question=0.0, background=8.0, far=0.5)
        return et.with_family(et.generate_records(task, count, length, seed=seed))
    if name == "scoped":
        task = et.ScopeTask(queries=q, revisions=(0, 2), wrong_scope=(1, 3), reference=(0, 1), same_value=(0, 1),
                            background=6.0, far=0.5)
        return et.with_family(et.generate_scoped(task, count, length, seed=seed))
    if name == "r3facts":                               # the R3 benchmark streams (same seeds as run_r3 facts)
        return et.with_family(et.generate_r3(count, length, seed=seed, events=min(112, length * 7 // 64),
                                             planted=R3_PLANTED))
    raise ValueError(name)


def probe_index(data: et.EvidenceStreams):
    """``{(row, input position): probe number}`` and the per-probe stale rows."""
    p = data.probes
    return {(int(r), int(a) - 1): i for i, (r, a) in enumerate(zip(p["row"].tolist(), p["answer"].tolist()))}


# -- training -------------------------------------------------------------------------------------------------

def _trainable(model, phase: str):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(phase == "A" or name.startswith(("encoder.", "lookup.")))
    return [p for p in model.parameters() if p.requires_grad]


def _optimizer(params, lr):
    return torch.optim.AdamW([{"params": [p for p in params if p.ndim >= 2], "weight_decay": OPTIM["weight_decay"]},
                              {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95))


def dense_loss(model, data: et.EvidenceStreams):
    """One chunk, every position a query (the 1,024-token phase and the rehearsal batch)."""
    tokens = data.tokens[:, :-1]
    batch, length = tokens.shape
    labels = data.query_source()[:, :-1]
    stale = torch.full((batch, length, et.SLOTS), -1, dtype=torch.long)
    p = data.probes
    keep = p["answer"] - 1 < length
    stale[p["row"][keep], p["answer"][keep] - 1] = p["stale"][keep]
    state = model.new_state(batch)
    logits, diag = model(tokens, state, evidence=labels, stale=stale, teacher=True, diagnostics="losses")
    targets = data.tokens[:, 1:]
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), reduction="none")
    ce = ce.view(batch, length)
    probe = data.kind[:, 1:] != 0
    answer = ce[probe].mean() + OPTIM["background"] * ce.mean()
    losses = diag.get("losses", {})
    coarse = losses.get("coarse_source_loss", answer.new_zeros(()))
    fine = losses.get("fine_evidence_loss", answer.new_zeros(()))
    return answer + coarse + fine, {"answer": float(answer.detach()), "coarse": float(coarse.detach()),
                                    "fine": float(fine.detach())}


def stream_backward(model, data: et.EvidenceStreams, *, chunk: int, rng: random.Random, extra: int = 4) -> dict:
    """Phase B: one long stream, chunk by chunk, the core frozen. Each chunk's loss is backpropagated at once
    (exact: no graph crosses a chunk, stored descriptors are detached), weighted by its share of the stream's
    probes; chunks without queries are ingested without a graph."""
    tokens = data.tokens[:, :-1]
    batch, length = tokens.shape
    index = probe_index(data)
    probes = len(index)
    p = data.probes
    by_pos: dict = {}
    for (row, q), i in index.items():
        by_pos.setdefault(q, []).append((row, i))
    state = model.new_state(batch)
    totals = {"answer": 0.0, "coarse": 0.0, "fine": 0.0}
    hist_total = max(1, probes)
    for at in range(0, length, chunk):
        piece = tokens[:, at:at + chunk]
        size = piece.shape[1]
        qs = sorted({q for q in by_pos if at <= q < at + size} | {at + rng.randrange(size) for _ in range(extra)})
        if not any(q in by_pos for q in qs):
            with torch.no_grad():
                model(piece, state, logit_positions=torch.zeros(0, dtype=torch.long))
            continue
        lp = torch.tensor([q - at for q in qs])
        src = data.source[:, [q + 1 for q in qs]]
        stl = torch.full((batch, len(qs), et.SLOTS), -1, dtype=torch.long)
        target = data.tokens[:, [q + 1 for q in qs]]
        is_probe = torch.zeros(batch, len(qs), dtype=torch.bool)
        for j, q in enumerate(qs):
            for row, i in by_pos.get(q, []):
                stl[row, j] = p["stale"][i]
                is_probe[row, j] = True
        logits, diag = model(piece, state, logit_positions=lp, evidence=src, stale=stl, teacher=True,
                             diagnostics="losses")
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1), reduction="none")
        ce = ce.view(batch, len(qs))
        answer = ce[is_probe].sum() / probes + OPTIM["background"] * ce[~is_probe].mean() * (len(qs) / length)
        losses = diag.get("losses", {})
        loss = answer
        for name, key in (("coarse", "coarse_source_loss"), ("fine", "fine_evidence_loss")):
            if key in losses:
                part = losses[key] * losses[key + "_n"] / hist_total
                loss = loss + part
                totals[name] += float(part.detach())
        totals["answer"] += float(answer.detach())
        loss.backward()
    return totals


def command_train(args) -> dict:
    torch.manual_seed(args.seed)
    model = build(args.seed, chunk=1024, **_overrides(args))
    started, history, status = time.perf_counter(), [], "COMPLETED"
    # Phase A
    params = _trainable(model, "A")
    optimizer = _optimizer(params, OPTIM["lr"])
    schedule = TrainConfig(steps=args.steps, lr=OPTIM["lr"], warmup_steps=min(OPTIM["warmup"], args.steps))
    batch = 4 if args.smoke else args.batch
    length = 256 if args.smoke else 1024
    for step in range(args.steps):
        model.train()
        data = mixture(TRAIN_SEED + args.seed * 10_000_000 + step, length, batch)
        loss, parts = dense_loss(model, data)
        if not torch.isfinite(loss):
            status = "DIVERGED"
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, OPTIM["grad_clip"])
        for group in optimizer.param_groups:
            group["lr"] = schedule.lr_at(step)
        optimizer.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            val = quick_val(model, 1024 if not args.smoke else 256, smoke=args.smoke)
            history.append({"phase": "A", "step": step + 1, "seconds": round(time.perf_counter() - started, 1),
                            **parts, "val": val})
            print(f"[r5 s{args.seed}] A {step + 1} {json.dumps(parts)} {json.dumps(val)}", file=sys.stderr, flush=True)
            if args.out:
                _save(model, args, history, status, started)
    # Phase B: the curriculum, core frozen
    if status == "COMPLETED" and not args.no_curriculum:
        params = _trainable(model, "B")
        optimizer = _optimizer(params, OPTIM["lr_b"])
        stages = ((512, 2, 2), (1024, 1, 1)) if args.smoke else CURRICULUM
        total = sum(steps for _, _, steps in stages)
        done = 0
        long_model = build(args.seed, chunk=256 if args.smoke else 4096, **_overrides(args))
        long_model.load_state_dict(model.state_dict())
        long_model.train()
        _trainable(long_model, "B")
        params = [p for p in long_model.parameters() if p.requires_grad]
        optimizer = _optimizer(params, OPTIM["lr_b"])
        rng = random.Random(f"elpis-torch-r5-curriculum:{args.seed}")
        for length, batch, steps in stages:
            if length > args.max_length:
                break
            for step in range(steps):
                lr = OPTIM["lr_b"] * 0.5 * (1 + math.cos(math.pi * done / total))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                data = mixture(TRAIN_SEED + args.seed * 10_000_000 + 5_000_000 + done, length, batch, r3=False)
                totals = stream_backward(long_model, data, chunk=long_model.ersa.chunk, rng=rng)
                rehearsal, rparts = dense_loss(long_model, mixture(TRAIN_SEED + args.seed * 10_000_000 + 7_000_000
                                                                   + done, 256 if args.smoke else 1024,
                                                                   4 if args.smoke else 8))
                rehearsal.backward()
                torch.nn.utils.clip_grad_norm_(params, OPTIM["grad_clip"])
                optimizer.step()
                done += 1
                record = {"phase": "B", "length": length, "step": done,
                          "seconds": round(time.perf_counter() - started, 1), **totals,
                          "rehearsal": rparts}
                if step + 1 == steps:
                    record["val"] = quick_val(long_model, min(length, 16384), smoke=args.smoke)
                history.append(record)
                print(f"[r5 s{args.seed}] B {length} {step + 1}/{steps} {json.dumps(totals)} "
                      f"{json.dumps(record.get('val', {}))}", file=sys.stderr, flush=True)
            if args.out:
                model.load_state_dict(long_model.state_dict())
                _save(model, args, history, status, started)
        model.load_state_dict(long_model.state_dict())
    record = _save(model, args, history, status, started) if args.out else None
    return {"status": status, "history": history, "checkpoint": record}


def _overrides(args) -> dict:
    out = {}
    for name in ("per_block", "block", "coarse_top", "evidence"):
        value = getattr(args, name, None)
        if value is not None:
            out[name] = value
    if out.get("block") and out["block"] != 32:
        out["capacity"] = 1 << 20
    return out


@torch.no_grad()
def quick_val(model, length: int, *, smoke: bool = False) -> dict:
    model.eval()
    out = {}
    for name in ("records", "scoped"):
        data = eval_set(name, length, streams=2 if smoke else 4, seed_offset=VAL_SEED - TEST_SEED)
        summary = summarize(evaluate(model, data, chunk=min(model.ersa.chunk, length)))
        out[name] = {k: summary["all"].get(k) for k in ("accuracy", "coarse_recall_64", "fine_recall_4")}
    model.train()
    return out


def _save(model, args, history, status, started) -> dict:
    """EXPLICIT OFFLINE ARTIFACT: the checkpoint requested with --out."""
    record = {"phase": PHASE, "seed": args.seed, "steps": args.steps, "status": status, "history": history,
              "train_seconds": round(time.perf_counter() - started, 1), "optim": OPTIM, "curriculum": CURRICULUM,
              "evidence": _overrides(args), "parameters": sum(p.numel() for p in model.parameters()),
              "parameter_digest": parameter_digest(model)}
    torch.save({"record": record, "state_dict": {k: v.detach() for k, v in model.state_dict().items()}}, args.out)
    return record


def load(path: str, *, chunk: int = 4096, **changes):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    record = saved["record"]
    model = build(record["seed"], chunk=chunk, **{**record.get("evidence", {}), **changes})
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, record


# -- evaluation -----------------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data: et.EvidenceStreams, *, chunk: int = 4096, mode: str = "learned", rerank=None,
             coarse_top=None, shuffle=None, timing: bool = False) -> dict:
    """Stream ``data`` through bounded state; the read is computed at every probe's input position (exactly the
    dense computation there). Returns per-probe arrays and system measurements."""
    tokens = data.tokens
    batch, length = tokens.shape
    index = probe_index(data)
    positions = sorted({q for (_, q) in index})
    where = {q: j for j, q in enumerate(positions)}
    state = model.new_state(batch)
    state.shuffle = shuffle
    state.timing = {} if timing else None
    n = len(positions)
    predicted = torch.full((batch, n), -1, dtype=torch.long)
    fields = ("coarse_rank", "fine_rank", "read_position", "null_weight", "historical", "gold_block", "local_block",
              "eligible")
    diag_all = {k: [] for k in fields}
    top_all, sel_all = [], []
    started = time.perf_counter()
    extra = {"mode": mode}
    if rerank:
        extra["rerank"] = rerank
    if coarse_top:
        extra["coarse_top"] = coarse_top
    for at in range(0, length, chunk):
        piece = tokens[:, at:at + chunk]
        qs = [q for q in positions[_bisect(positions, at):_bisect(positions, at + piece.shape[1])]]
        lp = torch.tensor([q - at for q in qs], dtype=torch.long)
        if not qs:
            model(piece, state, logit_positions=lp)
            continue
        src = data.source[:, [q + 1 for q in qs]]
        logits, diag = model(piece, state, logit_positions=lp, evidence=src, diagnostics=True, **extra)
        j0 = where[qs[0]]
        predicted[:, j0:j0 + len(qs)] = logits.argmax(-1)
        for k in fields:
            diag_all[k].append(diag[k])
        top_all.append(diag["fine_top"])
        sel_all.append(diag["selected"])
    seconds = time.perf_counter() - started
    cat = {k: torch.cat(v, dim=1) for k, v in diag_all.items()} if positions else {}
    fine_top = _pad_cat(top_all) if positions else None
    selected = _pad_cat(sel_all) if positions else None
    p = data.probes
    s = model.evidence.block
    rows, cols = [], []
    for i, (row, answer) in enumerate(zip(p["row"].tolist(), p["answer"].tolist())):
        rows.append(row)
        cols.append(where[answer - 1])
    rows, cols = torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)
    out = {"n": len(rows), "family": p["family"], "kind": p["kind"], "revised": p["revised"],
           "negatives": p["negatives"], "distance": p["answer"] - p["source"],
           "correct": predicted[rows, cols] == tokens[rows, p["answer"]] if len(rows) else torch.zeros(0, dtype=torch.bool)}
    if len(rows):
        gold = p["source"] // s
        out.update(historical=cat["historical"][rows, cols], local=cat["local_block"][rows, cols] == gold,
                   coarse_rank=cat["coarse_rank"][rows, cols], fine_rank=cat["fine_rank"][rows, cols],
                   read_hit=cat["read_position"][rows, cols] == p["source"],
                   null_weight=cat["null_weight"][rows, cols].float(),
                   selected_hit=(selected[rows, cols] == gold[:, None]).any(-1),
                   eligible=cat["eligible"][rows, cols])
        top1 = fine_top[rows, cols, 0]
        stale_blocks = torch.where(p["stale"] >= 0, p["stale"] // s, torch.full_like(p["stale"], -2))
        wrong_blocks = torch.where(p["wrong_scope"] >= 0, p["wrong_scope"] // s, torch.full_like(p["wrong_scope"], -2))
        out["stale_top1"] = (top1[:, None] == stale_blocks).any(-1) & (top1 != gold)
        out["wrong_scope_top1"] = (top1[:, None] == wrong_blocks).any(-1) & (top1 != gold)
    out["system"] = {"tokens": length, "streams": batch, "seconds": round(seconds, 2),
                     "tokens_per_second": round(batch * length / max(seconds, 1e-9)), "queries": n * batch,
                     "state_bytes": state.nbytes(), "ring_stats": dict(state.ring.stats),
                     "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if timing and state.timing:
        queries = max(1, state.timing.get("queries", 1))
        out["system"]["per_query_ms"] = {k: round(1000 * v / queries, 3) for k, v in state.timing.items()
                                         if k != "queries"}
    return out


def _pad_cat(parts):
    """Concatenate ``[B, n, k_i]`` id tensors along queries, padding the last dimension with -1."""
    width = max(t.shape[-1] for t in parts)
    return torch.cat([F.pad(t, (0, width - t.shape[-1]), value=-1) for t in parts], dim=1)


def _bisect(values, x):
    import bisect
    return bisect.bisect_left(values, x)


def wilson(k: int, n: int, z: float = 1.959964) -> list | None:
    if n == 0:
        return None
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return [round(centre - half, 4), round(centre + half, 4)]


def _rate(mask, subset) -> dict:
    n = int(subset.sum())
    k = int((mask & subset).sum())
    return {"n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def summarize(run: dict) -> dict:
    """Stage metrics over subsets of the probes (historical: the evidence is in an earlier block, so it must be
    retrieved; local: it is in the query's own block)."""
    if not run["n"]:
        return {"all": {"n": 0}}
    hist, local = run["historical"], run["local"]
    subsets = {"all": torch.ones_like(hist), "historical": hist, "local": local,
               "revised": run["revised"] & hist, "scoped": (run["family"] == 2), "records": (run["family"] == 1),
               "r3facts": (run["family"] == 3)}
    bounds = (0, 64, 1024, 16384, 262144, 1 << 62)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        subsets[f"distance_{lo}_{hi if hi < 1 << 62 else 'inf'}"] = (run["distance"] >= lo) & (run["distance"] < hi)
    for name, bit in et.NEG_BIT.items():
        subsets[f"neg_{name}"] = ((run["negatives"] & bit) != 0) & hist
    out = {}
    for name, sub in subsets.items():
        n = int(sub.sum())
        if not n:
            continue
        h = sub & hist
        entry = {"n": n, "accuracy": round(float(run["correct"][sub].double().mean()), 4),
                 "accuracy_ci95": wilson(int(run["correct"][sub].sum()), n), "historical_n": int(h.sum())}
        if int(h.sum()):
            rank = run["coarse_rank"][h].double()
            for k in (16, 32, 64, 128):
                entry[f"coarse_recall_{k}"] = _rate(run["coarse_rank"] <= k, h)["rate"]
            entry["coarse_recall_64_ci95"] = _rate(run["coarse_rank"] <= 64, h)["ci95"]
            entry["coarse_mrr"] = round(float((1.0 / rank).mean()), 4)
            entry["coarse_rank_median"] = float(rank.median())
            entry["coarse_rank_p95"] = float(torch.quantile(rank, 0.95))
            for k in (1, 4, 8):
                entry[f"fine_recall_{k}"] = _rate(run["fine_rank"] <= k, h)["rate"]
            entry["fine_recall_4_ci95"] = _rate(run["fine_rank"] <= 4, h)["ci95"]
            entry["fine_recall_8_ci95"] = _rate(run["fine_rank"] <= 8, h)["ci95"]
            hit64 = h & (run["coarse_rank"] <= 64)
            entry["fine_recall_4_given_coarse"] = _rate(run["fine_rank"] <= 4, hit64)["rate"]
            sel = h & run["selected_hit"]
            entry["conditional_accuracy"] = _rate(run["correct"], sel)
            entry["accuracy_when_not_selected"] = _rate(run["correct"], h & ~run["selected_hit"])["rate"]
            entry["read_hit_given_selected"] = _rate(run["read_hit"], sel)["rate"]
            entry["stale_top1_rate"] = _rate(run["stale_top1"], h & run["revised"])
            entry["wrong_scope_top1_rate"] = _rate(run["wrong_scope_top1"], h & (run["family"] == 2))
        out[name] = entry
    return out


def command_eval(args) -> dict:
    model, record = load(args.ckpt, chunk=args.chunk)
    out = {"phase": PHASE, "checkpoint": {k: v for k, v in record.items() if k != "history"}, "rows": []}
    lengths = (512, 1024) if args.smoke else [x for x in LADDER if x <= args.max_length]
    if getattr(args, "lengths", None):
        lengths = [int(x) for x in args.lengths.split(",")]
    sets = args.sets.split(",") if args.sets else SETS
    for length in lengths:
        for name in sets:
            data = eval_set(name, length, streams=2 if args.smoke else args.streams)
            run = evaluate(model, data, chunk=min(args.chunk, length), timing=True)
            row = {"set": name, "tokens": length, "streams": data.tokens.shape[0], "summary": summarize(run),
                   "system": run["system"]}
            out["rows"].append(row)
            a = row["summary"].get("all", {})
            hst = row["summary"].get("historical", {})
            print(f"[r5 eval s{record['seed']}] {name} {length}: acc={a.get('accuracy')} n={a.get('n')} "
                  f"hist_acc={hst.get('accuracy')} c@64={hst.get('coarse_recall_64')} f@4={hst.get('fine_recall_4')} "
                  f"f@8={hst.get('fine_recall_8')} cond={hst.get('conditional_accuracy', {}).get('rate')} "
                  f"{run['system']['tokens_per_second']} tok/s", file=sys.stderr, flush=True)
            if args.out:
                _write(args.out, out)
    return out


def command_ablate(args) -> dict:
    """Evaluation-time controls at one length on the given sets."""
    model, record = load(args.ckpt, chunk=args.chunk)
    length = args.length
    sets = args.sets.split(",") if args.sets else ("records", "revision", "scoped")
    variants = [("learned", {}), ("oracle_source", {"mode": "oracle"}), ("coarse_only", {"rerank": "coarse"}),
                ("coarse_reread_no_reranker", {"rerank": "all"}),
                ("coarse_top_16", {"coarse_top": 16}), ("coarse_top_32", {"coarse_top": 32}),
                ("coarse_top_128", {"coarse_top": 128}), ("shuffled_index", {"shuffle": 1234}),
                ("untrained_index", {"untrained": True})]
    if args.variants:
        keep = set(args.variants.split(","))
        variants = [v for v in variants if v[0] in keep]
    out = {"phase": PHASE, "checkpoint": {k: v for k, v in record.items() if k != "history"}, "tokens": length,
           "rows": []}
    for name in sets:
        data = eval_set(name, length, streams=args.streams)
        for label, options in variants:
            use = model
            options = dict(options)
            if options.pop("untrained", False):
                use, _ = load(args.ckpt, chunk=args.chunk, coarse_top=options.get("coarse_top", 64))
                fresh = build(record["seed"] + 1000, chunk=args.chunk, **record.get("evidence", {}))
                with torch.no_grad():
                    for part in ("pool", "descriptor", "coarse_q"):
                        getattr(use.lookup, part).weight.copy_(getattr(fresh.lookup, part).weight)
            if "coarse_top" in options and options["coarse_top"] > 64:
                use, _ = load(args.ckpt, chunk=args.chunk, coarse_top=options["coarse_top"])
            run = evaluate(use, data, chunk=min(args.chunk, length), **options)
            row = {"set": name, "variant": label, "summary": summarize(run), "system": run["system"]}
            out["rows"].append(row)
            hst = row["summary"].get("historical", {})
            print(f"[r5 ablate s{record['seed']}] {name} {length} {label}: acc={row['summary']['all'].get('accuracy')} "
                  f"hist={hst.get('accuracy')} c@64={hst.get('coarse_recall_64')} f@4={hst.get('fine_recall_4')}",
                  file=sys.stderr, flush=True)
            if args.out:
                _write(args.out, out)
    return out


# -- the R3 baseline on the R5 tasks (control) -----------------------------------------------------------------

R3_HISTORY = dict(block=32, topk=2, index_dim=32, per_block=2, read_heads=2, read_kv_heads=1, read_head_dim=32,
                  overlap=8, layers=(-1,))


def build_r3(seed: int, *, chunk: int = 1024, horizon: int = 4096):
    from ..history_model import ErsaHistoryLM, HistoryConfig
    widths = ModelConfig(vocab_size=et.VOCAB, **CORE, seed=seed)
    return ErsaHistoryLM(ErsaConfig(widths, **ERSA, chunk=chunk), HistoryConfig(horizon=horizon, **R3_HISTORY))


def command_r3base(args) -> dict:
    """The R3 architecture (top-2 learned addressing, readout, training-only addressing loss), trained on the
    same R5 mixture and budget as Phase A; evaluated on the same sets (``--ckpt`` with ``eval --r3``)."""
    from ..history_model import addressing_loss
    from .run_r3 import eligible_address
    torch.manual_seed(args.seed)
    length = 256 if args.smoke else 1024
    model = build_r3(args.seed, chunk=length, horizon=4 * length)
    params = list(model.parameters())
    optimizer = _optimizer(params, OPTIM["lr"])
    schedule = TrainConfig(steps=args.steps, lr=OPTIM["lr"], warmup_steps=min(OPTIM["warmup"], args.steps))
    started, history = time.perf_counter(), []
    for step in range(args.steps):
        model.train()
        data = mixture(TRAIN_SEED + args.seed * 10_000_000 + step, length, 4 if args.smoke else args.batch)
        tokens = data.tokens[:, :-1]
        gold = eligible_address(data.query_block(32)[:, :-1], model)
        logits, diag = model(tokens, model.new_state(tokens.shape[0]), diagnostics=True)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), data.tokens[:, 1:].reshape(-1),
                             reduction="none").view(tokens.shape)
        probe = data.kind[:, 1:] != 0
        aux = addressing_loss(diag, gold)
        loss = ce[probe].mean() + OPTIM["background"] * ce.mean() + aux
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, OPTIM["grad_clip"])
        for group in optimizer.param_groups:
            group["lr"] = schedule.lr_at(step)
        optimizer.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            history.append({"step": step + 1, "seconds": round(time.perf_counter() - started, 1),
                            "loss": float(loss.detach()), "aux": float(aux.detach())})
            print(f"[r3base s{args.seed}] {step + 1} loss {float(loss.detach()):.3f} aux {float(aux.detach()):.3f}",
                  file=sys.stderr, flush=True)
    record = {"phase": PHASE, "arch": "r3_history_on_r5_mixture", "seed": args.seed, "steps": args.steps,
              "history": history, "train_seconds": round(time.perf_counter() - started, 1),
              "parameters": sum(p.numel() for p in model.parameters()), "parameter_digest": parameter_digest(model)}
    if args.out:
        torch.save({"record": record, "state_dict": {k: v.detach() for k, v in model.state_dict().items()}},
                   args.out)
    return {"status": "COMPLETED", "history": history}


@torch.no_grad()
def evaluate_r3(model, data: et.EvidenceStreams, *, chunk: int = 4096) -> dict:
    """The R3 model streamed through the R4 tiled runtime; per-probe correctness and recall@2."""
    from ..context_runtime import RuntimeOptions
    tokens = data.tokens
    batch, length = tokens.shape
    state = model.new_state(batch)
    state.runtime = RuntimeOptions(scorer="tiled", tile=4096)
    index = probe_index(data)
    positions = sorted({q for (_, q) in index})
    where = {q: j for j, q in enumerate(positions)}
    predicted = torch.full((batch, len(positions)), -1, dtype=torch.long)
    chosen = torch.full((batch, len(positions), 2), -1, dtype=torch.long)
    started = time.perf_counter()
    for at in range(0, length, chunk):
        piece = tokens[:, at:at + chunk]
        qs = positions[_bisect(positions, at):_bisect(positions, at + piece.shape[1])]
        lp = torch.tensor([q - at for q in qs], dtype=torch.long)
        logits, diag = model(piece, state, logit_positions=lp, diagnostics=True)
        if qs:
            j0 = where[qs[0]]
            predicted[:, j0:j0 + len(qs)] = logits.argmax(-1)
            chosen[:, j0:j0 + len(qs)] = diag[model.read_layers[-1]]["selected"][:, lp]
    p = data.probes
    rows = p["row"]
    cols = torch.tensor([where[int(a) - 1] for a in p["answer"].tolist()], dtype=torch.long)
    gold = p["source"] // 32
    query_block = (p["answer"] - 1) // 32
    return {"n": rows.numel(), "family": p["family"], "kind": p["kind"], "revised": p["revised"],
            "negatives": p["negatives"], "distance": p["answer"] - p["source"],
            "correct": predicted[rows, cols] == tokens[rows, p["answer"]],
            "historical": gold < query_block, "recall_at_2": (chosen[rows, cols] == gold[:, None]).any(-1),
            "seconds": round(time.perf_counter() - started, 2)}


def summarize_r3(run: dict) -> dict:
    out = {}
    hist = run["historical"]
    for name, sub in (("all", torch.ones_like(hist)), ("historical", hist), ("revised", run["revised"] & hist)):
        n = int(sub.sum())
        if n:
            out[name] = {"n": n, "accuracy": round(float(run["correct"][sub].double().mean()), 4),
                         "accuracy_ci95": wilson(int(run["correct"][sub].sum()), n),
                         "recall_at_2": round(float(run["recall_at_2"][sub].double().mean()), 4)}
    return out


def command_r3eval(args) -> dict:
    saved = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = build_r3(saved["record"]["seed"], chunk=args.chunk, horizon=1 << 20)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    out = {"phase": PHASE, "checkpoint": saved["record"], "rows": []}
    lengths = (512, 1024) if args.smoke else [x for x in LADDER if x <= args.max_length]
    for length in lengths:
        for name in (args.sets.split(",") if args.sets else ("records", "revision", "scoped")):
            data = eval_set(name, length, streams=2 if args.smoke else (args.streams if length >= 1 << 20 else None))
            run = evaluate_r3(model, data, chunk=min(args.chunk, length))
            row = {"set": name, "tokens": length, "streams": data.tokens.shape[0], "summary": summarize_r3(run),
                   "seconds": run["seconds"]}
            out["rows"].append(row)
            print(f"[r3base eval] {name} {length}: {json.dumps(row['summary'].get('historical', {}))}",
                  file=sys.stderr, flush=True)
            if args.out:
                _write(args.out, out)
    return out


def memory_report(model: EvidenceLM, *, batch: int = 1) -> dict:
    """Memory by category for ``batch`` streams of the default geometry. EXACT: tensor sizes allocated now (or the
    same formulas on the meta device); SCRATCH: per-query-tile formulas of the largest transient tensors."""
    e, m = model.evidence, model.config
    params = sum(p.numel() for p in model.parameters())
    evidence_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith(("encoder.", "lookup.")))
    state = model.new_state(batch, ring_device="meta")
    ring = state.ring.nbytes()
    core_state = {k: v for k, v in state.nbytes().items() if k in ("ersa_cells", "conv_tail")}
    out = {"weights": {"parameters": params, "evidence_path_parameters": evidence_params,
                       "float32_bytes": 4 * params, "bfloat16_bytes": 2 * params}}
    per_block = {}
    for dtype, size in (("float32", 4), ("bfloat16", 2)):
        keys = batch * e.slots * e.per_block * e.key_dim * size
        per_block[dtype] = {"index_record_bytes": e.per_block * e.key_dim * size + 8,
                            "persistent_bytes": ring["history_tokens"] + keys + ring["index_ids"],
                            "index_keys_bytes": keys}
    out["persistent_retrieval_state"] = {
        "history_ring_bytes": ring["history_tokens"], "index_ids_bytes": ring["index_ids"],
        "by_key_dtype": per_block, "bytes_per_token_float32": round(per_block["float32"]["persistent_bytes"]
                                                                      / e.capacity, 3),
        "bytes_per_token_bfloat16": round(per_block["bfloat16"]["persistent_bytes"] / e.capacity, 3),
        "bytes_per_block_float32": round(per_block["float32"]["persistent_bytes"] / e.slots, 1),
        "bytes_per_block_bfloat16": round(per_block["bfloat16"]["persistent_bytes"] / e.slots, 1),
        "target": "far below 200 MiB"}
    out["local_working_state"] = core_state
    tile = max(1, min(e.query_tile, e.tile_elements // max(1, batch * e.slots * e.per_block)))
    span, w = e.span, e.encoder_dim
    out["scratch_per_query_tile"] = {
        "queries_per_tile": tile,
        "coarse_match_bytes": 4 * batch * tile * e.slots * e.per_block,
        "coarse_scores_bytes": 4 * batch * tile * e.slots * 3,
        "resident_descriptors_copy_bytes": 4 * batch * e.slots * e.per_block * e.key_dim,
        "reread_candidates_bytes_per_query": 4 * batch * e.coarse_top * span * w * (2 + 8 * e.encoder_layers),
        "fine_and_readout_bytes_per_query": 4 * batch * (e.coarse_top + 1) * e.block * (e.fine_dim + 2 * e.read_head_dim
                                                                                    + 2 * e.read_heads),
        "note": "transient; freed after each tile; resident blocks are re-encoded per tile (no per-token cache)"}
    trainable = evidence_params
    out["training"] = {"phase_A_gradients_bytes": 4 * params, "phase_A_adamw_state_bytes": 8 * params,
                       "phase_B_gradients_bytes": 4 * trainable, "phase_B_adamw_state_bytes": 8 * trainable}
    out["cuda"] = {"status": "NOT_RUN", "reason": "torch.cuda.is_available() is False on this host"} \
        if not torch.cuda.is_available() else {"status": "AVAILABLE_NOT_MEASURED_BY_THIS_COMMAND"}
    return out


@torch.no_grad()
def command_memory(args) -> dict:
    model = build(0, chunk=args.chunk)
    out = {"phase": PHASE, "accounting": memory_report(model)}
    if not args.smoke:                                   # measured: allocate the real state, stream, read RSS
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        data = eval_set("records", args.length, streams=1)
        run = evaluate(model, data, chunk=args.chunk, timing=True)
        out["measured_stream"] = {"tokens": args.length, "state_bytes": run["system"]["state_bytes"],
                                  "peak_rss_before_bytes": before, "peak_rss_after_bytes": run["system"]["peak_rss_bytes"],
                                  "tokens_per_second": run["system"]["tokens_per_second"],
                                  "per_query_ms": run["system"].get("per_query_ms"),
                                  "ring_stats": run["system"]["ring_stats"]}
    return out


def command_smoke(args) -> dict:
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "r5.pt")
        ns = argparse.Namespace(**{**vars(args), "steps": 3, "eval_every": 3, "out": path, "smoke": True,
                                   "max_length": 1024})
        trained = command_train(ns)
        ev = command_eval(argparse.Namespace(**{**vars(ns), "ckpt": path, "out": None, "chunk": 256,
                                                "sets": "records,scoped,r3facts", "streams": 2}))
        ab = command_ablate(argparse.Namespace(**{**vars(ns), "ckpt": path, "out": None, "chunk": 256, "length": 512,
                                                  "sets": "records", "streams": 2, "variants": None}))
    return {"train": trained["status"], "eval_rows": len(ev["rows"]), "ablate_rows": len(ab["rows"])}


def _finite(value):
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _finite(value.tolist())
    return value


def _environment() -> dict:
    return {"torch": torch.__version__, "python": platform.python_version(), "machine": platform.machine(),
            "threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available()}


def _write(path: str, record: dict) -> None:
    text = json.dumps(_finite(dict(record, environment=_environment())), indent=1, sort_keys=True, allow_nan=False)
    with open(path, "w", encoding="utf-8") as handle:           # explicit offline artifact, requested by --out
        handle.write(text + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["train", "eval", "ablate", "memory", "r3base", "r3eval", "smoke"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--max-length", type=int, default=1 << 20)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--length", type=int, default=1 << 20)
    parser.add_argument("--sets", default=None)
    parser.add_argument("--lengths", default=None, help="eval: comma-separated lengths (default: the ladder)")
    parser.add_argument("--streams", type=int, default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--threads", type=int, default=4)
    for name in ("per_block", "block", "coarse_top", "evidence"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=None)
    args = parser.parse_args(argv)
    args.smoke = args.command == "smoke"
    torch.set_num_threads(args.threads)
    result = {"train": command_train, "eval": command_eval, "ablate": command_ablate, "memory": command_memory,
              "r3base": command_r3base, "r3eval": command_r3eval, "smoke": command_smoke}[args.command](args)
    if args.command == "memory" and args.out:
        _write(args.out, result)
    if not args.out or args.command == "smoke":
        print(json.dumps(_finite(result if args.command != "train" else {k: v for k, v in result.items()
                                                                          if k != "checkpoint"}), indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
