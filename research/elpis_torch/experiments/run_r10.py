"""ELPIS_TORCH_R10 — CGC complementarity and in-context adaptation. RESEARCH_ONLY.

Does a CGC-augmented Elpis transformer adapt to new in-context relationships, and retain earlier conflicting ones,
more reliably than otherwise equivalent FFN-only transformers, WITHOUT inference-time weight updates?

The assay is ``adaptation_tasks`` (families A, B, C). Every arm is R9's d 64 backbone (R7's study model: two ERSA
blocks, routed channels in both, R7's PLE) trained from its own seed on the same episode stream, with the loss on
answer positions only; nothing is updated at evaluation (tested). Arms differ only where named:

    A      routed experts FFN 86 + CGC                      window attention (64 tokens) in both blocks
    B      routed experts FFN 86, no CGC                    same attention
    C      routed experts FFN 123 (parameters matched)      same attention
    C_mac  routed experts FFN 126 (multiply-adds matched)   same attention
    D_A    as A, attention-free (ERSA only)                 the attention-restricted pair
    D_C    as C, attention-free (ERSA only)

R5 history is off (no evidence head). The PLE is identical in every arm.

Commands
    plan       the arms' EXACT accounting and the frozen evaluation sets' digests (no training)
    pilot      PRE-REGISTRATION calibration: one arm, development seeds and development episodes only
    train      one preregistered run: train, evaluate every frozen set, run the diagnostics, record
    latency    MEASURED decode latency per arm, interleaved, one thread (untrained seed-0 weights)
    combine    the preregistered screening and confirmation analyses over the run records
    manifest   SHA-256 of every R10 record
    smoke      every path once, tiny

Records go only where ``--out`` says.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode

from .. import adaptation_tasks as at
from .. import lm_training as lmt
from ..cgc import CGCTransform
from ..ersa_model import ErsaConfig, ErsaLM
from ..sparse_channel import active_parameters
from ..training import TrainConfig
from . import run_r7 as r7

PHASE = "ELPIS_TORCH_R10_CGC_ADAPTATION"
REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "research" / "elpis_torch" / "results" / "r10"
PREREGISTRATION = RESULTS / "preregistration.json"
RUNNER = Path(__file__).resolve()
CHUNK = 32                                     # tokens per forward call (window attention needs window >= chunk + 1)
WINDOW = 64
LM = lmt.LMConfig(seq=256, chunk=CHUNK)
K_GRID = (0, 1, 2, 4, 8, 16)
K_ADAPT = (1, 2, 4, 8, 16)                     # k = 0 is unknowable by construction: reported, not averaged

ARMS = {
    "A": {"kind": "ffn+cgc", "ffn_hidden": 86, "attention": "window", "role": "FFN + CGC (the complementary design)"},
    "B": {"kind": "ffn", "ffn_hidden": 86, "attention": "window", "role": "same FFN, CGC absent (incremental)"},
    "C": {"kind": "ffn", "ffn_hidden": 123, "attention": "window", "role": "wider FFN, total parameters matched"},
    "C_mac": {"kind": "ffn", "ffn_hidden": 126, "attention": "window", "role": "wider FFN, multiply-adds matched"},
    "D_A": {"kind": "ffn+cgc", "ffn_hidden": 86, "attention": "none", "role": "A, attention-free (ERSA only)"},
    "D_C": {"kind": "ffn", "ffn_hidden": 123, "attention": "none", "role": "C, attention-free (ERSA only)"},
}
EVAL_COUNTS = {"main": 256, "diagnostic": 128, "dev": 32}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen() -> dict:
    """The preregistration, checked against this code's constants (the plan and the code cannot drift)."""
    plan = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if plan["phase"] != PHASE:
        raise ValueError("not the R10 preregistration")
    drift = {}
    if plan["arms"] != ARMS:
        drift["arms"] = "differ"
    spec = json.loads(json.dumps(at.TaskSpec().to_dict()))          # as JSON stores it (tuples become lists)
    if plan["task_spec"] != spec or plan["task_spec_digest"] != at.TaskSpec().digest():
        drift["task_spec"] = "differs"
    if plan["model"]["window"] != WINDOW or plan["model"]["chunk"] != CHUNK or plan["training"]["seq"] != LM.seq:
        drift["model"] = "differs"
    if list(plan["evaluation"]["k_grid"]) != list(K_GRID) or list(plan["evaluation"]["k_adapt"]) != list(K_ADAPT):
        drift["k"] = "differs"
    if not set(plan.get("confirmation_arms", ())) <= set(ARMS) or "A" not in plan.get("confirmation_arms", ()):
        drift["confirmation_arms"] = "invalid"
    if drift:
        raise ValueError(f"the preregistration does not describe this runner: {drift}")
    return plan


def provenance(plan: dict | None) -> dict:
    def git(*args):
        run = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
        return run.stdout.strip() if run.returncode == 0 else ""
    dirty = bool(git("status", "--porcelain", "--", ".", ":(exclude)research/elpis_torch/results"))
    return {"preregistration_sha256": sha256_file(PREREGISTRATION) if plan is not None else None,
            "runner_sha256": sha256_file(RUNNER), "tasks_sha256": sha256_file(Path(at.__file__)),
            "task_spec_digest": at.TaskSpec().digest(),
            "git_revision": git("rev-parse", "HEAD") + ("+dirty" if dirty else ""), "torch": torch.__version__,
            "python": platform.python_version(), "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "threads": torch.get_num_threads()}


# -- arms ------------------------------------------------------------------------------------------------------------

def build(arm: dict, seed: int, vocab: int = 259) -> ErsaLM:
    """R9's backbone (R7's study model) with the arm's routed-channel anatomy and attention."""
    sparse = dataclasses.replace(r7.STUDY_SPARSE, kind=arm["kind"], ffn_hidden=arm["ffn_hidden"])
    attention = {"attention": "window", "attn_heads": 4, "attn_kv_heads": 2, "attn_window": WINDOW} \
        if arm["attention"] == "window" else {}
    return ErsaLM(ErsaConfig(r7._widths(seed, vocab), conv_kernel=4, chunk=CHUNK, ersa_kernel="chunked",
                             scan_block=CHUNK, sparse=sparse, ple=r7.STUDY_PLE, **attention))


def _group(name: str) -> str:
    if name.endswith("cgc_gate"):
        return "cgc_gate"
    if ".cgc." in name:
        return "cgc"
    if ".experts." in name:
        return "channel_ffn"
    if ".router." in name:
        return "routing"
    if ".attn." in name:
        return "attention"
    if ".ersa." in name:
        return "ersa"
    if name.startswith("ple.table"):
        return "ple_table"
    if name.startswith("ple."):
        return "ple_other"
    if name.startswith("embed."):
        return "embedding"
    return "norms_and_head"


def exact_accounting(model) -> dict:
    """EXACT: parameter groups, resident and active parameters, routed multiply-adds per token (the CGC's cubic terms
    included), and the per-stream state bytes."""
    groups = {}
    for name, p in model.named_parameters():
        groups[_group(name)] = groups.get(_group(name), 0) + p.numel()
    counts = active_parameters(model)
    d = model.embed.weight.shape[1]
    channels = [b.sparse for b in model.blocks if getattr(b, "sparse", None) is not None]
    s = channels[0].config
    expert = channels[0].experts[0]
    ffn_macs = sum(p.numel() for p in expert.ffn.parameters() if p.ndim >= 2)
    cgc_macs = 0 if expert.cgc is None else CGCTransform.multiply_accumulates(
        d, expert.cgc.core.channels, expert.cgc.core.units,
        None if expert.cgc.proj_in is None else expert.cgc.proj_in.out_features)
    state = model.new_state(1).nbytes()
    return {"parameters_by_group": groups, "total_parameters": counts["total"],
            "active_parameters_per_token": counts["active_per_token"],
            "expert": {"kind": s.kind, "ffn_hidden": s.ffn_hidden, "parameters": sum(p.numel() for p in
                                                                                       expert.parameters()),
                       "ffn_macs": ffn_macs, "cgc_macs_including_cubic": cgc_macs, "macs": ffn_macs + cgc_macs},
            "channel_macs_per_token": len(channels) * (s.top_k * (ffn_macs + cgc_macs) + s.experts * d),
            "state_bytes_per_stream": state,
            "ple_table_bytes": groups.get("ple_table", 0) * model.embed.weight.element_size(),
            "labels": {"all": "EXACT"}}


def cgc_gates(model) -> list:
    return [float(e.cgc_gate.detach()) for b in model.blocks if getattr(b, "sparse", None) is not None
            for e in b.sparse.experts if e.cgc_gate is not None]


@torch.no_grad()
def set_cgc_gates(model, values) -> list:
    previous = cgc_gates(model)
    gates = [e.cgc_gate for b in model.blocks if getattr(b, "sparse", None) is not None
             for e in b.sparse.experts if e.cgc_gate is not None]
    for gate, value in zip(gates, values if isinstance(values, list) else [values] * len(gates)):
        gate.fill_(float(value))
    return previous


# -- data ------------------------------------------------------------------------------------------------------------

def _tensors(episodes) -> tuple:
    """``(inputs [n, L], targets [n, L])``: targets are the answers at probe positions and IGNORE elsewhere."""
    inputs = torch.tensor([e.tokens[:-1] for e in episodes], dtype=torch.long)
    targets = torch.full_like(inputs, lmt.IGNORE)
    for row, e in enumerate(episodes):
        for p in e.probes:
            targets[row, p.pos] = p.answer
    return inputs, targets


def batch_stream(run_seed: int, spec=at.TaskSpec(), rows: int = 16, signatures: set | None = None):
    """Step s, row j: ``train_episode(run_seed, s, j)``; every arm of a seed sees exactly these batches."""
    step = 0
    while True:
        episodes = [at.train_episode(run_seed, step, row, spec) for row in range(rows)]
        if signatures is not None:
            signatures.update(e.signature for e in episodes)
        yield _tensors(episodes)
        step += 1


def stage_of(plan: dict, seed: int) -> str:
    for stage, seeds in plan.get("seeds", {}).items():
        if seed in seeds:
            return stage
    return "development"


def eval_sets(plan_eval: dict, which: str, *, stage: str = "screening", smoke: bool = False) -> dict:
    """The frozen evaluation sets of a split (``main``, ``heldout``, ``diagnostic`` or ``dev``) for a stage: the
    screening and confirmation stages evaluate on independent episodes (their own split seeds)."""
    seed = plan_eval["split_seeds"][stage][which]
    count = 4 if smoke else plan_eval["counts"]["diagnostic" if which == "diagnostic" else
                                                ("dev" if which == "dev" else "main")]
    variants = plan_eval["variants"][which]
    return {v: at.eval_set(v, count, split_seed=seed) for v in variants}


# -- evaluation (prequential: teacher-forced, every answer scored before it is revealed) ------------------------------

def _reset(state, what: str) -> None:
    if what == "none":                                       # a chunk boundary only (chunking stability)
        return
    if what == "ersa":
        for c in state.cells:
            c.zero_()
        for t in state.tails:
            if t is not None:
                t.zero_()
    elif what == "attention":
        if state.window is not None:
            state.window.reset()
    else:
        raise ValueError(what)


@torch.no_grad()
def predict(model, episodes, *, reset_at: int | None = None, reset: str | None = None, batch: int = 64) -> list:
    """Per probe (in episode order): ``(correct, prediction, nats)``. ``reset_at``: a token index where the named
    state is cleared (every episode of the set shares it)."""
    model.eval()
    out = []
    for start in range(0, len(episodes), batch):
        part = episodes[start:start + batch]
        inputs, _ = _tensors(part)
        state = model.new_state(inputs.shape[0])
        logits = []
        bounds = list(range(0, inputs.shape[1], CHUNK))
        if reset_at is not None:
            bounds = sorted(set(bounds) | {reset_at})
        bounds.append(inputs.shape[1])
        for a, b in zip(bounds[:-1], bounds[1:]):
            if reset_at is not None and a == reset_at:
                _reset(state, reset)
            logits.append(model(inputs[:, a:b], state).float())
        logits = torch.cat(logits, dim=1)
        logp = F.log_softmax(logits, dim=-1)
        for row, e in enumerate(part):
            for p in e.probes:
                guess = int(logits[row, p.pos].argmax())
                out.append((guess == p.answer, guess, float(-logp[row, p.pos, p.answer])))
    return out


def _mean(values) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def summarize(episodes, results) -> dict:
    """Accuracy by k, by context condition and by kind, collision and recency errors, bits per answer."""
    flat = [p for e in episodes for p in e.probes]
    assert len(flat) == len(results)
    pairs = list(zip(flat, results))
    family = flat[0].family if flat else None
    out = {"answers": len(pairs), "accuracy": _mean(c for _, (c, _, _) in pairs),
           "bits_per_answer": _mean(n for _, (_, _, n) in pairs) / math.log(2) if pairs else None,
           "oracle": _mean(at.oracle_correct(p) for p in flat)}
    by_k = {}
    for k in K_GRID:
        sel = [c for p, (c, _, _) in pairs if p.k == k and p.kind in ("query", "pre")]
        by_k[str(k)] = {"accuracy": _mean(sel), "n": len(sel),
                        "oracle": _mean(at.oracle_correct(p) for p in flat if p.k == k and p.kind in ("query", "pre"))}
    out["by_k"] = by_k
    adapt = [by_k[str(k)]["accuracy"] for k in K_ADAPT if by_k[str(k)]["accuracy"] is not None]
    out["adapt_auc"] = _mean(adapt) if len(adapt) == len(K_ADAPT) else None
    if family in ("A", "C"):
        seen = [(p, res) for p, res in pairs if p.k >= 1 and p.kind in ("query", "pre")]
        out["inside_window"] = _mean(c for p, (c, _, _) in seen if p.distance is not None and p.distance <= WINDOW - 4)
        out["outside_window"] = _mean(c for p, (c, _, _) in seen if p.distance is not None and p.distance > WINDOW)
        wrong = [(p, g) for p, (c, g, _) in seen if not c]
        out["collision_error_rate"] = _mean(1.0 if g in p.conflicts else 0.0 for p, g in wrong) if wrong else 0.0
        out["recency_error_rate"] = _mean(1.0 if (p.latest is not None and g == p.latest and p.latest != p.answer)
                                          else 0.0 for p, (c, g, _) in seen)
    if family == "B":
        flags = []                                  # the last example before a regime switch
        for e in episodes:
            for n, p in enumerate(e.probes):
                nxt = e.probes[n + 1] if n + 1 < len(e.probes) else None
                flags.append(nxt is not None and nxt.segment != p.segment)
        out["pre_switch"] = _mean(c for (p, (c, _, _)), f in zip(pairs, flags) if f)
        out["post_switch_new_regime"] = _mean(c for p, (c, _, _) in pairs if p.since_switch == 0 and p.k == 0)
        out["post_switch_reentry"] = _mean(c for p, (c, _, _) in pairs if p.since_switch == 0 and p.k > 0)
        out["recovery_1_3"] = _mean(c for p, (c, _, _) in pairs if 1 <= p.since_switch <= 3)
    if family == "C":
        for kind in ("pre", "post", "mid", "reentry"):
            out[kind] = _mean(c for p, (c, _, _) in pairs if p.kind == kind)
        posts = [(p, g) for p, (c, g, _) in pairs if p.kind == "post"]
        out["post_recency_error_rate"] = _mean(1.0 if g == p.latest else 0.0 for p, g in posts)
        out["post_any_conflict_rate"] = _mean(1.0 if g in p.conflicts else 0.0 for p, g in posts)
        if out["pre"] is not None and out["post"] is not None:
            out["forgetting"] = out["pre"] - out["post"]
    return out


def evaluate(model, sets: dict, **kwargs) -> dict:
    return {v: summarize(eps, predict(model, eps, **kwargs)) for v, eps in sets.items()}


def endpoints(main: dict) -> dict:
    """The preregistered primary endpoints from the main-set summaries."""
    adapt = [main[v]["adapt_auc"] for v in ("A", "B") if v in main and main[v]["adapt_auc"] is not None]
    retain = [main[v]["post"] for v in ("C_short", "C_long", "C_gap") if v in main and main[v]["post"] is not None]
    pre = [main[v]["pre"] for v in ("C_short", "C_long", "C_gap") if v in main and main[v]["pre"] is not None]
    return {"ADAPT": _mean(adapt) if len(adapt) == 2 else None, "RETAIN": _mean(retain) if len(retain) == 3 else None,
            "RETAIN_pre": _mean(pre) if len(pre) == 3 else None,
            "RETAIN_change": (_mean(retain) - _mean(pre)) if len(retain) == 3 and len(pre) == 3 else None}


def diagnostics(model, arm: dict, diag: dict) -> dict:
    """Post-training interventions on the diagnostic sets (causal dependence, not matched controls)."""
    out = {}
    rng = random.Random(4242)
    c_sets = {v: diag[v] for v in ("C_short", "C_long", "C_gap") if v in diag}
    for v, eps in c_sets.items():
        boundary = at.first_probe_index(eps[0])
        assert all(at.first_probe_index(e) == boundary for e in eps)
        out[f"{v}/reset_ersa_before_probes"] = summarize(eps, predict(model, eps, reset_at=boundary, reset="ersa"))
        if arm["attention"] == "window":
            out[f"{v}/reset_attention_before_probes"] = summarize(
                eps, predict(model, eps, reset_at=boundary, reset="attention"))
    for v in ("A", "B"):
        shuffled = [at.with_shuffled_targets(e, rng) for e in diag[v]]
        out[f"{v}/shuffled_support_targets"] = summarize(shuffled, predict(model, shuffled))
    for v in ("A", "B", "C_long"):
        stripped = [at.without_regime_identity(e) for e in diag[v]]
        out[f"{v}/no_regime_identity"] = summarize(stripped, predict(model, stripped))
    irrelevant = [at.with_irrelevant_demonstrations(e, rng) for e in diag["C_long"]]
    out["C_long/irrelevant_demonstrations"] = summarize(irrelevant, predict(model, irrelevant))
    conflicting = [at.with_conflicting_recent(e) for e in diag["C_long"]]
    out["C_long/conflicting_recent"] = summarize(conflicting, predict(model, conflicting))
    if arm["kind"] == "ffn+cgc":
        previous = set_cgc_gates(model, 0.0)
        out["zero_cgc"] = {v: summarize(eps, predict(model, eps)) for v, eps in diag.items()}
        set_cgc_gates(model, previous)
    return {k: _slim(v) for k, v in out.items()}


def _slim(summary: dict) -> dict:
    if "by_k" not in summary:                              # zero_cgc: a dict of summaries
        return {k: _slim(v) for k, v in summary.items()}
    return summary


@torch.no_grad()
def measured_flops(model, inputs) -> dict:
    model.eval()
    with FlopCounterMode(display=False) as counter:
        state = model.new_state(inputs.shape[0])
        for a in range(0, inputs.shape[1], CHUNK):
            model(inputs[:, a:a + CHUNK], state)
    total = counter.get_total_flops()
    return {"matmul_flops_per_token": total / inputs.numel(), "label": "MEASURED (FlopCounterMode, forward)"}


def _parameters_digest(model) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


# -- commands ----------------------------------------------------------------------------------------------------------

def _plan_or_pilot(args) -> dict:
    return frozen() if not getattr(args, "pilot", False) else json.loads(json.dumps(PILOT_PLAN))


PILOT_PLAN = {
    "phase": PHASE, "training": {"steps": 1500, "lr": 3e-3, "warmup_steps": 50, "min_lr_fraction": 0.1,
                                 "weight_decay": 0.01, "grad_clip": 1.0, "eval_every": 250, "threads": 2,
                                 "seq": LM.seq, "batch": 16},
    "evaluation": {"split_seeds": {"development": {"main": 9_100, "heldout": 9_101, "diagnostic": 9_102,
                                                   "dev": 9_103}},
                   "counts": EVAL_COUNTS,
                   "variants": {"main": list(at.TRAIN_VARIANTS), "heldout": list(at.HELDOUT_VARIANTS),
                                "diagnostic": list(at.TRAIN_VARIANTS), "dev": ["A", "B", "C_long"]}},
}


def _train_and_evaluate(arm_name: str, seed: int, plan: dict, *, smoke: bool, label: str) -> dict:
    arm = ARMS[arm_name]
    t = plan["training"]
    torch.set_num_threads(t["threads"])
    model = build(arm, seed)
    record = {"phase": PHASE, "command": label, "arm": arm_name, "role": arm["role"], "seed": seed,
              "provenance": provenance(None if label == "pilot" else plan), "accounting": exact_accounting(model),
              "task_spec": at.TaskSpec().to_dict()}
    ev = plan["evaluation"]
    stage = stage_of(plan, seed) if label == "train" else "development"
    record["stage"] = stage
    dev = eval_sets(ev, "dev", stage=stage, smoke=smoke)
    steps = 6 if smoke else t["steps"]
    curve = []

    def log(entry):
        point = {"step": entry["step"], "train_loss": entry["loss"]}
        summary = evaluate(model, dev)
        point["dev"] = {v: {k: s.get(k) for k in ("accuracy", "adapt_auc", "post", "pre")} for v, s in summary.items()}
        if arm["kind"] == "ffn+cgc":
            point["cgc_gate_mean"] = _mean(cgc_gates(model))
        curve.append(point)
        model.train()

    signatures = set()
    config = TrainConfig(steps=steps, batch_size=16, lr=t["lr"], warmup_steps=min(t["warmup_steps"], steps),
                         min_lr_fraction=t["min_lr_fraction"], weight_decay=t["weight_decay"],
                         grad_clip=t["grad_clip"], seed=seed, eval_every=3 if smoke else t["eval_every"])
    started = time.perf_counter()
    result = lmt.train_lm(model, batch_stream(seed, signatures=signatures), config, LM, log=log)
    record["train"] = {"status": result["status"], "steps": result["steps"],
                       "seconds": round(time.perf_counter() - started, 1), "episodes": result["steps"] * 16,
                       "distinct_episode_signatures": len(signatures)}
    record["curve"] = curve
    record["status"] = "COMPLETED" if result["status"] == "COMPLETED" and result["steps"] == steps else "FAILED"
    if record["status"] != "COMPLETED":
        return record
    model.eval()
    before = _parameters_digest(model)
    started = time.perf_counter()
    main = eval_sets(ev, "main", stage=stage, smoke=smoke)
    heldout = eval_sets(ev, "heldout", stage=stage, smoke=smoke)
    record["eval_set_digests"] = {f"main/{v}": at.set_digest(e) for v, e in main.items()} | \
        {f"heldout/{v}": at.set_digest(e) for v, e in heldout.items()}
    overlap = sum(1 for eps in list(main.values()) + list(heldout.values()) for e in eps if e.signature in signatures)
    record["train_test_signature_overlap"] = overlap
    record["main"] = evaluate(model, main)
    record["heldout"] = evaluate(model, heldout)
    record["endpoints"] = endpoints(record["main"])
    record["heldout_endpoints"] = {"B_heldout_adapt": record["heldout"].get("B_heldout", {}).get("adapt_auc"),
                                   "C_gap128_post": record["heldout"].get("C_gap128", {}).get("post"),
                                   "C_4reg_post": record["heldout"].get("C_4reg", {}).get("post")}
    if label != "pilot":
        diag = eval_sets(ev, "diagnostic", stage=stage, smoke=smoke)
        record["diagnostics"] = diagnostics(model, arm, diag)
    if arm["kind"] == "ffn+cgc":
        record["cgc_gates_final"] = cgc_gates(model)
    record["evaluation_seconds"] = round(time.perf_counter() - started, 1)
    record["parameters_unchanged_by_evaluation"] = _parameters_digest(model) == before
    record["flops"] = measured_flops(model, _tensors(main["A"][:4])[0])
    record["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return record


def command_pilot(args) -> dict:
    args.pilot = True
    plan = _plan_or_pilot(args)
    if getattr(args, "steps", None):
        plan["training"]["steps"] = args.steps
    record = _train_and_evaluate(args.arm, args.seed, plan, smoke=False, label="pilot")
    record["label"] = "PILOT (pre-registration calibration; development seeds and development episodes only)"
    return record


def command_train(args, *, smoke: bool = False) -> dict:
    plan = frozen()
    if args.arm not in plan["arms"]:
        raise SystemExit(f"unknown arm {args.arm}")
    if stage_of(plan, args.seed) == "development":
        raise SystemExit(f"seed {args.seed} is not a preregistered screening or confirmation seed")
    if stage_of(plan, args.seed) == "confirmation" and args.arm not in plan["confirmation_arms"]:
        raise SystemExit(f"arm {args.arm} is not a confirmation arm")
    return _train_and_evaluate(args.arm, args.seed, plan, smoke=smoke, label="train")


def command_plan(args) -> dict:
    out = {}
    with torch.device("meta"):
        for name, arm in ARMS.items():
            out[name] = exact_accounting_meta(build(arm, 0))
    return {"phase": PHASE, "command": "plan", "arms": out, "task_spec_digest": at.TaskSpec().digest()}


def exact_accounting_meta(model) -> dict:
    groups = {}
    for name, p in model.named_parameters():
        groups[_group(name)] = groups.get(_group(name), 0) + p.numel()
    counts = active_parameters(model)
    expert = model.blocks[0].sparse.experts[0]
    d = 64
    ffn_macs = sum(p.numel() for p in expert.ffn.parameters() if p.ndim >= 2)
    cgc_macs = 0 if expert.cgc is None else CGCTransform.multiply_accumulates(
        d, expert.cgc.core.channels, expert.cgc.core.units, expert.cgc.proj_in.out_features)
    return {"parameters_by_group": groups, "total_parameters": counts["total"],
            "active_parameters_per_token": counts["active_per_token"],
            "expert_macs": ffn_macs + cgc_macs, "expert_parameters": sum(p.numel() for p in expert.parameters())}


@torch.no_grad()
def command_latency(args, *, smoke: bool = False) -> dict:
    """MEASURED: decode milliseconds per token for every arm, one thread, untrained seed-0 weights, interleaved in a
    rotating order after a 128-token prefill of the same family-A episode."""
    plan = frozen()
    torch.set_num_threads(1)
    episode = at.eval_set("A", 1, split_seed=plan["evaluation"]["split_seeds"]["screening"]["dev"])[0]
    tokens = torch.tensor([episode.tokens[:-1]], dtype=torch.long)
    models = {name: build(arm, 0).eval() for name, arm in ARMS.items()}
    steps = 8 if smoke else plan["latency"]["decode_steps"]
    samples = {name: [] for name in models}
    states = {}
    for name, model in models.items():
        states[name] = model.new_state(1)
        for a in range(0, 128, CHUNK):
            model(tokens[:, a:a + CHUNK], states[name])
    order = list(models)
    for step in range(steps):
        token = tokens[:, 128 + step:129 + step]
        for name in order[step % len(order):] + order[:step % len(order)]:
            started = time.perf_counter()
            models[name](token, states[name])
            samples[name].append(time.perf_counter() - started)
    out = {}
    for name, values in samples.items():
        ordered = sorted(values)
        out[name] = {"decode_ms_p50": 1000 * ordered[len(ordered) // 2],
                     "decode_ms_p95": 1000 * ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
                     "samples": len(values)}
    return {"phase": PHASE, "command": "latency", "provenance": provenance(plan), "arms": out,
            "label": "MEASURED, one thread, this VM, untrained seed-0 weights; PERFORMANCE_ONLY"}


# -- the preregistered analysis --------------------------------------------------------------------------------------

def _classify(differences: list, delta: float) -> str:
    mean = statistics.fmean(differences)
    if all(v > 0 for v in differences) and mean >= delta:
        return "A_BETTER"
    if all(v < 0 for v in differences) and mean <= -delta:
        return "X_BETTER"
    return "NOT_DISTINGUISHED"


def _paired(results: dict, endpoint: str, ref: str, other: str, seeds) -> list | None:
    values = []
    for s in seeds:
        a, b = results.get((ref, s)), results.get((other, s))
        if a is None or b is None or a.get(endpoint) is None or b.get(endpoint) is None:
            return None
        values.append(a[endpoint] - b[endpoint])
    return values


def _t(values: list) -> float | None:
    if len(values) < 2:
        return None
    sd = statistics.stdev(values)
    return math.inf if sd == 0 else statistics.fmean(values) / (sd / math.sqrt(len(values)))


def analyse(results: dict, seeds, analysis: dict, stage: str) -> dict:
    """``results[(arm, seed)]`` = the run's endpoints (and held-out endpoints). The rules of
    PREREGISTRATION_R10.md section 6, applied mechanically."""
    out = {"stage": stage, "seeds": list(seeds), "endpoints": {}}
    for endpoint, delta in (("ADAPT", analysis["delta_adapt"]), ("RETAIN", analysis["delta_retain"])):
        e = {"delta_min": delta, "comparisons": {}}
        for ref, other in (("A", "C"), ("A", "C_mac"), ("A", "B"), ("D_A", "D_C")):
            diffs = _paired(results, endpoint, ref, other, seeds)
            if diffs is None:
                e["comparisons"][f"{ref}-{other}"] = {"status": "MISSING"}
                continue
            entry = {"per_seed": diffs, "mean": statistics.fmean(diffs), "positive": sum(d > 0 for d in diffs),
                     "class": _classify(diffs, delta), "t": _t(diffs)}
            e["comparisons"][f"{ref}-{other}"] = entry
        def best(key):
            means = [statistics.fmean(results[(arm, s)][key] for s in seeds) for arm in ("A", "B", "C", "C_mac")
                     if all((arm, s) in results and results[(arm, s)].get(key) is not None for s in seeds)]
            return max(means) if means else None
        threshold = analysis["learned_threshold"]
        if endpoint == "ADAPT":
            e["best_primary_arm_mean"] = best("ADAPT")
            e["learned"] = e["best_primary_arm_mean"] is not None and e["best_primary_arm_mean"] >= threshold["ADAPT"]
        else:                          # the bindings were learned AND retention itself is measurable in some arm
            e["best_primary_arm_mean"] = best("RETAIN")
            e["best_primary_arm_pre_mean"] = best("RETAIN_pre")
            e["learned"] = (e["best_primary_arm_mean"] is not None and e["best_primary_arm_pre_mean"] is not None
                            and e["best_primary_arm_mean"] >= threshold["RETAIN"]
                            and e["best_primary_arm_pre_mean"] >= threshold["RETAIN_pre"])
        c, cm = e["comparisons"].get("A-C", {}), e["comparisons"].get("A-C_mac", {})
        if not e["learned"]:
            e["verdict"] = "ASSAY_NOT_LEARNED"
        elif "mean" not in c or "mean" not in cm:
            e["verdict"] = "NONQUAL"
        elif stage == "screening":
            favorable = all(x["mean"] >= delta and x["positive"] >= analysis["screen_min_positive"]
                            for x in (c, cm))
            e["verdict"] = "SCREEN_FAVORABLE" if favorable else "SCREEN_NOT_FAVORABLE"
        else:
            t_crit = analysis["t_critical_one_sided"]
            confirmed = all(x["mean"] >= delta and x["t"] is not None and x["t"] >= t_crit for x in (c, cm))
            e["verdict"] = "CONFIRMED" if confirmed else "NOT_CONFIRMED"
            held_key = "heldout_adapt" if endpoint == "ADAPT" else "heldout_retain"
            held = _paired(results, held_key, "A", "C", seeds)
            e["heldout_A-C"] = None if held is None else {"mean": statistics.fmean(held), "per_seed": held}
            if confirmed and (held is None or statistics.fmean(held) <= 0):
                e["verdict"] = "CONFIRMED_IN_DISTRIBUTION_ONLY"
            d = e["comparisons"].get("D_A-D_C", {})
            e["attribution"] = None if "mean" not in d else (
                "PERSISTS_WITHOUT_ATTENTION" if d["mean"] >= delta / 2 else "ATTENTION_DEPENDENT")
        out["endpoints"][endpoint] = e
    return out


def complete_blocks(results: dict, seeds, arms) -> list:
    """The seeds for which every listed arm has a completed run (PREREGISTRATION_R10.md section 6)."""
    return [s for s in seeds if all((a, s) in results for a in arms)]


def overall(out: dict, plan: dict, latency: dict | None) -> dict:
    """Map the stage verdicts to the preregistered overall verdicts (section 5), mechanically."""
    verdicts = {}
    ratio = None
    if latency is not None:
        arms = latency.get("arms", {})
        if "A" in arms and "C" in arms:
            ratio = arms["A"]["decode_ms_p50"] / arms["C"]["decode_ms_p50"]
    for endpoint, name in (("ADAPT", "ADAPTATION"), ("RETAIN", "RETENTION")):
        screen = out["screening"]["endpoints"][endpoint]["verdict"]
        conf = out.get("confirmation")
        v = {"screening": screen}
        if screen in ("ASSAY_NOT_LEARNED", "NONQUAL"):
            v["verdict"] = screen
        elif screen == "SCREEN_NOT_FAVORABLE":
            v["verdict"] = f"NO_INCREMENTAL_CGC_{name}_BENEFIT_ESTABLISHED"
        elif conf is None or conf.get("status") == "PARTIAL":
            v["verdict"] = "PARTIAL"                  # favorable screening without a complete confirmation
        else:
            c = conf["endpoints"][endpoint]
            v["confirmation"] = c["verdict"]
            if c["verdict"] in ("CONFIRMED", "CONFIRMED_IN_DISTRIBUTION_ONLY"):
                v["verdict"] = f"CGC_{name}_BENEFIT_CONFIRMED"
                v["held_out"] = "SURVIVES" if c["verdict"] == "CONFIRMED" else "IN_DISTRIBUTION_ONLY"
                v["attribution"] = c.get("attribution")
                v["latency_ratio_A_over_C"] = ratio
                v["latency"] = None if ratio is None else (
                    "QUALITY_LATENCY_TRADEOFF" if ratio > plan["analysis"]["latency_material_ratio"] else "NOT_MATERIAL")
            elif c["verdict"] in ("ASSAY_NOT_LEARNED", "NONQUAL"):
                v["verdict"] = c["verdict"]
            else:
                v["verdict"] = f"NO_INCREMENTAL_CGC_{name}_BENEFIT_ESTABLISHED"
        verdicts[endpoint] = v
    return verdicts


def command_combine(args) -> dict:
    plan = frozen()
    folder = Path(args.inputs) if args.inputs else RESULTS
    expected = {"preregistration_sha256": sha256_file(PREREGISTRATION), "runner_sha256": sha256_file(RUNNER),
                "tasks_sha256": sha256_file(Path(at.__file__))}
    results, refused, status, inputs = {}, [], {}, {}
    for path in sorted(folder.glob("run_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        prov = record.get("provenance", {})
        if any(prov.get(k) != v for k, v in expected.items()):
            refused.append({"file": path.name, "reason": "digest mismatch"})
            continue
        inputs[path.name] = sha256_file(path)
        key = (record["arm"], record["seed"])
        status[f"{record['arm']}_s{record['seed']}"] = record.get("status")
        if record.get("status") != "COMPLETED":
            continue
        ep = dict(record["endpoints"])
        held = record.get("heldout_endpoints", {})
        ep["heldout_adapt"] = held.get("B_heldout_adapt")
        posts = [v for v in (held.get("C_gap128_post"), held.get("C_4reg_post")) if v is not None]
        ep["heldout_retain"] = _mean(posts) if len(posts) == 2 else None
        results[key] = ep
    analysis = plan["analysis"]
    out = {"phase": PHASE, "command": "combine", "provenance": provenance(plan), "inputs": inputs,
           "refused": refused, "run_status": status,
           "screening": analyse(results, plan["seeds"]["screening"], analysis, "screening")}
    conf_seeds = plan["seeds"]["confirmation"]
    if any((a, s) in results for a in ARMS for s in conf_seeds):
        blocks = complete_blocks(results, conf_seeds, plan["confirmation_arms"])
        conf = analyse(results, blocks, analysis, "confirmation") if len(blocks) >= 2 else {"endpoints": {}}
        conf.update(complete_blocks=blocks, required_blocks=len(conf_seeds),
                    status="COMPLETE" if len(blocks) == len(conf_seeds) else "PARTIAL")
        out["confirmation"] = conf
    latency_path = folder / "latency.json"
    latency = json.loads(latency_path.read_text(encoding="utf-8")) if latency_path.exists() else None
    if latency is not None:
        inputs[latency_path.name] = sha256_file(latency_path)
    out["overall"] = overall(out, plan, latency)
    out["per_run"] = {f"{a}_s{s}": v for (a, s), v in sorted(results.items())}
    return out


def command_manifest(args) -> dict:
    files = sorted(p for p in RESULTS.glob("*.json") if p.name != "MANIFEST.json")
    return {"phase": PHASE, "command": "manifest", "runner_sha256": sha256_file(RUNNER),
            "files": {p.name: sha256_file(p) for p in files}}


def command_smoke(args) -> dict:
    plan = json.loads(json.dumps(PILOT_PLAN))
    out = {}
    for name in ("A", "D_C"):
        record = _train_and_evaluate(name, 0, plan, smoke=True, label="pilot")
        out[name] = {"status": record["status"], "endpoints": record.get("endpoints")}
    return {"phase": PHASE, "command": "smoke", "runs": out}


COMMANDS = {"plan": command_plan, "pilot": command_pilot, "train": command_train, "latency": command_latency,
            "combine": command_combine, "manifest": command_manifest, "smoke": command_smoke}


def _write(path: str, record: dict) -> None:
    Path(path).write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--arm", default="B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None, help="pilot only: training steps")
    parser.add_argument("--inputs", help="combine: the folder holding the run records (default: results/r10)")
    parser.add_argument("--out", help="the JSON record to write (nothing is written without it)")
    args = parser.parse_args(argv)
    record = COMMANDS[args.command](args)
    if args.out:
        _write(args.out, record)
    else:
        json.dump({k: record[k] for k in ("phase", "command", "status", "endpoints") if k in record}, sys.stdout,
                  default=str)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
