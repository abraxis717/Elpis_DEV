"""ELPIS_TORCH_R9 — Boardy adversarial qualification: does CGC earn its place? RESEARCH_ONLY.

The confirmatory CGC attribution study (P2) under the frozen preregistration ``results/r9/preregistration.json``
(``docs/research/elpis_torch/PREREGISTRATION_R9.md``). Every record carries the SHA-256 of that file and of this
runner; ``combine`` refuses records whose digests differ from the files it runs with.

Commands
    accounting  EXACT parameters, resident bytes, active parameters and multiply-adds per token of every arm (meta)
    integrity   arm A, seed 0, under R7's protocol as executed (one epoch, 1,432 steps): must reproduce R7's record
    train       one confirmatory (arm, seed) run of exactly 1,500 steps: learning curve, final held-out bits/byte,
                the gates-zeroed evaluation of a CGC arm (Arm D), gates, gradients, routing, generation mechanics,
                measured matmul FLOPs and peak RSS
    latency     decode and prefill cost of every trainable arm on one thread, arms interleaved step by step
    combine     the study record: per-seed results, paired differences and the frozen classes and verdicts
    manifest    SHA-256 of every R9 record
    smoke       a tiny end-to-end pass for CI (nothing is kept)

Records go only where ``--out`` says.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
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
from torch.utils.flop_counter import FlopCounterMode

from .. import lm_training as lmt
from ..cgc import CGCTransform
from ..ersa_model import ErsaConfig, ErsaLM
from ..sparse_channel import active_parameters
from ..training import TrainConfig
from . import run_r7 as r7

PHASE = "ELPIS_TORCH_R9_BOARDY_QUALIFICATION"
REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "research" / "elpis_torch" / "results" / "r9"
PREREGISTRATION = RESULTS / "preregistration.json"
RUNNER = Path(__file__).resolve()
TOKENIZER = r7.TOKENIZER
LM = lmt.LMConfig(seq=128, chunk=64)
DOMAINS = ("code", "prose")


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen() -> dict:
    """The preregistration, checked against the R7 constants it names (so the plan and the code cannot drift)."""
    plan = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if plan["phase"] != PHASE:
        raise ValueError("not the R9 preregistration")
    m, s, p = plan["model"], r7.STUDY_SPARSE, r7.STUDY_PLE
    widths = r7._widths(0)
    expected = {"d_model": widths.d_model, "n_layers": widths.n_layers, "experts": s.experts, "top_k": s.top_k,
                "sparse_layers": list(s.layers), "cgc_channels": s.cgc_channels, "cgc_bottleneck": s.cgc_bottleneck,
                "cgc_units": widths.cgc_units, "cgc_preact_bound": widths.cgc_preact_bound,
                "cgc_gate_init": s.gate_init}
    drift = {k: (m[k], v) for k, v in expected.items() if m[k] != v}
    ple = {k: m["ple"][k] for k in ("orders", "heads", "rows", "head_dim", "layer", "boundary")}
    if ple != {"orders": list(p.orders), "heads": p.heads, "rows": p.rows, "head_dim": p.head_dim, "layer": p.layer,
               "boundary": p.boundary}:
        drift["ple"] = (ple, p.to_dict())
    if drift:
        raise ValueError(f"the preregistration does not describe R7's study model: {drift}")
    return plan


def provenance(plan) -> dict:
    def git(*args):
        run = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
        return run.stdout.strip() if run.returncode == 0 else ""
    dirty = bool(git("status", "--porcelain", "--", ".", ":(exclude)research/elpis_torch/results"))
    return {"preregistration_sha256": sha256_file(PREREGISTRATION), "runner_sha256": sha256_file(RUNNER),
            "git_revision": git("rev-parse", "HEAD") + ("+dirty" if dirty else ""), "torch": torch.__version__,
            "python": platform.python_version(), "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "threads": torch.get_num_threads()}


# -- arms ----------------------------------------------------------------------------------------------------------

def build(arm: dict, seed: int, vocab: int = 259) -> ErsaLM:
    """A trainable arm: R7's study model with the routed channel's anatomy and width from the preregistration."""
    sparse = dataclasses.replace(r7.STUDY_SPARSE, kind=arm["kind"], ffn_hidden=arm["ffn_hidden"])
    return ErsaLM(ErsaConfig(r7._widths(seed, vocab), conv_kernel=4, chunk=64, ersa_kernel="chunked", scan_block=64,
                             sparse=sparse, ple=r7.STUDY_PLE if arm["ple"] else None))


def trainable(plan) -> list:
    return [name for name, arm in plan["arms"].items() if "kind" in arm]


def _group(name: str) -> str:
    if name.endswith("cgc_gate"):
        return "cgc_gate"
    if ".cgc." in name:
        return "cgc"
    if ".experts." in name:
        return "channel_ffn"
    if ".router." in name:
        return "routing"
    if name.startswith("ple.table"):
        return "ple_table"
    if name.startswith("ple."):
        return "ple_other"
    if name.startswith("embed."):
        return "embedding"
    return "ersa_norms_head"


def exact_accounting(model) -> dict:
    """EXACT: parameter groups, resident bytes, active parameters per token, and multiply-adds per token of the
    routed channel by formula (the CGC's cubic terms included; elementwise work otherwise not counted)."""
    groups = {}
    for name, p in model.named_parameters():
        groups[_group(name)] = groups.get(_group(name), 0) + p.numel()
    counts = active_parameters(model)
    d = model.embed.weight.shape[1]
    channels = [block.sparse for block in model.blocks if getattr(block, "sparse", None) is not None]
    s = channels[0].config
    expert = channels[0].experts[0]
    ffn_macs = sum(p.numel() for p in expert.ffn.parameters() if p.ndim >= 2)
    cgc_macs = 0 if expert.cgc is None else CGCTransform.multiply_accumulates(
        d, expert.cgc.core.channels, expert.cgc.core.units,
        None if expert.cgc.proj_in is None else expert.cgc.proj_in.out_features)
    per_expert = sum(p.numel() for p in expert.parameters())
    element = model.embed.weight.element_size()
    return {"parameters_by_group": groups, "total_parameters": counts["total"],
            "resident_weight_bytes": counts["total"] * element,
            "active_parameters_per_token": counts["active_per_token"],
            "weight_bytes_read_per_token_bound": counts["active_per_token"] * element,
            "expert": {"kind": s.kind, "ffn_hidden": s.ffn_hidden or None, "parameters": per_expert,
                       "ffn_macs": ffn_macs, "cgc_macs_including_cubic": cgc_macs, "macs": ffn_macs + cgc_macs},
            "channel_macs_per_token": len(channels) * (s.top_k * (ffn_macs + cgc_macs) + s.experts * d),
            "labels": {"all": "EXACT"}}


def cgc_gates(model) -> list:
    return [float(e.cgc_gate.detach()) for b in model.blocks if getattr(b, "sparse", None) is not None
            for e in b.sparse.experts if e.cgc_gate is not None]


@torch.no_grad()
def set_cgc_gates(model, values) -> list:
    """Set every CGC gate (in block, then channel order); returns the previous values."""
    previous = cgc_gates(model)
    gates = [e.cgc_gate for b in model.blocks if getattr(b, "sparse", None) is not None
             for e in b.sparse.experts if e.cgc_gate is not None]
    for gate, value in zip(gates, values if isinstance(values, list) else [values] * len(gates)):
        gate.fill_(float(value))
    return previous


def gradient_norms(model) -> dict:
    """Norms of the gradients present now, by parameter group (sparse table gradients by their values)."""
    sums = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            key = _group(name)
            sums[key] = sums.get(key, 0.0) + float(lmt._values(p.grad).double().pow(2).sum())
    return {k: math.sqrt(v) for k, v in sorted(sums.items())}


def held_out_gradient(model, batch) -> dict:
    model.zero_grad(set_to_none=True)
    model.train()
    lmt.batch_loss(model, *batch, LM, backward=True)
    norms = gradient_norms(model)
    model.zero_grad(set_to_none=True)
    model.eval()
    return norms


@torch.no_grad()
def routing(model, batches) -> list:
    """Per sparse block, over every chunk of ``batches``: channel shares, concentration and router entropy."""
    model.eval()
    blocks = [i for i, b in enumerate(model.blocks) if getattr(b, "sparse", None) is not None]
    counts = {i: None for i in blocks}
    entropy = {i: 0.0 for i in blocks}
    tokens = 0
    for inputs, _ in batches:
        state = model.new_state(inputs.shape[0])
        for at in range(0, inputs.shape[1], LM.chunk):
            model(inputs[:, at:at + LM.chunk], state)
            for i in blocks:
                stats = model.blocks[i].sparse.report["stats"]
                c = torch.tensor(stats["counts"], dtype=torch.float64)
                counts[i] = c if counts[i] is None else counts[i] + c
                entropy[i] += stats["router_entropy"] * stats["tokens"]
            tokens += stats["tokens"]
    out = []
    for i in blocks:
        c = counts[i]
        share = c / c.sum()
        out.append({"block": i, "share": [round(float(v), 5) for v in share],
                    "max_over_mean": float(c.max() / c.mean()), "cv": float(c.std(unbiased=False) / c.mean()),
                    "router_entropy": entropy[i] / tokens, "router_entropy_max": math.log(len(c)),
                    "below_1_percent": int((share < 0.01).sum())})
    return out


@torch.no_grad()
def measured_flops(model, batches) -> dict:
    """MEASURED: matmul-class FLOPs counted by FlopCounterMode over forward passes (elementwise work excluded)."""
    model.eval()
    tokens = 0
    with FlopCounterMode(display=False) as counter:
        for inputs, _ in batches:
            state = model.new_state(inputs.shape[0])
            for at in range(0, inputs.shape[1], LM.chunk):
                model(inputs[:, at:at + LM.chunk], state)
            tokens += inputs.numel()
    total = counter.get_total_flops()
    return {"matmul_flops": int(total), "tokens": tokens, "matmul_flops_per_token": total / tokens}


def _mechanics(ids: list, truth: bytes) -> dict:
    data = bytes(t for t in ids if t < 256)
    grams = [data[i:i + 4] for i in range(len(data) - 3)]
    longest = run = 0
    for i, b in enumerate(data):
        run = run + 1 if i and data[i - 1] == b else 1
        longest = max(longest, run)
    lead = 0
    while lead < min(len(data), len(truth)) and data[lead] == truth[lead]:
        lead += 1
    first = min(32, len(truth), len(data))
    return {"special_tokens": sum(1 for t in ids if t >= 256),
            "printable_fraction": sum(1 for b in data if 32 <= b < 127 or b in (9, 10, 13)) / max(1, len(data)),
            "distinct_4gram_fraction": len(set(grams)) / max(1, len(grams)), "longest_byte_run": longest,
            "leading_bytes_matching_source": lead,
            "first_32_bytes_agreement": sum(data[i] == truth[i] for i in range(first)) / max(1, first)}


@torch.no_grad()
def generation(model, documents, spec) -> dict:
    """Mechanics of greedy free-running continuation from fixed held-out prompts (not a quality claim)."""
    rows = []
    for doc in documents[:spec["prompts_per_domain"]]:
        prompt = torch.tensor([TOKENIZER.encode_bytes(doc[:spec["prompt_bytes"]], bos=True)])
        out = lmt.generate(model, prompt, spec["new_tokens"], chunk=LM.chunk, temperature=spec["temperature"])
        truth = doc[spec["prompt_bytes"]:spec["prompt_bytes"] + spec["new_tokens"]]
        rows.append(_mechanics(out[0].tolist(), truth))
    return {"prompts": rows, "mean": {k: statistics.fmean(r[k] for r in rows) for k in rows[0]}}


# -- data ----------------------------------------------------------------------------------------------------------

def load_data(plan, *, smoke: bool = False) -> dict:
    data = r7.corpus(None if smoke else plan["corpus"]["revision"])
    count = 1 if smoke else plan["evaluation"]["val_batches_per_domain"]
    seed = plan["evaluation"]["val_seed"]
    data["val"] = {d: r7._val_batches(data[f"val_{d}"], LM, count, seed=seed) for d in DOMAINS}
    data["rows"] = list(lmt.document_rows(data["train"], TOKENIZER, LM.seq))
    return data


def batch_stream(rows, seed: int):
    """R7's seeded epoch, continued by a second seeded epoch (seed + 1000): as many batches as asked for."""
    return itertools.chain(lmt.document_batches(rows, 16, seed=seed, ignore_pad=TOKENIZER.PAD),
                           lmt.document_batches(rows, 16, seed=seed + 1000, ignore_pad=TOKENIZER.PAD))


def _evaluate(model, data) -> dict:
    return {d: lmt.evaluate_lm(model, data["val"][d], LM)["bits_per_byte"] for d in DOMAINS}


# -- commands ------------------------------------------------------------------------------------------------------

def command_accounting(args) -> dict:
    plan = frozen()
    with torch.device("meta"):
        arms = {name: exact_accounting(build(plan["arms"][name], 0)) for name in trainable(plan)}
    ref = arms["A"]
    for name, a in arms.items():
        a["relative_to_A"] = {
            "total_parameters": a["total_parameters"] / ref["total_parameters"] - 1,
            "active_parameters_per_token": a["active_parameters_per_token"] / ref["active_parameters_per_token"] - 1,
            "expert_parameters": a["expert"]["parameters"] / ref["expert"]["parameters"] - 1,
            "expert_macs": a["expert"]["macs"] / ref["expert"]["macs"] - 1}
    return {"phase": PHASE, "command": "accounting", "provenance": provenance(plan), "arms": arms,
            "note": "Arm D is arm A evaluated with every CGC gate at 0: its parameters and compute are arm A's"}


def command_integrity(args) -> dict:
    plan = frozen()
    spec = plan["integrity"]["r7_reproduction"]
    torch.set_num_threads(plan["training"]["threads"])
    data = load_data(plan)
    identity = {}
    for seed in plan["seeds"]:
        ours, theirs = build(plan["arms"]["A"], seed).state_dict(), r7.study_model("ple", seed).state_dict()
        identity[str(seed)] = ours.keys() == theirs.keys() and all(torch.equal(ours[k], theirs[k]) for k in ours)
    model = r7.study_model("ple", spec["seed"])
    started = time.perf_counter()
    train = r7._train(model, data["rows"], plan["training"]["steps"], spec["seed"], LM)
    found = _evaluate(model, data)
    deviation = {d: abs(found[d] - spec["expected"][d]) for d in DOMAINS}
    ok = all(v <= spec["tolerance_bits_per_byte"] for v in deviation.values()) and all(identity.values())
    return {"phase": PHASE, "command": "integrity", "provenance": provenance(plan), "train": train,
            "seconds": round(time.perf_counter() - started, 1), "found": found, "expected": spec["expected"],
            "deviation": deviation, "construction_identity": identity, "status": "PASS" if ok else "FAIL",
            "corpus_digest": data["digest"]}


def command_train(args, *, smoke: bool = False) -> dict:
    plan = frozen()
    arm = plan["arms"][args.arm]
    if "kind" not in arm:
        raise SystemExit(f"arm {args.arm} is not trained (it is an evaluation of {arm.get('from')})")
    t = plan["training"]
    torch.set_num_threads(t["threads"])
    data = load_data(plan, smoke=smoke)
    record = {"phase": PHASE, "command": "train", "arm": args.arm, "role": arm["role"], "seed": args.seed,
              "provenance": provenance(plan),
              "corpus": {"digest": data["digest"], "revision": data["revision"], **data["summary"]}}
    if not smoke and data["digest"] != plan["corpus"]["digest"]:
        record["status"] = "NONQUAL"
        record["reason"] = "corpus digest differs from the preregistration"
        return record
    model = build(arm, args.seed)
    record["accounting"] = exact_accounting(model)
    steps = 6 if smoke else t["steps"]
    curve = []

    def log(entry):
        point = {"step": entry["step"], "train_loss": entry["loss"], "train_gradient_norms": gradient_norms(model)}
        point.update({f"val_{d}_bits_per_byte": v for d, v in _evaluate(model, data).items()})
        if arm["kind"] == "ffn+cgc":
            point["cgc_gates"] = cgc_gates(model)
        curve.append(point)

    config = TrainConfig(steps=steps, batch_size=16, lr=t["lr"], warmup_steps=min(t["warmup_steps"], steps),
                         min_lr_fraction=t["min_lr_fraction"], weight_decay=t["weight_decay"],
                         grad_clip=t["grad_clip"], seed=args.seed, eval_every=3 if smoke else t["eval_every"])
    started = time.perf_counter()
    result = lmt.train_lm(model, batch_stream(data["rows"], args.seed), config, LM, log=log)
    record["train"] = {"status": result["status"], "steps": result["steps"],
                       "seconds": round(time.perf_counter() - started, 1)}
    record["curve"] = curve
    record["status"] = "COMPLETED" if result["status"] == "COMPLETED" and result["steps"] == steps else "FAILED"
    if record["status"] != "COMPLETED":
        return record
    final = curve[-1]
    record["final"] = {f"val_{d}_bits_per_byte": final[f"val_{d}_bits_per_byte"] for d in DOMAINS}
    if arm["kind"] == "ffn+cgc":
        previous = set_cgc_gates(model, 0.0)
        record["gates_zeroed"] = {f"val_{d}_bits_per_byte": v for d, v in _evaluate(model, data).items()}
        set_cgc_gates(model, previous)
        record["cgc_gates_final"] = previous
    model.eval()
    few = {d: data["val"][d][:2 if not smoke else 1] for d in DOMAINS}
    record["routing"] = {d: routing(model, few[d]) for d in DOMAINS}
    record["held_out_gradient_norms"] = held_out_gradient(model, data["val"]["code"][0])
    spec = dict(plan["generation"], new_tokens=8, prompts_per_domain=1) if smoke else plan["generation"]
    record["generation"] = {d: generation(model, data[f"val_{d}"], spec) for d in DOMAINS}
    record["flops"] = measured_flops(model, few["code"])
    record["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return record


@torch.no_grad()
def command_latency(args, *, smoke: bool = False) -> dict:
    plan = frozen()
    spec = plan["latency"]
    torch.set_num_threads(spec["threads"])
    names = [n for n in trainable(plan) if not smoke or n in ("A", "C")]
    models = {n: build(plan["arms"][n], 0).eval() for n in names}
    data = r7.corpus(None if smoke else plan["corpus"]["revision"])
    prompt = torch.tensor([TOKENIZER.encode_bytes(data["val_code"][0][:spec["prompt_tokens"] - 1], bos=True)])
    prefill = {n: [] for n in names}
    rounds = 1 if smoke else spec["prefill_rounds"]
    for r in range(rounds):
        for n in names[r % len(names):] + names[:r % len(names)]:
            state = models[n].new_state(1)
            started = time.perf_counter()
            for at in range(0, prompt.shape[1], LM.chunk):
                models[n](prompt[:, at:at + LM.chunk], state)
            prefill[n].append((time.perf_counter() - started) * 1000)
    states, tokens = {}, {}
    for n in names:
        states[n] = models[n].new_state(1)
        for at in range(0, prompt.shape[1], LM.chunk):
            logits = models[n](prompt[:, at:at + LM.chunk], states[n])
        tokens[n] = logits[:, -1].argmax(-1)
    decode = {n: [] for n in names}
    for step in range(2 if smoke else spec["decode_steps"]):
        for n in names[step % len(names):] + names[:step % len(names)]:
            started = time.perf_counter()
            logits = models[n](tokens[n][:, None], states[n])
            decode[n].append((time.perf_counter() - started) * 1000)
            tokens[n] = logits[:, -1].argmax(-1)

    def summary(values):
        ordered = sorted(values)
        return {"p50": statistics.median(ordered), "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
                "mean": statistics.fmean(ordered), "samples": len(ordered)}
    out = {n: {"decode_ms": summary(decode[n]), "prefill_ms": summary(prefill[n]),
               "decode_tokens_per_second_p50": 1000 / summary(decode[n])["p50"]} for n in names}
    return {"phase": PHASE, "command": "latency", "provenance": provenance(plan), "arms": out,
            "labels": {"all": "MEASURED (this CPU, one thread, batch 1, untrained weights)"}}


def _classify(differences: list, delta: float) -> str:
    mean = statistics.fmean(differences)
    if all(v > 0 for v in differences) and mean >= delta:
        return "A_BETTER"
    if all(v < 0 for v in differences) and mean <= -delta:
        return "X_BETTER"
    return "NOT_DISTINGUISHED"


def _interval(values: list, t_critical: float) -> list:
    if len(values) < 2:
        return [float("nan"), float("nan")]
    mean, sd = statistics.fmean(values), statistics.stdev(values)
    half = t_critical * sd / math.sqrt(len(values))
    return [mean - half, mean + half]


def _compare(results: dict, other: str, ref: str, seeds, analysis) -> dict:
    out = {}
    for d in DOMAINS:
        key = f"val_{d}_bits_per_byte"
        diffs = [results[(other, s)][key] - results[(ref, s)][key] for s in seeds]
        out[d] = {"per_seed": diffs, "mean": statistics.fmean(diffs),
                  "interval_95": _interval(diffs, analysis["interval"]["t_critical_df2"]),
                  "class": _classify(diffs, analysis["delta_min_bits_per_byte"])}
    return out


def command_combine(args) -> dict:
    plan = frozen()
    analysis, seeds = plan["analysis"], plan["seeds"]
    folder = Path(args.inputs) if args.inputs else RESULTS
    expected = {"preregistration_sha256": sha256_file(PREREGISTRATION), "runner_sha256": sha256_file(RUNNER)}
    records, refused, inputs = {}, [], {}
    for path in sorted(folder.glob("cgc_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        inputs[path.name] = sha256_file(path)
        mismatch = {k: record["provenance"][k] for k in expected if record["provenance"][k] != expected[k]}
        if mismatch or record.get("corpus", {}).get("digest") != plan["corpus"]["digest"]:
            refused.append({"file": path.name, "mismatch": mismatch})
            continue
        records[(record["arm"], record["seed"])] = record
    status = {f"{a}_s{s}": records[(a, s)]["status"] if (a, s) in records else "MISSING"
              for a in trainable(plan) for s in seeds}
    complete = {a: all(status[f"{a}_s{s}"] == "COMPLETED" for s in seeds) for a in trainable(plan)}
    results = {}
    for (a, s), record in records.items():
        if record["status"] == "COMPLETED":
            results[(a, s)] = record["final"]
            if "gates_zeroed" in record:
                results[(a + "_gates_zeroed", s)] = record["gates_zeroed"]
    out = {"phase": PHASE, "command": "combine", "provenance": provenance(plan), "inputs": inputs,
           "refused": refused, "run_status": status,
           "per_seed": {f"{a}_s{s}": v for (a, s), v in sorted(results.items())}}
    comparisons = {}
    for other in ("C", "C_mac", "B"):
        if complete.get(other) and complete.get("A"):
            comparisons[f"{other}_vs_A"] = _compare(results, other, "A", seeds, analysis)
    if complete.get("A"):
        comparisons["D_vs_A"] = _compare(results, "A_gates_zeroed", "A", seeds, analysis)
    if all(complete.get(a) for a in ("A0", "B0", "C0")):
        comparisons["C0_vs_A0"] = _compare(results, "C0", "A0", seeds, analysis)
        comparisons["B0_vs_A0"] = _compare(results, "B0", "A0", seeds, analysis)
        comparisons["A0_gates_zeroed_vs_A0"] = _compare(results, "A0_gates_zeroed", "A0", seeds, analysis)
        if complete.get("A") and complete.get("C"):
            comparisons["interaction"] = {
                d: [(results[("C", s)][f"val_{d}_bits_per_byte"] - results[("A", s)][f"val_{d}_bits_per_byte"])
                    - (results[("C0", s)][f"val_{d}_bits_per_byte"] - results[("A0", s)][f"val_{d}_bits_per_byte"])
                    for s in seeds] for d in DOMAINS}
    out["comparisons"] = comparisons
    if not all(complete.get(a) for a in ("A", "C", "C_mac")) or refused:
        verdict = "NONQUAL"
    else:
        vs_c = {d: comparisons["C_vs_A"][d]["class"] for d in DOMAINS}
        vs_mac = {d: comparisons["C_mac_vs_A"][d]["class"] for d in DOMAINS}
        if all(v == "A_BETTER" for v in vs_c.values()) and all(v == "A_BETTER" for v in vs_mac.values()):
            verdict = "CGC_BENEFIT_QUALIFIED"
        elif any(v == "X_BETTER" for v in vs_c.values()) and not any(v == "A_BETTER" for v in vs_c.values()):
            verdict = "FFN_PREFERRED"
        else:
            verdict = "CGC_NOT_ESTABLISHED"
    out["primary_verdict"] = verdict
    if "B_vs_A" in comparisons:
        b = [comparisons["B_vs_A"][d]["class"] for d in DOMAINS]
        out["secondary_verdict_B"] = ("CGC_ADDS_OVER_SMALLER_FFN" if all(v == "A_BETTER" for v in b) else
                                      "SMALLER_FFN_NOT_WORSE" if not any(v == "A_BETTER" for v in b) else "MIXED")
    for name, field in (("integrity", "integrity.json"), ("latency", "latency.json")):
        path = folder / field
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            mismatch = {k: record["provenance"][k] for k in expected if record["provenance"][k] != expected[k]}
            if mismatch:
                out.setdefault("auxiliary_refused", []).append({"file": field, "mismatch": mismatch})
                continue
            inputs[field] = sha256_file(path)
            out[name] = record
    if "latency" in out and "A" in out["latency"]["arms"] and "C" in out["latency"]["arms"]:
        ratio = out["latency"]["arms"]["A"]["decode_ms"]["p50"] / out["latency"]["arms"]["C"]["decode_ms"]["p50"]
        out["latency_A_over_C"] = {"decode_p50_ratio": ratio,
                                   "material": ratio > analysis["latency_material_ratio"]}
    return out


def command_manifest(args) -> dict:
    files = sorted(p for p in RESULTS.glob("*.json") if p.name != "MANIFEST.json")
    return {"phase": PHASE, "command": "manifest", "files": {p.name: sha256_file(p) for p in files},
            "runner_sha256": sha256_file(RUNNER)}


def command_smoke(args) -> dict:
    frozen()
    out = {}
    for arm in ("A", "C"):
        record = command_train(argparse.Namespace(arm=arm, seed=0), smoke=True)
        assert record["status"] == "COMPLETED", record.get("train")
        out[arm] = record["final"]
        assert ("gates_zeroed" in record) == (arm == "A")
    command_accounting(args)
    command_latency(args, smoke=True)
    return {"phase": PHASE, "command": "smoke", "final": out}


COMMANDS = {"accounting": command_accounting, "integrity": command_integrity, "train": command_train,
            "latency": command_latency, "combine": command_combine, "manifest": command_manifest,
            "smoke": command_smoke}


def _write(path: str, record: dict) -> None:
    Path(path).write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inputs", help="combine: the folder holding the run records (default: results/r9)")
    parser.add_argument("--out", help="the JSON record to write (nothing is written without it)")
    args = parser.parse_args(argv)
    record = COMMANDS[args.command](args)
    if args.out:
        _write(args.out, record)
    else:
        json.dump({k: v for k, v in record.items() if k in ("phase", "command", "status", "final",
                                                              "primary_verdict")}, sys.stdout, default=str)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
