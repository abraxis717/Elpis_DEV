"""ELPIS_TORCH_R11 — bounded CGC–FFN expert specialization. RESEARCH_ONLY.

``Expert(u) = SwiGLU(u) + g * CGCTransform(u)`` keeps its anatomy; R11 bounds the CGC branch to R97W's microscopic
geometry (``cgc_specialist``), gives it an explicit specialist objective, and measures what it does:

    Stage A   microscopic reference (fresh R97W-protocol episodes): faithful E0, the projected adapter (fixed or
              learned projections), the bounded adapter, the 120-parameter GELU control; cold, shuffled-support
              and no-adaptation (k = 0) controls. Explicit inner SGD on the 120 core parameters, first-order Reptile.
    Stage B   the specialist inside the repository's ``ChannelExpert`` (Mode A, explicit offline adaptation of the
              120 core parameters): G0 FFN only, G1 learned scalar gate (historical), G2 fixed g = 1, G3 matched GELU
              specialist (g = 1), G2_POOLED (no specialist objective); causal ablation, specialist swap, expert
              misrouting, and the frozen-weight value (Mode B, k = 0).
    Stage C   frozen-weight transfer into the actual ERSA stack: preregistered as CONDITIONAL on Stage B.

Commands
    audit      exact CGC census of the legacy and micro scale layouts (meta tensors), budgets, operator costs
    parity     E0 parity of the micro core with the pinned upstream E0; the bound's deviation; float32 vs float64
    calibrate  development-only calibration (Stage A projection learning rates), recorded before the freeze
    stage_a    one Stage A group (family, seed): every arm, learning curves, controls
    stage_b    one Stage B seed: every arm for both families, evaluations, diagnostics
    latency    MEASURED CPU dispatch cost of the expert (d 64) and operator microbenchmarks (H 64 .. 1024)
    combine    the preregistered decision rules over the stage records
    manifest   SHA-256 of every R11 record

Nothing in this phase updates a model weight at inference: Stage A/B inner loops are offline research procedures.
Records go only where ``--out`` says.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
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

from .. import cgc_specialist as cs
from .. import specialist_tasks as st
from ..upstream import cgc_ecsg_e0 as upstream_e0

PHASE = "ELPIS_TORCH_R11_BOUNDED_CGC_SPECIALIST"
REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "research" / "elpis_torch" / "results" / "r11"
PREREGISTRATION = RESULTS / "preregistration.json"
RUNNER = Path(__file__).resolve()
SOURCES = {"runner": RUNNER, "tasks": REPO / "research/elpis_torch/specialist_tasks.py",
           "profile": REPO / "research/elpis_torch/cgc_specialist.py",
           "sparse_channel": REPO / "research/elpis_torch/sparse_channel.py",
           "cgc": REPO / "research/elpis_torch/cgc.py", "model": REPO / "research/elpis_torch/model.py",
           "config": REPO / "research/elpis_torch/config.py"}

STAGE_A_ARMS = {   # name: (core, hidden, bound, learn_projections, outer)  outer: "reptile" | "cold" | "shuffled"
    "E0_META": ("e0", None, None, False, "reptile"),
    "E0_COLD": ("e0", None, None, False, "cold"),
    "E0_SHUFFLED": ("e0", None, None, False, "shuffled"),
    "GELU_META": ("gelu", None, None, False, "reptile"),
    "GELU_COLD": ("gelu", None, None, False, "cold"),
    "PROJ_E0_FIXED_META": ("e0", 64, None, False, "reptile"),
    "PROJ_E0_COLD": ("e0", 64, None, False, "cold"),
    "PROJ_E0_LEARNED_META": ("e0", 64, None, True, "reptile"),
    "PROJ_GELU_FIXED_META": ("gelu", 64, None, False, "reptile"),
    "PROJ_GELU_COLD": ("gelu", 64, None, False, "cold"),
    "PROJ_GELU_LEARNED_META": ("gelu", 64, None, True, "reptile"),
    "PROJ_B3_FIXED_META": ("e0", 64, 3.0, False, "reptile"),
    "PROJ_B3_COLD": ("e0", 64, 3.0, False, "cold"),
    "PROJ_B3_LEARNED_META": ("e0", 64, 3.0, True, "reptile"),
}
LR_GRID = (0.01, 0.03, 0.1, 0.3)
DEV_SEEDS_A, DEV_SEEDS_B = (11, 12), (21,)


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen() -> dict:
    plan = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if plan["phase"] != PHASE:
        raise ValueError("not the R11 preregistration")
    drift = {}
    if plan["stage_a"]["arms"] != {k: list(v) for k, v in STAGE_A_ARMS.items()}:
        drift["stage_a_arms"] = "differ"
    if plan["stage_a"]["protocol"] != json.loads(json.dumps(st.PROTOCOL)):
        drift["stage_a_protocol"] = "differs"
    if plan["stage_b"]["arms"] != st.ARMS_B or plan["stage_b"]["constants"] != json.loads(json.dumps(st.STAGE_B)):
        drift["stage_b"] = "differs"
    if plan["profile"]["geometry"] != [cs.MICRO_BOTTLENECK, cs.MICRO_CHANNELS, cs.MICRO_UNITS] or \
            plan["profile"]["bounds"] != cs.PROFILES:
        drift["profile"] = "differs"
    if drift:
        raise ValueError(f"the preregistration does not describe this runner: {drift}")
    return plan


def provenance(plan) -> dict:
    def git(*args):
        run = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
        return run.stdout.strip() if run.returncode == 0 else ""
    dirty = bool(git("status", "--porcelain", "--", ".", ":(exclude)research/elpis_torch/results"))
    return {"preregistration_sha256": sha256_file(PREREGISTRATION) if plan is not None else None,
            **{f"{k}_sha256": sha256_file(v) for k, v in SOURCES.items()},
            "upstream_e0_sha256": sha256_file(REPO / "research/elpis_torch/upstream/cgc_ecsg_e0.py"),
            "git_revision": git("rev-parse", "HEAD") + ("+dirty" if dirty else ""), "torch": torch.__version__,
            "python": platform.python_version(), "threads": torch.get_num_threads()}


def _gm(ratios) -> float:
    return math.exp(statistics.fmean(math.log(max(r, 1e-300)) for r in ratios))


# -- audit ------------------------------------------------------------------------------------------------------------

def command_audit(args) -> dict:
    from . import run_r6 as r6
    from ..evidence_model import EvidenceLM
    from ..sparse_channel import active_parameters
    out = {"phase": PHASE, "command": "audit", "label": "EXACT (meta tensors; nothing allocated or trained)",
           "scales": {}}
    for name in r6.SCALES:
        legacy = r6.scale_model(name)
        census = cs.cgc_census(legacy)
        encoder = sum(p.numel() for n, p in legacy.named_parameters()
                      if n.startswith("encoder.") and (".cgc." in n or n.endswith("cgc_gate")))
        result = {"legacy": {"census": census, "evidence_encoder_cgc_parameters": encoder,
                             "core_model_cgc_parameters": census["resident_cgc_parameters"] - encoder,
                             "core_model_cgc_excluding_gates": census["resident_cgc_parameters"] - encoder
                             - sum(1 for n, _ in legacy.named_parameters()
                                   if n.endswith("cgc_gate") and not n.startswith("encoder.")),
                             "active": active_parameters(legacy)}}
        for profile in cs.PROFILES:
            ersa = dataclasses.replace(legacy.ersa, model=cs.micro_model(legacy.ersa.model, profile=profile),
                                       sparse=cs.micro_sparse(legacy.ersa.sparse, profile=profile))
            with torch.device("meta"):
                micro = EvidenceLM(ersa, legacy.evidence)
            mc = cs.cgc_census(micro)
            enc_as_built = sum(p.numel() for n, p in micro.named_parameters()
                               if n.startswith("encoder.") and (".cgc." in n or n.endswith("cgc_gate")))
            core_cgc = mc["resident_cgc_parameters"] - enc_as_built
            total_legacy_encoder = mc["model_total_parameters"] - enc_as_built + encoder
            result[profile] = {"census_as_built": mc, "core_model_cgc_parameters": core_cgc,
                               "evidence_encoder_cgc_as_built": enc_as_built,
                               "total_parameters_with_legacy_encoder": total_legacy_encoder,
                               "active_as_built": active_parameters(micro),
                               "note": "the R5 evidence encoder inherits cgc_units and the bound from the core "
                                       "ModelConfig, so the as-built micro model also changes the encoder's CGC "
                                       "(units 4 -> 8); R11 leaves the encoder to R5 and reports its legacy CGC "
                                       "separately (total_parameters_with_legacy_encoder)"}
        d = legacy.ersa.model.d_model
        s = legacy.ersa.sparse
        adapter = cs.adapter_parameters(d)
        result["micro_adapter"] = adapter
        result["budget_routed_expert"] = cs.specialist_budget(d, s.ffn_hidden, adapter["total"])
        result["budget_dense_channel"] = cs.specialist_budget(d, legacy.ersa.model.ffn_width, adapter["total"])
        result["operator_costs_micro_bounded"] = cs.operator_costs(d, 5, 3, 8, True)
        result["operator_costs_legacy_routed"] = cs.operator_costs(d, s.cgc_bottleneck, s.cgc_channels,
                                                                   legacy.ersa.model.cgc_units, True)
        out["scales"][name] = result
    expert, _, _ = st.build_arm("G1", "BOUNDED_SPECIALIST", 0)
    out["d64_expert"] = cs.check_adapter(expert, 64, 86)
    out["adapter_formula"] = {str(h): cs.adapter_parameters(h) for h in (64, 768, 1024)}
    out["causal_path"] = ("ErsaLM.forward computes each block's ERSA (and attention, evidence) mixing BEFORE its "
                          "channel: x -> mix -> pre -> pre + channel(norm(pre)). A block's CGC cannot change that "
                          "block's ERSA state; it changes the input of every later block and, in the last block, "
                          "only the readout (tested in test_r11.py).")
    return out


# -- parity -----------------------------------------------------------------------------------------------------------

def command_parity(args) -> dict:
    from ..cgc import CGCCore, CGCTransform
    out = {"phase": PHASE, "command": "parity"}
    g = torch.Generator().manual_seed(1101)
    core = CGCCore(5, 3, 8, preact_bound=None, dtype=torch.float64)
    with torch.no_grad():
        core.weight.copy_(torch.randn(5, 3, 8, generator=g, dtype=torch.float64) * 0.4)
    x = torch.randn(257, 5, generator=g, dtype=torch.float64)
    cells = core(x)
    ref = torch.stack([upstream_e0.forward(core.weight[:, c, :].detach().contiguous(), x) for c in range(3)], 1)
    y = torch.randn(257, generator=g, dtype=torch.float64)
    grads = []
    for c in range(3):
        w = core.weight[:, c, :].detach().clone().requires_grad_(True)
        loss = upstream_e0.mse_loss(w, x, y)
        loss.backward()
        grads.append(float((w.grad - upstream_e0.analytic_grad(w.detach(), x, y)).abs().max()))
    out["e0_reference_forward_max_abs"] = float((cells.detach() - ref).abs().max())
    out["e0_reference_grad_autograd_vs_analytic_max_abs"] = max(grads)
    bounded = CGCCore(5, 3, 8, preact_bound=3.0, dtype=torch.float64)
    deviation = []
    for scale in (0.1, 0.5, 1.0, 2.0, 4.0):
        with torch.no_grad():
            w = core.weight * scale
            bounded.weight.copy_(w)
            u = x @ w.reshape(5, 24)
            preact_gap = (bounded.preactivations(x).reshape(-1, 24) - u).abs()
            exact = torch.stack([upstream_e0.forward(w[:, c, :].contiguous(), x) for c in range(3)], 1)
            cell_gap = (bounded(x) - exact).abs()
        deviation.append({"weight_scale": scale, "preact_rms": float(u.pow(2).mean().sqrt()),
                          "preact_gap_max": float(preact_gap.max()),
                          "taylor_bound_holds": bool((preact_gap <= u.abs() ** 3 / (3 * 9.0) + 1e-15).all()),
                          "saturated_fraction": float((u.abs() > 3.0).double().mean()),
                          "cell_gap_max": float(cell_gap.max()),
                          "cell_relative_gap": float(cell_gap.max() / (exact.abs().max() + 1e-30))})
    out["bounded_vs_e0"] = deviation
    precision = {}
    for hidden in (64, 768, 1024):
        t64 = CGCTransform(hidden, 3, 8, bottleneck=5, preact_bound=3.0, dtype=torch.float64)
        with torch.no_grad():
            for p in t64.parameters():
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64) * (p.shape[-1] ** -0.5))
        t32 = CGCTransform(hidden, 3, 8, bottleneck=5, preact_bound=3.0, dtype=torch.float32)
        t32.load_state_dict({k: v.float() for k, v in t64.state_dict().items()})
        u = torch.randn(512, hidden, generator=g, dtype=torch.float64)
        a, b = t64(u), t32(u.float()).double()
        precision[str(hidden)] = {"max_abs": float((a - b).abs().max()),
                                  "max_rel": float(((a - b).abs() / (a.abs() + 1e-12)).max()),
                                  "rms": float(a.pow(2).mean().sqrt())}
    out["float32_vs_float64_micro_adapter"] = precision
    out["tolerance_e0"] = 1e-12
    out["e0_reference_qualified"] = out["e0_reference_forward_max_abs"] <= 1e-12
    return out


# -- Stage A ----------------------------------------------------------------------------------------------------------

def _learner(arm: str) -> st.Learner:
    core, hidden, bound, learn, _ = STAGE_A_ARMS[arm]
    return st.Learner(core, hidden=hidden, bound=bound, learn_projections=learn)


def _lr_key(learner: st.Learner) -> str:
    return learner.core + ("" if learner.bound is None else f"_b{learner.bound:g}")


def train_stage_a(arm: str, family: str, seed: int, namespace: str, lr_proj: dict, protocol=st.PROTOCOL,
                  dev=True) -> dict:
    learner = _learner(arm)
    outer = STAGE_A_ARMS[arm][4]
    fixed, core, proj = learner.fixed(seed, namespace), learner.init_core(seed, namespace), \
        learner.init_projections(seed, namespace)
    trace, forget = [], []
    p = protocol
    for episode in range(p["train_episodes"] + 1):
        if dev and episode in p["dev_checkpoints"]:
            losses = []
            for d in range(p["dev_episodes"]):
                e = st.teacher(namespace, family, seed, "DEV", d)
                _, tr = st.adapt(learner, core, proj, fixed, e.support, p["inner_steps"], p["inner_lr"],
                                 p["gradient_clip_l2"], record={p["inner_steps"]}, query=e.query, ood=e.ood)
                losses.append(tr[p["inner_steps"]][0])
            trace.append({"after_train_episodes": episode, "dev_final_query_median": statistics.median(losses)})
        if episode == p["train_episodes"] or outer == "cold":
            if outer == "cold" and episode < p["train_episodes"]:
                continue
            break
        e = st.teacher(namespace, family, seed, "TRAIN", episode)
        support = e.support
        if outer == "shuffled":
            support = (support[0], support[1][st.randperm(len(support[1]), namespace, "shuffle", family, seed,
                                                          episode)])
        fast, _ = st.adapt(learner, core, proj, fixed, support, p["inner_steps"], p["inner_lr"],
                           p["gradient_clip_l2"])
        if learner.learn_projections:
            proj = st.projection_step(learner, fast, proj, fixed, e.query, lr_proj[_lr_key(learner)],
                                      p["gradient_clip_l2"])
        new_core = tuple(a + p["outer_interpolation"] * (b - a) for a, b in zip(core, fast))
        if outer == "reptile" and episode >= p["train_episodes"] - p["forget_probe_episodes"]:
            with torch.no_grad():
                old_fast = float(st.mse(learner.predict(fast, proj, fixed, e.query[0]), e.query[1]))
                old_reset = float(st.mse(learner.predict(new_core, proj, fixed, e.query[0]), e.query[1]))
            forget.append({"episode": episode, "old_fast_loss": old_fast, "old_reset_loss": old_reset,
                           "ratio": old_reset / max(old_fast, 1e-12)})
        core = new_core
    return {"core": core, "projections": proj, "fixed": fixed, "trace": trace, "forget": forget}


def evaluate_stage_a(arm: str, state: dict, family: str, seed: int, namespace: str, split: str, episodes: int,
                     protocol=st.PROTOCOL) -> list:
    learner = _learner(arm)
    rows = []
    for index in range(episodes):
        e = st.teacher(namespace, family, seed, split, index)
        _, tr = st.adapt(learner, state["core"], state["projections"], state["fixed"], e.support,
                         protocol["inner_steps"], protocol["inner_lr"], protocol["gradient_clip_l2"],
                         record=set(protocol["eval_steps"]), query=e.query, ood=e.ood)
        rows.append({"episode": index, "curve": {str(k): v for k, v in tr.items()},
                     "null": [float(e.query[1].pow(2).mean()), float(e.ood[1].pow(2).mean())]})
    return rows


def summarize_curves(rows: list, steps=st.PROTOCOL["eval_steps"]) -> dict:
    final = str(max(steps))
    out = {"n": len(rows), "median": {str(k): statistics.median(r["curve"][str(k)][0] for r in rows) for k in steps},
           "mean": {str(k): statistics.fmean(r["curve"][str(k)][0] for r in rows) for k in steps},
           "ood_final_median": statistics.median(r["curve"][final][1] for r in rows)}
    aucs = [statistics.fmean(r["curve"][str(k)][0] for k in steps) for r in rows]
    out["auc_median"] = statistics.median(aucs)
    return out


def paired(rows_a: list, rows_b: list, step: str) -> dict:
    ratios = [a["curve"][step][0] / max(b["curve"][step][0], 1e-12) for a, b in zip(rows_a, rows_b)]
    return {"n": len(ratios), "geomean_ratio": _gm(ratios), "better_n": sum(r < 1 for r in ratios),
            "log_ratios": [math.log(max(r, 1e-300)) for r in ratios]}


def command_stage_a(args) -> dict:
    plan = None if args.dev else frozen()
    family, seed = args.family, args.seed
    namespace = "R11A_DEV" if args.dev else plan["stage_a"]["namespace"]
    lr_proj = (json.loads(args.lr_proj) if args.lr_proj else {"e0": 0.1, "gelu": 0.3, "e0_b3": 0.3}) if args.dev \
        else plan["stage_a"]["lr_proj"]
    if not args.dev and seed not in plan["stage_a"]["seeds"]:
        raise SystemExit(f"seed {seed} is not a preregistered Stage A seed")
    started = time.time()
    arms = {}
    for arm in STAGE_A_ARMS:
        state = train_stage_a(arm, family, seed, namespace, lr_proj)
        rows = evaluate_stage_a(arm, state, family, seed, namespace, "TEST", st.PROTOCOL["test_episodes"])
        arms[arm] = {"rows": rows, "summary": summarize_curves(rows), "trace": state["trace"],
                     "forget": state["forget"], "accounting": _learner(arm).parameter_accounting()}
    return {"phase": PHASE, "command": "stage_a", "stage": "development" if args.dev else "confirmation",
            "family": family, "seed": seed, "namespace": namespace, "lr_proj": lr_proj, "arms": arms,
            "seconds": round(time.time() - started, 1), "provenance": provenance(plan), "status": "COMPLETED"}


def command_calibrate(args) -> dict:
    """Development-only: Stage A projection learning rate per core type, by the rule frozen in advance (the grid
    value with the lowest mean development k = 8 query MSE over the development seeds and both families)."""
    out = {"phase": PHASE, "command": "calibrate", "label": "DEVELOPMENT ONLY (namespace R11A_DEV, split DEV)",
           "grid": LR_GRID, "seeds": DEV_SEEDS_A, "results": {}}
    for core, bound in (("e0", None), ("gelu", None), ("e0", 3.0)):
        learner = st.Learner(core, hidden=64, bound=bound, learn_projections=True)
        key = _lr_key(learner)
        scores = {}
        for lr in LR_GRID:
            values = []
            for family in st.FAMILIES:
                for seed in DEV_SEEDS_A:
                    fixed = learner.fixed(seed, "R11A_DEV")
                    c, proj = learner.init_core(seed, "R11A_DEV"), learner.init_projections(seed, "R11A_DEV")
                    p = st.PROTOCOL
                    for episode in range(p["train_episodes"]):
                        e = st.teacher("R11A_DEV", family, seed, "TRAIN", episode)
                        fast, _ = st.adapt(learner, c, proj, fixed, e.support, p["inner_steps"], p["inner_lr"],
                                           p["gradient_clip_l2"])
                        proj = st.projection_step(learner, fast, proj, fixed, e.query, lr, p["gradient_clip_l2"])
                        c = tuple(a + p["outer_interpolation"] * (b - a) for a, b in zip(c, fast))
                    for d in range(p["dev_episodes"]):
                        e = st.teacher("R11A_DEV", family, seed, "DEV", d)
                        _, tr = st.adapt(learner, c, proj, fixed, e.support, p["inner_steps"], p["inner_lr"],
                                         p["gradient_clip_l2"], record={p["inner_steps"]}, query=e.query, ood=e.ood)
                        values.append(tr[p["inner_steps"]][0])
            scores[str(lr)] = statistics.fmean(values)
        best = min(LR_GRID, key=lambda v: scores[str(v)])
        out["results"][key] = {"dev_mean_final_query_mse": scores, "selected": best}
    out["lr_proj"] = {k: v["selected"] for k, v in out["results"].items()}
    out["provenance"] = provenance(None)
    return out


# -- Stage B ----------------------------------------------------------------------------------------------------------

def _dev_score(expert, context, family, seed, namespace, episodes) -> float:
    rows = st.evaluate_arm(expert, context, family, seed, namespace, "DEV", episodes)
    return statistics.fmean(r["curve"][str(st.STAGE_B["inner_steps"])][0] for r in rows)


def train_stage_b(arm, profile, family, seed, namespace, plan_b) -> dict:
    """Train with per-arm checkpoint selection on DEV episodes (the same rule for every arm): one continuous run,
    scored every ``select_every`` outer steps; the best-scoring checkpoint is the arm's model."""
    best = {"score": math.inf, "step": None, "state": None}
    trace = []
    expert, context, info = _train_with_checkpoints(arm, profile, family, seed, namespace, plan_b, best, trace)
    expert.load_state_dict(best["state"])
    return {"expert": expert, "context": context, "info": info, "selected_step": best["step"],
            "selected_dev_score": best["score"], "selection_trace": trace}


def _train_with_checkpoints(arm, profile, family, seed, namespace, plan_b, best, trace):
    from ..cgc_specialist import branch_diagnostics, gradient_norms, specialist_core_names
    spec = st.ARMS_B[arm]
    expert, _, _ = st.build_arm(arm, profile, seed)
    context = st.hidden_context(namespace, seed, st.STAGE_B["hidden"])
    readout, sigma = context["R"], st.STAGE_B["nuisance"]
    diag_ep = st.teacher(namespace, family, seed, "DIAGNOSE", 0)
    diag_u = st.hidden_states(context, torch.cat([diag_ep.support[0], diag_ep.query[0]]), namespace, seed,
                              "DIAGNOSE", 0, "all", sigma=sigma)
    init = st.init_branch_like_stage_a(expert, seed, namespace)
    core_names = specialist_core_names(expert)
    optimizer = torch.optim.Adam(expert.parameters(), lr=plan_b["adam_lr"])
    steps, batch, every = plan_b["outer_steps"], plan_b["meta_batch"], plan_b["select_every"]
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda t: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / steps)))
    initial = {n: p.detach().clone() for n, p in expert.named_parameters()}
    for step in range(steps + 1):
        if step % every == 0 or step == steps:
            score = _dev_score(expert, context, family, seed, namespace, plan_b["dev_episodes"])
            d = branch_diagnostics(expert, diag_u)
            trace.append({"step": step, "dev_score": score,
                          **{k: d.get(k) for k in ("gate", "branch_rms", "branch_to_ffn_rms", "ffn_rms", "preact_rms",
                                                    "preact_abs_max", "saturated_fraction", "nonfinite")},
                          "proj_out_norm": st._proj_out_norm(expert)})
            if math.isfinite(score) and score < best["score"]:
                best.update(score=score, step=step,
                            state={k: v.detach().clone() for k, v in expert.state_dict().items()})
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for b in range(batch):
            index = step * batch + b
            ep = st.episode_states(context, st.teacher(namespace, family, seed, "TRAIN", index), namespace, seed,
                                   "TRAIN", index, sigma)
            su, sy = ep["support"]
            qu, qy = ep["query"]
            if spec["objective"] == "specialist" and core_names:
                fast, _ = st.inner_adapt(expert, core_names, su, sy, readout, st.STAGE_B["inner_steps"],
                                         st.STAGE_B["inner_lr"], st.STAGE_B["gradient_clip_l2"])
                live = dict(expert.named_parameters())
                overrides = {n: live[n] + (fast[n] - live[n].detach()) for n in core_names}
                loss = st.mse(st._functional(expert, overrides, qu, live=True) @ readout.T, qy)
            else:
                loss = st.mse(expert(torch.cat([su, qu])) @ readout.T, torch.cat([sy, qy]))
            (loss / batch).backward()
            total += float(loss.detach()) / batch
        if step % every == 0:
            trace[-1]["gradient_norms"] = gradient_norms(expert)
            trace[-1]["train_loss"] = total
        optimizer.step()
        schedule.step()
    change = {n: float((p.detach() - initial[n]).norm() / (initial[n].norm() + 1e-30))
              for n, p in expert.named_parameters()}
    return expert, context, {"branch_initialization": init, "relative_change_end": change,
                             "final_diagnostics": branch_diagnostics(expert, diag_u)}


def command_stage_b(args) -> dict:
    plan = None if args.dev else frozen()
    seed = args.seed
    namespace = "R11B_DEV" if args.dev else plan["stage_b"]["namespace"]
    plan_b = (json.loads(args.plan_b) if args.plan_b else DEFAULT_PLAN_B) if args.dev else plan["stage_b"]["training"]
    if args.dev:
        profile = args.profile or "BOUNDED_SPECIALIST"
    else:                    # the frozen rule: Stage A's complete confirmation decides the Stage B profile
        records = [json.loads(q.read_text(encoding="utf-8")) for q in sorted(RESULTS.glob("stage_a_*.json"))]
        stage_a = analyse_stage_a([x for x in records if x.get("stage") == "confirmation"], plan)
        if "stage_b_profile" not in stage_a:
            raise SystemExit("Stage B needs Stage A's complete confirmation first (the profile rule reads it)")
        profile = stage_a["stage_b_profile"]
    if not args.dev and seed not in plan["stage_b"]["seeds"]:
        raise SystemExit(f"seed {seed} is not a preregistered Stage B seed")
    torch.manual_seed(0)
    started = time.time()
    split, episodes = ("DEV_HOLDOUT", st.STAGE_B["dev_episodes"]) if args.dev else ("TEST", st.STAGE_B["test_episodes"])
    trained = {}
    for family in st.FAMILIES:
        for arm in st.ARMS_B:
            trained[(family, arm)] = train_stage_b(arm, profile, family, seed, namespace, plan_b)
    groups = {}
    for family in st.FAMILIES:
        other = [f for f in st.FAMILIES if f != family][0]
        arms = {}
        for arm in st.ARMS_B:
            t = trained[(family, arm)]
            rows = st.evaluate_arm(t["expert"], t["context"], family, seed, namespace, split, episodes)
            entry = {"rows": rows, "summary": summarize_curves(rows, st.STAGE_B["eval_steps"]),
                     "ablated_median": statistics.median(r["ablated"][0] for r in rows),
                     "selected_step": t["selected_step"], "selected_dev_score": t["selected_dev_score"],
                     "selection_trace": t["selection_trace"], "info": t["info"],
                     "parameters": {n: p.numel() for n, p in t["expert"].named_parameters()}}
            if arm != "G0":
                swap = st.evaluate_arm(t["expert"], t["context"], family, seed, namespace, split, episodes,
                                          start_from=trained[(other, arm)]["expert"])
                entry["specialist_swap_rows"] = swap
                misrouted = st.evaluate_arm(trained[(other, arm)]["expert"], t["context"], family, seed,
                                               namespace, split, episodes)
                entry["expert_misrouted_rows"] = misrouted
            arms[arm] = entry
        groups[family] = arms
    return {"phase": PHASE, "command": "stage_b", "stage": "development" if args.dev else "confirmation",
            "seed": seed, "namespace": namespace, "profile": profile, "plan_b": plan_b, "split": split,
            "groups": groups, "seconds": round(time.time() - started, 1), "provenance": provenance(plan),
            "status": "COMPLETED"}


DEFAULT_PLAN_B = {"outer_steps": 1000, "meta_batch": 4, "adam_lr": 3e-3, "select_every": 100, "dev_episodes": 5}


# -- latency ----------------------------------------------------------------------------------------------------------

def command_latency(args) -> dict:
    """MEASURED on this CPU, one thread: the expert's forward on 64 routed tokens (d 64) per arm, and the micro
    adapter against the legacy routed CGC at H = 64, 768, 1024. Interleaved repetitions; medians."""
    from ..cgc import CGCTransform
    torch.set_num_threads(1)
    out = {"phase": PHASE, "command": "latency", "label": "MEASURED CPU wall-clock, one thread; PERFORMANCE_ONLY"}
    u = torch.randn(64, 64)
    experts = {}
    for arm in st.ARMS_B:
        e, _, _ = st.build_arm(arm, "BOUNDED_SPECIALIST", 0, dtype=torch.float32)
        experts[arm] = e.eval()
    from ..sparse_channel import ChannelExpert, SparseConfig
    from ..config import ModelConfig
    m = ModelConfig(vocab_size=2, d_model=64, n_layers=1, n_heads=1, max_seq_len=8, channel="ffn+cgc", ffn_hidden=86)
    experts["LEGACY_R10_A"] = ChannelExpert(m, SparseConfig(experts=1, top_k=1, ffn_hidden=86, cgc_channels=48,
                                                            cgc_bottleneck=16, cgc_units=4), torch.float32).eval()
    samples = {k: [] for k in experts}
    with torch.no_grad():
        for _ in range(20):
            for e in experts.values():
                e(u)
        for rep in range(400):
            order = list(experts)
            order = order[rep % len(order):] + order[:rep % len(order)]
            for k in order:
                t = time.perf_counter()
                experts[k](u)
                samples[k].append(time.perf_counter() - t)
    out["expert_d64_64_tokens_us"] = {k: 1e6 * statistics.median(v) for k, v in samples.items()}
    ops = {}
    for hidden in (64, 768, 1024):
        cfgs = {"micro_bounded": (3, 8, 5, 3.0), "micro_e0": (3, 8, 5, None),
                "legacy_r6_routed": (hidden, 4, hidden // 4, 3.0)}
        mods = {k: CGCTransform(hidden, c, n, bottleneck=b, preact_bound=bd).eval() for k, (c, n, b, bd) in cfgs.items()}
        x = torch.randn(64, hidden)
        times = {k: [] for k in mods}
        with torch.no_grad():
            for _ in range(10):
                for m_ in mods.values():
                    m_(x)
            for rep in range(200):
                for k in mods:
                    t = time.perf_counter()
                    mods[k](x)
                    times[k].append(time.perf_counter() - t)
        ops[str(hidden)] = {k: {"median_us_64_tokens": 1e6 * statistics.median(v),
                                "parameters": sum(p.numel() for p in mods[k].parameters())} for k, v in times.items()}
    out["operator_64_tokens"] = ops
    out["provenance"] = provenance(None)
    return out


# -- combine ----------------------------------------------------------------------------------------------------------

def sign_test_p(log_ratios) -> float:
    """One-sided exact sign test: P(at least this many negative log-ratios | p = 1/2), ties dropped."""
    neg = sum(v < 0 for v in log_ratios)
    n = sum(v != 0 for v in log_ratios)
    return sum(math.comb(n, k) for k in range(neg, n + 1)) / 2 ** n if n else 1.0


def better(groups: list, rule: dict) -> dict:
    """``groups``: per group, the per-episode log ratios X/Y. X is better than Y when the pooled geometric mean
    ratio is at most ``rule['max_ratio']``, the one-sided sign test gives p < ``rule['alpha']`` and at least
    ``rule['min_group_fraction']`` of the groups have a geometric mean below 1."""
    pooled = [v for g in groups for v in g]
    gm = math.exp(statistics.fmean(pooled))
    group_gm = [math.exp(statistics.fmean(g)) for g in groups]
    p = sign_test_p(pooled)
    frac = sum(v < 1 for v in group_gm) / len(group_gm)
    return {"pooled_geomean_ratio": gm, "sign_test_p": p, "groups_below_one": frac, "group_geomeans": group_gm,
            "n_episodes": len(pooled), "better": gm <= rule["max_ratio"] and p < rule["alpha"]
            and frac >= rule["min_group_fraction"]}


def command_combine(args) -> dict:
    plan = frozen()
    folder = Path(args.inputs) if args.inputs else RESULTS
    expected = {"preregistration_sha256": sha256_file(PREREGISTRATION),
                **{f"{k}_sha256": sha256_file(v) for k, v in SOURCES.items()}}
    out = {"phase": PHASE, "command": "combine", "provenance": provenance(plan), "refused": [], "inputs": {}}
    a_records, b_records = [], []
    for path in sorted(folder.glob("stage_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        prov = record.get("provenance", {})
        if any(prov.get(k) != v for k, v in expected.items()) or record.get("stage") != "confirmation":
            out["refused"].append({"file": path.name, "reason": "digest mismatch or not a confirmation record"})
            continue
        out["inputs"][path.name] = sha256_file(path)
        (a_records if record["command"] == "stage_a" else b_records).append(record)
    out["stage_a"] = analyse_stage_a(a_records, plan)
    out["stage_b"] = analyse_stage_b(b_records, plan, out["stage_a"])
    out["stage_c"] = {"disposition": "NOT_RUN" if not out["stage_b"].get("stage_c_triggered") else "PENDING",
                      "reason": out["stage_b"].get("stage_c_reason")}
    return out


def analyse_stage_a(records: list, plan: dict) -> dict:
    rule, util = plan["analysis"]["better"], plan["analysis"]["stage_a_utility"]
    expected = {(f, s) for f in st.FAMILIES for s in plan["stage_a"]["seeds"]}
    have = {(r["family"], r["seed"]): r for r in records}
    if set(have) != expected:
        return {"disposition": "NONQUAL", "missing": sorted(map(list, expected - set(have)))}
    keys = sorted(have)
    final = str(st.PROTOCOL["inner_steps"])

    def utility(meta: str, cold: str) -> dict:
        per = []
        for k in keys:
            arms = have[k]["arms"]
            c = paired(arms[meta]["rows"], arms[cold]["rows"], final)
            auc_ratio = arms[meta]["summary"]["auc_median"] / arms[cold]["summary"]["auc_median"]
            forget = statistics.median(f["ratio"] for f in arms[meta]["forget"]) if arms[meta]["forget"] else None
            ok = (c["geomean_ratio"] <= util["max_geomean_meta_vs_cold"]
                  and c["better_n"] / c["n"] >= util["min_paired_positive_fraction"]
                  and auc_ratio <= util["max_auc_ratio"]
                  and (forget is None or forget >= util["min_reset_to_fast_ratio"]))
            per.append({"group": list(k), "geomean_meta_vs_cold": c["geomean_ratio"], "better_n": c["better_n"],
                        "n": c["n"], "auc_ratio": auc_ratio, "forget_ratio": forget, "pass": ok})
        return {"groups": per, "all_groups_pass": all(g["pass"] for g in per)}

    def compare(x: str, y: str) -> dict:
        return better([paired(have[k]["arms"][x]["rows"], have[k]["arms"][y]["rows"], final)["log_ratios"]
                       for k in keys], rule)

    out = {"utility": {arm: utility(arm, cold) for arm, cold in (
        ("E0_META", "E0_COLD"), ("GELU_META", "GELU_COLD"), ("PROJ_E0_FIXED_META", "PROJ_E0_COLD"),
        ("PROJ_E0_LEARNED_META", "PROJ_E0_COLD"), ("PROJ_GELU_FIXED_META", "PROJ_GELU_COLD"),
        ("PROJ_GELU_LEARNED_META", "PROJ_GELU_COLD"), ("PROJ_B3_FIXED_META", "PROJ_B3_COLD"),
        ("PROJ_B3_LEARNED_META", "PROJ_B3_COLD"))}}
    out["comparisons"] = {f"{x}/{y}": compare(x, y) for x, y in (
        ("E0_META", "GELU_META"), ("E0_META", "E0_SHUFFLED"), ("PROJ_E0_FIXED_META", "PROJ_GELU_FIXED_META"),
        ("PROJ_E0_LEARNED_META", "PROJ_GELU_LEARNED_META"), ("PROJ_B3_FIXED_META", "PROJ_GELU_FIXED_META"),
        ("PROJ_B3_LEARNED_META", "PROJ_GELU_LEARNED_META"), ("PROJ_E0_FIXED_META", "E0_META"),
        ("PROJ_B3_FIXED_META", "PROJ_E0_FIXED_META"), ("PROJ_B3_LEARNED_META", "PROJ_E0_LEARNED_META"))}
    bound_rule = plan["analysis"]["bound_destroys"]
    worse = {name: out["comparisons"][f"{b}/{u}"]["pooled_geomean_ratio"] for name, b, u in (
        ("fixed", "PROJ_B3_FIXED_META", "PROJ_E0_FIXED_META"), ("learned", "PROJ_B3_LEARNED_META",
                                                                "PROJ_E0_LEARNED_META"))}
    destroys = any(not out["utility"][f"PROJ_B3_{m.upper()}_META"]["all_groups_pass"]
                   and out["utility"][f"PROJ_E0_{m.upper()}_META"]["all_groups_pass"] for m in ("fixed", "learned")) \
        or any(v > bound_rule["max_ratio_bounded_over_unbounded"] for v in worse.values())
    out["bound_destroys_behavior"] = destroys
    dispositions = {"MICRO_REFERENCE": "MICRO_REFERENCE_QUALIFIED" if out["utility"]["E0_META"]["all_groups_pass"]
                    and plan["parity"]["e0_reference_qualified"] else "NOT_ESTABLISHED"}
    for arm in ("PROJ_E0_FIXED_META", "PROJ_E0_LEARNED_META", "PROJ_B3_FIXED_META", "PROJ_B3_LEARNED_META"):
        dispositions[arm] = ("PROJECTED_ADAPTATION_QUALIFIED" if out["utility"][arm]["all_groups_pass"]
                             else "NOT_ESTABLISHED")
    dispositions["MICRO_ADVANTAGE_E0_OVER_GELU"] = out["comparisons"]["E0_META/GELU_META"]["better"]
    for kind in ("FIXED", "LEARNED"):
        dispositions[f"PROJECTED_ADVANTAGE_{kind}_E0"] = out["comparisons"][
            f"PROJ_E0_{kind}_META/PROJ_GELU_{kind}_META"]["better"]
        dispositions[f"PROJECTED_ADVANTAGE_{kind}_B3"] = out["comparisons"][
            f"PROJ_B3_{kind}_META/PROJ_GELU_{kind}_META"]["better"]
    out["dispositions"] = dispositions
    out["stage_b_profile"] = "E0_REFERENCE" if destroys else "BOUNDED_SPECIALIST"
    return out


def analyse_stage_b(records: list, plan: dict, stage_a: dict) -> dict:
    rule = plan["analysis"]["better"]
    b_rule = plan["analysis"]["stage_b"]
    expected = set(plan["stage_b"]["seeds"])
    have = {r["seed"]: r for r in records}
    if set(have) != expected:
        return {"disposition": "NONQUAL", "missing": sorted(expected - set(have))}
    if stage_a.get("stage_b_profile") and any(r["profile"] != stage_a["stage_b_profile"] for r in records):
        return {"disposition": "NONQUAL", "reason": "Stage B ran a profile other than Stage A's rule selects"}
    final = str(st.STAGE_B["inner_steps"])
    groups = [(s, f) for s in sorted(have) for f in st.FAMILIES]

    def rows(arm, kind="rows"):
        return [have[s]["groups"][f][arm][kind] for s, f in groups]

    def compare(x_rows, y_rows, step=final):
        return better([paired(a, b, step)["log_ratios"] for a, b in zip(x_rows, y_rows)], rule)

    def ablation(arm):           # the adapted branch removed: ablated loss over intact k = 8 loss
        ratios = [[math.log(max(r["ablated"][0], 1e-300) / max(r["curve"][final][0], 1e-300)) for r in g]
                  for g in rows(arm)]
        pooled = [v for g in ratios for v in g]
        return {"pooled_geomean_ablated_over_intact": math.exp(statistics.fmean(pooled)),
                "causal": math.exp(statistics.fmean(pooled)) >= b_rule["min_ablation_ratio"]}

    out = {"comparisons": {}, "mode_b": {}, "controls": {}}
    for x, y in (("G1", "G0"), ("G2", "G0"), ("G3", "G0"), ("G2", "G3"), ("G1", "G3"), ("G2", "G1"),
                 ("G2", "G2_POOLED"), ("G1", "G2_POOLED"), ("G2_POOLED", "G0")):
        out["comparisons"][f"{x}/{y}"] = compare(rows(x), rows(y))
        out["mode_b"][f"{x}/{y}"] = compare(rows(x), rows(y), "0")
    for arm in ("G1", "G2", "G3", "G2_POOLED"):
        out["controls"][arm] = {"ablation": ablation(arm),
                                "specialist_swap_over_correct": compare(rows(arm, "specialist_swap_rows"), rows(arm)),
                                "expert_misrouted_over_correct": compare(rows(arm, "expert_misrouted_rows"),
                                                                         rows(arm))}
    g0 = [statistics.median(r["curve"]["0"][0] for r in have[s]["groups"][f]["G0"]["rows"]) for s, f in groups]
    null = [statistics.median(r["null"][0] for r in have[s]["groups"][f]["G0"]["rows"]) for s, f in groups]
    out["assay"] = {"g0_median_query_mse": g0, "null_median_query_mse": null,
                    "ffn_learned_shared_function": all(a <= b_rule["max_g0_over_null"] * b for a, b in zip(g0, null))}

    def qualified(arm):
        c = out["comparisons"]
        return {"adaptive_benefit": c[f"{arm}/G0"]["better"]
                and c[f"{arm}/G0"]["pooled_geomean_ratio"] <= b_rule["max_ratio_over_ffn"],
                "causal": out["controls"][arm]["ablation"]["causal"],
                "specialist_objective": c[f"{arm}/G2_POOLED"]["better"],
                "cgc_specific_over_gelu": c[f"{arm}/G3"]["better"]}

    if not out["assay"]["ffn_learned_shared_function"]:
        out["dispositions"] = {"G1": "ASSAY_NOT_LEARNED", "G2": "ASSAY_NOT_LEARNED"}
        out["stage_c_triggered"], out["stage_c_reason"] = False, "Stage B assay not learned"
        return out
    q = {arm: qualified(arm) for arm in ("G1", "G2")}
    out["qualification"] = q
    slot = {arm: q[arm]["adaptive_benefit"] and q[arm]["causal"] for arm in q}
    full = {arm: all(q[arm].values()) for arm in q}
    out["specialist_slot_functional"] = slot
    disp = {}
    for arm in ("G1", "G2"):
        disp[arm] = ("EXPERT_SPECIALIZATION_QUALIFIED" if full[arm] else
                     "SPECIALIST_SLOT_FUNCTIONAL_NO_CGC_SPECIFIC_ADVANTAGE" if slot[arm] and q[arm]["specialist_objective"]
                     and not q[arm]["cgc_specific_over_gelu"] else "NOT_ESTABLISHED")
    if slot["G2"] and not slot["G1"]:
        gate = "GATE_IDENTIFIABILITY_FAILURE"
    elif slot["G2"] and slot["G1"]:
        gate = "GATE_NOT_LIMITING"
    elif not slot["G2"] and not slot["G1"]:
        gate = "SPECIALIST_OPERATION_NOT_ESTABLISHED"
    else:
        gate = "LEARNED_GATE_ONLY"
    disp["gate_identifiability"] = gate
    mode_b_benefit = {arm: out["mode_b"][f"{arm}/G0"]["better"] for arm in ("G1", "G2")}
    disp["mode_b"] = {arm: ("FROZEN_BENEFIT" if mode_b_benefit[arm] else
                            "ADAPTIVE_SPECIALIST_REQUIRES_INNER_UPDATES" if slot[arm] else "NOT_ESTABLISHED")
                      for arm in ("G1", "G2")}
    out["dispositions"] = disp
    triggered = full["G1"] or full["G2"]
    out["stage_c_triggered"] = triggered
    out["stage_c_reason"] = ("Stage B qualified a CGC specialist: Stage C requires its own frozen amendment" if triggered
                             else "Stage C runs only after EXPERT_SPECIALIZATION_QUALIFIED (preregistration §6)")
    return out


def command_manifest(args) -> dict:
    files = sorted(p for p in RESULTS.glob("*.json") if p.name != "MANIFEST.json")
    return {"phase": PHASE, "command": "manifest", "files": {p.name: sha256_file(p) for p in files}}


COMMANDS = {"audit": command_audit, "parity": command_parity, "calibrate": command_calibrate,
            "stage_a": command_stage_a, "stage_b": command_stage_b, "latency": command_latency,
            "combine": command_combine, "manifest": command_manifest}


def _write(path: str, record: dict) -> None:
    Path(path).write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--family", default="smooth", choices=st.FAMILIES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev", action="store_true", help="development namespace and seeds (no preregistration)")
    parser.add_argument("--lr-proj", dest="lr_proj", help="development only: projection learning rates (JSON)")
    parser.add_argument("--plan-b", dest="plan_b", help="development only: Stage B training plan (JSON)")
    parser.add_argument("--profile", choices=sorted(cs.PROFILES), help="development only: Stage B profile")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--inputs", help="combine: folder of stage records (default results/r11)")
    parser.add_argument("--out", help="the JSON record to write (nothing is written without it)")
    args = parser.parse_args(argv)
    if not args.dev and (args.lr_proj or args.plan_b or args.profile):
        parser.error("--lr-proj, --plan-b and --profile are development-only")
    torch.set_num_threads(args.threads)
    record = COMMANDS[args.command](args)
    if args.out:
        _write(args.out, record)
    else:
        json.dump({k: record[k] for k in record if k in ("phase", "command", "status", "dispositions", "seconds")},
                  sys.stdout, default=str)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
