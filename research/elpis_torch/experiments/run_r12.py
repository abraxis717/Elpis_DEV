"""ELPIS_TORCH_R12: micro-CGC volatile fast-state integration (the Stage C seam experiment). RESEARCH_ONLY.

The question: inside R11's actual Stage C ERSA stack, does restoring the exact episode-local update of the LAST
block's 120-parameter specialist core recover CGC's specialist advantage? The update is R11 Stage B's: 8 clipped
SGD steps on the observed support, through a transactional ``fast_state.SpecialistFastState``. The plan is
``results/r12/preregistration.json``, frozen before any confirmation record.

Two training regimes, never pooled
    T1 ``reproduce``: R11's Stage C models, rebuilt from their recorded seeds with R11's own code
       (``run_r11_stage_c``'s loop: ``torch.manual_seed(0)``, then G0 to G3, each trained and evaluated in order).
       Each rebuilt model is checked against the committed R11 record row by row, then evaluated with fast
       adaptation on FRESH episodes (split ``R12TEST``). The training is unchanged: these models were never
       trained for adaptation.
    T2 ``episodic``: fresh seeds, the same stack and the same training plan, with ONE change. A specialist arm
       trains with the fast update in the loop (``fast_episodes.train_fast``; first-order outer gradient). G0, which
       has nothing to adapt, trains with R11's ``train_stack``.

Commands
    audit      exact state and parameter accounting, and the dependency facts (no training)
    parity     numerical parity record: cached and full recomputation, float32 and float64, the analytic gradient,
               batched and transactional updates, state coherence in an ERSA LM with window attention, refusal of
               earlier blocks
    reproduce  T1 for one (family, seed): record ``t1_<family>_s<seed>.json``
    episodic   T2 for one (family, seed): record ``t2_<family>_s<seed>.json``
    latency    measured CPU wall-clock of every fast-state operation (one thread)
    combine    the frozen decision rules (``study_r12.json``)
    manifest   every R12 record with its digest
    dev        development mechanics (namespaces ``R11C_DEV`` / ``R12C_DEV``; writes nothing unless ``--out``)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.func import functional_call

from .. import fast_episodes as fe
from .. import specialist_stack as ss
from .. import specialist_tasks as st
from ..cgc_specialist import specialist_core_names
from ..fast_state import FastStateConfig, FastStateError, ObservedSupport, SpecialistFastState, frozen_pass, last_channel

PHASE = "ELPIS_TORCH_R12_FAST_STATE"
REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "research" / "elpis_torch" / "results" / "r12"
R11 = REPO / "research" / "elpis_torch" / "results" / "r11"
PREREGISTRATION = RESULTS / "preregistration.json"
TORCH = REPO / "research" / "elpis_torch"
SOURCES = {"runner": Path(__file__).resolve(), "fast_state": TORCH / "fast_state.py",
           "fast_episodes": TORCH / "fast_episodes.py", "session": TORCH / "session.py",
           "lastblock": TORCH / "lastblock.py", "stack": TORCH / "specialist_stack.py",
           "tasks": TORCH / "specialist_tasks.py", "profile": TORCH / "cgc_specialist.py",
           "sparse_channel": TORCH / "sparse_channel.py", "cgc": TORCH / "cgc.py", "model": TORCH / "model.py",
           "config": TORCH / "config.py", "ersa": TORCH / "ersa.py", "ersa_model": TORCH / "ersa_model.py",
           "r11_stage_c_runner": TORCH / "experiments" / "run_r11_stage_c.py"}
R11_AMENDMENT = R11 / "amendment_stage_c.json"
EVAL_SPLIT_T1 = "R12TEST"        # fresh episodes of R11's families: never used by R11 or by R12's development
DEV_SPLIT = "R12DEV"


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(plan) -> dict:
    def git(*args):
        run = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
        return run.stdout.strip() if run.returncode == 0 else ""
    dirty = bool(git("status", "--porcelain", "--", ".", ":(exclude)research/elpis_torch/results"))
    return {"preregistration_sha256": sha(PREREGISTRATION) if plan is not None else None,
            **{f"{k}_sha256": sha(v) for k, v in SOURCES.items()},
            "r11_amendment_sha256": sha(R11_AMENDMENT),
            "git_revision": git("rev-parse", "HEAD") + ("+dirty" if dirty else ""), "torch": torch.__version__,
            "python": platform.python_version(), "threads": torch.get_num_threads()}


def frozen() -> dict:
    plan = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if plan["phase"] != PHASE or plan["fast_state"] != fast_config_dict() or plan["k_grid"] != list(ss.K_GRID) \
            or plan["stack"] != ss.STAGE_C or plan["arms"] != ss.ARMS_C:
        raise ValueError("the preregistration does not describe this runner")
    if plan["r11_amendment_sha256"] != sha(R11_AMENDMENT):
        raise ValueError("the R11 Stage C amendment changed: T1 would not rebuild R11's models")
    return plan


def fast_config_dict() -> dict:
    c = FastStateConfig()
    return {"lr": c.lr, "grad_clip": c.grad_clip, "max_steps": c.max_steps, "delta_bound": c.bound(),
            "momentum": 0.0, "reject_loss_increase": c.reject_loss_increase, "steps_recorded": list(fe.STEP_GRID)}


def _arm_summary(model, rows, info=None) -> dict:
    out = {"rows": rows, "parameters": sum(p.numel() for p in model.parameters()),
           "specialist_core": specialist_core_names(last_channel(model)[2]),
           "specialist_parameters": sum(p.numel() for n, p in model.named_parameters() if ".cgc" in n or ".mlp." in n)}
    if info is not None:
        out.update({"selected_step": info["selected_step"], "trace": info["trace"]})
    gates = [float(e.cgc_gate) for b in model.blocks for e in b.sparse.experts if e.cgc_gate is not None]
    out["gates"] = gates
    return out


@torch.no_grad()
def _ablated_last(model, namespace, family, seed, split, episodes, k=32) -> list:
    """Frozen query MSE with the last block's specialist removed (FFN only, same routing): reliance, not benefit."""
    out = []
    for index in range(episodes):
        tokens, y = ss.episode_tokens(namespace, family, seed, split, index, k)
        pred = model(tokens[None], model.new_state(1), ablate_blocks=(len(model.blocks) - 1,))[:, -y.shape[0]:]
        out.append(float((pred - y[None]).pow(2).mean()))
    return out


# -- T1: R11's Stage C models, rebuilt and verified -----------------------------------------------------------------

def command_reproduce(args) -> dict:
    plan = None if args.dev else frozen()
    r11 = json.loads(R11_AMENDMENT.read_text(encoding="utf-8"))
    namespace = r11["namespace"] if not args.dev else "R11C_DEV"
    split = EVAL_SPLIT_T1 if not args.dev else DEV_SPLIT
    if not args.dev and args.seed not in plan["regimes"]["T1"]["seeds"]:
        raise SystemExit(f"seed {args.seed} is not a T1 seed")
    started = time.time()
    torch.manual_seed(0)                                    # run_r11_stage_c.command_run, statement for statement
    models, r11_rows = {}, {}
    for arm in ss.ARMS_C:
        model = ss.RegressionStack(arm, r11["profile"], args.seed, namespace=namespace)
        info = ss.train_stack(model, namespace, [args.family], args.seed, r11["training"])
        r11_rows[arm] = ss.evaluate_stack(model, namespace, args.family, args.seed, r11["test_episodes"])
        models[arm] = (model, info)
    verification = {}
    if not args.dev:
        record = json.loads((R11 / f"c_{args.family}_s{args.seed}.json").read_text(encoding="utf-8"))
        for arm in ss.ARMS_C:
            mine, theirs = r11_rows[arm], record["arms"][arm]["rows"]
            worst = max(abs(a["curve"][k][0] - b["curve"][k][0]) for a, b in zip(mine, theirs) for k in a["curve"])
            verification[arm] = {"rows_identical": mine == theirs, "curve_max_abs_diff": worst,
                                 "selected_step_equal": models[arm][1]["selected_step"] == record["arms"][arm]["selected_step"],
                                 "trace_equal": models[arm][1]["trace"] == record["arms"][arm]["trace"]}
    episodes = plan["episodes"]["test"] if not args.dev else args.episodes
    arms = {}
    for arm, (model, info) in models.items():
        fe.freeze(model)
        rows = fe.evaluate_fast(model, namespace, args.family, args.seed, split, episodes)
        arms[arm] = _arm_summary(model, rows, info)
        arms[arm]["ablated_last_frozen"] = _ablated_last(model, namespace, args.family, args.seed, split, episodes)
    return {"phase": PHASE, "command": "reproduce", "regime": "T1", "stage": "development" if args.dev else "confirmation",
            "family": args.family, "seed": args.seed, "namespace": namespace, "eval_split": split,
            "r11_verification": verification, "arms": arms, "seconds": round(time.time() - started, 1),
            "provenance": provenance(plan), "status": "COMPLETED"}


# -- T2: the same stack trained with the fast update in the loop ------------------------------------------------------

def command_episodic(args) -> dict:
    plan = None if args.dev else frozen()
    r11 = json.loads(R11_AMENDMENT.read_text(encoding="utf-8"))
    namespace = plan["regimes"]["T2"]["namespace"] if not args.dev else "R12C_DEV"
    if not args.dev and args.seed not in plan["regimes"]["T2"]["seeds"]:
        raise SystemExit(f"seed {args.seed} is not a T2 seed")
    training = dict(r11["training"]) if not args.dev else {**r11["training"], "steps": args.steps}
    started = time.time()
    torch.manual_seed(0)
    arms = {}
    for arm in ss.ARMS_C:
        model = ss.RegressionStack(arm, r11["profile"], args.seed, namespace=namespace)
        t0 = time.time()
        if arm == "G0":
            info = ss.train_stack(model, namespace, [args.family], args.seed, training)
        else:
            info = fe.train_fast(model, namespace, [args.family], args.seed, training)
        seconds = round(time.time() - t0, 1)
        fe.freeze(model)
        episodes = plan["episodes"]["test"] if not args.dev else args.episodes
        rows = fe.evaluate_fast(model, namespace, args.family, args.seed, "TEST", episodes)
        arms[arm] = _arm_summary(model, rows, info)
        arms[arm]["training_seconds"] = seconds
        arms[arm]["ablated_last_frozen"] = _ablated_last(model, namespace, args.family, args.seed, "TEST", episodes)
    return {"phase": PHASE, "command": "episodic", "regime": "T2", "stage": "development" if args.dev else "confirmation",
            "family": args.family, "seed": args.seed, "namespace": namespace, "eval_split": "TEST",
            "training": training, "arms": arms, "seconds": round(time.time() - started, 1),
            "provenance": provenance(plan), "status": "COMPLETED"}


# -- parity ---------------------------------------------------------------------------------------------------------

def _analytic_core_gradient(fast, pre, targets, delta):
    """The PDF's/R97W's local update direction for the bounded core, computed by hand and checked against autograd:
    grad_{w_ci} L = sum_s delta_{s,c} phi'(r~_sci) kappa(r_sci) q_s, r~ = b tanh(r/b), kappa = sech^2(r/b)."""
    from ..cgc import phi_prime
    block, _, expert = last_channel(fast._model)
    core, w = expert.cgc.core, fast.active(delta)["cgc.core.weight"]
    h = block.channel_norm(pre)
    q = expert.cgc.proj_in(h)                                                      # [1, n, 5]
    r = torch.einsum("bnd,dcu->bncu", q, w)
    b = core.preact_bound
    rb, kappa = (r, torch.ones_like(r)) if b is None else (b * torch.tanh(r / b), 1 - torch.tanh(r / b) ** 2)
    cells = (0.5 * rb * (1 + rb * (1 + rb))).sum(-1).detach().requires_grad_(True)  # a_c: [1, n, 3]
    branch = expert.cgc.proj_out(cells)
    branch = branch if expert.cgc_gate is None else expert.cgc_gate * branch
    out = fe.head(fast._model, pre + expert.ffn(h) + branch)
    loss = (out - targets).pow(2).mean()
    (dcells,) = torch.autograd.grad(loss, [cells])                                  # delta_{s,c}
    return torch.einsum("bnc,bncu,bnd->dcu", dcells, phi_prime(rb) * kappa, q)


def command_parity(args) -> dict:
    out = {"phase": PHASE, "command": "parity", "checks": {}, "precision": {}}
    for arm in ("G1", "G2", "G3"):
        adapted = {}
        for dtype in (torch.float32, torch.float64):
            torch.manual_seed(0)
            model = fe.freeze(ss.RegressionStack(arm, "BOUNDED_SPECIALIST", 7, namespace="R12_PARITY").to(dtype))
            probes, py = fe.probe_inputs("R12_PARITY", "smooth", 7, "PARITY", 0, 32)
            probes, py = probes.to(dtype), py.to(dtype)
            tokens, y = ss.episode_tokens("R12_PARITY", "smooth", 7, "PARITY", 0, 32)
            tokens, y = tokens[None].to(dtype), y[None].to(dtype)
            _, probe_pre = frozen_pass(model, probes)
            _, pre = frozen_pass(model, tokens)
            fast = SpecialistFastState(model, session="parity")
            support = ObservedSupport.observed(probes, py, source="parity teacher")
            loss_of = fast.support_loss(support, probe_pre)
            results = fast.adapt(loss_of, steps=8)
            delta = {n: d.clone().requires_grad_(True) for n, d in fast.delta().items()}
            cached = (fast.output(probe_pre, delta) - py).pow(2).mean()
            g_cached = torch.autograd.grad(cached, list(delta.values()))
            full_out = functional_call(model, fast.overrides(delta), (probes, model.new_state(1)))
            full = (full_out - py).pow(2).mean()
            g_full = torch.autograd.grad(full, list(delta.values()))
            check = {"output_full_vs_cached_max_abs": float((functional_call(model, fast.overrides(), (tokens, model.new_state(1)))
                                                             - fast.output(pre)).abs().max()),
                     "loss_full_vs_cached_abs": abs(float(full.detach()) - float(cached.detach())),
                     "gradient_full_vs_cached_max_abs": max(float((a - b).abs().max()) for a, b in zip(g_full, g_cached)),
                     "states_unchanged_by_the_core": fe._state_equal(model, tokens, fast.overrides()),
                     "applied_steps": sum(r.applied for r in results), "projected": any(r.projected for r in results)}
            if arm != "G3":
                analytic = _analytic_core_gradient(fast, probe_pre, py, {n: d.detach() for n, d in delta.items()})
                check["analytic_vs_autograd_max_abs"] = float((analytic - g_cached[0]).abs().max())
            pres = probe_pre.expand(3, *probe_pre.shape[1:]).contiguous()
            batched = fe.batched_fast_update(model, pres, py.expand(3, *py.shape[1:]).contiguous(), FastStateConfig())
            check["batched_vs_transactional_max_abs"] = max(float((fast._base[n] + fast.delta()[n] - batched[n][0]).abs().max())
                                                            for n in fast.names)
            with torch.no_grad():
                adapted[dtype] = (fast.output(pre).double(), {n: d.double() for n, d in fast.delta().items()})
            fast.close()
            out["checks"][f"{arm}_{str(dtype).split('.')[-1]}"] = check
        (o32, d32), (o64, d64) = adapted[torch.float32], adapted[torch.float64]
        out["precision"][arm] = {"adapted_output_float32_vs_float64_max_abs": float((o32 - o64).abs().max()),
                                 "adapted_output_rms": float(o64.pow(2).mean().sqrt()),
                                 "delta_float32_vs_float64_max_abs": max(float((d32[n] - d64[n]).abs().max()) for n in d64)}
    # an ERSA language model with window attention: the last block's core changes no state (cells, tails, window)
    from ..config import ModelConfig
    from ..ersa_model import ErsaConfig, ErsaLM
    from ..sparse_channel import SparseConfig
    from ..cgc_specialist import micro_sparse
    m = ModelConfig(vocab_size=40, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=64, channel="ffn",
                    ffn_hidden=48, cgc_bottleneck=5, cgc_channels=3, cgc_units=8, seed=3)
    sparse = micro_sparse(SparseConfig(experts=1, top_k=1, layers=(0, 1), ffn_hidden=48), profile="BOUNDED_SPECIALIST",
                          gate_mode="fixed")
    lm = fe.freeze(ErsaLM(ErsaConfig(m, ersa_heads=2, key_dim=8, value_dim=8, conv_kernel=4, chunk=16,
                                     attention="window", attn_window=24, sparse=sparse)))
    tokens = torch.randint(0, 40, (1, 24), generator=torch.Generator().manual_seed(5))
    fast = SpecialistFastState(lm, session="lm")
    fast._delta = {n: torch.full_like(d, 0.05) for n, d in fast._delta.items()}   # a nonzero core (test only)
    a, b = lm.new_state(1), lm.new_state(1)
    with torch.no_grad():
        logits_frozen = torch.cat([lm(tokens[:, :16], a), lm(tokens[:, 16:], a)], dim=1)
        logits_fast = torch.cat([functional_call(lm, fast.overrides(), (tokens[:, :16], b)),
                                 functional_call(lm, fast.overrides(), (tokens[:, 16:], b))], dim=1)
    same = (all(torch.equal(x, y) for x, y in zip(a.cells, b.cells))
            and all(torch.equal(x, y) for x, y in zip(a.tails, b.tails) if x is not None)
            and all(torch.equal(la[key], lb[key]) for la, lb in zip(a.window.layers, b.window.layers) for key in la)
            and torch.equal(a.window.tokens, b.window.tokens))
    out["checks"]["ersa_lm_window"] = {"states_unchanged_by_the_core": same,
                                       "logits_changed": not torch.equal(logits_frozen, logits_fast)}
    try:
        SpecialistFastState(lm, session="earlier", block=0)
        out["checks"]["earlier_block_refused"] = False
    except FastStateError:
        out["checks"]["earlier_block_refused"] = True
    out["provenance"] = provenance(None)
    return out


# -- latency --------------------------------------------------------------------------------------------------------

def command_latency(args) -> dict:
    """MEASURED CPU wall-clock, one thread, median of repetitions: every fast-state operation on an R11 stack (G2,
    G3), and the same work by full recomputation. PERFORMANCE_ONLY; no GPU claim."""
    torch.set_num_threads(1)
    out = {"phase": PHASE, "command": "latency", "label": "MEASURED CPU wall-clock, one thread; PERFORMANCE_ONLY", "arms": {}}

    def timed(fn, reps=60):
        for _ in range(5):
            fn()
        samples = []
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t)
        return 1e6 * statistics.median(samples)

    for arm in ("G2", "G3"):
        torch.manual_seed(0)
        model = fe.freeze(ss.RegressionStack(arm, "BOUNDED_SPECIALIST", 7, namespace="R12_LATENCY"))
        probes, py = fe.probe_inputs("R12_LATENCY", "smooth", 7, "LATENCY", 0, 32)
        tokens, y = ss.episode_tokens("R12_LATENCY", "smooth", 7, "LATENCY", 0, 32)
        tokens = tokens[None]
        _, probe_pre = frozen_pass(model, probes)
        _, pre = frozen_pass(model, tokens)
        fast = SpecialistFastState(model, session="latency")
        support = ObservedSupport.observed(probes, py, source="latency teacher")
        loss_of = fast.support_loss(support, probe_pre)
        trial = {n: d.clone().requires_grad_(True) for n, d in fast.delta().items()}

        def gradient():
            loss = loss_of(trial)
            torch.autograd.grad(loss, list(trial.values()))

        def one_update():
            fast.reset()
            fast.adapt(loss_of, steps=1)

        def eight_updates():
            fast.reset()
            fast.adapt(loss_of, steps=8)

        def full_eight():                     # the same 8 updates with every step recomputing the whole stack
            fast.reset()
            fast.adapt(lambda d: (functional_call(model, fast.overrides(d), (probes, model.new_state(1))) - py).pow(2).mean(),
                       steps=8)

        with torch.no_grad():
            frozen_forward = timed(lambda: model(tokens, model.new_state(1)))
            support_eval = timed(lambda: loss_of(fast.delta()))
            adapted_forward = timed(lambda: fast.output(pre))
            full_adapted_forward = timed(lambda: functional_call(model, fast.overrides(), (tokens, model.new_state(1))))
        times = {"frozen_stack_forward_80_tokens": frozen_forward, "support_evaluation_cached_32": support_eval,
                 "gradient_cached_32": timed(gradient), "one_bounded_update": timed(one_update),
                 "eight_bounded_updates": timed(eight_updates, reps=20), "reset": timed(fast.reset),
                 "adapted_query_output_cached": adapted_forward, "adapted_query_output_full": full_adapted_forward,
                 "eight_updates_full_recomputation": timed(full_eight, reps=10)}
        sizes = fast.nbytes()
        out["arms"][arm] = {"median_us": times, "fast_state_bytes": sizes,
                            "support_activations_bytes": probe_pre.numel() * probe_pre.element_size(),
                            "support_targets_bytes": py.numel() * py.element_size()}
        fast.close()
    out["provenance"] = provenance(None)
    return out


# -- combine --------------------------------------------------------------------------------------------------------

def sign_test_p(log_ratios) -> float:
    negative = sum(v < 0 for v in log_ratios)
    n = sum(v != 0 for v in log_ratios)
    return sum(math.comb(n, k) for k in range(negative, n + 1)) / 2 ** n if n else 1.0


def better(groups: list, rule: dict) -> dict:
    pooled = [v for g in groups for v in g]
    gm = math.exp(statistics.fmean(pooled))
    group_gm = [math.exp(statistics.fmean(g)) for g in groups]
    p = sign_test_p(pooled)
    frac = sum(v < 1 for v in group_gm) / len(group_gm)
    return {"pooled_geomean_ratio": gm, "sign_test_p": p, "groups_below_one": frac, "group_geomeans": group_gm,
            "n_episodes": len(pooled), "better": gm <= rule["max_ratio"] and p < rule["alpha"]
            and frac >= rule["min_group_fraction"]}


def _value(row, measure: str) -> float:
    """``measure``: 'frozen' (k=32, no update), 'fast<t>', 'shuffled', 'fast_ood', 'frozen_ood', 'fast_cf',
    'frozen_cf', 'previous_core'."""
    if measure == "frozen":
        return row["frozen"]["query"]
    if measure.startswith("fast") and measure[4:].isdigit():
        return row["fast"][measure[4:]]
    return {"shuffled": lambda: row["shuffled"], "fast_ood": lambda: row["fast_ood"],
            "frozen_ood": lambda: row["frozen"]["ood"], "fast_cf": lambda: row["fast_context_free"],
            "frozen_cf": lambda: row["frozen"]["context_free"], "previous_core": lambda: row["previous_episode_core"]}[measure]()


def _integrity(rows) -> dict:
    """The state-integrity gates over every specialist row of a regime (PREREGISTRATION_R12.md section 6)."""
    checks = {"reset_exact": all(r["reset"]["exact"] for r in rows),
              "disabled_exact": all(r["disabled"]["exact"] for r in rows),
              "disabled_refused": all(set(r["disabled"]["status"]) == {"REFUSED_LEARNING_DISABLED"} for r in rows),
              "no_leakage": all(r["leakage"]["probe_tokens_carry_no_target"] and r["leakage"]["query_support_disjoint"]
                                for r in rows),
              "states_unchanged_by_the_core": all(r["parity"]["states_unchanged_by_the_core"] for r in rows),
              "full_vs_cached_parity": max(r["parity"]["full_vs_cached_max_abs"] for r in rows) <= 1e-5,
              "every_update_applied": all(t["status"] == "APPLIED" for r in rows for t in r["trace"]),
              "bound_never_binds": not any(t["projected"] for r in rows for t in r["trace"]),
              "finite": all(math.isfinite(v) for r in rows for v in (*r["fast"].values(), r["shuffled"],
                                                                     r["fast_ood"], r["fast_context_free"]))}
    checks["all"] = all(checks.values())
    return checks


def analyse_regime(records: dict, plan: dict, regime: str) -> dict:
    rule = plan["rules"]["better"]
    seeds = plan["regimes"][regime]["seeds"]
    groups = [(f, s) for s in seeds for f in st.FAMILIES]
    if any(g not in records for g in groups):
        return {"disposition": "NONQUAL", "missing": [list(g) for g in groups if g not in records]}

    def rows(arm):
        return [records[g]["arms"][arm]["rows"] for g in groups]

    def compare(x_arm, x_measure, y_arm, y_measure):
        logs = [[math.log(max(_value(a, x_measure), 1e-300) / max(_value(b, y_measure), 1e-300)) for a, b in zip(ga, gb)]
                for ga, gb in zip(rows(x_arm), rows(y_arm))]
        return better(logs, rule)

    out = {"groups": [list(g) for g in groups]}
    g0 = [statistics.median(r["frozen"]["query"] for r in grp) for grp in rows("G0")]
    null = [statistics.median(r["null"] for r in grp) for grp in rows("G0")]
    out["assay"] = {"g0_median_k32": g0, "null_median": null,
                    "learned": all(a <= plan["rules"]["max_g0_over_null"] * b for a, b in zip(g0, null))}
    if regime == "T1":
        out["r11_reproduction"] = {f"{f}_{s}": records[(f, s)]["r11_verification"] for f, s in groups}
        out["r11_reproduction_exact"] = all(v["rows_identical"] and v["selected_step_equal"] and v["trace_equal"]
                                            for rec in out["r11_reproduction"].values() for v in rec.values())
    final = f"fast{plan['fast_state']['max_steps']}"
    out["comparisons"] = {}
    for arm in ("G1", "G2"):
        out["comparisons"][arm] = {
            "fast_vs_ffn": compare(arm, final, "G0", "frozen"),
            "fast_vs_fast_gelu": compare(arm, final, "G3", final),
            "fast_vs_frozen_self": compare(arm, final, arm, "frozen"),
            "fast_vs_shuffled": compare(arm, final, arm, "shuffled"),
            "frozen_vs_ffn": compare(arm, "frozen", "G0", "frozen"),
            "fast_vs_frozen_gelu": compare(arm, final, "G3", "frozen"),
            "secondary_context_free_fast_vs_fast_gelu": compare(arm, "fast_cf", "G3", "fast_cf"),
            "secondary_ood_fast_vs_ffn": compare(arm, "fast_ood", "G0", "frozen_ood"),
            "secondary_ood_fast_vs_fast_gelu": compare(arm, "fast_ood", "G3", "fast_ood"),
            "secondary_previous_core_vs_frozen": compare(arm, "previous_core", arm, "frozen")}
    out["comparisons"]["G3"] = {"fast_vs_ffn": compare("G3", final, "G0", "frozen"),
                                "fast_vs_frozen_self": compare("G3", final, "G3", "frozen")}
    out["curves"] = {arm: {str(t): statistics.fmean(statistics.median(_value(r, "frozen" if t == 0 else f"fast{t}")
                                                                    for r in grp) for grp in rows(arm))
                           for t in fe.STEP_GRID} for arm in ("G1", "G2", "G3")}
    out["curves"]["G0"] = {"0": statistics.fmean(statistics.median(r["frozen"]["query"] for r in grp) for grp in rows("G0"))}
    out["integrity"] = {arm: _integrity([r for grp in rows(arm) for r in grp]) for arm in ("G1", "G2", "G3")}
    disp = {}
    for arm in ("G1", "G2"):
        c = out["comparisons"][arm]
        criteria = {"assay_learned": out["assay"]["learned"], "beats_ffn": c["fast_vs_ffn"]["better"],
                    "beats_fast_gelu": c["fast_vs_fast_gelu"]["better"],
                    "adaptation_curve": c["fast_vs_frozen_self"]["better"],
                    "target_driven": c["fast_vs_shuffled"]["better"],
                    "integrity": out["integrity"][arm]["all"] and out["integrity"]["G3"]["all"]}
        if not criteria["integrity"]:
            label = "NONQUAL_INTEGRITY"
        elif not criteria["assay_learned"]:
            label = "ASSAY_NOT_LEARNED"
        elif all(criteria.values()):
            label = "INTEGRATED_FAST_CGC_SPECIALIST_QUALIFIED"
        else:
            label = "NOT_ESTABLISHED"
        disp[arm] = {"disposition": label, "criteria": criteria}
    out["dispositions"] = disp
    return out


def command_combine(args) -> dict:
    plan = frozen()
    expected = {k: v for k, v in provenance(plan).items() if k.endswith("_sha256")}
    out = {"phase": PHASE, "command": "combine", "refused": [], "inputs": {}, "provenance": provenance(plan)}
    by_regime = {"T1": {}, "T2": {}}
    for path in sorted(RESULTS.glob("t[12]_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if any(record["provenance"].get(k) != v for k, v in expected.items()) or record.get("stage") != "confirmation":
            out["refused"].append(path.name)
            continue
        out["inputs"][path.name] = sha(path)
        by_regime[record["regime"]][(record["family"], record["seed"])] = record
    out["T1"] = analyse_regime(by_regime["T1"], plan, "T1")
    out["T2"] = analyse_regime(by_regime["T2"], plan, "T2")
    out["verdict"] = verdict(out, plan)
    return out


def verdict(out: dict, plan: dict) -> dict:
    """The preregistered reading of the two regimes (PREREGISTRATION_R12.md section 7)."""
    d = {r: {arm: out[r].get("dispositions", {}).get(arm, {}).get("disposition", "NONQUAL") for arm in ("G1", "G2")}
         for r in ("T1", "T2")}
    q = "INTEGRATED_FAST_CGC_SPECIALIST_QUALIFIED"
    t1, t2 = (any(v == q for v in d[r].values()) for r in ("T1", "T2"))
    if t1:
        reading = "DIRECT_RESTORATION_QUALIFIED"
    elif t2:
        reading = "QUALIFIED_ONLY_WITH_ADAPTATION_TRAINED_INITIALIZATION"
    elif any(v.startswith("NONQUAL") for r in d.values() for v in r.values()):
        reading = "NONQUAL"
    else:
        reading = "NOT_ESTABLISHED_REPRESENTATION_STAGE_TRIGGERED"
    return {"dispositions": d, "reading": reading,
            "representation_stage": "TRIGGERED" if reading.startswith("NOT_ESTABLISHED") else "NOT_RUN"}


def command_manifest(args) -> dict:
    files = sorted(p for p in RESULTS.glob("*.json") if p.name != "MANIFEST.json")
    return {"phase": PHASE, "command": "manifest", "files": {p.name: sha(p) for p in files}}


def command_audit(args) -> dict:
    """Exact accounting of the fast state and of what stays frozen, per arm (an untrained R11 stack: shapes only)."""
    out = {"phase": PHASE, "command": "audit", "arms": {}}
    for arm in ss.ARMS_C:
        model = fe.freeze(ss.RegressionStack(arm, "BOUNDED_SPECIALIST", 7, namespace="R12_AUDIT"))
        block, channel, expert = last_channel(model)
        names = specialist_core_names(expert)
        prefix = f"blocks.{len(model.blocks) - 1}.sparse.experts.0."
        frozen_names = [n for n, _ in model.named_parameters() if n not in {prefix + c for c in names}]
        entry = {"parameters_total": sum(p.numel() for p in model.parameters()),
                 "frozen_parameters": sum(p.numel() for n, p in model.named_parameters() if n in frozen_names),
                 "fast_core": {prefix + n: list(dict(expert.named_parameters())[n].shape) for n in names},
                 "fast_core_values": sum(dict(expert.named_parameters())[n].numel() for n in names)}
        if names:
            fast = SpecialistFastState(model, session="audit")
            entry["fast_state_bytes"] = fast.nbytes()
            entry["identity_fields"] = sorted(fast.identity())
            fast.close()
        out["arms"][arm] = entry
    out["provenance"] = provenance(None)
    return out


def command_dev(args) -> dict:
    args.dev = True
    record = command_reproduce(args) if args.regime == "T1" else command_episodic(args)
    record["label"] = "DEVELOPMENT ONLY"
    return record


COMMANDS = {"audit": command_audit, "parity": command_parity, "reproduce": command_reproduce,
            "episodic": command_episodic, "latency": command_latency, "combine": command_combine,
            "manifest": command_manifest, "dev": command_dev}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--family", default="smooth", choices=st.FAMILIES)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--regime", default="T1", choices=("T1", "T2"), help="dev only")
    parser.add_argument("--steps", type=int, default=2000, help="dev only (T2 training steps)")
    parser.add_argument("--episodes", type=int, default=12, help="dev only (evaluation episodes)")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", help="the JSON record to write")
    args = parser.parse_args(argv)
    args.dev = getattr(args, "dev", False)
    torch.set_num_threads(args.threads)
    record = COMMANDS[args.command](args)
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    else:
        json.dump({k: v for k, v in record.items() if k in ("phase", "command", "seconds", "verdict")}, sys.stdout,
                  default=str)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
