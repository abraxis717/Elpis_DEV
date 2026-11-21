"""ELPIS_TORCH_R12 P5: the representation seam. DIAGNOSIS ONLY; RESEARCH_ONLY.

Why it runs. The R12 combine read ``NOT_ESTABLISHED_REPRESENTATION_STAGE_TRIGGERED``: no arm qualified in either
regime. Before any change to the updater, directive P5 asks whether the last block's 120-parameter specialist
receives the information its adaptation protocol needs. P8 asks whether the failure is explained by initialization,
task-feature access, routing or a demonstrated absence of benefit. The plan is ``results/r12/amendment_p5.json``,
frozen before any P5 data.

Nothing here is a deployable path, and nothing here can qualify an arm. The Stage B condition reads the exact task
coordinates, an oracle by construction: it is a diagnosis only.

Commands
    stage_b  R11 Stage B's procedure, unchanged (``run_r11.train_stage_b`` with R11's frozen plan and profile, then
             ``specialist_tasks.evaluate_arm``). It is trained on ONE R12 group's own tasks and evaluated on the SAME
             episodes as that group's R12 record: arms G0, G2 and G3 for both families of one seed, plus R11's
             misrouted-expert control for G2. Record ``p5_b_<regime>_s<seed>.json``
    stack    one T2 seed, both families, arm G2:
             - rebuilt by ``fast_episodes.train_fast`` and verified against the T2 record;
             - trained again with the last block's ``P_in`` fixed at its initial value;
             - the other family's specialist branch, and its whole last-block expert, substituted (misrouting);
             - the linear decodability of the task coordinates from the specialist's code and from its input.
             Record ``p5_c_s<seed>.json``
    combine  the amendment's frozen readings (``study_p5.json``)
    dev      development mechanics only: R11's own Stage B group ``R11B``/201 rebuilt and compared with its record,
             or the stack mechanics in ``R12C_DEV`` at a few steps. Writes nothing unless ``--out``
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

from .. import fast_episodes as fe
from .. import specialist_stack as ss
from .. import specialist_tasks as st
from ..fast_state import frozen_pass, last_channel
from . import run_r11 as r11
from . import run_r12 as r12

PHASE = "ELPIS_TORCH_R12_P5_REPRESENTATION_SEAM"
RESULTS = r12.RESULTS
AMENDMENT = RESULTS / "amendment_p5.json"
STUDY = RESULTS / "study_r12.json"
SOURCES = {"p5_runner": Path(__file__).resolve(), "r11_runner": r12.TORCH / "experiments" / "run_r11.py"}
TRIGGER = "NOT_ESTABLISHED_REPRESENTATION_STAGE_TRIGGERED"
LAST = "blocks.1.sparse.experts.0."                 # the last block's single expert (R11's 2-block stack)
BRANCH = ("cgc.proj_in.weight", "cgc.core.weight", "cgc.proj_out.weight")
FIXED = "cgc.proj_in.weight"


def provenance(plan) -> dict:
    """R12's provenance (every R12 source digest, the git revision with the dirty check), plus this runner, R11's
    runner, the amendment, the R12 plan and the R12 study."""
    out = r12.provenance(None)
    out.update({f"{k}_sha256": r12.sha(v) for k, v in SOURCES.items()})
    out.update({"preregistration_sha256": r12.sha(r12.PREREGISTRATION),
                "study_r12_sha256": r12.sha(STUDY) if STUDY.exists() else None,
                "amendment_sha256": r12.sha(AMENDMENT) if plan is not None else None})
    return out


def frozen_p5() -> dict:
    plan = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    study = json.loads(STUDY.read_text(encoding="utf-8"))
    if plan["phase"] != PHASE:
        raise ValueError("the amendment does not describe this runner")
    if plan["trigger"]["study_r12_sha256"] != r12.sha(STUDY) or \
            plan["trigger"]["preregistration_sha256"] != r12.sha(r12.PREREGISTRATION):
        raise ValueError("the R12 study or plan changed after the P5 freeze")
    if study["verdict"]["reading"] != TRIGGER or plan["trigger"]["reading"] != TRIGGER:
        raise ValueError("P5 runs only after the reading " + TRIGGER)
    if plan["stage_b"]["plan_b"] != json.loads(r11.PREREGISTRATION.read_text(encoding="utf-8"))["stage_b"]["training"]:
        raise ValueError("the Stage B plan differs from R11's frozen plan")
    return plan


# -- the Stage B condition: R11's procedure on an R12 group's own tasks --------------------------------------------

def run_stage_b(namespace: str, seed: int, split: str, episodes: int, profile: str, plan_b: dict, arms) -> dict:
    """R11's ``command_stage_b`` loop, restricted to ``arms`` (every arm trains for both families before any
    evaluation; ``torch.manual_seed(0)`` first, as R11 did)."""
    torch.manual_seed(0)
    trained = {}
    for family in st.FAMILIES:
        for arm in arms:
            started = time.time()
            trained[(family, arm)] = r11.train_stage_b(arm, profile, family, seed, namespace, plan_b)
            trained[(family, arm)]["seconds"] = round(time.time() - started, 1)
    groups = {}
    for family in st.FAMILIES:
        other = next(f for f in st.FAMILIES if f != family)
        groups[family] = {}
        for arm in arms:
            t = trained[(family, arm)]
            entry = {"rows": st.evaluate_arm(t["expert"], t["context"], family, seed, namespace, split, episodes),
                     "selected_step": t["selected_step"], "selected_dev_score": t["selected_dev_score"],
                     "training_seconds": t["seconds"]}
            if arm == "G2":           # R11's expert_misrouted control: the other family's expert on these episodes
                entry["misrouted_rows"] = st.evaluate_arm(trained[(other, arm)]["expert"], t["context"], family,
                                                          seed, namespace, split, episodes)
            groups[family][arm] = entry
    return groups


def command_stage_b(args) -> dict:
    started = time.time()
    if args.dev:              # R11's own Stage B group, rebuilt through this path and compared with its record
        plan_b = json.loads(r11.PREREGISTRATION.read_text(encoding="utf-8"))["stage_b"]["training"]
        record = json.loads((r11.RESULTS / "stage_b_s201.json").read_text(encoding="utf-8"))
        groups = run_stage_b("R11B", 201, record["split"], st.STAGE_B["test_episodes"], record["profile"], plan_b,
                             ("G0", "G2", "G3"))
        check = {f"{f}_{a}": groups[f][a]["rows"] == record["groups"][f][a]["rows"] for f in groups for a in groups[f]}
        check.update({f"{f}_G2_misrouted": groups[f]["G2"]["misrouted_rows"] == record["groups"][f]["G2"]["expert_misrouted_rows"]
                      for f in groups})
        return {"phase": PHASE, "command": "stage_b", "stage": "development", "label": "DEVELOPMENT ONLY",
                "rebuilt": "R11B seed 201 (R11's own Stage B record)", "identical_to_r11": check,
                "seconds": round(time.time() - started, 1), "provenance": provenance(None)}
    plan = frozen_p5()
    group = plan["groups"][args.regime]
    if args.seed not in group["seeds"]:
        raise SystemExit(f"seed {args.seed} is not a {args.regime} seed")
    groups = run_stage_b(group["namespace"], args.seed, group["split"], group["episodes"], plan["stage_b"]["profile"],
                         plan["stage_b"]["plan_b"], tuple(plan["stage_b"]["arms"]))
    return {"phase": PHASE, "command": "stage_b", "stage": "diagnosis", "label": plan["label"], "regime": args.regime,
            "seed": args.seed, "namespace": group["namespace"], "split": group["split"], "groups": groups,
            "seconds": round(time.time() - started, 1), "provenance": provenance(plan), "status": "COMPLETED"}


# -- the stack conditions (T2, arm G2) -------------------------------------------------------------------------------

def train_g2(namespace: str, family: str, seed: int, training: dict, profile: str, *, fixed_projection=False):
    """T2's G2, exactly as ``run_r12 episodic`` trains it; ``fixed_projection``: the last block's ``P_in`` never
    learns (it keeps its initial value; nothing else changes)."""
    torch.manual_seed(0)
    model = ss.RegressionStack("G2", profile, seed, namespace=namespace)
    fixed = dict(model.named_parameters())[LAST + FIXED]
    initial = fixed.detach().clone()
    if fixed_projection:
        fixed.requires_grad_(False)
    started = time.time()
    info = fe.train_fast(model, namespace, [family], seed, training)
    info["seconds"] = round(time.time() - started, 1)
    info["projection_unchanged"] = bool(torch.equal(dict(model.named_parameters())[LAST + FIXED].detach(), initial))
    return fe.freeze(model), info


def substituted(model, donor, names, namespace: str, seed: int, profile: str):
    """A new frozen G2 stack: ``model``'s weights with ``donor``'s tensors at ``names``. Built before any session
    opens on it; neither source model is touched."""
    state = {k: v.clone() for k, v in model.state_dict().items()}
    theirs = donor.state_dict()
    for name in names:
        state[name] = theirs[name].clone()
    out = ss.RegressionStack("G2", profile, seed, namespace=namespace)
    out.load_state_dict(state)
    return fe.freeze(out)


def compact(rows) -> list:
    """The P5 measurements of ``evaluate_fast`` rows, each with its integrity gates (the full rows are not stored)."""
    return [{"episode": r["episode"], "frozen": r["frozen"]["query"], "fast": r["fast"], "shuffled": r["shuffled"],
             "frozen_context_free": r["frozen"]["context_free"], "fast_context_free": r["fast_context_free"],
             "grad_norm_first": r["trace"][0]["grad_norm"], "alignment_r2": r["alignment_r2"],
             "integrity": r12._integrity([r])} for r in rows]


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)


@torch.no_grad()
def decodability(model, namespace: str, family: str, seed: int, fit: dict, split: str, episodes: int, k: int = 32) -> dict:
    """Out-of-sample linear decodability (R²) of the 4 task coordinates ``z`` from the last block's specialist code
    ``q = P_in h`` (5 values) and from its input ``h = channel_norm(pre)`` (64 values). The map is fitted by least
    squares on ``fit`` (DEV) episodes and scored per evaluation episode, for the context-free probes and for the
    queries in context."""
    block, _, unit = last_channel(model)
    project = unit.cgc.proj_in

    def features(split_, index):
        e = st.teacher(namespace, family, seed, split_, index)
        probes, _ = fe.probe_inputs(namespace, family, seed, split_, index, k)
        tokens, _ = ss.episode_tokens(namespace, family, seed, split_, index, k)
        hp = block.channel_norm(frozen_pass(model, probes)[1])[0]
        hq = block.channel_norm(frozen_pass(model, tokens[None])[1])[0, k:]
        return {"probe": (hp, project(hp), e.support[0][:k, :st.FEATURES].float()),
                "query": (hq, project(hq), e.query[0][:, :st.FEATURES].float())}

    train = [features(fit["split"], i) for i in range(fit["episodes"])]
    test = [features(split, i) for i in range(episodes)]
    out = {}
    for kind in ("probe", "query"):
        for name, slot in (("code", 1), ("stream", 0)):
            x = torch.cat([f[kind][slot] for f in train]).double()
            z = torch.cat([f[kind][2] for f in train]).double()
            ones = torch.ones(x.shape[0], 1, dtype=x.dtype)
            coef = torch.linalg.lstsq(torch.cat([x, ones], 1), z).solution
            scores = []
            for f in test:
                xe, ze = f[kind][slot].double(), f[kind][2].double()
                pred = torch.cat([xe, torch.ones(xe.shape[0], 1, dtype=xe.dtype)], 1) @ coef
                total = float((ze - ze.mean(0, keepdim=True)).pow(2).sum())
                scores.append(1.0 - float((ze - pred).pow(2).sum()) / total)
            out[f"{kind}_{name}"] = scores
    return out


def run_stack(namespace: str, seed: int, training: dict, profile: str, episodes: int, fit: dict, records: dict) -> dict:
    """Every stack condition for one seed (both families). ``records``: family -> the T2 confirmation record (None in
    development: no verification)."""
    models, out = {}, {}
    for family in st.FAMILIES:
        model, info = train_g2(namespace, family, seed, training, profile)
        rows = fe.evaluate_fast(model, namespace, family, seed, "TEST", episodes)
        entry = {"rebuilt": {"rows": compact(rows), "selected_step": info["selected_step"],
                             "training_seconds": info["seconds"]}}
        if records.get(family) is not None:
            theirs = records[family]["arms"]["G2"]
            entry["rebuilt"]["verification"] = {
                "rows_identical": _canonical(rows) == _canonical(theirs["rows"]),
                "selected_step_equal": info["selected_step"] == theirs["selected_step"],
                "trace_equal": [(t["step"], t["dev_k32"]) for t in info["trace"]]
                == [(t["step"], t["dev_k32"]) for t in theirs["trace"]]}
        entry["decodability"] = {"learned": decodability(model, namespace, family, seed, fit, "TEST", episodes)}
        fixed, finfo = train_g2(namespace, family, seed, training, profile, fixed_projection=True)
        entry["fixed_projection"] = {"rows": compact(fe.evaluate_fast(fixed, namespace, family, seed, "TEST", episodes)),
                                     "selected_step": finfo["selected_step"], "training_seconds": finfo["seconds"],
                                     "projection_unchanged": finfo["projection_unchanged"]}
        entry["decodability"]["fixed"] = decodability(fixed, namespace, family, seed, fit, "TEST", episodes)
        models[family] = model
        out[family] = entry
    for family in st.FAMILIES:
        other = next(f for f in st.FAMILIES if f != family)
        mine, donor = models[family], models[other]
        everything = [n for n in mine.state_dict() if n.startswith(LAST)]
        for label, names in (("misrouted_branch", [LAST + n for n in BRANCH]), ("misrouted_unit", everything)):
            swapped = substituted(mine, donor, names, namespace, seed, profile)
            out[family][label] = {"substituted": names,
                                  "rows": compact(fe.evaluate_fast(swapped, namespace, family, seed, "TEST", episodes))}
    return out


def command_stack(args) -> dict:
    started = time.time()
    r11c = json.loads(r12.R11_AMENDMENT.read_text(encoding="utf-8"))
    if args.dev:
        training = {**r11c["training"], "steps": args.steps}
        families = run_stack("R12C_DEV", args.seed, training, r11c["profile"], args.episodes,
                             {"split": "DEV", "episodes": 8}, {})
        return {"phase": PHASE, "command": "stack", "stage": "development", "label": "DEVELOPMENT ONLY",
                "families": families, "seconds": round(time.time() - started, 1), "provenance": provenance(None)}
    plan = frozen_p5()
    group = plan["groups"]["T2"]
    if args.seed not in group["seeds"]:
        raise SystemExit(f"seed {args.seed} is not a T2 seed")
    records = {f: json.loads((RESULTS / f"t2_{f}_s{args.seed}.json").read_text(encoding="utf-8")) for f in st.FAMILIES}
    families = run_stack(group["namespace"], args.seed, dict(r11c["training"]), r11c["profile"], group["episodes"],
                         plan["stack"]["decodability_fit"], records)
    return {"phase": PHASE, "command": "stack", "stage": "diagnosis", "label": plan["label"], "seed": args.seed,
            "namespace": group["namespace"], "families": families, "seconds": round(time.time() - started, 1),
            "provenance": provenance(plan), "status": "COMPLETED"}


# -- combine ----------------------------------------------------------------------------------------------------------

def _paired(xs: list, ys: list, rule: dict) -> dict:
    """``xs``, ``ys``: per group, per episode values. The R12 rule for "X better than Y" (``run_r12.better``)."""
    logs = [[math.log(max(a, 1e-300) / max(b, 1e-300)) for a, b in zip(gx, gy)] for gx, gy in zip(xs, ys)]
    return r12.better(logs, rule)


def _three_way(x_vs_y: dict, y_vs_x: dict, yes: str, no: str, neither: str) -> str:
    return yes if x_vs_y["better"] else (no if y_vs_x["better"] else neither)


def command_combine(args) -> dict:
    plan = frozen_p5()
    r12plan = r12.frozen()
    rule = r12plan["rules"]["better"]
    study = json.loads(STUDY.read_text(encoding="utf-8"))
    expected = {k: v for k, v in provenance(plan).items() if k.endswith("_sha256")}
    out = {"phase": PHASE, "command": "combine", "label": plan["label"], "refused": [], "inputs": {},
           "provenance": provenance(plan)}
    records = {}
    for path in sorted(RESULTS.glob("p5_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if any(record["provenance"].get(k) != v for k, v in expected.items()) or record.get("stage") != "diagnosis":
            out["refused"].append(path.name)
            continue
        out["inputs"][path.name] = r12.sha(path)
        records[path.name] = record
    groups = {r: [(f, s) for s in plan["groups"][r]["seeds"] for f in st.FAMILIES] for r in ("T1", "T2")}
    missing = [f"p5_b_{r}_s{s}.json" for r in ("T1", "T2") for s in plan["groups"][r]["seeds"]
               if f"p5_b_{r}_s{s}.json" not in records]
    missing += [f"p5_c_s{s}.json" for s in plan["groups"]["T2"]["seeds"] if f"p5_c_s{s}.json" not in records]
    if missing:
        out.update({"disposition": "NONQUAL", "missing": missing})
        return out

    def confirmation(regime, family, seed):
        return json.loads((RESULTS / f"{regime.lower()}_{family}_s{seed}.json").read_text(encoding="utf-8"))

    def b_rows(regime, arm, key="rows"):
        return [records[f"p5_b_{regime}_s{s}.json"]["groups"][f][arm][key] for f, s in groups[regime]]

    def b_fast(rows):
        return [r["curve"]["8"][0] for r in rows]

    readings, comparisons = {}, {}
    for regime in ("T1", "T2"):
        g0_context = [[r["frozen"]["query"] for r in confirmation(regime, f, s)["arms"]["G0"]["rows"]] for f, s in groups[regime]]
        stack_g2 = [[r["fast"]["8"] for r in confirmation(regime, f, s)["arms"]["G2"]["rows"]] for f, s in groups[regime]]
        sb = {arm: [b_fast(rows) for rows in b_rows(regime, arm)] for arm in plan["stage_b"]["arms"]}
        sb_misrouted = [b_fast(rows) for rows in b_rows(regime, "G2", "misrouted_rows")]
        c = {"stage_b_g2_vs_stage_b_g0": _paired(sb["G2"], sb["G0"], rule),
             "stage_b_g2_vs_stage_b_g3": _paired(sb["G2"], sb["G3"], rule),
             "stage_b_g2_vs_in_context_g0": _paired(sb["G2"], g0_context, rule),
             "in_context_g0_vs_stage_b_g2": _paired(g0_context, sb["G2"], rule),
             "stage_b_g2_vs_stage_b_misrouted": _paired(sb["G2"], sb_misrouted, rule),
             "stage_b_g2_vs_stack_g2": _paired(sb["G2"], stack_g2, rule),
             "stack_g2_vs_stage_b_g2": _paired(stack_g2, sb["G2"], rule)}
        comparisons[regime] = c
        readings[regime] = {
            "stage_b_assay": "REPRODUCED" if c["stage_b_g2_vs_stage_b_g0"]["better"] else "NOT_REPRODUCED",
            "benefit_ceiling": _three_way(c["stage_b_g2_vs_in_context_g0"], c["in_context_g0_vs_stage_b_g2"],
                                          "HEADROOM_UNDER_COORDINATE_ACCESS", "IN_CONTEXT_INFERENCE_BETTER",
                                          "INDETERMINATE"),
            "task_feature_access": _three_way(c["stage_b_g2_vs_stack_g2"], c["stack_g2_vs_stage_b_g2"],
                                              "COORDINATE_ACCESS_BETTER", "STACK_REPRESENTATION_BETTER",
                                              "NO_DIFFERENCE_ESTABLISHED"),
            "stage_b_identity": "IDENTITY_MATTERS" if c["stage_b_g2_vs_stage_b_misrouted"]["better"] else "NOT_ESTABLISHED"}

    stack = {s: records[f"p5_c_s{s}.json"]["families"] for s in plan["groups"]["T2"]["seeds"]}

    def stack_fast(label):
        return [[r["fast"]["8"] for r in stack[s][f][label]["rows"]] for f, s in groups["T2"]]

    learned, fixed = stack_fast("rebuilt"), stack_fast("fixed_projection")
    c = {"learned_vs_fixed_projection": _paired(learned, fixed, rule), "fixed_vs_learned_projection": _paired(fixed, learned, rule),
         "correct_vs_misrouted_branch": _paired(learned, stack_fast("misrouted_branch"), rule),
         "correct_vs_misrouted_unit": _paired(learned, stack_fast("misrouted_unit"), rule)}
    comparisons["T2_stack"] = c
    readings["T2_stack"] = {
        "projection": _three_way(c["learned_vs_fixed_projection"], c["fixed_vs_learned_projection"],
                                 "LEARNED_PROJECTION_BETTER", "FIXED_PROJECTION_BETTER", "NO_DIFFERENCE_ESTABLISHED"),
        "identity_branch": "IDENTITY_MATTERS" if c["correct_vs_misrouted_branch"]["better"] else "NOT_ESTABLISHED",
        "identity_unit": "IDENTITY_MATTERS" if c["correct_vs_misrouted_unit"]["better"] else "NOT_ESTABLISHED"}
    decode = {}
    for model in ("learned", "fixed"):
        for key in ("probe_code", "probe_stream", "query_code", "query_stream"):
            values = [v for s in stack for f in stack[s] for v in stack[s][f]["decodability"][model][key]]
            decode[f"{model}_{key}"] = statistics.median(values)
    integrity = {
        "rebuilt_identical": all(stack[s][f]["rebuilt"]["verification"][k] for s in stack for f in stack[s]
                                 for k in ("rows_identical", "selected_step_equal", "trace_equal")),
        "fixed_projection_unchanged": all(stack[s][f]["fixed_projection"]["projection_unchanged"] for s in stack for f in stack[s]),
        "gates": all(r["integrity"]["all"] for s in stack for f in stack[s]
                     for label in ("rebuilt", "fixed_projection", "misrouted_branch", "misrouted_unit")
                     for r in stack[s][f][label]["rows"])}
    integrity["all"] = all(integrity.values())
    main_t1 = study["T1"]["dispositions"]["G2"]["criteria"]["adaptation_curve"]
    main_t2 = study["T2"]["dispositions"]["G2"]["criteria"]["adaptation_curve"]
    attribution = {
        "initialization": ("EXPLAINS_THE_T1_CURVE" if main_t2 and not main_t1 else
                           "NOT_THE_EXPLANATION" if main_t1 else "NOT_SUFFICIENT"),
        "task_feature_access": readings["T2"]["task_feature_access"],
        "routing": "FIXED_SINGLE_EXPERT_NOT_A_FACTOR; identity " + readings["T2_stack"]["identity_branch"],
        "absence_of_benefit": {r: readings[r]["benefit_ceiling"] for r in ("T1", "T2")}}
    out.update({"groups": {r: [list(g) for g in groups[r]] for r in groups}, "comparisons": comparisons,
                "readings": readings, "decodability_median": decode, "integrity": integrity,
                "attribution": attribution, "disposition": "COMPLETED" if integrity["all"] else "NONQUAL_INTEGRITY"})
    return out


def command_dev(args) -> dict:
    args.dev = True
    return command_stage_b(args) if args.part == "stage_b" else command_stack(args)


COMMANDS = {"stage_b": command_stage_b, "stack": command_stack, "combine": command_combine, "dev": command_dev}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--regime", default="T2", choices=("T1", "T2"))
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--part", default="stack", choices=("stage_b", "stack"), help="dev only")
    parser.add_argument("--steps", type=int, default=20, help="dev only (stack training steps)")
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
        json.dump({k: v for k, v in record.items() if k in ("phase", "command", "seconds", "identical_to_r11",
                                                             "readings", "attribution")}, sys.stdout, default=str)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
