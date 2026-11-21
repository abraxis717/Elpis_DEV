"""ELPIS_TORCH_R11 Stage C — frozen-weight transfer into the actual ERSA stack (amendment). RESEARCH_ONLY.

Triggered by Stage B's ``EXPERT_SPECIALIZATION_QUALIFIED`` (G2), as ``PREREGISTRATION_R11.md`` §5 requires; run only
under ``results/r11/amendment_stage_c.json``, frozen before any Stage C data. No inner loop and no weight update at
inference: every arm is trained end to end, then evaluated frozen, with the episode's support examples IN CONTEXT.

Commands
    run      one (family, seed) group: arms G0, G1, G2, G3 trained and evaluated (record c_<family>_s<seed>.json)
    router   G2 with two experts per block and a learned top-1 router, trained on mixed families: does the router
             separate the families' tokens? (record c_router_s<seed>.json)
    combine  the amendment's decision rules (record c_study.json)
    dev      development mechanics (namespace R11C_DEV; writes nothing unless --out)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

from .. import specialist_stack as ss
from .. import specialist_tasks as st
from . import run_r11 as r

PHASE = "ELPIS_TORCH_R11_STAGE_C"
AMENDMENT = r.RESULTS / "amendment_stage_c.json"
RUNNER = Path(__file__).resolve()
SOURCES = {**r.SOURCES, "stage_c_runner": RUNNER, "stack": r.REPO / "research/elpis_torch/specialist_stack.py"}


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen() -> dict:
    plan = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if plan["phase"] != PHASE or plan["arms"] != ss.ARMS_C or plan["k_grid"] != list(ss.K_GRID) \
            or plan["constants"] != ss.STAGE_C:
        raise ValueError("the Stage C amendment does not describe this runner")
    if plan["preregistration_sha256"] != sha(r.PREREGISTRATION):
        raise ValueError("the amendment does not belong to this preregistration")
    return plan


def provenance(plan) -> dict:
    out = r.provenance(None)
    out.update({f"{k}_sha256": sha(v) for k, v in SOURCES.items()})
    out["amendment_sha256"] = sha(AMENDMENT) if plan is not None else None
    out["preregistration_sha256"] = sha(r.PREREGISTRATION)
    return out


def command_run(args) -> dict:
    plan = frozen()
    if args.seed not in plan["seeds"]:
        raise SystemExit(f"seed {args.seed} is not a Stage C seed")
    torch.manual_seed(0)
    started = time.time()
    arms = {}
    for arm in ss.ARMS_C:
        model = ss.RegressionStack(arm, plan["profile"], args.seed, namespace=plan["namespace"])
        info = ss.train_stack(model, plan["namespace"], [args.family], args.seed, plan["training"])
        rows = ss.evaluate_stack(model, plan["namespace"], args.family, args.seed, plan["test_episodes"])
        gates = [float(e.cgc_gate) for b in model.blocks for e in b.sparse.experts if e.cgc_gate is not None]
        arms[arm] = {"rows": rows, "selected_step": info["selected_step"], "trace": info["trace"],
                     "gates": gates, "parameters": sum(p.numel() for p in model.parameters()),
                     "specialist_parameters": sum(p.numel() for n, p in model.named_parameters()
                                                  if ".cgc" in n or ".mlp." in n)}
    return {"phase": PHASE, "command": "run", "stage": "confirmation", "family": args.family, "seed": args.seed,
            "profile": plan["profile"], "arms": arms, "seconds": round(time.time() - started, 1),
            "provenance": provenance(plan), "status": "COMPLETED"}


def command_routing(args) -> dict:
    plan = frozen()
    if args.seed not in plan["seeds"]:
        raise SystemExit(f"seed {args.seed} is not a Stage C seed")
    torch.manual_seed(0)
    model = ss.RegressionStack("G2", plan["profile"], args.seed, experts=2, namespace=plan["namespace"])
    info = ss.train_stack(model, plan["namespace"], list(st.FAMILIES), args.seed, plan["training"])
    purity = ss.routing_purity(model, plan["namespace"], args.seed, plan["router_episodes"])
    mse = {f: statistics.median(r_["curve"]["32"][0] for r_ in ss.evaluate_stack(
        model, plan["namespace"], f, args.seed, plan["router_episodes"])) for f in st.FAMILIES}
    return {"phase": PHASE, "command": "router", "stage": "confirmation", "seed": args.seed, "purity": purity,
            "median_k32_mse": mse, "selected_step": info["selected_step"], "provenance": provenance(plan),
            "status": "COMPLETED"}


def command_combine(args) -> dict:
    plan = frozen()
    expected = {k: v for k, v in provenance(plan).items() if k.endswith("_sha256")}
    rule = plan["rules"]["better"]
    runs, routers, refused = {}, {}, []
    for path in sorted(r.RESULTS.glob("c_*.json")):
        if path.name == "c_study.json":
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        if any(rec["provenance"].get(k) != v for k, v in expected.items()):
            refused.append(path.name)
            continue
        (runs if rec["command"] == "run" else routers)[(rec.get("family"), rec["seed"])] = rec
    groups = [(f, s) for s in plan["seeds"] for f in st.FAMILIES]
    out = {"phase": PHASE, "command": "combine", "refused": refused, "provenance": provenance(plan)}
    if any(g not in runs for g in groups):
        out["disposition"] = "NONQUAL"
        out["missing"] = [list(g) for g in groups if g not in runs]
        return out

    def rows(arm):
        return [runs[g]["arms"][arm]["rows"] for g in groups]

    def compare(x, y, step="32"):
        return r.better([r.paired(a, b, step)["log_ratios"] for a, b in zip(rows(x), rows(y))], rule)

    def pooled(arm, key):        # geometric mean of (intervention / intact at k = 32)
        values = [math.log(max(rw[key], 1e-300) / max(rw["curve"]["32"][0], 1e-300)) for g in rows(arm) for rw in g]
        return math.exp(statistics.fmean(values))

    g0_k32 = [statistics.median(rw["curve"]["32"][0] for rw in g) for g in rows("G0")]
    null = [statistics.median(rw["null"] for rw in g) for g in rows("G0")]
    out["assay"] = {"g0_median_k32": g0_k32, "null_median": null,
                    "in_context_learned": all(a <= plan["rules"]["max_g0_k32_over_null"] * b
                                              for a, b in zip(g0_k32, null))}
    out["comparisons_k32"] = {f"{x}/{y}": compare(x, y) for x, y in (
        ("G2", "G0"), ("G1", "G0"), ("G3", "G0"), ("G2", "G3"), ("G1", "G3"), ("G2", "G1"), ("G0", "G2"),
        ("G0", "G1"), ("G0", "G3"))}
    out["comparisons_k1"] = {f"{x}/{y}": compare(x, y, "1") for x, y in (("G2", "G0"), ("G2", "G3"), ("G0", "G2"))}
    out["curves_median"] = {arm: {str(k): statistics.fmean(statistics.median(rw["curve"][str(k)][0] for rw in g)
                                                            for g in rows(arm)) for k in ss.K_GRID}
                            for arm in ss.ARMS_C}
    out["controls"] = {arm: {"ablate_all_over_intact": pooled(arm, "ablate_all"),
                             "ablate_block0_over_intact": pooled(arm, "ablate_block0"),
                             "ablate_block1_over_intact": pooled(arm, "ablate_block1"),
                             "ersa_reset_over_intact": pooled(arm, "ersa_reset_after_support")}
                       for arm in ss.ARMS_C}
    disp = {}
    if not out["assay"]["in_context_learned"]:
        disp = {"G1": "ASSAY_NOT_LEARNED", "G2": "ASSAY_NOT_LEARNED"}
    else:
        for arm in ("G1", "G2"):
            c = out["comparisons_k32"]
            qualified = (c[f"{arm}/G0"]["better"] and c[f"{arm}/G3"]["better"]
                         and out["controls"][arm]["ablate_all_over_intact"] >= plan["rules"]["min_ablation_ratio"])
            if qualified:
                disp[arm] = "FROZEN_INFERENCE_TRANSFER_QUALIFIED"
            elif c[f"G0/{arm}"]["better"]:
                disp[arm] = "NOT_ESTABLISHED_FROZEN_SPECIALIST_WORSE_THAN_FFN"
            else:
                disp[arm] = "NOT_ESTABLISHED"
    out["dispositions"] = disp
    if routers:
        separated = []
        purities = []
        for rec in routers.values():
            for block in rec["purity"].values():
                separated.append(block["families_separated"])
                purities.extend(block["purity"].values())
        out["router"] = {"seeds": sorted(s for _, s in routers), "blocks_separating_families": sum(separated),
                         "blocks_total": len(separated), "mean_purity": statistics.fmean(purities),
                         "disposition": "LEARNED_ROUTING_SEPARATES_FAMILIES"
                         if all(separated) and statistics.fmean(purities) >= plan["rules"]["min_router_purity"]
                         else "INTEGRATION_GAP_LEARNED_ROUTING"}
    return out


def command_dev(args) -> dict:
    plan = {"steps": args.steps, "batch": 16, "adam_lr": 3e-3, "select_every": 250, "dev_episodes": 8}
    out = {"phase": PHASE, "command": "dev", "label": "DEVELOPMENT ONLY (namespace R11C_DEV)", "arms": {}}
    for arm in ss.ARMS_C:
        model = ss.RegressionStack(arm, "BOUNDED_SPECIALIST", args.seed, namespace="R11C_DEV")
        info = ss.train_stack(model, "R11C_DEV", [args.family], args.seed, plan)
        rows = ss.evaluate_stack(model, "R11C_DEV", args.family, args.seed, 4)
        out["arms"][arm] = {"curve": {str(k): statistics.median(rw["curve"][str(k)][0] for rw in rows)
                                      for k in ss.K_GRID}, "selected_step": info["selected_step"]}
    return out


COMMANDS = {"run": command_run, "router": command_routing, "combine": command_combine, "dev": command_dev}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--family", default="smooth", choices=st.FAMILIES)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--steps", type=int, default=400, help="dev only")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", help="the JSON record to write")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    record = COMMANDS[args.command](args)
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    else:
        json.dump({k: v for k, v in record.items() if k in ("phase", "command", "dispositions", "seconds", "router")},
                  sys.stdout, default=str)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
