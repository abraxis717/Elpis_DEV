"""TRAIN-partition calibration: baseline coefficients and preregistered gate scales.

1. ``tune``: the SGD learning rate and the BETA prior strength are chosen on TRAIN router seeds 0-15 by
   the ALL-mixture terminal regret (each baseline gets its best shot on the deployment mixture).
   STATIC_ATTENUATION uses the a-priori half-scale proposal (0.5), not tuned.
2. ``levels``: every admissible baseline, on fresh TRAIN router seeds 16-47, under the ORIGINAL contract.
   The gate scales of PREREGISTRATION.md are computed from these levels by the fixed formulas below,
   before any DEV, VALIDATION or HOLDOUT evaluation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import evaluate
from .env import ALL_MIXTURE

LAB = Path(__file__).resolve().parents[2]
SGD_GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)
BETA_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
TUNE_SEEDS = range(0, 16)
LEVEL_SEEDS = range(16, 48)
EPISODES = 2048
ADMISSIBLE_BASELINES = ("NOOP", "SGD", "BETA", "CGC_REFERENCE", "STATIC_ATTENUATION", "CLEAN_BAYES")
CLEAN_MARGIN_FRACTION = 0.20
ALL_MARGIN_FRACTION = 0.20
ONLINE_BUDGET_FACTOR = 1.10


def all_mean(results: dict, arm: str, metric: str = "terminal") -> np.ndarray:
    return np.mean([results[r][arm][metric] for r in ALL_MIXTURE], axis=0)


def tune(workers: int = 3) -> dict:
    specs = {f"SGD@{e}": ("SGD", {"sgd_eta": e}) for e in SGD_GRID}
    specs.update({f"BETA@{k}": ("BETA", {"beta_kappa": k}) for k in BETA_GRID})
    rec = evaluate.run("TRAIN", TUNE_SEEDS, ALL_MIXTURE, specs, episodes=EPISODES, workers=workers)
    table = {label: round(float(all_mean(rec["results"], label).mean()), 8) for label in specs}
    eta = min(SGD_GRID, key=lambda e: table[f"SGD@{e}"])
    kappa = min(BETA_GRID, key=lambda k: table[f"BETA@{k}"])
    return {"record": rec, "all_terminal": table, "SGD": {"eta": eta}, "BETA": {"kappa": kappa}}


def levels(workers: int = 4) -> dict:
    names = list(ADMISSIBLE_BASELINES) + ["KNOWN_Q_ORACLE"]
    rec = evaluate.run("TRAIN", LEVEL_SEEDS, ALL_MIXTURE, evaluate.specs(names), episodes=EPISODES, workers=workers)
    res = rec["results"]
    summary = {}
    for arm in names:
        summary[arm] = {r: round(float(np.mean(res[r][arm]["terminal"])), 8) for r in ALL_MIXTURE}
        summary[arm]["ALL"] = round(float(all_mean(res, arm).mean()), 8)
        summary[arm]["ONLINE_ALL"] = round(float(all_mean(res, arm, "online").mean()), 8)
    clean_ref = min(ADMISSIBLE_BASELINES[1:], key=lambda a: summary[a]["CLEAN"])
    all_ref = min(ADMISSIBLE_BASELINES[1:], key=lambda a: summary[a]["ALL"])
    noop = summary["NOOP"]
    gates = {
        "clean_reference": clean_ref,
        "clean_gain_train": round(noop["CLEAN"] - summary[clean_ref]["CLEAN"], 8),
        "delta_clean": round(CLEAN_MARGIN_FRACTION * (noop["CLEAN"] - summary[clean_ref]["CLEAN"]), 8),
        "all_reference": all_ref,
        "all_gain_train": round(noop["ALL"] - summary[all_ref]["ALL"], 8),
        "delta_all": round(ALL_MARGIN_FRACTION * (noop["ALL"] - summary[all_ref]["ALL"]), 8),
        "online_budget": round(ONLINE_BUDGET_FACTOR * noop["ONLINE_ALL"], 8),
        "formulas": {
            "delta_clean": f"{CLEAN_MARGIN_FRACTION} x (TRAIN NOOP CLEAN - TRAIN clean reference CLEAN)",
            "delta_all": f"{ALL_MARGIN_FRACTION} x (TRAIN NOOP ALL - TRAIN ALL reference ALL)",
            "online_budget": f"{ONLINE_BUDGET_FACTOR} x TRAIN NOOP online regret on ALL (PROBE_EARLY)",
        },
    }
    return {"record": rec, "summary": summary, "gates": gates}


def main(argv) -> None:
    step = argv[1]
    out = LAB / "evidence" / "train"
    if step == "tune":
        res = tune()
        coeff = {"schema": "cgc-lao-solver-lab.baseline-coefficients.v1",
                 "coefficients": {"SGD": res["SGD"], "BETA": res["BETA"], "STATIC_ATTENUATION": {"attenuation": 0.5}},
                 "selection": "TRAIN router seeds 0-15, 2048 episodes, min ALL-mixture terminal regret; "
                              "STATIC_ATTENUATION fixed a priori at 0.5",
                 "all_terminal_table": res["all_terminal"]}
        (LAB / "coefficients" / "baselines.v1.json").write_text(json.dumps(coeff, indent=1) + "\n", encoding="utf-8")
        digest = evaluate.write(evaluate.seal(res["record"], "TRAIN-tune"), out / "tune.json")
        print(json.dumps(coeff, indent=1), digest)
    elif step == "levels":
        res = levels()
        rec = evaluate.seal(res["record"], "TRAIN-levels")
        rec["summary"], rec["gates"] = res["summary"], res["gates"]
        digest = evaluate.write(rec, out / "levels.json")
        print(json.dumps({"summary": res["summary"], "gates": res["gates"]}, indent=1), digest)
    else:
        raise SystemExit("usage: calibrate tune|levels")


if __name__ == "__main__":
    main(sys.argv)
