"""v2 TRAIN-partition calibration (fresh SLB2 TRAIN seeds; nothing from v1 TRAIN is reused).

1. ``cgc``: retrain the CGC_REFERENCE coefficients with the v1 ES procedure (same hyper-parameters and BCE
   objective, ``cgc_lao_lab.train_cgc``) on SLB2 TRAIN batches with SLB2-keyed ES noise.
2. ``tune``: original-contract baselines (SGD eta, BETA kappa) under ORIGINAL, and the equal-information
   controls (SGD_ALL eta, BETA_ALL kappa, DISAGREE_SGD eta) under DUP2_4 with every reading, on TRAIN router
   seeds 0-15 by ALL-mixture terminal regret.
3. ``levels``: every original-contract baseline (and the privileged oracle) on TRAIN seeds 16-47; the
   preregistered gate scales follow from fixed formulas.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from cgc_lao_lab import rng as rng1
from cgc_lao_lab import train_cgc as tc
from cgc_lao_lab.cgc import frozen_weights, weights_digest

from . import V2_ROOT, env2, evaluate2
from .env2 import ALL_MIXTURE

SGD_GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)
BETA_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
DISAGREE_GRID = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)
TUNE_SEEDS, LEVEL_SEEDS, EPISODES = range(0, 16), range(16, 48), 2048
ORIGINAL_BASELINES = ("NOOP", "SGD", "BETA", "CGC_REFERENCE", "STATIC_ATTENUATION", "CLEAN_BAYES")
CLEAN_MARGIN_FRACTION, ALL_MARGIN_FRACTION, ONLINE_BUDGET_FACTOR = 0.20, 0.20, 1.10
CGC_CONFIG = {**tc.CONFIG, "partition": "TRAIN"}


def _batches(g: int):
    seeds, n = CGC_CONFIG["train_seeds"], CGC_CONFIG["episodes_per_batch"]
    return [env2.episodes("TRAIN", seeds[(3 * g + j) % len(seeds)], r, n * (g // len(seeds)), n)
            for j, r in enumerate(CGC_CONFIG["regimes"])]


def train_cgc(log=print) -> dict:
    cfg = CGC_CONFIG
    theta = cfg["init_scale"] * rng1.normal(env2.derive_key("TRAIN", "cgc-es", "init"), 120)
    m, v = np.zeros(120), np.zeros(120)
    b1, b2, eps = cfg["adam"]
    history, t0 = [], time.time()
    for g in range(cfg["generations"]):
        batches = _batches(g)
        noise = rng1.normal(env2.derive_key("TRAIN", "cgc-es", "noise", g), cfg["pairs"] * 120).reshape(cfg["pairs"], 120)
        fit = np.array([f for e in noise for f in (tc.objective(theta + cfg["sigma"] * e, batches),
                                                   tc.objective(theta - cfg["sigma"] * e, batches))])
        shaped = tc.centred_ranks(fit)
        grad = ((shaped[0::2] - shaped[1::2])[:, None] * noise).sum(0) / (cfg["pairs"] * cfg["sigma"])
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad * grad
        theta = theta - cfg["lr"] * (m / (1 - b1 ** (g + 1))) / (np.sqrt(v / (1 - b2 ** (g + 1))) + eps)
        if g % 50 == 0 or g == cfg["generations"] - 1:
            history.append([g, round(float(fit.mean()), 7)])
            log(f"gen {g} pop-bce {fit.mean():.6f} ({time.time() - t0:.0f}s)")
    w = frozen_weights(theta.astype(np.float32))
    check = {}
    for regime in cfg["regimes"]:
        bw, b0 = [], []
        for seed in cfg["check_seeds"]:
            batch = env2.episodes("TRAIN", seed, regime, 0, 1024)
            bw.append(tc.objective(w.astype(np.float64).ravel(), [batch]))
            b0.append(tc.objective(np.zeros(120), [batch]))
        check[regime] = {"trained_bce": round(float(np.mean(bw)), 7), "zero_bce": round(float(np.mean(b0)), 7)}
    return {"schema": "cgc-lao-solver-lab.cgc-coefficients.v2", "arm": "CGC_REFERENCE", "shape": [3, 5, 8],
            "dtype": "float32", "weights_float32_hex": [float(x).hex() for x in w.ravel()],
            "weights_sha256": weights_digest(w),
            "training": {**cfg, "rng_root": env2.ROOT, "objective": "v1 procedure (cgc_lao_lab.train_cgc) on SLB2 TRAIN",
                         "history": history, "seconds": round(time.time() - t0, 1)},
            "check": check}


def all_mean(res, arm, metric="terminal"):
    return np.mean([res[r][arm][metric] for r in ALL_MIXTURE], axis=0)


def tune(workers=3) -> dict:
    s = evaluate2.spec
    specs = {f"SGD@{e}": s("SGD", "ORIGINAL", "FIRST_ONLY", sgd_eta=e) for e in SGD_GRID}
    specs.update({f"BETA@{k}": s("BETA", "ORIGINAL", "FIRST_ONLY", beta_kappa=k) for k in BETA_GRID})
    specs.update({f"SGD_ALL@{e}": s("SGD_ALL", sgd_all_eta=e) for e in SGD_GRID})
    specs.update({f"BETA_ALL@{k}": s("BETA_ALL", beta_all_kappa=k) for k in BETA_GRID})
    specs.update({f"DISAGREE_SGD@{e}": s("DISAGREE_SGD", disagree_eta=e) for e in DISAGREE_GRID})
    rec = evaluate2.run("TRAIN", TUNE_SEEDS, ALL_MIXTURE, specs, EPISODES, workers)
    table = {label: round(float(all_mean(rec["results"], label).mean()), 8) for label in specs}
    pick = lambda prefix, grid: min(grid, key=lambda g: table[f"{prefix}@{g}"])  # noqa: E731
    coeff = {"SGD": {"eta": pick("SGD", SGD_GRID)}, "BETA": {"kappa": pick("BETA", BETA_GRID)},
             "SGD_ALL": {"eta": pick("SGD_ALL", SGD_GRID)}, "BETA_ALL": {"kappa": pick("BETA_ALL", BETA_GRID)},
             "DISAGREE_SGD": {"eta": pick("DISAGREE_SGD", DISAGREE_GRID)}, "STATIC_ATTENUATION": {"attenuation": 0.5}}
    return {"record": rec, "table": table, "coefficients": coeff}


def levels(workers=4) -> dict:
    s = evaluate2.spec
    specs = {a: s(a, "ORIGINAL", "FIRST_ONLY") for a in ORIGINAL_BASELINES}
    specs["KNOWN_Q_ORACLE"] = s("KNOWN_Q_ORACLE", "ORIGINAL", "FIRST_ONLY")
    rec = evaluate2.run("TRAIN", LEVEL_SEEDS, ALL_MIXTURE, specs, EPISODES, workers)
    res = rec["results"]
    summary = {a: {**{r: round(float(np.mean(res[r][a]["terminal"])), 8) for r in ALL_MIXTURE},
                   "ALL": round(float(all_mean(res, a).mean()), 8),
                   "ONLINE_ALL": round(float(all_mean(res, a, "online").mean()), 8)} for a in specs}
    clean_ref = min(ORIGINAL_BASELINES[1:], key=lambda a: summary[a]["CLEAN"])
    all_ref = min(ORIGINAL_BASELINES[1:], key=lambda a: summary[a]["ALL"])
    noop = summary["NOOP"]
    gates = {"clean_reference": clean_ref, "clean_gain_train": round(noop["CLEAN"] - summary[clean_ref]["CLEAN"], 8),
             "delta_clean": round(CLEAN_MARGIN_FRACTION * (noop["CLEAN"] - summary[clean_ref]["CLEAN"]), 8),
             "all_reference": all_ref, "all_gain_train": round(noop["ALL"] - summary[all_ref]["ALL"], 8),
             "delta_all": round(ALL_MARGIN_FRACTION * (noop["ALL"] - summary[all_ref]["ALL"]), 8),
             "online_budget": round(ONLINE_BUDGET_FACTOR * noop["ONLINE_ALL"], 8)}
    return {"record": rec, "summary": summary, "gates": gates}


def main(argv) -> None:
    step = argv[1]
    out = V2_ROOT / "evidence" / "train"
    if step == "cgc":
        rec = train_cgc()
        (V2_ROOT / "coefficients" / "cgc_reference.v2.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(rec["check"]), rec["weights_sha256"])
    elif step == "tune":
        res = tune()
        coeff = {"schema": "cgc-lao-solver-lab.baseline-coefficients.v2", "coefficients": res["coefficients"],
                 "selection": "SLB2 TRAIN seeds 0-15, 2048 episodes, min ALL terminal regret; ORIGINAL-contract "
                              "baselines under ORIGINAL, equal-information controls under DUP2_4 with every reading",
                 "all_terminal_table": res["table"]}
        (V2_ROOT / "coefficients" / "baselines.v2.json").write_text(json.dumps(coeff, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(res["coefficients"]), evaluate2.write(evaluate2.seal(res["record"], "V2-TRAIN-tune"), out / "tune.json"))
    elif step == "levels":
        res = levels()
        rec = evaluate2.seal(res["record"], "V2-TRAIN-levels")
        rec["summary"], rec["gates"] = res["summary"], res["gates"]
        print(json.dumps({"summary": res["summary"], "gates": res["gates"]}, indent=1), evaluate2.write(rec, out / "levels.json"))
    else:
        raise SystemExit("usage: calibrate2 cgc|tune|levels")


if __name__ == "__main__":
    main(sys.argv)
