"""Post-holdout DEV diagnostic (never used for selection): learners updated on NOOP's action log.

Every arm observes the readings of exactly the experts NOOP's PROBE_EARLY selected, so acquisition is
held equal and only belief updates and terminal decision rules differ. DEV router seeds 64-79 (fresh).
Run from the lab root:  PYTHONPATH=src python3 scripts/fixed_log.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))

from cgc_lao_lab import arms, env, evaluate, harness, stats  # noqa: E402

SEEDS = range(64, 80)
EPISODES = 2048
ARMS = ["NOOP", "SGD", "BETA", "CGC_REFERENCE", "STATIC_ATTENUATION", "CLEAN_BAYES", "RMF_U3", "RMF_P67",
        "KNOWN_Q_ORACLE"]
CONTRACTS = {"ORIGINAL": ARMS, "DUP2_4": ["RMF_U3E25", "KNOWN_Q_ORACLE"]}


def main() -> None:
    results: dict = {}
    for cname, names in CONTRACTS.items():
        contract = env.CONTRACTS[cname]
        for regime in ("CLEAN", "FLIP20", "FLIP40"):
            per = {n: [] for n in names}
            for seed in SEEDS:
                batch = env.episodes("DEV", seed, regime, 0, EPISODES)
                log = harness.rollout(arms.factory("NOOP")(None, contract)[0], batch, contract).actions
                for n in names:
                    agent, priv = arms.factory(n)(env.REGIMES[regime], contract)
                    out = harness.rollout(agent, batch, contract, actions=log,
                                          privileged=harness.privileged_inputs(priv, batch))
                    per[n].append(float(out.terminal.mean()))
            results.setdefault(cname, {})[regime] = per
    noop = {r: results["ORIGINAL"][r]["NOOP"] for r in ("CLEAN", "FLIP20", "FLIP40")}
    for cname, by_regime in results.items():
        for regime, per in by_regime.items():
            for n, vals in per.items():
                d = stats.paired(vals, noop[regime])
                print(f"{cname:8s} {regime:6s} {n:18s} {np.mean(vals):.5f} d_vs_NOOP {d['mean']:+.5f} "
                      f"[{d['lo']:+.5f},{d['hi']:+.5f}]")
    script = Path(__file__).read_bytes()
    rec = evaluate.seal({"partition": "DEV", "seeds": list(SEEDS), "episodes_per_seed": EPISODES,
                         "mode": "fixed NOOP action log", "results": results,
                         "script_sha256": hashlib.sha256(script).hexdigest()},
                        "DEV-fixed-log", "post-holdout diagnostic; acquisition held equal across arms")
    print("sha256", evaluate.write(rec, LAB / "evidence" / "dev" / "fixed_log.json"))


if __name__ == "__main__":
    main()
