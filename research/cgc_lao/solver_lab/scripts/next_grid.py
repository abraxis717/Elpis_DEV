"""Post-holdout DEV diagnostic for the next experiment (NOT a qualification; SLB1 v1's holdout is spent).

Question: does a hypothesis grid that reaches a near-pure-noise channel (q = 0.48) close the FLIP45
boundary of the frozen DUP2_4 learner, does more event-level prior mass close EVENT40, and what does
either cost in the design regimes? DEV router seeds 80-95 (fresh), 2048 episodes; candidates under DUP2_4,
NOOP under ORIGINAL on the same episodes.  Run from the lab root:  python3 scripts/next_grid.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))

from cgc_lao_lab import agents as ag  # noqa: E402
from cgc_lao_lab import env, evaluate, harness, stats  # noqa: E402

SEEDS = range(80, 96)
EPISODES = 2048
REGIMES = ("CLEAN", "FLIP20", "FLIP40", "FLIP45", "EVENT40")
EVENT = ["channel", "channel", "channel", "event"]
CONFIGS = {
    "RMF_U3E25 (frozen)": ([0.0, 0.2, 0.4, 0.4], [1, 1, 1, 0.25], EVENT),
    "GRID48_E25": ([0.0, 0.2, 0.4, 0.48, 0.4], [1, 1, 1, 1, 0.25], EVENT[:3] + ["channel", "event"]),
    "GRID48_E1": ([0.0, 0.2, 0.4, 0.48, 0.4], [1, 1, 1, 1, 1], EVENT[:3] + ["channel", "event"]),
    "GRID48_E3": ([0.0, 0.2, 0.4, 0.48, 0.4], [1, 1, 1, 1, 3], EVENT[:3] + ["channel", "event"]),
}


def main() -> None:
    dup = env.CONTRACTS["DUP2_4"]
    orig = env.CONTRACTS["ORIGINAL"]
    out = {r: {name: [] for name in list(CONFIGS) + ["NOOP", "CLEAN_BAYES"]} for r in REGIMES}
    for seed in SEEDS:
        for regime in REGIMES:
            batch = env.episodes("DEV", seed, regime, 0, EPISODES)
            out[regime]["NOOP"].append(float(harness.rollout(ag.NoopAgent(), batch, orig).terminal.mean()))
            out[regime]["CLEAN_BAYES"].append(float(harness.rollout(ag.clean_bayes(), batch, orig).terminal.mean()))
            for name, (qs, prior, kinds) in CONFIGS.items():
                agent = ag.BayesAgent(qs, prior, kinds=kinds, name=name)
                out[regime][name].append(float(harness.rollout(agent, batch, dup).terminal.mean()))
    gain = np.mean(np.array(out["CLEAN"]["NOOP"]) - np.array(out["CLEAN"]["CLEAN_BAYES"]))
    for name in CONFIGS:
        retention = np.mean(np.array(out["CLEAN"]["NOOP"]) - np.array(out["CLEAN"][name])) / gain
        cells = []
        for regime in REGIMES[1:]:
            d = stats.paired(out[regime][name], out[regime]["NOOP"])
            cells.append(f"{regime} {d['mean']:+.5f} [{d['lo']:+.5f},{d['hi']:+.5f}]")
        print(f"{name:20s} retention {retention:.3f} | " + " | ".join(cells))
    rec = evaluate.seal({"partition": "DEV", "seeds": list(SEEDS), "episodes_per_seed": EPISODES,
                         "contract": "DUP2_4 (candidates), ORIGINAL (NOOP, CLEAN_BAYES)",
                         "configs": {k: {"qs": v[0], "prior": v[1], "kinds": v[2]} for k, v in CONFIGS.items()},
                         "results": out, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                        "DEV-next-grid", "post-holdout diagnostic for the next experiment; not a qualification")
    print("sha256", evaluate.write(rec, LAB / "evidence" / "dev" / "next_grid.json"))


if __name__ == "__main__":
    main()
