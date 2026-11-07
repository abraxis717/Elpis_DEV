"""DEV diagnostics (never used for qualification): what limits the original contract, and how much
independent evidence closes the gap.

1. Regime identification: AUC of the end-of-episode posterior P(clean | history) for separating CLEAN
   from FLIP40 episodes (paired router seeds), per contract and acquisition rule.
2. Information curve: terminal regret of one learner as the number of duplicated readings (0..8) or the
   horizon (8..16) grows. Ad-hoc contracts here are diagnostics only; the declared contracts live in
   ``env.CONTRACTS``.
3. Fixed-log decomposition: learners updated on NOOP's action log, so acquisition is held equal and
   only belief updates and terminal decisions differ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import arms, env, evaluate, harness, stats
from .env import Contract

LAB = Path(__file__).resolve().parents[2]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative (ties count one half)."""
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def regime_identification(arm: str, contracts, seeds, episodes: int = 1024) -> dict:
    out = {}
    fac = arms.factory(arm)
    for cname, contract in contracts.items():
        aucs, pc_clean, pc_flip = [], [], []
        for seed in seeds:
            post = {}
            for regime in ("CLEAN", "FLIP40"):
                batch = env.episodes("DEV", seed, regime, 0, episodes)
                agent, _ = fac(None, contract)
                harness.rollout(agent, batch, contract)
                post[regime] = agent.filter.weights()[:, list(agent.filter.qs).index(0.0)]
            aucs.append(auc(post["CLEAN"], post["FLIP40"]))
            pc_clean.append(post["CLEAN"].mean())
            pc_flip.append(post["FLIP40"].mean())
        out[cname] = {"auc": stats.mean_ci(aucs), "p_clean_CLEAN": float(np.mean(pc_clean)),
                      "p_clean_FLIP40": float(np.mean(pc_flip))}
    return out


def information_curve(arm: str, seeds, episodes: int = 2048) -> dict:
    curve = {}
    contracts = {f"DUP2@{k}": Contract(f"DUP2@{k}", (2,) * k + (1,) * (8 - k)) for k in range(0, 9)}
    contracts.update({f"T{h}": Contract(f"T{h}", (1,) * h) for h in (8, 10, 12, 14, 16)})
    fac = {arm: arms.factory(arm), "NOOP": arms.factory("NOOP"), "CLEAN_BAYES": arms.factory("CLEAN_BAYES")}
    per = {c: {r: [] for r in ("CLEAN", "FLIP40")} for c in contracts}
    ref = {r: {"NOOP": [], "CLEAN_BAYES": []} for r in ("CLEAN", "FLIP40")}
    for seed in seeds:
        for regime in ("CLEAN", "FLIP40"):
            batch = env.episodes("DEV", seed, regime, 0, episodes)
            for name in ("NOOP", "CLEAN_BAYES"):
                ref[regime][name].append(harness.rollout(fac[name](None, env.CONTRACTS["ORIGINAL"])[0], batch,
                                                         env.CONTRACTS["ORIGINAL"]).terminal.mean())
            for cname, contract in contracts.items():
                per[cname][regime].append(harness.rollout(fac[arm](None, contract)[0], batch, contract).terminal.mean())
    for cname, contract in contracts.items():
        g1 = stats.paired(per[cname]["FLIP40"], ref["FLIP40"]["NOOP"])
        gain_ref = np.mean(np.array(ref["CLEAN"]["NOOP"]) - np.array(ref["CLEAN"]["CLEAN_BAYES"]))
        retention = float(np.mean(np.array(ref["CLEAN"]["NOOP"]) - np.array(per[cname]["CLEAN"])) / gain_ref)
        curve[cname] = {"extra_reads": int(sum(contract.reads) - contract.horizon),
                        "extra_queries": contract.horizon - 8, "flip40_vs_noop": g1,
                        "clean_retention": retention}
    return curve


def main(argv) -> None:
    out = LAB / "evidence" / "dev"
    seeds = range(32, 48)
    rec = {"phase": "DEV-diagnostics", "partition": "DEV", "seeds": list(seeds)}
    acq = {"PROBE": "RMF_U3", "INFOQ01": "RMF_U3_INFOQ01", "GREEDY": "RMF_U3_GREEDY"}
    rec["regime_identification"] = {
        name: regime_identification(arm, {"ORIGINAL": env.CONTRACTS["ORIGINAL"], "T16": env.CONTRACTS["T16"],
                                          "DUP2_4": env.CONTRACTS["DUP2_4"], "DUP2": env.CONTRACTS["DUP2"]}, seeds)
        for name, arm in acq.items()}
    rec["information_curve"] = {arm: information_curve(arm, seeds) for arm in ("RMF_U3", "RMF_P67")}
    sealed = evaluate.seal(rec, "DEV-diagnostics", "regime identification AUC and information curves")
    digest = evaluate.write(sealed, out / "diagnostics.json")
    print(json.dumps({"regime_identification": rec["regime_identification"]}, indent=1, default=float)[:4000])
    for arm, curve in rec["information_curve"].items():
        for cname, row in curve.items():
            g = row["flip40_vs_noop"]
            print(f"{arm:8s} {cname:8s} +reads={row['extra_reads']} +queries={row['extra_queries']} "
                  f"FLIP40 {g['mean']:+.5f} [{g['lo']:+.5f},{g['hi']:+.5f}] retention {row['clean_retention']:.3f}")
    print("sha256", digest)


if __name__ == "__main__":
    main(sys.argv)
