"""Gate evaluation (PREREGISTRATION.md) from evidence records, and the DEV/VALIDATION/HOLDOUT CLI.

A candidate's record and the reference record must cover the same partition, seeds and episodes; the
candidate may run under a changed contract while the references run under ORIGINAL (paired episode by
episode, because every contract reads the same potential outcomes).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import evaluate, stats
from .env import ALL_MIXTURE, PRIMARY_SHIFTS, SHIFT_REGIMES

LAB = Path(__file__).resolve().parents[2]
LEVELS = LAB / "evidence" / "train" / "levels.json"
REFERENCES = ("NOOP", "SGD", "BETA", "CGC_REFERENCE", "STATIC_ATTENUATION", "CLEAN_BAYES", "KNOWN_Q_ORACLE")


def gate_scales() -> dict:
    return json.loads(LEVELS.read_text(encoding="utf-8"))["gates"]


def _series(rec: dict, regime: str, arm: str, metric: str = "terminal") -> np.ndarray:
    if regime == "ALL":
        return np.mean([rec["results"][r][arm][metric] for r in ALL_MIXTURE], axis=0)
    return np.asarray(rec["results"][regime][arm][metric])


def _same_episodes(a: dict, b: dict) -> None:
    for key in ("partition", "seeds", "episodes_per_seed"):
        if a[key] != b[key]:
            raise ValueError(f"records are not paired: {key}")


def gates(cand_rec: dict, cand: str, ref_rec: dict, scales: dict | None = None) -> dict:
    """Evaluate G1-G5 (and G7 when shift regimes are present) for one candidate."""
    _same_episodes(cand_rec, ref_rec)
    scales = scales or gate_scales()
    c = lambda r, m="terminal": _series(cand_rec, r, cand, m)  # noqa: E731
    ref = lambda r, a: _series(ref_rec, r, a)  # noqa: E731
    out = {}
    g1 = stats.paired(c("FLIP40"), ref("FLIP40", "NOOP"))
    out["G1"] = {**g1, "pass": g1["hi"] < 0}
    g2 = stats.paired(c("CLEAN"), ref("CLEAN", "NOOP"))
    out["G2"] = {**g2, "pass": g2["hi"] < 0}
    g3 = stats.paired(c("CLEAN"), ref("CLEAN", scales["clean_reference"]))
    out["G3"] = {**g3, "pass": g3["hi"] <= scales["delta_clean"], "margin": scales["delta_clean"]}
    g4 = stats.paired(c("ALL"), ref("ALL", scales["all_reference"]))
    out["G4"] = {**g4, "pass": g4["hi"] <= scales["delta_all"], "margin": scales["delta_all"]}
    g5 = stats.mean_ci(c("ALL", "online"))
    out["G5"] = {**g5, "pass": g5["hi"] <= scales["online_budget"], "budget": scales["online_budget"]}
    shifts = {}
    for regime in SHIFT_REGIMES:
        if regime in cand_rec["results"] and regime in ref_rec["results"]:
            shifts[regime] = stats.paired(c(regime), ref(regime, "NOOP"), level=0.975)
            shifts[regime]["useful"] = shifts[regime]["hi"] < 0
    if shifts:
        out["shift"] = shifts
        out["G7"] = {"pass": any(shifts[r]["useful"] for r in PRIMARY_SHIFTS if r in shifts),
                     "primary": {r: shifts[r]["hi"] for r in PRIMARY_SHIFTS if r in shifts}}
    clean_gain_ref = float(np.mean(ref("CLEAN", "NOOP") - ref("CLEAN", scales["clean_reference"])))
    out["clean_retention"] = float(np.mean(ref("CLEAN", "NOOP") - c("CLEAN"))) / clean_gain_ref
    out["passed"] = [g for g in ("G1", "G2", "G3", "G4", "G5", "G7") if g in out and out[g]["pass"]]
    return out


CORE_GATES = ("G1", "G2", "G3", "G4", "G5")


def disposition(phases: dict, cand: str, state_scalars: int, tests_passed: bool) -> dict:
    """Preregistered disposition from {phase: (candidate record, reference record)} for VALIDATION and
    HOLDOUT: G1-G5 per phase, G6 (both phases), G7 (primary shifts, per phase), G8 (boundaries)."""
    per = {phase: gates(c, cand, r) for phase, (c, r) in phases.items()}
    g6 = all(per[p][g]["pass"] for p in per for g in CORE_GATES)
    g7 = all(per[p].get("G7", {}).get("pass", False) for p in per)
    g8 = bool(tests_passed and state_scalars <= 64)
    failing = sorted({f"{g}@{p}" for p in per for g in CORE_GATES if not per[p][g]["pass"]})
    failing += [f"G7@{p}" for p in per if not per[p].get("G7", {}).get("pass", False)]
    if not g8:
        failing.append("G8")
    return {"candidate": cand, "phases": per, "G6": g6, "G7": g7, "G8": g8, "failing": failing,
            "status": "QUALIFIED_SYNTHETIC" if (g6 and g7 and g8) else "NOT_QUALIFIED"}


def arm_summary(rec: dict, ref_rec: dict | None = None, level: float = 0.95) -> dict:
    """Per regime and arm: mean terminal and online regret, and terminal delta vs NOOP with CI."""
    ref_rec = ref_rec or rec
    out = {}
    regimes = [r for r in rec["results"]] + (["ALL"] if all(r in rec["results"] for r in ALL_MIXTURE) else [])
    for r in regimes:
        arms = rec["results"][ALL_MIXTURE[0] if r == "ALL" else r]
        for arm in arms:
            t = _series(rec, r, arm)
            d = stats.paired(t, _series(ref_rec, r, "NOOP"), level)
            out.setdefault(r, {})[arm] = {"terminal": float(t.mean()), "online": float(_series(rec, r, arm, "online").mean()),
                                          "delta_vs_noop": d}
    return out


def regime_table(rec: dict, arms, ref_rec: dict | None = None) -> str:
    """Mean terminal regret per regime and Δ vs NOOP (with 95% CI) for a quick screen."""
    ref_rec = ref_rec or rec
    regimes = [r for r in list(ALL_MIXTURE) + ["ALL"] + list(SHIFT_REGIMES)
               if r == "ALL" or r in rec["results"]]
    lines = []
    for arm in arms:
        cells = []
        for r in regimes:
            if r != "ALL" and arm not in rec["results"].get(r, {}):
                continue
            d = stats.paired(_series(rec, r, arm), _series(ref_rec, r, "NOOP"))
            cells.append(f"{r}:{d['mean']:+.5f}[{d['lo']:+.5f},{d['hi']:+.5f}]")
        online = float(_series(rec, "ALL", arm, "online").mean())
        lines.append(f"{arm:22s} online={online:.4f} " + " ".join(cells))
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="DEV / VALIDATION / HOLDOUT runs for the solver lab")
    ap.add_argument("phase")
    ap.add_argument("--partition", required=True)
    ap.add_argument("--seeds", required=True, help="first:stop")
    ap.add_argument("--episodes", type=int, default=2048)
    ap.add_argument("--contract", default="ORIGINAL")
    ap.add_argument("--regimes", default=",".join(ALL_MIXTURE))
    ap.add_argument("--arms", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    first, stop = (int(v) for v in args.seeds.split(":"))
    names = [a for a in args.arms.split(",") if a]
    rec = evaluate.run(args.partition, range(first, stop), args.regimes.split(","), evaluate.specs(names),
                       contract=args.contract, episodes=args.episodes, workers=args.workers)
    digest = evaluate.write(evaluate.seal(rec, args.phase, args.note), Path(args.out))
    print(f"{args.out} sha256={digest} seconds={rec['seconds']}")
    print(regime_table(rec, names))


if __name__ == "__main__":
    main()
