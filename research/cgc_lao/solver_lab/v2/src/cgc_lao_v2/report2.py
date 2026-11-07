"""v2 gates, family claims, preregistered selection and the run CLI (PREREGISTRATION_V2.md).

All comparisons are paired by router seed within one record (every arm of a record ran on the same episodes).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cgc_lao_lab import stats

from . import V2_ROOT, arms2, candidates2, env2, evaluate2

LEVELS = V2_ROOT / "evidence" / "train" / "levels.json"
FAMILIES = ("CLEAN", "FLIP20", "FLIP40", "FLIP45", "FLIP48", "QUNIF", "BURST", "EVENT40")
FAMILY_LEVEL = 1.0 - 0.05 / len(FAMILIES)       # Bonferroni: 99.375% two-sided
TAU_SAFE = 0.0005
V1_CLEAN_MARGIN = 0.0005
EPSILON_OCCAM = 0.0002
PRIMARY_GATES = ("G1", "G2", "G3", "G4", "G5", "G9", "G10")
FALLBACK_GATES = ("F1", "F2", "F3", "F4", "F5")


def scales() -> dict:
    return json.loads(LEVELS.read_text(encoding="utf-8"))["gates"]


def series(rec, regime, arm, metric="terminal"):
    if regime == "ALL":
        return np.mean([rec["results"][r][arm][metric] for r in env2.ALL_MIXTURE], axis=0)
    return np.asarray(rec["results"][regime][arm][metric])


def has(rec, regime, arm):
    return regime in rec["results"] and arm in rec["results"][regime]


def _d(rec, regime, a, b, level=0.95):
    return stats.paired(series(rec, regime, a), series(rec, regime, b), level)


def primary_gates(rec, cand, sc=None) -> dict:
    sc = sc or scales()
    g = {}
    g["G1"] = {**_d(rec, "FLIP40", cand, "NOOP")}
    g["G1"]["pass"] = g["G1"]["hi"] < 0
    g["G2"] = {**_d(rec, "CLEAN", cand, "NOOP")}
    g["G2"]["pass"] = g["G2"]["hi"] < 0
    g["G3"] = {**_d(rec, "CLEAN", cand, sc["clean_reference"]), "margin": sc["delta_clean"]}
    g["G3"]["pass"] = g["G3"]["hi"] <= sc["delta_clean"]
    g["G4"] = {**_d(rec, "ALL", cand, sc["all_reference"]), "margin": sc["delta_all"]}
    g["G4"]["pass"] = g["G4"]["hi"] <= sc["delta_all"]
    g["G5"] = {**stats.mean_ci(series(rec, "ALL", cand, "online")), "budget": sc["online_budget"]}
    g["G5"]["pass"] = g["G5"]["hi"] <= sc["online_budget"]
    if has(rec, "FLIP45", cand):
        v1 = _d(rec, "FLIP45", cand, "RMF_V1")
        noop = _d(rec, "FLIP45", cand, "NOOP")
        g["G9"] = {"vs_RMF_V1": v1, "vs_NOOP": noop, "pass": v1["hi"] < 0 and noop["hi"] <= TAU_SAFE}
    g["G10"] = {**_d(rec, "CLEAN", cand, "RMF_V1"), "margin": V1_CLEAN_MARGIN}
    g["G10"]["pass"] = g["G10"]["hi"] <= V1_CLEAN_MARGIN
    shifts = {r: stats.paired(series(rec, r, cand), series(rec, r, "NOOP"), 0.975)
              for r in ("QUNIF", "BURST") if has(rec, r, cand)}
    if shifts:
        g["G7"] = {"shifts": shifts, "pass": any(s["hi"] < 0 for s in shifts.values())}
    g["families"] = family_claims(rec, cand, g)
    gain = float(np.mean(series(rec, "CLEAN", "NOOP") - series(rec, "CLEAN", sc["clean_reference"])))
    g["clean_retention"] = float(np.mean(series(rec, "CLEAN", "NOOP") - series(rec, "CLEAN", cand))) / gain
    return g


def family_claims(rec, cand, g=None) -> dict:
    out = {}
    for fam in FAMILIES:
        if not has(rec, fam, cand):
            continue
        d = stats.paired(series(rec, fam, cand), series(rec, fam, "NOOP"), FAMILY_LEVEL)
        claim = "USEFUL" if d["hi"] < 0 else ("SAFE" if d["hi"] <= TAU_SAFE else "NOT_QUALIFIED")
        if fam == "CLEAN" and g is not None and not g["G3"]["pass"]:
            claim = "NOT_QUALIFIED"
        out[fam] = {**d, "claim": claim}
    return out


def fallback_gates(rec, cand, sc=None) -> dict:
    sc = sc or scales()
    g = {}
    g["F1"] = {**_d(rec, "FLIP40", cand, "NOOP")}
    g["F1"]["pass"] = g["F1"]["hi"] < 0
    g["F2"] = {**_d(rec, "CLEAN", cand, "NOOP")}
    g["F2"]["pass"] = g["F2"]["hi"] < 0
    f3 = {r: _d(rec, r, cand, "NOOP") for r in ("FLIP45", "FLIP48") if has(rec, r, cand)}
    g["F3"] = {"regimes": f3, "pass": bool(f3) and all(v["hi"] <= TAU_SAFE for v in f3.values())}
    if has(rec, "EVENT40", cand):
        g["F4"] = {**_d(rec, "EVENT40", cand, "NOOP")}
        g["F4"]["pass"] = g["F4"]["hi"] < 0
    g["F5"] = {**stats.mean_ci(series(rec, "ALL", cand, "online")), "budget": sc["online_budget"]}
    g["F5"]["pass"] = g["F5"]["hi"] <= sc["online_budget"]
    shifts = {r: stats.paired(series(rec, r, cand), series(rec, r, "NOOP"), 0.975)
              for r in ("QUNIF", "BURST") if has(rec, r, cand)}
    if shifts:
        g["F6"] = {"shifts": shifts, "pass": any(s["hi"] < 0 for s in shifts.values())}
    gain = float(np.mean(series(rec, "CLEAN", "NOOP") - series(rec, "CLEAN", sc["clean_reference"])))
    g["clean_retention"] = float(np.mean(series(rec, "CLEAN", "NOOP") - series(rec, "CLEAN", cand))) / gain
    return g


def _point_primary(g) -> dict:
    sc = scales()
    return {"G1": g["G1"]["mean"] < 0, "G2": g["G2"]["mean"] < 0, "G3": g["G3"]["mean"] <= sc["delta_clean"],
            "G4": g["G4"]["mean"] <= sc["delta_all"], "G5": g["G5"]["mean"] <= sc["online_budget"],
            "G9": "G9" in g and g["G9"]["vs_RMF_V1"]["mean"] < 0 and g["G9"]["vs_NOOP"]["mean"] <= TAU_SAFE,
            "G10": g["G10"]["mean"] <= V1_CLEAN_MARGIN}


def complexity(name: str) -> tuple:
    """(logical state scalars, likelihood evaluations per update) of an arm, for the Occam tie-break."""
    agent, _ = arms2.factory(name)(None, env2.CONTRACTS["DUP2_4"])
    f = getattr(agent, "filter", None)
    evals = int(np.sum(np.asarray(f.wn) > 0) * 16) if f is not None else (16 if name == "PLUGIN_Q_BAYES" else 1)
    return agent.state_scalars(), evals


def select_primary(rec) -> dict:
    pool = list(candidates2.PRIMARY) + [c for c in candidates2.CONTROLS if c != "NOOP"]
    rows = {}
    for c in pool:
        g = primary_gates(rec, c)
        point = _point_primary(g)
        rows[c] = {"point": point, "passed": sum(point.values()), "eligible": all(point.values()),
                   "ALL": float(series(rec, "ALL", c).mean()), "complexity": complexity(c)}
    eligible = [c for c in pool if rows[c]["eligible"]]
    if eligible:
        best = min(rows[c]["ALL"] for c in eligible)
        near = [c for c in eligible if rows[c]["ALL"] <= best + EPSILON_OCCAM]
        chosen = min(near, key=lambda c: (rows[c]["complexity"], c))
        status = "ELIGIBLE"
    else:
        chosen = min(pool, key=lambda c: (-rows[c]["passed"], rows[c]["ALL"]))
        status = "NONE_ELIGIBLE"
    return {"chosen": chosen, "status": status, "rows": rows}


def select_fallback(rec) -> dict:
    rows = {}
    for c in candidates2.FALLBACK:
        g = fallback_gates(rec, c)
        point = {k: g[k]["mean"] < 0 if k in ("F1", "F2", "F4") else None for k in ("F1", "F2", "F4")}
        point["F3"] = all(v["mean"] <= TAU_SAFE for v in g["F3"]["regimes"].values())
        point["F5"] = g["F5"]["mean"] <= scales()["online_budget"]
        rows[c] = {"point": point, "eligible": all(point.values()), "CLEAN": float(series(rec, "CLEAN", c).mean())}
    eligible = [c for c in rows if rows[c]["eligible"]]
    chosen = min(eligible, key=lambda c: rows[c]["CLEAN"]) if eligible else None
    return {"chosen": chosen, "rows": rows}


def round_specs() -> dict:
    s = evaluate2.spec
    specs = {a: s(a, "ORIGINAL", "FIRST_ONLY") for a in candidates2.ORIGINAL_BASELINES}
    specs["KNOWN_Q_ORACLE"] = s("KNOWN_Q_ORACLE", "ORIGINAL", "FIRST_ONLY")
    for a in list(candidates2.PRIMARY) + [c for c in candidates2.CONTROLS if c != "NOOP"]:
        specs[a] = s(a, "DUP2_4", "DUPLICATE")
    specs["KNOWN_Q_DUP"] = s("KNOWN_Q_ORACLE", "DUP2_4", "DUPLICATE")
    for a in candidates2.FALLBACK:
        specs[a] = s(a, "DUP2_4", "FIRST_ONLY")
    return specs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--partition", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--episodes", type=int, default=2048)
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--arms", default="ROUND", help="ROUND (all DEV arms) or a JSON object label -> spec")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    first, stop = (int(v) for v in a.seeds.split(":"))
    specs = round_specs() if a.arms == "ROUND" else json.loads(a.arms)
    rec = evaluate2.run(a.partition, range(first, stop), a.regimes.split(","), specs, a.episodes, a.workers)
    print(Path(a.out), evaluate2.write(evaluate2.seal(rec, a.phase, a.note), Path(a.out)), rec["seconds"], "s")


if __name__ == "__main__":
    main()
