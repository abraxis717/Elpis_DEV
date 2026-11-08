"""v4 qualification layer: fresh root and partitions, candidates, runs, ROBUST-OCCAM selection, parity, freeze.

Selection (PREREGISTRATION_V4.md): a candidate is eligible if every gate holds with its interval computed at
the 99.9% level on the DEV round. The selected arm is the smallest eligible mechanism: fewest logical state
scalars, then fewest likelihood evaluations per update, then the lowest DEV ALL regret, then name.
Qualification gates on VALIDATION and HOLDOUT are the unchanged 95% gates of ``cgc_lao_v2.report2``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from cgc_lao_lab import evaluate as ev1
from cgc_lao_lab import stats
from cgc_lao_lab.env import logits
from cgc_lao_lab.harness import noop_decisions
from cgc_lao_v2 import agents2, arms2, candidates2 as c2, env2, evaluate2, harness2, native, report2
from cgc_lao_v2.scalar2 import ScalarRMF2
from cgc_lao_v3 import env3, study3

from . import V4_ROOT

ROOT = "elpis/cgc_lao/solver_lab/v4/"
PARTITIONS = {"DEV": 256, "VALIDATION": 32, "HOLDOUT": 64, "OOD": 64}
REGIME_USE = env3.REGIME_USE
PRIMARY4 = {name: study3.PRIMARY3[name] for name in ("RMF3", "RMF4_W", "RMF4_E25", "RMF_NOEVENT", "RMF5_E25")}
FALLBACK4 = {name: study3.FALLBACK3[name] for name in ("SAFE3_P4", "SAFE4_P8")}
CONTROLS4 = study3.CONTROLS3
CONFIGS4 = {**PRIMARY4, **FALLBACK4}
SELECTION_LEVEL = 0.999
PRIMARY_GATES, FALLBACK_GATES = study3.PRIMARY_GATES, study3.FALLBACK_GATES
SRC4 = V4_ROOT / "src" / "cgc_lao_v4"


def episodes(partition, seed, regime, first, count):
    if isinstance(regime, str):
        regime = env2.REGIMES[regime]
    if partition not in PARTITIONS or not (isinstance(seed, int) and 0 <= seed < PARTITIONS[partition]):
        raise ValueError(f"seed {seed!r} outside v4 partition {partition}")
    if regime.name not in REGIME_USE[partition]:
        raise ValueError(f"regime {regime.name} is not admitted on v4 partition {partition}")
    if first < 0 or count < 1:
        raise ValueError("EPISODE_RANGE")
    return env3.generate(ROOT, partition, seed, regime, first, count)


def factory(name: str, **ov):
    if name in CONFIGS4:
        cfg = CONFIGS4[name]
        return arms2._admissible(lambda: arms2.bayes(name, cfg))
    return arms2.factory(name, **ov)


def complexity(name: str) -> tuple:
    agent, _ = factory(name)(None, env2.CONTRACTS["DUP2_4"])
    f = getattr(agent, "filter", None)
    evals = int(np.sum(np.asarray(f.wn) > 0) * 16) if f is not None else (16 if name == "PLUGIN_Q_BAYES" else 1)
    return agent.state_scalars(), evals


def evaluate_seed(partition, seed, regimes, specs, n_episodes, chunk=2048) -> dict:
    out = {}
    for rname in regimes:
        sums = {}
        for first in range(0, n_episodes, chunk):
            batch = episodes(partition, seed, rname, first, min(chunk, n_episodes - first))
            nd = noop_decisions(batch)
            for label, spec in specs.items():
                for key, value in harness2.evaluate_batch2(spec, batch, nd).items():
                    acc = sums.setdefault(label, {})
                    acc[key] = acc.get(key, 0.0) + float(value)
        out[rname] = {label: {k: v / n_episodes for k, v in acc.items()} for label, acc in sums.items() if acc}
    return out


def _worker(task):
    partition, seed, regimes, specs, n_episodes = task
    built = {label: {**s, "factory": factory(s["arm"], **s["overrides"])} for label, s in specs.items()}
    return seed, evaluate_seed(partition, seed, regimes, built, n_episodes)


def run(partition, seeds, regimes, specs, n_episodes, workers=4) -> dict:
    seeds = list(seeds)
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("fork")) as ex:
        by_seed = dict(ex.map(_worker, [(partition, s, list(regimes), specs, n_episodes) for s in seeds]))
    results = {}
    for r in regimes:
        for label in specs:
            metrics = by_seed[seeds[0]][r].get(label)
            if metrics:
                results.setdefault(r, {})[label] = {m: [float(f"{by_seed[s][r][label][m]:.10g}") for s in seeds]
                                                    for m in metrics}
    return {"partition": partition, "seeds": seeds, "episodes_per_seed": n_episodes, "regimes": list(regimes),
            "arms": {k: {kk: v[kk] for kk in ("arm", "contract", "mode", "overrides")} for k, v in specs.items()},
            "results": results, "seconds": round(time.time() - t0, 1)}


def source_file_digests() -> dict:
    out = {f"v4/{p.name}": ev1.file_digest(p) for p in sorted(SRC4.glob("*.py"))}
    out.update(study3.source_file_digests())
    return out


def source_digest() -> str:
    h = hashlib.sha256()
    for name, digest in sorted(source_file_digests().items()):
        h.update(name.encode() + b"\0" + digest.encode() + b"\0")
    return h.hexdigest()


def seal(record: dict, phase: str, note: str = "") -> dict:
    return {"schema": "cgc-lao-solver-lab.record.v4", "phase": phase, "note": note, "rng_root": ROOT,
            "source_sha256": source_digest(), "source_files_sha256": source_file_digests(),
            "native_files_sha256": evaluate2.native_file_digests(),
            "spec_sha256": {"slb1": ev1.file_digest(ev1.SPEC), "slb2": ev1.file_digest(evaluate2.SPEC2)},
            "coefficients_sha256": evaluate2.coefficient_digests(), "environment": ev1.environment(), **record}


def round_specs() -> dict:
    s = evaluate2.spec
    specs = {a: s(a, "ORIGINAL", "FIRST_ONLY") for a in c2.ORIGINAL_BASELINES}
    specs["KNOWN_Q_ORACLE"] = s("KNOWN_Q_ORACLE", "ORIGINAL", "FIRST_ONLY")
    for a in list(PRIMARY4) + list(CONTROLS4):
        specs[a] = s(a, "DUP2_4", "DUPLICATE")
    specs["KNOWN_Q_DUP"] = s("KNOWN_Q_ORACLE", "DUP2_4", "DUPLICATE")
    for a in FALLBACK4:
        specs[a] = s(a, "DUP2_4", "FIRST_ONLY")
    return specs


# -- robust Occam selection -----------------------------------------------------------------------------------
def _d(rec, regime, a, b, level):
    return stats.paired(report2.series(rec, regime, a), report2.series(rec, regime, b), level)


def gates_at(rec, cand, level=SELECTION_LEVEL) -> dict:
    """The report2 primary gates with every interval at ``level`` (selection only)."""
    sc = report2.scales()
    g = {"G1": _d(rec, "FLIP40", cand, "NOOP", level)["hi"] < 0,
         "G2": _d(rec, "CLEAN", cand, "NOOP", level)["hi"] < 0,
         "G3": _d(rec, "CLEAN", cand, sc["clean_reference"], level)["hi"] <= sc["delta_clean"],
         "G4": _d(rec, "ALL", cand, sc["all_reference"], level)["hi"] <= sc["delta_all"],
         "G5": stats.mean_ci(report2.series(rec, "ALL", cand, "online"), level)["hi"] <= sc["online_budget"],
         "G7": any(_d(rec, r, cand, "NOOP", level)["hi"] < 0 for r in ("QUNIF", "BURST")),
         "G9": _d(rec, "FLIP45", cand, "RMF_V1", level)["hi"] < 0
         and _d(rec, "FLIP45", cand, "NOOP", level)["hi"] <= report2.TAU_SAFE,
         "G10": _d(rec, "CLEAN", cand, "RMF_V1", level)["hi"] <= report2.V1_CLEAN_MARGIN}
    return g


def fallback_gates_at(rec, cand, level=SELECTION_LEVEL) -> dict:
    sc = report2.scales()
    return {"F1": _d(rec, "FLIP40", cand, "NOOP", level)["hi"] < 0,
            "F2": _d(rec, "CLEAN", cand, "NOOP", level)["hi"] < 0,
            "F3": all(_d(rec, r, cand, "NOOP", level)["hi"] <= report2.TAU_SAFE for r in ("FLIP45", "FLIP48")),
            "F4": _d(rec, "EVENT40", cand, "NOOP", level)["hi"] < 0,
            "F5": stats.mean_ci(report2.series(rec, "ALL", cand, "online"), level)["hi"] <= sc["online_budget"],
            "F6": any(_d(rec, r, cand, "NOOP", level)["hi"] < 0 for r in ("QUNIF", "BURST"))}


def select(rec) -> dict:
    rows = {}
    for c in list(PRIMARY4) + list(CONTROLS4):
        g = gates_at(rec, c)
        rows[c] = {"gates_999": g, "eligible": all(g.values()), "ALL": float(report2.series(rec, "ALL", c).mean()),
                   "complexity": list(complexity(c))}
    eligible = [c for c in rows if rows[c]["eligible"]]
    chosen = min(eligible, key=lambda c: (tuple(rows[c]["complexity"]), rows[c]["ALL"], c)) if eligible else None
    frows = {}
    for c in FALLBACK4:
        g = fallback_gates_at(rec, c)
        frows[c] = {"gates_999": g, "eligible": all(g.values()), "CLEAN": float(report2.series(rec, "CLEAN", c).mean()),
                    "complexity": list(complexity(c))}
    feligible = [c for c in frows if frows[c]["eligible"]]
    fchosen = min(feligible, key=lambda c: (tuple(frows[c]["complexity"]), frows[c]["CLEAN"], c)) if feligible else None
    return {"primary": chosen, "fallback": fchosen, "primary_rows": rows, "fallback_rows": frows,
            "rule": "eligible: every gate at 99.9% on DEV; choose fewest state scalars, then fewest likelihood "
                    "evaluations, then lowest DEV ALL (fallback: CLEAN) regret, then name"}


# -- native parity, freeze, CLI -----------------------------------------------------------------------------------
def parity(partition, roles, seeds=range(0, 8), n_episodes=256, bitwise=32) -> dict:
    contract = env2.CONTRACTS["DUP2_4"]
    out = {}
    for role, (name, mode) in roles.items():
        cfg = CONFIGS4[name]
        r = {"name": name, "mode": mode, "episodes": 0, "trajectory_divergences": 0, "decision_mismatches": 0,
             "near_tie_mismatches": 0, "max_state_abs_diff": 0.0, "bitwise_episodes": 0, "bitwise_failures": 0}
        for seed in seeds:
            for regime in env2.DESIGN + env2.ROBUSTNESS:
                batch = episodes(partition, seed, regime, 0, n_episodes)
                agent = agents2.BayesAgent2(cfg["hypotheses"], cfg["prior"])
                ref = harness2.rollout2(agent, batch, contract, mode)
                for e in range(batch.count):
                    n = native.NativeRMF(cfg["hypotheses"], cfg["prior"])
                    s = ScalarRMF2(cfg["hypotheses"], cfg["prior"]) if e < bitwise else None
                    z = logits(batch.router_w, batch.x[e])
                    diverged, ok = False, True
                    for t in range(8):
                        a = n.select(list(z[t]))
                        ok &= s is None or s.select(list(z[t])) == a
                        diverged |= a != ref.actions[e, t]
                        y = [int(v) for v in batch.y[e, t, a, :contract.reads[t]]]
                        vals = y if mode == "DUPLICATE" else y[:1]
                        n.update(a, float(z[t, a]), vals)
                        if s is not None:
                            s.update(a, float(z[t, a]), vals)
                            ok &= s.mu == n.mu and s.v == n.v and s.lw == n.lw
                    r["episodes"] += 1
                    zq = logits(batch.router_w, batch.xq[e])
                    if s is not None:
                        ok &= all(s.decide(list(zq[j])) == n.decide(list(zq[j])) for j in range(16))
                        r["bitwise_episodes"] += 1
                        r["bitwise_failures"] += int(not ok)
                    if diverged:
                        r["trajectory_divergences"] += 1
                        continue
                    r["max_state_abs_diff"] = max(r["max_state_abs_diff"], float(max(
                        np.max(np.abs(np.array(n.mu) - agent.filter.mu[e])), np.max(np.abs(np.array(n.v) - agent.filter.v[e])),
                        np.max(np.abs(np.array(n.lw) - agent.filter.lw[e])))))
                    for j in range(16):
                        if n.decide(list(zq[j])) != ref.decisions[e, j]:
                            p = n.predictive(list(zq[j]))
                            gaps = [abs(p[i] - p[k]) for i in range(3) for k in range(i + 1, 3)] + [abs(max(p) - 0.6)]
                            r["near_tie_mismatches" if min(gaps) < 1e-9 else "decision_mismatches"] += 1
        r["pass"] = r["bitwise_failures"] == 0 and r["decision_mismatches"] == 0 and r["max_state_abs_diff"] <= 1e-9
        out[role] = r
    return out


PLANS = study3.PLANS


def qualification_specs(primary, fallback) -> dict:
    s = evaluate2.spec
    specs = {a: s(a, "ORIGINAL", "FIRST_ONLY") for a in c2.ORIGINAL_BASELINES}
    specs["KNOWN_Q_ORACLE"] = s("KNOWN_Q_ORACLE", "ORIGINAL", "FIRST_ONLY")
    specs[primary] = s(primary, "DUP2_4", "DUPLICATE")
    specs[fallback] = s(fallback, "DUP2_4", "FIRST_ONLY")
    for c in CONTROLS4:
        if c != primary:
            specs[c] = s(c, "DUP2_4", "DUPLICATE")
    specs["KNOWN_Q_DUP"] = s("KNOWN_Q_ORACLE", "DUP2_4", "DUPLICATE")
    return specs


def freeze() -> dict:
    sel = json.loads((V4_ROOT / "SELECTION_V4.json").read_text(encoding="utf-8"))
    primary, fallback = sel["primary"], sel["fallback"]
    return {"schema": "cgc-lao-solver-lab.freeze.v4",
            "note": "Frozen before any v4 VALIDATION, HOLDOUT or OOD run; selection by SELECTION_V4.json.",
            "candidates": {"PRIMARY": {"name": primary, "config": CONFIGS4[primary], "contract": "DUP2_4", "mode": "DUPLICATE"},
                           "FALLBACK": {"name": fallback, "config": CONFIGS4[fallback], "contract": "DUP2_4",
                                        "mode": "FIRST_ONLY"}},
            "rng_root": ROOT, "source_sha256": source_digest(), "source_files_sha256": source_file_digests(),
            "native_files_sha256": evaluate2.native_file_digests(),
            "spec_sha256": {"slb1": ev1.file_digest(ev1.SPEC), "slb2": ev1.file_digest(evaluate2.SPEC2)},
            "coefficients_sha256": evaluate2.coefficient_digests(), "plans": PLANS,
            "specs": {"VALIDATION": qualification_specs(primary, fallback), "HOLDOUT": qualification_specs(primary, fallback),
                      "OOD": study3.ood_specs(primary, fallback)},
            "holdout_rule": "open HOLDOUT only if the PRIMARY passes G1-G5, G7, G9, G10 on VALIDATION"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("dev", "select", "freeze", "qualify", "parity"))
    ap.add_argument("--phase", default="")
    a = ap.parse_args(argv)
    ev = V4_ROOT / "evidence"
    if a.command == "dev":
        rec = run("DEV", range(0, 32), list(env2.DESIGN + env2.ROBUSTNESS), round_specs(), 2048)
        print(evaluate2.write(seal(rec, "V4-DEV-selection", "the single preregistered v4 DEV round"),
                              ev / "dev" / "selection_round.json"))
    elif a.command == "select":
        path = ev / "dev" / "selection_round.json"
        out = {"schema": "cgc-lao-solver-lab.selection.v4", **select(json.loads(path.read_text(encoding="utf-8"))),
               "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (V4_ROOT / "SELECTION_V4.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(out["primary"], out["fallback"])
    elif a.command == "freeze":
        (V4_ROOT / "FROZEN_V4.json").write_text(json.dumps(freeze(), indent=1) + "\n", encoding="utf-8")
        print(source_digest())
    elif a.command == "qualify":
        frozen = json.loads((V4_ROOT / "FROZEN_V4.json").read_text(encoding="utf-8"))
        plan, specs = frozen["plans"][a.phase], frozen["specs"][a.phase]
        rec = run(plan["partition"], range(*plan["seeds"]), plan["regimes"], specs, plan["episodes"])
        print(evaluate2.write(seal(rec, f"V4-{a.phase}"), ev / a.phase.lower() / "qualification.json"), rec["seconds"])
    elif a.command == "parity":
        frozen = json.loads((V4_ROOT / "FROZEN_V4.json").read_text(encoding="utf-8"))
        roles = {r: (c["name"], c["mode"]) for r, c in frozen["candidates"].items()}
        t0 = time.time()
        rec = {"partition": a.phase, "seeds": list(range(8)), "episodes_per_seed_and_regime": 256,
               "bitwise_episodes_per_seed_and_regime": 32, "tolerances": {"state_abs": 1e-9, "near_tie": 1e-9},
               "arms": parity(a.phase, roles)}
        rec["seconds"] = round(time.time() - t0, 1)
        print({r: (x["pass"], x["episodes"], x["bitwise_failures"], x["decision_mismatches"]) for r, x in rec["arms"].items()})
        print(evaluate2.write(seal(rec, f"V4-{a.phase}-native-parity"), ev / a.phase.lower() / "native_parity.json"))


if __name__ == "__main__":
    main(sys.argv[1:])
