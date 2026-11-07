"""v3 qualification layer: candidates, registry, runs on v3 partitions, CI-bound selection, native parity, freeze.

Gate arithmetic, scales, agents, the native component and the host are the frozen v2 ones (imported). The only
behavioural change from v2 is the selection rule: a candidate is eligible only if the DEV 95% bounds (the gates
themselves), not the point estimates, satisfy G1-G5, G7, G9 and G10 (PREREGISTRATION_V3.md).
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
from cgc_lao_lab.env import logits
from cgc_lao_lab.harness import noop_decisions
from cgc_lao_v2 import agents2, arms2, candidates2 as c2, env2, evaluate2, harness2, native, report2
from cgc_lao_v2.scalar2 import ScalarRMF2

from . import V3_ROOT, env3

H3 = [(0.0, 0.0), (0.4, 0.0), (0.48, 0.0)]
PRIMARY3 = {
    "RMF3": {"hypotheses": H3, "prior": [1, 1, 1]},
    "RMF4_E25": c2.PRIMARY["RMF4_E25"],
    "RMF4_W": {"hypotheses": c2.H_NOEVENT, "prior": [1, 0.5, 1, 2]},
    "RMF_NOEVENT": c2.BAYES_CONTROLS["RMF_NOEVENT"],
    "RMF5_E25": c2.PRIMARY["RMF5_E25"],
}
FALLBACK3 = {
    "SAFE4_P8": c2.FALLBACK["SAFE4_P8"],
    "SAFE4_P4": c2.FALLBACK["SAFE4_P4"],
    "SAFE3_P4": {"hypotheses": H3, "prior": [1, 4, 4]},
}
CONTROLS3 = ("RMF_V1", "BETA_ALL", "SGD_ALL", "DISAGREE_SGD", "PLUGIN_Q_BAYES", "STATIC_ATT_ALL")
CONFIGS3 = {**PRIMARY3, **FALLBACK3}
EPSILON_OCCAM = 0.0002
SRC3 = V3_ROOT / "src" / "cgc_lao_v3"
PRIMARY_GATES = ("G1", "G2", "G3", "G4", "G5", "G7", "G9", "G10")
FALLBACK_GATES = ("F1", "F2", "F3", "F4", "F5", "F6")


def factory(name: str, **ov):
    if name in CONFIGS3:
        cfg = CONFIGS3[name]
        return arms2._admissible(lambda: arms2.bayes(name, cfg))
    return arms2.factory(name, **ov)


def complexity(name: str) -> tuple:
    agent, _ = factory(name)(None, env2.CONTRACTS["DUP2_4"])
    f = getattr(agent, "filter", None)
    evals = int(np.sum(np.asarray(f.wn) > 0) * 16) if f is not None else (16 if name == "PLUGIN_Q_BAYES" else 1)
    return agent.state_scalars(), evals


# -- runs ---------------------------------------------------------------------------------------------------
def evaluate_seed3(partition, seed, regimes, specs, episodes, chunk=2048) -> dict:
    out = {}
    for rname in regimes:
        sums = {}
        for first in range(0, episodes, chunk):
            batch = env3.episodes(partition, seed, rname, first, min(chunk, episodes - first))
            nd = noop_decisions(batch)
            for label, spec in specs.items():
                for key, value in harness2.evaluate_batch2(spec, batch, nd).items():
                    acc = sums.setdefault(label, {})
                    acc[key] = acc.get(key, 0.0) + float(value)
        out[rname] = {label: {k: v / episodes for k, v in acc.items()} for label, acc in sums.items() if acc}
    return out


def _worker(task):
    partition, seed, regimes, specs, episodes = task
    built = {label: {**s, "factory": factory(s["arm"], **s["overrides"])} for label, s in specs.items()}
    return seed, evaluate_seed3(partition, seed, regimes, built, episodes)


def run(partition, seeds, regimes, specs, episodes, workers=4) -> dict:
    seeds = list(seeds)
    t0 = time.time()
    tasks = [(partition, s, list(regimes), specs, episodes) for s in seeds]
    with ProcessPoolExecutor(workers, mp_context=get_context("fork")) as ex:
        by_seed = dict(ex.map(_worker, tasks))
    results = {}
    for r in regimes:
        for label in specs:
            metrics = by_seed[seeds[0]][r].get(label)
            if metrics:
                results.setdefault(r, {})[label] = {m: [float(f"{by_seed[s][r][label][m]:.10g}") for s in seeds]
                                                    for m in metrics}
    return {"partition": partition, "seeds": seeds, "episodes_per_seed": episodes, "regimes": list(regimes),
            "arms": {k: {kk: v[kk] for kk in ("arm", "contract", "mode", "overrides")} for k, v in specs.items()},
            "results": results, "seconds": round(time.time() - t0, 1)}


def source_file_digests() -> dict:
    out = {f"v3/{p.name}": ev1.file_digest(p) for p in sorted(SRC3.glob("*.py"))}
    out.update(evaluate2.source_file_digests())
    return out


def source_digest() -> str:
    h = hashlib.sha256()
    for name, digest in sorted(source_file_digests().items()):
        h.update(name.encode() + b"\0" + digest.encode() + b"\0")
    return h.hexdigest()


def seal(record: dict, phase: str, note: str = "") -> dict:
    return {"schema": "cgc-lao-solver-lab.record.v3", "phase": phase, "note": note, "rng_root": env3.ROOT,
            "source_sha256": source_digest(), "source_files_sha256": source_file_digests(),
            "native_files_sha256": evaluate2.native_file_digests(),
            "spec_sha256": {"slb1": ev1.file_digest(ev1.SPEC), "slb2": ev1.file_digest(evaluate2.SPEC2)},
            "coefficients_sha256": evaluate2.coefficient_digests(), "environment": ev1.environment(), **record}


def round_specs() -> dict:
    s = evaluate2.spec
    specs = {a: s(a, "ORIGINAL", "FIRST_ONLY") for a in c2.ORIGINAL_BASELINES}
    specs["KNOWN_Q_ORACLE"] = s("KNOWN_Q_ORACLE", "ORIGINAL", "FIRST_ONLY")
    for a in list(PRIMARY3) + list(CONTROLS3):
        specs[a] = s(a, "DUP2_4", "DUPLICATE")
    specs["KNOWN_Q_DUP"] = s("KNOWN_Q_ORACLE", "DUP2_4", "DUPLICATE")
    for a in FALLBACK3:
        specs[a] = s(a, "DUP2_4", "FIRST_ONLY")
    return specs


# -- selection (CI-bound) -----------------------------------------------------------------------------------
def select(rec) -> dict:
    rows = {}
    for c in list(PRIMARY3) + list(CONTROLS3):
        g = report2.primary_gates(rec, c)
        passed = {k: bool(g[k]["pass"]) for k in PRIMARY_GATES}
        rows[c] = {"gates": passed, "eligible": all(passed.values()), "ALL": float(report2.series(rec, "ALL", c).mean()),
                   "complexity": list(complexity(c))}
    eligible = [c for c in rows if rows[c]["eligible"]]
    chosen = None
    if eligible:
        best = min(rows[c]["ALL"] for c in eligible)
        chosen = min((c for c in eligible if rows[c]["ALL"] <= best + EPSILON_OCCAM),
                     key=lambda c: (tuple(rows[c]["complexity"]), c))
    frows = {}
    for c in FALLBACK3:
        g = report2.fallback_gates(rec, c)
        passed = {k: bool(g[k]["pass"]) for k in FALLBACK_GATES}
        frows[c] = {"gates": passed, "eligible": all(passed.values()),
                    "CLEAN": float(report2.series(rec, "CLEAN", c).mean()), "clean_retention": g["clean_retention"]}
    feligible = [c for c in frows if frows[c]["eligible"]]
    fchosen = min(feligible, key=lambda c: frows[c]["CLEAN"]) if feligible else None
    return {"primary": chosen, "fallback": fchosen, "primary_rows": rows, "fallback_rows": frows}


# -- native parity on qualification samples -----------------------------------------------------------------
def parity(partition: str, roles: dict, seeds=range(0, 8), episodes=256, bitwise=32) -> dict:
    contract = env2.CONTRACTS["DUP2_4"]
    regimes = env2.DESIGN + env2.ROBUSTNESS
    out = {}
    for role, (name, mode) in roles.items():
        cfg = CONFIGS3[name]
        r = {"name": name, "mode": mode, "episodes": 0, "trajectory_divergences": 0, "decision_mismatches": 0,
             "near_tie_mismatches": 0, "max_state_abs_diff": 0.0, "bitwise_episodes": 0, "bitwise_failures": 0}
        for seed in seeds:
            for regime in regimes:
                batch = env3.episodes(partition, seed, regime, 0, episodes)
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


# -- freeze -------------------------------------------------------------------------------------------------
PLANS = {
    "VALIDATION": {"partition": "VALIDATION", "seeds": [0, 32], "episodes": 4096, "regimes": list(env2.DESIGN + env2.ROBUSTNESS)},
    "HOLDOUT": {"partition": "HOLDOUT", "seeds": [0, 64], "episodes": 4096, "regimes": list(env2.DESIGN + env2.ROBUSTNESS)},
    "OOD": {"partition": "OOD", "seeds": [0, 64], "episodes": 2048,
            "regimes": ["FLIP40"] + list(env2.CORRELATION) + ["EVENT40"] + list(env2.OOD)},
}


def qualification_specs(primary, fallback) -> dict:
    s = evaluate2.spec
    specs = {a: s(a, "ORIGINAL", "FIRST_ONLY") for a in c2.ORIGINAL_BASELINES}
    specs["KNOWN_Q_ORACLE"] = s("KNOWN_Q_ORACLE", "ORIGINAL", "FIRST_ONLY")
    specs[primary] = s(primary, "DUP2_4", "DUPLICATE")
    specs[fallback] = s(fallback, "DUP2_4", "FIRST_ONLY")
    for c in CONTROLS3:
        if c != primary:
            specs[c] = s(c, "DUP2_4", "DUPLICATE")
    specs["KNOWN_Q_DUP"] = s("KNOWN_Q_ORACLE", "DUP2_4", "DUPLICATE")
    return specs


def ood_specs(primary, fallback) -> dict:
    s = evaluate2.spec
    return {"NOOP": s("NOOP", "ORIGINAL", "FIRST_ONLY"), primary: s(primary, "DUP2_4", "DUPLICATE"),
            f"{primary}@SECOND_EVENT": s(primary, "DUP2_4", "SECOND_EVENT"), fallback: s(fallback, "DUP2_4", "FIRST_ONLY"),
            "RMF_V1": s("RMF_V1", "DUP2_4", "DUPLICATE"), "CLEAN_BAYES": s("CLEAN_BAYES", "ORIGINAL", "FIRST_ONLY"),
            "KNOWN_Q_DUP": s("KNOWN_Q_ORACLE", "DUP2_4", "DUPLICATE")}


def freeze() -> dict:
    sel = json.loads((V3_ROOT / "SELECTION_V3.json").read_text(encoding="utf-8"))
    primary, fallback = sel["primary"], sel["fallback"]
    return {"schema": "cgc-lao-solver-lab.freeze.v3",
            "note": "Frozen before any v3 VALIDATION, HOLDOUT or OOD run; selection by SELECTION_V3.json.",
            "candidates": {"PRIMARY": {"name": primary, "config": CONFIGS3[primary], "contract": "DUP2_4", "mode": "DUPLICATE"},
                           "FALLBACK": {"name": fallback, "config": CONFIGS3[fallback], "contract": "DUP2_4",
                                        "mode": "FIRST_ONLY"}},
            "rng_root": env3.ROOT, "source_sha256": source_digest(), "source_files_sha256": source_file_digests(),
            "native_files_sha256": evaluate2.native_file_digests(),
            "spec_sha256": {"slb1": ev1.file_digest(ev1.SPEC), "slb2": ev1.file_digest(evaluate2.SPEC2)},
            "coefficients_sha256": evaluate2.coefficient_digests(), "plans": PLANS,
            "specs": {"VALIDATION": qualification_specs(primary, fallback), "HOLDOUT": qualification_specs(primary, fallback),
                      "OOD": ood_specs(primary, fallback)},
            "holdout_rule": "open HOLDOUT only if the PRIMARY passes G1-G5, G7, G9, G10 on VALIDATION"}


def write(record: dict, path: Path) -> str:
    return evaluate2.write(record, path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("dev", "select", "freeze", "qualify", "parity"))
    ap.add_argument("--phase", default="")
    ap.add_argument("--seeds", default="0:32")
    ap.add_argument("--episodes", type=int, default=2048)
    ap.add_argument("--regimes", default=",".join(env2.DESIGN + env2.ROBUSTNESS))
    a = ap.parse_args(argv)
    ev = V3_ROOT / "evidence"
    if a.command == "dev":
        first, stop = (int(v) for v in a.seeds.split(":"))
        rec = run("DEV", range(first, stop), a.regimes.split(","), round_specs(), a.episodes)
        print(write(seal(rec, "V3-DEV-selection", "the single preregistered v3 DEV round"), ev / "dev" / "selection_round.json"))
    elif a.command == "select":
        rec = json.loads((ev / "dev" / "selection_round.json").read_text(encoding="utf-8"))
        out = {"schema": "cgc-lao-solver-lab.selection.v3", **select(rec),
               "record_sha256": hashlib.sha256((ev / "dev" / "selection_round.json").read_bytes()).hexdigest()}
        (V3_ROOT / "SELECTION_V3.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(out["primary"], out["fallback"])
    elif a.command == "freeze":
        (V3_ROOT / "FROZEN_V3.json").write_text(json.dumps(freeze(), indent=1) + "\n", encoding="utf-8")
        print(source_digest())
    elif a.command == "qualify":
        frozen = json.loads((V3_ROOT / "FROZEN_V3.json").read_text(encoding="utf-8"))
        plan, specs = frozen["plans"][a.phase], frozen["specs"][a.phase]
        rec = run(plan["partition"], range(*plan["seeds"]), plan["regimes"], specs, plan["episodes"])
        print(write(seal(rec, f"V3-{a.phase}"), ev / a.phase.lower() / "qualification.json"), rec["seconds"])
    elif a.command == "parity":
        frozen = json.loads((V3_ROOT / "FROZEN_V3.json").read_text(encoding="utf-8"))
        roles = {r: (c["name"], c["mode"]) for r, c in frozen["candidates"].items()}
        t0 = time.time()
        rec = {"partition": a.phase, "seeds": list(range(8)), "episodes_per_seed_and_regime": 256,
               "bitwise_episodes_per_seed_and_regime": 32, "tolerances": {"state_abs": 1e-9, "near_tie": 1e-9},
               "arms": parity(a.phase, roles)}
        rec["seconds"] = round(time.time() - t0, 1)
        print(rec["arms"])
        print(write(seal(rec, f"V3-{a.phase}-native-parity"), ev / a.phase.lower() / "native_parity.json"))


if __name__ == "__main__":
    main(sys.argv[1:])
