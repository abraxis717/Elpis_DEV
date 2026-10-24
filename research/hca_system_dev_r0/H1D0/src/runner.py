"""H1D0 SOURCE_EVOLUTION_RECOVERY_GATE_R0 - main experiment runner.

Phases:
  0  freeze + digests
  1  contract validation, oracle/checker agreement, pair invariants
  2  exact engines differential (G1)
  3  real reference-substrate differential (native B0/B1/B2)
  4  TRAIN-only symbolic + control baselines (G3 gate)
  5  learned local recurrence comparator
  6  regime B: retained-state damage and recovery
  7  regime C: incomplete observation and identifiability
  8  cost accounting (isolated-process RSS, work counters, footprints)
  9  machine-readable report

Nothing here writes to the transferred mirror.  All evidence goes to runs/.
"""
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from hashlib import sha256

from contract import SCHEMA, DAMAGE_UNITS, KINDS
from generator import build_split
from oracle import gold_for_case, prefix_labels, identifiability
from checker import (ContractViolation, check_pairs, nuisance_ablation,
                     validate_dataset, checker_local_labels)
from engines import Replay, Incremental, CheckpointReplay, HcaNativeSemantics
from baselines import SymbolicInducer, Controls, features
from learned import FactorNet, edge_arrays
from learned_online import LearnedOnline

SRC = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = os.path.join(os.path.dirname(SRC), "runs")

TRAIN_SEED = 12345
TUNE_SEED = 987654
TRAIN_WORLDS = 4
TUNE_WORLDS = 2
PENDING_PROB = 0.25
PY = sys.executable


def digest(path):
    with open(path, "rb") as fh:
        return sha256(fh.read()).hexdigest()


def canon(obj):
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def all_cases(ds):
    return [c for w in ds["worlds"] for c in w["cases"]]


def nodes_of(case):
    toks = sorted({v for f in case["factors"] for v in f["roles"].values()}
                  | {q["from"] for q in case["queries"]} | {q["to"] for q in case["queries"]})
    return {t: i for i, t in enumerate(toks)}


def engine_registry(case):
    return {
        "replay": Replay(case),
        "incremental": Incremental(case),
        "checkpoint": CheckpointReplay(case, every=8),
        "hca_b0": HcaNativeSemantics(case, method="B0"),
        "hca_b1": HcaNativeSemantics(case, method="B1", rounds=4),
        "hca_b2": HcaNativeSemantics(case, method="B2", rounds=4),
    }


# ---------------------------------------------------------------------------
# phase 1
# ---------------------------------------------------------------------------
def phase_freeze():
    files = ["contract.py", "generator.py", "oracle.py", "checker.py", "engines.py",
             "baselines.py", "learned.py", "native_ref.py"]
    return {f: digest(os.path.join(SRC, f)) for f in files}


def phase_validate(ds, truth, which):
    case_ids, pairs, stats = validate_dataset(ds, truth, which)
    pair_stats = check_pairs(pairs, {k: 2 * TRAIN_WORLDS if which == "TRAIN" else 2 * TUNE_WORLDS
                                     for k in KINDS})
    for c in all_cases(ds):
        nuisance_ablation(c)
    disagree = 0
    decisions = 0
    positives = 0
    for c in all_cases(ds):
        a = checker_local_labels(c["events"], c["factors"], c["queries"])
        b = gold_for_case(c, truth[c["case_id"]])
        decisions += len(a) * len(c["queries"])
        positives += sum(sum(r.values()) for r in a)
        if a != b:
            disagree += 1
    return {"cases": len(all_cases(ds)), "decisions": decisions, "positives": positives,
           "oracle_checker_disagreement_cases": disagree, "pair_stats": pair_stats,
           "structural": dict(stats)}


# ---------------------------------------------------------------------------
# phase 2
# ---------------------------------------------------------------------------
def phase_exact(ds, truth):
    mismatches = Counter()
    decisions = 0
    footprints = Counter()
    work = Counter()
    times = defaultdict(list)
    for c in all_cases(ds):
        gold = gold_for_case(c, truth[c["case_id"]])
        eng = engine_registry(c)
        for name, e in eng.items():
            t0 = time.perf_counter()
            e.apply(c["events"][0]) if c["events"] else None
            t1 = time.perf_counter()
            times[name].append(t1 - t0)
        for i, ev in enumerate(c["events"]):
            for name, e in eng.items():
                t0 = time.perf_counter()
                if i > 0:
                    e.apply(ev)
                snap = e.snapshot()
                times[name].append(time.perf_counter() - t0)
                for q in c["queries"]:
                    decisions += 1
                    if snap[q["id"]] != gold[i + 1][q["id"]]:
                        mismatches[name] += 1
        for name, e in eng.items():
            footprints[name] += e.footprint()
            for k, v in e.work.items():
                work[(name, k)] += v
    return {"decisions": decisions, "mismatches": dict(mismatches),
           "footprint_bytes_total": dict(footprints),
           "work_counters": {f"{a}.{b}": v for (a, b), v in work.items()},
           "mean_step_ms": {k: round(1000 * sum(v) / len(v), 4) for k, v in times.items()}}


# ---------------------------------------------------------------------------
# phase 3
# ---------------------------------------------------------------------------
def phase_reference(ds, truth, limit=6):
    try:
        from native_ref import reference_predict
    except Exception as e:
        return {"error": "reference substrate unavailable: " + str(e)}
    out = {"comparisons": 0, "mismatches": Counter(), "work": 0, "seconds": 0.0}
    cases = all_cases(ds)[:limit]
    t0 = time.perf_counter()
    for c in cases:
        gold = gold_for_case(c, truth[c["case_id"]])
        n = len(c["events"])
        for q in c["queries"]:
            for method, rounds in (("B0", 4), ("B1", 4)):
                try:
                    pred, w = reference_predict(c, n, q, method, rounds=rounds)
                except Exception as e:
                    out["mismatches"]["reference_error_" + method] += 1
                    continue
                out["comparisons"] += 1
                out["work"] += w
                if pred != gold[n][q["id"]]:
                    out["mismatches"]["ref" + method + "_vs_gold"] += 1
        port = HcaNativeSemantics(c, method="B0")
        for ev in c["events"]:
            port.apply(ev)
        snap = port.snapshot()
        for q in c["queries"]:
            out["comparisons"] += 1
            try:
                ref, _w = reference_predict(c, n, q, "B0", rounds=4)
                if ref != snap[q["id"]]:
                    out["mismatches"]["portB0_vs_refB0"] += 1
            except Exception:
                out["mismatches"]["reference_error_port"] += 1
    out["seconds"] = round(time.perf_counter() - t0, 2)
    return out


# ---------------------------------------------------------------------------
# phase 4
# ---------------------------------------------------------------------------
def baseline_samples(ds, truth):
    out = []
    for c in all_cases(ds):
        gold = gold_for_case(c, truth[c["case_id"]])
        for n in range(len(c["events"]) + 1):
            for q in c["queries"]:
                out.append(((c["case_id"], q["id"], c["pair_id"]), features(c, n, q),
                            gold[n][q["id"]]))
    return out


def phase_baselines(tr, tr_truth, tu, tu_truth):
    S = baseline_samples(tr, tr_truth)
    T = baseline_samples(tu, tu_truth)
    res = {}
    for depth in ("full", "local"):
        ind = SymbolicInducer(depth=depth)
        t0 = time.perf_counter()
        ind.fit(S)
        secs = time.perf_counter() - t0
        acc_tr = sum(1 for k, f, y in S if ind.predict(f) == y) / len(S)
        acc_tu = sum(1 for k, f, y in T if ind.predict(f) == y) / len(T)
        res["symbolic_" + depth] = {"train": round(acc_tr, 4), "tune": round(acc_tu, 4),
                                    "rules": ind.rules, "fit_seconds": round(secs, 3)}
    ctl = Controls()
    ctl.fit(S)
    res["majority"] = round(sum(1 for k, f, y in T if ctl.predict_majority(f) == y) / len(T), 4)
    res["graph_statistic"] = round(sum(1 for k, f, y in T if ctl.predict_stat(f) == y) / len(T), 4)
    res["order_only"] = round(sum(1 for k, f, y in T if ctl.predict_order(f) == y) / len(T), 4)
    res["case_memorisation"] = round(sum(1 for k, f, y in T if ctl.predict(k, f) == y) / len(T), 4)
    res["query_identity_memorisation"] = round(sum(1 for k, f, y in T if ctl.predict_query_identity(k) == y) / len(T), 4)
    res["n_train_samples"] = len(S)
    res["n_tune_samples"] = len(T)
    return res


# ---------------------------------------------------------------------------
# phase 5
# ---------------------------------------------------------------------------
def learned_samples(ds, truth):
    out = []
    for c in all_cases(ds):
        gold = gold_for_case(c, truth[c["case_id"]])
        ni = nodes_of(c)
        cache = {}
        for n in range(len(c["events"]) + 1):
            for q in c["queries"]:
                key = (n, q["scope"], q["kind"])
                if key not in cache:
                    cache[key] = edge_arrays(c, n, ni, q)
                out.append((len(ni), ni[q["from"]], ni[q["to"]], cache[key], gold[n][q["id"]]))
    return out


def phase_learned(tr, tr_truth, tu, tu_truth, seeds, epochs=40, lr=0.05):
    S = learned_samples(tr, tr_truth)
    T = learned_samples(tu, tu_truth)
    runs = []
    for seed in seeds:
        net = FactorNet(rounds=3, seed=seed)
        t0 = time.perf_counter()
        net.fit(S, epochs=epochs, lr=lr)
        secs = time.perf_counter() - t0
        acc_tr = sum(1 for nc, a, b, e, y in S if net.predict(nc, a, b, e)[0] == y) / len(S)
        acc_tu = sum(1 for nc, a, b, e, y in T if net.predict(nc, a, b, e)[0] == y) / len(T)
        runs.append({"seed": seed, "train": round(acc_tr, 4), "tune": round(acc_tu, 4),
                     "seconds": round(secs, 2), "param_bytes": net.param_bytes(),
                     "admission_scores": net.t.tolist(), "theta": round(net.theta, 4)})
    return {"runs": runs, "n_train": len(S), "n_tune": len(T),
           "architecture": "3-round max-plus typed recurrence, 19 parameters, "
                           "trained with Adam, all seeds reported (no seed selection)"}


# ---------------------------------------------------------------------------
# phase 6
# ---------------------------------------------------------------------------
def phase_damage(ds, truth, units=DAMAGE_UNITS, learned_net=None):
    res = {}
    names = ["replay", "incremental", "checkpoint", "hca_b2"]
    if learned_net is not None:
        names.append("learned_online")
    for name in names:
        correct = 0
        total = 0
        rebuilds = 0
        for c in all_cases(ds):
            gold = gold_for_case(c, truth[c["case_id"]])
            k = c["damage_prefix"]
            eng = (LearnedOnline(learned_net, c) if name == "learned_online"
                   else engine_registry(c)[name])
            for ev in c["events"][:k]:
                eng.apply(ev)
            rng = __import__("random").Random(c["case_id"])
            damaged = eng.damage(rng, units)
            if hasattr(eng, "recover"):
                eng.recover()
            for ev in c["events"][k:]:
                eng.apply(ev)
            snap = eng.snapshot()
            for q in c["queries"]:
                total += 1
                correct += snap[q["id"]] == gold[len(c["events"])][q["id"]]
            rebuilds += getattr(eng, "rebuilds", 0)
        res[name] = {"decisions": total, "correct": correct,
                    "accuracy": round(correct / total, 4), "rebuilds": rebuilds}
    return res


# ---------------------------------------------------------------------------
# phase 7
# ---------------------------------------------------------------------------
def phase_incomplete(ds, truth):
    ident = Counter()
    naive = 0
    total = 0
    for c in all_cases(ds):
        info = identifiability(c)
        for qid, st in info["queries"].items():
            ident[st] += 1
        if not c["pending"]:
            continue
        gold = gold_for_case(c, truth[c["case_id"]])
        visible = prefix_labels(c["events"], c["factors"], c["queries"])[-1]
        for q in c["queries"]:
            total += 1
            naive += visible[q["id"]] == gold[len(truth[c["case_id"]])][q["id"]]
    return {"identifiability": dict(ident),
           "visible_only_accuracy_on_pending_cases": round(naive / total, 4) if total else None,
           "cases_with_pending": sum(1 for c in all_cases(ds) if c["pending"])}


# ---------------------------------------------------------------------------
# phase 8
# ---------------------------------------------------------------------------
BENCH = r'''
import json, resource, sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np
from generator import build_split
from engines import Replay, Incremental, CheckpointReplay, HcaNativeSemantics

seed = int(sys.argv[2]); worlds = int(sys.argv[3])
ds, truth = build_split("TUNE", seed, worlds)
cases = [c for w in ds["worlds"] for c in w["cases"]]
out = {}
from learned import FactorNet
from learned_online import LearnedOnline
spec = json.load(open(sys.argv[4]))
trained = FactorNet(rounds=3, seed=spec["seed"])
trained.t = np.array(spec["admission_scores"])
trained.theta = spec["theta"]
for name, ctor in (("replay", lambda c: Replay(c)),
                   ("incremental", lambda c: Incremental(c)),
                   ("checkpoint", lambda c: CheckpointReplay(c, every=8)),
                   ("hca_b0", lambda c: HcaNativeSemantics(c, method="B0")),
                   ("hca_b1", lambda c: HcaNativeSemantics(c, method="B1", rounds=4)),
                   ("hca_b2", lambda c: HcaNativeSemantics(c, method="B2", rounds=4)),
                   ("learned_online", lambda c: LearnedOnline(trained, c))):
    t0 = time.perf_counter()
    build = 0.0
    apply = 0.0
    snap = 0.0
    foot = 0
    work = 0
    decisions = 0
    for c in cases:
        a = time.perf_counter()
        eng = ctor(c)
        build += time.perf_counter() - a
        for ev in c["events"]:
            a = time.perf_counter()
            eng.apply(ev)
            apply += time.perf_counter() - a
            a = time.perf_counter()
            eng.snapshot()
            snap += time.perf_counter() - a
            decisions += len(c["queries"])
            foot += eng.footprint()
            work += sum(eng.work.values())
    out[name] = {"build_s": round(build, 4), "apply_s": round(apply, 4),
                 "snapshot_s": round(snap, 4), "total_s": round(build + apply + snap, 4),
                 "decisions": decisions, "footprint_bytes": foot, "logical_work": work}
out["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps(out))
'''


def phase_costs(src_dir, net_spec):
    p = subprocess.run([PY, "-c", BENCH, src_dir, str(TUNE_SEED), str(TUNE_WORLDS), net_spec],
                      capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": p.stderr[-1500:]}
    return json.loads(p.stdout)


def main():
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = os.path.join(RUN_ROOT, run_id + "_H1D0")
    os.makedirs(out_dir, exist_ok=True)
    report = {"run_id": run_id, "schema": SCHEMA,
              "classification": "PROSPECTIVE DEV CONTRACT, NOT QUAL. No B3 authorization."}
    report["frozen_digests"] = phase_freeze()
    tr, tr_truth = build_split("TRAIN", TRAIN_SEED, TRAIN_WORLDS)
    tu, tu_truth = build_split("TUNE", TUNE_SEED, TUNE_WORLDS)
    report["split_digests"] = {"train": canon(tr), "tune": canon(tu),
                               "train_truth": canon(tr_truth), "tune_truth": canon(tu_truth)}
    report["validation"] = {"TRAIN": phase_validate(tr, tr_truth, "TRAIN"),
                            "TUNE": phase_validate(tu, tu_truth, "TUNE")}
    report["exact_engines"] = {"TRAIN": phase_exact(tr, tr_truth),
                               "TUNE": phase_exact(tu, tu_truth)}
    report["reference_substrate"] = phase_reference(tr, tr_truth)
    report["baselines"] = phase_baselines(tr, tr_truth, tu, tu_truth)
    nets = {}
    for seed in (20261023, 20261029, 20261031):
        n = FactorNet(rounds=3, seed=seed)
        n.fit(learned_samples(tr, tr_truth), epochs=40, lr=0.05)
        nets[seed] = n
    report["learned"] = phase_learned(tr, tr_truth, tu, tu_truth, seeds=[20261023, 20261029, 20261031])
    report["regime_b_damage"] = {"TRAIN": phase_damage(tr, tr_truth, learned_net=nets[20261023]),
                                 "TUNE": phase_damage(tu, tu_truth, learned_net=nets[20261023])}
    tr_c, tr_c_truth = build_split("TRAIN", TRAIN_SEED, TRAIN_WORLDS, pending_prob=PENDING_PROB)
    tu_c, tu_c_truth = build_split("TUNE", TUNE_SEED, TUNE_WORLDS, pending_prob=PENDING_PROB)
    report["regime_c"] = {"TRAIN": phase_incomplete(tr_c, tr_c_truth),
                          "TUNE": phase_incomplete(tu_c, tu_c_truth)}
    net_spec_path = os.path.join(out_dir, "learned_net.json")
    with open(net_spec_path, "w") as fh:
        json.dump({"seed": 20261023, "rounds": 3,
                   "admission_scores": nets[20261023].t.tolist(),
                   "theta": nets[20261023].theta}, fh)
    report["costs"] = phase_costs(SRC, net_spec_path)
    mut = subprocess.run([PY, os.path.join(os.path.dirname(SRC), "tests", "test_mutations.py")],
                         capture_output=True, text=True, env={**os.environ, "PYTHONPATH": SRC})
    report["mutation_suite"] = {"exit_code": mut.returncode, "output": mut.stdout,
                                "stderr": mut.stderr[-2000:]}
    path = os.path.join(out_dir, "H1D0_REPORT.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    manifest = {}
    for f in sorted(os.listdir(out_dir)):
        manifest[f] = digest(os.path.join(out_dir, f))
    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(report, indent=2)[:6000])
    print("REPORT:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
