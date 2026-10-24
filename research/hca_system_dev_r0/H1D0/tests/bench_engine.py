"""Isolated-process per-engine cost benchmark.

Each engine runs in its own process under a fixed thread budget, measuring:
  build_s, apply_s, snapshot_s, total_s, retained footprint bytes, logical work,
  peak RSS (resource.getrusage), and accuracy against the independent oracle.
"""
import json, resource, sys, time
sys.path.insert(0, "src")
import numpy as np
from generator import build_split
from oracle import gold_for_case
from engines import Replay, Incremental, CheckpointReplay, HcaNativeSemantics
from learned import FactorNet, edge_arrays
from learned_online import LearnedOnline

name = sys.argv[1]
seed = int(sys.argv[2])
worlds = int(sys.argv[3])
spec_path = sys.argv[4]

ds, truth = build_split("TUNE", seed, worlds)
cases = [c for w in ds["worlds"] for c in w["cases"]]

if name == "learned_online":
    spec = json.load(open(spec_path))
    net = FactorNet(rounds=3, seed=spec["seed"])
    net.t = np.array(spec["admission_scores"])
    net.theta = spec["theta"]
else:
    net = None

t_build = 0.0
t_apply = 0.0
t_snap = 0.0
foot = 0
work = 0
correct = 0
decisions = 0
for case in cases:
    gold = gold_for_case(case, truth[case["case_id"]])
    t0 = time.perf_counter()
    if name == "replay":
        eng = Replay(case)
    elif name == "incremental":
        eng = Incremental(case)
    elif name == "incremental_integrity":
        eng = Incremental(case, integrity=True)
    elif name == "checkpoint":
        eng = CheckpointReplay(case, every=8)
    elif name == "hca_b0":
        eng = HcaNativeSemantics(case, method="B0")
    elif name == "hca_b1":
        eng = HcaNativeSemantics(case, method="B1")
    elif name == "hca_b2":
        eng = HcaNativeSemantics(case, method="B2")
    elif name == "learned_online":
        eng = LearnedOnline(net, case)
    else:
        raise SystemExit("unknown engine " + name)
    t_build += time.perf_counter() - t0
    for i, e in enumerate(case["events"]):
        t0 = time.perf_counter()
        eng.apply(e)
        t_apply += time.perf_counter() - t0
        t0 = time.perf_counter()
        snap = eng.snapshot()
        t_snap += time.perf_counter() - t0
        for q in case["queries"]:
            decisions += 1
            correct += (snap[q["id"]] == gold[i + 1][q["id"]])
    foot += eng.footprint()
    work += sum(eng.work.values())

print(json.dumps({
    "engine": name,
    "decisions": decisions,
    "accuracy": round(correct / decisions, 6),
    "build_s": round(t_build, 5),
    "apply_s": round(t_apply, 5),
    "snapshot_s": round(t_snap, 5),
    "total_s": round(t_build + t_apply + t_snap, 5),
    "retained_footprint_bytes": foot,
    "logical_work": work,
    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "param_bytes": (net.param_bytes() if net else 0),
}))
