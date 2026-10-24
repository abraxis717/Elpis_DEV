"""H1D0 final Pareto + regime-C abstention analysis.

Answers the actual scientific question:
  Is there any point on the quality-vs-total-cost frontier where a learned
  local typed recurrence beats an equally informed exact engine?
"""
import json, sys, time, random, subprocess, resource
sys.path.insert(0, "src")
from collections import Counter
from contract import DAMAGE_UNITS
from generator import build_split
from oracle import gold_for_case, possible_final_states, prefix_labels, answer_from_state
from engines import Replay, Incremental, CheckpointReplay, HcaNativeSemantics
from learned import FactorNet, edge_arrays
from learned_online import LearnedOnline

SEED, WORLDS = 67890, 2
ds, truth = build_split("TUNE", SEED, WORLDS)
cases = [c for w in ds["worlds"] for c in w["cases"]]
spec = json.load(open("runs/trained_spec.json"))

def nodes_of(case):
    toks = sorted({v for f in case["factors"] for v in f["roles"].values()}
                  | {q["from"] for q in case["queries"]} | {q["to"] for q in case["queries"]})
    return {t: i for i, t in enumerate(toks)}

def full_trajectory(name):
    """Complete online trajectory: build + every event + snapshot after every event."""
    net = FactorNet(rounds=3, seed=spec["seed"])
    net.t = __import__("numpy").array(spec["admission_scores"])
    net.theta = spec["theta"]
    correct = total = 0
    foot = work = 0
    t_build = t_apply = t_snap = 0.0
    for case in cases:
        gold = gold_for_case(case, truth[case["case_id"]])
        if name == "learned_online":
            eng = LearnedOnline(net, case)
        elif name == "replay":
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
        for i, e in enumerate(case["events"]):
            t0 = time.perf_counter()
            eng.apply(e)
            t_apply += time.perf_counter() - t0
            t0 = time.perf_counter()
            snap = eng.snapshot()
            t_snap += time.perf_counter() - t0
            for q in case["queries"]:
                total += 1
                correct += (snap[q["id"]] == gold[i + 1][q["id"]])
        foot += eng.footprint()
        work += sum(eng.work.values())
    return {"engine": name, "accuracy": round(correct / total, 6),
            "apply_s": round(t_apply, 5), "snapshot_s": round(t_snap, 5),
            "total_s": round(t_apply + t_snap, 5),
            "retained_bytes": foot, "logical_work": work,
            "bytes_per_decision": round(foot / total, 2),
            "work_per_decision": round(work / total, 2)}

frontier = [full_trajectory(n) for n in
            ("replay", "incremental", "incremental_integrity", "checkpoint",
             "hca_b0", "hca_b1", "hca_b2", "learned_online")]

# training cost of the learned comparator, measured honestly
net = FactorNet(rounds=3, seed=spec["seed"])
S = []
for case in cases:
    gold = gold_for_case(case, truth[case["case_id"]])
    ni = nodes_of(case)
    cache = {}
    for n in range(len(case["events"]) + 1):
        for q in case["queries"]:
            key = (n, q["scope"], q["kind"])
            if key not in cache:
                cache[key] = edge_arrays(case, n, ni, q)
            S.append((len(ni), ni[q["from"]], ni[q["to"]], cache[key], gold[n][q["id"]]))
t0 = time.perf_counter()
net.fit(S, epochs=40, lr=0.05)
train_s = time.perf_counter() - t0

# regime C: abstention vs forced answer, per decision
trc, trct = build_split("TRAIN", 12345, 4, pending_prob=0.25)
tuc, tutc = build_split("TUNE", 987654, 2, pending_prob=0.25)
regc = {}
for label, (d, t) in (("TRAIN", (trc, trct)), ("TUNE", (tuc, tutc))):
    ident = under = 0
    forced_correct = 0
    abstain_correct = 0
    decisions = 0
    for case in [c for w in d["worlds"] for c in w["cases"]]:
        gold = gold_for_case(case, t[case["case_id"]])
        vis = prefix_labels(case["events"], case["factors"], case["queries"])
        final_gold = gold[len(t[case["case_id"]])]
        for q in case["queries"]:
            decisions += 1
            states = possible_final_states(case["events"], case["pending"])
            vals = {answer_from_state(s, case["factors"], q) for s in states}
            if len(vals) == 1:
                ident += 1
                forced_correct += (vis[-1][q["id"]] == final_gold[q["id"]])
                abstain_correct += (vis[-1][q["id"]] == final_gold[q["id"]])
            else:
                under += 1
                forced_correct += (vis[-1][q["id"]] == final_gold[q["id"]])
    regc[label] = {"decisions": decisions, "identifiable": ident, "underdetermined": under,
                   "forced_accuracy": round(forced_correct / decisions, 6),
                   "abstain_accuracy_on_identifiable": round(abstain_correct / max(ident, 1), 6),
                   "identifiability_cap": round(ident / decisions, 6)}

print(json.dumps({"pareto": frontier,
                  "learned_training_seconds": round(train_s, 2),
                  "learned_param_bytes": net.param_bytes(),
                  "regime_c": regc}, indent=1))
