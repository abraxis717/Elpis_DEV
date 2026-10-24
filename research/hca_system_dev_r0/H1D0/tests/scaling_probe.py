"""H1D0 scaling probe: does the exact incremental engine's dependency maintenance
become the dominant cost as the query/factor ratio grows?

Sweeps the number of concurrent queries per case and the event-log length, and
measures per-event update cost for replay, incremental, checkpoint, hca_b0/b1/b2
and the learned online recurrence.
"""
import json, sys, time, random
sys.path.insert(0, "src")
import numpy as np
from generator import make_pair, KINDS
from oracle import gold_for_case
from engines import Replay, Incremental, CheckpointReplay, HcaNativeSemantics
from learned import FactorNet, edge_arrays
from learned_online import LearnedOnline

def nodes_of(case):
    toks = sorted({v for f in case["factors"] for v in f["roles"].values()}
                  | {q["from"] for q in case["queries"]} | {q["to"] for q in case["queries"]})
    return {t: i for i, t in enumerate(toks)}

def ctor(case, name, net=None):
    if name == "replay":
        return Replay(case)
    if name == "incremental":
        return Incremental(case)
    if name == "checkpoint":
        return CheckpointReplay(case, every=8)
    if name == "hca_b0":
        return HcaNativeSemantics(case, method="B0")
    if name == "hca_b1":
        return HcaNativeSemantics(case, method="B1")
    if name == "hca_b2":
        return HcaNativeSemantics(case, method="B2")
    if name == "learned_online":
        return LearnedOnline(net, case)
    raise KeyError(name)

wr = random.Random(99991)
cases = []
for kind in KINDS:
    cases.extend(make_pair(wr, kind))

# train a net on the probe cases (upper-bound reference, not a scientific claim)
S = []
truth = {c["case_id"]: [e for e in c["events"]] for c in cases}
for case in cases:
    gold = gold_for_case(case, truth[case["case_id"]])
    ni = nodes_of(case)
    for n in range(len(case["events"]) + 1):
        for q in case["queries"]:
            S.append((len(ni), ni[q["from"]], ni[q["to"]], edge_arrays(case, n, ni, q), gold[n][q["id"]]))
net = FactorNet(rounds=3, seed=1)
net.fit(S, epochs=25, lr=0.05)

names = ["replay", "incremental", "checkpoint", "hca_b0", "hca_b1", "hca_b2", "learned_online"]
rows = []
for case in cases:
    row = {"case_id": case["case_id"], "events": len(case["events"]),
           "queries": len(case["queries"]), "factors": len(case["factors"])}
    for name in names:
        eng = ctor(case, name, net)
        t0 = time.perf_counter()
        for e in case["events"]:
            eng.apply(e)
            eng.snapshot()
        row[name + "_s"] = round(time.perf_counter() - t0, 5)
        row[name + "_bytes"] = eng.footprint()
        row[name + "_work"] = sum(eng.work.values())
    rows.append(row)

agg = {}
for name in names:
    agg[name] = {"total_s": round(sum(r[name + "_s"] for r in rows), 4),
                "mean_bytes": round(sum(r[name + "_bytes"] for r in rows) / len(rows), 1),
                "total_work": sum(r[name + "_work"] for r in rows)}
print(json.dumps({"cases": len(rows),
                  "queries_per_case": [r["queries"] for r in rows],
                  "aggregate": agg}, indent=1))
json.dump({"cases": rows, "aggregate": agg}, open("runs/scaling.json", "w"), indent=1)
