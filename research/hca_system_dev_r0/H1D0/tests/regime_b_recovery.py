"""H1D0 regime-B recovery experiment (correct sequence).

Protocol, per case:
  1. apply events[:k]
  2. damage k retained-state units
  3a. measure snapshot immediately (accuracy without repair)
  3b. apply the remaining events, measure accuracy (self-recovery)
  3c. separately: damage, then explicit repair, then apply the rest (repair path)
Costs are measured per path and compared against the exact replay baseline,
which has no derived state to damage.
"""
import json, sys, time, random
sys.path.insert(0, "src")
from contract import DAMAGE_UNITS
from generator import build_split
from oracle import gold_for_case
from engines import Replay, Incremental, CheckpointReplay, HcaNativeSemantics
from learned import FactorNet, edge_arrays
from learned_online import LearnedOnline

TRAIN_SEED, TUNE_SEED = 12345, 67890
TW, TUW = 4, 2

def cases_of(ds):
    return [c for w in ds["worlds"] for c in w["cases"]]

def nodes_of(case):
    toks = sorted({v for f in case["factors"] for v in f["roles"].values()}
                  | {q["from"] for q in case["queries"]} | {q["to"] for q in case["queries"]})
    return {t: i for i, t in enumerate(toks)}

tr, truth_tr = build_split("TRAIN", TRAIN_SEED, TW)
tu, truth_tu = build_split("TUNE", TUNE_SEED, TUW)

net = FactorNet(rounds=3, seed=20261023)
S = []
for case in cases_of(tr):
    gold = gold_for_case(case, truth_tr[case["case_id"]])
    ni = nodes_of(case)
    cache = {}
    for n in range(len(case["events"]) + 1):
        for q in case["queries"]:
            key = (n, q["scope"], q["kind"])
            if key not in cache:
                cache[key] = edge_arrays(case, n, ni, q)
            S.append((len(ni), ni[q["from"]], ni[q["to"]], cache[key], gold[n][q["id"]]))
net.fit(S, epochs=40, lr=0.05)

def ctor(case, name):
    if name == "replay":
        return Replay(case)
    if name == "incremental":
        return Incremental(case)
    if name == "incremental_integrity":
        return Incremental(case, integrity=True)
    if name == "checkpoint":
        return CheckpointReplay(case, every=8)
    if name == "hca_b2":
        return HcaNativeSemantics(case, method="B2")
    if name == "learned_online":
        return LearnedOnline(net, case)
    raise KeyError(name)

def repair(eng, name):
    t0 = time.perf_counter()
    if name == "incremental_integrity":
        eng.rebuild()
    elif name == "learned_online":
        eng.repair()
    return time.perf_counter() - t0

def run(split, truth, units=DAMAGE_UNITS):
    res = {}
    for name in ("replay", "incremental", "incremental_integrity", "checkpoint", "hca_b2", "learned_online"):
        cs = cases_of(split)
        dmg = 0
        acc_damaged = 0
        acc_self = 0
        acc_repaired = 0
        repair_s = 0.0
        recovery_s = 0.0
        events_to_fix = []
        dec = 0
        for case in cs:
            gold = gold_for_case(case, truth[case["case_id"]])
            k = case["damage_prefix"]
            full = gold[len(case["events"])]
            rng = random.Random(case["case_id"])

            eng = ctor(case, name)
            for e in case["events"][:k]:
                eng.apply(e)
            dmg += eng.damage(rng, units)
            snap = eng.snapshot()
            for q in case["queries"]:
                dec += 1
                acc_damaged += (snap[q["id"]] == full[q["id"]])
            t0 = time.perf_counter()
            for e in case["events"][k:]:
                eng.apply(e)
            recovery_s += time.perf_counter() - t0
            snap = eng.snapshot()
            for q in case["queries"]:
                acc_self += (snap[q["id"]] == full[q["id"]])

            rng2 = random.Random(case["case_id"])
            eng2 = ctor(case, name)
            for e in case["events"][:k]:
                eng2.apply(e)
            eng2.damage(rng2, units)
            repair_s += repair(eng2, name)
            for e in case["events"][k:]:
                eng2.apply(e)
            snap = eng2.snapshot()
            for q in case["queries"]:
                acc_repaired += (snap[q["id"]] == full[q["id"]])

            # self-recovery latency: first prefix after damage where every query is right
            rng3 = random.Random(case["case_id"])
            eng3 = ctor(case, name)
            fixed = None
            for i, e in enumerate(case["events"]):
                eng3.apply(e)
                if i == k - 1:
                    eng3.damage(rng3, units)
                if i >= k:
                    snap = eng3.snapshot()
                    if all(snap[q["id"]] == gold[i + 1][q["id"]] for q in case["queries"]):
                        fixed = i - k + 1
                        break
            events_to_fix.append(fixed if fixed is not None else len(case["events"]) - k)

        nq = len(cs[0]["queries"])
        res[name] = {
            "decisions": dec,
            "damaged_units": dmg,
            "accuracy_immediately_after_damage": round(acc_damaged / dec, 6),
            "accuracy_after_self_recovery": round(acc_self / dec, 6),
            "accuracy_after_explicit_repair": round(acc_repaired / dec, 6),
            "explicit_repair_seconds_total": round(repair_s, 6),
            "explicit_repair_seconds_per_case": round(repair_s / len(cs), 7),
            "self_recovery_seconds_total": round(recovery_s, 6),
            "mean_events_until_self_recovery": round(sum(events_to_fix) / len(cs), 3),
            "cases_never_self_recovered": sum(1 for x in events_to_fix if x > len(cs[0]["events"]) // 2),
        }
    return res

out = {"protocol": "damage at damage_prefix, then either self-recover or explicitly repair",
       "damage_units": DAMAGE_UNITS,
       "TRAIN": run(tr, truth_tr),
       "TUNE": run(tu, truth_tu)}
print(json.dumps(out, indent=1))
json.dump(out, open("runs/regime_b_recovery.json", "w"), indent=1)
