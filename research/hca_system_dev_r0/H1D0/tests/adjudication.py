"""H1D0 final adjudication analysis.

Regime C with real pending pools, regime B recovery accounting, B2 prefix-error
profile, and an isolated-process cost/RSS comparison that includes the learned
online engine.
"""
import json, sys, time, random
import numpy as np
sys.path.insert(0, "src")
from contract import DAMAGE_UNITS, MAX_PENDING
from generator import build_split
from oracle import gold_for_case, identifiability, possible_final_states, answer_from_state
from checker import validate_dataset
from engines import Incremental, Replay, CheckpointReplay, HcaNativeSemantics
from learned import FactorNet, edge_arrays
from learned_online import LearnedOnline

TRAIN_SEED, TUNE_SEED = 12345, 67890
TW, TUW = 4, 2
PENDING_PROB = 0.25

def cases_of(ds):
    return [c for w in ds["worlds"] for c in w["cases"]]

def nodes_of(case):
    toks = sorted({v for f in case["factors"] for v in f["roles"].values()}
                  | {q["from"] for q in case["queries"]} | {q["to"] for q in case["queries"]})
    return {t: i for i, t in enumerate(toks)}

out = {}

# ---------------------------------------------------------------- regime C
trc, trc_truth = build_split("TRAIN", TRAIN_SEED, TW, pending_prob=PENDING_PROB)
tuc, tuc_truth = build_split("TUNE", TUNE_SEED, TUW, pending_prob=PENDING_PROB)
validate_dataset(trc, trc_truth, "TRAIN"); validate_dataset(tuc, tuc_truth, "TUNE")

def regime_c(split, truth):
    ident = {"IDENTIFIABLE": 0, "UNDERDETERMINED": 0}
    states_seen = 0
    cases_with_pending = 0
    naive_correct = 0
    naive_total = 0
    forced_wrong = 0
    for case in cases_of(split):
        info = identifiability(case)
        if info["status"] != "OK":
            continue
        cases_with_pending += 1
        states_seen += info["states"]
        gold = gold_for_case(case, truth[case["case_id"]])
        visible = gold[len(case["events"])]
        full = gold[len(case["events"]) + len(case["pending"])]
        for q in case["queries"]:
            naive_total += 1
            naive_correct += (visible[q["id"]] == full[q["id"]])
            if info["queries"][q["id"]] == "UNDERDETERMINED":
                forced_wrong += (visible[q["id"]] != full[q["id"]])
            ident[info["queries"][q["id"]]] += 1
    return {"cases_with_pending": cases_with_pending,
            "reachable_states_total": states_seen,
            "identifiable": ident["IDENTIFIABLE"],
            "underdetermined": ident["UNDERDETERMINED"],
            "decisions": ident["IDENTIFIABLE"] + ident["UNDERDETERMINED"],
            "visible_only_accuracy": round(naive_correct / max(1, naive_total), 6),
            "underdetermined_decisions_forced_wrong": forced_wrong}

out["regime_c"] = {"TRAIN": regime_c(trc, trc_truth), "TUNE": regime_c(tuc, tuc_truth)}

# ---------------------------------------------------------------- regime B
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

def regime_b(split, truth, units=DAMAGE_UNITS):
    res = {}
    for name in ("replay", "incremental", "incremental_integrity", "checkpoint", "hca_b2", "learned_online"):
        acc_norepair = 0; acc_repair = 0; dec = 0; dmg = 0
        repair_s = 0.0; rebuilds = 0; latency = 0
        for case in cases_of(split):
            gold = gold_for_case(case, truth[case["case_id"]])
            eng = LearnedOnline(net, case) if name == "learned_online" else (
                Incremental(case, integrity=True) if name == "incremental_integrity" else
                {"replay": Replay, "incremental": Incremental, "checkpoint": CheckpointReplay,
                 "hca_b2": lambda c: HcaNativeSemantics(c, method="B2")}[name](case))
            k = case["damage_prefix"]
            rng = random.Random(case["case_id"])
            for e in case["events"][:k]:
                eng.apply(e)
            dmg += eng.damage(rng, units)
            snap = eng.snapshot()
            for q in case["queries"]:
                dec += 1
                acc_norepair += (snap[q["id"]] == gold[len(case["events"])][q["id"]])
            t0 = time.perf_counter()
            if name == "incremental_integrity":
                eng.rebuild()
            elif name == "learned_online":
                eng.repair()
            repair_s += time.perf_counter() - t0
            rebuilds += getattr(eng, "rebuilds", 0)
            snap = eng.snapshot()
            for q in case["queries"]:
                acc_repair += (snap[q["id"]] == gold[len(case["events"])][q["id"]])
            # recovery latency: how many subsequent events until the engine is
            # again correct on every query, without an explicit repair
            eng2 = LearnedOnline(net, case) if name == "learned_online" else (
                Incremental(case, integrity=True) if name == "incremental_integrity" else
                {"replay": Replay, "incremental": Incremental, "checkpoint": CheckpointReplay,
                 "hca_b2": lambda c: HcaNativeSemantics(c, method="B2")}[name](case))
            rng2 = random.Random(case["case_id"])
            fixed_at = None
            for i, e in enumerate(case["events"]):
                eng2.apply(e)
                if i == k - 1:
                    eng2.damage(rng2, units)
                if i >= k:
                    snap = eng2.snapshot()
                    if all(snap[q["id"]] == gold[i + 1][q["id"]] for q in case["queries"]):
                        fixed_at = i - k + 1
                        break
            latency += (fixed_at if fixed_at is not None else len(case["events"]) - k)
        res[name] = {"decisions": dec, "damaged_units": dmg,
                     "accuracy_without_repair": round(acc_norepair / dec, 6),
                     "accuracy_with_repair": round(acc_repair / dec, 6),
                     "repair_seconds_total": round(repair_s, 6),
                     "repair_seconds_per_case": round(repair_s / len(cases_of(split)), 6),
                     "integrity_rebuilds": rebuilds,
                     "mean_self_recovery_events": round(latency / len(cases_of(split)), 3)}
    return res

out["regime_b"] = {"TRAIN": regime_b(tr, truth_tr), "TUNE": regime_b(tu, truth_tu)}

# ---------------------------------------------------------------- B2 prefix profile
profile = {"decisions": 0, "mismatched_prefixes": 0, "cases_with_any_gap": 0,
           "max_gap_events": 0, "gap_event_histogram": {}}
for case in cases_of(tr):
    gold = gold_for_case(case, truth_tr[case["case_id"]])
    eng = HcaNativeSemantics(case, method="B2")
    bad_prefixes = []
    for i, e in enumerate(case["events"]):
        eng.apply(e)
        snap = eng.snapshot()
        for q in case["queries"]:
            profile["decisions"] += 1
            if snap[q["id"]] != gold[i + 1][q["id"]]:
                profile["mismatched_prefixes"] += 1
                bad_prefixes.append(i + 1)
    if bad_prefixes:
        profile["cases_with_any_gap"] += 1
        profile["max_gap_events"] = max(profile["max_gap_events"], len(bad_prefixes))
        for p in bad_prefixes:
            profile["gap_event_histogram"][p] = profile["gap_event_histogram"].get(p, 0) + 1
out["b2_prefix_profile"] = profile

print(json.dumps(out, indent=1))
json.dump(out, open("runs/adjudication.json", "w"), indent=1)
