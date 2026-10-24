"""H1D0 decisive analysis: B2 convergence, regime-B recovery cost, Pareto frontier,
regime-C calibration/abstention, and learned-vs-exact equivalence."""
import json, sys, time, random
import numpy as np
sys.path.insert(0, "src")
from generator import build_split
from oracle import gold_for_case, possible_final_states
from engines import Incremental, Replay, CheckpointReplay, HcaNativeSemantics
from learned import FactorNet, edge_arrays
from learned_online import LearnedOnline

TRAIN_SEED, TUNE_SEED = 12345, 67890
TW, TUW = 4, 2
tr, truth_tr = build_split("TRAIN", TRAIN_SEED, TW)
tu, truth_tu = build_split("TUNE", TUNE_SEED, TUW)

def cases_of(ds):
    return [c for w in ds["worlds"] for c in w["cases"]]

def nodes_of(case):
    toks = sorted({v for f in case["factors"] for v in f["roles"].values()}
                  | {q["from"] for q in case["queries"]} | {q["to"] for q in case["queries"]})
    return {t: i for i, t in enumerate(toks)}

def make(case, name):
    if name == "replay":
        return Replay(case)
    if name == "incremental":
        return Incremental(case)
    if name == "incremental_integrity":
        return Incremental(case, integrity=True)
    if name == "checkpoint":
        return CheckpointReplay(case)
    if name == "hca_b0":
        return HcaNativeSemantics(case, method="B0")
    if name == "hca_b1":
        return HcaNativeSemantics(case, method="B1")
    if name == "hca_b2":
        return HcaNativeSemantics(case, method="B2")
    raise KeyError(name)

out = {}

# ---------- 1. B2 convergence study -----------------------------------------
cases = cases_of(tr)
conv = {}
for rounds, updates in [(4, 256), (4, 512), (8, 256), (8, 1024), (16, 256), (32, 256)]:
    mism = 0; dec = 0
    for c in cases:
        gold = gold_for_case(c, truth_tr[c["case_id"]])
        eng = HcaNativeSemantics(c, method="B2", rounds=rounds, updates_per_tick=updates)
        for i, e in enumerate(c["events"]):
            eng.apply(e)
            snap = eng.snapshot()
            for q in c["queries"]:
                dec += 1
                mism += (snap[q["id"]] != gold[i + 1][q["id"]])
    conv[f"rounds={rounds},updates={updates}"] = {"mismatches": mism, "decisions": dec}
out["b2_convergence"] = conv

ticks_to_fix = []
for c in cases:
    gold = gold_for_case(c, truth_tr[c["case_id"]])
    eng = HcaNativeSemantics(c, method="B2", rounds=4, updates_per_tick=256)
    for e in c["events"]:
        eng.apply(e)
    snap = eng.snapshot()
    bad = [q["id"] for q in c["queries"] if snap[q["id"]] != gold[len(c["events"])][q["id"]]]
    if not bad:
        continue
    extra = 0
    while bad and extra < 400:
        eng._tick()
        extra += 1
        snap = eng.snapshot()
        bad = [q["id"] for q in c["queries"] if snap[q["id"]] != gold[len(c["events"])][q["id"]]]
    ticks_to_fix.append(extra)
out["b2_extra_ticks_to_converge"] = {
    "cases_with_gap": len(ticks_to_fix),
    "max_extra_ticks": max(ticks_to_fix) if ticks_to_fix else 0,
    "mean_extra_ticks": round(sum(ticks_to_fix) / len(ticks_to_fix), 2) if ticks_to_fix else 0,
}

# ---------- 1b. Train the learned comparator (all seeds reported) ----------
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
t0 = time.perf_counter()
net.fit(S, epochs=40, lr=0.05)
train_secs = time.perf_counter() - t0

# ---------- 2. Regime B: damage + repair, cost and latency ------------------
def regime_b(split, truth, units=12, net=None):
    res = {}
    names = ["replay", "incremental", "incremental_integrity", "checkpoint", "hca_b2"]
    if net is not None:
        names.append("learned_online")
    for name in names:
        acc = 0; dec = 0; dmg = 0; recov_s = 0.0; rebuilds = 0
        for case in cases_of(split):
            gold = gold_for_case(case, truth[case["case_id"]])
            eng = LearnedOnline(net, case) if name == "learned_online" else make(case, name)
            rng = random.Random(4242)
            for e in case["events"]:
                eng.apply(e)
            dmg += eng.damage(rng, units)
            t0 = time.perf_counter()
            if name == "incremental_integrity":
                eng.rebuild()
            elif name == "learned_online":
                eng.repair()
            recov_s += time.perf_counter() - t0
            snap = eng.snapshot()
            for q in case["queries"]:
                dec += 1
                acc += (snap[q["id"]] == gold[len(case["events"])][q["id"]])
            rebuilds += getattr(eng, "rebuilds", 0)
        res[name] = {"decisions": dec, "damaged_units": dmg,
                     "accuracy_after_damage_and_repair": round(acc / dec, 6),
                     "repair_s": round(recov_s, 5), "rebuilds": rebuilds}
    return res

out["regime_b_with_repair"] = {"TRAIN": regime_b(tr, truth_tr, net=net), "TUNE": regime_b(tu, truth_tu, net=net)}

# ---------- 3. Pareto frontier: retained bytes vs accuracy vs time ----------
def frontier(split, truth):
    rows = []
    for name in ("replay", "incremental", "checkpoint", "hca_b0", "hca_b1", "hca_b2"):
        fp = 0; work = 0; t = 0.0; acc = 0; dec = 0
        for case in cases_of(split):
            gold = gold_for_case(case, truth[case["case_id"]])
            eng = make(case, name)
            t0 = time.perf_counter()
            for i, e in enumerate(case["events"]):
                eng.apply(e)
                snap = eng.snapshot()
                for q in case["queries"]:
                    dec += 1
                    acc += (snap[q["id"]] == gold[i + 1][q["id"]])
            t += time.perf_counter() - t0
            fp += eng.footprint()
            work += sum(eng.work.values())
        rows.append({"engine": name, "retained_bytes": fp, "logical_work": work,
                    "trajectory_s": round(t, 4), "accuracy": round(acc / dec, 6)})
    return sorted(rows, key=lambda r: r["retained_bytes"])

out["pareto_train"] = frontier(tr, truth_tr)

# ---------- 4. Regime C calibration / abstention ----------------------------
def live_from_state(state, factors):
    status, present, src = state
    live = []
    for f in factors:
        if not present.get(f["id"], False):
            continue
        s = src.get(f["id"], f["source"])
        st = status.get(s)
        if st is not None and st[0] == "VALID" and st[1] == f["version"]:
            live.append(f)
    return live


def answer_from_live(live, query):
    for f0 in live:
        if f0["rel"] != "R0":
            continue
        r = f0["roles"]
        if r["from"] != query["from"] or r["scope"] != query["scope"] or r["kind"] != query["kind"]:
            continue
        for f1 in live:
            if f1["rel"] != "R1":
                continue
            r1 = f1["roles"]
            if r1["from"] != r["to"] or r1["scope"] != query["scope"] or r1["kind"] != query["kind"]:
                continue
            for f2 in live:
                if f2["rel"] != "R2":
                    continue
                r2 = f2["roles"]
                if (r2["from"] == r1["to"] and r2["to"] == query["to"]
                        and r2["scope"] == query["scope"] and r2["kind"] == query["kind"]):
                    return 1
    return 0


def abstention(split, truth):
    res = {"pending_decisions": 0, "identifiable": 0, "underdetermined": 0,
           "wrong_if_forced_to_first_state": 0, "exact_visible_only_accuracy": 0.0}
    for case in cases_of(split):
        gold = gold_for_case(case, truth[case["case_id"]])
        n = len(case["events"])
        poss = possible_final_states(case["events"][:n - 1], case["events"][n - 1:])
        for q in case["queries"]:
            vals = sorted({answer_from_live(live_from_state(st, case["factors"]), q) for st in poss})
            vals = sorted(set(vals))
            res["pending_decisions"] += 1
            if len(vals) == 1:
                res["identifiable"] += 1
                res["exact_visible_only_accuracy"] += (vals[0] == gold[n][q["id"]])
            else:
                res["underdetermined"] += 1
                res["wrong_if_forced_to_first_state"] += (vals[0] != gold[n][q["id"]])
    res["exact_visible_only_accuracy"] = round(
        res["exact_visible_only_accuracy"] / max(1, res["pending_decisions"]), 6)
    return res

out["regime_c_abstention"] = {"TRAIN": abstention(tr, truth_tr), "TUNE": abstention(tu, truth_tu)}

# ---------- 5. Learned model vs exact engine --------------------------------
learned_acc = 0; n = 0
for case in cases_of(tu):
    gold = gold_for_case(case, truth_tu[case["case_id"]])
    ni = nodes_of(case)
    for q in case["queries"]:
        for pref in range(len(case["events"]) + 1):
            edges = edge_arrays(case, pref, ni, q)
            pred = net.predict(len(ni), ni[q["from"]], ni[q["to"]], edges)[0]
            n += 1
            learned_acc += (pred == gold[pref][q["id"]])
out["learned_vs_exact_tune"] = {"decisions": n, "learned_accuracy": round(learned_acc / n, 6),
                                "train_seconds": round(train_secs, 2),
                                "param_bytes": net.param_bytes()}

inc_acc = 0; n2 = 0; disagree = 0
for case in cases_of(tu):
    gold = gold_for_case(case, truth_tu[case["case_id"]])
    ni = nodes_of(case)
    eng = Incremental(case)
    for i, e in enumerate(case["events"]):
        eng.apply(e)
        snap = eng.snapshot()
        for q in case["queries"]:
            edges = edge_arrays(case, i + 1, ni, q)
            pred = net.predict(len(ni), ni[q["from"]], ni[q["to"]], edges)[0]
            n2 += 1
            inc_acc += (snap[q["id"]] == gold[i + 1][q["id"]])
            disagree += (pred != snap[q["id"]])
out["learned_vs_incremental"] = {"decisions": n2,
                                 "incremental_accuracy": round(inc_acc / n2, 6),
                                 "learned_incremental_disagreements": disagree}

print(json.dumps(out, indent=1))
json.dump(out, open("runs/analysis_decisive.json", "w"), indent=1)
