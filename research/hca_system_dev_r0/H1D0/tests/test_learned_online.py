"""Regime-B test for the learned online comparator.

Invariants tested:
  1. The stateful online engine must agree with the batch model at EVERY prefix
     (incremental propagation is not allowed to drift).
  2. Damage to retained state must produce an observable change in answers.
  3. The declared repair path must restore exact agreement.
  4. A damaged engine that is NOT repaired must be measurably worse.
"""
import random
from generator import build_split
from oracle import gold_for_case
from learned import FactorNet, edge_arrays
from learned_online import LearnedOnline

TRAIN_SEED = 12345
TUNE_SEED = 987654
UNITS = 6


def nodes_of(case):
    toks = sorted({v for f in case["factors"] for v in f["roles"].values()}
                  | {q["from"] for q in case["queries"]} | {q["to"] for q in case["queries"]})
    return {t: i for i, t in enumerate(toks)}


def samples(ds, truth):
    out = []
    for w in ds["worlds"]:
        for c in w["cases"]:
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


def main():
    tr, trt = build_split("TRAIN", TRAIN_SEED, 2)
    tu, tut = build_split("TUNE", TUNE_SEED, 2)
    net = FactorNet(rounds=3, seed=20261023)
    net.fit(samples(tr, trt), epochs=40, lr=0.05)

    drift = 0
    total = 0
    footprints = []
    damaged_correct = 0
    repaired_correct = 0
    damaged_total = 0
    cases = [c for w in tu["worlds"] for c in w["cases"]]
    for c in cases:
        gold = gold_for_case(c, tut[c["case_id"]])
        ni = nodes_of(c)
        eng = LearnedOnline(net, c)
        for i, e in enumerate(c["events"]):
            eng.apply(e)
            snap = eng.snapshot()
            for q in c["queries"]:
                total += 1
                batch = net.predict(len(ni), ni[q["from"]], ni[q["to"]],
                                    edge_arrays(c, i + 1, ni, q))[0]
                if snap[q["id"]] != batch:
                    drift += 1
                if snap[q["id"]] != gold[i + 1][q["id"]]:
                    drift += 1
        footprints.append(eng.footprint())

        k = c["damage_prefix"]
        eng2 = LearnedOnline(net, c)
        for e in c["events"][:k]:
            eng2.apply(e)
        units = eng2.damage(random.Random(c["case_id"]), UNITS)
        if units == 0:
            raise AssertionError("damage routine corrupted nothing")
        for e in c["events"][k:]:
            eng2.apply(e)
        snap2 = eng2.snapshot()
        for q in c["queries"]:
            damaged_total += 1
            damaged_correct += snap2[q["id"]] == gold[len(c["events"])][q["id"]]
        eng2.repair()
        snap3 = eng2.snapshot()
        for q in c["queries"]:
            repaired_correct += snap3[q["id"]] == gold[len(c["events"])][q["id"]]

    print("prefix/query decisions checked:", total)
    print("online-vs-batch / online-vs-gold drift:", drift)
    print("mean retained-state footprint bytes:", round(sum(footprints) / len(footprints), 1))
    print("damaged (no repair):", damaged_correct, "/", damaged_total,
          "=", round(damaged_correct / damaged_total, 4))
    print("damaged + repair:", repaired_correct, "/", damaged_total,
          "=", round(repaired_correct / damaged_total, 4))
    if drift:
        raise AssertionError("incremental learned propagation diverged from the batch model")
    if damaged_correct >= damaged_total:
        raise AssertionError("damage had no observable effect on retained state")
    if repaired_correct < damaged_total:
        raise AssertionError("repair did not restore exact state")
    print("LEARNED ONLINE REGIME-B: PASS")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
