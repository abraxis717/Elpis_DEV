"""H1D0 adversarial mutation tests.

Every mutation test PASSES only if the forbidden behaviour is actually
DETECTED.  A test that merely raises some exception is not a pass: the
exception must be the specific rejection the invariant demands, and a
semantic mutant must produce a DIFFERENT answer than the contract on at
least one case in the corpus.  No catch-all handler is used.
"""
import copy
import random
from collections import Counter

from contract import SCHEMA
from generator import build_split
from oracle import answer, prefix_labels
from checker import (ContractViolation, checker_local_labels, check_pairs,
                     nuisance_ablation, validate_dataset)
from engines import Incremental

TRAIN_SEED = 12345
TUNE_SEED = 987654

DS, TRUTH = build_split("TRAIN", TRAIN_SEED, 2)
CASES = [c for w in DS["worlds"] for c in w["cases"]]


def expect_reject(name, fn, exc=ContractViolation, want=None):
    """fn MUST raise exc (and, if `want` is given, the message must contain it)."""
    try:
        result = fn()
    except exc as e:
        if want is not None and want not in str(e):
            raise AssertionError(f"{name}: rejected for the wrong reason: {e}")
        return str(e)
    except Exception as e:
        raise AssertionError(f"{name}: wrong exception type {type(e).__name__}: {e}")
    raise AssertionError(f"{name}: forbidden behaviour NOT rejected (returned {result})")


def count_diff(fn):
    """Apply fn(case) -> gold; count cases where it differs from the contract gold."""
    n = 0
    for c in CASES:
        gold = prefix_labels(c["events"], c["factors"], c["queries"])
        alt = fn(c)
        if alt != gold:
            n += 1
    return n


# ---------------------------------------------------------------------------
# 1. stale-source acceptance
# ---------------------------------------------------------------------------
def live_of(events, factors, n, accept_stale=False, ignore_epoch=False):
    status, present, src = {}, {}, {}
    for e in events[:n]:
        t = e["type"]
        if t == "SOURCE":
            status[e["source"]] = (e["status"], e["epoch"])
        elif t == "EMIT":
            present[e["factor"]] = True
        elif t == "DROP":
            present[e["factor"]] = False
        else:
            src[e["factor"]] = e["source"]
    live = []
    for f in factors:
        if not present.get(f["id"], False):
            continue
        st = status.get(src.get(f["id"], f["source"]))
        if st is None:
            continue
        if st[0] == "VALID" and (ignore_epoch or st[1] == f["version"]):
            live.append(f)
        elif accept_stale:
            live.append(f)
    return live


def gold_with(live_fn):
    def run(case):
        rows = []
        for n in range(len(case["events"]) + 1):
            live = live_fn(case["events"], case["factors"], n)
            row = {}
            for q in case["queries"]:
                s1 = {f["roles"]["to"] for f in live if f["rel"] == "R0"
                      and f["roles"]["from"] == q["from"]
                      and f["roles"]["scope"] == q["scope"] and f["roles"]["kind"] == q["kind"]}
                s2 = {f["roles"]["to"] for f in live if f["rel"] == "R1" and f["roles"]["from"] in s1
                      and f["roles"]["scope"] == q["scope"] and f["roles"]["kind"] == q["kind"]}
                row[q["id"]] = int(any(f["rel"] == "R2" and f["roles"]["from"] in s2
                                       and f["roles"]["to"] == q["to"]
                                       and f["roles"]["scope"] == q["scope"]
                                       and f["roles"]["kind"] == q["kind"] for f in live))
            rows.append(row)
        return rows
    return run


def test_stale_acceptance():
    n = count_diff(gold_with(lambda e, f, k: live_of(e, f, k, accept_stale=True)))
    if n == 0:
        raise AssertionError("accepting STALE sources changes nothing: the corpus does not "
                             "exercise stale-source invalidation")
    return f"{n}/{len(CASES)} cases distinguish STALE from VALID"


def test_epoch_ignored():
    n = count_diff(gold_with(lambda e, f, k: live_of(e, f, k, ignore_epoch=True)))
    if n == 0:
        raise AssertionError("ignoring the source epoch changes nothing: epoch/version "
                             "consistency is not exercised")
    return f"{n}/{len(CASES)} cases distinguish epoch mismatch"


def test_relation_type_ignored():
    """Treat RX as if it were a real relation type (collapse the typed hypergraph)."""
    def run(case):
        promoted = [{**f, "rel": "R2"} if f["rel"] == "RX" else f for f in case["factors"]]
        return prefix_labels(case["events"], promoted, case["queries"])
    n = count_diff(run)
    if n == 0:
        raise AssertionError("RX decoys are inert: relation typing is not load-bearing")
    return f"{n}/{len(CASES)} cases are fooled by RX decoys"


def test_reversed_causal_order():
    def run(case):
        rev = list(reversed(case["events"]))
        return [{q["id"]: answer(rev, case["factors"], q, len(rev)) for q in case["queries"]}]
    n = 0
    for c in CASES:
        gold = prefix_labels(c["events"], c["factors"], c["queries"])[-1]
        rev = [{q["id"]: answer(list(reversed(c["events"])), c["factors"], q, len(c["events"]))
                for q in c["queries"]}]
        if gold != rev:
            n += 1
    if n == 0:
        raise AssertionError("log order is not causally load-bearing in this corpus")
    return f"{n}/{len(CASES)} cases have order-dependent final gold"


def test_provenance_forgery():
    """A factor re-bound to a source that was never declared VALID must not be live."""
    forged = 0
    for c in CASES:
        q = next(x for x in c["queries"] if x["id"] == c["target_query_id"])
        if answer(c["events"], c["factors"], q, len(c["events"])) != 1:
            continue
        r0 = next(f for f in c["factors"] if f["rel"] == "R0" and f["roles"]["from"] == q["from"]
                  and f["roles"]["scope"] == q["scope"] and f["roles"]["kind"] == q["kind"])
        r1 = next(f for f in c["factors"] if f["rel"] == "R1" and f["roles"]["from"] == r0["roles"]["to"]
                  and f["roles"]["scope"] == q["scope"] and f["roles"]["kind"] == q["kind"])
        ghost = "z" + "f" * 28
        factors = [dict(f) for f in c["factors"]]
        for f in factors:
            if f["id"] == r1["id"]:
                f["source"] = ghost
        if answer(c["events"], factors, q, len(c["events"])) == 1:
            raise AssertionError("forged provenance still admitted the witness")
        forged += 1
    if forged == 0:
        raise AssertionError("no live witness available to forge against")
    return f"{forged} witnesses correctly invalidated by provenance forgery"


def test_duplicate_identities():
    c = copy.deepcopy(CASES[0])
    c["factors"].append(copy.deepcopy(c["factors"][0]))
    msg = expect_reject("duplicate factor id",
                        lambda: validate_dataset({"schema": SCHEMA, "split": "TRAIN",
                                                 "worlds": [{"world_id": "w", "cases": [c]}]},
                                                {c["case_id"]: c["events"]}, "TRAIN"),
                        want="duplicate factor id")
    c2 = copy.deepcopy(CASES[0])
    c2["queries"][1] = copy.deepcopy(c2["queries"][0])
    expect_reject("duplicate query id",
                  lambda: validate_dataset({"schema": SCHEMA, "split": "TRAIN",
                                           "worlds": [{"world_id": "w", "cases": [c2]}]},
                                          {c2["case_id"]: c2["events"]}, "TRAIN"),
                  want="duplicate query id")
    expect_reject("duplicate world id",
                  lambda: validate_dataset({"schema": SCHEMA, "split": "TRAIN",
                                           "worlds": [{"world_id": "w", "cases": [CASES[0]]},
                                                     {"world_id": "w", "cases": [CASES[0]]}]},
                                          {CASES[0]["case_id"]: CASES[0]["events"]}, "TRAIN"),
                  want="duplicate world id")
    return msg


def test_target_leak():
    c = copy.deepcopy(CASES[0])
    c["gold_target"] = checker_local_labels(c["events"], c["factors"], c["queries"])[-1][c["target_query_id"]]
    return expect_reject("label leakage field",
                         lambda: validate_dataset({"schema": SCHEMA, "split": "TRAIN",
                                                  "worlds": [{"world_id": "w", "cases": [c]}]},
                                                 {c["case_id"]: c["events"]}, "TRAIN"),
                         want="unexpected field")


def test_pair_without_contrast():
    by_pair = {}
    for w in DS["worlds"]:
        for c in w["cases"]:
            by_pair.setdefault(c["pair_id"], []).append(c)
    a, b = by_pair[next(iter(by_pair))]
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    tgt = a["target_query_id"]
    ga = checker_local_labels(a["events"], a["factors"], a["queries"])[-1][tgt]
    gb = checker_local_labels(b["events"], b["factors"], b["queries"])[-1][tgt]
    if ga == gb:
        raise AssertionError("precondition: pair already has identical gold")
    b["events"] = copy.deepcopy(a["events"])
    b["factors"] = copy.deepcopy(a["factors"])
    pairs = {("w", a["pair_id"]): [(a, checker_local_labels(a["events"], a["factors"], a["queries"]), a["events"]),
                                   (b, checker_local_labels(b["events"], b["factors"], b["queries"]), b["events"])]}
    return expect_reject("pair without contrast",
                         lambda: check_pairs(pairs, {a["kind"]: 1}),
                         want="opposite gold")


def test_nuisance_independence():
    for c in CASES:
        if not nuisance_ablation(c):
            raise AssertionError("nuisance ablation failed")
    # negative control: renaming ONE role binding inconsistently must change the gold
    c = copy.deepcopy(CASES[0])
    tq = next(x for x in c["queries"] if x["id"] == c["target_query_id"])
    for f in c["factors"]:
        if f["rel"] == "R0" and f["roles"]["from"] == tq["from"] and f["roles"]["scope"] == tq["scope"]:
            f["roles"]["scope"] = "z" + "1" * 28
            break
    g1 = checker_local_labels(CASES[0]["events"], CASES[0]["factors"], CASES[0]["queries"])[-1][tq["id"]]
    g2 = checker_local_labels(c["events"], c["factors"], c["queries"])[-1][tq["id"]]
    if g1 == g2:
        raise AssertionError("scope binding is not semantically load-bearing")
    return f"token renaming is neutral on {len(CASES)} cases; single role rename flips gold"


def test_incremental_damage_detection():
    c = CASES[0]
    gold = prefix_labels(c["events"], c["factors"], c["queries"])
    eng = Incremental(c, integrity=True)
    for e in c["events"]:
        eng.apply(e)
    if eng.snapshot() != gold[-1]:
        raise AssertionError("clean incremental engine disagrees with oracle")
    rng = random.Random(7)
    damaged = eng.damage(rng, 4)
    if damaged == 0:
        raise AssertionError("damage routine corrupted nothing")
    if eng._checksum() == eng.check:
        raise AssertionError("damage did not change the retained-state checksum")
    eng.apply({"step": len(c["events"]) + 1, "type": "SOURCE",
               "source": c["factors"][0]["source"], "status": "VALID", "epoch": 0})
    if eng.rebuilds == 0:
        raise AssertionError("integrity check did not trigger a rebuild")
    if eng.snapshot() != gold[-1]:
        raise AssertionError("integrity rebuild failed to restore correct state")
    return f"{damaged} units corrupted, {eng.rebuilds} rebuild(s), state restored"


def test_damage_without_repair_is_caught():
    """A damaged incremental engine that is NOT repaired must produce a wrong answer."""
    c = CASES[0]
    gold = prefix_labels(c["events"], c["factors"], c["queries"])
    eng = Incremental(c, integrity=False)
    for e in c["events"]:
        eng.apply(e)
    rng = random.Random(7)
    eng.damage(rng, 4)
    snap = eng.snapshot()
    if snap == gold[-1]:
        raise AssertionError("corrupted retained state produced the correct answer: "
                             "damage is not observable, so the resilience experiment is void")
    return "damaged incremental engine demonstrably wrong"


def test_drop_semantics():
    n = 0
    for c in CASES:
        if not any(e["type"] == "DROP" for e in c["events"]):
            continue
        g1 = prefix_labels(c["events"], c["factors"], c["queries"])[-1]
        g2 = prefix_labels([e for e in c["events"] if e["type"] != "DROP"], c["factors"], c["queries"])[-1]
        if g1 != g2:
            n += 1
    if n == 0:
        raise AssertionError("DROP events are semantically inert")
    return f"{n} cases depend on DROP ordering"


def test_split_overlap_and_freeze():
    tr, _ = build_split("TRAIN", TRAIN_SEED, 2)
    tu, _ = build_split("TUNE", TUNE_SEED, 2)
    ids_tr = {c["case_id"] for w in tr["worlds"] for c in w["cases"]}
    ids_tu = {c["case_id"] for w in tu["worlds"] for c in w["cases"]}
    if ids_tr & ids_tu:
        raise AssertionError("TRAIN/TUNE case ids overlap")
    if {w["world_id"] for w in tr["worlds"]} & {w["world_id"] for w in tu["worlds"]}:
        raise AssertionError("TRAIN/TUNE worlds overlap")
    tr2, _ = build_split("TRAIN", TRAIN_SEED, 2)
    if tr != tr2:
        raise AssertionError("generator is not deterministic for a fixed seed")
    try:
        build_split("SEALED", 1, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("a SEALED split was generated")
    return "disjoint splits, deterministic freeze, no SEALED generator"


def test_denominator_accounting():
    stats = Counter()
    for c in CASES:
        gold = prefix_labels(c["events"], c["factors"], c["queries"])
        stats["rows"] += len(gold) * len(c["queries"])
        stats["positives"] += sum(sum(r.values()) for r in gold)
        stats["cases"] += 1
    if stats["rows"] != stats["cases"] * (len(CASES[0]["events"]) + 1) * 8:
        # event counts differ per case; recompute exactly
        exact = sum((len(c["events"]) + 1) * len(c["queries"]) for c in CASES)
        if stats["rows"] != exact:
            raise AssertionError("decision denominator mismatch")
    if stats["positives"] == 0 or stats["positives"] == stats["rows"]:
        raise AssertionError("degenerate label distribution")
    return f"{stats['rows']} decisions, {stats['positives']} positive"


def main():
    tests = [
        ("stale_source_acceptance", test_stale_acceptance),
        ("epoch_ignored", test_epoch_ignored),
        ("relation_type_ignored_rx", test_relation_type_ignored),
        ("reversed_causal_order", test_reversed_causal_order),
        ("provenance_forgery", test_provenance_forgery),
        ("duplicate_identities", test_duplicate_identities),
        ("target_label_leak", test_target_leak),
        ("pair_without_contrast", test_pair_without_contrast),
        ("nuisance_independence", test_nuisance_independence),
        ("incremental_damage_detection", test_incremental_damage_detection),
        ("damage_without_repair_caught", test_damage_without_repair_is_caught),
        ("drop_semantics", test_drop_semantics),
        ("split_overlap_and_freeze", test_split_overlap_and_freeze),
        ("denominator_accounting", test_denominator_accounting),
    ]
    results = {}
    for name, fn in tests:
        try:
            results[name] = ("PASS", fn())
        except AssertionError as e:
            results[name] = ("FAIL", str(e))
    for name, (status, detail) in results.items():
        print(f"{name}: {status} ({detail})")
    failed = [n for n, (s, _d) in results.items() if s == "FAIL"]
    print("MUTATION SUITE:", "PASS" if not failed else "FAIL", failed)
    return 0 if not failed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
