"""H1D0 adversarial BREAK suite.

Every test asserts a FORBIDDEN behaviour is actually detected, not merely that
some exception was raised.  Catch-all handlers are not used.
"""
import copy
import json
import random
import sys
from hashlib import sha256

import numpy as np

sys.path.insert(0, "src")
from contract import (CONTRACT_VERSION, SCHEMA, PUBLIC_FIELDS, MAX_EVENTS, MAX_FACTORS,
                      MAX_PENDING, DAMAGE_UNITS)
from generator import build_split, make_pair, withhold, token
from oracle import (gold_for_case, prefix_labels, identifiability,
                   possible_final_states, answer_from_state)
from checker import (validate_dataset, check_pairs, nuisance_ablation, ContractViolation,
                     checker_local_labels)
from engines import Replay, Incremental, CheckpointReplay, HcaNativeSemantics
from learned import FactorNet, edge_arrays
from learned_online import LearnedOnline

TRAIN_SEED, TUNE_SEED = 12345, 67890
TW, TUW = 4, 2
TR, TR_TRUTH = build_split("TRAIN", TRAIN_SEED, TW)
TU, TU_TRUTH = build_split("TUNE", TUNE_SEED, TUW)
TRC, TRC_TRUTH = build_split("TRAIN", TRAIN_SEED, TW, pending_prob=0.25)
TU_C, TU_C_TRUTH = build_split("TUNE", TUNE_SEED, TUW, pending_prob=0.25)


def cases_of(ds):
    return [c for w in ds["worlds"] for c in w["cases"]]


def nodes_of(case):
    toks = sorted({v for f in case["factors"] for v in f["roles"].values()}
                  | {q["from"] for q in case["queries"]} | {q["to"] for q in case["queries"]})
    return {t: i for i, t in enumerate(toks)}


# ---------------------------------------------------------------- contract
def test_contract_freeze_is_deterministic():
    a, ta = build_split("TRAIN", TRAIN_SEED, TW)
    b, tb = build_split("TRAIN", TRAIN_SEED, TW)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert json.dumps(ta, sort_keys=True) == json.dumps(tb, sort_keys=True)
    assert CONTRACT_VERSION == "H1D0_SOURCE_EVOLUTION_R0/v1"
    assert SCHEMA == "elpis.hca.h1d0.source_evolution.public.v1"


def test_no_sealed_split_exists():
    try:
        build_split("SEALED", 1, 1)
    except ValueError as exc:
        assert "sealed" in str(exc).lower()
        return
    raise AssertionError("a SEALED split generator exists: leakage path")


def test_dataset_validation_passes_and_denominators_are_exact():
    _, _, st_tr = validate_dataset(TR, TR_TRUTH, "TRAIN")
    _, _, st_tu = validate_dataset(TU, TU_TRUTH, "TUNE")
    # denominators must be exactly the sum over cases of (prefixes x queries)
    def manual(ds):
        rows = events = pending = positives = 0
        for c in cases_of(ds):
            rows += (len(c["events"]) + 1) * len(c["queries"])
            events += len(c["events"])
            pending += len(c["pending"])
            positives += sum(sum(r.values()) for r in gold_for_case(c, c["events"]))
        return rows, events, pending, positives
    m_tr = manual(TR)
    m_tu = manual(TU)
    assert st_tr["cases"] == 96 and st_tr["rows"] == m_tr[0]
    assert st_tu["cases"] == 48 and st_tu["rows"] == m_tu[0]
    assert st_tr["positives"] == m_tr[3] == 4096
    assert st_tr["events"] == m_tr[1] <= 96 * MAX_EVENTS
    assert st_tr["pending"] == m_tr[2] == 0
    # every case carries exactly 8 queries and a full prefix sweep
    for c in cases_of(TR) + cases_of(TU):
        assert len(c["queries"]) == 8
        assert len(gold_for_case(c, c["events"])) == len(c["events"]) + 1


def test_split_disjointness():
    tr_ids = {c["case_id"] for c in cases_of(TR)}
    tu_ids = {c["case_id"] for c in cases_of(TU)}
    assert not (tr_ids & tu_ids)
    tr_worlds = {w["world_id"] for w in TR["worlds"]}
    tu_worlds = {w["world_id"] for w in TU["worlds"]}
    assert not (tr_worlds & tu_worlds)


def test_truth_log_is_not_in_the_public_dataset():
    for c in cases_of(TR) + cases_of(TU):
        assert set(c) == set(PUBLIC_FIELDS), "public case carries extra fields"
        for f in c:
            assert not any(k in f.lower() for k in ("gold", "truth", "oracle", "label", "sealed"))
        # no nested object inside a public case may carry a 0/1 answer vector
        for v in c.values():
            if isinstance(v, dict):
                assert not all(x in (0, 1) for x in v.values()) or len(v) != len(c["queries"])
    # the truth log is a separate object keyed by case id
    assert set(TR_TRUTH) == {c["case_id"] for c in cases_of(TR)}
    assert set(TU_TRUTH) == {c["case_id"] for c in cases_of(TU)}


# ---------------------------------------------------------------- oracle vs checker
def test_oracle_and_checker_agree_on_every_decision():
    for ds, truth in ((TR, TR_TRUTH), (TU, TU_TRUTH)):
        for c in cases_of(ds):
            g = gold_for_case(c, truth[c["case_id"]])
            cl = checker_local_labels(truth[c["case_id"]], c["factors"], c["queries"])
            assert len(cl) == len(g)
            for n, lab in enumerate(cl):
                for q in c["queries"]:
                    assert g[n][q["id"]] == lab[q["id"]], (c["case_id"], n, q["id"])


def test_pair_interventions_always_flip_the_target():
    pairs = {}
    for w in TR["worlds"]:
        for c in w["cases"]:
            pairs.setdefault((w["world_id"], c["pair_id"]), []).append(
                (c, gold_for_case(c, TR_TRUTH[c["case_id"]]), c["factors"]))
    stats = check_pairs(pairs, {k: 8 for k in ("source_swap", "event_order", "epoch_rebind",
                                              "structural", "cancellation", "provenance")})
    for kind in ("source_swap", "event_order", "epoch_rebind", "structural",
                 "cancellation", "provenance"):
        assert stats["total_" + kind] == 8
        assert stats["both_" + kind] == 8


def test_nuisance_ablation_is_neutral():
    for c in cases_of(TR):
        assert nuisance_ablation(c) is True


def test_budgets_are_respected():
    for c in cases_of(TR):
        assert len(c["events"]) <= MAX_EVENTS
        assert len(c["factors"]) <= MAX_FACTORS
        assert len(c["pending"]) <= MAX_PENDING


# ---------------------------------------------------------------- exact engines
def test_every_exact_engine_matches_the_independent_oracle():
    for name, ctor in (("replay", Replay), ("incremental", Incremental),
                       ("checkpoint", lambda c: CheckpointReplay(c, every=8)),
                       ("hca_b0", lambda c: HcaNativeSemantics(c, method="B0")),
                       ("hca_b1", lambda c: HcaNativeSemantics(c, method="B1"))):
        mism = 0
        decisions = 0
        for c in cases_of(TR):
            gold = gold_for_case(c, TR_TRUTH[c["case_id"]])
            eng = ctor(c)
            for i, e in enumerate(c["events"]):
                eng.apply(e)
                snap = eng.snapshot()
                for q in c["queries"]:
                    decisions += 1
                    mism += (snap[q["id"]] != gold[i + 1][q["id"]])
        assert mism == 0, f"{name} disagrees with the oracle on {mism}/{decisions} decisions"
        expected = sum(len(c["events"]) for c in cases_of(TR)) * 8
        assert decisions == expected, (decisions, expected)


def test_incremental_and_replay_agree_at_every_prefix_not_only_the_final_one():
    bad = 0
    for c in cases_of(TR):
        r = Replay(c)
        inc = Incremental(c)
        for e in c["events"]:
            r.apply(e)
            inc.apply(e)
            if r.snapshot() != inc.snapshot():
                bad += 1
    assert bad == 0


def test_replay_is_deterministic_under_repeated_runs():
    for c in cases_of(TR)[:20]:
        outs = []
        for _ in range(3):
            r = Replay(c)
            for e in c["events"]:
                r.apply(e)
            outs.append(json.dumps(r.snapshot(), sort_keys=True))
        assert len(set(outs)) == 1


def test_b2_is_not_exact_and_the_suite_knows_it():
    """The unlearned asynchronous recurrence is NOT an exact solver.  Asserting
    this is a guard against ever reporting B2 as a correctness baseline."""
    mism = 0
    for c in cases_of(TR):
        gold = gold_for_case(c, TR_TRUTH[c["case_id"]])
        eng = HcaNativeSemantics(c, method="B2")
        for i, e in enumerate(c["events"]):
            eng.apply(e)
            snap = eng.snapshot()
            for q in c["queries"]:
                mism += (snap[q["id"]] != gold[i + 1][q["id"]])
    assert mism > 0, "B2 unexpectedly became exact; the cost comparison would need re-derivation"


# ---------------------------------------------------------------- leakage / privilege
def test_engines_cannot_read_truth():
    """AST-level: no engine may name, receive, or read a truth/gold/label object."""
    import ast
    for fname in ("engines.py", "learned.py", "learned_online.py", "baselines.py"):
        tree = ast.parse(open("src/" + fname).read())
        bad = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id.lower() in ("truth", "gold", "oracle", "sealed"):
                bad.add(node.id)
            if isinstance(node, ast.Attribute) and node.attr.lower() in ("truth", "gold", "oracle", "sealed", "labels"):
                bad.add(node.attr)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.lower() in ("truth", "gold", "oracle", "sealed", "labels"):
                    bad.add(node.value)
            if isinstance(node, ast.keyword) and node.arg and node.arg.lower() in ("truth", "gold", "labels"):
                bad.add(node.arg)
        assert not bad, f"{fname} references forbidden identifiers: {bad}"
    # every engine constructor takes the public case and no truth-bearing argument
    for ctor in (Replay, Incremental, CheckpointReplay, HcaNativeSemantics, LearnedOnline):
        code = ctor.__init__.__code__
        params = list(code.co_varnames[:code.co_argcount])
        assert "case" in params, (ctor.__name__, params)
        assert not any(p.lower() in ("truth", "gold", "labels", "oracle") for p in params)


def test_learned_model_sees_no_more_than_the_incremental_engine():
    """Both receive exactly the public case.  Neither receives the truth log."""
    c = cases_of(TR)[0]
    ni = nodes_of(c)
    eng = Incremental(c)
    net = FactorNet(rounds=3, seed=1)
    for i, e in enumerate(c["events"]):
        eng.apply(e)
        for q in c["queries"]:
            net.predict(len(ni), ni[q["from"]], ni[q["to"]], edge_arrays(c, i + 1, ni, q))
    # the only inputs are case events/factors/queries; nothing else exists
    assert set(c) == set(PUBLIC_FIELDS)


# ---------------------------------------------------------------- regime B
def test_damage_is_detected_by_the_integrity_checked_incremental_engine():
    recovered = 0
    total = 0
    for c in cases_of(TR):
        gold = gold_for_case(c, TR_TRUTH[c["case_id"]])
        k = c["damage_prefix"]
        rng = random.Random(c["case_id"])
        eng = Incremental(c, integrity=True)
        for e in c["events"][:k]:
            eng.apply(e)
        damaged = eng.damage(rng, DAMAGE_UNITS)
        if damaged == 0:
            continue
        total += 1
        for e in c["events"][k:]:
            eng.apply(e)
        snap = eng.snapshot()
        ok = all(snap[q["id"]] == gold[len(c["events"])][q["id"]] for q in c["queries"])
        recovered += ok
        assert eng.rebuilds > 0, "damage was silently absorbed without an integrity rebuild"
    assert total > 0
    assert recovered == total


def test_undetected_damage_produces_a_measurable_wrong_answer():
    wrong = 0
    checked = 0
    for c in cases_of(TR):
        gold = gold_for_case(c, TR_TRUTH[c["case_id"]])
        k = c["damage_prefix"]
        rng = random.Random(c["case_id"])
        eng = Incremental(c)
        for e in c["events"][:k]:
            eng.apply(e)
        if eng.damage(rng, DAMAGE_UNITS) == 0:
            continue
        snap = eng.snapshot()
        for q in c["queries"]:
            checked += 1
            wrong += (snap[q["id"]] != gold[k][q["id"]])
    assert checked > 0 and wrong > 0, "damage had no observable effect: the test proves nothing"


def test_learned_online_recovers_only_after_an_explicit_repair():
    net = FactorNet(rounds=3, seed=20261023)
    S = []
    for c in cases_of(TR):
        gold = gold_for_case(c, TR_TRUTH[c["case_id"]])
        ni = nodes_of(c)
        for n in range(len(c["events"]) + 1):
            for q in c["queries"]:
                S.append((len(ni), ni[q["from"]], ni[q["to"]],
                          edge_arrays(c, n, ni, q), gold[n][q["id"]]))
    net.fit(S, epochs=40, lr=0.05)
    self_ok = 0
    repaired_ok = 0
    total = 0
    for c in cases_of(TR):
        gold = gold_for_case(c, TR_TRUTH[c["case_id"]])
        k = c["damage_prefix"]
        rng = random.Random(c["case_id"])
        eng = LearnedOnline(net, c)
        for e in c["events"][:k]:
            eng.apply(e)
        eng.damage(rng, DAMAGE_UNITS)
        for e in c["events"][k:]:
            eng.apply(e)
        snap = eng.snapshot()
        for q in c["queries"]:
            total += 1
            self_ok += (snap[q["id"]] == gold[len(c["events"])][q["id"]])
        rng2 = random.Random(c["case_id"])
        eng2 = LearnedOnline(net, c)
        for e in c["events"][:k]:
            eng2.apply(e)
        eng2.damage(rng2, DAMAGE_UNITS)
        eng2.repair()
        for e in c["events"][k:]:
            eng2.apply(e)
        snap = eng2.snapshot()
        for q in c["queries"]:
            repaired_ok += (snap[q["id"]] == gold[len(c["events"])][q["id"]])
    assert repaired_ok == total
    assert self_ok < total, "the learned recurrence recovered without repair: the regime is not adversarial"


def test_replay_has_no_derived_state_to_damage():
    for c in cases_of(TR)[:20]:
        r = Replay(c)
        for e in c["events"]:
            r.apply(e)
        before = r.snapshot()
        assert r.damage(random.Random(1), 10) == 0
        assert r.snapshot() == before


# ---------------------------------------------------------------- regime C
def test_underdetermined_decisions_are_labelled_not_scored():
    ident = {"IDENTIFIABLE": 0, "UNDERDETERMINED": 0}
    for c in cases_of(TRC):
        info = identifiability(c)
        if info["status"] != "OK":
            continue
        for st in info["queries"].values():
            ident[st] += 1
    assert ident["UNDERDETERMINED"] > 0, "regime C contains no genuinely unknowable decision"
    assert ident["IDENTIFIABLE"] > 0


def test_identifiable_decisions_are_identifiable_by_construction():
    for c in cases_of(TRC):
        info = identifiability(c)
        if info["status"] != "OK":
            continue
        for q in c["queries"]:
            states = possible_final_states(c["events"], c["pending"])
            vals = {answer_from_state(st, c["factors"], q) for st in states}
            if info["queries"][q["id"]] == "IDENTIFIABLE":
                assert len(vals) == 1
            else:
                assert len(vals) > 1


# ---------------------------------------------------------------- learned model
def test_learned_gradient_is_exact():
    net = FactorNet(rounds=3, seed=7)
    c = cases_of(TR)[0]
    ni = nodes_of(c)
    q = c["queries"][0]
    edges = edge_arrays(c, len(c["events"]), ni, q)
    nc = len(ni)
    logit, _vs, grads = net.forward(nc, ni[q["from"]], ni[q["to"]], edges, want_grad=True)
    analytic = np.zeros_like(net.t)
    gv = np.zeros(nc)
    gv[ni[q["to"]]] = 1.0
    for r in range(net.rounds - 1, -1, -1):
        a, b, cc, weight = grads[r]
        np.add.at(analytic[r], cc, gv[b] * weight)
        nxt = np.zeros(nc)
        np.add.at(nxt, a, gv[b] * weight)
        gv = nxt
    eps = 1e-7
    numeric = np.zeros_like(net.t)
    for r in range(net.rounds):
        for k in range(net.t.shape[1]):
            net.t[r][k] += eps
            hi = net.forward(nc, ni[q["from"]], ni[q["to"]], edges)
            net.t[r][k] -= 2 * eps
            lo = net.forward(nc, ni[q["from"]], ni[q["to"]], edges)
            net.t[r][k] += eps
            numeric[r][k] = (hi - lo) / (2 * eps)
    assert np.max(np.abs(analytic - numeric)) < 1e-6


def test_learned_gradient_negative_control():
    net = FactorNet(rounds=3, seed=7)
    c = cases_of(TR)[0]
    ni = nodes_of(c)
    q = c["queries"][0]
    edges = edge_arrays(c, len(c["events"]), ni, q)
    nc = len(ni)
    logit, _vs, grads = net.forward(nc, ni[q["from"]], ni[q["to"]], edges, want_grad=True)
    analytic = np.zeros_like(net.t)
    gv = np.zeros(nc)
    gv[ni[q["to"]]] = 1.0
    for r in range(net.rounds - 1, -1, -1):
        a, b, cc, weight = grads[r]
        np.add.at(analytic[r], cc, gv[b] * weight)
        nxt = np.zeros(nc)
        np.add.at(nxt, a, gv[b] * weight)
        gv = nxt
    perturbed = analytic.copy()
    perturbed[0][0] *= 2.0
    assert np.max(np.abs(perturbed - analytic)) > 0.0


def test_learned_online_matches_the_batch_model_on_clean_data():
    net = FactorNet(rounds=3, seed=20261023)
    S = []
    for c in cases_of(TR):
        gold = gold_for_case(c, TR_TRUTH[c["case_id"]])
        ni = nodes_of(c)
        for n in range(len(c["events"]) + 1):
            for q in c["queries"]:
                S.append((len(ni), ni[q["from"]], ni[q["to"]],
                          edge_arrays(c, n, ni, q), gold[n][q["id"]]))
    net.fit(S, epochs=40, lr=0.05)
    disagree = 0
    decisions = 0
    for c in cases_of(TU):
        ni = nodes_of(c)
        eng = LearnedOnline(net, c)
        for i, e in enumerate(c["events"]):
            eng.apply(e)
            snap = eng.snapshot()
            for q in c["queries"]:
                pred = net.predict(len(ni), ni[q["from"]], ni[q["to"]],
                                   edge_arrays(c, i + 1, ni, q))[0]
                decisions += 1
                disagree += (snap[q["id"]] != pred)
    assert decisions > 0
    assert disagree == 0, "the online recurrence diverges from the batch model"


# ---------------------------------------------------------------- cost accounting
def test_footprint_is_measured_from_retained_state_not_transient_allocation():
    for c in cases_of(TR)[:10]:
        for ctor in (Replay, Incremental, lambda x: CheckpointReplay(x, every=8),
                     lambda x: HcaNativeSemantics(x, method="B2")):
            eng = ctor(c)
            for e in c["events"]:
                eng.apply(e)
            fp = eng.footprint()
            assert isinstance(fp, int) and fp > 0
            assert eng.work, "no logical-work counter recorded"


def test_logical_work_and_wall_time_are_reported_separately():
    rep = json.load(open("runs/20261023T195041Z_H1D0/H1D0_REPORT.json"))
    assert "work_counters" in rep["exact_engines"]["TRAIN"]
    assert "mean_step_ms" in rep["exact_engines"]["TRAIN"]
    assert "peak_rss_kib" in rep["costs"]
    assert "footprint_bytes" in rep["costs"]["incremental"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
