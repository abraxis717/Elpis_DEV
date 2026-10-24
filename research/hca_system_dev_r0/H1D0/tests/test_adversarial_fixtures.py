"""Hand-built adversarial fixtures for invariants the generated corpus does not
exercise on its own.

Findings that drove these fixtures:
  * no generated case places an RX factor in a position where it could complete
    a witness chain, so "RX is a decoy" was untested by corpus agreement;
  * a pending pool in the generated corpus rarely changes a decision, so
    underdetermination was only weakly exercised;
  * structural rejection (duplicate ids, unknown factor references, duplicate
    roles, non-monotonic steps) lives in validate_dataset and must be proven to
    actually reject, not merely to raise.

Every fixture is itself contract-valid apart from the single injected defect.
"""
import json
import random
import sys

sys.path.insert(0, "src")
from contract import SCHEMA
from oracle import answer, possible_final_states, answer_from_state
from checker import witness, validate_dataset, ContractViolation
from engines import Incremental, HcaNativeSemantics


def tok(i):
    return "e%d" % i


S0, S1, S2, S3 = tok(40), tok(41), tok(44), tok(45)
K0, K1 = tok(42), tok(43)


def f(fid, rel, a, b, scope, kind, src, ver):
    return {"id": fid, "rel": rel,
            "roles": {"from": a, "to": b, "scope": scope, "kind": kind},
            "source": src, "version": ver}


def q(qid, a, b, scope, kind):
    return {"id": qid, "from": a, "to": b, "scope": scope, "kind": kind}


def make_case(cid, kind, factors, queries, events, pending=(), prefix=None):
    """Build a contract-valid case with exactly 8 queries."""
    assert len(queries) == 8, "fixture must carry the declared query denominator"
    return {"case_id": cid, "pair_id": cid + "-p", "kind": kind,
            "target_query_id": queries[0]["id"], "factors": factors,
            "queries": queries, "events": events, "pending": list(pending),
            "damage_prefix": (len(events) // 2) if prefix is None else prefix}


def eight_queries(base_from, base_to):
    """Eight queries with distinct (scope, kind) pairs.

    Only the first pair is intended to match any factor chain, so a fixture
    built from this helper has exactly one witness path for queries[0].
    """
    scopes = [S0, S1, S2, S3]
    kinds = [K0, K1]
    return [q("q%d" % i, base_from, base_to, scopes[i // 2], kinds[i % 2])
            for i in range(8)]


def dataset(c):
    return {"schema": SCHEMA, "split": "TRAIN",
            "worlds": [{"world_id": "w1", "cases": [c]}]}


def expect_reject(c, needle):
    """validate_dataset must raise ContractViolation whose message names the defect."""
    try:
        validate_dataset(dataset(c), {c["case_id"]: c["events"]}, "TRAIN")
    except ContractViolation as e:
        msg = str(e)
        if needle not in msg:
            raise AssertionError("rejected for the wrong reason: %r (expected %r)"
                                 % (msg, needle))
        return msg
    raise AssertionError("defective case was accepted")


# ---------------------------------------------------------------- RX decoy ----
RX_FACTORS = [f("f0", "RX", tok(0), tok(1), S0, K0, "s1", 0),
              f("f1", "R1", tok(1), tok(2), S0, K0, "s2", 0),
              f("f2", "R2", tok(2), tok(3), S0, K0, "s3", 0),
              f("f3", "R0", tok(0), tok(5), S1, K1, "s1", 0),
              f("f4", "R1", tok(5), tok(6), S1, K1, "s2", 0),
              f("f5", "R2", tok(6), tok(7), S1, K1, "s3", 0)]
RX_EVENTS = [{"step": 1, "type": "SOURCE", "source": "s1", "status": "VALID", "epoch": 0},
             {"step": 2, "type": "SOURCE", "source": "s2", "status": "VALID", "epoch": 0},
             {"step": 3, "type": "SOURCE", "source": "s3", "status": "VALID", "epoch": 0},
             {"step": 4, "type": "EMIT", "factor": "f0"},
             {"step": 5, "type": "EMIT", "factor": "f1"},
             {"step": 6, "type": "EMIT", "factor": "f2"},
             {"step": 7, "type": "EMIT", "factor": "f3"},
             {"step": 8, "type": "EMIT", "factor": "f4"},
             {"step": 9, "type": "EMIT", "factor": "f5"}]
RX_CASE = make_case("fx-rx", "structural", RX_FACTORS,
                    eight_queries(tok(0), tok(3)), RX_EVENTS)


def test_rx_factor_completes_a_chain_but_is_never_admitted():
    """The chain is complete except that its first link is an RX factor.
    Every independent implementation must answer 0, and the same fixture with
    the rel changed to R0 must answer 1 - proving the fixture actually bites."""
    n = len(RX_CASE["events"])
    query = RX_CASE["queries"][0]
    assert answer(RX_CASE["events"], RX_CASE["factors"], query, n) == 0
    status = {"s1": ("VALID", 0), "s2": ("VALID", 0), "s3": ("VALID", 0)}
    present = {fid: True for fid in ("f0", "f1", "f2", "f3", "f4", "f5")}
    assert witness(RX_CASE["events"], RX_CASE["factors"], {}, status, present, query) == 0
    for method in ("B0", "B1", "B2"):
        eng = HcaNativeSemantics(RX_CASE, method=method)
        for e in RX_CASE["events"]:
            eng.apply(e)
        assert eng.snapshot()[query["id"]] == 0, (method, eng.snapshot())
    promoted = json.loads(json.dumps(RX_CASE))
    promoted["factors"][0]["rel"] = "R0"
    assert answer(promoted["events"], promoted["factors"], query, n) == 1


def test_rx_decoy_is_rejected_by_every_engine_at_every_prefix():
    for n in range(len(RX_CASE["events"]) + 1):
        for method in ("B0", "B1", "B2"):
            eng = HcaNativeSemantics(RX_CASE, method=method)
            for e in RX_CASE["events"][:n]:
                eng.apply(e)
            assert eng.snapshot()[RX_CASE["queries"][0]["id"]] == 0, (method, n)


# ------------------------------------------------------- genuine underdetermination
FLIP_CASE = make_case(
    "fx-flip", "provenance",
    [f("g0", "R0", tok(10), tok(11), S0, K0, "s1", 1),
     f("g1", "R1", tok(11), tok(12), S0, K0, "s2", 0),
     f("g2", "R2", tok(12), tok(13), S0, K0, "s3", 0),
     f("g3", "R0", tok(10), tok(14), S1, K1, "s1", 0),
     f("g4", "R1", tok(14), tok(15), S1, K1, "s2", 0),
     f("g5", "R2", tok(15), tok(16), S1, K1, "s3", 0)],
    eight_queries(tok(10), tok(13)),
    [{"step": 1, "type": "SOURCE", "source": "s1", "status": "STALE", "epoch": 0},
     {"step": 2, "type": "SOURCE", "source": "s2", "status": "VALID", "epoch": 0},
     {"step": 3, "type": "SOURCE", "source": "s3", "status": "VALID", "epoch": 0},
     {"step": 4, "type": "EMIT", "factor": "g0"},
     {"step": 5, "type": "EMIT", "factor": "g1"},
     {"step": 6, "type": "EMIT", "factor": "g2"},
     {"step": 7, "type": "EMIT", "factor": "g3"},
     {"step": 8, "type": "EMIT", "factor": "g4"},
     {"step": 9, "type": "EMIT", "factor": "g5"}],
    pending=[{"step": 10, "type": "SOURCE", "source": "s1", "status": "VALID", "epoch": 1}],
)


def test_pending_pool_can_genuinely_flip_a_decision():
    states = possible_final_states(FLIP_CASE["events"], FLIP_CASE["pending"])
    query = FLIP_CASE["queries"][0]
    vals = {answer_from_state(s, FLIP_CASE["factors"], query) for s in states}
    assert vals == {0, 1}, vals
    assert answer(FLIP_CASE["events"], FLIP_CASE["factors"], query,
                  len(FLIP_CASE["events"])) == 0


def brute_force_final_states(visible, pool):
    """Independent enumeration: every interleaving of the pool into the log."""
    from itertools import combinations
    n, k = len(visible), len(pool)
    out = set()
    for positions in combinations(range(n + k), k):
        merged = [None] * (n + k)
        for slot, pos in enumerate(positions):
            merged[pos] = pool[slot]
        vi = 0
        for i in range(n + k):
            if merged[i] is None:
                merged[i] = visible[vi]
                vi += 1
        assert all(m is not None for m in merged)
        status, present, src = {}, {}, {}
        for e in merged:
            t = e["type"]
            if t == "SOURCE":
                status[e["source"]] = (e["status"], e["epoch"])
            elif t == "EMIT":
                present[e["factor"]] = True
            elif t == "DROP":
                present[e["factor"]] = False
            else:
                src[e["factor"]] = e["source"]
        out.add(witness(merged, FLIP_CASE["factors"], src, status, present,
                        FLIP_CASE["queries"][0]))
    return out


def test_identifiability_enumeration_matches_brute_force_interleavings():
    """possible_final_states must agree with an exhaustive interleaving search."""
    fast = {answer_from_state(s, FLIP_CASE["factors"], FLIP_CASE["queries"][0])
            for s in possible_final_states(FLIP_CASE["events"], FLIP_CASE["pending"])}
    slow = brute_force_final_states(FLIP_CASE["events"], FLIP_CASE["pending"])
    assert fast == slow, (fast, slow)
    assert len(fast) > 1, "the fixture must actually be underdetermined"


# ------------------------------------------------------- observable state damage
COMPLETE_CASE = make_case(
    "fx-damage", "cancellation",
    [f("h0", "R0", tok(20), tok(21), S0, K0, "s1", 0),
     f("h1", "R1", tok(21), tok(22), S0, K0, "s2", 0),
     f("h2", "R2", tok(22), tok(23), S0, K0, "s3", 0),
     f("h3", "R0", tok(20), tok(24), S1, K1, "s1", 0),
     f("h4", "R1", tok(24), tok(25), S1, K1, "s2", 0),
     f("h5", "R2", tok(25), tok(26), S1, K1, "s3", 0)],
    eight_queries(tok(20), tok(23)),
    [{"step": 1, "type": "SOURCE", "source": "s1", "status": "VALID", "epoch": 0},
     {"step": 2, "type": "SOURCE", "source": "s2", "status": "VALID", "epoch": 0},
     {"step": 3, "type": "SOURCE", "source": "s3", "status": "VALID", "epoch": 0},
     {"step": 4, "type": "EMIT", "factor": "h0"},
     {"step": 5, "type": "EMIT", "factor": "h1"},
     {"step": 6, "type": "EMIT", "factor": "h2"},
     {"step": 7, "type": "EMIT", "factor": "h3"},
     {"step": 8, "type": "EMIT", "factor": "h4"},
     {"step": 9, "type": "EMIT", "factor": "h5"}],
    prefix=9,
)


def test_damage_to_retained_state_changes_the_answer_and_repair_restores_it():
    eng = Incremental(COMPLETE_CASE, integrity=False)
    for e in COMPLETE_CASE["events"]:
        eng.apply(e)
    before = eng.snapshot()
    assert before[COMPLETE_CASE["target_query_id"]] == 1
    damaged = eng.damage(random.Random(11), 1)
    assert damaged == 1
    after = eng.snapshot()
    assert after[COMPLETE_CASE["target_query_id"]] == 0, \
        "damage to a complete witness path must change the answer"
    eng.rebuild()
    assert eng.snapshot()[COMPLETE_CASE["target_query_id"]] == 1


def test_integrity_checked_engine_detects_and_repairs_the_damage():
    eng = Incremental(COMPLETE_CASE, integrity=True)
    for e in COMPLETE_CASE["events"]:
        eng.apply(e)
    eng.damage(random.Random(11), 1)
    eng.apply({"step": 10, "type": "EMIT", "factor": "h0"})
    assert eng.rebuilds >= 1, "integrity check failed to detect the corruption"
    assert eng.snapshot()[COMPLETE_CASE["target_query_id"]] == 1


# ------------------------------------------------------- structural rejection
def test_duplicate_factor_id_is_rejected():
    c = json.loads(json.dumps(RX_CASE))
    c["factors"].append(json.loads(json.dumps(c["factors"][0])))
    expect_reject(c, "duplicate factor id")


def test_event_referencing_an_unknown_factor_is_rejected():
    c = json.loads(json.dumps(RX_CASE))
    c["events"].append({"step": 10, "type": "EMIT", "factor": "ghost"})
    expect_reject(c, "unknown factor in event")


def test_missing_role_key_is_rejected():
    c = json.loads(json.dumps(RX_CASE))
    del c["factors"][1]["roles"]["scope"]
    expect_reject(c, "role schema")


def test_empty_role_token_is_rejected():
    c = json.loads(json.dumps(RX_CASE))
    c["factors"][1]["roles"]["scope"] = ""
    expect_reject(c, "empty role token")


def test_self_loop_factor_is_accepted_because_the_contract_permits_it():
    """The contract declares role NAMES are distinct, not role VALUES.
    A factor with from == to is therefore legal and must not be rejected."""
    c = json.loads(json.dumps(RX_CASE))
    c["factors"][1]["roles"]["to"] = c["factors"][1]["roles"]["from"]
    validate_dataset(dataset(c), {c["case_id"]: c["events"]}, "TRAIN")


def test_non_monotonic_step_order_is_rejected():
    c = json.loads(json.dumps(RX_CASE))
    c["events"][3]["step"] = 1
    expect_reject(c, "steps not strictly increasing")


def test_duplicate_query_id_is_rejected():
    c = json.loads(json.dumps(RX_CASE))
    c["queries"][3] = json.loads(json.dumps(c["queries"][0]))
    expect_reject(c, "duplicate query id")


def test_truth_log_shorter_than_visible_log_is_rejected():
    c = json.loads(json.dumps(RX_CASE))
    try:
        validate_dataset(dataset(c), {c["case_id"]: c["events"][:3]}, "TRAIN")
    except ContractViolation as e:
        assert "truth log shorter than visible log" in str(e)
        return
    raise AssertionError("short truth log was accepted")


if __name__ == "__main__":
    import inspect
    out = {}
    for name, fn in sorted(inspect.getmembers(sys.modules[__name__],
                                             lambda x: inspect.isfunction(x)
                                             and x.__name__.startswith("test_"))):
        try:
            fn()
            out[name] = "PASS"
        except AssertionError as e:
            out[name] = "FAIL: " + str(e)
    print(json.dumps(out, indent=1))
    print("FIXTURE VERDICT:", "PASS" if all(v == "PASS" for v in out.values()) else "FAIL")
