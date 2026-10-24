"""Hand-built adversarial fixtures for invariants the generated corpus does not
exercise.

The generated H1D0 corpus never places an RX factor in a position where it could
complete a witness chain, and never produces a pending pool that changes a
decision.  Those invariants are therefore untested by corpus-level agreement.
These fixtures construct the minimal cases that DO exercise them.
"""
import json, sys
sys.path.insert(0, "src")
from oracle import answer, possible_final_states, identifiability
from checker import witness, checker_local_labels, ContractViolation
from engines import Incremental, HcaNativeSemantics, Replay
import random

def tok(i):
    return "e%d" % i

# ---- fixture 1: an RX factor that WOULD complete the chain if rel were ignored
RX_CASE = {
    "case_id": "fx-rx", "pair_id": "p1", "kind": "structural",
    "target_query_id": "q1",
    "factors": [
        {"id": "f0", "rel": "RX", "roles": {"from": tok(0), "to": tok(1),
                                           "scope": tok(4), "kind": tok(5)},
         "source": "s1", "version": 0},
        {"id": "f1", "rel": "R1", "roles": {"from": tok(1), "to": tok(2),
                                           "scope": tok(4), "kind": tok(5)},
         "source": "s2", "version": 0},
        {"id": "f2", "rel": "R2", "roles": {"from": tok(2), "to": tok(3),
                                           "scope": tok(4), "kind": tok(5)},
         "source": "s3", "version": 0},
    ],
    "queries": [{"id": "q1", "from": tok(0), "to": tok(3), "scope": tok(4), "kind": tok(5)}],
    "events": [{"type": "SOURCE", "source": "s1", "status": "VALID", "epoch": 0},
               {"type": "SOURCE", "source": "s2", "status": "VALID", "epoch": 0},
               {"type": "SOURCE", "source": "s3", "status": "VALID", "epoch": 0},
               {"type": "EMIT", "factor": "f0"},
               {"type": "EMIT", "factor": "f1"},
               {"type": "EMIT", "factor": "f2"}],
    "pending": [], "damage_prefix": 3,
}

# ---- fixture 2: a pending pool that genuinely flips the decision
FLIP_CASE = {
    "case_id": "fx-flip", "pair_id": "p2", "kind": "provenance",
    "target_query_id": "q1",
    "factors": [
        {"id": "g0", "rel": "R0", "roles": {"from": tok(0), "to": tok(1),
                                           "scope": tok(4), "kind": tok(5)},
         "source": "s1", "version": 1},
        {"id": "g1", "rel": "R1", "roles": {"from": tok(1), "to": tok(2),
                                           "scope": tok(4), "kind": tok(5)},
         "source": "s2", "version": 0},
        {"id": "g2", "rel": "R2", "roles": {"from": tok(2), "to": tok(3),
                                           "scope": tok(4), "kind": tok(5)},
         "source": "s3", "version": 0},
    ],
    "queries": [{"id": "q1", "from": tok(0), "to": tok(3), "scope": tok(4), "kind": tok(5)}],
    # visible log: g0's source is STALE at epoch 0, so the chain is broken
    "events": [{"type": "SOURCE", "source": "s1", "status": "STALE", "epoch": 0},
               {"type": "SOURCE", "source": "s2", "status": "VALID", "epoch": 0},
               {"type": "SOURCE", "source": "s3", "status": "VALID", "epoch": 0},
               {"type": "EMIT", "factor": "g0"},
               {"type": "EMIT", "factor": "g1"},
               {"type": "EMIT", "factor": "g2"}],
    # withheld: the rebind that would make g0 live
    "pending": [{"type": "SOURCE", "source": "s1", "status": "VALID", "epoch": 1}],
    "damage_prefix": 3,
}

results = {}

# RX must never participate in a witness
results["rx_never_witness_oracle"] = answer(RX_CASE["events"], RX_CASE["factors"],
                                           RX_CASE["queries"][0], len(RX_CASE["events"]))
results["rx_never_witness_checker"] = witness(RX_CASE["events"], RX_CASE["factors"],
                                             {}, {"s1": ("VALID", 0), "s2": ("VALID", 0),
                                                 "s3": ("VALID", 0)},
                                             {"f0": True, "f1": True, "f2": True},
                                             RX_CASE["queries"][0])
eng = HcaNativeSemantics(RX_CASE, method="B0")
for e in RX_CASE["events"]:
    eng.apply(e)
results["rx_never_witness_b0"] = eng.snapshot()["q1"]
eng2 = HcaNativeSemantics(RX_CASE, method="B2")
for e in RX_CASE["events"]:
    eng2.apply(e)
results["rx_never_witness_b2"] = eng2.snapshot()["q1"]

# identifiability: the pool genuinely changes the answer
states = possible_final_states(FLIP_CASE["events"], FLIP_CASE["pending"])
from oracle import answer_from_state
vals = {answer_from_state(s, FLIP_CASE["factors"], FLIP_CASE["queries"][0]) for s in states}
results["flip_states"] = len(states)
results["flip_distinct_answers"] = sorted(vals)
results["flip_is_underdetermined"] = len(vals) > 1
results["flip_visible_only_answer"] = answer(FLIP_CASE["events"], FLIP_CASE["factors"],
                                            FLIP_CASE["queries"][0], len(FLIP_CASE["events"]))

# damage must be observable and repairable
inc = Incremental(FLIP_CASE, integrity=True)
for e in FLIP_CASE["events"]:
    inc.apply(e)
before = inc.snapshot()
d = inc.damage(random.Random(7), 6)
after = inc.snapshot()
inc.rebuild()
repaired = inc.snapshot()
results["damage_units"] = d
results["damage_changed_snapshot"] = before != after
results["damage_repair_restores"] = repaired == before

# a duplicate factor id must be rejected by the checker
dup = json.loads(json.dumps(RX_CASE))
dup["factors"].append(json.loads(json.dumps(dup["factors"][0])))
try:
    checker_local_labels(dup["events"], dup["factors"], dup["queries"])
    results["duplicate_id_rejected"] = False
except ContractViolation as e:
    results["duplicate_id_rejected"] = True
    results["duplicate_id_reason"] = str(e)

# an event referencing an unknown factor must be rejected
bad = json.loads(json.dumps(RX_CASE))
bad["events"].append({"type": "EMIT", "factor": "nonexistent"})
try:
    checker_local_labels(bad["events"], bad["factors"], bad["queries"])
    results["unknown_factor_event_rejected"] = False
except ContractViolation as e:
    results["unknown_factor_event_rejected"] = True
    results["unknown_factor_event_reason"] = str(e)

# arity/role invariants
bad2 = json.loads(json.dumps(RX_CASE))
bad2["factors"][1]["roles"]["scope"] = bad2["factors"][1]["roles"]["from"]
try:
    checker_local_labels(bad2["events"], bad2["factors"], bad2["queries"])
    results["duplicate_role_rejected"] = False
except ContractViolation as e:
    results["duplicate_role_rejected"] = True
    results["duplicate_role_reason"] = str(e)

results["verdict"] = "FIXTURES_PASS" if (
    results["rx_never_witness_oracle"] == 0 and results["rx_never_witness_checker"] == 0
    and results["rx_never_witness_b0"] == 0 and results["rx_never_witness_b2"] == 0
    and results["flip_is_underdetermined"]
    and results["damage_changed_snapshot"] and results["damage_repair_restores"]
    and results["duplicate_id_rejected"] and results["unknown_factor_event_rejected"]
    and results["duplicate_role_rejected"]) else "FIXTURES_FAIL"
print(json.dumps(results, indent=1))
