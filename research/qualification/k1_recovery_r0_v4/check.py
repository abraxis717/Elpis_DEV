"""K1 Recovery R0 v4: the pre-registered gates, evaluated from the run records alone.

    python -I check.py RECORDS [--plan plan.json] [--out VERDICT]

The checker reads nothing but the frozen plan and the records the runner wrote (it imports no Elpis code and runs
nothing native), so anyone holding the evidence can recompute the verdict. Each record is judged against its point's
pre-registered expectation; identities are resolved from what the child reported before it was killed (D0 the
anchored state, D1 the first durable publication, D2 the candidate, D3 the volatile state never requested; K and C
for the import). A record that is missing, duplicated or incomplete fails G1.

G1 dispositions as pre-registered · G2 every resumed state is continuity's, with the memory transition present exactly
when authorized · G3 no volatile work recovered, no implicit operator action, no publication over an unresolved
candidate · G4 the acknowledgment law; identity-only never recoverable · G5 an uncertain publication faults the lane,
the HOT lane continues, restart resolves one complete record · G6 two fixed slot files, sizes and inodes unchanged.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
GATES = ("G1", "G2", "G3", "G4", "G5", "G6")
VERDICT_SCHEMA = "elpis.k1-recovery-r0-v4.verdict.v1"
SLOT_NAMES = ("k1-checkpoint.a", "k1-checkpoint.b")
HEADER = 128


class Judge:
    def __init__(self):
        self.failures = collections.defaultdict(list)
        self.checked = collections.Counter()

    def gate(self, gate: str, record: dict, ok: bool, why: str) -> None:
        self.checked[gate] += 1
        if not ok:
            self.failures[gate].append({"id": record["id"], "why": why})


def _labels(child: dict) -> dict:
    return {k: child.get(k) for k in ("D0", "D1", "D2", "D3", "K", "C")}


def _final(record: dict, expect: dict) -> str:
    """The label continuity must name after the pre-registered operator action."""
    if "final" in expect:
        return expect["final"]
    if expect.get("disposition") == "CANDIDATE_UNRESOLVED":
        return "D2" if record["run_index"] % 2 == 0 else "D1"   # adopt when even, discard when odd
    return expect["authority"]


def _footprint(j: Judge, r: dict, provisioned: dict, final: dict, capacity_known=True) -> None:
    names_ok = sorted(provisioned) == list(SLOT_NAMES)
    sizes = {v["size"] for v in provisioned.values()}
    j.gate("G6", r, names_ok and len(sizes) == 1 and final == provisioned,
           f"provisioned {provisioned} final {final}")


def _resumed(j: Judge, r: dict, resumed: dict, want: str | None, keys: list, authorized_keys: set) -> None:
    ok = (resumed.get("disposition") == "RESUMABLE" and resumed.get("restored_identity") is not None
          and resumed.get("restored_identity") == resumed.get("continuity_identity") == want)
    j.gate("G2", r, ok, f"resumed {resumed.get('restored_identity')} continuity {resumed.get('continuity_identity')} "
                        f"want {want}")
    if r["lineage"] == "composite" and r["family"] != "import":
        resident = resumed.get("resident", {})
        expected = {k: (1 if k in authorized_keys else None) for k in keys if k is not None}
        j.gate("G2", r, resident == expected, f"resident {resident} expected {expected}")
    continued = resumed.get("continued", {})
    t = continued.get("ticket", {})
    j.gate("G2", r, t.get("status") == "ACKNOWLEDGED" and t.get("recoverable") is True
           and t.get("identity") == continued.get("identity"), f"continued {continued}")


def _keys_upto(keys: list, label: str) -> set:
    """The relation keys an authorized state holds: LEARN 0 for D1, LEARNs 0 and 1 for D2."""
    n = {"D0": 0, "D1": 1, "D2": 2, "D3": 3}.get(label, 0)
    return {k for k in keys[:n] if k is not None}


def check_store_family(j: Judge, r: dict) -> None:
    """publish, discard, adopt, uncertain: a lineage with a checkpoint store and a candidate scenario."""
    c, rs, e = r["child"], r["restart"], r["expect"]
    L = _labels(c)
    o = rs["observed"]
    family = r["family"]
    if family == "uncertain":
        t = c["ticket"]
        j.gate("G5", r, t["status"] == e["status"] and t["recoverable"] is False, f"ticket {t}")
        j.gate("G5", r, c["lane_fault"] == e["fault"] and c["request_under_fault"] == e["fault"],
               f"lane fault {c['lane_fault']} request {c['request_under_fault']}")
        j.gate("G5", r, c["hot_after_fault"] is True and c["D3"] not in (None, c["D2"]), "HOT lane after the fault")
        j.gate("G5", r, o["authorized"] in (L["D1"], L["D2"]), "restart names exactly one complete record")
    else:
        j.gate("G1", r, r["child_signal"] == -9, f"child signal {r['child_signal']}")
        j.gate("G1", r, c["ticket"]["status"] == e.get("status", "PENDING"), f"ticket {c['ticket']}")
    if family in ("discard", "adopt"):
        found = r["reconcile"]["found"]
        j.gate("G1", r, (found["disposition"], found["authorized"], found["candidate"])
               == ("CANDIDATE_UNRESOLVED", L["D1"], L["D2"]), f"reconciliation precondition {found}")
        j.gate("G1", r, r["reconcile_signal"] == -9 and r["reconcile"]["control"] == {},
               f"parked inside the reconciliation {r['reconcile']['control']}")
    j.gate("G1", r, o["disposition"] == e["disposition"], f"disposition {o['disposition']} expected {e['disposition']}")
    j.gate("G1", r, o["authorized"] == L[e["authority"]], f"authorized {o['authorized']} expected {e['authority']}")
    j.gate("G1", r, o["candidate"] == (L[e["candidate"]] if "candidate" in e else None),
           f"candidate {o['candidate']} expected {e.get('candidate')}")
    # G3: nothing implicit; the pre-registered action only; never over an unresolved candidate; never D3.
    unresolved = o["disposition"] == "CANDIDATE_UNRESOLVED"
    want_action = (family if family in ("discard", "adopt") else ("adopt" if r["run_index"] % 2 == 0 else "discard"))
    j.gate("G3", r, rs["action"] == (want_action if unresolved else None), f"action {rs['action']}")
    if unresolved:
        p = rs["publish_over_candidate"]
        j.gate("G3", r, p["attempted"] and p["ticket"]["status"] == "FAILED"
               and p["ticket"]["code"] == "CHECKPOINT_UNRESOLVED", f"publication over the candidate {p}")
    final = _final(r, e)
    after = rs["after_action"] or o
    j.gate("G3", r, after["authorized"] == L[final] and after["authorized"] != L["D3"],
           f"after the action {after['authorized']} expected {final}")
    # G4: the acknowledgment law.
    first = c["first_ticket"]
    j.gate("G4", r, (first["status"], first["recoverable"], first["identity"]) == ("ACKNOWLEDGED", True, L["D1"]),
           f"first ticket {first}")
    t = c["ticket"]
    j.gate("G4", r, (t["status"] != "ACKNOWLEDGED" or (t["recoverable"] is True and t["identity"] == L["D2"]))
           and (t["recoverable"] is False or t["status"] == "ACKNOWLEDGED"), f"ticket {t}")
    _resumed(j, r, rs["resumed"], L[final], c.get("keys", []), _keys_upto(c.get("keys", []), final))
    _footprint(j, r, r["provisioned"], rs["final_footprint"])


def check_identity(j: Judge, r: dict) -> None:
    c, rs, e = r["child"], r["restart"], r["expect"]
    L = _labels(c)
    j.gate("G1", r, r["child_signal"] == -9 and c["ticket"]["status"] == e["status"], f"child {c['ticket']}")
    j.gate("G1", r, rs["before"]["authorized"] == L[e["authority"]], f"continuity {rs['before']['authorized']} "
                                                                     f"expected {e['authority']}")
    j.gate("G4", r, rs["observed"]["disposition"] == "CHECKPOINT_MISSING", f"restart {rs['observed']['disposition']}")
    first = c["first_ticket"]
    j.gate("G4", r, (first["kind"], first["status"], first["recoverable"], first["identity"])
           == ("IDENTITY", "IDENTITY_PUBLISHED", False, L["D1"]), f"first ticket {first}")
    j.gate("G4", r, c["ticket"]["status"] != "ACKNOWLEDGED" and c["ticket"]["recoverable"] is False,
           f"ticket {c['ticket']}")
    j.gate("G3", r, rs["caller_bytes"]["authorized"] == [e["authority"]]
           and rs.get("wrong_bytes_refusal") == "CONTINUITY_STATE_MISMATCH", f"caller bytes {rs['caller_bytes']} "
                                                                             f"{rs.get('wrong_bytes_refusal')}")
    j.gate("G3", r, rs["before"]["authorized"] != L["D3"], "volatile state never named")
    continued = rs["continued"]
    j.gate("G2", r, continued["ticket"]["status"] == "ACKNOWLEDGED" and rs["after"]["disposition"] == "RESUMABLE"
           and rs["after"]["authorized"] == continued["identity"], f"continued {continued}")
    _footprint(j, r, rs["provisioned"], rs["final_footprint"])


def check_identity_refused(j: Judge, r: dict) -> None:
    c, rs, e = r["child"], r["restart"], r["expect"]
    L = _labels(c)
    j.gate("G4", r, c["refusal"] == e["refusal"] and c["recoverable_refusal"] is not None,
           f"refusals {c['refusal']} {c['recoverable_refusal']}")
    j.gate("G3", r, c["durable_after"] == L["D0"] and rs["observed"]["authorized"] == L["D0"],
           f"durable {c['durable_after']} restart {rs['observed']['authorized']}")


def check_anchor(j: Judge, r: dict) -> None:
    c, rs, e = r["child"], r["restart"], r["expect"]
    d0 = c["D0"]
    o = rs["observed"]
    j.gate("G1", r, r["child_signal"] == -9 and c["control"] == {}, f"parked inside the anchor {c['control']}")
    j.gate("G1", r, o["disposition"] == e["disposition"], f"disposition {o['disposition']} expected {e['disposition']}")
    held = any(s and s["identity"] == d0 and s["verified"] for s in o["slots"].values())
    j.gate("G1", r, held == e["slot_holds_d0"], f"slots {o['slots']}")
    if e["disposition"] == "RESUMABLE":
        j.gate("G1", r, o["authorized"] == d0, f"authorized {o['authorized']}")
    j.gate("G3", r, rs["action"] == ("anchor" if e["disposition"] == "NOTHING_TO_RECOVER" else None),
           f"action {rs['action']}")
    _resumed(j, r, rs["resumed"], d0, [], set())
    _footprint(j, r, r["provisioned"], rs["final_footprint"])


def check_import(j: Judge, r: dict) -> None:
    c, rs, e = r["child"], r["restart"], r["expect"]
    L = _labels(c)
    oc, ok1 = rs["observed_composite_store"], rs["observed_k1_store"]
    j.gate("G1", r, r["child_signal"] == -9 and c["control"] == {}, f"parked inside the import {c['control']}")
    j.gate("G4", r, (c["first_ticket"]["status"], c["first_ticket"]["identity"]) == ("ACKNOWLEDGED", L["K"]),
           f"first ticket {c['first_ticket']}")
    j.gate("G1", r, (oc["format"], oc["authorized"]) == (e["format"], L[e["authority"]]),
           f"continuity {oc['format']} {oc['authorized']}")
    j.gate("G1", r, (oc["disposition"], ok1["disposition"]) == (e["composite_store"], e["k1_store"]),
           f"composite store {oc['disposition']} K1 store {ok1['disposition']}")
    j.gate("G3", r, rs["action"] == ("import" if e["format"] == "K1" else None), f"action {rs['action']}")
    if e["format"] == "K1":
        j.gate("G3", r, rs["reimported"] == {"identity": L["C"], "format": "COMPOSITE"}, f"reimport {rs['reimported']}")
    _resumed(j, r, rs["resumed"], L["C"], [], set())
    j.gate("G2", r, rs["resumed"].get("resident", {}) == {}, "an imported composite holds an empty memory")
    _footprint(j, r, r["provisioned"], rs["final_footprint"])
    _footprint(j, r, r["provisioned_composite"], rs["final_footprint_composite"])


def check_random(j: Judge, r: dict) -> None:
    rs = r["restart"]
    events = r["events"]
    order, keys = {}, []
    anchor = next(e["identity"] for e in events if e.get("event") == "anchor")
    order[anchor] = -1
    for e in events:
        if e.get("event") == "learn":
            order[e["identity"]] = e["i"]
            keys.append(e["key"])
    requested = {e["identity"] for e in events if e.get("event") == "request"}
    acked = [e["identity"] for e in events if e.get("event") == "ticket" and e["status"] == "ACKNOWLEDGED"]
    tickets_ok = all(e["recoverable"] is (e["status"] == "ACKNOWLEDGED") for e in events if e.get("event") == "ticket")
    j.gate("G4", r, tickets_ok, "acknowledgment law of every delivered ticket")
    floor = max((order[a] for a in acked), default=-1)
    o = rs["observed"]
    authorized = o["authorized"]
    j.gate("G1", r, r["child_signal"] == -9, f"child signal {r['child_signal']}")
    j.gate("G1", r, o["disposition"] in ("RESUMABLE", "CANDIDATE_UNRESOLVED"), f"disposition {o['disposition']}")
    j.gate("G3", r, authorized == anchor or authorized in requested, "never a volatile state never requested")
    j.gate("G1", r, authorized in order and order[authorized] >= floor,
           f"authorized at {order.get(authorized)}, last acknowledgment at {floor}")
    unresolved = o["disposition"] == "CANDIDATE_UNRESOLVED"
    if unresolved:
        cand = o["candidate"]
        j.gate("G3", r, cand in requested and order.get(cand, -2) > order[authorized], f"candidate {cand}")
    want_action = ("adopt" if r["run_index"] % 2 == 0 else "discard") if unresolved else None
    j.gate("G3", r, rs["action"] == want_action, f"action {rs['action']}")
    final = (o["candidate"] if want_action == "adopt" else authorized)
    after = rs["after_action"] or o
    j.gate("G3", r, after["authorized"] == final, f"after the action {after['authorized']}")
    upto = order.get(final, -1)
    _resumed(j, r, rs["resumed"], final, keys, {k for i, k in enumerate(keys) if i <= upto and k is not None})
    _footprint(j, r, r["provisioned"], rs["final_footprint"])


CHECKS = {"publish": check_store_family, "discard": check_store_family, "adopt": check_store_family,
          "uncertain": check_store_family, "identity": check_identity, "identity_refused": check_identity_refused,
          "anchor": check_anchor, "import": check_import, "random": check_random}


def planned(plan: dict) -> collections.Counter:
    want = collections.Counter()
    for family, fam in plan["families"].items():
        points = [p["point"] for p in fam.get("points", [{"point": "random"}])]
        runs = fam.get("runs_per_lineage", plan["runs_per_point"])
        for kind in fam["lineages"]:
            for point in points:
                want[(family, kind, point)] += runs
    return want


def evaluate(plan: dict, records: list) -> dict:
    j = Judge()
    got = collections.Counter((r["family"], r["lineage"], r["point"]) for r in records)
    want = planned(plan)
    complete = got == want and len({r["id"] for r in records}) == len(records)
    for r in records:
        if r.get("child_error") or r.get("reconcile_error") or r.get("restart_error") or not r.get("restart"):
            j.gate("G1", r, False, "incomplete run: " + str(r.get("child_error") or r.get("reconcile_error")
                                                             or r.get("restart_error"))[:300])
            continue
        try:
            CHECKS[r["family"]](j, r)
        except (KeyError, TypeError, StopIteration) as exc:
            j.gate("G1", r, False, f"malformed record: {exc!r}")
    gates = {g: {"checks": j.checked[g], "failures": j.failures[g], "verdict": "PASS" if j.checked[g] and not
                 j.failures[g] else "FAIL"} for g in GATES}
    verdict = "PASS" if complete and all(v["verdict"] == "PASS" for v in gates.values()) else "FAIL"
    return {"schema": VERDICT_SCHEMA, "records": len(records), "planned": sum(want.values()), "complete": complete,
            "missing": sorted("/".join(k) for k in (want - got)), "unexpected": sorted("/".join(k) for k in
                                                                                   (got - want)),
            "gates": gates, "verdict": verdict}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="check.py")
    p.add_argument("records")
    p.add_argument("--plan", default=str(HERE / "plan.json"))
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    plan_bytes = Path(args.plan).read_bytes()
    records = [json.loads(line) for line in Path(args.records).read_text().splitlines() if line.strip()]
    result = evaluate(json.loads(plan_bytes), records)
    result["plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
    result["records_sha256"] = hashlib.sha256(Path(args.records).read_bytes()).hexdigest()
    text = json.dumps(result, indent=1, sort_keys=True) + "\n"
    if args.out:
        with open(args.out, "x") as f:
            f.write(text)
    summary = {g: (v["verdict"], v["checks"], len(v["failures"])) for g, v in result["gates"].items()}
    print(json.dumps({"verdict": result["verdict"], "complete": result["complete"], "gates": summary}))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
