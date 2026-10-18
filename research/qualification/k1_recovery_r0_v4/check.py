"""K1 Recovery R0 v4: the pre-registered gates, evaluated from the frozen plan and the run records alone.

    python -I -B check.py verdict RECORDS --inputs qual [--out VERDICT]
    python -I -B check.py evidence EVIDENCE_DIR

The checker imports no Elpis code, no runner and nothing native: anyone holding the plan and the records can recompute
the verdict. It never takes an expectation from a record. From the plan alone it derives every trial's coordinate
(family, lineage, point, run index), its index, seed, fault argument, random-family draws, its expected disposition,
authority, candidate, operator action and continuity generation; a record must sit at exactly its coordinate, once,
in order. It parses the raw continuity records itself (magic, format byte, generation, digest, checksum under the
format's domain) and judges the raw checkpoint-slot comparison the restart recorded, so a tear must be shown, never
assumed. ``evidence`` validates a sealed evidence directory: exactly the required files, every SHA-256, the binding of
run, records, verdict and freeze to each other, and the verdict recomputed.

G1 dispositions and coordinates as pre-registered; every injected death a real SIGKILL; every tear real · G2 every
resumed state is continuity's, with the memory transition present exactly when authorized · G3 nothing implicit: no
volatile work recovered, the registered operator action only, no publication over an unresolved candidate, exactly
the expected continuity generations (no retry, no duplicate publication) · G4 the acknowledgment law; identity-only
never recoverable; a refused publication moves nothing · G5 an uncertain publication faults the lane, HOT continues,
restart names one complete record · G6 two fixed slot files, sizes and inodes unchanged · S4 (within G1/G2) continuity's
actual record format is the lineage's, and another format's state is refused.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
GATES = ("G1", "G2", "G3", "G4", "G5", "G6")
RECORD_SCHEMA = "elpis.k1-recovery-r0-v4.record.v2"
VERDICT_SCHEMA = "elpis.k1-recovery-r0-v4.verdict.v2"
RUN_SCHEMA = "elpis.k1-recovery-r0-v4.run.v2"
FREEZE_SCHEMA = "elpis.k1-recovery-r0-v4.freeze.v2"
SLOT_NAMES = ("k1-checkpoint.a", "k1-checkpoint.b")
EVIDENCE_FILES = ("SHA256SUMS", "freeze.json", "records.jsonl", "run.json", "verdict.json")
FORMAT = {"k1": "K1", "composite": "COMPOSITE"}
GENESIS_GENERATION = 1
_REGISTER = {b"ELPCONT\x02": ("K1", b"elpis.continuity.register.v2\x00"),
             b"ELPCONT\x03": ("COMPOSITE", b"elpis.continuity.register.v3\x00")}


# =====================================================================================================================
# The plan's coordinates, derived independently of the runner
# =====================================================================================================================

def trials(plan: dict, runs_override: int | None = None) -> list:
    out = []
    for family, fam in plan["families"].items():
        points = [p["point"] for p in fam.get("points", [{"point": "random"}])]
        runs = runs_override or fam.get("runs_per_lineage", plan["runs_per_point"])
        for lineage in fam["lineages"]:
            for point in points:
                for run_index in range(runs):
                    out.append((family, lineage, point, run_index))
    return out


def fault_arg(plan: dict, lineage: str, point: str, run_index: int) -> int:
    capacity = plan["lineages"][lineage]["envelope_bytes"]
    if point == "step:3":
        return (17, 127, 128 + capacity // 2, 128 + capacity - 32)[run_index % 4]
    if point in ("step:6", "control:4"):
        return plan["torn_offsets"]["record"][run_index % 4]
    return 0


def coordinates(plan: dict, inputs: str, runs_override: int | None = None) -> list:
    """Every trial's full coordinate: what the runner must have used for it."""
    rng = random.Random(plan["inputs"][inputs]["random_seed"])
    out = []
    for index, (family, lineage, point, run_index) in enumerate(trials(plan, runs_override)):
        c = {"index": index, "family": family, "lineage": lineage, "point": point, "run_index": run_index,
             "seed": plan["inputs"][inputs]["lineage_seed_base"] + index, "inputs": inputs,
             "arg": fault_arg(plan, lineage, point, run_index)}
        if family == "random":
            c["rng_seed"] = rng.randrange(1 << 63)
            c["kill_after"] = rng.randint(1, plan["families"]["random"]["max_learns"][lineage] - 1)
        out.append(c)
    return out


def expectation(plan: dict, family: str, point: str) -> dict:
    for p in plan["families"][family].get("points", []):
        if p["point"] == point:
            return p["expect"]
    return {}


def operator_action(family: str, run_index: int) -> str:
    """plan.operator_actions, CANDIDATE_UNRESOLVED."""
    return family if family in ("discard", "adopt") else ("adopt" if run_index % 2 == 0 else "discard")


# =====================================================================================================================
# Raw continuity records, parsed here (docs/CONTINUITY.md layout; native/continuity/src/record.rs)
# =====================================================================================================================

def parse_record(raw_hex: str | None) -> dict | None:
    if raw_hex is None:
        return None
    raw = bytes.fromhex(raw_hex)
    if len(raw) != 176 or raw[:8] not in _REGISTER:
        return {"valid": False}
    fmt, domain = _REGISTER[raw[:8]]
    valid = hashlib.sha256(domain + raw[:144]).digest() == raw[144:176]
    return {"valid": valid, "format": fmt, "generation": int.from_bytes(raw[16:24], "big"), "anchored": raw[24],
            "format_byte": raw[25], "digest": raw[32:64].hex()}


def current_record(raw: dict) -> dict | None:
    """The authority a restart resolves: the valid record of the highest generation."""
    parsed = [parse_record(v) for v in raw.values()]
    valid = [p for p in parsed if p and p["valid"]]
    return max(valid, key=lambda p: p["generation"]) if valid else None


def _diff(before_hex: str, after_hex: str) -> list:
    a, b = bytes.fromhex(before_hex), bytes.fromhex(after_hex)
    return [i for i in range(max(len(a), len(b))) if (a[i:i + 1] or b"") != (b[i:i + 1] or b"")]


# =====================================================================================================================
# The judge
# =====================================================================================================================

class Judge:
    def __init__(self):
        self.failures = collections.defaultdict(list)
        self.checked = collections.Counter()

    def gate(self, gate: str, record: dict, ok: bool, why: str) -> None:
        self.checked[gate] += 1
        if not ok:
            self.failures[gate].append({"index": record.get("index"), "id": record.get("id"), "why": why})


def _labels(child: dict) -> dict:
    return {k: (child or {}).get(k) for k in ("D0", "D1", "D2", "D3", "K", "C")}


def _killed(j: Judge, r: dict, key: str = "child_exit") -> None:
    e = r.get(key) or {}
    j.gate("G1", r, e.get("signaled") is True and e.get("returncode") == -9, f"{key}: not a SIGKILL death {e}")


def _exited(j: Judge, r: dict, key: str) -> None:
    e = r.get(key) or {}
    j.gate("G1", r, e.get("returncode") == 0 and e.get("signaled") is False, f"{key}: {e}")


def _format(j: Judge, r: dict, raw: dict, observed: dict, fmt: str) -> None:
    """S4: continuity's actual record (parsed from its raw bytes) names the observed authority in the lineage's
    format, and the runtime reports the same format."""
    cur = current_record(raw)
    if observed.get("authorized") is None:   # unanchored: the current record names nothing
        j.gate("G1", r, cur is not None and cur["anchored"] == 0, f"unanchored continuity raw {cur}")
        return
    j.gate("G1", r, observed.get("format") == fmt, f"continuity format {observed.get('format')} expected {fmt}")
    j.gate("G1", r, cur is not None and cur["format"] == fmt and cur["digest"] == observed["authorized"]
           and cur["generation"] == observed["generation"] and cur["anchored"] == 1
           and cur["format_byte"] == (1 if fmt == "COMPOSITE" else 0),
           f"raw continuity record {cur} does not name {observed['authorized']} as {fmt}")


def _cross_format(j: Judge, r: dict, rs: dict, family: str) -> None:
    if family in ("publish", "anchor") and rs.get("after_action", {}).get("disposition") == "RESUMABLE":
        c = rs.get("cross_format") or {}
        j.gate("G1", r, c.get("refusal") == "CONTINUITY_STATE_MISMATCH" and c.get("unchanged") is True,
               f"a state of another format was not refused without effect: {c}")


def _footprint(j: Judge, r: dict, provisioned: dict, final: dict, envelope_bytes: int, header: int) -> None:
    sizes = {v["size"] for v in provisioned.values()}
    j.gate("G6", r, sorted(provisioned) == list(SLOT_NAMES) and sizes == {header + envelope_bytes}
           and final == provisioned, f"provisioned {provisioned} final {final}")


def _torn_slot(j: Judge, r: dict, rs: dict, c: dict) -> None:
    """S3: exactly one slot file changed, only below the tear, at least one byte, and it is not a valid slot."""
    slots = (rs.get("files_after") or {}).get("slots", {})
    changed = {name: e for name, e in slots.items() if e["changed"]}
    ok = len(changed) == 1
    if ok:
        name, e = next(iter(changed.items()))
        raw = rs["observed"]["slots"].get(name.split("/", 1)[1])
        # The bytes around the tear are recorded as evidence; they may coincide (a composite's empty banks are zeros
        # in both states), so the proof is the difference range and the slot's invalidity, not the window.
        ok = (e["within_arg"] is True and e["diff_bytes"] >= 1 and e["last_diff"] < c["arg"]
              and (raw is None or raw["verified"] is False))
    j.gate("G1", r, ok, f"no genuine partial slot write at {c['arg']}: {changed}")


def _torn_record(j: Judge, r: dict, before: dict, after: dict, c: dict) -> None:
    """S3: exactly one continuity record changed, only below the tear, at least one byte, and it is invalid."""
    changed = [n for n in after if before.get(n) != after[n]]
    ok = len(changed) == 1
    if ok:
        diff = _diff(before.get(changed[0]) or "00" * 176, after[changed[0]])
        torn = parse_record(after[changed[0]])
        ok = bool(diff) and max(diff) < c["arg"] and torn is not None and torn["valid"] is False
    j.gate("G1", r, ok, f"no genuine torn record at {c['arg']}: changed {changed}")


def _resumed(j: Judge, r: dict, resumed: dict, want: str | None, keys: list, authorized_keys: set,
             composite: bool) -> None:
    ok = (resumed.get("disposition") == "RESUMABLE" and resumed.get("restored_identity") is not None
          and resumed.get("restored_identity") == resumed.get("continuity_identity") == want)
    j.gate("G2", r, ok, f"resumed {resumed.get('restored_identity')} continuity {resumed.get('continuity_identity')} "
                        f"want {want}")
    if composite:
        expected = {k: (1 if k in authorized_keys else None) for k in keys if k is not None}
        j.gate("G2", r, resumed.get("resident") == expected, f"resident {resumed.get('resident')} expected {expected}")
        j.gate("G2", r, resumed.get("memory_slots") == len(authorized_keys)
               and resumed.get("memory_slots_after") == len(authorized_keys) + 1,
               f"memory occupancy {resumed.get('memory_slots')} -> {resumed.get('memory_slots_after')}, "
               f"expected {len(authorized_keys)} -> {len(authorized_keys) + 1}")
    continued = resumed.get("continued", {})
    t = continued.get("ticket", {})
    j.gate("G2", r, t.get("status") == "ACKNOWLEDGED" and t.get("recoverable") is True
           and t.get("identity") == continued.get("identity"), f"continued {continued}")


def _keys_upto(keys: list, label: str) -> set:
    n = {"D0": 0, "D1": 1, "D2": 2, "D3": 3}[label]
    return {k for k in keys[:n] if k is not None}


GEN = {"D0": 0, "D1": 1, "D2": 2}


def check_store_family(j: Judge, plan: dict, r: dict, c: dict) -> None:
    """publish, discard, adopt, uncertain: a lineage with a checkpoint store and a candidate scenario."""
    family, kind = c["family"], c["lineage"]
    e = expectation(plan, family, c["point"])
    ch, rs = r["child"], r["restart"]
    L = _labels(ch)
    o = rs["observed"]
    if family == "uncertain":
        t = ch["ticket"]
        _exited(j, r, "child_exit")
        j.gate("G5", r, t["status"] == e["status"] and t["recoverable"] is False, f"ticket {t}")
        j.gate("G5", r, ch["lane_fault"] == e["fault"] and ch["request_under_fault"] == e["fault"],
               f"lane fault {ch['lane_fault']} request {ch['request_under_fault']}")
        j.gate("G5", r, ch["hot_after_fault"] is True, "the HOT lane did not continue under the lane fault")
        j.gate("G5", r, current_record(rs["continuity_raw"]) is not None
               and o["authorized"] in (L["D1"], L["D2"]), "restart names exactly one complete record")
    else:
        _killed(j, r)
        # The request is captured at once and returns PENDING: a request that waited for the worker would not.
        j.gate("G1", r, ch["ticket"]["status"] == e.get("status", "PENDING"), f"ticket {ch['ticket']}")
    if family in ("discard", "adopt"):
        found = r["reconcile"]["found"]
        j.gate("G1", r, (found["disposition"], found["authorized"], found["candidate"])
               == ("CANDIDATE_UNRESOLVED", L["D1"], L["D2"]), f"reconciliation precondition {found}")
        _killed(j, r, "reconcile_exit")
        j.gate("G1", r, r["reconcile"]["control"] == {}, f"not parked inside the reconciliation {r['reconcile']}")
    j.gate("G1", r, o["disposition"] == e["disposition"], f"disposition {o['disposition']} expected {e['disposition']}")
    j.gate("G1", r, o["authorized"] == L[e["authority"]], f"authorized {o['authorized']} expected {e['authority']}")
    j.gate("G1", r, o["candidate"] == (L[e["candidate"]] if "candidate" in e else None),
           f"candidate {o['candidate']} expected {e.get('candidate')}")
    _format(j, r, rs["continuity_raw"], o, FORMAT[kind])
    if c["point"] == "step:3":
        _torn_slot(j, r, rs, c)
    if c["point"] in ("step:6", "control:4"):
        before = (r["reconcile"] if family in ("discard", "adopt") else ch)["files_before"]["continuity"]
        _torn_record(j, r, before, rs["continuity_raw"], c)
    unresolved = o["disposition"] == "CANDIDATE_UNRESOLVED"
    action = operator_action(family, c["run_index"]) if unresolved else None
    j.gate("G3", r, rs["action"] == action, f"action {rs['action']} expected {action}")
    if unresolved:
        p = rs["publish_over_candidate"]
        j.gate("G3", r, p["attempted"] and p["ticket"]["status"] == "FAILED"
               and p["ticket"]["code"] == "CHECKPOINT_UNRESOLVED", f"publication over the candidate {p}")
    final = e.get("final") or ({"adopt": "D2", "discard": "D1"}[action] if unresolved else e["authority"])
    after = rs["after_action"]
    j.gate("G3", r, after["authorized"] == L[final] and after["authorized"] != L["D3"]
           and after["disposition"] == "RESUMABLE", f"after the action {after} expected {final}")
    base = ch["anchor"]
    j.gate("G3", r, o["generation"] == base + GEN[e["authority"]], f"generation {o['generation']} at restart")
    j.gate("G3", r, after["generation"] == o["generation"] + (1 if action == "adopt" else 0),
           f"generation after the action {after['generation']}")
    j.gate("G3", r, rs["final"]["generation"] == after["generation"] + 1 and rs["final"]["authorized"]
           == rs["resumed"].get("continued", {}).get("identity"), "the continuation published exactly once")
    first = ch["first_ticket"]
    j.gate("G4", r, (first["status"], first["recoverable"], first["identity"]) == ("ACKNOWLEDGED", True, L["D1"]),
           f"first ticket {first}")
    t = ch["ticket"]
    j.gate("G4", r, (t["status"] != "ACKNOWLEDGED" or (t["recoverable"] is True and t["identity"] == L["D2"]))
           and (t["recoverable"] is False or t["status"] == "ACKNOWLEDGED"), f"ticket {t}")
    _cross_format(j, r, rs, family)
    keys = ch.get("keys", [])
    _resumed(j, r, rs["resumed"], L[final], keys, _keys_upto(keys, final), kind == "composite")
    _footprint(j, r, r["provisioned"], rs["final_footprint"], plan["lineages"][kind]["envelope_bytes"],
               plan["slot_header_bytes"])


def check_identity(j: Judge, plan: dict, r: dict, c: dict) -> None:
    e = expectation(plan, "identity", c["point"])
    ch, rs = r["child"], r["restart"]
    L = _labels(ch)
    _killed(j, r)
    j.gate("G1", r, ch["ticket"]["status"] == e["status"], f"child {ch['ticket']}")
    j.gate("G1", r, rs["before"]["authorized"] == L[e["authority"]],
           f"continuity {rs['before']['authorized']} expected {e['authority']}")
    j.gate("G3", r, rs["before"]["generation"] == ch["anchor"] + GEN[e["authority"]],
           f"generation {rs['before']['generation']}")
    _format(j, r, rs["continuity_raw"], rs["before"], FORMAT["k1"])
    if c["point"] == "step:6":
        _torn_record(j, r, ch["files_before"]["continuity"], rs["continuity_raw"], c)
    j.gate("G4", r, rs["observed"]["disposition"] == e["disposition"] and rs["observed"]["candidate"] is None,
           f"restart {rs['observed']['disposition']}")
    first = ch["first_ticket"]
    j.gate("G4", r, (first["kind"], first["status"], first["recoverable"], first["identity"])
           == ("IDENTITY", "IDENTITY_PUBLISHED", False, L["D1"]), f"first ticket {first}")
    j.gate("G4", r, ch["ticket"]["status"] != "ACKNOWLEDGED" and ch["ticket"]["recoverable"] is False,
           f"ticket {ch['ticket']}")
    j.gate("G3", r, rs["caller_bytes"]["authorized"] == [e["authority"]]
           and rs.get("wrong_bytes_refusal") == "CONTINUITY_STATE_MISMATCH",
           f"caller bytes {rs['caller_bytes']} {rs.get('wrong_bytes_refusal')}")
    j.gate("G3", r, rs["before"]["authorized"] != L["D3"], "volatile state never named")
    continued = rs["continued"]
    j.gate("G2", r, continued["ticket"]["status"] == "ACKNOWLEDGED" and rs["after"]["disposition"] == "RESUMABLE"
           and rs["after"]["authorized"] == continued["identity"]
           and rs["after"]["generation"] == rs["before"]["generation"] + 1, f"continued {continued}")
    _footprint(j, r, rs["provisioned"], rs["final_footprint"], plan["lineages"]["k1"]["envelope_bytes"],
               plan["slot_header_bytes"])


def check_identity_refused(j: Judge, plan: dict, r: dict, c: dict) -> None:
    e = expectation(plan, "identity_refused", c["point"])
    ch, rs = r["child"], r["restart"]
    L = _labels(ch)
    _exited(j, r, "child_exit")
    j.gate("G4", r, ch["refusal"] == e["refusal"] and ch["recoverable_refusal"] == e["recoverable_refusal"],
           f"refusals {ch['refusal']} {ch['recoverable_refusal']}")
    durable = ch["durable_after"]
    j.gate("G4", r, (durable["authorized"], durable["generation"], durable["format"])
           == (L[e["authority"]], ch["anchor"], "COMPOSITE"), f"the refused publication moved continuity: {durable}")
    j.gate("G4", r, ch["last_acknowledged"]["status"] in ("NONE", None) and ch["last_acknowledged"]["recoverable"]
           is False, f"an acknowledgment exists: {ch['last_acknowledged']}")
    j.gate("G4", r, rs["continuity_raw"] == ch["files_before"]["continuity"], "the continuity records changed")
    _format(j, r, rs["continuity_raw"], rs["before"], "COMPOSITE")
    j.gate("G4", r, rs["before"]["authorized"] == L[e["authority"]] and rs["before"]["generation"] == ch["anchor"],
           f"restart {rs['before']}")
    o = rs["observed"]
    j.gate("G4", r, o["disposition"] == e["disposition"] and o["candidate"] is None
           and all(v is None for v in o["slots"].values()), f"a candidate or slot exists: {o}")


def check_anchor(j: Judge, plan: dict, r: dict, c: dict) -> None:
    e = expectation(plan, "anchor", c["point"])
    ch, rs = r["child"], r["restart"]
    d0, o = ch["D0"], rs["observed"]
    _killed(j, r)
    j.gate("G1", r, ch["control"] == {}, f"not parked inside the anchor {ch['control']}")
    j.gate("G1", r, o["disposition"] == e["disposition"], f"disposition {o['disposition']} expected {e['disposition']}")
    held = any(s and s["identity"] == d0 and s["verified"] for s in o["slots"].values())
    j.gate("G1", r, held == e["slot_holds_d0"], f"slots {o['slots']}")
    _format(j, r, rs["continuity_raw"], o, FORMAT[c["lineage"]])
    if c["point"] == "control:4":
        _torn_record(j, r, ch["files_before"]["continuity"], rs["continuity_raw"], c)
    anchored = e["disposition"] == "RESUMABLE"
    j.gate("G1", r, o["authorized"] == (d0 if anchored else None), f"authorized {o['authorized']}")
    j.gate("G3", r, rs["action"] == (None if anchored else "anchor"), f"action {rs['action']}")
    j.gate("G3", r, o["generation"] == GENESIS_GENERATION + (1 if anchored else 0)
           and rs["after_action"]["generation"] == GENESIS_GENERATION + 1
           and rs["after_action"]["authorized"] == d0 and rs["final"]["generation"] == GENESIS_GENERATION + 2,
           f"generations {o['generation']} {rs['after_action']['generation']} {rs['final']['generation']}")
    _cross_format(j, r, rs, "anchor")
    _resumed(j, r, rs["resumed"], d0, [], set(), c["lineage"] == "composite")
    _footprint(j, r, r["provisioned"], rs["final_footprint"], plan["lineages"][c["lineage"]]["envelope_bytes"],
               plan["slot_header_bytes"])


def check_import(j: Judge, plan: dict, r: dict, c: dict) -> None:
    e = expectation(plan, "import", c["point"])
    ch, rs = r["child"], r["restart"]
    L = _labels(ch)
    oc, ok1 = rs["observed_composite_store"], rs["observed_k1_store"]
    _killed(j, r)
    j.gate("G1", r, ch["control"] == {}, f"not parked inside the import {ch['control']}")
    j.gate("G4", r, (ch["first_ticket"]["status"], ch["first_ticket"]["identity"]) == ("ACKNOWLEDGED", L["K"]),
           f"first ticket {ch['first_ticket']}")
    j.gate("G1", r, (oc["format"], oc["authorized"]) == (e["format"], L[e["authority"]]),
           f"continuity {oc['format']} {oc['authorized']}")
    j.gate("G1", r, (oc["disposition"], ok1["disposition"]) == (e["composite_store"], e["k1_store"]),
           f"composite store {oc['disposition']} K1 store {ok1['disposition']}")
    _format(j, r, rs["continuity_raw"], oc, e["format"])
    if c["point"] == "control:4":
        _torn_record(j, r, ch["files_before"]["continuity"], rs["continuity_raw"], c)
    pending = e["format"] == "K1"
    j.gate("G3", r, rs["action"] == ("import" if pending else None), f"action {rs['action']}")
    if pending:
        j.gate("G3", r, rs.get("reimported") == {"identity": L["C"], "format": "COMPOSITE"},
               f"reimport {rs.get('reimported')}")
    base = ch["anchor"] + 1   # K published once after the anchor
    j.gate("G3", r, oc["generation"] == base + (0 if pending else 1)
           and rs["after_action"]["generation"] == base + 1 and rs["after_action"]["format"] == "COMPOSITE"
           and rs["final"]["generation"] == base + 2, "import generations")
    _resumed(j, r, rs["resumed"], L["C"], [], set(), True)   # an imported composite holds an empty memory
    _footprint(j, r, r["provisioned"], rs["final_footprint"], plan["lineages"]["k1"]["envelope_bytes"],
               plan["slot_header_bytes"])
    _footprint(j, r, r["provisioned_composite"], rs["final_footprint_composite"],
               plan["lineages"]["composite"]["envelope_bytes"], plan["slot_header_bytes"])


def check_random(j: Judge, plan: dict, r: dict, c: dict) -> None:
    rs, events = r["restart"], r["events"]
    order, keys = {}, []
    anchor = next(e["identity"] for e in events if e.get("event") == "anchor")
    order[anchor] = -1
    for e in events:
        if e.get("event") == "learn":
            order[e["identity"]] = e["i"]
            keys.append(e["key"])
    requested = {e["identity"] for e in events if e.get("event") == "request"}
    acked = [e["identity"] for e in events if e.get("event") == "ticket" and e["status"] == "ACKNOWLEDGED"]
    acked += [e["requested"] for e in events if e.get("event") == "resolved" and e["status"] == "ACKNOWLEDGED"]
    never = {e["requested"] for e in events if e.get("event") == "resolved" and e["status"] in ("SUPERSEDED", "FAILED")}
    resolved_ok = all(e["identity"] == e["requested"] for e in events
                      if e.get("event") == "resolved" and e["status"] == "ACKNOWLEDGED")
    tickets_ok = all(e["recoverable"] is (e["status"] == "ACKNOWLEDGED") for e in events
                     if e.get("event") in ("ticket", "resolved"))
    j.gate("G4", r, tickets_ok and resolved_ok, "acknowledgment law of every delivered or resolved ticket")
    floor = max((order[a] for a in acked), default=-1)
    o = rs["observed"]
    authorized = o["authorized"]
    _killed(j, r)
    j.gate("G1", r, o["disposition"] in ("RESUMABLE", "CANDIDATE_UNRESOLVED"), f"disposition {o['disposition']}")
    _format(j, r, rs["continuity_raw"], o, FORMAT[c["lineage"]])
    j.gate("G3", r, authorized == anchor or authorized in requested, "never a volatile state never requested")
    j.gate("G3", r, authorized not in never, "a superseded or failed request became the authority")
    j.gate("G1", r, authorized in order and order[authorized] >= floor,
           f"authorized at {order.get(authorized)}, last acknowledgment at {floor}")
    unresolved = o["disposition"] == "CANDIDATE_UNRESOLVED"
    if unresolved:
        cand = o["candidate"]
        j.gate("G3", r, cand in requested and cand not in never and order.get(cand, -2) > order[authorized],
               f"candidate {cand}")
    action = operator_action("random", c["run_index"]) if unresolved else None
    j.gate("G3", r, rs["action"] == action, f"action {rs['action']}")
    final = o["candidate"] if action == "adopt" else authorized
    j.gate("G3", r, rs["after_action"]["authorized"] == final, f"after the action {rs['after_action']['authorized']}")
    j.gate("G3", r, rs["final"]["generation"] == rs["after_action"]["generation"] + 1, "continuation published once")
    upto = order.get(final, -1)
    _resumed(j, r, rs["resumed"], final, keys, {k for i, k in enumerate(keys) if i <= upto and k is not None},
             c["lineage"] == "composite")
    _footprint(j, r, r["provisioned"], rs["final_footprint"], plan["lineages"][c["lineage"]]["envelope_bytes"],
               plan["slot_header_bytes"])


CHECKS = {"publish": check_store_family, "discard": check_store_family, "adopt": check_store_family,
          "uncertain": check_store_family, "identity": check_identity, "identity_refused": check_identity_refused,
          "anchor": check_anchor, "import": check_import, "random": check_random}
COORDINATE_KEYS = ("index", "family", "lineage", "point", "run_index", "seed", "inputs", "arg", "rng_seed",
                   "kill_after")


def evaluate(plan: dict, records: list, *, inputs: str, runs_override: int | None = None,
             families: set | None = None) -> dict:
    """The verdict. ``runs_override`` and ``families`` describe a DEV subset (completeness is then judged against
    that subset; a QUAL verdict uses neither)."""
    j = Judge()
    expected = [c for c in coordinates(plan, inputs, runs_override) if families is None or c["family"] in families]
    by_index = {c["index"]: c for c in expected}
    seen, last = set(), -1
    for r in records:
        c = by_index.get(r.get("index"))
        ok = (c is not None and r.get("schema") == RECORD_SCHEMA and r["index"] not in seen and r["index"] > last
              and all(r.get(k) == c.get(k) for k in COORDINATE_KEYS))
        j.gate("G1", r, ok, f"not the plan's trial at its coordinate (or duplicated or out of order): "
                            f"{ {k: r.get(k) for k in COORDINATE_KEYS} }")
        seen.add(r.get("index"))
        last = max(last, r.get("index") if isinstance(r.get("index"), int) else last)
        if not ok:
            continue
        if r.get("child_error") or r.get("reconcile_error") or r.get("restart_error") or not r.get("restart"):
            j.gate("G1", r, False, "incomplete trial: " + str(r.get("child_error") or r.get("reconcile_error")
                                                                or r.get("restart_error"))[:300])
            continue
        try:
            CHECKS[c["family"]](j, plan, r, c)
        except (KeyError, TypeError, StopIteration, AttributeError, ValueError) as exc:
            j.gate("G1", r, False, f"malformed record: {exc!r}")
    complete = seen == set(by_index) and len(records) == len(by_index)
    gates = {g: {"checks": j.checked[g], "failures": j.failures[g],
                 "verdict": "PASS" if j.checked[g] and not j.failures[g] else "FAIL"} for g in GATES}
    verdict = "PASS" if complete and all(v["verdict"] == "PASS" for v in gates.values()) else "FAIL"
    return {"schema": VERDICT_SCHEMA, "inputs": inputs, "records": len(records), "planned": len(by_index),
            "complete": complete, "missing": sorted(set(by_index) - seen), "gates": gates, "verdict": verdict}


# =====================================================================================================================
# The sealed evidence directory
# =====================================================================================================================

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_evidence(directory: Path, plan_path: Path, *, inputs: str = "qual") -> list:
    """Every problem with a sealed evidence directory (empty: complete, intact, bound and recomputed). Evidence is a
    complete QUAL run; ``inputs="dev"`` exists only so a sealed full DEV rehearsal exercises every other check before
    any QUAL data exists (its records are never evidence)."""
    problems = []
    present = sorted(p.name for p in directory.iterdir())
    if present != sorted(EVIDENCE_FILES):
        return [f"the evidence directory holds {present}, not exactly {sorted(EVIDENCE_FILES)}"]
    listed = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, _, name = line.partition("  ")
        listed[name] = digest
    if sorted(listed) != sorted(set(EVIDENCE_FILES) - {"SHA256SUMS"}):
        problems.append(f"SHA256SUMS lists {sorted(listed)}")
    for name, digest in listed.items():
        if (directory / name).is_file() and _sha(directory / name) != digest:
            problems.append(f"{name}: SHA-256 differs from SHA256SUMS")
    try:
        freeze = json.loads((directory / "freeze.json").read_text())
        run = json.loads((directory / "run.json").read_text())
        verdict = json.loads((directory / "verdict.json").read_text())
        records = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        return problems + [f"an evidence file is unreadable: {exc}"]
    if not all(isinstance(d, dict) for d in (freeze, run, verdict)):
        return problems + ["an evidence document is not a JSON object"]
    if freeze.get("schema") != FREEZE_SCHEMA or run.get("schema") != RUN_SCHEMA or verdict.get("schema") != \
            VERDICT_SCHEMA:
        problems.append("a schema is not the registered one")
    if run.get("inputs") != inputs or run.get("dev_runs") is not None or run.get("only") is not None:
        problems.append(f"the run is not a complete {inputs.upper()} run")
    if run.get("freeze_sha256") != _sha(directory / "freeze.json"):
        problems.append("the run does not name this freeze record")
    if run.get("records_sha256") != _sha(directory / "records.jsonl"):
        problems.append("the run does not name these records")
    if run.get("post_run_problems"):
        problems.append(f"the checkout changed during the run: {run['post_run_problems']}")
    observed = run.get("observed") or {}
    for key in ("source", "instrument", "subject", "native"):
        want, got = freeze.get(key) or {}, observed.get(key) or {}
        if key == "source":
            want, got = {k: want.get(k) for k in ("commit", "tree")}, {k: got.get(k) for k in ("commit", "tree")}
        if not want or want != got:
            problems.append(f"the run's observed {key} is not the freeze's")
    if _sha(plan_path) != ((freeze.get("instrument") or {}).get("files") or {}).get("plan.json"):
        problems.append("the plan is not the frozen plan")
    plan = json.loads(plan_path.read_text())
    recomputed = evaluate(plan, records, inputs=inputs)
    for key in ("records", "planned", "complete", "missing", "gates", "verdict"):
        if recomputed[key] != verdict.get(key):
            problems.append(f"the verdict's {key} is not what the records give")
    return problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="check.py")
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("records")
    v.add_argument("--inputs", choices=("dev", "qual"), required=True)
    v.add_argument("--plan", default=str(HERE / "plan.json"))
    v.add_argument("--runs", type=int, default=None, help="DEV subsets only")
    v.add_argument("--families", default=None, help="DEV subsets only")
    v.add_argument("--out", default=None)
    e = sub.add_parser("evidence")
    e.add_argument("directory")
    e.add_argument("--plan", default=str(HERE / "plan.json"))
    args = p.parse_args(argv)
    if args.command == "evidence":
        problems = verify_evidence(Path(args.directory), Path(args.plan))
        print(json.dumps({"evidence": args.directory, "valid": not problems, "problems": problems}))
        return 0 if not problems else 1
    if args.inputs == "qual" and (args.runs or args.families):
        raise SystemExit("a QUAL verdict judges the complete plan")
    plan_bytes = Path(args.plan).read_bytes()
    records = [json.loads(line) for line in Path(args.records).read_text().splitlines() if line.strip()]
    result = evaluate(json.loads(plan_bytes), records, inputs=args.inputs, runs_override=args.runs,
                      families=set(args.families.split(",")) if args.families else None)
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
