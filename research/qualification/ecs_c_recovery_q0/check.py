"""ECS_C Recovery Q0: the independent checker.

    python -I -B check.py verdict records.jsonl --inputs dev|qual --out verdict.json
    python -I -B check.py evidence EVIDENCE_DIR

``verdict`` judges the gates G1-G6 from plan.json and the records alone. It imports neither Elpis nor the runner: it
re-derives the enumeration (every trial exactly once, in order, with its seed and random draws) from the plan's own
coordinate rule, and every expectation from the plan's points. Identities are the records' own values, named by the
plan's symbols:
- D0, the identity `create` reports;
- D1..D6, the durable identities of the trial's acknowledged publications, in order;
- D2 in the crash families, the live identity the child reported last before the fault.

A record the runner could not complete (``instrument_error``) or a missing or extra record is a **mechanical**
failure, reported apart from the gates: the verdict is then INCOMPLETE, never PASS or NONPASS. ``evidence`` checks
a sealed directory: exactly its files, every digest, and the verdict recomputed from its records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "plan.json").read_text())
VERDICT_SCHEMA = "elpis.ecs-c-recovery-q0.verdict.v1"
RECORD_SCHEMA = "elpis.ecs-c-recovery-q0.record.v1"
RUN_SCHEMA = "elpis.ecs-c-recovery-q0.run.v1"
GATES = ("G1", "G2", "G3", "G4", "G5", "G6")
PROCESS_DEATH = ("publish", "failure", "anchor", "discard", "adopt", "random")


# ---- the enumeration, re-derived ----------------------------------------------------------------------------------

def expected_trials(inputs: str, plan: dict = PLAN) -> list:
    base = plan["inputs"][inputs]["lineage_seed_base"]
    rng = random.Random(plan["inputs"][inputs]["random_seed"])
    out, index = [], 0
    for family, f in plan["families"].items():
        single = family in ("random", "bounded")
        points = [{"point": family}] if single else f["points"]
        runs = f["runs"] if single else plan["runs_per_point"]
        for p in points:
            for run in range(runs):
                t = {"index": index, "family": family, "point": p["point"], "run": run, "seed": base + index,
                     "spec": p, "random": None}
                if family == "random":
                    turns = rng.randint(1, f["max_turns"])
                    publish = [rng.random() < f["publish_probability"] for _ in range(turns)]
                    lines = 2 * turns + sum(publish)
                    t["random"] = {"publish": publish, "kill_after": rng.randint(1, lines - 1) if lines > 1 else 1}
                out.append(t)
                index += 1
    return out


# ---- reading records ----------------------------------------------------------------------------------------------

def lines(run) -> list:
    return (run or {}).get("lines") or []


def first(run) -> dict:
    out = lines(run)
    return out[0] if out else {}


def ok_all(run) -> bool:
    return bool(run) and run.get("rc") == 0 and bool(lines(run)) and all(o.get("ok") for o in lines(run))


def report(seen) -> dict:
    return first(seen["authority"]).get("report") or {}


def recovery(seen) -> dict:
    return first(seen["status"]).get("recovery") or {}


def durable_identity(seen):
    return (first(seen["status"]).get("durable") or {}).get("identity")


def last_identity(child_lines, before_op=None):
    """The live identity the child reported last (learn and evidence lines carry it)."""
    ident = None
    for o in child_lines:
        if before_op and o.get("op") == before_op:
            break
        if o.get("op") in ("learn", "evidence") and o.get("ok"):
            ident = o.get("identity")
    return ident


def acked(child_lines) -> list:
    return [o["durable"]["identity"] for o in child_lines
            if o.get("op") == "publish" and o.get("ok") and (o.get("ticket") or {}).get("acknowledged")]


class Judge:
    def __init__(self):
        self.failures = {g: [] for g in GATES}

    def check(self, gate: str, condition: bool, t: dict, what: str):
        if not condition:
            self.failures[gate].append(f"#{t['index']} {t['family']} {t['point']} run {t['run']}: {what}")


# ---- shared checks ------------------------------------------------------------------------------------------------

def check_ack_law(j: Judge, t: dict, observed: dict):
    """G4: no ticket violates the acknowledgment law, anywhere in the record."""
    def walk(x):
        if isinstance(x, dict):
            if x.get("op") == "publish" and "ticket" in x:
                ticket = x["ticket"] or {}
                ack = ticket.get("acknowledged") is True
                j.check("G4", ack == (ticket.get("status") == "ACKNOWLEDGED"), t,
                        "a ticket acknowledged without the ACKNOWLEDGED status, or the reverse")
                if ack:
                    j.check("G4", ticket.get("identity") == (x.get("durable") or {}).get("identity"), t,
                            "an acknowledged ticket that does not name the durable state")
                if x.get("ok"):
                    # Waited: ACKNOWLEDGED, or refused. Not waited: still PENDING or RUNNING.
                    j.check("G4", ticket.get("status") in ("ACKNOWLEDGED", "PENDING", "RUNNING"), t,
                            f"a publish reported ok with a {ticket.get('status')} ticket")
                else:
                    j.check("G4", not ack, t, "a refused publish acknowledged")
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(observed)


def check_candidate_action(j: Judge, t: dict, action: dict, want_action: str, authorized, candidate):
    """G4 and G2: the unresolved candidate refuses a publication; then exactly the named reconciliation."""
    j.check("G4", action.get("taken") == want_action, t, f"reconciliation {action.get('taken')}, not {want_action}")
    probe = first(action.get("probe"))
    j.check("G4", probe.get("op") == "publish" and probe.get("ok") is False, t,
            "a recoverable publication succeeded over an unresolved candidate")
    result = lines(action.get("result"))
    live = result[1] if len(result) > 1 else {}
    want = candidate if want_action == "adopt" else authorized
    j.check("G2", bool(result) and result[0].get("ok") and (live.get("live") or {}).get("identity") == want, t,
            f"after {want_action} the live state is not {want}")


def check_continuation(j: Judge, t: dict, observed: dict, evidence_expected):
    c = observed.get("continuation")
    if c is None:
        j.check("G2", False, t, "no continuation")
        return
    pub = lines(c["published"])
    published = pub[-1] if pub else {}
    ident = (published.get("durable") or {}).get("identity")
    j.check("G2", ok_all(c["published"]) and (published.get("ticket") or {}).get("acknowledged") is True, t,
            "the continuation's publication was not acknowledged")
    live = first(c["resumed"])
    j.check("G2", (live.get("live") or {}).get("identity") == ident and
            (live.get("durable") or {}).get("identity") == ident, t,
            "a fresh process does not resume exactly the continuation's publication")
    a = first(c["authority"])
    j.check("G2", (a.get("report") or {}).get("disposition") == "CLEAN" and a.get("reconciliation") is None, t,
            "the authority is not CLEAN after the continuation")
    if evidence_expected is not None:
        got = (a.get("authority") or {}).get("evidence_count")
        j.check("G2", got == evidence_expected, t, f"evidence count {got}, the plan says {evidence_expected}")
    check_files(j, t, observed, c["files"])


def check_files(j: Judge, t: dict, observed: dict, final: dict):
    """G5: exactly the 17 files, at their provisioned sizes and inodes."""
    names = sorted(PLAN["deployment"]["files"])
    j.check("G5", sorted(final) == names, t, f"the home holds {sorted(set(final) ^ set(names))} unexpectedly")
    baseline = observed.get("baseline") or (observed.get("deploy") or {}).get("baseline")
    j.check("G5", baseline is not None and final == baseline, t,
            "a file's size or inode differs from the provisioned file set")
    if final.get("state/authority.a"):
        j.check("G5", final["state/authority.a"][0] == PLAN["deployment"]["record_bytes"] and
                final["authority/witness.a"][0] == PLAN["deployment"]["witness_bytes"], t,
                "an authority slot is not its fixed size")


def check_process_death_observation(j: Judge, t: dict, seen: dict, e: dict, symbols: dict):
    rep, rec = report(seen), recovery(seen)
    j.check("G1", rep.get("disposition") == e["authority"], t,
            f"authority {rep.get('disposition')}, the crash law says {e['authority']}")
    j.check("G1", first(seen["authority"]).get("reconciliation") is None, t,
            "a process death needs authority reconciliation")
    j.check("G1", first(seen["status"]).get("session_refusal") is None, t, "the runtime refused to open")
    j.check("G1", rec.get("disposition") == e["recovery"], t,
            f"recovery {rec.get('disposition')}, the crash law says {e['recovery']}")
    if "durable" in e:
        j.check("G2", durable_identity(seen) == symbols.get(e["durable"]) is not None, t,
                f"durable {durable_identity(seen)} is not {e['durable']}")
    if e["recovery"] == "CANDIDATE_UNRESOLVED":
        j.check("G1", rec.get("candidate") == symbols.get(e.get("candidate")) is not None, t,
                f"candidate {rec.get('candidate')} is not {e.get('candidate')}")
    elif e["recovery"] == "RESUMABLE":
        j.check("G1", rec.get("candidate") is None, t, "an unexpected candidate")


# ---- families ------------------------------------------------------------------------------------------------------

def _crash_symbols(o: dict, child_lines: list, fault_op: str) -> dict:
    return {"D0": first(o["deploy"].get("create")).get("identity"),
            "D1": (acked(child_lines) or [None])[0],
            "D2": last_identity(child_lines, before_op=fault_op)}


def distinct(j: Judge, t: dict, symbols: dict):
    """G2: the symbols name distinct states, so no expectation over them holds vacuously."""
    values = [v for v in symbols.values()]
    j.check("G2", None not in values and len(set(values)) == len(values), t,
            f"the trial's states are not distinct identities: {symbols}")


def judge_publish(j, t, o, e):
    child = lines(o["child"])
    j.check("G1", o["child"].get("reached") and o["child"].get("signal") == 9, t, "the child was not killed parked")
    symbols = _crash_symbols(o, child, "testing_durable_fault")
    distinct(j, t, symbols)
    check_process_death_observation(j, t, o["observe"], e, symbols)
    if e["recovery"] == "CANDIDATE_UNRESOLVED":
        check_candidate_action(j, t, o["action"], "adopt" if t["run"] % 2 == 0 else "discard", symbols["D1"],
                               symbols["D2"])
    else:
        j.check("G4", o["action"].get("taken") is None, t, "a reconciliation without a candidate")
    check_continuation(j, t, o, e.get("evidence"))


def judge_failure(j, t, o, e):
    child = lines(o["child"])
    failed = child[-1] if child else {}
    ticket = failed.get("ticket") or {}
    j.check("G1", o["child"].get("rc") == 3 and failed.get("op") == "publish" and failed.get("code") == e["code"], t,
            f"the failed publication is {failed.get('code')}, not {e['code']}")
    j.check("G1", ticket.get("code") == e["cause"] and ticket.get("acknowledged") is False, t,
            f"the ticket's cause is {ticket.get('code')}, not {e['cause']}")
    symbols = _crash_symbols(o, child, "testing_durable_fault")
    distinct(j, t, symbols)
    check_process_death_observation(j, t, o["observe"], e, symbols)
    if e["recovery"] == "CANDIDATE_UNRESOLVED":
        check_candidate_action(j, t, o["action"], "adopt" if t["run"] % 2 == 0 else "discard", symbols["D1"],
                               symbols["D2"])
    else:
        j.check("G4", o["action"].get("taken") is None, t, "a reconciliation without a candidate")
    check_continuation(j, t, o, e.get("evidence"))


def judge_anchor(j, t, o, e):
    point = int(t["point"].split(":")[1])
    child = lines(o["child"])
    j.check("G1", o["child"].get("reached") and child and child[-1].get("parked") == 100 + point, t,
            "the anchor was not killed parked at its point")
    created = first(o["action"]["result"])
    d0 = created.get("identity")
    check_process_death_observation(j, t, o["observe"], e, {"D0": d0})
    j.check("G4", o["action"]["result"].get("rc") == 0 and created.get("ok") and created.get("resumed") is e["resumed"],
            t, f"the same create did not complete the lineage (resumed {created.get('resumed')})")
    other = first(o["action"]["other_seed"])
    j.check("G4", o["action"]["other_seed"].get("rc") == 3 and other.get("code") == "LINEAGE_EXISTS", t,
            "another lineage's create was not refused")
    check_continuation(j, t, o, e.get("evidence"))


def judge_control(j, t, o, e, kind, own):
    point = int(t["point"].split(":")[1])
    first_child = lines(o["child"])
    symbols = _crash_symbols(o, first_child, "testing_durable_fault")
    distinct(j, t, symbols)
    pre = recovery(o["observe_candidate"])
    j.check("G1", pre.get("disposition") == "CANDIDATE_UNRESOLVED" and pre.get("candidate") == symbols["D2"], t,
            "the step-5 kill left no candidate")
    control = lines(o["control"])
    j.check("G1", o["control"].get("reached") and control and control[-1].get("parked") == 100 * kind + point, t,
            "the reconciliation was not killed parked at its point")
    check_process_death_observation(j, t, o["observe"], e, symbols)
    if e["recovery"] == "CANDIDATE_UNRESOLVED":
        check_candidate_action(j, t, o["action"], own, symbols["D1"], symbols["D2"])
    check_continuation(j, t, o, e.get("evidence"))


def judge_discard(j, t, o, e):
    judge_control(j, t, o, e, 3, "discard")


def judge_adopt(j, t, o, e):
    judge_control(j, t, o, e, 4, "adopt")


def judge_random(j, t, o, f):
    e = f["expect"]
    script_ops = []
    for publish in t["random"]["publish"]:
        script_ops += ["learn", "evidence"] + (["publish"] if publish else [])
    child = lines(o["child"])
    j.check("G1", o["child"].get("signal") == 9 or len(child) == len(script_ops), t, "the child was not killed")
    d0 = first(o["deploy"].get("create")).get("identity")
    ack = acked(child)
    last_ack = ack[-1] if ack else d0
    allowed = {last_ack}
    in_flight = len(child) < len(script_ops) and script_ops[len(child)] == "publish"
    flight_identity = last_identity(child) if in_flight else None
    if in_flight and flight_identity:
        allowed.add(flight_identity)
    seen = o["observe"]
    rep, rec = report(seen), recovery(seen)
    j.check("G1", rep.get("disposition") in e["authority_any_of"], t, f"authority {rep.get('disposition')}")
    j.check("G1", first(seen["authority"]).get("reconciliation") is None, t, "a process death needs reconciliation")
    j.check("G1", rec.get("disposition") in e["recovery_any_of"], t, f"recovery {rec.get('disposition')}")
    j.check("G2", durable_identity(seen) in allowed, t, "the durable state is not the last acknowledged publication "
            "or the one in flight: an acknowledged publication was lost")
    if rec.get("disposition") == "CANDIDATE_UNRESOLVED":
        reported = {x.get("identity") for x in child if x.get("op") in ("learn", "evidence")}
        j.check("G1", rec.get("candidate") in reported and rec.get("candidate") != durable_identity(seen), t,
                "the candidate is not a state the child reported")
        check_candidate_action(j, t, o["action"], "adopt" if t["run"] % 2 == 0 else "discard",
                               durable_identity(seen), rec.get("candidate"))
    check_continuation(j, t, o, None)


def _history_symbols(o: dict) -> dict:
    out = {"D0": first(o["deploy"].get("create")).get("identity")}
    for k, run in enumerate(o["history"], start=1):
        ident = acked(lines(run))
        out[f"D{k}"] = ident[0] if ident else None
    return out


def judge_attack(j, t, o, e):
    symbols = _history_symbols(o)
    j.check("G2", all(ok_all(r) for r in o["history"]) and None not in symbols.values(), t,
            "the history's publications were not all acknowledged")
    distinct(j, t, symbols)
    before = first(o["before_attack"]).get("authority") or {}
    seen = o["observe"]
    status, auth = first(seen["status"]), first(seen["authority"])
    rec_ = auth.get("reconciliation") or {}
    j.check("G3", (status.get("session_refusal") or {}).get("code") == e["open"], t,
            f"the runtime opened ({status.get('session_refusal')})")
    j.check("G3", report(seen).get("disposition") == e["authority"], t,
            f"authority {report(seen).get('disposition')}, the law says {e['authority']}")
    j.check("G3", rec_.get("kind") == e["reconcile"], t, f"the authority names {rec_.get('kind')}")
    if "class" in e:
        j.check("G3", rec_.get("class") == e["class"], t, f"class {rec_.get('class')}, the law says {e['class']}")
    if "witness_classes" in e:
        j.check("G3", rec_.get("witness_classes") == e["witness_classes"], t, "the witness classes")
    if e["reconcile"] == "ROLLBACK":
        j.check("G3", rec_.get("abandoned_generation") == before.get("generation"), t,
                "the abandoned generation is not the acknowledged one")
    steps = o["reconcile"]
    if t["run"] % 2 == 1:
        wrong, after_wrong = first(steps[0]), first(steps[1])
        j.check("G3", wrong.get("ok") is False and wrong.get("code") == "AUTHORITY_RECONCILE_MISMATCH", t,
                "a wrong reconciliation was not refused")
        j.check("G3", after_wrong.get("reconciliation") == auth.get("reconciliation") and
                (after_wrong.get("report") or {}).get("slot_generations") == report(seen).get("slot_generations"), t,
                "a refused reconciliation changed something")
    exact = first(steps[-1])
    j.check("G3", steps[-1].get("rc") == 0 and exact.get("ok"), t, "the exact reconciliation was refused")
    if e["reconcile"] == "ROLLBACK":
        j.check("G3", exact.get("generation") == (rec_.get("abandoned_generation") or -9) + 1, t,
                "the rollback acceptance did not continue at the abandoned generation + 1")
    after = o["after_reconcile"]
    j.check("G3", report(after).get("disposition") == "CLEAN" and first(after["authority"]).get("reconciliation") is None,
            t, f"after reconciliation the authority is {report(after).get('disposition')}")
    j.check("G3", durable_identity(after) == symbols.get(e["accepted"]), t,
            f"the authority continues from {durable_identity(after)}, not the accepted {e['accepted']}")
    rec = recovery(after)
    j.check("G3", rec.get("disposition") == e["recovery"], t, f"checkpoint {rec.get('disposition')}, not {e['recovery']}")
    if e["recovery"] == "CANDIDATE_UNRESOLVED":
        j.check("G3", rec.get("candidate") == symbols.get(e["candidate"]), t, "the checkpoint candidate")
        check_candidate_action(j, t, o["action"], "adopt" if t["run"] % 2 == 0 else "discard",
                               symbols[e["accepted"]], symbols[e["candidate"]])
    if e["recovery"] == "CHECKPOINT_MISSING":
        final = o["final"]
        j.check("G3", first(final["learn"]).get("ok") is False, t, "cognition proceeded without its authorized state")
        j.check("G2", (first(final["authority"]).get("authority") or {}).get("evidence_count") == e["evidence"], t,
                "the accepted record's evidence count")
        check_files(j, t, o, final["files"])
    else:
        check_continuation(j, t, o, e.get("evidence"))


judge_witness = judge_attack


def judge_rearrangement(j, t, o, e):
    symbols = _history_symbols(o)
    distinct(j, t, symbols)
    seen = o["observe"]
    j.check("G3", report(seen).get("disposition") == e["authority"], t,
            f"authority {report(seen).get('disposition')}, the law says {e['authority']}")
    j.check("G3", first(seen["authority"]).get("reconciliation") is None and
            first(seen["status"]).get("session_refusal") is None, t, "an accepted rearrangement needed the operator")
    j.check("G3", recovery(seen).get("disposition") == e["recovery"] and
            durable_identity(seen) == symbols.get(e["durable"]), t, "the acknowledged state was not resumed exactly")
    check_continuation(j, t, o, e.get("evidence"))


def judge_bounded(j, t, o, f):
    marks = o["marks"]
    j.check("G6", [m.get("publications") for m in marks] == f["publications"] and
            all("authority" in m for m in marks), t, "a block of publications failed")
    works = []
    for m in marks:
        if "authority" not in m:
            continue
        a = first(m["authority"])
        r = a.get("report") or {}
        works.append((r.get("reads"), r.get("read_bytes"), r.get("replays")))
        j.check("G6", (a.get("authority") or {}).get("publications") == 1 + m["publications"], t,
                "the authority's publication count")
        j.check("G6", (a.get("authority") or {}).get("evidence_count") == m["publications"], t,
                "the evidence count of the history")
        j.check("G5", sorted(m["files"]) == sorted(PLAN["deployment"]["files"]) and
                m["files"] == o["deploy"]["baseline"], t, f"the file set after {m['publications']} publications")
    j.check("G6", len(set(works)) == 1, t, f"recovery work differs with history length: {works}")


JUDGES = {"publish": judge_publish, "failure": judge_failure, "anchor": judge_anchor, "discard": judge_discard,
          "adopt": judge_adopt, "random": judge_random, "attack": judge_attack, "rearrangement": judge_rearrangement,
          "witness": judge_witness, "bounded": judge_bounded}


# ---- the verdict ---------------------------------------------------------------------------------------------------

def read_records(path: Path) -> list:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verdict(records_path: Path, inputs: str) -> dict:
    return evaluate(read_records(records_path), inputs)


def evaluate(rows: list, inputs: str, plan: dict = PLAN) -> dict:
    """The verdict of a run's rows (its header line, then its records) under ``plan``."""
    if not rows:
        return {"schema": VERDICT_SCHEMA, "inputs": inputs, "plan": plan["schema"], "trials": 0, "judged": 0,
                "subset": False, "mechanical": ["no records"], "gates": {g: {"pass": False, "failures": []}
                                                                         for g in GATES}, "verdict": "INCOMPLETE"}
    head, records = rows[0], rows[1:]
    mechanical = []
    if head.get("schema") != RUN_SCHEMA or head.get("inputs") != inputs:
        mechanical.append("the records are not a run of these inputs")
    expected = expected_trials(inputs, plan)
    families = set(head["families"]) if head.get("families") else None
    runs = head.get("runs")
    expected = [t for t in expected
                if (families is None or t["family"] in families) and (runs is None or t["run"] < runs)]
    subset = families is not None or runs is not None
    if [r.get("index") for r in records] != [t["index"] for t in expected]:
        mechanical.append("the records are not exactly the planned trials, in order")
    j = Judge()
    judged = 0
    for t, r in zip(expected, records):
        if r.get("schema") != RECORD_SCHEMA or any(r.get(k) != t[k] for k in ("family", "point", "run", "seed")) \
                or r.get("random") != t["random"]:
            mechanical.append(f"#{t['index']}: the record is not the planned trial")
            continue
        if r.get("instrument_error") or r.get("observed") is None:
            mechanical.append(f"#{t['index']}: instrument error: {r.get('instrument_error')}")
            continue
        f = plan["families"][t["family"]]
        spec = f if t["family"] in ("random", "bounded") else t["spec"]["expect"]
        try:
            check_ack_law(j, t, r["observed"])
            JUDGES[t["family"]](j, t, r["observed"], spec)
            judged += 1
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            mechanical.append(f"#{t['index']}: the record lacks what the checker reads ({type(exc).__name__}: {exc})")
    gates = {g: {"pass": not j.failures[g], "failures": j.failures[g]} for g in GATES}
    if mechanical:
        result = "INCOMPLETE"
    elif subset:
        result = "PARTIAL_PASS" if all(v["pass"] for v in gates.values()) else "PARTIAL_NONPASS"
    else:
        result = "PASS" if all(v["pass"] for v in gates.values()) else "NONPASS"
    return {"schema": VERDICT_SCHEMA, "inputs": inputs, "plan": plan["schema"], "trials": len(expected),
            "judged": judged, "subset": subset, "mechanical": mechanical, "gates": gates, "verdict": result}


def evidence(directory: Path) -> list:
    """Every problem of a sealed evidence directory (empty: it is exactly what was sealed, and its verdict stands)."""
    names = {"FREEZE.json", "records.jsonl", "verdict.json", "SHA256SUMS"}
    problems = []
    present = {p.name for p in directory.iterdir()}
    if present != names:
        problems.append(f"the directory holds {sorted(present ^ names)} unexpectedly")
        return problems
    sums = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        sums[name] = digest
    for name in names - {"SHA256SUMS"}:
        if sums.get(name) != hashlib.sha256((directory / name).read_bytes()).hexdigest():
            problems.append(f"{name}: digest mismatch")
    recorded = json.loads((directory / "verdict.json").read_text())
    again = verdict(directory / "records.jsonl", recorded.get("inputs", "qual"))
    if again != recorded:
        problems.append("the verdict does not recompute from the records")
    if recorded.get("inputs") != "qual" or recorded.get("subset"):
        problems.append("evidence must be a complete QUAL run")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("records")
    v.add_argument("--inputs", choices=("dev", "qual"), required=True)
    v.add_argument("--out", required=True)
    e = sub.add_parser("evidence")
    e.add_argument("directory")
    args = ap.parse_args(argv)
    if args.command == "verdict":
        result = verdict(Path(args.records), args.inputs)
        with open(args.out, "x") as f:
            json.dump(result, f, indent=1, sort_keys=True)
        print(result["verdict"], {g: len(x["failures"]) for g, x in result["gates"].items()},
              len(result["mechanical"]), "mechanical")
        return 0
    problems = evidence(Path(args.directory))
    print("\n".join(problems) or "evidence verified")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
