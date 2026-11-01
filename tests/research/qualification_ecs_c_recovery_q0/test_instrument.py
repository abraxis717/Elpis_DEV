"""ECS_C Recovery Q0: the instrument, and the checker cannot be talked into a PASS.

The subject of this qualification, the durable ECS_C authority and its crash-recovery law, was retired by the
volatile-native migration (RuntimeCore ABI v8, docs/RUNTIME_CORE.md): no host can publish, recover or reconcile any
more, so the DEV rehearsal that drove a live testing host (in git history before that migration) cannot run, and the
host suite's crash-law tables it was cross-checked against are retired with it. The frozen instrument fails closed
over this checkout (its premise cannot even be stated). The checker's adversarial guarantees are kept over the sealed
QUAL run's own records (research/qualification/ecs_c_recovery_q0/evidence, proved byte-identical to what was sealed by
test_evidence.py): they recompute the recorded PASS, and every mutant of them must fail the gate that exists to catch
it:
- a wrong recovery disposition, candidate or authority report;
- a lost acknowledged publication, a wrong evidence count, a resumption that is not exact;
- an attack that opened the runtime, a wrong reconciliation accepted, a rollback continuing at the wrong generation;
- a publication over an unresolved candidate, a broken acknowledgment law;
- an extra or resized file;
- a missing, duplicated, reordered or re-seeded trial, and an instrument error (mechanical, never a gate).
The plan's statement of the crash law is cross-checked against the host suite's own tables.
"""
from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[3]
INSTRUMENT = REPO / "research" / "qualification" / "ecs_c_recovery_q0"
PLAN = json.loads((INSTRUMENT / "plan.json").read_text())


def _load(name: str):
    if str(INSTRUMENT) not in sys.path:
        sys.path.insert(0, str(INSTRUMENT))             # the runner imports its sibling freeze module
    spec = importlib.util.spec_from_file_location(f"q0_{name}", INSTRUMENT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _load("check")
runner = _load("runner")
freeze = _load("freeze")


# ---- the plan --------------------------------------------------------------------------------------------------

def test_the_plan_is_coherent_with_the_record_layout():
    d = PLAN["deployment"]
    assert d["record_bytes"] == 1344 + 160 * d["capacity"] and d["witness_bytes"] == 240
    assert len(d["files"]) == len(set(d["files"])) == 17
    regions = {"flip:header": (0, 256), "flip:generation": (32, 40), "flip:state": (256, d["record_bytes"] - 576),
               "flip:mac": (d["record_bytes"] - 64, d["record_bytes"] - 32),
               "flip:checksum": (d["record_bytes"] - 32, d["record_bytes"])}
    for p in PLAN["families"]["attack"]["points"]:
        a = p["attack"]
        if a["kind"] == "flip":
            lo, hi = regions[p["point"]]
            assert lo <= a["offset"] < hi, p["point"]
        if a["kind"] == "shorten":
            assert 0 <= a["length"] < d["record_bytes"]
    for family in ("publish", "failure", "anchor", "discard", "adopt", "attack", "rearrangement", "witness"):
        for p in PLAN["families"][family]["points"]:
            assert "evidence" in p["expect"], (family, p["point"])


@pytest.mark.parametrize("inputs", ["dev", "qual"])
def test_the_runner_and_the_checker_enumerate_the_same_trials(inputs):
    keys = ("index", "family", "point", "run", "seed")
    ran = [({k: t[k] for k in keys}, t.get("random")) for t in runner.enumerate_trials(inputs)]
    judged = [({k: t[k] for k in keys}, t["random"]) for t in check.expected_trials(inputs)]
    assert ran == judged
    assert len({t[0]["seed"] for t in ran}) == len(ran)
    counts = {f: sum(1 for t, _ in ran if t["family"] == f) for f in PLAN["families"]}
    assert counts["random"] == 32 and counts["bounded"] == 1 and counts["publish"] == 36 and counts["attack"] == 52
    # A subset keeps every trial's coordinates.
    subset = runner.enumerate_trials(inputs, {"random", "witness"}, 2)
    assert [t["index"] for t in subset] == [t["index"] for t, _ in ran
                                            if t["family"] in ("random", "witness") and t["run"] < 2]


# ---- the freeze ------------------------------------------------------------------------------------------------

def _observed():
    return {"source": {"commit": "c", "tree": "t", "clean": True, "status": []},
            "instrument": {"files": {"plan.json": "p"}, "specification": "s"}, "subject": {"a.rs": "x"},
            "premise": {"production_views": {"a.rs": "v"}, "testing_symbols": ["s"], "problems": []},
            "installed": {"bin": {"elpis": "h", "elpis_testing": "t"}, "lib": {"libelpis_runtime.so": "r"}},
            "python": {"executable": "/py", "version": "3", "implementation": "CPython", "isolated": 1,
                       "no_bytecode": 1, "elpis_imported": []},
            "toolchain": {"cc": "gcc"}, "platform": {"system": "Linux"}, "filesystem": {"type": "ext4"}}


def test_the_freeze_comparison_fails_closed():
    frozen = {"schema": freeze.FREEZE_SCHEMA, **_observed()}
    assert freeze.compare(frozen, _observed()) == []
    assert freeze.compare({}, _observed()) == [f"not a {freeze.FREEZE_SCHEMA} record"]
    for path, value, expected in [
        (("source", "commit"), "other", "source.commit differs from the freeze"),
        (("instrument", "files"), {"plan.json": "q"}, "instrument differs from the freeze"),
        (("premise", "production_views"), {"a.rs": "w"}, "premise differs from the freeze"),
        (("installed", "lib"), {"libelpis_runtime.so": "s"}, "installed differs from the freeze"),
        (("source", "clean"), False, "the checkout is not clean: []"),
        (("python", "isolated"), 0, "the interpreter is not in isolated mode (python -I)"),
        (("python", "elpis_imported"), ["elpis"], "the instrument imported Elpis modules: ['elpis']"),
        (("installed", "bin"), {"elpis": None, "elpis_testing": "t"}, "the installed prefix lacks ['elpis']"),
        (("premise", "problems"), ["P2: x"], "P2: x"),
    ]:
        observed = _observed()
        target = observed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert expected in freeze.compare(frozen, observed), path
    missing = copy.deepcopy(frozen)
    del missing["subject"]
    assert "subject: missing from the freeze record" in freeze.compare(missing, _observed())


def test_a_qual_run_without_its_freeze_creates_nothing(tmp_path):
    work = tmp_path / "work"
    done = subprocess.run([sys.executable, "-I", "-B", str(INSTRUMENT / "runner.py"), "run", "--prefix",
                           str(tmp_path), "--work", str(work), "--inputs", "qual", "--out", str(tmp_path / "r.jsonl")],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode != 0 and "needs --freeze" in done.stderr
    assert not work.exists() and not (tmp_path / "r.jsonl").exists()


# ---- the retired subject ----------------------------------------------------------------------------------------

def test_the_frozen_instrument_cannot_state_its_premise_over_the_volatile_runtime(tmp_path):
    """The premise binds the durable RuntimeCore sources (native/runtime/src/durable.rs among them) and a testing
    host with publication and recovery operations. Both are retired: the frozen instrument fails closed here, so the
    recorded PASS can never be re-derived from, or extended to, the volatile runtime."""
    assert not (REPO / "native" / "runtime" / "src" / "durable.rs").exists()
    with pytest.raises(FileNotFoundError):
        freeze.premise(REPO, tmp_path)


# ---- the checker over the sealed QUAL records -------------------------------------------------------------------

@pytest.fixture(scope="module")
def recorded():
    data = gzip.decompress((INSTRUMENT / "evidence" / "records.jsonl.gz").read_bytes()).decode()
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def test_the_sealed_qual_records_recompute_the_recorded_pass(recorded):
    rows = recorded
    result = check.evaluate(rows, "qual")
    assert result["mechanical"] == [] and result["verdict"] == "PASS", \
        {g: v["failures"][:3] for g, v in result["gates"].items()}
    assert result == json.loads((INSTRUMENT / "evidence" / "verdict.json").read_text())
    assert {r["family"] for r in rows[1:]} == set(PLAN["families"])
    assert result["judged"] == len(rows) - 1


def _where(rows, **match):
    return next(i for i, r in enumerate(rows) if i and all(r.get(k) == v for k, v in match.items()))


def _fails(rows, gate, plan=PLAN):
    result = check.evaluate(rows, "qual", plan)
    if gate == "mechanical":
        return result["verdict"] == "INCOMPLETE"
    return result["verdict"] == "NONPASS" and not result["gates"][gate]["pass"]


def _edit(rows, index, fn):
    out = copy.deepcopy(rows)
    fn(out[index]["observed"])
    return out


def _line(run, op):
    return next(o for o in run["lines"] if o.get("op") == op)


def test_every_mutant_fails_the_gate_that_exists_to_catch_it(recorded):
    rows = recorded
    trailing = _where(rows, family="publish", point="step:6", run=0)
    candidate = _where(rows, family="publish", point="step:5", run=0)
    completed = _where(rows, family="publish", point="step:7", run=0)
    rollback = _where(rows, family="attack", point="rollback:1", run=1)
    deep = _where(rows, family="attack", point="rollback:2", run=0)
    swap = _where(rows, family="rearrangement", point="swap", run=0)
    rnd = _where(rows, family="random", run=0)

    def set_report(key, value):
        return lambda o: _line(o["observe"]["authority"], "authority")["report"].__setitem__(key, value)

    def set_recovery(key, value):
        return lambda o: _line(o["observe"]["status"], "status")["recovery"].__setitem__(key, value)

    def plan_with(family, point, key, value):
        plan = copy.deepcopy(PLAN)
        next(p for p in plan["families"][family]["points"] if p["point"] == point)["expect"][key] = value
        return plan

    mutants = {
        "authority report": (_edit(rows, trailing, set_report("disposition", "CLEAN")), "G1"),
        "recovery disposition": (_edit(rows, candidate, set_recovery("disposition", "RESUMABLE")), "G1"),
        "candidate": (_edit(rows, candidate, set_recovery("candidate", "0" * 64)), "G1"),
        "an operator was needed": (_edit(rows, completed, lambda o: _line(o["observe"]["authority"], "authority")
                                         .__setitem__("reconciliation", {"kind": "ROLLBACK"})), "G1"),
        "lost acknowledged publication": (_edit(rows, completed, lambda o: _line(o["observe"]["status"], "status")
                                                ["durable"].__setitem__("identity", "1" * 64)), "G2"),
        "evidence count": (_edit(rows, completed, lambda o: _line(o["continuation"]["authority"], "authority")
                                 ["authority"].__setitem__("evidence_count", 99)), "G2"),
        "inexact resumption": (_edit(rows, swap, lambda o: _line(o["continuation"]["resumed"], "live")
                                     ["live"].__setitem__("identity", "2" * 64)), "G2"),
        "attack opened the runtime": (_edit(rows, rollback, lambda o: _line(o["observe"]["status"], "status")
                                            .__setitem__("session_refusal", None)), "G3"),
        "wrong reconciliation accepted": (_edit(rows, rollback, lambda o: o["reconcile"][0]["lines"][0]
                                                .update(ok=True, code=None)), "G3"),
        "wrong generation": (_edit(rows, rollback, lambda o: o["reconcile"][-1]["lines"][0]
                                   .__setitem__("generation", 1)), "G3"),
        "missing checkpoint resumed": (_edit(rows, deep, lambda o: o["final"]["learn"]["lines"][0]
                                             .__setitem__("ok", True)), "G3"),
        "rearrangement needed the operator": (_edit(rows, swap, set_report("disposition", "ROLLBACK")), "G3"),
        "publication over a candidate": (_edit(rows, candidate, lambda o: o["action"]["probe"]["lines"][0]
                                               .__setitem__("ok", True)), "G4"),
        "acknowledgment law": (_edit(rows, completed, lambda o: _line(o["continuation"]["published"], "publish")
                                     ["ticket"].__setitem__("status", "FAILED")), "G4"),
        "implicit reconciliation": (_edit(rows, candidate, lambda o: o["action"].__setitem__("taken", "none")), "G4"),
        "extra file": (_edit(rows, swap, lambda o: o["continuation"]["files"].__setitem__("log", [1, 1])), "G5"),
        "resized slot": (_edit(rows, swap, lambda o: o["continuation"]["files"]["state/authority.b"]
                               .__setitem__(0, 1)), "G5"),
        "random: acknowledged state lost": (_edit(rows, rnd, lambda o: _line(o["observe"]["status"], "status")
                                                  ["durable"].__setitem__("identity", "3" * 64)), "G2"),
        "plan: another disposition": (rows, "G1", plan_with("publish", "step:6", "authority", "CLEAN")),
        "plan: another accepted record": (rows, "G3", plan_with("attack", "rollback:1", "accepted", "D4")),
    }
    for name, (mutant, gate, *plan) in mutants.items():
        assert _fails(mutant, gate, *plan), name
    # Mechanical: the enumeration itself.
    assert _fails(rows[:-1], "mechanical")
    assert _fails(rows + [rows[-1]], "mechanical")
    assert _fails([rows[0], rows[2], rows[1], *rows[3:]], "mechanical")
    reseeded = copy.deepcopy(rows)
    reseeded[1]["seed"] += 1
    assert _fails(reseeded, "mechanical")
    errored = copy.deepcopy(rows)
    errored[1].update(observed=None, instrument_error="OSError: x")
    assert _fails(errored, "mechanical")
    assert check.evaluate(errored, "qual")["verdict"] == "INCOMPLETE"


def test_growing_recovery_work_or_files_fail_the_bounded_gate(recorded):
    """The bounded family over the sealed run's real outputs: constant work passes, growing work fails G6."""
    rows = recorded
    swap = rows[_where(rows, family="rearrangement", point="swap", run=0)]["observed"]
    authority = copy.deepcopy(swap["continuation"]["authority"])
    baseline = swap["deploy"]["baseline"]
    marks = []
    for n in PLAN["families"]["bounded"]["publications"]:
        a = copy.deepcopy(authority)
        a["lines"][0]["authority"].update(publications=1 + n, evidence_count=n)
        marks.append({"publications": n, "authority": a, "files": dict(baseline)})
    t = {"index": 0, "family": "bounded", "point": "bounded", "run": 0}
    j = check.Judge()
    check.judge_bounded(j, t, {"deploy": {"baseline": baseline}, "marks": marks}, PLAN["families"]["bounded"])
    assert not any(j.failures.values()), j.failures
    marks[-1]["authority"]["lines"][0]["report"]["read_bytes"] += 1
    j = check.Judge()
    check.judge_bounded(j, t, {"deploy": {"baseline": baseline}, "marks": marks}, PLAN["families"]["bounded"])
    assert j.failures["G6"]


def test_a_partial_run_is_never_evidence_and_a_sealed_directory_is_verified(recorded, tmp_path):
    head = dict(recorded[0], families=["publish"])
    rows = [head] + [r for r in recorded[1:] if r["family"] == "publish"]
    expected = [t for t in check.expected_trials("qual", PLAN) if t["family"] == "publish"]
    assert [r["index"] for r in rows[1:]] == [t["index"] for t in expected]
    records = tmp_path / "records.jsonl"
    records.write_text("".join(json.dumps(r) + "\n" for r in rows))
    verdict = tmp_path / "verdict.json"
    result = check.evaluate(rows, "qual")
    assert result["verdict"] == "PARTIAL_PASS" and result["subset"]
    verdict.write_text(json.dumps(result))
    frozen = tmp_path / "FREEZE.json"
    frozen.write_text("{}")
    sealed = tmp_path / "evidence"
    runner.main(["seal", "--freeze", str(frozen), "--records", str(records), "--verdict", str(verdict),
                 "--out", str(sealed)])
    assert check.evidence(sealed) == ["evidence must be a complete QUAL run"]   # sealed exactly, yet not evidence
    (sealed / "records.jsonl").chmod(0o644)
    with open(sealed / "records.jsonl", "a") as f:
        f.write("\n")
    assert "records.jsonl: digest mismatch" in check.evidence(sealed)
