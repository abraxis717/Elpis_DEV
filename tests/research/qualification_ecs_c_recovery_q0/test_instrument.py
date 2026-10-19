"""ECS_C Recovery Q0: the instrument, rehearsed on DEV inputs, and the checker cannot be talked into a PASS.

A DEV run (every family but the long bounded history, two runs per point so both operator parities run; DEV seeds,
never QUAL inputs) drives the real testing host and is judged by the checker. Then it is tampered with one way at a
time, and every mutant must fail the gate that exists to catch it:
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
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ...conftest import native_build_dir, native_required

REPO = Path(__file__).resolve().parents[3]
INSTRUMENT = REPO / "research" / "qualification" / "ecs_c_recovery_q0"
PLAN = json.loads((INSTRUMENT / "plan.json").read_text())
# The installed layout, staged from a build tree when no prefix is given: binary and library build paths.
_BINARIES = {"elpis": "native/runtime/elpis", "elpis_testing": "native/runtime/elpis_testing"}
_LIBRARIES = ("native/ECS_G/libelpis_ecsg_math.so", "native/ECS_G/libelpis_ecsg_k1.so",
              "native/ECS_C/libelpis_ecs_memory.so", "native/runtime/composite/libelpis_ecs_composite.so",
              "native/runtime/libelpis_hacf_observation.so", "native/ECS_C/authority/libelpis_ecs_c_authority.so",
              "native/runtime/libelpis_runtime.so", "native/runtime/libelpis_runtime_testing.so")


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


def test_the_plan_states_the_crash_law_the_host_suite_states():
    """Two independent statements of the same law (plan.json and tests/integration/test_native_host.py) agree."""
    from ...integration import test_native_host as suite
    for p in PLAN["families"]["publish"]["points"]:
        durable, candidate, disposition = suite.CRASH[int(p["point"].split(":")[1])]
        e = p["expect"]
        assert e["authority"] == disposition and e["durable"] == ("D2" if durable == "new" else "D1")
        assert (e["recovery"] == "CANDIDATE_UNRESOLVED") is candidate
    for p in PLAN["families"]["failure"]["points"]:
        code, cause, disposition, durable, authority = suite.FAILURES[int(p["point"].split(":")[1])]
        e = p["expect"]
        assert (e["code"], e["cause"], e["recovery"], e["authority"]) == (code, cause, disposition, authority)
        assert e["durable"] == ("D2" if durable == "new" else "D1")
    for p in PLAN["families"]["anchor"]["points"]:
        anchored, disposition = suite.ANCHOR[int(p["point"].split(":")[1])]
        e = p["expect"]
        assert e["authority"] == disposition and e["resumed"] is anchored
        assert e["recovery"] == ("RESUMABLE" if anchored else "NOTHING_TO_RECOVER")


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


# ---- the DEV rehearsal -----------------------------------------------------------------------------------------

def _prefix(root: Path) -> Path:
    if os.environ.get("ELPIS_NATIVE_HOST_PREFIX"):
        return Path(os.environ["ELPIS_NATIVE_HOST_PREFIX"]).resolve()
    build = native_build_dir()
    if not (build / _BINARIES["elpis_testing"]).is_file():
        message = f"the native host is not built under {build}"
        if native_required():
            pytest.fail(message)
        pytest.skip(message + " (never an implicit PASS)")
    prefix = root / "prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib" / "elpis").mkdir(parents=True)
    for name, rel in _BINARIES.items():
        shutil.copy2(build / rel, prefix / "bin" / name)
    for rel in _LIBRARIES:
        shutil.copy2(build / rel, prefix / "lib" / "elpis" / Path(rel).name)
    return prefix


@pytest.fixture(scope="module")
def dev(tmp_path_factory):
    root = tmp_path_factory.mktemp("q0-dev")
    prefix = _prefix(root)
    families = ",".join(f for f in PLAN["families"] if f != "bounded")
    out = root / "records.jsonl"
    done = subprocess.run([sys.executable, "-I", "-B", str(INSTRUMENT / "runner.py"), "run", "--prefix", str(prefix),
                           "--work", str(root / "work"), "--inputs", "dev", "--families", families, "--runs", "2",
                           "--out", str(out)], capture_output=True, text=True, timeout=5400,
                          env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
    assert done.returncode == 0 and out.exists(), (done.stdout[-3000:], done.stderr[-3000:])
    return prefix, check.read_records(out)


def test_the_premise_holds_for_the_prefix(dev):
    prefix, _ = dev
    assert freeze.premise(REPO, prefix)["problems"] == []


def test_the_dev_rehearsal_passes_every_gate(dev):
    _, rows = dev
    result = check.evaluate(rows, "dev")
    assert result["mechanical"] == [] and result["verdict"] == "PARTIAL_PASS", \
        {g: v["failures"][:3] for g, v in result["gates"].items()}
    assert {r["family"] for r in rows[1:]} == set(PLAN["families"]) - {"bounded"}
    assert result["judged"] == len(rows) - 1


def _where(rows, **match):
    return next(i for i, r in enumerate(rows) if i and all(r.get(k) == v for k, v in match.items()))


def _fails(rows, gate, plan=PLAN):
    result = check.evaluate(rows, "dev", plan)
    if gate == "mechanical":
        return result["verdict"] == "INCOMPLETE"
    return result["verdict"] == "PARTIAL_NONPASS" and not result["gates"][gate]["pass"]


def _edit(rows, index, fn):
    out = copy.deepcopy(rows)
    fn(out[index]["observed"])
    return out


def _line(run, op):
    return next(o for o in run["lines"] if o.get("op") == op)


def test_every_mutant_fails_the_gate_that_exists_to_catch_it(dev):
    _, rows = dev
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
    assert check.evaluate(errored, "dev")["verdict"] == "INCOMPLETE"


def test_growing_recovery_work_or_files_fail_the_bounded_gate(dev):
    """The bounded family over the DEV rehearsal's real outputs: constant work passes, growing work fails G6."""
    _, rows = dev
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


def test_a_dev_run_is_never_evidence_and_a_sealed_directory_is_verified(dev, tmp_path):
    _, rows = dev
    records = tmp_path / "records.jsonl"
    records.write_text("".join(json.dumps(r) + "\n" for r in rows))
    verdict = tmp_path / "verdict.json"
    verdict.write_text(json.dumps(check.evaluate(rows, "dev")))
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
