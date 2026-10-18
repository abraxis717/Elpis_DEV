"""K1 Recovery R0 v4: the checker cannot be talked into a PASS (controlled mutants over a real DEV run).

A DEV run of the frozen instrument (every family, one run per point and lineage; DEV seeds, never QUAL inputs) is
generated over the real libraries, judged PASS by the checker, and then tampered with one way at a time. Every
mutant must fail the gate that exists to catch it: an altered expectation (in the plan, since records carry none), a
wrong seed, fault argument or coordinate, a missing, duplicated or reordered trial, no actual torn write, a wrong
continuity format, an incorrect restart state, an unregistered operator action, a duplicate publication, a broken
acknowledgment law, a lost lane fault, a moved slot file, a never-requested state as the authority, and an
incomplete or altered evidence manifest.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ...conftest import native_build_dir, require_runtime_library

REPO = Path(__file__).resolve().parents[3]
INSTRUMENT = REPO / "research" / "qualification" / "k1_recovery_r0_v4"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"k1r4_{name}", INSTRUMENT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _load("check")
PLAN = json.loads((INSTRUMENT / "plan.json").read_text())


@pytest.fixture(scope="module")
def dev(tmp_path_factory):
    require_runtime_library(testing=True)
    root = tmp_path_factory.mktemp("dev")
    out = root / "records.jsonl"
    done = subprocess.run([sys.executable, "-I", "-B", str(INSTRUMENT / "runner.py"), "run", "--native-build",
                           str(native_build_dir()), "--work", str(root / "work"), "--inputs", "dev", "--dev-runs", "1",
                           "--out", str(out)], capture_output=True, text=True, timeout=1800)
    assert out.exists(), done.stderr[-4000:]
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    return root, out, records


def _verdict(records, plan=PLAN):
    return check.evaluate(plan, records, inputs="dev", runs_override=1)


def _first(records, **match):
    return next(i for i, r in enumerate(records) if all(r.get(k) == v for k, v in match.items()))


def test_the_runner_keeps_every_line_a_child_wrote_past_the_one_it_waited_for(tmp_path):
    """A chunk can carry the LEARN the runner waits for and the request the child made next, before its kill: the
    request must reach the record (a lost one made a requested candidate look never requested)."""
    runner = _load("runner")
    child = ("import json, os, sys, time\n"
             "lines = [json.dumps({'event': 'learn', 'i': 0}), json.dumps({'event': 'request', 'identity': 'aa'}),"
             " json.dumps({'event': 'learn', 'i': 1})]\n"
             "os.write(1, ('\\n'.join(lines[:2]) + '\\n').encode()); time.sleep(0.5)\n"
             "os.write(1, (lines[2] + '\\n').encode()); time.sleep(3600)\n")
    proc = subprocess.Popen([sys.executable, "-I", "-B", "-c", child], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    out = runner._Output(proc)
    first, done = out.read(30, until=lambda line: line.get("event") == "learn")
    assert done and first == [{"event": "learn", "i": 0}]
    second, done = out.read(30, until=lambda line: line.get("i") == 1)
    assert done and second == [{"event": "request", "identity": "aa"}, {"event": "learn", "i": 1}]
    assert runner._kill(proc)["signaled"] and out.rest() == []


def test_the_dev_run_is_complete_and_passes_every_gate(dev):
    _, _, records = dev
    result = _verdict(records)
    assert result["complete"] and result["verdict"] == "PASS", {g: v["failures"][:3] for g, v in result["gates"].items()}
    assert {r["family"] for r in records} == set(PLAN["families"])


def _fails(records, gate, plan=PLAN):
    result = _verdict(records, plan)
    return result["verdict"] == "FAIL" and (gate == "complete" and not result["complete"]
                                            or gate != "complete" and result["gates"][gate]["verdict"] == "FAIL")


def _mutate(records, index, edit):
    out = copy.deepcopy(records)
    edit(out[index])
    return out


def _flip_magic(raw: dict) -> dict:
    """Relabel every record's format (v2 <-> v3) keeping its digest, generation and checksum bytes."""
    out = {}
    for name, value in raw.items():
        b = bytearray(bytes.fromhex(value))
        b[7] = 3 if b[7] == 2 else 2
        out[name] = bytes(b).hex()
    return out


def test_every_mutant_fails_the_gate_that_exists_to_catch_it(dev):
    _, _, records = dev
    pub = lambda kind, point: _first(records, family="publish", lineage=kind, point=point)
    unresolved, resumable = pub("composite", "step:5"), pub("composite", "step:2")
    slot_torn, record_torn = pub("k1", "step:3"), pub("k1", "step:6")
    volatile_key = records[resumable]["child"]["keys"][2]
    lost = _first(records, family="uncertain", point="lane:8")
    refused = _first(records, family="identity_refused")
    rnd = _first(records, family="random", lineage="k1")

    def no_slot_tear(r):
        for e in r["restart"]["files_after"]["slots"].values():
            e.update(changed=False, diff_bytes=0, first_diff=None, last_diff=None, within_arg=False)

    def no_record_tear(r):
        r["restart"]["continuity_raw"] = copy.deepcopy(r["child"]["files_before"]["continuity"])

    def never_requested(r):
        requested = {e["identity"] for e in r["events"] if e.get("event") == "request"}
        learned = [e["identity"] for e in r["events"] if e.get("event") == "learn"]
        r["restart"]["observed"]["authorized"] = next(i for i in learned if i not in requested)

    mutants = {
        "wrong seed": (resumable, lambda r: r.update(seed=r["seed"] + 1), "G1"),
        "wrong fault argument": (slot_torn, lambda r: r.update(arg=r["arg"] + 1), "G1"),
        "wrong coordinate": (resumable, lambda r: r.update(point="step:7"), "G1"),
        "wrong run index": (resumable, lambda r: r.update(run_index=1), "G1"),
        "wrong random draw": (rnd, lambda r: r.update(kill_after=r["kill_after"] + 1), "G1"),
        "no actual torn slot write": (slot_torn, no_slot_tear, "G1"),
        "no actual torn record": (record_torn, no_record_tear, "G1"),
        "wrong continuity format (runtime)": (resumable, lambda r: r["restart"]["observed"].update(format="K1"), "G1"),
        "wrong continuity format (raw record)": (
            resumable, lambda r: r["restart"].update(continuity_raw=_flip_magic(r["restart"]["continuity_raw"])), "G1"),
        "another format accepted": (resumable, lambda r: r["restart"]["cross_format"].update(refusal=None), "G1"),
        "no SIGKILL": (resumable, lambda r: r["child_exit"].update(signaled=False, returncode=0), "G1"),
        "incorrect restart state": (
            resumable, lambda r: r["restart"]["resumed"].update(restored_identity="00" * 32), "G2"),
        "volatile memory recovered": (
            resumable, lambda r: r["restart"]["resumed"]["resident"].update({volatile_key: 1}), "G2"),
        "memory written twice": (
            resumable, lambda r: r["restart"]["resumed"].update(memory_slots_after=3), "G2"),
        "unauthorized candidate adoption": (resumable, lambda r: r["restart"].update(action="adopt"), "G3"),
        "wrong reconciliation": (unresolved, lambda r: r["restart"].update(action="discard"), "G3"),
        "a duplicate publication": (
            resumable, lambda r: r["restart"]["final"].update(generation=r["restart"]["final"]["generation"] + 1), "G3"),
        "a never-requested state recovered": (rnd, never_requested, "G3"),
        "an acknowledgment without recovery": (
            resumable, lambda r: r["child"]["first_ticket"].update(recoverable=False), "G4"),
        "a refused publication that moved continuity": (
            refused, lambda r: r["child"]["durable_after"].update(generation=r["child"]["anchor"] + 1), "G4"),
        "a lost lane fault": (lost, lambda r: r["child"].update(lane_fault=None), "G5"),
        "a replaced slot file": (
            resumable, lambda r: next(iter(r["restart"]["final_footprint"].values())).update(inode=1), "G6"),
    }
    for name, (index, edit, gate) in mutants.items():
        assert _fails(_mutate(records, index, edit), gate), name
    # An altered expectation: records carry none, so it can only be altered in the plan, which the freeze binds.
    plan = copy.deepcopy(PLAN)
    next(p for p in plan["families"]["publish"]["points"] if p["point"] == "step:5")["expect"]["disposition"] = \
        "RESUMABLE"
    assert _fails(records, "G1", plan)
    # Missing, duplicated and reordered trials.
    assert _fails(records[1:], "complete")
    assert _fails(records + records[-1:], "G1")
    swapped = list(records)
    swapped[3], swapped[4] = swapped[4], swapped[3]
    assert _fails(swapped, "G1")


def test_the_evidence_manifest_is_complete_intact_and_bound(dev, tmp_path):
    root, out, records = dev
    verdict = root / "verdict.json"
    if not verdict.exists():
        verdict.write_text(json.dumps(dict(_verdict(records), plan_sha256=check._sha(INSTRUMENT / "plan.json"),
                                           records_sha256=check._sha(out))))
    stand_in = root / "freeze.json"   # a DEV stand-in: the manifest checks are what is exercised here
    if not stand_in.exists():
        stand_in.write_text(json.dumps({"schema": check.FREEZE_SCHEMA}))
    sealed = tmp_path / "evidence"
    subprocess.run([sys.executable, "-I", "-B", str(INSTRUMENT / "runner.py"), "seal", "--freeze", str(stand_in),
                    "--records", str(out), "--verdict", str(verdict), "--out", str(sealed)], check=True,
                   capture_output=True)
    assert sorted(p.name for p in sealed.iterdir()) == sorted(check.EVIDENCE_FILES)
    baseline = check.verify_evidence(sealed, INSTRUMENT / "plan.json")
    assert "the run is not a complete QUAL run" in baseline     # a DEV run is never QUAL evidence

    def copy_of():
        target = tmp_path / f"copy-{len(list(tmp_path.iterdir()))}"
        shutil.copytree(sealed, target)
        for p in target.iterdir():
            p.chmod(0o644)
        return target

    missing = copy_of()
    (missing / "verdict.json").unlink()
    assert any("not exactly" in p for p in check.verify_evidence(missing, INSTRUMENT / "plan.json"))
    extra = copy_of()
    (extra / "notes.txt").write_text("x")
    assert any("not exactly" in p for p in check.verify_evidence(extra, INSTRUMENT / "plan.json"))
    altered = copy_of()
    data = bytearray((altered / "records.jsonl").read_bytes())
    data[100] ^= 1
    (altered / "records.jsonl").write_bytes(bytes(data))
    assert any("records.jsonl: SHA-256" in p for p in check.verify_evidence(altered, INSTRUMENT / "plan.json"))
