"""Elpis Torch R10 records: provenance, integrity and the preregistered verdict recomputed independently.

These tests read JSON only (no PyTorch). They re-implement the decision rules of ``PREREGISTRATION_R10.md`` §5
rather than calling the runner, and check that every run was produced by the committed preregistration, runner and
task generator (byte digests), that no evaluation episode was a training episode, that evaluation changed no
parameter, and that the pilot used no CGC arm.
"""
from __future__ import annotations

import hashlib
import json
import statistics

import pytest

from .conftest import REPO

RESULTS = REPO / "research" / "elpis_torch" / "results" / "r10"
PREREGISTRATION = RESULTS / "preregistration.json"
RUNNER = REPO / "research" / "elpis_torch" / "experiments" / "run_r10.py"
TASKS = REPO / "research" / "elpis_torch" / "adaptation_tasks.py"


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def plan():
    return json.loads(PREREGISTRATION.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runs():
    out = {}
    for path in sorted(RESULTS.glob("run_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        out[(record["arm"], record["seed"])] = record
    return out


def test_every_run_was_produced_by_the_committed_plan_runner_and_generator(runs, plan):
    assert runs, "no run records"
    expected = (_sha(PREREGISTRATION), _sha(RUNNER), _sha(TASKS))
    for key, record in runs.items():
        p = record["provenance"]
        assert (p["preregistration_sha256"], p["runner_sha256"], p["tasks_sha256"]) == expected, key
        assert p["task_spec_digest"] == plan["task_spec_digest"], key


def test_the_preregistration_predates_every_run():
    import subprocess
    revisions = {_load(p.name)["provenance"]["git_revision"].split("+")[0] for p in RESULTS.glob("run_*.json")}
    for revision in revisions:
        shown = subprocess.run(["git", "show", f"{revision}:research/elpis_torch/results/r10/preregistration.json"],
                               cwd=REPO, capture_output=True)
        if shown.returncode != 0:
            pytest.skip("the recorded revision is not in this checkout's history (shallow clone)")
        assert hashlib.sha256(shown.stdout).hexdigest() == _sha(PREREGISTRATION), revision


def test_runs_completed_their_steps_on_disjoint_episodes_without_updating_weights(runs, plan):
    """A failed run is kept as FAILED (PREREGISTRATION_R10.md section 8) and makes its stage NONQUAL; a completed
    run trained every preregistered step on episodes disjoint from evaluation and left the weights unchanged."""
    for key, record in runs.items():
        stage = "screening" if record["seed"] in plan["seeds"]["screening"] else "confirmation"
        assert record["stage"] == stage, key
        if record["status"] != "COMPLETED":
            assert record["status"] == "FAILED", key
            continue
        assert record["train"]["steps"] == plan["training"]["steps"], key
        assert record["train_test_signature_overlap"] == 0, key
        assert record["parameters_unchanged_by_evaluation"] is True, key


def test_every_planned_screening_run_has_a_record_and_no_unplanned_run_exists(runs, plan):
    for seed in plan["seeds"]["screening"]:
        for arm in plan["arms"]:
            assert (arm, seed) in runs, (arm, seed)
    for arm, seed in runs:
        assert seed in plan["seeds"]["screening"] or (
            seed in plan["seeds"]["confirmation"] and arm in plan["confirmation_arms"]), (arm, seed)


def test_confirmation_ran_only_after_a_favorable_screening(runs, plan):
    combined = _load("study_cgc_adaptation.json")
    confirmation_runs = [k for k in runs if k[1] in plan["seeds"]["confirmation"]]
    favorable = [e for e, v in combined["screening"]["endpoints"].items() if v["verdict"] == "SCREEN_FAVORABLE"]
    assert not confirmation_runs or favorable


def test_the_latency_record_was_measured_on_the_frozen_code():
    record = _load("latency.json")
    p = record["provenance"]
    assert (p["preregistration_sha256"], p["runner_sha256"], p["tasks_sha256"]) == (
        _sha(PREREGISTRATION), _sha(RUNNER), _sha(TASKS))
    assert p["threads"] == 1 and set(record["arms"]) == set(_load("preregistration.json")["arms"])


def _classify_screen(a_minus_c, a_minus_cm, delta, need):
    ok = all(statistics.fmean(d) >= delta and sum(x > 0 for x in d) >= need for d in (a_minus_c, a_minus_cm))
    return "SCREEN_FAVORABLE" if ok else "SCREEN_NOT_FAVORABLE"


def test_the_recorded_screening_verdict_follows_from_the_runs(runs, plan):
    combined = _load("study_cgc_adaptation.json")["screening"]["endpoints"]
    a = plan["analysis"]
    seeds = plan["seeds"]["screening"]
    done = {k: r for k, r in runs.items() if r["status"] == "COMPLETED"}

    def best(key):
        means = [statistics.fmean(done[(arm, s)]["endpoints"][key] for s in seeds) for arm in ("A", "B", "C", "C_mac")
                 if all((arm, s) in done for s in seeds)]
        return max(means) if means else -1.0

    for endpoint, delta in (("ADAPT", a["delta_adapt"]), ("RETAIN", a["delta_retain"])):
        t = a["learned_threshold"]
        learned = best("ADAPT") >= t["ADAPT"] if endpoint == "ADAPT" else \
            best("RETAIN") >= t["RETAIN"] and best("RETAIN_pre") >= t["RETAIN_pre"]
        if not learned:
            expected = "ASSAY_NOT_LEARNED"
        elif not all((arm, s) in done for arm in ("A", "C", "C_mac") for s in seeds):
            expected = "NONQUAL"
        else:
            diff = lambda other: [done[("A", s)]["endpoints"][endpoint] - done[(other, s)]["endpoints"][endpoint]  # noqa
                                  for s in seeds]
            expected = _classify_screen(diff("C"), diff("C_mac"), delta, a["screen_min_positive"])
        assert combined[endpoint]["verdict"] == expected, endpoint


def test_shuffled_support_targets_leave_only_chance(runs):
    """Negative control (PREREGISTRATION_R10.md §7): with every revealed value randomized, an answer scored against
    the truth is unknowable; accuracy within chance + 0.05 in every arm."""
    for key, record in runs.items():
        if record["status"] != "COMPLETED":
            continue
        for v in ("A", "B"):
            acc = record["diagnostics"][f"{v}/shuffled_support_targets"]["accuracy"]
            assert acc <= 0.05 + 0.05, (key, v, acc)


def test_the_pilot_used_no_cgc_arm_and_is_labelled():
    pilots = sorted(RESULTS.glob("pilot_*.json"))
    assert pilots
    for path in pilots:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["command"] == "pilot" and record["label"].startswith("PILOT")
        assert record["arm"] in ("B", "D_C") and record["seed"] == 900


def test_the_manifest_lists_every_r10_record_with_its_digest():
    manifest = _load("MANIFEST.json")
    files = {p.name for p in RESULTS.glob("*.json") if p.name != "MANIFEST.json"}
    assert set(manifest["files"]) == files
    for name, digest in manifest["files"].items():
        assert _sha(RESULTS / name) == digest, name
