"""Retention R0 evidence guards: chronology, frozen authority, write-once evidence, exact reproduction, honest report.

The specification and candidates predate DEV; the frozen laboratory source is
the source in this tree; every DEV and QUAL record carries the full
implementation binding; QUAL worlds reproduce their recorded measurements
exactly when the bound implementation is the one present; the results
document states the disposition the evidence yields and claims no more.
"""
from __future__ import annotations

import ctypes
import json
import subprocess

import pytest

from elpis.ECS_G.native import ECSGLibrary

from research.ecs_retention_r0 import experiment as X
from research.ecs_retention_r0 import protocol as P
from research.ecs_retention_r0 import run as R

from ...ECS_G.test_math_r0 import REPO, _library_path

RESULTS = REPO / "docs" / "research" / "ECS_RETENTION_R0_RESULTS.md"
FROZEN = P.load(R.FROZEN_PATH, "frozen")
QUAL = P.load(R.QUAL_PATH, "qual")


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def test_frozen_authority_still_matches_the_laboratory_and_specification():
    body = FROZEN["body"]
    assert body["lab_source_digest"] == P.source_digest()
    digests = P.spec_digests(P.load_spec())
    assert all(body[k] == digests[k] for k in ("spec", "pass_rule", "candidates_sha256"))
    assert body["pass_rule_text"] == P.load_spec()["pass_rule"]
    assert body["qual_worlds"] == list(P.world_ids(P.load_spec(), "QUAL"))


def test_the_specification_predates_dev_and_the_freeze_predates_qual():
    log = _git("log", "--format=%H %s", "--", "research/ecs_retention_r0").splitlines()
    if not log:
        pytest.skip("not a git checkout")
    order = {}
    for i, line in enumerate(log):
        subject = line.split(" ", 1)[1]
        if subject.startswith("RET0"):
            order.setdefault(subject.split(" ", 1)[0], i)
    # git log is newest first: a larger index is older.
    assert order["RET0A.1"] > order["RET0B"] > order["RET0C"] > order["RET0D"]
    first_spec = _git("log", "--diff-filter=A", "--format=%s", "--",
                      "research/ecs_retention_r0/specs/ecsg-retention-r0.v1.spec.json")
    assert first_spec.startswith("RET0A.1")
    for path, step in (("evidence/dev/ecsg-retention-r0.v1.dev.json", "RET0B "),
                       ("frozen/ecsg-retention-r0.v1.frozen.json", "RET0C"),
                       ("evidence/qual/ecsg-retention-r0.v1.qual.json", "RET0D")):
        assert _git("log", "--diff-filter=A", "--format=%s", "--",
                    f"research/ecs_retention_r0/{path}").startswith(step), path


def test_dev_records_chain_and_qual_is_bound_to_the_freeze():
    first, rerun = P.load(R.DEV_PATH, "dev"), P.load(R.DEV_RERUN_PATH, "dev")
    assert rerun["body"]["supersedes"]["digest"] == first["digest"]
    assert R._results(first["body"]) == R._results({k: v for k, v in rerun["body"].items() if k != "supersedes"})
    assert FROZEN["body"]["dev_evidence_digest"] == rerun["digest"]
    assert FROZEN["body"]["dev_records"] == [first["digest"], rerun["digest"]]
    q = QUAL["body"]
    assert q["frozen_digest"] == FROZEN["digest"] and q["choices"] == FROZEN["body"]["choices"]
    # The QUAL body's "worlds" key holds the world count (the verdict's "worlds" overrides the list when spread);
    # the world ids are the per-world keys, which equal the frozen QUAL split.
    assert q["worlds"] == 24 and list(q["per_world"]) == FROZEN["body"]["qual_worlds"]
    assert not set(q["per_world"]) & set(first["body"]["worlds"])
    for record in (first["body"], rerun["body"], q):
        impl = record["implementation"]
        assert not impl["dirty"] and set(impl["files"]) == set(P.BOUND_FILES)
        assert impl["library"]["sha256"] and impl["build"]["compiler"] and impl["numerical_profile"]["numpy"]
    assert P.binding_mismatch(FROZEN["body"]["implementation"], q["implementation"]) == []


def test_qual_disposition_is_recorded_as_found():
    q = QUAL["body"]
    assert q["disposition"] == "PARTIAL_REDUCTION" and q["outcome"] == "OUTCOME_C"
    assert all(q["mechanics"].values())
    assert q["gates"]["B_acquisition"] is False and q["gates"]["C_retention"] is False
    assert q["gates"]["H_negative_baseline"] and q["gates"]["E_state_causality"] and q["gates"]["G_determinism"]
    assert q["choices"]["selected"] == {"family": "C1", "lambda": 4.0}


def _strip_timing(value):
    if isinstance(value, dict):
        return {k: _strip_timing(v) for k, v in value.items() if k != "learn_seconds"}
    if isinstance(value, list):
        return [_strip_timing(v) for v in value]
    return value


@pytest.mark.parametrize("world", ["qual-0000", "qual-0013", "qual-0020"])
def test_qual_measurements_reproduce_exactly(world):
    """A QUAL world re-run gives its recorded measurements bit for bit (same implementation and profile)."""
    current = P.implementation(_library_path())
    stale = P.binding_mismatch(QUAL["body"]["implementation"], current)
    stale = [s for s in stale if s != "lab_source_digest"] + (["lab_source_digest"] if "lab_source_digest" in stale
                                                              else [])
    if stale:
        pytest.skip(f"evidence bound to a different implementation or profile: {stale}")
    lab = X.Lab(ECSGLibrary(ctypes.CDLL(str(_library_path()))), P.load_spec())
    got = json.loads(P.canonical_json(X.qual_world(lab, world, QUAL["body"]["choices"])))
    assert _strip_timing(got) == _strip_timing(QUAL["body"]["per_world"][world])


def test_results_report_the_disposition_and_stay_bounded():
    text = RESULTS.read_text(encoding="utf-8")
    q = QUAL["body"]
    for needle in ("PARTIAL_REDUCTION", "OUTCOME_C", QUAL["digest"], FROZEN["digest"],
                   q["implementation"]["library"]["sha256"], "No candidate is eligible for canonical promotion",
                   "Retention is not solved", "SEMANTICS=NONE", "18/24", "0/24"):
        assert needle in text, needle
    lowered = text.lower()
    for overclaim in ("retention is solved", "elpis now remembers", "continual learning is achieved",
                      "understands language", "promoted to canonical"):
        assert overclaim not in lowered, overclaim
