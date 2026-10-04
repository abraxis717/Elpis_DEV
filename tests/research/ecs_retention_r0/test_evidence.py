"""Retention R0 evidence guards: chronology, frozen authority, write-once evidence, exact reproduction, honest report.

The specification and candidates predate DEV (proven from the git history, which CI checks out in
full); the frozen laboratory source is the source in this tree; the write-once records are byte-identical
to the ones written; every DEV and QUAL record carries the full implementation binding; QUAL worlds
reproduce their recorded measurements exactly when the bound implementation is the one present; the
results document states the disposition the evidence yields and claims no more.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

from research.ecs_retention_r0 import protocol as P
from research.ecs_retention_r0 import run as R

from ...ECS_G.test_math_r0 import REPO, _library_path

RESULTS = REPO / "docs" / "research" / "ECS_RETENTION_R0_RESULTS.md"
FROZEN = P.load(R.FROZEN_PATH, "frozen")
QUAL = P.load(R.QUAL_PATH, "qual")


# SHA-256 of each write-once record as written (one commit each: e444814, e35ffdc, 57f238b, 71eb4e3, f7b6ab6).
EVIDENCE_SHA256 = {
    "specs/ecsg-retention-r0.v1.spec.json": "4317c658472c2c992c44d5f1d0c4a1fd69679c957b5a4848b091f03d8ad89d40",
    "evidence/dev/ecsg-retention-r0.v1.dev.json": "9857f3bbb47152415a862d219d84ddae083e122b95c51ee86aa1e3e321bde95f",
    "evidence/dev/ecsg-retention-r0.v1.dev.r2.json": "4a51e3629a7d58dd50932b4eaaadad5dde9a3ab4cba87d8d363d1839d8067453",
    "frozen/ecsg-retention-r0.v1.frozen.json": "c54bebeab17dad336c000ecbd332b796d9a1f1b0152319d423a15f0c45b222bf",
    "evidence/qual/ecsg-retention-r0.v1.qual.json": "b7864fdbae08383b05f4f6ce9fe03979297895f92fbc356ed013e3807b74020b",
}
# The pre-registered order (newest last) and the step that must have added each record.
CHRONOLOGY = ("RET0A.1", "RET0B", "RET0C", "RET0D")
ADDED_BY = (("specs/ecsg-retention-r0.v1.spec.json", "RET0A.1"),
            ("evidence/dev/ecsg-retention-r0.v1.dev.json", "RET0B"),
            ("frozen/ecsg-retention-r0.v1.frozen.json", "RET0C"),
            ("evidence/qual/ecsg-retention-r0.v1.qual.json", "RET0D"))
HISTORY_INCOMPLETE = "RETENTION_AUTHORITY_HISTORY_INCOMPLETE"
CHRONOLOGY_VIOLATED = "RETENTION_CHRONOLOGY_VIOLATED"


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


@pytest.mark.parametrize("name", sorted(EVIDENCE_SHA256))
def test_write_once_evidence_bytes_are_unchanged(name):
    data = (R.ROOT / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == EVIDENCE_SHA256[name], f"{name} changed after it was written"


def test_frozen_authority_still_matches_the_laboratory_and_specification():
    body = FROZEN["body"]
    assert body["lab_source_digest"] == P.source_digest()
    digests = P.spec_digests(P.load_spec())
    assert all(body[k] == digests[k] for k in ("spec", "pass_rule", "candidates_sha256"))
    assert body["pass_rule_text"] == P.load_spec()["pass_rule"]
    assert body["qual_worlds"] == list(P.world_ids(P.load_spec(), "QUAL"))


def test_the_specification_predates_dev_and_the_freeze_predates_qual():
    """RET0A.1 (specification) before RET0B (DEV) before RET0C (freeze) before RET0D (QUAL), from git history.

    CI checks out full history (``fetch-depth: 0``) in every job that runs this. A shallow or malformed
    checkout fails here by name, listing what it cannot see, never with a bare lookup error.
    """
    inside = _git("rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        pytest.fail(f"{HISTORY_INCOMPLETE}: {REPO} is not a git checkout "
                    f"({(inside.stderr or inside.stdout).strip()}); the chronology gate needs the repository history")
    shallow = _git("rev-parse", "--is-shallow-repository").stdout.strip() == "true"
    where = " in a shallow checkout; check out full history (actions/checkout fetch-depth: 0)" if shallow else ""
    log = _git("log", "--format=%H %s", "--", "research/ecs_retention_r0").stdout.splitlines()
    order = {}
    for i, line in enumerate(log):
        tag = line.partition(" ")[2].split(" ", 1)[0]
        if tag.startswith("RET0"):
            order.setdefault(tag, i)
    missing = [m for m in CHRONOLOGY if m not in order]
    assert not missing, (f"{HISTORY_INCOMPLETE}: missing {', '.join(missing)} among the {len(log)} visible "
                         f"commit(s) touching research/ecs_retention_r0{where}")
    # git log is newest first: an older step has a larger index.
    ranks = [order[m] for m in CHRONOLOGY]
    assert ranks == sorted(ranks, reverse=True), f"{CHRONOLOGY_VIOLATED}: {dict(zip(CHRONOLOGY, ranks))}"
    for path, step in ADDED_BY:
        adds = _git("log", "--diff-filter=A", "--format=%s", "--",
                    f"research/ecs_retention_r0/{path}").stdout.splitlines()
        assert adds, f"{HISTORY_INCOMPLETE}: no visible commit adds research/ecs_retention_r0/{path}{where}"
        assert [s.split(" ", 1)[0] for s in adds] == [step], (
            f"{CHRONOLOGY_VIOLATED}: {path} added by {adds!r}; expected exactly one commit, {step}")


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


# Runs in a fresh interpreter whose BLAS/OpenMP thread counts are fixed in the environment before NumPy loads.
# In-process, an earlier test may already have loaded NumPy with a multi-threaded OpenBLAS while the environment
# (and hence the recorded profile) says 1; LAPACK least squares (the representability ceiling) then differs in its
# last bits. QUAL itself ran with the laboratory package imported before NumPy (effective single thread).
_REPRODUCE = """
import ctypes, json, sys
from elpis.ECS_G.native import ECSGLibrary
from research.ecs_retention_r0 import experiment as X, protocol as P, run as R
library, world = sys.argv[1], sys.argv[2]
qual = P.load(R.QUAL_PATH, "qual")["body"]
stale = P.binding_mismatch(qual["implementation"], P.implementation(library))
if stale:
    got = None
else:
    lab = X.Lab(ECSGLibrary(ctypes.CDLL(library)), P.load_spec())
    got = json.loads(P.canonical_json(X.qual_world(lab, world, qual["choices"])))
sys.stdout.write(json.dumps({"stale": stale, "world": got}))
"""


@pytest.mark.parametrize("world", ["qual-0000", "qual-0013", "qual-0020"])
def test_qual_measurements_reproduce_exactly(world):
    """A QUAL world re-run gives its recorded measurements bit for bit (same implementation and profile)."""
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", _REPRODUCE, str(_library_path()), world], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True)
    result = json.loads(out.stdout)
    if result["stale"]:
        pytest.skip(f"evidence bound to a different implementation or profile: {result['stale']}")
    assert _strip_timing(result["world"]) == _strip_timing(QUAL["body"]["per_world"][world])


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
