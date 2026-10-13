"""ECS Memory MF-R0 laboratory: preregistration, freeze and write-once evidence (research/ecs_memory_mf_r0).

SCIENTIFIC lane. Every recorded verdict is recomputed from the recorded counts by the frozen scorer; the QUAL
record is bound to the freeze manifest, which is bound to the DEV record and the sources; the chronology
preregistration -> laboratory -> DEV -> freeze -> QUAL is read from git history; and while the bound sources
are unchanged, one QUAL seed is re-run on the current build and must reproduce every integer count exactly.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from tests.conftest import native_build_dir, native_required

REPO = Path(__file__).resolve().parents[3]
LAB = REPO / "research" / "ecs_memory_mf_r0"
EVIDENCE = LAB / "evidence"
FREEZE = EVIDENCE / "FREEZE.json"
DEV = EVIDENCE / "dev" / "RECORD.json"
QUAL = EVIDENCE / "qual" / "RECORD.json"

# Write-once: the exact bytes of the three evidence files.
PINNED = {
    FREEZE: "3cf46ea56655720abad68904f220da65f012517a4652aa679145f7e99dc2d199",
    DEV: "f33167b1916542534a4bac5bc90d0577d450718b056069caaa80192a2e3d80ce",
    QUAL: "5454cbf136d947d6ccf3bbd495de7cd49c6dc29dd2d9961ce421bece0d6f8f20",
}
QUAL_VERDICTS = {"H1": "PASS", "H2": "PASS", "H3": "PASS", "H4": "NONPASS", "H5": "PASS"}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"ecs_mem_lab_{name}", LAB / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict:
    return json.loads(path.read_text())


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def test_evidence_bytes_are_pinned():
    for path, digest in PINNED.items():
        assert _sha(path) == digest, path


def test_records_are_sealed():
    run = _load("run")
    for path in PINNED:
        record = _record(path)
        assert run.seal(record)["record_sha256"] == record["record_sha256"], path


def test_verdicts_recompute_under_the_frozen_scorer():
    score = _load("score").score
    for path in (DEV, QUAL):
        record = _record(path)
        rows = [row for doc in record["runs"] for row in doc["rows"]]
        assert score(rows) == record["scored"], path
    verdicts = {h: v["verdict"] for h, v in _record(QUAL)["scored"]["verdicts"].items()}
    assert verdicts == QUAL_VERDICTS


def test_qual_is_bound_to_the_freeze_and_the_freeze_to_dev():
    freeze, dev, qual = _record(FREEZE), _record(DEV), _record(QUAL)
    assert qual["phase"] == "QUAL" and dev["phase"] == "DEV"
    assert qual["freeze_sha256"] == _sha(FREEZE)
    assert freeze["dev_record_sha256"] == _sha(DEV)
    assert qual["sources"] == freeze["sources"] == dev["sources"]
    assert qual["seeds"] == freeze["qual_seeds"] == list(range(1001, 1009))
    assert not set(qual["seeds"]) & set(dev["seeds"])
    assert qual["parameters"] == dev["parameters"]
    for record in (dev, qual):
        assert all(doc["seed"] in record["seeds"] and not doc["smoke"] for doc in record["runs"])
        assert sorted(doc["seed"] for doc in record["runs"]) == record["seeds"]
        assert record["build"]["sanitizers"] == {k: "OFF" for k in record["build"]["sanitizers"]}


def _added(path: Path) -> str:
    return _git("log", "--diff-filter=A", "--format=%H", "--", str(path.relative_to(REPO))).splitlines()[-1]


def _ancestor(older: str, newer: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", older, newer], cwd=REPO).returncode == 0


def test_chronology_preregistration_laboratory_dev_freeze_qual():
    if _git("rev-parse", "--is-shallow-repository") == "true":
        pytest.fail("chronology needs full history (docs/CI_POLICY.md: fetch-depth 0)")
    order = [LAB / "PREREGISTRATION.md", LAB / "lab" / "ecs_mem_lab.c", DEV, FREEZE, QUAL]
    commits = [_added(p) for p in order]
    for older, newer in zip(commits, commits[1:]):
        assert older != newer and _ancestor(older, newer), (older, newer)
    # Write-once: each evidence file was committed exactly once.
    for path in (DEV, FREEZE, QUAL):
        assert len(_git("log", "--format=%H", "--", str(path.relative_to(REPO))).splitlines()) == 1, path


def test_current_runtime_reproduces_a_qual_seed_exactly():
    """Integer arms are exact by construction (ecs_mem.h); D31C is binary32 arithmetic and is compared within
    one answer per row. Historical replay is demanded only while the bound sources are unchanged."""
    run = _load("run")
    freeze = _record(FREEZE)
    if run.source_digests() != freeze["sources"]:
        pytest.skip("bound sources changed after the freeze: historical replay is not demanded (never a PASS)")
    binary = native_build_dir() / "research" / "ecs_memory_mf_r0" / "ecs_mem_lab"
    if not binary.is_file():
        message = f"laboratory binary missing under {native_build_dir()}"
        if native_required():
            pytest.fail(message)
        pytest.skip(message)
    out = subprocess.run([str(binary), "--seed", "1001"], check=True, capture_output=True, text=True).stdout
    fresh = json.loads(out)
    recorded = next(doc for doc in _record(QUAL)["runs"] if doc["seed"] == 1001)
    assert fresh["parameters"] == recorded["parameters"]
    assert len(fresh["rows"]) == len(recorded["rows"])
    for a, b in zip(fresh["rows"], recorded["rows"]):
        assert (a["scenario"], a["condition"], a["arm"]) == (b["scenario"], b["condition"], b["arm"])
        keys = set(a) - {"query_ns"}
        if a["arm"] == "D31C":
            assert abs(a["correct"] - b["correct"]) <= 1 and a["total"] == b["total"], a
        else:
            assert {k: a[k] for k in keys} == {k: b[k] for k in keys}, (a, b)
