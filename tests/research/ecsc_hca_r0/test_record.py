"""The ECS_C/HCA R0-R19 research record of PR #51 (research/ecsc_hca_r0, DEV/ecsc_hca): kept, never promoted.

SCIENTIFIC lane. The record is historical DEV research. This guard keeps it exactly as migrated (every file
byte-identical to DEV/ecsc_hca/MIGRATION_PROVENANCE.md), keeps it out of the runtime (no src/ module imports
it), keeps its recorded negative dispositions readable, and replays its own mechanics suite against the
current runtime with the record's historical invocation. Passing grants nothing: no HCA law, contextual
reasoning, learned parsing or language claim follows from this record (ELPIS_SYSTEM.json, research).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LAB = REPO / "research" / "ecsc_hca_r0"
DOCS = REPO / "DEV" / "ecsc_hca"
MANIFEST = DOCS / "MIGRATION_PROVENANCE.md"

# BASELINE_R1_FREEZE.json pins this file of the original workspace; the migration did not carry it, so the
# record's own freeze test cannot be completed in this repository (the other eleven revisions are checked below).
NOT_MIGRATED = {"research/ecsc_hca_r0/CHECKSUMS.sha256"}
FREEZE_TEST = "research/ecsc_hca_r0/test_b2.py::B2Mechanics::test_baseline_freeze_is_unchanged"


def _manifest() -> dict[str, str]:
    rows = re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|", MANIFEST.read_text(encoding="utf-8"), re.M)
    return dict(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_migrated_file_is_byte_identical_and_nothing_was_added():
    rows = _manifest()
    assert len(rows) == 87
    for name, digest in rows.items():
        assert _sha(REPO / name) == digest, name
    tracked = subprocess.run(["git", "ls-files", "DEV/ecsc_hca", "research/ecsc_hca_r0"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.split()
    assert set(tracked) == set(rows) | {"DEV/ecsc_hca/MIGRATION_PROVENANCE.md"}


def test_the_baseline_freeze_holds_for_every_revision_the_migration_carried():
    freeze = json.loads((LAB / "BASELINE_R1_FREEZE.json").read_text(encoding="utf-8"))
    checked = 0
    for name, digest in freeze["revisions"].items():
        if name in NOT_MIGRATED:
            assert not (REPO / name).exists() and not (DOCS / Path(name).name).exists(), name
            continue
        # The migration moved the record's Markdown documents from the laboratory into DEV/ecsc_hca.
        path = REPO / name if (REPO / name).exists() else DOCS / Path(name).name
        assert _sha(path) == digest, name
        checked += 1
    assert checked == len(freeze["revisions"]) - len(NOT_MIGRATED) == 11
    assert _sha(REPO / "src/elpis/runtime/context.py") == freeze["canonical_context_sha256"]


def test_no_runtime_module_imports_the_record():
    pattern = re.compile(r"^\s*(from|import)\s+research(\.ecsc_hca_r0)?\b|ecsc_hca", re.M)
    hits = [p for p in (REPO / "src").rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
    assert not hits, hits


def test_the_recorded_negative_dispositions_are_kept():
    provenance = MANIFEST.read_text(encoding="utf-8")
    assert "R19 disposition: `DEV_DESCRIPTIVE_NOT_SCIENTIFIC_QUAL`" in provenance
    assert "destination runtime compatibility is **not qualified**" in provenance
    r19 = (DOCS / "R19_PROTOCOL.md").read_text(encoding="utf-8")
    assert "root-operator-only rule reproduced all 448 outcomes" in r19
    assert "learned primitive gates were correct on only 6/10 cases" in r19
    assert "not an independent scientific qualification" in r19


def test_the_record_replays_its_own_mechanics_suite_against_the_current_runtime():
    """The record's historical invocation, live output captured, exit code and counts checked exactly."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO), str(LAB)]),
               PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "research/ecsc_hca_r0", "-q", "-p", "no:cacheprovider", "--tb=short",
         "-c", os.devnull, "--rootdir", str(REPO), "--deselect", FREEZE_TEST],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    tail = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else run.stderr[-2000:]
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-2000:]
    assert re.fullmatch(r"275 passed, 1 deselected in [\d.]+s(?: \(\d+:\d+:\d+\))?", tail), tail
