"""R2 DEV-only reproducibility audit. No Codex, dependencies, source rewrites, or QUAL."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "research/ecsc_hca_r0"
PIN = "cb530e7a0c290b9ceeb6614ab1db4ab45ada54e3"
TESTS = (
    "research.ecsc_hca_r0.test_baselines",
    "research.ecsc_hca_r0.test_task",
    "research.ecsc_hca_r0.test_b2",
    "research.ecsc_hca_r0.test_r2_mechanics",
)
CANONICAL_TESTS = (
    "tests/integration/test_context_lane.py::test_successor_is_pure_and_deterministic",
    "tests/integration/test_context_lane.py::test_references_and_relations_keep_their_epistemic_status",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def verify_inputs(frozen: dict) -> dict:
    assert git("rev-parse", "HEAD") == PIN, "SOURCE_PIN_DRIFT"
    assert git("branch", "--show-current") == "research/ecsc_hca_r0", "RESEARCH_BRANCH_DRIFT"
    assert not subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--exit-code", "HEAD", "--", ".", ":!research/ecsc_hca_r0"],
        stdout=subprocess.DEVNULL, check=False,
    ).returncode, "TRACKED_CANONICAL_DRIFT"
    source = {}
    for rel, expected in frozen["revisions"].items():
        actual = sha(ROOT / rel)
        assert actual == expected, f"FROZEN_INPUT_DRIFT:{rel}"
        source[rel] = actual
    canonical = sha(ROOT / "src/elpis/runtime/context.py")
    assert canonical == frozen["canonical_context_sha256"], "CANONICAL_CONTEXT_DRIFT"
    return {"head": PIN, "canonical": canonical, "frozen": source}


def r2_hashes() -> dict:
    # Explicit original R2 evidence set: the verifier itself must not be treated as
    # an experimental input or manufacture a recorded-source mismatch.
    names = ("r2_b0i.py", "r2_b2s.py", "r2_common.py",
             "r2_run.py", "r2_trajectories.py", "test_r2_mechanics.py")
    return {name: sha(LOCAL / name) for name in names}


def strip_time(item):
    if isinstance(item, dict):
        return {k: strip_time(v) for k, v in item.items()
                if k not in ("wall_ns_samples", "shared_event_construction_wall_ns")}
    if isinstance(item, list):
        return [strip_time(v) for v in item]
    return item


def main():
    print("===== R2 LOCAL VERIFICATION — NO REMOTE AGENT =====", flush=True)
    manifest = json.loads((LOCAL / "BASELINE_R1_FREEZE.json").read_text())
    before = verify_inputs(manifest)
    before_hashes = r2_hashes()
    previous = json.loads((LOCAL / "R2_DEV_RESULTS.json").read_text())
    old_hashes = previous.get("r2_source_hashes", {})
    drift = sorted(k for k, h in before_hashes.items() if old_hashes.get(k) != h)
    print("FROZEN_INPUTS=PASS", "R2_RECORDED_HASH_DRIFT=" + (",".join(drift) or "none"), flush=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="src:.")
    commands = [
        [sys.executable, "-m", "unittest", *TESTS, "-v"],
        [sys.executable, "-m", "pytest", *CANONICAL_TESTS, "-q", "-p", "no:cacheprovider"],
    ]
    for command in commands:
        print("===== EXEC =====", " ".join(command), flush=True)
        rc = subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode
        print("TEST_EXIT=", rc, flush=True)
        if rc:
            print("BLOCKED_TEST_FAILURE; original evidence left unchanged", flush=True)
            return rc
    # Source/contract hashes also verified after the test run, before evidence is accepted.
    assert verify_inputs(manifest) == before, "FROZEN_INPUT_CHANGED_DURING_TESTS"
    assert r2_hashes() == before_hashes, "R2_SOURCE_CHANGED_DURING_TESTS"
    print("===== IN-PROCESS R2 REPRODUCTION =====", flush=True)
    from .r2_run import run_trajectory, damage_control
    current_events = run_trajectory(4) + run_trajectory(8)
    current_damage = damage_control()
    assert len(current_events) == 40, "R2_EVENT_COUNT"
    event_equal = strip_time(current_events) == strip_time(previous["events"])
    damage_equal = strip_time(current_damage) == strip_time(previous["cycle_damage"])
    counts = Counter((name, record["outcome"])
                     for row in current_events for name, record in row["methods"].items())
    source_after = verify_inputs(manifest)
    assert source_after == before and r2_hashes() == before_hashes, "POSTCHECK_DRIFT"
    summary = {
        "phase": "R2_LOCAL_MECHANICS_REPRODUCTION_NOT_SCIENCE",
        "source_pin": PIN,
        "scientific_status": "NOT_RUN_NOT_QUALIFIED",
        "tests": "UNITTEST_AND_CANONICAL_SELECTED_PASS",
        "frozen_input_hashes_verified": len(before["frozen"]),
        "canonical_context_sha256": before["canonical"],
        "recorded_source_hash_drift": drift,
        "current_r2_source_hashes": before_hashes,
        "original_result_sha256": sha(LOCAL / "R2_DEV_RESULTS.json"),
        "events": len(current_events),
        "outcomes": {f"{method}.{status}": n for (method, status), n in sorted(counts.items())},
        "event_results_identical_excluding_timing": event_equal,
        "damage_results_identical_excluding_timing": damage_equal,
        "result": "REPRODUCED_WITH_CURRENT_SOURCE" if event_equal and damage_equal else "RESULT_DRIFT_REVIEW_REQUIRED",
        "limit": "Descriptive mechanics only; no external source attestation or scientific qualification",
    }
    # A single compact atomic audit record; preserve the original R2 result unchanged.
    output = LOCAL / "R2_LOCAL_VERIFICATION.json"
    tmp = LOCAL / ".R2_LOCAL_VERIFICATION.json.tmp"
    tmp.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    tmp.replace(output)
    print("===== AUDIT RESULT =====", summary["result"], flush=True)
    print("RESULTS_METADATA_DRIFT=", drift, flush=True)
    print("REPRODUCED_EVENTS=", len(current_events), flush=True)
    print("REPORT=", output, flush=True)
    return 0 if event_equal and damage_equal else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, OSError, ValueError, KeyError) as exc:
        print("BLOCKED_R2_VERIFICATION:", exc, flush=True)
        sys.exit(2)
