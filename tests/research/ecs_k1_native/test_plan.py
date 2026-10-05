"""The native K1 qualification plan: write-once, bound to the frozen Retention R3 authority, committed before any
differential evidence (research/ecs_k1_native/PLAN.md). Needs neither NumPy nor the native library."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "research" / "ecs_k1_native"
PLAN = ROOT / "specs" / "ecsg-k1-native.v1.plan.json"
PINS = {
    "specs/ecsg-k1-native.v1.plan.json": "a06c81f6fdac302e9470c6e6530a121c6851fa10d7b74c2a92a0fa57aa64d4d4",
    "PLAN.md": "b62f6c008a42785b8070caacefa5d03f2fe42ef8914be319de04fe7ce32911ba",
}
EVIDENCE = "research/ecs_k1_native/evidence"


def _plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout


def test_plan_files_are_write_once():
    for name, sha in PINS.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, f"{name} changed after it was committed"


def test_plan_is_bound_to_the_frozen_r3_authority():
    authority = _plan()["authority"]
    for key in ("r3_spec", "r3_frozen", "r3_qual"):
        entry = authority[key]
        assert hashlib.sha256((REPO / entry["path"]).read_bytes()).hexdigest() == entry["sha256"], key
    qual = json.loads((REPO / authority["r3_qual"]["path"]).read_bytes())
    frozen = json.loads((REPO / authority["r3_frozen"]["path"]).read_bytes())
    assert qual["digest"] == authority["r3_qual"]["digest"] and qual["body"]["outcome"] == "OUTCOME_A"
    assert frozen["digest"] == authority["r3_frozen"]["digest"] == qual["body"]["frozen_digest"]
    assert _plan()["worlds"]["count"] == qual["body"]["world_count"] == len(qual["body"]["world_ids"]) == 32


def test_plan_fixes_every_tolerance_and_forbids_host_specific_rescue():
    plan = _plan()
    assert plan["tolerances"] == {"w_relative": 1e-9, "h_relative": 1e-9, "a_relative": 1e-9, "query_relative": 1e-9,
                                  "nmse_relative": 1e-7, "e_reset_relative": 1e-7}
    for word in ("CPU-model branches", "per-host expected outputs", "post-hoc tolerances"):
        assert word in plan["forbidden"]
    assert set(plan["verdict"]) == {"QUALIFIED", "NOT_QUALIFIED"}
    for key in ("zero_k1_parity", "boundaries", "reset_challenge", "w_only_negative_control",
                "transplant_continuation", "classification", "disposition", "resident_equals_standalone"):
        assert key in plan["comparisons"], key


def test_the_plan_was_committed_before_any_differential_evidence():
    plan_adds = _git("log", "--diff-filter=A", "--format=%H", "--", str(PLAN.relative_to(REPO))).split()
    evidence_adds = _git("log", "--diff-filter=A", "--format=%H", "--", EVIDENCE).split()
    if not plan_adds:
        assert not (REPO / EVIDENCE).exists(), "evidence exists although the plan was never committed"
        return
    assert len(plan_adds) == 1
    for sha in evidence_adds:
        assert sha != plan_adds[0], "the plan and the evidence must be separate commits"
        ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", plan_adds[0], sha], cwd=REPO)
        assert ancestor.returncode == 0, f"evidence commit {sha[:12]} does not follow the plan commit"
