"""The native K1 qualification record: write-once, bound to the plan and the harness commit, and its verdict follows
from its own rows under the plan (research/ecs_k1_native/PLAN.md). The record is NOT_QUALIFIED as found.

The never-skipped current-runtime regression is test_runtime_regression.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "research" / "ecs_k1_native"
RECORD = ROOT / "evidence" / "ecsg-k1-native.v1.qualification.json"
RECORD_SHA256 = "df730b5f1542faace08fad9079bdae403c049b5d051b29a3a0bc73409018f374"
HARNESS_COMMIT = "44d4d853035a3577e2972807c818ff786945eb59"      # K1N-C
PLAN = json.loads((ROOT / "specs" / "ecsg-k1-native.v1.plan.json").read_text(encoding="utf-8"))
BODY = json.loads(RECORD.read_bytes())["body"]
BOUNDARY_KEYS = ("w_relative", "h_relative", "a_relative", "query_relative", "nmse_relative")


def test_record_bytes_are_write_once():
    assert hashlib.sha256(RECORD.read_bytes()).hexdigest() == RECORD_SHA256, "the qualification record changed"


def test_record_digest_matches_its_content():
    pytest.importorskip("numpy")
    from research.ecs_k1_native import run as R
    R.load_record()


def test_record_is_bound_to_the_plan_the_r3_authority_and_the_clean_harness_commit():
    impl = BODY["implementation"]
    assert impl["base_commit"] == HARNESS_COMMIT and impl["dirty"] is False
    assert impl["plan_sha256"] == hashlib.sha256((ROOT / "specs" / "ecsg-k1-native.v1.plan.json").read_bytes()).hexdigest()
    assert impl["plan_md_sha256"] == hashlib.sha256((ROOT / "PLAN.md").read_bytes()).hexdigest()
    assert BODY["r3_qual_digest"] == PLAN["authority"]["r3_qual"]["digest"]
    assert BODY["tolerances"] == PLAN["tolerances"]
    r3 = json.loads((REPO / PLAN["authority"]["r3_qual"]["path"]).read_bytes())["body"]
    assert BODY["world_ids"] == r3["world_ids"] and len(BODY["per_world"]) == 32


def test_every_tolerance_check_follows_from_the_recorded_numbers():
    tol = BODY["tolerances"]
    for wid, row in BODY["per_world"].items():
        c = row["comparisons"]
        bounds = c["boundaries"].values()
        within = all(b["epoch"] == b["epoch_expected"] and b["provenance"] == "COMPLETE"
                     and all(b[key] <= tol[key] for key in BOUNDARY_KEYS) for b in bounds)
        assert row["checks"]["boundaries"] == within, wid
        if row["checks"]["reset_challenge"]:
            assert c["reset"]["e_reset_relative"] <= tol["e_reset_relative"], wid
        assert row["pass"] == all(row["checks"].values()), wid


def test_recorded_verdict_follows_from_the_plan_rule():
    failed = sorted(w for w, r in BODY["per_world"].items() if not r["pass"])
    assert failed == BODY["failed_worlds"]
    qualified = (not failed and BODY["determinism"]["identical"] and BODY["clean_transplant"]["bitwise"]
                 and all(BODY["decision_record_equal"].values()) and BODY["outcome"] == "OUTCOME_A")
    assert BODY["verdict"] == ("QUALIFIED" if qualified else "NOT_QUALIFIED")


def test_qualification_is_recorded_as_found():
    """NOT_QUALIFIED: two worlds exceed the planned float tolerances; every exact comparison held."""
    assert BODY["verdict"] == "NOT_QUALIFIED"
    assert BODY["failed_worlds"] == ["r3qual-0019", "r3qual-0028"]
    assert BODY["worlds_passing"] == 30
    for wid in BODY["failed_worlds"]:
        failing = {k for k, v in BODY["per_world"][wid]["checks"].items() if not v}
        assert failing == {"boundaries", "reset_challenge"}, (wid, failing)
    exact = ("zero_k1_parity", "w_a", "resident_equals_standalone", "classification", "state_removal_classification",
             "w_only_negative_control", "transplant_in_process", "state_semantics")
    for key in exact:
        assert BODY["summary"]["checks_passing"][key] == 32, key
    assert BODY["determinism"]["identical"] and BODY["clean_transplant"]["bitwise"]
    assert all(BODY["decision_record_equal"].values())
    assert BODY["outcome"] == "OUTCOME_A" and all(BODY["gates"].values())


def test_the_failing_reset_challenges_keep_their_recorded_classes():
    r3 = json.loads((REPO / PLAN["authority"]["r3_qual"]["path"]).read_bytes())["body"]["per_world"]
    for wid in BODY["failed_worlds"]:
        rec = r3[wid]["P"]["causality"]["reset_challenge"]
        assert BODY["per_world"][wid]["comparisons"]["reset"]["RESET_DEGRADED"] == rec["RESET_DEGRADED"], wid
