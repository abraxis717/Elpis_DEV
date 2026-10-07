"""H-ECS R0 evidence guards (research/hecs_r0/PREREGISTRATION.md).

The study stopped at DEV calibration with TASK_INVALID_ON_DEV. These guards keep that record honest: the
calibration ran under the frozen specification, its decision follows the preregistered rule from its own
numbers, and no DEV run or QUAL evidence exists (nothing was run past the stop).
"""
from __future__ import annotations

import json
from pathlib import Path

LAB = Path(__file__).resolve().parents[3] / "research" / "hecs_r0"
CALIBRATION = LAB / "evidence" / "dev" / "hecs-r0.v1.calibration.json"
RESULTS = LAB.parents[1] / "docs" / "research" / "HECS_R0_RESULTS.md"


def _calibration():
    return json.loads(CALIBRATION.read_text())


def test_calibration_ran_under_the_frozen_specification():
    frozen = json.loads((LAB / "specs" / "hecs-r0.v1.spec.json").read_text())
    evidence = _calibration()
    assert evidence["spec"] == frozen
    assert evidence["phase"] == "DEV_CALIBRATION"
    assert frozen["dev_seeds"] == [0, 1, 2, 3]


def test_the_decision_follows_the_preregistered_rule_from_its_own_numbers():
    evidence = _calibration()
    limit = evidence["spec"]["gates"]["V1_L1_ONE_STEP_NMSE_MAX"]
    grid = evidence["spec"]["ecs"]["calibration_steps"]
    rows = evidence["grid"]
    assert [r["steps"] for r in rows] == grid[:len(rows)]
    chosen = None
    for row in rows:
        values = row["l1_one_step_nmse"]
        assert len(values) == 2 * len(evidence["spec"]["dev_seeds"])  # both worlds, every DEV seed
        assert row["worst"] == max(values)
        if max(values) <= limit:
            chosen = row["steps"]
            break
    assert chosen == evidence["chosen_steps"] is None
    assert len(rows) == len(grid)  # every budget was tried before invalidity was declared
    assert evidence["disposition"] == "TASK_INVALID_ON_DEV"


def test_the_stop_was_honoured_and_reported():
    assert sorted(p.name for p in (LAB / "evidence").rglob("*.json")) == [CALIBRATION.name]
    text = RESULTS.read_text()
    assert "TASK_INVALID_ON_DEV" in text and "NO_CANONICAL_PROMOTION" in text
