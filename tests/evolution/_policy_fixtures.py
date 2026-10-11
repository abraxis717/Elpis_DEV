"""TEST_ONLY evolution policy fixtures: a test-fixture policy pinned by its author (this module) and its evaluator.

The evaluator is an interface fixture, not a fitness environment: it reads an integer ``agent/score.txt`` from each
workspace. Nothing here measures or claims improvement of anything.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from elpis.evolution.policy import (
    EVOLUTION_POLICY_SCHEMA,
    EvaluatorReport,
    EvolutionPolicy,
    EvolutionPolicyAuthority,
    implementation_identity,
)


def d(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


PARTITIONS = {"EVOLVE": d("p-evolve"), "CALIBRATION": d("p-calibration"), "HELD_OUT": d("p-held-out"),
              "OOD": d("p-ood")}


def _score(root: Path) -> int:
    return int((Path(root) / "agent" / "score.txt").read_text())


def score(*, parent_root, candidate_root, partitions) -> EvaluatorReport:
    """TEST_ONLY evaluator: held-out delta = candidate score - parent score."""
    assert set(partitions) == set(PARTITIONS)
    files = sum(1 for p in Path(candidate_root).rglob("*") if p.is_file())
    return EvaluatorReport(correctness_pass=not (Path(candidate_root) / "agent" / "broken").exists(),
                           leakage_pass=True, resource_cost=10 * files,
                           held_out_delta=_score(candidate_root) - _score(parent_root), ood_delta=0)


def scribble(*, parent_root, candidate_root, partitions) -> EvaluatorReport:
    """A TEST_ONLY evaluator with a side effect: it writes into the candidate workspace."""
    (Path(candidate_root) / "agent" / "score.txt").write_text("999")
    return score(parent_root=parent_root, candidate_root=candidate_root, partitions=partitions)


def unreported(*, parent_root, candidate_root, partitions):
    """A TEST_ONLY evaluator that returns a self-made verdict instead of an EvaluatorReport."""
    return {"held_out_delta": 100}


def spec(evaluator=score, **overrides) -> dict:
    implementation, sha256 = implementation_identity(evaluator)
    out = {
        "schema": EVOLUTION_POLICY_SCHEMA,
        "source": "TEST fixture evolution policy (TEST_ONLY)",
        "provenance": "test-fixture",
        "objective": {"objective_id": "test-held-out-score", "metric": "held_out_delta", "direction": "MAXIMIZE"},
        "evaluator": {"evaluator_id": "test-score", "implementation": implementation,
                      "implementation_sha256": sha256},
        "candidate_scopes": ["agent", "evolution/population"],
        "protected_scopes": ["evaluator", "policy"],
        "budget": {"max_edit_budget": 4, "max_resource_cost": 100, "max_candidates": 4},
        "evaluation_contract": {"partitions": dict(PARTITIONS), "noise_envelope": 3, "ood_regression_floor": -1},
        "side_effects": "CANDIDATE_WORKSPACE_ONLY",
        "confinement": "TRUSTED_OPERATOR_CALLBACKS",
        "promotion": {"requires_operator_approval": True},
    }
    out.update(overrides)
    return out


def document(value: dict) -> tuple[bytes, str]:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return raw, hashlib.sha256(raw).hexdigest()


def authority(evaluator=score, **overrides) -> EvolutionPolicyAuthority:
    raw, pin = document(spec(evaluator, **overrides))
    return EvolutionPolicyAuthority(EvolutionPolicy(raw, expected_sha256=pin), evaluator)


TEST_POLICY, TEST_EVOLUTION_PIN = document(spec())
TEST_AUTHORITY = EvolutionPolicyAuthority(EvolutionPolicy(TEST_POLICY, expected_sha256=TEST_EVOLUTION_PIN), score)
