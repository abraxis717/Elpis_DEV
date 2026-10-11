"""The independent fitness evaluator: it recomputes fitness itself. RESEARCH_ONLY. UNQUALIFIED. NO_CLAIM.

:func:`evaluate` has the evaluator signature of ``elpis.evolution.policy`` and is meant to be pinned by an evolution
policy whose evaluation contract names exactly this environment's frozen partitions (:func:`policy_spec`). It:

1. refuses partitions other than the frozen ones (``PARTITION_MISMATCH``): the policy cannot point it elsewhere;
2. reads the parent's and the candidate's *declarative* policies (a malformed candidate fails correctness; a
   malformed incumbent refuses the evaluation);
3. checks itself on CALIBRATION: the parent's fitness is recomputed twice and every calibration trajectory replays
   to its own identity, or the evaluation is refused (``EVALUATOR_NONDETERMINISTIC``);
4. runs both policies on HELD_OUT and OOD and computes the deltas from the episode scores the environment assigns
   (anything the candidate writes about its own fitness is never read);
5. fails correctness unless the candidate beats every negative baseline on HELD_OUT and its trajectories replay;
6. fails leakage when the candidate workspace contains any evaluation world identifier or partition digest;
7. reports a resource cost of the candidate's HELD_OUT steps / 10.

It writes nothing and executes nothing from a workspace.
"""
from __future__ import annotations

from pathlib import Path

from elpis.evolution.policy import EvaluatorReport, implementation_identity

from .environment import Trajectory, World, run, verify
from .partitions import (
    NEGATIVE_BASELINES,
    PARTITIONS,
    PolicyRefusal,
    load_policy,
    partition_digest,
    promotion_partitions,
)


class FitnessRefusal(ValueError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def trajectories(table, partition: str) -> tuple[Trajectory, ...]:
    spec, worlds = PARTITIONS[partition]
    return tuple(run(World(spec, w), table.__getitem__) for w in worlds)


def fitness(table, partition: str) -> int:
    return sum(t.score for t in trajectories(table, partition))


def _leaks(root: Path) -> bool:
    secrets = [w.encode() for name in ("CALIBRATION", "HELD_OUT", "OOD") for w in PARTITIONS[name][1]]
    secrets += [partition_digest(name).encode() for name in PARTITIONS]
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            if any(secret in data for secret in secrets):
                return True
    return False


def evaluate(*, parent_root, candidate_root, partitions) -> EvaluatorReport:
    if dict(partitions) != promotion_partitions():
        raise FitnessRefusal("PARTITION_MISMATCH", "the contract does not name this environment's frozen partitions")
    try:
        parent = load_policy(parent_root)
    except PolicyRefusal as exc:
        raise FitnessRefusal("INCUMBENT_INVALID", str(exc)) from exc
    calibration = trajectories(parent, "CALIBRATION")
    if sum(t.score for t in calibration) != fitness(parent, "CALIBRATION") or not all(map(verify, calibration)):
        raise FitnessRefusal("EVALUATOR_NONDETERMINISTIC")
    leaked = _leaks(Path(candidate_root))
    try:
        candidate = load_policy(candidate_root)
    except PolicyRefusal:
        return EvaluatorReport(correctness_pass=False, leakage_pass=not leaked, resource_cost=0, held_out_delta=0,
                               ood_delta=0)
    held = trajectories(candidate, "HELD_OUT")
    held_score = sum(t.score for t in held)
    best_negative = max(fitness(baseline, "HELD_OUT") for baseline in NEGATIVE_BASELINES.values())
    correct = held_score > best_negative and all(map(verify, held))
    return EvaluatorReport(correctness_pass=correct, leakage_pass=not leaked,
                           resource_cost=sum(len(t.steps) for t in held) // 10,
                           held_out_delta=held_score - fitness(parent, "HELD_OUT"),
                           ood_delta=fitness(candidate, "OOD") - fitness(parent, "OOD"))


def policy_spec(*, provenance: str = "test-fixture", candidate_scopes=("agent",),
                protected_scopes=("evaluator", "environment", "policy")) -> dict:
    """An ``elpis.evolution-policy.v1`` spec that pins this evaluator and this environment's frozen partitions."""
    implementation, sha256 = implementation_identity(evaluate)
    return {
        "schema": "elpis.evolution-policy.v1",
        "source": "RESEARCH_ONLY Evolution Fitness R0 corridor environment (research/evolution_fitness_r0)",
        "provenance": provenance,
        "objective": {"objective_id": "corridor-held-out-score", "metric": "held_out_delta", "direction": "MAXIMIZE"},
        "evaluator": {"evaluator_id": "corridor-fitness-r0", "implementation": implementation,
                      "implementation_sha256": sha256},
        "candidate_scopes": list(candidate_scopes),
        "protected_scopes": list(protected_scopes),
        "budget": {"max_edit_budget": 8, "max_resource_cost": 60, "max_candidates": 8},
        "evaluation_contract": {"partitions": promotion_partitions(), "noise_envelope": 3, "ood_regression_floor": 0},
        "side_effects": "CANDIDATE_WORKSPACE_ONLY",
        "confinement": "TRUSTED_OPERATOR_CALLBACKS",
        "promotion": {"requires_operator_approval": True},
    }
