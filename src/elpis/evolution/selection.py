"""Population state and deterministic, self-verified truncation selection.

Selection takes no externally proposed solution. The commit itself:

1. binds the exact population revision the caller observed;
2. requires exactly one fitness record per eligible organism (ALIVE or
   REPRODUCTIVE), each bound to that organism's current revision;
3. recomputes every scalar fitness under the stated fitness policy;
4. ranks by scalar fitness (descending), then organism id, keeps the policy's
   survivor count and transitions every other eligible organism to DYING.

Any defect rejects the whole commit and leaves the population unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .canonical import (
    payload_digest,
    require_sha256,
)
from .fitness import FitnessPolicyV1, OrganismFitnessRecord
from .lifecycle import (
    LifecycleState,
    transition_lifecycle,
)
from .organism import OrganismState


POPULATION_STATE_SCHEMA = "darwinian.life.population-state.v1"
TRUNCATION_SELECTION_POLICY_SCHEMA = "elpis.evolution.truncation-selection-policy.v1"
SELECTION_COMMIT_RESULT_SCHEMA = "elpis.evolution.selection-commit-result.v1"

ELIGIBLE_LIFECYCLES = frozenset({LifecycleState.ALIVE, LifecycleState.REPRODUCTIVE})


def _require_nonnegative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(field_name + " must be an integer.")
    if value < 0:
        raise ValueError(field_name + " cannot be negative.")
    return value


class SelectionRejectionCode(str, Enum):
    STALE_POPULATION = "STALE_POPULATION"
    NO_ELIGIBLE_CANDIDATE = "NO_ELIGIBLE_CANDIDATE"
    FITNESS_POLICY_MISMATCH = "FITNESS_POLICY_MISMATCH"
    DUPLICATE_FITNESS_RECORD = "DUPLICATE_FITNESS_RECORD"
    UNEXPECTED_FITNESS_RECORD = "UNEXPECTED_FITNESS_RECORD"
    MISSING_FITNESS_RECORD = "MISSING_FITNESS_RECORD"
    STALE_FITNESS_RECORD = "STALE_FITNESS_RECORD"
    FITNESS_SCORE_MISMATCH = "FITNESS_SCORE_MISMATCH"


@dataclass(frozen=True)
class PopulationState:
    """Canonical population bound to one source world digest."""

    source_world_state_digest: str
    revision: int
    organisms: tuple[OrganismState, ...]
    schema: str = POPULATION_STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != POPULATION_STATE_SCHEMA:
            raise ValueError("Unsupported population-state schema.")
        require_sha256(self.source_world_state_digest, field_name="source_world_state_digest")
        _require_nonnegative_integer(self.revision, field_name="revision")
        organisms = tuple(self.organisms)
        if not organisms:
            raise ValueError("A population requires at least one organism.")
        if any(not isinstance(organism, OrganismState) for organism in organisms):
            raise TypeError("organisms must contain OrganismState objects.")
        ordered = tuple(sorted(organisms, key=lambda organism: organism.organism_id))
        identifiers = tuple(organism.organism_id for organism in ordered)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Population organism identities must be unique.")
        object.__setattr__(self, "organisms", ordered)

    @property
    def organism_ids(self) -> tuple[str, ...]:
        return tuple(organism.organism_id for organism in self.organisms)

    def organism(self, organism_id: str) -> OrganismState:
        for organism in self.organisms:
            if organism.organism_id == organism_id:
                return organism
        raise KeyError(organism_id)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_world_state_digest": self.source_world_state_digest,
            "revision": self.revision,
            "organisms": [organism.canonical_payload() for organism in self.organisms],
        }

    def digest(self) -> str:
        return payload_digest(self.canonical_payload())


@dataclass(frozen=True)
class TruncationSelectionPolicyV1:
    """Keep the ``survivor_count`` fittest eligible organisms."""

    survivor_count: int
    schema: str = TRUNCATION_SELECTION_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRUNCATION_SELECTION_POLICY_SCHEMA:
            raise ValueError("Unsupported truncation-selection policy schema.")
        if _require_nonnegative_integer(self.survivor_count, field_name="survivor_count") < 1:
            raise ValueError("survivor_count must be at least one.")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "survivor_count": self.survivor_count,
            "eligible_lifecycles": sorted(state.value for state in ELIGIBLE_LIFECYCLES),
            "ranking": "SCALAR_FITNESS_DESC_THEN_ORGANISM_ID",
            "excluded_candidate_transition": LifecycleState.DYING.value,
            "selected_candidate_transition": "UNCHANGED",
            "noncandidate_transition": "UNCHANGED",
            "atomic": True,
        }

    def digest(self) -> str:
        return payload_digest(self.canonical_payload())


@dataclass(frozen=True)
class SelectionCommitResult:
    accepted: bool
    rejection_code: SelectionRejectionCode | None
    policy_digest: str
    fitness_policy_digest: str
    population_before: PopulationState
    population_after: PopulationState
    ranking: tuple[tuple[str, int], ...]
    selected_ids: tuple[str, ...]
    transitioned_to_dying: tuple[str, ...]
    schema: str = SELECTION_COMMIT_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SELECTION_COMMIT_RESULT_SCHEMA:
            raise ValueError("Unsupported selection-commit result schema.")
        require_sha256(self.policy_digest, field_name="policy_digest")
        require_sha256(self.fitness_policy_digest, field_name="fitness_policy_digest")
        if not isinstance(self.population_before, PopulationState) or not isinstance(
                self.population_after, PopulationState):
            raise TypeError("population_before and population_after must be PopulationState.")
        object.__setattr__(self, "selected_ids", tuple(sorted(self.selected_ids)))
        object.__setattr__(self, "transitioned_to_dying", tuple(sorted(self.transitioned_to_dying)))
        if self.accepted:
            if self.rejection_code is not None:
                raise ValueError("Accepted selection cannot carry a rejection code.")
            if self.population_after.revision != self.population_before.revision + 1:
                raise ValueError("Accepted selection must increment the population revision once.")
        else:
            if not isinstance(self.rejection_code, SelectionRejectionCode):
                raise ValueError("Rejected selection requires a rejection code.")
            if self.population_after != self.population_before:
                raise ValueError("Rejected selection must leave the population unchanged.")
            if self.transitioned_to_dying or self.selected_ids or self.ranking:
                raise ValueError("Rejected selection cannot select or transition organisms.")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "accepted": self.accepted,
            "rejection_code": None if self.rejection_code is None else self.rejection_code.value,
            "policy_digest": self.policy_digest,
            "fitness_policy_digest": self.fitness_policy_digest,
            "population_before_digest": self.population_before.digest(),
            "population_after_digest": self.population_after.digest(),
            "ranking": [[organism_id, score] for organism_id, score in self.ranking],
            "selected_ids": list(self.selected_ids),
            "transitioned_to_dying": list(self.transitioned_to_dying),
        }

    def digest(self) -> str:
        return payload_digest(self.canonical_payload())


def _rejected(code: SelectionRejectionCode, population: PopulationState,
              policy: TruncationSelectionPolicyV1, fitness_policy: FitnessPolicyV1) -> SelectionCommitResult:
    return SelectionCommitResult(
        accepted=False, rejection_code=code, policy_digest=policy.digest(),
        fitness_policy_digest=fitness_policy.digest(), population_before=population,
        population_after=population, ranking=(), selected_ids=(), transitioned_to_dying=())


def commit_selection(
    *,
    population: PopulationState,
    expected_population_digest: str,
    fitness_records: Sequence[OrganismFitnessRecord],
    fitness_policy: FitnessPolicyV1,
    policy: TruncationSelectionPolicyV1,
) -> SelectionCommitResult:
    """Atomically apply deterministic truncation selection to one population revision."""
    if not isinstance(population, PopulationState):
        raise TypeError("population must be a PopulationState.")
    if not isinstance(fitness_policy, FitnessPolicyV1):
        raise TypeError("fitness_policy must be a FitnessPolicyV1.")
    if not isinstance(policy, TruncationSelectionPolicyV1):
        raise TypeError("policy must be a TruncationSelectionPolicyV1.")
    require_sha256(expected_population_digest, field_name="expected_population_digest")
    records = tuple(fitness_records)
    if any(not isinstance(record, OrganismFitnessRecord) for record in records):
        raise TypeError("fitness_records must contain OrganismFitnessRecord objects.")

    def reject(code: SelectionRejectionCode) -> SelectionCommitResult:
        return _rejected(code, population, policy, fitness_policy)

    if population.digest() != expected_population_digest:
        return reject(SelectionRejectionCode.STALE_POPULATION)
    candidates = {o.organism_id: o for o in population.organisms if o.lifecycle in ELIGIBLE_LIFECYCLES}
    if not candidates:
        return reject(SelectionRejectionCode.NO_ELIGIBLE_CANDIDATE)

    policy_digest = fitness_policy.digest()
    by_organism: dict[str, OrganismFitnessRecord] = {}
    for record in records:
        if record.fitness_policy_digest != policy_digest:
            return reject(SelectionRejectionCode.FITNESS_POLICY_MISMATCH)
        if record.organism_id in by_organism:
            return reject(SelectionRejectionCode.DUPLICATE_FITNESS_RECORD)
        if record.organism_id not in candidates:
            return reject(SelectionRejectionCode.UNEXPECTED_FITNESS_RECORD)
        by_organism[record.organism_id] = record
    if set(by_organism) != set(candidates):
        return reject(SelectionRejectionCode.MISSING_FITNESS_RECORD)

    scores: dict[str, int] = {}
    for organism_id, record in by_organism.items():
        if not record.validate_organism(candidates[organism_id]):
            return reject(SelectionRejectionCode.STALE_FITNESS_RECORD)
        score = fitness_policy.score(record.observation)
        if score != record.scalar_fitness:
            return reject(SelectionRejectionCode.FITNESS_SCORE_MISMATCH)
        scores[organism_id] = score

    ranking = tuple(sorted(scores.items(), key=lambda item: (-item[1], item[0])))
    survivors = frozenset(organism_id for organism_id, _ in ranking[:policy.survivor_count])
    excluded = tuple(organism_id for organism_id, _ in ranking[policy.survivor_count:])
    organisms_after = tuple(
        transition_lifecycle(organism, LifecycleState.DYING) if organism.organism_id in excluded else organism
        for organism in population.organisms
    )
    population_after = PopulationState(
        source_world_state_digest=population.source_world_state_digest,
        revision=population.revision + 1,
        organisms=organisms_after,
    )
    return SelectionCommitResult(
        accepted=True, rejection_code=None, policy_digest=policy.digest(),
        fitness_policy_digest=policy_digest, population_before=population,
        population_after=population_after, ranking=ranking,
        selected_ids=tuple(survivors), transitioned_to_dying=excluded)


__all__ = (
    "ELIGIBLE_LIFECYCLES",
    "PopulationState",
    "SelectionCommitResult",
    "SelectionRejectionCode",
    "TruncationSelectionPolicyV1",
    "commit_selection",
)
