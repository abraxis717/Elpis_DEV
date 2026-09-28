"""Generic evolution: heredity, lifecycle, fitness, selection and gated promotion.

* ``genotype``, ``mutation``, ``lineage`` — canonical integer genotypes,
  bounded deterministic mutation seeded from content-addressed lineage;
* ``organism``, ``lifecycle``, ``reproduction`` — persistent organism state,
  the lifecycle law and atomic, energy-conserving asexual reproduction;
* ``fitness`` — exact integer scalarization of measured outcomes;
* ``selection`` — deterministic, self-verified truncation selection;
* ``path_gate`` — one bounded attempt per admitted, ECS-history-bound path
  assertion, recorded as a chained transition receipt;
* ``promotion`` — evaluation-gated promotion of a candidate workspace over its
  incumbent and atomic materialization of the selected child.

Nothing here randomizes implicitly, loads a model or imports numerics.
Measured outcomes (fitness observations, evaluation evidence) are supplied by
the caller; no environment producing them is part of this package.
"""

from .fitness import FitnessObservation, FitnessPolicyV1, OrganismFitnessRecord
from .genotype import Genotype, IntegerGene
from .lifecycle import LifecycleState, can_transition, transition_lifecycle
from .lineage import LineageIdentity, ParentLineageRef, derive_mutation_seed
from .mutation import MutationEvent, MutationPolicyV1, MutationResult, mutate_genotype
from .organism import OrganismState, ResourceQuantity
from .path_gate import (
    GENESIS_DIGEST,
    EvolutionAttempt,
    EvolutionGateError,
    EvolutionPathAssertion,
    EvolutionPathGate,
    GateExecuted,
    GateRejected,
    PathTransitionReceipt,
)
from .promotion import (
    CandidateRecord,
    EvaluationEvidence,
    HarnessManifest,
    SelectionReceipt,
    atomic_materialize,
    build_manifest_from_tree,
    content_map_digest,
    digest_tree,
    eligibility,
    select,
)
from .reproduction import (
    BirthRequest,
    ReproductionPolicyV1,
    ReproductionRejectionCode,
    ReproductionResult,
    execute_reproduction,
)
from .selection import (
    PopulationState,
    SelectionCommitResult,
    SelectionRejectionCode,
    TruncationSelectionPolicyV1,
    commit_selection,
)

__all__ = [
    "BirthRequest", "CandidateRecord", "EvaluationEvidence", "EvolutionAttempt",
    "EvolutionGateError", "EvolutionPathAssertion", "EvolutionPathGate", "FitnessObservation",
    "FitnessPolicyV1", "GENESIS_DIGEST", "GateExecuted", "GateRejected", "Genotype",
    "HarnessManifest", "IntegerGene", "LifecycleState", "LineageIdentity", "MutationEvent",
    "MutationPolicyV1", "MutationResult", "OrganismFitnessRecord", "OrganismState",
    "ParentLineageRef", "PathTransitionReceipt", "PopulationState", "ReproductionPolicyV1",
    "ReproductionRejectionCode", "ReproductionResult", "ResourceQuantity", "SelectionCommitResult",
    "SelectionReceipt", "SelectionRejectionCode", "TruncationSelectionPolicyV1",
    "atomic_materialize", "build_manifest_from_tree", "can_transition", "commit_selection",
    "content_map_digest", "derive_mutation_seed", "digest_tree", "eligibility",
    "execute_reproduction", "mutate_genotype", "select", "transition_lifecycle",
]
