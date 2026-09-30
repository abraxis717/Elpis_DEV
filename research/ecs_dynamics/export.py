"""Inert, typed observation of a laboratory result. RESEARCH_ONLY. NO_RUNTIME_AUTHORITY.

:class:`DynamicsObservation` is plain frozen data: identities, dispositions and
finding statements. It holds no reference to a system, a kernel or a port, it
has no method that acts on anything, and no production module imports it.
It is the only shape in which a laboratory result is meant to leave this
directory, for example as a log record.
"""
from __future__ import annotations

from dataclasses import dataclass

from .results import RESEARCH_FLAGS, ExperimentResult
from .spec import canonical_json


@dataclass(frozen=True)
class DynamicsObservation:
    experiment: str
    version: int
    split: str
    result_digest: str
    frozen_digest: str | None
    mechanics: str
    scientific: str
    statements: tuple
    research_only: bool = True
    no_runtime_authority: bool = True
    no_production_neural_claim: bool = True

    @classmethod
    def from_result(cls, result: ExperimentResult) -> "DynamicsObservation":
        return cls(result.experiment, result.version, result.split, result.digest, result.frozen_digest,
                   result.mechanics, result.scientific, tuple(f.statement() for f in result.findings),
                   RESEARCH_FLAGS["RESEARCH_ONLY"], RESEARCH_FLAGS["NO_RUNTIME_AUTHORITY"],
                   RESEARCH_FLAGS["NO_PRODUCTION_NEURAL_CLAIM"])

    def to_json(self) -> bytes:
        return canonical_json({k: getattr(self, k) for k in self.__dataclass_fields__})
