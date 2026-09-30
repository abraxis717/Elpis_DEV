"""Result schema: evidence categories, target-relative sufficiency, dispositions. RESEARCH_ONLY.

The schema keeps apart:

* evidence categories: MECHANICS, ALGEBRAIC_IDENTITY, DESCRIPTIVE_RESULT and
  INTERVENTION_SUPPORTED_RESULT;
* mechanics disposition (does the harness run correctly) from scientific
  disposition (what the frozen experiment found);
* sufficiency outcomes, which always name the representation, the target and
  the regime. There is no field or value meaning "sufficient" in general;
* sources: every paper-derived statement cites ``arXiv:<id> <section/eq>``;
  laboratory constructions are marked ``LAB:``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .observables import TARGETS
from .spec import canonical_json, research_digest

RESULT_SCHEMA = "elpis.research.ecs-dynamics.result.v1"
RESULT_DOMAIN = "elpis.research.ecs-dynamics.result-digest.v1"

EVIDENCE = ("MECHANICS", "ALGEBRAIC_IDENTITY", "DESCRIPTIVE_RESULT", "INTERVENTION_SUPPORTED_RESULT")
MECHANICS = ("MECHANICS_PASS", "MECHANICS_FAIL")
SCIENTIFIC = ("SUPPORTS_UNDER_FROZEN_SYNTHETIC_REGIME", "DID_NOT_SUPPORT", "DESCRIPTIVE_ONLY", "NOT_TESTED")
SUFFICIENCY_OUTCOMES = (
    "COUNTEREXAMPLE_FOUND",               # evidence against sufficiency for this target and regime
    "NO_COUNTEREXAMPLE_UNDER_REGIME",     # not a proof: only what was tried
    "NOT_TESTED",
)

PAPERS = ("2609.19288", "2609.29834", "2609.19424", "2609.07341")

RESEARCH_FLAGS = {"RESEARCH_ONLY": True, "NO_RUNTIME_AUTHORITY": True, "NO_PRODUCTION_NEURAL_CLAIM": True}


class ResultError(ValueError):
    pass


@dataclass(frozen=True)
class SufficiencyFinding:
    """``representation`` was tested for sufficiency with respect to ``target`` under ``regime``."""
    representation: str
    target: str
    regime: str
    outcome: str
    evidence: str
    statistic: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.target not in TARGETS:
            raise ResultError(f"unknown target {self.target!r}")
        if self.outcome not in SUFFICIENCY_OUTCOMES:
            raise ResultError(f"unknown sufficiency outcome {self.outcome!r}")
        if self.evidence not in EVIDENCE:
            raise ResultError(f"unknown evidence category {self.evidence!r}")
        if not self.representation or not self.regime:
            raise ResultError("representation and regime must be named")

    def statement(self) -> str:
        verb = {"COUNTEREXAMPLE_FOUND": "a counterexample was found",
                "NO_COUNTEREXAMPLE_UNDER_REGIME": "no counterexample was found",
                "NOT_TESTED": "it was not tested"}[self.outcome]
        return (f"{self.representation} was tested for sufficiency with respect to target {self.target} "
                f"under regime {self.regime}: {verb}.")


@dataclass(frozen=True)
class ExperimentResult:
    experiment: str
    version: int
    split: str
    spec_digest: str
    frozen_digest: str | None
    source_digest: str
    source_commit: str
    model_family: str
    numerical_profile: dict
    world_ids: tuple
    coarse_representation: str
    target: str
    intervention_policy: str
    evidence: str
    metrics: dict
    findings: tuple = ()
    pass_rule: dict = field(default_factory=dict)
    mechanics: str = "MECHANICS_PASS"
    scientific: str = "NOT_TESTED"
    limitations: tuple = ()
    sources: tuple = ()
    schema: str = RESULT_SCHEMA

    def __post_init__(self):
        if self.split not in ("DEV", "QUAL", "FIXTURE"):
            raise ResultError("split")
        if self.target not in TARGETS:
            raise ResultError("target")
        if self.evidence not in EVIDENCE:
            raise ResultError("evidence")
        if self.mechanics not in MECHANICS or self.scientific not in SCIENTIFIC:
            raise ResultError("disposition")
        if self.split == "QUAL" and self.frozen_digest is None:
            raise ResultError("a QUAL result must bind a frozen specification")
        if self.mechanics == "MECHANICS_FAIL" and self.scientific != "NOT_TESTED":
            raise ResultError("a mechanics failure has no scientific disposition")
        if not all(type(f) is SufficiencyFinding for f in self.findings):
            raise ResultError("findings must be SufficiencyFinding values")
        for source in self.sources:
            if not (source.startswith("LAB:") or any(source.startswith(f"arXiv:{p}") for p in PAPERS)):
                raise ResultError(f"a source must cite one of the four papers or the laboratory: {source!r}")

    def as_dict(self) -> dict:
        data = asdict(self)
        data["findings"] = [dict(asdict(f), statement=f.statement()) for f in self.findings]
        data["flags"] = dict(RESEARCH_FLAGS)
        return data

    def to_json(self) -> bytes:
        return canonical_json(self.as_dict())

    @property
    def digest(self) -> str:
        return research_digest(RESULT_DOMAIN, self.as_dict())
