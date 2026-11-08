"""Fail-closed host admission for the frozen SLB2 v4 CGC-LAO research learner.

DO NOT modify or silently substitute a file pinned by FROZEN_V4.json. This is an
additive, post-freeze integration-boundary correction. Historical v4 experiments
continue to replay against their exact frozen source. New host integrations
must use StrictResearchHost, not the historical cgc_lao_v2.host.ResearchHost.

Only the synthetic 8-step DUP2_4 / FIRST_ONLY contract is implemented here;
no real-world provider is certified by this package.
"""
from __future__ import annotations

import math
from itertools import islice
from numbers import Integral, Real

from cgc_lao_v2.host import (
    Admission, HostError, ResearchHost, _Episode, admit as historical_admit,
)
from cgc_lao_v2.providers import ProviderDeclaration, RELATIONS

QUALIFIED_READ_SCHEDULE = (2, 2, 2, 2, 1, 1, 1, 1)
MAX_QUALIFIED_RHO = 0.25  # frozen v4 OOD grid threshold, NOT a general correlation guarantee


def _fallback(reason: str) -> Admission:
    return Admission("II", "FALLBACK", "FIRST_ONLY", (reason,))


def _unit_float(value: object) -> bool:
    return (isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0)


def strict_admit(declaration: object, registry, rho_admit: float = 0.0) -> Admission:
    """Return Contract I only for a FULL qualified read schedule and finite audited metadata.

    Invalid or unverifiable declarations fall back to Contract II. This function
    does not authenticate the provider; its digest is meaningful only when the
    host independently owns and audits the registry.
    """
    if not _unit_float(rho_admit) or float(rho_admit) > MAX_QUALIFIED_RHO:
        return _fallback("RHO_ADMISSION_OUT_OF_QUALIFIED_RANGE")
    if not isinstance(declaration, ProviderDeclaration):
        return _fallback("MALFORMED_DECLARATION")
    if (not isinstance(declaration.max_readings, Integral)
            or isinstance(declaration.max_readings, bool)
            or not 1 <= declaration.max_readings <= 2):
        return _fallback("MALFORMED_READING_CAPACITY")
    if declaration.relation not in RELATIONS:
        return _fallback("UNKNOWN_PROVIDER_RELATION")
    if declaration.correlation_bound is not None and not _unit_float(declaration.correlation_bound):
        return _fallback("INVALID_DECLARED_CORRELATION")
    if declaration.relation in ("INDEPENDENT_REREAD", "CORRELATED"):
        if declaration.max_readings != 2:
            return _fallback("REQUIRED_DUPLICATE_CAPABILITY_MISSING")
        if not isinstance(declaration.mechanism_sha256, str) or len(declaration.mechanism_sha256) != 64:
            return _fallback("INVALID_MECHANISM_DIGEST")
        if declaration.relation == "INDEPENDENT_REREAD":
            if declaration.correlation_bound not in (None, 0, 0.0):
                return _fallback("CONTRADICTORY_INDEPENDENCE_DECLARATION")
        elif declaration.correlation_bound is None:
            return _fallback("MISSING_CORRELATION_BOUND")
        audited = registry.lookup(declaration.mechanism_sha256)
        if audited is None:
            return _fallback("UNREGISTERED_MECHANISM")
        if not _unit_float(audited.correlation_bound):
            return _fallback("INVALID_AUDITED_CORRELATION")
    # Preserve historical predicate as a necessary (but no longer sufficient) check.
    return historical_admit(declaration, registry, float(rho_admit))


class StrictResearchHost(ResearchHost):
    """Post-freeze boundary enforcing exactly DUP2_4 or fail-closed FIRST_ONLY.

    Historical ResearchHost remains unmodified solely so the scientific records
    can be reproduced bit-for-bit. Use THIS host in any new research integration.
    """

    def __init__(self, frozen: dict, registry, rho_admit: float = 0.0,
                 backend: str = "native"):
        if (not _unit_float(rho_admit) or float(rho_admit) > MAX_QUALIFIED_RHO):
            raise ValueError("RHO_ADMISSION_OUT_OF_QUALIFIED_RANGE")
        if (not isinstance(frozen, dict)
                or frozen.get("horizon") != len(QUALIFIED_READ_SCHEDULE)
                or tuple(frozen.get("reads", ())) != QUALIFIED_READ_SCHEDULE
                or not all(k in frozen for k in ("PRIMARY", "FALLBACK"))):
            raise ValueError("UNQUALIFIED_READING_SCHEDULE_OR_CONFIGURATION")
        if backend not in ("scalar", "native"):
            raise ValueError("UNKNOWN_BACKEND")
        super().__init__(frozen, registry, float(rho_admit), backend)

    def initialize(self, declaration: ProviderDeclaration) -> Admission:
        if self.admission is not None:
            raise HostError("ALREADY_INITIALIZED")
        if not isinstance(declaration, ProviderDeclaration):
            raise HostError("MALFORMED_DECLARATION")
        if (not isinstance(declaration.max_readings, Integral)
                or isinstance(declaration.max_readings, bool)
                or not 1 <= declaration.max_readings <= 2):
            raise HostError("MALFORMED_READING_CAPACITY")
        decision = strict_admit(declaration, self.registry, self.rho_admit)
        cfg = self.frozen[decision.configuration]
        # Instantiate BEFORE committing admission state: failures cannot leave
        # a half-initialized host believing the provider was admitted.
        if self.backend == "native":
            from cgc_lao_v2.native import NativeRMF
            learner = NativeRMF(cfg["hypotheses"], cfg["prior"], horizon=self.frozen["horizon"])
        else:
            from cgc_lao_v2.scalar2 import ScalarRMF2
            learner = ScalarRMF2(cfg["hypotheses"], cfg["prior"], horizon=self.frozen["horizon"])
        self.declaration, self.admission, self.learner = declaration, decision, learner
        return decision

    def _zeroize_strict(self) -> None:
        learner = self.learner
        if learner is None:
            return
        if hasattr(learner, "zeroize"):
            learner.zeroize()
            return
        # The historical scalar reference lacks zeroize(). Wipe its bounded
        # episode state explicitly without modifying its frozen coefficients.
        from cgc_lao_v2.scalar2 import ScalarRMF2
        if not isinstance(learner, ScalarRMF2):
            raise HostError("UNRECOGNIZED_LEARNER_CANNOT_ZEROIZE")
        learner.lw[:] = [0.0] * len(learner.lw)
        for rows in (learner.mu, learner.v):
            for row in rows:
                row[:] = [0.0] * len(row)
        learner.counts[:] = [0] * len(learner.counts)
        learner.t = 0

    def _fail(self, reason: str):
        self.ep.status, self.ep.failure = "FAILED", reason
        try:
            self._zeroize_strict()
        except Exception as exc:
            raise HostError(reason + ":ZEROIZE_FAILED") from exc
        raise HostError(reason)

    def observe_selected(self, selected_expert: int, readings) -> None:
        # Bounded intake: never materialize an unbounded iterator of readings.
        try:
            bounded = list(islice(iter(readings), 3))
        except (TypeError, ValueError) as exc:
            self._fail("INVALID_READING_CONTAINER")
            raise AssertionError("unreachable") from exc
        if (self.admission is not None and self.admission.mode == "DUPLICATE"
                and self.ep.status == "ACTIVE" and self.ep.proposed is not None
                and 0 <= self.ep.t < len(QUALIFIED_READ_SCHEDULE)):
            expected = QUALIFIED_READ_SCHEDULE[self.ep.t]
            self._require(len(bounded) == expected, "QUALIFIED_READING_COUNT_MISMATCH")
        return super().observe_selected(selected_expert, bounded)

    def close_episode(self) -> dict:
        summary = {"status": self.ep.status, "steps": self.ep.t,
                   "failure": self.ep.failure,
                   "contract": self.admission.contract if self.admission else None}
        self._zeroize_strict()
        self.ep = _Episode()
        return summary
