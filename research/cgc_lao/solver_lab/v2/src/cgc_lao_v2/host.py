"""Research-only host for the bounded learner (Milestone E). RESEARCH_ONLY, NO_RUNTIME_AUTHORITY.

The host owns provider admission, independence verification, input and resource validation, the session
lifecycle, permission boundaries and the final acceptance of every proposal. The learner (the native C11
component, or the explicit-loop scalar reference) owns only its bounded posterior and episode-local state, and
its proposals. Operations:

    initialize(declaration)            -> admission decision (Contract I or II, configuration, provider mode)
    reset_episode(router_w)            -> fresh learner state for one episode
    propose_expert(public_context)     -> the learner's proposal; the host records what it accepts
    observe_selected(expert, readings) -> verified, routed readings reach the learner; anything else fails closed
    propose_terminal_decision(queries) -> one expert or STOP per public query
    close_episode()                    -> zeroises the learner state

Failure of any capability, integrity or lifecycle check fails closed: the learner state is zeroised, the episode
is marked FAILED and refuses further operations until the next reset_episode. Nothing is persisted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cgc_lao_lab.env import logits

from .providers import RELATIONS, MechanismRegistry, ProviderDeclaration, Reading

K, D, Q_MAX, STOP = 3, 4, 64, 3


class HostError(RuntimeError):
    pass


@dataclass(frozen=True)
class Admission:
    contract: str        # "I" (independent duplicates admitted) or "II" (fallback)
    configuration: str   # "PRIMARY" or "FALLBACK"
    mode: str            # provider mode the learner sees: DUPLICATE or FIRST_ONLY
    reasons: tuple


@dataclass
class _Episode:
    status: str = "IDLE"            # IDLE, ACTIVE, TERMINAL, FAILED
    t: int = 0
    proposed: int | None = None
    events: set = field(default_factory=set)
    reading_ids: set = field(default_factory=set)
    failure: str | None = None


def admit(decl: ProviderDeclaration, registry: MechanismRegistry, rho_admit: float = 0.0) -> Admission:
    """Contract I only for audited independent re-reads (or audited correlation within rho_admit)."""
    if not isinstance(decl, ProviderDeclaration) or decl.relation not in RELATIONS or not (1 <= decl.max_readings <= 2):
        return Admission("II", "FALLBACK", "FIRST_ONLY", ("MALFORMED_DECLARATION",))
    audited = registry.lookup(decl.mechanism_sha256)
    if decl.relation == "INDEPENDENT_REREAD":
        if audited is None:
            return Admission("II", "FALLBACK", "FIRST_ONLY", ("UNREGISTERED_MECHANISM",))
        if audited.relation != "INDEPENDENT_REREAD" or audited.correlation_bound != 0.0:
            return Admission("II", "FALLBACK", "FIRST_ONLY", ("DECLARATION_CONTRADICTS_AUDIT",))
        return Admission("I", "PRIMARY", "DUPLICATE", ("AUDITED_INDEPENDENT_REREAD",))
    if decl.relation == "CORRELATED":
        if audited is None or audited.relation != "CORRELATED" or decl.correlation_bound is None:
            return Admission("II", "FALLBACK", "FIRST_ONLY", ("UNVERIFIED_CORRELATION",))
        bound = max(audited.correlation_bound, decl.correlation_bound)
        if bound <= rho_admit:
            return Admission("I", "PRIMARY", "DUPLICATE", (f"AUDITED_CORRELATION<={bound}<=RHO_ADMIT",))
        return Admission("II", "FALLBACK", "FIRST_ONLY", (f"CORRELATION_BOUND_{bound}>RHO_ADMIT_{rho_admit}",))
    reason = {"SINGLE": "NO_REREAD", "SHARED_SOURCE": "SHARED_CORRUPTION", "COPY": "COPIES_CARRY_NO_RELIABILITY_EVIDENCE",
              "SECOND_EVENT": "SECOND_EVENTS_NOT_QUALIFIED_AS_DUPLICATES"}[decl.relation]
    return Admission("II", "FALLBACK", "FIRST_ONLY", (reason,))


class ResearchHost:
    def __init__(self, frozen: dict, registry: MechanismRegistry, rho_admit: float = 0.0, backend: str = "native"):
        """``frozen``: {"PRIMARY": {"hypotheses", "prior"}, "FALLBACK": {...}, "horizon": 8, "reads": [...]}."""
        self.frozen = frozen
        self.registry = registry
        self.rho_admit = float(rho_admit)
        self.backend = backend
        self.admission: Admission | None = None
        self.learner = None
        self.declaration = None
        self.ep = _Episode()

    # -- session ------------------------------------------------------------------------------------------
    def initialize(self, declaration: ProviderDeclaration) -> Admission:
        if self.admission is not None:
            raise HostError("ALREADY_INITIALIZED")
        self.declaration = declaration
        self.admission = admit(declaration, self.registry, self.rho_admit)
        cfg = self.frozen[self.admission.configuration]
        if self.backend == "native":
            from .native import NativeRMF
            self.learner = NativeRMF(cfg["hypotheses"], cfg["prior"], horizon=self.frozen["horizon"])
        else:
            from .scalar2 import ScalarRMF2
            self.learner = ScalarRMF2(cfg["hypotheses"], cfg["prior"], horizon=self.frozen["horizon"])
        return self.admission

    def _fail(self, reason: str):
        self.ep.status, self.ep.failure = "FAILED", reason
        if self.learner is not None and hasattr(self.learner, "zeroize"):
            self.learner.zeroize()
        raise HostError(reason)

    def _require(self, condition: bool, reason: str):
        if not condition:
            self._fail(reason)

    def reset_episode(self, router_w) -> None:
        if self.admission is None:
            raise HostError("NOT_INITIALIZED")
        w = np.asarray(router_w, dtype=np.float64)
        if w.shape != (K, D) or not np.isfinite(w).all():
            raise HostError("ROUTER_CONTRACT")
        self.router_w = w.copy()
        self.router_w.setflags(write=False)
        self.learner.reset()
        self.ep = _Episode(status="ACTIVE")

    # -- acquisition ----------------------------------------------------------------------------------------
    def propose_expert(self, public_context) -> int:
        self._require(self.ep.status == "ACTIVE", f"PROPOSE_IN_{self.ep.status}")
        self._require(self.ep.proposed is None, "OBSERVATION_MISSING")
        self._require(self.ep.t < self.frozen["horizon"], "HORIZON_EXHAUSTED")
        x = np.asarray(public_context, dtype=np.float64)
        self._require(x.shape == (D,) and bool(np.isfinite(x).all()), "CONTEXT_CONTRACT")
        self.z = logits(self.router_w, x)
        expert = int(self.learner.select([float(v) for v in self.z]))
        self._require(0 <= expert < K, "LEARNER_PROPOSAL_INVALID")
        self.ep.proposed = expert
        return expert

    def observe_selected(self, selected_expert: int, readings) -> None:
        self._require(self.ep.status == "ACTIVE", f"OBSERVE_IN_{self.ep.status}")
        self._require(self.ep.proposed is not None, "SELECT_REQUIRED")
        self._require(selected_expert == self.ep.proposed, "SELECTION_NOT_ACCEPTED")
        readings = list(readings)
        n_allowed = self.frozen["reads"][self.ep.t]
        self._require(1 <= len(readings) <= min(n_allowed, self.declaration.max_readings), "READING_COUNT")
        if self.declaration.relation == "COPY":
            # a declared copy presents one reading several times: the copies must be identical, and count once
            self._require(all(r == readings[0] for r in readings), "COPY_MISMATCH")
            readings = readings[:1]
        for r in readings:
            self._require(isinstance(r, Reading), "READING_TYPE")
            self._require(r.expert == selected_expert, "READING_FOR_UNSELECTED_EXPERT")
            self._require(r.value in (0, 1) and not isinstance(r.value, bool), "READING_VALUE")
            self._require(r.reading_id not in self.ep.reading_ids, "READING_REPLAYED")
            self.ep.reading_ids.add(r.reading_id)
        self._require(readings[0].event_id not in self.ep.events, "STALE_EVENT")
        mode = self.admission.mode
        z = float(self.z[selected_expert])
        if mode == "DUPLICATE" and len(readings) == 2:
            self._require(readings[1].event_id == readings[0].event_id, "REREAD_OF_ANOTHER_EVENT")
            self._require(readings[1].channel_id != readings[0].channel_id, "REREAD_ON_SAME_CHANNEL")
            values = [readings[0].value, readings[1].value]
        else:
            values = [readings[0].value]          # Contract II: only the first reading reaches the learner
        self.ep.events.add(readings[0].event_id)
        self.learner.update(selected_expert, z, values)
        self.ep.proposed = None
        self.ep.t += 1

    # -- terminal ---------------------------------------------------------------------------------------------
    def propose_terminal_decision(self, public_queries) -> list:
        self._require(self.ep.status == "ACTIVE", f"TERMINAL_IN_{self.ep.status}")
        self._require(self.ep.t == self.frozen["horizon"] and self.ep.proposed is None, "TERMINAL_PREMATURE")
        q = np.asarray(public_queries, dtype=np.float64)
        self._require(q.ndim == 2 and q.shape[1] == D and 1 <= q.shape[0] <= Q_MAX and bool(np.isfinite(q).all()),
                      "QUERY_CONTRACT")
        out = []
        for row in logits(self.router_w, q):
            d = int(self.learner.decide([float(v) for v in row]))
            self._require(0 <= d <= STOP, "LEARNER_DECISION_INVALID")
            out.append(d)
        self.ep.status = "TERMINAL"
        return out

    def close_episode(self) -> dict:
        summary = {"status": self.ep.status, "steps": self.ep.t, "failure": self.ep.failure,
                   "contract": self.admission.contract if self.admission else None}
        if self.learner is not None and hasattr(self.learner, "zeroize"):
            self.learner.zeroize()
        self.ep = _Episode()
        return summary


def run_episode(host: ResearchHost, batch, e: int, provider) -> list:
    """Drive one SLB2 episode through the host with an instrumented provider; returns terminal decisions."""
    host.reset_episode(batch.router_w)
    for t in range(host.frozen["horizon"]):
        a = host.propose_expert(batch.x[e, t])
        host.observe_selected(a, provider.acquire(t, a))
    decisions = host.propose_terminal_decision(batch.xq[e])
    host.close_episode()
    return decisions


def finite(x) -> bool:
    return all(math.isfinite(float(v)) for v in np.ravel(x))
