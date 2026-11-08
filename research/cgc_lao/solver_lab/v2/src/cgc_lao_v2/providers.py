"""Research-only observation-provider contract (Milestone B).

A provider serves readings of the expert the host selected. A reading carries only an opaque event id, a unique
reading id, the id of the measurement channel that produced it and its binary value: never a corruption label or
any simulator variable. Each provider declares the statistical relation between the readings it returns for one
acquisition:

    SINGLE              one reading per event
    INDEPENDENT_REREAD  several readings of the same event whose corruptions are independent (rho = 0)
    CORRELATED          several readings of the same event with corruption correlation at most a certified bound
    SHARED_SOURCE       several readings of the same event that share one corruption source
    COPY                the same reading presented more than once
    SECOND_EVENT        each reading belongs to its own independent event of the selected expert

The relations are not statistically equivalent, and no observable can tell some of them apart (agreeing
readings fit both a clean channel and a shared corruption). So a declaration is backed by a mechanism digest
that the HOST must find in its own registry of audited measurement mechanisms. In this laboratory the
registry is built from the SLB2 mechanism specifications, and the test suite audits each one statistically
with privileged access to the simulator. That qualifies the interface mechanics; it cannot certify an
external data source.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

RELATIONS = ("SINGLE", "INDEPENDENT_REREAD", "CORRELATED", "SHARED_SOURCE", "COPY", "SECOND_EVENT")


@dataclass(frozen=True)
class Reading:
    event_id: str
    reading_id: str
    channel_id: str
    expert: int
    value: int


@dataclass(frozen=True)
class ProviderDeclaration:
    provider_id: str
    relation: str
    max_readings: int
    mechanism_sha256: str
    correlation_bound: float | None = None


@dataclass(frozen=True)
class AuditedMechanism:
    name: str
    relation: str
    correlation_bound: float     # audited upper bound on reading-level corruption correlation
    audit: str                   # where the audit lives (test name)


def mechanism_digest(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# SLB2 measurement mechanisms (specs/slb2.spec.json) as auditable specifications.
SLB2_MECHANISMS = {
    "slb2/independent-reread": ({"bench": "SLB2", "readings": "symmetric(q) with independent flips per reading"},
                                AuditedMechanism("slb2/independent-reread", "INDEPENDENT_REREAD", 0.0,
                                                 "test_v2_host.TestProviderAudit.test_independent_reread")),
    "slb2/correlated-0.1": ({"bench": "SLB2", "readings": "correlated(q, rho=0.1)"},
                            AuditedMechanism("slb2/correlated-0.1", "CORRELATED", 0.1,
                                             "test_v2_host.TestProviderAudit.test_correlated")),
    "slb2/correlated-0.5": ({"bench": "SLB2", "readings": "correlated(q, rho=0.5)"},
                            AuditedMechanism("slb2/correlated-0.5", "CORRELATED", 0.5,
                                             "test_v2_host.TestProviderAudit.test_correlated")),
    "slb2/shared-source": ({"bench": "SLB2", "readings": "event(q): readings share one flip"},
                           AuditedMechanism("slb2/shared-source", "SHARED_SOURCE", 1.0,
                                            "test_v2_host.TestProviderAudit.test_shared_source")),
    "slb2/second-event": ({"bench": "SLB2", "readings": "second_event(q): independent events"},
                          AuditedMechanism("slb2/second-event", "SECOND_EVENT", 0.0,
                                           "test_v2_host.TestProviderAudit.test_second_event")),
}


class MechanismRegistry:
    """Host-owned: mechanism digest -> audited properties. Only the host populates it."""

    def __init__(self, entries: dict[str, AuditedMechanism]):
        self._entries = dict(entries)

    @classmethod
    def slb2(cls) -> "MechanismRegistry":
        return cls({mechanism_digest(spec): audited for spec, audited in SLB2_MECHANISMS.values()})

    def lookup(self, digest: str) -> AuditedMechanism | None:
        return self._entries.get(digest)


def slb2_digest(name: str) -> str:
    return mechanism_digest(SLB2_MECHANISMS[name][0])


class SimulatedProvider:
    """Serves SLB2 readings of the selected expert for one episode (the world side; it may read the batch).

    ``behaviour`` decides what is actually served: 'first' (one reading), 'reread' (the generated readings of the
    selected event, distinct channels), 'copy' (the first reading twice under one reading id), 'second_event'
    (the generated second-event reading). The declaration is whatever the provider claims, honest or not.
    """

    def __init__(self, batch, e: int, declaration: ProviderDeclaration, behaviour: str, reads_per_step,
                 tamper: str | None = None):
        self.batch, self.e, self.declaration, self.behaviour = batch, e, declaration, behaviour
        self.reads_per_step = tuple(reads_per_step)
        self.tamper = tamper

    def acquire(self, t: int, expert: int) -> list[Reading]:
        y = self.batch.y[self.e, t, expert]
        event = f"{self.declaration.provider_id}/e{self.e}/t{t}/a{expert}"
        n = self.reads_per_step[t]
        if self.behaviour == "first" or n == 1:
            out = [Reading(event, f"{event}/r0", "chan-A", expert, int(y[0]))]
        elif self.behaviour == "reread":
            out = [Reading(event, f"{event}/r{r}", f"chan-{'AB'[r]}", expert, int(y[r])) for r in range(n)]
        elif self.behaviour == "copy":
            out = [Reading(event, f"{event}/r0", "chan-A", expert, int(y[0]))] * n
        elif self.behaviour == "second_event":
            out = [Reading(event, f"{event}/r0", "chan-A", expert, int(y[0])),
                   Reading(f"{event}/second", f"{event}/second/r0", "chan-A", expert, int(y[1]))]
        else:
            raise ValueError("UNKNOWN_BEHAVIOUR")
        if self.tamper == "replay_reading_id" and len(out) == 2:
            out = [out[0], Reading(out[1].event_id, out[0].reading_id, out[1].channel_id, expert, out[1].value)]
        elif self.tamper == "same_channel" and len(out) == 2:
            out = [out[0], Reading(out[1].event_id, out[1].reading_id, out[0].channel_id, expert, out[1].value)]
        elif self.tamper == "foreign_event" and len(out) == 2:
            out = [out[0], Reading(out[0].event_id + "/x", out[1].reading_id, out[1].channel_id, expert, out[1].value)]
        elif self.tamper == "wrong_expert":
            out = [Reading(r.event_id, r.reading_id, r.channel_id, (expert + 1) % 3, r.value) for r in out]
        elif self.tamper == "bad_value":
            out = [Reading(out[0].event_id, out[0].reading_id, out[0].channel_id, expert, 2)] + out[1:]
        elif self.tamper == "stale_event" and t > 0:
            prev = f"{self.declaration.provider_id}/e{self.e}/t0/a"
            out = [Reading(prev + str(expert), r.reading_id, r.channel_id, expert, r.value) for r in out]
        return out
