"""Layers A + D: the research host of the relational learner over the real ECS_C relational view.

The host owns admission (host_closure.strict_admit: Contract I only for an audited two-reading provider delivering
exactly (2,2,2,2,1,1,1,1); everything else Contract II, first reading only), the canonical slot table (relation keys
in ascending order, capacity 8, refused beyond it), the admissible actions (relations whose reporting source is LIVE
in the view), reading integrity, authority syncs and invalidation, and the terminal decisions. The learner owns only
its bounded posterior and its proposals.

Evidence flows one way: every reading is ADMITTED into the view under its source's binding before the learner sees
it, and a refused admission fails the episode closed. The learner's projection (u16 saliences) is exported read-only;
nothing reads back into the view. Binding status of buffered events is the VIEW's: LIVE or SUSPENDED as
elpis_ecs_c_rel_support_of reports it, RETIRED when the view released it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from cgc_lao_v2.host import HostError
from cgc_lao_v2.providers import ProviderDeclaration
from host_closure.strict_host import QUALIFIED_READ_SCHEDULE, strict_admit

from . import ecsc
from .scalar import R_MAX, ScalarRelational

FROZEN_DEFAULT = {
    "PRIMARY": {"hypotheses": ((0.0, 0.0), (0.4, 0.0), (0.48, 0.0)), "prior": (1.0, 1.0, 1.0), "readout": "coupled"},
    "FALLBACK": {"hypotheses": ((0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.48, 0.0)), "prior": (1.0, 1.0, 8.0, 8.0),
                 "readout": "coupled"},
    "horizon": 8, "reads": QUALIFIED_READ_SCHEDULE, "invalidation": "replay",
}


def config_from_freeze(frozen_r6: dict) -> dict:
    """The host configuration bound by FROZEN_R6.json: the primary (hypotheses, prior, readout, invalidation) and
    the Contract II fallback (hypotheses, prior; coupled readout of the first reading only)."""
    hyp, prior, readout, invalidation = frozen_r6["primary"]["config"]
    fhyp, fprior = frozen_r6["fallback"]["config"]
    return {"PRIMARY": {"hypotheses": tuple(map(tuple, hyp)), "prior": tuple(prior), "readout": readout},
            "FALLBACK": {"hypotheses": tuple(map(tuple, fhyp)), "prior": tuple(fprior), "readout": "coupled"},
            "horizon": 8, "reads": QUALIFIED_READ_SCHEDULE, "invalidation": invalidation}


@dataclass
class _Episode:
    status: str = "IDLE"
    t: int = 0
    proposed: int | None = None
    failure: str | None = None
    events: set = field(default_factory=set)
    readings: set = field(default_factory=set)


class RelationalHost:
    def __init__(self, registry, rho_admit: float = 0.0, frozen: dict | None = None):
        self.frozen = dict(FROZEN_DEFAULT if frozen is None else frozen)
        if tuple(self.frozen["reads"]) != QUALIFIED_READ_SCHEDULE or self.frozen["horizon"] != 8:
            raise ValueError("UNQUALIFIED_READING_SCHEDULE")
        if self.frozen["invalidation"] not in ("replay", "slot"):
            raise ValueError("UNKNOWN_INVALIDATION")
        self.registry, self.rho_admit = registry, float(rho_admit)
        self.admission = self.learner = self.view = None
        self.ep = _Episode()

    # -- admission ------------------------------------------------------------------------------------------------
    def initialize(self, declaration) -> object:
        if self.admission is not None:
            raise HostError("ALREADY_INITIALIZED")
        if not isinstance(declaration, ProviderDeclaration):
            raise HostError("MALFORMED_DECLARATION")
        self.admission = strict_admit(declaration, self.registry, self.rho_admit)
        return self.admission

    def _fail(self, reason: str):
        self.ep.status, self.ep.failure = "FAILED", reason
        if self.learner is not None:
            self.learner.zeroize()
        raise HostError(reason)

    def _require(self, ok: bool, reason: str) -> None:
        if not ok:
            self._fail(reason)

    # -- episode --------------------------------------------------------------------------------------------------
    def reset_episode(self, world: ecsc.AuthorityWorld, relations, W, reporters) -> None:
        """relations: [(entity_from, entity_to)] (32-byte ids); W: [R][4] public weights; reporters(t, rel) -> the
        authority row reporting an event of relation `rel` at step t (the host reads its binding from the table)."""
        if self.admission is None:
            raise HostError("NOT_INITIALIZED")
        if not (1 <= len(relations) <= R_MAX):
            raise HostError("RELATION_CAPACITY")
        keys = [ecsc.relation_key(a, b) for a, b in relations]
        if len(set(keys)) != len(keys):
            raise HostError("DUPLICATE_RELATION")
        order = sorted(range(len(keys)), key=lambda r: keys[r])
        self.slot_of = {r: s for s, r in enumerate(order)}           # template relation -> canonical slot
        self.rel_of = {s: r for r, s in self.slot_of.items()}
        self.keys = [keys[self.rel_of[s]] for s in range(len(keys))]
        self.relations, self.W, self.reporters, self.world = relations, W, reporters, world
        if self.view is None:
            self.view = ecsc.RelationalView()
        else:
            self.view.reset()
        code, _ = self.view.sync(world)
        if code != ecsc.OK:
            raise HostError(f"SYNC_{code}")
        cfg = self.frozen[self.admission.configuration]
        self.learner = ScalarRelational(cfg["hypotheses"], cfg["prior"], slots=len(keys),
                                        readout=cfg.get("readout", "coupled"))
        self.ep = _Episode(status="ACTIVE")

    def _z(self, x, rel: int) -> float:
        w = self.W[rel]
        acc = x[0] * w[0]
        for d in range(1, 4):
            acc = acc + x[d] * w[d]
        return float(acc)

    def z_row(self, x) -> list:
        return [self._z(x, self.rel_of[s]) for s in range(len(self.keys))]

    def binding_status(self, slot: int, binding) -> str:
        code, supports = self.view.support_of(self.keys[slot])
        if code != ecsc.OK:
            return "RETIRED"
        for s in supports:
            if (bytes(s.source), int(s.version), bytes(s.scope)) == binding:
                return "LIVE" if s.state == ecsc.LIVE else "SUSPENDED"
        return "RETIRED"

    def sync(self, world: ecsc.AuthorityWorld) -> dict:
        """Apply an authority transition: the view syncs (its successor law); then each buffered event takes the
        VIEW's status of its binding. replay (B): RETIRED events are dropped and the state is rebuilt from the LIVE
        ones. slot (A): every slot with RETIRED support is reset (no replay), and slots with SUSPENDED support are
        excluded until restored. Either way the state equals a fresh learner fed exactly the applied events."""
        self._require(self.ep.status == "ACTIVE", "EPISODE_NOT_ACTIVE")
        code, res = self.view.sync(world)
        self._require(code == ecsc.OK, f"SYNC_{code}")
        self.world = world
        status = {(e[0], e[1]): self.binding_status(e[1], e[4]) for e in self.learner.buffer}
        retired = {k for k, st in status.items() if st == "RETIRED"}
        if self.frozen["invalidation"] == "slot":
            for slot in sorted({k[1] for k in retired}):
                self.learner.reset_slot(slot)
            excluded = {k[1] for k, st in status.items() if st == "SUSPENDED"}
            self.learner.rebuild(lambda e: e[1] not in excluded)
        else:
            self.learner.drop(lambda e: (e[0], e[1]) in retired)
            self.learner.rebuild(lambda e: status[(e[0], e[1])] == "LIVE")
        return {"retired": res.contributions_retired, "flipped": res.contributions_flipped,
                "status": sorted(status.values())}

    def admissible(self, t: int) -> list:
        out = []
        for s in range(len(self.keys)):
            row = self.reporters(t, self.rel_of[s])
            src, version, scope = self.world.binding(row)
            out.append(self.view.check(src, version, scope) == ecsc.OK)
        return out

    def propose_relation(self, x) -> int:
        """PROBE_R over admissible slots (argmax predictive; best unseen probed at steps 1-2 and forced by the
        coverage deadline)."""
        self._require(self.ep.status == "ACTIVE" and self.ep.proposed is None, "PROPOSAL_STATE")
        self._require(self.ep.t < self.frozen["horizon"], "HORIZON_EXHAUSTED")
        adm = self.admissible(self.ep.t)
        self._require(any(adm), "NO_ADMISSIBLE_RELATION")
        pred = self.learner.predictive(self.z_row(x))
        unseen = [adm[s] and self.learner.counts[s] == 0 for s in range(len(adm))]
        need = sum(unseen)

        def first_argmax(mask):
            best, bv = -1, -math.inf
            for s in range(len(adm)):
                if mask[s] and (best < 0 or pred[s] > bv):
                    best, bv = s, pred[s]
            return best
        force = need > 0 and (need >= self.frozen["horizon"] - self.ep.t or self.ep.t in (1, 2))
        self.ep.proposed = first_argmax(unseen) if force else first_argmax(adm)
        self.x = list(x)
        return self.ep.proposed

    def observe_selected(self, slot: int, readings) -> None:
        self._require(self.ep.status == "ACTIVE", "EPISODE_NOT_ACTIVE")
        self._require(self.ep.proposed is not None, "SELECT_REQUIRED")
        self._require(slot == self.ep.proposed, "SELECTION_NOT_ACCEPTED")
        readings = list(readings)[:3]
        expected = QUALIFIED_READ_SCHEDULE[self.ep.t] if self.admission.mode == "DUPLICATE" else 1
        if self.admission.mode != "DUPLICATE":
            readings = readings[:1] if len(readings) == QUALIFIED_READ_SCHEDULE[self.ep.t] else readings
        self._require(len(readings) == expected, "QUALIFIED_READING_COUNT_MISMATCH")
        events = {r.event_id for r in readings}
        self._require(len(events) == 1 and not (events & self.ep.events), "STALE_OR_FOREIGN_EVENT")
        self._require(len({r.reading_id for r in readings}) == len(readings), "READING_REPLAYED")
        self._require(not ({r.reading_id for r in readings} & self.ep.readings), "READING_REPLAYED")
        self._require(len({r.channel_id for r in readings}) == len(readings), "REREAD_ON_SAME_CHANNEL")
        self._require(all(r.expert == slot for r in readings), "READING_FOR_UNSELECTED_RELATION")
        self._require(all(r.value in (0, 1) for r in readings), "READING_VALUE")
        rel = self.rel_of[slot]
        row = self.reporters(self.ep.t, rel)
        src = self.world.rows[row]
        a, b = self.relations[rel]
        for k, r in enumerate(readings):
            obs = ecsc.observation(ecsc.ASSERT if r.value else ecsc.NEGATE, self.ep.t * 4 + k,
                                   int(src.source_authority), a, b, ecsc.digest("provenance", r.event_id, r.reading_id))
            code, key = self.view.admit(self.world, row, obs)
            self._require(code == ecsc.OK and key == self.keys[slot], f"ADMISSION_REFUSED_{code}")
        self.ep.events |= events
        self.ep.readings |= {r.reading_id for r in readings}
        self.learner.update(slot, self._z(self.x, rel), tuple(r.value for r in readings), self.ep.t,
                            self.world.binding(row))
        self.ep.t += 1
        self.ep.proposed = None

    def propose_terminal(self, queries, contexts) -> list:
        """queries: [(alternatives in template relation indices, stop payoff)]; contexts: [x'] -> per query the
        chosen alternative or -1 (STOP)."""
        self._require(self.ep.status == "ACTIVE" and self.ep.t == self.frozen["horizon"], "TERMINAL_PREMATURE")
        out = []
        for (alts, stop), x in zip(queries, contexts):
            slots = tuple(tuple(tuple(self.slot_of[r] for r in p) for p in alt) for alt in alts)
            out.append(self.learner.decide(slots, stop, self.z_row(x)))
        return out

    def export_projection(self, x) -> list:
        """Layer C: a bounded, NON-AUTHORITATIVE vector of slot saliences (u16, 0..1000). Read-only."""
        return [max(0, min(1000, int(round(1000.0 * p)))) for p in self.learner.predictive(self.z_row(x))]

    def close_episode(self) -> dict:
        summary = {"status": self.ep.status, "steps": self.ep.t, "failure": self.ep.failure,
                   "contract": self.admission.contract if self.admission else None}
        if self.learner is not None:
            self.learner.zeroize()
        self.ep = _Episode()
        return summary
