"""B2S: issued immutable warm snapshots with conservative dependency refresh.

No B2 Activation is imported or altered. No HOT mutation, I/O, history or training.
All positive values originate from valid seed clamps/local factors, or explicitly
tainted damage that must be invalidated before observation or another warm tick.
"""
from dataclasses import dataclass
from hashlib import sha256

from .baselines import Refusal, integer, require
from .task import message, readout
from .r2_common import (affected, carry, memory_check, meter_for, prepared,
                        seal_scores, transition)

SCHEDULE_DOMAIN = b"ecsc-hca-r2.B2S.schedule\0"


@dataclass(frozen=True)
class Snapshot:
    scores: object
    step: int
    tainted: frozenset = frozenset()


@dataclass(frozen=True, init=False)
class B2S:
    verified: object
    task: object
    factors: object
    deps: object
    reservation: int
    snapshot: Snapshot

    def __init__(self, *args, **kwargs):
        raise TypeError("Use B2S.start or an issued session's explicit WARM interface")

    @classmethod
    def _issue(cls, verified, task, factors, deps, reservation, snapshot):
        session = object.__new__(cls)
        for name, value in zip(("verified", "task", "factors", "deps", "reservation", "snapshot"),
                               (verified, task, factors, deps, reservation, snapshot)):
            object.__setattr__(session, name, value)
        return session

    def _bound(self, snapshot, old_graph, old_task, meter):
        meter.tick("snapshot_identity")
        require(snapshot is self.snapshot and old_graph is self.verified
                and old_task == self.task, "UNISSUED_OR_UNBOUND_STATE")

    @classmethod
    def start(cls, verified, task):
        return cls._run(None, None, None, None, verified, task)

    def warm_rebase(self, snapshot, old_graph, old_task, verified, task):
        """Explicit WARM transaction: validate -> invalidate -> fire -> read -> publish.

        Old immutable snapshot/graph/task are explicit. A failed transaction returns
        nothing and leaves them untouched. Every call has one shared task work cap.
        """
        return self._run(self, snapshot, old_graph, old_task, verified, task)

    @classmethod
    def _run(cls, old, snapshot, old_graph, old_task, verified, task):
        meter = meter_for(verified, task)
        graph = verified.graph
        phase = {"setup": 0, "transition": 0, "update": 0, "prediction": 0}
        try:
            if old is not None:
                old._bound(snapshot, old_graph, old_task, meter)
                transition(old.verified, verified, old.task, task, meter)
            phase["transition"] = meter.used
            scores, factors, deps, reservation = prepared(verified, task, meter)
            phase["setup"] = meter.used - phase["transition"]
            before = meter.used
            if old is None:
                stats, region, step = (0, 0, len(scores)), frozenset(scores), 0
                mode = "COLD_SEED_INITIALIZATION"
            else:
                region = affected(old.verified.graph, graph, old.task, task, snapshot.scores,
                                  scores, old.factors, factors, old.deps, deps, meter, snapshot.tainted)
                stats = carry(snapshot.scores, scores, region, meter)
                step = snapshot.step
                mode = "TOTAL_REINITIALIZATION" if stats[0] == 0 else "SELECTIVE_CARRYOVER"
            phase["transition"] += meter.used - before
            before = meter.used
            # Only explicit WARM calls reach this loop. New schedules do not change
            # the validity of unaffected support; the counter never resets on rebase.
            seed_set = set(task.seeds)
            meter.tick("seed_index", len(seed_set))
            for _ in range(task.steps):
                require(step < 2**64 - 1, "STEP_LIMIT")
                entries = []
                for key in scores:
                    meter.tick("schedule_hash_byte", len(SCHEDULE_DOMAIN) + 48)
                    entries.append((sha256(SCHEDULE_DOMAIN + task.schedule_seed.to_bytes(8, "big")
                                           + step.to_bytes(8, "big") + bytes.fromhex(key)).digest(), key))
                meter.tick("schedule_order", len(entries))
                for _, key in sorted(entries)[:task.updates_per_tick]:
                    meter.tick("cell_fire")
                    value = 1000 if key in seed_set else 0
                    for rk, position in graph.incidence[key]:
                        meter.tick("incidence_visit")
                        hit = factors.get(rk)
                        if hit is not None and position == len(hit.roles) - 1:
                            value = max(value, message(graph, hit, scores, meter))
                    meter.tick("cell_update")
                    scores[key] = value if graph.axioms[key].source_valid else 0
                step += 1
            phase["update"] = meter.used - before
            before = meter.used
            snap = Snapshot(seal_scores(scores, meter), step)
            session = cls._issue(verified, task, factors, deps, reservation, snap)
            memory_check(session, graph, task, reservation, meter)
            phase["setup"] += meter.used - before
            before = meter.used
            result = readout(graph, task, snap.scores, factors, meter)
            phase["prediction"] = meter.used - before
            before = meter.used
            retained_derived = 0 if old is None else sum(
                key not in region and key not in seed_set and value > 0
                for key, value in snapshot.scores.items() if key in scores)
            meter.tick("retention_read", 0 if old is None else len(snapshot.scores))
            phase["transition"] += meter.used - before
            from dataclasses import replace
            result = replace(result, operations=meter.result())
            return session, result, dict(mode=mode, retained=stats[0], retained_positive=stats[1],
                                         retained_derived_positive=retained_derived,
                                         reset=stats[2], affected=len(region), step=step, phase_work=phase)
        except Refusal as exc:
            exc.used = meter.used
            raise

    def observe(self, snapshot):
        meter = meter_for(self.verified, self.task)
        try:
            self._bound(snapshot, self.verified, self.task, meter)
            require(not snapshot.tainted, "TAINTED_STATE")
            return readout(self.verified.graph, self.task, snapshot.scores, self.factors, meter)
        except Refusal as exc:
            exc.used = meter.used
            raise

    def damage(self, snapshot, changes):
        """Harness-only explicit intervention; every touched cell becomes tainted.

        Tainted snapshots cannot be read as predictions. The next WARM transaction
        clears their entire downstream closure, even across unchanged evidence.
        """
        meter = meter_for(self.verified, self.task)
        try:
            self._bound(snapshot, self.verified, self.task, meter)
            require(type(changes) is tuple and len(changes) <= len(snapshot.scores), "INVALID_DAMAGE")
            tainted = set(snapshot.tainted)
            seen = set()
            for change in changes:
                meter.tick("damage_validate")
                require(type(change) is tuple and len(change) == 2, "INVALID_DAMAGE")
                key, value = change
                require(type(key) is str and key in snapshot.scores and key not in seen
                        and integer(value, 0, 1000), "INVALID_DAMAGE")
                seen.add(key)
                tainted.add(key)
            meter.tick("damage_copy", len(snapshot.scores))
            scores = dict(snapshot.scores)
            for key, value in changes:
                meter.tick("damage_write")
                scores[key] = value
            result = self._issue(self.verified, self.task, self.factors, self.deps, self.reservation,
                                 Snapshot(seal_scores(scores, meter), snapshot.step, frozenset(tainted)))
            memory_check(result, self.verified.graph, self.task, self.reservation, meter)
            return result, meter.result()
        except Refusal as exc:
            exc.used = meter.used
            raise
