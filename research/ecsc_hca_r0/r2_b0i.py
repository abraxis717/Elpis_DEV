"""B0I: exact task-conditioned incremental least closure, independent R2 provenance."""
from dataclasses import dataclass
from heapq import heappop, heappush

from .baselines import Refusal, require
from .task import message, predict, readout
from .r2_common import (affected, carry, memory_check, meter_for, prepared,
                        seal_scores, transition)


def close_region(graph, scores, factors, deps, region, meter):
    """Recompute region from seed clamps with exact unaffected boundary values.

    No old values survive in region. Union-dependency closure means its exterior
    cannot depend on it. Monotone integer worklist reaches the same least fixed
    point as frozen B0; cycles without external support therefore remain zero.
    """
    meter.tick("closure_factor_scan", len(factors))
    queue = sorted(k for k, h in factors.items() if h.relation.members[-1] in region)
    queued = set(queue)
    meter.tick("closure_queue_init", len(queue))
    while queue:
        rk = heappop(queue)
        queued.remove(rk)
        meter.tick("closure_queue_pop")
        hit = factors[rk]
        value = message(graph, hit, scores, meter)
        out = hit.relation.members[-1]
        meter.tick("closure_compare")
        if value > scores[out]:
            scores[out] = value
            meter.tick("closure_write")
            for dependent in deps[out]:
                meter.tick("closure_dependency")
                if factors[dependent].relation.members[-1] in region and dependent not in queued:
                    heappush(queue, dependent)
                    queued.add(dependent)
                    meter.tick("closure_queue_push")


@dataclass(frozen=True, init=False)
class B0I:
    verified: object
    task: object
    scores: object
    factors: object
    deps: object
    reservation: int

    def __init__(self, *args, **kwargs):
        raise TypeError("Use B0I.start or an issued cache's warm interface")

    @classmethod
    def _issue(cls, verified, task, scores, factors, deps, reservation):
        session = object.__new__(cls)
        for name, value in zip(("verified", "task", "scores", "factors", "deps", "reservation"),
                               (verified, task, scores, factors, deps, reservation)):
            object.__setattr__(session, name, value)
        return session

    @classmethod
    def start(cls, verified, task):
        return cls._run(None, verified, task)

    def warm(self, verified, task):
        return self._run(self, verified, task)

    @classmethod
    def _run(cls, old, verified, task):
        meter = meter_for(verified, task)
        graph = verified.graph
        phase = {"setup": 0, "transition": 0, "update": 0, "prediction": 0}
        try:
            if old is not None:
                # Identical immutable inputs need neither graph scans nor copies.
                # Readout is still charged; the cache is never a stored answer key.
                meter.tick("cache_identity")
                if old.verified is verified and old.task is not task:
                    meter.tick("task_compare", 1 + len(task.seeds) + len(task.predicates) + len(task.output_kinds))
                if old.verified is verified and old.task == task:
                    phase["transition"] = meter.used
                    result = readout(graph, task, old.scores, old.factors, meter)
                    phase["prediction"] = meter.used - phase["transition"]
                    return old, result, {"mode": "EXACT_CACHE_REUSE", "cache_retained": True,
                                         "retained": len(old.scores), "reset": 0, "affected": 0,
                                         "phase_work": phase}
                transition(old.verified, verified, old.task, task, meter, same_scope=False)
                if old.task.scope != task.scope:
                    # Explicit fallback to the actual frozen B0. No substituted result
                    # on refusal and no pretend cache: caller receives None session.
                    remaining = task.work_budget - meter.used
                    require(remaining > 0, "WORK_LIMIT")
                    from dataclasses import replace
                    try:
                        result = predict(graph, replace(task, work_budget=remaining), "B0")
                    except Refusal as exc:
                        meter.tick("fallback_completed_work", exc.used)
                        raise
                    for name, count in result.operations:
                        meter.tick("fallback_" + name, count)
                    result = replace(result, query=task, operations=meter.result())
                    fallback_work = sum(n for name, n in result.operations if name.startswith("fallback_"))
                    return None, result, {"mode": "FULL_RECOMPUTE_FROZEN_B0_SCOPE", "cache_retained": False,
                                          "phase_work": {"transition": result.work - fallback_work,
                                                         "fallback_setup_update_prediction": fallback_work}}
            phase["transition"] = meter.used
            scores, factors, deps, reservation = prepared(verified, task, meter)
            phase["setup"] = meter.used - phase["transition"]
            before = meter.used
            if old is None:
                region = frozenset(scores)
                stats = (0, 0, len(scores))
                mode = "COLD_EXACT_CACHE_CONSTRUCTION"
            else:
                region = affected(old.verified.graph, graph, old.task, task, old.scores,
                                  scores, old.factors, factors, old.deps, deps, meter)
                stats = carry(old.scores, scores, region, meter)
                mode = "INCREMENTAL_EXACT" if region else "EXACT_CACHE_REUSE"
            phase["transition"] += meter.used - before
            before = meter.used
            close_region(graph, scores, factors, deps, region, meter)
            phase["update"] = meter.used - before
            before = meter.used
            sealed = seal_scores(scores, meter)
            session = cls._issue(verified, task, sealed, factors, deps, reservation)
            memory_check(session, graph, task, reservation, meter)
            phase["setup"] += meter.used - before
            before = meter.used
            result = readout(graph, task, sealed, factors, meter)
            phase["prediction"] = meter.used - before
            return session, result, dict(mode=mode, retained=stats[0], retained_positive=stats[1],
                                         reset=stats[2], affected=len(region), cache_retained=True, phase_work=phase)
        except Refusal as exc:
            exc.used = meter.used
            raise
