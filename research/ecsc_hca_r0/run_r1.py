"""Small DEV comparison and recovery controls. Stdout only; no persisted active state."""
from dataclasses import asdict, replace
import json
from time import perf_counter_ns

from .b2 import B2, footprint, predict_b2
from .baselines import Baselines, Refusal, SOURCE_PIN
from .evaluation import score
from .fixtures import coding
from .run_dev import dev_task, measured
from .task import predict


def summarize(call, truth):
    started = perf_counter_ns()
    try:
        result = call()
    except Refusal as exc:
        return {"outcome": "MECHANICAL_REFUSAL", "code": exc.code, "logical_work": exc.used,
                "wall_ns": perf_counter_ns() - started}
    elapsed = perf_counter_ns() - started
    return {"outcome": "COMPLETED", "logical_work": result.work, "wall_ns": elapsed,
            "operations": dict(result.operations), "score": asdict(score(result, truth))}


def run():
    f = coding()
    graph, build_ns = measured(lambda: Baselines.build(f.view, f.authority, f.checks))
    task = dev_task(f)
    truth = frozenset((f.key(task.scope, "plan"),))
    comparisons = []
    for cap in (64, 128, 2048, 4096, 100_000):
        q = replace(task, work_budget=cap)
        row = {"common_work_cap": cap, "B0": summarize(lambda: predict(graph, q, "B0"), truth),
               "B1": summarize(lambda: predict(graph, q, "B1"), truth),
               "B2": summarize(lambda: predict_b2(graph, q), truth)}
        comparisons.append(row)
    session, initial = B2.start(graph, task)
    healthy = initial
    for _ in range(task.steps):
        healthy = session.warm(healthy)
    damaged, damage_ns = measured(lambda: session.damage(healthy, tuple((k, 0) for k in session.cells)))

    def repair():
        meter = graph._meter(task.work_budget)
        state = damaged
        for _ in range(task.steps):
            state = session._warm(state, meter)
        return session._observe(state, meter)

    def reinitialize():
        meter = graph._meter(task.work_budget)
        fresh, state = B2._start(graph, task, meter)
        for _ in range(task.steps):
            state = fresh._warm(state, meter)
        return fresh._observe(state, meter)

    recovery = {"damage": {"logical_work": damaged.work, "wall_ns": damage_ns},
                "B2_local_repair": summarize(repair, truth),
                "B2_reinitialize_then_recur": summarize(reinitialize, truth),
                "B0_exact_recompute": summarize(lambda: predict(graph, task, "B0"), truth),
                "B1_fresh_recompute": summarize(lambda: predict(graph, task, "B1"), truth)}
    # No assertion of utility: the output makes inferior repair cost visible.
    return {
        "phase": "R1_DEV_MECHANICS_ONLY", "source_pin": SOURCE_PIN, "seed": 1729,
        "scientific_status": "NOT_EVALUATED; NO_QUAL; NONLEARNED_EXPLORATORY_COMPARATOR",
        "view_digest": graph.view.digest, "policy_digest": graph.policy_digest,
        "evidence_digest": graph.evidence_digest,
        "shared_limits": asdict(graph.limits), "shared_query": asdict(task),
        "capacity": {"axioms": len(graph.axioms), "relations": len(graph.relations),
                     "incidences": sum(len(v) for v in graph.incidence.values()),
                     "max_degree": max(map(len, graph.incidence.values())),
                     "max_arity": max(len(r.relation.members) for r in graph.relations.values()),
                     "activation_payload_bytes": len(initial.values),
                     "auxiliary_owned_bytes_at_start": footprint((session, initial), (graph, task)),
                     "auxiliary_reservation_bytes": session.auxiliary_reservation},
        "shared_build": {"logical_work": sum(n for _, n in graph.build_operations), "wall_ns": build_ns},
        "comparisons": comparisons, "recovery": recovery,
        "limitations": ["Logical work is not instructions/FLOPs; wall times are single noisy observations.",
                        "Direct incoming provenance is not a transitive native proof; all scores are salience.",
                        "Unit-weight unsupported cycles can retain damaged positives; exact reset is a control.",
                        "B2 repair has no demonstrated advantage over exact recomputation."],
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
