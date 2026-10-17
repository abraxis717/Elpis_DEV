"""B0/B1 shared-task DEV mechanics record; stdout only, never QUAL."""
from dataclasses import asdict
import json
from time import perf_counter_ns

from .baselines import Baselines, SOURCE_PIN
from .evaluation import score
from .fixtures import SEED, coding
from .task import Predicate, Task, predict


def dev_task(fixture):
    return Task("project-arch", tuple(k for s, kind, k in fixture.keys
                                     if s == "project-arch" and kind != "plan"),
                tuple(Predicate(s.relation_type, s.roles, s.kinds) for s in fixture.authority.schemas),
                ("plan",), steps=3)


def measured(call):
    started = perf_counter_ns()
    result = call()
    return result, perf_counter_ns() - started


def run():
    fixture = coding(SEED)
    baseline, build_ns = measured(lambda: Baselines.build(fixture.view, fixture.authority, fixture.checks))
    task = dev_task(fixture)
    # Withheld answer only crosses into score(), never predict() or Task.
    truth = frozenset((fixture.key(task.scope, "plan"),))
    methods = {}
    for method in ("B0", "B1"):
        result, wall_ns = measured(lambda: predict(baseline, task, method))
        methods[method] = {"operations": dict(result.operations), "logical_work": result.work,
                           "wall_ns": wall_ns, "score": asdict(score(result, truth))}
    return {
        "phase": "DEV_MECHANICS_ONLY", "authority": "TEST_ONLY_SYNTHETIC_ALLOWLIST",
        "source_pin": SOURCE_PIN, "seed": SEED, "limits": asdict(baseline.limits),
        "view_digest": fixture.view.digest, "policy_digest": baseline.policy_digest,
        "evidence_digest": baseline.evidence_digest, "task": asdict(task),
        "visible_axioms": len(baseline.axioms), "visible_relations": len(baseline.relations),
        "incidences": sum(len(v) for v in baseline.incidence.values()),
        "build": {"operations": dict(baseline.build_operations),
                  "logical_work": sum(n for _, n in baseline.build_operations), "wall_ns": build_ns},
        "methods": methods,
        "scientific_status": "NOT_EVALUATED; NO_QUAL; NO_UTILITY_OR_STABILITY_CLAIM",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
