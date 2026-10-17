"""R8 DEV_ONLY: method-paired epistemic challenge on the existing frozen R2 trajectories.

This is a diagnostic/falsification experiment, NOT a scientific qualification:
all labels and relationship admissions originate in TEST_ONLY authored fixtures.
Source grant truth, semantic competence, language generation and task utility are
not tested. Work counters are instrumented logical counts, NOT elapsed time.

An exact-index comparator, admitted-only factor traversal, unrestricted exact
factor traversal, incremental exact state, and asynchronous recurrent state all
see the same immutable input graph and task. Only the scorer receives the
fixture's withheld synthetic labels; no output-key oracle enters a Task.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json

from elpis.runtime.context import ADMITTED

from .baselines import Refusal
from .evaluation import score
from .r2_b0i import B0I
from .r2_b2s import B2S
from .r2_common import semantic
from .r2_trajectories import trajectory
from .task import predict, prepare, readout

SCHEMA = "elpis.ecsc.r8.dev-epistemic-falsification.v1"
METHODS = ("INDEX_ONLY", "ADMITTED_ONLY", "B0", "B0I", "B2S")


def _identity(result):
    """Query/evidence binding AND predictions. Never equate counters with semantics."""
    return semantic(result)


def _attempt(fn):
    try:
        return fn(), None
    except Refusal as exc:
        return None, exc.code


def index_only(graph, task):
    """No propagation, but identical admission, budget, seed checks and readout."""
    meter = graph._meter(task.work_budget)
    try:
        scores, factors = prepare(graph, task, meter)
        return readout(graph, task, scores, factors, meter)
    except Refusal as exc:
        exc.used = meter.used
        raise


def admitted_task(task):
    """Explicit comparator: no CANDIDATE edge may contribute salience."""
    return replace(task, predicates=tuple(replace(p, statuses=(ADMITTED,)) for p in task.predicates))


def _extract(result, truth):
    if result is None:
        return None
    metric = score(result, truth)
    keys = tuple(sorted(pred.hit.grant.axiom.key for pred in result.predictions))
    return {
        "predicted_keys": keys,
        "tp": metric.true_positive,
        "fp": metric.false_positive,
        "fn": metric.false_negative,
        "invalid": metric.invalid_predictions,
        "omitted": metric.omitted,
        "binding_count": len(result.constraints),
        "candidate_evidence_links": sum(
            r.relation.status == "CANDIDATE"
            for p in result.predictions for r in p.relations),
        "admitted_evidence_links": sum(
            r.relation.status == "ADMITTED"
            for p in result.predictions for r in p.relations),
        "logical_work": result.work,
    }


def profile(arity, reverse):
    """One chronological replay. Refusals are results, not post-hoc rescues."""
    b0i = b2s = None
    rows = []
    for event in trajectory(arity=arity, reverse=reverse):
        graph, task = event.verified.graph, event.task
        assert not set(task.seeds).intersection(event.withheld), "WITHHELD_IN_TASK_SEEDS"
        exact, e_exact = _attempt(lambda: predict(graph, task, "B0"))
        zero, e_zero = _attempt(lambda: index_only(graph, task))
        admitted, e_admitted = _attempt(lambda: predict(graph, admitted_task(task), "B0"))

        def incremental():
            nonlocal b0i
            new, result, _ = (B0I.start(event.verified, task)
                              if b0i is None else b0i.warm(event.verified, task))
            b0i = new
            return result

        inc, e_inc = _attempt(incremental)
        if e_exact != e_inc:
            raise AssertionError((event.name, "B0_B0I_REFUSAL_MISMATCH", e_exact, e_inc))
        if exact is not None and _identity(exact) != _identity(inc):
            raise AssertionError((event.name, "B0_B0I_SEMANTIC_MISMATCH"))

        def recurrent():
            nonlocal b2s
            new, result, _ = (B2S.start(event.verified, task)
                              if b2s is None else b2s.warm_rebase(
                                  b2s.snapshot, b2s.verified, b2s.task,
                                  event.verified, task))
            b2s = new
            return result

        recur, e_recur = _attempt(recurrent)
        results = {
            "INDEX_ONLY": (zero, e_zero),
            "ADMITTED_ONLY": (admitted, e_admitted),
            "B0": (exact, e_exact),
            "B0I": (inc, e_inc),
            "B2S": (recur, e_recur),
        }
        row = {
            "event": event.name,
            "view_coordinate": (graph.view.version, graph.view.digest),
            "scope": task.scope,
            "withheld_count": len(event.withheld),
            "methods": {name: {"refusal": err, "score": _extract(result, event.withheld)}
                        for name, (result, err) in results.items()},
        }
        # A method must never expose a source-invalid prediction. The admitted
        # comparator cannot report candidate edges as evidence. Mechanical gates.
        for name, (result, err) in results.items():
            if result is None:
                continue
            metrics = row["methods"][name]["score"]
            if metrics["invalid"]:
                raise AssertionError((event.name, name, "INVALID_SOURCE_LEAK"))
            if name == "ADMITTED_ONLY" and metrics["candidate_evidence_links"]:
                raise AssertionError((event.name, "UNADMITTED_EVIDENCE_IN_ADMITTED_CONTROL"))
            if result.coordinate != graph.view.coordinate:
                raise AssertionError((event.name, name, "CROSS_VIEW_LEAK"))
            if result.query != (admitted_task(task) if name == "ADMITTED_ONLY" else task):
                raise AssertionError((event.name, name, "CROSS_TASK_LEAK"))
        rows.append(row)
    return {"arity": arity, "reverse": reverse, "events": rows}


def summarize(profiles):
    summary = {}
    for name in METHODS:
        scores = [r["methods"][name]["score"]
                  for p in profiles for r in p["events"]
                  if r["methods"][name]["score"] is not None]
        refusals = {}
        for p in profiles:
            for row in p["events"]:
                code = row["methods"][name]["refusal"]
                if code is not None:
                    refusals[code] = refusals.get(code, 0) + 1
        summary[name] = {
            "answered": len(scores), "refusals": dict(sorted(refusals.items())),
            "tp": sum(s["tp"] for s in scores),
            "fp": sum(s["fp"] for s in scores),
            "fn": sum(s["fn"] for s in scores),
            "invalid": sum(s["invalid"] for s in scores),
            "candidate_evidence_links": sum(s["candidate_evidence_links"] for s in scores),
            "admitted_evidence_links": sum(s["admitted_evidence_links"] for s in scores),
            "logical_work": sum(s["logical_work"] for s in scores),
        }
    return summary


def run():
    profiles = [profile(arity, reverse) for arity in (4, 8) for reverse in (False, True)]
    assert len(profiles) == 4 and all(len(p["events"]) == 20 for p in profiles)
    summary = summarize(profiles)
    assert summary["B0"]["tp"] == summary["B0I"]["tp"]
    assert summary["B0"]["fp"] == summary["B0I"]["fp"]
    assert summary["B0"]["fn"] == summary["B0I"]["fn"]
    # No demand that B2S equal exact closure; its failure to converge is the point.
    return {"schema": SCHEMA, "classification": "DEV_ONLY_SYNTHETIC_DESCRIPTIVE_NOT_SCIENTIFIC_QUAL",
            "oracle": "R2_AUTHORED_WITHHELD_KEYS_NOT_SEMANTIC_GROUND_TRUTH",
            "work": "INSTRUMENTED_LOGICAL_OPERATIONS_NOT_CPU_WORK_OR_TIME",
            "gate": "B0_EQUALS_B0I; NO_INVALID_SOURCE_LEAK; ADMITTED_ONLY_NO_CANDIDATE_EVIDENCE",
            "summary": summary, "profiles": profiles}


def write_report(path: Path):
    report = run()
    wire = (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != wire:
            raise RuntimeError("R8_EXISTING_REPORT_CONFLICT_PRESERVE_EVIDENCE")
    else:
        path.write_bytes(wire)
    return report, sha256(wire).hexdigest()


if __name__ == "__main__":
    p = Path(__file__).with_name("R8_DEV_FALSIFICATION.json")
    result, digest = write_report(p)
    print("R8_REPORT=", p)
    print("R8_REPORT_SHA256=", digest)
    for name, stats in result["summary"].items():
        print(name, "TP", stats["tp"], "FP", stats["fp"], "FN", stats["fn"],
              "REFUSALS", stats["refusals"], "OPS", stats["logical_work"])
    print("R8_MECHANICS=PASS_DEV_ONLY_NOT_SCIENTIFIC_QUAL")
