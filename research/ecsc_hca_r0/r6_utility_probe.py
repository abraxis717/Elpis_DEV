"""R6 DEV_ONLY: contextual-utility observability, not a scientific experiment.

Replays the *existing* R2 authored synthetic tasks, comparing a fresh exact
closure with retained exact B0I. The harness owns withheld target keys; those
keys are never passed to either predictor. Includes admission work, cold cache
construction, refusals and whole-trajectory cost. Source-verified WARM refresh
is measured separately with real local bytes and bounded HACF resolution.

This deliberately makes no claim about semantic relevance, language generation,
real-user tasks, scientific significance or CPU-level operation equivalence.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter_ns
import json

from .baselines import Refusal
from .evaluation import score
from .r2_b0i import B0I
from .r2_common import semantic
from .r2_trajectories import trajectory
from .task import predict


def _methods(event, cache):
    """Exact paired results with no oracle routed into the predictors."""
    graph, task = event.verified.graph, event.task
    assert not set(task.seeds).intersection(event.withheld), "WITHHELD_LEAKAGE"
    try:
        fresh = predict(graph, task, "B0")
        fresh_error = None
    except Refusal as exc:
        fresh, fresh_error = None, exc.code
    try:
        updated, incremental, info = (B0I.start(event.verified, task) if cache is None
                                       else cache.warm(event.verified, task))
        incremental_error = None
    except Refusal as exc:
        # A refusal MUST preserve the caller's prior valid cache; no rescue.
        updated, incremental, info, incremental_error = cache, None, {}, exc.code
    assert fresh_error == incremental_error, (event.name, fresh_error, incremental_error)
    if fresh is None:
        assert incremental is None
        return updated, {"event": event.name, "disposition": fresh_error}
    assert semantic(fresh) == semantic(incremental), f"EXACT_MISMATCH: {event.name}"
    a, b = score(fresh, event.withheld), score(incremental, event.withheld)
    assert a == b, "SCORE_MISMATCH"
    return updated, {
        "event": event.name, "disposition": "EXACT",
        "gold_count": len(event.withheld),
        "true_positive": a.true_positive,
        "false_positive": a.false_positive,
        "false_negative": a.false_negative,
        "omitted": a.omitted,
        "fresh_work": fresh.work,
        "incremental_work": incremental.work,
        "cache_mode": info["mode"],
        "incremental_cache_retained": info["cache_retained"],
        "context_coordinate": [graph.view.version, graph.view.digest],
    }


def replay_profiles():
    """DEV replay, including graph admission (counted once per new graph)."""
    report = []
    for arity in (4, 8):
        for reverse in (False, True):
            cache = None
            previous_graph = None
            rows = []
            fresh_total = incremental_total = graph_total = refused = 0
            for event in trajectory(arity=arity, reverse=reverse):
                # Work of each newly constructed immutable graph is shared by
                # both candidates; avoid counting it again for task-only turns.
                graph_work = (sum(n for _, n in event.verified.graph.build_operations)
                              if event.verified is not previous_graph else 0)
                previous_graph = event.verified
                cache, row = _methods(event, cache)
                row["shared_admission_work"] = graph_work
                graph_total += graph_work
                if row["disposition"] == "EXACT":
                    fresh_total += row["fresh_work"]
                    incremental_total += row["incremental_work"]
                else:
                    refused += 1
                rows.append(row)
            report.append({
                "arity": arity, "reverse_observation_order": reverse,
                "event_count": len(rows), "refusals": refused,
                "sum_shared_admission_work": graph_total,
                "sum_fresh_solver_work": fresh_total,
                "sum_incremental_solver_work": incremental_total,
                "sum_fresh_including_shared_admission": fresh_total + graph_total,
                "sum_incremental_including_shared_admission": incremental_total + graph_total,
                "events": rows,
            })
    return report


def warm_refresh_probe(repetitions=12):
    """Small real-byte HACF fixture: cold open versus explicit same-view refresh.

    The same file is verified on every call. Wall timings are descriptive, not
    scientifically frozen or stable benchmarks, and no cold-page-cache claims.
    """
    from .r5_warm_publication import WarmExactPublication
    from .test_r4_hacf_sources import HACFSourceBoundaryTests
    from .r2_common import semantic

    fixture = HACFSourceBoundaryTests(
        methodName="test_original_bytes_reverified_in_place_and_normalized_digest_bound")
    fixture.setUp()
    try:
        def publish(publisher):
            return publisher.refresh(fixture.root, fixture.manifest, fixture.view,
                                     fixture.authority, fixture.bindings, fixture.task)

        warm = WarmExactPublication()
        first = publish(warm)
        cold_ns, warm_ns = [], []
        work_cold, work_warm = [], []
        for i in range(repetitions):
            fresh = WarmExactPublication()
            # Alternate call order to reduce systematic cache-order advantage.
            ops = (("cold", fresh), ("warm", warm)) if i % 2 == 0 else (
                ("warm", warm), ("cold", fresh))
            for kind, obj in ops:
                t0 = perf_counter_ns()
                published = publish(obj)
                elapsed = perf_counter_ns() - t0
                assert semantic(published.result) == semantic(first.result), "HACF_RESULT_DRIFT"
                assert published.coordinate == first.coordinate
                assert obj.peek(fixture.view.coordinate) is published
                if kind == "cold":
                    cold_ns.append(elapsed)
                    work_cold.append(published.result.work)
                else:
                    warm_ns.append(elapsed)
                    work_warm.append(published.result.work)
        assert len(cold_ns) == len(warm_ns) == repetitions
        return {
            "trials_per_arm": repetitions,
            "source_bytes": len(fixture.data),
            "document_sha256": sha256(fixture.data).hexdigest(),
            "cold_wall_ns": cold_ns,
            "warm_wall_ns": warm_ns,
            "median_cold_wall_ns": median(cold_ns),
            "median_warm_wall_ns": median(warm_ns),
            "cold_result_work": work_cold,
            "warm_result_work": work_warm,
            "semantics": "EXACT_EQUAL",
            "attestation": "HACF_BYTES_VERIFIED_PER_REFRESH; GRANTS_TEST_ONLY",
        }
    finally:
        fixture.doCleanups()


def run(repetitions=12):
    rows = replay_profiles()
    measured = warm_refresh_probe(repetitions)
    return {
        "schema": "elpis.ecsc.r6.dev-utility-probe.r0",
        "disposition": "DEV_DESCRIPTIVE_ONLY_NOT_SCIENTIFIC_QUAL",
        "source_authority": "Research B0/B0I and R5 unchanged; DEV branch input authority",
        "limitations": [
            "Withheld labels are synthetic scenario keys, not ground-truth semantic meanings.",
            "Work units are instrumentation counts; graph admission is a shared cost.",
            "Wall times are single-process descriptive measurements over tiny source files.",
            "No frozen scientific protocol, deployment, ECS_G encoder or trained recurrence.",
            "Reading verified source bytes never admits semantic truth.",
        ],
        "trajectories": rows,
        "hacf_refresh": measured,
    }


if __name__ == "__main__":
    data = run()
    print(json.dumps(data, sort_keys=True, indent=2))
