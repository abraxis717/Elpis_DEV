"""Bounded sequential DEV harness. Only --write writes the requested R2 JSON file."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from time import perf_counter_ns
from unittest.mock import patch

from . import task as frozen_task
from .baselines import Refusal, SOURCE_PIN
from .b2 import B2, predict_b2
from .evaluation import score
from .r2_b0i import B0I
from .r2_b2s import B2S
from .r2_common import VerifiedGraph, semantic
from .r2_trajectories import query, trajectory
from .test_b2 import ring


def frozen_profile(graph, task, method):
    """Counters at frozen function boundaries, without changing their algorithms.

    B0/B1 wrappers merely observe the same Meter. B2 uses its frozen cold call path.
    Harness-only wrappers add Python wall overhead, never logical solver work.
    """
    if method == "B2_cold":
        meter = graph._meter(task.work_budget)
        try:
            session, state = B2._start(graph, task, meter)
            setup = meter.used
            for _ in range(task.steps):
                state = session._warm(state, meter)
            update = meter.used - setup
            result = session._observe(state, meter)
            return result, dict(setup=setup, transition=0, update=update,
                                prediction=meter.used - setup - update)
        except Refusal as exc:
            exc.used = meter.used
            raise
    checkpoints = {}
    prepare, readout = frozen_task.prepare, frozen_task.readout

    def traced_prepare(g, t, meter):
        values = prepare(g, t, meter)
        checkpoints["setup"] = meter.used
        return values

    def traced_readout(g, t, scores, factors, meter):
        checkpoints["before_prediction"] = meter.used
        return readout(g, t, scores, factors, meter)

    with patch.object(frozen_task, "prepare", traced_prepare), patch.object(frozen_task, "readout", traced_readout):
        result = frozen_task.predict(graph, task, method)
    return result, dict(setup=checkpoints["setup"], transition=0,
                        update=checkpoints["before_prediction"] - checkpoints["setup"],
                        prediction=result.work - checkpoints["before_prediction"])


def record(result, withheld, wall, meta):
    return dict(outcome="COMPLETED", logical_work=result.work, wall_ns_samples=[wall],
                operations=dict(result.operations), phase_work=meta.pop("phase_work", {}),
                transition=meta, score=asdict(score(result, withheld)),
                predictions=[dict(key=p.hit.grant.axiom.key, salience=p.salience,
                                  source_state=p.hit.source_state, axiom_status=p.hit.grant.axiom.status,
                                  claim=p.claim, relation_statuses=[r.relation.status for r in p.relations])
                             for p in result.predictions],
                constraints=[dict(key=h.grant.axiom.key, source_state=h.source_state, status=h.grant.axiom.status)
                             for h in result.constraints], omitted=result.omitted)


def refusal(exc, wall):
    return dict(outcome="MECHANICAL_REFUSAL", code=exc.code, logical_work=exc.used,
                wall_ns_samples=[wall], predictions=None, omitted=None,
                phase_work=None, note="No replacement result; prior session unchanged. Refusal exposes total completed work only.")


def run_trajectory(arity):
    inc = warm = previous_graph = None
    rows = []
    events = iter(trajectory(arity=arity))
    while True:
        started = perf_counter_ns()
        try:
            event = next(events)
        except StopIteration:
            break
        event_ns = perf_counter_ns() - started
        graph, task = event.verified.graph, event.task
        assert not set(task.seeds) & event.withheld, "WITHHELD_KEY_IN_SOLVER_SEEDS"
        rebuilt = previous_graph is not event.verified
        row = dict(name=event.name, arity=arity, task=asdict(task), evidence=graph.evidence_digest,
                   policy=graph.policy_digest, view=asdict(graph.view.coordinate),
                   shared_event_construction_wall_ns=event_ns,
                   shared_build_work=sum(n for _, n in graph.build_operations) if rebuilt else 0,
                   shared_build_reused=not rebuilt, axioms=len(graph.axioms), relations=len(graph.relations),
                   incidence_entries=sum(len(v) for v in graph.incidence.values()),
                   dropped=graph.view.dropped, methods={})
        previous_graph = event.verified
        exact = None
        for method in ("B0", "B1", "B2_cold", "B0I", "B2S"):
            started = perf_counter_ns()
            try:
                if method in ("B0", "B1", "B2_cold"):
                    result, phases = frozen_profile(graph, task, method)
                    meta = {"mode": "COLD", "phase_work": phases}
                elif method == "B0I":
                    inc, result, meta = B0I.start(event.verified, task) if inc is None else inc.warm(event.verified, task)
                else:
                    warm, result, meta = B2S.start(event.verified, task) if warm is None else warm.warm_rebase(
                        warm.snapshot, warm.verified, warm.task, event.verified, task)
                elapsed = perf_counter_ns() - started
                assert sum(meta["phase_work"].values()) == result.work
                if method == "B0":
                    exact = result
                if method == "B0I":
                    assert exact is not None and semantic(result) == semantic(exact)
                if method in ("B0", "B1", "B2_cold"):
                    # Untimed verification that boundary instrumentation preserves
                    # every result field including original operation counters.
                    check = predict_b2(graph, task) if method == "B2_cold" else frozen_task.predict(graph, task, method)
                    assert result == check
                row["methods"][method] = record(result, event.withheld, elapsed, dict(meta))
                if method == "B0I":
                    row["methods"][method]["equality_to_frozen_B0_excluding_costs"] = True
            except Refusal as exc:
                row["methods"][method] = refusal(exc, perf_counter_ns() - started)
        rows.append(row)
        print("DEV arity", arity, event.name, " | ".join(
            f"{m}: {r.get('code', str(r['logical_work']) + ' ops')}"
            for m, r in row["methods"].items()), flush=True)
    return rows


def damage_control():
    f = ring()
    g = VerifiedGraph(f.view, f.authority, f.checks)
    q = query(f)
    withheld = frozenset()
    rows = {}
    inc, _, _ = B0I.start(g, q)
    warm, _, _ = B2S.start(g, q)
    frozen, initial = B2.start(g.graph, q)
    started = perf_counter_ns()
    damaged = frozen.damage(initial, tuple((k, 1000) for k in frozen.cells))
    rows["B2_damage"] = dict(logical_work=damaged.work, wall_ns_samples=[perf_counter_ns() - started])
    started = perf_counter_ns()
    warm, ops = warm.damage(warm.snapshot, tuple((k, 1000) for k in warm.snapshot.scores))
    rows["B2S_damage"] = dict(logical_work=sum(n for _, n in ops), wall_ns_samples=[perf_counter_ns() - started])
    try:
        warm.observe(warm.snapshot)
        raise AssertionError("Tainted readout accepted")
    except Refusal as exc:
        assert exc.code == "TAINTED_STATE"
        rows["B2S_tainted_observe"] = dict(outcome="MECHANICAL_REFUSAL", code=exc.code, logical_work=exc.used)
    for name in ("B0", "B1", "B2_cold", "B0I", "B2S", "B2_damaged_warm"):
        started = perf_counter_ns()
        if name in ("B0", "B1", "B2_cold"):
            r, phases = frozen_profile(g.graph, q, name)
            meta = dict(mode="COLD", phase_work=phases)
        elif name == "B0I":
            inc, r, meta = inc.warm(g, q)
        elif name == "B2S":
            warm, r, meta = warm.warm_rebase(warm.snapshot, g, q, g, q)
        else:
            meter = g.graph._meter(q.work_budget)
            state = damaged
            for _ in range(q.steps):
                state = frozen._warm(state, meter)
            update = meter.used
            r = frozen._observe(state, meter)
            meta = dict(mode="FROZEN_NEGATIVE_CONTROL", phase_work=dict(setup=0, transition=0,
                        update=update, prediction=meter.used - update))
        rows[name] = record(r, withheld, perf_counter_ns() - started, dict(meta))
    assert rows["B2_damaged_warm"]["score"]["false_positive"] == 8
    assert rows["B2S"]["score"]["false_positive"] == 0
    return dict(name="unsupported_cycle_damage_then_explicit_recovery", task=asdict(q),
                shared_build_work=sum(n for _, n in g.graph.build_operations), methods=rows,
                noncomparability="Only B2/B2S receive explicit activation damage. B0/B1 recompute; B0I retains its exact zero cache. B2S damage is tracked/tainted, frozen B2 damage is not.")


def main():
    rows = run_trajectory(4) + run_trajectory(8)
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "BASELINE_R1_FREEZE.json").read_text())
    frozen_hashes = {}
    for name, expected in manifest["revisions"].items():
        actual = sha256((root.parent.parent / name).read_bytes()).hexdigest()
        assert actual == expected, name
        frozen_hashes[name] = actual
    canonical = sha256((root.parent.parent / "src/elpis/runtime/context.py").read_bytes()).hexdigest()
    assert canonical == manifest["canonical_context_sha256"]
    report = dict(phase="R2_DEV_MECHANICS_ONLY", scientific_status="NOT_EVALUATED; NO_QUAL",
                  source_pin=SOURCE_PIN, canonical_context_sha256=canonical,
                  baseline_freeze_sha256=sha256((root / "BASELINE_R1_FREEZE.json").read_bytes()).hexdigest(),
                  frozen_hashes=frozen_hashes,
                  r2_source_hashes={p.name: sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(root.glob("*r2*.py"))},
                  environment=dict(python=platform.python_version(), platform=platform.platform()),
                  shared_limits=asdict(next(trajectory()).verified.graph.limits), events=rows,
                  cycle_damage=damage_control(),
                  result="RETAINED_ACTIVATION_DEMONSTRATED_MECHANICALLY; NO_RECURRENT_UTILITY_ADVANTAGE_ESTABLISHED",
                  limitations=[
                      "Single descriptive wall sample per event/method; no calibration or statistical inference.",
                      "B0/B1 phase wrappers add harness wall overhead; logical counters and results verified identical.",
                      "Shared event wall includes fixture/successor/admission; not attributed to a solver.",
                      "Logical work counts records, visits, copying and hash bytes, not CPU instructions or FLOPs.",
                      "Policy/order checks and all dependency construction charged. Python allocation/sorting/footprint overhead only in wall time.",
                      "B0I scope fallback uses frozen B0 and discards cache; its setup/update/readout costs are disclosed jointly.",
                      "Refusals expose completed total work only; per-phase counters unavailable on refused calls.",
                      "Warm and cold methods have different prior work; each event discloses that history and setup cost.",
                      "Partial asynchronous schedules have no finite fairness or convergence guarantee.",
                      "Opaque in-process receipts/snapshots are not a sandbox against hostile Python reflection or arbitrary memory corruption.",
                      "No external source attestation, persistence, authority migration, learning, HOT integration or scientific qualification."])
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    assert len(encoded.encode()) < 1_000_000
    if sys.argv[1:] == ["--write"]:
        (root / "R2_DEV_RESULTS.json").write_text(encoded)
        print("WROTE R2_DEV_RESULTS.json", len(encoded.encode()), "bytes", flush=True)
    elif sys.argv[1:]:
        raise SystemExit("Only --write is supported")
    else:
        print("DEV record prepared", len(rows), "events; use --write for the authorized bounded JSON")


if __name__ == "__main__":
    main()
