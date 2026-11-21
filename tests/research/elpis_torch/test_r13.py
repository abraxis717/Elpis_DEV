"""Elpis Torch R13: the Geodome cognitive substrate, P1-P4 (mechanics and gradients only). RESEARCH_ONLY.

The admitted Geodome sources and their identity; the plan, working-state and executor contracts against Geodome's
own scheduler; gradients over the bounded horizon; the bridge to the actual EvidenceLM (disabled-path and zero-gate
parity, the context clock, evidence admission, refusals, atomicity, isolation, immutability, reset). Nothing here
tests or claims competence.
"""
from __future__ import annotations

import copy
import pickle
import shutil
import sys
import types

import pytest

from .conftest import REPO, SKIP_REASON

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch.context_runtime import BudgetError  # noqa: E402
from research.elpis_torch.experiments import run_r13 as r13  # noqa: E402
from research.elpis_torch.geodome_cognitive import source as gsource  # noqa: E402
from research.elpis_torch.geodome_cognitive.binding import CognitiveError, CognitiveSession  # noqa: E402
from research.elpis_torch.geodome_cognitive.plan import ExecutionPlan, PlanError, compile_plan  # noqa: E402
from research.elpis_torch.geodome_cognitive.reference import scheduler_reference  # noqa: E402
from research.elpis_torch.geodome_cognitive.state import TensorWorkingState, WorkingStateError  # noqa: E402
from research.elpis_torch.geodome_cognitive.topology import substrate  # noqa: E402
from research.elpis_torch.model import parameter_digest  # noqa: E402


@pytest.fixture(scope="module")
def sub():
    return substrate()


@pytest.fixture(scope="module")
def backbone():
    return r13.build_backbone()


def _two_hop(sub, v):
    return next(w for w in sub.neighbors(sub.neighbors(v)[0]) if w != v and w not in sub.neighbors(v))


# -- P1: the frozen dependency -----------------------------------------------------------------------------------

def test_the_vendored_geodome_is_the_pinned_upstream_bytes():
    digests = gsource.verify_sources()
    assert gsource.MANIFEST_SHA256 == "e44d288fec19a09c17be0ce503f011d400b499e4d393f1275d305e2b23d31cb8"
    assert digests == {gsource.MANIFEST: gsource.MANIFEST_SHA256, **{k: pin for k, (_, pin) in gsource.FILES.items()}}
    assert gsource.GEODOME_COMMIT == "4e5fe56670c61a0d8906770b03033658c3087c3a"
    assert (gsource.GEODOME_API_VERSION, gsource.GEODOME_LICENSE) == ("1.0.0", "Apache-2.0")


@pytest.mark.parametrize("damage", ["edited", "extra", "missing"])
def test_a_damaged_vendored_tree_is_refused(tmp_path, damage):
    root = tmp_path / "geodome_v1"
    shutil.copytree(gsource.VENDORED, root, ignore=shutil.ignore_patterns("__pycache__"))
    assert gsource.verify_sources(root)
    if damage == "edited":
        path = root / "geodome_scheduler.py"
        path.write_bytes(path.read_bytes() + b"\n")
    elif damage == "extra":
        (root / "geodome_helpers.py").write_text("x = 1\n", encoding="utf-8")
    else:
        (root / "GEO144K" / "g2b_bridge.py").unlink()
    with pytest.raises(gsource.GeodomeSourceError):
        gsource.verify_sources(root)


def test_a_shadowing_module_is_refused_before_anything_is_imported(monkeypatch):
    fake = types.ModuleType("Hexaflag")
    fake.__file__ = str(REPO / "Hexaflag.py")
    monkeypatch.setattr(gsource, "_LOADED", {})
    monkeypatch.setitem(sys.modules, "Hexaflag", fake)
    path = list(sys.path)
    with pytest.raises(gsource.GeodomeSourceError, match="already loaded from elsewhere"):
        gsource.load_geodome()
    assert sys.path == path


def test_geodome_resolves_inside_the_vendored_tree_only(sub):
    for name in gsource.MODULES:
        assert gsource._inside(sys.modules[name]), name
    assert str(gsource.VENDORED) not in sys.path
    assert not list(gsource.VENDORED.rglob("*.pyc"))          # no bytecode is written into the vendored tree


def test_the_frequency_120_identity_and_its_invariants(sub):
    record = sub.identity_record()
    for key in ("vertices", "edges", "arcs", "capacity", "depth", "mapping_sha256", "edge_sha256",
                "index_payload_bytes"):
        assert record[key] == gsource.IDENTITY[key], key
    assert record["degree_five_vertices"] == 12
    assert substrate() is sub                                     # built once per process, never per token


def test_addresses_coordinates_arcs_and_ports_round_trip_and_non_edges_are_refused(sub):
    for v in (0, 1, 71999, 144001):
        assert sub.resolve(sub.address(v)) == v and sub.resolve_coordinate(sub.coordinate(v)) == v
        for w in sub.neighbors(v):
            arc = sub.arc(v, w)
            assert sub.endpoints(arc) == (v, w) and sub.endpoints(arc ^ 1) == (w, v)
            assert sub.neighbors(v)[sub.port(v, w)] == w
    with pytest.raises(KeyError, match="NONEDGE"):
        sub.arc(0, _two_hop(sub, 0))
    with pytest.raises(KeyError):
        sub.port(0, _two_hop(sub, 0))
    with pytest.raises(ValueError):
        sub.endpoints(sub.arcs)


def test_a_logical_prefix_is_not_a_physical_region_and_no_edge_is_fabricated(sub):
    hxf = sys.modules["Hexaflag"]
    prefix = (3, hxf.DIGITS[1], hxf.DIGITS[0], hxf.DIGITS[0])
    members = sub.prefix_page(prefix, 0, 1024)
    assert len(members) == sub.prefix_count(prefix) == 420
    assert len(sub.components(members)) == 2                  # one logical group, two physical components
    inside = set(members)
    for arc in sub.physical_arcs(members):
        u, v = sub.endpoints(arc)
        assert u in inside and v in inside and v in sub.neighbors(u)


# -- P2: plans, working state, executor --------------------------------------------------------------------------

def _plan_cases(sub, path, far):
    arcs = sub.physical_arcs(path)
    base = dict(vertices=path, arcs=arcs, rounds=3, readout=(path[-1],), evidence=(path[0],))
    cases = {
        "ARC_AUTHORITY": dict(arcs=(*arcs, sub.arc(far[0], far[1]))),
        "ARC_RANGE": dict(arcs=(sub.arcs,)),
        "DUPLICATE_ARC": dict(arcs=(arcs[0], arcs[0])),
        "DUPLICATE_VERTEX": dict(vertices=(path[0], *path[:-1])),
        "VERTEX_RANGE": dict(vertices=(*path[:-1], sub.vertices)),
        "VERTEX_TYPE": dict(vertices=tuple(float(v) for v in path)),
        "NONRESIDENT": dict(vertices=(*path[:-1], far[0]), readout=(far[0],)),
        "ROUND_RANGE": dict(rounds=17),
        "VISIT_BUDGET": dict(max_visits=len(arcs)),
        "READOUT_BINDING": dict(readout=()),
        "BINDING_OVERLAP": dict(query=(path[0],)),
        "EVIDENCE_BINDING": dict(evidence=(path[0], path[1])),
        "START": dict(start=(path[0],)),
        "SCHEDULE": dict(schedule="eager"),
        "ACTIVE_BUDGET": dict(max_active=1025),
    }
    return {code: ExecutionPlan(**{**base, **change}) for code, change in cases.items()}


def test_every_plan_refusal_names_its_rule(sub):
    path, far = r13.chain(sub, 5, start=300), r13.chain(sub, 2, start=90000)
    working = TensorWorkingState(sub, capacity=8, width=3)
    working.acquire(path)
    limits = dict(max_rounds=16, max_arcs=1024, max_visits=16384, max_evidence=8, evidence_available=1)
    for code, plan in _plan_cases(sub, path, far).items():
        with pytest.raises(PlanError) as refused:
            compile_plan(plan, working, **limits)
        assert refused.value.code == code
    with pytest.raises(PlanError, match="PLAN_TYPE"):
        compile_plan(object(), working, **limits)
    with pytest.raises(PlanError, match="START"):
        compile_plan(ExecutionPlan(vertices=path, arcs=(), rounds=2, schedule="changed_destination",
                                   readout=(path[0],)), working, **limits)
    compiled = compile_plan(ExecutionPlan(vertices=path, arcs=tuple(reversed(sub.physical_arcs(path))), rounds=3,
                                          readout=(path[-1],)), working, **limits)
    assert compiled.arc_ids.tolist() == sorted(sub.physical_arcs(path))        # Geodome's aggregation order


def test_working_state_acquisition_is_all_or_none_and_bounded(sub):
    path = r13.chain(sub, 5, start=4000)
    working = TensorWorkingState(sub, capacity=4, width=3)
    working.acquire(path[:3])
    for vertices, error in (((path[3], path[4]), "STATE_CAPACITY"), ((path[3], path[0]), "ALREADY_RESIDENT"),
                            ((path[3], path[3]), "DUPLICATE_VERTEX"), ((sub.vertices,), "VERTEX_RANGE")):
        with pytest.raises((ValueError, OverflowError), match=error):
            working.acquire(vertices)
        assert working.resident == 3
    with pytest.raises(ValueError):
        TensorWorkingState(sub, capacity=4097, width=3)


def test_handles_are_owned_epoch_scoped_and_generation_checked(sub):
    path = r13.chain(sub, 3, start=5000)
    a, b = TensorWorkingState(sub, capacity=4, width=3), TensorWorkingState(sub, capacity=4, width=3)
    handles = a.acquire(path)
    b.acquire(path)
    with pytest.raises(ValueError, match="FOREIGN_OR_STALE_HANDLE"):
        b.validate(handles)                                       # another state's handle
    a.release(handles[:1])
    again = a.acquire(path[:1])
    with pytest.raises(ValueError, match="STALE_HANDLE"):
        a.validate(handles[:1])                                   # a reused slot's earlier generation
    a.reset()
    with pytest.raises(ValueError, match="FOREIGN_OR_STALE_HANDLE"):
        a.validate(again)                                         # an earlier epoch
    assert a.resident == 0 and not bool(a.latent.any())


def test_publish_validates_everything_first_and_keeps_values_only(sub):
    path = r13.chain(sub, 3, start=6000)
    working = TensorWorkingState(sub, capacity=4, width=3)
    handles = working.acquire(path)
    slots = working.slots(handles)
    rows = torch.randn(3, 3, generator=torch.Generator().manual_seed(0), requires_grad=True)
    before = working.latent.clone()
    infinite = rows.detach().index_fill(0, torch.tensor([1]), float("inf"))
    for bad, bad_slots in ((rows[:2], slots), (rows.double(), slots), (infinite, slots), (rows, slots.flip(0))):
        with pytest.raises(WorkingStateError):
            working.publish(handles, bad_slots, bad)
        assert torch.equal(working.latent, before)
    working.publish(handles, slots, rows * 2)
    assert torch.equal(working.gather(slots), (rows * 2).detach()) and working.latent.grad_fn is None


def test_volatile_objects_refuse_serialization_and_copies(sub, backbone):
    working = TensorWorkingState(sub, capacity=2, width=2)
    session = CognitiveSession(backbone, r13.kernels_for(backbone))
    for thing in (working, session):
        for fn in (pickle.dumps, copy.deepcopy, copy.copy):
            with pytest.raises(TypeError):
                fn(thing)
    working.close()
    with pytest.raises(WorkingStateError):
        working.acquire((0,))
    session.close()


@pytest.mark.parametrize("residency", ["sparse", "dense"])
def test_the_changed_destination_executor_is_geodomes_scheduler(sub, residency):
    kernels = r13.reference_kernels(width=4, hidden=8)
    region = r13.ball(sub, 0, 20)
    arcs = sub.physical_arcs(region)
    path = r13.chain(sub, 2, start=777)
    workloads = ((region, arcs, (0,), 6, 20, None, "ROUND_BUDGET"), (region, arcs, (0,), 6, 20, 20, "ARC_BUDGET"),
                 (region, arcs, region[:6], 4, 8, None, "ACTIVE_BUDGET"),
                 (path, (sub.arc(*path),), (path[1],), 4, 2, None, "QUIESCENT"))
    for index, (vertices, admitted, start, rounds, active, visits, status) in enumerate(workloads):
        init = r13.seeded_rows(len(vertices), 4, seed=index)
        visits = visits if visits is not None else rounds * len(admitted)
        mine, _ = r13.tensor_run(sub, kernels, vertices, admitted, init, schedule="changed_destination",
                                 rounds=rounds, start=start, max_active=active, max_visits=visits,
                                 residency=residency)
        ref = scheduler_reference(sub, kernels, vertices, admitted, init, start, rounds=rounds, max_active=active,
                                  max_visits=visits, mode=residency)
        assert mine.status == ref["status"] == status
        assert (mine.rounds, mine.visits) == (ref["rounds"], ref["visits"])
        assert [(r.status, r.sources, r.arcs, r.destinations, r.changed, r.touched)
                for r in mine.records] == ref["records"]
        assert float((mine.rows - ref["rows"]).abs().max()) <= r13.PARITY


def test_the_synchronous_schedule_equals_the_scheduler_only_where_they_coincide(sub):
    kernels = r13.reference_kernels(width=4, hidden=8)
    region = r13.ball(sub, 777, 24)
    arcs = sub.physical_arcs(region)
    init = r13.seeded_rows(len(region), 4, seed=2)
    sync, compiled = r13.tensor_run(sub, kernels, region, arcs, init, schedule="synchronous", rounds=4)
    ref = scheduler_reference(sub, kernels, region, arcs, init, region, rounds=4, max_active=len(region),
                              max_visits=4 * len(arcs))
    assert bool(compiled.incoming.all()) and all(r[3] == r[4] == len(region) for r in ref["records"][:4])
    assert float((sync.rows - ref["rows"]).abs().max()) <= r13.PARITY
    path = r13.chain(sub, 6, start=777)
    forward = tuple(sub.arc(path[i], path[i + 1]) for i in range(5))
    init = r13.seeded_rows(6, 4, seed=3)
    a, _ = r13.tensor_run(sub, kernels, path, forward, init, schedule="synchronous", rounds=4)
    b, _ = r13.tensor_run(sub, kernels, path, forward, init, schedule="changed_destination", rounds=4,
                          start=(path[0],), max_active=6, max_visits=20)
    assert not torch.allclose(a.rows, b.rows)                 # different policies: never claimed equal


def test_information_moves_at_most_one_physical_hop_per_round(sub):
    kernels = r13.reference_kernels(width=4, hidden=8)
    path = r13.chain(sub, 7, start=1234)
    forward = tuple(sub.arc(path[i], path[i + 1]) for i in range(6))
    base = r13.seeded_rows(7, 4, seed=5)
    bumped = base.clone()
    bumped[0] += 0.25
    for rounds in (1, 2, 5):
        a, _ = r13.tensor_run(sub, kernels, path, forward, base, schedule="synchronous", rounds=rounds)
        b, _ = r13.tensor_run(sub, kernels, path, forward, bumped, schedule="synchronous", rounds=rounds)
        moved = [i for i in range(1, 7) if not torch.equal(a.rows[i], b.rows[i])]
        assert moved == list(range(1, rounds + 1)), rounds


# -- gradients -----------------------------------------------------------------------------------------------------

def test_float64_gradcheck_of_the_executor_and_readout(sub):
    small = r13.reference_kernels(width=2, hidden=3)
    tiny = r13.chain(sub, 4, start=31)
    working = TensorWorkingState(sub, capacity=4, width=2, dtype=torch.float64)
    working.acquire(tiny)
    compiled = compile_plan(ExecutionPlan(vertices=tiny, arcs=sub.physical_arcs(tiny), rounds=3, readout=(tiny[-1],)),
                            working, max_rounds=8, max_arcs=64, max_visits=1024, max_evidence=1)
    names = ("message_in.weight", "port_out", "port_in", "update_gate.weight", "update_cand.weight",
             "readout_proj.weight")
    params = dict(small.named_parameters())
    fn = r13.substituted(small, names, lambda rows: small.readout(
        r13.run_synchronous(rows, compiled, small).rows.index_select(0, compiled.readout)))
    inputs = (r13.seeded_rows(4, 2, seed=4).requires_grad_(True),
              *(params[n].detach().clone().requires_grad_(True) for n in names))
    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-8, rtol=1e-6)


def test_gradients_reach_every_kernel_over_the_hops_and_stop_at_the_hop_bound(sub, backbone):
    tokens = r13.stream_tokens(600)
    path = r13.chain(sub, 6)
    arcs = sub.physical_arcs(path)
    for rounds, reaches in ((6, True), (4, False)):              # evidence at path[0], readout 5 hops away
        kernels = r13.kernels_for(backbone, gate=0.5)
        session = r13._prepared(backbone, kernels, tokens, path)
        plan = ExecutionPlan(vertices=path, arcs=arcs, rounds=rounds, evidence=(path[0],), query=(path[2],),
                             readout=(path[-1],))
        with torch.enable_grad():
            session.reason(plan)
            torch.nn.functional.cross_entropy(session.readout().logits, torch.tensor([3])).backward()
        if reaches:
            for name, p in kernels.named_parameters():
                assert p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.any()), name
        else:
            grad = kernels.evidence_in.weight.grad
            assert grad is None or not bool(grad.any())
            assert bool(kernels.query_in.weight.grad.any())        # the query is 3 hops away: reached
        session.close()


def test_a_zero_gate_starts_exactly_at_the_backbone_and_only_the_gate_learns_first(sub, backbone):
    tokens = r13.stream_tokens(600)
    path = r13.chain(sub, 5)
    kernels = r13.kernels_for(backbone, gate=0.0)
    session = r13._prepared(backbone, kernels, tokens, path)
    plan = ExecutionPlan(vertices=path, arcs=sub.physical_arcs(path), rounds=4, evidence=(path[0],),
                         query=(path[2],), readout=(path[-1],))
    with torch.enable_grad():
        session.reason(plan)
        read = session.readout()
        torch.nn.functional.cross_entropy(read.logits, torch.tensor([3])).backward()
    assert torch.equal(read.logits.detach(), session._observation.logits)
    learning = sorted(n for n, p in kernels.named_parameters() if p.grad is not None and bool(p.grad.any()))
    assert learning == ["gate"]
    session.close()


def test_one_reason_call_is_one_bounded_graph_and_the_cognitive_path_never_reads_the_host(sub, backbone):
    tokens = r13.stream_tokens(600)
    path = r13.chain(sub, 5)
    kernels = r13.kernels_for(backbone, gate=0.5)
    session = r13._prepared(backbone, kernels, tokens, path)
    plan = ExecutionPlan(vertices=path, arcs=sub.physical_arcs(path), rounds=4, evidence=(path[0],),
                         query=(path[2],), readout=(path[-1],))
    sizes = []
    with torch.enable_grad():
        for _ in range(3):
            session.reason(plan)
            sizes.append(r13.graph_size(session._reasoning.latent))
    assert len(set(sizes)) == 1 and session.working.latent.grad_fn is None
    compiled = compile_plan(plan, session.working, max_rounds=16, max_arcs=1024, max_visits=16384, max_evidence=8,
                            evidence_available=1)
    rows = session.working.gather(compiled.slots)
    with torch.enable_grad(), r13.HostExtraction() as watch:
        _, latent = session.compute(compiled, rows, session._observation, session._evidence)
        kernels.injection(latent)
    assert watch.calls == []
    with r13.HostExtraction() as control:                          # the detector does see an extraction
        latent.sum().item()
    assert control.calls == ["item"]
    session.observe(tokens[:, 560:600])
    assert session._reasoning is None                             # an external step drops every result
    session.close()


def test_end_to_end_float64_gradcheck_through_the_actual_final_channel(sub, backbone):
    tokens = r13.stream_tokens(600)
    path = r13.chain(sub, 4, start=2024)
    kernels = r13.kernels_for(backbone, width=3, hidden=4, gate=0.5)
    session = r13._prepared(backbone, kernels, tokens, path)
    compiled = compile_plan(ExecutionPlan(vertices=path, arcs=sub.physical_arcs(path), rounds=3,
                                          evidence=(path[0],), query=(path[1],), readout=(path[-1],)),
                            session.working, max_rounds=16, max_arcs=1024, max_visits=16384, max_evidence=8,
                            evidence_available=1)
    rows = session.working.gather(compiled.slots)
    projection = torch.randn(backbone.config.vocab_size, 2, generator=torch.Generator().manual_seed(8),
                             dtype=torch.float64)
    names = ("query_in.weight", "evidence_in.weight", "port_out", "readout_proj.weight", "inject.weight", "gate")

    def end_to_end():
        _, latent = session.compute(compiled, rows, session._observation, session._evidence)
        return session.refine(latent, session._observation) @ projection
    params = dict(kernels.named_parameters())
    inputs = tuple(params[n].detach().clone().requires_grad_(True) for n in names)
    assert torch.autograd.gradcheck(r13.substituted(kernels, names, end_to_end), inputs, eps=1e-6, atol=1e-8,
                                    rtol=1e-6)
    session.close()


# -- P3/P4: the bridge on actual EvidenceLM --------------------------------------------------------------------------

@pytest.mark.parametrize("variant", [r13.VARIANTS[i] for i in (0, 6, 13, 16, 17)],
                         ids=lambda v: "-".join(f"{k}={v[k]}" for k in ("mode", "fine_keys", "ple", "origin", "window")))
def test_the_bridge_on_actual_evidence_lm(variant):
    out = r13.run_variant(variant)
    assert out["pass"], out
    assert out["ring"]["ok_wrapped"] and out["ring"]["ok_evicted"]
    assert out["context_clock"]["counter_deltas"] == {"blocks_reencoded": 2, "ring_rereads": 2}


def test_every_refusal_is_coded_and_changes_nothing():
    battery = r13.refusal_battery()
    assert r13._gate(battery) == "PASS", battery
    for name, result in battery.items():
        assert result.get("refused", True) and result.get("ok_code", True), (name, result)


def test_sparse_and_dense_residency_give_the_same_bridge_result():
    assert r13.residency_bridge() == {"equal_latent": True, "equal_logits": True}


def test_training_the_cognitive_kernels_never_writes_the_backbone(sub, backbone):
    digest = parameter_digest(backbone)
    path = r13.chain(sub, 5)
    kernels = r13.kernels_for(backbone, gate=0.5)
    session = r13._prepared(backbone, kernels, r13.stream_tokens(600), path)
    optimizer = torch.optim.SGD(kernels.parameters(), lr=0.1)
    plan = ExecutionPlan(vertices=path, arcs=sub.physical_arcs(path), rounds=4, evidence=(path[0],),
                         query=(path[2],), readout=(path[-1],))
    for step in range(2):
        with torch.enable_grad():
            session.reason(plan)
            loss = torch.nn.functional.cross_entropy(session.readout().logits, torch.tensor([step + 1]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert parameter_digest(backbone) == digest and all(p.grad is None for p in backbone.parameters())
    session.close()


def test_a_session_refuses_an_unfrozen_or_mismatched_backbone_and_a_budget_overrun(backbone):
    kernels = r13.kernels_for(backbone)
    unfrozen = r13.build_backbone()
    unfrozen.blocks[0].mix_norm.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        CognitiveSession(unfrozen, kernels)
    with pytest.raises(ValueError, match="width and dtype"):
        CognitiveSession(backbone, r13.kernels_for(r13.build_backbone(dtype="float32")))
    with pytest.raises(ValueError, match="evidence_options"):
        CognitiveSession(backbone, kernels, evidence_options={"teacher": True})
    with pytest.raises(BudgetError):
        CognitiveSession(backbone, kernels, budget_bytes=1)
    session = CognitiveSession(backbone, kernels)
    with pytest.raises(CognitiveError, match="NO_OBSERVATION"):
        session.admit_evidence([])
    session.close()
