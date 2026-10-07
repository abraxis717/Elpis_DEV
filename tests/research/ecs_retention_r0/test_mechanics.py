"""Retention R0 laboratory mechanics (no DEV or QUAL data: test-* worlds only).

The specification is complete and pre-registered; the S3 identity behind the
candidates holds against the canonical native kernel; candidate gradients are
the derivatives of their declared penalties; the candidate engine without
consolidation is bitwise the canonical core; consolidation never receives
targets; the selection rule and the write-once protocol behave as specified.
"""
from __future__ import annotations

import ctypes
import json

import numpy as np
import pytest

from elpis.ECS.native import ECSGLibrary

from research.ecs_retention_r0 import engine as E
from research.ecs_retention_r0 import experiment as X
from research.ecs_retention_r0 import protocol as P
from research.ecs_retention_r0 import task as T

from .._renamed_sources import renamed_lab_binding
from ...ECS.test_math_r0 import _library_path

SPEC = P.load_spec()
IDX = T.indices(SPEC["regime"]["dim"])


@pytest.fixture(scope="module")
def api():
    return ECSGLibrary(ctypes.CDLL(str(_library_path())))


@pytest.fixture(scope="module")
def world():
    return T.build_world(SPEC, "test-0000", 0.5)



@pytest.fixture(autouse=True)
def _renamed_binding(monkeypatch):
    """The frozen laboratory binds the pre-rename layout; point it at the renamed tree in memory only."""
    for attr, value in renamed_lab_binding(P).items():
        monkeypatch.setattr(P, attr, value)


def _small_spec(steps=40):
    spec = json.loads(json.dumps(SPEC))
    spec["regime"]["steps"] = steps
    return spec


def test_specification_is_complete_before_dev():
    for key in ("regime", "tasks", "offset_grid", "splits", "mechanisms", "metrics", "dev_rules", "pass_rule",
                "mechanics_checks", "secondary", "binding_requirements", "labels"):
        assert key in SPEC, key
    assert set(SPEC["pass_rule"]) == {"A_acquisition", "B_acquisition", "C_retention", "D_joint_state",
                                      "E_state_causality", "F_no_external_answer_store", "G_determinism",
                                      "H_negative_baseline", "I_control_honesty", "J_capacity_accounting",
                                      "disposition"}
    assert SPEC["splits"]["qual_worlds"] >= 3 * 8 and SPEC["splits"]["dev_worlds"] >= 4
    dev, qual = P.world_ids(SPEC, "DEV"), P.world_ids(SPEC, "QUAL")
    assert not set(dev) & set(qual) and all(w.startswith("dev-") for w in dev)
    assert SPEC["mechanisms"]["M1"]["eligible_for_selection"] is False
    assert SPEC["regime"]["steps"] == 4000 and SPEC["regime"]["learning_rate"] == 0.002
    assert "SEMANTICS=NONE" in SPEC["labels"] and "NO_LANGUAGE_CLAIM" in SPEC["labels"]
    digests = P.spec_digests(SPEC)
    assert digests == P.spec_digests(P.load_spec()) and len(digests["candidates_sha256"]) == 64


def test_the_response_is_one_half_phi_dot_s3_exactly(api, world):
    for W in (world.w0, world.w0 * 1.7 + 0.01):
        x = world.tasks["A"].x_test
        native = E.responses(api, W, x)
        coarse = 0.5 * T.features(x, IDX) @ E.s3(W, IDX)
        assert np.max(np.abs(native - coarse)) <= 1e-12 * max(1.0, np.max(np.abs(native)))
        assert np.max(np.abs(E.s3(W, IDX) - E.native_s3(api, W))) <= 1e-12
    assert IDX.size == 83 and len(IDX.pairs) == 21 and len(IDX.triples) == 56


def test_jacobian_and_vector_jacobian_product_are_the_derivatives_of_s3(world):
    W, h = world.w0, 1e-6
    J = E.s3_jacobian(W, IDX)
    rng = np.random.default_rng(3)
    for k in rng.choice(W.size, 12, replace=False):
        plus, minus = W.copy(), W.copy()
        plus.reshape(-1)[k] += h
        minus.reshape(-1)[k] -= h
        assert np.allclose((E.s3(plus, IDX) - E.s3(minus, IDX)) / (2 * h), J[:, k], atol=1e-8)
    u = rng.normal(size=IDX.size)
    assert np.allclose(E.s3_vjp(W, u, IDX).reshape(-1), J.T @ u, atol=1e-12)


def test_c1_penalty_is_the_function_change_over_consolidated_inputs(api, world):
    W_A = world.w0
    W = W_A + np.random.default_rng(5).normal(0.0, 0.02, size=W_A.shape)
    a = world.tasks["A"]
    c1 = E.C1(IDX, 0.7)
    c1.consolidate(W_A, a.x_train)
    direct = 0.7 * float(np.mean((E.responses(api, W, a.x_train) - E.responses(api, W_A, a.x_train)) ** 2))
    assert abs(c1.penalty(W, E.s3(W_A, IDX)) - direct) <= 1e-10 * max(1.0, direct)
    h, g = 1e-6, c1.grad(W)
    for k in (0, 17, 101, 200):
        plus, minus = W.copy(), W.copy()
        plus.reshape(-1)[k] += h
        minus.reshape(-1)[k] -= h
        fd = (c1.penalty(plus, E.s3(W_A, IDX)) - c1.penalty(minus, E.s3(W_A, IDX))) / (2 * h)
        assert abs(fd - g.reshape(-1)[k]) <= 1e-6 * max(1.0, abs(fd))


def test_c2_importance_is_the_diagonal_of_the_same_curvature(world):
    W_A, a = world.w0, world.tasks["A"]
    c2 = E.C2(IDX, 3.0, W_A.shape)
    c2.consolidate(W_A, a.x_train)
    J, sigma = E.s3_jacobian(W_A, IDX), E.input_statistics(a.x_train, IDX)
    assert np.allclose(c2.Omega.reshape(-1), 0.25 * np.diag(J.T @ sigma @ J), rtol=1e-12, atol=1e-14)
    W = W_A + 0.01
    penalty = lambda V: 3.0 * float(np.sum(c2.Omega * (V - W_A) ** 2))  # noqa: E731
    h = 1e-6
    for k in (3, 77, 150):
        plus, minus = W.copy(), W.copy()
        plus.reshape(-1)[k] += h
        minus.reshape(-1)[k] -= h
        assert abs((penalty(plus) - penalty(minus)) / (2 * h) - c2.grad(W).reshape(-1)[k]) <= 1e-7
    assert c2.state_bytes() == 2 * 8 * W_A.size and E.C1(IDX, 1.0).state_bytes() == 28552


def test_engine_without_consolidation_is_bitwise_the_canonical_core(api, world):
    a = world.tasks["A"]
    canonical = E.canonical_learn(api, world.w0, a.x_train, a.y_train, 0.002, 30)
    engine, refused, _ = E.engine_learn(api, world.w0, a.x_train, a.y_train, 0.002, 30)
    assert refused == 0 and np.array_equal(canonical, engine)


def test_a_refused_candidate_learn_leaves_the_start_state(api, world):
    b = world.tasks["B"]
    c1 = E.C1(IDX, 1e12)
    c1.consolidate(world.w0 + 0.3, world.tasks["A"].x_train)
    W, refused, _ = E.engine_learn(api, world.w0, b.x_train, b.y_train, 0.002, 50, c1)
    assert refused > 0 and np.array_equal(W, world.w0)


def test_consolidation_never_receives_targets():
    inputs = X.consolidation_inputs()
    assert inputs == {"C1": {"parameters": ["W", "inputs"], "receives_targets": False},
                      "C2": {"parameters": ["W", "inputs"], "receives_targets": False}}


def test_worlds_are_deterministic_and_offsets_shift_the_same_draws():
    a, b = T.build_world(SPEC, "test-0001", 0.5), T.build_world(SPEC, "test-0001", 0.5)
    assert np.array_equal(a.w0, b.w0) and all(np.array_equal(a.tasks[t].y_test, b.tasks[t].y_test) for t in T.TASKS)
    far = T.build_world(SPEC, "test-0001", 1.5)
    shift = far.tasks["B"].x_train - a.tasks["B"].x_train
    assert np.allclose(shift[:, 0], -1.0) and np.allclose(shift[:, 1:], 0.0)
    assert np.array_equal(far.w0, a.w0)
    assert not np.array_equal(T.build_world(SPEC, "test-0002", 0.5).w0, a.w0)


def test_the_pipeline_runs_end_to_end_on_a_test_world(api):
    lab = X.Lab(api, _small_spec(40))
    world = lab.world("test-0003", 0.5)
    results = {m: lab.run(world, m) for m in ("M0", "M1", "ENGINE0")}
    results["C1"] = lab.run(world, "C1", 1.0)
    results["C2"] = lab.run(world, "C2", 4.0)
    results["mismatched"] = lab.run(world, "C1", 1.0, "mismatched")
    results["uninformed"] = lab.run(world, "C2", 4.0, "uninformed")
    metrics = {k: lab.metrics(world, r) for k, r in results.items()}
    assert np.array_equal(results["ENGINE0"]["W"], results["M0"]["W"])
    assert metrics["M1"]["extra_state_bytes"] == 64 * 7 * 8 and metrics["M0"]["extra_state_bytes"] == 0
    assert metrics["C1"]["extra_state_bytes"] == 28552 and metrics["C2"]["extra_state_bytes"] == 3456
    for m in metrics.values():
        assert {"LEARNED_A", "LEARNED_B", "RETAINED_A", "retention_ratio", "W_digest"} <= set(m)
    assert lab.causality(world, results["C1"]["W"]) == {"transplant_bitwise": True, "reset_bitwise": True}
    assert lab.pte_check(world, "C1", 1.0)["pass"] and lab.pte_check(world, "C2", 4.0)["pass"]
    assert lab.feature_identity(world) <= 1e-10
    seq = lab.sequence(world, "C1", 1.0)
    assert set(seq["after_stage"]) == set(T.TASKS) and 0 <= seq["held_count"] <= 4
    assert set(lab.ceiling(world)) == {"nmse_A", "nmse_B"}


def test_selection_rule_is_the_registered_one():
    def row(learned, retained, a, b, state):
        return {"LEARNED_B": learned, "RETAINED_A": retained, "nmse_A_after_B": a, "nmse_B_after": b,
                "extra_state_bytes": state}
    table = {}
    for fam, lam in X.configurations(SPEC):
        table[f"{fam}:{lam}"] = {"w0": row(False, False, 1.0, 1.0, 8), "w1": row(False, False, 1.0, 1.0, 8)}
    table["C2:16.0"] = {"w0": row(True, True, 0.2, 0.2, 3456), "w1": row(True, False, 0.6, 0.2, 3456)}
    table["C1:2.0"] = {"w0": row(True, True, 0.3, 0.3, 28552), "w1": row(True, False, 0.6, 0.2, 28552)}
    selected, secondary, ranking = X.select(table, SPEC)
    assert selected == {"family": "C2", "lambda": 16.0}       # same count, smaller median joint nmse
    assert secondary == {"family": "C1", "lambda": 2.0}
    assert all(r["family"] != "M1" for r in ranking)


def test_evidence_is_written_once(tmp_path):
    path = tmp_path / "x.json"
    d = P.write(path, "dev", {"a": 1.5})
    assert P.load(path, "dev")["digest"] == d == P.digest("dev", {"a": 1.5})
    with pytest.raises(P.ProtocolError):
        P.write(path, "dev", {"a": 2})
    with pytest.raises(P.ProtocolError):
        P.canonical_json({"x": float("nan")})


def test_implementation_binding_names_every_required_field():
    impl = P.implementation(_library_path())
    assert set(impl["files"]) == set(P.BOUND_FILES) and len(P.BOUND_FILES) == 8
    assert impl["library"]["sha256"] and impl["build"]["compiler"] and impl["build"]["cmake"]["CMAKE_BUILD_TYPE"]
    assert impl["numerical_profile"]["numpy"] and impl["lab_source_digest"] == P.source_digest()
    assert P.binding_mismatch(impl, impl) == []


def test_the_freeze_record_keeps_every_digest_and_the_dev_rerun_must_reproduce(tmp_path, monkeypatch):
    from research.ecs_retention_r0 import run as R
    impl = P.implementation(_library_path())
    body = {"experiment": R.NAME, "split": "DEV", "worlds": ["dev-0000"], **P.spec_digests(SPEC),
            "implementation": impl, "choices": {"offset": 1.0, "selected": {"family": "C1", "lambda": 4.0},
                                                "secondary": {"family": "C2", "lambda": 16.0}},
            "candidate_table": {"C1:4.0": {"dev-0000": {"learn_seconds": 1.0, "nmse_A_after_B": 0.1}}}}
    monkeypatch.setattr(R, "DEV_PATH", tmp_path / "dev.json")
    monkeypatch.setattr(R, "DEV_RERUN_PATH", tmp_path / "dev.r2.json")
    monkeypatch.setattr(R, "FROZEN_PATH", tmp_path / "frozen.json")
    monkeypatch.setattr(R, "source_digest_at", lambda commit: impl["lab_source_digest"])
    P.write(R.DEV_PATH, "dev", body)
    R.freeze("0" * 40, _library_path())
    frozen = P.load(R.FROZEN_PATH, "frozen")["body"]
    digests = P.spec_digests(SPEC)
    assert all(frozen[k] == digests[k] for k in ("spec", "pass_rule", "candidates_sha256"))
    assert frozen["pass_rule_text"] == SPEC["pass_rule"] and frozen["dev_records"]
    # Re-execution comparison ignores only the implementation binding and wall-clock timings.
    other = json.loads(json.dumps(body))
    other["candidate_table"]["C1:4.0"]["dev-0000"]["learn_seconds"] = 9.0
    other["implementation"] = {"anything": 1}
    assert R._results(other) == R._results(body)
    other["candidate_table"]["C1:4.0"]["dev-0000"]["nmse_A_after_B"] = 0.2
    assert R._results(other) != R._results(body)
