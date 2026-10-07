"""Retention R2 laboratory mechanics (RET2B). No DEV or QUAL world is evaluated: test-* worlds only.

The laboratory implements the write-once specification exactly: the registered constants; the S3 identity and
calculus behind the candidates (feature identity, Jacobian, vector-Jacobian product, Hessian contraction, Omega
gradient); every correction is the derivative of its declared quantity; K2's projected step has no protected
component; K3's reconditioning preserves S3 and every response; the engine without its mechanism is bitwise the
canonical core; W_A is shared; the witness is exact; the PTE microstates diverge; a refused or non-finite
experience leaves the complete state unchanged; consolidation receives no target; the declared state is fixed in
size, serializes deterministically and restores exactly; the numerical environment is effectively single
threaded; the selection rule, classes, task rule and disposition behave as registered.
"""
from __future__ import annotations

import ast
import ctypes
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from elpis.ECS.native import ECSGLibrary

from research.ecs_retention_r2 import engine as E
from research.ecs_retention_r2 import experiment as X
from research.ecs_retention_r2 import protocol as P
from research.ecs_retention_r2 import task as T

from .._renamed_sources import current_path, renamed_lab_binding
from ...ECS.test_math_r0 import REPO, _library_path

SPEC = P.load_spec()
IDX = T.indices(SPEC["regime"]["dim"])
LAB_DIR = REPO / "research" / "ecs_retention_r2"



@pytest.fixture(autouse=True)
def _renamed_binding(monkeypatch):
    """The frozen laboratory binds the pre-rename layout; point it at the renamed tree in memory only."""
    for attr, value in renamed_lab_binding(P).items():
        monkeypatch.setattr(P, attr, value)


def _short_spec(steps=40) -> dict:
    spec = json.loads(json.dumps(SPEC))
    spec["regime"]["steps_per_experience"] = steps
    return spec


@pytest.fixture(scope="module")
def api():
    return ECSGLibrary(ctypes.CDLL(str(_library_path())))


@pytest.fixture(scope="module")
def lab(api):
    return X.Lab(api, _short_spec())


@pytest.fixture(scope="module")
def world():
    return T.build_world(SPEC, "test-0000", SPEC["task"]["hidden_gain_grid"][0])


def _fd(fun, W, h=1e-6, every=5):
    g, mask = np.zeros_like(W), np.zeros_like(W, dtype=bool)
    for e in range(W.shape[0]):
        for i in range(0, W.shape[1], every):
            wp, wm = W.copy(), W.copy()
            wp[e, i] += h
            wm[e, i] -= h
            g[e, i], mask[e, i] = (fun(wp) - fun(wm)) / (2 * h), True
    return g, mask


def test_registered_constants_are_the_specification():
    k2, k3 = SPEC["mechanisms"]["K2"]["hyperparameters"], SPEC["mechanisms"]["K3"]["procedure"]
    assert E.PROTECTED_CUTOFF == k2["protected_eigenvalue_cutoff"] and E.GRAM_CUTOFF == k2["gram_pseudoinverse_cutoff"]
    assert E.FIBRE_MAX_ITERATIONS == k3["max_iterations"] and E.FIBRE_INITIAL_STEP == k3["initial_relative_step"]
    assert E.FIBRE_MAX_CONSECUTIVE_REJECTIONS == k3["stop_after_consecutive_rejections"]
    assert "at most 10 iterations" in k3["retraction"] and E.RETRACTION_MAX_ITERATIONS == 10
    assert "1e-13" in k3["retraction"] and E.RETRACTION_TOLERANCE == 1e-13
    assert E.make("C1R", IDX).lam == SPEC["mechanisms"]["C1R"]["hyperparameters"]["lambda"] == 4.0
    assert E.make("K1", IDX).constants()["lambda"] == SPEC["mechanisms"]["K1"]["hyperparameters"]["lambda"] == 1.0
    assert IDX.size == SPEC["regime"]["features"] == 83
    assert X.CANDIDATES == ("K1", "K2", "K3") and SPEC["task"]["hidden_gain_grid"] == [2.0, 2.5, 1.5]


def test_feature_identity_and_native_s3(api, world):
    for W in (world.w0, world.w0 * 1.7 + 0.01, world.witness):
        x = world.tasks["B"].x_test
        native = E.responses(api, W, x)
        coarse = 0.5 * T.features(x, IDX) @ E.s3(W, IDX)
        assert np.max(np.abs(native - coarse)) <= 1e-12 * max(1.0, np.max(np.abs(native)))
        assert np.max(np.abs(E.s3(W, IDX) - E.native_s3(api, W))) <= 1e-12 * max(1.0, np.max(np.abs(E.s3(W, IDX))))


def test_s3_calculus_matches_finite_differences(world):
    W = world.w0
    rng = np.random.default_rng(5)
    u = rng.normal(size=IDX.size)
    g, m = _fd(lambda w: float(u @ E.s3(w, IDX)), W)
    assert np.max(np.abs(E.s3_vjp(W, u, IDX)[m] - g[m])) <= 1e-7 * np.max(np.abs(g[m]))
    J = E.jacobian(W, IDX)
    assert np.allclose(J.T @ u, E.s3_vjp(W, u, IDX).reshape(-1), rtol=1e-12, atol=1e-12)
    G = rng.normal(size=(IDX.size,) + W.shape)
    g, m = _fd(lambda w: float(np.sum(G * E.jacobian3(w, IDX))), W)
    assert np.max(np.abs(E.hessian_vjp(W, G, IDX)[m] - g[m])) <= 1e-7 * np.max(np.abs(g[m]))


def test_corrections_are_derivatives_of_their_declared_quantities(api, lab, world):
    gi = lab.gradient_identity(world)
    assert gi["pass"], gi
    assert max(gi["K1"], gi["C1R"], gi["K3"]) <= 1e-6 and gi["K2"] <= 1e-10


def test_k1_is_c1_at_lambda_one_for_one_experience(lab, world):
    """For the pair A -> B, K1 equals R0's C1 law at lambda 1 (up to the exact symmetrization of Sigma)."""
    WA = lab.w_a(world)
    k1, c1 = E.make("K1", IDX), E.make("C1R", IDX, lam=1.0)
    k1.consolidate(WA, world.tasks["A"].x_train)
    c1.consolidate(WA, world.tasks["A"].x_train)
    W = WA + 0.01 * np.random.default_rng(1).normal(size=WA.shape)
    assert np.allclose(k1.grad(W), c1.grad(W), rtol=1e-10, atol=1e-13)


def test_k3_reconditioning_preserves_function_and_moves_the_microstate(api, lab, world):
    WA = lab.w_a(world)
    k3 = E.make("K3", IDX)
    W2, info = lab._consolidate(world, "A", WA, k3)[0:3:2]
    assert info["iterations"] >= 1 and info["omega_after"] <= info["omega_before"]
    assert info["s3_relative"] <= SPEC["thresholds"]["fibre_s3_relative"]
    assert info["response_relative"] <= SPEC["thresholds"]["fibre_response_relative"]
    assert np.array_equal(k3.a, E.s3(W2, IDX)) and info["protected_dimension"] == 19   # cubics on a 3-plane
    assert info["cold_ops"] > 0


def test_engine_without_mechanism_is_the_canonical_core_and_w_a_is_shared(lab, world):
    m0 = lab.run_mechanism(world, "M0")
    for kind in X.ENGINE_KINDS:
        removed = lab.run_mechanism(world, kind, "removed")
        assert all(removed["stages"][s]["W_digest"] == m0["stages"][s]["W_digest"] for s in X.STAGES), kind
        live = lab.run_mechanism(world, kind)
        assert live["W_A_digest"] == m0["stages"]["A"]["W_digest"], kind


def test_k2_with_nothing_protected_is_canonical_and_projects_out_protected_change(api, lab, world):
    WA = lab.w_a(world)
    b = world.tasks["B"]
    plain = E.engine_learn(api, WA, b.x_train, b.y_train, 0.002, 3, E.make("K1", IDX, removed=True))[0]
    empty = E.make("K2", IDX)                      # H = 0: no protected direction
    assert np.array_equal(E.engine_learn(api, WA, b.x_train, b.y_train, 0.002, 3, empty)[0], plain)
    k2 = E.make("K2", IDX)
    k2.consolidate(WA, world.tasks["A"].x_train)
    assert k2.U.shape[1] == 19 and np.allclose(k2.U.T @ k2.U, np.eye(19), atol=1e-12)   # A's 3-plane
    stepped = E.engine_learn(api, WA, b.x_train, b.y_train, 0.002, 1, E.make("K1", IDX, removed=True))[0]
    after = k2.correct(WA, stepped, 0.002)
    C = k2.U.T @ E.jacobian(WA, IDX)
    assert np.linalg.norm(C @ (after - WA).reshape(-1)) <= 1e-10 * np.linalg.norm(C @ (stepped - WA).reshape(-1))


def test_witness_is_exact_and_the_task_has_its_registered_structure(api, lab, world):
    assert max(lab.witness(world).values()) <= SPEC["thresholds"]["witness_max_nmse"]
    Q = world.basis
    assert np.allclose(Q.T @ Q, np.eye(6), atol=1e-12)
    c = s = np.sqrt(0.5)
    assert np.allclose(world.planes["B"][:, 0], c * Q[:, 0] + s * Q[:, 3], atol=1e-15)
    W = world.w0
    for k, t in enumerate("ABCD"):
        M, exp = world.planes[t], world.tasks[t]
        assert np.allclose(M.T @ M, np.eye(3), atol=1e-12)
        assert np.allclose(exp.x_train - exp.x_train @ M @ M.T, 0.0, atol=1e-12)          # inputs on the plane
        # structural novelty: each later hidden ridge vanishes on every earlier plane
        for later in range(k + 1, 4):
            assert np.allclose(exp.x_test @ Q[:, 2 + later], 0.0, atol=1e-12), (t, later)
        # every G1 entity update lies in the current plane
        stepped = E.engine_learn(api, W, exp.x_train, exp.y_train, 0.002, 1, E.make("K1", IDX, removed=True))[0]
        delta = stepped - W
        assert np.linalg.norm(delta - M @ (M.T @ delta)) <= 1e-12 * np.linalg.norm(delta), t
    # aliasing: on B's plane a ridge along q4 and one along q1 are the same function; on A's they are not
    xb, xa = world.tasks["B"].x_test, world.tasks["A"].x_test
    r4, r1 = 2.0 * Q[:, 3:4], 2.0 * Q[:, 0:1]
    assert np.allclose(T.cubic(r4, xb), T.cubic(r1, xb), atol=1e-12)
    assert not np.allclose(T.cubic(r4, xa), T.cubic(r1, xa))


def test_ceilings_novelty_and_capacity_are_computed_as_registered(lab, world):
    ceil = lab.ceilings(world)
    assert set(ceil["previous"]) == {"B", "C", "D"} and ceil["capacity_max"] <= 1e-6
    assert all(ceil["stage"][s][t] <= 1e-6 for i, s in enumerate("ABCD") for t in "ABCD"[:i + 1])


def test_pte_microstates_diverge_for_every_candidate(lab, world):
    """Along B's aliased axis: identical S3, and a B step reads exactly the coordinate where they differ."""
    p, q = T.pte_pair(world.w0, world.pte_axis)
    sp, sq = E.s3(p, IDX), E.s3(q, IDX)
    assert np.max(np.abs(sp - sq)) <= 1e-12 * max(1.0, np.max(np.abs(sp))) and np.any(p != q)
    for kind in X.CANDIDATES:
        assert lab.pte_check(world, kind)["pass"], kind


def test_refusal_leaves_the_complete_state_unchanged(api, lab, world):
    WA = lab.w_a(world)
    b = world.tasks["B"]
    k1 = E.make("K1", IDX)
    k1.consolidate(WA, world.tasks["A"].x_train)
    before = E.state_digest(k1, WA, 0)
    W, refused, _ = E.engine_learn(api, WA, b.x_train, b.y_train, 1e6, 5, k1)   # diverges: non-finite or refusal
    assert refused > 0 and np.array_equal(W, WA) and E.state_digest(k1, WA, 0) == before
    blown = lab.world("test-0001", 2.0)
    bad = X.Lab(lab.api, _short_spec())
    bad.rate = 1e6
    run = bad.candidate_run(blown, "K1")
    for s in ("B", "C", "D"):
        assert run["stages"][s]["refused"] != 0
        assert run["stages"][s]["state_digest"] == run["stages"]["A"]["state_digest"]


def test_consolidation_receives_no_target():
    interface = X.consolidation_interface()
    assert all(v["parameters"] == ["W", "inputs", "rng"] and not v["receives_targets"] for v in interface.values())


def test_declared_state_is_fixed_size_and_serializes_exactly(lab, world):
    F = IDX.size
    declared = {"K1": SPEC["mechanisms"]["K1"]["state"]["extra_persistent_bytes"],
                "K2": SPEC["mechanisms"]["K2"]["state"]["extra_persistent_bytes"],
                "K3": SPEC["mechanisms"]["K3"]["state"]["extra_persistent_bytes"]}
    for kind in X.CANDIDATES:
        run = lab.run_mechanism(world, kind, keep=True)
        sizes = {st["persistent_bytes"] for st in run["stages"].values()}
        assert sizes == {declared[kind]}, kind
        mech, W, epoch = run["_run"]["_final"]
        blob = E.serialize(mech, W, epoch)
        header = 4 + int.from_bytes(blob[:4], "little")
        assert len(blob) - header == 8 * W.size + declared[kind], kind
        m2, W2, e2 = E.deserialize(blob, IDX, T.uninformed_rng(SPEC, world.world, "D"))
        assert E.serialize(m2, W2, e2) == blob and e2 == epoch and np.array_equal(W2, W)
        if kind == "K2":
            assert np.array_equal(m2.U, mech.U)
        again = lab.run_mechanism(world, kind)
        assert [again["stages"][s]["state_digest"] for s in X.STAGES] == \
            [run["stages"][s]["state_digest"] for s in X.STAGES], kind
    assert F * (F + 1) // 2 * 8 == declared["K2"]


def test_extended_state_transplant_and_reset(lab, world):
    run = lab.run_mechanism(world, "K1", keep=True)["_run"]
    t = X.transplant_at_b(lab, world, "K1", run)
    assert t["continuation_bitwise"] and t["restored_digest_equal"]
    r = X.reset_at_b(lab, world, "K1", run)
    assert set(r) >= {"stages", "both_retained_at_D"}


def test_effective_numerical_environment_is_single_threaded():
    code = ("import research.ecs_retention_r2, numpy\n"
            "from research.ecs_retention_r2 import numerics, protocol\n"
            "protocol.require_single_thread()\n"
            "p = numerics.profile()\n"
            "print(p['blas']['threads'], p['numpy'])")
    env = {k: v for k, v in os.environ.items() if not k.endswith("_NUM_THREADS")}
    env["PYTHONPATH"] = f"{REPO / 'src'}{os.pathsep}{REPO}"
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split()[0] == "1"


def test_write_once_records_and_binding(tmp_path):
    path = tmp_path / "r.json"
    digest = P.write(path, "dev", {"x": 1})
    assert P.load(path, "dev")["digest"] == digest
    with pytest.raises(P.ProtocolError):
        P.write(path, "dev", {"x": 2})
    impl = P.implementation(_library_path())
    assert {"files", "library", "build", "numerical_profile", "lab_source_digest"} <= set(impl)
    # Closed laboratory: its frozen binding names the pre-rename layout; under the one stated rename
    # (tests/research/_renamed_sources.py, applied in memory by the fixture above) it binds the canonical ECS.
    assert "native/ECS/src/ecsg_executor.c" in impl["files"] and "src/elpis/ECS/native.py" in impl["files"]
    prof = impl["numerical_profile"]
    for key in ("python", "numpy", "numpy_build", "blas", "cpu", "machine", "thread_environment"):
        assert key in prof, key
    assert P.binding_mismatch(impl, impl) == []


def test_classes_selection_task_rule_and_disposition_follow_the_specification():
    th = SPEC["thresholds"]
    init = {t: 1.0 for t in X.STAGES}
    good = {s: {t: (0.01 if X.STAGES.index(t) <= X.STAGES.index(s) else 1.0) for t in X.STAGES} for s in X.STAGES}
    c = X.classify(good, init, th)
    assert c["SEQUENCE_HELD"] and not any(c["CATASTROPHIC"].values())
    forget = json.loads(json.dumps(good))
    forget["D"]["A"] = 1.2
    c2 = X.classify(forget, init, th)
    assert not c2["SEQUENCE_HELD"] and c2["CATASTROPHIC"]["A@D"] and not c2["RETAINED"]["A@D"]

    def row(held, final):
        return {"SEQUENCE_HELD": held, "final_mean_nmse": final, "stages": {"A": {"persistent_bytes": 1,
                                                                                   "hot_ops": 1}}}
    table = {w: {"P": {"K1": row(True, 0.2), "K2": row(True, 0.1), "K3": row(False, 0.0)}} for w in ("a", "b")}
    assert X.select(table)[0] == {"candidate": "K2"}
    table = {w: {"P": {"K1": row(True, 0.1), "K2": row(True, 0.1), "K3": row(True, 0.1)}} for w in ("a", "b")}
    assert X.select(table)[0] == {"candidate": "K1"}
    ok = {"x": True}
    assert X.disposition({"v": True}, {"m": False}, ok, [], [], [], th)["disposition"] == "MECHANICS_FAIL"
    assert X.disposition({"v": False}, ok, ok, [], [], [], th)["outcome"] == "OUTCOME_V"
    assert X.disposition(ok, ok, ok, [], [], [], th)["outcome"] == "OUTCOME_A"
    # the task rule needs all of V1-V6 (synthetic control rows)
    seq = {"SEQUENCE_HELD": False, "final_earlier_mean_nmse": 0.9, "A_end_ratio": 50.0,
           "LEARNED": {t: True for t in X.STAGES}}
    m1 = dict(seq, SEQUENCE_HELD=True)
    good = {"M0": seq, "M1": m1, "witness": {t: 0.0 for t in X.STAGES},
            "ceilings": {"previous": {t: 0.9 for t in "BCD"}, "capacity_max": 0.0}}
    rows = {w: json.loads(json.dumps(good)) for w in ("a", "b", "c", "d")}
    assert X.task_validity_dev(rows, th)["valid"]
    for path, bad in ((("M0", "final_earlier_mean_nmse"), 0.2), (("ceilings", "capacity_max"), 1e-3),
                      (("M1", "SEQUENCE_HELD"), False)):
        broken = json.loads(json.dumps(rows))
        broken["a"][path[0]][path[1]] = bad
        if path[1] == "final_earlier_mean_nmse":
            for w in broken:
                broken[w]["M0"]["final_earlier_mean_nmse"] = bad
        assert not X.task_validity_dev(broken, th)["valid"], path


def test_laboratory_uses_no_other_model_and_is_never_imported_by_canonical_code():
    forbidden = ("research.dsv41_tower", "research.ecs_dynamics", "elpis.inference", "elpis.runtime", "torch",
                 "transformers", "sklearn", "scipy", "jax")
    for path in sorted(LAB_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not [n for n in names if any(n == f or n.startswith(f + ".") for f in forbidden)], path.name
    for path in sorted((REPO / "src").rglob("*.py")):
        assert "ecs_retention_r2" not in path.read_text(encoding="utf-8"), path


def test_k2_projection_is_the_registered_gram_pseudoinverse_operator(lab, world):
    """K2 computes C^T (C C^T)^+ C from the SVD of C (same operator, same cutoff, better conditioned)."""
    WA = lab.w_a(world)
    k2 = E.make("K2", IDX)
    k2.consolidate(WA, world.tasks["A"].x_train)
    C = k2.U.T @ E.jacobian(WA, IDX)
    for seed in range(3):
        d = np.random.default_rng(seed).normal(size=WA.shape)
        gram = C.T @ (E.pinv_sym(C @ C.T, E.GRAM_CUTOFF) @ (C @ d.reshape(-1)))
        assert np.linalg.norm(k2.projection(WA, d) - gram) <= 1e-8 * np.linalg.norm(gram)
