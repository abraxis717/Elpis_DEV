"""Retention R3 laboratory mechanics (RET3B). No DEV or QUAL world is evaluated: test-* worlds only.

Established before DEV:

* canonical forward equivalence and the S3 identity; the K1 derivative/correction identity;
* exact reduction to canonical G1 when the K1 state is absent (H = 0) or removed;
* deterministic, fixed-size complete-state serialization; refusal of foreign, truncated or W-only bytes as complete;
* no target enters consolidation (closed form, no iteration); no task identity enters query or learning;
* query ignores (H, a); (H, a) shapes later learning, and the first step from the consolidation anchor carries a
  zero correction by construction (the RET3B decision recorded in the README);
* the reset challenge mechanics, the full-state transplant and the W-only negative control;
* refusal atomicity and non-finite refusal; an effective single thread; the source and binary binding;
* the classes, the DEV rules, the gates, the decision record and the disposition behave as registered.
"""
from __future__ import annotations

import ast
import ctypes
import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from elpis.ECS_G.native import ECSGLibrary, Executor

from research.ecs_retention_r3 import engine as E
from research.ecs_retention_r3 import experiment as X
from research.ecs_retention_r3 import protocol as P
from research.ecs_retention_r3 import task as T

from ...ECS_G.test_math_r0 import REPO, _library_path

SPEC = P.load_spec()
IDX = T.indices(SPEC["regime"]["dim"])
LAB_DIR = REPO / "research" / "ecs_retention_r3"
GAIN = SPEC["task"]["hidden_gain"]


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
def world(lab):
    return lab.world("test-0000")


@pytest.fixture(scope="module")
def k1_world(lab, world):
    return X.k1_world(lab, world)


def _fd(fun, W, h=1e-6, every=5):
    g, mask = np.zeros_like(W), np.zeros_like(W, dtype=bool)
    for e in range(W.shape[0]):
        for i in range(0, W.shape[1], every):
            wp, wm = W.copy(), W.copy()
            wp[e, i] += h
            wm[e, i] -= h
            g[e, i], mask[e, i] = (fun(wp) - fun(wm)) / (2 * h), True
    return g, mask


# --- registered constants and the inherited task ------------------------------------------------------------------


def test_registered_constants_are_the_specification():
    assert E.make("K1", IDX).constants()["lambda"] == SPEC["mechanisms"]["K1"]["hyperparameters"]["lambda"] == 1.0
    assert E.make("C1R", IDX).lam == SPEC["mechanisms"]["C1R"]["hyperparameters"]["lambda"] == 4.0
    assert IDX.size == SPEC["regime"]["features"] == 83 and GAIN == 2.0
    assert X.ENGINE_KINDS == ("C1R", "K1")
    for kind in ("K2", "K3"):
        with pytest.raises(ValueError):
            E.make(kind, IDX)
    reg = SPEC["regime"]
    assert E.k1_ops(reg["dim"], reg["width"], IDX) == SPEC["native_budget"]["k1_hot_path_extra_ops_per_step"]
    assert E.g1_ops(reg["dim"], reg["width"], reg["train_rows"]) == SPEC["native_budget"]["g1_ops_per_step_at_R64"]
    assert E.k1_consolidation_ops(reg["dim"], reg["width"], reg["train_rows"], IDX) == 462782
    assert E.make("K1", IDX).persistent_bytes() == SPEC["thresholds"]["native_persistent_extra_bytes"]


def test_world_ids_are_fresh_and_disjoint_from_r2():
    dev, qual = P.world_ids(SPEC, "DEV"), P.world_ids(SPEC, "QUAL")
    assert dev[0] == "r3dev-0000" and len(dev) == 8 and qual[-1] == "r3qual-0031" and len(qual) == 32
    r2 = json.loads((REPO / "research/ecs_retention_r2/specs/ecsg-retention-r2.v1.spec.json").read_text())
    r2_ids = {f"dev-{i:04d}" for i in range(r2["splits"]["dev_worlds"])} | {
        f"qual-{i:04d}" for i in range(r2["splits"]["qual_worlds"])}
    assert not (set(dev) | set(qual)) & r2_ids
    a = T.build_world(SPEC, "test-0000", GAIN)
    b = T.build_world(r2 | {"task": dict(r2["task"])}, "test-0000", GAIN)
    assert not np.array_equal(a.w0, b.w0), "R3's seed and name must give different worlds from R2's"


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
    assert np.allclose(E.jacobian(W, IDX).T @ u, E.s3_vjp(W, u, IDX).reshape(-1), rtol=1e-12, atol=1e-12)


def test_k1_correction_is_the_derivative_of_its_declared_quantity(lab, world):
    gi = lab.gradient_identity(world)
    assert gi["pass"] and max(gi["K1"], gi["C1R"]) <= SPEC["thresholds"]["gradient_identity_relative"], gi


def test_k1_is_c1_at_lambda_one_for_one_experience(lab, world):
    WA = lab.w_a(world)
    k1, c1 = E.make("K1", IDX), E.make("C1R", IDX, lam=1.0)
    k1.consolidate(WA, world.tasks["A"].x_train)
    c1.consolidate(WA, world.tasks["A"].x_train)
    W = WA + 0.01 * np.random.default_rng(1).normal(size=WA.shape)
    assert np.allclose(k1.grad(W), c1.grad(W), rtol=1e-10, atol=1e-13)


def test_witness_is_exact_and_the_task_has_its_inherited_structure(api, lab, world):
    assert max(lab.witness(world).values()) <= SPEC["thresholds"]["witness_max_nmse"]
    Q = world.basis
    assert np.allclose(Q.T @ Q, np.eye(6), atol=1e-12)
    for k, t in enumerate("ABCD"):
        M, exp = world.planes[t], world.tasks[t]
        assert np.allclose(exp.x_train - exp.x_train @ M @ M.T, 0.0, atol=1e-12)
        for later in range(k + 1, 4):
            assert np.allclose(exp.x_test @ Q[:, 2 + later], 0.0, atol=1e-12), (t, later)
    xb, xa = world.tasks["B"].x_test, world.tasks["A"].x_test
    r4, r1 = GAIN * Q[:, 3:4], GAIN * Q[:, 0:1]
    assert np.allclose(T.cubic(r4, xb), T.cubic(r1, xb), atol=1e-12)
    assert not np.allclose(T.cubic(r4, xa), T.cubic(r1, xa))
    ceil = lab.ceilings(world)
    assert set(ceil["previous"]) == {"B", "C", "D"} and ceil["capacity_max"] <= 1e-6


# --- reduction to the canonical core ----------------------------------------------------------------------------


def test_engine_without_k1_state_is_the_canonical_core_and_w_a_is_shared(api, lab, world):
    m0 = lab.run_mechanism(world, "M0")
    for kind in X.ENGINE_KINDS:
        removed = lab.run_mechanism(world, kind, removed=True)
        assert all(removed["stages"][s]["W_digest"] == m0["stages"][s]["W_digest"] for s in X.STAGES), kind
        assert lab.run_mechanism(world, kind)["W_A_digest"] == m0["stages"]["A"]["W_digest"], kind
    WA, b = lab.w_a(world), world.tasks["B"]
    absent = E.engine_learn(api, WA, b.x_train, b.y_train, 0.002, 7, E.make("K1", IDX))[0]   # H = 0, a = 0
    plain = E.engine_learn(api, WA, b.x_train, b.y_train, 0.002, 7, E.make("K1", IDX, removed=True))[0]
    canonical = E.canonical_learn(api, WA, b.x_train, b.y_train, 0.002, 7)[0]
    assert np.array_equal(absent, plain) and np.array_equal(plain, canonical)


def test_pte_microstates_diverge_under_k1(lab, world):
    p, q = T.pte_pair(world.w0, world.pte_axis)
    sp, sq = E.s3(p, IDX), E.s3(q, IDX)
    assert np.max(np.abs(sp - sq)) <= 1e-12 * max(1.0, np.max(np.abs(sp))) and np.any(p != q)
    assert lab.pte_check(world)["pass"]


# --- refusal ------------------------------------------------------------------------------------------------------


def test_refusal_and_non_finite_refusal_leave_the_complete_state_unchanged(api, lab, world):
    WA = lab.w_a(world)
    b = world.tasks["B"]
    k1 = E.make("K1", IDX)
    k1.consolidate(WA, world.tasks["A"].x_train)
    before = E.state_digest(k1, WA, 0)
    W, refused, _ = E.engine_learn(api, WA, b.x_train, b.y_train, 1e6, 5, k1)          # diverges
    assert refused > 0 and np.array_equal(W, WA) and E.state_digest(k1, WA, 0) == before
    bad_x = b.x_train.copy()
    bad_x[3, 1] = np.nan
    W, refused, _ = E.engine_learn(api, WA, bad_x, b.y_train, 0.002, 5, k1)            # non-finite input
    assert refused > 0 and np.array_equal(W, WA) and E.state_digest(k1, WA, 0) == before
    blown = X.Lab(lab.api, _short_spec())
    blown.rate = 1e6
    run = blown.mechanism_run(blown.world("test-0001"), "K1")
    for s in ("B", "C", "D"):
        assert run["stages"][s]["refused"] != 0
        assert run["stages"][s]["state_digest"] == run["stages"]["A"]["state_digest"]


# --- consolidation, query and task identity -----------------------------------------------------------------------


def test_consolidation_receives_no_target_and_is_closed_form():
    interface = X.consolidation_interface()
    assert set(interface) == {"C1R", "K1"}
    for kind, v in interface.items():
        assert v["parameters"] == ["W", "inputs"] and not v["receives_targets"] and v["closed_form"], kind


def test_no_task_identity_enters_query_or_learning():
    assert list(inspect.signature(E.query).parameters) == ["api", "state", "X"]
    assert list(inspect.signature(E.engine_learn).parameters) == ["api", "W_start", "X", "y", "rate", "steps", "mech"]
    assert list(inspect.signature(X.Lab.evaluate).parameters) == ["self", "W", "world"]
    for fn in (E.query, E.engine_learn, E.K1.correct, E.K1.grad, E.K1.consolidate):
        assert not {"task", "task_id", "label", "stage"} & set(inspect.signature(fn).parameters), fn.__name__


def test_query_ignores_h_and_a_and_the_anchor_step_carries_no_correction(api, lab, world):
    WA = lab.w_a(world)
    k1 = E.make("K1", IDX)
    k1.consolidate(WA, world.tasks["A"].x_train)                   # a = S3(W_A): W_A is the anchor
    full, bare = (k1, WA, 4000), E.reset((k1, WA, 4000))
    x = lab.heldout_inputs(world)
    assert np.array_equal(E.query(api, full, x), E.query(api, bare, x))
    assert not np.any(k1.grad(WA)), "u(W_anchor) = 1/2 H (S3(W_anchor) - a) = 0 exactly"
    b = world.tasks["B"]
    one_full = E.engine_learn(api, WA, b.x_train, b.y_train, 0.002, 1, k1)[0]
    one_bare = E.engine_learn(api, WA, b.x_train, b.y_train, 0.002, 1, bare[0])[0]
    assert np.array_equal(one_full, one_bare), "one step from the anchor is identical by construction"
    many_full = E.engine_learn(api, WA, b.x_train, b.y_train, 0.002, 20, k1)[0]
    many_bare = E.engine_learn(api, WA, b.x_train, b.y_train, 0.002, 20, bare[0])[0]
    assert not np.array_equal(many_full, many_bare), "(H, a) shape later learning"


# --- the declared complete state ----------------------------------------------------------------------------------


def test_complete_state_is_fixed_size_and_serializes_deterministically(lab, world):
    run = lab.run_mechanism(world, "K1", keep=True)
    declared = SPEC["thresholds"]["native_persistent_extra_bytes"]
    assert {st["persistent_bytes"] for st in run["stages"].values()} == {declared}
    mech, W, epoch = run["_run"]["_final"]
    blob = E.serialize(mech, W, epoch)
    header = 4 + int.from_bytes(blob[:4], "little")
    assert len(blob) - header == 8 * W.size + declared
    m2, W2, e2 = E.deserialize(blob, IDX)
    assert E.serialize(m2, W2, e2) == blob and e2 == epoch and np.array_equal(W2, W) and m2.provenance == "COMPLETE"
    again = lab.run_mechanism(world, "K1")
    assert [again["stages"][s]["state_digest"] for s in X.STAGES] == [run["stages"][s]["state_digest"]
                                                                       for s in X.STAGES]
    with pytest.raises(ValueError):
        E.deserialize(blob[:-8], IDX)
    with pytest.raises(ValueError):
        E.deserialize(blob + b"\0" * 8, IDX)
    foreign = blob.replace(E.STATE_FORMAT.encode(), E.STATE_FORMAT.replace("r3", "r2").encode())
    with pytest.raises(ValueError):
        E.deserialize(foreign, IDX)


def test_w_only_snapshot_imports_only_as_unconsolidated(api, lab, world):
    WA = lab.w_a(world)
    with Executor.create(api, lab.dim, lab.width, E._flat(WA)) as e:
        snapshot = e.snapshot()
    mech, W, epoch = E.import_w_only(snapshot, IDX, lab.dim, lab.width)
    assert mech.provenance == "UNCONSOLIDATED_IMPORT" and not mech.H.any() and not mech.a.any()
    assert np.array_equal(W, WA) and epoch == 0
    with pytest.raises(ValueError):
        E.deserialize(snapshot, IDX)                                 # never a complete retained state
    with pytest.raises(ValueError):
        E.import_w_only(b"XXXXXXXX" + snapshot[8:], IDX, lab.dim, lab.width)
    with pytest.raises(ValueError):
        E.import_w_only(snapshot, IDX, lab.dim, lab.width + 1)


def test_reset_keeps_w_and_epoch_and_empties_h_and_a(lab, world):
    k1 = E.make("K1", IDX)
    WA = lab.w_a(world)
    k1.consolidate(WA, world.tasks["A"].x_train)
    mech, W, epoch = E.reset((k1, WA, 4000))
    assert np.array_equal(W, WA) and W is not WA and epoch == 4000
    assert mech.provenance == "RESET" and not mech.H.any() and not mech.a.any() and k1.H.any()
    with pytest.raises(TypeError):
        E.reset((E.make("C1R", IDX), WA, 0))


# --- the causal branches ------------------------------------------------------------------------------------------


def test_full_state_transplant_continues_bitwise(k1_world):
    t = k1_world["causality"]["full_state_transplant_at_B"]
    assert t["continuation_bitwise"] and t["restored_digest_equal"]


def test_reset_challenge_reads_the_c_boundary_and_stops(k1_world):
    r = k1_world["causality"]["reset_challenge"]
    assert set(r["stages"]) == {"C"} and r["reset_mechanics"]
    th = SPEC["thresholds"]
    assert r["RESET_DEGRADED"] == (r["e_reset"] >= th["reset_min_factor"] * r["e_uninterrupted"]
                                   and r["e_reset"] >= th["reset_min_absolute_nmse"])
    assert r["reset_ratio"] == pytest.approx(r["e_reset"] / r["e_uninterrupted"])


def test_w_only_negative_control_is_not_a_retained_state(k1_world):
    w = k1_world["causality"]["w_only_negative_control_at_B"]
    assert w["flagged_unconsolidated"] and w["state_digest_differs"]
    assert w["reproduces_reset_at_C"] and w["differs_from_uninterrupted_at_C"] and w["pass"]


def test_state_semantics(k1_world):
    assert k1_world["state_semantics"] == {"query_identical": True, "learning_differs": True}


# --- environment and binding ------------------------------------------------------------------------------------


def test_effective_numerical_environment_is_single_threaded():
    code = ("import research.ecs_retention_r3, numpy\n"
            "from research.ecs_retention_r3 import numerics, protocol\n"
            "protocol.require_single_thread()\n"
            "print(numerics.profile()['blas']['threads'])")
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
    assert {"files", "library", "build", "numerical_profile", "lab_source_digest", "tree", "dirty"} <= set(impl)
    assert "native/ECS_G/src/ecsg_executor.c" in impl["files"] and "src/elpis/ECS_G/native.py" in impl["files"]
    assert P.binding_mismatch(impl, impl) == []
    assert P.spec_digests(SPEC)["candidates_sha256"] == P.sha256_file(LAB_DIR / "PREREGISTRATION.md")


# --- rules, gates, decision record, disposition (synthetic rows) ----------------------------------------------------


def _row(*, held=True, final=0.01, removed_held=False, removed_final=0.6, e_reset=0.4, e_main=0.001):
    stages = {s: {"hot_ops": 42506, "consolidation_ops": 462782, "persistent_bytes": 28552, "W_digest": "x",
                  "refused": 0} for s in X.STAGES}
    k1 = {"LEARNED": {s: held for s in X.STAGES}, "RETAINED": {"A@B": held}, "CATASTROPHIC": {"A@B": False},
          "HELD": {s: held for s in X.STAGES}, "SEQUENCE_HELD": held, "final_mean_nmse": final,
          "final_earlier_mean_nmse": final, "stages": stages}
    removed = {"SEQUENCE_HELD": removed_held, "final_earlier_mean_nmse": removed_final}
    reset = {"stages": {"C": {}}, "e_reset": e_reset, "e_uninterrupted": e_main, "reset_ratio": e_reset / e_main,
             "RESET_DEGRADED": e_reset >= 10 * e_main and e_reset >= 0.05}
    return {"mechanisms": {"K1": k1}, "causality": {
        "state_removed": removed, "reset_challenge": reset,
        "full_state_transplant_at_B": {"continuation_bitwise": True, "restored_digest_equal": True},
        "w_only_negative_control_at_B": {"pass": True}}}


def test_k1_dev_rule_follows_the_specification():
    th = SPEC["thresholds"]
    good = {f"w{i}": _row() for i in range(8)}
    rule = X.k1_dev_rule(good, th, SPEC)
    assert rule["proceeds"] and all(rule["criteria"].values())
    for name, change in (("acquisition", {"held": False}), ("state_removal", {"removed_held": True}),
                         ("reset_challenge", {"e_reset": 0.04})):
        bad = dict(good)
        for w in list(bad)[:3]:
            bad[w] = _row(**change)
        assert not X.k1_dev_rule(bad, th, SPEC)["criteria"][name], name
    two_bad = dict(good, w0=_row(e_reset=0.004), w1=_row(e_reset=0.004))
    assert X.k1_dev_rule(two_bad, th, SPEC)["criteria"]["reset_challenge"]          # 6/8 = 75% still degraded
    three_bad = dict(two_bad, w2=_row(e_reset=0.004))
    assert not X.k1_dev_rule(three_bad, th, SPEC)["criteria"]["reset_challenge"]
    hot = {w: json.loads(json.dumps(r)) for w, r in good.items()}
    hot["w0"]["mechanisms"]["K1"]["stages"]["B"]["hot_ops"] = 82945
    assert not X.k1_dev_rule(hot, th, SPEC)["criteria"]["native_budget"]


def test_classes_task_rule_and_disposition_follow_the_specification():
    th = SPEC["thresholds"]
    init = {t: 1.0 for t in X.STAGES}
    good = {s: {t: (0.01 if X.STAGES.index(t) <= X.STAGES.index(s) else 1.0) for t in X.STAGES} for s in X.STAGES}
    assert X.classify(good, init, th)["SEQUENCE_HELD"]
    forget = json.loads(json.dumps(good))
    forget["C"]["A"] = 1.2
    c = X.classify(forget, init, th)
    assert not c["SEQUENCE_HELD"] and c["CATASTROPHIC"]["A@C"]
    ok = {"x": True}
    assert X.disposition({"v": True}, {"m": False}, ok, [], [], [], th)["disposition"] == "MECHANICS_FAIL"
    assert X.disposition({"v": False}, ok, ok, [], [], [], th)["outcome"] == "OUTCOME_V"
    assert X.disposition(ok, ok, ok, [], [], [], th)["outcome"] == "OUTCOME_A"
    seq = {"SEQUENCE_HELD": False, "final_earlier_mean_nmse": 0.9, "A_end_ratio": 50.0,
           "LEARNED": {t: True for t in X.STAGES}}
    rows = {w: {"M0": seq, "M1": dict(seq, SEQUENCE_HELD=True), "witness": {t: 0.0 for t in X.STAGES},
                "ceilings": {"previous": {t: 0.9 for t in "BCD"}, "capacity_max": 0.0}} for w in "abcd"}
    assert X.task_validity_dev(rows, th)["valid"]
    broken = json.loads(json.dumps(rows))
    broken["a"]["M1"]["SEQUENCE_HELD"] = False
    assert not X.task_validity_dev(broken, th)["valid"]


def test_decision_record_carries_every_decision_bearing_quantity_and_nothing_descriptive(lab, world, k1_world):
    P_ = dict(k1_world)
    P_["mechanisms"] = dict(P_["mechanisms"], M0=X.strip_private(lab.run_mechanism(world, "M0")),
                            M1=X.strip_private(lab.run_mechanism(world, "M1")))
    verdict = {"validity": {"V": True}, "mechanics": {"m": True}, "gates": {"A": True}, "disposition": "D",
               "outcome": "O"}
    record = X.decision_record({"test-0000": {"P": P_}}, verdict)
    assert set(record) == {"validity", "mechanics", "gates", "disposition", "outcome", "K1", "causality", "controls"}
    assert set(record["K1"]["test-0000"]) == {"LEARNED", "RETAINED", "CATASTROPHIC", "HELD", "SEQUENCE_HELD"}
    assert set(record["causality"]["test-0000"]) == {"state_removed_SEQUENCE_HELD", "RESET_DEGRADED",
                                                     "transplant_bitwise", "w_only_control_pass"}
    assert "C1R" not in json.dumps(record) and "nmse" not in json.dumps(record)


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
        assert "ecs_retention_r3" not in path.read_text(encoding="utf-8"), path


def test_only_the_gate_l_children_force_a_kernel():
    """Portability: no CPU-model branch anywhere in the laboratory; OPENBLAS_CORETYPE is set only for gate L."""
    import re
    kernels = re.compile(r"(?i)\b(zen|skylakex|cooperlake|sandybridge)\b")
    for path in sorted(LAB_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert not kernels.search(text), path.name
        if path.name != "run.py":
            assert "OPENBLAS_CORETYPE\"]" not in text and "OPENBLAS_CORETYPE'] =" not in text, path.name
    run = (LAB_DIR / "run.py").read_text(encoding="utf-8")
    assert run.count('env["OPENBLAS_CORETYPE"] = core') == 1 and 'ROBUST_CORES = ("Prescott", "Haswell")' in run
