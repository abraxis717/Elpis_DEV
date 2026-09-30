"""Recurrent systems, delay pipeline, and the paper-derived computations."""
from __future__ import annotations

import numpy as np
import pytest

from research.ecs_dynamics import associative as A
from research.ecs_dynamics import delay as L
from research.ecs_dynamics import feedback as F
from research.ecs_dynamics import systems as Y
from research.ecs_dynamics.experiments import EXPERIMENTS, _cubic_world, _delay_rows
from research.ecs_dynamics.spec import canonical_json, world_rng
from research.ecs_dynamics.sweep import cell_world


def test_same_rng_same_system_and_trajectory():
    make = lambda: Y.make_tanh_rnn(np.random.default_rng(5), 12, 1.5, 0.7, bias_scale=0.1, input_dimension=1)
    a, b = make(), make()
    assert np.array_equal(a.J, b.J) and np.array_equal(a.b, b.b)
    x0 = np.linspace(-1, 1, 12)
    assert np.array_equal(Y.rollout(a, x0, 50), Y.rollout(b, x0, 50))


def test_clone_is_exact_and_independent():
    s = Y.make_tanh_rnn(np.random.default_rng(6), 8, 2.0, 1.0)
    c = s.clone()
    assert c is not s and np.array_equal(c.J, s.J) and not np.shares_memory(c.J, s.J)
    x0 = np.ones(8) * 0.3
    assert np.array_equal(Y.rollout(c, x0, 20), Y.rollout(s, x0, 20))
    with pytest.raises(ValueError):
        s.J[0, 0] = 1.0  # parameters are read-only


def test_input_and_perturbation():
    s = Y.make_tanh_rnn(np.random.default_rng(7), 6, 1.0, 1.0, input_dimension=2)
    x = np.zeros(6)
    assert not np.array_equal(s.step(x, np.array([1.0, 0.0])), s.step(x))
    with pytest.raises(ValueError):
        Y.LinearSystem(np.eye(2)).step(np.ones(2), np.ones(1))
    rng = np.random.default_rng(8)
    assert np.linalg.norm(Y.perturb(np.zeros(5), rng, 1e-3)) == pytest.approx(1e-3)


def test_nonreciprocal_normalization():
    d = 400
    for gamma in (0.0, 1.0, 3.0):
        J = Y.nonreciprocal_coupling(np.random.default_rng(9), d, gamma)
        off = ~np.eye(d, dtype=bool)
        assert np.var(J[off]) * d == pytest.approx(1.0, rel=0.02)
        tau = np.corrcoef(J[np.triu_indices(d, 1)], J.T[np.triu_indices(d, 1)])[0, 1]
        assert tau == pytest.approx((1 - gamma ** 2) / (1 + gamma ** 2), abs=0.02)


def test_ridge_matches_closed_form():
    rng = np.random.default_rng(10)
    X, Y_ = rng.standard_normal((50, 3)), rng.standard_normal((50, 2))
    model = L.Ridge(0.0).fit(X, Y_)
    A_ = np.column_stack([np.ones(50), X])
    coef, *_ = np.linalg.lstsq(A_, Y_, rcond=None)
    assert np.allclose(model.predict(X), A_ @ coef)


def test_delay_pipeline_uses_one_split_one_target_for_every_representation():
    spec = EXPERIMENTS["tanh-delay-ladder"].spec.with_choices(horizon=80, parameters=dict(
        EXPERIMENTS["tanh-delay-ladder"].spec.parameters, initial_conditions=4))
    cell = {"gain": 2.0, "gamma": 1.0}
    for target in ("ONE_STEP_TRANSITION", "K_STEP_TRAJECTORY", "EVENT"):
        train, test = _delay_rows(spec, "dev-0000", cell, target)
        n_train, n_test = len(train["_target"]), len(test["_target"])
        for rep in L.REPRESENTATIONS:
            assert len(train[rep]) == n_train and len(test[rep]) == n_test
        assert train["delay"].shape[1] == train["shuffled_delay"].shape[1] == train["random_projection"].shape[1]
        assert np.array_equal(train["delay"][:, :2], train["coarse"])  # the delay vector starts with C_t
    # choose_lambda only ever sees the splits it is given (DEV); it never receives test rows.
    splits = [_delay_rows(spec, w, cell, "ONE_STEP_TRANSITION") for w in ("dev-0000", "dev-0001")]
    lam, table = L.choose_lambda(splits, "delay", "ONE_STEP_TRANSITION", [1e-3, 1e-1])
    assert lam in table and set(table) == {1e-3, 1e-1}


def test_ladder_identity_and_nested_monotonicity():
    spec = EXPERIMENTS["tanh-delay-ladder"].spec.with_choices(horizon=80, parameters=dict(
        EXPERIMENTS["tanh-delay-ladder"].spec.parameters, initial_conditions=4))
    train, test = _delay_rows(spec, "dev-0000", {"gain": 2.0, "gamma": 1.0}, "ONE_STEP_TRANSITION")
    lad = L.ladder_decomposition(train, test, {"full": 1e-3, "coarse": 1e-3, "delay": 1e-3})
    assert lad["identity_residual"] < 1e-12
    small, big = L.insample_least_squares_monotone(train["coarse"], train["delay"], train["_target"])
    assert big <= small


def test_experiment_worlds_are_deterministic():
    cubic = EXPERIMENTS["cubic-control"].spec
    assert canonical_json(_cubic_world(cubic, "dev-0000")) == canonical_json(_cubic_world(cubic, "dev-0000"))
    sweep_spec = EXPERIMENTS["tanh-phase-sweep"].spec.with_choices(warmup=50, horizon=40, lyapunov_horizon=20,
                                                                   parameters=dict(EXPERIMENTS["tanh-phase-sweep"]
                                                                                   .spec.parameters, initial_conditions=2))
    assert canonical_json(cell_world(sweep_spec, "dev-0000", 1.5, 1.0)) == \
        canonical_json(cell_world(sweep_spec, "dev-0000", 1.5, 1.0))


# --- arXiv:2609.19288 (Vaidya) --------------------------------------------------------------


def test_vaidya_critical_feedback_reproduced_from_eqs_4_5_and_4_6():
    crit = F.critical_feedback(1.3)
    assert crit["chi_at_u_c"] == pytest.approx(1.0, abs=1e-9)
    assert crit["yhat_c"] == pytest.approx(0.2, abs=0.01)  # paper: yhat_c = 0.2 at g = 1.3
    assert crit["u_c"] == pytest.approx(0.50, abs=0.01)    # so "u_c ... is 0.2" (Sec. 5.3) refers to yhat_c
    assert F.critical_feedback(2.0)["yhat_c"] > crit["yhat_c"]  # "will increase with increasing value of g"


def test_vaidya_meanfield_growth_changes_sign_at_yhat_c():
    yc = F.critical_feedback(1.3)["yhat_c"]
    assert F.meanfield_growth(1.3, 0.9 * yc)["chi"] > 1.0 > F.meanfield_growth(1.3, 1.1 * yc)["chi"]


# --- arXiv:2609.07341 (Aguilera & De Martino) -----------------------------------------------


@pytest.mark.parametrize("phi", [0.0, 0.1 * np.pi, 0.25 * np.pi, 0.4 * np.pi])
def test_hopf_line_and_frequency_match_linearization(phi):
    chk = A.linearization_check(phi, 0.1)
    assert chk["spectral_radius"] == pytest.approx(1.0, abs=1e-12)
    assert chk["argument"] == pytest.approx(chk["omega_formula"], abs=1e-12)


def test_overlap_map_jacobian_and_onset():
    s = A.OverlapMeanField(A.rotation(0.2 * np.pi), 3.0, 0.1)
    m = np.array([0.3, -0.2])
    h = 1e-6
    fd = np.column_stack([(s.step(m + h * e) - s.step(m - h * e)) / (2 * h) for e in np.eye(2)])
    assert np.allclose(fd, s.jacobian(m), atol=1e-8)
    bc = A.hopf_beta(0.2 * np.pi, 0.1)
    below = Y.run(A.OverlapMeanField(A.rotation(0.2 * np.pi), 0.9 * bc, 0.1), np.array([0.5, 0.1]), 5000)
    above = Y.run(A.OverlapMeanField(A.rotation(0.2 * np.pi), 1.1 * bc, 0.1), np.array([0.5, 0.1]), 5000)
    assert np.linalg.norm(below) < 1e-6 < 0.1 < np.linalg.norm(above)


def test_microscopic_couplings_and_zero_load_retrieval():
    rng = np.random.default_rng(11)
    J, xi = A.couplings(rng, 400, 0.0, 10, "uniform")
    assert np.all(np.diag(J) == 0.0) and J.shape == (400, 400)
    J0, xi0 = A.couplings(rng, 400, 0.0, 0, "coherent")
    ov = A.glauber_overlaps(rng, J0, xi0, xi0[:, 0].copy(), beta=4.0, delta=0.1, steps=100)
    assert A.retrieval_norm(ov, 20) > 0.9
    with pytest.raises(ValueError):
        A.couplings(rng, 10, 0.0, 1, "partial")


def test_world_rng_drives_the_associative_simulation_deterministically():
    spec = EXPERIMENTS["associative-cycles"].spec
    a = world_rng(spec, "dev-0000", "s").random(3)
    assert np.array_equal(a, world_rng(spec, "dev-0000", "s").random(3))
