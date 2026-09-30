"""Exact cubic control, the target-relative distinction, and fibre capacity/prediction loss."""
from __future__ import annotations

import numpy as np
import pytest

from research.ecs_dynamics import cubic as C
from research.ecs_dynamics import intervention as I
from research.ecs_dynamics.results import SufficiencyFinding


def _world(k, d=3, n=12):
    rng = np.random.default_rng(1000 + k)
    return 0.5 * rng.standard_normal((d, n)), rng.standard_normal((16, d))


def _fd(f, W, h=1e-6):
    w = W.ravel()
    cols = []
    for k in range(w.size):
        e = np.zeros_like(w)
        e[k] = h
        cols.append((f((w + e).reshape(W.shape)) - f((w - e).reshape(W.shape))) / (2 * h))
    return np.array(cols).T


@pytest.mark.parametrize("k", range(8))
def test_microscopic_and_contracted_forward_agree(k):
    W, X = _world(k, d=2 + k % 3, n=5 + k)
    micro = C.forward_microscopic(W, X)
    contracted = C.forward_contracted(C.raw_moments(W), X)
    assert np.max(np.abs(micro - contracted)) <= 1e-12 * max(1.0, np.max(np.abs(micro)))


def test_contraction_uses_only_moments():
    W, X = _world(0)
    moments = C.raw_moments(W)
    W[:] = 0.0  # the contraction must not read W
    assert not np.allclose(C.forward_contracted(moments, X), 0.0)


def test_analytic_jacobians_match_finite_differences():
    W, X = _world(1)
    for order in (1, 2, 3):
        assert np.allclose(_fd(lambda M: C.moment_coordinates(M, order), W), C.moment_jacobian(W, order), atol=1e-7)
    assert np.allclose(_fd(lambda M: C.forward_microscopic(M, X), W), C.output_jacobian(W, X), atol=1e-6)
    assert np.allclose(_fd(lambda M: C.transition_coarse_target(M, 0.1), W), C.transition_coarse_jacobian(W, 0.1),
                       atol=1e-6)
    assert np.allclose(_fd(lambda M: C.trajectory_output_target(M, 0.1, X, 2), W),
                       C.trajectory_output_jacobian(W, 0.1, X, 2), atol=1e-5)


def test_pte_witness_separates_instantaneous_from_transition_target():
    W, X = _world(2)
    wa, wb = C.pte_collision(W, 3)
    assert np.array_equal(C.moment_coordinates(wa, 3), C.moment_coordinates(wb, 3))
    # Same S3 => same instantaneous output (the identity) ...
    assert np.max(np.abs(C.forward_microscopic(wa, X) - C.forward_microscopic(wb, X))) < 1e-12
    # ... but not the same coarse state after one rule update: S3 is not transition-sufficient.
    assert np.max(np.abs(C.transition_coarse_target(wa, 0.1) - C.transition_coarse_target(wb, 0.1))) > 1e-4
    # S2 is not even instantaneous-sufficient.
    wa2, wb2 = C.pte_collision(W, 2)
    assert np.array_equal(C.moment_coordinates(wa2, 2), C.moment_coordinates(wb2, 2))
    assert np.max(np.abs(C.forward_microscopic(wa2, X) - C.forward_microscopic(wb2, X))) > 1e-4


def test_fibre_response_capacity_is_target_relative():
    W, X = _world(3)
    eta = 0.1
    caps, losses = {}, {}
    for order in (1, 2, 3):
        ns = I.null_space(C.moment_jacobian(W, order), 1e-10)
        for name, J in (("inst", C.output_jacobian(W, X)), ("trans", C.transition_coarse_jacobian(W, eta))):
            caps[order, name] = I.response_capacity(J, ns.basis, 1e-8)[0]
            losses[order, name] = I.prediction_loss(J, ns.basis)
    assert caps[3, "inst"] == 0 and losses[3, "inst"] < 1e-10
    assert caps[2, "inst"] >= 1 and caps[3, "trans"] >= 1
    for name in ("inst", "trans"):  # more matched moments => smaller fibre => no larger response
        assert caps[1, name] >= caps[2, name] >= caps[3, name]
        assert losses[1, name] + 1e-12 >= losses[2, name] >= losses[3, name] - 1e-12


def test_prediction_loss_is_first_order_worst_case_response():
    W, X = _world(4)
    ns = I.null_space(C.moment_jacobian(W, 2), 1e-10)
    J = C.output_jacobian(W, X)
    _, s, vt = np.linalg.svd(J @ ns.basis)
    eps = 1e-5
    dz = eps * ns.basis @ vt[0]
    change = C.forward_microscopic(W + dz.reshape(W.shape), X) - C.forward_microscopic(W, X)
    assert abs(np.linalg.norm(change) - eps * I.prediction_loss(J, ns.basis)) < 1e-3 * eps * s[0]


def test_instantaneous_sufficiency_is_the_only_claim():
    doc = C.__doc__
    assert "INSTANTANEOUS_OUTPUT" in doc
    for word in ("one-step", "transition", "trajectory", "attractor", "intervention response", "production ECS"):
        assert word in doc
    assert "does not" in doc
    with pytest.raises(ValueError):
        SufficiencyFinding("S3", "SUFFICIENT", "r", "NO_COUNTEREXAMPLE_UNDER_REGIME", "ALGEBRAIC_IDENTITY")
    f = SufficiencyFinding("S3", "INSTANTANEOUS_OUTPUT", "cubic", "NO_COUNTEREXAMPLE_UNDER_REGIME", "ALGEBRAIC_IDENTITY")
    assert "with respect to target INSTANTANEOUS_OUTPUT" in f.statement()
    assert "is sufficient" not in f.statement()
