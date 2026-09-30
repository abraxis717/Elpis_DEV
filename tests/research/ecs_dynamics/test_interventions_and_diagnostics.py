"""Null-space interventions (tolerance, rejection), phase detectors and recovery classes."""
from __future__ import annotations

import numpy as np
import pytest

from research.ecs_dynamics import diagnostics as D
from research.ecs_dynamics import intervention as I
from research.ecs_dynamics import observables as O
from research.ecs_dynamics import systems as Y
from research.ecs_dynamics.attractors import AttractorInventory


def test_linear_coarse_map_is_preserved_exactly():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16)
    C = O.projection([0, 3])
    acc, rep = I.coarse_preserving_interventions(x, C, C.jacobian, rng, magnitude=0.5, count=8, tolerance=1e-12)
    assert rep["accepted"] == 8 and rep["rejected"] == 0
    assert rep["nominal_dimension"] == 2 and rep["rank"] == 2 and rep["null_dimension"] == 14
    assert all(np.array_equal(C(a), C(x)) for a in acc)


def test_nonlinear_map_rejects_over_tolerance_and_retraction_repairs():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(16)
    C = O.MEAN_AND_SECOND
    acc, rep = I.coarse_preserving_interventions(x, C, C.jacobian, rng, magnitude=0.3, count=6, tolerance=1e-10)
    assert rep["accepted"] == 0 and rep["rejected"] == 6  # first-order preservation only: residual O(eps^2)
    acc, rep = I.coarse_preserving_interventions(x, C, C.jacobian, rng, magnitude=0.3, count=6, tolerance=1e-10,
                                                 retraction_steps=6)
    assert rep["accepted"] == 6 and rep["residual_max"] <= 1e-10
    assert rep["accepted"] + rep["rejected"] == rep["attempted"]


def test_degenerate_fibre_yields_no_accepted_intervention():
    # At x = 0 the level set {mean = 0, second moment = 0} is the single point 0.
    rng = np.random.default_rng(2)
    acc, rep = I.coarse_preserving_interventions(np.zeros(8), O.MEAN_AND_SECOND, O.MEAN_AND_SECOND.jacobian, rng,
                                                 magnitude=0.1, count=4, tolerance=1e-9, retraction_steps=6)
    assert acc == [] and rep["rejected"] == 4


def test_divergence_statistics():
    stats = I.divergence_statistics(np.zeros(3), [np.ones(3), np.zeros(3)], 0.5)
    assert stats["count"] == 2 and stats["fraction_above_threshold"] == 0.5


def _classify(system, x0, warmup=2000, tail=400, max_period=16):
    x = Y.run(system, x0, warmup)
    return D.classify_tail(Y.rollout(system, x, tail - 1), fixed_tol=1e-10, recurrence_tol=1e-8,
                           max_period=max_period)


def test_fixed_point_detector():
    assert _classify(Y.LinearSystem(0.5 * np.eye(3)), np.ones(3))["class"] == "FIXED_POINT"
    assert _classify(Y.LogisticMap(2.8), np.array([0.3]))["class"] == "FIXED_POINT"


def test_period_detectors():
    two = _classify(Y.CyclicShift(2), np.array([0.1, 0.7]), warmup=3)
    assert two["class"] == "PERIODIC" and two["period"] == 2 and two["cycles"] >= 3
    logistic = _classify(Y.LogisticMap(3.2), np.array([0.3]))
    assert logistic["class"] == "PERIODIC" and logistic["period"] == 2
    assert _classify(Y.LogisticMap(3.5), np.array([0.3]))["period"] == 4
    assert _classify(Y.CyclicShift(5), np.arange(5.0), warmup=0)["period"] == 5


def test_period_needs_repetition_across_cycles():
    tail = np.array([[0.0], [1.0], [0.0], [1.0]])  # only two cycles of period 2
    assert D.classify_tail(tail, fixed_tol=1e-12, recurrence_tol=1e-9, max_period=2, min_cycles=3)["class"] == \
        "UNRESOLVED_NONPERIODIC"


def test_unresolved_is_never_called_chaotic():
    info = _classify(Y.LogisticMap(3.9), np.array([0.3]), max_period=64)
    assert info["class"] == "UNRESOLVED_NONPERIODIC"
    assert "CHAOTIC" not in D.CLASSES
    assert _classify(Y.LinearSystem(1.5 * np.eye(2)), np.ones(2), warmup=100)["class"] == "UNBOUNDED"


def test_finite_time_growth_and_recovery():
    rng = np.random.default_rng(3)
    stable, unstable = Y.LinearSystem(0.5 * np.eye(3)), Y.LinearSystem(np.diag([1.2, 0.3, 0.3]))
    assert D.finite_time_growth(stable, np.ones(3), rng, 50) == pytest.approx(np.log(0.5))
    # Finite-time: the random initial tangent direction needs time to align, an O(1/T) bias.
    assert D.finite_time_growth(unstable, np.ones(3), rng, 200) == pytest.approx(np.log(1.2), abs=1e-2)
    assert D.recovery(stable, np.ones(3), rng, magnitude=1e-6, horizon=50)["class"] == "DECAY"
    assert D.recovery(unstable, np.ones(3), rng, magnitude=1e-6, horizon=50)["class"] == "GROW"
    assert D.recovery(Y.CyclicShift(3), np.arange(3.0), rng, magnitude=1e-6, horizon=30)["class"] == "PERSIST"


def test_correlations():
    ac = D.autocorrelation(np.sin(np.arange(200) * 0.3), 5)
    assert ac[0] == pytest.approx(1.0) and ac[1] < 1.0
    traj = np.array([[1.0, 1.0], [1.0, -1.0], [2.0, 0.0]])
    assert D.two_time_correlation(traj, [0], [0, 1, 2])[0].tolist() == [1.0, 0.0, 1.0]


def test_attractor_inventory_separates_unresolved():
    inv = AttractorInventory(warmup=1000, tail=200, fixed_tol=1e-10, recurrence_tol=1e-8, max_period=16)
    for x0 in (0.2, 0.4, 0.6):
        inv.label(Y.LogisticMap(3.2), np.array([x0]))
    s = inv.summary(3)
    assert s["attractor_count_proxy"] == 1 and s["periods"] == [2] and s["attractors"][0]["basin_count"] == 3
    inv2 = AttractorInventory(warmup=1000, tail=200, fixed_tol=1e-10, recurrence_tol=1e-8, max_period=16)
    inv2.label(Y.LogisticMap(3.9), np.array([0.3]))
    assert inv2.summary(1)["attractor_count_proxy"] == 0 and inv2.summary(1)["unresolved_fraction"] == 1.0


def test_symmetric_tanh_probe_has_several_sampled_fixed_points_or_two_cycles():
    rng = np.random.default_rng(4)
    system = Y.make_tanh_rnn(rng, 16, 3.0, 0.0)
    inv = AttractorInventory(warmup=500, tail=200, fixed_tol=1e-10, recurrence_tol=1e-8, max_period=8)
    labels = [inv.label(system, system.initial_state(rng)) for _ in range(12)]
    s = inv.summary(12)
    assert s["attractor_count_proxy"] >= 2
    assert set(s["periods"]) <= {2}
    # Slow transients stay unresolved: counted apart, never merged into an attractor.
    assert all(lab >= 0 or lab == -1 for lab in labels)
    assert s["unresolved_fraction"] == labels.count(-1) / 12
    assert sum(a["basin_count"] for a in s["attractors"]) == sum(lab >= 0 for lab in labels)
    inv.add_recovery(system, rng, 1e-6)
    assert all(a["recovery"] == "RETURNS" for a in inv.summary(12)["attractors"])
