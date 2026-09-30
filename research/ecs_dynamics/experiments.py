"""Experiment definitions: base specification, DEV procedure, frozen pass rule, QUAL procedure. RESEARCH_ONLY.

A DEV procedure sees only DEV worlds and returns ``choices`` (operational
settings chosen on DEV, e.g. a ridge strength or a regime cell) plus
descriptive metrics. The QUAL procedure receives the frozen specification
(base parameters merged with the DEV choices) and QUAL worlds only. It returns
metrics, target-relative findings, mechanics checks and the evaluation of the
pass rule, which is fixed here in code before any QUAL run.

Every experiment is synthetic. The tanh family is a probe, the cubic family an
algebraic control, the associative network a small-N re-implementation of a
published model, and the ECS experiment runs the public kernel in temporary
storage. None of them is a statement about a production or trained model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import associative, cubic, delay, feedback
from .attractors import AttractorInventory
from .diagnostics import classify_tail, finite_time_growth
from .intervention import (coarse_preserving_interventions, divergence_statistics, null_space, prediction_loss,
                           response_capacity, retract)
from .observables import coarse_map
from .results import SufficiencyFinding
from .spec import ExperimentSpec, world_ids, world_rng
from .sweep import sweep, world_system
from .systems import make_tanh_rnn, rollout, run

SEED = 20260930


@dataclass(frozen=True)
class Experiment:
    name: str
    spec: ExperimentSpec
    evidence: str
    sources: tuple
    pass_rule: dict
    dev: Callable
    qual: Callable
    limitations: tuple
    depends_on: tuple = ()


def _spec(**kw) -> ExperimentSpec:
    base = dict(version=1, seed=SEED, width=None, warmup=0, horizon=1, perturbation=1e-6, recurrence_tolerance=1e-8,
                fixed_point_tolerance=1e-10, max_period=16, lyapunov_horizon=0, delay_depth=0,
                intervention_protocol="none", parameters={}, protocol={})
    base.update(kw)
    return ExperimentSpec(**base)


def _finite(x) -> bool:
    return bool(np.all(np.isfinite(np.asarray(x, dtype=np.float64))))


# ===========================================================================
# E1. Exact cubic control (+ Wen & Lu fibre response capacity and design)
# ===========================================================================

CUBIC = _spec(
    name="cubic-control", model_family="cubic-moment-control", dimension=3, width=12, dev_worlds=4, qual_worlds=16,
    horizon=3, perturbation=0.05, recurrence_tolerance=1e-9, fixed_point_tolerance=1e-12, max_period=1,
    coarse_observable="S3=(m1,m2,m3)", target="INSTANTANEOUS_OUTPUT",
    intervention_protocol="fibre-null-space+gauss-newton-retraction",
    parameters={"eta": 0.1, "probes": 16, "weight_scale": 0.5, "rank_rtol": 1e-8, "retraction_steps": 6,
                "interventions": 24, "design_fraction": 0.5},
    protocol={"preservation_tolerance": 1e-9, "divergence_threshold": 1e-6},
)

CUBIC_RULE = {
    "hypothesis": ("For the cubic family, S3 is sufficient for INSTANTANEOUS_OUTPUT and not for ONE_STEP_TRANSITION; "
                   "the S3 fibre has zero response capacity for the instantaneous output and positive capacity for the "
                   "transition target."),
    "exactness_max_relative_error": 1e-10,
    "capacity_S3_instantaneous": 0, "capacity_S2_instantaneous_min": 1, "capacity_S3_transition_min": 1,
    "pte_coarse_max": 1e-12, "pte_output_max": 1e-10, "pte_transition_min": 1e-6,
    "s3_intervention_output_divergence_max": 1e-8, "s3_intervention_transition_fraction_min": 0.5,
    "design_relative_error_max": 0.1,
}


def _cubic_world(spec: ExperimentSpec, world: str) -> dict:
    p, q = spec.parameters, spec.protocol
    rng = world_rng(spec, world, "cubic")
    d, n = spec.dimension, spec.width
    W = p["weight_scale"] * rng.standard_normal((d, n))
    X = rng.standard_normal((p["probes"], d))
    eta, k = p["eta"], spec.horizon
    micro = cubic.forward_microscopic(W, X)
    contracted = cubic.forward_contracted(cubic.raw_moments(W), X)
    out = {"exactness_relative_error": float(np.max(np.abs(micro - contracted)) / max(1.0, np.max(np.abs(micro))))}
    targets = {
        "INSTANTANEOUS_OUTPUT": (lambda M: cubic.forward_microscopic(M, X), lambda M: cubic.output_jacobian(M, X)),
        "ONE_STEP_TRANSITION": (lambda M: cubic.transition_coarse_target(M, eta),
                                lambda M: cubic.transition_coarse_jacobian(M, eta)),
        "K_STEP_TRAJECTORY": (lambda M: cubic.trajectory_output_target(M, eta, X, k),
                              lambda M: cubic.trajectory_output_jacobian(M, eta, X, k)),
    }
    fibre = {}
    for order in (1, 2, 3):
        ns = null_space(cubic.moment_jacobian(W, order), p["rank_rtol"] * 1e-2)
        row = dict(ns.summary())
        for tname, (_, jac) in targets.items():
            g, _ = response_capacity(jac(W), ns.basis, p["rank_rtol"])
            row[tname] = {"capacity": g, "prediction_loss": prediction_loss(jac(W), ns.basis)}
        fibre[f"S{order}"] = row
    out["fibre"] = fibre
    out["loss_monotone"] = all(fibre["S1"][t]["prediction_loss"] + 1e-12 >= fibre["S2"][t]["prediction_loss"]
                               >= fibre["S3"][t]["prediction_loss"] - 1e-12 for t in targets)
    out["capacity_monotone"] = all(fibre["S1"][t]["capacity"] >= fibre["S2"][t]["capacity"] >= fibre["S3"][t]["capacity"]
                                   for t in targets)
    # Coarse-preserving interventions on the rule, S2 and S3 fibres.
    shape = W.shape
    z = W.ravel()
    interventions = {}
    for order in (2, 3):
        co = (lambda o: lambda v: cubic.moment_coordinates(v.reshape(shape), o))(order)
        ja = (lambda o: lambda v: cubic.moment_jacobian(v.reshape(shape), o))(order)
        acc, rep = coarse_preserving_interventions(z, co, ja, rng, magnitude=spec.perturbation,
                                                   count=p["interventions"], tolerance=q["preservation_tolerance"],
                                                   retraction_steps=p["retraction_steps"], rtol=p["rank_rtol"] * 1e-2)
        row = {"report": rep}
        for tname, (fn, _) in targets.items():
            row[tname] = divergence_statistics(fn(W), [fn(a.reshape(shape)) for a in acc], q["divergence_threshold"])
        interventions[f"S{order}"] = row
    out["interventions"] = interventions
    # Exact Prouhet-Tarry-Escott witnesses built on this world's rule.
    wa, wb = cubic.pte_collision(W, 3)
    wa2, wb2 = cubic.pte_collision(W, 2)
    out["pte"] = {
        "S3_coarse_difference": float(np.max(np.abs(cubic.moment_coordinates(wa, 3) - cubic.moment_coordinates(wb, 3)))),
        "S3_output_difference": float(np.max(np.abs(cubic.forward_microscopic(wa, X) - cubic.forward_microscopic(wb, X)))),
        "S3_transition_difference": float(np.max(np.abs(cubic.transition_coarse_target(wa, eta)
                                                        - cubic.transition_coarse_target(wb, eta)))),
        "S2_coarse_difference": float(np.max(np.abs(cubic.moment_coordinates(wa2, 2) - cubic.moment_coordinates(wb2, 2)))),
        "S2_output_difference": float(np.max(np.abs(cubic.forward_microscopic(wa2, X)
                                                    - cubic.forward_microscopic(wb2, X)))),
    }
    # Fibre design (after Wen & Lu, "Designed oscillations"): reach a prescribed first-order change of the
    # transition target inside the S3 fibre, then retract onto the fibre and measure what was realized.
    co3 = lambda v: cubic.moment_coordinates(v.reshape(shape), 3)
    ja3 = lambda v: cubic.moment_jacobian(v.reshape(shape), 3)
    ns3 = null_space(ja3(z), p["rank_rtol"] * 1e-2)
    fn_t, jac_t = targets["ONE_STEP_TRANSITION"]
    R = jac_t(W) @ ns3.basis
    u, s, _ = np.linalg.svd(R, full_matrices=False)
    wanted = p["design_fraction"] * spec.perturbation * s[0] * u[:, 0]
    coeff, *_ = np.linalg.lstsq(R, wanted, rcond=None)
    designed = retract(z + ns3.basis @ coeff, co3, ja3, co3(z), p["retraction_steps"])
    realized = fn_t(designed.reshape(shape)) - fn_t(W)
    out["design"] = {
        "budget_l2": float(np.linalg.norm(ns3.basis @ coeff)),
        "target_change_norm": float(np.linalg.norm(wanted)),
        "relative_error": float(np.linalg.norm(realized - wanted) / np.linalg.norm(wanted)),
        "coarse_residual": float(np.linalg.norm(co3(designed) - co3(z)) / max(1.0, np.linalg.norm(co3(z)))),
        "instantaneous_output_change": float(np.max(np.abs(cubic.forward_microscopic(designed.reshape(shape), X)
                                                           - cubic.forward_microscopic(W, X)))),
    }
    return out


def _cubic_dev(spec, context):
    worlds = [_cubic_world(spec, w) for w in world_ids(spec, "DEV")]
    return {}, {"worlds": worlds}


def _cubic_qual(spec, choices, context):
    rule = CUBIC_RULE
    worlds = [_cubic_world(spec, w) for w in world_ids(spec, "QUAL")]
    checks = {
        "all_finite": all(_finite(w["exactness_relative_error"]) for w in worlds),
        "interventions_accepted": all(w["interventions"]["S3"]["report"]["accepted"] > 0 for w in worlds),
        "rejections_counted": all(w["interventions"][o]["report"]["accepted"] + w["interventions"][o]["report"]["rejected"]
                                  == w["interventions"][o]["report"]["attempted"] for w in worlds for o in ("S2", "S3")),
    }
    passes = {
        "exactness": max(w["exactness_relative_error"] for w in worlds) <= rule["exactness_max_relative_error"],
        "capacity_S3_instantaneous": all(w["fibre"]["S3"]["INSTANTANEOUS_OUTPUT"]["capacity"]
                                         == rule["capacity_S3_instantaneous"] for w in worlds),
        "capacity_S2_instantaneous": all(w["fibre"]["S2"]["INSTANTANEOUS_OUTPUT"]["capacity"]
                                         >= rule["capacity_S2_instantaneous_min"] for w in worlds),
        "capacity_S3_transition": all(w["fibre"]["S3"]["ONE_STEP_TRANSITION"]["capacity"]
                                      >= rule["capacity_S3_transition_min"] for w in worlds),
        "monotone": all(w["loss_monotone"] and w["capacity_monotone"] for w in worlds),
        "pte": all(w["pte"]["S3_coarse_difference"] <= rule["pte_coarse_max"]
                   and w["pte"]["S3_output_difference"] <= rule["pte_output_max"]
                   and w["pte"]["S3_transition_difference"] >= rule["pte_transition_min"] for w in worlds),
        "s3_interventions": all(
            (w["interventions"]["S3"]["INSTANTANEOUS_OUTPUT"]["relative_max"] or 0.0)
            <= rule["s3_intervention_output_divergence_max"]
            and (w["interventions"]["S3"]["ONE_STEP_TRANSITION"]["fraction_above_threshold"] or 0.0)
            >= rule["s3_intervention_transition_fraction_min"] for w in worlds),
        "design": all(w["design"]["relative_error"] <= rule["design_relative_error_max"] for w in worlds),
    }
    regime = f"cubic phi family, d={spec.dimension}, N={spec.width}, eta={spec.parameters['eta']}, {len(worlds)} QUAL worlds"
    s2_div = min(w["interventions"]["S2"]["INSTANTANEOUS_OUTPUT"]["relative_max"] or 0.0 for w in worlds)
    findings = (
        SufficiencyFinding("S3=(m1,m2,m3)", "INSTANTANEOUS_OUTPUT", regime,
                           "NO_COUNTEREXAMPLE_UNDER_REGIME" if passes["exactness"] and passes["capacity_S3_instantaneous"]
                           else "COUNTEREXAMPLE_FOUND", "ALGEBRAIC_IDENTITY",
                           {"max_relative_error": max(w["exactness_relative_error"] for w in worlds)}),
        SufficiencyFinding("S2=(m1,m2)", "INSTANTANEOUS_OUTPUT", regime, "COUNTEREXAMPLE_FOUND"
                           if all(w["pte"]["S2_output_difference"] > 1e-6 for w in worlds) else "NOT_TESTED",
                           "ALGEBRAIC_IDENTITY", {"pte2_min_output_difference": min(w["pte"]["S2_output_difference"]
                                                                                   for w in worlds),
                                                  "fibre_intervention_min_divergence": s2_div}),
        SufficiencyFinding("S3=(m1,m2,m3)", "ONE_STEP_TRANSITION", regime, "COUNTEREXAMPLE_FOUND"
                           if passes["pte"] else "NOT_TESTED", "ALGEBRAIC_IDENTITY",
                           {"pte3_min_transition_difference": min(w["pte"]["S3_transition_difference"] for w in worlds)}),
        SufficiencyFinding("S3=(m1,m2,m3)", "K_STEP_TRAJECTORY", regime, "COUNTEREXAMPLE_FOUND"
                           if all((w["interventions"]["S3"]["K_STEP_TRAJECTORY"]["fraction_above_threshold"] or 0) > 0
                                  for w in worlds) else "NO_COUNTEREXAMPLE_UNDER_REGIME",
                           "INTERVENTION_SUPPORTED_RESULT",
                           {"min_fraction_above_threshold": min(w["interventions"]["S3"]["K_STEP_TRAJECTORY"]
                                                                ["fraction_above_threshold"] or 0 for w in worlds)}),
    )
    return {"worlds": worlds, "passes": passes}, findings, checks, all(passes.values())


# ===========================================================================
# E2. Gain x gamma sweep of the tanh probe (descriptive)
# ===========================================================================

SWEEP = _spec(
    name="tanh-phase-sweep", model_family="tanh-rnn-nonreciprocal", dimension=32, dev_worlds=3, qual_worlds=6,
    warmup=1000, horizon=400, perturbation=1e-6, max_period=16, lyapunov_horizon=300,
    coarse_observable="projection[0, 1]", target="ATTRACTOR_CLASS", intervention_protocol="none",
    parameters={"gains": [0.5, 0.8, 1.0, 1.2, 1.6, 2.0, 3.0], "gammas": [0.0, 0.5, 1.0, 2.0, 4.0],
                "initial_conditions": 6, "sweep_lambda": 1e-3, "bias_scale": 0.0},
)


def _sweep_choices(cells) -> dict:
    """Regime cells for later experiments, chosen on DEV cells only."""
    unresolved = [c for c in cells if c["gamma"] == 1.0 and c["unresolved_fraction"] >= 0.9
                  and c["growth_positive_fraction"] >= 0.9]
    chosen = min(unresolved, key=lambda c: c["gain"]) if unresolved else max(
        cells, key=lambda c: (c["unresolved_fraction"], c["growth_mean"]))
    multistable = max(cells, key=lambda c: (c["attractor_count_proxy_mean"], -c["unresolved_fraction"]))
    stable = [c for c in cells if c["gamma"] == 1.0 and c["fixed_fraction"] == 1.0]
    return {"unresolved_cell": {"gain": chosen["gain"], "gamma": chosen["gamma"]},
            "multistable_cell": {"gain": multistable["gain"], "gamma": multistable["gamma"]},
            "stable_cell": {"gain": max(c["gain"] for c in stable), "gamma": 1.0} if stable else {"gain": 0.5, "gamma": 1.0}}


def _sweep_dev(spec, context):
    cells = sweep(spec, world_ids(spec, "DEV"))
    return _sweep_choices(cells), {"cells": cells}


def _sweep_qual(spec, choices, context):
    cells = sweep(spec, world_ids(spec, "QUAL"))
    checks = {"fractions_sum_to_one": all(abs(c["fixed_fraction"] + c["periodic_fraction"] + c["unresolved_fraction"]
                                              - 1.0) < 1e-9 for c in cells),
              "all_finite": all(_finite([c["growth_mean"], c["norm_mean"]]) for c in cells)}
    return {"cells": cells}, (), checks, None


# ===========================================================================
# E3. Sampled attractors and the ATTRACTOR_CLASS target
# ===========================================================================

ATTRACTORS = _spec(
    name="tanh-attractors", model_family="tanh-rnn-nonreciprocal", dimension=32, dev_worlds=3, qual_worlds=8,
    warmup=1000, horizon=400, perturbation=1e-4, max_period=16, lyapunov_horizon=300,
    coarse_observable="projection[0, 1]", target="ATTRACTOR_CLASS", intervention_protocol="exact-linear-null-space",
    parameters={"initial_conditions": 24, "pairs": 24, "pair_magnitude": 1.0, "match_tol": 1e-6,
                "reference_cells": [{"gain": 0.9, "gamma": 0.0}, {"gain": 2.5, "gamma": 4.0},
                                    {"gain": 2.0, "gamma": 1.0}]},
    protocol={"preservation_tolerance": 1e-12},
)

ATTRACTOR_RULE = {
    "hypothesis": ("In the DEV-selected multistable cell, the initial coarse state projection[0,1] is not sufficient "
                   "for ATTRACTOR_CLASS: coarse-matched initial states reach different resolved attractors."),
    "worlds_with_split_pair_fraction_min": 0.5,
}


def _attractor_world(spec, world, cell, pairs: bool) -> dict:
    p = spec.parameters
    system, rng = world_system(spec, world, cell["gain"], cell["gamma"], stream="attractors")
    inv = AttractorInventory(warmup=spec.warmup, tail=spec.horizon, fixed_tol=spec.fixed_point_tolerance,
                             recurrence_tol=spec.recurrence_tolerance, max_period=spec.max_period,
                             match_tol=p["match_tol"])
    for _ in range(p["initial_conditions"]):
        inv.label(system, system.initial_state(rng))
    inv.add_recovery(system, rng, spec.perturbation)
    out = {"cell": cell, "inventory": inv.summary(p["initial_conditions"])}
    if pairs:
        C = coarse_map(spec.coarse_observable)
        split = both = 0
        residuals = []
        for _ in range(p["pairs"]):
            x0 = system.initial_state(rng)
            acc, rep = coarse_preserving_interventions(x0, C, C.jacobian, rng, magnitude=p["pair_magnitude"], count=1,
                                                       tolerance=spec.protocol["preservation_tolerance"])
            if not acc:
                continue
            residuals.append(rep["residual_max"])
            a, b = inv.label(system, x0, register=False), inv.label(system, acc[0], register=False)
            if a == -3 or b == -3:  # resolved but new: register both to compare identities
                a, b = inv.label(system, x0), inv.label(system, acc[0])
            if a >= 0 and b >= 0:
                both += 1
                split += int(a != b)
        out["pairs"] = {"resolved_pairs": both, "split_pairs": split,
                        "max_preservation_residual": max(residuals) if residuals else None}
    return out


def _attractor_dev(spec, context):
    cell = context["tanh-phase-sweep"]["multistable_cell"]
    worlds = [_attractor_world(spec, w, cell, True) for w in world_ids(spec, "DEV")]
    return {"multistable_cell": cell}, {"worlds": worlds}


def _attractor_qual(spec, choices, context):
    cell = choices["multistable_cell"]
    ws = world_ids(spec, "QUAL")
    main = [_attractor_world(spec, w, cell, True) for w in ws]
    refs = [[_attractor_world(spec, w, c, False) for w in ws] for c in spec.parameters["reference_cells"]]
    frac = float(np.mean([m["pairs"]["split_pairs"] > 0 for m in main]))
    checks = {"preservation_exact": all((m["pairs"]["max_preservation_residual"] or 0.0) <= 1e-12 for m in main),
              "resolved_pairs_exist": sum(m["pairs"]["resolved_pairs"] for m in main) > 0}
    regime = f"tanh probe d={spec.dimension} gain={cell['gain']} gamma={cell['gamma']}, {len(ws)} QUAL worlds"
    findings = (SufficiencyFinding(
        "C_0=projection[0,1] of the initial state", "ATTRACTOR_CLASS", regime,
        "COUNTEREXAMPLE_FOUND" if any(m["pairs"]["split_pairs"] for m in main) else "NO_COUNTEREXAMPLE_UNDER_REGIME",
        "INTERVENTION_SUPPORTED_RESULT", {"worlds_with_split_pair_fraction": frac,
                                          "split_pairs": sum(m["pairs"]["split_pairs"] for m in main),
                                          "resolved_pairs": sum(m["pairs"]["resolved_pairs"] for m in main)}),)
    return ({"main": main, "reference_cells": refs, "worlds_with_split_pair_fraction": frac}, findings, checks,
            frac >= ATTRACTOR_RULE["worlds_with_split_pair_fraction_min"])


# ===========================================================================
# E4. Delay / memory with the Lozano-Duran reference-penalty-gain surrogate
# ===========================================================================

DELAY = _spec(
    name="tanh-delay-ladder", model_family="tanh-rnn-nonreciprocal", dimension=32, dev_worlds=4, qual_worlds=8,
    warmup=500, horizon=600, delay_depth=4, coarse_observable="projection[0, 1]", target="ONE_STEP_TRANSITION",
    intervention_protocol="none",
    parameters={"targets": ["ONE_STEP_TRANSITION", "K_STEP_TRAJECTORY", "EVENT"], "k_horizon": 5,
                "initial_conditions": 8, "seed_dim": 8,
                "lambda_grid": [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]},
)

DELAY_RULE = {
    "hypothesis": ("In the DEV-selected unresolved cell, the delay vector (C_t..C_{t-4}) predicts ONE_STEP_TRANSITION "
                   "and K_STEP_TRAJECTORY targets better than C_t alone under the same ridge class, split and metric; "
                   "the gain is not reproduced by a dimension-matched shuffled delay; and independent seed columns do "
                   "not improve on C_t."),
    "median_delay_gain_min": 0.05, "delay_beats_shuffled_fraction_min": 0.75, "seed_gain_median_max": 0.01,
    "continuous_targets": ["ONE_STEP_TRANSITION", "K_STEP_TRAJECTORY"],
}


def _delay_rows(spec, world, cell, target):
    p = spec.parameters
    system, rng = world_system(spec, world, cell["gain"], cell["gamma"], stream="delay")
    C = coarse_map(spec.coarse_observable)
    proj = delay.random_projection(world_rng(spec, world, "projection"), spec.dimension, len(C(np.zeros(spec.dimension))),
                                   spec.delay_depth)
    rows = []
    for i in range(p["initial_conditions"]):
        x = run(system, system.initial_state(rng), spec.warmup)
        traj = rollout(system, x, spec.horizon - 1)
        coarse = np.array([C(s) for s in traj])
        erng = world_rng(spec, world, f"extras:{i}")
        extras = delay.trajectory_extras(erng, len(traj), spec.delay_depth, p["seed_dim"], proj)
        rows.append(delay.make_rows(traj, coarse, depth=spec.delay_depth, horizon=p["k_horizon"], target=target,
                                    extras=extras))
    half = p["initial_conditions"] // 2
    return delay.stack(rows[:half]), delay.stack(rows[half:])


def _delay_dev(spec, context):
    cell = context["tanh-phase-sweep"]["unresolved_cell"]
    lambdas, tables = {}, {}
    for target in spec.parameters["targets"]:
        splits = [_delay_rows(spec, w, cell, target) for w in world_ids(spec, "DEV")]
        for rep in delay.REPRESENTATIONS:
            lam, table = delay.choose_lambda(splits, rep, target, spec.parameters["lambda_grid"])
            lambdas[f"{target}/{rep}"] = lam
            tables[f"{target}/{rep}"] = table
    return {"cell": cell, "lambdas": lambdas}, {"validation_tables": tables}


def _delay_qual(spec, choices, context):
    cell, lambdas = choices["cell"], choices["lambdas"]
    per_target = {}
    for target in spec.parameters["targets"]:
        scores, ladders, monotone = [], [], []
        for w in world_ids(spec, "QUAL"):
            train, test = _delay_rows(spec, w, cell, target)
            scores.append({rep: delay.score(train, test, rep, lambdas[f"{target}/{rep}"], target)
                           for rep in delay.REPRESENTATIONS})
            if target != "EVENT":
                ladders.append(delay.ladder_decomposition(train, test, {r: lambdas[f"{target}/{r}"]
                                                                        for r in ("full", "coarse", "delay")}))
                small, big = delay.insample_least_squares_monotone(train["coarse"], train["delay"], train["_target"])
                monotone.append(big <= small * (1 + 1e-12))
        gains = [s["delay"] - s["coarse"] for s in scores]
        per_target[target] = {
            "scores": scores,
            "median_delay_gain": float(np.median(gains)),
            "delay_beats_shuffled_fraction": float(np.mean([s["delay"] > s["shuffled_delay"] for s in scores])),
            "median_seed_gain": float(np.median([s["seed"] - s["coarse"] for s in scores])),
            "median_scores": {rep: float(np.median([s[rep] for s in scores])) for rep in delay.REPRESENTATIONS},
            "ladder": ladders, "insample_monotone": monotone,
        }
    rule = DELAY_RULE
    passes = {t: per_target[t]["median_delay_gain"] >= rule["median_delay_gain_min"]
              and per_target[t]["delay_beats_shuffled_fraction"] >= rule["delay_beats_shuffled_fraction_min"]
              and per_target[t]["median_seed_gain"] <= rule["seed_gain_median_max"] for t in rule["continuous_targets"]}
    checks = {"same_split_all_representations": True,  # enforced by construction: one (train, test) per world
              "all_finite": all(_finite(list(s.values())) for t in per_target.values() for s in t["scores"]),
              "ladder_identity": all(l["identity_residual"] < 1e-9 for t in per_target.values() for l in t["ladder"]),
              "insample_monotone": all(all(t["insample_monotone"]) for t in per_target.values())}
    dims = {"weak": 1, "coarse": 2, "delay": 2 * (spec.delay_depth + 1), "shuffled_delay": 2 * (spec.delay_depth + 1),
            "random_projection": 2 * (spec.delay_depth + 1), "seed": 2 + spec.parameters["seed_dim"],
            "full": spec.dimension}
    return {"cell": cell, "per_target": per_target, "passes": passes, "representation_dimensions": dims}, (), checks, \
        all(passes.values())


# ===========================================================================
# E5. Coarse-preserving state interventions: trajectory vs long-run vs response
# ===========================================================================

INTERVENE = _spec(
    name="tanh-intervention", model_family="tanh-rnn-nonreciprocal", dimension=32, dev_worlds=3, qual_worlds=8,
    warmup=500, horizon=10, perturbation=0.05, coarse_observable="projection[0, 1]", target="K_STEP_TRAJECTORY",
    intervention_protocol="null-space (exact for linear C; Gauss-Newton retraction otherwise)",
    parameters={"coarse_maps": ["projection[0, 1]", "mean+second_moment"], "interventions": 16,
                "retraction_steps": 6, "long_run_offset": 300, "long_run_window": 2000, "long_run_samples": 4, "pulse_amplitude": 1.0,
                "input_dimension": 1},
    protocol={"preservation_tolerance": 1e-9, "trajectory_threshold": 1e-3, "response_threshold": 1e-3,
              "long_run_floor_ratio_max": 2.0, "long_run_absolute_floor": 1e-12},
)

INTERVENE_RULE = {
    "hypothesis": ("In the DEV-selected unresolved cell, coarse-preserving interventions change the K-step coarse "
                   "trajectory (fraction above threshold >= 0.5) while the long-run coarse statistic stays within "
                   "twice the within-trajectory sampling floor (median over 4 disjoint windows; median ratio <= 2)."),
    "trajectory_fraction_min": 0.5, "long_run_median_ratio_max": 2.0,
}


def _long_run(system, x, C, offset, window):
    x = run(system, x, offset)
    vals = []
    for _ in range(window):
        x = system.step(x)
        vals.append(C(x))
    return np.mean(vals, axis=0)


def _intervention_world(spec, world, cell) -> dict:
    p, q = spec.parameters, spec.protocol
    system, rng = world_system(spec, world, cell["gain"], cell["gamma"], stream="intervention")
    x = run(system, system.initial_state(rng), spec.warmup)
    out = {}
    for cname in p["coarse_maps"]:
        C = coarse_map(cname)
        acc, rep = coarse_preserving_interventions(x, C, C.jacobian, rng, magnitude=spec.perturbation,
                                                   count=p["interventions"], tolerance=q["preservation_tolerance"],
                                                   retraction_steps=0 if C.linear else p["retraction_steps"])
        traj = lambda s: np.concatenate([C(v) for v in rollout(system, s, spec.horizon)[1:]])
        pulse = np.zeros((spec.horizon, system.input_dimension))
        pulse[0, :] = p["pulse_amplitude"]

        def response(s):
            with_pulse = rollout(system, s, spec.horizon, inputs=pulse)[1:]
            without = rollout(system, s, spec.horizon)[1:]
            return np.concatenate([C(a) - C(b) for a, b in zip(with_pulse, without)])
        lr_ref = _long_run(system, x, C, p["long_run_offset"], p["long_run_window"])
        # Sampling floor: the same statistic over later disjoint windows of the unperturbed trajectory.
        floors = [float(np.linalg.norm(_long_run(system, run(system, x, j * p["long_run_window"]), C,
                                                 p["long_run_offset"], p["long_run_window"]) - lr_ref))
                  for j in range(1, p["long_run_samples"] + 1)]
        floor = float(np.median(floors))
        lr = [float(np.linalg.norm(_long_run(system, a, C, p["long_run_offset"], p["long_run_window"]) - lr_ref))
              for a in acc[:p["long_run_samples"]]]
        out[cname] = {
            "report": rep,
            "K_STEP_TRAJECTORY": divergence_statistics(traj(x), [traj(a) for a in acc], q["trajectory_threshold"]),
            "INTERVENTION_RESPONSE": divergence_statistics(response(x), [response(a) for a in acc],
                                                           q["response_threshold"]),
            "LONG_RUN_STATISTIC": {"floor": floor, "floor_samples": floors, "differences": lr,
                                   "median_ratio": (float(np.median(lr) / max(floor, q["long_run_absolute_floor"]))
                                                    if lr else None)},
        }
    return out


def _intervention_dev(spec, context):
    cells = context["tanh-phase-sweep"]
    worlds = {name: [_intervention_world(spec, w, cells[name]) for w in world_ids(spec, "DEV")]
              for name in ("unresolved_cell", "stable_cell")}
    return {"unresolved_cell": cells["unresolved_cell"], "stable_cell": cells["stable_cell"]}, {"worlds": worlds}


def _intervention_qual(spec, choices, context):
    ws = world_ids(spec, "QUAL")
    main = [_intervention_world(spec, w, choices["unresolved_cell"]) for w in ws]
    stable = [_intervention_world(spec, w, choices["stable_cell"]) for w in ws]
    cmain = spec.coarse_observable
    traj_frac = float(np.median([m[cmain]["K_STEP_TRAJECTORY"]["fraction_above_threshold"] for m in main]))
    ratios = [m[cmain]["LONG_RUN_STATISTIC"]["median_ratio"] for m in main
              if m[cmain]["LONG_RUN_STATISTIC"]["median_ratio"] is not None]
    lr_ratio = float(np.median(ratios)) if ratios else None
    resp_frac = float(np.median([m[cmain]["INTERVENTION_RESPONSE"]["fraction_above_threshold"] for m in main]))
    checks = {"accepted": all(m[c]["report"]["accepted"] > 0 for m in main for c in spec.parameters["coarse_maps"]),
              "residuals_within_tolerance": all((m[c]["report"]["residual_max"] or 0.0)
                                                <= spec.protocol["preservation_tolerance"]
                                                for m in main + stable for c in spec.parameters["coarse_maps"])}
    cell = choices["unresolved_cell"]
    regime = f"tanh probe d={spec.dimension} gain={cell['gain']} gamma={cell['gamma']}, {len(ws)} QUAL worlds"
    findings = (
        SufficiencyFinding(f"C_t={cmain}", "K_STEP_TRAJECTORY", regime,
                           "COUNTEREXAMPLE_FOUND" if traj_frac > 0 else "NO_COUNTEREXAMPLE_UNDER_REGIME",
                           "INTERVENTION_SUPPORTED_RESULT", {"median_fraction_above_threshold": traj_frac}),
        SufficiencyFinding(f"C_t={cmain}", "INTERVENTION_RESPONSE", regime,
                           "COUNTEREXAMPLE_FOUND" if resp_frac > 0 else "NO_COUNTEREXAMPLE_UNDER_REGIME",
                           "INTERVENTION_SUPPORTED_RESULT", {"median_fraction_above_threshold": resp_frac}),
        SufficiencyFinding(f"C_t={cmain}", "LONG_RUN_STATISTIC", regime,
                           "NOT_TESTED" if lr_ratio is None else
                           "NO_COUNTEREXAMPLE_UNDER_REGIME" if lr_ratio <= spec.protocol["long_run_floor_ratio_max"]
                           else "COUNTEREXAMPLE_FOUND", "INTERVENTION_SUPPORTED_RESULT",
                           {"median_ratio_to_sampling_floor": lr_ratio}),
    )
    passed = (traj_frac >= INTERVENE_RULE["trajectory_fraction_min"] and lr_ratio is not None
              and lr_ratio <= INTERVENE_RULE["long_run_median_ratio_max"])
    return {"unresolved": main, "stable": stable, "summary": {"trajectory_fraction_median": traj_frac,
                                                              "long_run_ratio_median": lr_ratio,
                                                              "response_fraction_median": resp_frac}}, \
        findings, checks, passed


# ===========================================================================
# E6. Vaidya: critical feedback (Eqs. 4.5-4.6) and a static-feedback probe
# ===========================================================================

FEEDBACK = _spec(
    name="feedback-transition", model_family="tanh-rnn-nonreciprocal", dimension=200, dev_worlds=4, qual_worlds=8,
    warmup=1000, horizon=300, lyapunov_horizon=300, coarse_observable="none (full-state diagnostics)",
    target="LONG_RUN_STATISTIC", intervention_protocol="none",
    parameters={"gains": [1.3, 2.0], "factors": [0.0, 0.5, 0.75, 1.0, 1.33, 2.0], "gamma": 1.0,
                "paper_gain": 1.3, "paper_yhat_c": 0.2, "paper_tolerance": 0.01},
)

FEEDBACK_RULE = {
    "hypothesis": ("(a) Eqs. (4.5)-(4.6) of arXiv:2609.19288 give yhat_c = 0.2 +- 0.01 at g = 1.3. (b) For each "
                   "DEV-resolvable gain, the discrete static-feedback probe has positive mean finite-time growth at "
                   "0.5 yhat_c and negative at 2 yhat_c, and the fixed-point fraction rises by >= 0.5 between them."),
    "fixed_fraction_rise_min": 0.5,
}


def _feedback_cell(spec, world, g, yhat) -> dict:
    rng = world_rng(spec, world, f"feedback:g={g!r}:yhat={yhat!r}")
    s = make_tanh_rnn(rng, spec.dimension, g, spec.parameters["gamma"], bias_scale=yhat)
    x = run(s, s.initial_state(rng), spec.warmup)
    tail = rollout(s, x, spec.horizon - 1)
    c = classify_tail(tail, fixed_tol=spec.fixed_point_tolerance, recurrence_tol=spec.recurrence_tolerance,
                      max_period=spec.max_period)
    return {"class": c["class"], "growth": finite_time_growth(s, x, rng, spec.lyapunov_horizon)}


def _feedback_table(spec, worlds, gains):
    table = []
    for g in gains:
        yc = feedback.critical_feedback(g)["yhat_c"]
        for f in spec.parameters["factors"]:
            cells = [_feedback_cell(spec, w, g, f * yc) for w in worlds]
            gr = np.array([c["growth"] for c in cells])
            table.append({"gain": g, "factor": f, "yhat": f * yc, "yhat_c": yc,
                          "growth_mean": float(gr.mean()), "growth_se": float(gr.std(ddof=1) / np.sqrt(len(gr))),
                          "fixed_fraction": float(np.mean([c["class"] == "FIXED_POINT" for c in cells])),
                          "unresolved_fraction": float(np.mean([c["class"] == "UNRESOLVED_NONPERIODIC" for c in cells])),
                          "meanfield_log_norm_rate": feedback.meanfield_growth(g, f * yc)["log_norm_rate"]})
    return table


def _row(table, g, f):
    return next(r for r in table if r["gain"] == g and r["factor"] == f)


def _feedback_dev(spec, context):
    table = _feedback_table(spec, world_ids(spec, "DEV"), spec.parameters["gains"])
    resolvable = []
    for g in spec.parameters["gains"]:
        lo, hi = _row(table, g, 0.5), _row(table, g, 2.0)
        if lo["growth_mean"] > 2 * lo["growth_se"] and hi["growth_mean"] < -2 * hi["growth_se"]:
            resolvable.append(g)
    return {"resolvable_gains": resolvable}, {"table": table}


def _feedback_qual(spec, choices, context):
    p = spec.parameters
    crit = feedback.critical_feedback(p["paper_gain"])
    table = _feedback_table(spec, world_ids(spec, "QUAL"), p["gains"])
    analytic = abs(crit["yhat_c"] - p["paper_yhat_c"]) <= p["paper_tolerance"]
    probe = {}
    for g in choices["resolvable_gains"]:
        lo, hi = _row(table, g, 0.5), _row(table, g, 2.0)
        probe[str(g)] = (lo["growth_mean"] > 0 > hi["growth_mean"]
                         and hi["fixed_fraction"] - lo["fixed_fraction"] >= FEEDBACK_RULE["fixed_fraction_rise_min"])
    crossings = {}
    for g in p["gains"]:
        rows = [_row(table, g, f) for f in p["factors"]]
        cross = None
        for a, b in zip(rows, rows[1:]):
            if a["growth_mean"] > 0 >= b["growth_mean"]:
                cross = a["factor"] + (b["factor"] - a["factor"]) * a["growth_mean"] / (a["growth_mean"] - b["growth_mean"])
                break
        crossings[str(g)] = cross
    checks = {"all_finite": all(_finite([r["growth_mean"]]) for r in table), "chi_at_u_c": abs(crit["chi_at_u_c"] - 1) < 1e-9}
    passed = analytic and bool(probe) and all(probe.values())
    return {"critical": crit, "analytic_reproduction": analytic, "table": table, "probe_passes": probe,
            "interpolated_crossing_factor": crossings}, (), checks, passed


# ===========================================================================
# E7. Aguilera & De Martino: Hopf line, alpha=0 map, small-N eigenphase comparison
# ===========================================================================

ASSOC = _spec(
    name="associative-cycles", model_family="nonreciprocal-associative", dimension=2000, width=2, dev_worlds=3,
    qual_worlds=8, horizon=400, fixed_point_tolerance=1e-12, recurrence_tolerance=1e-9, max_period=64,
    coarse_observable="target overlaps m_t", target="ATTRACTOR_CLASS", intervention_protocol="none",
    parameters={"delta": 0.1, "last": 100, "candidates": [[0.2, 3.0], [0.2, 6.0], [0.25, 3.0]],
                "alphas": [0.02, 0.05, 0.08, 0.12], "check_phis": [0.0, 0.1, 0.2, 0.3],
                "meanfield_steps": 20000, "meanfield_tail": 2000},
)

ASSOC_RULE = {
    "hypothesis": ("(a) The End-Matter Hopf line beta_c(phi) and frequency omega of arXiv:2609.07341 equal the modulus-one "
                   "point and eigen-argument of the alpha=0 linearization. (b) The alpha=0 map decays to m=0 at "
                   "0.9 beta_c and does not settle to m=0 at 1.1 beta_c. (c) At the DEV-selected (phi, beta, alpha), uniform "
                   "disorder eigenphases give higher retrieval than coherent ones in >= 75% of QUAL realizations, "
                   "with median difference >= 0.1."),
    "uniform_wins_fraction_min": 0.75, "median_difference_min": 0.1, "linearization_tol": 1e-12,
}


def _meanfield_class(phi, beta, spec):
    p = spec.parameters
    s = associative.OverlapMeanField(associative.rotation(phi), beta, p["delta"])
    x = run(s, np.array([0.5, 0.1]), p["meanfield_steps"])
    tail = rollout(s, x, p["meanfield_tail"] - 1)
    c = classify_tail(tail, fixed_tol=spec.fixed_point_tolerance, recurrence_tol=spec.recurrence_tolerance,
                      max_period=spec.max_period)
    return {"class": c["class"], "period": c["period"], "norm_mean": float(np.mean(np.linalg.norm(tail, axis=1)))}


def _retrieval(spec, world, phi, beta, alpha, disorder):
    n = spec.dimension
    # One stream per (world, phi, beta, alpha): uniform and coherent disorder share the target and disorder
    # patterns and differ only in the block eigenphases (a paired comparison).
    rng = world_rng(spec, world, f"assoc:{phi!r}:{beta!r}:{alpha!r}")
    J, xi = associative.couplings(rng, n, phi, int(round(alpha * n / 2)), disorder)
    ov = associative.glauber_overlaps(rng, J, xi, xi[:, 0].copy(), beta=beta, delta=spec.parameters["delta"],
                                      steps=spec.horizon)
    return associative.retrieval_norm(ov, spec.parameters["last"])


def _assoc_dev(spec, context):
    p = spec.parameters
    oscillatory = []
    classes = {}
    for phi_pi, beta in p["candidates"]:
        c = _meanfield_class(phi_pi * np.pi, beta, spec)
        classes[f"{phi_pi}pi/{beta}"] = c
        if c["class"] != "FIXED_POINT" and c["norm_mean"] >= 0.5:
            oscillatory.append((phi_pi, beta))
    table = {}
    for phi_pi, beta in oscillatory:
        for a in p["alphas"]:
            diffs = [_retrieval(spec, w, phi_pi * np.pi, beta, a, "uniform")
                     - _retrieval(spec, w, phi_pi * np.pi, beta, a, "coherent") for w in world_ids(spec, "DEV")]
            table[f"{phi_pi}/{beta}/{a}"] = float(np.median(diffs))
    best = max(table, key=table.get) if table else None
    choice = ({"phi_pi": float(best.split("/")[0]), "beta": float(best.split("/")[1]), "alpha": float(best.split("/")[2])}
              if best else None)
    return {"regime": choice}, {"meanfield_classes": classes, "median_difference_table": table}


def _assoc_qual(spec, choices, context):
    p = spec.parameters
    lin = [associative.linearization_check(f * np.pi, p["delta"]) for f in p["check_phis"]]
    lin_ok = all(abs(l["spectral_radius"] - 1) <= ASSOC_RULE["linearization_tol"]
                 and abs(l["argument"] - l["omega_formula"]) <= ASSOC_RULE["linearization_tol"] for l in lin)
    reg = choices["regime"]
    phi = reg["phi_pi"] * np.pi
    bc = associative.hopf_beta(phi, p["delta"])
    below, above = _meanfield_class(phi, 0.9 * bc, spec), _meanfield_class(phi, 1.1 * bc, spec)
    onset_ok = below["class"] == "FIXED_POINT" and below["norm_mean"] < 1e-6 and above["norm_mean"] > 0.1 \
        and above["class"] != "FIXED_POINT"
    ws = world_ids(spec, "QUAL")
    pairs = [(_retrieval(spec, w, phi, reg["beta"], reg["alpha"], "uniform"),
              _retrieval(spec, w, phi, reg["beta"], reg["alpha"], "coherent")) for w in ws]
    control = [(_retrieval(spec, w, 0.0, reg["beta"], reg["alpha"], "uniform"),
                _retrieval(spec, w, 0.0, reg["beta"], reg["alpha"], "coherent")) for w in ws]
    zero_load = [_retrieval(spec, w, phi, reg["beta"], 0.0, "uniform") for w in ws]
    wins = float(np.mean([u > c for u, c in pairs]))
    med = float(np.median([u - c for u, c in pairs]))
    checks = {"all_finite": _finite(pairs) and _finite(control)}
    passed = lin_ok and onset_ok and wins >= ASSOC_RULE["uniform_wins_fraction_min"] \
        and med >= ASSOC_RULE["median_difference_min"]
    return {"linearization": lin, "linearization_ok": lin_ok, "onset": {"beta_c": bc, "below": below, "above": above,
                                                                        "ok": onset_ok},
            "retrieval_pairs_uniform_coherent": pairs, "phi0_control_uniform_coherent": control,
            "zero_load_retrieval": zero_load, "uniform_wins_fraction": wins, "median_difference": med}, (), checks, passed


# ===========================================================================
# E8. Public ECS: shared topology-analysis coarse state, different futures
# ===========================================================================

ECS = _spec(
    name="ecs-topology-collision", model_family="ecs-topology", dimension=5, dev_worlds=6, qual_worlds=24,
    horizon=2, delay_depth=1, coarse_observable="topology analysis minus topology_digest",
    target="EVENT", intervention_protocol="reorder committed enqueues (same multiset)",
    parameters={"min_entities": 3, "max_entities": 5, "min_messages": 3, "max_messages": 6},
)

ECS_RULE = {
    "hypothesis": ("The topology-analysis coarse state (without its projection-binding digest) is not sufficient for "
                   "the next-processed-edge EVENT nor for the reply-policy K_STEP_TRAJECTORY: the minimal pair and at "
                   "least one QUAL pair share it exactly and diverge; every sampled reordering shares it; reading it never "
                   "changes the kernel state root."),
}


def _ecs_pair(spec, world):
    from . import ecs_collision as E
    p = spec.parameters
    rng = world_rng(spec, world, "ecs")
    n = int(rng.integers(p["min_entities"], p["max_entities"] + 1))
    m = int(rng.integers(p["min_messages"], p["max_messages"] + 1))
    sends = []
    for _ in range(m):
        s = int(rng.integers(n))
        r = int(rng.integers(n - 1))
        sends.append((s, r if r < s else r + 1))
    other = list(sends)
    for _ in range(10):
        other = [sends[i] for i in rng.permutation(m)]
        if other != sends:
            break
    labels = tuple(f"e{i}" for i in range(n))
    return dict(E.compare_pair(labels, sends, other, depth=spec.delay_depth, rounds=spec.horizon), entities=n,
                messages=m, reordered=other != sends)


def _ecs_minimal(spec):
    from . import ecs_collision as E
    labels, a, b = E.MINIMAL_PAIR
    return E.compare_pair(labels, a, b, depth=spec.delay_depth, rounds=spec.horizon)


def _ecs_dev(spec, context):
    return {}, {"minimal_pair": _ecs_minimal(spec), "pairs": [_ecs_pair(spec, w) for w in world_ids(spec, "DEV")]}


def _ecs_qual(spec, choices, context):
    minimal = _ecs_minimal(spec)
    pairs = [_ecs_pair(spec, w) for w in world_ids(spec, "QUAL")]
    reordered = [q for q in pairs if q["reordered"] and q["histories_distinct"]]
    event_div = [q for q in reordered if q["event_differs"]]
    traj_div = [q for q in reordered if q["trajectory_differs"]]
    delay_unresolved = [q for q in event_div if q["delay_equal"]]
    checks = {"analysis_read_only": minimal["analysis_read_only"] and all(q["analysis_read_only"] for q in pairs),
              "coarse_equal_every_reordering": all(q["coarse_equal"] for q in reordered)}
    regime = f"public ECS kernel, scheduler v2, {len(reordered)} reordered QUAL pairs + minimal pair"
    findings = (
        SufficiencyFinding("topology analysis minus topology_digest", "EVENT", regime,
                           "COUNTEREXAMPLE_FOUND" if minimal["event_differs"] and minimal["coarse_equal"]
                           else "NO_COUNTEREXAMPLE_UNDER_REGIME", "INTERVENTION_SUPPORTED_RESULT",
                           {"divergent_pairs": len(event_div), "pairs": len(reordered)}),
        SufficiencyFinding("topology analysis minus topology_digest", "K_STEP_TRAJECTORY", regime,
                           "COUNTEREXAMPLE_FOUND" if minimal["trajectory_differs"] and minimal["coarse_equal"]
                           else "NO_COUNTEREXAMPLE_UNDER_REGIME", "INTERVENTION_SUPPORTED_RESULT",
                           {"divergent_pairs": len(traj_div), "pairs": len(reordered), "reply_rounds": spec.horizon}),
        SufficiencyFinding("weak (node_count, total_message_count)", "EVENT", regime,
                           "COUNTEREXAMPLE_FOUND" if minimal["weak_equal"] and minimal["event_differs"]
                           else "NO_COUNTEREXAMPLE_UNDER_REGIME", "INTERVENTION_SUPPORTED_RESULT", {}),
        SufficiencyFinding(f"delay-augmented (C_t..C_t-{spec.delay_depth}) over history prefixes", "EVENT", regime,
                           "COUNTEREXAMPLE_FOUND" if delay_unresolved else "NO_COUNTEREXAMPLE_UNDER_REGIME",
                           "INTERVENTION_SUPPORTED_RESULT",
                           {"divergent_pairs_not_resolved": len(delay_unresolved), "divergent_pairs": len(event_div)}),
        SufficiencyFinding("full microscopic state (state-root digest)", "EVENT", regime,
                           "COUNTEREXAMPLE_FOUND" if any(q["full_equal"] and q["event_differs"] for q in pairs)
                           else "NO_COUNTEREXAMPLE_UNDER_REGIME", "INTERVENTION_SUPPORTED_RESULT", {}),
    )
    passed = (minimal["coarse_equal"] and minimal["event_differs"] and minimal["trajectory_differs"]
              and bool(event_div) and bool(traj_div) and all(checks.values()))
    return {"minimal_pair": minimal, "pairs": pairs, "reordered_pairs": len(reordered),
            "event_divergent": len(event_div), "trajectory_divergent": len(traj_div),
            "delay_unresolved_divergent": len(delay_unresolved)}, findings, checks, passed


# ===========================================================================
# Registry
# ===========================================================================

_COMMON_LIMITS = ("synthetic regime only", "finite-time, finite-size diagnostics",
                  "no statement about production ECS, trained models or Elpis neural components")

EXPERIMENTS = {e.name: e for e in (
    Experiment("cubic-control", CUBIC, "ALGEBRAIC_IDENTITY",
               ("LAB: cubic identity, PTE witnesses", "arXiv:2609.29834 Eq. (2) response capacity",
                "arXiv:2609.29834 Eq. (3) prediction loss (Euclidean-budget adaptation)",
                "arXiv:2609.29834 'Designed oscillations and their limits' (design within a fibre)"),
               CUBIC_RULE, _cubic_dev, _cubic_qual,
               _COMMON_LIMITS + ("establishes instantaneous-forward sufficiency of S3 only; not transition, trajectory, "
                                 "attractor, intervention or production-ECS sufficiency",)),
    Experiment("tanh-phase-sweep", SWEEP, "DESCRIPTIVE_RESULT",
               ("LAB: nonreciprocal coupling J=(S+gamma A)/sqrt(1+gamma^2)",
                "arXiv:2609.19288 Eq. (2.5) two-time correlation definition"),
               {}, _sweep_dev, _sweep_qual,
               _COMMON_LIMITS + ("classification boundaries are not bifurcations",
                                 "UNRESOLVED_NONPERIODIC includes slow transients and quasi-periodic motion")),
    Experiment("tanh-attractors", ATTRACTORS, "INTERVENTION_SUPPORTED_RESULT",
               ("LAB: conservative attractor inventory",), ATTRACTOR_RULE, _attractor_dev, _attractor_qual,
               _COMMON_LIMITS + ("sampled attractors for the initial conditions tried, not a census",),
               ("tanh-phase-sweep",)),
    Experiment("tanh-delay-ladder", DELAY, "DESCRIPTIVE_RESULT",
               ("arXiv:2609.19424 Eq. (3.11) delay vector", "arXiv:2609.19424 Eq. (3.24) reference-penalty-gain form",
                "arXiv:2609.19424 Eq. (3.44) independent seed adds no information",
                "arXiv:2609.19424 Eqs. (3.17)-(3.18) nested predictors"),
               DELAY_RULE, _delay_dev, _delay_qual,
               _COMMON_LIMITS + ("errors are held-out errors of one linear ridge class: upper-bound surrogates of the "
                                 "paper's irreducible errors, not estimates of them",),
               ("tanh-phase-sweep",)),
    Experiment("tanh-intervention", INTERVENE, "INTERVENTION_SUPPORTED_RESULT",
               ("arXiv:2609.19424 Sec. 3.8 statistics vs trajectories", "arXiv:2609.29834 fibre ker DC (main text)"),
               INTERVENE_RULE, _intervention_dev, _intervention_qual,
               _COMMON_LIMITS + ("long-run comparison uses a within-trajectory sampling floor, not a stationary-law test",),
               ("tanh-phase-sweep",)),
    Experiment("feedback-transition", FEEDBACK, "DESCRIPTIVE_RESULT",
               ("arXiv:2609.19288 Eq. (4.2) quasi-static feedback field", "arXiv:2609.19288 Eqs. (4.5)-(4.6)",
                "arXiv:2609.19288 Sec. 4 and Fig. 1(f): yhat_c = 0.2 at g = 1.3",
                "LAB: discrete-time static-feedback mean field"),
               FEEDBACK_RULE, _feedback_dev, _feedback_qual,
               _COMMON_LIMITS + ("learning dynamics (Eq. 2.4), t_cr and the two-time DMFT solution are not tested",
                                 "discrete-time probe, not the paper's continuous-time network")),
    Experiment("associative-cycles", ASSOC, "DESCRIPTIVE_RESULT",
               ("arXiv:2609.07341 Eqs. (1)-(3) model", "arXiv:2609.07341 End Matter: alpha=0 map, beta_c(phi), omega",
                "arXiv:2609.07341 Fig. 1 and Discussion: eigenphase decoherence and capacity"),
               ASSOC_RULE, _assoc_dev, _assoc_qual,
               _COMMON_LIMITS + ("N = 2000 microscopic networks; the paper uses N = 50,000-500,000",
                                 "no capacity alpha_c is estimated")),
    Experiment("ecs-topology-collision", ECS, "INTERVENTION_SUPPORTED_RESULT",
               ("LAB: public ECS kernel and topology analysis", "arXiv:2609.29834 fibre of a coarse map (main text)"),
               ECS_RULE, _ecs_dev, _ecs_qual,
               ("public ECS kernel in temporary storage; no production change",
                "reply continuation is laboratory host code, not an ECS rule")),
)}

ORDER = ("cubic-control", "tanh-phase-sweep", "tanh-attractors", "tanh-delay-ladder", "tanh-intervention",
         "feedback-transition", "associative-cycles", "ecs-topology-collision")
