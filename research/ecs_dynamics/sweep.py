"""Gain x gamma sweep of the tanh probe, with several worlds per cell. RESEARCH_ONLY. DESCRIPTIVE.

Each cell reports the sampled fractions of fixed, periodic and unresolved
trajectories; finite-time growth; recovery classes; an attractor-count proxy
and basin occupancy; state norm and activity; and one-step coarse prediction
quality. Changes in these fractions across cells are empirical boundaries of
a finite-time classification. They are not identified as bifurcations.
"""
from __future__ import annotations

import numpy as np

from . import delay
from .attractors import AttractorInventory
from .diagnostics import finite_time_growth, recovery
from .observables import coarse_map
from .spec import ExperimentSpec, world_rng
from .systems import make_tanh_rnn, rollout, run


def world_system(spec: ExperimentSpec, world: str, gain: float, gamma: float, stream: str = "system"):
    rng = world_rng(spec, world, f"{stream}:g={gain!r}:gamma={gamma!r}")
    return make_tanh_rnn(rng, spec.dimension, gain, gamma, bias_scale=float(spec.parameters.get("bias_scale", 0.0)),
                         input_dimension=int(spec.parameters.get("input_dimension", 0))), rng


def cell_world(spec: ExperimentSpec, world: str, gain: float, gamma: float) -> dict:
    p = spec.parameters
    system, rng = world_system(spec, world, gain, gamma)
    inv = AttractorInventory(warmup=spec.warmup, tail=spec.horizon, fixed_tol=spec.fixed_point_tolerance,
                             recurrence_tol=spec.recurrence_tolerance, max_period=spec.max_period)
    n_init = int(p["initial_conditions"])
    inits = [system.initial_state(rng) for _ in range(n_init)]
    for x0 in inits:
        inv.label(system, x0)
    inv.add_recovery(system, rng, spec.perturbation)
    growth, rec, norms, activity, series = [], [], [], [], []
    C = coarse_map(spec.coarse_observable)
    for x0 in inits:
        x = run(system, x0, spec.warmup)
        growth.append(finite_time_growth(system, x, rng, spec.lyapunov_horizon))
        rec.append(recovery(system, x, rng, magnitude=spec.perturbation, horizon=spec.lyapunov_horizon)["class"])
        traj = rollout(system, x, spec.horizon - 1)
        norms.append(float(np.mean(np.linalg.norm(traj, axis=1) / np.sqrt(spec.dimension))))
        activity.append(float(np.mean(np.abs(np.diff(traj, axis=0)))))
        series.append(np.array([C(s) for s in traj]))
    # One-step prediction quality of C_{t+1} from C_t: train on the first half of the
    # initial conditions, test on the rest, fixed lambda from the spec.
    half = max(1, n_init // 2)
    rows = [delay.make_rows(np.zeros((len(s), 1)), s, depth=0, horizon=1, target="ONE_STEP_TRANSITION",
                            extras={"perm": np.arange(len(s)), "proj": np.zeros((s.shape[1], 1)),
                                    "seed": np.zeros((len(s), 1))}) for s in series]
    train, test = delay.stack(rows[:half]), delay.stack(rows[half:])
    pred = delay.score(train, test, "coarse", float(p["sweep_lambda"]), "ONE_STEP_TRANSITION") if n_init > 1 else None
    if pred is not None and not np.isfinite(pred):
        pred = None  # a constant coarse signal (e.g. every trajectory at a fixed point) has no defined R^2
    summary = inv.summary(n_init)
    return {"world": world, "attractors": summary, "growth": growth, "recovery": rec, "norm": norms,
            "activity": activity, "one_step_r2": pred}


def _frac(labels, value):
    return float(np.mean([v == value for v in labels])) if labels else 0.0


def sweep(spec: ExperimentSpec, worlds) -> list[dict]:
    p = spec.parameters
    cells = []
    for gamma in p["gammas"]:
        for gain in p["gains"]:
            per_world = [cell_world(spec, w, float(gain), float(gamma)) for w in worlds]
            n = sum(pw["attractors"]["samples"] for pw in per_world)
            fixed = sum(a["basin_count"] for pw in per_world for a in pw["attractors"]["attractors"]
                        if a["type"] == "FIXED_POINT")
            periodic = sum(a["basin_count"] for pw in per_world for a in pw["attractors"]["attractors"]
                           if a["type"] == "PERIODIC")
            unresolved = sum(round(pw["attractors"]["unresolved_fraction"] * pw["attractors"]["samples"])
                             for pw in per_world)
            growth = [g for pw in per_world for g in pw["growth"]]
            rec = [r for pw in per_world for r in pw["recovery"]]
            preds = [pw["one_step_r2"] for pw in per_world if pw["one_step_r2"] is not None]
            cells.append({
                "gain": float(gain), "gamma": float(gamma), "worlds": len(per_world), "trajectories": n,
                "fixed_fraction": fixed / n, "periodic_fraction": periodic / n, "unresolved_fraction": unresolved / n,
                "periods": sorted({q for pw in per_world for q in pw["attractors"]["periods"]}),
                "growth_mean": float(np.mean(growth)), "growth_median": float(np.median(growth)),
                "growth_positive_fraction": float(np.mean(np.array(growth) > 0)),
                "recovery_decay": _frac(rec, "DECAY"), "recovery_persist": _frac(rec, "PERSIST"),
                "recovery_grow": _frac(rec, "GROW"),
                "attractor_count_proxy_mean": float(np.mean([pw["attractors"]["attractor_count_proxy"]
                                                             for pw in per_world])),
                "largest_basin_fraction_mean": float(np.mean([pw["attractors"]["largest_basin_fraction"]
                                                              for pw in per_world])),
                "norm_mean": float(np.mean([v for pw in per_world for v in pw["norm"]])),
                "activity_mean": float(np.mean([v for pw in per_world for v in pw["activity"]])),
                "one_step_r2_mean": float(np.mean(preds)) if preds else None,
            })
    return cells
