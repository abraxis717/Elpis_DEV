//! Deterministic random-shooting planning over the levels' own ECS world models (top-down subgoals).
//!
//! FLAT: level 1 plans `horizon` environment steps (actions held in blocks) toward the goal latent.
//! Hierarchical (L levels): the top level plans `horizon` of its steps toward the goal in its own space;
//! its first predicted state is the subgoal of the level below, which plans exactly one upper tick
//! (`stride` of its steps), encodes its predicted window into the upper space and is scored by the
//! distance to that subgoal. Level 1 returns environment actions. Every native forward row is counted.

use crate::encoder::Encoder;
use crate::hierarchy::HierarchySpec;
use crate::k1::K1Error;
use crate::model::{LevelModel, LATENT};
use crate::rng::Rng;

pub const ACTION_GRID: [f64; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];

/// K candidate sequences of `horizon` actions held in blocks of `block`; the first three are the constant
/// anchors -1, 0, +1 (when K >= 3), the rest uniform on the grid per block.
pub fn candidates(k: usize, horizon: usize, block: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let blocks = horizon.div_ceil(block);
    (0..k)
        .map(|i| {
            let values: Vec<f64> = if i < 3 && k >= 3 {
                vec![[-1.0, 0.0, 1.0][i]; blocks]
            } else {
                (0..blocks).map(|_| ACTION_GRID[rng.below(5) as usize]).collect()
            };
            (0..horizon).map(|t| values[t / block]).collect()
        })
        .collect()
}

fn dist2(a: &[f64; LATENT], b: &[f64; LATENT]) -> f64 {
    (0..LATENT).map(|k| (a[k] - b[k]).powi(2)).sum()
}

/// Batched rollout of all candidates from z0: one native QUERY call per step for the live candidates. A
/// candidate whose rollout diverges (NONFINITE) is dead: `None`, never predicted again, infinite cost.
pub fn rollouts(
    model: &mut LevelModel,
    z0: [f64; LATENT],
    cands: &[Vec<f64>],
) -> Result<Vec<Option<Vec<[f64; LATENT]>>>, K1Error> {
    let horizon = cands[0].len();
    let mut out: Vec<Option<Vec<[f64; LATENT]>>> = vec![Some(Vec::with_capacity(horizon)); cands.len()];
    let mut states = vec![z0; cands.len()];
    for t in 0..horizon {
        let live: Vec<usize> = (0..cands.len()).filter(|&i| out[i].is_some()).collect();
        if live.is_empty() {
            break;
        }
        let inputs: Vec<_> = live.iter().map(|&i| (states[i], cands[i][t])).collect();
        for (&i, next) in live.iter().zip(model.predict_lossy(&inputs)?) {
            match next {
                Some(z) => {
                    states[i] = z;
                    out[i].as_mut().unwrap().push(z);
                }
                None => out[i] = None,
            }
        }
    }
    Ok(out)
}

fn cost_last(r: &Option<Vec<[f64; LATENT]>>, goal: &[f64; LATENT]) -> f64 {
    r.as_ref().map_or(f64::INFINITY, |r| dist2(r.last().unwrap(), goal))
}

fn argmin(costs: &[f64]) -> usize {
    let mut best = 0;
    for (i, c) in costs.iter().enumerate() {
        if *c < costs[best] || (costs[best].is_nan() && !c.is_nan()) {
            best = i;
        }
    }
    best
}

pub struct Plan {
    /// Environment actions to execute now.
    pub actions: Vec<f64>,
    /// The subgoal level 1 was steered to (in level-2 space), for hierarchical plans.
    pub subgoal: Option<[f64; LATENT]>,
}

/// FLAT: horizon `horizon` env steps in blocks of `block`; execute the first `execute` actions.
pub fn plan_flat(
    model: &mut LevelModel,
    z: [f64; LATENT],
    goal: [f64; LATENT],
    k: usize,
    horizon: usize,
    block: usize,
    execute: usize,
    rng: &mut Rng,
) -> Result<Plan, K1Error> {
    let cands = candidates(k, horizon, block, rng);
    let rolls = rollouts(model, z, &cands)?;
    let costs: Vec<f64> = rolls.iter().map(|r| cost_last(r, &goal)).collect();
    let best = argmin(&costs);
    Ok(Plan { actions: cands[best][..execute].to_vec(), subgoal: None })
}

/// Hierarchical top-down plan. `current[l]` and `goals[l]` are per-level latents.
pub fn plan_hierarchical(
    spec: &HierarchySpec,
    models: &mut [LevelModel],
    encoders: &[Encoder],
    current: &[[f64; LATENT]],
    goals: &[[f64; LATENT]],
    k: usize,
    rng: &mut Rng,
) -> Result<Plan, K1Error> {
    let top = spec.levels.len() - 1;
    let cands = candidates(k, spec.levels[top].horizon, 1, rng);
    let rolls = rollouts(&mut models[top], current[top], &cands)?;
    let costs: Vec<f64> = rolls.iter().map(|r| cost_last(r, &goals[top])).collect();
    // If every top rollout diverged, the subgoal is the current top latent (hold).
    let mut subgoal = rolls[argmin(&costs)].as_ref().map_or(current[top], |r| r[0]);
    let mut level1_subgoal = subgoal;
    let mut actions = Vec::new();
    for l in (0..top).rev() {
        let upper = &spec.levels[l + 1];
        let cands = candidates(k, upper.stride, 1, rng);
        let rolls = rollouts(&mut models[l], current[l], &cands)?;
        let costs: Vec<f64> = rolls
            .iter()
            .map(|r| match r {
                Some(r) => dist2(&encoders[l + 1].encode(&r[upper.stride - upper.window..upper.stride]), &subgoal),
                None => f64::INFINITY,
            })
            .collect();
        let best = argmin(&costs);
        if l == 0 {
            level1_subgoal = subgoal;
            actions = cands[best].clone();
        }
        subgoal = rolls[best].as_ref().map_or(current[l], |r| r[0]);
    }
    Ok(Plan { actions, subgoal: Some(level1_subgoal) })
}
