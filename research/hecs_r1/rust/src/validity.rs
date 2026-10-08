//! R1's task-validity diagnostics (research/hecs_r1/PREREGISTRATION.md, section 5).
//!
//! R0's V1 was an absolute one-step NMSE threshold and failed because the SEPARATED world's stochastic fast
//! factor puts an irreducible floor near it. R1 measures, on exactly the held-out examples the ECS is scored
//! on and with the same normalizer:
//!
//! * the **analytic floor** F: the generator-derived one-step error of the whitened latent (innovation of both
//!   factors, next-observation noise, and the current observation's noise propagated through the dynamics);
//! * the **linear reference** L: the best linear one-step predictor z' ~ [z, b, 1], fit on the training
//!   partition only;
//! * the **constant** C (the training mean of z') and **persistence** P (z' = z) baselines;
//! * the **ECS** E, from `analysis::horizon_error` at horizon 1.
//!
//! All are NMSE: squared error averaged over the latent coordinates, divided by the mean held-out latent
//! variance (the normalizer of `horizon_error`).

use crate::encoder::Encoder;
use crate::hierarchy::LevelSequence;
use crate::linalg::{ridge, ridge_apply, Mat};
use crate::model::LATENT;
use crate::world::World;

/// The held-out normalizer of `analysis::horizon_error`: the mean per-coordinate variance of every
/// held-out latent.
pub fn normalizer(test: &[LevelSequence]) -> f64 {
    let all: Vec<[f64; LATENT]> = test.iter().flat_map(|s| s.z.iter().copied()).collect();
    (0..LATENT)
        .map(|k| {
            let m = all.iter().map(|z| z[k]).sum::<f64>() / all.len() as f64;
            all.iter().map(|z| (z[k] - m).powi(2)).sum::<f64>() / all.len() as f64
        })
        .sum::<f64>()
        / LATENT as f64
}

/// One-step transitions (z_t, b_t) -> z_{t+1}: the examples `horizon_error(h = 1)` scores.
fn transitions(seqs: &[LevelSequence]) -> Vec<([f64; LATENT], f64, [f64; LATENT])> {
    seqs.iter().flat_map(|s| (0..s.b.len()).map(move |t| (s.z[t], s.b[t], s.z[t + 1]))).collect()
}

fn nmse(pairs: impl Iterator<Item = ([f64; LATENT], [f64; LATENT])>, var: f64) -> f64 {
    let (mut sum, mut n) = (0.0, 0usize);
    for (p, t) in pairs {
        sum += (0..LATENT).map(|k| (p[k] - t[k]).powi(2)).sum::<f64>() / LATENT as f64;
        n += 1;
    }
    sum / n as f64 / var
}

/// Held-out one-step NMSE of the best linear predictor fit on `train` (ridge 1e-9: an unregularized fit).
pub fn linear_reference(train: &[LevelSequence], test: &[LevelSequence]) -> f64 {
    let tr = transitions(train);
    let x: Vec<Vec<f64>> = tr.iter().map(|(z, b, _)| vec![z[0], z[1], *b]).collect();
    let y: Vec<Vec<f64>> = tr.iter().map(|(_, _, n)| n.to_vec()).collect();
    let (w, c) = ridge(&x, &y, 1e-9);
    let var = normalizer(test);
    nmse(
        transitions(test).into_iter().map(|(z, b, n)| {
            let p = ridge_apply(&w, &c, &[z[0], z[1], b]);
            ([p[0], p[1]], n)
        }),
        var,
    )
}

/// Held-out one-step NMSE of the constant predictor (the training mean of the next latent).
pub fn constant_baseline(train: &[LevelSequence], test: &[LevelSequence]) -> f64 {
    let tr = transitions(train);
    let mut m = [0.0; LATENT];
    for (_, _, n) in &tr {
        for k in 0..LATENT {
            m[k] += n[k] / tr.len() as f64;
        }
    }
    nmse(transitions(test).into_iter().map(|(_, _, n)| (m, n)), normalizer(test))
}

/// Held-out one-step NMSE of persistence (z' = z).
pub fn persistence_baseline(test: &[LevelSequence]) -> f64 {
    nmse(transitions(test).into_iter().map(|(z, _, n)| (z, n)), normalizer(test))
}

fn diag_rot(angle: f64, a: f64, b: f64) -> Mat {
    // R diag(a, b) R^T, with R the world's mixing rotation (columns: the s and f directions).
    let (c, s) = (angle.cos(), angle.sin());
    let r = Mat::from_rows(&[vec![c, -s], vec![s, c]]);
    let d = Mat::from_rows(&[vec![a, 0.0], vec![0.0, b]]);
    r.mul(&d).mul(&r.t())
}

/// The generator-derived one-step floor in the whitened level-1 latent (unclipped dynamics):
///
///   Q = R diag(slow_noise^2, (1 - rho^2) second_std^2) R^T  +  n^2 I  +  n^2 A A^T,   A = R diag(1, rho) R^T
///
/// (factor innovations, the next observation's noise, and the current observation's noise carried by the
/// dynamics), whitened by the level-1 map P: F = tr(P Q P^T) / LATENT / normalizer.
pub fn analytic_floor(world: &World, enc1: &Encoder, test: &[LevelSequence]) -> f64 {
    let w = &world.spec;
    let n2 = w.obs_noise * w.obs_noise;
    let innovation = diag_rot(world.angle, w.slow_noise * w.slow_noise, (1.0 - w.rho * w.rho) * w.second_std.powi(2));
    let a = diag_rot(world.angle, 1.0, w.rho);
    let carried = a.mul(&a.t());
    let mut q = Mat::zeros(2, 2);
    for i in 0..2 {
        for j in 0..2 {
            q[(i, j)] = innovation[(i, j)] + n2 * carried[(i, j)] + if i == j { n2 } else { 0.0 };
        }
    }
    let p = match enc1 {
        Encoder::Affine { p, .. } => p,
        Encoder::Shared => panic!("level 1 is a whitening map"),
    };
    let qz = p.mul(&q).mul(&p.t());
    (qz[(0, 0)] + qz[(1, 1)]) / LATENT as f64 / normalizer(test)
}

/// The R1 validity law for one world instance (thresholds in `spec::gates`).
#[derive(Clone, Copy, Debug)]
pub struct Diagnostics {
    pub floor: f64,
    pub linear: f64,
    pub constant: f64,
    pub persistence: f64,
    pub ecs: f64,
}

impl Diagnostics {
    /// A: the held-out linear reference explains at least `min_explained` of the constant baseline's error.
    pub fn reference_predictable(&self, min_explained: f64) -> bool {
        self.linear <= (1.0 - min_explained) * self.constant
    }

    /// B: the linear reference sits at the generator's floor (the world is what the generator says).
    pub fn reference_at_floor(&self, tolerance_abs: f64, tolerance_rel: f64) -> bool {
        (self.linear - self.floor).abs() <= tolerance_abs + tolerance_rel * self.floor
    }

    /// C: the ECS captures at least `min_capture` of the one-step error the linear reference removes from the
    /// constant baseline, (constant - ecs) >= min_capture (constant - linear), with no divergence.
    pub fn ecs_adequate(&self, min_capture: f64) -> bool {
        self.ecs.is_finite() && self.constant - self.ecs >= min_capture * (self.constant - self.linear)
    }

    /// The fraction of the linearly removable one-step error the ECS removes.
    pub fn capture(&self) -> f64 {
        (self.constant - self.ecs) / (self.constant - self.linear)
    }
}
