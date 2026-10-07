//! Level encoders (bounded, deterministic, closed-form training).
//!
//! * Level 1, every condition: whitening of the 2-dimensional observation, z = S^-1/2 (o - m).
//! * TEMPORAL_SHARED upper levels: no learned map; the upper latent is the lower latent at the window end
//!   (window 1). Every level lives in the level-1 space (the temporal-hierarchy control).
//! * HIERARCHICAL_DISTINCT upper levels: a learned affine map E(u) = P (u - m) of the window u of the last
//!   `window` lower latents (2 * window inputs) to 2 outputs. Training ("predictable components"): fit the
//!   linear one-upper-step model u' ~ A u + B b + c on training data, take the residual covariance R and
//!   the input covariance S, and keep the 2 directions p minimising p^T R p subject to p^T S p = 1 (the
//!   generalized eigenproblem R p = lambda S p). The unit-covariance constraint prevents collapse. This is
//!   a linear predictability proxy, not end-to-end training through the ECS.

use crate::linalg::{covariance, mean_rows, ridge, ridge_apply, sym_eigen, Mat};
use crate::model::LATENT;

#[derive(Clone, Debug)]
pub enum Encoder {
    /// z = P (u - m), u the concatenated window (oldest first).
    Affine { mean: Vec<f64>, p: Mat, predictability: Vec<f64> },
    /// Pass-through of the window's last lower latent.
    Shared,
}

/// S^-1/2 of a symmetric positive definite matrix.
fn inv_sqrt(s: &Mat) -> Mat {
    let (vals, vecs) = sym_eigen(s);
    let n = s.rows;
    let mut d = Mat::zeros(n, n);
    for i in 0..n {
        d[(i, i)] = 1.0 / vals[i].max(1e-12).sqrt();
    }
    vecs.mul(&d).mul(&vecs.t())
}

impl Encoder {
    pub fn whitening(obs: &[Vec<f64>]) -> Encoder {
        let mean = mean_rows(obs);
        let p = inv_sqrt(&covariance(obs));
        Encoder::Affine { mean, p, predictability: vec![] }
    }

    /// Fit the predictable-components map from (window, pooled action, next window) triples.
    pub fn predictable(windows: &[Vec<f64>], actions: &[f64], next: &[Vec<f64>]) -> Encoder {
        let inputs: Vec<Vec<f64>> = windows.iter().zip(actions).map(|(u, b)| [u.clone(), vec![*b]].concat()).collect();
        let (w, c) = ridge(&inputs, next, 1e-6);
        let residuals: Vec<Vec<f64>> = inputs
            .iter()
            .zip(next)
            .map(|(x, y)| ridge_apply(&w, &c, x).iter().zip(y).map(|(p, t)| t - p).collect())
            .collect();
        let r = covariance(&residuals);
        let s = covariance(windows);
        let k = inv_sqrt(&s);
        let (vals, q) = sym_eigen(&k.mul(&r).mul(&k));
        // The LATENT most predictable whitened directions (smallest residual variance), as rows of P.
        let dirs = k.mul(&q);
        let n = s.rows;
        let mut p = Mat::zeros(LATENT, n);
        for out in 0..LATENT {
            for i in 0..n {
                p[(out, i)] = dirs[(i, out)];
            }
        }
        Encoder::Affine { mean: mean_rows(windows), p, predictability: vals }
    }

    /// Encode a window (oldest lower latent first). Shared returns the last lower latent.
    pub fn encode(&self, window: &[[f64; LATENT]]) -> [f64; LATENT] {
        match self {
            Encoder::Shared => *window.last().expect("non-empty window"),
            Encoder::Affine { mean, p, .. } => {
                let u: Vec<f64> = window.iter().flat_map(|z| z.iter().copied()).collect();
                let centered: Vec<f64> = u.iter().zip(mean).map(|(a, m)| a - m).collect();
                let z = p.mul_vec(&centered);
                [z[0], z[1]]
            }
        }
    }

    /// Learned parameters (0 for the shared pass-through).
    pub fn parameters(&self) -> usize {
        match self {
            Encoder::Shared => 0,
            Encoder::Affine { mean, p, .. } => mean.len() + p.data.len(),
        }
    }
}

/// Encode an observation through the level-1 whitening.
pub fn encode_obs(e: &Encoder, o: &[f64; 2]) -> [f64; LATENT] {
    e.encode(&[[o[0], o[1]]])
}
