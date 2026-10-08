//! One hierarchy level's world model: exactly one K1 state (W, epoch, H, a) of the qualified ECS.
//!
//! The ECS map is scalar, f_W(x) = sum_i phi(x . w_i). A level predicts its next 2-dimensional latent with
//! one state by a coordinate selector in the input, keeping the qualified dimension d = 6:
//!
//!     x = [z_0, z_1, b, sel_0, sel_1, 1]      y = z'_k   for k = argmax(sel)
//!
//! where z is the level latent, b the level action (pooled for upper levels) and z' the next latent at the
//! level's own clock. Learning is the qualified LEARN call only (K1 steps on mini-batches; H stays zero, so
//! each step is exactly G1). Nothing here changes the ECS equations.

use crate::k1::{K1Error, K1State, K1};
use crate::rng::Rng;

pub const DIM: usize = 6;
pub const LATENT: usize = 2;

/// One transition at a level's clock.
#[derive(Clone, Copy, Debug)]
pub struct Transition {
    pub z: [f64; LATENT],
    pub b: f64,
    pub next: [f64; LATENT],
}

#[derive(Clone, Debug)]
pub struct TrainSpec {
    /// Total K1 learning steps (the budget is fixed in the specification, not adapted).
    pub steps: u64,
    pub steps_per_call: u64,
    pub batch_transitions: usize,
    pub base_rate: f64,
    pub reference_width: usize,
    pub init_scale: f64,
}

impl TrainSpec {
    /// Width-normalized rate: base_rate * reference_width / width (0.002 at the reference width 36).
    pub fn rate(&self, width: usize) -> f64 {
        self.base_rate * self.reference_width as f64 / width as f64
    }
}

#[derive(Clone, Debug, Default)]
pub struct TrainTrace {
    /// (epoch, training MSE, ||W||_F) at 0, 25, 50, 75 and 100 percent of the budget.
    pub checkpoints: Vec<(u64, f64, f64)>,
    pub rows_learned: u64,
    pub multiply_adds: u64,
    pub refused: Option<String>,
}

pub fn row(z: &[f64; LATENT], b: f64, k: usize) -> [f64; DIM] {
    [z[0], z[1], b, (k == 0) as u8 as f64, (k == 1) as u8 as f64, 1.0]
}

pub struct LevelModel<'a> {
    pub ecs: K1State<'a>,
    pub width: usize,
    pub forward_rows: u64,
}

impl<'a> LevelModel<'a> {
    pub fn new(lib: &'a K1, width: usize, max_rows: usize, init_scale: f64, rng: &mut Rng) -> Result<Self, K1Error> {
        let w: Vec<f64> = (0..DIM * width).map(|_| init_scale * rng.normal()).collect();
        Ok(LevelModel { ecs: lib.create(DIM, width, max_rows, &w)?, width, forward_rows: 0 })
    }

    /// Next latents for a batch of (z, b): one native QUERY call for all rows.
    pub fn predict(&mut self, inputs: &[([f64; LATENT], f64)]) -> Result<Vec<[f64; LATENT]>, K1Error> {
        let mut x = Vec::with_capacity(inputs.len() * LATENT * DIM);
        for (z, b) in inputs {
            for k in 0..LATENT {
                x.extend_from_slice(&row(z, *b, k));
            }
        }
        let mut out = vec![0.0; inputs.len() * LATENT];
        self.ecs.forward(&x, &mut out)?;
        self.forward_rows += out.len() as u64;
        Ok(out.chunks(LATENT).map(|c| [c[0], c[1]]).collect())
    }

    /// Like `predict`, but a row whose native QUERY is refused as NONFINITE (a diverging cubic map) yields
    /// `None` instead of failing the batch. Refused batches are retried row by row; every row computed is
    /// counted. Other refusals are errors.
    pub fn predict_lossy(&mut self, inputs: &[([f64; LATENT], f64)]) -> Result<Vec<Option<[f64; LATENT]>>, K1Error> {
        match self.predict(inputs) {
            Ok(v) => Ok(v.into_iter().map(Some).collect()),
            Err(K1Error(-2)) => {
                let mut out = Vec::with_capacity(inputs.len());
                for input in inputs {
                    out.push(match self.predict(std::slice::from_ref(input)) {
                        Ok(v) => Some(v[0]),
                        Err(K1Error(-2)) => None,
                        Err(e) => return Err(e),
                    });
                }
                Ok(out)
            }
            Err(e) => Err(e),
        }
    }

    pub fn mse(&mut self, data: &[Transition]) -> Result<f64, K1Error> {
        let inputs: Vec<_> = data.iter().map(|t| (t.z, t.b)).collect();
        let pred = self.predict(&inputs)?;
        let sum: f64 =
            pred.iter().zip(data).map(|(p, t)| (0..LATENT).map(|k| (p[k] - t.next[k]).powi(2)).sum::<f64>()).sum();
        Ok(sum / (data.len() * LATENT) as f64)
    }

    /// Fixed-budget training over mini-batches in a fixed order (deterministic).
    pub fn train(&mut self, data: &[Transition], spec: &TrainSpec) -> TrainTrace {
        let mut trace = TrainTrace::default();
        let batches: Vec<&[Transition]> = data.chunks(spec.batch_transitions).collect();
        let rate = spec.rate(self.width);
        let calls = spec.steps / spec.steps_per_call;
        let marks: Vec<u64> = (0..=4).map(|q| calls * q / 4).collect();
        let record = |m: &mut Self, trace: &mut TrainTrace| -> Result<(), K1Error> {
            let mse = m.mse(data)?;
            let w = m.ecs.w()?;
            trace.checkpoints.push((m.ecs.epoch(), mse, w.iter().map(|v| v * v).sum::<f64>().sqrt()));
            Ok(())
        };
        if let Err(e) = record(self, &mut trace) {
            trace.refused = Some(e.to_string());
            return trace;
        }
        for call in 0..calls {
            let batch = batches[call as usize % batches.len()];
            let mut x = Vec::with_capacity(batch.len() * LATENT * DIM);
            let mut y = Vec::with_capacity(batch.len() * LATENT);
            for t in batch {
                for k in 0..LATENT {
                    x.extend_from_slice(&row(&t.z, t.b, k));
                    y.push(t.next[k]);
                }
            }
            if let Err(e) = self.ecs.learn(&x, &y, rate, spec.steps_per_call) {
                trace.refused = Some(format!("{e} at call {call}"));
                return trace;
            }
            trace.rows_learned += y.len() as u64 * spec.steps_per_call;
            if marks[1..].contains(&(call + 1)) {
                if let Err(e) = record(self, &mut trace) {
                    trace.refused = Some(e.to_string());
                    return trace;
                }
            }
        }
        // A G1 step costs a forward and a gradient pass: about 2 * rows * width * dim multiply-adds each.
        trace.multiply_adds = 2 * trace.rows_learned * (self.width * DIM) as u64;
        trace
    }

    /// Open-loop rollout of h steps under the given level actions.
    pub fn rollout(&mut self, z0: [f64; LATENT], actions: &[f64]) -> Result<Vec<[f64; LATENT]>, K1Error> {
        let mut z = z0;
        let mut out = Vec::with_capacity(actions.len());
        for &b in actions {
            z = self.predict(&[(z, b)])?[0];
            out.push(z);
        }
        Ok(out)
    }
}
