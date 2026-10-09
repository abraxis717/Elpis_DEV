//! H-ECS R3 research-only candidate optimizer mechanics. No scientific authority.
//! Cubic forward = K1 G1 for a frozen W. All data passed in; no global state.
//! This module is a donor for a future preregistered DESIGN candidate; its knobs are NOT frozen.

use hecs_r2::model::{row, DIM, LATENT};

pub type Z = [f64; LATENT];

#[derive(Clone)]
pub struct Window {
    pub z: Vec<Z>,          // True z_0 .. z_h; not a predicted or teacher-leaked input.
    pub actions: Vec<f64>, // True actions b_0 .. b_{h-1}.
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Objective {
    Teacher, // Every transition takes the *true* preceding latent.
    Rollout, // Only z_0 is true; all subsequent latents are self-generated.
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Work {
    pub forward_rows: u64,
    pub d_weight_rows: u64,
    pub state_adjoint_rows: u64,
    pub counted_multiplies: u64,
    pub samples: u64,
}

impl Work {
    pub fn accumulate(&mut self, other: Self) {
        self.forward_rows += other.forward_rows;
        self.d_weight_rows += other.d_weight_rows;
        self.state_adjoint_rows += other.state_adjoint_rows;
        self.counted_multiplies += other.counted_multiplies;
        self.samples += other.samples;
    }
}

fn phi(u: f64) -> f64 {
    0.5 * u + 0.5 * (u * u) + 0.5 * (u * u) * u
}
fn dphi(u: f64) -> f64 {
    0.5 + u + 1.5 * u * u
}

pub fn forward(w: &[f64], width: usize, z: Z, b: f64) -> Z {
    assert_eq!(w.len(), width * DIM);
    let mut out = [0.0; LATENT];
    for (k, value) in out.iter_mut().enumerate() {
        let x = row(&z, b, k);
        for i in 0..width {
            let mut u = 0.0;
            for a in 0..DIM {
                u += x[a] * w[a * width + i];
            }
            *value += phi(u);
        }
    }
    out
}

fn checked_window(win: &Window, h: usize) -> Result<(), String> {
    if win.actions.len() != h || win.z.len() != h + 1 {
        return Err("INVALID_WINDOW_SHAPE".to_string());
    }
    if !win.actions.iter().all(|v| v.is_finite()) || !win.z.iter().flatten().all(|v| v.is_finite()) {
        return Err("NONFINITE_WINDOW".to_string());
    }
    Ok(())
}

/// Returns mean loss and the exact mean gradient for a collection of independent windows.
/// Training normalization must originate from *training* latents only, never held-out data.
/// Teacher objective: average_{t=1..h} 0.5*||G(z_true[t-1],b)-z_true[t]||^2/(2*var).
/// Rollout objective: the first step with coefficient 1 plus lambda/(h-1) for t>=2.
/// Gradient through each self-generated latent is exact reverse-mode differentiation.
/// Work counts only coordinate multiplies in u, dW, and state adjoint; activation ops,
/// accumulation, memory traffic, and optimizer overhead are NOT included (surrogate, not FLOPs).
pub fn objective_gradient(
    w: &[f64],
    width: usize,
    windows: &[Window],
    h: usize,
    variance: f64,
    lambda: f64,
    objective: Objective,
) -> Result<(f64, Vec<f64>, Work), String> {
    if width == 0 || w.len() != DIM * width || windows.is_empty() || h < 2 || h > 16 {
        return Err("INVALID_SHAPE_OR_HORIZON".into());
    }
    if !(variance.is_finite() && variance > 0.0 && lambda.is_finite() && lambda >= 0.0)
        || !w.iter().all(|v| v.is_finite())
    {
        return Err("INVALID_PARAMETER_OR_NORMALIZER".into());
    }
    let mut grad = vec![0.0; w.len()];
    let mut loss = 0.0;
    let mut work = Work::default();
    for win in windows {
        checked_window(win, h)?;
        let mut input_states = Vec::with_capacity(h);
        let mut predictions = Vec::with_capacity(h);
        let mut state = win.z[0];
        for t in 0..h {
            let input = if objective == Objective::Teacher { win.z[t] } else { state };
            let pred = forward(w, width, input, win.actions[t]);
            if !pred.iter().all(|v| v.is_finite()) || pred.iter().any(|v| v.abs() > 1.0e3) {
                return Err("NUMERICAL_ESCAPE".into());
            }
            input_states.push(input);
            predictions.push(pred);
            state = pred;
        }
        let mut upstream = [0.0; LATENT];
        for t in (0..h).rev() {
            let coefficient = match objective {
                Objective::Teacher => 1.0 / h as f64,
                Objective::Rollout if t == 0 => 1.0,
                Objective::Rollout => lambda / (h - 1) as f64,
            };
            let mut next_upstream = [0.0; LATENT];
            for k in 0..LATENT {
                let error = predictions[t][k] - win.z[t + 1][k];
                loss += coefficient * 0.5 * error * error / (LATENT as f64 * variance);
                let q = coefficient * error / (LATENT as f64 * variance)
                    + if objective == Objective::Rollout { upstream[k] } else { 0.0 };
                let x = row(&input_states[t], win.actions[t], k);
                for i in 0..width {
                    let mut u = 0.0;
                    for a in 0..DIM {
                        u += x[a] * w[a * width + i];
                    }
                    let c = q * dphi(u);
                    for a in 0..DIM {
                        grad[a * width + i] += c * x[a];
                    }
                    if objective == Objective::Rollout {
                        for j in 0..LATENT {
                            next_upstream[j] += c * w[j * width + i];
                        }
                    }
                }
            }
            upstream = next_upstream;
        }
        work.samples += 1;
        work.forward_rows += (h * LATENT * width) as u64;
        work.d_weight_rows += (h * LATENT * width) as u64;
        if objective == Objective::Rollout {
            work.state_adjoint_rows += (h * LATENT * width) as u64;
        }
    }
    let n = windows.len() as f64;
    loss /= n;
    for g in &mut grad {
        *g /= n;
    }
    work.counted_multiplies = DIM as u64 * (work.forward_rows + work.d_weight_rows)
        + LATENT as u64 * work.state_adjoint_rows;
    if !loss.is_finite() || !grad.iter().all(|v| v.is_finite()) {
        return Err("NONFINITE_LOSS_OR_GRADIENT".into());
    }
    Ok((loss, grad, work))
}

/// Same update/clip rule for teacher and rollout, with refusal on any nonfinite result.
pub fn update(w: &mut [f64], gradient: &[f64], rate: f64, clip: f64) -> Result<f64, String> {
    if w.len() != gradient.len() || w.is_empty() || !(rate.is_finite() && rate > 0.0)
        || !(clip.is_finite() && clip > 0.0)
    {
        return Err("INVALID_UPDATE".into());
    }
    let norm = gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !norm.is_finite() {
        return Err("NONFINITE_GRADIENT_NORM".into());
    }
    let scale = if norm > clip { clip / norm } else { 1.0 };
    for (wi, &gi) in w.iter_mut().zip(gradient) {
        *wi -= rate * scale * gi;
    }
    if !w.iter().all(|v| v.is_finite()) {
        return Err("NONFINITE_UPDATED_WEIGHTS".into());
    }
    Ok(norm)
}

/// Deterministic trainer for a paired batch, with matching *counted multiply surrogate*.
/// Teacher gets 7 windows; rollout gets 6: 7*(2*DIM) == 6*(2*DIM+LATENT) for DIM=6,LATENT=2.
/// These are explicitly unequal sample counts. Equal surrogate work does NOT certify equal wallclock/FLOPs.
pub fn paired_step(
    teacher_w: &mut [f64],
    rollout_w: &mut [f64],
    width: usize,
    windows: &[Window],
    h: usize,
    variance: f64,
    lambda: f64,
    rate: f64,
    clip: f64,
    index: usize,
) -> Result<(f64, f64, Work, Work), String> {
    if windows.len() < 8 {
        return Err("WINDOW_POOL_TOO_SMALL".into());
    }
    let picks: Vec<Window> = (0..7).map(|j| windows[(index * 7 + j) % windows.len()].clone()).collect();
    let (teacher_loss, teacher_grad, teacher_work) =
        objective_gradient(teacher_w, width, &picks, h, variance, lambda, Objective::Teacher)?;
    let (rollout_loss, rollout_grad, rollout_work) =
        objective_gradient(rollout_w, width, &picks[..6], h, variance, lambda, Objective::Rollout)?;
    if teacher_work.counted_multiplies != rollout_work.counted_multiplies {
        return Err("COMPUTE_SURROGATE_NOT_MATCHED".into());
    }
    update(teacher_w, &teacher_grad, rate, clip)?;
    update(rollout_w, &rollout_grad, rate, clip)?;
    Ok((teacher_loss, rollout_loss, teacher_work, rollout_work))
}
