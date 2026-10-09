//! H-ECS R3 successor C2 candidate mechanics: true-target terminal rollout anchoring. Research-only; no authority.
//! The unchanged R3 rollout objective plus w_T * 0.5*||zhat_h - z_h||^2/(LATENT*var), where zhat_h is
//! free-running from the TRUE z_0 through all h recursive transitions and the target z_h is the TRUE training
//! latent (never a teacher prediction). Exact reverse-mode gradient through every transition.
use hecs_r2::model::{row, DIM, LATENT};
use super::r3_training_core::{self, Objective, Window, Work};

fn dphi(u: f64) -> f64 {
    0.5 + u + 1.5 * u * u
}

/// Mean over windows of sum_t c_t * 0.5*||zhat_{t+1} - z_{t+1}||^2 / (LATENT*var) for the free-running rollout
/// (only z_0 is true) and its exact mean gradient. The arithmetic mirrors r3_training_core::objective_gradient
/// (Objective::Rollout) operation for operation, so coefficients [1, lambda/(h-1), ..., lambda/(h-1)] reproduce
/// that function bitwise; work is counted with the same surrogate.
pub fn weighted_rollout_gradient(
    w: &[f64],
    width: usize,
    windows: &[Window],
    h: usize,
    variance: f64,
    coefficients: &[f64],
) -> Result<(f64, Vec<f64>, Work), String> {
    if width == 0 || w.len() != DIM * width || windows.is_empty() || h < 2 || h > 16 || coefficients.len() != h {
        return Err("INVALID_SHAPE_OR_HORIZON".into());
    }
    if !(variance.is_finite() && variance > 0.0)
        || !coefficients.iter().all(|c| c.is_finite() && *c >= 0.0)
        || !w.iter().all(|v| v.is_finite())
    {
        return Err("INVALID_PARAMETER_OR_NORMALIZER".into());
    }
    let mut grad = vec![0.0; w.len()];
    let mut loss = 0.0;
    let mut work = Work::default();
    for win in windows {
        if win.actions.len() != h || win.z.len() != h + 1 {
            return Err("INVALID_WINDOW_SHAPE".to_string());
        }
        if !win.actions.iter().all(|v| v.is_finite()) || !win.z.iter().flatten().all(|v| v.is_finite()) {
            return Err("NONFINITE_WINDOW".to_string());
        }
        let mut input_states = Vec::with_capacity(h);
        let mut predictions = Vec::with_capacity(h);
        let mut state = win.z[0];
        for t in 0..h {
            let input = state;
            let pred = r3_training_core::forward(w, width, input, win.actions[t]);
            if !pred.iter().all(|v| v.is_finite()) || pred.iter().any(|v| v.abs() > 1.0e3) {
                return Err("NUMERICAL_ESCAPE".into());
            }
            input_states.push(input);
            predictions.push(pred);
            state = pred;
        }
        let mut upstream = [0.0; LATENT];
        for t in (0..h).rev() {
            let coefficient = coefficients[t];
            let mut next_upstream = [0.0; LATENT];
            for k in 0..LATENT {
                let error = predictions[t][k] - win.z[t + 1][k];
                loss += coefficient * 0.5 * error * error / (LATENT as f64 * variance);
                let q = coefficient * error / (LATENT as f64 * variance) + upstream[k];
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
                    for j in 0..LATENT {
                        next_upstream[j] += c * w[j * width + i];
                    }
                }
            }
            upstream = next_upstream;
        }
        work.samples += 1;
        work.forward_rows += (h * LATENT * width) as u64;
        work.d_weight_rows += (h * LATENT * width) as u64;
        work.state_adjoint_rows += (h * LATENT * width) as u64;
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

/// One C2 update: unchanged R3 rollout objective (base lambda) plus terminal_weight times the true-target
/// terminal loss at step h, summed gradients, then the unchanged R3 update/clip rule. Returns
/// (combined loss, base work, terminal work, pre-clip gradient norm).
pub fn c2_step(
    w: &mut [f64],
    width: usize,
    windows: &[Window],
    h: usize,
    variance: f64,
    base_lambda: f64,
    terminal_weight: f64,
    rate: f64,
    clip: f64,
) -> Result<(f64, Work, Work, f64), String> {
    if !(terminal_weight.is_finite() && terminal_weight >= 0.0) {
        return Err("INVALID_TERMINAL_WEIGHT".into());
    }
    let (base_loss, mut grad, base_work) =
        r3_training_core::objective_gradient(w, width, windows, h, variance, base_lambda, Objective::Rollout)?;
    let mut coefficients = vec![0.0; h];
    coefficients[h - 1] = 1.0;
    let (terminal_loss, terminal_grad, terminal_work) =
        weighted_rollout_gradient(w, width, windows, h, variance, &coefficients)?;
    for (g, t) in grad.iter_mut().zip(&terminal_grad) {
        *g += terminal_weight * t;
    }
    let loss = base_loss + terminal_weight * terminal_loss;
    if !loss.is_finite() || !grad.iter().all(|v| v.is_finite()) {
        return Err("NONFINITE_LOSS_OR_GRADIENT".into());
    }
    let norm = r3_training_core::update(w, &grad, rate, clip)?;
    Ok((loss, base_work, terminal_work, norm))
}
