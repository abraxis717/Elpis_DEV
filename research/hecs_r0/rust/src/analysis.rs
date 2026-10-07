//! Measurements per level: factor recoverability (linear probes), prediction error by horizon, and the
//! attractor structure of each level's free-running ECS map.

use crate::k1::K1Error;
use crate::linalg::{ridge, ridge_apply};
use crate::model::{LevelModel, LATENT};

/// Normalized MSE of a ridge probe from level latents to a scalar target: fit on train, score on test.
/// 1 means no better than predicting the test mean; 0 means exact recovery.
pub fn probe_nmse(train_z: &[[f64; LATENT]], train_y: &[f64], test_z: &[[f64; LATENT]], test_y: &[f64]) -> f64 {
    let x: Vec<Vec<f64>> = train_z.iter().map(|z| z.to_vec()).collect();
    let y: Vec<Vec<f64>> = train_y.iter().map(|v| vec![*v]).collect();
    let (w, b) = ridge(&x, &y, 1e-3);
    let mean = test_y.iter().sum::<f64>() / test_y.len() as f64;
    let var = test_y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / test_y.len() as f64;
    let mse = test_z.iter().zip(test_y).map(|(z, t)| (ridge_apply(&w, &b, z)[0] - t).powi(2)).sum::<f64>()
        / test_y.len() as f64;
    if var > 0.0 {
        mse / var
    } else {
        f64::NAN
    }
}

/// Open-loop prediction error at horizon h (level steps) over every start of every held-out sequence,
/// driven by the true level actions: (NMSE over the finite rollouts, fraction of rollouts that diverged).
/// Normalized by the held-out latent variance.
pub fn horizon_error(
    model: &mut LevelModel,
    seqs: &[(Vec<[f64; LATENT]>, Vec<f64>)],
    h: usize,
) -> Result<(f64, f64), K1Error> {
    let (mut sum, mut count, mut diverged) = (0.0, 0usize, 0usize);
    let all: Vec<[f64; LATENT]> = seqs.iter().flat_map(|(z, _)| z.iter().copied()).collect();
    let var: f64 = (0..LATENT)
        .map(|k| {
            let m = all.iter().map(|z| z[k]).sum::<f64>() / all.len() as f64;
            all.iter().map(|z| (z[k] - m).powi(2)).sum::<f64>() / all.len() as f64
        })
        .sum::<f64>()
        / LATENT as f64;
    for (z, b) in seqs {
        if b.len() < h {
            continue;
        }
        let starts: Vec<usize> = (0..=b.len() - h).collect();
        let mut states: Vec<Option<[f64; LATENT]>> = starts.iter().map(|&t| Some(z[t])).collect();
        for step in 0..h {
            let live: Vec<usize> = (0..states.len()).filter(|&i| states[i].is_some()).collect();
            let inputs: Vec<_> = live.iter().map(|&i| (states[i].unwrap(), b[starts[i] + step])).collect();
            for (&i, next) in live.iter().zip(model.predict_lossy(&inputs)?) {
                states[i] = next;
            }
        }
        for (s, &t) in states.iter().zip(&starts) {
            match s {
                Some(s) => {
                    sum += (0..LATENT).map(|k| (s[k] - z[t + h][k]).powi(2)).sum::<f64>() / LATENT as f64;
                    count += 1;
                }
                None => diverged += 1,
            }
        }
    }
    let nmse = if count > 0 && var > 0.0 { sum / count as f64 / var } else { f64::NAN };
    Ok((nmse, diverged as f64 / (count + diverged).max(1) as f64))
}

#[derive(Clone, Debug, Default)]
pub struct Attractors {
    pub fixed_points: usize,
    pub occupancy_fixed: f64,
    pub occupancy_diverged: f64,
    pub occupancy_other: f64,
    pub mean_steps_to_settle: f64,
}

/// Free-run z <- F(z, b = 0) from a 5 x 5 grid on [-2, 2]^2 for 200 steps (one native call per step).
pub fn attractors(model: &mut LevelModel) -> Result<Attractors, K1Error> {
    let starts: Vec<[f64; LATENT]> = (0..25).map(|i| [-2.0 + (i % 5) as f64, -2.0 + (i / 5) as f64]).collect();
    let mut z = starts.clone();
    let mut settled: Vec<Option<usize>> = vec![None; z.len()];
    let mut diverged = vec![false; z.len()];
    for step in 1..=200 {
        let live: Vec<usize> = (0..z.len()).filter(|&i| !diverged[i]).collect();
        if live.is_empty() {
            break;
        }
        let inputs: Vec<_> = live.iter().map(|&i| (z[i], 0.0)).collect();
        let next = match model.predict(&inputs) {
            Ok(n) => n,
            Err(K1Error(-2)) => {
                // Non-finite somewhere: step the live points one by one to isolate the divergent ones.
                let mut n = Vec::with_capacity(live.len());
                for &i in &live {
                    n.push(model.predict(&[(z[i], 0.0)]).map(|v| v[0]).unwrap_or([f64::NAN; LATENT]));
                }
                n
            }
            Err(e) => return Err(e),
        };
        for (j, &i) in live.iter().enumerate() {
            let n = next[j];
            if n.iter().any(|v| !v.is_finite() || v.abs() > 1e3) {
                diverged[i] = true;
                continue;
            }
            let moved = (0..LATENT).map(|k| (n[k] - z[i][k]).abs()).fold(0.0, f64::max);
            if moved < 1e-6 {
                settled[i].get_or_insert(step);
            } else {
                settled[i] = None;
            }
            z[i] = n;
        }
    }
    let n = z.len() as f64;
    let fixed: Vec<usize> = (0..z.len()).filter(|&i| !diverged[i] && settled[i].is_some()).collect();
    let mut points: Vec<(i64, i64)> =
        fixed.iter().map(|&i| ((z[i][0] * 100.0).round() as i64, (z[i][1] * 100.0).round() as i64)).collect();
    points.sort();
    points.dedup();
    let nd = diverged.iter().filter(|d| **d).count() as f64;
    Ok(Attractors {
        fixed_points: points.len(),
        occupancy_fixed: fixed.len() as f64 / n,
        occupancy_diverged: nd / n,
        occupancy_other: (n - nd - fixed.len() as f64) / n,
        mean_steps_to_settle: if fixed.is_empty() {
            f64::NAN
        } else {
            fixed.iter().map(|&i| settled[i].unwrap() as f64).sum::<f64>() / fixed.len() as f64
        },
    })
}
