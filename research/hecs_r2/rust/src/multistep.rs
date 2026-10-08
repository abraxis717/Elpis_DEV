//! R2's multi-step instrumentation (research/hecs_r2/PREREGISTRATION.md, sections 5-6). NEW in R2.
//!
//! A level model is a one-step map z' = F(z, b). The planner consumes it recursively: it feeds its own
//! prediction back as the next input (free-running, open loop, with the actions of the plan). This module
//! measures that recursive object against references on exactly the same held-out starts and targets:
//!
//! * **ECS free-running** rollouts with the true level actions, scored at horizons h (level steps); a rollout
//!   that leaves the finite region or the escape bound (|z_k| > 1e3 in a whitened, unit-variance latent)
//!   is *escaped* and is not scored further;
//! * **ECS teacher-forced** prediction: the one-step prediction of the same target z_{t+h} from the true
//!   z_{t+h-1}. Good teacher-forced and bad free-running error identifies compounding instability, not an
//!   inability to model the next transition;
//! * the **iterated linear reference**: the one-step least-squares model z' ~ [z, b, 1] fit on the training
//!   partition only (R1's one-step reference), iterated open loop like the ECS;
//! * the **generator reference** (level 1 and temporal-shared levels, whose latent is a whitened observation):
//!   the true dynamics run from the factor estimate the current latent implies; and the **analytic floor**:
//!   the generator's h-step innovation and observation noise in the latent's units;
//! * the **constant baseline** (the training mean of the next latent);
//! * **capture** at h: (constant - ecs) / (constant - linear), the fraction of the linearly predictable h-step
//!   structure the free-running ECS keeps (R1's one-step capture, at horizon h).
//!
//! Diagnostics (described, not gates unless the specification names them): norm growth, passive (b = 0)
//! escape, local perturbation amplification of the recursive map, and the spectral radius of the empirical
//! one-step Jacobian at held-out states. Every number is NMSE with `validity::normalizer`'s held-out
//! variance, as in R1.

use crate::encoder::Encoder;
use crate::hierarchy::LevelSequence;
use crate::k1::K1Error;
use crate::linalg::{ridge, ridge_apply, Mat};
use crate::model::{LevelModel, LATENT};
use crate::rng::Rng;
use crate::validity::normalizer;
use crate::world::World;

/// A rollout state beyond this (any coordinate, whitened units) has escaped.
pub const ESCAPE_BOUND: f64 = 1e3;
/// Finite-difference step for the Jacobian and initial perturbation size for amplification.
pub const EPSILON: f64 = 1e-4;
/// Starts used for amplification and Jacobian statistics (evenly subsampled, deterministic).
pub const DIAGNOSTIC_STARTS: usize = 256;

type Z = [f64; LATENT];

fn escaped(z: &Z) -> bool {
    z.iter().any(|v| !v.is_finite() || v.abs() > ESCAPE_BOUND)
}

fn d2(a: &Z, b: &Z) -> f64 {
    (0..LATENT).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>() / LATENT as f64
}

fn norm(z: &Z) -> f64 {
    (z[0] * z[0] + z[1] * z[1]).sqrt()
}

/// A one-step predictor over batches (ECS or reference).
pub trait Step {
    /// Next latents for a batch; `None` for a non-finite prediction.
    fn step(&mut self, inputs: &[(Z, f64)]) -> Result<Vec<Option<Z>>, K1Error>;
}

/// Inputs per native QUERY batch (LATENT rows each). FORWARD has no row cap; the chunk bounds the work of
/// `predict_lossy`'s row-by-row retry when a batch holds a non-finite row. Results do not depend on it.
const CHUNK: usize = crate::spec::MAX_ROWS / LATENT;

impl Step for LevelModel<'_> {
    fn step(&mut self, inputs: &[(Z, f64)]) -> Result<Vec<Option<Z>>, K1Error> {
        let mut out = Vec::with_capacity(inputs.len());
        for chunk in inputs.chunks(CHUNK) {
            out.extend(self.predict_lossy(chunk)?);
        }
        Ok(out)
    }
}

/// The iterated one-step linear reference: least squares z' ~ [z0, z1, b, 1] on training transitions only.
pub struct Linear {
    w: Mat,
    c: Vec<f64>,
}

impl Linear {
    pub fn fit(train: &[LevelSequence]) -> Linear {
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for s in train {
            for t in 0..s.b.len() {
                x.push(vec![s.z[t][0], s.z[t][1], s.b[t]]);
                y.push(s.z[t + 1].to_vec());
            }
        }
        let (w, c) = ridge(&x, &y, 1e-9);
        Linear { w, c }
    }

    pub fn apply(&self, z: &Z, b: f64) -> Z {
        let p = ridge_apply(&self.w, &self.c, &[z[0], z[1], b]);
        [p[0], p[1]]
    }

    /// The spectral radius of the latent part of the linear map (its local, and global, Jacobian).
    pub fn spectral_radius(&self) -> f64 {
        let j = [[self.w[(0, 0)], self.w[(0, 1)]], [self.w[(1, 0)], self.w[(1, 1)]]];
        spectral_radius(&j)
    }
}

impl Step for Linear {
    fn step(&mut self, inputs: &[(Z, f64)]) -> Result<Vec<Option<Z>>, K1Error> {
        Ok(inputs.iter().map(|(z, b)| Some(self.apply(z, *b))).collect())
    }
}

/// How a level latent relates to the world's factors: a whitened observation at `env_stride` environment
/// steps per level step (level 1, temporal-shared levels), or a learned window encoding (no generator frame).
#[derive(Clone, Copy)]
pub enum Frame<'a> {
    Observation { world: &'a World, enc1: &'a Encoder, env_stride: usize },
    Opaque,
}

fn rotation(angle: f64) -> [[f64; 2]; 2] {
    let (c, s) = (angle.cos(), angle.sin());
    [[c, -s], [s, c]]
}

fn whitening(enc1: &Encoder) -> (&Mat, &[f64]) {
    match enc1 {
        Encoder::Affine { p, mean, .. } => (p, mean),
        Encoder::Shared => panic!("level 1 is a whitening map"),
    }
}

/// The true dynamics as a latent predictor: z -> (s, f) estimate -> one level step of the generator (the level
/// action held for `env_stride` environment steps, unclipped within the step, clipped at its end) -> z.
pub struct Generator<'a> {
    world: &'a World,
    p: &'a Mat,
    mean: &'a [f64],
    env_stride: usize,
}

impl<'a> Generator<'a> {
    pub fn new(world: &'a World, enc1: &'a Encoder, env_stride: usize) -> Generator<'a> {
        let (p, mean) = whitening(enc1);
        Generator { world, p, mean, env_stride }
    }

    fn apply(&self, z: &Z, b: f64) -> Z {
        let p = self.p;
        let det = p[(0, 0)] * p[(1, 1)] - p[(0, 1)] * p[(1, 0)];
        let pinv = [[p[(1, 1)] / det, -p[(0, 1)] / det], [-p[(1, 0)] / det, p[(0, 0)] / det]];
        let o = [
            pinv[0][0] * z[0] + pinv[0][1] * z[1] + self.mean[0],
            pinv[1][0] * z[0] + pinv[1][1] * z[1] + self.mean[1],
        ];
        let r = rotation(self.world.angle);
        // (s, f) = R^T o
        let (s, f) = (r[0][0] * o[0] + r[1][0] * o[1], r[0][1] * o[0] + r[1][1] * o[1]);
        let w = &self.world.spec;
        let e = self.env_stride as f64;
        let s2 = (s + w.kappa * e * b.clamp(-1.0, 1.0)).clamp(-1.0, 1.0);
        let f2 = w.rho.powi(self.env_stride as i32) * f;
        let o2 = [r[0][0] * s2 + r[0][1] * f2 - self.mean[0], r[1][0] * s2 + r[1][1] * f2 - self.mean[1]];
        [p[(0, 0)] * o2[0] + p[(0, 1)] * o2[1], p[(1, 0)] * o2[0] + p[(1, 1)] * o2[1]]
    }
}

impl Step for Generator<'_> {
    fn step(&mut self, inputs: &[(Z, f64)]) -> Result<Vec<Option<Z>>, K1Error> {
        Ok(inputs.iter().map(|(z, b)| Some(self.apply(z, *b))).collect())
    }
}

/// The generator's h-step error floor in the latent, as NMSE: with n = env_stride * h environment steps,
///   Q = R diag(n slow_noise^2, (1 - rho^(2n)) second_std^2) R^T + obs^2 I + obs^2 A A^T,  A = R diag(1, rho^n) R^T,
/// whitened by P: tr(P Q P^T) / LATENT / normalizer (unclipped). R1's one-step floor is the case n = 1.
pub fn analytic_floor(world: &World, enc1: &Encoder, env_stride: usize, h: usize, test: &[LevelSequence]) -> f64 {
    let w = &world.spec;
    let n = (env_stride * h) as i32;
    let r = rotation(world.angle);
    let rdr = |a: f64, b: f64| -> [[f64; 2]; 2] {
        let mut m = [[0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                m[i][j] = r[i][0] * a * r[j][0] + r[i][1] * b * r[j][1];
            }
        }
        m
    };
    let innovation = rdr(n as f64 * w.slow_noise.powi(2), (1.0 - w.rho.powi(2 * n)) * w.second_std.powi(2));
    let a = rdr(1.0, w.rho.powi(n));
    let o2 = w.obs_noise * w.obs_noise;
    let mut q = Mat::zeros(2, 2);
    for i in 0..2 {
        for j in 0..2 {
            let aat = a[i][0] * a[j][0] + a[i][1] * a[j][1];
            q[(i, j)] = innovation[i][j] + o2 * aat + if i == j { o2 } else { 0.0 };
        }
    }
    let (p, _) = whitening(enc1);
    let qz = p.mul(&q).mul(&p.t());
    (qz[(0, 0)] + qz[(1, 1)]) / LATENT as f64 / normalizer(test)
}

/// Batched open-loop rollouts from `starts` under per-start action sequences, up to `hmax` steps. Entry k of a
/// rollout is the state after k + 1 steps; `None` once the rollout has escaped (and for every later step).
pub fn rollouts(
    model: &mut dyn Step,
    starts: &[Z],
    actions: &[Vec<f64>],
    hmax: usize,
) -> Result<Vec<Vec<Option<Z>>>, K1Error> {
    let mut out: Vec<Vec<Option<Z>>> = vec![Vec::with_capacity(hmax); starts.len()];
    let mut state: Vec<Option<Z>> = starts.iter().map(|z| Some(*z)).collect();
    for k in 0..hmax {
        let live: Vec<usize> = (0..starts.len()).filter(|&i| state[i].is_some() && k < actions[i].len()).collect();
        let inputs: Vec<(Z, f64)> = live.iter().map(|&i| (state[i].unwrap(), actions[i][k])).collect();
        let next = if inputs.is_empty() { Vec::new() } else { model.step(&inputs)? };
        let mut done = vec![false; starts.len()];
        for (&i, n) in live.iter().zip(next) {
            let n = n.filter(|z| !escaped(z));
            state[i] = n;
            out[i].push(n);
            done[i] = true;
        }
        for i in 0..starts.len() {
            if !done[i] && k < actions[i].len() {
                state[i] = None;
                out[i].push(None);
            }
        }
    }
    Ok(out)
}

/// The held-out starts of every sequence: (start latent, the actions that follow, the true latents that follow).
struct Starts {
    z0: Vec<Z>,
    actions: Vec<Vec<f64>>,
    truth: Vec<Vec<Z>>,
}

fn starts(test: &[LevelSequence], hmax: usize) -> Starts {
    let (mut z0, mut actions, mut truth) = (Vec::new(), Vec::new(), Vec::new());
    for s in test {
        for t in 0..s.b.len() {
            let n = hmax.min(s.b.len() - t);
            z0.push(s.z[t]);
            actions.push(s.b[t..t + n].to_vec());
            truth.push(s.z[t + 1..t + 1 + n].to_vec());
        }
    }
    Starts { z0, actions, truth }
}

/// One horizon's reference numbers (no ECS involved): what DESIGN may look at.
#[derive(Clone, Debug, Default)]
pub struct RefStats {
    pub h: usize,
    /// Held-out starts with a target at h.
    pub starts: usize,
    pub linear: f64,
    pub constant: f64,
    pub generator: Option<f64>,
    pub floor: Option<f64>,
}

impl RefStats {
    /// The fraction of the constant baseline's h-step error the iterated linear reference removes.
    pub fn explained(&self) -> f64 {
        1.0 - self.linear / self.constant
    }

    /// The generator's capture on the same footing as the ECS's: (constant - generator) / (constant - linear).
    pub fn generator_capture(&self) -> Option<f64> {
        self.generator.map(|g| (self.constant - g) / (self.constant - self.linear))
    }
}

fn train_mean(train: &[LevelSequence]) -> Z {
    let (mut m, mut n) = ([0.0; LATENT], 0.0);
    for s in train {
        for z in &s.z[1..] {
            m[0] += z[0];
            m[1] += z[1];
            n += 1.0;
        }
    }
    [m[0] / n, m[1] / n]
}

/// The reference numbers at every horizon: iterated linear (fit on `train` only), generator and analytic floor
/// (observation frames only) and the constant baseline, on the held-out starts and targets of `test`.
pub fn references(train: &[LevelSequence], test: &[LevelSequence], frame: Frame, hs: &[usize]) -> Vec<RefStats> {
    let hmax = *hs.iter().max().unwrap();
    let st = starts(test, hmax);
    let var = normalizer(test);
    let mut linear = Linear::fit(train);
    let lin = rollouts(&mut linear, &st.z0, &st.actions, hmax).expect("linear reference is infallible");
    let gen = match frame {
        Frame::Observation { world, enc1, env_stride } => {
            let mut g = Generator::new(world, enc1, env_stride);
            Some(rollouts(&mut g, &st.z0, &st.actions, hmax).expect("generator reference is infallible"))
        }
        Frame::Opaque => None,
    };
    let mean = train_mean(train);
    hs.iter()
        .map(|&h| {
            let (linear, _, n, _) = score(&lin, &st.truth, h, var);
            let mut csum = 0.0;
            for t in st.truth.iter().filter(|t| t.len() >= h) {
                csum += d2(&mean, &t[h - 1]);
            }
            RefStats {
                h,
                starts: n,
                linear,
                constant: csum / n.max(1) as f64 / var,
                generator: gen.as_ref().map(|g| score(g, &st.truth, h, var).0),
                floor: match frame {
                    Frame::Observation { world, enc1, env_stride } => {
                        Some(analytic_floor(world, enc1, env_stride, h, test))
                    }
                    Frame::Opaque => None,
                },
            }
        })
        .collect()
}

/// One horizon's numbers for one level model.
#[derive(Clone, Debug, Default)]
pub struct HorizonStats {
    pub h: usize,
    /// Held-out starts with a target at h.
    pub starts: usize,
    pub ecs_free: f64,
    pub ecs_escaped: f64,
    pub ecs_teacher: f64,
    pub linear: f64,
    pub constant: f64,
    pub generator: Option<f64>,
    pub floor: Option<f64>,
    /// (constant - ecs_free) / (constant - linear); NaN if every rollout escaped.
    pub capture: f64,
    /// Mean |z_hat| over unescaped rollouts / mean |z| of the same targets.
    pub norm_ratio: f64,
    /// Escaped fraction of passive (b = 0) ECS rollouts from the same starts.
    pub passive_escaped: f64,
    /// Perturbation amplification |dz_h| / |dz_0| of the ECS (median, 90th percentile, fraction > 10) and the
    /// linear reference's median, on `DIAGNOSTIC_STARTS` subsampled starts.
    pub amp_median: f64,
    pub amp_p90: f64,
    pub amp_over_10: f64,
    pub linear_amp_median: f64,
}

fn score(rolls: &[Vec<Option<Z>>], truth: &[Vec<Z>], h: usize, var: f64) -> (f64, f64, usize, f64) {
    let (mut sum, mut n, mut esc, mut targets, mut pn, mut tn) = (0.0, 0usize, 0usize, 0usize, 0.0, 0.0);
    for (r, t) in rolls.iter().zip(truth) {
        if t.len() < h {
            continue;
        }
        targets += 1;
        match r.get(h - 1).copied().flatten() {
            Some(z) => {
                sum += d2(&z, &t[h - 1]);
                n += 1;
                pn += norm(&z);
                tn += norm(&t[h - 1]);
            }
            None => esc += 1,
        }
    }
    let nmse = if n > 0 { sum / n as f64 / var } else { f64::NAN };
    let ratio = if n > 0 && tn > 0.0 { pn / tn } else { f64::NAN };
    (nmse, esc as f64 / targets.max(1) as f64, targets, ratio)
}

fn quantile(mut v: Vec<f64>, q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// Every horizon's statistics for one level model on its held-out sequences.
pub fn horizons(
    model: &mut LevelModel,
    train: &[LevelSequence],
    test: &[LevelSequence],
    frame: Frame,
    hs: &[usize],
    seed_path: &[u64],
) -> Result<Vec<HorizonStats>, K1Error> {
    let hmax = *hs.iter().max().unwrap();
    let st = starts(test, hmax);
    let var = normalizer(test);
    let refs = references(train, test, frame, hs);
    let ecs = rollouts(model, &st.z0, &st.actions, hmax)?;
    let passive_actions: Vec<Vec<f64>> = st.actions.iter().map(|a| vec![0.0; a.len()]).collect();
    let passive = rollouts(model, &st.z0, &passive_actions, hmax)?;
    // Teacher-forced: the one-step prediction from every true state (index j of a sequence predicts z_{j+1}).
    let mut tf_inputs = Vec::new();
    for s in test {
        for j in 0..s.b.len() {
            tf_inputs.push((s.z[j], s.b[j]));
        }
    }
    let tf_pred = model.step(&tf_inputs)?;
    // Perturbation amplification on evenly subsampled starts with a deterministic direction per start.
    let mut linear = Linear::fit(train);
    let stride = (st.z0.len() / DIAGNOSTIC_STARTS).max(1);
    let picked: Vec<usize> = (0..st.z0.len()).step_by(stride).take(DIAGNOSTIC_STARTS).collect();
    let mut rng = Rng::new(0x5eed, seed_path);
    let dirs: Vec<Z> = picked
        .iter()
        .map(|_| {
            let a = rng.range(0.0, std::f64::consts::TAU);
            [a.cos(), a.sin()]
        })
        .collect();
    let p_z0: Vec<Z> = picked.iter().map(|&i| st.z0[i]).collect();
    let p_z1: Vec<Z> =
        picked.iter().zip(&dirs).map(|(&i, d)| [st.z0[i][0] + EPSILON * d[0], st.z0[i][1] + EPSILON * d[1]]).collect();
    let p_act: Vec<Vec<f64>> = picked.iter().map(|&i| st.actions[i].clone()).collect();
    let base = rollouts(model, &p_z0, &p_act, hmax)?;
    let pert = rollouts(model, &p_z1, &p_act, hmax)?;
    let lbase = rollouts(&mut linear, &p_z0, &p_act, hmax)?;
    let lpert = rollouts(&mut linear, &p_z1, &p_act, hmax)?;
    let amps = |a: &[Vec<Option<Z>>], b: &[Vec<Option<Z>>], h: usize| -> Vec<f64> {
        a.iter()
            .zip(b)
            .filter_map(|(x, y)| match (x.get(h - 1).copied().flatten(), y.get(h - 1).copied().flatten()) {
                (Some(x), Some(y)) => Some(((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2)).sqrt() / EPSILON),
                _ => None,
            })
            .collect()
    };

    let mut out = Vec::new();
    for (r, &h) in refs.iter().zip(hs) {
        let (ecs_free, ecs_escaped, n, norm_ratio) = score(&ecs, &st.truth, h, var);
        let (_, passive_escaped, _, _) = score(&passive, &st.truth, h, var);
        // Teacher-forced on the same targets: start t of a sequence, target z_{t+h}, input the true z_{t+h-1}.
        let (mut tsum, mut tcount, mut offset) = (0.0, 0usize, 0usize);
        for s in test {
            for t in 0..s.b.len() {
                if t + h > s.b.len() {
                    continue;
                }
                if let Some(p) = tf_pred[offset + t + h - 1] {
                    tsum += d2(&p, &s.z[t + h]);
                    tcount += 1;
                }
            }
            offset += s.b.len();
        }
        let ecs_teacher = if tcount == n { tsum / tcount.max(1) as f64 / var } else { f64::INFINITY };
        let a = amps(&base, &pert, h);
        let la = amps(&lbase, &lpert, h);
        let over =
            if a.is_empty() { f64::NAN } else { a.iter().filter(|v| **v > 10.0).count() as f64 / a.len() as f64 };
        out.push(HorizonStats {
            h,
            starts: n,
            ecs_free,
            ecs_escaped,
            ecs_teacher,
            linear: r.linear,
            constant: r.constant,
            generator: r.generator,
            floor: r.floor,
            capture: (r.constant - ecs_free) / (r.constant - r.linear),
            norm_ratio,
            passive_escaped,
            amp_median: quantile(a.clone(), 0.5),
            amp_p90: quantile(a, 0.9),
            amp_over_10: over,
            linear_amp_median: quantile(la, 0.5),
        });
    }
    Ok(out)
}

/// Spectral radius of a 2 x 2 matrix.
pub fn spectral_radius(j: &[[f64; 2]; 2]) -> f64 {
    let tr = j[0][0] + j[1][1];
    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    let disc = tr * tr / 4.0 - det;
    if disc >= 0.0 {
        let r = disc.sqrt();
        (tr / 2.0 + r).abs().max((tr / 2.0 - r).abs())
    } else {
        det.abs().sqrt()
    }
}

/// The empirical one-step Jacobian dF/dz (central differences) at `DIAGNOSTIC_STARTS` held-out states with
/// their actions: (mean spectral radius, median, fraction > 1). States where a difference is non-finite are
/// skipped.
pub fn jacobians(model: &mut LevelModel, test: &[LevelSequence]) -> Result<(f64, f64, f64), K1Error> {
    let all: Vec<(Z, f64)> = test.iter().flat_map(|s| (0..s.b.len()).map(move |t| (s.z[t], s.b[t]))).collect();
    let stride = (all.len() / DIAGNOSTIC_STARTS).max(1);
    let picked: Vec<(Z, f64)> = all.into_iter().step_by(stride).take(DIAGNOSTIC_STARTS).collect();
    let mut inputs = Vec::with_capacity(picked.len() * 4);
    for (z, b) in &picked {
        for k in 0..LATENT {
            for sign in [1.0, -1.0] {
                let mut x = *z;
                x[k] += sign * EPSILON;
                inputs.push((x, *b));
            }
        }
    }
    let pred = model.step(&inputs)?;
    let mut radii = Vec::new();
    for c in pred.chunks(4) {
        if let [Some(a), Some(b), Some(c2), Some(d)] = [c[0], c[1], c[2], c[3]] {
            let j = [
                [(a[0] - b[0]) / (2.0 * EPSILON), (c2[0] - d[0]) / (2.0 * EPSILON)],
                [(a[1] - b[1]) / (2.0 * EPSILON), (c2[1] - d[1]) / (2.0 * EPSILON)],
            ];
            radii.push(spectral_radius(&j));
        }
    }
    let mean = radii.iter().sum::<f64>() / radii.len().max(1) as f64;
    let over = radii.iter().filter(|r| **r > 1.0).count() as f64 / radii.len().max(1) as f64;
    Ok((mean, quantile(radii, 0.5), over))
}
