//! R2's multi-step instrumentation (research/hecs_r2/PREREGISTRATION.md, sections 5-8). NEW in R2.
//!
//! A level model is a one-step map z' = G(z, b). The planner consumes it recursively: it feeds its own
//! prediction back as the next input (free-running, open loop, with the actions of the plan). This module
//! measures that recursive object against references on exactly the same held-out starts and targets:
//!
//! * **ECS free-running** rollouts with the true level actions, scored at horizons h (level steps). A rollout
//!   whose state is non-finite or has any |z| > 1e3 (whitened, unit-variance latent units) has met
//!   NUMERICAL_ESCAPE: a catastrophic-divergence sentinel; it is not scored further and its escape is counted;
//! * **DOMAIN_EXCURSION**, a different statement: a rollout whose state leaves the level's training envelope (the
//!   axis-aligned box of every training latent, `Domain`) at some step up to h. It says the model left the
//!   region its training data supports, not that it diverged; the truth's own excursion rate is reported next
//!   to it;
//! * **ECS teacher-forced** prediction: the one-step prediction of the same target z_{t+h} from the true
//!   z_{t+h-1}. Good teacher-forced and bad free-running error identifies compounding, not an inability to
//!   model the next transition;
//! * the **iterated linear reference** (R1's one-step least-squares model, training data only, iterated open
//!   loop); the **generator reference** (level 1 and temporal-shared levels, whose latent is a whitened
//!   observation): the true dynamics, clipping included, run from the factor estimate the current latent
//!   implies; the **analytic h-step floor** (unclipped dynamics; a consistency diagnostic, never a gate); the
//!   **constant** and **persistence** baselines;
//! * **capture** at h: (constant - ecs) / (constant - linear).
//!
//! Finite-horizon tangent diagnostics (never gates). With J_t = D_z G(z_t, b_t), the state Jacobian of the
//! level map at fixed action (`EcsTangent`: exact, from the level's W through the existing ABI), the ordered
//! derivative product P_h = J_{t+h-1} ... J_{t+1} J_t is the derivative of the h-step open-loop map (chain
//! rule), and sigma_max(P_h), its induced 2-norm, is the worst-case local amplification of a small state error
//! over h steps. It is measured on two different paths: the teacher path (J at the true held-out z_t: local
//! sensitivity on states of the real process) and the free-running path (J at the model's own recursive
//! predictions: sensitivity on the state distribution the planner's model creates for itself). The spectral
//! radius of one J does not control sigma_max(P_h) for a non-normal map (the mechanics tests construct a
//! counterexample), so the one-step radius is reported only descriptively. gamma_h = log sigma_max(P_h) / h is a
//! finite-time tangent growth rate, not a Lyapunov exponent: an exponent is an asymptotic limit that needs an
//! invariant measure and an almost-everywhere convergence argument (donor: openai/math
//! fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb, lean/OAI/Dynamics/StandardMap/Lyapunov/), none of which holds for a
//! learned, action-driven map over 16 steps. sigma_max depends on the coordinates (a change of basis C moves
//! log sigma_max by at most log cond(C), which vanishes only after dividing by h -> infinity); every number here
//! is in the level's whitened latent coordinates, the coordinates the planner uses.
//!
//! The generic rollout-error recurrence (`recurrence_bound`) is exact mathematics, not a measurement; the
//! empirical products above are local, not certified Lipschitz bounds. Every error is NMSE with
//! `validity::normalizer`'s held-out variance, as in R1.

use crate::encoder::Encoder;
use crate::hierarchy::LevelSequence;
use crate::k1::K1Error;
use crate::linalg::{ridge, ridge_apply, Mat};
use crate::model::{row, LevelModel, DIM, LATENT};
use crate::planner::candidates;
use crate::rng::Rng;
use crate::validity::normalizer;
use crate::world::World;

/// A rollout state beyond this (any coordinate, whitened units) has met NUMERICAL_ESCAPE.
pub const ESCAPE_BOUND: f64 = 1e3;
/// Initial perturbation size for the direct amplification diagnostic.
pub const EPSILON: f64 = 1e-4;
/// Starts used for the amplification, tangent and planner-candidate diagnostics (evenly subsampled).
pub const DIAGNOSTIC_STARTS: usize = 256;

pub type Z = [f64; LATENT];
/// A 2 x 2 matrix, rows first: m[i][j] = d out_i / d in_j.
pub type M2 = [[f64; 2]; 2];

fn escaped(z: &Z) -> bool {
    z.iter().any(|v| !v.is_finite() || v.abs() > ESCAPE_BOUND)
}

fn d2(a: &Z, b: &Z) -> f64 {
    (0..LATENT).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>() / LATENT as f64
}

fn norm(z: &Z) -> f64 {
    (z[0] * z[0] + z[1] * z[1]).sqrt()
}

// -- 2 x 2 linear algebra -------------------------------------------------------------------------------------

pub fn mat_mul(a: &M2, b: &M2) -> M2 {
    [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]
}

/// The largest singular value (induced 2-norm) of a 2 x 2 matrix, in closed form:
/// for M = [[a, b], [c, d]], sigma_max = (sqrt((a + d)^2 + (c - b)^2) + sqrt((a - d)^2 + (b + c)^2)) / 2.
pub fn sigma_max(m: &M2) -> f64 {
    let (a, b, c, d) = (m[0][0], m[0][1], m[1][0], m[1][1]);
    0.5 * (((a + d).powi(2) + (c - b).powi(2)).sqrt() + ((a - d).powi(2) + (b + c).powi(2)).sqrt())
}

/// Spectral radius of a 2 x 2 matrix (descriptive only: it does not bound sigma_max of a product).
pub fn spectral_radius(j: &M2) -> f64 {
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

/// The ordered derivative products of a path: entry k is P_{k+1} = J_k ... J_1 J_0 (the latest Jacobian on the
/// left), the derivative of the (k + 1)-step map.
pub fn ordered_products(js: &[M2]) -> Vec<M2> {
    let mut out = Vec::with_capacity(js.len());
    let mut p = [[1.0, 0.0], [0.0, 1.0]];
    for j in js {
        p = mat_mul(j, &p);
        out.push(p);
    }
    out
}

// -- the generic rollout-error recurrence (mathematics, not a measurement) -----------------------------------

/// The rollout-error bound. True and model transitions under the same action sequence, x_{t+1} = F_t(x_t) and
/// y_{t+1} = G_t(y_t), from x_0 = y_0. If on a domain D that contains both trajectories
/// ||F_t(x) - G_t(x)|| <= eps_t (uniform one-step discrepancy) and ||G_t(x) - G_t(y)|| <= L_t ||x - y||
/// (a Lipschitz bound of the model map on D), then with e_t = ||x_t - y_t||:
///   e_{t+1} = ||F_t(x_t) - G_t(x_t) + G_t(x_t) - G_t(y_t)|| <= eps_t + L_t e_t,
/// hence e_h <= sum_{i<h} eps_i prod_{i<j<h} L_j. Returns the bounds on e_1 .. e_n (e_0 = 0).
pub fn recurrence_bound(eps: &[f64], lip: &[f64]) -> Vec<f64> {
    assert_eq!(eps.len(), lip.len());
    let mut e = 0.0;
    eps.iter()
        .zip(lip)
        .map(|(ep, l)| {
            e = ep + l * e;
            e
        })
        .collect()
}

/// The closed form for constant eps and L: eps (L^h - 1) / (L - 1), or h eps when L = 1.
pub fn constant_bound(eps: f64, l: f64, h: usize) -> f64 {
    if l == 1.0 {
        h as f64 * eps
    } else {
        eps * (l.powi(h as i32) - 1.0) / (l - 1.0)
    }
}

// -- predictors -----------------------------------------------------------------------------------------------

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

/// The exact state Jacobian of a level's K1 map, from its W (read through the existing `copy_w` ABI; nothing
/// native is changed). K1 FORWARD computes, for output k with input row x = [z0, z1, b, sel0, sel1, 1],
/// y_k = sum_i phi(u_ki), u_ki = sum_a x_a W[a, i], phi(u) = u/2 + u^2/2 + u^3/2; hence
/// dy_k / dz_j = sum_i phi'(u_ki) W[j, i] with phi'(u) = 1/2 + u + 3u^2/2. `value` re-evaluates y with the same
/// operation order as the native kernel; the mechanics tests check it against FORWARD and the Jacobian against
/// centered differences of FORWARD.
pub struct EcsTangent {
    w: Vec<f64>,
    width: usize,
}

impl EcsTangent {
    pub fn of(model: &mut LevelModel) -> Result<EcsTangent, K1Error> {
        Ok(EcsTangent { w: model.ecs.w()?, width: model.width })
    }

    fn pre(&self, x: &[f64; DIM]) -> Vec<f64> {
        let mut u = vec![0.0; self.width];
        for (a, xa) in x.iter().enumerate() {
            let wa = &self.w[a * self.width..(a + 1) * self.width];
            for (ui, wi) in u.iter_mut().zip(wa) {
                *ui += xa * wi;
            }
        }
        u
    }

    pub fn value(&self, z: &Z, b: f64) -> Z {
        let mut out = [0.0; LATENT];
        for (k, o) in out.iter_mut().enumerate() {
            *o = self.pre(&row(z, b, k)).iter().map(|&u| 0.5 * u + 0.5 * (u * u) + 0.5 * (u * u) * u).sum();
        }
        out
    }

    pub fn jacobian(&self, z: &Z, b: f64) -> M2 {
        let mut j = [[0.0; 2]; 2];
        for (k, jk) in j.iter_mut().enumerate() {
            let u = self.pre(&row(z, b, k));
            for (input, jkj) in jk.iter_mut().enumerate() {
                let wj = &self.w[input * self.width..(input + 1) * self.width];
                *jkj = u.iter().zip(wj).map(|(&ui, &wi)| (0.5 + ui + 1.5 * ui * ui) * wi).sum();
            }
        }
        j
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

    /// The state Jacobian (constant): d z'_k / d z_j = W[j, k] (ridge maps inputs to outputs row-wise).
    pub fn jacobian(&self) -> M2 {
        [[self.w[(0, 0)], self.w[(1, 0)]], [self.w[(0, 1)], self.w[(1, 1)]]]
    }

    pub fn spectral_radius(&self) -> f64 {
        spectral_radius(&self.jacobian())
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

fn rotation(angle: f64) -> M2 {
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
/// action held for `env_stride` environment steps, slow factor clipped at the end of the level step) -> z.
pub struct Generator<'a> {
    world: &'a World,
    p: M2,
    pinv: M2,
    mean: [f64; 2],
    env_stride: usize,
}

impl<'a> Generator<'a> {
    pub fn new(world: &'a World, enc1: &'a Encoder, env_stride: usize) -> Generator<'a> {
        let (p, mean) = whitening(enc1);
        let p = [[p[(0, 0)], p[(0, 1)]], [p[(1, 0)], p[(1, 1)]]];
        let det = p[0][0] * p[1][1] - p[0][1] * p[1][0];
        let pinv = [[p[1][1] / det, -p[0][1] / det], [-p[1][0] / det, p[0][0] / det]];
        Generator { world, p, pinv, mean: [mean[0], mean[1]], env_stride }
    }

    /// (s, f) from a latent, and the unclipped slow value after the level step.
    fn factors(&self, z: &Z, b: f64) -> (f64, f64, f64) {
        let o = [
            self.pinv[0][0] * z[0] + self.pinv[0][1] * z[1] + self.mean[0],
            self.pinv[1][0] * z[0] + self.pinv[1][1] * z[1] + self.mean[1],
        ];
        let r = rotation(self.world.angle);
        let (s, f) = (r[0][0] * o[0] + r[1][0] * o[1], r[0][1] * o[0] + r[1][1] * o[1]);
        (s, f, s + self.world.spec.kappa * self.env_stride as f64 * b.clamp(-1.0, 1.0))
    }

    fn apply(&self, z: &Z, b: f64) -> Z {
        let (_, f, s_next) = self.factors(z, b);
        let s2 = s_next.clamp(-1.0, 1.0);
        let f2 = self.world.spec.rho.powi(self.env_stride as i32) * f;
        let r = rotation(self.world.angle);
        let o2 = [r[0][0] * s2 + r[0][1] * f2 - self.mean[0], r[1][0] * s2 + r[1][1] * f2 - self.mean[1]];
        [self.p[0][0] * o2[0] + self.p[0][1] * o2[1], self.p[1][0] * o2[0] + self.p[1][1] * o2[1]]
    }

    /// The generator's state Jacobian: P R diag(c, rho^e) R^T P^-1, c = 1 where the slow factor is not clipped
    /// at the end of the step and 0 where it is.
    pub fn jacobian(&self, z: &Z, b: f64) -> M2 {
        let (_, _, s_next) = self.factors(z, b);
        let c = if s_next.abs() <= 1.0 { 1.0 } else { 0.0 };
        let r = rotation(self.world.angle);
        let rt = [[r[0][0], r[1][0]], [r[0][1], r[1][1]]];
        let d = [[c, 0.0], [0.0, self.world.spec.rho.powi(self.env_stride as i32)]];
        mat_mul(&self.p, &mat_mul(&r, &mat_mul(&d, &mat_mul(&rt, &self.pinv))))
    }
}

impl Step for Generator<'_> {
    fn step(&mut self, inputs: &[(Z, f64)]) -> Result<Vec<Option<Z>>, K1Error> {
        Ok(inputs.iter().map(|(z, b)| Some(self.apply(z, *b))).collect())
    }
}

/// The generator's h-step error in the latent for UNCLIPPED dynamics, as NMSE: with n = env_stride * h
/// environment steps,
///   Q = R diag(n slow_noise^2, (1 - rho^(2n)) second_std^2) R^T + obs^2 I + obs^2 A A^T,  A = R diag(1, rho^n) R^T,
/// whitened by P: tr(P Q P^T) / LATENT / normalizer. R1's one-step floor is the case n = 1. Clipping is a
/// 1-Lipschitz contraction applied after independent noise, so for level 1 (per-step actions known) it can only
/// lower the slow factor's error of the plug-in predictor (Efron-Stein); for coarser levels the within-stride
/// action order is unknown and clipping can raise it. It is an expectation compared with finite-sample estimates.
/// R2 therefore reports it as a consistency diagnostic only.
pub fn analytic_floor(world: &World, enc1: &Encoder, env_stride: usize, h: usize, test: &[LevelSequence]) -> f64 {
    let w = &world.spec;
    let n = (env_stride * h) as i32;
    let r = rotation(world.angle);
    let rdr = |a: f64, b: f64| -> M2 {
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

// -- the training-envelope domain -----------------------------------------------------------------------------

/// The axis-aligned box of every training latent of a level (training data only; no margin).
#[derive(Clone, Copy, Debug)]
pub struct Domain {
    pub lo: Z,
    pub hi: Z,
}

impl Domain {
    pub fn of(train: &[LevelSequence]) -> Domain {
        let (mut lo, mut hi) = ([f64::INFINITY; LATENT], [f64::NEG_INFINITY; LATENT]);
        for z in train.iter().flat_map(|s| s.z.iter()) {
            for k in 0..LATENT {
                lo[k] = lo[k].min(z[k]);
                hi[k] = hi[k].max(z[k]);
            }
        }
        Domain { lo, hi }
    }

    pub fn contains(&self, z: &Z) -> bool {
        (0..LATENT).all(|k| z[k] >= self.lo[k] && z[k] <= self.hi[k])
    }
}

// -- rollouts and starts --------------------------------------------------------------------------------------

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

/// The held-out starts of every sequence: (start latent, the actions that follow, the true latents that follow,
/// and the start's (sequence, step) origin).
struct Starts {
    z0: Vec<Z>,
    actions: Vec<Vec<f64>>,
    truth: Vec<Vec<Z>>,
    origin: Vec<(usize, usize)>,
}

fn starts(test: &[LevelSequence], hmax: usize) -> Starts {
    let (mut z0, mut actions, mut truth, mut origin) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (si, s) in test.iter().enumerate() {
        for t in 0..s.b.len() {
            let n = hmax.min(s.b.len() - t);
            z0.push(s.z[t]);
            actions.push(s.b[t..t + n].to_vec());
            truth.push(s.z[t + 1..t + 1 + n].to_vec());
            origin.push((si, t));
        }
    }
    Starts { z0, actions, truth, origin }
}

/// Evenly subsampled diagnostic starts (at most `DIAGNOSTIC_STARTS`).
fn picked(n: usize) -> Vec<usize> {
    let stride = (n / DIAGNOSTIC_STARTS).max(1);
    (0..n).step_by(stride).take(DIAGNOSTIC_STARTS).collect()
}

fn score(
    rolls: &[Vec<Option<Z>>],
    truth: &[Vec<Z>],
    h: usize,
    var: f64,
    mask: Option<&[bool]>,
) -> (f64, f64, usize, f64) {
    let (mut sum, mut n, mut esc, mut targets, mut pn, mut tn) = (0.0, 0usize, 0usize, 0usize, 0.0, 0.0);
    for (i, (r, t)) in rolls.iter().zip(truth).enumerate() {
        if t.len() < h || mask.is_some_and(|m| !m[i]) {
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

/// Fraction of the starts with a target at h whose path left the domain (or escaped) at some step <= h.
fn excursion(paths: &[Vec<Option<Z>>], truth: &[Vec<Z>], h: usize, domain: &Domain) -> f64 {
    let (mut out, mut n) = (0usize, 0usize);
    for (p, t) in paths.iter().zip(truth) {
        if t.len() < h {
            continue;
        }
        n += 1;
        if p.iter().take(h).any(|z| z.is_none_or(|z| !domain.contains(&z))) {
            out += 1;
        }
    }
    out as f64 / n.max(1) as f64
}

fn quantile(mut v: Vec<f64>, q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn fraction(v: &[f64], f: impl Fn(f64) -> bool) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().filter(|x| f(**x)).count() as f64 / v.len() as f64
    }
}

// -- references (no ECS) --------------------------------------------------------------------------------------

/// One horizon's reference numbers (no ECS involved): what DESIGN may look at.
#[derive(Clone, Debug, Default)]
pub struct RefStats {
    pub h: usize,
    /// Held-out starts with a target at h.
    pub starts: usize,
    pub linear: f64,
    pub constant: f64,
    pub persistence: f64,
    pub generator: Option<f64>,
    pub floor: Option<f64>,
    /// Observation levels: the fraction of starts whose window [t, t + h) contains an environment step at which
    /// the world clipped the slow factor; and the linear and generator errors on the unclipped windows only.
    pub clip_fraction: Option<f64>,
    pub linear_unclipped: Option<f64>,
    pub generator_unclipped: Option<f64>,
    /// DOMAIN_EXCURSION of the truth itself and of the iterated linear reference.
    pub truth_excursion: f64,
    pub linear_excursion: f64,
    /// Tangent growth of the references: sigma_max(A^h) of the linear map; the generator's sigma_max(P_h) on
    /// the teacher path (median over the diagnostic starts).
    pub linear_sigma: f64,
    pub generator_sigma_median: Option<f64>,
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

    /// The persistence baseline's capture (z_{t+h} = z_t): what a trivial predictor achieves.
    pub fn persistence_capture(&self) -> f64 {
        (self.constant - self.persistence) / (self.constant - self.linear)
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
/// (observation frames only), constant and persistence baselines, on the held-out starts and targets of `test`.
/// `clips` (observation frames): per test sequence and level step, whether the world clipped the slow factor
/// during that level step.
pub fn references(
    train: &[LevelSequence],
    test: &[LevelSequence],
    frame: Frame,
    hs: &[usize],
    clips: Option<&[Vec<bool>]>,
) -> Vec<RefStats> {
    let hmax = *hs.iter().max().unwrap();
    let st = starts(test, hmax);
    let var = normalizer(test);
    let domain = Domain::of(train);
    let mut linear = Linear::fit(train);
    let lin = rollouts(&mut linear, &st.z0, &st.actions, hmax).expect("linear reference is infallible");
    let truth_paths: Vec<Vec<Option<Z>>> = st.truth.iter().map(|t| t.iter().map(|z| Some(*z)).collect()).collect();
    let mut generator = match frame {
        Frame::Observation { world, enc1, env_stride } => Some(Generator::new(world, enc1, env_stride)),
        Frame::Opaque => None,
    };
    let gen =
        generator.as_mut().map(|g| rollouts(g, &st.z0, &st.actions, hmax).expect("generator reference is infallible"));
    let mean = train_mean(train);
    let pick = picked(st.z0.len());
    let a = linear.jacobian();
    hs.iter()
        .map(|&h| {
            let (lin_e, _, n, _) = score(&lin, &st.truth, h, var, None);
            let (mut csum, mut psum) = (0.0, 0.0);
            for (z0, t) in st.z0.iter().zip(&st.truth).filter(|(_, t)| t.len() >= h) {
                csum += d2(&mean, &t[h - 1]);
                psum += d2(z0, &t[h - 1]);
            }
            let unclipped: Option<Vec<bool>> = clips.map(|c| {
                st.origin.iter().map(|&(s, t)| !c[s][t..(t + h).min(c[s].len())].iter().any(|x| *x)).collect()
            });
            let clip_fraction = unclipped.as_ref().map(|u| {
                let eligible: Vec<bool> = st.truth.iter().map(|t| t.len() >= h).collect();
                let clipped = u.iter().zip(&eligible).filter(|(u, e)| **e && !**u).count();
                clipped as f64 / eligible.iter().filter(|e| **e).count().max(1) as f64
            });
            let gen_sigma = generator.as_ref().map(|g| {
                let sig: Vec<f64> = pick
                    .iter()
                    .filter(|&&i| st.truth[i].len() >= h)
                    .map(|&i| {
                        let (s, t) = st.origin[i];
                        let js: Vec<M2> = (0..h).map(|k| g.jacobian(&test[s].z[t + k], test[s].b[t + k])).collect();
                        sigma_max(&ordered_products(&js)[h - 1])
                    })
                    .collect();
                quantile(sig, 0.5)
            });
            RefStats {
                h,
                starts: n,
                linear: lin_e,
                constant: csum / n.max(1) as f64 / var,
                persistence: psum / n.max(1) as f64 / var,
                generator: gen.as_ref().map(|g| score(g, &st.truth, h, var, None).0),
                floor: match frame {
                    Frame::Observation { world, enc1, env_stride } => {
                        Some(analytic_floor(world, enc1, env_stride, h, test))
                    }
                    Frame::Opaque => None,
                },
                clip_fraction,
                linear_unclipped: unclipped.as_ref().map(|u| score(&lin, &st.truth, h, var, Some(u)).0),
                generator_unclipped: unclipped
                    .as_ref()
                    .and_then(|u| gen.as_ref().map(|g| score(g, &st.truth, h, var, Some(u)).0)),
                truth_excursion: excursion(&truth_paths, &st.truth, h, &domain),
                linear_excursion: excursion(&lin, &st.truth, h, &domain),
                linear_sigma: sigma_max(&ordered_products(&vec![a; h])[h - 1]),
                generator_sigma_median: gen_sigma,
            }
        })
        .collect()
}

// -- the ECS level model ---------------------------------------------------------------------------------------

/// Finite-horizon derivative-product statistics of the ECS at one horizon (diagnostics, never gates).
#[derive(Clone, Debug, Default)]
pub struct TangentStats {
    /// sigma_max(P_h) on the teacher path (J at the true held-out states): median, 90th percentile, fraction > 1,
    /// and the median finite-time growth rate gamma_h = log sigma_max(P_h) / h.
    pub teacher_median: f64,
    pub teacher_p90: f64,
    pub teacher_over_1: f64,
    pub teacher_gamma_median: f64,
    /// The same on the free-running path (J at the model's own predictions, same actions); defined only for
    /// rollouts that have not escaped by step h - 1 (`free_defined` is that fraction of the diagnostic starts).
    pub free_median: f64,
    pub free_p90: f64,
    pub free_over_1: f64,
    pub free_gamma_median: f64,
    pub free_defined: f64,
    /// Median ||e_lin_h|| / ||e_h||: the first-order propagation of the true one-step residuals along the
    /// free-running tangent path (e_{k+1} = r_k + J_k e_k) against the actual free-running error. Descriptive.
    pub recursion_ratio_median: f64,
}

/// One horizon's numbers for one level model.
#[derive(Clone, Debug, Default)]
pub struct HorizonStats {
    pub h: usize,
    /// Held-out starts with a target at h.
    pub starts: usize,
    pub ecs_free: f64,
    /// NUMERICAL_ESCAPE: fraction of the free-running rollouts escaped by step h.
    pub ecs_escaped: f64,
    pub ecs_teacher: f64,
    pub linear: f64,
    pub constant: f64,
    pub persistence: f64,
    pub generator: Option<f64>,
    pub floor: Option<f64>,
    /// (constant - ecs_free) / (constant - linear); NaN if every rollout escaped.
    pub capture: f64,
    /// Mean |z_hat| over unescaped rollouts / mean |z| of the same targets.
    pub norm_ratio: f64,
    /// DOMAIN_EXCURSION of the ECS free-running rollouts; the truth's and the linear reference's.
    pub ecs_excursion: f64,
    pub truth_excursion: f64,
    pub linear_excursion: f64,
    /// Escaped fraction of passive (b = 0) ECS rollouts from the same starts.
    pub passive_escaped: f64,
    /// Planner-candidate rollouts (the planner's own action generator, block 1, anchors included) from the
    /// diagnostic starts: NUMERICAL_ESCAPE and DOMAIN_EXCURSION by step h.
    pub candidate_escaped: f64,
    pub candidate_excursion: f64,
    /// Direct perturbation amplification |dz_h| / |dz_0| of the ECS (median, 90th percentile, fraction > 10)
    /// and the linear reference's median, on the diagnostic starts.
    pub amp_median: f64,
    pub amp_p90: f64,
    pub amp_over_10: f64,
    pub linear_amp_median: f64,
    pub tangent: TangentStats,
    pub linear_sigma: f64,
    pub generator_sigma_median: Option<f64>,
}

/// Every horizon's statistics for one level model on its held-out sequences.
pub fn horizons(
    model: &mut LevelModel,
    train: &[LevelSequence],
    test: &[LevelSequence],
    frame: Frame,
    hs: &[usize],
    clips: Option<&[Vec<bool>]>,
    seed_path: &[u64],
) -> Result<Vec<HorizonStats>, K1Error> {
    let hmax = *hs.iter().max().unwrap();
    let st = starts(test, hmax);
    let var = normalizer(test);
    let domain = Domain::of(train);
    let refs = references(train, test, frame, hs, clips);
    let tangent = EcsTangent::of(model)?;
    let ecs = rollouts(model, &st.z0, &st.actions, hmax)?;
    let passive_actions: Vec<Vec<f64>> = st.actions.iter().map(|a| vec![0.0; a.len()]).collect();
    let passive = rollouts(model, &st.z0, &passive_actions, hmax)?;
    // Teacher-forced: the one-step prediction from every true state (index j of a sequence predicts z_{j+1}).
    let mut tf_inputs = Vec::new();
    let mut offsets = Vec::with_capacity(test.len());
    for s in test {
        offsets.push(tf_inputs.len());
        for j in 0..s.b.len() {
            tf_inputs.push((s.z[j], s.b[j]));
        }
    }
    let tf_pred = model.step(&tf_inputs)?;
    // Diagnostic starts: perturbation amplification with a deterministic direction per start, planner candidates.
    let mut linear = Linear::fit(train);
    let pick = picked(st.z0.len());
    let mut rng = Rng::new(0x5eed, seed_path);
    let dirs: Vec<Z> = pick
        .iter()
        .map(|_| {
            let a = rng.range(0.0, std::f64::consts::TAU);
            [a.cos(), a.sin()]
        })
        .collect();
    let p_z0: Vec<Z> = pick.iter().map(|&i| st.z0[i]).collect();
    let p_z1: Vec<Z> =
        pick.iter().zip(&dirs).map(|(&i, d)| [st.z0[i][0] + EPSILON * d[0], st.z0[i][1] + EPSILON * d[1]]).collect();
    let p_act: Vec<Vec<f64>> = pick.iter().map(|&i| st.actions[i].clone()).collect();
    let p_truth: Vec<Vec<Z>> = pick.iter().map(|&i| st.truth[i].clone()).collect();
    let base = rollouts(model, &p_z0, &p_act, hmax)?;
    let pert = rollouts(model, &p_z1, &p_act, hmax)?;
    let lbase = rollouts(&mut linear, &p_z0, &p_act, hmax)?;
    let lpert = rollouts(&mut linear, &p_z1, &p_act, hmax)?;
    let cand_path: Vec<u64> = seed_path.iter().copied().chain([1]).collect();
    let cand_actions = candidates(pick.len(), hmax, 1, &mut Rng::new(0x5eed, &cand_path));
    let cand = rollouts(model, &p_z0, &cand_actions, hmax)?;
    // Candidate rollouts have no truth: every one is eligible at every horizon.
    let cand_eligible: Vec<Vec<Z>> = vec![vec![[0.0; LATENT]; hmax]; pick.len()];
    let amps = |a: &[Vec<Option<Z>>], b: &[Vec<Option<Z>>], h: usize| -> Vec<f64> {
        a.iter()
            .zip(b)
            .filter_map(|(x, y)| match (x.get(h - 1).copied().flatten(), y.get(h - 1).copied().flatten()) {
                (Some(x), Some(y)) => Some(((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2)).sqrt() / EPSILON),
                _ => None,
            })
            .collect()
    };
    // Jacobians along the teacher and the free-running paths of the diagnostic starts (up to hmax steps).
    let teacher_js: Vec<Vec<M2>> = pick
        .iter()
        .map(|&i| {
            let (s, t) = st.origin[i];
            (0..st.truth[i].len()).map(|k| tangent.jacobian(&test[s].z[t + k], test[s].b[t + k])).collect()
        })
        .collect();
    let free_js: Vec<Vec<M2>> = pick
        .iter()
        .enumerate()
        .map(|(pi, &i)| {
            let mut js = Vec::new();
            let mut z = Some(st.z0[i]);
            for k in 0..st.actions[i].len() {
                match z {
                    Some(zk) => js.push(tangent.jacobian(&zk, st.actions[i][k])),
                    None => break,
                }
                z = base[pi].get(k).copied().flatten();
            }
            js
        })
        .collect();
    let teacher_p: Vec<Vec<M2>> = teacher_js.iter().map(|js| ordered_products(js)).collect();
    let free_p: Vec<Vec<M2>> = free_js.iter().map(|js| ordered_products(js)).collect();

    let mut out = Vec::new();
    for (r, &h) in refs.iter().zip(hs) {
        let (ecs_free, ecs_escaped, n, norm_ratio) = score(&ecs, &st.truth, h, var, None);
        let (_, passive_escaped, _, _) = score(&passive, &st.truth, h, var, None);
        // Teacher-forced on the same targets: start t of a sequence, target z_{t+h}, input the true z_{t+h-1}.
        let (mut tsum, mut tcount) = (0.0, 0usize);
        for (s, &offset) in test.iter().zip(&offsets) {
            for t in 0..s.b.len() {
                if t + h > s.b.len() {
                    continue;
                }
                if let Some(p) = tf_pred[offset + t + h - 1] {
                    tsum += d2(&p, &s.z[t + h]);
                    tcount += 1;
                }
            }
        }
        let ecs_teacher = if tcount == n { tsum / tcount.max(1) as f64 / var } else { f64::INFINITY };
        let a = amps(&base, &pert, h);
        let la = amps(&lbase, &lpert, h);
        let sig = |ps: &[Vec<M2>]| -> Vec<f64> {
            ps.iter()
                .zip(&p_truth)
                .filter(|(p, t)| t.len() >= h && p.len() >= h)
                .map(|(p, _)| sigma_max(&p[h - 1]))
                .collect()
        };
        let (ts, fs) = (sig(&teacher_p), sig(&free_p));
        let eligible = p_truth.iter().filter(|t| t.len() >= h).count().max(1) as f64;
        let gamma = |v: &[f64]| quantile(v.iter().map(|s| s.ln() / h as f64).collect(), 0.5);
        // First-order error recursion along the free path: e_{k+1} = r_k + J_k e_k, r_k the true one-step residual.
        let ratios: Vec<f64> = pick
            .iter()
            .enumerate()
            .filter(|&(pi, &i)| st.truth[i].len() >= h && free_js[pi].len() >= h && base[pi][h - 1].is_some())
            .filter_map(|(pi, &i)| {
                let (s, t) = st.origin[i];
                let mut e = [0.0; LATENT];
                for k in 0..h {
                    let pred = tf_pred[offsets[s] + t + k]?;
                    let r = [test[s].z[t + k + 1][0] - pred[0], test[s].z[t + k + 1][1] - pred[1]];
                    let j = free_js[pi][k];
                    e = [r[0] + j[0][0] * e[0] + j[0][1] * e[1], r[1] + j[1][0] * e[0] + j[1][1] * e[1]];
                }
                let zh = base[pi][h - 1]?;
                let actual = norm(&[st.truth[i][h - 1][0] - zh[0], st.truth[i][h - 1][1] - zh[1]]);
                (actual > 0.0 && e.iter().all(|v| v.is_finite())).then(|| norm(&e) / actual)
            })
            .collect();
        out.push(HorizonStats {
            h,
            starts: n,
            ecs_free,
            ecs_escaped,
            ecs_teacher,
            linear: r.linear,
            constant: r.constant,
            persistence: r.persistence,
            generator: r.generator,
            floor: r.floor,
            capture: (r.constant - ecs_free) / (r.constant - r.linear),
            norm_ratio,
            ecs_excursion: excursion(&ecs, &st.truth, h, &domain),
            truth_excursion: r.truth_excursion,
            linear_excursion: r.linear_excursion,
            passive_escaped,
            candidate_escaped: score(&cand, &cand_eligible, h, 1.0, None).1,
            candidate_excursion: excursion(&cand, &cand_eligible, h, &domain),
            amp_median: quantile(a.clone(), 0.5),
            amp_p90: quantile(a.clone(), 0.9),
            amp_over_10: fraction(&a, |v| v > 10.0),
            linear_amp_median: quantile(la, 0.5),
            tangent: TangentStats {
                teacher_median: quantile(ts.clone(), 0.5),
                teacher_p90: quantile(ts.clone(), 0.9),
                teacher_over_1: fraction(&ts, |v| v > 1.0),
                teacher_gamma_median: gamma(&ts),
                free_median: quantile(fs.clone(), 0.5),
                free_p90: quantile(fs.clone(), 0.9),
                free_over_1: fraction(&fs, |v| v > 1.0),
                free_gamma_median: gamma(&fs),
                free_defined: fs.len() as f64 / eligible,
                recursion_ratio_median: quantile(ratios, 0.5),
            },
            linear_sigma: r.linear_sigma,
            generator_sigma_median: r.generator_sigma_median,
        });
    }
    Ok(out)
}

/// The one-step state Jacobian (exact) at `DIAGNOSTIC_STARTS` held-out states with their actions:
/// (mean spectral radius, median spectral radius, fraction of radii > 1, median sigma_max). Descriptive only.
pub fn one_step_jacobians(tangent: &EcsTangent, test: &[LevelSequence]) -> (f64, f64, f64, f64) {
    let all: Vec<(Z, f64)> = test.iter().flat_map(|s| (0..s.b.len()).map(move |t| (s.z[t], s.b[t]))).collect();
    let js: Vec<M2> = picked(all.len()).into_iter().map(|i| tangent.jacobian(&all[i].0, all[i].1)).collect();
    let radii: Vec<f64> = js.iter().map(spectral_radius).filter(|r| r.is_finite()).collect();
    let sig: Vec<f64> = js.iter().map(sigma_max).filter(|r| r.is_finite()).collect();
    let mean = radii.iter().sum::<f64>() / radii.len().max(1) as f64;
    (mean, quantile(radii.clone(), 0.5), fraction(&radii, |r| r > 1.0), quantile(sig, 0.5))
}
