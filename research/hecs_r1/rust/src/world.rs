//! The controlled action-conditioned synthetic world (docs/research/HECS_R0_PREREGISTRATION.md).
//!
//! Ground truth: a slow factor `s` driven by the action, and a second factor `f` the action cannot
//! influence. In the SEPARATED world `f` is fast (an AR(1) with negative coefficient: high-frequency,
//! predictable one step ahead, unpredictable over long strides). In the MATCHED control `f` evolves on a
//! timescale close to `s` (an AR(1) with coefficient near one). Observations mix the two factors by a
//! world-specific rotation plus small noise; no level ever sees the ground truth directly.

use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldKind {
    Separated,
    Matched,
}

impl WorldKind {
    pub fn name(self) -> &'static str {
        match self {
            WorldKind::Separated => "SEPARATED",
            WorldKind::Matched => "MATCHED",
        }
    }
}

#[derive(Clone, Debug)]
pub struct WorldSpec {
    pub kind: WorldKind,
    /// Slow factor gain per unit action per step.
    pub kappa: f64,
    pub slow_noise: f64,
    /// AR(1) coefficient of the second factor.
    pub rho: f64,
    /// Stationary standard deviation of the second factor.
    pub second_std: f64,
    pub obs_noise: f64,
}

/// One world instance: its spec plus the mixing rotation drawn from the world seed.
#[derive(Clone, Debug)]
pub struct World {
    pub spec: WorldSpec,
    pub angle: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Truth {
    pub s: f64,
    pub f: f64,
}

impl World {
    pub fn new(spec: WorldSpec, rng: &mut Rng) -> World {
        let angle = rng.range(std::f64::consts::PI / 8.0, 3.0 * std::f64::consts::PI / 8.0);
        World { spec, angle }
    }

    pub fn observe(&self, t: Truth, rng: &mut Rng) -> [f64; 2] {
        let (c, s) = (self.angle.cos(), self.angle.sin());
        let n = self.spec.obs_noise;
        [c * t.s - s * t.f + n * rng.normal(), s * t.s + c * t.f + n * rng.normal()]
    }

    /// The noise-free observation of a ground-truth state (goal rendering).
    pub fn render(&self, t: Truth) -> [f64; 2] {
        let (c, s) = (self.angle.cos(), self.angle.sin());
        [c * t.s - s * t.f, s * t.s + c * t.f]
    }

    pub fn step(&self, t: Truth, action: f64, rng: &mut Rng) -> Truth {
        let a = action.clamp(-1.0, 1.0);
        let s = (t.s + self.spec.kappa * a + self.spec.slow_noise * rng.normal()).clamp(-1.0, 1.0);
        let r = self.spec.rho;
        let f = r * t.f + (1.0 - r * r).sqrt() * self.spec.second_std * rng.normal();
        Truth { s, f }
    }

    pub fn initial(&self, rng: &mut Rng) -> Truth {
        Truth { s: rng.range(-1.0, 1.0), f: self.spec.second_std * rng.normal() }
    }
}

/// One recorded trajectory: observations o_0..o_T, actions a_0..a_{T-1}, truths at every step.
#[derive(Clone, Debug)]
pub struct Trajectory {
    pub obs: Vec<[f64; 2]>,
    pub actions: Vec<f64>,
    pub truth: Vec<Truth>,
}

/// Exploration: piecewise-constant uniform actions held for 1..=8 steps.
pub fn explore(world: &World, steps: usize, rng: &mut Rng) -> Trajectory {
    let mut t = world.initial(rng);
    let mut obs = vec![world.observe(t, rng)];
    let mut truth = vec![t];
    let mut actions = Vec::with_capacity(steps);
    let (mut held, mut a) = (0u64, 0.0);
    for _ in 0..steps {
        if held == 0 {
            a = rng.range(-1.0, 1.0);
            held = 1 + rng.below(8);
        }
        held -= 1;
        actions.push(a);
        t = world.step(t, a, rng);
        truth.push(t);
        obs.push(world.observe(t, rng));
    }
    Trajectory { obs, actions, truth }
}

/// Spectral centroid (in cycles per step) of a series: sum_k k P_k / sum_k P_k over the positive DFT bins
/// of the mean-removed series. Deterministic O(n^2) DFT; n is small.
pub fn spectral_centroid(x: &[f64]) -> f64 {
    let n = x.len();
    let m = x.iter().sum::<f64>() / n as f64;
    let (mut num, mut den) = (0.0, 0.0);
    for k in 1..=n / 2 {
        let (mut re, mut im) = (0.0, 0.0);
        for (t, v) in x.iter().enumerate() {
            let ang = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
            re += (v - m) * ang.cos();
            im += (v - m) * ang.sin();
        }
        let p = re * re + im * im;
        num += k as f64 / n as f64 * p;
        den += p;
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}
