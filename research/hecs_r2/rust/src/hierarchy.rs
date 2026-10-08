//! The declarative hierarchy specification, level datasets and the bounded streaming controller.
//!
//! Clock law. Level 1 runs at the environment clock. Level l > 1 ticks every `stride` level-(l-1) steps.
//! Its latent at tick t encodes the window of the last `window` level-(l-1) latents ending at lower step
//! t * stride (past only, so the controller can compute it online). Its action b_t is the mean of the
//! level-(l-1) actions taken between ticks t and t + 1 (fixed mean pooling, every hierarchical condition).
//! A level sees only its lower level's latents and actions: never W, H or a of another level, and never
//! the ground truth.

use crate::encoder::Encoder;
use crate::model::{Transition, LATENT};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Representation {
    /// One level only.
    Flat,
    /// Temporal hierarchy in one latent space (no learned map between levels).
    TemporalShared,
    /// Temporal hierarchy with a separately learned encoder per boundary.
    HierarchicalDistinct,
}

impl Representation {
    pub fn name(self) -> &'static str {
        match self {
            Representation::Flat => "FLAT",
            Representation::TemporalShared => "TEMPORAL_SHARED",
            Representation::HierarchicalDistinct => "HIERARCHICAL_DISTINCT",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LevelSpec {
    /// Lower-level steps per tick (1 for level 1).
    pub stride: usize,
    /// Lower-level latents encoded per tick (1 for level 1).
    pub window: usize,
    /// ECS width N of this level's K1 state.
    pub width: usize,
    /// Planning horizon in this level's steps (top level only; lower levels plan exactly one upper tick).
    pub horizon: usize,
}

#[derive(Clone, Debug)]
pub struct HierarchySpec {
    pub name: String,
    pub representation: Representation,
    pub levels: Vec<LevelSpec>,
}

impl HierarchySpec {
    /// Environment steps per tick of level `l` (0-based).
    pub fn env_stride(&self, l: usize) -> usize {
        self.levels[..=l].iter().map(|s| s.stride).product()
    }

    /// Structural validation: every quantity bounded and consistent.
    pub fn validate(&self) -> Result<(), String> {
        let first = self.levels.first().ok_or("no levels")?;
        if first.stride != 1 || first.window != 1 {
            return Err("level 1 runs at the environment clock with window 1".into());
        }
        if self.levels.len() > 3 {
            return Err("R0 stops at three levels".into());
        }
        if (self.representation == Representation::Flat) != (self.levels.len() == 1) {
            return Err("exactly the flat representation has one level".into());
        }
        for l in &self.levels[1..] {
            if !(2..=8).contains(&l.stride) || !(1..=l.stride).contains(&l.window) {
                return Err(format!("bounded stride/window violated: {l:?}"));
            }
            let shared = self.representation == Representation::TemporalShared;
            if shared != (l.window == 1) {
                return Err("shared levels use window 1; distinct levels encode windows > 1".into());
            }
        }
        if self.levels.iter().any(|l| l.width == 0 || l.width > 256 || l.horizon == 0 || l.horizon > 64) {
            return Err("width or horizon out of bounds".into());
        }
        Ok(())
    }

    pub fn total_width(&self) -> usize {
        self.levels.iter().map(|l| l.width).sum()
    }
}

/// One level's sequence over an episode: latents z_0..z_n and actions b_0..b_{n-1}.
#[derive(Clone, Debug, Default)]
pub struct LevelSequence {
    pub z: Vec<[f64; LATENT]>,
    pub b: Vec<f64>,
}

impl LevelSequence {
    pub fn transitions(&self) -> Vec<Transition> {
        (0..self.b.len()).map(|t| Transition { z: self.z[t], b: self.b[t], next: self.z[t + 1] }).collect()
    }
}

/// Window ending at lower step `end` (oldest first); before the episode start the first latent repeats.
pub fn window(lower: &[[f64; LATENT]], end: usize, w: usize) -> Vec<[f64; LATENT]> {
    (0..w).map(|i| lower[(end + i + 1).saturating_sub(w)]).collect()
}

/// The upper sequence of a lower sequence under (stride, window) and an encoder.
pub fn lift(lower: &LevelSequence, spec: &LevelSpec, enc: &Encoder) -> LevelSequence {
    let n = lower.b.len() / spec.stride; // ticks with a complete action span
    let z = (0..=n).map(|t| enc.encode(&window(&lower.z, t * spec.stride, spec.window))).collect();
    let b = (0..n)
        .map(|t| lower.b[t * spec.stride..(t + 1) * spec.stride].iter().sum::<f64>() / spec.stride as f64)
        .collect();
    LevelSequence { z, b }
}

/// Raw (window, pooled action, next window) triples for fitting a distinct encoder.
pub fn window_triples(lower: &LevelSequence, spec: &LevelSpec) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let n = lower.b.len() / spec.stride;
    let flat = |t: usize| -> Vec<f64> {
        window(&lower.z, t * spec.stride, spec.window).iter().flat_map(|z| z.iter().copied()).collect()
    };
    let (mut u, mut b, mut next) = (Vec::new(), Vec::new(), Vec::new());
    for t in 0..n {
        u.push(flat(t));
        b.push(lower.b[t * spec.stride..(t + 1) * spec.stride].iter().sum::<f64>() / spec.stride as f64);
        next.push(flat(t + 1));
    }
    (u, b, next)
}

/// Bounded streaming controller state: per level, a ring of the last `window` lower latents at tick
/// boundaries and the action sum since the last tick. Its size is fixed by the spec, whatever the episode
/// length; no trajectory is stored.
#[derive(Clone, Debug)]
pub struct Controller {
    pub spec: HierarchySpec,
    /// rings[l]: the last levels[l].window latents of level l - 1 (rings[0] unused).
    rings: Vec<Vec<[f64; LATENT]>>,
    /// Lower steps since the last tick of each upper level.
    phase: Vec<usize>,
    /// Current latent of each level.
    pub current: Vec<[f64; LATENT]>,
}

impl Controller {
    /// Start an episode at level-1 latent z1 (history before the start repeats the first latent).
    pub fn start(spec: &HierarchySpec, encoders: &[Encoder], z1: [f64; LATENT]) -> Controller {
        let mut current = vec![z1];
        let mut rings = vec![vec![]];
        for l in 1..spec.levels.len() {
            let w = spec.levels[l].window;
            let ring = vec![current[l - 1]; w];
            current.push(encoders[l].encode(&ring));
            rings.push(ring);
        }
        Controller { spec: spec.clone(), rings, phase: vec![0; spec.levels.len()], current }
    }

    /// Advance one environment step to the new level-1 latent. Returns the levels that ticked.
    ///
    /// Each step of level l - 1 enters level l's ring (the last `window` lower latents); level l ticks,
    /// encoding its ring, every `stride` lower steps. This is exactly the window law of `lift`.
    pub fn step(&mut self, encoders: &[Encoder], z1: [f64; LATENT]) -> Vec<usize> {
        self.current[0] = z1;
        let mut ticked = vec![0];
        for l in 1..self.spec.levels.len() {
            if ticked.last() != Some(&(l - 1)) {
                break;
            }
            let ring = &mut self.rings[l];
            ring.remove(0);
            ring.push(self.current[l - 1]);
            self.phase[l] += 1;
            if self.phase[l] < self.spec.levels[l].stride {
                break;
            }
            self.phase[l] = 0;
            self.current[l] = encoders[l].encode(&self.rings[l]);
            ticked.push(l);
        }
        ticked
    }

    /// The latents planning starts from: level 1's current latent and, for each upper level, the encoding
    /// of its most recent window (the window ending at the latest lower step). At an upper tick this equals
    /// that level's tick latent; between ticks it is the up-to-date encoding of the present, as H-JEPA
    /// encodes the current observation at every level before planning.
    pub fn planning_latents(&self, encoders: &[Encoder]) -> Vec<[f64; LATENT]> {
        (0..self.spec.levels.len())
            .map(|l| if l == 0 { self.current[0] } else { encoders[l].encode(&self.rings[l]) })
            .collect()
    }

    /// Resident bytes of the controller state (constant for a spec).
    pub fn resident_bytes(&self) -> usize {
        let latents: usize = self.rings.iter().map(|r| r.len()).sum::<usize>() + self.current.len();
        latents * LATENT * 8 + self.phase.len() * 8
    }
}
