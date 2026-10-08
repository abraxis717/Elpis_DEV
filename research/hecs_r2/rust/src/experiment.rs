//! H-ECS R2 experiment driver: worlds, level models, one- and multi-step measurements, paired planning, gates
//! and disposition.
//!
//! Derived from H-ECS R1 (research/hecs_r1/rust/src/experiment.rs; research/hecs_r2/SOURCE_EQUIVALENCE.json).
//! Unchanged from R1: the world data, the pool of trained levels (`build_pool`), the probes, the planning
//! episodes and the oracle. Changed: every level reports R2's multi-step block (`multistep`) next to R1's
//! one-step adequacy; task validity covers every level and every consumed horizon (T1-T3); `design_instance`
//! (reference numbers only, no ECS rollout statistic) and `calibration_instance` serve DESIGN and DEV
//! calibration; `evaluate` applies R2's law (task validity, then M, then H2); seeds are recorded as decimal
//! strings.

use std::collections::HashMap;

use crate::analysis::{attractors, horizon_error, probe_nmse};
use crate::encoder::{encode_obs, Encoder};
use crate::hierarchy::{lift, window_triples, Controller, HierarchySpec, LevelSequence, Representation};
use crate::json::Json;
use crate::k1::{K1Error, K1};
use crate::model::{LevelModel, TrainTrace, DIM, LATENT};
use crate::multistep::{self, Frame, HorizonStats, RefStats};
use crate::planner::{candidates, plan_flat, plan_hierarchical};
use crate::rng::Rng;
use crate::spec::{self, gates};
use crate::validity::{analytic_floor, constant_baseline, linear_reference, persistence_baseline, Diagnostics};
use crate::world::{explore, spectral_centroid, Trajectory, Truth, World, WorldKind};

fn log(msg: &str) {
    eprintln!("[hecs] {msg}");
}

const KIND_TAG: [u64; 2] = [11, 13];
fn kind_tag(k: WorldKind) -> u64 {
    match k {
        WorldKind::Separated => KIND_TAG[0],
        WorldKind::Matched => KIND_TAG[1],
    }
}

/// One world instance with its fixed training and held-out exploration data.
pub struct WorldData {
    pub world: World,
    pub train: Vec<Trajectory>,
    pub test: Vec<Trajectory>,
    pub enc1: Encoder,
}

pub fn world_data(kind: WorldKind, seed: u64) -> WorldData {
    let mut rng = Rng::new(spec::BASE_SEED, &[seed, kind_tag(kind), 1]);
    let world = World::new(spec::world(kind), &mut rng);
    let train: Vec<_> = (0..spec::TRAIN_EPISODES).map(|_| explore(&world, spec::EPISODE_STEPS, &mut rng)).collect();
    let test: Vec<_> = (0..spec::TEST_EPISODES).map(|_| explore(&world, spec::EPISODE_STEPS, &mut rng)).collect();
    let obs: Vec<Vec<f64>> = train.iter().flat_map(|t| t.obs.iter().map(|o| o.to_vec())).collect();
    let enc1 = Encoder::whitening(&obs);
    WorldData { world, train, test, enc1 }
}

fn level1(enc1: &Encoder, t: &Trajectory) -> LevelSequence {
    LevelSequence { z: t.obs.iter().map(|o| encode_obs(enc1, o)).collect(), b: t.actions.clone() }
}

/// A trained level shared by every condition whose levels up to it are identical.
pub struct PoolEntry<'a> {
    pub key: String,
    pub encoder: Encoder,
    pub model: Option<LevelModel<'a>>,
    pub train: Vec<LevelSequence>,
    pub test: Vec<LevelSequence>,
    pub trace: TrainTrace,
    pub env_stride: usize,
}

fn shuffled(mut v: Vec<crate::model::Transition>, rng: &mut Rng) -> Vec<crate::model::Transition> {
    for i in (1..v.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        v.swap(i, j);
    }
    v
}

fn train_entry<'a>(
    k1: &'a K1,
    key: &str,
    encoder: Encoder,
    train: Vec<LevelSequence>,
    test: Vec<LevelSequence>,
    width: usize,
    env_stride: usize,
    steps: u64,
    seed: u64,
    kind: WorldKind,
) -> Result<PoolEntry<'a>, K1Error> {
    let tag: u64 = key.bytes().fold(7u64, |h, b| h.wrapping_mul(131).wrapping_add(b as u64));
    let mut rng = Rng::new(spec::BASE_SEED, &[seed, kind_tag(kind), 2, tag]);
    let mut model = LevelModel::new(k1, width, spec::MAX_ROWS, spec::train_spec(steps).init_scale, &mut rng)?;
    let data = shuffled(train.iter().flat_map(|s| s.transitions()).collect(), &mut rng);
    let trace = model.train(&data, &spec::train_spec(steps));
    Ok(PoolEntry { key: key.into(), encoder, model: Some(model), train, test, trace, env_stride })
}

/// One level's data before any ECS training: its pool key, encoder, width, lifted sequences and environment
/// stride per level step.
pub struct LevelData {
    pub key: String,
    pub encoder: Encoder,
    pub width: usize,
    pub train: Vec<LevelSequence>,
    pub test: Vec<LevelSequence>,
    pub env_stride: usize,
}

/// Every distinct level used by the conditions (FLAT widths, and each hierarchy path), before any ECS trains:
/// R1's `build_pool` without the training. The encoders are closed form, fixed from lower training data alone.
pub fn level_sequences(data: &WorldData) -> Vec<LevelData> {
    let l1_train: Vec<_> = data.train.iter().map(|t| level1(&data.enc1, t)).collect();
    let l1_test: Vec<_> = data.test.iter().map(|t| level1(&data.enc1, t)).collect();
    let mut levels = Vec::new();
    for width in [36usize, 72, 108] {
        levels.push(LevelData {
            key: format!("L1/N{width}"),
            encoder: data.enc1.clone(),
            width,
            train: l1_train.clone(),
            test: l1_test.clone(),
            env_stride: 1,
        });
    }
    for (repr, tag) in [(Representation::TemporalShared, "TS"), (Representation::HierarchicalDistinct, "HD")] {
        let cond = spec::conditions().into_iter().find(|c| c.representation == repr && c.levels.len() == 3).unwrap();
        let (mut lower_train, mut lower_test) = (l1_train.clone(), l1_test.clone());
        for l in 1..3 {
            let ls = cond.levels[l];
            let encoder = match repr {
                Representation::TemporalShared => Encoder::Shared,
                _ => {
                    let (mut u, mut b, mut n) = (Vec::new(), Vec::new(), Vec::new());
                    for s in &lower_train {
                        let (u1, b1, n1) = window_triples(s, &ls);
                        u.extend(u1);
                        b.extend(b1);
                        n.extend(n1);
                    }
                    Encoder::predictable(&u, &b, &n)
                }
            };
            let up_train: Vec<_> = lower_train.iter().map(|s| lift(s, &ls, &encoder)).collect();
            let up_test: Vec<_> = lower_test.iter().map(|s| lift(s, &ls, &encoder)).collect();
            levels.push(LevelData {
                key: format!("{tag}/L{}", l + 1),
                encoder,
                width: ls.width,
                train: up_train.clone(),
                test: up_test.clone(),
                env_stride: cond.env_stride(l),
            });
            lower_train = up_train;
            lower_test = up_test;
        }
    }
    levels
}

/// Train every distinct level (R1's pool: the same keys, data, widths and per-key training streams).
pub fn build_pool<'a>(
    k1: &'a K1,
    data: &WorldData,
    steps: u64,
    seed: u64,
    kind: WorldKind,
) -> Result<Vec<PoolEntry<'a>>, K1Error> {
    level_sequences(data)
        .into_iter()
        .map(|d| {
            log(&format!("  train {} ({steps} steps)", d.key));
            train_entry(k1, &d.key, d.encoder, d.train, d.test, d.width, d.env_stride, steps, seed, kind)
        })
        .collect()
}

/// Level 1's one-step task diagnostics (R1's, unchanged): analytic floor, linear reference, constant and
/// persistence baselines and the ECS one-step NMSE (infinite on any divergence or a refused training).
pub fn diagnostics(entry: &mut PoolEntry, data: &WorldData) -> Result<Diagnostics, K1Error> {
    let seqs: Vec<_> = entry.test.iter().map(|s| (s.z.clone(), s.b.clone())).collect();
    let ecs = match entry.model.as_mut() {
        Some(m) if entry.trace.refused.is_none() => match horizon_error(m, &seqs, 1)? {
            (e, 0.0) => e,
            _ => f64::INFINITY,
        },
        _ => f64::INFINITY,
    };
    Ok(Diagnostics {
        floor: analytic_floor(&data.world, &data.enc1, &entry.test),
        linear: linear_reference(&entry.train, &entry.test),
        constant: constant_baseline(&entry.train, &entry.test),
        persistence: persistence_baseline(&entry.test),
        ecs,
    })
}

/// T1 on one instance: level 1's one-step reference removes enough of the constant error and sits at its
/// analytic floor (R1's V1A and V1B).
pub fn t1_holds(d: &Diagnostics) -> bool {
    use gates::*;
    d.reference_predictable(T1A_MIN_EXPLAINED) && d.reference_at_floor(T1B_FLOOR_TOL_ABS, T1B_FLOOR_TOL_REL)
}

/// Any trained level's one-step numbers (ecs, linear, constant); ecs is infinite on divergence or refusal.
pub fn one_step(entry: &mut PoolEntry) -> Result<(f64, f64, f64), K1Error> {
    let seqs: Vec<_> = entry.test.iter().map(|s| (s.z.clone(), s.b.clone())).collect();
    let ecs = match entry.model.as_mut() {
        Some(m) if entry.trace.refused.is_none() => match horizon_error(m, &seqs, 1)? {
            (e, 0.0) => e,
            _ => f64::INFINITY,
        },
        _ => f64::INFINITY,
    };
    Ok((ecs, linear_reference(&entry.train, &entry.test), constant_baseline(&entry.train, &entry.test)))
}

/// (constant - ecs) / (constant - linear): the fraction of the linearly removable error the ECS removes.
pub fn capture(ecs: f64, linear: f64, constant: f64) -> f64 {
    if ecs.is_finite() {
        (constant - ecs) / (constant - linear)
    } else {
        f64::NEG_INFINITY
    }
}

/// Level 1 and the temporal-shared levels: the latent is a whitened observation (generator frame).
pub fn observation_level(key: &str) -> bool {
    key.starts_with("L1/") || key.starts_with("TS/")
}

/// How a pool level's latent relates to the world: level 1 and temporal-shared levels are whitened observations
/// at their environment stride; distinct levels are window encodings (no generator frame).
pub fn frame<'a>(key: &str, env_stride: usize, data: &'a WorldData) -> Frame<'a> {
    if observation_level(key) {
        Frame::Observation { world: &data.world, enc1: &data.enc1, env_stride }
    } else {
        Frame::Opaque
    }
}

/// T3 at one (level, consumed horizon): (T3A, T3B). T3B applies to observation levels only (true elsewhere).
pub fn t3_holds(r: &RefStats, observation: bool) -> (bool, bool) {
    use gates::*;
    let a = r.explained() >= T3A_MIN_EXPLAINED;
    let b = !observation || r.floor.is_some_and(|f| (r.linear - f).abs() <= T3B_FLOOR_TOL_ABS + T3B_FLOOR_TOL_REL * f);
    (a, b)
}

fn key_tag(key: &str) -> u64 {
    key.bytes().fold(7u64, |h, b| h.wrapping_mul(131).wrapping_add(b as u64))
}

pub fn diagnostics_json(d: &Diagnostics) -> Json {
    use gates::*;
    Json::obj(vec![
        ("analytic_floor", Json::Num(d.floor)),
        ("linear_reference", Json::Num(d.linear)),
        ("constant_baseline", Json::Num(d.constant)),
        ("persistence_baseline", Json::Num(d.persistence)),
        ("ecs_one_step", Json::Num(d.ecs)),
        ("explained_by_linear", Json::Num(1.0 - d.linear / d.constant)),
        ("linear_minus_floor", Json::Num(d.linear - d.floor)),
        ("ecs_capture", Json::Num(d.capture())),
        ("T1A_reference_predictable", Json::Bool(d.reference_predictable(T1A_MIN_EXPLAINED))),
        ("T1B_reference_at_floor", Json::Bool(d.reference_at_floor(T1B_FLOOR_TOL_ABS, T1B_FLOOR_TOL_REL))),
    ])
}

fn opt(v: Option<f64>) -> Json {
    v.map_or(Json::Null, Json::Num)
}

pub fn refs_json(rs: &[RefStats]) -> Json {
    Json::Arr(
        rs.iter()
            .map(|r| {
                Json::obj(vec![
                    ("h", Json::Int(r.h as i64)),
                    ("starts", Json::Int(r.starts as i64)),
                    ("linear", Json::Num(r.linear)),
                    ("constant", Json::Num(r.constant)),
                    ("generator", opt(r.generator)),
                    ("analytic_floor", opt(r.floor)),
                    ("explained_by_linear", Json::Num(r.explained())),
                    ("generator_capture", opt(r.generator_capture())),
                ])
            })
            .collect(),
    )
}

pub fn horizons_json(hs: &[HorizonStats]) -> Json {
    Json::Arr(
        hs.iter()
            .map(|h| {
                Json::obj(vec![
                    ("h", Json::Int(h.h as i64)),
                    ("starts", Json::Int(h.starts as i64)),
                    ("ecs_free_running", Json::Num(h.ecs_free)),
                    ("ecs_escaped", Json::Num(h.ecs_escaped)),
                    ("ecs_teacher_forced", Json::Num(h.ecs_teacher)),
                    ("linear", Json::Num(h.linear)),
                    ("constant", Json::Num(h.constant)),
                    ("generator", opt(h.generator)),
                    ("analytic_floor", opt(h.floor)),
                    ("capture", Json::Num(h.capture)),
                    ("norm_ratio", Json::Num(h.norm_ratio)),
                    ("passive_escaped", Json::Num(h.passive_escaped)),
                    ("amplification_median", Json::Num(h.amp_median)),
                    ("amplification_p90", Json::Num(h.amp_p90)),
                    ("amplification_over_10", Json::Num(h.amp_over_10)),
                    ("linear_amplification_median", Json::Num(h.linear_amp_median)),
                ])
            })
            .collect(),
    )
}

/// One trained level's numbers used by the law.
#[derive(Clone, Debug)]
pub struct LevelMetrics {
    pub ecs1: f64,
    pub linear1: f64,
    pub constant1: f64,
    pub capture1: f64,
    pub horizons: Vec<HorizonStats>,
    pub probe_s: f64,
    pub probe_f: f64,
    /// Empirical one-step Jacobian spectral radius: (mean, median, fraction > 1); the linear reference's.
    pub jacobian: (f64, f64, f64),
    pub linear_radius: f64,
}

impl LevelMetrics {
    pub fn at(&self, h: usize) -> &HorizonStats {
        self.horizons.iter().find(|s| s.h == h).expect("a measured horizon")
    }
}

/// Pool indices of a condition's levels.
pub fn level_keys(c: &HierarchySpec) -> Vec<String> {
    let tag = match c.representation {
        Representation::TemporalShared => "TS",
        Representation::HierarchicalDistinct => "HD",
        Representation::Flat => "",
    };
    (0..c.levels.len())
        .map(|l| if l == 0 { format!("L1/N{}", c.levels[0].width) } else { format!("{tag}/L{}", l + 1) })
        .collect()
}

fn find(pool: &[PoolEntry], key: &str) -> usize {
    pool.iter().position(|e| e.key == key).expect("pool key")
}

/// Probe the level latents for the ground-truth factors at the window-end environment times.
fn probes(entry: &PoolEntry, data: &WorldData) -> (f64, f64) {
    let collect = |seqs: &[LevelSequence], trajs: &[Trajectory]| {
        let mut z = Vec::new();
        let mut truth: Vec<Truth> = Vec::new();
        for (s, t) in seqs.iter().zip(trajs) {
            for (i, zi) in s.z.iter().enumerate() {
                z.push(*zi);
                truth.push(t.truth[i * entry.env_stride]);
            }
        }
        (z, truth)
    };
    let (ztr, ttr) = collect(&entry.train, &data.train);
    let (zte, tte) = collect(&entry.test, &data.test);
    let s = probe_nmse(
        &ztr,
        &ttr.iter().map(|t| t.s).collect::<Vec<_>>(),
        &zte,
        &tte.iter().map(|t| t.s).collect::<Vec<_>>(),
    );
    let f = probe_nmse(
        &ztr,
        &ttr.iter().map(|t| t.f).collect::<Vec<_>>(),
        &zte,
        &tte.iter().map(|t| t.f).collect::<Vec<_>>(),
    );
    (s, f)
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// Every quantity of one world instance that needs no ECS: the references of every level at every horizon, T3
/// at every consumed (condition, level, horizon), the oracle planner (V2), the level-1 probes (V3) and the
/// spectral centroids (V4). Deterministic from (kind, seed): the evidence embeds `json` verbatim and the
/// mechanics tests recompute it from the recorded seed.
pub struct References {
    pub json: Json,
    pub keys: Vec<String>,
    pub refs: HashMap<String, Vec<RefStats>>,
    pub oracle: f64,
    pub l1_probe: (f64, f64),
    pub centroid_ratio: f64,
}

pub fn ref_at(rs: &[RefStats], h: usize) -> &RefStats {
    rs.iter().find(|r| r.h == h).expect("a measured horizon")
}

/// Every (condition, level key, consumed horizon) in report order.
pub fn consumed_pairs() -> Vec<(String, String, usize)> {
    let mut out = Vec::new();
    for c in spec::conditions() {
        for (l, key) in level_keys(&c).into_iter().enumerate() {
            out.push((c.name.clone(), key, spec::consumed_horizon(&c, l)));
        }
    }
    out
}

pub fn references(data: &WorldData, seed: u64, kind: WorldKind) -> References {
    let centroid = |f: fn(&Truth) -> f64| -> f64 {
        data.train.iter().map(|t| spectral_centroid(&t.truth.iter().map(f).collect::<Vec<_>>())).sum::<f64>()
            / data.train.len() as f64
    };
    let (cs, cf) = (centroid(|t| t.s), centroid(|t| t.f));
    let oracle = oracle_success(data, seed, kind);
    let (mut keys, mut refs, mut level_json) = (Vec::new(), HashMap::new(), Vec::new());
    let mut l1_probe = (f64::NAN, f64::NAN);
    for d in level_sequences(data) {
        let r = multistep::references(&d.train, &d.test, frame(&d.key, d.env_stride, data), &spec::HORIZONS);
        let entry = PoolEntry {
            key: d.key.clone(),
            encoder: d.encoder,
            model: None,
            train: d.train,
            test: d.test,
            trace: TrainTrace::default(),
            env_stride: d.env_stride,
        };
        let (ps, pf) = probes(&entry, data);
        if d.key == "L1/N36" {
            l1_probe = (ps, pf);
        }
        level_json.push(Json::obj(vec![
            ("key", Json::str(&d.key)),
            ("env_stride", Json::Int(d.env_stride as i64)),
            ("probe_nmse_slow", Json::Num(ps)),
            ("probe_nmse_fast", Json::Num(pf)),
            ("by_horizon", refs_json(&r)),
        ]));
        keys.push(d.key.clone());
        refs.insert(d.key, r);
    }
    let consumed: Vec<Json> = consumed_pairs()
        .into_iter()
        .map(|(c, key, h)| {
            let r = ref_at(&refs[&key], h);
            let (a, b) = t3_holds(r, observation_level(&key));
            Json::obj(vec![
                ("condition", Json::str(c)),
                ("key", Json::str(&key)),
                ("h", Json::Int(h as i64)),
                ("explained_by_linear", Json::Num(r.explained())),
                ("linear", Json::Num(r.linear)),
                ("constant", Json::Num(r.constant)),
                ("generator", opt(r.generator)),
                ("analytic_floor", opt(r.floor)),
                ("T3A", Json::Bool(a)),
                ("T3B", Json::Bool(b)),
            ])
        })
        .collect();
    let json = Json::obj(vec![
        ("world", Json::str(kind.name())),
        ("seed", Json::str(seed.to_string())),
        ("mixing_angle", Json::Num(data.world.angle)),
        ("spectral_centroid_slow", Json::Num(cs)),
        ("spectral_centroid_second", Json::Num(cf)),
        ("centroid_ratio", Json::Num(cf / cs)),
        ("oracle_success", Json::Num(oracle)),
        ("l1_probe_nmse_slow_fast", Json::nums(&[l1_probe.0, l1_probe.1])),
        ("levels", Json::Arr(level_json)),
        ("consumed", Json::Arr(consumed)),
    ]);
    References { json, keys, refs, oracle, l1_probe, centroid_ratio: cf / cs }
}

/// Every trained level's one-step adequacy (ecs, linear, constant, capture, refusal) and the smallest capture
/// over the pool (negative infinity on a refusal or a divergence).
fn one_step_levels(pool: &mut [PoolEntry]) -> Result<(Json, f64), K1Error> {
    let (mut rows, mut min) = (Vec::new(), f64::INFINITY);
    for e in pool.iter_mut() {
        let (ecs, linear, constant) = one_step(e)?;
        let c = if e.trace.refused.is_some() { f64::NEG_INFINITY } else { capture(ecs, linear, constant) };
        min = if c.is_nan() { f64::NEG_INFINITY } else { min.min(c) };
        rows.push(Json::obj(vec![
            ("key", Json::str(&e.key)),
            ("ecs_one_step", Json::Num(ecs)),
            ("linear_reference", Json::Num(linear)),
            ("constant_baseline", Json::Num(constant)),
            ("capture", Json::Num(c)),
            ("refused", e.trace.refused.as_ref().map_or(Json::Null, Json::str)),
        ]));
    }
    Ok((Json::Arr(rows), min))
}

/// DESIGN, one world instance: `references` (no ECS) and, at every grid budget, T1 and every trained level's
/// one-step adequacy (T2 and the calibration margin) and refusals (V5). No ECS prediction beyond one step and
/// no planning with an ECS: no multi-step or hypothesis quantity is computed.
pub fn design_instance(k1: &K1, kind: WorldKind, seed: u64) -> Result<Json, K1Error> {
    let data = world_data(kind, seed);
    let refs = references(&data, seed, kind);
    let mut grid = Vec::new();
    for steps in spec::CALIBRATION_STEPS {
        let mut pool = build_pool(k1, &data, steps, seed, kind)?;
        let l1 = find(&pool, "L1/N36");
        let d = diagnostics(&mut pool[l1], &data)?;
        let (levels, min) = one_step_levels(&mut pool)?;
        grid.push(Json::obj(vec![
            ("steps", Json::Int(steps as i64)),
            ("t1", diagnostics_json(&d)),
            ("T1", Json::Bool(t1_holds(&d))),
            ("levels", levels),
            ("min_capture", Json::Num(min)),
            ("T2", Json::Bool(min >= gates::T2_MIN_CAPTURE)),
            ("calibration_margin", Json::Bool(t1_holds(&d) && min >= gates::CAL_MIN_CAPTURE)),
        ]));
    }
    Ok(Json::obj(vec![
        ("world", Json::str(kind.name())),
        ("seed", Json::str(seed.to_string())),
        ("references", refs.json),
        ("grid", Json::Arr(grid)),
    ]))
}

/// DEV calibration, one world instance at one budget: T1 and the calibration margin on every level.
pub fn calibration_instance(k1: &K1, kind: WorldKind, seed: u64, steps: u64) -> Result<(Json, bool), K1Error> {
    let data = world_data(kind, seed);
    let mut pool = build_pool(k1, &data, steps, seed, kind)?;
    let l1 = find(&pool, "L1/N36");
    let d = diagnostics(&mut pool[l1], &data)?;
    let (levels, min) = one_step_levels(&mut pool)?;
    let ok = t1_holds(&d) && min >= gates::CAL_MIN_CAPTURE;
    Ok((
        Json::obj(vec![
            ("world", Json::str(kind.name())),
            ("seed", Json::str(seed.to_string())),
            ("t1", diagnostics_json(&d)),
            ("T1", Json::Bool(t1_holds(&d))),
            ("levels", levels),
            ("min_capture", Json::Num(min)),
            ("calibrated", Json::Bool(ok)),
        ]),
        ok,
    ))
}

/// Per-level measurements of one trained pool entry: R1's one-step adequacy and R2's multi-step block.
fn level_report(
    k1: &K1,
    entry: &mut PoolEntry,
    data: &WorldData,
    seed: u64,
    kind: WorldKind,
) -> Result<(Json, LevelMetrics), K1Error> {
    let (probe_s, probe_f) = probes(entry, data);
    let (ecs1, linear1, constant1) = one_step(entry)?;
    let persistence = persistence_baseline(&entry.test);
    let fr = frame(&entry.key, entry.env_stride, data);
    let path = [seed, kind_tag(kind), 7, key_tag(&entry.key)];
    let model = entry.model.as_mut().unwrap();
    let horizons = multistep::horizons(model, &entry.train, &entry.test, fr, &spec::HORIZONS, &path)?;
    let jacobian = multistep::jacobians(model, &entry.test)?;
    let linear_radius = multistep::Linear::fit(&entry.train).spectral_radius();
    let att = attractors(model)?;
    let digest = model.ecs.digest()?;
    let counters = model.ecs.counters()?;
    let image = k1.image_bytes(DIM, model.width);
    let json = Json::obj(vec![
        ("key", Json::str(&entry.key)),
        ("width", Json::Int(model.width as i64)),
        ("env_stride", Json::Int(entry.env_stride as i64)),
        ("epoch", Json::Int(model.ecs.epoch() as i64)),
        ("state_digest", Json::str(hex(&digest))),
        ("probe_nmse_slow", Json::Num(probe_s)),
        ("probe_nmse_fast", Json::Num(probe_f)),
        (
            "one_step",
            Json::obj(vec![
                ("ecs", Json::Num(ecs1)),
                ("linear_reference", Json::Num(linear1)),
                ("constant_baseline", Json::Num(constant1)),
                ("persistence_baseline", Json::Num(persistence)),
                ("capture", Json::Num(capture(ecs1, linear1, constant1))),
            ]),
        ),
        ("multistep", horizons_json(&horizons)),
        (
            "jacobian",
            Json::obj(vec![
                ("ecs_spectral_radius_mean", Json::Num(jacobian.0)),
                ("ecs_spectral_radius_median", Json::Num(jacobian.1)),
                ("ecs_fraction_over_1", Json::Num(jacobian.2)),
                ("linear_spectral_radius", Json::Num(linear_radius)),
            ]),
        ),
        (
            "training",
            Json::obj(vec![
                (
                    "checkpoints",
                    Json::Arr(
                        entry
                            .trace
                            .checkpoints
                            .iter()
                            .map(|(e, mse, wn)| {
                                Json::obj(vec![
                                    ("epoch", Json::Int(*e as i64)),
                                    ("train_mse", Json::Num(*mse)),
                                    ("w_frobenius", Json::Num(*wn)),
                                ])
                            })
                            .collect(),
                    ),
                ),
                ("rows_learned", Json::Int(entry.trace.rows_learned as i64)),
                ("multiply_adds", Json::Int(entry.trace.multiply_adds as i64)),
                ("refused", entry.trace.refused.as_ref().map_or(Json::Null, Json::str)),
            ]),
        ),
        (
            "attractors",
            Json::obj(vec![
                ("fixed_points", Json::Int(att.fixed_points as i64)),
                ("occupancy_fixed", Json::Num(att.occupancy_fixed)),
                ("occupancy_diverged", Json::Num(att.occupancy_diverged)),
                ("occupancy_other", Json::Num(att.occupancy_other)),
                ("mean_steps_to_settle", Json::Num(att.mean_steps_to_settle)),
            ]),
        ),
        (
            "memory_bytes",
            Json::obj(vec![
                ("k1_image", Json::Int(image as i64)),
                ("k1_workspace", Json::Int(counters.workspace_bytes as i64)),
                ("encoder_parameters", Json::Int((entry.encoder.parameters() * 8) as i64)),
            ]),
        ),
        (
            "encoder_predictability",
            match &entry.encoder {
                Encoder::Affine { predictability, .. } if !predictability.is_empty() => Json::nums(predictability),
                _ => Json::Null,
            },
        ),
    ]);
    let metrics = LevelMetrics {
        ecs1,
        linear1,
        constant1,
        capture1: capture(ecs1, linear1, constant1),
        horizons,
        probe_s,
        probe_f,
        jacobian,
        linear_radius,
    };
    Ok((json, metrics))
}

/// One planning episode's paired setup (identical for every condition and budget).
fn episode_setup(seed: u64, kind: WorldKind, world: &World, e: usize) -> (Truth, Truth) {
    let mut rng = Rng::new(spec::BASE_SEED, &[seed, kind_tag(kind), 3, e as u64]);
    let s0 = rng.range(-spec::START_RANGE, spec::START_RANGE);
    let sg = loop {
        let g = rng.range(-spec::START_RANGE, spec::START_RANGE);
        if (g - s0).abs() >= spec::MIN_GOAL_DISTANCE {
            break g;
        }
    };
    let std = world.spec.second_std;
    (Truth { s: s0, f: std * rng.normal() }, Truth { s: sg, f: std * rng.normal() })
}

#[derive(Default, Clone)]
struct EpisodeStats {
    success: f64,
    final_error: f64,
    rows: Vec<f64>,
    subgoal_error: f64,
    subgoal_count: f64,
}

fn run_episode(
    cond: &HierarchySpec,
    models: &mut [LevelModel],
    encoders: &[Encoder],
    data: &WorldData,
    seed: u64,
    kind: WorldKind,
    e: usize,
    k: usize,
    cond_tag: u64,
    budget: usize,
) -> Result<EpisodeStats, K1Error> {
    let world = &data.world;
    let (start, goal) = episode_setup(seed, kind, world, e);
    let mut noise = Rng::new(spec::BASE_SEED, &[seed, kind_tag(kind), 4, e as u64]);
    let mut plan_rng = Rng::new(spec::BASE_SEED, &[seed, kind_tag(kind), 5, e as u64, cond_tag, budget as u64]);
    let g1 = encode_obs(&data.enc1, &world.render(goal));
    let mut goals = vec![g1];
    for l in 1..cond.levels.len() {
        let w = cond.levels[l].window;
        goals.push(encoders[l].encode(&vec![goals[l - 1]; w]));
    }
    let mut truth = start;
    let mut ctl = Controller::start(cond, encoders, encode_obs(&data.enc1, &world.observe(truth, &mut noise)));
    let rows0: Vec<u64> = models.iter().map(|m| m.forward_rows).collect();
    let mut stats = EpisodeStats::default();
    let mut steps = 0;
    while steps < spec::PLAN_STEPS {
        let plan = if cond.levels.len() == 1 {
            plan_flat(
                &mut models[0],
                ctl.current[0],
                goals[0],
                k,
                spec::FLAT_HORIZON,
                spec::FLAT_BLOCK,
                spec::REPLAN_EVERY,
                &mut plan_rng,
            )?
        } else {
            plan_hierarchical(cond, models, encoders, &ctl.planning_latents(encoders), &goals, k, &mut plan_rng)?
        };
        for a in plan.actions.iter().take(spec::REPLAN_EVERY) {
            truth = world.step(truth, *a, &mut noise);
            ctl.step(encoders, encode_obs(&data.enc1, &world.observe(truth, &mut noise)));
            steps += 1;
        }
        if let Some(sub) = plan.subgoal {
            stats.subgoal_error +=
                (0..LATENT).map(|i| (ctl.current[1][i] - sub[i]).powi(2)).sum::<f64>() / LATENT as f64;
            stats.subgoal_count += 1.0;
        }
    }
    let err = (truth.s - goal.s).abs();
    stats.success = (err <= spec::SUCCESS_TOLERANCE) as u8 as f64;
    stats.final_error = err;
    stats.rows = models.iter().zip(&rows0).map(|(m, r)| (m.forward_rows - r) as f64).collect();
    Ok(stats)
}

/// The oracle planner: the flat planner over the true slow dynamics (validity: the task is solvable).
pub fn oracle_success(data: &WorldData, seed: u64, kind: WorldKind) -> f64 {
    let world = &data.world;
    let mut wins = 0.0;
    for e in 0..spec::PLAN_EPISODES {
        let (start, goal) = episode_setup(seed, kind, world, e);
        let mut noise = Rng::new(spec::BASE_SEED, &[seed, kind_tag(kind), 4, e as u64]);
        let mut plan_rng = Rng::new(spec::BASE_SEED, &[seed, kind_tag(kind), 6, e as u64]);
        let mut truth = start;
        let _ = world.observe(truth, &mut noise);
        let mut steps = 0;
        while steps < spec::PLAN_STEPS {
            let cands = candidates(spec::ORACLE_K, spec::FLAT_HORIZON, spec::FLAT_BLOCK, &mut plan_rng);
            // The summed squared distance of every predicted slow value to the goal (the planners' path cost).
            let cost = |c: &Vec<f64>| {
                let mut s = truth.s;
                c.iter()
                    .map(|a| {
                        s = (s + world.spec.kappa * a).clamp(-1.0, 1.0);
                        (s - goal.s).powi(2)
                    })
                    .sum::<f64>()
            };
            let best = (0..cands.len())
                .min_by(|&i, &j| cost(&cands[i]).partial_cmp(&cost(&cands[j])).unwrap().then(i.cmp(&j)))
                .unwrap();
            for a in cands[best].iter().take(spec::REPLAN_EVERY) {
                truth = world.step(truth, *a, &mut noise);
                let _ = world.observe(truth, &mut noise);
                steps += 1;
            }
        }
        wins += ((truth.s - goal.s).abs() <= spec::SUCCESS_TOLERANCE) as u8 as f64;
    }
    wins / spec::PLAN_EPISODES as f64
}

/// Results of one condition in one world instance (for aggregation).
#[derive(Clone, Default)]
pub struct CondResult {
    pub name: String,
    pub levels: usize,
    pub keys: Vec<String>,
    pub consumed: Vec<usize>,
    pub probe_s: Vec<f64>,
    pub probe_f: Vec<f64>,
    pub success: [f64; 3],
    pub rows_per_replan: [f64; 3],
}

pub struct SeedResult {
    pub json: Json,
    pub seed: u64,
    pub t1: Diagnostics,
    pub oracle: f64,
    pub l1_probe: (f64, f64),
    pub centroid_ratio: f64,
    pub refused: bool,
    pub keys: Vec<String>,
    pub refs: HashMap<String, Vec<RefStats>>,
    pub levels: HashMap<String, LevelMetrics>,
    pub conds: Vec<CondResult>,
}

pub fn run_seed(k1: &K1, kind: WorldKind, seed: u64, steps: u64) -> Result<SeedResult, K1Error> {
    log(&format!("world {} seed {seed}: data and references", kind.name()));
    let data = world_data(kind, seed);
    let refs = references(&data, seed, kind);
    let mut pool = build_pool(k1, &data, steps, seed, kind)?;
    let refused = pool.iter().any(|e| e.trace.refused.is_some());
    let mut level_json = Vec::new();
    let mut levels = HashMap::new();
    for i in 0..pool.len() {
        if pool[i].trace.refused.is_some() {
            level_json.push(Json::obj(vec![
                ("key", Json::str(&pool[i].key)),
                ("refused", Json::str(pool[i].trace.refused.clone().unwrap())),
            ]));
            continue;
        }
        log(&format!("world {} seed {seed}: measure {}", kind.name(), pool[i].key));
        let (j, m) = level_report(k1, &mut pool[i], &data, seed, kind)?;
        levels.insert(pool[i].key.clone(), m);
        level_json.push(j);
    }
    let l1 = find(&pool, "L1/N36");
    let t1 = diagnostics(&mut pool[l1], &data)?;
    let mut conds = Vec::new();
    let mut cond_json = Vec::new();
    for (ci, cond) in spec::conditions().iter().enumerate() {
        cond.validate().expect("valid spec");
        let keys = level_keys(cond);
        let consumed: Vec<usize> = (0..cond.levels.len()).map(|l| spec::consumed_horizon(cond, l)).collect();
        let mut cr = CondResult {
            name: cond.name.clone(),
            levels: cond.levels.len(),
            keys: keys.clone(),
            consumed: consumed.clone(),
            ..Default::default()
        };
        for k in &keys {
            let m = levels.get(k);
            cr.probe_s.push(m.map_or(f64::NAN, |m: &LevelMetrics| m.probe_s));
            cr.probe_f.push(m.map_or(f64::NAN, |m: &LevelMetrics| m.probe_f));
        }
        let consumed_json: Vec<Json> = keys
            .iter()
            .zip(&consumed)
            .map(|(k, &h)| match levels.get(k) {
                Some(m) => {
                    let s = m.at(h);
                    Json::obj(vec![
                        ("key", Json::str(k)),
                        ("h", Json::Int(h as i64)),
                        ("capture", Json::Num(s.capture)),
                        ("ecs_escaped", Json::Num(s.ecs_escaped)),
                        ("ecs_free_running", Json::Num(s.ecs_free)),
                        ("ecs_teacher_forced", Json::Num(s.ecs_teacher)),
                        ("linear", Json::Num(s.linear)),
                        ("constant", Json::Num(s.constant)),
                        ("generator", opt(s.generator)),
                        ("analytic_floor", opt(s.floor)),
                    ])
                }
                None => {
                    Json::obj(vec![("key", Json::str(k)), ("h", Json::Int(h as i64)), ("refused", Json::Bool(true))])
                }
            })
            .collect();
        let idx: Vec<usize> = keys.iter().map(|k| find(&pool, k)).collect();
        let any_refused = idx.iter().any(|&i| pool[i].trace.refused.is_some());
        let mut budgets_json = Vec::new();
        if !any_refused {
            let encoders: Vec<Encoder> = idx.iter().map(|&i| pool[i].encoder.clone()).collect();
            let mut models: Vec<LevelModel> = idx.iter().map(|&i| pool[i].model.take().unwrap()).collect();
            log(&format!("world {} seed {seed}: plan {}", kind.name(), cond.name));
            for (bi, &k) in spec::budgets(cond).iter().enumerate() {
                let mut acc = EpisodeStats { rows: vec![0.0; models.len()], ..Default::default() };
                for e in 0..spec::PLAN_EPISODES {
                    let s = run_episode(cond, &mut models, &encoders, &data, seed, kind, e, k, ci as u64, bi)?;
                    acc.success += s.success;
                    acc.final_error += s.final_error;
                    acc.subgoal_error += s.subgoal_error;
                    acc.subgoal_count += s.subgoal_count;
                    for (a, r) in acc.rows.iter_mut().zip(&s.rows) {
                        *a += r;
                    }
                }
                let n = spec::PLAN_EPISODES as f64;
                let replans = (spec::PLAN_STEPS / spec::REPLAN_EVERY) as f64;
                let rows: Vec<f64> = acc.rows.iter().map(|r| r / n).collect();
                let total_rows: f64 = rows.iter().sum();
                let madds: f64 = rows.iter().zip(&cond.levels).map(|(r, l)| r * (l.width * DIM) as f64).sum();
                cr.success[bi] = acc.success / n;
                cr.rows_per_replan[bi] = total_rows / replans;
                budgets_json.push(Json::obj(vec![
                    ("planner_k", Json::Int(k as i64)),
                    ("success", Json::Num(acc.success / n)),
                    ("mean_final_error", Json::Num(acc.final_error / n)),
                    ("forward_rows_per_episode_by_level", Json::nums(&rows)),
                    ("forward_rows_per_replan", Json::Num(total_rows / replans)),
                    ("forward_multiply_adds_per_episode", Json::Num(madds)),
                    (
                        "subgoal_tracking_nmse",
                        if acc.subgoal_count > 0.0 {
                            Json::Num(acc.subgoal_error / acc.subgoal_count)
                        } else {
                            Json::Null
                        },
                    ),
                ]));
            }
            for (j, &i) in idx.iter().enumerate().rev() {
                pool[i].model = Some(models.remove(j));
            }
        }
        let params_w: usize = cond.levels.iter().map(|l| l.width * DIM).sum();
        let params_enc: usize = idx.iter().skip(1).map(|&i| pool[i].encoder.parameters()).sum();
        let ctl_bytes =
            Controller::start(cond, &idx.iter().map(|&i| pool[i].encoder.clone()).collect::<Vec<_>>(), [0.0; LATENT])
                .resident_bytes();
        let image: usize = cond.levels.iter().map(|l| k1.image_bytes(DIM, l.width)).sum();
        let train_madds: u64 = idx.iter().map(|&i| pool[i].trace.multiply_adds).sum();
        cond_json.push(Json::obj(vec![
            ("name", Json::str(&cond.name)),
            ("representation", Json::str(cond.representation.name())),
            ("levels", Json::Arr(keys.iter().map(|k| Json::str(k.clone())).collect())),
            ("consumed_horizons", Json::Arr(consumed_json)),
            (
                "accounting",
                Json::obj(vec![
                    ("ecs_w_parameters", Json::Int(params_w as i64)),
                    ("encoder_parameters", Json::Int(params_enc as i64)),
                    ("total_width", Json::Int(cond.total_width() as i64)),
                    ("k1_image_bytes", Json::Int(image as i64)),
                    ("controller_resident_bytes", Json::Int(ctl_bytes as i64)),
                    ("training_multiply_adds", Json::Int(train_madds as i64)),
                ]),
            ),
            ("probe_nmse_slow_by_level", Json::nums(&cr.probe_s)),
            ("probe_nmse_fast_by_level", Json::nums(&cr.probe_f)),
            (
                "planning",
                if any_refused {
                    Json::str("NOT_RUN: a level's training was refused")
                } else {
                    Json::Arr(budgets_json)
                },
            ),
        ]));
        conds.push(cr);
    }
    let json = Json::obj(vec![
        ("world", Json::str(kind.name())),
        ("seed", Json::str(seed.to_string())),
        ("references", refs.json),
        ("t1", diagnostics_json(&t1)),
        ("levels", Json::Arr(level_json)),
        ("conditions", Json::Arr(cond_json)),
    ]);
    Ok(SeedResult {
        json,
        seed,
        t1,
        oracle: refs.oracle,
        l1_probe: refs.l1_probe,
        centroid_ratio: refs.centroid_ratio,
        refused,
        keys: refs.keys,
        refs: refs.refs,
        levels,
        conds,
    })
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// The median of the finite values (NaN if there is none).
fn median(v: &[f64]) -> f64 {
    let mut f: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if f.is_empty() {
        return f64::NAN;
    }
    f.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = f.len();
    if n % 2 == 1 {
        f[n / 2]
    } else {
        0.5 * (f[n / 2 - 1] + f[n / 2])
    }
}

fn cond<'a>(r: &'a SeedResult, name: &str) -> &'a CondResult {
    r.conds.iter().find(|c| c.name == name).unwrap()
}

/// Task validity of one instance: (T1, T2, T3A, T3B).
pub fn instance_validity(r: &SeedResult) -> (bool, bool, bool, bool) {
    let t1 = t1_holds(&r.t1);
    let t2 = !r.refused && r.keys.iter().all(|k| r.levels.get(k).is_some_and(|m| m.capture1 >= gates::T2_MIN_CAPTURE));
    let (mut t3a, mut t3b) = (true, true);
    for (_, key, h) in consumed_pairs() {
        let (a, b) = t3_holds(ref_at(&r.refs[&key], h), observation_level(&key));
        t3a &= a;
        t3b &= b;
    }
    (t1, t2, t3a, t3b)
}

/// M for one condition over one world's seeds: (M-valid, evidence).
pub fn m_condition(rs: &[SeedResult], c: &HierarchySpec) -> (bool, Json) {
    use gates::*;
    let mut seed_ok = vec![true; rs.len()];
    let (mut means_ok, mut levels) = (true, Vec::new());
    for (l, key) in level_keys(c).iter().enumerate() {
        let h = spec::consumed_horizon(c, l);
        let (caps, escs): (Vec<f64>, Vec<f64>) = rs
            .iter()
            .map(|r| r.levels.get(key).map_or((f64::NAN, 1.0), |m| (m.at(h).capture, m.at(h).ecs_escaped)))
            .unzip();
        let (mc, me) = (mean(&caps), mean(&escs));
        let ok = mc >= M_MIN_CAPTURE && me <= M_MAX_ESCAPED;
        means_ok &= ok;
        for (i, (c, e)) in caps.iter().zip(&escs).enumerate() {
            seed_ok[i] &= *c >= M_MIN_CAPTURE && *e <= M_MAX_ESCAPED;
        }
        levels.push(Json::obj(vec![
            ("key", Json::str(key)),
            ("h", Json::Int(h as i64)),
            ("capture_by_seed", Json::nums(&caps)),
            ("escaped_by_seed", Json::nums(&escs)),
            ("capture_mean", Json::Num(mc)),
            ("escaped_mean", Json::Num(me)),
            ("means_meet_M", Json::Bool(ok)),
        ]));
    }
    let frac = seed_ok.iter().filter(|o| **o).count() as f64 / rs.len() as f64;
    let valid = means_ok && frac >= M_SEED_FRACTION;
    (
        valid,
        Json::obj(vec![
            ("condition", Json::str(&c.name)),
            ("levels", Json::Arr(levels)),
            ("seed_meets_M_at_every_level", Json::Arr(seed_ok.iter().map(|b| Json::Bool(*b)).collect())),
            ("seed_fraction", Json::Num(frac)),
            ("M", Json::Bool(valid)),
        ]),
    )
}

/// The multi-step diagnosis of one world: per pool level and horizon, medians over the seeds (finite values)
/// of the free-running, teacher-forced and reference errors, capture, escape and amplification; per level the
/// one-step Jacobian. Descriptive only: no gate reads it.
fn diagnosis(rs: &[SeedResult]) -> Json {
    let keys = rs.first().map_or(Vec::new(), |r| r.keys.clone());
    Json::Arr(
        keys.iter()
            .map(|key| {
                let ms: Vec<&LevelMetrics> = rs.iter().filter_map(|r| r.levels.get(key)).collect();
                let by_h: Vec<Json> = spec::HORIZONS
                    .iter()
                    .map(|&h| {
                        let med =
                            |f: fn(&HorizonStats) -> f64| median(&ms.iter().map(|m| f(m.at(h))).collect::<Vec<_>>());
                        Json::obj(vec![
                            ("h", Json::Int(h as i64)),
                            ("ecs_free_running", Json::Num(med(|s| s.ecs_free))),
                            ("ecs_teacher_forced", Json::Num(med(|s| s.ecs_teacher))),
                            ("linear", Json::Num(med(|s| s.linear))),
                            ("generator", Json::Num(med(|s| s.generator.unwrap_or(f64::NAN)))),
                            ("analytic_floor", Json::Num(med(|s| s.floor.unwrap_or(f64::NAN)))),
                            ("constant", Json::Num(med(|s| s.constant))),
                            ("capture", Json::Num(med(|s| s.capture))),
                            (
                                "escaped_mean",
                                Json::Num(mean(&ms.iter().map(|m| m.at(h).ecs_escaped).collect::<Vec<_>>())),
                            ),
                            (
                                "passive_escaped_mean",
                                Json::Num(mean(&ms.iter().map(|m| m.at(h).passive_escaped).collect::<Vec<_>>())),
                            ),
                            ("norm_ratio", Json::Num(med(|s| s.norm_ratio))),
                            ("amplification_median", Json::Num(med(|s| s.amp_median))),
                            ("linear_amplification_median", Json::Num(med(|s| s.linear_amp_median))),
                            ("amplification_over_10", Json::Num(med(|s| s.amp_over_10))),
                        ])
                    })
                    .collect();
                Json::obj(vec![
                    ("key", Json::str(key)),
                    ("seeds_measured", Json::Int(ms.len() as i64)),
                    ("one_step_capture", Json::Num(median(&ms.iter().map(|m| m.capture1).collect::<Vec<_>>()))),
                    ("jacobian_radius_mean", Json::Num(median(&ms.iter().map(|m| m.jacobian.0).collect::<Vec<_>>()))),
                    (
                        "jacobian_fraction_over_1",
                        Json::Num(median(&ms.iter().map(|m| m.jacobian.2).collect::<Vec<_>>())),
                    ),
                    ("linear_radius", Json::Num(median(&ms.iter().map(|m| m.linear_radius).collect::<Vec<_>>()))),
                    ("by_horizon", Json::Arr(by_h)),
                ])
            })
            .collect(),
    )
}

/// Gates and disposition over one phase's seeds (both worlds). `binding` is true for QUAL only: DEV applies
/// task validity (TASK_INVALID_ON_DEV or QUAL_AUTHORIZED) and reports M and H2 as non-binding values.
pub fn evaluate(sep: &[SeedResult], mat: &[SeedResult], binding: bool) -> Json {
    use gates::*;
    let both: Vec<&SeedResult> = sep.iter().chain(mat).collect();
    let per: Vec<(bool, bool, bool, bool)> = both.iter().map(|r| instance_validity(r)).collect();
    let t1 = per.iter().all(|v| v.0);
    let t2 = per.iter().all(|v| v.1);
    let t3a = per.iter().all(|v| v.2);
    let t3b = per.iter().all(|v| v.3);
    let oracle = |rs: &[SeedResult]| mean(&rs.iter().map(|r| r.oracle).collect::<Vec<_>>());
    let ratio = |rs: &[SeedResult]| mean(&rs.iter().map(|r| r.centroid_ratio).collect::<Vec<_>>());
    let v2 = oracle(sep) >= V2_ORACLE_SUCCESS && oracle(mat) >= V2_ORACLE_SUCCESS;
    let v3 = both.iter().all(|r| r.l1_probe.0 <= V3_L1_PROBE_NMSE && r.l1_probe.1 <= V3_L1_PROBE_NMSE);
    let v4 = ratio(sep) >= V4_SEPARATED_CENTROID_RATIO && ratio(mat) <= V4_MATCHED_CENTROID_RATIO;
    let v5 = both.iter().all(|r| !r.refused);
    let valid = t1 && t2 && t3a && t3b && v2 && v3 && v4 && v5;

    // M: every condition in each world; the H2 conditions in SEPARATED decide whether H2 is adjudicated.
    let conds = spec::conditions();
    let m_world = |rs: &[SeedResult]| -> (Vec<(String, bool)>, Json) {
        let (flags, json): (Vec<(String, bool)>, Vec<Json>) = conds
            .iter()
            .map(|c| {
                let (ok, j) = m_condition(rs, c);
                ((c.name.clone(), ok), j)
            })
            .unzip();
        (flags, Json::Arr(json))
    };
    let (m_sep, m_sep_json) = m_world(sep);
    let (m_mat, m_mat_json) = m_world(mat);
    let m_of = |flags: &[(String, bool)], name: &str| flags.iter().find(|(n, _)| n == name).unwrap().1;
    let m_h2 = spec::H2_CONDITIONS.iter().all(|n| m_of(&m_sep, n));

    let succ = |name: &str, b: usize| mean(&sep.iter().map(|r| cond(r, name).success[b]).collect::<Vec<_>>());
    let diff = |a: &str, b: &str| -> Vec<f64> { (0..3).map(|k| succ(a, k) - succ(b, k)).collect() };
    let h2a_by = diff("HIERARCHICAL_DISTINCT_L2", "FLAT_N72");
    let h2b_by = diff("HIERARCHICAL_DISTINCT_L2", "TEMPORAL_SHARED_L2");
    let h2c_by = diff("TEMPORAL_SHARED_L2", "FLAT_N72");
    let holds = |v: &[f64], t: f64| v.iter().filter(|d| **d >= t).count() >= H2_BUDGETS_REQUIRED;
    let (h2a, h2b, h2c) = (
        holds(&h2a_by, H2A_DISTINCT_OVER_WIDTH),
        holds(&h2b_by, H2B_DISTINCT_OVER_TEMPORAL),
        holds(&h2c_by, H2C_TEMPORAL_OVER_WIDTH),
    );
    let class = |d: f64| {
        if d >= DEPTH_EFFECT {
            "IMPROVES"
        } else if d <= -DEPTH_EFFECT {
            "DEGRADES"
        } else {
            "NO_CHANGE"
        }
    };
    let depth = |names: [&str; 3]| -> Json {
        let s: Vec<f64> = names.iter().map(|n| succ(n, 1)).collect();
        Json::obj(vec![
            ("success_L1_L2_L3_middle_budget", Json::nums(&s)),
            ("L2_vs_L1", Json::str(class(s[1] - s[0]))),
            ("L3_vs_L2", Json::str(class(s[2] - s[1]))),
            ("M_separated_L2_L3", Json::Arr(names[1..].iter().map(|n| Json::Bool(m_of(&m_sep, n))).collect())),
        ])
    };
    let width: Vec<f64> = ["FLAT_N36", "FLAT_N72", "FLAT_N108"].iter().map(|n| succ(n, 1)).collect();

    let adjudicated = binding && valid && m_h2;
    let disposition = match (binding, valid, m_h2) {
        (false, false, _) => "TASK_INVALID_ON_DEV",
        (false, true, _) => "QUAL_AUTHORIZED",
        (true, false, _) => "TASK_INVALID",
        (true, true, false) => "WORLD_MODEL_INVALID",
        (true, true, true) => "HIERARCHY_ADJUDICATED",
    };
    let outcome = if !adjudicated {
        "NOT_ADJUDICATED"
    } else if h2a && h2b {
        "DISTINCT_ADVANTAGE"
    } else if h2c {
        "TEMPORAL_ADVANTAGE"
    } else if h2a {
        "UNATTRIBUTED_ADVANTAGE"
    } else {
        "NO_ADVANTAGE_OVER_WIDTH"
    };
    let instances: Vec<Json> = both
        .iter()
        .zip(&per)
        .zip(sep.iter().map(|_| "SEPARATED").chain(mat.iter().map(|_| "MATCHED")))
        .map(|((r, v), w)| {
            Json::obj(vec![
                ("world", Json::str(w)),
                ("seed", Json::str(r.seed.to_string())),
                ("t1", diagnostics_json(&r.t1)),
                ("T1", Json::Bool(v.0)),
                (
                    "one_step_capture_by_level",
                    Json::Arr(
                        r.keys
                            .iter()
                            .map(|k| {
                                Json::obj(vec![
                                    ("key", Json::str(k)),
                                    ("capture", Json::Num(r.levels.get(k).map_or(f64::NEG_INFINITY, |m| m.capture1))),
                                ])
                            })
                            .collect(),
                    ),
                ),
                ("T2", Json::Bool(v.1)),
                ("T3A", Json::Bool(v.2)),
                ("T3B", Json::Bool(v.3)),
                ("oracle_success", Json::Num(r.oracle)),
                ("l1_probe_nmse_slow_fast", Json::nums(&[r.l1_probe.0, r.l1_probe.1])),
                ("centroid_ratio", Json::Num(r.centroid_ratio)),
                ("refused", Json::Bool(r.refused)),
            ])
        })
        .collect();
    Json::obj(vec![
        ("binding", Json::Bool(binding)),
        (
            "task_validity",
            Json::obj(vec![
                ("instances", Json::Arr(instances)),
                ("T1", Json::Bool(t1)),
                ("T2", Json::Bool(t2)),
                ("T3A", Json::Bool(t3a)),
                ("T3B", Json::Bool(t3b)),
                ("V2_oracle_success", Json::nums(&[oracle(sep), oracle(mat)])),
                ("V2", Json::Bool(v2)),
                ("V3", Json::Bool(v3)),
                ("V4_centroid_ratio", Json::nums(&[ratio(sep), ratio(mat)])),
                ("V4", Json::Bool(v4)),
                ("V5_no_refusal", Json::Bool(v5)),
                ("valid", Json::Bool(valid)),
            ]),
        ),
        (
            "multistep",
            Json::obj(vec![
                ("SEPARATED", m_sep_json),
                ("MATCHED", m_mat_json),
                ("M_separated_h2_conditions", Json::Bool(m_h2)),
            ]),
        ),
        ("diagnosis", Json::obj(vec![("SEPARATED", diagnosis(sep)), ("MATCHED", diagnosis(mat))])),
        (
            "planning",
            Json::obj(vec![
                ("adjudicated", Json::Bool(adjudicated)),
                ("hd_l2_minus_flat_n72_by_budget", Json::nums(&h2a_by)),
                ("hd_l2_minus_ts_l2_by_budget", Json::nums(&h2b_by)),
                ("ts_l2_minus_flat_n72_by_budget", Json::nums(&h2c_by)),
                ("H2A", Json::Bool(h2a)),
                ("H2B", Json::Bool(h2b)),
                ("H2C", Json::Bool(h2c)),
                ("width_success_n36_n72_n108_middle_budget", Json::nums(&width)),
                (
                    "width_M_separated_n36_n72_n108",
                    Json::Arr(
                        ["FLAT_N36", "FLAT_N72", "FLAT_N108"].iter().map(|n| Json::Bool(m_of(&m_sep, n))).collect(),
                    ),
                ),
                ("depth_temporal_shared", depth(["FLAT_N36", "TEMPORAL_SHARED_L2", "TEMPORAL_SHARED_L3"])),
                (
                    "depth_hierarchical_distinct",
                    depth(["FLAT_N36", "HIERARCHICAL_DISTINCT_L2", "HIERARCHICAL_DISTINCT_L3"]),
                ),
                (
                    "M_matched_by_condition",
                    Json::Arr(
                        m_mat
                            .iter()
                            .map(|(n, ok)| Json::obj(vec![("condition", Json::str(n)), ("M", Json::Bool(*ok))]))
                            .collect(),
                    ),
                ),
            ]),
        ),
        ("disposition", Json::str(disposition)),
        ("hierarchy_outcome", Json::str(outcome)),
        ("integration_authorized", Json::Bool(adjudicated && h2a && h2b)),
    ])
}
