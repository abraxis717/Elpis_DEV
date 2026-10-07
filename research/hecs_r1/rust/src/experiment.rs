//! H-ECS R1 experiment driver: worlds, level models, measurements, paired planning, gates and disposition.
//!
//! Copied from H-ECS R0 (research/hecs_r0/rust/src/experiment.rs) with exactly these changes
//! (research/hecs_r1/SOURCE_EQUIVALENCE.json): the R1 validity diagnostics (`validity`) replace R0's
//! absolute V1; every level reports its own linear reference and baselines (adequacy, not gated); a
//! level-1-only path (`level1_validity`) serves DESIGN and DEV calibration; `oracle_success` is public (DESIGN
//! checks V2) and scores the path like the planners (planner.rs); `evaluate` applies R1's law.

use crate::analysis::{attractors, horizon_error, probe_nmse};
use crate::encoder::{encode_obs, Encoder};
use crate::hierarchy::{lift, window_triples, Controller, HierarchySpec, LevelSequence, Representation};
use crate::json::Json;
use crate::k1::{K1Error, K1};
use crate::model::{LevelModel, TrainTrace, DIM, LATENT};
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

/// Train every distinct level used by the conditions (FLAT widths, and each hierarchy path).
pub fn build_pool<'a>(
    k1: &'a K1,
    data: &WorldData,
    steps: u64,
    seed: u64,
    kind: WorldKind,
) -> Result<Vec<PoolEntry<'a>>, K1Error> {
    let l1_train: Vec<_> = data.train.iter().map(|t| level1(&data.enc1, t)).collect();
    let l1_test: Vec<_> = data.test.iter().map(|t| level1(&data.enc1, t)).collect();
    let mut pool = Vec::new();
    for width in [36usize, 72, 108] {
        log(&format!("  train L1 width {width}"));
        pool.push(train_entry(
            k1,
            &format!("L1/N{width}"),
            data.enc1.clone(),
            l1_train.clone(),
            l1_test.clone(),
            width,
            1,
            steps,
            seed,
            kind,
        )?);
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
            log(&format!("  train {tag} L{}", l + 1));
            pool.push(train_entry(
                k1,
                &format!("{tag}/L{}", l + 1),
                encoder,
                up_train.clone(),
                up_test.clone(),
                ls.width,
                cond.env_stride(l),
                steps,
                seed,
                kind,
            )?);
            lower_train = up_train;
            lower_test = up_test;
        }
    }
    Ok(pool)
}

/// The R1 validity diagnostics of one world instance from its FLAT_N36 level-1 model (the same model, seeded
/// identically, that `build_pool` trains as `L1/N36`).
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

/// Level 1 only (DESIGN and DEV calibration): train FLAT_N36's level-1 model at `steps` and return its
/// validity diagnostics, the level-1 probes and whether training was refused. No hierarchy, no planning.
pub fn level1_validity(
    k1: &K1,
    data: &WorldData,
    steps: u64,
    seed: u64,
    kind: WorldKind,
) -> Result<(Diagnostics, (f64, f64), bool), K1Error> {
    let l1_train: Vec<_> = data.train.iter().map(|t| level1(&data.enc1, t)).collect();
    let l1_test: Vec<_> = data.test.iter().map(|t| level1(&data.enc1, t)).collect();
    let mut entry = train_entry(k1, "L1/N36", data.enc1.clone(), l1_train, l1_test, 36, 1, steps, seed, kind)?;
    let d = diagnostics(&mut entry, data)?;
    Ok((d, probes(&entry, data), entry.trace.refused.is_some()))
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
        ("ecs_excess_over_linear", Json::Num(d.ecs - d.linear)),
        ("linear_minus_floor", Json::Num(d.linear - d.floor)),
        ("A_reference_predictable", Json::Bool(d.reference_predictable(V1A_MIN_EXPLAINED))),
        ("B_reference_at_floor", Json::Bool(d.reference_at_floor(V1B_FLOOR_TOL_ABS, V1B_FLOOR_TOL_REL))),
        ("ecs_capture", Json::Num(d.capture())),
        ("C_ecs_adequate", Json::Bool(d.ecs_adequate(V1C_MIN_CAPTURE))),
    ])
}

/// The R1 V1 law on one instance: A and B and C.
pub fn v1_holds(d: &Diagnostics) -> bool {
    use gates::*;
    d.reference_predictable(V1A_MIN_EXPLAINED)
        && d.reference_at_floor(V1B_FLOOR_TOL_ABS, V1B_FLOOR_TOL_REL)
        && d.ecs_adequate(V1C_MIN_CAPTURE)
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

/// Per-level measurements of one pool entry.
fn level_report(k1: &K1, entry: &mut PoolEntry, data: &WorldData) -> Result<(Json, f64, f64, f64), K1Error> {
    let (probe_s, probe_f) = probes(entry, data);
    let model = entry.model.as_mut().unwrap();
    let seqs: Vec<_> = entry.test.iter().map(|s| (s.z.clone(), s.b.clone())).collect();
    let (mut hz, mut dv) = (Vec::new(), Vec::new());
    for h in spec::HORIZONS {
        let (e, d) = horizon_error(model, &seqs, h)?;
        hz.push(e);
        dv.push(d);
    }
    // The ECS one-step value: the NMSE, or infinity if any one-step prediction diverged.
    let one_step = if dv[0] > 0.0 { f64::INFINITY } else { hz[0] };
    let linear = linear_reference(&entry.train, &entry.test);
    let constant = constant_baseline(&entry.train, &entry.test);
    let persistence = persistence_baseline(&entry.test);
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
        ("prediction_nmse_by_horizon", Json::nums(&hz)),
        ("prediction_diverged_fraction_by_horizon", Json::nums(&dv)),
        (
            "one_step_adequacy",
            Json::obj(vec![
                ("ecs", Json::Num(one_step)),
                ("linear_reference", Json::Num(linear)),
                ("constant_baseline", Json::Num(constant)),
                ("persistence_baseline", Json::Num(persistence)),
                ("ecs_excess_over_linear", Json::Num(one_step - linear)),
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
    Ok((json, one_step, probe_s, probe_f))
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
    pub probe_s: Vec<f64>,
    pub probe_f: Vec<f64>,
    pub success: [f64; 3],
    pub rows_per_replan: [f64; 3],
}

pub struct SeedResult {
    pub json: Json,
    pub validity: Diagnostics,
    pub oracle: f64,
    pub l1_probe: (f64, f64),
    pub centroid_ratio: f64,
    pub refused: bool,
    pub conds: Vec<CondResult>,
}

pub fn run_seed(k1: &K1, kind: WorldKind, seed: u64, steps: u64) -> Result<SeedResult, K1Error> {
    log(&format!("world {} seed {seed}: data", kind.name()));
    let data = world_data(kind, seed);
    let series = |f: fn(&Truth) -> f64| -> f64 {
        data.train.iter().map(|t| spectral_centroid(&t.truth.iter().map(f).collect::<Vec<_>>())).sum::<f64>()
            / data.train.len() as f64
    };
    let (cs, cf) = (series(|t| t.s), series(|t| t.f));
    let mut pool = build_pool(k1, &data, steps, seed, kind)?;
    let refused = pool.iter().any(|e| e.trace.refused.is_some());
    let mut level_json = Vec::new();
    let mut level_metrics = std::collections::HashMap::new();
    for i in 0..pool.len() {
        if pool[i].trace.refused.is_some() {
            level_json.push(Json::obj(vec![
                ("key", Json::str(&pool[i].key)),
                ("refused", Json::str(pool[i].trace.refused.clone().unwrap())),
            ]));
            continue;
        }
        let (j, one_step, ps, pf) = level_report(k1, &mut pool[i], &data)?;
        level_metrics.insert(pool[i].key.clone(), (one_step, ps, pf));
        level_json.push(j);
    }
    let l1 = find(&pool, "L1/N36");
    let validity = diagnostics(&mut pool[l1], &data)?;
    let l1_probe = level_metrics.get("L1/N36").map_or((f64::NAN, f64::NAN), |m| (m.1, m.2));
    log(&format!("world {} seed {seed}: oracle", kind.name()));
    let oracle = oracle_success(&data, seed, kind);
    let mut conds = Vec::new();
    let mut cond_json = Vec::new();
    for (ci, cond) in spec::conditions().iter().enumerate() {
        cond.validate().expect("valid spec");
        let keys = level_keys(cond);
        let mut cr = CondResult { name: cond.name.clone(), levels: cond.levels.len(), ..Default::default() };
        for k in &keys {
            let m = level_metrics.get(k).copied().unwrap_or((f64::NAN, f64::NAN, f64::NAN));
            cr.probe_s.push(m.1);
            cr.probe_f.push(m.2);
        }
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
        ("seed", Json::Int(seed as i64)),
        ("mixing_angle", Json::Num(data.world.angle)),
        ("spectral_centroid_slow", Json::Num(cs)),
        ("spectral_centroid_second", Json::Num(cf)),
        ("centroid_ratio", Json::Num(cf / cs)),
        ("oracle_success", Json::Num(oracle)),
        ("validity", diagnostics_json(&validity)),
        ("levels", Json::Arr(level_json)),
        ("conditions", Json::Arr(cond_json)),
    ]);
    Ok(SeedResult { json, validity, oracle, l1_probe, centroid_ratio: cf / cs, refused, conds })
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn cond<'a>(r: &'a SeedResult, name: &str) -> &'a CondResult {
    r.conds.iter().find(|c| c.name == name).unwrap()
}

/// Gates and disposition over one phase's seeds (both worlds).
pub fn evaluate(sep: &[SeedResult], mat: &[SeedResult]) -> Json {
    use gates::*;
    let both: Vec<&SeedResult> = sep.iter().chain(mat).collect();
    let v1 = both.iter().all(|r| v1_holds(&r.validity));
    let v2 = mean(&sep.iter().map(|r| r.oracle).collect::<Vec<_>>()) >= V2_ORACLE_SUCCESS
        && mean(&mat.iter().map(|r| r.oracle).collect::<Vec<_>>()) >= V2_ORACLE_SUCCESS;
    let v3 = both.iter().all(|r| r.l1_probe.0 <= V3_L1_PROBE_NMSE && r.l1_probe.1 <= V3_L1_PROBE_NMSE);
    let v4 = mean(&sep.iter().map(|r| r.centroid_ratio).collect::<Vec<_>>()) >= V4_SEPARATED_CENTROID_RATIO
        && mean(&mat.iter().map(|r| r.centroid_ratio).collect::<Vec<_>>()) <= V4_MATCHED_CENTROID_RATIO;
    let v5 = both.iter().all(|r| !r.refused);
    let valid = v1 && v2 && v3 && v4 && v5;

    let loss = |r: &SeedResult, name: &str| {
        let c = cond(r, name);
        c.probe_f[1] - c.probe_f[0]
    };
    let hd_sep: Vec<f64> = sep.iter().map(|r| loss(r, "HIERARCHICAL_DISTINCT_L2")).collect();
    let hd_mat: Vec<f64> = mat.iter().map(|r| loss(r, "HIERARCHICAL_DISTINCT_L2")).collect();
    let ts_sep: Vec<f64> = sep.iter().map(|r| loss(r, "TEMPORAL_SHARED_L2")).collect();
    let slow_l2 = mean(&sep.iter().map(|r| cond(r, "HIERARCHICAL_DISTINCT_L2").probe_s[1]).collect::<Vec<_>>());
    let frac = hd_sep.iter().filter(|d| **d >= H1_FAST_LOSS).count() as f64 / hd_sep.len() as f64;
    let h1 = mean(&hd_sep) >= H1_FAST_LOSS && slow_l2 <= H1_SLOW_RETAINED && frac >= H1_SEED_FRACTION;
    let h1c = mean(&hd_mat) <= mean(&hd_sep) - H1C_CONTROL_MARGIN;
    let h1t = mean(&ts_sep) < H1T_SHARED_MAX_LOSS;

    let succ = |name: &str, b: usize| mean(&sep.iter().map(|r| cond(r, name).success[b]).collect::<Vec<_>>());
    let h2a_by: Vec<f64> = (0..3).map(|b| succ("HIERARCHICAL_DISTINCT_L2", b) - succ("FLAT_N72", b)).collect();
    let h2b_by: Vec<f64> =
        (0..3).map(|b| succ("HIERARCHICAL_DISTINCT_L2", b) - succ("TEMPORAL_SHARED_L2", b)).collect();
    let h2a = h2a_by.iter().filter(|d| **d >= H2A_OVER_WIDTH).count() >= H2_BUDGETS_REQUIRED;
    let h2b = h2b_by.iter().filter(|d| **d >= H2B_OVER_TEMPORAL).count() >= H2_BUDGETS_REQUIRED;
    let h1_all = h1 && h1c && h1t;
    let h2 = h2a && h2b;
    let depth = |names: [&str; 3]| -> Json {
        let s: Vec<f64> = names.iter().map(|n| succ(n, 1)).collect();
        let class = |d: f64| {
            if d >= DEPTH_EFFECT {
                "IMPROVES"
            } else if d <= -DEPTH_EFFECT {
                "DEGRADES"
            } else {
                "NO_CHANGE"
            }
        };
        Json::obj(vec![
            ("success_L1_L2_L3", Json::nums(&s)),
            ("L2_vs_L1", Json::str(class(s[1] - s[0]))),
            ("L3_vs_L2", Json::str(class(s[2] - s[1]))),
        ])
    };
    let width: Vec<f64> = ["FLAT_N36", "FLAT_N72", "FLAT_N108"].iter().map(|n| succ(n, 1)).collect();
    let disposition = if !valid {
        "TASK_INVALID"
    } else {
        match (h1_all, h2) {
            (true, true) => "OUTCOME_A",
            (true, false) => "OUTCOME_B",
            (false, true) => "OUTCOME_C",
            (false, false) => "OUTCOME_D",
        }
    };
    Json::obj(vec![
        (
            "validity",
            Json::obj(vec![
                ("V1_instances", Json::Arr(both.iter().map(|r| diagnostics_json(&r.validity)).collect())),
                ("V1", Json::Bool(v1)),
                (
                    "V2_oracle_success",
                    Json::nums(&[
                        mean(&sep.iter().map(|r| r.oracle).collect::<Vec<_>>()),
                        mean(&mat.iter().map(|r| r.oracle).collect::<Vec<_>>()),
                    ]),
                ),
                ("V2", Json::Bool(v2)),
                ("V3", Json::Bool(v3)),
                (
                    "V4_centroid_ratio",
                    Json::nums(&[
                        mean(&sep.iter().map(|r| r.centroid_ratio).collect::<Vec<_>>()),
                        mean(&mat.iter().map(|r| r.centroid_ratio).collect::<Vec<_>>()),
                    ]),
                ),
                ("V4", Json::Bool(v4)),
                ("V5_no_refusal", Json::Bool(v5)),
                ("valid", Json::Bool(valid)),
            ]),
        ),
        (
            "selective_abstraction",
            Json::obj(vec![
                ("hd_fast_loss_separated", Json::nums(&hd_sep)),
                ("hd_fast_loss_matched", Json::nums(&hd_mat)),
                ("ts_fast_loss_separated", Json::nums(&ts_sep)),
                ("hd_slow_nmse_l2_separated_mean", Json::Num(slow_l2)),
                ("hd_seed_fraction", Json::Num(frac)),
                ("H1", Json::Bool(h1)),
                ("H1C", Json::Bool(h1c)),
                ("H1T", Json::Bool(h1t)),
            ]),
        ),
        (
            "planning",
            Json::obj(vec![
                ("hd_l2_minus_flat_n72_by_budget", Json::nums(&h2a_by)),
                ("hd_l2_minus_ts_l2_by_budget", Json::nums(&h2b_by)),
                ("H2A", Json::Bool(h2a)),
                ("H2B", Json::Bool(h2b)),
                ("width_success_n36_n72_n108_middle_budget", Json::nums(&width)),
                ("depth_temporal_shared", depth(["FLAT_N36", "TEMPORAL_SHARED_L2", "TEMPORAL_SHARED_L3"])),
                (
                    "depth_hierarchical_distinct",
                    depth(["FLAT_N36", "HIERARCHICAL_DISTINCT_L2", "HIERARCHICAL_DISTINCT_L3"]),
                ),
            ]),
        ),
        ("disposition", Json::str(disposition)),
    ])
}
