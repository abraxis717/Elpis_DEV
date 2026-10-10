//! H-ECS R3 successor C2M planning module: the R2 H2 planning comparison (FLAT_N72, TEMPORAL_SHARED_L2,
//! HIERARCHICAL_DISTINCT_L2) over SUPPLIED level weights. QUERY-only computation: native K1 forward rollouts of fixed
//! W; no training. kind_tag, episode_setup, EpisodeStats and run_episode are verbatim from hecs_r2 experiment.rs;
//! plan_conditions mirrors the per-condition planning loop and accounting of hecs_r2 experiment::run_seed.
use hecs_r2::encoder::{encode_obs, Encoder};
use hecs_r2::experiment::{level_keys, WorldData};
use hecs_r2::hierarchy::{Controller, HierarchySpec};
use hecs_r2::k1::{K1Error, K1};
use hecs_r2::model::{LevelModel, DIM, LATENT};
use hecs_r2::planner::{plan_flat, plan_hierarchical};
use hecs_r2::rng::Rng;
use hecs_r2::spec;
use hecs_r2::world::{Truth, World, WorldKind};

const KIND_TAG: [u64; 2] = [11, 13];
fn kind_tag(k: WorldKind) -> u64 {
    match k {
        WorldKind::Separated => KIND_TAG[0],
        WorldKind::Matched => KIND_TAG[1],
    }
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

pub struct PlanRow {
    pub condition: String,
    pub budget_index: usize,
    pub planner_k: usize,
    pub success: f64,
    pub mean_final_error: f64,
    pub forward_rows_per_replan: f64,
    pub subgoal_tracking_nmse: f64,
}

/// Plans every H2 condition (in hecs_r2 spec::conditions() order; cond_tag = its index there) with the supplied
/// (key, W, width, encoder) levels. Mirrors hecs_r2 experiment::run_seed's planning loop and accounting.
pub fn plan_conditions(k1: &K1, data: &WorldData, seed: u64, kind: WorldKind,
    levels: &[(String, Vec<f64>, usize, Encoder)]) -> Result<Vec<PlanRow>, K1Error> {
    let mut out = Vec::new();
    for (ci, cond) in spec::conditions().iter().enumerate() {
        if !spec::H2_CONDITIONS.contains(&cond.name.as_str()) {
            continue;
        }
        cond.validate().expect("valid spec");
        let keys = level_keys(cond);
        let mut encoders: Vec<Encoder> = Vec::new();
        let mut models: Vec<LevelModel> = Vec::new();
        for k in &keys {
            let (_, w, width, enc) = levels.iter().find(|(key, ..)| key == k).expect("PLANNING_LEVEL_MISSING");
            encoders.push(enc.clone());
            models.push(LevelModel { ecs: k1.create(DIM, *width, spec::MAX_ROWS, w)?, width: *width, forward_rows: 0 });
        }
        for (bi, &k) in spec::budgets(cond).iter().enumerate() {
            let mut acc = EpisodeStats { rows: vec![0.0; models.len()], ..Default::default() };
            for e in 0..spec::PLAN_EPISODES {
                let s = run_episode(cond, &mut models, &encoders, data, seed, kind, e, k, ci as u64, bi)?;
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
            out.push(PlanRow {
                condition: cond.name.clone(),
                budget_index: bi,
                planner_k: k,
                success: acc.success / n,
                mean_final_error: acc.final_error / n,
                forward_rows_per_replan: total_rows / replans,
                subgoal_tracking_nmse: if acc.subgoal_count > 0.0 { acc.subgoal_error / acc.subgoal_count } else { f64::NAN },
            });
        }
    }
    Ok(out)
}
