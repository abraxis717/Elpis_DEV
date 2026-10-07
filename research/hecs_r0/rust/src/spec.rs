//! The preregistered H-ECS R0 specification (research/hecs_r0/PREREGISTRATION.md). Single source of every
//! parameter, condition and gate. `hecs spec` renders it; the committed specs/hecs-r0.v1.spec.json must equal
//! that rendering byte for byte, and every evidence file embeds it.

use crate::hierarchy::{HierarchySpec, LevelSpec, Representation};
use crate::json::Json;
use crate::model::TrainSpec;
use crate::world::{WorldKind, WorldSpec};

pub const NAME: &str = "hecs-r0.v1";
pub const BASE_SEED: u64 = 20261007;
pub const DEV_SEEDS: [u64; 4] = [0, 1, 2, 3];
pub const QUAL_SEEDS: [u64; 8] = [100, 101, 102, 103, 104, 105, 106, 107];

pub const TRAIN_EPISODES: usize = 16;
pub const TEST_EPISODES: usize = 8;
pub const EPISODE_STEPS: usize = 256;
pub const MAX_ROWS: usize = 512;
/// Calibration grid for the per-level training budget (K1 steps). DEV chooses the smallest value whose
/// level-1 FLAT_N36 one-step NMSE meets V1 in every DEV world; only that criterion is consulted.
pub const CALIBRATION_STEPS: [u64; 3] = [6000, 24000, 96000];
pub const HORIZONS: [usize; 4] = [1, 2, 4, 8];

pub const PLAN_EPISODES: usize = 24;
pub const PLAN_STEPS: usize = 64;
pub const REPLAN_EVERY: usize = 4;
pub const FLAT_HORIZON: usize = 16;
pub const FLAT_BLOCK: usize = 4;
pub const SUCCESS_TOLERANCE: f64 = 0.1;
pub const MIN_GOAL_DISTANCE: f64 = 0.8;
pub const START_RANGE: f64 = 0.9;
pub const ORACLE_K: usize = 32;

pub fn train_spec(steps: u64) -> TrainSpec {
    TrainSpec {
        steps,
        steps_per_call: 25,
        batch_transitions: 128,
        base_rate: 0.002,
        reference_width: 36,
        init_scale: 0.18,
    }
}

pub fn world(kind: WorldKind) -> WorldSpec {
    let rho = match kind {
        WorldKind::Separated => -0.7,
        WorldKind::Matched => 0.995,
    };
    WorldSpec { kind, kappa: 0.04, slow_noise: 0.005, rho, second_std: 0.5, obs_noise: 0.01 }
}

pub const WORLDS: [WorldKind; 2] = [WorldKind::Separated, WorldKind::Matched];

fn level(stride: usize, window: usize, width: usize, horizon: usize) -> LevelSpec {
    LevelSpec { stride, window, width, horizon }
}

/// The conditions, in report order. Total ECS width is matched: FLAT_N72 with the two-level hierarchies
/// (2 x 36), FLAT_N108 with the three-level hierarchies (3 x 36).
pub fn conditions() -> Vec<HierarchySpec> {
    let l1_flat = |w| level(1, 1, w, FLAT_HORIZON);
    let l1 = level(1, 1, 36, 4);
    let mk = |name: &str, representation, levels| HierarchySpec { name: name.into(), representation, levels };
    vec![
        mk("FLAT_N36", Representation::Flat, vec![l1_flat(36)]),
        mk("FLAT_N72", Representation::Flat, vec![l1_flat(72)]),
        mk("FLAT_N108", Representation::Flat, vec![l1_flat(108)]),
        mk("TEMPORAL_SHARED_L2", Representation::TemporalShared, vec![l1, level(4, 1, 36, 4)]),
        mk("TEMPORAL_SHARED_L3", Representation::TemporalShared, vec![l1, level(4, 1, 36, 4), level(4, 1, 36, 4)]),
        mk("HIERARCHICAL_DISTINCT_L2", Representation::HierarchicalDistinct, vec![l1, level(4, 2, 36, 4)]),
        mk(
            "HIERARCHICAL_DISTINCT_L3",
            Representation::HierarchicalDistinct,
            vec![l1, level(4, 2, 36, 4), level(4, 2, 36, 4)],
        ),
    ]
}

/// Planner candidate counts per condition at the three compute points. Native forward rows per replan:
/// flat K x 16 x 2 = {256, 1024, 4096}; two levels K x (4 + 4) x 2 (equal); three levels K x (4 + 4 + 4) x 2
/// = {264, 1032, 4104} (within 4 percent: K is an integer). Divergent candidates are not predicted further,
/// so measured rows can be lower; the evidence reports the measured rows.
pub fn budgets(spec: &HierarchySpec) -> [usize; 3] {
    match spec.levels.len() {
        1 => [8, 32, 128],
        2 => [16, 64, 256],
        _ => [11, 43, 171],
    }
}

/// Preregistered gate thresholds.
pub mod gates {
    pub const V1_L1_ONE_STEP_NMSE: f64 = 0.25;
    pub const V2_ORACLE_SUCCESS: f64 = 0.8;
    pub const V3_L1_PROBE_NMSE: f64 = 0.1;
    pub const V4_SEPARATED_CENTROID_RATIO: f64 = 4.0;
    pub const V4_MATCHED_CENTROID_RATIO: f64 = 1.5;
    pub const H1_FAST_LOSS: f64 = 0.3;
    pub const H1_SLOW_RETAINED: f64 = 0.2;
    pub const H1_SEED_FRACTION: f64 = 0.75;
    pub const H1C_CONTROL_MARGIN: f64 = 0.2;
    pub const H1T_SHARED_MAX_LOSS: f64 = 0.1;
    pub const H2A_OVER_WIDTH: f64 = 0.15;
    pub const H2B_OVER_TEMPORAL: f64 = 0.10;
    pub const H2_BUDGETS_REQUIRED: usize = 2;
    pub const DEPTH_EFFECT: f64 = 0.05;
}

pub fn to_json() -> Json {
    let worlds = WORLDS
        .iter()
        .map(|k| {
            let w = world(*k);
            Json::obj(vec![
                ("kind", Json::str(k.name())),
                ("kappa", Json::Num(w.kappa)),
                ("slow_noise", Json::Num(w.slow_noise)),
                ("rho", Json::Num(w.rho)),
                ("second_std", Json::Num(w.second_std)),
                ("obs_noise", Json::Num(w.obs_noise)),
            ])
        })
        .collect();
    let conds = conditions()
        .iter()
        .map(|c| {
            Json::obj(vec![
                ("name", Json::str(&c.name)),
                ("representation", Json::str(c.representation.name())),
                (
                    "levels",
                    Json::Arr(
                        c.levels
                            .iter()
                            .map(|l| {
                                Json::obj(vec![
                                    ("stride", Json::Int(l.stride as i64)),
                                    ("window", Json::Int(l.window as i64)),
                                    ("width", Json::Int(l.width as i64)),
                                    ("horizon", Json::Int(l.horizon as i64)),
                                ])
                            })
                            .collect(),
                    ),
                ),
                ("planner_k", Json::ints(&budgets(c))),
            ])
        })
        .collect();
    let t = train_spec(0);
    use gates::*;
    Json::obj(vec![
        ("name", Json::str(NAME)),
        ("labels", Json::Arr(["RESEARCH_ONLY", "NO_RUNTIME_AUTHORITY", "NO_LANGUAGE_CLAIM", "SYNTHETIC", "SEMANTICS=NONE",
            "PREREGISTERED"].iter().map(|s| Json::str(*s)).collect())),
        ("base_seed", Json::Int(BASE_SEED as i64)),
        ("dev_seeds", Json::Arr(DEV_SEEDS.iter().map(|s| Json::Int(*s as i64)).collect())),
        ("qual_seeds", Json::Arr(QUAL_SEEDS.iter().map(|s| Json::Int(*s as i64)).collect())),
        ("worlds", Json::Arr(worlds)),
        ("data", Json::obj(vec![
            ("train_episodes", Json::Int(TRAIN_EPISODES as i64)),
            ("test_episodes", Json::Int(TEST_EPISODES as i64)),
            ("episode_steps", Json::Int(EPISODE_STEPS as i64)),
            ("exploration", Json::str("uniform actions in [-1, 1] held for 1..=8 steps")),
        ])),
        ("ecs", Json::obj(vec![
            ("primitive", Json::str("native K1 (elpis_ecsg_k1), one state per level; LEARN only (H stays 0: exact G1)")),
            ("dim", Json::Int(6)),
            ("input", Json::str("[z0, z1, b, sel0, sel1, 1]; target z'_k")),
            ("init_scale", Json::Num(t.init_scale)),
            ("base_rate", Json::Num(t.base_rate)),
            ("rate_rule", Json::str("base_rate * 36 / width")),
            ("batch_transitions", Json::Int(t.batch_transitions as i64)),
            ("steps_per_call", Json::Int(t.steps_per_call as i64)),
            ("max_rows", Json::Int(MAX_ROWS as i64)),
            ("calibration_steps", Json::Arr(CALIBRATION_STEPS.iter().map(|s| Json::Int(*s as i64)).collect())),
        ])),
        ("encoders", Json::obj(vec![
            ("level1", Json::str("whitening of the observation (every condition)")),
            ("temporal_shared", Json::str("none: the window's last lower latent (window 1)")),
            ("hierarchical_distinct", Json::str("affine 4 -> 2, two most predictable whitened directions of the window (linear one-tick predictor residual vs input covariance)")),
            ("action_pooling", Json::str("mean of the lower actions over the stride")),
        ])),
        ("conditions", Json::Arr(conds)),
        ("measurements", Json::obj(vec![
            ("horizons", Json::ints(&HORIZONS)),
            ("probes", Json::str("ridge (1e-3) linear probe, latent -> s and f at the window-end time; fit on train, NMSE on test")),
            ("attractors", Json::str("free run at b = 0 from a 5 x 5 grid on [-2, 2]^2 for 200 steps")),
        ])),
        ("planning", Json::obj(vec![
            ("episodes", Json::Int(PLAN_EPISODES as i64)),
            ("steps", Json::Int(PLAN_STEPS as i64)),
            ("replan_every", Json::Int(REPLAN_EVERY as i64)),
            ("flat_horizon", Json::Int(FLAT_HORIZON as i64)),
            ("flat_block", Json::Int(FLAT_BLOCK as i64)),
            ("success_tolerance", Json::Num(SUCCESS_TOLERANCE)),
            ("min_goal_distance", Json::Num(MIN_GOAL_DISTANCE)),
            ("start_range", Json::Num(START_RANGE)),
            ("goal", Json::str("observation rendered from (s_goal, f_goal), f_goal drawn from the second factor's stationary law")),
            ("oracle_k", Json::Int(ORACLE_K as i64)),
        ])),
        ("gates", Json::obj(vec![
            ("V1_L1_ONE_STEP_NMSE_MAX", Json::Num(V1_L1_ONE_STEP_NMSE)),
            ("V2_ORACLE_SUCCESS_MIN", Json::Num(V2_ORACLE_SUCCESS)),
            ("V3_L1_PROBE_NMSE_MAX", Json::Num(V3_L1_PROBE_NMSE)),
            ("V4_SEPARATED_CENTROID_RATIO_MIN", Json::Num(V4_SEPARATED_CENTROID_RATIO)),
            ("V4_MATCHED_CENTROID_RATIO_MAX", Json::Num(V4_MATCHED_CENTROID_RATIO)),
            ("H1_FAST_LOSS_MIN", Json::Num(H1_FAST_LOSS)),
            ("H1_SLOW_RETAINED_MAX", Json::Num(H1_SLOW_RETAINED)),
            ("H1_SEED_FRACTION_MIN", Json::Num(H1_SEED_FRACTION)),
            ("H1C_CONTROL_MARGIN_MIN", Json::Num(H1C_CONTROL_MARGIN)),
            ("H1T_SHARED_LOSS_MAX", Json::Num(H1T_SHARED_MAX_LOSS)),
            ("H2A_OVER_WIDTH_MIN", Json::Num(H2A_OVER_WIDTH)),
            ("H2B_OVER_TEMPORAL_MIN", Json::Num(H2B_OVER_TEMPORAL)),
            ("H2_BUDGETS_REQUIRED", Json::Int(H2_BUDGETS_REQUIRED as i64)),
            ("DEPTH_EFFECT", Json::Num(DEPTH_EFFECT)),
        ])),
    ])
}
