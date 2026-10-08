//! The preregistered H-ECS R2 specification (research/hecs_r2/PREREGISTRATION.md). Single source of every
//! parameter, condition, seed, gate and stop law. `hecs2 spec` renders it; the committed
//! specs/hecs-r2.v1.spec.json must equal that rendering byte for byte, and every evidence file embeds it.
//!
//! Derived from the R1 specification (research/hecs_r1/rust/src/spec.rs). Unchanged from R1: the worlds, the
//! data, the ECS primitive and its one-step training, the encoders, the seven conditions, the compute budgets,
//! the planner and its oracle, and the V2, V3, V4 (R1's corrected MATCHED bound) and V5 thresholds. New in R2:
//! the name and fresh seeds (recorded as decimal strings); the horizons and the horizon each planner consumes;
//! task validity extended to every level's one-step adequacy and to each consumed horizon; a calibration rule
//! with a margin; the primary multi-step gate M; planning hypotheses that are adjudicated only when M holds;
//! and the stop and integration laws.

use crate::hierarchy::{HierarchySpec, LevelSpec, Representation};
use crate::json::Json;
use crate::model::TrainSpec;
use crate::world::{WorldKind, WorldSpec};

pub const NAME: &str = "hecs-r2.v1";
pub const BASE_SEED: u64 = 20261007;
/// Seed i of a phase is the first 8 bytes (big-endian) of SHA-256("elpis.hecs-r2.<phase>.v1:<i>"), fixed before
/// any R2 observation (tests/research/hecs_r2 recomputes them). Disjoint from each other and from R0's and R1's.
/// Evidence records every seed as a decimal string (R1 recorded u64 seeds through a signed integer).
pub const SEED_DERIVATION: &str = "u64 big-endian of SHA-256(\"elpis.hecs-r2.<phase>.v1:<index>\")[0..8]";
pub const DESIGN_SEEDS: [u64; 4] = [8318582482988895235, 48883900885208620, 1842212770792861585, 15282691279643327234];
pub const DEV_SEEDS: [u64; 4] = [14754993789186315071, 12154359753255553410, 6287278211154781243, 10323661857439540229];
pub const QUAL_SEEDS: [u64; 8] = [
    1081895251583953854,
    2761248050979241786,
    10349694206428623058,
    4316384433055186278,
    17377210329154756622,
    1947799111408248483,
    2136955805524040833,
    17712258514216225143,
];

pub const TRAIN_EPISODES: usize = 16;
pub const TEST_EPISODES: usize = 8;
pub const EPISODE_STEPS: usize = 256;
pub const MAX_ROWS: usize = 512;
/// Calibration grid for the per-level training budget (K1 steps).
pub const CALIBRATION_STEPS: [u64; 4] = [6000, 24000, 96000, 384000];
/// Free-running horizons measured at every level (level steps).
pub const HORIZONS: [usize; 5] = [1, 2, 4, 8, 16];

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

/// The conditions, in report order (R1's). Total ECS width is matched: FLAT_N72 with the two-level hierarchies
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

/// The horizon (in its own steps) the planner consumes from level `l` of a condition: FLAT rolls level 1 out
/// over the whole flat horizon; in a hierarchy the top level over its horizon and every lower level over one
/// tick of the level above (its stride).
pub fn consumed_horizon(c: &HierarchySpec, l: usize) -> usize {
    if c.levels.len() == 1 {
        c.levels[0].horizon
    } else if l + 1 == c.levels.len() {
        c.levels[l].horizon
    } else {
        c.levels[l + 1].stride
    }
}

/// The conditions whose models the H2 planning comparisons consume.
pub const H2_CONDITIONS: [&str; 3] = ["FLAT_N72", "TEMPORAL_SHARED_L2", "HIERARCHICAL_DISTINCT_L2"];

/// Planner candidate counts per condition at the three compute points (R1's). Native forward rows per replan:
/// flat K x 16 x 2 = {256, 1024, 4096}; two levels K x (4 + 4) x 2 (equal); three levels K x (4 + 4 + 4) x 2
/// = {264, 1032, 4104} (within 4 percent).
pub fn budgets(spec: &HierarchySpec) -> [usize; 3] {
    match spec.levels.len() {
        1 => [8, 32, 128],
        2 => [16, 64, 256],
        _ => [11, 43, 171],
    }
}

/// Preregistered gate thresholds.
pub mod gates {
    // -- task validity (every instance, both worlds) ----------------------------------------------------------
    /// T1A: level 1's held-out linear reference removes at least this fraction of the constant baseline's
    /// one-step error (R1's V1A).
    pub const T1A_MIN_EXPLAINED: f64 = 0.5;
    /// T1B: |level-1 linear reference - analytic floor| <= abs + rel * floor, one step (R1's V1B).
    pub const T1B_FLOOR_TOL_ABS: f64 = 0.005;
    pub const T1B_FLOOR_TOL_REL: f64 = 0.15;
    /// T2 (training adequacy): every trained level model, one step, no divergence and capture >= this.
    pub const T2_MIN_CAPTURE: f64 = 0.98;
    /// T3A: at each consumed horizon the iterated linear reference removes at least this fraction of the
    /// constant baseline's error (the horizon still carries predictable structure).
    pub const T3A_MIN_EXPLAINED: f64 = 0.25;
    /// T3B: at each consumed horizon of level 1 and the temporal-shared levels, |iterated linear reference -
    /// analytic floor| <= abs + rel * floor (the reference is the generator's floor there too).
    pub const T3B_FLOOR_TOL_ABS: f64 = 0.01;
    pub const T3B_FLOOR_TOL_REL: f64 = 0.15;
    pub const V2_ORACLE_SUCCESS: f64 = 0.8;
    pub const V3_L1_PROBE_NMSE: f64 = 0.1;
    pub const V4_SEPARATED_CENTROID_RATIO: f64 = 4.0;
    pub const V4_MATCHED_CENTROID_RATIO: f64 = 2.0;
    // -- calibration ----------------------------------------------------------------------------------------
    /// The calibration margin: every DEV instance and every level's one-step capture at least this (above
    /// T2's 0.98), with T1 holding; the chosen budget is then one grid step above the smallest such budget.
    pub const CAL_MIN_CAPTURE: f64 = 0.99;
    pub const CAL_GRID_STEPS_ABOVE: usize = 1;
    // -- M: multi-step validity (the primary question) --------------------------------------------------------
    /// At the consumed horizon: free-running capture (constant - ecs) / (constant - linear) at least this ...
    pub const M_MIN_CAPTURE: f64 = 0.90;
    /// ... and escaped (non-finite or beyond the escape bound) rollouts at most this fraction.
    pub const M_MAX_ESCAPED: f64 = 0.01;
    /// A condition is M-valid in a world when, for every level, the seed means meet both bounds and at least
    /// this fraction of seeds meet both at every level.
    pub const M_SEED_FRACTION: f64 = 0.75;
    // -- H2: planning (SEPARATED), adjudicated only if M holds there for every H2 condition -------------------
    pub const H2A_DISTINCT_OVER_WIDTH: f64 = 0.15;
    pub const H2B_DISTINCT_OVER_TEMPORAL: f64 = 0.10;
    pub const H2C_TEMPORAL_OVER_WIDTH: f64 = 0.10;
    pub const H2_BUDGETS_REQUIRED: usize = 2;
    pub const DEPTH_EFFECT: f64 = 0.05;
}

fn seeds(v: &[u64]) -> Json {
    Json::Arr(v.iter().map(|s| Json::str(s.to_string())).collect())
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
                            .enumerate()
                            .map(|(l, s)| {
                                Json::obj(vec![
                                    ("stride", Json::Int(s.stride as i64)),
                                    ("window", Json::Int(s.window as i64)),
                                    ("width", Json::Int(s.width as i64)),
                                    ("horizon", Json::Int(s.horizon as i64)),
                                    ("consumed_horizon", Json::Int(consumed_horizon(c, l) as i64)),
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
    let num = |v: f64| Json::Num(v);
    Json::obj(vec![
        ("name", Json::str(NAME)),
        ("labels", Json::Arr(["RESEARCH_ONLY", "NO_RUNTIME_AUTHORITY", "NO_LANGUAGE_CLAIM", "SYNTHETIC", "SEMANTICS=NONE",
            "PREREGISTERED"].iter().map(|s| Json::str(*s)).collect())),
        ("question", Json::str("Can the qualified ECS/K1 world-model primitive, trained one step ahead, produce a multi-step predictive object stable enough for planning at the horizons H-ECS consumes? Hierarchy is adjudicated only if it can.")),
        ("base_seed", Json::str(BASE_SEED.to_string())),
        ("seed_derivation", Json::str(SEED_DERIVATION)),
        ("design_seeds", seeds(&DESIGN_SEEDS)),
        ("dev_seeds", seeds(&DEV_SEEDS)),
        ("qual_seeds", seeds(&QUAL_SEEDS)),
        ("worlds", Json::Arr(worlds)),
        ("data", Json::obj(vec![
            ("train_episodes", Json::Int(TRAIN_EPISODES as i64)),
            ("test_episodes", Json::Int(TEST_EPISODES as i64)),
            ("episode_steps", Json::Int(EPISODE_STEPS as i64)),
            ("exploration", Json::str("uniform actions in [-1, 1] held for 1..=8 steps")),
        ])),
        ("ecs", Json::obj(vec![
            ("primitive", Json::str("native K1 (elpis_ecsg_k1), one state (W, epoch, H, a) per level; LEARN only (H stays 0: exact G1); W is the authority, S3 never stands in for it")),
            ("dim", Json::Int(6)),
            ("input", Json::str("[z0, z1, b, sel0, sel1, 1]; target z'_k")),
            ("training", Json::str("one step ahead (teacher-forced transitions), unchanged from R0/R1")),
            ("init_scale", num(t.init_scale)),
            ("base_rate", num(t.base_rate)),
            ("rate_rule", Json::str("base_rate * 36 / width")),
            ("batch_transitions", Json::Int(t.batch_transitions as i64)),
            ("steps_per_call", Json::Int(t.steps_per_call as i64)),
            ("max_rows", Json::Int(MAX_ROWS as i64)),
            ("calibration_steps", Json::Arr(CALIBRATION_STEPS.iter().map(|s| Json::Int(*s as i64)).collect())),
        ])),
        ("encoders", Json::obj(vec![
            ("level1", Json::str("whitening of the observation (every condition)")),
            ("temporal_shared", Json::str("none: the window's last lower latent (window 1)")),
            ("hierarchical_distinct", Json::str("affine 4 -> 2, two most predictable whitened directions of the window (closed form, fixed before the ECS trains): any abstraction it shows is the encoder's, not learned by the ECS")),
            ("action_pooling", Json::str("mean of the lower actions over the stride")),
        ])),
        ("conditions", Json::Arr(conds)),
        ("measurements", Json::obj(vec![
            ("horizons", Json::ints(&HORIZONS)),
            ("free_running", Json::str("open-loop rollouts with the true level actions from every held-out start; escaped (non-finite or any |z| > 1e3) rollouts stop and are counted")),
            ("teacher_forced", Json::str("the one-step prediction of the same target from the true previous latent")),
            ("references", Json::str("iterated one-step linear reference (fit on training data only); generator dynamics from the latent's factor estimate (level 1, temporal-shared levels); analytic h-step floor (same levels); constant baseline (training mean)")),
            ("capture", Json::str("(constant - ecs_free) / (constant - linear) at horizon h")),
            ("diagnostics", Json::str("norm ratio, passive (b = 0) escape, perturbation amplification (eps 1e-4, 256 starts), empirical one-step Jacobian spectral radius (256 held-out states); reported, never gates")),
            ("probes", Json::str("ridge (1e-3) linear probe, latent -> s and f at the window-end time; fit on train, NMSE on test")),
            ("attractors", Json::str("free run at b = 0 from a 5 x 5 grid on [-2, 2]^2 for 200 steps")),
        ])),
        ("planning", Json::obj(vec![
            ("episodes", Json::Int(PLAN_EPISODES as i64)),
            ("steps", Json::Int(PLAN_STEPS as i64)),
            ("replan_every", Json::Int(REPLAN_EVERY as i64)),
            ("flat_horizon", Json::Int(FLAT_HORIZON as i64)),
            ("flat_block", Json::Int(FLAT_BLOCK as i64)),
            ("cost", Json::str("summed squared distance of every predicted state to the goal (R1's path cost); subgoal tracking scored at the tick")),
            ("success_tolerance", num(SUCCESS_TOLERANCE)),
            ("min_goal_distance", num(MIN_GOAL_DISTANCE)),
            ("start_range", num(START_RANGE)),
            ("goal", Json::str("observation rendered from (s_goal, f_goal), f_goal drawn from the second factor's stationary law")),
            ("oracle_k", Json::Int(ORACLE_K as i64)),
        ])),
        ("gates", Json::obj(vec![
            ("T1A_MIN_EXPLAINED", num(T1A_MIN_EXPLAINED)),
            ("T1B_FLOOR_TOL_ABS", num(T1B_FLOOR_TOL_ABS)),
            ("T1B_FLOOR_TOL_REL", num(T1B_FLOOR_TOL_REL)),
            ("T2_MIN_CAPTURE", num(T2_MIN_CAPTURE)),
            ("T3A_MIN_EXPLAINED", num(T3A_MIN_EXPLAINED)),
            ("T3B_FLOOR_TOL_ABS", num(T3B_FLOOR_TOL_ABS)),
            ("T3B_FLOOR_TOL_REL", num(T3B_FLOOR_TOL_REL)),
            ("V2_ORACLE_SUCCESS_MIN", num(V2_ORACLE_SUCCESS)),
            ("V3_L1_PROBE_NMSE_MAX", num(V3_L1_PROBE_NMSE)),
            ("V4_SEPARATED_CENTROID_RATIO_MIN", num(V4_SEPARATED_CENTROID_RATIO)),
            ("V4_MATCHED_CENTROID_RATIO_MAX", num(V4_MATCHED_CENTROID_RATIO)),
            ("CAL_MIN_CAPTURE", num(CAL_MIN_CAPTURE)),
            ("CAL_GRID_STEPS_ABOVE", Json::Int(CAL_GRID_STEPS_ABOVE as i64)),
            ("M_MIN_CAPTURE", num(M_MIN_CAPTURE)),
            ("M_MAX_ESCAPED", num(M_MAX_ESCAPED)),
            ("M_SEED_FRACTION", num(M_SEED_FRACTION)),
            ("H2A_DISTINCT_OVER_WIDTH", num(H2A_DISTINCT_OVER_WIDTH)),
            ("H2B_DISTINCT_OVER_TEMPORAL", num(H2B_DISTINCT_OVER_TEMPORAL)),
            ("H2C_TEMPORAL_OVER_WIDTH", num(H2C_TEMPORAL_OVER_WIDTH)),
            ("H2_BUDGETS_REQUIRED", Json::Int(H2_BUDGETS_REQUIRED as i64)),
            ("DEPTH_EFFECT", num(DEPTH_EFFECT)),
        ])),
        ("laws", Json::Arr([
            "TASK VALIDITY (every instance, both worlds): T1A, T1B (level 1, one step); T2 (every trained level, one step); T3A at every (condition, level, consumed horizon); T3B at the consumed horizons of level 1 and the temporal-shared levels; V2 (oracle mean per world); V3; V4 (means); V5 (no K1 refusal)",
            "CALIBRATION (DEV): B* = the smallest grid budget at which every DEV instance has T1 and every level's one-step capture >= CAL_MIN_CAPTURE; the chosen budget is the grid value CAL_GRID_STEPS_ABOVE above B*; no such B*, or no grid value above it: TASK_INVALID_ON_DEV, stop",
            "DEV RUN at the chosen budget: task validity fails: TASK_INVALID_ON_DEV, stop; else QUAL is authorized (DEV M and H2 values are non-binding)",
            "QUAL, once, at the DEV budget: task validity fails: TASK_INVALID",
            "M (QUAL): condition c is M-valid in a world iff for every level l of c at its consumed horizon the seed-mean capture >= M_MIN_CAPTURE and the seed-mean escaped fraction <= M_MAX_ESCAPED, and at least M_SEED_FRACTION of the seeds meet both at every level",
            "M fails in SEPARATED for any H2 condition: WORLD_MODEL_INVALID; the hierarchy hypotheses are not adjudicated",
            "else HIERARCHY_ADJUDICATED: H2A, H2B, H2C on SEPARATED success at the three compute points; width and depth reported",
            "INTEGRATION (experimental, noncanonical) is authorized only by HIERARCHY_ADJUDICATED with H2A and H2B both holding",
            "no gate, threshold, world parameter, budget, seed, horizon or condition changes after DEV data exist; a mechanics failure after the freeze is classified MECHANICS_FAILURE and repaired in the smallest mechanical layer, never in the scientific object",
        ].iter().map(|s| Json::str(*s)).collect())),
    ])
}
