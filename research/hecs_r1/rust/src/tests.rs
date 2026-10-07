//! FAST MECHANICS: structure, bounds, determinism and level-state authority. No scientific outcome is read.
//!
//! Tests that need the native ECS load libelpis_ecsg_k1.so from ELPIS_ECS_K1_LIBRARY. Without it they
//! fail when ELPIS_REQUIRE_NATIVE=1 and are reported as not run otherwise (never an implicit pass).

use crate::encoder::Encoder;
use crate::hierarchy::{lift, Controller, LevelSequence};
use crate::k1::K1;
use crate::model::{LevelModel, Transition, LATENT};
use crate::planner::{plan_flat, plan_hierarchical};
use crate::rng::Rng;
use crate::spec;
use crate::world::{explore, spectral_centroid, World, WorldKind};

fn k1() -> Option<K1> {
    match std::env::var("ELPIS_ECS_K1_LIBRARY") {
        Ok(p) => Some(K1::load(std::path::Path::new(&p)).expect("load K1")),
        Err(_) => {
            assert!(std::env::var("ELPIS_REQUIRE_NATIVE").as_deref() != Ok("1"), "ELPIS_ECS_K1_LIBRARY required");
            eprintln!("NOT RUN: ELPIS_ECS_K1_LIBRARY unset");
            None
        }
    }
}

fn random_sequence(n: usize, rng: &mut Rng) -> LevelSequence {
    LevelSequence {
        z: (0..=n).map(|_| [rng.normal(), rng.normal()]).collect(),
        b: (0..n).map(|_| rng.range(-1.0, 1.0)).collect(),
    }
}

#[test]
fn every_condition_is_valid_bounded_and_width_matched() {
    let conds = spec::conditions();
    assert_eq!(conds.len(), 7);
    for c in &conds {
        c.validate().unwrap();
    }
    let width = |n: &str| conds.iter().find(|c| c.name == n).unwrap().total_width();
    assert_eq!(width("FLAT_N72"), width("HIERARCHICAL_DISTINCT_L2"));
    assert_eq!(width("FLAT_N72"), width("TEMPORAL_SHARED_L2"));
    assert_eq!(width("FLAT_N108"), width("HIERARCHICAL_DISTINCT_L3"));
    assert_eq!(width("FLAT_N108"), width("TEMPORAL_SHARED_L3"));
    // Compute matching (native forward rows per replan) within 4 percent across representations.
    for b in 0..3 {
        let rows = |c: &crate::hierarchy::HierarchySpec| -> f64 {
            let k = spec::budgets(c)[b] as f64;
            if c.levels.len() == 1 {
                k * spec::FLAT_HORIZON as f64 * LATENT as f64
            } else {
                let top = c.levels.last().unwrap().horizon as f64;
                let lower: f64 = c.levels[1..].iter().map(|l| l.stride as f64).sum();
                k * (top + lower) * LATENT as f64
            }
        };
        let flat = rows(&conds[0]);
        for c in &conds {
            assert!((rows(c) / flat - 1.0).abs() < 0.04, "{} budget {b}", c.name);
        }
    }
}

#[test]
fn the_committed_spec_is_the_rendered_spec_and_rejects_unbounded_hierarchies() {
    let committed = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../specs/hecs-r1.v1.spec.json"));
    if let Ok(text) = committed {
        assert_eq!(text, spec::to_json().render(), "specs/hecs-r1.v1.spec.json is frozen: it must equal `hecs1 spec`");
    }
    let mut c = spec::conditions()[6].clone();
    c.levels.push(c.levels[1]);
    assert!(c.validate().is_err(), "R1 stops at three levels");
    let mut c = spec::conditions()[5].clone();
    c.levels[1].stride = 64;
    assert!(c.validate().is_err());
}

#[test]
fn streaming_controller_equals_offline_lift_for_both_representations() {
    let mut rng = Rng::new(3, &[]);
    for cond in [&spec::conditions()[4], &spec::conditions()[6]] {
        let lower = random_sequence(200, &mut rng);
        let mut encoders = vec![Encoder::Shared];
        let mut seqs = vec![lower.clone()];
        for l in 1..cond.levels.len() {
            let enc = if cond.levels[l].window == 1 {
                Encoder::Shared
            } else {
                let (u, b, n) = crate::hierarchy::window_triples(&seqs[l - 1], &cond.levels[l]);
                Encoder::predictable(&u, &b, &n)
            };
            seqs.push(lift(&seqs[l - 1], &cond.levels[l], &enc));
            encoders.push(enc);
        }
        let mut ctl = Controller::start(cond, &encoders, lower.z[0]);
        let bytes = ctl.resident_bytes();
        let mut tick = vec![0usize; cond.levels.len()];
        for t in 1..=192 {
            for l in ctl.step(&encoders, lower.z[t]).into_iter().skip(1) {
                tick[l] += 1;
                assert_eq!(ctl.current[l], seqs[l].z[tick[l]], "{} level {l} tick {}", cond.name, tick[l]);
                // At a tick, the planning latent of that level is its tick latent.
                assert_eq!(ctl.planning_latents(&encoders)[l], ctl.current[l]);
            }
            assert_eq!(ctl.resident_bytes(), bytes, "bounded controller state");
        }
        assert_eq!(tick[1], 48);
    }
}

#[test]
fn predictable_components_keep_the_predictable_factor() {
    // Window of [slow, fast] lower latents: the slow factor persists across a tick, the fast one does not.
    let mut rng = Rng::new(9, &[]);
    let (mut s, mut f) = (0.0f64, 0.0f64);
    let mut z = Vec::new();
    for _ in 0..4000 {
        s = (0.995 * s + 0.1 * rng.normal()).clamp(-3.0, 3.0);
        f = -0.7 * f + 0.7 * rng.normal();
        z.push([s, f]);
    }
    let seq = LevelSequence { z, b: vec![0.0; 3999] };
    let spec = crate::hierarchy::LevelSpec { stride: 4, window: 2, width: 36, horizon: 4 };
    let (u, b, n) = crate::hierarchy::window_triples(&seq, &spec);
    let enc = Encoder::predictable(&u, &b, &n);
    let Encoder::Affine { p, predictability, .. } = &enc else { panic!() };
    // The most predictable direction loads on the slow coordinates (0 and 2 of the window), not the fast ones.
    let slow = p[(0, 0)].abs() + p[(0, 2)].abs();
    let fast = p[(0, 1)].abs() + p[(0, 3)].abs();
    assert!(slow > 5.0 * fast, "{slow} vs {fast}");
    assert!(predictability[0] < predictability[3]);
}

#[test]
fn worlds_are_deterministic_and_the_second_factor_frequency_is_as_constructed() {
    for kind in spec::WORLDS {
        let a = explore(&World::new(spec::world(kind), &mut Rng::new(1, &[2])), 256, &mut Rng::new(1, &[3]));
        let b = explore(&World::new(spec::world(kind), &mut Rng::new(1, &[2])), 256, &mut Rng::new(1, &[3]));
        assert_eq!(a.obs, b.obs);
        assert_eq!(a.actions, b.actions);
    }
    let fast = World::new(spec::world(WorldKind::Separated), &mut Rng::new(1, &[]));
    let close = World::new(spec::world(WorldKind::Matched), &mut Rng::new(1, &[]));
    let c = |w: &World| {
        let t = explore(w, 1024, &mut Rng::new(5, &[]));
        spectral_centroid(&t.truth.iter().map(|x| x.f).collect::<Vec<_>>())
    };
    assert!(c(&fast) > 3.0 * c(&close));
}

#[test]
fn each_level_owns_one_independent_k1_state() {
    let Some(lib) = k1() else { return };
    let mut rng = Rng::new(4, &[]);
    let mut l1 = LevelModel::new(&lib, 36, 512, 0.18, &mut rng).unwrap();
    let mut l2 = LevelModel::new(&lib, 36, 512, 0.18, &mut rng).unwrap();
    let before2 = l2.ecs.digest().unwrap();
    let data: Vec<Transition> = (0..64)
        .map(|i| {
            let z = [((i * 7) % 13) as f64 / 13.0 - 0.5, ((i * 5) % 11) as f64 / 11.0 - 0.5];
            Transition { z, b: 0.0, next: [0.9 * z[0], 0.5 * z[1]] }
        })
        .collect();
    let mse0 = l1.mse(&data).unwrap();
    let trace = l1.train(&data, &crate::model::TrainSpec { steps: 500, ..spec::train_spec(500) });
    assert!(trace.refused.is_none());
    assert!(l1.mse(&data).unwrap() < mse0);
    assert_eq!(l1.ecs.epoch(), 500);
    // Learning in level 1 never touches level 2's state.
    assert_eq!(l2.ecs.digest().unwrap(), before2);
    assert_eq!(l2.ecs.epoch(), 0);
}

#[test]
fn planning_counts_native_rows_exactly_and_is_deterministic() {
    let Some(lib) = k1() else { return };
    let conds = spec::conditions();
    let mut rng = Rng::new(8, &[]);
    // Small initial weights keep these untrained maps finite, so every row is counted exactly once.
    let mut flat = LevelModel::new(&lib, 36, 512, 0.02, &mut rng).unwrap();
    let p1 = plan_flat(&mut flat, [0.1, 0.2], [1.0, 0.0], 32, 16, 4, 4, &mut Rng::new(1, &[])).unwrap();
    assert_eq!(flat.forward_rows, 32 * 16 * 2);
    let p2 = plan_flat(&mut flat, [0.1, 0.2], [1.0, 0.0], 32, 16, 4, 4, &mut Rng::new(1, &[])).unwrap();
    assert_eq!(p1.actions, p2.actions);
    let hd = &conds[5];
    let mut models = vec![
        LevelModel::new(&lib, 36, 512, 0.02, &mut rng).unwrap(),
        LevelModel::new(&lib, 36, 512, 0.02, &mut rng).unwrap(),
    ];
    let seq = random_sequence(400, &mut rng);
    let (u, b, n) = crate::hierarchy::window_triples(&seq, &hd.levels[1]);
    let encoders = vec![Encoder::Shared, Encoder::predictable(&u, &b, &n)];
    let plan = plan_hierarchical(
        hd,
        &mut models,
        &encoders,
        &[[0.0, 0.0], [0.0, 0.0]],
        &[[1.0, 0.0], [1.0, 0.0]],
        64,
        &mut Rng::new(2, &[]),
    )
    .unwrap();
    assert_eq!(plan.actions.len(), 4);
    assert!(plan.subgoal.is_some());
    assert_eq!((models[0].forward_rows, models[1].forward_rows), (64 * 4 * 2, 64 * 4 * 2));
}

#[test]
fn validity_reference_and_law_are_exact_on_constructed_cases() {
    use crate::validity::{constant_baseline, linear_reference, persistence_baseline, Diagnostics};
    // An exactly linear level: z' = A z + B b, so the linear reference is exact and persistence is not.
    let mut rng = Rng::new(21, &[]);
    let seq = |rng: &mut Rng| {
        let mut z = vec![[rng.normal(), rng.normal()]];
        let b: Vec<f64> = (0..300).map(|_| rng.range(-1.0, 1.0)).collect();
        for t in 0..300 {
            let p = z[t];
            z.push([0.5 * p[0] - 0.2 * p[1] + 0.3 * b[t], 0.1 * p[0] + 0.7 * p[1] - 0.1 * b[t]]);
        }
        LevelSequence { z, b }
    };
    let (train, test) = (vec![seq(&mut rng), seq(&mut rng)], vec![seq(&mut rng)]);
    assert!(linear_reference(&train, &test) < 1e-12);
    assert!(constant_baseline(&train, &test) > 0.5 && persistence_baseline(&test) > 0.05);
    let d = Diagnostics { floor: 0.2, linear: 0.21, constant: 1.0, persistence: 1.5, ecs: 0.215 };
    assert!(d.reference_predictable(0.5) && !d.reference_predictable(0.8));
    assert!(d.reference_at_floor(0.005, 0.15) && !d.reference_at_floor(0.005, 0.0));
    assert!((d.capture() - 0.785 / 0.79).abs() < 1e-12 && d.ecs_adequate(0.98));
    assert!(!Diagnostics { ecs: f64::INFINITY, ..d }.ecs_adequate(0.0));
    assert!(!Diagnostics { ecs: 0.3, ..d }.ecs_adequate(0.98));
}

#[test]
fn the_seeds_are_fresh_and_distinct() {
    let all: Vec<u64> = spec::DEV_SEEDS.iter().chain(&spec::QUAL_SEEDS).chain(&spec::DESIGN_SEEDS).copied().collect();
    for (i, a) in all.iter().enumerate() {
        assert!(![0, 1, 2, 3, 100, 101, 102, 103, 104, 105, 106, 107, 9999].contains(a), "an R0 seed");
        assert!(!all[i + 1..].contains(a), "duplicate seed");
    }
}
