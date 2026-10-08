//! FAST MECHANICS: structure, bounds, determinism, level-state authority, the multi-step instrumentation and
//! seed provenance. No scientific outcome is read.
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
    let committed = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../specs/hecs-r2.v1.spec.json"));
    if let Ok(text) = committed {
        assert_eq!(text, spec::to_json().render(), "specs/hecs-r2.v1.spec.json is frozen: it must equal `hecs2 spec`");
    }
    let mut c = spec::conditions()[6].clone();
    c.levels.push(c.levels[1]);
    assert!(c.validate().is_err(), "R2 stops at three levels");
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

/// R1's seeds (research/hecs_r1/specs/hecs-r1.v1.spec.json), as decimal strings.
const R1_SEEDS: [&str; 16] = [
    "209439022501430681",
    "15142748718241856473",
    "3237598911337905887",
    "513197779884602256",
    "16186020972637718675",
    "14588923398969713564",
    "3901883399138462192",
    "2748090251381783525",
    "15966254688005471966",
    "18144351229204560108",
    "14670577078401291347",
    "2581806976561750120",
    "15236446594267265809",
    "88216728179822102",
    "3724062739189718506",
    "15985262823488282381",
];

#[test]
fn the_seeds_are_fresh_distinct_and_recorded_as_exact_decimal_strings() {
    let all: Vec<u64> = spec::DEV_SEEDS.iter().chain(&spec::QUAL_SEEDS).chain(&spec::DESIGN_SEEDS).copied().collect();
    let r1: Vec<u64> = R1_SEEDS.iter().map(|s| s.parse().unwrap()).collect();
    for (i, a) in all.iter().enumerate() {
        assert!(![0, 1, 2, 3, 100, 101, 102, 103, 104, 105, 106, 107, 9999].contains(a), "an R0 seed");
        assert!(!r1.contains(a), "an R1 seed");
        assert!(!all[i + 1..].contains(a), "duplicate seed");
    }
    // Seeds above i64::MAX exist, so a signed record would wrap: the specification records decimal strings that
    // parse back to the computation seed exactly.
    assert!(all.iter().any(|s| *s > i64::MAX as u64));
    let rendered = spec::to_json().render();
    for s in &all {
        assert!(rendered.contains(&format!("\"{s}\"")), "seed {s} recorded as a decimal string");
        assert_eq!(s.to_string().parse::<u64>().unwrap(), *s);
        assert!(!rendered.contains(&format!("{}", *s as i64)) || *s <= i64::MAX as u64, "no signed wrap of {s}");
    }
}

#[test]
fn consumed_horizons_are_what_the_planners_roll_out() {
    for c in spec::conditions() {
        for l in 0..c.levels.len() {
            let h = spec::consumed_horizon(&c, l);
            assert!(spec::HORIZONS.contains(&h), "{} level {l}", c.name);
            let expected = match (c.levels.len(), l) {
                (1, _) => spec::FLAT_HORIZON,
                (n, l) if l + 1 == n => c.levels[l].horizon,
                (_, l) => c.levels[l + 1].stride,
            };
            assert_eq!(h, expected);
        }
    }
    let flat = &spec::conditions()[1];
    assert_eq!(spec::consumed_horizon(flat, 0), 16);
    let hd = &spec::conditions()[5];
    assert_eq!((spec::consumed_horizon(hd, 0), spec::consumed_horizon(hd, 1)), (4, 4));
}

/// An exactly linear level z' = A z + B b.
fn linear_sequence(rng: &mut Rng, n: usize) -> LevelSequence {
    let mut z = vec![[rng.normal(), rng.normal()]];
    let b: Vec<f64> = (0..n).map(|_| rng.range(-1.0, 1.0)).collect();
    for t in 0..n {
        let p = z[t];
        z.push([0.5 * p[0] - 0.2 * p[1] + 0.3 * b[t], 0.1 * p[0] + 0.7 * p[1] - 0.1 * b[t]]);
    }
    LevelSequence { z, b }
}

#[test]
fn rollouts_stop_at_escape_and_the_iterated_linear_reference_is_exact_on_a_linear_level() {
    use crate::multistep::{references, rollouts, Frame, Step, ESCAPE_BOUND};
    struct Doubling;
    impl Step for Doubling {
        fn step(&mut self, inputs: &[([f64; 2], f64)]) -> Result<Vec<Option<[f64; 2]>>, crate::k1::K1Error> {
            Ok(inputs.iter().map(|(z, _)| Some([2.0 * z[0], z[1]])).collect())
        }
    }
    let r = rollouts(&mut Doubling, &[[1.0, 0.0], [0.0, 1.0]], &[vec![0.0; 12], vec![0.0; 3]], 12).unwrap();
    // 2^9 = 512 <= 1e3 < 2^10: the first start escapes at step 10 and stays escaped; the second runs 3 steps.
    assert!(ESCAPE_BOUND == 1e3 && r[0][8] == Some([512.0, 0.0]) && r[0][9].is_none() && r[0][11].is_none());
    assert_eq!(r[1].len(), 3);
    assert!(r[1].iter().all(|z| *z == Some([0.0, 1.0])));

    let mut rng = Rng::new(22, &[]);
    let (train, test) =
        (vec![linear_sequence(&mut rng, 300), linear_sequence(&mut rng, 300)], vec![linear_sequence(&mut rng, 300)]);
    let refs = references(&train, &test, Frame::Opaque, &[1, 2, 4, 8, 16]);
    for r in &refs {
        assert!(r.linear < 1e-12, "h {}: {}", r.h, r.linear);
        assert!(r.constant > 0.1 && r.generator.is_none() && r.floor.is_none());
        assert_eq!(r.starts, 300 + 1 - r.h);
    }
}

#[test]
fn the_multistep_references_at_one_step_are_r1s_and_the_generator_sits_at_its_floor() {
    use crate::experiment::{frame, level_sequences, world_data};
    use crate::multistep::references;
    use crate::validity::{analytic_floor, constant_baseline, linear_reference};
    for kind in spec::WORLDS {
        let data = world_data(kind, 5);
        for d in level_sequences(&data) {
            let rs = references(&d.train, &d.test, frame(&d.key, d.env_stride, &data), &spec::HORIZONS);
            let one = &rs[0];
            let rel = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1e-12);
            // h = 1 is R1's one-step reference and baseline, on the same examples.
            assert!(rel(one.linear, linear_reference(&d.train, &d.test)), "{} linear", d.key);
            assert!(rel(one.constant, constant_baseline(&d.train, &d.test)), "{} constant", d.key);
            if d.key.starts_with("L1/") {
                assert!(rel(one.floor.unwrap(), analytic_floor(&data.world, &data.enc1, &d.test)), "{} floor", d.key);
            }
            // In observation frames the generator predictor's error is its analytic floor (sampling and the
            // unmodelled clipping of the slow factor aside); elsewhere there is no generator frame.
            for r in &rs {
                match (r.generator, r.floor) {
                    (Some(g), Some(f)) if r.starts >= 100 => {
                        assert!(
                            (g - f).abs() <= 0.25 * f + 0.01,
                            "{} {} h {}: generator {g} floor {f}",
                            kind.name(),
                            d.key,
                            r.h
                        )
                    }
                    (Some(_), Some(_)) => {}
                    (None, None) => assert!(d.key.starts_with("HD/")),
                    _ => panic!("generator and floor come together"),
                }
            }
        }
    }
}

#[test]
fn spectral_radius_is_exact_on_constructed_matrices() {
    use crate::multistep::spectral_radius;
    assert!((spectral_radius(&[[0.5, 0.0], [0.0, -0.9]]) - 0.9).abs() < 1e-15);
    // Rotation by any angle scaled by 0.8: complex eigenvalues of modulus 0.8.
    let (c, s) = (0.3f64.cos(), 0.3f64.sin());
    assert!((spectral_radius(&[[0.8 * c, -0.8 * s], [0.8 * s, 0.8 * c]]) - 0.8).abs() < 1e-12);
    assert!((spectral_radius(&[[1.0, 1.0], [0.0, 1.0]]) - 1.0).abs() < 1e-12);
}

#[test]
fn level_sequences_are_the_trained_pool_and_free_running_at_one_step_is_teacher_forced() {
    use crate::experiment::{build_pool, frame, level_sequences, world_data};
    use crate::multistep::{horizons, jacobians};
    let Some(lib) = k1() else { return };
    let data = world_data(WorldKind::Separated, 6);
    let mut pool = build_pool(&lib, &data, 500, 6, WorldKind::Separated).unwrap();
    let seqs = level_sequences(&data);
    assert_eq!(pool.len(), 7);
    for (e, d) in pool.iter().zip(&seqs) {
        assert_eq!((&e.key, e.env_stride, e.model.as_ref().unwrap().width), (&d.key, d.env_stride, d.width));
        assert!(e.train.iter().zip(&d.train).all(|(a, b)| a.z == b.z && a.b == b.b));
        assert!(e.test.iter().zip(&d.test).all(|(a, b)| a.z == b.z && a.b == b.b));
    }
    let e = &mut pool[0];
    let fr = frame(&e.key, e.env_stride, &data);
    let model = e.model.as_mut().unwrap();
    let digest = model.ecs.digest().unwrap();
    let hs = horizons(model, &e.train, &e.test, fr, &[1, 2, 4], &[1]).unwrap();
    // Measuring never learns: the level state is unchanged.
    assert_eq!(model.ecs.digest().unwrap(), digest);
    assert_eq!(hs[0].ecs_free, hs[0].ecs_teacher);
    let seq: Vec<_> = e.test.iter().map(|s| (s.z.clone(), s.b.clone())).collect();
    let (one, _) = crate::analysis::horizon_error(model, &seq, 1).unwrap();
    assert!((hs[0].ecs_free - one).abs() <= 1e-12 * one, "{} vs {one}", hs[0].ecs_free);
    // Teacher-forced at h: the one-step prediction of z_{t+h} from the true z_{t+h-1}, over the starts with a
    // target at h, recomputed here one input at a time.
    let var = crate::validity::normalizer(&e.test);
    for s in &hs[1..] {
        let (mut sum, mut n) = (0.0, 0usize);
        for q in &e.test {
            for t in 0..q.b.len() {
                if t + s.h <= q.b.len() {
                    let p = model.predict(&[(q.z[t + s.h - 1], q.b[t + s.h - 1])]).unwrap()[0];
                    sum += ((p[0] - q.z[t + s.h][0]).powi(2) + (p[1] - q.z[t + s.h][1]).powi(2)) / 2.0;
                    n += 1;
                }
            }
        }
        assert_eq!(n, s.starts);
        assert!((s.ecs_teacher - sum / n as f64 / var).abs() <= 1e-9 * s.ecs_teacher, "h {}", s.h);
        assert!(s.ecs_escaped == 0.0 && s.capture.is_finite() && s.amp_median.is_finite());
    }
    let (mean, median, over) = jacobians(model, &e.test).unwrap();
    assert!(mean.is_finite() && median.is_finite() && (0.0..=1.0).contains(&over));
}

/// Whitespace-free text (the evidence is pretty-printed at a depth the fragment does not know).
fn compact(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

#[test]
fn every_recorded_seed_is_the_computation_seed_and_reproduces_its_evidence() {
    use crate::experiment::{references, world_data};
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../");
    let frozen = std::path::Path::new(root).join("specs/hecs-r2.v1.spec.json").exists();
    let files = [
        ("evidence/design/hecs-r2.v1.design.json", &spec::DESIGN_SEEDS[..]),
        ("evidence/dev/hecs-r2.v1.dev.json", &spec::DEV_SEEDS[..]),
        ("evidence/qual/hecs-r2.v1.qual.json", &spec::QUAL_SEEDS[..]),
    ];
    for (i, (rel, seeds)) in files.iter().enumerate() {
        let Ok(text) = std::fs::read_to_string(std::path::Path::new(root).join(rel)) else {
            // The DESIGN evidence is frozen with the specification; DEV and QUAL exist only once run.
            assert!(!(frozen && i == 0), "{rel} must exist once the specification is frozen");
            eprintln!("NOT RUN: {rel} absent");
            continue;
        };
        let body = compact(&text);
        // Every recorded seed is a decimal string of the phase, never a (possibly wrapped) integer.
        assert!(body.match_indices("\"seed\":").all(|(at, m)| body[at + m.len()..].starts_with('"')));
        for kind in spec::WORLDS {
            for &seed in seeds.iter() {
                let fragment = compact(&references(&world_data(kind, seed), seed, kind).json.render());
                assert!(body.contains(&fragment), "{rel}: {} seed {seed} does not reproduce its record", kind.name());
            }
        }
    }
}
