//! hecs: the H-ECS R0 command line (research/hecs_r0/README.md).
//!
//!   hecs spec                                      the preregistered specification (JSON)
//!   hecs calibrate --k1 LIB [--out FILE]           DEV calibration of the training budget (V1 only)
//!   hecs run --phase dev|qual --steps N --k1 LIB --out FILE

use std::path::PathBuf;
use std::time::Instant;

use hecs_r0::experiment::{evaluate, level_keys, run_seed, world_data};
use hecs_r0::json::Json;
use hecs_r0::k1::K1;
use hecs_r0::model::LevelModel;
use hecs_r0::rng::Rng;
use hecs_r0::spec::{self, gates};
use hecs_r0::world::WorldKind;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn write(out: Option<String>, json: &Json) {
    let text = json.render();
    match out {
        Some(path) => std::fs::write(&path, text).unwrap_or_else(|e| panic!("cannot write {path}: {e}")),
        None => print!("{text}"),
    }
}

fn load(args: &[String]) -> K1 {
    let path = PathBuf::from(arg(args, "--k1").expect("--k1 PATH to libelpis_ecsg_k1.so"));
    let k1 = K1::load(&path).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(k1.abi_version(), 1, "K1 ABI version");
    k1
}

fn calibrate(args: &[String]) {
    let k1 = load(args);
    let started = Instant::now();
    let mut rows = Vec::new();
    let mut chosen = None;
    for steps in spec::CALIBRATION_STEPS {
        let mut worst = 0.0f64;
        let mut values = Vec::new();
        for kind in spec::WORLDS {
            for seed in spec::DEV_SEEDS {
                let data = world_data(kind, seed);
                let train: Vec<_> = data
                    .train
                    .iter()
                    .map(|t| hecs_r0::hierarchy::LevelSequence {
                        z: t.obs.iter().map(|o| hecs_r0::encoder::encode_obs(&data.enc1, o)).collect(),
                        b: t.actions.clone(),
                    })
                    .collect();
                let test: Vec<_> = data
                    .test
                    .iter()
                    .map(|t| {
                        (
                            t.obs.iter().map(|o| hecs_r0::encoder::encode_obs(&data.enc1, o)).collect::<Vec<_>>(),
                            t.actions.clone(),
                        )
                    })
                    .collect();
                let tag: u64 = "L1/N36".bytes().fold(7u64, |h, b| h.wrapping_mul(131).wrapping_add(b as u64));
                let kt = if kind == WorldKind::Separated { 11 } else { 13 };
                let mut rng = Rng::new(spec::BASE_SEED, &[seed, kt, 2, tag]);
                let mut m = LevelModel::new(&k1, 36, spec::MAX_ROWS, spec::train_spec(steps).init_scale, &mut rng)
                    .expect("create");
                let mut data_t: Vec<_> = train.iter().flat_map(|s| s.transitions()).collect();
                for i in (1..data_t.len()).rev() {
                    let j = rng.below(i as u64 + 1) as usize;
                    data_t.swap(i, j);
                }
                let trace = m.train(&data_t, &spec::train_spec(steps));
                let v = if trace.refused.is_some() {
                    f64::INFINITY
                } else {
                    match hecs_r0::analysis::horizon_error(&mut m, &test, 1) {
                        Ok((e, 0.0)) => e,
                        _ => f64::INFINITY,
                    }
                };
                eprintln!("[hecs] calibrate steps {steps} {} seed {seed}: L1 one-step NMSE {v:.4}", kind.name());
                worst = worst.max(v);
                values.push(Json::Num(v));
            }
        }
        rows.push(Json::obj(vec![
            ("steps", Json::Int(steps as i64)),
            ("l1_one_step_nmse", Json::Arr(values)),
            ("worst", Json::Num(worst)),
        ]));
        if worst <= gates::V1_L1_ONE_STEP_NMSE {
            chosen = Some(steps);
            break;
        }
    }
    eprintln!("[hecs] calibration done in {:.1}s", started.elapsed().as_secs_f64());
    let _ = level_keys;
    write(
        arg(args, "--out"),
        &Json::obj(vec![
            ("spec", spec::to_json()),
            ("phase", Json::str("DEV_CALIBRATION")),
            (
                "rule",
                Json::str("smallest grid budget whose FLAT_N36 level-1 one-step NMSE meets V1 in every DEV world"),
            ),
            ("grid", Json::Arr(rows)),
            ("chosen_steps", chosen.map_or(Json::Null, |s| Json::Int(s as i64))),
            ("disposition", Json::str(if chosen.is_some() { "CALIBRATED" } else { "TASK_INVALID_ON_DEV" })),
        ]),
    );
}

fn run(args: &[String]) {
    let k1 = load(args);
    let phase = arg(args, "--phase").expect("--phase dev|qual");
    let steps: u64 = arg(args, "--steps").expect("--steps N").parse().expect("integer steps");
    assert!(spec::CALIBRATION_STEPS.contains(&steps), "steps must be a calibration grid value");
    let seeds: Vec<u64> = match phase.as_str() {
        "dev" => spec::DEV_SEEDS.to_vec(),
        "qual" => spec::QUAL_SEEDS.to_vec(),
        _ => panic!("--phase dev|qual"),
    };
    let started = Instant::now();
    let mut per_world = Vec::new();
    let mut results = Vec::new();
    for kind in spec::WORLDS {
        let mut rs = Vec::new();
        for &seed in &seeds {
            let t = Instant::now();
            let r = run_seed(&k1, kind, seed, steps).unwrap_or_else(|e| panic!("{} seed {seed}: {e}", kind.name()));
            eprintln!("[hecs] {} seed {seed} done in {:.1}s", kind.name(), t.elapsed().as_secs_f64());
            rs.push(r);
        }
        per_world.push(Json::obj(vec![
            ("world", Json::str(kind.name())),
            ("seeds", Json::Arr(rs.iter().map(|r| r.json.clone()).collect())),
        ]));
        results.push(rs);
    }
    let evaluation = evaluate(&results[0], &results[1]);
    eprintln!("[hecs] {phase} done in {:.1}s", started.elapsed().as_secs_f64());
    write(
        arg(args, "--out"),
        &Json::obj(vec![
            ("spec", spec::to_json()),
            ("phase", Json::str(phase.to_uppercase())),
            ("training_steps", Json::Int(steps as i64)),
            ("k1_abi_version", Json::Int(k1.abi_version() as i64)),
            ("worlds", Json::Arr(per_world)),
            ("evaluation", evaluation),
        ]),
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("spec") => print!("{}", spec::to_json().render()),
        Some("calibrate") => calibrate(&args),
        Some("run") => run(&args),
        _ => {
            eprintln!(
                "usage: hecs spec | calibrate --k1 LIB [--out F] | run --phase dev|qual --steps N --k1 LIB --out F"
            );
            std::process::exit(2);
        }
    }
}
