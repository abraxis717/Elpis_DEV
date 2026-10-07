//! hecs1: the H-ECS R1 command line (research/hecs_r1/README.md).
//!
//!   hecs1 spec                                       the preregistered specification (JSON)
//!   hecs1 design --k1 LIB [--out FILE]               DESIGN seeds only: validity diagnostics (no hypotheses)
//!   hecs1 calibrate --k1 LIB [--out FILE]            DEV calibration of the training budget (V1 law only)
//!   hecs1 run --phase dev|qual --steps N --k1 LIB --out FILE
//!
//! Derived from R0's `hecs` (research/hecs_r0/rust/src/main.rs): `design` is new, `calibrate` applies the R1
//! V1 law through `experiment::level1_validity` (the same level-1 model the run trains), `run` is unchanged.

use std::path::PathBuf;
use std::time::Instant;

use hecs_r1::experiment::{
    diagnostics_json, evaluate, level1_validity, oracle_success, run_seed, v1_holds, world_data,
};
use hecs_r1::json::Json;
use hecs_r1::k1::K1;
use hecs_r1::spec;
use hecs_r1::world::spectral_centroid;

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

/// DESIGN: level-1 validity diagnostics, level-1 probes, spectral centroids, the oracle planner's success (V2)
/// and wall time on the DESIGN seeds at every grid budget. It trains no upper level and plans with no ECS: no
/// hypothesis quantity is computed.
fn design(args: &[String]) {
    let k1 = load(args);
    let started = Instant::now();
    let mut rows = Vec::new();
    for steps in spec::CALIBRATION_STEPS {
        let mut instances = Vec::new();
        for kind in spec::WORLDS {
            for seed in spec::DESIGN_SEEDS {
                let t = Instant::now();
                let data = world_data(kind, seed);
                let centroid = |f: fn(&hecs_r1::world::Truth) -> f64| {
                    data.train
                        .iter()
                        .map(|t| spectral_centroid(&t.truth.iter().map(f).collect::<Vec<_>>()))
                        .sum::<f64>()
                        / data.train.len() as f64
                };
                let ratio = centroid(|t| t.f) / centroid(|t| t.s);
                let (d, probes, refused) = level1_validity(&k1, &data, steps, seed, kind).expect("level 1");
                let oracle = oracle_success(&data, seed, kind);
                eprintln!(
                    "[hecs1] design steps {steps} {} {seed}: floor {:.4} linear {:.4} ecs {:.4} ({:.1}s)",
                    kind.name(),
                    d.floor,
                    d.linear,
                    d.ecs,
                    t.elapsed().as_secs_f64()
                );
                instances.push(Json::obj(vec![
                    ("world", Json::str(kind.name())),
                    ("seed", Json::str(seed.to_string())),
                    ("validity", diagnostics_json(&d)),
                    ("v1", Json::Bool(v1_holds(&d))),
                    ("l1_probe_nmse_slow_fast", Json::nums(&[probes.0, probes.1])),
                    ("centroid_ratio", Json::Num(ratio)),
                    ("oracle_success", Json::Num(oracle)),
                    ("refused", Json::Bool(refused)),
                    ("seconds", Json::Num(t.elapsed().as_secs_f64())),
                ]));
            }
        }
        rows.push(Json::obj(vec![("steps", Json::Int(steps as i64)), ("instances", Json::Arr(instances))]));
    }
    eprintln!("[hecs1] design done in {:.1}s", started.elapsed().as_secs_f64());
    write(
        arg(args, "--out"),
        &Json::obj(vec![
            ("spec", spec::to_json()),
            ("phase", Json::str("DESIGN")),
            ("note", Json::str("DESIGN seeds only; no hierarchy, planning or hypothesis quantity is computed")),
            ("grid", Json::Arr(rows)),
        ]),
    );
}

fn calibrate(args: &[String]) {
    let k1 = load(args);
    let started = Instant::now();
    let mut rows = Vec::new();
    let mut chosen = None;
    for steps in spec::CALIBRATION_STEPS {
        let mut instances = Vec::new();
        let mut all = true;
        for kind in spec::WORLDS {
            for seed in spec::DEV_SEEDS {
                let data = world_data(kind, seed);
                let (d, _, refused) = level1_validity(&k1, &data, steps, seed, kind).expect("level 1");
                let ok = !refused && v1_holds(&d);
                eprintln!(
                    "[hecs1] calibrate steps {steps} {} {seed}: floor {:.4} linear {:.4} ecs {:.4} -> {}",
                    kind.name(),
                    d.floor,
                    d.linear,
                    d.ecs,
                    if ok { "V1" } else { "not V1" }
                );
                all &= ok;
                instances.push(Json::obj(vec![
                    ("world", Json::str(kind.name())),
                    ("seed", Json::str(seed.to_string())),
                    ("validity", diagnostics_json(&d)),
                    ("refused", Json::Bool(refused)),
                    ("v1", Json::Bool(ok)),
                ]));
            }
        }
        rows.push(Json::obj(vec![
            ("steps", Json::Int(steps as i64)),
            ("instances", Json::Arr(instances)),
            ("v1_all", Json::Bool(all)),
        ]));
        if all {
            chosen = Some(steps);
            break;
        }
    }
    eprintln!("[hecs1] calibration done in {:.1}s", started.elapsed().as_secs_f64());
    write(
        arg(args, "--out"),
        &Json::obj(vec![
            ("spec", spec::to_json()),
            ("phase", Json::str("DEV_CALIBRATION")),
            ("rule", Json::str("smallest grid budget at which V1 (A, B and C) holds in every DEV world instance")),
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
            eprintln!("[hecs1] {} seed {seed} done in {:.1}s", kind.name(), t.elapsed().as_secs_f64());
            rs.push(r);
        }
        per_world.push(Json::obj(vec![
            ("world", Json::str(kind.name())),
            ("seeds", Json::Arr(rs.iter().map(|r| r.json.clone()).collect())),
        ]));
        results.push(rs);
    }
    let evaluation = evaluate(&results[0], &results[1]);
    eprintln!("[hecs1] {phase} done in {:.1}s", started.elapsed().as_secs_f64());
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
        Some("design") => design(&args),
        Some("calibrate") => calibrate(&args),
        Some("run") => run(&args),
        _ => {
            eprintln!(
                "usage: hecs1 spec | design --k1 LIB [--out F] | calibrate --k1 LIB [--out F] | \
                 run --phase dev|qual --steps N --k1 LIB --out F"
            );
            std::process::exit(2);
        }
    }
}
