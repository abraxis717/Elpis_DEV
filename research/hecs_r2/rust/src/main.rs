//! hecs2: the H-ECS R2 command line (research/hecs_r2/README.md).
//!
//!   hecs2 spec                                       the preregistered specification (JSON)
//!   hecs2 design --k1 LIB [--out FILE]               DESIGN seeds only: references, oracle, one-step adequacy
//!   hecs2 calibrate --k1 LIB [--out FILE]            DEV calibration of the training budget (margin rule)
//!   hecs2 run --phase dev|qual --steps N --k1 LIB --out FILE
//!
//! Derived from R1's `hecs1` (research/hecs_r1/rust/src/main.rs): `design` and `calibrate` apply R2's task
//! validity and calibration rule over every level (`experiment::design_instance`, `calibration_instance`);
//! `run` passes the phase to R2's law (DEV non-binding, QUAL binding).

use std::path::PathBuf;
use std::time::Instant;

use hecs_r2::experiment::{calibration_instance, design_instance, evaluate, run_seed};
use hecs_r2::json::Json;
use hecs_r2::k1::K1;
use hecs_r2::spec::{self, gates};

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

/// DESIGN: on the DESIGN seeds only, every level's references at every horizon (no ECS), T3, the oracle (V2),
/// probes (V3), centroids (V4) and, at every grid budget, T1 and every level's one-step adequacy (T2, the
/// calibration margin, V5). No ECS prediction beyond one step and no ECS planning: no multi-step or hypothesis
/// quantity is computed.
fn design(args: &[String]) {
    let k1 = load(args);
    let started = Instant::now();
    let mut instances = Vec::new();
    for kind in spec::WORLDS {
        for seed in spec::DESIGN_SEEDS {
            let t = Instant::now();
            let j = design_instance(&k1, kind, seed).unwrap_or_else(|e| panic!("{} seed {seed}: {e}", kind.name()));
            eprintln!("[hecs2] design {} {seed} done in {:.1}s", kind.name(), t.elapsed().as_secs_f64());
            instances.push(j);
        }
    }
    eprintln!("[hecs2] design done in {:.1}s", started.elapsed().as_secs_f64());
    write(
        arg(args, "--out"),
        &Json::obj(vec![
            ("spec", spec::to_json()),
            ("phase", Json::str("DESIGN")),
            (
                "note",
                Json::str(
                    "DESIGN seeds only: references (no ECS), oracle, probes, centroids; at every grid budget T1 and \
                     every level's one-step adequacy. No ECS prediction beyond one step, no ECS planning, no \
                     hypothesis quantity",
                ),
            ),
            ("instances", Json::Arr(instances)),
        ]),
    );
}

/// DEV calibration: B* is the smallest grid budget at which every DEV instance holds T1 and every level's
/// one-step capture >= CAL_MIN_CAPTURE; the chosen budget is CAL_GRID_STEPS_ABOVE grid values above B*.
fn calibrate(args: &[String]) {
    let k1 = load(args);
    let started = Instant::now();
    let mut rows = Vec::new();
    let mut smallest = None;
    for (gi, steps) in spec::CALIBRATION_STEPS.into_iter().enumerate() {
        let mut instances = Vec::new();
        let mut all = true;
        for kind in spec::WORLDS {
            for seed in spec::DEV_SEEDS {
                let (j, ok) = calibration_instance(&k1, kind, seed, steps)
                    .unwrap_or_else(|e| panic!("{} seed {seed}: {e}", kind.name()));
                eprintln!(
                    "[hecs2] calibrate steps {steps} {} {seed}: {}",
                    kind.name(),
                    if ok { "margin" } else { "no margin" }
                );
                all &= ok;
                instances.push(j);
            }
        }
        rows.push(Json::obj(vec![
            ("steps", Json::Int(steps as i64)),
            ("instances", Json::Arr(instances)),
            ("all_calibrated", Json::Bool(all)),
        ]));
        if all {
            smallest = Some(gi);
            break;
        }
    }
    let chosen = smallest.and_then(|i| spec::CALIBRATION_STEPS.get(i + gates::CAL_GRID_STEPS_ABOVE)).copied();
    eprintln!("[hecs2] calibration done in {:.1}s: chosen {chosen:?}", started.elapsed().as_secs_f64());
    write(
        arg(args, "--out"),
        &Json::obj(vec![
            ("spec", spec::to_json()),
            ("phase", Json::str("DEV_CALIBRATION")),
            (
                "rule",
                Json::str(
                    "B* = the smallest grid budget at which every DEV instance holds T1 and every level's one-step \
                     capture >= CAL_MIN_CAPTURE; chosen = the grid value CAL_GRID_STEPS_ABOVE above B*",
                ),
            ),
            ("grid", Json::Arr(rows)),
            ("smallest_margin_steps", smallest.map_or(Json::Null, |i| Json::Int(spec::CALIBRATION_STEPS[i] as i64))),
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
            eprintln!("[hecs2] {} seed {seed} done in {:.1}s", kind.name(), t.elapsed().as_secs_f64());
            rs.push(r);
        }
        per_world.push(Json::obj(vec![
            ("world", Json::str(kind.name())),
            ("seeds", Json::Arr(rs.iter().map(|r| r.json.clone()).collect())),
        ]));
        results.push(rs);
    }
    let evaluation = evaluate(&results[0], &results[1], phase == "qual");
    eprintln!("[hecs2] {phase} done in {:.1}s", started.elapsed().as_secs_f64());
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
                "usage: hecs2 spec | design --k1 LIB [--out F] | calibrate --k1 LIB [--out F] | \
                 run --phase dev|qual --steps N --k1 LIB --out F"
            );
            std::process::exit(2);
        }
    }
}
