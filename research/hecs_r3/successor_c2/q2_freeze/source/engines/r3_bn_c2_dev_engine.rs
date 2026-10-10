//! C2-Q2: the frozen C2 engine with only its QUAL registry constant, QUAL seal path and the DESIGN-mode
//! refusal of the spent Q1 QUAL registry changed (fresh Q2 QUAL execution of the unchanged C2 candidate).
//! H-ECS R3 successor C2 engine: the frozen R3 successor engine (C1 source, byte-for-byte outside the C2 diff)
//! with the HD/L2 ROLLOUT arm trained by the C2 law (true-target terminal rollout anchoring, convergent budget).
//! Historical header follows.
//! H-ECS R3: R2-equivalent S1/M DESIGN-only observer for the frozen R3-G candidate.
//! The R2/G1 reference is exact native K1 96,000-step training on same training sequences.
//! The additional R3 teacher-vs-rollout fine-tuning (both start from G1 W) matches only the declared work surrogate.
//! In particular, the R3 teacher optimizer is NOT the frozen R2/G1 learning algorithm.
//! R3-R adds native refusal-to-V5 mechanics on SYNTHETIC ONLY fixtures.
//! No DEV/QUAL seeds or scientific hierarchy-planner evaluations occur in this phase.
//! Reconstructs frozen R3-G training exactly, then additionally measures R2 multistep statistics.

#[path = "support/r3_training_core.rs"]
mod candidate;
#[path = "support/r3_bd_emitter.rs"]
mod r3_bd_emitter;
#[path = "support/r3_c2_terminal.rs"]
mod successor_c2;
mod r3_training_core { pub use super::candidate::*; }

use hecs_r2::analysis::horizon_error;
use hecs_r2::experiment::{level_sequences, world_data, LevelData, WorldData, frame, clip_flags, observation_level};
use hecs_r2::multistep;
use hecs_r2::validity::{linear_reference, constant_baseline};
use hecs_r2::hierarchy::LevelSequence;
use hecs_r2::k1::{K1, K1Error};
use hecs_r2::model::{LevelModel, Transition, DIM, LATENT};
use hecs_r2::rng::Rng;
use hecs_r2::spec;
use hecs_r2::world::WorldKind;
use std::fs::{OpenOptions, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use candidate::{paired_step, forward, objective_gradient, update, Objective, Window, Work};

const DESIGN: [u64; 4] = [
    3303124151261597934, 3300425290757990491,
    17908635065518423054, 11843937136941043377,
];
const BINDINGS: [(&str, usize); 4] = [
    ("L1/N72", 16), ("L1/N36", 4), ("TS/L2", 4), ("HD/L2", 4),
];
const G1_STEPS: u64 = 96_000;
const CANDIDATE_UPDATES: usize = 256;
const LAMBDA: f64 = 0.37;
const CLIP: f64 = 10.0;
const RATE_AT_N36: f64 = 0.0005;
const TAG_DESIGN_CANDIDATE: u64 = 0x52334443414e4431;

fn require<T, E: std::fmt::Display>(v: Result<T, E>, context: &str) -> T {
    v.unwrap_or_else(|err| panic!("{context}: {err}"))
}
fn key_tag(key: &str) -> u64 {
    key.bytes().fold(7u64, |x, b| x.wrapping_mul(131).wrapping_add(b as u64))
}
fn training_variance(data: &[LevelSequence]) -> f64 {
    let mut sum = [0.0; LATENT];
    let mut count = 0usize;
    for s in data {
        for z in &s.z {
            for (i, x) in z.iter().enumerate() { sum[i] += *x; }
            count += 1;
        }
    }
    assert!(count > 0);
    let mean = [sum[0] / count as f64, sum[1] / count as f64];
    let mut variance = 0.0;
    for s in data {
        for z in &s.z {
            for i in 0..LATENT { variance += (z[i] - mean[i]).powi(2); }
        }
    }
    variance /= (LATENT * count) as f64;
    assert!(variance.is_finite() && variance > 0.0, "invalid train-only variance");
    variance
}
fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = rng.below((i + 1) as u64) as usize;
        v.swap(i, j);
    }
}
fn training_windows(data: &[LevelSequence], h: usize, rng: &mut Rng) -> Vec<Window> {
    let mut result = Vec::new();
    for s in data {
        assert_eq!(s.z.len(), s.b.len() + 1);
        if s.b.len() < h { continue; }
        for start in 0..=s.b.len() - h {
            result.push(Window {
                z: s.z[start..=start+h].to_vec(),
                actions: s.b[start..start+h].to_vec(),
            });
        }
    }
    shuffle(&mut result, rng);
    assert!(result.len() >= 8, "WINDOW_POOL_TOO_SMALL");
    result
}
#[derive(Debug)]
enum RunFailure {
    Scientific(String),
    Mechanics(String),
}
fn k1_failure(err: K1Error, stage: &str) -> RunFailure {
    if err.0 == -2 { RunFailure::Scientific(format!("K1_NONFINITE stage={stage}")) }
    else { RunFailure::Mechanics(format!("K1_STATUS_{} stage={stage}",err.0)) }
}
fn scientific_optimizer_failure(err: String, stage: &str) -> RunFailure {
    match err.as_str() {
        "NUMERICAL_ESCAPE" | "NONFINITE_LOSS_OR_GRADIENT" | "NONFINITE_GRADIENT_NORM" |
        "NONFINITE_UPDATED_WEIGHTS" => RunFailure::Scientific(format!("{err} stage={stage}")),
        _ => RunFailure::Mechanics(format!("OPTIMIZER_CONTRACT_{err} stage={stage}")),
    }
}
fn status<T>(r: Result<T, RunFailure>, stage: &str) -> Option<T> {
    match r {
        Ok(v)=>Some(v),
        Err(RunFailure::Scientific(msg))=>{eprintln!("SCIENTIFIC_NONPASS stage={stage} cause={msg}");None},
        Err(RunFailure::Mechanics(msg))=>panic!("MECHANICS_NONQUAL stage={stage} cause={msg}"),
    }
}
fn native<T>(r: Result<T, K1Error>, stage: &str) -> Result<T,RunFailure> {
    r.map_err(|e| k1_failure(e,stage))
}
fn measure_simple(k1: &K1, w: &[f64], width: usize, test: &[LevelSequence], h: usize)
    -> Result<(f64,f64,f64,f64),RunFailure> {
    let mut native_model = native(k1.create(DIM,width,spec::MAX_ROWS,w),"evaluation create")?;
    let mut parity=0.0_f64;
    for seq in test.iter().take(2) {
        for t in 0..seq.b.len().min(4) {
            let mut input=Vec::with_capacity(DIM*LATENT);
            for k in 0..LATENT { input.extend_from_slice(&hecs_r2::model::row(&seq.z[t],seq.b[t],k)); }
            let mut output=[0.0;LATENT];
            native(native_model.forward(&input,&mut output),"parity forward")?;
            let oracle=forward(w,width,seq.z[t],seq.b[t]);
            for k in 0..LATENT { parity=parity.max((oracle[k]-output[k]).abs()); }
        }
    }
    if !(parity.is_finite() && parity<=2.0e-11) {
        return Err(RunFailure::Mechanics(format!("REAL_WORLD_NATIVE_PARITY_FAIL={parity}")));
    }
    drop(native_model);
    let mut model=LevelModel{ecs:native(k1.create(DIM,width,spec::MAX_ROWS,w),"horizon create")?,width,forward_rows:0};
    let seqs:Vec<_>=test.iter().map(|s|(s.z.clone(),s.b.clone())).collect();
    let (one, one_escape)=native(horizon_error(&mut model,&seqs,1),"one step horizon")?;
    let (consumed,escaped)=native(horizon_error(&mut model,&seqs,h),"consumed horizon")?;
    if !one.is_finite() || !consumed.is_finite() || !one_escape.is_finite() || !escaped.is_finite() {
        return Err(RunFailure::Scientific("NONFINITE_HELDOUT_METRIC".into()));
    }
    if one < 0.0 || consumed < 0.0 || !(0.0..=1.0).contains(&escaped) || !(0.0..=1.0).contains(&one_escape) {
        return Err(RunFailure::Mechanics("INVALID_HELDOUT_METRIC_DOMAIN".into()));
    }
    Ok((one,consumed,escaped,parity))
}
// R2 multistep arithmetic is unchanged; collecting rows BEFORE writing makes failures atomic.
fn measure_r2(k1:&K1,seed:u64,key:&str,consumed:usize,width:usize,w:&[f64],
    level:&LevelData,world:&WorldData,arm:&str) -> Result<Vec<String>,RunFailure> {
    let mut model=LevelModel{ecs:native(k1.create(DIM,width,spec::MAX_ROWS,w),"R2 observer create")?,width,forward_rows:0};
    let seqs:Vec<_>=level.test.iter().map(|s|(s.z.clone(),s.b.clone())).collect();
    let (one_nmse,one_escaped)=native(horizon_error(&mut model,&seqs,1),"R2 one-step S1")?;
    if one_nmse < 0.0 || !(0.0..=1.0).contains(&one_escaped) {return Err(RunFailure::Mechanics("INVALID_R2_S1_DOMAIN".into()));}
    let linear_one=linear_reference(&level.train,&level.test);
    let constant_one=constant_baseline(&level.train,&level.test);
    let capture_one=if constant_one>linear_one && one_nmse.is_finite() && one_escaped==0.0 {
        (constant_one-one_nmse)/(constant_one-linear_one)
    }else{f64::NEG_INFINITY};
    let s1=capture_one.is_finite() && capture_one>=spec::gates::S1_MIN_CAPTURE;
    let clips=observation_level(key).then(||clip_flags(world,level.env_stride,&level.test));
    let fr=frame(key,level.env_stride,world);
    let path=[seed,11,7,key_tag(key)];
    eprintln!("[r3-r2-observer] begin seed={seed} key={key} arm={arm} all R2 horizons 1,2,4,8,16");
    let horizons=native(multistep::horizons(&mut model,&level.train,&level.test,fr,
        &spec::HORIZONS,clips.as_deref(),&path),"R2 multistep")?;
    let mut rows=Vec::new();
    for r in horizons {
        if r.ecs_free < 0.0 || r.linear < 0.0 || r.constant < 0.0 || !(0.0..=1.0).contains(&r.ecs_escaped) {
            return Err(RunFailure::Mechanics("INVALID_R2_HORIZON_DOMAIN".into()));
        }
        let binding=r.h<=consumed;
        let m_at_h=r.capture.is_finite() && r.capture>=spec::gates::M_MIN_CAPTURE
            && r.ecs_escaped<=spec::gates::M_MAX_ESCAPED;
        let fields=[seed.to_string(),key.to_string(),arm.to_string(),consumed.to_string(),r.h.to_string(),
            (binding as u8).to_string(),format!("{one_nmse:.17e}"),format!("{linear_one:.17e}"),
            format!("{constant_one:.17e}"),format!("{capture_one:.17e}"),(s1 as u8).to_string(),
            format!("{:.17e}",r.ecs_free),format!("{:.17e}",r.linear),format!("{:.17e}",r.constant),
            format!("{:.17e}",r.capture),format!("{:.17e}",r.ecs_escaped),
            format!("{:.17e}",r.candidate_escaped),(m_at_h as u8).to_string()];
        rows.push(fields.join("\t"));
    }
    if rows.len()!=5{return Err(RunFailure::Mechanics("INCOMPLETE_R2_METRIC_GROUP".into()));}
    eprintln!("[r3-r2-observer] complete seed={seed} key={key} arm={arm} S1_capture={capture_one:.6}");
    Ok(rows)
}
fn failure_rows(seed:u64,key:&str,consumed:usize,level:&LevelData,world:&WorldData,arm:&str) -> Vec<String> {
    let clips=observation_level(key).then(||clip_flags(world,level.env_stride,&level.test));
    let refs=multistep::references(&level.train,&level.test,frame(key,level.env_stride,world),
        &spec::HORIZONS,clips.as_deref());
    let linear_one=linear_reference(&level.train,&level.test);
    let constant_one=constant_baseline(&level.train,&level.test);
    refs.iter().map(|r|[
        seed.to_string(),key.into(),arm.into(),consumed.to_string(),r.h.to_string(),
        ((r.h<=consumed) as u8).to_string(),"inf".into(),format!("{linear_one:.17e}"),
        format!("{constant_one:.17e}"),"-inf".into(),"0".into(),
        "nan".into(),format!("{:.17e}",r.linear),format!("{:.17e}",r.constant),
        "nan".into(),"1.00000000000000000e0".into(),"1.00000000000000000e0".into(),"0".into()
    ].join("\t")).collect()
}
fn write_event(f:&mut BufWriter<File>,seed:u64,key:&str,arm:&str,stage:&str,cause:&str,synthetic:bool) {
    assert!(!cause.contains('\t') && !cause.contains('\n'),"BAD_EVENT_CAUSE");
    writeln!(f,"{seed}\t{key}\t{arm}\t{stage}\tSCIENTIFIC_NONPASS\t{cause}\t{}",synthetic as u8).expect("write failure event");
    f.flush().expect("flush event");
}
fn handle_arm(k1:&K1,seed:u64,key:&str,h:usize,width:usize,w:&[f64],
    level:&LevelData,world:&WorldData,arm:&str,r2_out:&mut BufWriter<File>,events:&mut BufWriter<File>,
    synthetic_failure:Option<&str>) -> Option<(f64,f64,f64,f64)> {
    let measured=if let Some(fault)=synthetic_failure {
        Err(RunFailure::Scientific(format!("SYNTHETIC_INJECTION_{fault}")))
    } else {
        measure_simple(k1,w,width,&level.test,h).and_then(|numbers| {
            let rows=measure_r2(k1,seed,key,h,width,w,level,world,arm)?;
            Ok((numbers,rows))
        })
    };
    match measured {
        Ok((numbers,rows))=>{
            for row in rows { writeln!(r2_out,"{row}").expect("metric write"); }
            r2_out.flush().expect("metric flush");
            Some(numbers)
        }
        Err(RunFailure::Scientific(cause))=>{
            write_event(events,seed,key,arm,"MEASURE",&cause,synthetic_failure.is_some());
            for row in failure_rows(seed,key,h,level,world,arm) {
                writeln!(r2_out,"{row}").expect("failure metric write");
            }
            r2_out.flush().expect("failure metric flush");
            None
        }
        Err(RunFailure::Mechanics(cause))=>panic!("MECHANICS_NONQUAL cause={cause}"),
    }
}
fn run_binding(k1:&K1,seed:u64,data:LevelData,world:&WorldData,h:usize,
    r2_out:&mut BufWriter<File>,events:&mut BufWriter<File>,fault:Option<&str>,
    binding_native_refusals:&mut Vec<String>, successor:bool) -> Vec<String> {
    let width=data.width;
    let key=data.key.clone();
    let mut rng=Rng::new(spec::BASE_SEED,&[seed,11,2,key_tag(&key)]);
    let mut native_g1=require(LevelModel::new(k1,width,spec::MAX_ROWS,
        spec::train_spec(G1_STEPS).init_scale,&mut rng),"R2/G1 initialize");
    let mut train:Vec<Transition>=data.train.iter().flat_map(|s|s.transitions()).collect();
    assert!(train.iter().all(|t|t.z.iter().chain(t.next.iter()).all(|v|v.is_finite()) && t.b.is_finite()),
        "MECHANICS_NONQUAL_NONFINITE_WORLD_INPUT");
    shuffle(&mut train,&mut rng);
    // MECHANICS ONLY: an invalid synthetic TARGET causes a REAL K1 LEARN -2 refusal.
    // This branch is unreachable in released DEV/QUAL modes; it never changes the
    // registered world generator or the normal native training law.
    let native_fault=fault==Some("NATIVE_G1");
    if native_fault {
        assert_eq!(key,"HD/L2","NATIVE_INJECTION_TARGET_MUST_BE_HD_L2");
        assert!(!train.is_empty(),"NATIVE_INJECTION_EMPTY_TRAIN");
        train[0].next[0]=f64::NAN;
    }
    let trace=native_g1.train(&train,&spec::train_spec(G1_STEPS));
    let injected_g1=fault==Some("G1");
    let g1_failed=trace.refused.is_some() || injected_g1;
    if let Some(refused)=&trace.refused {
        if !refused.contains("NONFINITE (-2)") {panic!("MECHANICS_NONQUAL_UNRECOGNIZED_NATIVE_G1_REFUSAL {refused}");}
        // Unlike a post-hoc G1 flag, this is grounded in the native LEARN refusal.
        assert!(!binding_native_refusals.contains(&key),"DUPLICATE_BINDING_NATIVE_REFUSAL");
        binding_native_refusals.push(key.clone());
        if native_fault {println!("PASS_R3_R_SYNTHETIC_STIMULUS_REAL_NATIVE_K1_NONFINITE seed={seed} key={key} refusal={refused}");}
    } else if native_fault {
        panic!("MECHANICS_NONQUAL_NATIVE_K1_INJECTION_DID_NOT_REFUSE");
    }
    let variance=training_variance(&data.train);
    if g1_failed {
        let cause=if injected_g1{"SYNTHETIC_INJECTION_G1".to_string()}
                  else if native_fault {format!("SYNTHETIC_INJECTION_NATIVE_G1_K1_NONFINITE {}",trace.refused.as_deref().unwrap_or(""))}
                  else{format!("G1_NATIVE_NONFINITE {}",trace.refused.as_deref().unwrap_or(""))};
        for arm in ["G1","TEACHER","ROLLOUT"] {
            write_event(events,seed,&key,arm,"G1_TRAIN_DEPENDENCY",&cause,injected_g1 || native_fault);
            for row in failure_rows(seed,&key,h,&data,world,arm) {
                writeln!(r2_out,"{row}").expect("failure metric write");
            }
        }
        r2_out.flush().expect("failure metric flush");
        return failure_observation(seed,&key,h,width,variance,trace.rows_learned,trace.multiply_adds);
    }
    assert_eq!(native_g1.ecs.epoch(),G1_STEPS,"G1_TRAIN_EPOCH_MISMATCH");
    let w_g1=require(native_g1.ecs.w(),"copy G1 W");
    drop(native_g1);
    let mut windows_rng=Rng::new(spec::BASE_SEED,&[seed,11,TAG_DESIGN_CANDIDATE,key_tag(&key)]);
    let windows=training_windows(&data.train,h,&mut windows_rng);
    let rate=RATE_AT_N36*36.0/width as f64;
    let (mut teacher,mut rollout)=(w_g1.clone(),w_g1.clone());
    let (mut work_teacher,mut work_rollout)=(Work::default(),Work::default());
    let mut optimizer_failed=None::<String>;
    let mut c2_failed=None::<String>; // a C2 ROLLOUT-arm failure; the TEACHER arm stays complete and measured
    for step in 0..CANDIDATE_UPDATES {
        // C2 is the successor candidate arm at HD/L2 only; historical modes use the original path.
        let step_result = if successor {
            // TEACHER keeps its historical trajectory: exactly the teacher half of paired_step (the same seven
            // picks windows[(7s+j)%n], objective_gradient(Teacher), update). No rollout step is evaluated here;
            // the C2 ROLLOUT arm is trained separately below, from the same G1 W.
            let n=windows.len();
            let picks:Vec<Window>=(0..7).map(|j|windows[(step*7+j)%n].clone()).collect();
            match objective_gradient(&teacher,width,&picks,h,variance,LAMBDA,Objective::Teacher) {
                Err(e)=>Err(e),
                Ok((lt,grad,wa))=>update(&mut teacher,&grad,rate,CLIP).map(|_|(lt,f64::NAN,wa,Work::default())),
            }
        } else {paired_step(&mut teacher,&mut rollout,width,&windows,h,variance,LAMBDA,rate,CLIP,step)};
        match step_result {
            Ok((_,_,wa,wb))=>{work_teacher.accumulate(wa);work_rollout.accumulate(wb);}
            Err(err)=>{
                match scientific_optimizer_failure(err,"paired_step") {
                    RunFailure::Scientific(msg)=>{optimizer_failed=Some(msg);break;}
                    RunFailure::Mechanics(msg)=>panic!("MECHANICS_NONQUAL {msg}"),
                }
            }
        }
        if (step+1)%64==0 {
            eprintln!("[r3-design-candidate] seed={seed} key={key} update={}/{} surrogate_teacher={} surrogate_rollout={}",
                step+1,CANDIDATE_UPDATES,work_teacher.counted_multiplies,work_rollout.counted_multiplies);
        }
    }
    if optimizer_failed.is_none() && !successor {assert_eq!(work_teacher.counted_multiplies,work_rollout.counted_multiplies,
        "SURROGATE_WORK_NOT_MATCHED");}
    if successor && optimizer_failed.is_none() {
        // C2 law (frozen): from G1 W, C2_UPDATES updates; update s uses the C0 rollout picks
        // windows[(s*7+j)%n], j<6; loss = R3 rollout objective (LAMBDA) + C2_TERMINAL_WEIGHT * true-target terminal
        // loss at step h; exact reverse-mode gradient; unchanged rate law and CLIP. Train-only, deterministic.
        let n=windows.len();
        for step in 0..C2_UPDATES {
            let batch:Vec<Window>=(0..C2_WINDOWS).map(|j|windows[(step*7+j)%n].clone()).collect();
            match successor_c2::c2_step(&mut rollout,width,&batch,h,variance,LAMBDA,C2_TERMINAL_WEIGHT,rate,CLIP) {
                Ok((_,base_work,mut terminal_work,_))=>{
                    work_rollout.accumulate(base_work);
                    terminal_work.samples=0; // the terminal pass revisits the same windows
                    work_rollout.accumulate(terminal_work);
                }
                Err(err)=>{
                    match scientific_optimizer_failure(err,"c2_step") {
                        RunFailure::Scientific(msg)=>{c2_failed=Some(msg);break;}
                        RunFailure::Mechanics(msg)=>panic!("MECHANICS_NONQUAL {msg}"),
                    }
                }
            }
            if (step+1)%512==0 {
                eprintln!("[r3-c2-candidate] seed={seed} key={key} update={}/{} surrogate_c2={}",
                    step+1,C2_UPDATES,work_rollout.counted_multiplies);
            }
        }
        println!("R3_C2_SURROGATE teacher={} c2={} UNMATCHED=YES EXACT_FLOPS_MATCHED=NO",work_teacher.counted_multiplies,work_rollout.counted_multiplies);
    }
    if fault==Some("PAIRED") {
        let (mut huge_t,mut huge_r)=(vec![1.0e75;DIM*width],vec![1.0e75;DIM*width]);
        let err=paired_step(&mut huge_t,&mut huge_r,width,&windows,h,variance,
            LAMBDA,rate,CLIP,0).expect_err("INJECTION_EXPECTED_ESCAPED_OPTIMIZER");
        optimizer_failed=Some(match scientific_optimizer_failure(err,"synthetic paired_step") {
            RunFailure::Scientific(msg)=>format!("SYNTHETIC_INJECTION_{msg}"),
            RunFailure::Mechanics(msg)=>panic!("MECHANICS_NONQUAL_FAULT_CLASSIFIER {msg}"),
        });
    }
    // Once a joint paired_step fails, neither additional arm has completed its frozen training law.
    // A completed G1 arm remains independently measured and untouched. (C2: a C2_OPTIMIZER failure fails only the
    // ROLLOUT arm; the completed TEACHER arm is measured as usual.)
    let g1=handle_arm(k1,seed,&key,h,width,&w_g1,&data,world,"G1",r2_out,events,None);
    let optimizer_bad=optimizer_failed.is_some();
    let teacher_result=if optimizer_bad { None } else {
        handle_arm(k1,seed,&key,h,width,&teacher,&data,world,"TEACHER",r2_out,events,None)
    };
    let rollout_result=if optimizer_bad || c2_failed.is_some() { None } else {
        handle_arm(k1,seed,&key,h,width,&rollout,&data,world,"ROLLOUT",r2_out,events,
            if fault==Some("ROLLOUT"){Some("ROLLOUT")}else{None})
    };
    if let Some(cause)=c2_failed {
        write_event(events,seed,&key,"ROLLOUT","C2_OPTIMIZER",&cause,false);
        for row in failure_rows(seed,&key,h,&data,world,"ROLLOUT") {
            writeln!(r2_out,"{row}").expect("C2 optimizer failed metric write");
        }
        r2_out.flush().expect("C2 optimizer failed metric flush");
    }
    if let Some(cause)=optimizer_failed {
        for arm in ["TEACHER","ROLLOUT"] {
            write_event(events,seed,&key,arm,"PAIRED_OPTIMIZER",&cause,fault==Some("PAIRED"));
            for row in failure_rows(seed,&key,h,&data,world,arm) {
                writeln!(r2_out,"{row}").expect("optimizer failed metric write");
            }
        }
        r2_out.flush().expect("optimizer failed metric flush");
    }
    let unpack=|v:Option<(f64,f64,f64,f64)>| v.unwrap_or((f64::INFINITY,f64::NAN,1.0,f64::NAN));
    let (g1_one,g1_h,g1_escape,g1_parity)=unpack(g1);
    let (teacher_one,teacher_h,teacher_escape,teacher_parity)=unpack(teacher_result);
    let (rollout_one,rollout_h,rollout_escape,rollout_parity)=unpack(rollout_result);
    let delta=rollout_h-teacher_h;
    println!("ROW seed={seed} key={key} h={h} G1_h={g1_h:.8e} teacher_h={teacher_h:.8e} rollout_h={rollout_h:.8e} rollout_minus_teacher={delta:+.8e} G1_work={} matched_surrogate={} exact_flops_matched=NO",
        trace.multiply_adds,work_teacher.counted_multiplies);
    vec![seed.to_string(),key,h.to_string(),width.to_string(),format!("{variance:.17e}"),
        format!("{g1_one:.17e}"),format!("{g1_h:.17e}"),format!("{g1_escape:.17e}"),
        format!("{teacher_one:.17e}"),format!("{teacher_h:.17e}"),format!("{teacher_escape:.17e}"),
        format!("{rollout_one:.17e}"),format!("{rollout_h:.17e}"),format!("{rollout_escape:.17e}"),
        format!("{delta:.17e}"),trace.rows_learned.to_string(),trace.multiply_adds.to_string(),
        work_teacher.samples.to_string(),work_rollout.samples.to_string(),
        work_teacher.counted_multiplies.to_string(),work_rollout.counted_multiplies.to_string(),
        work_teacher.forward_rows.to_string(),work_teacher.d_weight_rows.to_string(),work_teacher.state_adjoint_rows.to_string(),
        work_rollout.forward_rows.to_string(),work_rollout.d_weight_rows.to_string(),work_rollout.state_adjoint_rows.to_string(),
        format!("{g1_parity:.17e}"),format!("{teacher_parity:.17e}"),format!("{rollout_parity:.17e}")]
}
fn failure_observation(seed:u64,key:&str,h:usize,width:usize,variance:f64,rows:u64,multiplies:u64)->Vec<String>{
    let mut v=vec![seed.to_string(),key.into(),h.to_string(),width.to_string(),format!("{variance:.17e}")];
    for _ in 0..3{v.extend(["inf".into(),"nan".into(),"1.00000000000000000e0".into()]);}
    v.push("nan".into());v.push(rows.to_string());v.push(multiplies.to_string());
    for _ in 0..10{v.push("0".into());}
    for _ in 0..3{v.push("nan".into());}
    assert_eq!(v.len(),30,"FAILURE_OBSERVATION_SCHEMA");v
}
// R3-L: four synthetic seed worlds exercise the complete prospective DEV census.
// The executable deliberately has NO DEV/QUAL mode and must never be used on registered seeds.
const SYNTHETIC_FULL_CENSUS: [u64;4] = [571334596706648308, 17629385886378724003, 13540456215938820495, 3963545577634352514];

fn task_row_full(k1: &K1, seed: u64, kind: WorldKind, v5: &mut BufWriter<File>,
    separated_binding_native_refusals: &[String]) -> Vec<String> {
    let world = world_data(kind, seed);
    let reference = hecs_r2::experiment::references(&world, seed, kind);
    let levels = level_sequences(&world);
    let l1 = levels.iter().find(|x| x.key == "L1/N36").expect("L1 data");
    let linear = linear_reference(&l1.train, &l1.test);
    let constant = constant_baseline(&l1.train, &l1.test);
    let floor = hecs_r2::validity::analytic_floor(&world.world, &world.enc1, &l1.test);
    let mut min_explained = f64::INFINITY;
    let mut max_l_over_g = f64::NEG_INFINITY;
    for (key, h) in BINDINGS {
        let rf = reference.refs.get(key).expect("R2 binding reference");
        let r = rf.iter().find(|r| r.h == h).expect("consumed R2 reference");
        min_explained = min_explained.min(r.explained());
        if hecs_r2::experiment::observation_level(key) {
            let g = r.generator.expect("observation generator reference");
            assert!(g > 0.0, "invalid generator reference");
            max_l_over_g = max_l_over_g.max(r.linear / g);
        }
    }
    assert!(min_explained.is_finite() && max_l_over_g.is_finite(), "missing R2 T3 reference");
    // Full 96,000-step native G1 budget for every R2 level on each of the two world kinds.
    // This proves the executor exercises V5's COMPLETE level/budget census on synthetic data.
    // The returned V5 bit is an observed outcome, never a forced pass.
    let pool = require(hecs_r2::experiment::build_pool(k1, &world, G1_STEPS, seed, kind),
        "synthetic full-budget R2 V5 pool");
    assert_eq!(pool.len(), 7, "R2 pool seven-level census");
    let mut no_refusal = true;
    for entry in &pool {
        let epoch = entry.model.as_ref().map(|m|m.ecs.epoch()).unwrap_or(0);
        let independent_pool_good = entry.trace.refused.is_none() && epoch == G1_STEPS;
        let binding_g1_refused = kind==WorldKind::Separated &&
            separated_binding_native_refusals.contains(&entry.key);
        // "refused" records INDEPENDENT pool training only; do not forge that outcome.
        // "full_budget_no_refusal" is the joint V5 obligation across the independent
        // R2 pool and the separately trained, planning-bound R3 G1 constituent.
        let full_budget_no_refusal = independent_pool_good && !binding_g1_refused;
        no_refusal &= full_budget_no_refusal;
        writeln!(v5,"{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            seed,kind.name(),entry.key,entry.model.as_ref().map(|m|m.width).unwrap_or(0),
            G1_STEPS,epoch,entry.trace.rows_learned,
            if entry.trace.refused.is_none(){"0"}else{"1"},
            if full_budget_no_refusal{"1"}else{"0"})
            .expect("write independent V5 level census");
    }
    v5.flush().expect("flush V5");
    println!("PASS_R3_L_V5_FULL_BUDGET_CENSUS seed={seed} world={} levels={} no_refusal={}",
        kind.name(),pool.len(),if no_refusal{1}else{0});
    vec![seed.to_string(),kind.name().to_string(),
        format!("{linear:.17e}"),format!("{constant:.17e}"),format!("{floor:.17e}"),
        format!("{min_explained:.17e}"),format!("{max_l_over_g:.17e}"),
        format!("{:.17e}",reference.oracle),format!("{:.17e}",reference.l1_probe.0),
        format!("{:.17e}",reference.l1_probe.1),format!("{:.17e}",reference.centroid_ratio),
        (no_refusal as u8).to_string()]
}


// R3-O: fixed preregistered science splits; neither is executed in R3-O.
// Prospective successor DEV cohort from accepted R3-AM deterministic registry.
// C2: fresh DEV registry (never executed; disjoint from every H-ECS registry, the burned successor DEV cohort
// and all C2 DESIGN worlds) and the reserved, never-executed successor QUAL registry. The C2 law changes only
// the HD/L2 ROLLOUT arm; every other binding, the TEACHER/G1 arms and all task rows are the C1/C0 code paths.
const BIN_CANDIDATE: &str = "C2";
const SUCCESSOR_DEV_SEEDS: [u64;8] = [684004886576317082,1703137016795210571,10088827697117853828,8960933697477120594,134760612940272399,7871577751992445249,17179523806848842400,13174565413729237818];
const SUCCESSOR_QUAL_SEEDS: [u64;12] = [614105602224108045,13431973024848494695,8015593928169052856,4368982056838705008,12192592809587942193,97214863508322549,6955956690753739111,18105131785740719551,17684696095471157875,11665294448906309688,16885114571500087964,12577501025278960659];
// Spent Q1 QUAL registry (infrastructure-interrupted, retired): refused in every mode.
const SPENT_QUAL_SEEDS: [u64;12] = [10405860717610874745,17760219202278299362,14564880931009575015,15447012595138726224,8070250311440129494,7015268779680225457,12995885642664273032,4683001325509031006,5700582020531935958,15439875992717053813,2654106804044592861,1474235648324239766];
const C2_UPDATES: usize = 4096;
const C2_TERMINAL_WEIGHT: f64 = 1.0;
const C2_WINDOWS: usize = 6;
const SUCCESSOR_DEV_SEAL: &str = "/mnt/primesauce/Elpis_DEV/evidence/hecs_r3_successor_c2/release/DEV_AUTHORITY.txt";
const SUCCESSOR_QUAL_SEAL: &str = "/mnt/primesauce/Elpis_DEV/evidence/hecs_r3_successor_c2/release/QUAL2_AUTHORITY.txt";
fn require_successor_seal(phase:&str, path:&str) {
    let contents=std::fs::read_to_string(path)
        .unwrap_or_else(|_|panic!("SUCCESSOR_C2_{phase}_NOT_RELEASED"));
    let lines:Vec<_>=contents.lines().collect();
    assert!(lines.len()>=5 && lines[0]=="HECS_R3_SUCCESSOR_C2_RELEASE_V1" &&
        lines.iter().any(|x|*x==format!("PHASE={phase}")) &&
        lines.iter().any(|x|*x=="CANDIDATES=C2") &&
        lines.iter().any(|x|x.starts_with("ENGINE_BINARY_SHA256_C2=")) &&
        lines.iter().any(|x|x.starts_with("PROTOCOL_SHA256=")),
        "INVALID_SUCCESSOR_C2_RELEASE_SEAL");
    // Structural seal is not self-authorization: the external C2 supervisor authenticates the release digest,
    // sources, binaries and split before invocation.
}
// DESIGN mode refuses every registered DEV/QUAL world of every H-ECS R3 line.
fn design_seed_allowed(seed:u64)->bool {
    !SUCCESSOR_DEV_SEEDS.contains(&seed) && !SUCCESSOR_QUAL_SEEDS.contains(&seed) && !SPENT_QUAL_SEEDS.contains(&seed) &&
    !R3_DEV_SEEDS.contains(&seed) && !R3_QUAL_SEEDS.contains(&seed)
}
const R3_DEV_SEEDS: [u64;4] = [15078392486590401649,17287511265161880591,1765155105912321725,8032847830432719464];
const R3_QUAL_SEEDS: [u64;8] = [15687576931411260316,16012458119835386863,1709660507283341147,4632183407265633503,11290532906745579263,17062616526593206115,10414294774421817650,14398233175459340262];
const DEV_SEAL: &str = "/mnt/primesauce/Elpis_DEV/evidence/hecs_r3/release/R3_P_DEV_FREEZE_V1/DEV_AUTHORITY.txt";
const QUAL_SEAL: &str = "/mnt/primesauce/Elpis_DEV/evidence/hecs_r3/release/R3_Q_QUAL_AUTHORIZATION_V1/QUAL_AUTHORITY.txt";

// This checks a structurally frozen release entitlement, not an authorization by itself.
// An independent external supervisor MUST verify the complete file, source and binary
// hashes and one-shot evidence protocol before invoking a scientific mode.
fn require_release_seal(phase: &str, seal_path: &str) {
    let content=std::fs::read_to_string(seal_path)
        .unwrap_or_else(|_|panic!("RELEASE_AUTHORITY_MISSING phase={phase}"));
    let lines:Vec<_>=content.lines().collect();
    assert!(lines.len()>=4 && lines[0]=="HECS_R3_FROZEN_PHASE_AUTHORITY_V1",
        "INVALID_RELEASE_AUTHORITY_FORMAT");
    assert!(lines.iter().any(|v| *v==format!("PHASE={phase}")),
        "INVALID_RELEASE_PHASE");
    assert!(lines.iter().any(|v| v.starts_with("ENGINE_BINARY_SHA256=")),
        "MISSING_BINARY_AUTHORITY");
    assert!(lines.iter().any(|v| v.starts_with("PROTOCOL_SHA256=")),
        "MISSING_PROTOCOL_AUTHORITY");
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len()==3 && args[1]=="--mechanics-classifier-selftest" {
        let k1=K1::load(Path::new(&args[2])).expect("synthetic classifier K1");
        let w=vec![0.0;DIM*36];
        let mut model=k1.create(DIM,36,spec::MAX_ROWS,&w).expect("synthetic native create");
        let err=model.learn(&vec![0.0;DIM],&[0.0],-1.0,1).expect_err("invalid negative rate must refuse");
        assert_eq!(err.0,-1,"expected native INVALID");
        assert!(matches!(k1_failure(err,"synthetic invalid rate"),RunFailure::Mechanics(_)));
        let mut y=[0.0];
        let mut x=vec![0.0;DIM];x[0]=f64::NAN;
        let err=model.forward(&x,&mut y).expect_err("nonfinite input must refuse");
        assert_eq!(err.0,-2,"expected native NONFINITE");
        assert!(matches!(k1_failure(err,"synthetic nonfinite input"),RunFailure::Scientific(_)));
        for str_err in ["NUMERICAL_ESCAPE","NONFINITE_LOSS_OR_GRADIENT","NONFINITE_GRADIENT_NORM","NONFINITE_UPDATED_WEIGHTS"] {
            assert!(matches!(scientific_optimizer_failure(str_err.into(),"synthetic"),RunFailure::Scientific(_)));
        }
        for str_err in ["INVALID_PARAMETER_OR_NORMALIZER","INVALID_SHAPE_OR_HORIZON","NONFINITE_WINDOW"] {
            assert!(matches!(scientific_optimizer_failure(str_err.into(),"synthetic"),RunFailure::Mechanics(_)));
        }
        println!("PASS_R3_N_TYPED_NATIVE_AND_OPTIMIZER_FAILURE_CLASSIFIER SCIENCE=NO");
        return;
    }
    if args.len()==2 && (args[1]=="--preflight-dev" || args[1]=="--preflight-qual") {
        let p=if args[1]=="--preflight-dev" {"DEV"} else {"QUAL"};
        let seeds=if p=="DEV" {R3_DEV_SEEDS.len()} else {R3_QUAL_SEEDS.len()};
        println!("PASS_R3_O_RELEASE_PHASE_PREVIEW phase={p} seed_count={seeds} SCIENTIFIC_EXECUTION=NO");
        return;
    }
    if args.len()==2 && args[1]=="--successor-dev-preview" {
        println!("PASS_R3_C2_PROSPECTIVE_ENGINE_PREVIEW candidate=C2 dev_seed_count=8 qual_seed_count=12");
        println!("R3_C2_DEV_SEEDS={:?}",SUCCESSOR_DEV_SEEDS);
        println!("R3_C2_UPDATES={} C2_TERMINAL_WEIGHT={} C2_WINDOWS={} C2_BASE_LAMBDA={} TEACHER_LEGACY_PAIRING=YES",C2_UPDATES,C2_TERMINAL_WEIGHT,C2_WINDOWS,LAMBDA);
        println!("R3_C2_COMPUTE_MATCHED=NO EXACT_FLOPS_MATCHED=NO");
        println!("SCIENTIFIC_EXECUTION=NO DEV_RELEASE=NO");
        return;
    }
    let mode=args.get(1).map(String::as_str).unwrap_or("");
    let is_successor_dev=mode=="--successor-dev-c2-one-seed";
    let is_successor_qual=mode=="--successor-qual-c2-one-seed";
    let is_design=mode=="--design-c2-one-seed";
    let is_successor=is_successor_dev || is_successor_qual || is_design;
    let is_single=matches!(mode,"--successor-dev-c2-one-seed"|"--successor-qual-c2-one-seed"|"--design-c2-one-seed"|"--mechanics-one-seed"|"--mechanics-successor-c2-one-seed"|"--mechanics-native-g1-refusal-one-seed"|"--frozen-dev-one-seed"|"--frozen-qual-one-seed");
    let is_mechanics=matches!(mode,"--mechanics-full-census"|"--mechanics-inject-g1"|
        "--mechanics-inject-rollout"|"--mechanics-inject-paired"|"--mechanics-one-seed"|"--mechanics-successor-c2-one-seed"|"--mechanics-native-g1-refusal-one-seed");
    let is_dev=matches!(mode,"--frozen-dev"|"--frozen-dev-one-seed");
    let is_qual=matches!(mode,"--frozen-qual"|"--frozen-qual-one-seed");
    assert!((is_single && args.len()==9 || !is_single && args.len()==8) && (is_mechanics || is_dev || is_qual || is_successor),
        "R3_P_REQUIRE_EXACT_MODE_OUTPUTS_AND_SINGLE_INDEX");
    if is_dev { require_release_seal("DEV",DEV_SEAL); }
    if is_qual { require_release_seal("QUAL",QUAL_SEAL); }
    if is_successor_dev { require_successor_seal("DEV",SUCCESSOR_DEV_SEAL); }
    if is_successor_qual { require_successor_seal("QUAL",SUCCESSOR_QUAL_SEAL); }
    if mode=="--mechanics-successor-c2-one-seed" {println!("R3_C2_SUCCESSOR_MECHANICS_ONLY; C2_TREATED_AS_ROLLOUT_COLUMN_FOR_SCHEMA_TEST; NOT_R3_SCIENCE");}
    let design_seed:[u64;1]=if is_design {
        let s=args[8].parse::<u64>().expect("INVALID_DESIGN_SEED");
        assert!(design_seed_allowed(s),"DESIGN_SEED_IS_A_REGISTERED_DEV_OR_QUAL_WORLD");
        [s]
    } else {[0]};
    let seeds:&[u64]=if is_successor_dev {&SUCCESSOR_DEV_SEEDS} else if is_successor_qual {&SUCCESSOR_QUAL_SEEDS}
        else if is_design {&design_seed} else if is_dev {&R3_DEV_SEEDS} else if is_qual {&R3_QUAL_SEEDS} else {&SYNTHETIC_FULL_CENSUS};
    let selected:Option<usize>=if is_design {Some(0)} else if is_single {
        let i=args[8].parse::<usize>().expect("INVALID_SEED_INDEX");
        assert!(i<seeds.len(),"SEED_INDEX_OUT_OF_RANGE");
        Some(i)
    } else {None};
    assert!(!(is_dev || is_qual || is_successor) || args[3..8].iter().all(|s| Path::new(s).is_absolute()),
        "RELEASE_OUTPUTS_MUST_BE_ABSOLUTE");
    let k1=K1::load(Path::new(&args[2])).expect("qualified K1 loading");
    assert_eq!(k1.abi_version(),1,"K1 ABI");
    let obsfile=require(OpenOptions::new().write(true).create_new(true).open(&args[3]),"exclusive synthetic observations");
    let metfile=require(OpenOptions::new().write(true).create_new(true).open(&args[4]),"exclusive synthetic metrics");
    let taskfile=require(OpenOptions::new().write(true).create_new(true).open(&args[5]),"exclusive synthetic tasks");
    let v5file=require(OpenOptions::new().write(true).create_new(true).open(&args[6]),"exclusive synthetic V5 census");
    let mut obs=BufWriter::new(obsfile);
    let mut metrics=BufWriter::new(metfile);
    let mut task=BufWriter::new(taskfile);
    let mut v5=BufWriter::new(v5file);
    let eventfile=require(OpenOptions::new().write(true).create_new(true).open(&args[7]),"exclusive synthetic failure events");
    let mut events=BufWriter::new(eventfile);
    writeln!(events,"seed\tkey\tarm\tstage\tstatus\tcause\tsynthetic_injection").unwrap();events.flush().unwrap();
    writeln!(metrics,"seed\tkey\tarm\tconsumed_h\th\tbinding\tS1_ecs_one\tS1_linear_one\tS1_constant_one\tS1_capture\tS1_ge_095\tM_ecs_free_nmse\tM_linear_nmse\tM_constant_nmse\tM_capture\tM_escape\tM_planner_candidate_escape\tM_at_h_ge_090_escape_le_001").unwrap();
    writeln!(obs,"seed\tkey\thorizon\twidth\ttrain_variance\tG1_one_nmse\tG1_h_nmse\tG1_h_escape\tteacher_one_nmse\tteacher_h_nmse\tteacher_h_escape\trollout_one_nmse\trollout_h_nmse\trollout_h_escape\trollout_minus_teacher_h_nmse\tG1_rows_learned\tG1_multiply_surrogate\tteacher_windows\trollout_windows\tteacher_multiply_surrogate\trollout_multiply_surrogate\tteacher_forward_rows\tteacher_gradient_rows\tteacher_state_adjoint_rows\trollout_forward_rows\trollout_gradient_rows\trollout_state_adjoint_rows\tG1_native_max_abs\tteacher_native_max_abs\trollout_native_max_abs").unwrap();
    writeln!(task,"seed\tworld\tT1_linear\tT1_constant\tT1_floor\tT3_min_explained\tT3_max_observation_L_over_G\toracle_success\tprobe_slow_nmse\tprobe_fast_nmse\tcentroid_ratio\tV5_no_refusal").unwrap();
    writeln!(v5,"seed\tworld\tkey\twidth\tbudget\tepoch\trows_learned\trefused\tfull_budget_no_refusal").unwrap();
    obs.flush().unwrap();metrics.flush().unwrap();task.flush().unwrap();v5.flush().unwrap();
    if is_mechanics {
        println!("PHASE=R3_O_RELEASE_CANDIDATE_MECHANICS_ONLY; SYNTHETIC_SEEDS=4; SCIENTIFIC_SEEDS=NONE");
    } else if is_successor {println!("PHASE=R3_SUCCESSOR_C2_{} candidate={} COMPUTE_MATCHED=NO",
        if is_successor_dev{"FROZEN_DEV"}else if is_successor_qual{"FROZEN_QUAL"}else{"DESIGN_ONLY"},BIN_CANDIDATE);}
    else { println!("PHASE=R3_O_FROZEN_{}",if is_dev{"DEV"}else{"QUAL"}); }
    println!("FULL_BUDGET_NATIVE_G1=96000; R3_TREATMENT_UPDATES=256");
    if is_successor {println!("SUCCESSOR_CANDIDATE={} C2_UPDATES={} C2_TERMINAL_WEIGHT={} C2_WINDOWS={} C2_BASE_LAMBDA={} HD_L2_ONLY=YES NO_SUPERIORITY_CLAIM",
        BIN_CANDIDATE,C2_UPDATES,C2_TERMINAL_WEIGHT,C2_WINDOWS,LAMBDA);}
    for (i,seed) in seeds.iter().enumerate() {
        if let Some(index)=selected {if i!=index {continue;}}
        if is_mechanics && !is_single && args[1]!="--mechanics-full-census" && i>0 {break;}
        let seed=*seed;
        if is_successor {println!("R3_C2_SEED_BEGIN candidate={} phase={} index={} seed={seed}",BIN_CANDIDATE,
            if is_successor_dev{"DEV"}else if is_successor_qual{"QUAL"}else{"DESIGN"},i);}
        else {println!("R3_L_SYNTHETIC_SEED_BEGIN index={} seed={seed}",i);}
        let world=world_data(WorldKind::Separated,seed);
        let mut levels=level_sequences(&world);
        let mut binding_native_refusals:Vec<String>=Vec::new();
        for (key,h) in BINDINGS {
            if is_mechanics && !is_single && args[1]!="--mechanics-full-census" && key!="HD/L2" {continue;}
            let index=levels.iter().position(|x|x.key==key).expect("missing binding level");
            let data=levels.remove(index);
            println!("R3_L_BINDING_BEGIN index={} seed={seed} key={} h={}",i,key,h);
            let fault=if args[1]=="--mechanics-inject-g1"{Some("G1")}
                      else if args[1]=="--mechanics-inject-rollout"{Some("ROLLOUT")}
                      else if args[1]=="--mechanics-inject-paired"{Some("PAIRED")}
                      else if args[1]=="--mechanics-native-g1-refusal-one-seed" && key=="HD/L2"{Some("NATIVE_G1")}
                      else{None};
            let row=run_binding(&k1,seed,data,&world,h,&mut metrics,&mut events,fault,
                &mut binding_native_refusals, (mode=="--mechanics-successor-c2-one-seed" || is_successor) && key=="HD/L2");
            writeln!(obs,"{}",row.join("\t")).unwrap();obs.flush().unwrap();metrics.flush().unwrap();
            println!("R3_L_BINDING_DONE index={} key={}",i,key);
        }
        for kind in [WorldKind::Separated,WorldKind::Matched] {
            if is_mechanics && !is_single && args[1]!="--mechanics-full-census" {continue;}
            let row=task_row_full(&k1,seed,kind,&mut v5,&binding_native_refusals);
            writeln!(task,"{}",row.join("\t")).unwrap();task.flush().unwrap();
        }
        if args[1]=="--mechanics-native-g1-refusal-one-seed" {
            assert_eq!(binding_native_refusals,vec!["HD/L2".to_string()],
                "NATIVE_K1_REFUSAL_NOT_OBSERVED_AND_PROPAGATED");
            println!("PASS_R3_R_NATIVE_REFUSAL_PROPAGATION_INTERNAL seed={seed} binding=HD/L2");
        }
        if is_successor {println!("R3_C2_SEED_COMPLETE candidate={} index={} seed={seed}",BIN_CANDIDATE,i);}
        else {println!("R3_L_SYNTHETIC_SEED_COMPLETE index={} seed={seed}",i);}
    }
    if matches!(mode,"--mechanics-one-seed"|"--mechanics-successor-c2-one-seed") {
        obs.flush().unwrap(); metrics.flush().unwrap(); task.flush().unwrap(); v5.flush().unwrap(); events.flush().unwrap();
        drop(obs);drop(metrics);drop(task);drop(v5);drop(events);
        let candidate=if mode=="--mechanics-one-seed"{"C0"}else{"C2"};
        let input=Path::new(&args[3]).parent().expect("INPUT_PARENT");
        assert_eq!(input.file_name().unwrap().to_str().unwrap(),candidate,"INPUT_CANDIDATE_DRIFT");
        let root=std::env::var("R3_BD_MECHANICS_EMIT_ROOT").expect("EMIT_ROOT_REQUIRED");
        let fault=std::env::var("R3_BD_INJECT_EMITTER_FAULT").unwrap_or_default()=="1";
        r3_bd_emitter::publish(input,&Path::new(&root).join(candidate),fault);
        println!("PASS_R3_BD_IN_ENGINE_NATIVE_EMITTER candidate={} files=5",candidate);
    }
    println!("PASS_R3_O_ENGINE_EXECUTION_MODE mode={}",args[1]);
    if is_mechanics {println!("SCIENTIFIC_EXECUTION=NO; DEV_PROTOCOL_FROZEN=NO; EXACT_FLOPS_MATCHED=NO");}
    else {println!("SCIENTIFIC_RUN_COMPLETE_PENDING_EXTERNAL_EVIDENCE_ACCEPTANCE");}
}
