//! R3-BP: native Rust successor per-seed and complete-DEV-cohort science observer.
//! Read-only. Does not authorize, launch, retry, or commit scientific work.
//! Canonical numerical decision law follows historical frozen R3-J R1.
use std::{collections::HashSet, env, fs, path::Path};
type R<T> = Result<T,String>;
const SEEDS:[u64;8]=[
15007341021985616615,12334837942893278080,5526746150047008937,2644957107391819373,
14747431184225333660,2338945401063472337,2859738232811479868,229129266962718901];
const NAMES:[&str;5]=["observations.tsv","metrics.tsv","tasks.tsv","v5.tsv","events.tsv"];
const BINDINGS:[(&str,u64);4]=[("L1/N72",16),("L1/N36",4),("TS/L2",4),("HD/L2",4)];
const ARMS:[&str;3]=["G1","TEACHER","ROLLOUT"];
const HORIZONS:[u64;5]=[1,2,4,8,16];
const OBS_HEADER:&str="seed\tkey\thorizon\twidth\ttrain_variance\tG1_one_nmse\tG1_h_nmse\tG1_h_escape\tteacher_one_nmse\tteacher_h_nmse\tteacher_h_escape\trollout_one_nmse\trollout_h_nmse\trollout_h_escape\trollout_minus_teacher_h_nmse\tG1_rows_learned\tG1_multiply_surrogate\tteacher_windows\trollout_windows\tteacher_multiply_surrogate\trollout_multiply_surrogate\tteacher_forward_rows\tteacher_gradient_rows\tteacher_state_adjoint_rows\trollout_forward_rows\trollout_gradient_rows\trollout_state_adjoint_rows\tG1_native_max_abs\tteacher_native_max_abs\trollout_native_max_abs";
const MET_HEADER:&str="seed\tkey\tarm\tconsumed_h\th\tbinding\tS1_ecs_one\tS1_linear_one\tS1_constant_one\tS1_capture\tS1_ge_095\tM_ecs_free_nmse\tM_linear_nmse\tM_constant_nmse\tM_capture\tM_escape\tM_planner_candidate_escape\tM_at_h_ge_090_escape_le_001";
const TASK_HEADER:&str="seed\tworld\tT1_linear\tT1_constant\tT1_floor\tT3_min_explained\tT3_max_observation_L_over_G\toracle_success\tprobe_slow_nmse\tprobe_fast_nmse\tcentroid_ratio\tV5_no_refusal";
const V5_HEADER:&str="seed\tworld\tkey\twidth\tbudget\tepoch\trows_learned\trefused\tfull_budget_no_refusal";
const EVENT_HEADER:&str="seed\tkey\tarm\tstage\tstatus\tcause\tsynthetic_injection";
fn demand(b:bool,c:&str)->R<()> {if b{Ok(())}else{Err(c.to_string())}}
fn u(s:&str)->R<u64> {s.parse().map_err(|_|format!("INVALID_U64:{s}"))}
fn f(s:&str)->R<f64> {s.parse().map_err(|_|format!("INVALID_F64:{s}"))}
fn bit(s:&str)->R<bool> {match s{"1"=>Ok(true),"0"=>Ok(false),_=>Err("INVALID_BOOLEAN".into())}}
fn eq_capture(a:f64,b:f64)->bool {
 if a.is_nan()||b.is_nan() {return a.is_nan()&&b.is_nan();}
 if a.is_infinite()||b.is_infinite(){return a==b;}
 (a-b).abs()<=1e-10_f64*(1.0_f64.max(a.abs()).max(b.abs()))
}
fn capture(ecs:f64,lin:f64,constant:f64)->f64 {
 let den=constant-lin;
 if den<=0.0 {f64::NEG_INFINITY} else {(constant-ecs)/den}
}
fn rows(p:&Path,header:&str,cols:usize)->R<Vec<Vec<String>>> {
 let md=fs::symlink_metadata(p).map_err(|e|format!("INPUT_FILE_MISSING:{}:{e}",p.display()))?;
 demand(md.file_type().is_file()&&!md.file_type().is_symlink(),"UNSAFE_FILE")?;
 demand(md.len()<64*1024*1024,"INPUT_OVERSIZE")?;
 let b=fs::read(p).map_err(|e|format!("INPUT_READ:{e}"))?;
 demand(b.last()==Some(&b'\n'),"TRUNCATED_FILE")?;
 let s=std::str::from_utf8(&b).map_err(|_|"INVALID_UTF8")?;
 let mut lines=s.lines();demand(lines.next()==Some(header),"HEADER_MISMATCH")?;
 let mut out=Vec::new();
 for line in lines {let v:Vec<String>=line.split('\t').map(str::to_owned).collect();demand(v.len()==cols,"COLUMN_COUNT")?;out.push(v);}
 Ok(out)
}
fn binding(s:&str)->R<usize>{BINDINGS.iter().position(|(k,_)|*k==s).ok_or("UNKNOWN_BINDING".into())}
fn arm(s:&str)->R<usize>{ARMS.iter().position(|x|*x==s).ok_or("UNKNOWN_ARM".into())}
fn world(s:&str)->R<usize>{match s{"SEPARATED"=>Ok(0),"MATCHED"=>Ok(1),_=>Err("UNKNOWN_WORLD".into())}}
#[derive(Debug,Clone)]
struct ResultSeed {seed:u64,task_local:bool,s1:bool,m:bool,oracle:[f64;2],centroid:[f64;2],v3:bool,m_fail:usize,events:usize}
fn inspect(dir:&Path,candidate:&str,seed:u64)->R<ResultSeed>{
 demand(candidate=="C0"||candidate=="C1","CANDIDATE_INVALID")?;
 demand(dir.file_name().and_then(|n|n.to_str())==Some(candidate),"CANDIDATE_PATH_MISMATCH")?;
 let dm=fs::symlink_metadata(dir).map_err(|_|"CANDIDATE_DIR_ABSENT")?;
 demand(dm.is_dir()&&!dm.file_type().is_symlink(),"UNSAFE_CANDIDATE_DIR")?;
 let mut entries=HashSet::new();
 for e in fs::read_dir(dir).map_err(|_|"READDIR_FAIL")? {
  let e=e.map_err(|_|"DIRENT_FAIL")?;let name=e.file_name().into_string().map_err(|_|"NONUTF8_FILENAME")?;
  demand(NAMES.contains(&name.as_str())||name=="CHILD_STREAM.log","UNAUTHORIZED_EVIDENCE_FILE")?;
  demand(entries.insert(name),"DUPLICATE_ENTRY")?;
 }
 for n in NAMES {demand(entries.contains(n),"INCOMPLETE_FIVE_FILE_CENSUS")?;}
 let obs=rows(&dir.join(NAMES[0]),OBS_HEADER,30)?;
 let metrics=rows(&dir.join(NAMES[1]),MET_HEADER,18)?;
 let tasks=rows(&dir.join(NAMES[2]),TASK_HEADER,12)?;
 let v5=rows(&dir.join(NAMES[3]),V5_HEADER,9)?;
 let events=rows(&dir.join(NAMES[4]),EVENT_HEADER,7)?;
 demand(obs.len()==4&&metrics.len()==60&&tasks.len()==2&&v5.len()==14,"ROW_CENSUS")?;
 let mut seen_obs=HashSet::new();
 for r in &obs {
  demand(u(&r[0])?==seed,"OBS_SEED")?;
  let key=binding(&r[1])?;demand(seen_obs.insert(key),"OBS_DUPLICATE")?;
  demand(u(&r[2])?==BINDINGS[key].1,"OBS_CONSUMED")?;
  demand(u(&r[3])?==(if key==0{72}else{36}),"OBS_WIDTH")?;
  for i in 4..30 {if (15..27).contains(&i){let _=u(&r[i])?;}else{let _=f(&r[i])?;}}
 }
 let mut seen=HashSet::new();let mut first_s1=[None;12];let mut s1=true;let mut m=true;let mut m_fail=0;
 for r in &metrics {
  demand(u(&r[0])?==seed,"METRIC_SEED")?;
  let key=binding(&r[1])?;let ar=arm(&r[2])?;let h=u(&r[4])?;
  demand(HORIZONS.contains(&h),"METRIC_HORIZON")?;
  demand(u(&r[3])?==BINDINGS[key].1,"METRIC_CONSUMED")?;
  let bound=h<=BINDINGS[key].1;
  demand(bit(&r[5])?==bound,"METRIC_BINDING_FLAG")?;
  demand(seen.insert((key,ar,h)),"METRIC_DUPLICATE")?;
  let ecs_one=f(&r[6])?;let lin_one=f(&r[7])?;let const_one=f(&r[8])?;let stored_s1=f(&r[9])?;
  let ecs_m=f(&r[11])?;let lin_m=f(&r[12])?;let const_m=f(&r[13])?;let stored_m=f(&r[14])?;
  let escape=f(&r[15])?;let planner=f(&r[16])?;
  demand(lin_one.is_finite()&&const_one.is_finite()&&lin_m.is_finite()&&const_m.is_finite(),"NONFINITE_REFERENCE")?;
  demand(!ecs_one.is_nan()||stored_s1.is_nan()||stored_s1.is_infinite(),"S1_NAN_CAPTURE")?;
  demand(!(ecs_one<0.0||ecs_m<0.0),"NEGATIVE_SQUARED_ERROR")?;
  demand(escape.is_finite()&&(0.0..=1.0).contains(&escape),"ESCAPE_DOMAIN")?;
  demand(planner.is_finite()&&(0.0..=1.0).contains(&planner),"PLANNER_ESCAPE_DOMAIN")?;
  let cs=capture(ecs_one,lin_one,const_one);let cm=capture(ecs_m,lin_m,const_m);
  demand(eq_capture(cs,stored_s1),"S1_CAPTURE_ARITHMETIC_DRIFT")?;
  demand(eq_capture(cm,stored_m),"M_CAPTURE_ARITHMETIC_DRIFT")?;
  let is_s1=ecs_one.is_finite()&&ecs_one>=0.0&&cs>=0.95;
  let is_m=cm>=0.90&&escape<=0.01;
  demand(bit(&r[10])?==is_s1,"S1_FLAG_DRIFT")?;
  demand(bit(&r[17])?==is_m,"M_FLAG_DRIFT")?;
  let k=key*3+ar;
  if let Some(x)=first_s1[k] {demand(eq_capture(x,cs),"S1_HORIZON_INCONSISTENCY")?;}else{first_s1[k]=Some(cs);}
  if ar==2 {if !is_s1 {s1=false;}if bound&&!is_m {m=false;m_fail+=1;}}
 }
 demand(seen.len()==60,"METRIC_CENSUS")?;
 let(mut task,mut v3)=(true,true);let mut oracles=[0.0;2];let mut centroids=[0.0;2];let mut task_seen=HashSet::new();
 for r in &tasks {
  demand(u(&r[0])?==seed,"TASK_SEED")?;
  let w=world(&r[1])?;demand(task_seen.insert(w),"TASK_DUPLICATE")?;
  let lin=f(&r[2])?;let constant=f(&r[3])?;let floor=f(&r[4])?;
  let t3exp=f(&r[5])?;let t3ratio=f(&r[6])?;let oracle=f(&r[7])?;let slow=f(&r[8])?;let fast=f(&r[9])?;let centroid=f(&r[10])?;
  demand(lin.is_finite()&&constant.is_finite()&&floor.is_finite()&&lin>=0.0&&constant>0.0&&floor>0.0,"T1_REFERENCE_DOMAIN")?;
  demand(oracle.is_finite()&&(0.0..=1.0).contains(&oracle),"ORACLE_DOMAIN")?;
  demand(centroid.is_finite()&&centroid>0.0,"CENTROID_DOMAIN")?;
  demand(!(slow<0.0||fast<0.0),"NEGATIVE_PROBE_ERROR")?;
  let t1a=lin<=(1.0-0.5)*constant;
  let t1b=(lin-floor).abs()<=0.005+0.15*floor;
  let t3a=t3exp>=0.25;let t3b=t3ratio<=1.20;
  let probe=slow<=0.1&&fast<=0.1;
  oracles[w]=oracle;centroids[w]=centroid;
  if !(t1a&&t1b&&t3a&&t3b&&bit(&r[11])?) {task=false;}
  if !probe {v3=false;}
 }
 demand(task_seen.len()==2,"TASK_CENSUS")?;
 let mut seen_v5=HashSet::new();
 for r in &v5 {
  demand(u(&r[0])?==seed,"V5_SEED")?;let w=world(&r[1])?;
  demand(seen_v5.insert((w,r[2].clone())),"V5_DUPLICATE")?;
  let _=u(&r[3])?;demand(u(&r[4])?==96000,"V5_BUDGET")?;
  let epoch=u(&r[5])?;let _=u(&r[6])?;
  let refusal=bit(&r[7])?;let good=bit(&r[8])?;
  if epoch!=96000||refusal||!good{task=false;}
 }
 for w in 0..2 {demand(seen_v5.iter().filter(|(ww,_)|*ww==w).count()==7,"V5_SEVEN_LEVEL_CENSUS")?;}
 for r in &events {
  demand(u(&r[0])?==seed,"EVENT_SEED")?;let _=binding(&r[1])?;let _=arm(&r[2])?;
  demand(r[4]=="SCIENTIFIC_NONPASS"&& !bit(&r[6])?,"EVENT_INVALID")?;
 }
 if !events.is_empty(){task=false;}
 Ok(ResultSeed{seed,task_local:task,s1,m,oracle:oracles,centroid:centroids,v3,m_fail,events:events.len()})
}
fn resultline(arm:&str,x:&ResultSeed){
 println!("R3_BP_PER_SEED candidate={} seed={} TASK_LOCAL={} V3={} S1={} M={} M_FAILED={} EVENTS={} ORACLE_SEPARATED={} ORACLE_MATCHED={} CENTROID_SEPARATED={} CENTROID_MATCHED={} SCIENCE_RELEASE=NO",arm,x.seed,x.task_local as u8,x.v3 as u8,x.s1 as u8,x.m as u8,x.m_fail,x.events,x.oracle[0],x.oracle[1],x.centroid[0],x.centroid[1]);
}
fn cohort(root:&Path,arm:&str,fixture:bool)->R<()> {
 demand(arm=="C0"||arm=="C1","COHORT_ARM")?;
 let m=fs::symlink_metadata(root).map_err(|_|"COHORT_ROOT_MISSING")?;
 demand(m.is_dir()&&!m.file_type().is_symlink(),"COHORT_ROOT_UNSAFE")?;
 let mut count=0;
 for e in fs::read_dir(root).map_err(|_|"COHORT_READDIR")? {let e=e.map_err(|_|"COHORT_ENTRY")?;
  let n=e.file_name().into_string().map_err(|_|"COHORT_NON_UTF8")?;
  demand((0..8).any(|i|n==format!("seed_{i}")),"COHORT_UNKNOWN_SEED_SLOT")?;count+=1;
 }
 demand(count==8,"COHORT_INCOMPLETE_EIGHT_SEED_CENSUS")?;
 let mut xs=Vec::new();
 for (i,seed) in SEEDS.iter().enumerate() {
  let d=root.join(format!("seed_{i}")).join(arm);
  let result=inspect(&d,arm,*seed)?;
  resultline(arm,&result);xs.push(result);
 }
 let mut means=[0.0;4];for x in &xs{means[0]+=x.oracle[0]/8.0;means[1]+=x.oracle[1]/8.0;means[2]+=x.centroid[0]/8.0;means[3]+=x.centroid[1]/8.0;}
 let v2=means[0]>=0.80&&means[1]>=0.80;
 let v4=means[2]>=4.0&&means[3]<=2.0;
 let tl=xs.iter().all(|x|x.task_local);let v3=xs.iter().all(|x|x.v3);
 let task=tl&&v2&&v3&&v4;let s1=xs.iter().all(|x|x.s1);let m=xs.iter().all(|x|x.m);
 let disposition=if !task{"TASK_INVALID_ON_DEV"}else if !s1{"ONE_STEP_MODEL_INVALID_ON_DEV"}else if !m{"WORLD_MODEL_INVALID_ON_DEV"}else{"DEV_WORLD_MODEL_VALID_CONDITIONAL_QUAL_ELIGIBLE"};
 println!("R3_BP_FULL_DEV_PHASE candidate={} n=8 TASK_LOCAL={} V2={} V3={} V4={} S1={} M={} V2_MEAN_SEPARATED={} V2_MEAN_MATCHED={} V4_MEAN_SEPARATED={} V4_MEAN_MATCHED={} DISPOSITION={} SYNTHETIC_FIXTURE={} QUAL_AUTHORIZED=NO INTEGRATION_AUTHORIZED=NO COMPUTE_MATCHED=NO",arm,tl as u8,v2 as u8,v3 as u8,v4 as u8,s1 as u8,m as u8,means[0],means[1],means[2],means[3],disposition,if fixture{"YES"}else{"NO"});
 Ok(())
}
fn main(){let a:Vec<String>=env::args().collect();let r=match a.as_slice(){
 [_,mode,dir,arm,seed] if mode=="--one"=>seed.parse::<u64>().map_err(|_|"BAD_SEED_ARGUMENT".to_string()).and_then(|s|inspect(Path::new(dir),arm,s)).map(|x|resultline(arm,&x)),
 [_,mode,root,arm] if mode=="--cohort"||mode=="--fixture-cohort"=>cohort(Path::new(root),arm,mode=="--fixture-cohort"),
 _=>Err("USAGE: r3_bp_observer --one PATH/C0 C0 SEED | --cohort ROOT C0".into()),
 };if let Err(e)=r {eprintln!("MECHANICS_NONQUAL_R3_BP_OBSERVER={e}");std::process::exit(2);}
}
#[cfg(test)]mod tests {
 use super::*;
 #[test]fn capture_is_recomputed(){assert!(eq_capture(capture(0.4,0.5,1.0),1.2));assert!(!eq_capture(capture(0.4,0.5,1.0),1.1));}
 #[test]fn finite_and_nonfinite(){assert!(eq_capture(f64::NEG_INFINITY,f64::NEG_INFINITY));assert!(eq_capture(f64::NAN,f64::NAN));assert!(!eq_capture(f64::NAN,1.0));}
 #[test]fn seed_census(){assert_eq!(SEEDS.len(),8);assert_eq!(SEEDS.iter().collect::<HashSet<_>>().len(),8);}
 #[test]fn gate_thresholds(){assert!(capture(0.54,0.5,1.0)>=0.90);assert!(capture(0.52,0.5,1.0)>=0.95);}
}
