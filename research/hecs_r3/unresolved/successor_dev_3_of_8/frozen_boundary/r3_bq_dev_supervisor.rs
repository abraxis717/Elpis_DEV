// H-ECS R3-BQ prospective Rust DEV scientific supervisor; std only.
// Preflight/proposal is mechanics-only. --run-pair requires a separately approved,
// exact pinned release; no user/CI workflow in this tranche invokes scientific runs.
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path,PathBuf};
use std::process::{Command,Stdio};
use std::sync::{Arc,Mutex};
use std::thread;

const ROOT:&str="/mnt/primesauce/Elpis_DEV";
const NAMES:[&str;5]=["observations.tsv","metrics.tsv","tasks.tsv","v5.tsv","events.tsv"];
const DEV:[u64;8]=[15007341021985616615,12334837942893278080,5526746150047008937,2644957107391819373,14747431184225333660,2338945401063472337,2859738232811479868,229129266962718901];
const ACCEPTED:[(&str,&str,&str);4]=[
 ("BM","evidence/hecs_r3/successor_mechanics/R3_BM_RUST_NATIVE_SUPERVISOR_R0/ADMISSION.json","f4b11353c39ffc889a18ac583dd99bd02469551df78c871bd53ef60e549ba0cf"),
 ("BN","evidence/hecs_r3/successor_mechanics/R3_BN_SUCCESSOR_DEV_ENGINE_R0/ADMISSION.json","973839a464ed82fa93e2d38f445d7d50ed5188b68a14fefacda5c85bd52e0cf4"),
 ("BO","evidence/hecs_r3/successor_mechanics/R3_BO_NATIVE_DEV_OBSERVER_R0/ADMISSION.json","d3b813dc6fb005a3371014381ca6a223031c9e0943e557c8048f0def42af5dda"),
 ("BP","evidence/hecs_r3/successor_mechanics/R3_BP_PHASE_GATE_MECHANICS_R0/ADMISSION.json","d6937e9f056a255ed93f384645874a83125aaa76bd27a1299e08b3303edd0756"),
];
const SRC_BYTES:&[u8]=include_bytes!("r3_bq_dev_supervisor.rs");
type R<T>=Result<T,String>;
fn fail<T>(why:&str)->R<T>{Err(why.into())}
fn need(ok:bool,why:&str)->R<()> {if ok{Ok(())}else{fail(why)}}
fn ioerr(where_:&str,e:io::Error)->String{format!("{}:{}",where_,e)}
fn sha256(buf: &[u8]) -> String {
 const K: [u32;64] = [
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
 ];
 let mut h: [u32;8] = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
 let mut data=buf.to_vec(); let bits=(buf.len() as u64).wrapping_mul(8);
 data.push(0x80); while data.len()%64!=56 {data.push(0)} data.extend_from_slice(&bits.to_be_bytes());
 for blk in data.chunks_exact(64) {
  let mut w=[0u32;64];
  for i in 0..16 {w[i]=u32::from_be_bytes([blk[i*4],blk[i*4+1],blk[i*4+2],blk[i*4+3]])}
  for i in 16..64 {
   let s0=w[i-15].rotate_right(7)^w[i-15].rotate_right(18)^(w[i-15]>>3);
   let s1=w[i-2].rotate_right(17)^w[i-2].rotate_right(19)^(w[i-2]>>10);
   w[i]=w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
  }
  let (mut a,mut b,mut c,mut d,mut e,mut f,mut g,mut z)=(h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7]);
  for i in 0..64 {
   let s1=e.rotate_right(6)^e.rotate_right(11)^e.rotate_right(25);
   let ch=(e&f)^(!e&g);
   let t1=z.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
   let s0=a.rotate_right(2)^a.rotate_right(13)^a.rotate_right(22);
   let maj=(a&b)^(a&c)^(b&c);
   let t2=s0.wrapping_add(maj);
   z=g;g=f;f=e;e=d.wrapping_add(t1);d=c;c=b;b=a;a=t1.wrapping_add(t2);
  }
  for (x,y) in h.iter_mut().zip([a,b,c,d,e,f,g,z]) {*x=x.wrapping_add(y);}
 }
 h.iter().map(|v|format!("{:08x}",v)).collect::<String>()
}
fn checked_file(path: &Path) -> R<Vec<u8>> {
 let m=fs::symlink_metadata(path).map_err(|e|ioerr("FILE_METADATA",e))?;
 need(m.file_type().is_file() && !m.file_type().is_symlink(),"UNSAFE_OR_MISSING_REGULAR_FILE")?;
 fs::read(path).map_err(|e|ioerr("FILE_READ",e))
}
fn sha_file(path:&Path)->R<String> {Ok(sha256(&checked_file(path)?))}
fn exists(path:&Path)->bool {fs::symlink_metadata(path).is_ok()}
fn safe_dir(path:&Path)->R<()> {
 let meta=fs::symlink_metadata(path).map_err(|e|ioerr("DIRECTORY_METADATA",e))?;
 need(meta.file_type().is_dir() && !meta.file_type().is_symlink(),"UNSAFE_DIRECTORY")
}
fn mkdir_new(path:&Path)->R<()> {
 need(!exists(path),"EXISTING_PATH_REPLAY_REFUSED")?;
 fs::create_dir(path).map_err(|e|ioerr("MKDIR",e))?;
 sync_dir(path.parent().ok_or("MISSING_PARENT")?)
}
fn sync_dir(path:&Path)->R<()> {
 File::open(path).and_then(|f|f.sync_all()).map_err(|e|ioerr("DIRECTORY_FSYNC",e))
}
fn write_once(path:&Path,text:&str)->R<()> {
 let mut file=OpenOptions::new().write(true).create_new(true).open(path).map_err(|e|ioerr("CREATE_NEW",e))?;
 file.write_all(text.as_bytes()).and_then(|_|file.sync_all()).map_err(|e|ioerr("WRITE_FSYNC",e))?;
 sync_dir(path.parent().ok_or("WRITE_PARENT")?)
}
fn report_field(report:&str,key:&str)->R<String>{
 let needle=format!("\"{}\":\"",key);
 let chunks:Vec<&str>=report.split(&needle).collect();
 need(chunks.len()==2,"REPORT_FIELD_ABSENT_OR_DUPLICATED")?;
 let found=chunks[1].split('"').next().unwrap_or("");
 need(found.len()==64&&found.bytes().all(|b|b.is_ascii_hexdigit()),"REPORT_DIGEST_FORMAT")?;
 Ok(found.to_string())
}
fn proposal(project:&Path,proto:&Path,supervisor_bin:&Path)->R<String>{
 need(project==Path::new(ROOT),"PROJECT_ROOT")?;
 safe_dir(project)?;
 let mut reports=std::collections::BTreeMap::new();
 for (label,path,hash) in ACCEPTED {
  let p=project.join(path);
  need(sha_file(&p)?==hash,"PINNED_REPORT_DRIFT")?;
  let contents=String::from_utf8(checked_file(&p)?).map_err(|_|"REPORT_UTF8")?;
  let allowed=if label=="BN" {contents.contains("\"science_authorized\":false")}
   else{contents.contains("\"scientific_execution\":false")};
  need(allowed,"PREDECESSOR_SCIENCE_SCOPE")?;
  reports.insert(label,contents);
  println!("PASS_R3_BQ_ACCEPTED_REPORT {} sha256={}",label,hash);
 }
 let bp=reports.get("BP").ok_or("BP_MISSING")?;
 let bn=reports.get("BN").ok_or("BN_MISSING")?;
 need(bp.contains("\"fixture_only\":true")&&bp.contains("\"successor_dev_started\":0"),"BP_FIXTURE_SCOPE")?;
 need(bn.contains("\"prospective_gate_status\":\"PROPOSAL_NOT_FROZEN\""),"BN_UNEXPECTED_FREEZE")?;
 let bn_root=project.join("build/hecs_r3_bn_successor_dev_engine_r0/cargo_target/debug");
 let k1=project.join("build/hecs_r3_k1/native/ECS/libelpis_ecsg_k1.so");
 let observer=project.join("build/hecs_r3_bp_phase_gate_mechanics_r0/r3_bp_observer");
 let c0=bn_root.join("r3_bn_c0_dev_engine");let c1=bn_root.join("r3_bn_c1_dev_engine");
 for f in [&c0,&c1,&k1,&observer,proto,supervisor_bin]{let _=sha_file(f)?;}
 let c0h=sha_file(&c0)?;let c1h=sha_file(&c1)?;
 // BN's frozen report includes the exact two candidate binary digests.
 for h in [&c0h,&c1h]{need(bn.contains(&format!("\"{}\"",h)),"BN_BINARY_IDENTITY_DRIFT")?;}
 let oh=sha_file(&observer)?;
 need(report_field(bp,"observer_binary_sha256")?==oh,"BP_OBSERVER_BINARY_DRIFT")?;
 let seed_text=DEV.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
 need(bn.contains(&DEV[0].to_string())&&bn.contains(&DEV[7].to_string())&&bp.contains(&DEV[0].to_string())&&bp.contains(&DEV[7].to_string()),"SEED_REGISTRY_DRIFT")?;
let exe=std::env::current_exe().map_err(|e|ioerr("CURRENT_EXE",e))?;
 need(fs::canonicalize(&exe).map_err(|e|ioerr("EXE_CANONICAL",e))?==fs::canonicalize(supervisor_bin).map_err(|e|ioerr("BIN_CANONICAL",e))?,"SUPERVISOR_EXECUTABLE_IDENTITY")?;
 let t=format!(concat!(
  "HECS_R3_SUCCESSOR_DEV_RELEASE_V1\nPHASE=DEV\nCANDIDATES=C0,C1\n",
  "ENGINE_BINARY_SHA256_C0={}\nENGINE_BINARY_SHA256_C1={}\n",
  "OBSERVER_BINARY_SHA256={}\nPROTOCOL_SHA256={}\n",
  "SUPERVISOR_SOURCE_SHA256={}\nSUPERVISOR_BINARY_SHA256={}\n",
  "NATIVE_K1_SHA256={}\nR3_BN_REPORT_SHA256={}\nR3_BP_REPORT_SHA256={}\n",
  "DEV_SEEDS={}\nNATIVE_G1_STEPS=96000\nTREATMENT_UPDATES=256\n",
  "C1_LAMBDA4=0.1\nC1_WINDOWS_PER_UPDATE=8\nCOMPUTE_MATCHED=NO\n",
  "DEV_GATE=S1_0.95_M_0.90_ESCAPE_0.01_FULL_R3_BP\n",
  "NO_QUAL_AUTHORIZATION=YES\nNO_GIT_MUTATION=YES\n"
 ),c0h,c1h,oh,sha_file(proto)?,sha256(SRC_BYTES),sha_file(supervisor_bin)?,sha_file(&k1)?,ACCEPTED[1].2,ACCEPTED[3].2,seed_text);
 Ok(t)
}
fn proposal_path(project:&Path)->PathBuf{
 project.join("evidence/hecs_r3/successor_mechanics/R3_BQ_DEV_RELEASE_CANDIDATE_R0/PROPOSED_DEV_AUTHORITY.txt")
}
fn write_proposal(project:&Path,proto:&Path,supervisor_bin:&Path)->R<()> {
 let expected=proposal(project,proto,supervisor_bin)?;
 let p=proposal_path(project);
 need(!exists(&p),"PROPOSAL_REPLAY_REFUSED")?;
 need(!exists(&project.join("evidence/hecs_r3_successor/release/DEV_AUTHORITY.txt")),"UNAUTHORIZED_EARLY_SEAL")?;
 write_once(&p,&expected)?;
 println!("PASS_R3_BQ_WRITE_ONCE_PROSPECTIVE_PROPOSAL sha256={}",sha256(expected.as_bytes()));
 Ok(())
}
fn show_proposal(project:&Path,proto:&Path,supervisor_bin:&Path)->R<()> {
 let expected=proposal(project,proto,supervisor_bin)?;
 let p=proposal_path(project);
 need(checked_file(&p)?==expected.as_bytes(),"PROPOSED_RELEASE_CONTENT_DRIFT")?;
 need(!exists(&project.join("evidence/hecs_r3_successor/release/DEV_AUTHORITY.txt")),"UNAUTHORIZED_EARLY_SEAL")?;
 println!("PASS_R3_BQ_RUST_RELEASE_PRECHECK proposal_sha256={}",sha256(expected.as_bytes()));
 Ok(())
}
fn approval(project:&Path,proto:&Path,supervisor_bin:&Path)->R<String>{
 let expected=proposal(project,proto,supervisor_bin)?;
 let p=proposal_path(project);
 need(checked_file(&p)?==expected.as_bytes(),"PROPOSAL_CHANGED")?;
 let seal=project.join("evidence/hecs_r3_successor/release/DEV_AUTHORITY.txt");
 need(checked_file(&seal).map_err(|_|"DEV_RELEASE_NOT_AUTHORIZED")?==expected.as_bytes(),"DEV_RELEASE_NOT_AUTHORIZED")?;
 let ap=project.join("evidence/hecs_r3_successor/release/DEV_RELEASE_APPROVAL.txt");
 let approv=format!("HECS_R3_SUCCESSOR_DEV_OPERATOR_RELEASE_V1\nPROPOSAL_SHA256={}\nSUPERVISOR_BINARY_SHA256={}\n",sha256(expected.as_bytes()),sha_file(supervisor_bin)?);
 need(checked_file(&ap).map_err(|_|"DEV_RELEASE_NOT_AUTHORIZED")?==approv.as_bytes(),"DEV_RELEASE_NOT_AUTHORIZED")?;
 Ok(sha256(expected.as_bytes()))
}
fn pump<T:Read+Send+'static>(reader:T,log:Arc<Mutex<File>>,label:&'static str)->thread::JoinHandle<R<()>>{
 thread::spawn(move||{
  let mut br=BufReader::new(reader);let mut line=Vec::new();
  loop {line.clear();let n=br.read_until(b'\n',&mut line).map_err(|e|ioerr("STREAM_READ",e))?;
   if n==0 {break;}
   {let mut f=log.lock().map_err(|_|"LOG_LOCK")?;f.write_all(label.as_bytes()).and_then(|_|f.write_all(b" ")).and_then(|_|f.write_all(&line)).map_err(|e|ioerr("LOG_WRITE",e))?;}
   print!("{} {}",label,String::from_utf8_lossy(&line));io::stdout().flush().map_err(|e|ioerr("LIVE_STDOUT",e))?;
  }Ok(())
 })
}
fn command_live(mut cmd:Command,logpath:&Path)->R<i32>{
 let lf=OpenOptions::new().write(true).create_new(true).open(logpath).map_err(|e|ioerr("CHILD_LOG_CREATE",e))?;
 let log=Arc::new(Mutex::new(lf));cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
 let mut child=cmd.spawn().map_err(|e|ioerr("CHILD_SPAWN",e))?;
 let so=child.stdout.take().ok_or("NO_STDOUT_PIPE")?;let se=child.stderr.take().ok_or("NO_STDERR_PIPE")?;
 let o=pump(so,log.clone(),"CHILD_STDOUT");let e=pump(se,log.clone(),"CHILD_STDERR");
 let status=child.wait().map_err(|e|ioerr("CHILD_WAIT",e))?;
 o.join().map_err(|_|"STDOUT_THREAD_PANIC")??;e.join().map_err(|_|"STDERR_THREAD_PANIC")??;
 log.lock().map_err(|_|"LOG_LOCK")?.sync_all().map_err(|e|ioerr("LOG_SYNC",e))?;
 Ok(status.code().unwrap_or(-15))
}
fn verify_commit(slot:&Path,index:usize,pred:Option<&str>,pseal:&str)->R<String>{
 let committed=slot.join("COMMITTED");safe_dir(&committed)?;
 let start=checked_file(&committed.join("START.txt"))?;
 let predecessor=pred.unwrap_or("NONE");
 let st=format!("HECS_R3_BQ_DEV_PAIR_START_V1\nINDEX={}\nSEED={}\nPREDECESSOR_COMMIT_SHA256={}\nRELEASE_PROPOSAL_SHA256={}\n",index,DEV[index],predecessor,pseal);
 need(start==st.as_bytes(),"COMMITTED_START_DRIFT")?;
 let mut record=format!("HECS_R3_BQ_DEV_PAIR_COMMIT_V1\nSTART_SHA256={}\n",sha256(&start));
 for arm in ["C0","C1"] {
  let directory=committed.join("raw").join(arm);safe_dir(&directory)?;
  for name in NAMES {record.push_str(&format!("{}_{}_SHA256={}\n",arm,name,sha_file(&directory.join(name))?));}
  record.push_str(&format!("{}_CHILD_LOG_SHA256={}\n",arm,sha_file(&committed.join(format!("{}_CHILD_STREAM.log",arm)))?));
  record.push_str(&format!("{}_OBSERVER_LOG_SHA256={}\n",arm,sha_file(&committed.join(format!("{}_OBSERVER_STREAM.log",arm)))?));
 }
 need(checked_file(&committed.join("COMMIT.txt"))?==record.as_bytes(),"COMMITTED_RECORD_DRIFT")?;
 Ok(sha256(record.as_bytes()))
}
fn pair(project:&Path,proto:&Path,supervisor_bin:&Path,index:usize)->R<()> {
 need(index<8,"DEV_SEED_INDEX_OUT_OF_RANGE")?;
 let pseal=approval(project,proto,supervisor_bin)?;
 let work=project.join("build/hecs_r3_successor_dev_science_r0");
 let slot=work.join(format!("seed_{}",index));
 need(!exists(&slot),"STARTED_SEED_REPLAY_REFUSED")?;
 // Re-admit the entire predecessor prefix before reserving a new seed.
 let mut last:Option<String>=None;
 for i in 0..index {
  let s=work.join(format!("seed_{}",i));
  last=Some(verify_commit(&s,i,last.as_deref(),&pseal)?);
 }
 if !exists(&work){fs::create_dir(&work).map_err(|e|ioerr("SCIENCE_ROOT_CREATE",e))?;sync_dir(work.parent().ok_or("ROOT_PARENT")?)?;}
 mkdir_new(&slot)?;
 let q=slot.join("STARTED_UNCOMMITTED");mkdir_new(&q)?;
 let start=format!("HECS_R3_BQ_DEV_PAIR_START_V1\nINDEX={}\nSEED={}\nPREDECESSOR_COMMIT_SHA256={}\nRELEASE_PROPOSAL_SHA256={}\n",index,DEV[index],last.as_deref().unwrap_or("NONE"),pseal);
 write_once(&q.join("START.txt"),&start)?;
 let outcome=(||->R<()> {
  mkdir_new(&q.join("raw"))?;
  let mut accepted_digests=std::collections::BTreeMap::<String,String>::new();
  let bn=project.join("build/hecs_r3_bn_successor_dev_engine_r0/cargo_target/debug");
  let k1=project.join("build/hecs_r3_k1/native/ECS/libelpis_ecsg_k1.so");
  let observer=project.join("build/hecs_r3_bp_phase_gate_mechanics_r0/r3_bp_observer");
  for arm in ["C0","C1"] {
   let raw=q.join("raw").join(arm);mkdir_new(&raw)?;
   let mut c=Command::new(bn.join(if arm=="C0"{"r3_bn_c0_dev_engine"}else{"r3_bn_c1_dev_engine"}));
   c.arg(if arm=="C0"{"--successor-dev-c0-one-seed"}else{"--successor-dev-c1-one-seed"}).arg(&k1);
   for f in NAMES{c.arg(raw.join(f));}c.arg(index.to_string());
   c.current_dir("/").env_remove("R3_BD_MECHANICS_EMIT_ROOT").env_remove("R3_BD_INJECT_EMITTER_FAULT");
   println!("R3_BQ_SCIENTIFIC_CHILD_START candidate={} index={} seed={}",arm,index,DEV[index]);
   let rc=command_live(c,&q.join(format!("{}_CHILD_STREAM.log",arm)))?;
   need(rc==0,"SCIENTIFIC_CHILD_NONZERO_TERMINAL")?;
   let mut obs=Command::new(&observer);
   obs.arg("--one").arg(&raw).arg(arm).arg(DEV[index].to_string()).current_dir("/");
   let result=command_live(obs,&q.join(format!("{}_OBSERVER_STREAM.log",arm)))?;
   need(result==0,"SCIENTIFIC_EVIDENCE_INVALID_TERMINAL")?;
   for f in NAMES{let h=sha_file(&raw.join(f))?;accepted_digests.insert(format!("{}_{}",arm,f),h);}
   println!("PASS_R3_BQ_SCIENTIFIC_CANDIDATE_EVIDENCE_ADMITTED candidate={} seed={}",arm,DEV[index]);
  }
  let mut commit=format!("HECS_R3_BQ_DEV_PAIR_COMMIT_V1\nSTART_SHA256={}\n",sha_file(&q.join("START.txt"))?);
  for arm in ["C0","C1"] {
   for f in NAMES{
    let h=sha_file(&q.join("raw").join(arm).join(f))?;
    need(accepted_digests.get(&format!("{}_{}",arm,f))==Some(&h),"EVIDENCE_POST_OBSERVER_MUTATION")?;
    commit.push_str(&format!("{}_{}_SHA256={}\n",arm,f,h));
   }
   commit.push_str(&format!("{}_CHILD_LOG_SHA256={}\n",arm,sha_file(&q.join(format!("{}_CHILD_STREAM.log",arm)))?));
   commit.push_str(&format!("{}_OBSERVER_LOG_SHA256={}\n",arm,sha_file(&q.join(format!("{}_OBSERVER_STREAM.log",arm)))?));
  }
  write_once(&q.join("COMMIT.txt"),&commit)?;
  fs::rename(&q,slot.join("COMMITTED")).map_err(|e|ioerr("ATOMIC_COMMIT_RENAME",e))?;
  sync_dir(&slot)?;
  let admitted=verify_commit(&slot,index,last.as_deref(),&pseal)?;
  println!("PASS_R3_BQ_ONE_SHOT_SCIENTIFIC_PAIR_COMMITTED index={} commit_sha256={} SCIENCE=YES QUAL=NO GIT=NO",index,admitted);
  Ok(())
 })();
 if let Err(ref why)=outcome {
  if exists(&q) && !exists(&q.join("FAULT.txt")) {
   let fault=format!("HECS_R3_BQ_DEV_PAIR_TERMINAL_V1\nINDEX={}\nSEED={}\nSTART_SHA256={}\nREASON={}\nREPLAY_PERMITTED=NO\n",index,DEV[index],sha_file(&q.join("START.txt")).unwrap_or_else(|_|"START_MISSING".into()),why);
   if let Err(err)=write_once(&q.join("FAULT.txt"),&fault){eprintln!("R3_BQ_FAULT_RECORD_UNCERTAIN {}",err);}
  }
 }
 outcome
}
fn selftest()->R<()> {
 need(sha256(b"")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","SHA256_EMPTY")?;
 need(sha256(b"abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","SHA256_ABC")?;
 need(sha256(&vec![b'a';1_000_000])=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0","SHA256_MILLION_A")?;
 need(DEV.iter().copied().collect::<std::collections::HashSet<_>>().len()==8,"SEED_DUPLICATES")?;
 println!("PASS_R3_BQ_RUST_SHA256_SEED_AND_EXCLUSIVE_RECORD_SELFTEST");Ok(())
}
fn main(){
 let a=std::env::args().collect::<Vec<_>>();
 let outcome=match a.as_slice(){
  [_,m] if m=="--selftest"=>selftest(),
  [_,m,p,proto,bin] if m=="--write-proposal"=>write_proposal(Path::new(p),Path::new(proto),Path::new(bin)),
  [_,m,p,proto,bin] if m=="--verify-proposal"=>show_proposal(Path::new(p),Path::new(proto),Path::new(bin)),
  [_,m,p,proto,bin,i] if m=="--run-pair"=>i.parse::<usize>().map_err(|_|"INVALID_INDEX".into()).and_then(|n|pair(Path::new(p),Path::new(proto),Path::new(bin),n)),
  _=>fail("USAGE: r3_bq_supervisor --selftest | --verify-proposal PROJECT PROTOCOL BINARY | --run-pair PROJECT PROTOCOL BINARY INDEX"),
 };
 if let Err(e)=outcome {eprintln!("MECHANICS_OR_SCIENCE_REFUSAL_R3_BQ={}",e);std::process::exit(1)}
}
