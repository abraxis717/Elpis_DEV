// RECONSTRUCTION ADAPTER of the frozen H-ECS R3-BQ DEV supervisor (frozen source sha256
// 35df429c5648739593f64a0908d61eb060a9c05c0909e992c98d9abb6d34e49d, unmodified in Git); std only.
// NOT the historical supervisor. Pair execution, child invocation, stream capture, START/COMMIT formats,
// prefix chain, write-once/no-replay and FAULT semantics are the frozen ones, verbatim. Changed: the release
// identity. The historical binaries are not reproduced by the cold rebuild; the rebuilt engines/observer/K1 are
// pinned as RECONSTRUCTED artifacts, admitted only by exact raw-output identity on committed pairs 0..2. Pairs
// 0..2 stay bound to the frozen release (cffe...) and their archived commit digests; pairs 3..7 bind a
// reconstructed release record R' that binds the frozen authority, the source closure and the qualification.
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::Path;
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
const SRC_BYTES:&[u8]=include_bytes!("r3_bq_dev_supervisor_recon_adapter.rs");
// Reconstruction pins (identity only; no scientific constant is defined here).
const SOURCE_CLOSURE_COMMIT:&str="6776cd4c18c83f9bf75b6ddabfa22428cb2ff090";
const PKG:&str="research/hecs_r3/unresolved/successor_dev_3_of_8";
const FROZEN_RELEASE_SHA256:&str="cffe1fffab6e26e16e30e4d03fd72727ab55ffe115487703b10715989335a516";
const FROZEN_APPROVAL_SHA256:&str="2809b438a6487a9e15f4d2ccf4fc3e8d24d30337eb747201a0ddf01ea0ea535f";
const FROZEN_PROTOCOL_SHA256:&str="9ad9d1d35ce40756aaac91c47a1f26e635265e6115576f820b16d0d000ca262a";
const FROZEN_SUPERVISOR_SOURCE_SHA256:&str="35df429c5648739593f64a0908d61eb060a9c05c0909e992c98d9abb6d34e49d";
const FROZEN_PAIR_COMMITS:[&str;3]=["a731f84c5d77aa556d261e28075c777439839bfb630007e841ebcf18bee85c26",
 "5c48e9ec9446578d4030731fc06a1b437646fcd0317fd4fa0fcd5ed082aeabfd",
 "f5992e836342e1ac11daf7063101a34cc119d23a366e067fa7214adb4e55cf0b"];
const HISTORICAL:[(&str,&str);5]=[("ENGINE_BINARY_SHA256_C0","69a374c3259dae8d79151af3f3e78e262c1587f6b2d268d05dc33fcb48172130"),
 ("ENGINE_BINARY_SHA256_C1","0bc42613ed2c5407260bb65738f94ef9c2f44fd958e32d5cc66a2bbaf4b422b5"),
 ("OBSERVER_BINARY_SHA256","3c93e9122033b5401120a4a58c65de0bd7f23818cebc24cca0cfa3460b2af495"),
 ("NATIVE_K1_SHA256","a9bf232ddc108dd920fade613cf3901327ebc876717b410d549ec9a1c2551c8b"),
 ("SUPERVISOR_BINARY_SHA256","f0860e92067b927cef539a3f283c733b5ab8f056e3a3bc02573ff3f97033009e")];
// Frozen sources (archive path, built path) — byte identity required at both.
const SOURCES:[(&str,&str,&str);8]=[
 ("ENGINE_SOURCE_SHA256_C0","source/engines/r3_bn_c0_dev_engine.rs","build/hecs_r3_bn_successor_dev_engine_r0/native_source/src/bin/r3_bn_c0_dev_engine.rs"),
 ("ENGINE_SOURCE_SHA256_C1","source/engines/r3_bn_c1_dev_engine.rs","build/hecs_r3_bn_successor_dev_engine_r0/native_source/src/bin/r3_bn_c1_dev_engine.rs"),
 ("SUPPORT_SOURCE_SHA256_R3_TRAINING_CORE","source/engines/support/r3_training_core.rs","build/hecs_r3_bn_successor_dev_engine_r0/native_source/src/bin/support/r3_training_core.rs"),
 ("SUPPORT_SOURCE_SHA256_R3_BD_EMITTER","source/engines/support/r3_bd_emitter.rs","build/hecs_r3_bn_successor_dev_engine_r0/native_source/src/bin/support/r3_bd_emitter.rs"),
 ("SUPPORT_SOURCE_SHA256_R3_AX_C1","source/engines/support/r3_ax_c1.rs","build/hecs_r3_bn_successor_dev_engine_r0/native_source/src/bin/support/r3_ax_c1.rs"),
 ("CARGO_TOML_SHA256","source/build/Cargo.toml","build/hecs_r3_bn_successor_dev_engine_r0/native_source/Cargo.toml"),
 ("CARGO_LOCK_SHA256","source/build/Cargo.lock","build/hecs_r3_bn_successor_dev_engine_r0/native_source/Cargo.lock"),
 ("OBSERVER_SOURCE_SHA256","source/observer/r3_bp_observer.rs","build/hecs_r3_bp_phase_gate_mechanics_r0/r3_bp_observer.rs")];
const SOURCE_SHA256:[&str;8]=["2426143228444895fe5d7e9be74bd3c0141e5771c5626b512d2096215b3a23b7",
 "360d81b899bece6bd344c67e0362eaa879b7bcf2aada42c5d5307cf6d9108fa0",
 "51488b07c8a5677c2b06737c5922b6b5c5919440110d37f5d75c2245b3dfc103",
 "35bdfb020db3adcf74465360d84dd08399bd97c55f0a801249b72294b6c64064",
 "461ecb9f203be79b00b50ccead08cd6cfcab5322e38979dc0981839c03a8e3b2",
 "5ef432a6bdb917f1d6cc59f9d7b8d94ddf5b39f372cf3b008baf54d9669cdf04",
 "1ebc8d90479de75a43d35a982674cb4fcd26c81d97a332ad3c2b2668eb0118e6",
 "c63f71f5a4f0730c93ba82219fc6e597cc52b74147224b65160e67d8b45d2183"];
const RECONSTRUCTED:[(&str,&str,&str);4]=[
 ("RECONSTRUCTED_ENGINE_BINARY_SHA256_C0","build/hecs_r3_bn_successor_dev_engine_r0/cargo_target/debug/r3_bn_c0_dev_engine","d2ec51455329b52612a5dca568e94b2fdd99b3e7d10b00e3527fdfd81fbeeefd"),
 ("RECONSTRUCTED_ENGINE_BINARY_SHA256_C1","build/hecs_r3_bn_successor_dev_engine_r0/cargo_target/debug/r3_bn_c1_dev_engine","e137ac103180ace84d94c0d17600d5a667ee13dbb8354b7b838296eabef45e57"),
 ("RECONSTRUCTED_OBSERVER_BINARY_SHA256","build/hecs_r3_bp_phase_gate_mechanics_r0/r3_bp_observer","a878fd56c9eb44f7517533a406ce773e3cb6cfc46e97864ca2ddedda5d86c298"),
 ("RECONSTRUCTED_NATIVE_K1_SHA256","build/hecs_r3_k1/native/ECS/libelpis_ecsg_k1.so","8d8b7ffcf37039a4af85d8d5a8017406b2e13044ae1045477118ce76fdb8db9d")];
const TOOLCHAIN:&str="rustc 1.97.0 (2d8144b78 2026-07-07); cargo 1.97.0 (c980f4866 2026-06-30); cc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0; cmake 3.28.3; engines cargo dev profile --offline --locked; observer/supervisor rustc --edition 2021 -O; K1 cmake default build type";
const QUALIFICATION_PATH:&str="evidence/hecs_r3_successor/reconstruction/QUALIFICATION.json";
const QUALIFICATION_SHA256:&str="b9a8b5a60137b4c9e3b510eb874179127decf0e11cebf87e41b862a0d69a16e8";
const RECON_RELEASE_PATH:&str="evidence/hecs_r3_successor/reconstruction/DEV_RECONSTRUCTED_RELEASE.txt";
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
fn reconstructed_release(project:&Path,proto:&Path,supervisor_bin:&Path)->R<String>{
 need(project==Path::new(ROOT),"PROJECT_ROOT")?;
 safe_dir(project)?;
 let pkg=project.join(PKG);
 // The frozen release and its operator approval stay in force, byte-identical (the engines' structural seal).
 need(sha_file(&project.join("evidence/hecs_r3_successor/release/DEV_AUTHORITY.txt"))?==FROZEN_RELEASE_SHA256,"FROZEN_RELEASE_DRIFT")?;
 need(sha_file(&project.join("evidence/hecs_r3_successor/release/DEV_RELEASE_APPROVAL.txt"))?==FROZEN_APPROVAL_SHA256,"FROZEN_APPROVAL_DRIFT")?;
 need(sha_file(&pkg.join("frozen_boundary/DEV_AUTHORITY.txt"))?==FROZEN_RELEASE_SHA256,"ARCHIVED_RELEASE_DRIFT")?;
 need(sha_file(&pkg.join("frozen_boundary/DEV_RELEASE_APPROVAL.txt"))?==FROZEN_APPROVAL_SHA256,"ARCHIVED_APPROVAL_DRIFT")?;
 need(sha_file(proto)?==FROZEN_PROTOCOL_SHA256,"PROTOCOL_DRIFT")?;
 need(sha_file(&pkg.join("frozen_boundary/r3_bq_dev_supervisor.rs"))?==FROZEN_SUPERVISOR_SOURCE_SHA256,"FROZEN_SUPERVISOR_SOURCE_DRIFT")?;
 for (label,_,hash) in ACCEPTED {
  need(sha_file(&pkg.join(format!("frozen_boundary/R3_{}_ADMISSION.json",label)))?==hash,"PINNED_REPORT_DRIFT")?;
  println!("PASS_R3_BQ_ACCEPTED_REPORT {} sha256={}",label,hash);
 }
 let mut t=String::from("HECS_R3_SUCCESSOR_DEV_RECONSTRUCTED_RELEASE_V1\nPHASE=DEV\nCANDIDATES=C0,C1\nRECONSTRUCTED=YES\nHISTORICAL_BINARIES_REPRODUCED=NO\n");
 t.push_str(&format!("SOURCE_CLOSURE_COMMIT={}\nFROZEN_DEV_AUTHORITY_SHA256={}\nFROZEN_DEV_RELEASE_APPROVAL_SHA256={}\nPROTOCOL_SHA256={}\nFROZEN_SUPERVISOR_SOURCE_SHA256={}\n",
  SOURCE_CLOSURE_COMMIT,FROZEN_RELEASE_SHA256,FROZEN_APPROVAL_SHA256,FROZEN_PROTOCOL_SHA256,FROZEN_SUPERVISOR_SOURCE_SHA256));
 for (i,(key,archived,built)) in SOURCES.iter().enumerate() {
  let a=sha_file(&pkg.join(archived))?;let b=sha_file(&project.join(built))?;
  need(a==SOURCE_SHA256[i]&&b==SOURCE_SHA256[i],"FROZEN_SOURCE_DRIFT")?;
  t.push_str(&format!("{}={}\n",key,a));
 }
 for (key,path,hash) in RECONSTRUCTED {
  need(sha_file(&project.join(path))?==hash,"RECONSTRUCTED_BINARY_DRIFT")?;
  t.push_str(&format!("{}={}\n",key,hash));
 }
 for (key,hash) in HISTORICAL {t.push_str(&format!("HISTORICAL_{}={}\n",key,hash));}
 let exe=std::env::current_exe().map_err(|e|ioerr("CURRENT_EXE",e))?;
 need(fs::canonicalize(&exe).map_err(|e|ioerr("EXE_CANONICAL",e))?==fs::canonicalize(supervisor_bin).map_err(|e|ioerr("BIN_CANONICAL",e))?,"SUPERVISOR_EXECUTABLE_IDENTITY")?;
 t.push_str(&format!("ADAPTER_SUPERVISOR_SOURCE_SHA256={}\nADAPTER_SUPERVISOR_BINARY_SHA256={}\nTOOLCHAIN={}\n",sha256(SRC_BYTES),sha_file(supervisor_bin)?,TOOLCHAIN));
 let q=checked_file(&project.join(QUALIFICATION_PATH))?;
 need(sha256(&q)==QUALIFICATION_SHA256,"QUALIFICATION_RECORD_DRIFT")?;
 need(String::from_utf8(q).map_err(|_|"QUALIFICATION_UTF8")?.contains("\"qualified_exact_identity\": true"),"RECONSTRUCTION_NOT_QUALIFIED")?;
 let seed_text=DEV.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
 t.push_str(&format!(concat!("QUALIFICATION_RECORD_SHA256={}\nQUALIFICATION=EXACT_RAW_OUTPUT_IDENTITY_COMMITTED_PAIRS_0_1_2\n",
  "FROZEN_PAIR_COMMIT_SHA256={}\nRECONSTRUCTED_PAIR_INDICES=3,4,5,6,7\n",
  "DEV_SEEDS={}\nNATIVE_G1_STEPS=96000\nTREATMENT_UPDATES=256\n",
  "C1_LAMBDA4=0.1\nC1_WINDOWS_PER_UPDATE=8\nCOMPUTE_MATCHED=NO\n",
  "DEV_GATE=S1_0.95_M_0.90_ESCAPE_0.01_FULL_R3_BP\n",
  "NO_QUAL_AUTHORIZATION=YES\nNO_R4_AUTHORIZATION=YES\nNO_INTEGRATION_AUTHORIZATION=YES\nNO_GIT_MUTATION=YES\n"),
  QUALIFICATION_SHA256,FROZEN_PAIR_COMMITS.join(","),seed_text));
 Ok(t)
}
fn write_reconstructed_release(project:&Path,proto:&Path,supervisor_bin:&Path)->R<()> {
 let expected=reconstructed_release(project,proto,supervisor_bin)?;
 let p=project.join(RECON_RELEASE_PATH);
 need(!exists(&p),"RECONSTRUCTED_RELEASE_REPLAY_REFUSED")?;
 need(!exists(&project.join("build/hecs_r3_successor_dev_science_r0/seed_3")),"RECONSTRUCTED_RELEASE_AFTER_START_REFUSED")?;
 write_once(&p,&expected)?;
 println!("PASS_R3_BQ_RECON_WRITE_ONCE_RECONSTRUCTED_RELEASE sha256={}",sha256(expected.as_bytes()));
 Ok(())
}
fn show_reconstructed_release(project:&Path,proto:&Path,supervisor_bin:&Path)->R<()> {
 let expected=reconstructed_release(project,proto,supervisor_bin)?;
 need(checked_file(&project.join(RECON_RELEASE_PATH))?==expected.as_bytes(),"RECONSTRUCTED_RELEASE_CONTENT_DRIFT")?;
 println!("PASS_R3_BQ_RECON_RELEASE_PRECHECK reconstructed_release_sha256={}",sha256(expected.as_bytes()));
 Ok(())
}
fn approval(project:&Path,proto:&Path,supervisor_bin:&Path)->R<String>{
 let expected=reconstructed_release(project,proto,supervisor_bin)?;
 need(checked_file(&project.join(RECON_RELEASE_PATH)).map_err(|_|"DEV_RECONSTRUCTED_RELEASE_ABSENT")?==expected.as_bytes(),"RECONSTRUCTED_RELEASE_CHANGED")?;
 Ok(sha256(expected.as_bytes()))
}
// The release identity a pair's START binds: the frozen release for archived pairs 0..2, R' for 3..7.
fn release_for(index:usize,recon:&str)->String{if index<3{FROZEN_RELEASE_SHA256.to_string()}else{recon.to_string()}}
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
 need(index>=3&&index<8,"RECONSTRUCTED_PAIR_INDEX_OUT_OF_RANGE")?;
 let recon=approval(project,proto,supervisor_bin)?;
 let pseal=release_for(index,&recon);
 let work=project.join("build/hecs_r3_successor_dev_science_r0");
 let slot=work.join(format!("seed_{}",index));
 need(!exists(&slot),"STARTED_SEED_REPLAY_REFUSED")?;
 // Re-admit the entire predecessor prefix before reserving a new seed.
 let mut last:Option<String>=None;
 for i in 0..index {
  let s=work.join(format!("seed_{}",i));
  let c=verify_commit(&s,i,last.as_deref(),&release_for(i,&recon))?;
  if i<3 {need(c==FROZEN_PAIR_COMMITS[i],"ARCHIVED_PAIR_COMMIT_DRIFT")?;}
  last=Some(c);
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
  println!("PASS_R3_BQ_RECON_ONE_SHOT_SCIENTIFIC_PAIR_COMMITTED index={} commit_sha256={} RECONSTRUCTED=YES SCIENCE=YES QUAL=NO GIT=NO",index,admitted);
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
  [_,m,p,proto,bin] if m=="--write-reconstructed-release"=>write_reconstructed_release(Path::new(p),Path::new(proto),Path::new(bin)),
  [_,m,p,proto,bin] if m=="--verify-reconstructed-release"=>show_reconstructed_release(Path::new(p),Path::new(proto),Path::new(bin)),
  [_,m,p,proto,bin,i] if m=="--run-pair"=>i.parse::<usize>().map_err(|_|"INVALID_INDEX".into()).and_then(|n|pair(Path::new(p),Path::new(proto),Path::new(bin),n)),
  _=>fail("USAGE: r3_bq_dev_supervisor_recon_adapter --selftest | --write-reconstructed-release|--verify-reconstructed-release PROJECT PROTOCOL BINARY | --run-pair PROJECT PROTOCOL BINARY INDEX(3..7)"),
 };
 if let Err(e)=outcome {eprintln!("MECHANICS_OR_SCIENCE_REFUSAL_R3_BQ_RECON={}",e);std::process::exit(1)}
}
