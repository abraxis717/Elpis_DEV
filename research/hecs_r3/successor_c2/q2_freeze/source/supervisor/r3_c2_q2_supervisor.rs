// C2-Q2: the frozen C2 supervisor (binary 726cf1f9..., source 50582d3e...) re-pinned for the fresh Q2 QUAL execution
// of the UNCHANGED C2 candidate after the infrastructure-interrupted Q1. Only the QUAL2 phase exists here.
// H-ECS R3 successor C2 scientific supervisor (DEV and QUAL); std only. RESEARCH_ONLY.
// Derived from the frozen R3-BQ supervisor (source 35df429c...): SHA-256, file safety, write-once, replay refusal,
// child stream capture, START/COMMIT/FAULT formats and the prefix chain are verbatim; the slot holds the single
// candidate C2. A release binds every artifact identity; QUAL release additionally requires the committed DEV
// disposition DEV_WORLD_MODEL_VALID_CONDITIONAL_QUAL_ELIGIBLE. No retry, replay, rescue or retuning exists here.
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path,PathBuf};
use std::process::{Command,Stdio};
use std::sync::{Arc,Mutex};
use std::thread;

const ROOT:&str="/mnt/primesauce/Elpis_DEV";
const NAMES:[&str;5]=["observations.tsv","metrics.tsv","tasks.tsv","v5.tsv","events.tsv"];
const ARM:&str="C2";
const DEV:[u64;8]=[684004886576317082,1703137016795210571,10088827697117853828,8960933697477120594,134760612940272399,7871577751992445249,17179523806848842400,13174565413729237818];
const QUAL:[u64;12]=[614105602224108045,13431973024848494695,8015593928169052856,4368982056838705008,12192592809587942193,97214863508322549,6955956690753739111,18105131785740719551,17684696095471157875,11665294448906309688,16885114571500087964,12577501025278960659];
const ENGINE:&str="build/hecs_r3_successor_c2_q2_engine_r0/r3_bn_c2_dev_engine";
const OBSERVER:&str="build/hecs_r3_successor_c2_q2_engine_r0/r3_c2_observer";
const K1:&str="build/hecs_r3_k1/native/ECS/libelpis_ecsg_k1.so";
// Pinned artifact identities (frozen at C2 freeze; a drift refuses the release and every run).
const ENGINE_SHA256:&str="38b08ff26c092c88527008ae5ec93578c11a1e022db402bb177a4e26fcc5b0ee";
const OBSERVER_SHA256:&str="93c4a5e8461743509c65ef5f8a1f718e79ff6710275203444a6cfd8e321de372";
const K1_SHA256:&str="8d8b7ffcf37039a4af85d8d5a8017406b2e13044ae1045477118ce76fdb8db9d";
const PROTOCOL_SHA256:&str="df9d2a57a97798c835e18aec03b2a9a14efbd0fb0292d9cb679f4f64279a2da1";
const DESIGN_RECORD_SHA256:&str="04291ebd385cb6ff46d6bf5e029f4c0dec5f6e2650f371c5626e66f5658aa6b9";
// Pinned source identities of the built artifacts (paths relative to the project root).
const SOURCES:[(&str,&str,&str);13]=[
 ("ENGINE_SOURCE_SHA256_C2","build/hecs_r3_successor_c2_q2_engine_r0/native_source/src/bin/r3_bn_c2_dev_engine.rs","69ee89e58817ca836cf2143e07acdacded4a08dc723083759394333299209ea8"),
 ("C2_TERMINAL_SOURCE_SHA256","build/hecs_r3_successor_c2_q2_engine_r0/native_source/src/bin/support/r3_c2_terminal.rs","344879473c17a10193b1b0422b9eea68e33bc66fb98dea4cb1beb9902e3eb86f"),
 ("TRAINING_CORE_SOURCE_SHA256","build/hecs_r3_successor_c2_q2_engine_r0/native_source/src/bin/support/r3_training_core.rs","51488b07c8a5677c2b06737c5922b6b5c5919440110d37f5d75c2245b3dfc103"),
 ("EMITTER_SOURCE_SHA256","build/hecs_r3_successor_c2_q2_engine_r0/native_source/src/bin/support/r3_bd_emitter.rs","35bdfb020db3adcf74465360d84dd08399bd97c55f0a801249b72294b6c64064"),
 ("CARGO_TOML_SHA256","build/hecs_r3_successor_c2_q2_engine_r0/native_source/Cargo.toml","5ef432a6bdb917f1d6cc59f9d7b8d94ddf5b39f372cf3b008baf54d9669cdf04"),
 ("CARGO_LOCK_SHA256","build/hecs_r3_successor_c2_q2_engine_r0/native_source/Cargo.lock","1ebc8d90479de75a43d35a982674cb4fcd26c81d97a332ad3c2b2668eb0118e6"),
 ("OBSERVER_SOURCE_SHA256","build/hecs_r3_successor_c2_q2_engine_r0/r3_c2_observer.rs","07b9edcb74f75c0085f3ccd4014896b5adf56810ed2f0f84410e88b06422cb79"),
 ("PROTOCOL_COPY_SHA256","evidence/hecs_r3_successor_c2/PROTOCOL_R3_C2_Q2.md","df9d2a57a97798c835e18aec03b2a9a14efbd0fb0292d9cb679f4f64279a2da1"),
 ("REGISTRIES_SHA256","evidence/hecs_r3_successor_c2/C2_Q2_REGISTRY.json","22a932cbf3fc06ff7bed5b29afa1e1e9979ca6aaabe0d95ae55e278626e13c23"),
 ("C2_PROTOCOL_SHA256","evidence/hecs_r3_successor_c2/PROTOCOL_R3_C2.md","09f7278c2a230d18328c3127df05a017002150c2f62b46d8fc30ef03859e9d79"),
 ("C2_REGISTRIES_SHA256","evidence/hecs_r3_successor_c2/C2_REGISTRIES.json","2f034eceb3e191bd9cf493c8167f894eeb2e4a998e1d29b177e434ab0d336cfb"),
 ("Q1_INCIDENT_SHA256","evidence/hecs_r3_successor_c2/Q1_INCIDENT.txt","05dda85336dacb07019a03a5460e8636d1dc388fa53f3dd7dcbbbc1e518c95ce"),
 ("Q1_QUAL_RELEASE_SHA256","evidence/hecs_r3_successor_c2/release/QUAL_AUTHORITY.txt","d29b06223ab7de52edd3b86d7aa9fa2a3d91ea2e81270d6b98f3e312e7507406")];
// hecs_r2 library source tree: sha256 over "<file> <sha256>\n" for every src/*.rs, sorted by file name.
const LIB_TREE_SHA256:&str="f030ed4978567f6520443748463ea7fec4deb39f20321499c996822aea021d90";
const LIB_BUILT:&str="build/hecs_r3_successor_c2_q2_engine_r0/native_source/src";
const LIB_REPOSITORY:&str="research/hecs_r2/rust/src";
const RECONSTRUCTION_QUALIFICATION_SHA256:&str="b9a8b5a60137b4c9e3b510eb874179127decf0e11cebf87e41b862a0d69a16e8";
const SRC_BYTES:&[u8]=include_bytes!("r3_c2_q2_supervisor.rs");
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
fn phase_of(p:&str)->R<(&'static str,&'static [u64],&'static str,&'static str,&'static str)>{match p{
 "QUAL2"=>Ok(("QUAL",&QUAL,"build/hecs_r3_successor_c2_qual2_r0","--successor-qual-c2-one-seed","C2_Q2_QUAL")),
 _=>fail("UNKNOWN_PHASE")}}
fn seal_path(project:&Path,phase:&str)->PathBuf{project.join(format!("evidence/hecs_r3_successor_c2/release/{}_AUTHORITY.txt",phase))}
fn disposition_path(project:&Path,phase:&str)->PathBuf{project.join(format!("evidence/hecs_r3_successor_c2/{}_DISPOSITION.txt",phase))}
fn lib_tree(dir:&Path)->R<String>{
 safe_dir(dir)?;
 let mut names=Vec::new();
 for e in fs::read_dir(dir).map_err(|e|ioerr("LIB_READDIR",e))? {
  let e=e.map_err(|e|ioerr("LIB_DIRENT",e))?;let n=e.file_name().into_string().map_err(|_|"LIB_NON_UTF8")?;
  if n.ends_with(".rs") {names.push(n);}
 }
 names.sort();
 need(!names.is_empty(),"LIB_EMPTY")?;
 let mut text=String::new();
 for n in names {text.push_str(&format!("{} {}\n",n,sha_file(&dir.join(&n))?));}
 Ok(sha256(text.as_bytes()))
}
fn release(project:&Path,proto:&Path,supervisor_bin:&Path,phase:&str)->R<String>{
 need(project==Path::new(ROOT),"PROJECT_ROOT")?;
 safe_dir(project)?;
 let (ph,seeds,_,_,_)=phase_of(phase)?;
 need(sha_file(&project.join(ENGINE))?==ENGINE_SHA256,"ENGINE_BINARY_DRIFT")?;
 need(sha_file(&project.join(OBSERVER))?==OBSERVER_SHA256,"OBSERVER_BINARY_DRIFT")?;
 need(sha_file(&project.join(K1))?==K1_SHA256,"K1_DRIFT")?;
 need(sha_file(proto)?==PROTOCOL_SHA256,"PROTOCOL_DRIFT")?;
 need(sha_file(&project.join("evidence/hecs_r3_successor_c2/DESIGN_RECORD.json"))?==DESIGN_RECORD_SHA256,"DESIGN_RECORD_DRIFT")?;
 let exe=std::env::current_exe().map_err(|e|ioerr("CURRENT_EXE",e))?;
 need(fs::canonicalize(&exe).map_err(|e|ioerr("EXE_CANONICAL",e))?==fs::canonicalize(supervisor_bin).map_err(|e|ioerr("BIN_CANONICAL",e))?,"SUPERVISOR_EXECUTABLE_IDENTITY")?;
 let seed_text=seeds.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
 let mut t=format!(concat!(
  "HECS_R3_SUCCESSOR_C2_RELEASE_V1\nPHASE={}\nCANDIDATES=C2\n",
  "ENGINE_BINARY_SHA256_C2={}\nOBSERVER_BINARY_SHA256={}\nNATIVE_K1_SHA256={}\nPROTOCOL_SHA256={}\n",
  "SUPERVISOR_SOURCE_SHA256={}\nSUPERVISOR_BINARY_SHA256={}\nDESIGN_RECORD_SHA256={}\n",
  "RECONSTRUCTION_QUALIFICATION_SHA256={}\nSEEDS={}\n",
  "NATIVE_G1_STEPS=96000\nTREATMENT_UPDATES=256\nC2_BINDING=HD/L2\nC2_UPDATES=4096\nC2_TERMINAL_WEIGHT=1.0\n",
  "C2_WINDOWS=6\nC2_BASE_LAMBDA=0.37\nCOMPUTE_MATCHED=NO\n",
  "GATE=TASK_S1_0.95_M_0.90_ESCAPE_0.01_EVERY_SEED_BINDING_HORIZON\n"),
  ph,ENGINE_SHA256,OBSERVER_SHA256,K1_SHA256,PROTOCOL_SHA256,sha256(SRC_BYTES),sha_file(supervisor_bin)?,
  DESIGN_RECORD_SHA256,RECONSTRUCTION_QUALIFICATION_SHA256,seed_text);
 for (key,path,hash) in SOURCES {
  need(sha_file(&project.join(path))?==hash,"SOURCE_DRIFT")?;
  t.push_str(&format!("{}={}\n",key,hash));
 }
 need(lib_tree(&project.join(LIB_BUILT))?==LIB_TREE_SHA256,"LIB_TREE_DRIFT")?;
 need(lib_tree(&project.join(LIB_REPOSITORY))?==LIB_TREE_SHA256,"LIB_REPOSITORY_DRIFT")?;
 t.push_str(&format!("HECS_R2_LIB_TREE_SHA256={}\n",LIB_TREE_SHA256));
 if ph=="QUAL" {
  let d=checked_file(&disposition_path(project,"DEV")).map_err(|_|"DEV_DISPOSITION_ABSENT")?;
  let text=String::from_utf8(d.clone()).map_err(|_|"DEV_DISPOSITION_UTF8")?;
  need(text.lines().any(|l|l=="DISPOSITION=DEV_WORLD_MODEL_VALID_CONDITIONAL_QUAL_ELIGIBLE"),"DEV_NOT_PASSED_QUAL_REFUSED")?;
  t.push_str(&format!("DEV_DISPOSITION_SHA256={}\n",sha256(&d)));
  need(!exists(&disposition_path(project,"QUAL")),"Q1_UNEXPECTEDLY_CLOSED")?;
  t.push_str(concat!("QUAL_EXECUTION=Q2\nQ1_DISPOSITION=INCOMPLETE_INFRASTRUCTURE_INTERRUPTION_REGISTRY_SPENT\n",
   "CANDIDATE_CHANGED=NO\nRETUNED_AFTER_Q1=NO\nOLD_QUAL_RESULTS_USED_FOR_SELECTION=NO\nQ1_REGISTRY_REUSED=NO\nQ2_REGISTRY_FRESH=YES\n",
   "FROZEN_C2_ENGINE_BINARY_SHA256=3661b21ffabadc53b760e9be887416fb6b53bbcacfe5ab264b7ab106715d8204\n",
   "FROZEN_C2_OBSERVER_BINARY_SHA256=49566a769eed7d95d0de5606bafaf5d0f1600e860945842effc11b6550d5f718\n",
   "FROZEN_C2_SUPERVISOR_BINARY_SHA256=726cf1f9a2a847334d4eef0d2c96e949834eae40508c64f9cfcdd5aa23c1b72e\n",
   "OPERATOR_AUTHORIZATION=ONE_Q2_QUAL_EXECUTION_OF_UNCHANGED_C2_AFTER_Q1_INFRASTRUCTURE_INTERRUPTION\n"));
 } else {
  t.push_str("OPERATOR_AUTHORIZATION=ONE_FRESH_DEV_EXECUTION_OF_THE_FROZEN_C2_CANDIDATE\nNO_QUAL_AUTHORIZATION=YES\n");
 }
 t.push_str("NO_RETUNING=YES\nNO_REPLAY=YES\nNO_R4_AUTHORIZATION=YES\nNO_INTEGRATION_AUTHORIZATION=YES\n");
 Ok(t)
}
fn write_release(project:&Path,proto:&Path,supervisor_bin:&Path,phase:&str)->R<()> {
 let (_,_,work,_,_)=phase_of(phase)?;
 let expected=release(project,proto,supervisor_bin,phase)?;
 let p=seal_path(project,phase);
 need(!exists(&p),"RELEASE_REPLAY_REFUSED")?;
 need(!exists(&project.join(work)),"RELEASE_AFTER_START_REFUSED")?;
 write_once(&p,&expected)?;
 println!("PASS_R3_C2_WRITE_ONCE_RELEASE phase={} sha256={}",phase,sha256(expected.as_bytes()));
 Ok(())
}
fn approval(project:&Path,proto:&Path,supervisor_bin:&Path,phase:&str)->R<String>{
 let expected=release(project,proto,supervisor_bin,phase)?;
 need(checked_file(&seal_path(project,phase)).map_err(|_|"PHASE_NOT_RELEASED")?==expected.as_bytes(),"RELEASE_CHANGED")?;
 Ok(sha256(expected.as_bytes()))
}
fn start_text(phase:&str,index:usize,seed:u64,pred:&str,pseal:&str)->String{
 format!("HECS_R3_SUCCESSOR_C2_SEED_START_V1\nPHASE={}\nINDEX={}\nSEED={}\nPREDECESSOR_COMMIT_SHA256={}\nRELEASE_SHA256={}\n",phase,index,seed,pred,pseal)
}
fn verify_commit(slot:&Path,phase:&str,index:usize,seed:u64,pred:Option<&str>,pseal:&str)->R<String>{
 let committed=slot.join("COMMITTED");safe_dir(&committed)?;
 let start=checked_file(&committed.join("START.txt"))?;
 need(start==start_text(phase,index,seed,pred.unwrap_or("NONE"),pseal).as_bytes(),"COMMITTED_START_DRIFT")?;
 let mut record=format!("HECS_R3_SUCCESSOR_C2_SEED_COMMIT_V1\nSTART_SHA256={}\n",sha256(&start));
 let directory=committed.join("raw").join(ARM);safe_dir(&directory)?;
 for name in NAMES {record.push_str(&format!("{}_{}_SHA256={}\n",ARM,name,sha_file(&directory.join(name))?));}
 record.push_str(&format!("{}_CHILD_LOG_SHA256={}\n",ARM,sha_file(&committed.join(format!("{}_CHILD_STREAM.log",ARM)))?));
 record.push_str(&format!("{}_OBSERVER_LOG_SHA256={}\n",ARM,sha_file(&committed.join(format!("{}_OBSERVER_STREAM.log",ARM)))?));
 need(checked_file(&committed.join("COMMIT.txt"))?==record.as_bytes(),"COMMITTED_RECORD_DRIFT")?;
 Ok(sha256(record.as_bytes()))
}
fn prefix(project:&Path,phase:&str,upto:usize,pseal:&str)->R<Option<String>>{
 let (ph,seeds,work,_,_)=phase_of(phase)?;
 let mut last:Option<String>=None;
 for i in 0..upto {
  last=Some(verify_commit(&project.join(work).join(format!("seed_{}",i)),ph,i,seeds[i],last.as_deref(),pseal)?);
 }
 Ok(last)
}
fn run_seed(project:&Path,proto:&Path,supervisor_bin:&Path,phase:&str,index:usize)->R<()> {
 let (ph,seeds,work_rel,mode,_)=phase_of(phase)?;
 need(index<seeds.len(),"SEED_INDEX_OUT_OF_RANGE")?;
 let pseal=approval(project,proto,supervisor_bin,phase)?;
 need(!exists(&disposition_path(project,phase)),"PHASE_ALREADY_CLOSED")?;
 let work=project.join(work_rel);
 let slot=work.join(format!("seed_{}",index));
 need(!exists(&slot),"STARTED_SEED_REPLAY_REFUSED")?;
 // Re-admit the entire predecessor prefix before reserving a new seed.
 let last=prefix(project,ph,index,&pseal)?;
 if !exists(&work){fs::create_dir(&work).map_err(|e|ioerr("SCIENCE_ROOT_CREATE",e))?;sync_dir(work.parent().ok_or("ROOT_PARENT")?)?;}
 mkdir_new(&slot)?;
 let q=slot.join("STARTED_UNCOMMITTED");mkdir_new(&q)?;
 write_once(&q.join("START.txt"),&start_text(ph,index,seeds[index],last.as_deref().unwrap_or("NONE"),&pseal))?;
 let outcome=(||->R<()> {
  mkdir_new(&q.join("raw"))?;
  let raw=q.join("raw").join(ARM);mkdir_new(&raw)?;
  let mut c=Command::new(project.join(ENGINE));
  c.arg(mode).arg(project.join(K1));
  for f in NAMES{c.arg(raw.join(f));}c.arg(index.to_string());
  c.current_dir("/").env_remove("R3_BD_MECHANICS_EMIT_ROOT").env_remove("R3_BD_INJECT_EMITTER_FAULT");
  println!("R3_C2_SCIENTIFIC_CHILD_START phase={} candidate={} index={} seed={}",ph,ARM,index,seeds[index]);
  let rc=command_live(c,&q.join(format!("{}_CHILD_STREAM.log",ARM)))?;
  need(rc==0,"SCIENTIFIC_CHILD_NONZERO_TERMINAL")?;
  let mut obs=Command::new(project.join(OBSERVER));
  obs.arg("--one").arg(&raw).arg(ARM).arg(seeds[index].to_string()).current_dir("/");
  let result=command_live(obs,&q.join(format!("{}_OBSERVER_STREAM.log",ARM)))?;
  need(result==0,"SCIENTIFIC_EVIDENCE_INVALID_TERMINAL")?;
  let mut commit=format!("HECS_R3_SUCCESSOR_C2_SEED_COMMIT_V1\nSTART_SHA256={}\n",sha_file(&q.join("START.txt"))?);
  for f in NAMES{commit.push_str(&format!("{}_{}_SHA256={}\n",ARM,f,sha_file(&raw.join(f))?));}
  commit.push_str(&format!("{}_CHILD_LOG_SHA256={}\n",ARM,sha_file(&q.join(format!("{}_CHILD_STREAM.log",ARM)))?));
  commit.push_str(&format!("{}_OBSERVER_LOG_SHA256={}\n",ARM,sha_file(&q.join(format!("{}_OBSERVER_STREAM.log",ARM)))?));
  write_once(&q.join("COMMIT.txt"),&commit)?;
  fs::rename(&q,slot.join("COMMITTED")).map_err(|e|ioerr("ATOMIC_COMMIT_RENAME",e))?;
  sync_dir(&slot)?;
  let admitted=verify_commit(&slot,ph,index,seeds[index],last.as_deref(),&pseal)?;
  println!("PASS_R3_C2_ONE_SHOT_SCIENTIFIC_SEED_COMMITTED phase={} index={} commit_sha256={} GIT=NO",ph,index,admitted);
  Ok(())
 })();
 if let Err(ref why)=outcome {
  if exists(&q) && !exists(&q.join("FAULT.txt")) {
   let fault=format!("HECS_R3_SUCCESSOR_C2_SEED_TERMINAL_V1\nPHASE={}\nINDEX={}\nSEED={}\nSTART_SHA256={}\nREASON={}\nREPLAY_PERMITTED=NO\n",ph,index,seeds[index],sha_file(&q.join("START.txt")).unwrap_or_else(|_|"START_MISSING".into()),why);
   if let Err(err)=write_once(&q.join("FAULT.txt"),&fault){eprintln!("R3_C2_FAULT_RECORD_UNCERTAIN {}",err);}
  }
 }
 outcome
}
fn close(project:&Path,proto:&Path,supervisor_bin:&Path,phase:&str)->R<()> {
 let (ph,seeds,work_rel,_,registry)=phase_of(phase)?;
 let pseal=approval(project,proto,supervisor_bin,phase)?;
 let dpath=disposition_path(project,phase);
 need(!exists(&dpath),"DISPOSITION_REPLAY_REFUSED")?;
 let last=prefix(project,ph,seeds.len(),&pseal)?.ok_or("EMPTY_REGISTRY")?;
 // Cohort view: plain copies of the committed five-file sets (the observer refuses symlinks).
 let view=project.join(format!("build/hecs_r3_successor_c2_{}_cohort_r0",phase.to_ascii_lowercase()));
 mkdir_new(&view)?;
 for i in 0..seeds.len() {
  let d=view.join(format!("seed_{}",i));mkdir_new(&d)?;let a=d.join(ARM);mkdir_new(&a)?;
  for f in NAMES {
   let src=project.join(work_rel).join(format!("seed_{}",i)).join("COMMITTED").join("raw").join(ARM).join(f);
   let bytes=checked_file(&src)?;
   let mut out=OpenOptions::new().write(true).create_new(true).open(a.join(f)).map_err(|e|ioerr("VIEW_CREATE",e))?;
   out.write_all(&bytes).and_then(|_|out.sync_all()).map_err(|e|ioerr("VIEW_WRITE",e))?;
  }
 }
 let mut obs=Command::new(project.join(OBSERVER));
 obs.arg("--cohort").arg(&view).arg(ARM).arg(registry).current_dir("/");
 // The observer refuses unknown entries in the cohort root, so its stream log lives one level up.
 let logq=project.join(format!("build/hecs_r3_successor_c2_{}_cohort_r0.log",phase.to_ascii_lowercase()));
 let rc=command_live(obs,&logq)?;
 need(rc==0,"COHORT_OBSERVER_REFUSED")?;
 let log=String::from_utf8(checked_file(&logq)?).map_err(|_|"COHORT_LOG_UTF8")?;
 let line=log.lines().find(|l|l.starts_with(&format!("CHILD_STDOUT R3_BP_FULL_{}_PHASE ",ph))).ok_or("COHORT_PHASE_LINE_ABSENT")?;
 let disposition=line.split_whitespace().find_map(|w|w.strip_prefix("DISPOSITION=")).ok_or("COHORT_DISPOSITION_ABSENT")?;
 let record=format!("HECS_R3_SUCCESSOR_C2_DISPOSITION_V1\nPHASE={}\nRELEASE_SHA256={}\nLAST_COMMIT_SHA256={}\nCOHORT_OBSERVER_LOG_SHA256={}\nDISPOSITION={}\n",
  ph,pseal,last,sha_file(&logq)?,disposition);
 write_once(&dpath,&record)?;
 println!("PASS_R3_C2_PHASE_CLOSED phase={} disposition={} record_sha256={}",ph,disposition,sha256(record.as_bytes()));
 Ok(())
}
fn selftest()->R<()> {
 need(sha256(b"")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","SHA256_EMPTY")?;
 need(sha256(b"abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","SHA256_ABC")?;
 need(sha256(&vec![b'a';1_000_000])=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0","SHA256_MILLION_A")?;
 let all:Vec<u64>=DEV.iter().chain(QUAL.iter()).copied().collect();
 need(all.iter().copied().collect::<std::collections::HashSet<_>>().len()==20,"SEED_DUPLICATES")?;
 println!("PASS_R3_C2_SHA256_SEED_AND_EXCLUSIVE_RECORD_SELFTEST");Ok(())
}
fn main(){
 let a=std::env::args().collect::<Vec<_>>();
 let outcome=match a.as_slice(){
  [_,m] if m=="--selftest"=>selftest(),
  [_,m,ph,p,proto,bin] if m=="--write-release"=>write_release(Path::new(p),Path::new(proto),Path::new(bin),ph),
  [_,m,ph,p,proto,bin] if m=="--verify-release"=>approval(Path::new(p),Path::new(proto),Path::new(bin),ph).map(|s|println!("PASS_R3_C2_RELEASE_VERIFIED phase={} sha256={}",ph,s)),
  [_,m,ph,p,proto,bin,i] if m=="--run-seed"=>i.parse::<usize>().map_err(|_|"INVALID_INDEX".into()).and_then(|n|run_seed(Path::new(p),Path::new(proto),Path::new(bin),ph,n)),
  [_,m,ph,p,proto,bin] if m=="--close"=>close(Path::new(p),Path::new(proto),Path::new(bin),ph),
  _=>fail("USAGE: r3_c2_q2_supervisor (PHASE=QUAL2 only) --selftest | --write-release|--verify-release|--close PHASE PROJECT PROTOCOL BINARY | --run-seed PHASE PROJECT PROTOCOL BINARY INDEX"),
 };
 if let Err(e)=outcome {eprintln!("MECHANICS_OR_SCIENCE_REFUSAL_R3_C2={}",e);std::process::exit(1)}
}
