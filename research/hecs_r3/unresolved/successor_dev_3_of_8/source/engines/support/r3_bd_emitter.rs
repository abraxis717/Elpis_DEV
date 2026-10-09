use std::env;
use std::fs::{self,OpenOptions};
use std::io::{Write,Read};
use std::path::{Path,PathBuf};
fn reject(msg:&str)->!{eprintln!("MECHANICS_NONQUAL_R3_BC={msg}");std::process::exit(2)}
pub fn publish(input:&Path,output:&Path,inject:bool){
 if output.exists(){reject("REPLAY_DESTINATION_EXISTS")}
 if !input.is_dir(){reject("INPUT_MISSING")}
 let candidate=input.file_name().and_then(|x|x.to_str()).unwrap_or("");
 if candidate!="C0" && candidate!="C1"{reject("CANDIDATE_ID")}
 let names=["observations.tsv","metrics.tsv","tasks.tsv","v5.tsv","events.tsv"];
 let mut evidence=Vec::new();
 for name in names {
  let path=input.join(name);
  let meta=fs::symlink_metadata(&path).unwrap_or_else(|_|reject("SOURCE_MISSING"));
  if !meta.is_file()||meta.file_type().is_symlink(){reject("UNSAFE_SOURCE")}
  let mut data=Vec::new();fs::File::open(&path).unwrap().read_to_end(&mut data).unwrap();
  if data.is_empty()||!data.contains(&b'\n'){reject("MALFORMED_HEADER")}
  let header=data.split(|b|*b==b'\n').next().unwrap();
  if header.is_empty()||header[0]==b'\t'{reject("MALFORMED_HEADER")}
  let row_count=data.split(|b|*b==b'\n').skip(1).filter(|line|!line.is_empty()).count();
  if name=="events.tsv" && row_count!=0{reject("UNEXPECTED_SYNTHETIC_EVENT")}
  evidence.push((name.to_string(),data,row_count));
 }
 let stage=output.with_extension("r3bc_incomplete");
 if stage.exists(){reject("PARTIAL_STAGE_EXISTS")}
 fs::create_dir(&stage).unwrap_or_else(|_|reject("STAGE_CREATE"));
 for (i,(name,data,_)) in evidence.iter().enumerate(){
  let mut f=OpenOptions::new().write(true).create_new(true).open(stage.join(name)).unwrap();
  f.write_all(data).unwrap();f.sync_all().unwrap();
  if inject && i==0{reject("INJECTED_AFTER_FIRST_WRITE_NO_PUBLICATION")}
 }
 let mut manifest=String::new();
 manifest.push_str("R3_BC_NATIVE_CANDIDATE_EMITTER_V1\nsynthetic_only=true\nscientific_execution=false\nexact_flops_matched=false\n");
 manifest.push_str(&format!("candidate_id={candidate}\n"));
 for (name,data,rows) in &evidence {manifest.push_str(&format!("{name}\t{rows}\t{}\n",data.len()));}
 let mut mf=OpenOptions::new().write(true).create_new(true).open(stage.join("NATIVE_MANIFEST.txt")).unwrap();
 mf.write_all(manifest.as_bytes()).unwrap();mf.sync_all().unwrap();
 fs::rename(&stage,output).unwrap_or_else(|_|reject("PUBLICATION_RENAME"));
 if let Ok(parent)=fs::File::open(output.parent().unwrap()){let _=parent.sync_all();}
 println!("PASS_R3_BC_NATIVE_EMITTER_SYNTHETIC_CANDIDATE={} files=5",candidate);
}
fn main(){let args:Vec<String>=env::args().collect();if args.len()!=4{reject("USAGE_INPUT_OUTPUT_INJECT_ZERO_ONE")};
 if args[3]!="0"&&args[3]!="1"{reject("INJECTION_FLAG")};
 publish(&PathBuf::from(&args[1]),&PathBuf::from(&args[2]),args[3]=="1");}
