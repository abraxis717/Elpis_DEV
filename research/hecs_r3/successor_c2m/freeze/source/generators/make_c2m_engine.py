"""Derive the C2M engine from the frozen C2 engine source by exact, counted replacements (scratch tool).
C2M = the C2 learning law applied to EVERY binding (L1/N72 h16, L1/N36 h4, TS/L2 h4, HD/L2 h4) + the R2 H2 planning
comparison over the trained ROLLOUT weights (QUERY-only native forward), written to planning.tsv next to the five files.
usage: python3 -I make_c2m_engine.py C2_ENGINE_RS OUT_RS C2M_REGISTRIES_JSON C2_REGISTRIES_JSON C2_Q2_REGISTRY_JSON"""
import json, re, sys

src, out, regp, c2p, q2p = sys.argv[1:6]
reg, c2, q2 = json.load(open(regp)), json.load(open(c2p)), json.load(open(q2p))
s = open(src).read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), count, old[:90])
    s = s.replace(old, new)


rep("//! H-ECS R3 successor C2 engine:", "//! H-ECS R3 successor C2M engine (matched constituent family): the frozen C2 engine with the C2 learning law applied\n"
    "//! to EVERY binding and the R2 H2 planning comparison over the trained ROLLOUT weights (planning.tsv). Training is\n"
    "//! offline LEARN mechanics of a qualification replicate; planning is QUERY-only native forward computation.\n"
    "//! H-ECS R3 successor C2 engine:")
rep("#[path = \"support/r3_c2_terminal.rs\"]\nmod successor_c2;\n",
    "#[path = \"support/r3_c2_terminal.rs\"]\nmod successor_c2;\n#[path = \"support/r3_c2m_planning.rs\"]\nmod planning;\n")
rep("    binding_native_refusals:&mut Vec<String>, successor:bool) -> Vec<String> {",
    "    binding_native_refusals:&mut Vec<String>, successor:bool,\n"
    "    trained:&mut Vec<(String,Vec<f64>,usize,hecs_r2::encoder::Encoder)>) -> Vec<String> {")
rep("    if let Some(cause)=c2_failed {\n",
    "    // C2M: keep each successfully measured matched ROLLOUT W for the H2 planning comparison.\n"
    "    if successor && rollout_result.is_some() {trained.push((key.clone(),rollout.clone(),width,data.encoder.clone()));}\n"
    "    if let Some(cause)=c2_failed {\n")
# registries and seals
s = re.sub(r"const SUCCESSOR_DEV_SEEDS: \[u64;8\] = \[[0-9,]*\];", "const SUCCESSOR_DEV_SEEDS: [u64;8] = [%s];" % ",".join(reg["dev_seeds"]), s, count=1)
s = re.sub(r"const SUCCESSOR_QUAL_SEEDS: \[u64;12\] = \[[0-9,]*\];",
           "const SUCCESSOR_QUAL_SEEDS: [u64;12] = [%s];\n// Executed or spent registries of the C2 line: refused in every mode.\n"
           "const SPENT_SEEDS: [u64;32] = [%s];" % (",".join(reg["qual_seeds"]),
           ",".join(c2["dev_seeds"] + c2["qual_seeds"] + q2["qual_q2_seeds"])), s, count=1)
rep('const BIN_CANDIDATE: &str = "C2";', 'const BIN_CANDIDATE: &str = "C2M";')
rep('"/mnt/primesauce/Elpis_DEV/evidence/hecs_r3_successor_c2/release/DEV_AUTHORITY.txt"',
    '"/mnt/primesauce/Elpis_DEV/evidence/hecs_r3_successor_c2m/release/DEV_AUTHORITY.txt"')
rep('"/mnt/primesauce/Elpis_DEV/evidence/hecs_r3_successor_c2/release/QUAL_AUTHORITY.txt"',
    '"/mnt/primesauce/Elpis_DEV/evidence/hecs_r3_successor_c2m/release/QUAL_AUTHORITY.txt"')
rep('lines[0]=="HECS_R3_SUCCESSOR_C2_RELEASE_V1"', 'lines[0]=="HECS_R3_SUCCESSOR_C2M_RELEASE_V1"')
rep('lines.iter().any(|x|*x=="CANDIDATES=C2")', 'lines.iter().any(|x|*x=="CANDIDATES=C2M")')
rep('lines.iter().any(|x|x.starts_with("ENGINE_BINARY_SHA256_C2="))', 'lines.iter().any(|x|x.starts_with("ENGINE_BINARY_SHA256_C2M="))')
rep("    !SUCCESSOR_DEV_SEEDS.contains(&seed) && !SUCCESSOR_QUAL_SEEDS.contains(&seed) &&\n",
    "    !SUCCESSOR_DEV_SEEDS.contains(&seed) && !SUCCESSOR_QUAL_SEEDS.contains(&seed) && !SPENT_SEEDS.contains(&seed) &&\n")
# modes
for a, b in [("--successor-dev-c2-one-seed", "--successor-dev-c2m-one-seed"), ("--successor-qual-c2-one-seed", "--successor-qual-c2m-one-seed"),
             ("--design-c2-one-seed", "--design-c2m-one-seed"), ("--mechanics-successor-c2-one-seed", "--mechanics-successor-c2m-one-seed")]:
    s = s.replace(a, b)
# every binding is a C2M successor binding in C2M modes
rep("        let mut binding_native_refusals:Vec<String>=Vec::new();\n",
    "        let mut binding_native_refusals:Vec<String>=Vec::new();\n"
    "        let mut trained:Vec<(String,Vec<f64>,usize,hecs_r2::encoder::Encoder)>=Vec::new();\n")
rep('                &mut binding_native_refusals, (mode=="--mechanics-successor-c2m-one-seed" || is_successor) && key=="HD/L2");',
    '                &mut binding_native_refusals, is_successor || (mode=="--mechanics-successor-c2m-one-seed" && key=="HD/L2"), &mut trained);')
rep("        for kind in [WorldKind::Separated,WorldKind::Matched] {\n",
    "        if is_successor {\n"
    "            // H2 planning (SEPARATED world) over the matched ROLLOUT weights; QUERY-only native forward.\n"
    "            let planning_path=Path::new(&args[3]).parent().expect(\"OUTPUT_PARENT\").join(\"planning.tsv\");\n"
    "            let pf=require(OpenOptions::new().write(true).create_new(true).open(&planning_path),\"exclusive planning output\");\n"
    "            let mut pw=BufWriter::new(pf);\n"
    "            writeln!(pw,\"seed\\tcondition\\tbudget_index\\tplanner_k\\tsuccess\\tmean_final_error\\tforward_rows_per_replan\\tsubgoal_tracking_nmse\").unwrap();\n"
    "            if [\"L1/N72\",\"L1/N36\",\"TS/L2\",\"HD/L2\"].iter().all(|k| trained.iter().any(|(t,..)| t==k)) {\n"
    "                let rows=planning::plan_conditions(&k1,&world,seed,WorldKind::Separated,&trained)\n"
    "                    .unwrap_or_else(|e| panic!(\"MECHANICS_NONQUAL_PLANNING K1_STATUS_{}\",e.0));\n"
    "                for r in rows {\n"
    "                    writeln!(pw,\"{}\\t{}\\t{}\\t{}\\t{:.17e}\\t{:.17e}\\t{:.17e}\\t{:.17e}\",seed,r.condition,r.budget_index,r.planner_k,\n"
    "                        r.success,r.mean_final_error,r.forward_rows_per_replan,r.subgoal_tracking_nmse).unwrap();\n"
    "                }\n"
    "                println!(\"R3_C2M_PLANNING_DONE seed={seed}\");\n"
    "            } else {println!(\"R3_C2M_PLANNING_NOT_RUN seed={seed} reason=A_BINDING_ROLLOUT_ARM_FAILED\");}\n"
    "            pw.flush().unwrap();\n"
    "        }\n"
    "        for kind in [WorldKind::Separated,WorldKind::Matched] {\n")
rep('let candidate=if mode=="--mechanics-one-seed"{"C0"}else{"C2"};', 'let candidate=if mode=="--mechanics-one-seed"{"C0"}else{"C2M"};')
rep("HD_L2_ONLY=YES NO_SUPERIORITY_CLAIM", "C2M_ALL_BINDINGS=YES H2_PLANNING=QUERY_ONLY_NATIVE_FORWARD NO_SUPERIORITY_CLAIM")
rep('println!("PHASE=R3_SUCCESSOR_C2_{} candidate', 'println!("PHASE=R3_SUCCESSOR_C2M_{} candidate')
rep("PASS_R3_C2_PROSPECTIVE_ENGINE_PREVIEW candidate=C2 ", "PASS_R3_C2M_PROSPECTIVE_ENGINE_PREVIEW candidate=C2M ")
# C2M terminal weight per consumed horizon (DESIGN selection, DESIGN_RECORD.json): w_T(h) = (4/h)^1.5, exact literals.
rep("const C2_TERMINAL_WEIGHT: f64 = 1.0;\n",
    "const C2_TERMINAL_WEIGHT: f64 = 1.0;\n"
    "// C2M: terminal weight per consumed horizon, w_T(h) = (4/h)^C2M_TERMINAL_EXPONENT as exact literals; w_T(4) is the\n"
    "// qualified C2 weight (1.0), w_T(16) = 0.125 (DESIGN selection). Any other horizon is undeclared and refused.\n"
    "const C2M_TERMINAL_EXPONENT: f64 = 1.5;\n"
    "const C2M_TERMINAL_WEIGHT_H16: f64 = 0.125;\n"
    "fn c2m_terminal_weight(h:usize)->f64 {\n"
    "    match h {4=>C2_TERMINAL_WEIGHT,16=>C2M_TERMINAL_WEIGHT_H16,_=>panic!(\"MECHANICS_NONQUAL C2M_UNDECLARED_HORIZON h={h}\")}\n"
    "}\n")
rep("successor_c2::c2_step(&mut rollout,width,&batch,h,variance,LAMBDA,C2_TERMINAL_WEIGHT,rate,CLIP)",
    "successor_c2::c2_step(&mut rollout,width,&batch,h,variance,LAMBDA,c2m_terminal_weight(h),rate,CLIP)")
rep("loss = R3 rollout objective (LAMBDA) + C2_TERMINAL_WEIGHT * true-target terminal",
    "loss = R3 rollout objective (LAMBDA) + c2m_terminal_weight(h) * true-target terminal")
rep('C2_BASE_LAMBDA={} C2M_ALL_BINDINGS=YES H2_PLANNING=QUERY_ONLY_NATIVE_FORWARD NO_SUPERIORITY_CLAIM",\n        BIN_CANDIDATE,C2_UPDATES,C2_TERMINAL_WEIGHT,C2_WINDOWS,LAMBDA);}',
    'C2_BASE_LAMBDA={} C2M_TERMINAL_EXPONENT={} C2M_TERMINAL_WEIGHT_H16={} C2M_ALL_BINDINGS=YES H2_PLANNING=QUERY_ONLY_NATIVE_FORWARD NO_SUPERIORITY_CLAIM",\n        BIN_CANDIDATE,C2_UPDATES,C2_TERMINAL_WEIGHT,C2_WINDOWS,LAMBDA,C2M_TERMINAL_EXPONENT,C2M_TERMINAL_WEIGHT_H16);}')
open(out, "x").write(s)
print("C2M engine written", out)
