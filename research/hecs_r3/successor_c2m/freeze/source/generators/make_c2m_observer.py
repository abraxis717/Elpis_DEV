"""Derive the C2M observer from the frozen C2 observer by counted replacements (scratch tool).
Per-seed five-file arithmetic and the R3-BP cohort validity law are unchanged. Added for candidate C2M only: the
planning.tsv census and the R2 H2 law (hecs_r2 experiment::evaluate: mean SEPARATED success per condition/budget,
H2A>=0.15, H2B>=0.10, H2C>=0.10 at >=2 of 3 budgets; outcome DISTINCT > TEMPORAL > UNATTRIBUTED > NO_ADVANTAGE),
adjudicated only on QUAL with validity WORLD_MODEL_VALID.
usage: python3 -I make_c2m_observer.py C2_OBSERVER_RS OUT_RS C2M_REGISTRIES_JSON"""
import json, sys

src, out, regp = sys.argv[1:4]
reg = json.load(open(regp))
s = open(src).read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), count, old[:90])
    s = s.replace(old, new)


rep("//! R3 successor C2 observer:", "//! R3 successor C2M observer: the C2 observer plus, for candidate C2M only, the planning.tsv census and the R2 H2\n"
    "//! hierarchy law (hecs_r2 experiment::evaluate), adjudicated only on QUAL with validity WORLD_MODEL_VALID.\n//! R3 successor C2 observer:")
rep("const C2_QUAL_SEEDS:[u64;12]=[", "const C2M_DEV_SEEDS:[u64;8]=[%s];\nconst C2M_QUAL_SEEDS:[u64;12]=[%s];\nconst C2_QUAL_SEEDS:[u64;12]=["
    % (",".join(reg["dev_seeds"]), ",".join(reg["qual_seeds"])))
rep('"C2_QUAL"=>Ok(("QUAL",&C2_QUAL_SEEDS)),', '"C2_QUAL"=>Ok(("QUAL",&C2_QUAL_SEEDS)),"C2M_DEV"=>Ok(("DEV",&C2M_DEV_SEEDS)),"C2M_QUAL"=>Ok(("QUAL",&C2M_QUAL_SEEDS)),')
rep("struct ResultSeed {seed:u64,task_local:bool,s1:bool,m:bool,oracle:[f64;2],centroid:[f64;2],v3:bool,m_fail:usize,events:usize}",
    "struct ResultSeed {seed:u64,task_local:bool,s1:bool,m:bool,oracle:[f64;2],centroid:[f64;2],v3:bool,m_fail:usize,events:usize,plan:Option<[[f64;3];3]>}\n"
    "// H2 conditions in hecs_r2 spec::H2_CONDITIONS order, and their spec::budgets (planner K per budget index).\n"
    "const H2_CONDITIONS:[(&str,[u64;3]);3]=[(\"FLAT_N72\",[8,32,128]),(\"TEMPORAL_SHARED_L2\",[16,64,256]),(\"HIERARCHICAL_DISTINCT_L2\",[16,64,256])];\n"
    "const PLAN_HEADER:&str=\"seed\\tcondition\\tbudget_index\\tplanner_k\\tsuccess\\tmean_final_error\\tforward_rows_per_replan\\tsubgoal_tracking_nmse\";\n"
    "const PLAN_EPISODES:f64=24.0;\n"
    "fn planning(dir:&Path,seed:u64,events:usize)->R<Option<[[f64;3];3]>>{\n"
    " let p=rows(&dir.join(\"planning.tsv\"),PLAN_HEADER,8)?;\n"
    " if p.is_empty(){demand(events>0,\"PLANNING_ABSENT_WITHOUT_FAILURE_EVENT\")?;return Ok(None);}\n"
    " demand(p.len()==9,\"PLANNING_ROW_CENSUS\")?;\n"
    " let mut m=[[f64::NAN;3];3];let mut seen=HashSet::new();\n"
    " for r in &p {\n"
    "  demand(u(&r[0])?==seed,\"PLANNING_SEED\")?;\n"
    "  let c=H2_CONDITIONS.iter().position(|(n,_)|*n==r[1]).ok_or(\"PLANNING_UNKNOWN_CONDITION\")?;\n"
    "  let b=u(&r[2])? as usize;demand(b<3&&seen.insert((c,b)),\"PLANNING_BUDGET_INDEX\")?;\n"
    "  demand(u(&r[3])?==H2_CONDITIONS[c].1[b],\"PLANNING_K_DRIFT\")?;\n"
    "  let sc=f(&r[4])?;demand(sc.is_finite()&&(0.0..=1.0).contains(&sc)&&(sc*PLAN_EPISODES-(sc*PLAN_EPISODES).round()).abs()<1e-9,\"PLANNING_SUCCESS_DOMAIN\")?;\n"
    "  let _=f(&r[5])?;let _=f(&r[6])?;let _=f(&r[7])?;\n"
    "  m[c][b]=sc;\n"
    " }\n"
    " Ok(Some(m))\n"
    "}\n")
rep('demand(candidate=="C0"||candidate=="C1"||candidate=="C2","CANDIDATE_INVALID")?;',
    'demand(candidate=="C0"||candidate=="C1"||candidate=="C2"||candidate=="C2M","CANDIDATE_INVALID")?;')
rep('demand(NAMES.contains(&name.as_str())||name=="CHILD_STREAM.log","UNAUTHORIZED_EVIDENCE_FILE")?;',
    'demand(NAMES.contains(&name.as_str())||name=="CHILD_STREAM.log"||(candidate=="C2M"&&name=="planning.tsv"),"UNAUTHORIZED_EVIDENCE_FILE")?;')
# Rehearsal finding (DESIGN root, 2026-10-10): the cohort arm admission must also admit C2M.
rep('demand(arm=="C0"||arm=="C1"||arm=="C2","COHORT_ARM")?;', 'demand(arm=="C0"||arm=="C1"||arm=="C2"||arm=="C2M","COHORT_ARM")?;')
rep(" Ok(ResultSeed{seed,task_local:task,s1,m,oracle:oracles,centroid:centroids,v3,m_fail,events:events.len()})",
    " let plan=if candidate==\"C2M\" {demand(entries.contains(\"planning.tsv\"),\"PLANNING_FILE_ABSENT\")?;planning(dir,seed,events.len())?} else {None};\n"
    " Ok(ResultSeed{seed,task_local:task,s1,m,oracle:oracles,centroid:centroids,v3,m_fail,events:events.len(),plan})")
rep('demand(if registry_name=="R3_SUCCESSOR_DEV"{arm=="C0"||arm=="C1"}else{arm=="C2"},"COHORT_ARM_REGISTRY_MISMATCH")?;',
    'demand(if registry_name=="R3_SUCCESSOR_DEV"{arm=="C0"||arm=="C1"}else if registry_name.starts_with("C2M_"){arm=="C2M"}else{arm=="C2"},"COHORT_ARM_REGISTRY_MISMATCH")?;')
rep("  resultline(arm,&result);xs.push(result);",
    "  resultline(arm,&result);\n"
    "  if let Some(p)=&result.plan {println!(\"R3_C2M_PLANNING_PER_SEED seed={} FLAT_N72={:?} TEMPORAL_SHARED_L2={:?} HIERARCHICAL_DISTINCT_L2={:?}\",result.seed,p[0],p[1],p[2]);}\n"
    "  xs.push(result);")
rep(" Ok(())\n}\nfn main()", " if arm==\"C2M\" {h2(&xs,phase,disposition);}\n Ok(())\n}\n"
    "// hecs_r2 experiment::evaluate H2 law: succ = mean over seeds; diff per budget; holds at >= 2 of 3 budgets.\n"
    "fn h2(xs:&[ResultSeed],phase:&str,disposition:&str){\n"
    " let plans:Vec<[[f64;3];3]>=xs.iter().filter_map(|x|x.plan).collect();\n"
    " let adjudicated=phase==\"QUAL\"&&disposition==\"WORLD_MODEL_VALID\"&&plans.len()==xs.len();\n"
    " if plans.len()!=xs.len() {println!(\"R3_C2M_H2 phase={} HIERARCHY=HIERARCHY_NOT_ADJUDICATED OUTCOME=NOT_ADJUDICATED REASON=PLANNING_INCOMPLETE\",phase);return;}\n"
    " let mean=|v:&[f64]|v.iter().sum::<f64>()/v.len() as f64;\n"
    " let succ=|c:usize,b:usize|mean(&plans.iter().map(|p|p[c][b]).collect::<Vec<_>>());\n"
    " let diff=|a:usize,b:usize|->Vec<f64>{(0..3).map(|k|succ(a,k)-succ(b,k)).collect()};\n"
    " let (h2a_by,h2b_by,h2c_by)=(diff(2,0),diff(2,1),diff(1,0));\n"
    " let holds=|v:&[f64],t:f64|v.iter().filter(|d|**d>=t).count()>=2;\n"
    " let (h2a,h2b,h2c)=(holds(&h2a_by,0.15),holds(&h2b_by,0.10),holds(&h2c_by,0.10));\n"
    " let outcome=if !adjudicated{\"NOT_ADJUDICATED\"}else if h2a&&h2b{\"DISTINCT_ADVANTAGE\"}else if h2c{\"TEMPORAL_ADVANTAGE\"}else if h2a{\"UNATTRIBUTED_ADVANTAGE\"}else{\"NO_ADVANTAGE_OVER_WIDTH\"};\n"
    " let hierarchy=if adjudicated{\"HIERARCHY_ADJUDICATED\"}else if phase==\"DEV\"{\"DEV_NON_BINDING\"}else{\"HIERARCHY_NOT_ADJUDICATED\"};\n"
    " println!(\"R3_C2M_H2 phase={} HIERARCHY={} OUTCOME={} H2A={} H2B={} H2C={} H2A_BY={:?} H2B_BY={:?} H2C_BY={:?} SUCCESS_FLAT_N72={:?} SUCCESS_TEMPORAL_SHARED_L2={:?} SUCCESS_HIERARCHICAL_DISTINCT_L2={:?} INTEGRATION_AUTHORIZED=NO\",\n"
    "  phase,hierarchy,outcome,h2a as u8,h2b as u8,h2c as u8,h2a_by,h2b_by,h2c_by,(0..3).map(|b|succ(0,b)).collect::<Vec<_>>(),(0..3).map(|b|succ(1,b)).collect::<Vec<_>>(),(0..3).map(|b|succ(2,b)).collect::<Vec<_>>());\n"
    "}\nfn main()")
rep(".chain(C2_QUAL_SEEDS.iter()).copied().collect();\n  assert_eq!(all.iter().collect::<HashSet<_>>().len(),28);",
    ".chain(C2_QUAL_SEEDS.iter()).chain(C2M_DEV_SEEDS.iter()).chain(C2M_QUAL_SEEDS.iter()).copied().collect();\n  assert_eq!(all.iter().collect::<HashSet<_>>().len(),48);")
open(out, "x").write(s)
print("C2M observer written", out)
