"""Assemble the C2M supervisor from the C2 supervisor template parts (head + verbatim R3-BQ helpers + pump + body) with
counted replacements, then fill pins (scratch tool).
usage: python3 -I make_c2m_supervisor.py OUT_RS C2M_REGISTRIES_JSON KEY=VALUE...  (ENGINE OBSERVER K1 PROTOCOL DESIGN
       SRC_ENGINE SRC_TERMINAL SRC_PLANNING SRC_OBSERVER REGISTRIES LIBTREE)"""
import json, sys

d = "/tmp/claude-0/-home-user-Elpis-DEV/8df46079-3148-5dbf-b5bf-abf159f57c90/scratchpad/r4/"
out, regp = sys.argv[1:3]
pins = dict(a.split("=", 1) for a in sys.argv[3:])
reg = json.load(open(regp))
s = "".join(open(d + p).read() for p in ("sup_head.rs.part", "sup_helpers.rs.part", "sup_pump.rs.part", "sup_body.rs.part"))


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), count, old[:90])
    s = s.replace(old, new)


rep("// H-ECS R3 successor C2 scientific supervisor (DEV and QUAL); std only. RESEARCH_ONLY.\n",
    "// H-ECS R3 successor C2M scientific supervisor (matched constituent family: DEV, then QUAL with H2 hierarchy\n"
    "// adjudication); derived from the C2 supervisor template; std only. RESEARCH_ONLY. Training inside a seed is an\n"
    "// offline LEARN qualification replicate, never per-turn QUERY work.\n")
rep('const NAMES:[&str;5]=["observations.tsv","metrics.tsv","tasks.tsv","v5.tsv","events.tsv"];',
    '// The engine receives the first five as argv outputs and writes planning.tsv beside them; all six are committed.\n'
    'const NAMES:[&str;6]=["observations.tsv","metrics.tsv","tasks.tsv","v5.tsv","events.tsv","planning.tsv"];')
rep("  for f in NAMES{c.arg(raw.join(f));}c.arg(index.to_string());", "  for f in &NAMES[..5]{c.arg(raw.join(f));}c.arg(index.to_string());")
rep('const ARM:&str="C2";', 'const ARM:&str="C2M";')
for a, b, n in [("hecs_r3_successor_c2_", "hecs_r3_successor_c2m_", None), ("evidence/hecs_r3_successor_c2/", "evidence/hecs_r3_successor_c2m/", None),
                ("r3_bn_c2_dev_engine", "r3_bn_c2m_dev_engine", None), ("r3_c2_observer", "r3_c2m_observer", None),
                ('include_bytes!("r3_c2_supervisor.rs")', 'include_bytes!("r3_c2m_supervisor.rs")', 1),
                ('"C2_DEV"', '"C2M_DEV"', 1), ('"C2_QUAL"', '"C2M_QUAL"', 1),
                ("HECS_R3_SUCCESSOR_C2_", "HECS_R3_SUCCESSOR_C2M_", None), ("CANDIDATES=C2\\n", "CANDIDATES=C2M\\n", 1),
                ("ENGINE_BINARY_SHA256_C2=", "ENGINE_BINARY_SHA256_C2M=", 1), ("ENGINE_SOURCE_SHA256_C2\"", "ENGINE_SOURCE_SHA256_C2M\"", 1),
                ("PROTOCOL_R3_C2.md", "PROTOCOL_R3_C2M.md", 1), ("C2_REGISTRIES.json", "C2M_REGISTRIES.json", 1),
                ("--successor-dev-c2-one-seed", "--successor-dev-c2m-one-seed", 1), ("--successor-qual-c2-one-seed", "--successor-qual-c2m-one-seed", 1)]:
    assert s.count(a) >= 1 and (n is None or s.count(a) == n), (a, s.count(a))
    s = s.replace(a, b)
rep("const SOURCES:[(&str,&str,&str);9]=[", "const SOURCES:[(&str,&str,&str);10]=[")
rep(' ("TRAINING_CORE_SOURCE_SHA256",',
    ' ("C2M_PLANNING_SOURCE_SHA256","build/hecs_r3_successor_c2m_engine_r0/native_source/src/bin/support/r3_c2m_planning.rs","@@SRC_PLANNING@@"),\n ("TRAINING_CORE_SOURCE_SHA256",')
rep('C2_BINDING=HD/L2\\n', 'C2M_LAW_BINDINGS=L1/N72:16,L1/N36:4,TS/L2:4,HD/L2:4\\nC2M_TERMINAL_WEIGHT_BY_HORIZON=4:1.0,16:0.125\\nC2M_TERMINAL_EXPONENT=1.5\\n'
    'H2_CONDITIONS=FLAT_N72,TEMPORAL_SHARED_L2,HIERARCHICAL_DISTINCT_L2\\nH2_LAW=R2_SPEC_12_H2A_0.15_H2B_0.10_H2C_0.10_AT_2_OF_3_BUDGETS\\n'
    'LEARN_IS_OFFLINE_QUALIFICATION_REPLICATE_NOT_QUERY=YES\\nNO_RUNTIME_READINESS_CLAIM=YES\\n')
rep('OPERATOR_AUTHORIZATION=ONE_QUAL_EXECUTION_CONDITIONAL_ON_FRESH_DEV_PASS_EVERY_SEED',
    'OPERATOR_AUTHORIZATION=ONE_QUAL_EXECUTION_WITH_H2_ADJUDICATION_CONDITIONAL_ON_FRESH_DEV_PASS_EVERY_SEED')
rep('OPERATOR_AUTHORIZATION=ONE_FRESH_DEV_EXECUTION_OF_THE_FROZEN_C2_CANDIDATE', 'OPERATOR_AUTHORIZATION=ONE_FRESH_DEV_EXECUTION_OF_THE_FROZEN_C2M_FAMILY')
rep(" let last=prefix(project,ph,index,&pseal)?;", " let last=prefix(project,phase,index,&pseal)?;")
rep(' let last=prefix(project,ph,seeds.len(),&pseal)?.ok_or("EMPTY_REGISTRY")?;', ' let last=prefix(project,phase,seeds.len(),&pseal)?.ok_or("EMPTY_REGISTRY")?;')
rep(' let record=format!("HECS_R3_SUCCESSOR_C2M_DISPOSITION_V1\\nPHASE={}\\nRELEASE_SHA256={}\\nLAST_COMMIT_SHA256={}\\nCOHORT_OBSERVER_LOG_SHA256={}\\nDISPOSITION={}\\n",\n  ph,pseal,last,sha_file(&logq)?,disposition);',
    ' let h2=log.lines().find(|l|l.starts_with("CHILD_STDOUT R3_C2M_H2 ")).ok_or("H2_LINE_ABSENT")?;\n'
    ' let hierarchy=h2.split_whitespace().find_map(|w|w.strip_prefix("HIERARCHY=")).ok_or("H2_HIERARCHY_ABSENT")?;\n'
    ' let outcome=h2.split_whitespace().find_map(|w|w.strip_prefix("OUTCOME=")).ok_or("H2_OUTCOME_ABSENT")?;\n'
    ' let record=format!("HECS_R3_SUCCESSOR_C2M_DISPOSITION_V1\\nPHASE={}\\nRELEASE_SHA256={}\\nLAST_COMMIT_SHA256={}\\nCOHORT_OBSERVER_LOG_SHA256={}\\nDISPOSITION={}\\nHIERARCHY={}\\nH2_OUTCOME={}\\n",\n  ph,pseal,last,sha_file(&logq)?,disposition,hierarchy,outcome);')
rep('"USAGE: r3_c2_supervisor', '"USAGE: r3_c2m_supervisor')
for k, v in {"@@DEV@@": ",".join(reg["dev_seeds"]), "@@QUAL@@": ",".join(reg["qual_seeds"]), "@@ENGINE@@": pins["ENGINE"],
             "@@OBSERVER@@": pins["OBSERVER"], "@@K1@@": pins["K1"], "@@DESIGN@@": pins["DESIGN"], "@@SRC_ENGINE@@": pins["SRC_ENGINE"],
             "@@SRC_TERMINAL@@": pins["SRC_TERMINAL"], "@@SRC_PLANNING@@": pins["SRC_PLANNING"], "@@SRC_OBSERVER@@": pins["SRC_OBSERVER"],
             "@@REGISTRIES@@": pins["REGISTRIES"], "@@LIBTREE@@": pins["LIBTREE"]}.items():
    rep(k, v)
rep("@@PROTOCOL@@", pins["PROTOCOL"], count=2)
open(out, "x").write(s)
print("C2M supervisor written", out)
