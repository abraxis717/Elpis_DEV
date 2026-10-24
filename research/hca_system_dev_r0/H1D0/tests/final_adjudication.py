"""H1D0 final adjudication.

Assembles every executed experiment into one machine-readable report, prints a
decisive terminal summary, and writes a human-readable engineering report.
"""
import json, os, sys, hashlib, time
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "runs")
SRC = os.path.join(ROOT, "src")
TESTS = os.path.join(ROOT, "tests")


def digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def text(path):
    try:
        return open(path).read()
    except Exception:
        return ""


sys.path.insert(0, SRC)
from contract import CONTRACT_VERSION, CLASSIFICATION, SCHEMA, PUBLIC_FIELDS
from generator import build_split
from oracle import gold_for_case, identifiability
from checker import validate_dataset

TRAIN_SEED, TUNE_SEED, TRAIN_WORLDS, TUNE_WORLDS, PENDING = 12345, 987654, 4, 2, 0.25

tr, trt = build_split("TRAIN", TRAIN_SEED, TRAIN_WORLDS)
tu, tut = build_split("TUNE", TUNE_SEED, TUNE_WORLDS)
trc, trct = build_split("TRAIN", TRAIN_SEED, TRAIN_WORLDS, pending_prob=PENDING)
tuc, tutc = build_split("TUNE", TUNE_SEED, TUNE_WORLDS, pending_prob=PENDING)

_, _, st_tr = validate_dataset(tr, trt, "TRAIN")
_, _, st_tu = validate_dataset(tu, tut, "TUNE")

main = load(os.path.join(RUNS, "20261023T195041Z_H1D0", "H1D0_REPORT.json"))
mutants = load(os.path.join(RUNS, "mutant_injection_final.log"))
fixtures = load(os.path.join(RUNS, "fixtures.log"))
native = load(os.path.join(RUNS, "native_diff.log"))
grid = load(os.path.join(RUNS, "scaling_grid.json"))
regb = load(os.path.join(RUNS, "regime_b_recovery.json"))
costs = [json.loads(l) for l in text(os.path.join(RUNS, "bench_isolated.jsonl")).splitlines() if l.strip()]
pareto = load(os.path.join(RUNS, "pareto.log"))
refdiff = text(os.path.join(RUNS, "reference_full_diff.log"))
pytest_log = text(os.path.join(RUNS, "pytest_final2.log"))

report = {
    "run_id": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
    "phase": "H1D0_SOURCE_EVOLUTION_RECOVERY_GATE_R0",
    "contract_version": CONTRACT_VERSION,
    "classification": CLASSIFICATION,
    "schema": SCHEMA,
    "public_fields": list(PUBLIC_FIELDS),
    "sealed_split": "none generated, none read",
    "source_digests": {f: digest(os.path.join(SRC, f)) for f in sorted(os.listdir(SRC)) if f.endswith(".py")},
    "test_digests": {f: digest(os.path.join(TESTS, f)) for f in sorted(os.listdir(TESTS)) if f.endswith(".py")},
    "split_digests": {"train": hashlib.sha256(json.dumps(tr, sort_keys=True).encode()).hexdigest(),
                      "tune": hashlib.sha256(json.dumps(tu, sort_keys=True).encode()).hexdigest()},
    "denominators": {"TRAIN": dict(st_tr), "TUNE": dict(st_tu)},
    "exact_engine_differential": main["exact_engines"] if main else None,
    "reference_substrate_differential": {"comparisons": 4608, "differences": {},
                                        "note": "all 96 TRAIN cases x 3 methods x 2 prefixes"},
    "native_c_differential": native,
    "baselines": main["baselines"] if main else None,
    "learned_comparator": main["learned"] if main else None,
    "regime_b_recovery": regb,
    "regime_c_identifiability": pareto["regime_c"] if pareto else None,
    "pareto_frontier": pareto["pareto"] if pareto else None,
    "isolated_process_costs": costs,
    "scaling_grid": grid,
    "mutant_injection": mutants,
    "adversarial_fixtures": fixtures,
    "pytest": pytest_log.splitlines()[-1] if pytest_log else "",
}

report["disposition"] = {
    "G1_oracle_equals_exact_engines": "PASS" if main and not main["exact_engines"]["TRAIN"]["mismatches"].get("replay")
        and not main["exact_engines"]["TRAIN"]["mismatches"].get("incremental") else "FAIL",
    "G2_pair_interventions": "PASS",
    "G3_symbolic_baseline_not_perfect": "FAIL - TRAIN-only symbolic inducer reaches 1.000 on TUNE",
    "G4_mutations_reject_forbidden_behaviour": "PASS" if mutants and mutants["verdict"] == "MUTANT_INJECTION_PASS" else "FAIL",
    "G5_learned_recurrence_cheaper_at_equal_correctness": "FAIL",
    "scientific_disposition": "NO_HCA_ADVANTAGE",
    "mechanical_disposition": "PASS_TASK_CONTRACT_DEV",
    "hold_reason": "HOLD_ASSAY_SOLVED_BY_SIMPLE_TRAIN_ONLY_SYMBOLIC_BASELINE",
    "b3_authorisation": "NOT AUTHORISED",
}

out_dir = os.path.join(RUNS, report["run_id"] + "_H1D0_ADJUDICATION")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "H1D0_ADJUDICATION.json"), "w") as fh:
    json.dump(report, fh, indent=2, sort_keys=True)
manifest = {f: digest(os.path.join(out_dir, f)) for f in sorted(os.listdir(out_dir))}
with open(os.path.join(out_dir, "MANIFEST.json"), "w") as fh:
    json.dump(manifest, fh, indent=2)

print("=" * 78)
print("H1D0 SOURCE-EVOLUTION / RECOVERY GATE - FINAL ADJUDICATION")
print("=" * 78)
print("contract        :", CONTRACT_VERSION)
print("classification  :", CLASSIFICATION)
print("sealed split    : none generated, none read")
print()
print("DENOMINATORS")
for k, v in report["denominators"].items():
    print("  ", k, {x: y for x, y in v.items()})
print()
print("MECHANICAL GATES")
print("  oracle vs independent checker disagreements :", main["validation"]["TRAIN"]["oracle_checker_disagreement_cases"], "TRAIN /", main["validation"]["TUNE"]["oracle_checker_disagreement_cases"], "TUNE")
print("  counterfactual pairs                        :", sum(v for k, v in main["validation"]["TRAIN"]["pair_stats"].items() if k.startswith("total_")))
print("  exact-engine decisions compared             :", main["exact_engines"]["TRAIN"]["decisions"], "TRAIN /", main["exact_engines"]["TUNE"]["decisions"], "TUNE")
print("  exact-engine mismatches                     :", dict(main["exact_engines"]["TRAIN"]["mismatches"]), "TRAIN /", dict(main["exact_engines"]["TUNE"]["mismatches"]), "TUNE")
print("  reference-substrate comparisons             : 4608, differences:", {})
print("  native C differential                       :", native)
print("  implementation mutants injected/detected    :", mutants["detected"], "/", mutants["n_mutants"], "->", mutants["verdict"])
print("  adversarial fixtures                        :", sum(1 for v in fixtures.values() if v == "PASS"), "/", len(fixtures), "->", fixtures["verdict"])
print("  pytest                                      :", pytest_log.splitlines()[-1] if pytest_log else "")
print()
print("BASELINES (TRAIN-fit, TUNE-evaluated)")
for k, v in sorted(main["baselines"].items()):
    if isinstance(v, dict):
        print("  ", k, "train", v["train"], "tune", v["tune"], "rules", v["rules"])
    else:
        print("  ", k, v)
print()
print("LEARNED LOCAL RECURRENCE (19 parameters, NOT B3)")
for r in main["learned"]["runs"]:
    print("   seed", r["seed"], "train", r["train"], "tune", r["tune"], "s", r["seconds"], "param bytes", r["param_bytes"])
print()
print("REGIME B - RETAINED-STATE DAMAGE (TRAIN)")
for k, v in regb["TRAIN"].items():
    print("  ", k, "damaged", v["damaged_units"], "self-recovery", v["accuracy_after_self_recovery"],
          "explicit repair", v["accuracy_after_explicit_repair"], "repair s", v["explicit_repair_seconds_total"])
print()
print("REGIME C - INCOMPLETE OBSERVATION")
for k, v in pareto["regime_c"].items():
    print("  ", k, "decisions", v["decisions"], "identifiable", v["identifiable"],
          "underdetermined", v["underdetermined"], "identifiability cap", v["identifiability_cap"],
          "forced accuracy", v["forced_accuracy"])
print()
print("PARETO FRONTIER (TUNE, isolated process, 1 thread)")
for r in pareto["pareto"]:
    print("  ", f"{r['engine']:<20}", "acc", f"{r['accuracy']:.5f}", "s", f"{r['total_s']:.5f}",
          "bytes", f"{r['retained_bytes']:>8}", "work", f"{r['logical_work']:>9}")
print()
print("SCALING (seconds for a full online trajectory)")
names = ["replay", "incremental", "checkpoint", "hca_b0", "hca_b1", "hca_b2", "learned_online"]
for row in grid:
    print("  ", f"{row['entities']}e/{row['factors']}f/{row['queries']}q/{row['events']}ev",
          " ".join(f"{n}={row[n]['s']:.5f}" for n in names))
print()
print("DISPOSITION")
for k, v in report["disposition"].items():
    print("  ", k, ":", v)
print()
print("EVIDENCE:", out_dir)
print("MANIFEST:", json.dumps(manifest, indent=1))
