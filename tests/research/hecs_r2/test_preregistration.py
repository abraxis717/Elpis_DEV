"""H-ECS R2 preregistration and evidence guards (research/hecs_r2/PREREGISTRATION.md). SCIENTIFIC lane; seconds.

They keep the frozen authority honest without running any science: the seeds are the documented SHA-256
derivation, disjoint from R0's and R1's, and recorded as exact decimal strings; every R2 source file is either
byte-identical to R1's or declared changed with a reason; R0's and R1's closed authority and evidence are
byte-for-byte unchanged. Raw runner output is telemetry and is never committed: each phase keeps one compact record
(evidence/<phase>/RECORD.json) that attests the raw output by SHA-256 and byte count, names its execution source
and engine, and carries the decisive numbers from which every recorded decision is recomputed here under the
frozen law.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LAB = REPO / "research" / "hecs_r2"
R1 = REPO / "research" / "hecs_r1"
SPEC = json.loads((LAB / "specs" / "hecs-r2.v1.spec.json").read_text())
GATES = SPEC["gates"]
EVIDENCE = LAB / "evidence"
R0_SEEDS = {0, 1, 2, 3, 100, 101, 102, 103, 104, 105, 106, 107, 9999}
WORLDS = ("SEPARATED", "MATCHED")
BINDING = [("L1/N72", 16), ("L1/N36", 4), ("TS/L2", 4), ("HD/L2", 4)]
BINDING_KEYS = [k for k, _ in BINDING]

# The closed R0 and R1 authority, byte for byte (the merge base 02fa89e of this work; R1's results document is
# the one permitted R1 edit and is not pinned here).
CLOSED = {
    "research/hecs_r0/PREREGISTRATION.md": "b0d3dbb8da8a118be979f4f0ee984d10fcc85373044390d82e53c9d174c76530",
    "research/hecs_r0/specs/hecs-r0.v1.spec.json": "ddb868917e22d77515653aebc83c771c39c0aa4d11a6924df1e5ce4887a31ce0",
    "research/hecs_r0/evidence/dev/hecs-r0.v1.calibration.json":
        "d6d33a93cd97f2129f90c1375563f775d7a7ad920862bce3986e185853b3f3e2",
    "research/hecs_r1/PREREGISTRATION.md": "dc1d2d5735c25dd7648f0ea2ee5ab6eb6074dfa0e185f5931e1066bdc142ce18",
    "research/hecs_r1/SOURCE_EQUIVALENCE.json": "1480925f42829c23c74ecacdf96a8c891d0444e75160c1d77207947c1affa4e0",
    "research/hecs_r1/specs/hecs-r1.v1.spec.json": "af889a11933c0f3cd9203a90008111ceaf1e8668dd620fb2f45aac6ca3daaa00",
    "research/hecs_r1/evidence/design/hecs-r1.v1.design.json":
        "23a27a085fe62102121ac08f46e231251852389dabafe4b22895131472511dfd",
    "research/hecs_r1/evidence/dev/hecs-r1.v1.calibration.json":
        "eb03d6dd3bf1cf3c805476a582f4f459cd0487147bea763e5cacf5df9c3b3a98",
    "research/hecs_r1/evidence/dev/hecs-r1.v1.dev.json":
        "3e897d5c23d87b332707816eba141230653ce87f6322c53db7fe79c2f834f9ca",
    "research/hecs_r1/evidence/qual/hecs-r1.v1.qual.json":
        "fa9b1441a06faea2fac1e3902fe537f6403d4c51c6cf5f1554b0adf23eb88938",
}

# The raw runner output each phase produced (not committed): name -> (phase, sha256, bytes).
RAW = {
    "hecs-r2.v1.design.json": ("DESIGN", "fb786f4fa6691c950afdd856c251dd937433b8bec65c80b34425a682c05a95bd", 457980),
    "hecs-r2.v1.calibration.json":
        ("DEV_CALIBRATION", "db4f375d68fe7b0c5ea55ff248a2092ff89f0ad56bddee9419207fc112ad0c00", 80322),
    "hecs-r2.v1.dev.json": ("DEV", "5dac78e9eccb4af58afb9bd30771ece8ffc5157044ec8ccf0a66f98551ae35b8", 1409992),
    "hecs-r2.v1.qual.json": ("QUAL", "5377637677672a6e03f323baf4fa9fefec58c17bf5dfae119a818db3a472d9ac", 2690878),
}
PHASE_RAW = {"design": ["hecs-r2.v1.design.json"], "dev": ["hecs-r2.v1.calibration.json", "hecs-r2.v1.dev.json"],
             "qual": ["hecs-r2.v1.qual.json"]}
# The commit whose tree built the hecs2 binary that ran each phase. b4b84ed is in the final history; aa918b3 and
# c1d38d2 belong to the pre-cleanup lineage, whose hecs2 sources the records pin file by file.
EXECUTED_FROM = {"design": "b4b84ed45f1503e4324abd50f5bd5e7b2320b47b",
                 "dev": "aa918b37eeb17b1b24d6809be25acbabb8472504",
                 "qual": "c1d38d220f4bc557d7b52d03dcc680e37306451e"}
TESTS_RS_AT_EXECUTION = "2d94d222d55e02209cd0bf976f6edb9a38122c1fd6f7a3aa689073edcc3331d6"
K1_LIBRARY_SHA256 = "a7dd1f6415e57a34a4f62ff978764176be8e89493fedb8afd42e80340d85c3b8"
HECS2_BINARY_SHA256 = "6130aff50fbcb81da2488372defafb3c8ea80ad0d4efc761e2dec0e40785c0d3"
DONOR = "fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb"
RECORDS = ["design/RECORD.json", "dev/RECORD.json", "qual/RECORD.json"]
MAX_RECORD_BYTES = 64 * 1024
MAX_EVIDENCE_BYTES = 256 * 1024


def _seed(phase: str, i: int) -> int:
    return int.from_bytes(hashlib.sha256(f"elpis.hecs-r2.{phase}.v1:{i}".encode()).digest()[:8], "big")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(phase: str) -> dict:
    return json.loads((EVIDENCE / phase / "RECORD.json").read_text())


def _seeds(phase: str) -> list[str]:
    return SPEC[f"{phase}_seeds"]


def _is_hex(value, n: int) -> bool:
    return isinstance(value, str) and len(value) == n and all(c in "0123456789abcdef" for c in value)


def test_seeds_are_the_documented_derivation_fresh_disjoint_and_exact_strings():
    r1 = json.loads((R1 / "specs" / "hecs-r1.v1.spec.json").read_text())
    r1_seeds = {int(s) for p in ("dev", "qual", "design") for s in r1[f"{p}_seeds"]}
    for phase, n in (("dev", 4), ("qual", 8), ("design", 4)):
        recorded = _seeds(phase)
        assert all(isinstance(s, str) and s.isdigit() for s in recorded), phase   # no sign, no float, no wrap
        assert [int(s) for s in recorded] == [_seed(phase, i) for i in range(n)]
        assert all(str(int(s)) == s and int(s) < 2**64 for s in recorded)
    seeds = [int(s) for p in ("dev", "qual", "design") for s in _seeds(p)]
    assert len(set(seeds)) == len(seeds) and not set(seeds) & R0_SEEDS and not set(seeds) & r1_seeds
    assert any(s > 2**63 - 1 for s in seeds)   # an i64 record would have wrapped: the strings are necessary


def test_sources_are_r1_byte_for_byte_except_the_declared_changes():
    record = json.loads((LAB / "SOURCE_EQUIVALENCE.json").read_text())
    listed = {row["path"] for row in record["files"]}
    actual = {p.relative_to(LAB / "rust").as_posix() for p in (LAB / "rust" / "src").glob("*.rs")}
    actual |= {"Cargo.toml", "Cargo.lock", "rustfmt.toml"}
    assert listed == actual
    for row in record["files"]:
        assert _sha(LAB / "rust" / row["path"]) == row["r2_sha256"], row["path"]
        if row["status"] == "NEW":
            assert row["r1_sha256"] is None and row["reason"]
            continue
        assert _sha(R1 / "rust" / row["path"]) == row["r1_sha256"], f"R1 changed: {row['path']}"
        if row["status"] == "IDENTICAL":
            assert row["r1_sha256"] == row["r2_sha256"]
        else:
            assert row["status"] == "MODIFIED" and row["reason"] and row["r1_sha256"] != row["r2_sha256"]
    # The ECS binding and the level model (one-step training) are R1's bytes: the cubic mathematics is untouched.
    status = {row["path"]: row["status"] for row in record["files"]}
    for path in ("src/k1.rs", "src/model.rs", "src/world.rs", "src/encoder.rs", "src/hierarchy.rs", "src/planner.rs",
                 "src/validity.rs", "src/analysis.rs"):
        assert status[path] == "IDENTICAL", path


def test_the_closed_r0_and_r1_authority_and_evidence_are_unchanged():
    for rel, digest in CLOSED.items():
        assert _sha(REPO / rel) == digest, f"closed authority changed: {rel}"


def test_inherited_specification_and_gates_are_r1s():
    r1 = json.loads((R1 / "specs" / "hecs-r1.v1.spec.json").read_text())
    for key in ("worlds", "data"):
        assert SPEC[key] == r1[key], key
    # No status label inside the specification: the freeze commit is the status.
    assert SPEC["labels"] == [x for x in r1["labels"] if x != "PREREGISTERED"]
    # R1's conditions with the consumed horizon added per level.
    stripped = [{**c, "levels": [{k: v for k, v in lv.items() if k != "consumed_horizon"} for lv in c["levels"]]}
                for c in SPEC["conditions"]]
    assert stripped == r1["conditions"]
    assert {k: v for k, v in SPEC["planning"].items() if k != "cost"} == r1["planning"]
    g1 = r1["gates"]
    assert (GATES["T1A_MIN_EXPLAINED"], GATES["T1B_FLOOR_TOL_ABS"], GATES["T1B_FLOOR_TOL_REL"]) == (
        g1["V1A_MIN_EXPLAINED"], g1["V1B_FLOOR_TOL_ABS"], g1["V1B_FLOOR_TOL_REL"])
    for k in ("V2_ORACLE_SUCCESS_MIN", "V3_L1_PROBE_NMSE_MAX", "V4_SEPARATED_CENTROID_RATIO_MIN",
              "V4_MATCHED_CENTROID_RATIO_MAX", "H2A_DISTINCT_OVER_WIDTH", "H2B_DISTINCT_OVER_TEMPORAL",
              "H2_BUDGETS_REQUIRED", "DEPTH_EFFECT"):
        r1_key = {"H2A_DISTINCT_OVER_WIDTH": "H2A_OVER_WIDTH_MIN", "H2B_DISTINCT_OVER_TEMPORAL": "H2B_OVER_TEMPORAL_MIN"}.get(k, k)
        assert GATES[k] == g1[r1_key], k
    # The calibration margin is real: the chosen budget lies at least one grid step past measured convergence.
    assert GATES["CAL_GRID_STEPS_ABOVE"] >= 1 and 0 < GATES["CAL_MAX_IMPROVEMENT"] <= 0.01
    # M binds every seed; the one-step stage sits above M's capture bound.
    assert GATES["M_SEED_FRACTION"] == 1.0 and GATES["S1_MIN_CAPTURE"] > GATES["M_MIN_CAPTURE"]
    # No tangent-growth, excursion or amplification quantity is a gate.
    assert not [k for k in GATES if any(w in k for w in ("SIGMA", "TANGENT", "EXCURSION", "AMPL", "LYAPUNOV"))]
    # The mathematical donor is pinned in the specification and the source-equivalence record.
    donor = json.loads((LAB / "SOURCE_EQUIVALENCE.json").read_text())["mathematical_donor"]
    assert donor["commit"] == DONOR in SPEC["measurements"]["donor"]
    # Every consumed horizon is measured.
    for c in SPEC["conditions"]:
        for lv in c["levels"]:
            assert lv["consumed_horizon"] in SPEC["measurements"]["horizons"]


# -- Evidence surface: one compact record per phase, never raw runner output ----------------------------------------

def test_the_evidence_tree_holds_only_compact_phase_records():
    files = sorted(p.relative_to(EVIDENCE).as_posix() for p in EVIDENCE.rglob("*") if p.is_file())
    assert "design/RECORD.json" in files and set(files) <= set(RECORDS), files
    assert not [f for f in files if Path(f).name in RAW], "raw runner output is telemetry, never evidence"
    sizes = [(EVIDENCE / f).stat().st_size for f in files]
    assert max(sizes) < MAX_RECORD_BYTES and sum(sizes) < MAX_EVIDENCE_BYTES
    # The raw output names cannot be staged by accident.
    ignored = set((REPO / ".gitignore").read_text().split())
    assert set(RAW) <= ignored


def test_every_record_is_bound_to_spec_seeds_raw_output_execution_source_and_engine():
    compiled = {p.relative_to(LAB / "rust").as_posix() for p in (LAB / "rust" / "src").glob("*.rs")}
    compiled = (compiled - {"src/tests.rs"}) | {"Cargo.toml", "Cargo.lock"}
    for path in sorted(EVIDENCE.glob("*/RECORD.json")):
        phase = path.parent.name
        record = json.loads(path.read_text())
        assert record["schema"] == "elpis.hecs-r2.record.v1" and record["phase"] == phase.upper()
        assert record["specification"] == {"path": "specs/hecs-r2.v1.spec.json", "name": SPEC["name"],
                                           "sha256": _sha(LAB / "specs" / "hecs-r2.v1.spec.json")}
        assert record["seeds"] == _seeds(phase) and record["mathematical_donor"] == DONOR
        # Every ECS-free references block, attested by digest; tests.rs recomputes each from its recorded seed.
        assert [(r["world"], r["seed"]) for r in record["references"]] == [(w, s) for w in WORLDS for s in _seeds(phase)]
        assert all(_is_hex(r["references_sha256"], 64) for r in record["references"])
        assert [r["name"] for r in record["raw_evidence"]] == PHASE_RAW[phase]
        for raw in record["raw_evidence"]:
            assert (raw["phase"], raw["sha256"], raw["bytes"]) == RAW[raw["name"]] and raw["committed"] is False
        engine = record["engine"]
        assert engine["k1_abi_version"] == 1 and engine["k1_library_sha256"] == K1_LIBRARY_SHA256
        assert engine["hecs2_binary_sha256"] == (None if phase == "design" else HECS2_BINARY_SHA256)
        # The hecs2 sources that executed the phase are the final ones byte for byte, except a declared difference.
        execution = record["execution"]
        assert execution["source_original_commit"] == EXECUTED_FROM[phase]
        assert _is_hex(execution["final_compact_history_commit"], 40)
        sources = execution["hecs2_sources_sha256"]
        differs = execution.get("sources_differing_from_final", {})
        assert set(sources) == compiled
        assert {p for p in sources if _sha(LAB / "rust" / p) != sources[p]} == set(differs)
        assert execution["test_module"]["sha256_at_execution"] == TESTS_RS_AT_EXECUTION
    assert set(_record("design")["execution"]["sources_differing_from_final"]) == {"src/experiment.rs"}


# -- DESIGN: the findings the frozen law rests on --------------------------------------------------------------------

DESIGN_FINDINGS = {
    "ONE_STEP_CAPTURE_PLATEAUS_PER_INSTANCE", "PROVISIONAL_0_99_EVERYWHERE_CALIBRATION_REJECTED",
    "CONVERGENCE_TOLERANCE_SUPPORTED", "S1_THRESHOLD_BETWEEN_UNTRAINED_AND_CONVERGED", "ANALYTIC_FLOOR_DIAGNOSTIC_ONLY",
    "CLIPPING_NON_NEGLIGIBLE", "T3B_MEASURED_CLIPPING_AWARE_GENERATOR", "MATCHED_M_NON_INFORMATIVE",
    "TANGENT_AND_EXCURSION_DESCRIPTIVE", "M_EVERY_SEED", "DESIGN_TASK_VALID",
}


def test_design_record_supports_the_frozen_thresholds():
    design = _record("design")
    assert design["result"] == "FREEZE_AUTHORIZED" and design["calibration_grid"] == SPEC["ecs"]["calibration_steps"]
    assert design["execution"]["final_compact_history_commit"] == EXECUTED_FROM["design"]
    found = {f["id"]: f["measured"] for f in design["findings"]}
    assert set(found) == DESIGN_FINDINGS
    # Capture plateaus, so the provisional "0.99 everywhere" rule is met at no budget while convergence is measurable.
    assert max(found["PROVISIONAL_0_99_EVERYWHERE_CALIBRATION_REJECTED"]["min_capture_over_every_level_by_budget"]) < 0.99
    assert any(x <= GATES["CAL_MAX_IMPROVEMENT"]
               for x in found["CONVERGENCE_TOLERANCE_SUPPORTED"]["max_binding_improvement_by_step"])
    # S1 lies between untrained and converged one-step capture.
    s1 = found["S1_THRESHOLD_BETWEEN_UNTRAINED_AND_CONVERGED"]
    assert s1["untrained_capture_range_every_level"][1] < GATES["S1_MIN_CAPTURE"] <= s1["converged_binding_capture_min"]
    # T3 holds at every binding pair against the measured clipping-aware generator.
    t3 = found["T3B_MEASURED_CLIPPING_AWARE_GENERATOR"]
    assert max(hi for _, hi in t3["binding_linear_over_generator_range"].values()) <= GATES["T3B_MAX_LINEAR_OVER_GENERATOR"]
    assert t3["binding_t3a_explained_by_linear_min"] >= GATES["T3A_MIN_EXPLAINED"] and t3["t3a_and_t3b_every_binding_pair"]
    # A correct predictor clears M widely; persistence clears it in MATCHED (non-informative) but not in SEPARATED.
    m = found["MATCHED_M_NON_INFORMATIVE"]
    assert m["binding_generator_capture_range"][0] > GATES["M_MIN_CAPTURE"]
    assert m["binding_persistence_capture_range"]["SEPARATED"][1] < GATES["M_MIN_CAPTURE"]
    assert m["binding_persistence_capture_range"]["MATCHED"][1] >= GATES["M_MIN_CAPTURE"]
    task = found["DESIGN_TASK_VALID"]
    assert task["oracle_success_min"] >= GATES["V2_ORACLE_SUCCESS_MIN"] and task["l1_probe_nmse_max"] <= GATES["V3_L1_PROBE_NMSE_MAX"]
    assert task["centroid_ratio_range"]["SEPARATED"][0] >= GATES["V4_SEPARATED_CENTROID_RATIO_MIN"]
    assert task["centroid_ratio_range"]["MATCHED"][1] <= GATES["V4_MATCHED_CENTROID_RATIO_MAX"]


# == DEV: calibration and run follow the frozen law from the record's own numbers ===================================

def _task_flags(tv: dict) -> dict:
    flags = {
        "T1": tv["T1A_explained_by_linear_min"] >= GATES["T1A_MIN_EXPLAINED"] and tv["T1B_tolerance_slack_min"] >= 0,
        "T3A": all(v >= GATES["T3A_MIN_EXPLAINED"] for v in tv["T3A_explained_by_linear_min"].values()),
        "T3B": all(v <= GATES["T3B_MAX_LINEAR_OVER_GENERATOR"] for v in tv["T3B_linear_over_generator_max"].values()),
        "V2": all(v >= GATES["V2_ORACLE_SUCCESS_MIN"] for v in tv["V2_oracle_success_mean"].values()),
        "V3": tv["V3_l1_probe_nmse_max"] <= GATES["V3_L1_PROBE_NMSE_MAX"],
        "V4": (tv["V4_centroid_ratio_mean"]["SEPARATED"] >= GATES["V4_SEPARATED_CENTROID_RATIO_MIN"]
               and tv["V4_centroid_ratio_mean"]["MATCHED"] <= GATES["V4_MATCHED_CENTROID_RATIO_MAX"]),
        "V5_no_refusal": tv["V5_refusals"] == 0,
    }
    flags["valid"] = all(flags.values())
    return flags


def _seed_meets_m(world: dict, i: int) -> bool:
    return all(world["by_pair"][k]["min_capture_by_seed"][i] >= GATES["M_MIN_CAPTURE"]
               and world["by_pair"][k]["max_numerical_escape_by_seed"][i] <= GATES["M_MAX_ESCAPED"] for k in BINDING_KEYS)


def _law_is_recomputed(record: dict, phase: str) -> dict:
    """Recompute every flag the frozen law derives (task validity, S1, M, M informativeness) and return them."""
    assert record["task_validity"]["flags"] == _task_flags(record["task_validity"])
    assert record["task_validity"]["instances"] == 2 * len(_seeds(phase))
    s1 = {w: all(min(record["one_step"][w][k]) >= GATES["S1_MIN_CAPTURE"] for k in BINDING_KEYS) for w in WORLDS}
    assert record["one_step"]["S1"] == s1
    ms = record["multistep"]
    m = {}
    for w in WORLDS:
        world = ms[w]
        assert [(k, world["by_pair"][k]["consumed_horizon"]) for k in world["by_pair"]] == BINDING
        n = len(_seeds(phase))
        assert all(len(v) == n for p in world["by_pair"].values() for v in p.values() if isinstance(v, list))
        meets = [_seed_meets_m(world, i) for i in range(n)]
        assert world["seeds"] == n and world["seeds_meeting_every_binding"] == sum(meets)
        # The listed failures are exactly the (seed, pair) entries that miss a bound.
        missed = {(_seeds(phase)[i], k) for i in range(n) for k in BINDING_KEYS
                  if world["by_pair"][k]["min_capture_by_seed"][i] < GATES["M_MIN_CAPTURE"]
                  or world["by_pair"][k]["max_numerical_escape_by_seed"][i] > GATES["M_MAX_ESCAPED"]}
        assert {(f["seed"], f["pair"]) for f in world["failures"]} == missed
        m[w] = all(meets) and GATES["M_SEED_FRACTION"] == 1.0
        assert ms["M_informative"][w] == (ms["persistence_capture_max"][w] < GATES["M_MIN_CAPTURE"])
    assert ms["M"] == m
    return {"valid": record["task_validity"]["flags"]["valid"], "S1": s1, "M": m}


def test_dev_calibration_and_run_follow_the_frozen_law():
    dev = _record("dev")
    cal = dev["calibration"]
    grid = SPEC["ecs"]["calibration_steps"]
    previous, converged_at = None, None
    for g, row in enumerate(cal["by_budget"]):
        assert row["steps"] == grid[g]
        imp = row["max_binding_improvement_from_previous"]
        if previous is None:
            assert imp is None and row["previous_converged"] is False
        else:
            assert row["previous_converged"] == (imp <= GATES["CAL_MAX_IMPROVEMENT"])
            if row["previous_converged"]:
                converged_at = g - 1
                assert g == len(cal["by_budget"]) - 1, "the search stops once the chosen budget is measured"
        previous = row
    assert converged_at is not None and cal["converged_steps"] == grid[converged_at]
    assert cal["chosen_steps"] == grid[converged_at + GATES["CAL_GRID_STEPS_ABOVE"]] == 96000
    assert cal["disposition"] == "CALIBRATED" and dev["training_steps"] == cal["chosen_steps"]
    law = _law_is_recomputed(dev, "dev")
    assert dev["binding"] is False and dev["disposition"] == ("QUAL_AUTHORIZED" if law["valid"] else "TASK_INVALID_ON_DEV")
    assert dev["disposition"] == "QUAL_AUTHORIZED"
    assert dev["hierarchy_outcome"] == "NOT_ADJUDICATED" and dev["integration_authorized"] is False
