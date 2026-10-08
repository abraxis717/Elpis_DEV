"""H-ECS R1 preregistration guards (research/hecs_r1/PREREGISTRATION.md). SCIENTIFIC lane; seconds.

They keep the frozen authority honest without running any science: the seeds are the documented SHA-256
derivation and disjoint from R0's; every R1 source file is either byte-identical to R0's or declared modified
with a reason; R0's closed authority is unchanged; the DESIGN evidence ran under the frozen specification on
DESIGN seeds only and computed no hypothesis quantity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LAB = REPO / "research" / "hecs_r1"
R0 = REPO / "research" / "hecs_r0"
SPEC = json.loads((LAB / "specs" / "hecs-r1.v1.spec.json").read_text())
R0_SEEDS = {0, 1, 2, 3, 100, 101, 102, 103, 104, 105, 106, 107, 9999}


def _seed(phase: str, i: int) -> int:
    return int.from_bytes(hashlib.sha256(f"elpis.hecs-r1.{phase}.v1:{i}".encode()).digest()[:8], "big")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_seeds_are_the_documented_derivation_fresh_and_disjoint():
    for phase, n in (("dev", 4), ("qual", 8), ("design", 4)):
        assert [int(s) for s in SPEC[f"{phase}_seeds"]] == [_seed(phase, i) for i in range(n)]
    seeds = [int(s) for p in ("dev", "qual", "design") for s in SPEC[f"{p}_seeds"]]
    assert len(set(seeds)) == len(seeds) and not set(seeds) & R0_SEEDS


def test_sources_are_r0_byte_for_byte_except_the_declared_changes():
    record = json.loads((LAB / "SOURCE_EQUIVALENCE.json").read_text())
    listed = {row["path"] for row in record["files"]}
    actual = {p.relative_to(LAB / "rust").as_posix() for p in (LAB / "rust" / "src").glob("*.rs")}
    actual |= {"Cargo.toml", "Cargo.lock", "rustfmt.toml"}
    assert listed == actual
    for row in record["files"]:
        r1 = LAB / "rust" / row["path"]
        assert _sha(r1) == row["r1_sha256"], row["path"]
        if row["status"] == "NEW":
            assert row["r0_sha256"] is None and row["reason"]
            continue
        r0 = R0 / "rust" / row["path"]
        assert _sha(r0) == row["r0_sha256"], f"R0 changed: {row['path']}"   # R0's closed authority is untouched
        if row["status"] == "IDENTICAL":
            assert row["r0_sha256"] == row["r1_sha256"]
        else:
            assert row["status"] == "MODIFIED" and row["reason"] and row["r0_sha256"] != row["r1_sha256"]


def test_the_validity_law_and_inherited_gates_are_as_preregistered():
    g = SPEC["gates"]
    assert (g["V1A_MIN_EXPLAINED"], g["V1B_FLOOR_TOL_ABS"], g["V1B_FLOOR_TOL_REL"], g["V1C_MIN_CAPTURE"]) == (
        0.5, 0.005, 0.15, 0.98)
    assert "V1_L1_ONE_STEP_NMSE_MAX" not in g   # R0's absolute gate is gone
    r0 = json.loads((R0 / "specs" / "hecs-r0.v1.spec.json").read_text())
    unchanged = {k: v for k, v in r0["gates"].items() if not k.startswith("V1") and k != "V4_MATCHED_CENTROID_RATIO_MAX"}
    assert {k: g[k] for k in unchanged} == unchanged
    assert g["V4_MATCHED_CENTROID_RATIO_MAX"] == 2.0
    for key in ("worlds", "data", "conditions", "measurements", "planning", "encoders"):
        assert SPEC[key] == r0[key], key
    ecs = dict(SPEC["ecs"]), dict(r0["ecs"])
    assert ecs[0] == ecs[1]


def test_design_evidence_is_design_only_and_computes_no_hypothesis():
    design = json.loads((LAB / "evidence" / "design" / "hecs-r1.v1.design.json").read_text())
    assert design["spec"] == SPEC and design["phase"] == "DESIGN"
    seeds = {i["seed"] for row in design["grid"] for i in row["instances"]}
    assert seeds == set(SPEC["design_seeds"])
    # Only level-1 validity quantities, probes, centroids and the oracle: no condition, plan or hypothesis.
    assert set(design) == {"spec", "phase", "note", "grid"}
    for row in design["grid"]:
        for instance in row["instances"]:
            assert set(instance) == {"world", "seed", "validity", "v1", "l1_probe_nmse_slow_fast", "centroid_ratio",
                                     "oracle_success", "refused", "seconds"}


# -- evidence: the decisions follow the frozen law from their own numbers -----------------------------------------

EVIDENCE = LAB / "evidence"


def _load(rel: str) -> dict:
    return json.loads((EVIDENCE / rel).read_text())


def _v1(d: dict, g: dict) -> bool:
    return (d["linear_reference"] <= (1 - g["V1A_MIN_EXPLAINED"]) * d["constant_baseline"]
            and abs(d["linear_reference"] - d["analytic_floor"]) <= g["V1B_FLOOR_TOL_ABS"] + g["V1B_FLOOR_TOL_REL"]
            * d["analytic_floor"]
            and d["ecs_one_step"] != float("inf")
            and d["constant_baseline"] - d["ecs_one_step"] >= g["V1C_MIN_CAPTURE"] * (d["constant_baseline"]
                                                                                     - d["linear_reference"]))


def test_calibration_follows_the_rule():
    c = _load("dev/hecs-r1.v1.calibration.json")
    assert c["spec"] == SPEC and c["phase"] == "DEV_CALIBRATION"
    chosen = None
    for row in c["grid"]:
        assert {i["seed"] for i in row["instances"]} == set(SPEC["dev_seeds"])
        assert row["v1_all"] == all(_v1(i["validity"], SPEC["gates"]) and not i["refused"] for i in row["instances"])
        if row["v1_all"]:
            chosen = row["steps"]
            break
    assert chosen == c["chosen_steps"] == 6000 and c["disposition"] == "CALIBRATED"


def _evaluation_is_recomputed(phase: str, seeds_key: str) -> dict:
    run = _load(f"{phase}/hecs-r1.v1.{phase}.json")
    assert run["spec"] == SPEC and run["phase"] == phase.upper() and run["training_steps"] == 6000
    for world in run["worlds"]:
        # EVIDENCE_PROVENANCE: the run evidence records each u64 seed as a signed i64 (`seed as i64`); the
        # computation used the u64 seed. Decoded exactly here; the write-once evidence is never rewritten.
        assert [s["seed"] % 2**64 for s in world["seeds"]] == [int(x) for x in SPEC[seeds_key]]
    validity = run["evaluation"]["validity"]
    instances = validity["V1_instances"]
    assert len(instances) == 2 * len(SPEC[seeds_key])
    assert validity["V1"] == all(_v1(d, SPEC["gates"]) for d in instances)
    return run


def test_dev_is_valid_and_authorized_qual():
    dev = _evaluation_is_recomputed("dev", "dev_seeds")
    assert dev["evaluation"]["validity"]["valid"] is True


def test_qual_ran_once_and_is_task_invalid_by_v1c_alone():
    qual = _evaluation_is_recomputed("qual", "qual_seeds")
    ev = qual["evaluation"]
    assert ev["disposition"] == "TASK_INVALID" and ev["validity"]["valid"] is False
    failing = [d for d in ev["validity"]["V1_instances"] if not _v1(d, SPEC["gates"])]
    assert len(failing) == 1 and failing[0]["C_ecs_adequate"] is False
    assert failing[0]["A_reference_predictable"] and failing[0]["B_reference_at_floor"]
    assert all(ev["validity"][k] for k in ("V2", "V3", "V4", "V5_no_refusal"))
    # Write-once: exactly one QUAL evidence file; nothing beyond DESIGN, calibration, DEV and QUAL.
    assert sorted(p.relative_to(EVIDENCE).as_posix() for p in EVIDENCE.rglob("*.json")) == [
        "design/hecs-r1.v1.design.json", "dev/hecs-r1.v1.calibration.json", "dev/hecs-r1.v1.dev.json",
        "qual/hecs-r1.v1.qual.json"]


def test_results_report_the_disposition_and_withhold_integration():
    text = (REPO / "docs" / "research" / "HECS_R1_RESULTS.md").read_text()
    for needle in ("`TASK_INVALID`", "NOT_AUTHORIZED", "NO_CANONICAL_PROMOTION", "not adjudicated", "5c57dec"):
        assert needle in text, needle
    # No hierarchy controller was integrated: nothing outside research names the laboratory's crates.
    for path in (REPO / "src").rglob("*.py"):
        assert "hecs_r1" not in path.read_text() and "hecs_r0" not in path.read_text(), path
