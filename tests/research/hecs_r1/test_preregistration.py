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
