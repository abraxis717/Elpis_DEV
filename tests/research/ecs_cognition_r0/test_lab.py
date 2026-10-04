"""Cognitive R0 laboratory guards: frozen authority, immutable evidence, historical replay, honest reporting.

Three questions are kept apart (docs/research/COGNITION_R0_RESULTS.md, "Reproduction contract"):

* evidence integrity (always tested here): the frozen records are byte-identical to the ones written at
  DEV / freeze / QUAL time and internally consistent;
* current implementation correctness (tests/ECS_G, tests/research/ecs_runtime_r1, the native ctest suite):
  the executor equals the scalar reference bitwise on the host that runs the tests;
* historical replay (``test_qual_measurements_reproduce_exactly``): the current process reproduces the
  recorded QUAL bytes. Demanded bitwise only under the numerical profile the evidence recorded.

Whether the current runtime still reaches the recorded scientific result on any host is
``test_runtime_regression.py``, which never skips for a profile difference.
"""
from __future__ import annotations

import ctypes
import hashlib

import pytest

from elpis.ECS_G.native import ECSGLibrary
from research.ecs_cognition_r0 import experiment as E
from research.ecs_cognition_r0 import run as R
from research.ecs_cognition_r0.protocol import load, numerical_profile, source_digest

from ...ECS_G.test_math_r0 import REPO, _library_path
from .._blas import openblas_core

RESULTS = REPO / "docs" / "research" / "COGNITION_R0_RESULTS.md"


def test_frozen_authority_still_matches_the_laboratory():
    """Editing the laboratory after freezing fails here until a new experiment version is frozen."""
    frozen_record = load(R.FROZEN_PATH, "frozen")
    frozen = frozen_record["body"]
    assert frozen["source_digest"] == source_digest()
    assert frozen["spec"] == E.SPEC and frozen["pass_rule"] == E.PASS_RULE
    assert frozen["dev_evidence_digest"] == load(R.DEV_PATH, "dev")["digest"]
    assert frozen["choices"]["steps"] in E.SPEC["step_grid"]


# SHA-256 of each record as written (each file has exactly one commit: bdb8823 spec and DEV, 9ea274f freeze
# and QUAL). Any byte change to frozen evidence fails here; a repaired experiment is a new version.
EVIDENCE_SHA256 = {
    "specs/ecsg-cognition-r0.v1.spec.json": "83e2427655928f51ffd8c2d49fd6e4a9a8fe41cd591727ccbe6d3960b449ee3f",
    "evidence/dev/ecsg-cognition-r0.v1.dev.json": "43d8ca83d58979934fd378a082ffa39ef3534d325819cf47d6ed3cfb4490635f",
    "frozen/ecsg-cognition-r0.v1.frozen.json": "46ad3a6907ce6a5a502767968d580c94c66685de8d73941a6c2fb084e61d374f",
    "evidence/qual/ecsg-cognition-r0.v1.qual.json": "cd801cc9796dcfdc692b541564735c9281608443b0a6f5bc44fa7376a405d429",
}


@pytest.mark.parametrize("name", sorted(EVIDENCE_SHA256))
def test_frozen_evidence_bytes_are_unchanged(name):
    data = (R.ROOT / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == EVIDENCE_SHA256[name], f"{name} changed after it was written"


def test_dev_saw_only_dev_worlds_and_qual_only_qual_worlds():
    dev = load(R.DEV_PATH, "dev")["body"]
    qual = load(R.QUAL_PATH, "qual")["body"]
    assert all(w.startswith("dev-") for w in dev["worlds"]) and set(dev["curves"]) == set(dev["worlds"])
    assert all(w.startswith("qual-") for w in qual["worlds"]) and not set(dev["worlds"]) & set(qual["worlds"])


def test_qual_evidence_is_intact_and_bound_to_the_freeze():
    frozen_record = load(R.FROZEN_PATH, "frozen")
    qual = load(R.QUAL_PATH, "qual")["body"]
    assert qual["frozen_digest"] == frozen_record["digest"]
    assert qual["source_digest"] == frozen_record["body"]["source_digest"]
    assert qual["steps"] == frozen_record["body"]["choices"]["steps"]
    assert qual["mechanics"] == "MECHANICS_PASS"
    assert qual["scientific"] == "SUPPORTS_UNDER_FROZEN_SYNTHETIC_REGIME"
    assert all(qual["gates"].values()) and set(qual["gates"]) == {
        "A_learning", "B_state_causality", "C_persistence", "D_continual", "E_controls", "F_determinism",
        "G_microstate_authority"}


def test_interference_is_reported_as_found():
    qual = load(R.QUAL_PATH, "qual")["body"]
    classes = [r["class"] for r in qual["retention_measured"].values()]
    catastrophic = sum(r["catastrophic"] for r in qual["retention_measured"].values())
    text = RESULTS.read_text(encoding="utf-8")
    assert f"{classes.count('INTERFERENCE')} of {len(classes)}" in text
    assert f"catastrophic in {catastrophic} of {len(classes)}" in text


QUAL_WORLDS = tuple(load(R.QUAL_PATH, "qual")["body"]["worlds"])


def _profile_mismatch(recorded: dict) -> list[str]:
    """Recorded numerical-profile fields this process does not satisfy (every field v1 recorded, exactly)."""
    current = numerical_profile()
    unknown = sorted(set(recorded) - set(current))
    if unknown:
        return [f"unrecognized recorded field(s) {unknown}"]
    return [f"{k} {current[k]} != recorded {recorded[k]}" for k in sorted(recorded) if current[k] != recorded[k]]


# OpenBLAS kernel families under which the v1 QUAL bytes were shown to reproduce exactly, established after
# the fact (CE0) by forcing each family with OPENBLAS_CORETYPE on Cooper Lake and Emerald Rapids hosts: the
# FMA DGEMM kernels reproduce all 8 worlds; Prescott, Nehalem, Sandybridge (no FMA) reproduce none. This is
# not part of the v1 record, which bound no kernel, CPU, compiler or native-binary digest.
REPRODUCING_OPENBLAS_CORES = frozenset({"Haswell", "Zen", "SkylakeX", "Cooperlake"})


@pytest.mark.parametrize("world", QUAL_WORLDS)
def test_qual_measurements_reproduce_exactly(world):
    """HISTORICAL REPLAY: re-running a QUAL world reproduces its recorded measurements bit for bit.

    Demanded only when this process satisfies every numerical-profile field the evidence recorded (Python,
    NumPy, machine, float format; exact versions) and its OpenBLAS kernel family is one under which the v1
    bytes are known to reproduce. The v1 record binds no native-binary digest, compiler, CPU or BLAS kernel,
    so its profile is necessary but not sufficient: NumPy's bundled OpenBLAS selects its DGEMM kernel per
    CPU, falls back to the generic Prescott kernel on CPUs it does not recognize (e.g. family 6 model 207),
    and the non-FMA kernels round the teacher targets differently in the last bit, which changes every
    learned snapshot digest (the 61f81e6 CI failure). Whether the science still holds on such a host is
    ``test_runtime_regression.py``, which never skips.
    """
    qual = load(R.QUAL_PATH, "qual")["body"]
    mismatch = _profile_mismatch(qual["numerical_profile"])
    if mismatch:
        pytest.skip("HISTORICAL_REPLAY_PROFILE_MISMATCH: exact replay of v1 QUAL bytes is demanded only under the "
                    f"recorded numerical profile; {'; '.join(mismatch)} (OpenBLAS core {openblas_core()})")
    core = openblas_core()
    if core not in REPRODUCING_OPENBLAS_CORES:
        pytest.skip(f"HISTORICAL_REPLAY_KERNEL_MISMATCH: OpenBLAS core {core} is outside the kernel families under "
                    f"which the v1 bytes reproduce ({', '.join(sorted(REPRODUCING_OPENBLAS_CORES))}; established "
                    "after the fact, not recorded in v1)")
    api = ECSGLibrary(ctypes.CDLL(str(_library_path())))
    metrics, checks, _ = E.evaluate_world(api, E.SPEC, world, qual["steps"])
    assert metrics == qual["per_world"][world], f"OpenBLAS core {core}"
    assert checks == qual["mechanics_checks"][world]


OVERCLAIMS = ("elpis now thinks", "elpis thinks", "sufficient for intelligence", "architecture is solved",
              "understands language", "general intelligence is", "ecs is the final")


def test_admitted_claims_cite_the_evidence_and_stay_bounded():
    import json
    qual_digest = load(R.QUAL_PATH, "qual")["digest"]
    frozen_digest = load(R.FROZEN_PATH, "frozen")["digest"]
    authority = (REPO / "docs" / "COGNITION_R0.md").read_text(encoding="utf-8")
    assert qual_digest in authority and frozen_digest in authority
    system = json.loads((REPO / "ELPIS_SYSTEM.json").read_text(encoding="utf-8"))
    ecs_g = next(s for s in system["subsystems"] if s["id"] == "ECS_G")
    assert qual_digest[:8] in ecs_g["maturity"] and frozen_digest[:8] in ecs_g["maturity"]
    assert "under the qualified Cognitive R0 regime" in ecs_g["maturity"]
    for name in ("docs/COGNITION_R0.md", "docs/research/COGNITION_R0_RESULTS.md", "native/ECS_G/README.md",
                 "docs/ARCHITECTURE.md", "README.md", "ELPIS_SYSTEM.json"):
        text = (REPO / name).read_text(encoding="utf-8").lower()
        assert not [o for o in OVERCLAIMS if o in text], name
