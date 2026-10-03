"""Cognitive R0 laboratory guards: frozen authority, immutable evidence, exact reproduction, honest reporting."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from elpis.ECS_G.native import ECSGLibrary
from research.ecs_cognition_r0 import experiment as E
from research.ecs_cognition_r0 import run as R
from research.ecs_cognition_r0.protocol import load, source_digest

from ...ECS_G.test_math_r0 import REPO, _library_path

RESULTS = REPO / "docs" / "research" / "COGNITION_R0_RESULTS.md"


def test_frozen_authority_still_matches_the_laboratory():
    """Editing the laboratory after freezing fails here until a new experiment version is frozen."""
    frozen_record = load(R.FROZEN_PATH, "frozen")
    frozen = frozen_record["body"]
    assert frozen["source_digest"] == source_digest()
    assert frozen["spec"] == E.SPEC and frozen["pass_rule"] == E.PASS_RULE
    assert frozen["dev_evidence_digest"] == load(R.DEV_PATH, "dev")["digest"]
    assert frozen["choices"]["steps"] in E.SPEC["step_grid"]


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


@pytest.mark.parametrize("world", ["qual-0000", "qual-0007"])
def test_qual_measurements_reproduce_exactly(world):
    """Re-running a QUAL world reproduces its recorded measurements bit for bit (same numerical profile)."""
    qual = load(R.QUAL_PATH, "qual")["body"]
    if qual["numerical_profile"]["numpy"].split(".")[:2] != np.__version__.split(".")[:2]:
        pytest.skip(f"evidence recorded with numpy {qual['numerical_profile']['numpy']}; reproduction is exact "
                    f"only under the recorded numerical profile (have {np.__version__})")
    api = ECSGLibrary(ctypes.CDLL(str(_library_path())))
    metrics, checks, _ = E.evaluate_world(api, E.SPEC, world, qual["steps"])
    assert metrics == qual["per_world"][world]
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
