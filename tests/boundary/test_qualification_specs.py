"""Qualification specifications grant nothing by existing (docs/qualification/README.md).

Every specification states its status: NOT RUN (no evidence may exist), RUN PASS or RUN FAIL. A run status needs
write-once evidence that the specification's independent validator verifies (exactly its files, every digest, their
binding to the freeze) and recomputes, and the status must repeat the recomputed verdict. Only a RUN PASS backed so
may state a QUALIFIED subject, with its scope in its own document; no other document declares a qualification.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[2]
SPECS = REPO / "docs" / "qualification"
EVIDENCE = {
    "COGNITIVE_BOUNDARY_R1.md": "research/qualification/cognitive_boundary_r1/evidence",
    "K1_RECOVERY_R0.md": "research/qualification/k1_recovery_r0/evidence",
    "HACF_ECS_BRIDGE_R0.md": "research/hacf_ecs_bridge_r0/evidence",
    "EVOLUTION_FITNESS_R0.md": "research/evolution_fitness_r0/evidence",
    # RuntimeCore ABI v4 successors (the ABI v3 specifications above stay as pre-registered) and the codec interface.
    "COGNITIVE_BOUNDARY_R2.md": "research/qualification/cognitive_boundary_r2/evidence",
    "K1_RECOVERY_R0_V4.md": "research/qualification/k1_recovery_r0_v4/evidence",
    "ECS_CODEC_R0.md": "research/qualification/ecs_codec_r0/evidence",
    # The ABI v5 successor of Cognitive Boundary R2 (R2 stays as pre-registered, unrun authority for ABI v4).
    "COGNITIVE_BOUNDARY_R2_V5.md": "research/qualification/cognitive_boundary_r2_v5/evidence",
}
EXPECTED = set(EVIDENCE)
STATUS = re.compile(r"^Status: \*\*(?P<status>[A-Z ]+)\*\* · Subject: \*\*(?P<subject>[A-Z]+)\*\*", re.M)
# Specifications whose evidence has an independent validator (a sealed directory it can digest-check and recompute).
VALIDATORS = {"K1_RECOVERY_R0_V4.md": REPO / "research" / "qualification" / "k1_recovery_r0_v4" / "check.py"}
STATUSES = ("NOT RUN", "RUN PASS", "RUN FAIL")


def _status(name):
    found = STATUS.search((SPECS / name).read_text(encoding="utf-8"))
    assert found, name
    return found["status"], found["subject"]


def _validate(name):
    """The validator's problems with the specification's sealed evidence, and the verdict it holds."""
    spec = importlib.util.spec_from_file_location(f"validator_{name}", VALIDATORS[name])
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    evidence = REPO / EVIDENCE[name]
    problems = check.verify_evidence(evidence, VALIDATORS[name].parent / "plan.json")
    verdict = json.loads((evidence / "verdict.json").read_text()).get("verdict") if not problems else None
    return problems, verdict


def test_every_specification_is_present_and_its_status_is_backed_by_its_evidence():
    assert {p.name for p in SPECS.glob("*.md")} - {"README.md"} == EXPECTED
    for name in EXPECTED:
        text = (SPECS / name).read_text(encoding="utf-8")
        status, subject = _status(name)
        assert status in STATUSES, (name, status)
        evidence = REPO / EVIDENCE[name]
        if status == "NOT RUN":
            assert subject == "UNQUALIFIED", name
            assert not evidence.exists(), f"{name}: evidence exists but the status says NOT RUN"
        else:   # a run happened: its write-once evidence must exist, validate, and say what the status says
            assert name in VALIDATORS, f"{name}: a run status needs an evidence validator"
            assert evidence.is_dir() and any(evidence.iterdir()), f"{name}: status {status} without evidence"
            problems, verdict = _validate(name)
            assert problems == [], (name, problems)
            assert verdict == status.split()[1], (name, status, verdict)
            assert subject == ("QUALIFIED" if status == "RUN PASS" else "UNQUALIFIED"), (name, status, subject)
        for gate in ("## Question", "## Gates (pre-registered)", "## Would establish / would not establish"):
            assert gate in text, (name, gate)


def test_no_document_declares_a_subject_qualified_without_validated_evidence():
    passed = {n for n in EXPECTED if _status(n)[0] == "RUN PASS"}
    documents = [SPECS / n for n in EXPECTED - passed] + [REPO / "docs" / n for n in ("K1_RECOVERY_R0.md",
                                                                                    "EVOLUTION_POLICY.md")]
    documents += [REPO / "research" / lab / "README.md" for lab in ("hacf_ecs_bridge_r0", "evolution_fitness_r0")]
    for path in documents:
        text = path.read_text(encoding="utf-8")
        assert "UNQUALIFIED" in text, path
        assert not re.search(r"(Status|Subject|Classification|Qualification):\s*\**\s*`?QUALIFIED", text), path
    index = (SPECS / "README.md").read_text(encoding="utf-8")
    assert index.count("NOT RUN · UNQUALIFIED") == len([n for n in EXPECTED if _status(n)[0] == "NOT RUN"])
    for name in passed:   # a qualified subject is indexed as such, with its scope stated in its own document
        assert f"RUN PASS · QUALIFIED" in index and "Qualified scope" in (SPECS / name).read_text(encoding="utf-8")
