"""Qualification specifications grant nothing by existing (docs/qualification/README.md).

Every specification states its status; a status other than NOT RUN must name evidence that exists and is
digest-bound. None exists yet, so every subject stays UNQUALIFIED, and no specification or implementation document
of the post-red-team subjects declares a qualification.
"""
from __future__ import annotations

from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[2]
SPECS = REPO / "docs" / "qualification"
EXPECTED = {"COGNITIVE_BOUNDARY_R1.md", "K1_RECOVERY_R0.md", "HACF_ECS_BRIDGE_R0.md", "EVOLUTION_FITNESS_R0.md"}
EVIDENCE = {
    "COGNITIVE_BOUNDARY_R1.md": "research/qualification/cognitive_boundary_r1/evidence",
    "K1_RECOVERY_R0.md": "research/qualification/k1_recovery_r0/evidence",
    "HACF_ECS_BRIDGE_R0.md": "research/hacf_ecs_bridge_r0/evidence",
    "EVOLUTION_FITNESS_R0.md": "research/evolution_fitness_r0/evidence",
}
STATUS = re.compile(r"^Status: \*\*(?P<status>[A-Z ]+)\*\* · Subject: \*\*(?P<subject>[A-Z]+)\*\*", re.M)


def test_every_specification_is_present_and_not_run():
    assert {p.name for p in SPECS.glob("*.md")} - {"README.md"} == EXPECTED
    for name in EXPECTED:
        text = (SPECS / name).read_text(encoding="utf-8")
        found = STATUS.search(text)
        assert found, name
        evidence = REPO / EVIDENCE[name]
        if found["status"] == "NOT RUN":
            assert found["subject"] == "UNQUALIFIED", name
            assert not evidence.exists(), f"{name}: evidence exists but the status says NOT RUN"
        else:   # a run happened: its write-once evidence must exist
            assert evidence.is_dir() and any(evidence.iterdir()), f"{name}: status {found['status']} without evidence"
        for gate in ("## Question", "## Gates (pre-registered)", "## Would establish / would not establish"):
            assert gate in text, (name, gate)


def test_no_post_red_team_document_declares_its_subject_qualified():
    documents = [SPECS / n for n in EXPECTED] + [REPO / "docs" / n for n in ("K1_RECOVERY_R0.md",
                                                                          "EVOLUTION_POLICY.md")]
    documents += [REPO / "research" / lab / "README.md" for lab in ("hacf_ecs_bridge_r0", "evolution_fitness_r0")]
    for path in documents:
        text = path.read_text(encoding="utf-8")
        assert "UNQUALIFIED" in text, path
        assert not re.search(r"(Status|Subject|Classification|Qualification):\s*\**\s*`?QUALIFIED", text), path
    index = (SPECS / "README.md").read_text(encoding="utf-8")
    assert index.count("NOT RUN · UNQUALIFIED") == len(EXPECTED)
