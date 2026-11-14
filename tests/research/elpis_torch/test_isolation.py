"""The prototype stays where it belongs: research only, PyTorch optional, canonical authority untouched.

These run with or without PyTorch (they import nothing from the prototype that needs it).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib

from .conftest import REPO

LAB = REPO / "research" / "elpis_torch"
CANONICAL = (REPO / "src", REPO / "native")
SOURCE_SUFFIXES = (".py", ".c", ".h", ".cpp", ".hpp", ".rs", ".toml", ".cmake", ".txt", ".json")
PROPOSAL = REPO / "DEV" / "proposals" / "CGC_NATIVE_GENERATIVE_SUBSTRATE_R0.md"


def test_canonical_code_does_not_know_the_prototype_exists():
    offenders = []
    for root in CANONICAL:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="replace")
                if "elpis_torch" in text or re.search(r"(?m)^\s*(import|from)\s+torch\b", text):
                    offenders.append(str(path.relative_to(REPO)))
    assert not offenders


def test_pytorch_is_not_a_dependency_of_the_distribution():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = list(project["dependencies"]) + [d for extra in project["optional-dependencies"].values() for d in extra]
    assert not [d for d in declared if d.lower().startswith("torch")]
    assert project["dependencies"] == []
    policy = json.loads((REPO / "ELPIS_SYSTEM.json").read_text(encoding="utf-8"))["boundary_policy"]
    assert "torch" in policy["forbidden_python_imports"] and "research" in policy["forbidden_python_imports"]


def test_importing_the_package_does_not_import_pytorch():
    probe = "import sys, research.elpis_torch as lab; print(lab.PHASE, 'torch' in sys.modules)"
    run = subprocess.run([sys.executable, "-B", "-c", probe], capture_output=True, text=True, cwd=REPO,
                         env={"PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"})
    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == ["ELPIS_TORCH_GENERATIVE_SUBSTRATE_R0", "False"]


def test_no_weights_or_build_products_are_committed_with_the_prototype():
    forbidden = tuple(json.loads((REPO / "ELPIS_SYSTEM.json").read_text())["boundary_policy"]["forbidden_file_suffixes"])
    roots = (LAB, REPO / "tests" / "research" / "elpis_torch", REPO / "docs" / "research" / "elpis_torch")
    files = [p for root in roots for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    assert files and not [str(p) for p in files if p.name.endswith(forbidden)]
    assert max(p.stat().st_size for p in files) < 400_000         # proportional: no evidence archive


def test_the_only_storage_writers_are_the_two_explicit_offline_operations():
    """Sources of the package that can open a file for writing: the checkpoint saver and the experiment's --out."""
    writers = {}
    for path in sorted(LAB.rglob("*.py")):
        if "upstream" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        hits = re.findall(r"torch\.save\(|\bopen\([^)]*[\"'][wax+b]+[\"']|\.write_text\(|\.write_bytes\(|pickle\.dump", text)
        if hits:
            writers[str(path.relative_to(LAB))] = len(hits)
    assert writers == {"training.py": 1, "experiments/run_r0.py": 1}
    session = (LAB / "session.py").read_text(encoding="utf-8")
    assert "save_checkpoint" not in session and "torch.save" not in session and "open(" not in session


def test_the_deferred_systems_are_not_implemented():
    """No supervisor, expert selector, halting agent, energy function or persistent learner hides in the package."""
    names = {p.stem.lower() for p in LAB.rglob("*.py")}
    assert not names & {"ralph", "moe", "engram", "mhc", "hca", "awr", "gdn", "ersa", "halt", "energy", "router",
                        "controller", "supervisor", "memory", "retrieval"}
    code = "\n".join(p.read_text(encoding="utf-8") for p in LAB.rglob("*.py") if "upstream" not in p.parts)
    assert not re.search(r"(?im)^\s*(class|def)\s+\w*(ralph|halt|expert|router|energy|metacog|supervis)\w*", code)


def test_the_mission_does_not_name_the_prototype_and_the_amendment_is_only_a_proposal():
    """The prototype claims no place in the mission; what it would take is a proposal the owner may decline."""
    assert "elpis_torch" not in (REPO / "docs" / "ELPIS_MISSION.md").read_text(encoding="utf-8")
    text = " ".join(PROPOSAL.read_text(encoding="utf-8").split())
    for needle in ("PROPOSAL", "NOT ADOPTED", "not repository authority", "docs/ELPIS_MISSION.md",
                   "prohibited topology", "research/elpis_torch"):
        assert needle in text, needle
