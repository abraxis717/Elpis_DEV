"""Proof that the laboratory has no authority over production code or state.

The laboratory's public-kernel collision experiment (``ecs_collision.py``) ran on the event-history
kernel that Elpis has since abolished (docs/CONTINUITY.md). The laboratory source stays byte-frozen and its
recorded results stay historical, but that experiment can no longer be executed and is not tested here.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from research.ecs_dynamics.export import DynamicsObservation
from research.ecs_dynamics.results import ExperimentResult, ResultError

REPO = Path(__file__).resolve().parents[3]


# --- no authority mutation --------------------------------------------------------------------


def test_production_code_never_imports_the_laboratory():
    system = json.loads((REPO / "ELPIS_SYSTEM.json").read_text())
    assert "research" in system["boundary_policy"]["forbidden_python_imports"]
    assert "research" in system["repository_surfaces"]
    assert all(s["id"] != "research" for s in system["subsystems"])
    for path in (REPO / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import research" not in text and "from research" not in text, path


def test_importing_ecs_does_not_load_the_laboratory_or_numpy():
    probe = ("import sys, elpis.ECS, elpis.ECS.native, elpis.ECS.k1, elpis.continuity; "
             "print(sorted(m for m in sys.modules if m.startswith('research') or m.startswith('numpy')))")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=REPO,
                         env={"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"}, check=True)
    assert out.stdout.strip() == "[]"


def test_neural_law_is_not_in_the_ecs_or_continuity_and_numpy_is_not_a_base_dependency():
    import ast

    for root in ("ECS", "continuity"):
        for path in (REPO / "src" / "elpis" / root).rglob("*.py"):
            tree = ast.parse(path.read_text())
            imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
            imported |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
            assert not [n for n in imported if n.split(".")[0] in ("numpy", "research")], path
            names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
            assert "tanh" not in names, path
    import tomllib
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert not any(dep.lower().startswith("numpy") for dep in project.get("dependencies", []))
    packages = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]["packages"]["find"]
    assert packages["include"] == ["elpis*"]  # the laboratory is never packaged


def test_exported_observation_is_inert_data():
    result = ExperimentResult("x", 1, "FIXTURE", "a" * 64, None, "b" * 64, "c", "cubic-moment-control", {}, (), "S3",
                              "INSTANTANEOUS_OUTPUT", "none", "ALGEBRAIC_IDENTITY", {})
    obs = DynamicsObservation.from_result(result)
    assert obs.research_only and obs.no_runtime_authority and obs.no_production_neural_claim
    public = [n for n in dir(obs) if not n.startswith("_")]
    callables = {n for n in public if callable(getattr(obs, n))}
    assert callables == {"from_result", "to_json"}
    with pytest.raises(Exception):
        obs.scientific = "SUPPORTS_UNDER_FROZEN_SYNTHETIC_REGIME"
    assert json.loads(obs.to_json())["result_digest"] == result.digest


def test_result_schema_separates_mechanics_from_science_and_binds_qual_to_a_freeze():
    common = ("x", 1)
    tail = ("a" * 64, "b" * 64, "c", "cubic-moment-control", {}, (), "S3", "INSTANTANEOUS_OUTPUT", "none",
            "ALGEBRAIC_IDENTITY", {})
    with pytest.raises(ResultError, match="frozen"):
        ExperimentResult(*common, "QUAL", tail[0], None, *tail[1:])
    with pytest.raises(ResultError, match="mechanics"):
        ExperimentResult(*common, "FIXTURE", tail[0], None, *tail[1:], mechanics="MECHANICS_FAIL",
                         scientific="SUPPORTS_UNDER_FROZEN_SYNTHETIC_REGIME")
    with pytest.raises(ResultError, match="source"):
        ExperimentResult(*common, "FIXTURE", tail[0], None, *tail[1:], sources=("a blog post",))
    flags = ExperimentResult(*common, "FIXTURE", tail[0], None, *tail[1:]).as_dict()["flags"]
    assert flags == {"RESEARCH_ONLY": True, "NO_RUNTIME_AUTHORITY": True, "NO_PRODUCTION_NEURAL_CLAIM": True}
