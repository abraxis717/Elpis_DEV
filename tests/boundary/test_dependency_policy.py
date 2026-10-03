"""Import-level architecture rules: no model-path resurrection, declared edges only."""
from __future__ import annotations

import json
import re
import subprocess
import sys

import pytest

from ._system import REPO, SYSTEM, imported_modules, live_subsystems, module_name, python_modules

SRC = REPO / "src" / "elpis"
POLICY = SYSTEM["boundary_policy"]


def _owner(name: str) -> str | None:
    parts = name.split(".")
    if parts[0] != "elpis" or len(parts) < 2:
        return None
    return parts[1]


def test_no_forbidden_imports_anywhere_in_the_package():
    forbidden = set(POLICY["forbidden_python_imports"])
    offenders = []
    for path in python_modules(SRC):
        for name in imported_modules(path):
            if name.split(".")[0] in forbidden:
                offenders.append((str(path.relative_to(REPO)), name))
    assert not offenders


def test_no_learned_solver_identifiers_in_source():
    patterns = [re.compile(p) for p in POLICY["forbidden_identifier_patterns"]]
    # Implementation sources only: tests may name forbidden tokens in order to
    # assert their absence.
    roots = [REPO / "src"] + sorted(
        d for d in (REPO / "native").glob("**/*") if d.is_dir() and d.name in ("src", "include")
    ) if (REPO / "native").is_dir() else [REPO / "src"]
    exempt = {REPO / e["path"] for e in POLICY.get("identifier_scan_exemptions", [])}
    for path in exempt:
        assert path.is_file(), f"stale scan exemption {path}"
    offenders = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in (".py", ".c", ".h", ".cpp", ".hpp", ".txt") or path in exempt:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    offenders.append((str(path.relative_to(REPO)), match.group(0)))
    assert not offenders


def test_subsystem_imports_follow_declared_edges():
    declared = {s["id"]: set(s["depends_on"]) for s in SYSTEM["subsystems"]}
    shared = {m["module"] for m in SYSTEM["shared_modules"]}
    violations = []
    for path in python_modules(SRC):
        owner = _owner(module_name(path))
        for name in imported_modules(path):
            if not name.startswith("elpis"):
                continue
            if any(name == s or name.startswith(s + ".") for s in shared):
                continue
            target = _owner(name)
            if target is None or owner is None or target == owner:
                continue
            if module_name(path).startswith("elpis.") and ("elpis." + (owner or "")) in shared:
                violations.append((module_name(path), name))  # shared modules import nothing
                continue
            if target not in declared.get(owner, set()):
                violations.append((module_name(path), name))
    assert not violations, "undeclared cross-subsystem imports: " + json.dumps(violations, indent=1)


_PROBE = r"""
import importlib, json, sys
mod = importlib.import_module(sys.argv[1])
print(json.dumps(sorted(m.split('.')[0] for m in sys.modules)))
"""


@pytest.mark.parametrize("sub", [s["id"] for s in live_subsystems()] or ["<none>"])
def test_subsystem_import_does_not_pull_models_or_numerics(sub):
    if sub == "<none>":
        pytest.skip("no live subsystem yet")
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, "elpis." + sub],
        capture_output=True, text=True, cwd=REPO,
        env={"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    loaded = set(json.loads(result.stdout))
    forbidden = set(POLICY["forbidden_python_imports"])
    assert not loaded & forbidden
    numeric = next(s for s in SYSTEM["subsystems"] if s["id"] == sub).get("uses_numpy", False)
    if not numeric:
        assert "numpy" not in loaded, f"elpis.{sub} must not import numpy"


def test_base_package_import_is_inert():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, "elpis"],
        capture_output=True, text=True, cwd=REPO,
        env={"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    loaded = set(json.loads(result.stdout))
    assert "numpy" not in loaded and not loaded & set(POLICY["forbidden_python_imports"])


SLOW_LANES = ("elpis.ECS_C", "elpis.structure", "elpis.pipeline", "elpis.evolution", "elpis.runtime")


def _inference_modules():
    return [path for path in python_modules(SRC / "inference")]


def test_inference_token_lane_imports_no_slow_lane_statically():
    """The token lane (all of elpis.inference) names no ECS, structure, pipeline, evolution or runtime module."""
    offenders = []
    for path in _inference_modules():
        for name in imported_modules(path):
            if any(name == lane or name.startswith(lane + ".") for lane in SLOW_LANES):
                offenders.append((module_name(path), name))
    assert not offenders, offenders


_CLOSURE_PROBE = r"""
import importlib, json, pkgutil, sys
import elpis.inference as root
names = [m.name for m in pkgutil.walk_packages(root.__path__, "elpis.inference.")]
for name in names:
    importlib.import_module(name)
print(json.dumps({"imported": names, "loaded": sorted(sys.modules)}))
"""


def test_inference_token_lane_import_closure_loads_no_slow_lane():
    """Importing every inference module in a clean process loads no slow-lane module transitively."""
    result = subprocess.run(
        [sys.executable, "-c", _CLOSURE_PROBE],
        capture_output=True, text=True, cwd=REPO,
        env={"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert "elpis.inference.sequence" in report["imported"] and "elpis.inference.structural" in report["imported"]
    loaded = [m for m in report["loaded"] if any(m == lane or m.startswith(lane + ".") for lane in SLOW_LANES)]
    assert not loaded, loaded
