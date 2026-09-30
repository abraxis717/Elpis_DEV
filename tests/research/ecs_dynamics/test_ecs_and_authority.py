"""Public-ECS coarse collision, and proof that the laboratory has no authority over production code or state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from research.ecs_dynamics import ecs_collision as E
from research.ecs_dynamics.export import DynamicsObservation
from research.ecs_dynamics.results import ExperimentResult, ResultError

REPO = Path(__file__).resolve().parents[3]


def test_minimal_pair_shares_coarse_state_and_diverges_on_targets():
    labels, a, b = E.MINIMAL_PAIR
    r = E.compare_pair(labels, a, b, depth=1, rounds=2)
    assert r["histories_distinct"] and not r["full_equal"]
    assert r["coarse_equal"] and r["weak_equal"]
    assert r["event_differs"] and r["trajectory_differs"]
    assert r["first_divergent_round"] == 1 and not r["endpoint_differs"]  # diverges, then re-converges
    assert not r["delay_equal"]  # C_{t-1} tells the two histories apart
    assert r["analysis_read_only"]


def test_identical_histories_do_not_diverge():
    labels, a, _ = E.MINIMAL_PAIR
    r = E.compare_pair(labels, a, a, depth=1, rounds=2)
    assert r["coarse_equal"] and r["full_equal"] and not r["event_differs"] and not r["trajectory_differs"]


def _tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def test_futures_run_on_copies_and_never_touch_the_original_history():
    labels, a, _ = E.MINIMAL_PAIR
    h = E.History(labels, a)
    try:
        before = _tree_digest(Path(h.path))
        E.next_processed_edge(h)
        E.reply_policy_future(h, 2)
        h.prefix_coarse(1)
        assert _tree_digest(Path(h.path)) == before
    finally:
        h.cleanup()
    assert not Path(h.root).exists()


def test_coarse_state_drops_only_the_projection_binding_digest():
    labels, a, _ = E.MINIMAL_PAIR
    h = E.History(labels, a)
    try:
        c = E.coarse_state(h.analysis)
        assert "topology_digest" not in c
        assert set(h.analysis.to_dict()) - set(c) == {"topology_digest"}
    finally:
        h.cleanup()


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
    probe = "import sys, elpis.ecs, elpis.ecs.kernel; print(sorted(m for m in sys.modules if m.startswith('research')))"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=REPO,
                         env={"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"}, check=True)
    assert out.stdout.strip() == "[]"


def test_neural_law_is_not_in_the_ecs_kernel_and_numpy_is_not_a_base_dependency():
    kernel = (REPO / "src" / "elpis" / "ecs" / "kernel.py").read_text()
    for token in ("tanh", "numpy", "research", "phi(z)"):
        assert token not in kernel
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
