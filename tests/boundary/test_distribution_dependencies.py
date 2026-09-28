# Distribution dependency boundary: base Elpis is model- and numerics-neutral.
from __future__ import annotations

import tomllib

from ._system import REPO


def test_base_distribution_has_no_runtime_python_dependencies():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project.get("dependencies", []) == []


def test_inference_numerics_are_explicit_optional_dependencies():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    assert extras["inference"] == ["numpy>=1.26,<2"]
    assert "numpy>=1.26,<2" in extras["test"]
    assert "pytest>=8" in extras["test"]
