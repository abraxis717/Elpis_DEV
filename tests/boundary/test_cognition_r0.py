"""Cognitive R0 anti-drift gate, static part (docs/COGNITION_R0.md).

ECS is the model: the cognitive core lives in ECS_G and depends on nothing but
ECS_G and the standard library, and neither it nor its qualification can be
satisfied by a DSV model, attention, MoE/experts or an external learned
predictor. The behavioural half (responses depend on ECS state) is in
tests/ECS_G/test_cognition_r0_contract.py, next to the native library.
"""
from __future__ import annotations

import ast
import re
import sys

import pytest

from . import _mission as M
from ._system import REPO

CORE = REPO / "src" / "elpis" / "ECS_G" / "cognition.py"
LAB = REPO / "research" / "ecs_cognition_r0"

# Strict xfail for what is not built yet; each later commit deletes its entries.
PENDING = {
    "core": "Cognitive R0 core not implemented yet (K2)",
    "lab": "Cognitive R0 qualification laboratory not created yet (K3)",
}


def pending(key):
    return pytest.mark.xfail(strict=True, reason=PENDING[key]) if key in PENDING else (lambda f: f)


# Anything that would put another model in the middle.
FORBIDDEN_MODULES = ("research.dsv41_tower", "research.ecs_dynamics", "elpis.inference", "elpis.runtime",
                     "torch", "transformers", "sklearn", "jax", "tensorflow")
FORBIDDEN_NAMES = re.compile(r"(?i)\b(DSV41Target|PrincipalEngine|InferenceEngine|dsv41_tower|attention|"
                             r"moe|experts?|transformer|logits?|tokeni[sz]er)\b")


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(("." * node.level) + (node.module or ""))
    return names


@pending("core")
def test_cognitive_core_lives_in_ecs_g_and_imports_only_ecs_g_and_stdlib():
    assert CORE.is_file()
    stdlib = set(sys.stdlib_module_names)
    for name in _imports(CORE):
        if name.startswith("."):
            assert name in (".native",), name
            continue
        top = name.split(".")[0]
        assert top in stdlib or name == "elpis.ECS_G.native", name


@pending("core")
def test_cognitive_core_names_no_model_machinery():
    text = CORE.read_text(encoding="utf-8")
    code = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    assert not FORBIDDEN_NAMES.search(code), FORBIDDEN_NAMES.search(code).group(0)


@pending("lab")
def test_cognition_qualification_lab_uses_no_other_model():
    sources = sorted(LAB.glob("*.py"))
    assert sources, "laboratory missing"
    for path in sources:
        for name in _imports(path):
            assert not any(name == m or name.startswith(m + ".") for m in FORBIDDEN_MODULES), (path.name, name)
        code = "\n".join(line.split("#", 1)[0] for line in path.read_text(encoding="utf-8").splitlines())
        assert not FORBIDDEN_NAMES.search(code), (path.name, FORBIDDEN_NAMES.search(code).group(0))


def test_ecs_g_package_stays_dependency_free():
    """ECS_G (binding and any cognition) imports nothing beyond itself and the standard library."""
    assert not M.ecs_independence(REPO)
