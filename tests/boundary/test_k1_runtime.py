"""K1 runtime hot-path gate, static part (docs/ECS_K1_RUNTIME.md).

PYTHON MAY CONTROL THE ECS. PYTHON MUST NOT EXECUTE THE ECS HOT PATH.

``elpis.ECS_G.k1`` admits, calls native code once and packages: no loop on a query, learn, consolidate or commit path,
no step-count loop anywhere, no cognitive mathematics, no NumPy. The residency adapter keeps FMS generic: it uses the
public FMS API only, and FMS names no K1 vocabulary. The behavioural half lives in tests/ECS_G/test_k1_runtime.py and
the native tests (ctest ECS_G.test_ecsg_k1*).
"""
from __future__ import annotations

import ast
import re

import pytest

from . import _mission as M
from ._system import REPO

K1 = REPO / "src" / "elpis" / "ECS_G" / "k1.py"
ADAPTER = REPO / "native" / "ECS_G" / "src" / "ecsg_k1_fms.c"
HOT = ("K1State.query", "K1State.query_into", "K1State.learn", "K1State.consolidate", "K1State.reset",
       "K1Transaction.learn", "K1Transaction.consolidate", "K1Transaction.query", "K1Transaction.commit",
       "K1FMSRuntime.query", "K1FMSRuntime.query_into", "K1FMSRuntime.learn", "K1FMSRuntime.consolidate",
       "_FMSTransaction.learn", "_FMSTransaction.consolidate", "_FMSTransaction.query", "_FMSTransaction.commit",
       "K1Transaction.run_schedule", "_FMSTransaction.run_schedule", "_admit_schedule", "_prepared")
LOOPS = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _functions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    found[f"{node.name}.{item.name}"] = item
        elif isinstance(node, ast.FunctionDef):
            found[node.name] = node
    return tree, found


@pytest.mark.parametrize("name", HOT)
def test_hot_path_functions_never_iterate_in_python(name):
    _, functions = _functions(K1)
    assert name in functions, name
    loops = [type(n).__name__ for n in ast.walk(functions[name]) if isinstance(n, LOOPS)]
    assert not loops, f"{name} iterates in Python: {loops}"


def test_no_step_count_loop_and_no_cognitive_mathematics_in_the_control_plane():
    tree, _ = _functions(K1)
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.comprehension)) and isinstance(node.iter, ast.Call) and \
                isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
            raise AssertionError(f"k1.py:{node.lineno} ranges in Python")
    code = "\n".join(line.split("#", 1)[0] for line in K1.read_text(encoding="utf-8").splitlines())
    for forbidden in (r"\bmath\.", r"\*\*\s*3", r"\bsum\(", r"\bnumpy\b", r"\bs3_vjp\b", r"\bjacobian\b"):
        assert not re.search(forbidden, code), forbidden
    imports = M.imports_of(REPO, K1)
    roots = {name if name.startswith("elpis.") else name.split(".", 1)[0] for name in imports}
    roots = {"elpis.ECS_G.native" if name.startswith("elpis.ECS_G.native") else name for name in roots}
    assert roots <= {"__future__", "ctypes", "struct", "types", "elpis.ECS_G.native"}, imports


def test_residency_adapter_uses_public_fms_and_keeps_fms_generic():
    source = ADAPTER.read_text(encoding="utf-8")
    for call in ("fms_acquire(", "fms_release(", "fms_register(", "fms_unregister(", "fms_query("):
        assert call in source, call
    for forbidden in ("fms_core", "slot_t", "PyObject", "DSV", "elpis_ecsg_executor_restore", "fopen(", "pwrite("):
        assert forbidden not in source, forbidden
    assert not M.fms_genericity(REPO)
    assert not M.ecs_independence(REPO)
    assert not M.ecsg_kernel_unchanged(REPO), "the pinned Runtime R1 kernel sources must not change"


# --- the canonical turn (elpis.runtime.cognition): codec -> native K1 schedule -> codec ------------------------

TURN = REPO / "src" / "elpis" / "runtime" / "cognition.py"
# Calls that would execute or copy ECS state in Python on the turn: none may appear in run_turn or in admission.
ECS_DATA_PLANE = {"learn", "consolidate", "learn_schedule", "s3", "w", "h_packed", "a", "snapshot", "restore",
                  "query", "query_into", "forward", "reset", "copy_w", "tolist", "tobytes"}


def _calls(node):
    return {n.func.attr for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}


def test_the_canonical_turn_runs_the_experience_schedule_natively():
    _, functions = _functions(TURN)
    turn = functions["run_turn"]
    calls = _calls(turn)
    assert {"transaction", "run_schedule", "commit"} <= calls, calls
    assert not calls & ECS_DATA_PLANE, calls & ECS_DATA_PLANE
    # The only Python iteration in the turn is over the decoded output token IDs (the token boundary).
    for node in ast.walk(turn):
        if isinstance(node, (ast.For, ast.comprehension)):
            assert isinstance(node.iter, ast.Name) and node.iter.id == "output", ast.unparse(node.iter)
        assert not isinstance(node, (ast.While, ast.AsyncFor)), ast.unparse(node)


def test_stimulus_admission_never_walks_ecs_values():
    _, functions = _functions(TURN)
    for name in ("Stimulus.__init__", "_owned"):
        node = functions[name]
        assert not _calls(node) & ECS_DATA_PLANE, (name, _calls(node) & ECS_DATA_PLANE)
        for loop in ast.walk(node):
            if isinstance(loop, (ast.For, ast.comprehension)):   # only the fixed tuple of field names
                assert isinstance(loop.iter, ast.Tuple), (name, ast.unparse(loop.iter))
            assert not isinstance(loop, (ast.While, ast.AsyncFor, ast.ListComp, ast.GeneratorExp)), name


def test_the_canonical_turn_substrate_is_native_k1_with_no_model_framework():
    names = M.imports_of(REPO, TURN)
    assert any(n.startswith("elpis.ECS_G.k1") for n in names), names
    assert not [n for n in names if n.startswith("elpis.ECS_G.native")], names   # not the Runtime R1 Executor
    for root in ("numpy", "torch", "elpis.inference", "research", "elpis.ECS_C"):
        assert not [n for n in names if n == root or n.startswith(root + ".")], (root, names)
    code = "\n".join(line.split("#", 1)[0] for line in TURN.read_text(encoding="utf-8").splitlines())
    for forbidden in (r"\bmath\.(exp|tanh|sqrt)", r"\*\*\s*3", r"\bs3_vjp\b", r"\bjacobian\b", r"\bSigma\b ="):
        assert not re.search(forbidden, code), forbidden
