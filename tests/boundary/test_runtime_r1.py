"""Runtime R1 hot-path gate, static part (docs/ECS_RUNTIME_R1.md).

PYTHON MAY CONTROL THE ECS. PYTHON MUST NOT EXECUTE THE ECS HOT PATH.

The Python control plane of ECS_G (``elpis.ECS_G.native`` executor binding,
``elpis.ECS_G.cognition``) and the runtime turn admit, call native code once
and package. They contain no loop on the query/learn/commit path, never step
the state from Python, and never commit by comparing Python snapshot hashes.
The behavioural half (one native call per learn whatever K, no steady-state
allocation, native commit and staleness) lives next to the native library in
tests/ECS_G.
"""
from __future__ import annotations

import ast

import pytest

from ._system import REPO

NATIVE = REPO / "src" / "elpis" / "ECS_G" / "native.py"
CORE = REPO / "src" / "elpis" / "ECS_G" / "cognition.py"
TURN = REPO / "src" / "elpis" / "runtime" / "cognition.py"

# Strict xfail for what is not built yet; each later commit deletes its entries.
PENDING = {
}


def pending(key):
    return pytest.mark.xfail(strict=True, reason=PENDING[key]) if key in PENDING else (lambda f: f)


# Hot-path functions: admission, one native call, packaging. Nothing in them may
# iterate in Python (no for/while, no comprehension or generator).
HOT = {
    NATIVE: ("Executor.forward", "Executor.forward_into", "Executor.learn", "Transaction.learn",
             "Transaction.forward", "Transaction.commit", "_admit_rows", "_admit_vector", "_admit_out"),
    CORE: ("CognitiveCore.query", "CognitiveCore.query_into", "CognitiveCore.learn"),
}
LOOPS = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _functions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found[f"{node.name}.{item.name}"] = item
    return tree, found


def _calls(node):
    return {n.func.attr for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}


@pending("executor")
@pytest.mark.parametrize("path", sorted(HOT), ids=lambda p: p.name)
def test_hot_path_functions_exist_and_never_iterate_in_python(path):
    _, functions = _functions(path)
    for name in HOT[path]:
        assert name in functions, f"{path.name}: hot-path function {name} missing"
        loops = [type(n).__name__ for n in ast.walk(functions[name]) if isinstance(n, LOOPS)]
        assert not loops, f"{path.name}:{name} iterates in Python: {loops}"


@pending("executor")
def test_no_step_count_loop_anywhere_in_the_control_plane():
    """The K loop is native: nothing in the binding, the core or the turn ranges over a step count."""
    for path in (NATIVE, CORE, TURN):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.comprehension)) and isinstance(node.iter, ast.Call) and \
                    isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                names = {n.id for n in ast.walk(node.iter) if isinstance(n, ast.Name)} | \
                        {n.attr for n in ast.walk(node.iter) if isinstance(n, ast.Attribute)}
                assert not names & {"steps", "k", "K", "n_steps"}, f"{path.name}:{node.lineno} loops over steps"


@pending("executor")
def test_cognitive_core_learns_through_one_native_executor_call():
    _, functions = _functions(CORE)
    learn = functions["CognitiveCore.learn"]
    calls = _calls(learn)
    assert "learn" in calls, "CognitiveCore.learn must delegate to the native executor learn"
    assert not calls & {"step", "fork", "adopt"}, calls & {"step", "fork", "adopt"}
    query = functions["CognitiveCore.query"]
    assert not _calls(query) & {"snapshot", "sha256", "dumps"}


@pending("executor")
def test_commit_is_native_not_a_python_snapshot_hash_swap():
    """The binding exposes no Python-side adopt; native generation, not SHA-256 of snapshots, detects staleness."""
    _, functions = _functions(NATIVE)
    assert "WorldState.adopt" not in functions
    for name in ("Transaction.commit", "Transaction.abort", "Executor.learn"):
        assert name in functions, name
        assert not _calls(functions[name]) & {"sha256", "snapshot", "hexdigest", "digest"}, name


@pending("executor")
def test_runtime_turn_commits_through_a_native_transaction():
    """The canonical turn's substrate is now native K1 (tests/boundary/test_k1_runtime.py); its commit stays one
    native transaction commit, never a Python fork/adopt/step."""
    _, functions = _functions(TURN)
    calls = set().union(*(_calls(f) for f in functions.values()))
    assert "transaction" in calls and "commit" in calls
    assert not calls & {"fork", "adopt", "step"}, calls & {"fork", "adopt", "step"}
