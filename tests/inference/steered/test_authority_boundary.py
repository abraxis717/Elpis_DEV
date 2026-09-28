import ast
import dataclasses
import json
import os
import subprocess
import sys

import pytest

from elpis.inference.contracts import ProposalOnly
from elpis.inference.transaction import InferenceEngine
from elpis.inference.steered import SteeredInferenceEngine
from elpis.inference.steering import SteeringOutcome as O

from ._helpers import (AUTHORITY, CONTROLLER, SRC_PATHS, TOOL_MODULES, scripted,
                           slow_lane_module, slow_lane_path, trace_calls)

IMPORT_PROBE = """
import json, sys
sys.path[:0] = %r
import elpis.inference.steered
print(json.dumps(sorted([n, getattr(m, '__file__', None) or ''] for n, m in sys.modules.items())))
"""
def _imports(path):
    modules = set()
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            modules.add('.' * node.level + (node.module or ''))
    return modules


def _names(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return ({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} |
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)})


def test_controller_static_imports_are_exact():
    assert _imports(CONTROLLER) == {'__future__', 'dataclasses', 'elpis.identity',
                                    'elpis.inference.contracts', 'elpis.inference.transaction',
                                    'elpis.inference.steering'}


def test_no_slow_lane_ecs_tool_persistence_or_direct_digest_surface():
    for path in (CONTROLLER,):
        names = _names(path)
        lowered = {name.lower() for name in names}
        assert not [name for name in names if slow_lane_module(name)]
        assert not lowered & {'entityport', 'kernel', 'ecscontextprojector', 'branch43globalevent',
                              'global_event_fields', 'darwinianmatrix', 'evolutionpathgate', 'wiring'}
        assert not lowered & {'subprocess', 'system', 'popen', 'eval', 'exec', 'compile', '__import__', 'open',
                              'write_text', 'write_bytes', 'mkdir', 'sqlite3', 'pickle', 'shelve', 'dump', 'dumps',
                              'socket'}
        assert not lowered & {'hashlib', 'sha256', 'sha3_256', 'blake2b', 'hexdigest'}  # digest-sink census clean
        assert not {module.split('.')[0] for module in _imports(path)} & set(TOOL_MODULES)


def test_clean_interpreter_import_closure_has_no_slow_lane():
    probe = subprocess.run([sys.executable, '-c', IMPORT_PROBE % (SRC_PATHS,)], capture_output=True, text=True,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert probe.returncode == 0, probe.stderr
    modules = json.loads(probe.stdout)
    assert 'elpis.inference.steered' in {name for name, _ in modules}
    assert not [(name, path) for name, path in modules if slow_lane_module(name) or (path and slow_lane_path(path))]


def test_composed_token_path_makes_zero_slow_lane_calls(rt):
    before = {name for name in sys.modules if slow_lane_module(name)}
    (_, _, epochs), frames, c_calls = trace_calls(lambda: scripted(rt, greedy_tokens=4, block_size=1))
    files = {path for path, _ in frames}
    assert any(path.endswith(os.path.join('inference', 'steered.py')) for path in files)
    assert any(path.endswith(os.path.join('inference', 'transaction.py')) for path in files)
    assert any(e.application is not None and e.application.outcome is O.APPLIED for e in epochs)
    assert not sorted(path for path in files if slow_lane_path(path))
    assert not sorted((module, name) for module, name in c_calls if module and slow_lane_module(module))
    assert {name for name in sys.modules if slow_lane_module(name)} == before  # the token path loads no slow-lane module; the clean-process import closure is probed in a subprocess


def test_model_execution_only_through_engine_execute(rt, monkeypatch):
    calls, original = [], InferenceEngine.execute
    monkeypatch.setattr(InferenceEngine, 'execute', lambda self, *a, **k: calls.append(1) or original(self, *a, **k))
    _, _, epochs = scripted(rt, greedy_tokens=4, block_size=2)
    assert len(calls) == len(epochs)
    tree = ast.parse(CONTROLLER.read_text(encoding='utf-8'))
    attribute_calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert 'execute' in attribute_calls
    assert not attribute_calls & {'step', 'advance', '_advance_validated', 'replay', 'initial', 'propose', 'verify_draft'}


def test_request_transformation_only_via_apply_fast_steering():
    tree = ast.parse(CONTROLLER.read_text(encoding='utf-8'))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert 'apply_fast_steering' in names and 'LatentInput' not in names
    assert not attributes & {'latents', 'logits', 'hidden', 'tokens', 'weights'}
    replaces = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == 'replace']
    assert replaces and all({keyword.arg for keyword in n.keywords} == {'count'} for n in replaces)


def test_session_state_is_authority_zero(rt):
    session = SteeredInferenceEngine(rt).initial_session('req-A')
    assert isinstance(session, ProposalOnly)
    for flag in AUTHORITY:
        assert getattr(session, flag) is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        session.next_epoch_index = 5
