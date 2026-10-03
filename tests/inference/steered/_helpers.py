"""Shared steered-decoding test helpers."""
import hashlib
import os
import sys
import threading
from pathlib import Path

from elpis.inference.context import initial_snapshot
from elpis.inference.transaction import InferenceRequest
from elpis.inference.steered import SteeredInferenceEngine

REPO = Path(__file__).resolve().parents[3]
CONTROLLER = REPO / 'src' / 'elpis' / 'inference' / 'steered.py'
SRC_PATHS = [str(REPO / 'src')]
AUTHORITY = ('semantic_authority', 'admission_authority', 'execution_authority', 'mutation_authority',
             'runtime_admission')
# Subsystems the steered decoding path must never touch.
SLOW_LANE_MODULES = ('elpis.evolution', 'elpis.ECS_C', 'elpis.pipeline', 'elpis.runtime', 'elpis.structure.grid81')
SLOW_LANE_PREFIXES = ()
SLOW_LANE_TAGS = ('darwin', 'evolution')
SLOW_LANE_ROOTS = tuple(str(REPO / 'src' / 'elpis' / name) for name in ('evolution', 'ecs', 'pipeline', 'runtime'))
TOOL_MODULES = ('subprocess', 'os', 'shutil', 'importlib', 'ctypes', 'socket', 'multiprocessing', 'sqlite3',
                'pickle', 'shelve', 'json', 'threading', 'asyncio', 'time', 'random', 'hashlib', 'pathlib')


def D(label):
    return hashlib.sha256(label.encode()).hexdigest()


def prefill(request_id, ctx, tokens=(1, 2, 3), **kwargs):
    return InferenceRequest(request_id, ctx.digest, 'PREFILL', tokens, **kwargs)


def greedy(request_id, ctx, count=1, **kwargs):
    return InferenceRequest(request_id, ctx.digest, 'GREEDY', count=count, **kwargs)


def scripted(runtime, request_id='req-A', greedy_tokens=4, block_size=1, ctx=None):
    """Prefill epoch, then block-partitioned GREEDY epochs, through the composed runtime."""
    ctx = initial_snapshot() if ctx is None else ctx
    controller = SteeredInferenceEngine(runtime)
    first = controller.execute_epoch(controller.initial_session(request_id), runtime.initial(ctx),
                                     prefill(request_id, ctx))
    rest = controller.generate(first.session, first.runtime_result.state,
                               greedy(request_id, ctx, count=greedy_tokens), block_size=block_size)
    return ctx, controller, (first,) + rest


def two_epochs(runtime, count=2):
    ctx = initial_snapshot()
    controller = SteeredInferenceEngine(runtime)
    e0 = controller.execute_epoch(controller.initial_session('req-A'), runtime.initial(ctx), prefill('req-A', ctx))
    e1 = controller.execute_epoch(e0.session, e0.runtime_result.state, greedy('req-A', ctx, count=count))
    return ctx, controller, e0, e1


def digest_vector(epochs):
    return [dict(
        epoch=e.epoch_index, provenance=e.provenance_digest, state=e.runtime_result.state.digest,
        receipt=e.runtime_result.receipt.digest, host_request=e.host_request.digest,
        executed_request=e.executed_request.digest, binding=e.binding.digest, observer=e.observer.digest,
        proposal=e.proposal.digest, application=None if e.application is None else e.application.digest,
        outcome=None if e.application is None else e.application.outcome.value,
        applied_latent=None if e.application is None else e.application.applied_latent_digest,
        control=e.session.control_state.digest, session=e.session.digest) for e in epochs]


def strip_timing(value):
    if isinstance(value, dict):
        return {key: strip_timing(item) for key, item in value.items() if not key.endswith('_ns')}
    if isinstance(value, (list, tuple)):
        return [strip_timing(item) for item in value]
    return value


def key_paths(value, prefix=''):
    paths = set()
    if isinstance(value, dict):
        for key, item in value.items():
            paths.add(prefix + '/' + key)
            paths |= key_paths(item, prefix + '/' + key)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            paths |= key_paths(item, prefix + '/' + str(index))
    return paths


def slow_lane_module(name):
    lowered = name.lower()
    return (any(name == m or name.startswith(m + '.') for m in SLOW_LANE_MODULES) or
            (bool(SLOW_LANE_PREFIXES) and name.startswith(SLOW_LANE_PREFIXES)) or
            any(tag in lowered for tag in SLOW_LANE_TAGS))


def slow_lane_path(path):
    path = os.path.abspath(path)
    repo_local = path.startswith(str(REPO) + os.sep)
    return (any(path == root or path.startswith(root + os.sep) for root in SLOW_LANE_ROOTS) or
            (repo_local and any(tag in path.lower() for tag in SLOW_LANE_TAGS)))


def trace_calls(action):
    frames, c_calls = set(), set()

    def profiler(frame, event, arg):
        if event == 'call':
            frames.add((frame.f_code.co_filename, frame.f_code.co_name))
        elif event == 'c_call':
            c_calls.add((getattr(arg, '__module__', None) or '', getattr(arg, '__name__', None) or ''))

    set_all = getattr(threading, 'setprofile_all_threads', None)
    if set_all:
        set_all(profiler)
    sys.setprofile(profiler)
    threading.setprofile(profiler)
    try:
        result = action()
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
        if set_all:
            set_all(None)
    return result, frames, c_calls
