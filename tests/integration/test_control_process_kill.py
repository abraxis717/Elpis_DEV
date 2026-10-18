"""Real process death inside a control operation: the anchor, the legacy import, operator discard and adopt.

These run outside the durability lane's step machine (docs/RUNTIME_CORE.md, "Control-operation crash points"), so the
testing library parks their caller instead of the worker (``testing_control_fault``: the caller blocks, holding the
handle, at exactly the point's on-disk effect). A child process parks inside the operation on a thread; the parent
sends SIGKILL and restarts in a fresh process over the same directories:

| operation, point | restart |
|---|---|
| anchor, 3 slot synced (continuity not begun) | NOTHING_TO_RECOVER: a slot never creates a lineage; an explicit anchor then resumes |
| anchor, 5 record written (page cache) | RESUMABLE: the anchored state |
| discard, 2 the candidate's header zeroed, not synced | RESUMABLE: the authorized state, no candidate |
| adopt, 4 record torn | CANDIDATE_UNRESOLVED: a torn record is not an authority; the operator adopts again |
| adopt, 6 record synced, not returned | RESUMABLE: the candidate |
| import, 3 composite slot synced | the K1 authority (the composite slot is no candidate); an explicit import then resumes |
| import, 5 record written | the composite, RESUMABLE from the composite store |

The full matrix (every point, both lineages, repetitions) is K1 Recovery R0 v4's. The production library exports
none of the testing entry points. SIGKILL is process death, not power loss. The codec and the observations are
TEST_ONLY TRAINING=NONE SEMANTICS=NONE fixtures: mechanics only.
"""
from __future__ import annotations

import ctypes
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from elpis.ECS.composite import CompositeState
from elpis.ECS.k1 import K1State
from elpis.runtime import Runtime

from .test_codec_ecs_turn import _config, k1, world  # noqa: F401 (the k1 fixture)
from .test_composite_runtime import CONFIG, _store, library  # noqa: F401 (fixture)
from .test_k1_recovery import _learn, _provision
from ..conftest import require_runtime_library

REPO = Path(__file__).resolve().parents[2]

_CHILD = r"""
import json, sys, threading, time
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tests.conftest import admit_composite_set, admit_k1_set
from tests.integration.test_codec_ecs_turn import _config, world
from tests.integration.test_composite_runtime import CONFIG
from tests.integration.test_k1_recovery import _learn
from elpis.ECS.composite import CompositeLibrary, CompositeState
from elpis.ECS.k1 import K1Library
from elpis.runtime import Runtime

root, mode = Path(sys.argv[2]), sys.argv[3]
k1 = K1Library(admit_k1_set()[1].lib)
say = lambda **kw: (sys.stdout.write(json.dumps(kw) + "\n"), sys.stdout.flush())
open_at = lambda store: Runtime(replace(_config(root / "continuity", testing=True), k1_checkpoint_dir=store)).open()
OPS = {"anchor": 1, "import": 2, "discard": 3, "adopt": 4}


def parked(runtime, value):
    deadline = time.monotonic() + 60                  # a hook that never parks fails the test, never hangs it
    while runtime.durability().parked != value:
        assert time.monotonic() < deadline, f"never parked at {value}"
        time.sleep(0.002)


def park(runtime, op, point, call):
    runtime._core.testing_control_fault(OPS[op], 2, point, 40)
    threading.Thread(target=call, daemon=True).start()
    parked(runtime, 100 * OPS[op] + point)


op, point = mode.split(":")[0], int(mode.split(":")[1]) if ":" in mode else 0
if op == "candidate":                                 # D1 acknowledged, D2 a candidate (worker killed at step 5)
    state = world(k1, max_rows=8)
    runtime = open_at(root / "checkpoint")
    runtime.anchor_cognition(state)
    runtime.run_learn(state, _learn("one"))
    d1 = runtime.publish_recoverable(state).identity
    runtime.run_learn(state, _learn("two"))
    d2 = runtime.live_identity(state)[0]
    runtime._core.testing_durable_fault(2, 5, 0)
    runtime.publish_recoverable(state, wait=False)
    parked(runtime, 5)
    say(d1=d1.hex(), d2=d2.hex())
elif op == "anchor":
    state = world(k1, max_rows=8)
    runtime = open_at(root / "checkpoint")
    park(runtime, op, point, lambda: runtime.anchor_cognition(state))
    say(d0=state.state_digest().hex())
elif op in ("discard", "adopt"):
    runtime = open_at(root / "checkpoint")
    candidate = runtime.recover_k1().candidate
    call = runtime.discard_k1_candidate if op == "discard" else runtime.adopt_k1_candidate
    park(runtime, op, point, lambda: call(candidate))
    say(candidate=candidate.hex())
else:                                                 # import: a K1 lineage publishes K, then moves to a composite
    state = world(k1, max_rows=8)
    runtime = open_at(root / "checkpoint")
    runtime.anchor_cognition(state)
    runtime.run_learn(state, _learn("one"))
    k = runtime.publish_recoverable(state).identity
    runtime.close()
    composite = CompositeState.import_k1(CompositeLibrary(admit_composite_set()[3].lib), CONFIG, state.snapshot())
    runtime = open_at(root / "composite")
    park(runtime, op, point, lambda: runtime.import_composite(composite))
    say(k=k.hex(), c=composite.state_digest().hex())
time.sleep(3600)                                      # killed here
"""


def _killed(tmp_path, mode) -> dict:
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1")
    child = subprocess.Popen([sys.executable, "-c", _CHILD, str(REPO), str(tmp_path), mode], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=env, cwd=REPO)
    try:
        line = child.stdout.readline()
        assert line, child.stderr.read()
        return json.loads(line)
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait(timeout=30)
        assert child.returncode == -signal.SIGKILL


def _runtime(tmp_path, store):
    return Runtime(replace(_config(tmp_path / "continuity"), k1_checkpoint_dir=store))


@pytest.mark.parametrize("point,disposition", [(3, "NOTHING_TO_RECOVER"), (5, "RESUMABLE")])
def test_sigkill_inside_the_anchor_leaves_no_lineage_or_exactly_the_anchored_one(k1, tmp_path, point,   # noqa: F811
                                                                                 disposition):
    store = _provision(k1, tmp_path / "checkpoint")
    d0 = bytes.fromhex(_killed(tmp_path, f"anchor:{point}")["d0"])
    with _runtime(tmp_path, store) as runtime:
        assert runtime.recover_k1().disposition == disposition
        if disposition == "NOTHING_TO_RECOVER":
            with world(k1, max_rows=8) as again:                      # the caller's same state, anchored explicitly
                assert again.state_digest() == d0
                runtime.anchor_cognition(again)
                runtime.release(again)
    with _runtime(tmp_path, store) as runtime:
        recovery = runtime.recover_k1()
        assert (recovery.disposition, recovery.authorized) == ("RESUMABLE", d0)
        with K1State.restore(k1, recovery.envelope, max_rows=8) as resumed:
            runtime.run_learn(resumed, _learn("after"))
            assert runtime.publish_recoverable(resumed).acknowledged


@pytest.mark.parametrize("op,point,disposition,final", [("discard", 2, "RESUMABLE", "d1"),
                                                        ("adopt", 4, "CANDIDATE_UNRESOLVED", "d2"),
                                                        ("adopt", 6, "RESUMABLE", "d2")])
def test_sigkill_inside_a_reconciliation_leaves_the_candidate_or_its_resolution(k1, tmp_path, op, point,  # noqa: F811
                                                                                disposition, final):
    store = _provision(k1, tmp_path / "checkpoint")
    ids = {k: bytes.fromhex(v) for k, v in _killed(tmp_path, "candidate").items()}
    with _runtime(tmp_path, store) as runtime:
        assert runtime.recover_k1().candidate == ids["d2"]
    assert bytes.fromhex(_killed(tmp_path, f"{op}:{point}")["candidate"]) == ids["d2"]
    with _runtime(tmp_path, store) as runtime:
        recovery = runtime.recover_k1()
        assert recovery.disposition == disposition, recovery
        if disposition == "CANDIDATE_UNRESOLVED":
            assert (recovery.authorized, recovery.candidate) == (ids["d1"], ids["d2"])
            runtime.adopt_k1_candidate(ids["d2"])                    # nothing is adopted but explicitly
            recovery = runtime.recover_k1()
        assert (recovery.disposition, recovery.authorized, recovery.candidate) == ("RESUMABLE", ids[final], None)


@pytest.mark.parametrize("point,state_format", [(3, "K1"), (5, "COMPOSITE")])
def test_sigkill_inside_the_import_leaves_the_k1_lineage_or_the_composite(k1, library, tmp_path, point,   # noqa: F811
                                                                          state_format):
    _provision(k1, tmp_path / "checkpoint")
    composite_store = _store(library, tmp_path / "composite")
    ids = {k: bytes.fromhex(v) for k, v in _killed(tmp_path, f"import:{point}").items()}
    with _runtime(tmp_path, composite_store) as runtime:
        durable = runtime.continuity.snapshot()
        assert durable.state_format == state_format
        recovery = runtime.recover_k1()
        if state_format == "K1":
            assert (durable.k1_state_digest, recovery.disposition) == (ids["k"], "CHECKPOINT_MISSING")
            assert recovery.candidate is None                         # the composite slot authorizes nothing
    if state_format == "K1":
        with _runtime(tmp_path, tmp_path / "checkpoint") as runtime:
            envelope = runtime.recover_k1().envelope                  # the authorized K1 state, from its own store
        with CompositeState.import_k1(library, CONFIG, envelope) as again, \
                _runtime(tmp_path, composite_store) as runtime:
            assert again.state_digest() == ids["c"]
            runtime.import_composite(again)                           # the operator imports again, explicitly
    with _runtime(tmp_path, composite_store) as runtime:
        recovery = runtime.recover_k1()
        assert (recovery.disposition, recovery.authorized) == ("RESUMABLE", ids["c"])
        with CompositeState.restore(library, CONFIG, recovery.envelope) as resumed:
            assert resumed.state_digest() == ids["c"]


def test_the_production_library_exports_no_testing_entry_point():
    production, testing = require_runtime_library(testing=False), require_runtime_library(testing=True)
    names = ("elpis_runtime_testing_fault", "elpis_runtime_testing_io_counters", "elpis_runtime_testing_durable_hold",
             "elpis_runtime_testing_durable_fault", "elpis_runtime_testing_control_fault",
             "elpis_runtime_testing_checkpoint_io_counters", "elpis_continuity_testing_fault")
    prod, test = ctypes.CDLL(str(production)), ctypes.CDLL(str(testing))
    assert not [n for n in names if hasattr(prod, n)]
    assert all(hasattr(test, n) for n in names)


def _slot_identities(store):
    """The identities the slot headers name on disk (evidence only; a slot authorizes nothing)."""
    out = set()
    for p in store.iterdir():
        header = p.read_bytes()[:64]
        if header[:8] == b"ELPK1CK\x01":
            out.add(header[32:64])
    return out


@pytest.mark.parametrize("point,state_format,slot_holds_composite", [
    (1, "K1", False), (2, "K1", True), (3, "K1", True), (4, "K1", True), (5, "COMPOSITE", True),
    (6, "COMPOSITE", True)])
def test_a_death_at_every_import_point_leaves_the_k1_lineage_or_the_composite(k1, library, tmp_path, point,  # noqa: F811
                                                                              state_format, slot_holds_composite):
    """In process, over the real libraries (simulated death, testing_control_fault action 1), every registered point
    of the legacy import: the death has exactly its point's on-disk effect, and a fresh runtime decides from continuity
    alone. The real SIGKILL matrix of every point is K1 Recovery R0 v4's."""
    from elpis.runtime.core import ContinuityProcessDeath

    k1_store = _provision(k1, tmp_path / "checkpoint")
    composite_store = _store(library, tmp_path / "composite")
    testing = lambda store: Runtime(replace(_config(tmp_path / "continuity", testing=True), k1_checkpoint_dir=store))
    with world(k1, max_rows=8) as state:
        with testing(k1_store) as runtime:
            runtime.anchor_cognition(state)
            runtime.run_learn(state, _learn("one"))
            k = runtime.publish_recoverable(state).identity
        with CompositeState.import_k1(library, CONFIG, state.snapshot()) as composite:
            c = composite.state_digest()
            runtime = testing(composite_store).open()
            runtime._core.testing_control_fault(2, 1, point, 40)
            with pytest.raises(ContinuityProcessDeath):
                runtime.import_composite(composite)
            assert runtime.fault == "CONTINUITY_TESTING_PROCESS_DEATH"   # the process is "gone"
            runtime.close()
    assert (c in _slot_identities(composite_store)) is slot_holds_composite
    with _runtime(tmp_path, composite_store) as runtime:
        durable, recovery = runtime.continuity.snapshot(), runtime.recover_k1()
    assert durable.state_format == state_format
    if state_format == "K1":
        assert (durable.k1_state_digest, recovery.disposition, recovery.candidate) == (k, "CHECKPOINT_MISSING", None)
        with _runtime(tmp_path, k1_store) as runtime:
            assert (runtime.recover_k1().disposition, runtime.recover_k1().authorized) == ("RESUMABLE", k)
    else:
        assert (durable.k1_state_digest, recovery.disposition) == (c, "RESUMABLE")
        with CompositeState.restore(library, CONFIG, recovery.envelope) as resumed:
            assert resumed.state_digest() == c and resumed.inspect()["memory_slots_used"] == 0
