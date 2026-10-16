"""Real process death inside a composite durable publication (docs/ECS_COMPOSITE_STATE.md, "Authority").

``test_durable_process_kill.py`` kills a K1 lineage at every step of a recoverable publication; the composite's
simulated deaths are native (``runtime.test_runtime_composite``). Here a child process runs a managed runtime over a
real ECS composite (K1 + memory, ELPISEC1) with a composite-sized checkpoint store: it anchors, learns with one
observation and publishes recoverably (acknowledged), learns with a second observation (the candidate), parks the
durability worker at one step of the next recoverable publication (or receives that acknowledgment), then learns
with a third observation that is never requested (volatile). The parent sends SIGKILL and restarts in a fresh
process over the same directories. The memory transition must be present exactly when continuity authorizes it:

| crash point | restart | observations in the recovered composite |
|---|---|---|
| 5 slot synced, before continuity | CANDIDATE_UNRESOLVED; the operator adopts it | adopted: first and second, once each |
| 7 record written (page cache), 8 synced, not delivered | RESUMABLE at the candidate | first and second, once each |
| after the acknowledgment was delivered | RESUMABLE at the candidate | first and second, once each |
| every point | the volatile third observation is never recovered | |

SIGKILL is process death, not power loss (see ``test_durable_process_kill.py``). Bounded: four child processes.
The codec and the observations are TEST_ONLY TRAINING=NONE SEMANTICS=NONE fixtures: mechanics only.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from elpis.ECS.composite import CompositeState
from elpis.runtime import Runtime

from .test_codec_ecs_turn import _config
from .test_composite_runtime import AUTHORITY, CONFIG, _learn, _store, library, obs  # noqa: F401 (fixture)

REPO = Path(__file__).resolve().parents[2]

_CHILD = r"""
import json, sys, time
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tests.integration.test_codec_ecs_turn import _config
from tests.integration.test_composite_runtime import AUTHORITY, CONFIG, _learn, fresh, obs
from tests.conftest import admit_composite_set
from elpis.ECS.composite import CompositeLibrary
from elpis.runtime import Runtime

continuity, checkpoint, mode = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
say = lambda **kw: (sys.stdout.write(json.dumps(kw) + "\n"), sys.stdout.flush())
x = fresh(CompositeLibrary(admit_composite_set()[3].lib))
runtime = Runtime(replace(_config(continuity, testing=True), k1_checkpoint_dir=checkpoint)).open()
runtime.anchor_cognition(x)
learn = lambda observation: runtime.run_learn(x, _learn(), observations=(observation,),
                                              observation_authority=AUTHORITY).observed[0]
first = learn(obs(1, 2, 10))
acked = runtime.publish_recoverable(x)
assert acked.acknowledged and acked.recoverable
second = learn(obs(3, 4, 11))
candidate = runtime.live_identity(x)[0]
if mode.startswith("park:"):
    step = int(mode[5:])
    runtime._core.testing_durable_fault(2, step, 0)      # park the worker at `step`
    pending = runtime.publish_recoverable(x, wait=False)
    third = learn(obs(5, 6, 12))                          # the HOT lane is not blocked by the park
    while runtime.durability().parked != step:
        time.sleep(0.005)
    status = pending.status
else:                                                     # "delivered": the acknowledgment reached the caller
    ticket = runtime.publish_recoverable(x)
    assert ticket.acknowledged and ticket.identity == candidate
    third = learn(obs(5, 6, 12))
    status = ticket.status
say(authorized=acked.identity.hex(), candidate=candidate.hex(), volatile=runtime.live_identity(x)[0].hex(),
    keys=[first.hex(), second.hex(), third.hex()], status=status)
time.sleep(3600)                                          # killed here
"""


def _child(tmp_path, mode):
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.Popen([sys.executable, "-c", _CHILD, str(REPO), str(tmp_path / "continuity"),
                             str(tmp_path / "checkpoint"), mode],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=REPO)


def _killed_report(child):
    try:
        line = child.stdout.readline()
        assert line, child.stderr.read()
        return json.loads(line)
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait(timeout=30)
        assert child.returncode == -signal.SIGKILL


@pytest.mark.parametrize("mode,status,disposition", [("park:5", "PENDING", "CANDIDATE_UNRESOLVED"),
                                                     ("park:7", "PENDING", "RESUMABLE"),
                                                     ("park:8", "PENDING", "RESUMABLE"),
                                                     ("delivered", "ACKNOWLEDGED", "RESUMABLE")])
def test_sigkill_in_a_composite_publication_recovers_the_memory_transition_exactly_once_when_authorized(
        library, tmp_path, mode, status, disposition):   # noqa: F811
    _store(library, tmp_path / "checkpoint")
    report = _killed_report(_child(tmp_path, mode))
    assert report["status"] == status
    authorized, candidate, volatile = (bytes.fromhex(report[k]) for k in ("authorized", "candidate", "volatile"))
    first, second, third = (bytes.fromhex(k) for k in report["keys"])
    assert len({authorized, candidate, volatile}) == 3 and len({first, second, third}) == 3
    with Runtime(replace(_config(tmp_path / "continuity"), k1_checkpoint_dir=tmp_path / "checkpoint")) as runtime:
        durable = runtime.continuity.snapshot()
        assert durable.state_format == "COMPOSITE"
        recovery = runtime.recover_k1()
        assert recovery.disposition == disposition, (mode, recovery)
        if disposition == "CANDIDATE_UNRESOLVED":
            assert (durable.k1_state_digest, recovery.authorized, recovery.candidate) == (authorized, authorized,
                                                                                           candidate)
            runtime.adopt_k1_candidate(candidate)             # the operator's explicit decision
            recovery = runtime.recover_k1()
        assert recovery.disposition == "RESUMABLE" and recovery.candidate is None
        assert runtime.continuity.snapshot().k1_state_digest == candidate != volatile
        with CompositeState.restore(library, CONFIG, recovery.envelope) as y:
            assert y.state_digest() == candidate
            assert [y.slot(k).assert_count if y.slot(k) else 0 for k in (first, second, third)] == [1, 1, 0]
            again = runtime.run_learn(y, _learn(), observations=(obs(5, 6, 12),), observation_authority=AUTHORITY)
            assert again.observed == (third,) and y.slot(third).assert_count == 1
            ticket = runtime.publish_recoverable(y)
            assert ticket.acknowledged and runtime.continuity.snapshot().k1_state_digest == y.state_digest()
