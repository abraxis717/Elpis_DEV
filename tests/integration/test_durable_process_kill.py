"""Gate D with real process death: SIGKILL inside a recoverable publication, then restart in a fresh process.

Durable publication, caller acknowledgment and unknown outcome are three different things. A child process runs a
managed runtime over real native K1 with an operator-provisioned checkpoint store: it anchors, learns, publishes
recoverably (acknowledged), learns again (volatile work), and requests one more recoverable publication whose
durability worker is parked at one step of the publication (the testing library's park: the worker stops with exactly
that step's effect on disk), or (the last crash point) waits until that acknowledgment was delivered to it. The parent
then kills the child with SIGKILL: no destructor, no close, no flush. A fresh process restarts over the same
directories and must recover exactly what continuity authorizes:

| crash point | parked at | on disk | restart |
|---|---|---|---|
| before the checkpoint write | 1 taken, 2 sealed | nothing written | RESUMABLE: the acknowledged state; no candidate |
| during a partial checkpoint write | 3 slot torn | a partial slot | RESUMABLE: a torn slot is not a checkpoint |
| after the checkpoint write / sync, before continuity | 4 slot written, 5 slot synced | a complete candidate slot | CANDIDATE_UNRESOLVED until an operator decides |
| during a partial continuity write | 6 record torn | the slot, a partial record | CANDIDATE_UNRESOLVED: a torn record is not an authority |
| after the continuity write, before its sync | 7 record written | the slot and the complete record (page cache) | RESUMABLE: the new state |
| after the durable sync, before the acknowledgment was delivered | 8 record synced, 9 acknowledged | the slot and the synced record | RESUMABLE: the new state |
| after the acknowledgment was delivered | (not parked) | the slot and the synced record | RESUMABLE: the acknowledged state |

At steps 7..9 the caller never received its acknowledgment, yet restart resumes the requested state: a lost
acknowledgment is not evidence that nothing became durable. Restart recovers only a complete, authorized durable
state, which may be a publication that completed immediately before the caller lost its acknowledgment; never the
volatile work after the last request, never an unauthorized candidate.

Process kill is not power loss. SIGKILL leaves the kernel's page cache intact, so a written but unsynced record (step
7) survives and is the authority; after a power loss it may not, and restart would find the previous authority (the
lane's simulated sync-loss faults cover that branch in-process: tests/integration/test_k1_recovery.py). Nothing here
proves power-loss durability, which depends on the device honoring fdatasync. A randomized loop kills a
child that learns and publishes continuously at random moments. The fixture map is TEST_ONLY TRAINING=NONE
SEMANTICS=NONE: mechanics only.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys

import pytest

from elpis.ECS.k1 import K1State
from elpis.runtime import Runtime
from elpis.runtime.cognition import LearnRequest
from elpis.runtime.composition import CompositionError

from ._turn_fixtures import LEARN, ByteTokens, FixtureMap, admitted
from .test_codec_ecs_turn import _config, k1, world  # noqa: F401 (the k1 fixture)
from .test_k1_recovery import _provision

REPO = Path(__file__).resolve().parents[2]

_CHILD = r"""
import json, sys, time
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tests.conftest import admit_k1_set
from tests.integration._turn_fixtures import LEARN, ByteTokens, FixtureMap, admitted
from tests.integration.test_codec_ecs_turn import _config, world
from elpis.ECS.k1 import K1Library
from elpis.runtime import Runtime
from elpis.runtime.cognition import LearnRequest

continuity, checkpoint, mode = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
k1 = K1Library(admit_k1_set()[1].lib)
learn = lambda text: LearnRequest(text, ByteTokens(), admitted(FixtureMap()), LEARN)
say = lambda **kw: (sys.stdout.write(json.dumps(kw) + "\n"), sys.stdout.flush())
state = world(k1, max_rows=8)
runtime = Runtime(replace(_config(continuity, testing=True), k1_checkpoint_dir=checkpoint)).open()
runtime.anchor_cognition(state)
if mode.startswith("park:"):
    step, arg = (int(v) for v in mode[5:].split(","))
    for n in range(3):
        runtime.run_learn(state, learn(f"before {n}"))
    acked = runtime.publish_recoverable(state)
    assert acked.acknowledged and acked.recoverable
    for n in range(2):
        runtime.run_learn(state, learn(f"candidate {n}"))
    candidate = runtime.live_identity(state)[0]
    runtime._core.testing_durable_fault(2, step, arg)      # park the worker at `step`
    pending = runtime.publish_recoverable(state, wait=False)
    runtime.run_learn(state, learn("volatile, never requested"))   # the HOT lane is not blocked by the park
    volatile = runtime.live_identity(state)[0]
    while runtime.durability().parked != step:
        time.sleep(0.005)
    say(parked=step, authorized=acked.identity.hex(), candidate=candidate.hex(), volatile=volatile.hex(),
        pending=pending.status)
    time.sleep(3600)                                          # killed here
elif mode == "delivered":                                     # killed after the acknowledgment reached the caller
    runtime.run_learn(state, learn("before"))
    acked = runtime.publish_recoverable(state)
    runtime.run_learn(state, learn("volatile, never requested"))
    say(acked=acked.identity.hex(), status=acked.status, recoverable=acked.recoverable,
        volatile=runtime.live_identity(state)[0].hex())
    time.sleep(3600)                                          # killed here
else:                                                         # "loop": learn and publish until killed
    n = 0
    while True:
        for _ in range(2):
            runtime.run_learn(state, learn(f"loop {n}"))
            n += 1
        say(requested=runtime.live_identity(state)[0].hex())
        ticket = runtime.publish_recoverable(state)
        say(acked=ticket.identity.hex(), status=ticket.status)
"""


def _child(tmp_path, mode):
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.Popen([sys.executable, "-c", _CHILD, str(REPO), str(tmp_path / "authority"),
                             str(tmp_path / "checkpoint"), mode],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=REPO)


def _kill(child):
    child.send_signal(signal.SIGKILL)
    child.wait(timeout=30)
    assert child.returncode == -signal.SIGKILL


def _restart(tmp_path):
    return Runtime(replace(_config(tmp_path / "authority"), k1_checkpoint_dir=tmp_path / "checkpoint"))


def _resume(k1, runtime, envelope):   # noqa: F811
    """The recovered envelope binds and the lineage continues: one LEARN, one acknowledged publication."""
    with K1State.restore(k1, envelope, max_rows=8) as resumed:
        result = runtime.run_learn(resumed, LearnRequest("after", ByteTokens(), admitted(FixtureMap()), LEARN))
        assert result.durability == "VOLATILE"
        ticket = runtime.publish_recoverable(resumed)
        assert ticket.acknowledged and ticket.recoverable
        assert runtime.continuity.snapshot().k1_state_digest == resumed.state_digest()


# step -> (arg, the authority restart must find, the disposition)
_STEPS = {1: (0, "authorized", "RESUMABLE"), 2: (0, "authorized", "RESUMABLE"), 3: (512, "authorized", "RESUMABLE"),
          4: (0, "authorized", "CANDIDATE_UNRESOLVED"), 5: (0, "authorized", "CANDIDATE_UNRESOLVED"),
          6: (64, "authorized", "CANDIDATE_UNRESOLVED"), 7: (0, "candidate", "RESUMABLE"),
          8: (0, "candidate", "RESUMABLE"), 9: (0, "candidate", "RESUMABLE")}


@pytest.mark.parametrize("step", sorted(_STEPS))
def test_sigkill_at_each_publication_step_restarts_at_exactly_the_authorized_state(k1, tmp_path, step):   # noqa: F811
    arg, expected, disposition = _STEPS[step]
    _provision(k1, tmp_path / "checkpoint")
    child = _child(tmp_path, f"park:{step},{arg}")
    try:
        line = child.stdout.readline()
        assert line, child.stderr.read()
        report = json.loads(line)
    finally:
        _kill(child)
    assert report["parked"] == step and report["pending"] == "PENDING"
    authorized, candidate, volatile = (bytes.fromhex(report[k]) for k in ("authorized", "candidate", "volatile"))
    assert len({authorized, candidate, volatile}) == 3
    with _restart(tmp_path) as runtime:
        authority = runtime.continuity.snapshot().k1_state_digest
        assert authority == {"authorized": authorized, "candidate": candidate}[expected], step
        assert authority != volatile                          # volatile work is never recovered
        recovery = runtime.recover_k1()
        assert recovery.disposition == disposition, (step, recovery)
        if disposition == "RESUMABLE":
            assert recovery.candidate is None and recovery.envelope[-32:] == authority
            _resume(k1, runtime, recovery.envelope)
        else:
            assert recovery.candidate == candidate and recovery.authorized == authorized
            with pytest.raises(CompositionError):             # nothing durable over an unresolved candidate
                runtime.adopt_k1_candidate(authorized)
            runtime.discard_k1_candidate(candidate)           # the operator's explicit decision
            recovery = runtime.recover_k1()
            assert recovery.disposition == "RESUMABLE" and recovery.envelope[-32:] == authorized
            _resume(k1, runtime, recovery.envelope)


def test_sigkill_after_the_acknowledgment_was_delivered_resumes_exactly_the_acknowledged_state(
        k1, tmp_path):   # noqa: F811
    _provision(k1, tmp_path / "checkpoint")
    child = _child(tmp_path, "delivered")
    try:
        line = child.stdout.readline()
        assert line, child.stderr.read()
        report = json.loads(line)
    finally:
        _kill(child)
    assert (report["status"], report["recoverable"]) == ("ACKNOWLEDGED", True)
    acked, volatile = bytes.fromhex(report["acked"]), bytes.fromhex(report["volatile"])
    assert acked != volatile
    with _restart(tmp_path) as runtime:
        assert runtime.continuity.snapshot().k1_state_digest == acked
        recovery = runtime.recover_k1()
        assert recovery.disposition == "RESUMABLE" and recovery.candidate is None
        assert recovery.envelope[-32:] == acked
        _resume(k1, runtime, recovery.envelope)


KILLS = 6


def _events(text):
    """Every complete event line the child flushed before it died (a line cut by the kill is not an event)."""
    out = []
    for line in text.splitlines(keepends=True):
        if line.endswith("\n"):
            out.append(json.loads(line))
    return out


def test_random_sigkill_while_learning_and_publishing_recovers_only_a_complete_authorized_state(
        k1, tmp_path):   # noqa: F811
    rng = random.Random(20261014)
    with world(k1, max_rows=8) as fresh:
        anchor = fresh.state_digest()
    for kill in range(KILLS):
        for d in ("authority", "checkpoint"):
            subprocess.run(["rm", "-rf", str(tmp_path / d)], check=True)
        _provision(k1, tmp_path / "checkpoint")
        child = _child(tmp_path, "loop")
        seen = []
        try:
            for _ in range(rng.randrange(1, 30)):      # let it run for a random number of events
                line = child.stdout.readline()
                assert line, child.stderr.read()
                seen.append(line)
        finally:
            _kill(child)
        events = _events("".join(seen) + child.stdout.read())   # everything flushed before the kill
        acked = [bytes.fromhex(e["acked"]) for e in events if "acked" in e]
        requested = [bytes.fromhex(e["requested"]) for e in events if "requested" in e]
        assert all(e.get("status", "ACKNOWLEDGED") == "ACKNOWLEDGED" for e in events)
        # The last acknowledged state (the anchor before any), or the last requested one whose publication completed
        # before the kill although its acknowledgment never reached the child (a lost acknowledgment is not proof of
        # non-durability). Never a volatile state: none was ever requested.
        allowed = {acked[-1] if acked else anchor} | set(requested[-1:])
        with _restart(tmp_path) as runtime:
            authority = runtime.continuity.snapshot().k1_state_digest
            assert authority in allowed, (kill, len(events), authority.hex())
            recovery = runtime.recover_k1()
            assert recovery.disposition in ("RESUMABLE", "CANDIDATE_UNRESOLVED"), recovery
            if recovery.disposition == "CANDIDATE_UNRESOLVED":
                assert recovery.candidate in set(requested[-1:]) and recovery.candidate != authority
                runtime.discard_k1_candidate(recovery.candidate)
                recovery = runtime.recover_k1()
            assert recovery.disposition == "RESUMABLE" and recovery.envelope[-32:] == authority
            _resume(k1, runtime, recovery.envelope)
