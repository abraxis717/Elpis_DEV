"""The observe-only managed turn (RuntimeCore ABI v5; docs/RUNTIME_CORE.md, "Observe-only turns").

Typed evidence enters the memory of an ECS composite lineage through ``Runtime.run_observe`` under an explicit
``ObservationAuthority``, with NO K1 learning: RuntimeCore verifies and binds the lineage, opens the composite
transaction at the live generation, runs no experience schedule, records the observations on the memory candidate
and commits one VOLATILE transition. Over real native libraries these tests check:

* the K1 component's envelope is byte-identical before and after (its recorded identity too), the memory and the
  composite identity moved, no K1 schedule crossed into native code, QUERY answers exactly as before, and LOOKUP
  answers the recorded relation exactly;
* volatile: the session's anchor did not move (RuntimeCore ABI v8 publishes nothing);
* all or nothing: an unauthorized, unadmitted, over-bound (in Python and natively) or memory-refused observation
  installs nothing (composite identity, K1 bytes, live generation and continuity unchanged) and the runtime stays
  healthy;
* real process death: a child records two observations and is killed (SIGKILL); it wrote nothing, and a new session
  holds neither (nothing is ever recovered); a refused observation in the new session moves nothing.

Mechanics only: the observations are TEST_ONLY fixture records; evidence is storage, never learning or truth.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from elpis.ECS.composite import CompositeState
from elpis.runtime import Runtime
from elpis.runtime.cognition import ObservationAuthority
from elpis.runtime.composition import CompositionError
from elpis.runtime.fuel import CEILING

from .test_codec_ecs_turn import _config, k1, world  # noqa: F401 (module fixtures)
from .test_composite_runtime import (AUTHORITY, CONFIG, _query, _refused, ent, fresh, library,  # noqa: F401
                                     obs)

REPO = Path(__file__).resolve().parents[2]


def _k1_bytes(envelope: bytes) -> bytes:
    """The ELPISGK1 envelope inside an ELPISEC1 one (header offset 16: its size; it starts at byte 128)."""
    return envelope[128:128 + int.from_bytes(envelope[16:24], "little")]


def test_an_observe_only_turn_records_evidence_and_leaves_k1_byte_identical(library, tmp_path):
    with fresh(library) as x, Runtime(_config()) as runtime:
        anchored = runtime.anchor_cognition(x)
        before = x.snapshot()
        answer_before = runtime.run_query(x, _query())
        live = runtime.live_identity(x)
        runtime._core.counters(reset=True)
        result = runtime.run_observe(x, (obs(1, 2, 10), obs(3, 4, 11)), AUTHORITY)
        counted = runtime._core.counters()
        assert (result.durability, result.k1_steps, result.generation_before, result.generation_after) == \
            ("VOLATILE", 0, live[1], live[1] + 1)
        assert result.grant == AUTHORITY.grant and len(result.observed) == 2 and len(set(result.observed)) == 2
        # No K1 schedule crossed into native code: begin, two observations, commit.
        assert {k: counted[k] for k in ("k1_txn_begins", "k1_run_schedules", "k1_observes", "k1_commits",
                                        "session_transitions")} == \
            {"k1_txn_begins": 1, "k1_run_schedules": 0, "k1_observes": 2, "k1_commits": 1, "session_transitions": 0}
        after = x.snapshot()
        assert _k1_bytes(after) == _k1_bytes(before)                    # K1 byte-identical
        c0, k0, m0 = library.envelope_identities(before)
        c1, k1_after, m1 = library.envelope_identities(after)
        assert k1_after == k0 and m1 != m0 and c1 != c0 == anchored.k1_state_digest
        # QUERY answers exactly as before (K1 did not move); it reads the live generation.
        answer_after = runtime.run_query(x, _query())
        assert answer_after.readout.values == answer_before.readout.values
        assert answer_after.live_generation == live[1] + 1
        # LOOKUP answers the recorded relation exactly; the slot keeps provenance, authority and participants.
        found = x.lookup(3, 2, ((1, 0, ent(1)),))
        assert (found.found, found.answer, found.state_digest) == (True, ent(2), x.state_digest())
        slot = x.slot(result.observed[0])
        assert (slot.assert_count, slot.negate_count, slot.last_authority) == (1, 0, 1)
        assert slot.participants == obs(1, 2, 10).participants and slot.first_provenance == obs(1, 2, 10).provenance
        # Volatile: the session's anchor did not move, and nothing can publish the new state.
        assert runtime.continuity.snapshot() == anchored and not hasattr(runtime, "publish_recoverable")


def test_refused_observe_only_turns_install_nothing_and_leave_the_runtime_healthy(library, tmp_path):
    with fresh(library) as x, Runtime(_config()) as runtime:
        anchored = runtime.anchor_cognition(x)
        runtime.run_observe(x, (obs(1, 2, 10),), AUTHORITY)
        state, envelope, live = x.state_digest(), x.snapshot(), runtime.live_identity(x)

        def unchanged():
            assert (x.state_digest(), x.snapshot(), runtime.live_identity(x)) == (state, envelope, live)
            assert runtime.continuity.snapshot() == anchored and runtime.fault is None

        # Refused before anything native is called.
        _refused(lambda: runtime.run_observe(x, (obs(3, 4, 11),), None), "OBSERVATION_UNAUTHORIZED")
        _refused(lambda: runtime.run_observe(x, (obs(3, 4, 11, source=9),), AUTHORITY), "OBSERVATION_AUTHORITY")
        _refused(lambda: runtime.run_observe(x, (), AUTHORITY), "OBSERVATION")
        _refused(lambda: runtime.run_observe(x, [obs(3, 4, 11)], AUTHORITY), "OBSERVATION")
        one = ObservationAuthority("test: one observation per turn", frozenset({1}), max_observations=1)
        _refused(lambda: runtime.run_observe(x, (obs(3, 4, 11), obs(5, 6, 12)), one), "OBSERVATION_AUTHORITY")
        _refused(lambda: runtime.run_observe(x, (obs(3, 4, 11),), AUTHORITY, budget={"max_observations": 1}),
                 "OBSERVATION")
        unchanged()
        # The native bound: the budget's max_observations is enforced by RuntimeCore; the turn is aborted whole.
        tight = replace(CEILING, max_observations=1)
        _refused(lambda: runtime.run_observe(x, (obs(3, 4, 11), obs(5, 6, 12)), AUTHORITY, budget=tight),
                 "COGNITION_FUEL_EXCEEDED")
        unchanged()
        assert x.slot(hashlib.sha256(b"never").digest()) is None
        # A memory refusal (an observation older than the memory's clock) after a valid one: all or nothing.
        info = _refused(lambda: runtime.run_observe(x, (obs(3, 4, 12), obs(5, 6, 1)), AUTHORITY), "ECS_REFUSED")
        assert "CLOCK" in str(info)
        unchanged()
        assert x.lookup(3, 2, ((1, 0, ent(3)),)).found is False
        # A K1 lineage has no memory (refused before any native call).
        with pytest.raises(CompositionError) as other:
            runtime.run_observe(object(), (obs(3, 4, 13),), AUTHORITY)
        assert other.value.code == "OBSERVATION"
        unchanged()
        # Still healthy: the next authorized observation commits.
        later = runtime.run_observe(x, (obs(3, 4, 13),), AUTHORITY)
        assert later.generation_after == live[1] + 1 and x.slot(later.observed[0]).assert_count == 1


def test_a_k1_lineage_refuses_the_observe_only_turn_natively(k1, tmp_path):   # noqa: F811
    from elpis.runtime.core import describe
    with world(k1) as state, Runtime(_config()) as runtime:
        runtime.anchor_cognition(state)
        live = runtime.live_identity(state)
        descriptor, _ = describe(state)
        with pytest.raises(CompositionError) as info:
            runtime._core.turn_begin_observe(descriptor, CEILING)
        assert info.value.code == "RUNTIME_INVALID"
        assert runtime.live_identity(state) == live and runtime.fault is None


_CHILD = r"""
import json, sys, time
sys.path.insert(0, sys.argv[1])
from tests.integration.test_codec_ecs_turn import _config
from tests.integration.test_composite_runtime import AUTHORITY, fresh, obs
from tests.conftest import admit_composite_set
from elpis.ECS.composite import CompositeLibrary
from elpis.runtime import Runtime

library = CompositeLibrary(admit_composite_set()[3].lib)
x = fresh(library)
initial = x.snapshot()
runtime = Runtime(_config()).open()
runtime.anchor_cognition(x)
first = runtime.run_observe(x, (obs(1, 2, 10),), AUTHORITY).observed[0]
second = runtime.run_observe(x, (obs(3, 4, 11),), AUTHORITY).observed[0]
print(json.dumps({"live": x.state_digest().hex(), "keys": [first.hex(), second.hex()],
                  "initial": initial.hex()}), flush=True)
time.sleep(3600)                                                            # killed here
"""


def test_observations_are_volatile_and_a_killed_process_leaves_none_of_them(library, tmp_path):
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1")
    child = subprocess.Popen([sys.executable, "-c", _CHILD, str(REPO)], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=env, cwd=tmp_path)
    try:
        line = child.stdout.readline()
        assert line, child.stderr.read()
        report = json.loads(line)
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait(timeout=30)
    assert child.returncode == -signal.SIGKILL
    assert list(tmp_path.iterdir()) == []                                   # the killed process wrote nothing
    first, second = (bytes.fromhex(k) for k in report["keys"])
    assert first != second
    with Runtime(_config()) as runtime, fresh(library) as y:
        assert not runtime.continuity.snapshot().anchored                   # nothing is ever recovered
        runtime.anchor_cognition(y)
        assert y.state_digest() != bytes.fromhex(report["live"])
        assert y.slot(first) is None and y.slot(second) is None
        assert _k1_bytes(y.snapshot()) == _k1_bytes(bytes.fromhex(report["initial"]))
        # A refused observation moves nothing; the session continues.
        before, session = y.state_digest(), runtime.continuity.snapshot()
        _refused(lambda: runtime.run_observe(y, (obs(3, 4, 11), obs(7, 8, 2)), AUTHORITY), "ECS_REFUSED")
        assert y.state_digest() == before and runtime.continuity.snapshot() == session
        again = runtime.run_observe(y, (obs(3, 4, 11),), AUTHORITY)
        assert again.observed == (second,) and y.slot(second).assert_count == 1


def test_the_native_observation_bound_is_the_authority_bound():
    """RuntimeCore itself enforces what the ObservationAuthority admitted: the turn's native ``max_observations`` is
    the smaller of the budget's and the authority's (a LEARN without observations keeps its budget unchanged)."""
    from elpis.runtime.composition import _observation_budget
    three = ObservationAuthority("test: three per turn", frozenset({1}), max_observations=3)
    assert _observation_budget(CEILING, three).max_observations == 3
    assert _observation_budget(replace(CEILING, max_observations=2), three).max_observations == 2
    assert _observation_budget(CEILING, None) is CEILING
