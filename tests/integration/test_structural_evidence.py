"""Admitted typed structural evidence through the canonical runtime (``elpis.runtime.observation.record_evidence``).

validated HACF hyperedge + assertion (structure computes and checks every identity) -> native typed observation
(``libelpis_hacf_observation``) -> ONE observe-only managed turn (RuntimeCore ABI v5: no K1 learning) under an
explicit ObservationAuthority -> atomic volatile commit -> exact slot and LOOKUP -> recoverable publication ->
restart -> restore. Real native code throughout; no mock storage.

* two sources asserting and negating one hyperedge: one relation, both polarities and provenances kept; LOOKUP
  answers by the memory's net-evidence law (asserts - negates > 0), so it withholds an answer while they balance and
  never decides which source is right; K1 byte-identical;
* duplicate evidence is recorded as repeated observations (counted), never merged or dropped;
* invalid provenance, another hyperedge's assertion, unsupported (9) and query-local participants, an undefined
  polarity, an unadmitted source: refused before the runtime is touched (no native turn begun);
* over-capacity evidence aborts its whole turn (the memory's slot capacity); nothing of it is installed;
* the evidence survives recoverable publication, process death (SIGKILL) and restore exactly, and the inverse map
  rebuilds the identical hyperedge;
* simultaneous requests from several threads: RuntimeCore admits one turn at a time (``RUNTIME_TURN_OPEN`` for the
  others, nothing partial); every item is recorded exactly once;
* the resource footprint is fixed: the envelope and the allocated bytes never grow with the evidence.

Mechanics only: fixture records; evidence is storage, never truth, learning or authority.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

import pytest

from elpis.ECS.composite import CompositeState
from elpis.runtime import Runtime
from elpis.runtime.cognition import ObservationAuthority
from elpis.runtime.composition import CompositionError
from elpis.runtime.observation import (INCIDENCE_NAMESPACE, HacfParticipant, StructuralEvidence, _Assertion,
                                       record_evidence)

from .test_composite_runtime import CONFIG, _store, fresh, library  # noqa: F401 (fixture)
from .test_hacf_composite_observation import RELATION, SOURCES, _edge, _tampered, hacf, node, provenance  # noqa: F401
from .test_k1_recovery import _cfg
from .test_observe_only import _k1_bytes

REPO = Path(__file__).resolve().parents[2]


def _refused(call, code):
    with pytest.raises(CompositionError) as info:
        call()
    assert info.value.code == code, (info.value.code, str(info.value))
    return info.value


def _evidence(hacf, nodes, source, tick, polarity="ASSERT", prov=None):   # noqa: F811
    edge = _edge(hacf, nodes=nodes)
    return StructuralEvidence(edge, hacf.assertion(edge, provenance(prov if prov is not None else source), source),
                              polarity, tick)


def test_two_sources_asserting_and_negating_one_hyperedge_keep_both_and_lookup_reports_the_conflict(
        hacf, library, tmp_path):   # noqa: F811
    store = _store(library, tmp_path / "ck")
    with fresh(library) as x, Runtime(_cfg(tmp_path, store)) as runtime:
        runtime.anchor_cognition(x)
        before = x.snapshot()
        live = runtime.live_identity(x)
        yes, no = _evidence(hacf, (0, 1, 2), 1, 30), _evidence(hacf, (0, 1, 2), 2, 31, "NEGATE")
        receipt = record_evidence(runtime, x, hacf, (yes, no), SOURCES)
        assert (receipt.durability, receipt.k1_steps, receipt.generation_before, receipt.generation_after) == \
            ("VOLATILE", 0, live[1], live[1] + 1)
        first, second = receipt.entries
        assert first.relation_key == second.relation_key                  # one relation, two observations
        assert (first.hyperedge_identity, first.assertion_identity, first.provenance_digest, first.source_authority,
                first.polarity, first.tick) == (yes.hyperedge.identity, yes.assertion.identity, provenance(1), 1,
                                                "ASSERT", 30)
        assert (second.source_authority, second.polarity, second.provenance_digest) == (2, "NEGATE", provenance(2))
        slot = x.slot(first.relation_key)
        assert (slot.assert_count, slot.negate_count, slot.last_polarity, slot.last_authority) == (1, 1, "NEGATE", 2)
        assert (slot.first_provenance, slot.last_provenance, slot.structure_ref) == \
            (provenance(1), provenance(2), yes.hyperedge.identity)
        assert hacf.hyperedge_of(slot) == yes.hyperedge                   # the exact structure, recomputed
        cue = tuple(slot.participants[:2])
        answer = x.lookup(RELATION, slot.participants[2][0], cue)
        assert (answer.found, answer.answer, answer.matches) == (False, None, 1)   # balanced: no answer withheld
        # A third, asserting observation makes the net evidence positive: LOOKUP answers the exact filler.
        record_evidence(runtime, x, hacf, (_evidence(hacf, (0, 1, 2), 1, 32),), SOURCES)
        answer = x.lookup(RELATION, slot.participants[2][0], cue)
        assert (answer.found, answer.answer) == (True, slot.participants[2][2])
        assert _k1_bytes(x.snapshot()) == _k1_bytes(before)               # no K1 learning


def test_duplicate_evidence_is_recorded_as_repeated_observations_never_merged_or_dropped(
        hacf, library, tmp_path):   # noqa: F811
    store = _store(library, tmp_path / "ck")
    with fresh(library) as x, Runtime(_cfg(tmp_path, store)) as runtime:
        runtime.anchor_cognition(x)
        same = _evidence(hacf, (0, 1, 2), 1, 40)
        key = record_evidence(runtime, x, hacf, (same,), SOURCES).entries[0].relation_key
        again = record_evidence(runtime, x, hacf, (same, same), SOURCES)
        assert [e.relation_key for e in again.entries] == [key, key]
        assert x.slot(key).assert_count == 3 and x.inspect()["memory_slots_used"] == 1


def test_each_receipt_entry_names_the_slot_of_exactly_its_own_evidence(hacf, library, tmp_path):   # noqa: F811
    store = _store(library, tmp_path / "ck")
    with fresh(library) as x, Runtime(_cfg(tmp_path, store)) as runtime:
        runtime.anchor_cognition(x)
        batch = tuple(_evidence(hacf, (30 + i, 80 + i, 150 + i), 1 + i % 2, 45, prov=0x40 + i) for i in range(3))
        receipt = record_evidence(runtime, x, hacf, batch, SOURCES)
        assert len({e.relation_key for e in receipt.entries}) == 3
        for entry, item in zip(receipt.entries, batch):
            slot = x.slot(entry.relation_key)
            assert (slot.structure_ref, slot.first_provenance, slot.last_authority) == \
                (item.hyperedge.identity, item.assertion.provenance_digest, item.assertion.authority)
            assert (entry.hyperedge_identity, entry.provenance_digest) == (item.hyperedge.identity,
                                                                          item.assertion.provenance_digest)


def test_invalid_or_unsupported_evidence_is_refused_before_the_runtime_is_touched(hacf, library, tmp_path):   # noqa: F811
    store = _store(library, tmp_path / "ck")
    with fresh(library) as x, Runtime(_cfg(tmp_path, store)) as runtime:
        runtime.anchor_cognition(x)
        state = x.state_digest()
        runtime._core.counters(reset=True)
        good = _evidence(hacf, (0, 1, 2), 1, 50)
        other = _evidence(hacf, (3, 4, 5), 1, 50)
        forged = replace_assertion(good, _tampered(good.assertion, _Assertion, provenance_digest=provenance(0x77)))
        cases = [
            ((forged,), "OBSERVATION_IDENTITY"),                                      # provenance changed
            ((StructuralEvidence(good.hyperedge, other.assertion, "ASSERT", 50),), "OBSERVATION_IDENTITY"),
            ((StructuralEvidence(good.hyperedge, good.assertion, "TRUE", 50),), "OBSERVATION_INVALID"),
            ((good, StructuralEvidence(good.hyperedge, good.assertion, "ASSERT", -1)), "OBSERVATION_INVALID"),
            ((_evidence(hacf, (0, 1, 2), 3, 50),), "OBSERVATION_AUTHORITY"),         # a source not admitted
            ((), "OBSERVATION"),
        ]
        for evidence, code in cases:
            with pytest.raises(CompositionError) as info:
                record_evidence(runtime, x, hacf, evidence, SOURCES)
            assert info.value.code.startswith(code), (code, info.value.code, str(info.value))
        nine = tuple(HacfParticipant(node(100 + i), INCIDENCE_NAMESPACE | (i + 1)) for i in range(9))
        wide = hacf.hyperedge(RELATION, nine)
        _refused(lambda: record_evidence(runtime, x, hacf, (StructuralEvidence(
            wide, hacf.assertion(wide, provenance(1), 1), "ASSERT", 50),), SOURCES), "OBSERVATION_UNSUPPORTED")
        local = hacf.hyperedge(RELATION, (HacfParticipant(node(1), INCIDENCE_NAMESPACE | 1, query_local=True),
                                          HacfParticipant(node(2), INCIDENCE_NAMESPACE | 2)))
        _refused(lambda: record_evidence(runtime, x, hacf, (StructuralEvidence(
            local, hacf.assertion(local, provenance(1), 1), "ASSERT", 50),), SOURCES), "OBSERVATION_UNSUPPORTED")
        counted = runtime._core.counters()
        assert counted["k1_txn_begins"] == 0 and counted["k1_observes"] == 0   # nothing native was begun
        assert x.state_digest() == state and runtime.fault is None


def replace_assertion(evidence, assertion):
    return StructuralEvidence(evidence.hyperedge, assertion, evidence.polarity, evidence.tick)


def test_over_capacity_evidence_aborts_its_whole_turn_and_the_footprint_is_fixed(hacf, library, tmp_path):   # noqa: F811
    store = _store(library, tmp_path / "ck")
    with fresh(library) as x, Runtime(_cfg(tmp_path, store)) as runtime:
        runtime.anchor_cognition(x)
        footprint = x.inspect()
        capacity = CONFIG.slot_capacity
        for i in range(capacity - 1):
            record_evidence(runtime, x, hacf, (_evidence(hacf, (10 + i, 60 + i, 120 + i), 1, 60 + i),), SOURCES)
        state, live = x.state_digest(), runtime.live_identity(x)
        batch = (_evidence(hacf, (200, 201, 202), 1, 100), _evidence(hacf, (210, 211, 212), 1, 101))
        info = _refused(lambda: record_evidence(runtime, x, hacf, batch, SOURCES), "ECS_REFUSED")
        assert "CAPACITY" in str(info)
        assert (x.state_digest(), runtime.live_identity(x)) == (state, live)   # nothing of the batch installed
        used = x.inspect()
        assert used["memory_slots_used"] == capacity - 1 and runtime.fault is None
        # Fixed resources: the envelope and the allocation never grow with the evidence.
        for name in ("envelope_bytes", "allocated_bytes", "memory_capacity_bytes", "k1_envelope_bytes"):
            assert used[name] == footprint[name], name
        assert used["memory_state_bytes"] <= used["memory_capacity_bytes"]
        # One more relation still fits; the next is refused.
        record_evidence(runtime, x, hacf, (batch[0],), SOURCES)
        _refused(lambda: record_evidence(runtime, x, hacf, (batch[1],), SOURCES), "ECS_REFUSED")


def test_simultaneous_requests_record_every_item_exactly_once(hacf, library, tmp_path):   # noqa: F811
    store = _store(library, tmp_path / "ck")
    with fresh(library) as x, Runtime(_cfg(tmp_path, store)) as runtime:
        runtime.anchor_cognition(x)
        items = [_evidence(hacf, (20 + i, 70 + i, 140 + i), 1 + i % 2, 70) for i in range(8)]
        receipts, refusals, errors = [None] * len(items), [0] * len(items), []

        def worker(i):
            while True:
                try:
                    receipts[i] = record_evidence(runtime, x, hacf, (items[i],), SOURCES)
                    return
                except CompositionError as e:
                    if e.code != "RUNTIME_TURN_OPEN":
                        errors.append(e)
                        return
                    refusals[i] += 1     # another request's turn is open: nothing of this one was begun

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(items))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert not errors and all(r is not None for r in receipts)
        generations = sorted(r.generation_after for r in receipts)
        assert generations == list(range(generations[0], generations[0] + len(items)))   # one commit each
        assert all(x.slot(r.entries[0].relation_key).assert_count == 1 for r in receipts)
        assert x.inspect()["memory_slots_used"] == len(items) and runtime.fault is None


_CHILD = r"""
import json, sys, time
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tests.integration.test_codec_ecs_turn import _config
from tests.integration.test_composite_runtime import fresh
from tests.integration.test_hacf_composite_observation import SOURCES, _edge, provenance
from tests.conftest import admit_composite_set, admit_native, native_build_dir
from elpis.ECS.composite import CompositeLibrary
from elpis.runtime import Runtime
from elpis.runtime.observation import HacfObservationLibrary, StructuralEvidence, record_evidence

hacf = HacfObservationLibrary(admit_native(native_build_dir() / "native" / "runtime" / "libelpis_hacf_observation.so"))
x = fresh(CompositeLibrary(admit_composite_set()[3].lib))
runtime = Runtime(replace(_config(Path(sys.argv[2])), k1_checkpoint_dir=Path(sys.argv[3]))).open()
runtime.anchor_cognition(x)
edge = _edge(hacf)
ev = lambda src, pol, tick: StructuralEvidence(edge, hacf.assertion(edge, provenance(src), src), pol, tick)
first = record_evidence(runtime, x, hacf, (ev(1, "ASSERT", 10), ev(2, "NEGATE", 11)), SOURCES)
acked = runtime.publish_recoverable(x)
assert acked.acknowledged
lost = record_evidence(runtime, x, hacf, (ev(1, "ASSERT", 12),), SOURCES)          # volatile: never published
print(json.dumps({"authorized": acked.identity.hex(), "key": first.entries[0].relation_key.hex(),
                  "edge": edge.identity.hex(), "volatile": x.state_digest().hex()}), flush=True)
time.sleep(3600)
"""


def test_evidence_survives_process_death_and_restore_exactly(hacf, library, tmp_path):   # noqa: F811
    _store(library, tmp_path / "checkpoint")
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1")
    child = subprocess.Popen([sys.executable, "-c", _CHILD, str(REPO), str(tmp_path / "authority"),
                              str(tmp_path / "checkpoint")], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, env=env, cwd=REPO)
    try:
        line = child.stdout.readline()
        assert line, child.stderr.read()
        report = json.loads(line)
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait(timeout=30)
    assert child.returncode == -signal.SIGKILL
    authorized, key = bytes.fromhex(report["authorized"]), bytes.fromhex(report["key"])
    from elpis.runtime.cognition import ObservationAuthority  # noqa: F401
    from .test_codec_ecs_turn import _config
    from dataclasses import replace
    with Runtime(replace(_config(tmp_path / "authority"), k1_checkpoint_dir=tmp_path / "checkpoint")) as runtime:
        recovery = runtime.recover_k1()
        assert recovery.disposition == "RESUMABLE" and recovery.authorized == authorized
        with CompositeState.restore(library, CONFIG, recovery.envelope) as y:
            assert y.state_digest() == authorized != bytes.fromhex(report["volatile"])
            slot = y.slot(key)
            assert (slot.assert_count, slot.negate_count) == (1, 1)              # the lost third never recovered
            assert (slot.first_provenance, slot.last_provenance) == (provenance(1), provenance(2))
            assert hacf.hyperedge_of(slot).identity == bytes.fromhex(report["edge"])
            again = record_evidence(runtime, y, hacf, (StructuralEvidence(
                hacf.hyperedge_of(slot), hacf.assertion(hacf.hyperedge_of(slot), provenance(1), 1), "ASSERT", 12),),
                ObservationAuthority("test: restart", frozenset({1})))
            assert again.entries[0].relation_key == key and y.slot(key).assert_count == 2
