"""ECS composite lineages through the Python runtime composition (docs/ECS_COMPOSITE_STATE.md, "Authority").

The one ECS retained state is the qualified K1 state and the ECS memory under one ELPISEC1 identity. RuntimeCore
binds it as substrate kind 3; these tests drive it through the public Python runtime with real native libraries:
the composite's lineage is anchored and bound under RuntimeCore's lease, a managed LEARN carries explicitly
authorized typed observations into the same turn (all or nothing), the commit is volatile, an explicit recoverable
publication makes the complete composite durable, restart restores exactly the acknowledged composite, a K1 identity
never stands for a composite one (or the reverse), and a K1 lineage becomes a composite one only by the explicit
legacy import. Only the admitted memory operations are reachable (OBSERVE, LOOKUP); memory LEARN is not.

Mechanics only: the codec is the TEST_ONLY TRAINING=NONE SEMANTICS=NONE fixture map and the observations are fixture
records; an observation is evidence storage, never truth, a learning instruction or authority. Process death inside
a composite durable publication (slot synced; record torn) and its recovery are native:
``runtime.test_runtime_composite``; composite format corruption and unadmitted learned content on restore:
``ECS.test_ecs_composite``.
"""
from __future__ import annotations

import ctypes
import hashlib

import numpy as np
import pytest

from elpis.ECS.composite import CompositeConfig, CompositeError, CompositeLibrary, CompositeState, MemoryObservation
from elpis.continuity import ContinuityError, ContinuityStore
from elpis.ECS.k1 import K1State
from elpis.runtime import Runtime
from elpis.runtime.cognition import LearnRequest, ObservationAuthority, QueryRequest
from elpis.runtime.composition import CompositionError
from elpis.runtime.recovery import provision_k1_checkpoint

from ..conftest import admit_composite_set
from ._turn_fixtures import LEARN, ByteTokens, FixtureMap, admitted
from .test_codec_ecs_turn import DIM, WIDTH, _config, initial_w, k1  # noqa: F401 (module fixtures)
from ..conftest import native_build_dir
from .test_k1_recovery import _cfg, _library

HEADER = 128
CONFIG = CompositeConfig(DIM, WIDTH, max_rows=8)
AUTHORITY = ObservationAuthority("test: fixture evidence records (TEST_ONLY TRAINING=NONE SEMANTICS=NONE)",
                                 frozenset({1, 2}))


@pytest.fixture(scope="module")
def library():
    return CompositeLibrary(admit_composite_set()[3].lib)


def _w(seed=36):
    return memoryview(np.ascontiguousarray(initial_w(seed)).reshape(-1))


def fresh(library, config=CONFIG, seed=36):
    return CompositeState.create(library, config, _w(seed))


def _store(library, path, config=CONFIG):
    provision_k1_checkpoint(_library(), path, library.envelope_bytes(config))
    return path


def ent(n):
    return hashlib.sha256(b"fixture-entity" + bytes([n])).digest()


def obs(subject, obj, tick, *, polarity="ASSERT", source=1, relation=3):
    return MemoryObservation(relation, polarity, ((1, 0, ent(subject)), (2, 0, ent(obj))), tick, source,
                             hashlib.sha256(b"fixture-provenance" + tick.to_bytes(8, "little")).digest())


def _learn(text="learn"):
    return LearnRequest(text, ByteTokens(), admitted(FixtureMap()), LEARN)


def _query():
    return QueryRequest("query", ByteTokens(), admitted(FixtureMap()))


def _refused(call, code):
    with pytest.raises(CompositionError) as info:
        call()
    assert info.value.code == code, (info.value.code, str(info.value))
    return info.value


def _footprint(path):
    return sorted((p.name, p.stat().st_size, p.stat().st_ino) for p in path.iterdir())


def test_a_composite_lineage_learns_and_observes_in_one_volatile_turn_under_runtime_ownership(library, tmp_path):
    store = _store(library, tmp_path / "ck")
    with fresh(library) as x, Runtime(_cfg(tmp_path, store)) as runtime:
        anchored = runtime.anchor_cognition(x)
        assert (anchored.state_format, anchored.k1_state_digest) == ("COMPOSITE", x.state_digest())
        recovery = runtime.recover_k1()
        assert (recovery.disposition, recovery.envelope) == ("RESUMABLE", x.snapshot())   # durable first
        assert x.managed_lease != 0 and x.inspect()["capabilities"] == 0   # admitted memory operations only
        with pytest.raises(CompositeError) as leased:
            x.transaction()                                   # unmanaged mutation of a bound composite
        assert leased.value.code == "LEASED"

        live = runtime.live_identity(x)
        result = runtime.run_learn(x, _learn(), observations=(obs(1, 2, 10), obs(3, 4, 11)),
                                   observation_authority=AUTHORITY)
        assert (result.durability, result.generation_before, result.generation_after) == \
            ("VOLATILE", live[1], live[1] + 1)
        assert len(result.observed) == 2 and len(set(result.observed)) == 2
        # Both the K1 candidate and the memory candidate were installed together; the observation is exactly the
        # recorded evidence (provenance, authority, polarity, participants), and LOOKUP answers it exactly.
        slot = x.slot(result.observed[0])
        assert (slot.assert_count, slot.negate_count, slot.last_authority, slot.relation_type) == (1, 0, 1, 3)
        assert slot.participants == obs(1, 2, 10).participants and slot.first_provenance == obs(1, 2, 10).provenance
        answer = x.lookup(3, 2, ((1, 0, ent(1)),))
        assert (answer.found, answer.answer, answer.state_digest) == (True, ent(2), x.state_digest())
        assert runtime.live_identity(x) == (x.state_digest(), live[1] + 1) != live
        # Volatile: the durable authority and the checkpoint slots did not move.
        assert runtime.continuity.snapshot() == anchored
        assert runtime.recover_k1().envelope != x.snapshot()
        # QUERY answers from the live composite, at its live generation; it changes nothing.
        before = x.snapshot()
        assert runtime.run_query(x, _query()).readout.live_generation == live[1] + 1
        assert x.snapshot() == before
        assert runtime.fault is None
        runtime.release(x)
        assert x.managed_lease == 0
        with x.transaction() as t:                            # released: its owner may use it unmanaged again
            t.abort()
    assert [n for n, _, _ in _footprint(store)] == ["k1-checkpoint.a", "k1-checkpoint.b"]   # still its two slots


def test_an_unauthorized_or_refused_observation_installs_nothing(library, k1, tmp_path):   # noqa: F811
    small = CompositeConfig(DIM, WIDTH, max_rows=8, slot_capacity=2)   # REFUSE when full
    with fresh(library, small) as x, Runtime(_cfg(tmp_path, _store(library, tmp_path / "ck", small))) as runtime:
        runtime.anchor_cognition(x)
        runtime.run_learn(x, _learn(), observations=(obs(1, 2, 10),), observation_authority=AUTHORITY)
        state, live = x.snapshot(), runtime.live_identity(x)

        def unchanged():
            assert (x.snapshot(), runtime.live_identity(x), runtime.fault) == (state, live, None)

        # Refused at the boundary, before anything native: no authority, a source it does not admit, more than it
        # admits, an authority authorizing nothing, a value that is not a typed observation.
        _refused(lambda: runtime.run_learn(x, _learn(), observations=(obs(5, 6, 20),)), "OBSERVATION_UNAUTHORIZED")
        _refused(lambda: runtime.run_learn(x, _learn(), observations=(obs(5, 6, 20, source=9),),
                                           observation_authority=AUTHORITY), "OBSERVATION_AUTHORITY")
        narrow = ObservationAuthority("test: one per turn", frozenset({1}), max_observations=1)
        _refused(lambda: runtime.run_learn(x, _learn(), observations=(obs(5, 6, 20), obs(7, 8, 21)),
                                           observation_authority=narrow), "OBSERVATION_AUTHORITY")
        _refused(lambda: runtime.run_learn(x, _learn(), observation_authority=AUTHORITY), "OBSERVATION_AUTHORITY")
        _refused(lambda: runtime.run_learn(x, _learn(), observations=[obs(5, 6, 20)],
                                           observation_authority=AUTHORITY), "OBSERVATION")
        unchanged()
        # Refused natively inside the turn: a stale tick after a valid observation; the memory's capacity. The whole
        # turn is aborted: neither its K1 schedule nor its first, valid observation is installed.
        stale = _refused(lambda: runtime.run_learn(x, _learn(), observations=(obs(5, 6, 20), obs(7, 8, 1)),
                                                   observation_authority=AUTHORITY), "ECS_REFUSED")
        assert "memory CLOCK" in str(stale)
        unchanged()
        full = _refused(lambda: runtime.run_learn(x, _learn(), observations=(obs(5, 6, 20), obs(7, 8, 21)),
                                                  observation_authority=AUTHORITY), "ECS_REFUSED")
        assert "memory CAPACITY" in str(full)
        unchanged()
        malformed = MemoryObservation(3, "ASSERT", ((1, 0, ent(5)), (1, 0, ent(6))), 30, 1, ent(0))   # same role+ordinal
        _refused(lambda: runtime.run_learn(x, _learn(), observations=(malformed,), observation_authority=AUTHORITY),
                 "ECS_REFUSED")
        unchanged()
        # The runtime stays healthy: the next authorized turn commits exactly once.
        ok = runtime.run_learn(x, _learn(), observations=(obs(5, 6, 20),), observation_authority=AUTHORITY)
        assert ok.generation_after == live[1] + 1 and x.slot(ok.observed[0]).assert_count == 1
    # Observations need a composite: a K1 lineage has no memory to observe into.
    with K1State.create(k1, DIM, WIDTH, _w(), max_rows=8) as plain, \
            Runtime(_config(tmp_path / "k1-lineage")) as runtime:
        runtime.anchor_cognition(plain)
        _refused(lambda: runtime.run_learn(plain, _learn(), observations=(obs(1, 2, 10),),
                                           observation_authority=AUTHORITY), "OBSERVATION")


def test_a_published_composite_is_restored_exactly_and_volatile_work_is_lost_on_restart(library, tmp_path):
    store = _store(library, tmp_path / "ck")
    envelope_bytes = library.envelope_bytes(CONFIG)
    with fresh(library) as x, Runtime(_cfg(tmp_path, store)) as runtime:
        runtime.anchor_cognition(x)
        published = runtime.run_learn(x, _learn(), observations=(obs(1, 2, 10),), observation_authority=AUTHORITY)
        ticket = runtime.publish_recoverable(x)
        assert (ticket.status, ticket.recoverable, ticket.kind, ticket.identity) == \
            ("ACKNOWLEDGED", True, "RECOVERABLE", x.state_digest())
        durable = runtime.continuity.snapshot()
        assert (durable.state_format, durable.k1_state_digest) == ("COMPOSITE", x.state_digest())
        acknowledged = x.snapshot()
        allocated = x.inspect()["allocated_bytes"]
        lost = runtime.run_learn(x, _learn(), observations=(obs(3, 4, 11),), observation_authority=AUTHORITY)
        for n in range(20):   # bounded resources: no allocation, the store never grows
            runtime.run_learn(x, _learn(), observations=(obs(1, 2, 12 + n),), observation_authority=AUTHORITY)
        assert x.inspect()["allocated_bytes"] == allocated
        assert runtime.continuity.snapshot() == durable        # volatile work since the acknowledgment
    footprint = _footprint(store)
    assert [(n, s) for n, s, _ in footprint] == [("k1-checkpoint.a", HEADER + envelope_bytes),
                                                 ("k1-checkpoint.b", HEADER + envelope_bytes)]

    with Runtime(_cfg(tmp_path, store)) as runtime:
        recovery = runtime.recover_k1()
        assert (recovery.disposition, recovery.envelope) == ("RESUMABLE", acknowledged)
        with CompositeState.restore(library, CONFIG, recovery.envelope) as y:
            assert y.state_digest() == durable.k1_state_digest and y.snapshot() == acknowledged
            assert y.slot(published.observed[0]).assert_count == 1     # the published observation, exactly once
            assert y.slot(lost.observed[0]) is None                     # the volatile one is gone
            again = runtime.run_learn(y, _learn(), observations=(obs(3, 4, 11),), observation_authority=AUTHORITY)
            assert again.observed == lost.observed and y.slot(again.observed[0]).assert_count == 1
            assert runtime.continuity.snapshot() == durable
    assert _footprint(store) == footprint


def test_a_k1_identity_never_stands_for_a_composite_lineage_or_the_reverse(library, k1, tmp_path):   # noqa: F811
    store = _store(library, tmp_path / "ck")
    with fresh(library) as x:
        with Runtime(_cfg(tmp_path, store)) as runtime:
            runtime.anchor_cognition(x)
            durable = runtime.continuity.snapshot()
        with K1State.create(k1, DIM, WIDTH, _w(), max_rows=8) as plain:
            with Runtime(_cfg(tmp_path, store)) as runtime:
                _refused(lambda: runtime.run_learn(plain, _learn()), "CONTINUITY_STATE_MISMATCH")
                assert runtime.fault == "CONTINUITY_STATE_MISMATCH"      # fail-stop before any K1 mutation
            with Runtime(_config(tmp_path / "k1-lineage")) as runtime:
                runtime.anchor_cognition(plain)
                assert runtime.continuity.snapshot().state_format == "K1"
                _refused(lambda: runtime.run_query(x, _query()), "COGNITION_SUBSTRATE_SWITCH")   # bound elsewhere
                runtime.release(plain)
                _refused(lambda: runtime.run_query(x, _query()), "CONTINUITY_STATE_MISMATCH")    # another format
                assert runtime.fault == "CONTINUITY_STATE_MISMATCH"
        with Runtime(_cfg(tmp_path, store)) as runtime:
            assert runtime.continuity.snapshot() == durable                  # no refusal moved the authority
            runtime.run_learn(x, _learn())                                   # the composite lineage continues


def test_a_k1_lineage_becomes_a_composite_lineage_only_by_the_explicit_import(library, k1, tmp_path):   # noqa: F811
    with K1State.create(k1, DIM, WIDTH, _w(), max_rows=8) as plain:
        with Runtime(_config(tmp_path / "c")) as runtime:       # a K1 lineage, published identity-only
            runtime.anchor_cognition(plain)
            runtime.run_learn(plain, _learn())
            assert runtime.publish_identity(plain).status == "IDENTITY_PUBLISHED"
            k1_durable = runtime.continuity.snapshot()
        k1_envelope = plain.snapshot()
        assert k1_durable.k1_state_digest == plain.state_digest()
    store = _store(library, tmp_path / "ck")
    with CompositeState.import_k1(library, CONFIG, k1_envelope) as imported, fresh(library) as other, \
            CompositeState.import_k1(library, CONFIG, k1_envelope) as observed, \
            Runtime(_cfg(tmp_path, store)) as runtime:
        assert imported.state_digest() != k1_durable.k1_state_digest      # a new (composite) identity
        # Not the authorized K1 state; or the right one with a non-empty memory: refused, nothing published.
        _refused(lambda: runtime.import_composite(other), "CONTINUITY_STATE_MISMATCH")
        with observed.transaction() as t:
            t.observe(obs(9, 9, 1))
            t.commit()
        _refused(lambda: runtime.import_composite(observed), "CONTINUITY_STATE_MISMATCH")
        _refused(lambda: runtime.import_composite(plain), "ECS_STATE")    # a K1 state is not an import target
        assert runtime.continuity.snapshot() == k1_durable
        snapshot = runtime.import_composite(imported)
        assert (snapshot.state_format, snapshot.k1_state_digest) == ("COMPOSITE", imported.state_digest())
        assert (runtime.recover_k1().disposition, runtime.recover_k1().envelope) == ("RESUMABLE", imported.snapshot())
        result = runtime.run_learn(imported, _learn(), observations=(obs(1, 2, 10),), observation_authority=AUTHORITY)
        assert imported.slot(result.observed[0]).assert_count == 1


def test_a_composite_from_an_unadmitted_library_is_refused_before_any_native_call(library, tmp_path):
    """The managed runtime drives only native code it admitted: the same bytes loaded by a pathname are not."""
    path = native_build_dir() / "native" / "ECS" / "libelpis_ecs_composite.so"
    unadmitted = CompositeLibrary(ctypes.CDLL(str(path)))
    with CompositeState.create(unadmitted, CONFIG, _w()) as x, Runtime(_config(tmp_path / "c")) as runtime:
        _refused(lambda: runtime.anchor_cognition(x), "ECS_NATIVE_UNADMITTED")
        _refused(lambda: runtime.run_query(x, _query()), "ECS_NATIVE_UNADMITTED")
        assert x.managed_lease == 0 and runtime.continuity.snapshot().k1_state_digest is None


def test_a_composite_record_is_read_only_by_a_format_aware_reader(library, tmp_path):
    """Continuity's v3 record names a composite identity and its format. RuntimeCore reads it as COMPOSITE; the
    K1-only continuity adapter fails closed rather than misread a composite identity as a K1 digest."""
    with fresh(library) as x, Runtime(_config(tmp_path / "c")) as runtime:
        snapshot = runtime.anchor_cognition(x)
        assert (snapshot.state_format, snapshot.k1_state_digest) == ("COMPOSITE", x.state_digest())
        continuity = runtime._core.library.continuity
    with pytest.raises(ContinuityError) as refused:
        ContinuityStore(continuity, tmp_path / "c").open()
    assert refused.value.code == "CONTINUITY_STATE_MISMATCH"
    # Nor does the adapter ever encode a composite authority as a (v2, K1) record.
    with pytest.raises(ContinuityError) as refused:
        continuity.encode_record(snapshot)
    assert refused.value.code == "CONTINUITY_CORRUPT"
