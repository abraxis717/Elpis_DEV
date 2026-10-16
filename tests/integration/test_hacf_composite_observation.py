"""Typed HACF observations carried through the managed composite runtime (docs/ECS_MEMORY.md, "HACF observations").

validated HACF hyperedge + assertion -> typed observation (``libelpis_hacf_observation``; structure computes and
checks every identity) -> managed composite OBSERVE under an explicit ObservationAuthority -> atomic volatile commit
-> exact slot and LOOKUP -> recoverable publication -> restart -> restore -> the identical slot, and the inverse map
rebuilds the identical hyperedge. Adversarial: a tampered identity, a changed provenance, an assertion of another
hyperedge, undefined flags, query-local and unrepresentable participants, a noncanonical order, a source the
authority does not admit, a failed turn, an unauthorized mutation, an unadmitted adapter.

This establishes mechanical transport and preservation only: original hyperedge identity, canonical participant
order, relation type, source authority, the caller's explicit polarity, provenance and uncertainty survive exactly.
It establishes no semantic interpretation, no useful cognition and no qualified HACF -> ECS observation map; an
observation is evidence storage, never truth, a learning instruction or authority. Fixture records only.
"""
from __future__ import annotations

from dataclasses import replace
import ctypes
import hashlib

import pytest

from elpis.ECS.composite import CompositeError, CompositeState
from elpis.runtime import Runtime
from elpis.runtime.cognition import ObservationAuthority
from elpis.runtime.composition import CompositionError
from elpis.runtime.observation import (HYPEREDGE_NAMESPACE, INCIDENCE_NAMESPACE, HacfHyperedge,
                                       HacfObservationLibrary, HacfParticipant, _Assertion, _Hyperedge, _Participant)

from ..conftest import admit_native, native_build_dir
from .test_composite_runtime import AUTHORITY, CONFIG, _learn, _store, fresh, library, obs  # noqa: F401 (fixture)
from .test_k1_recovery import _cfg

RELATION = HYPEREDGE_NAMESPACE | 1
SOURCES = ObservationAuthority("test: two fixture HACF sources (TEST_ONLY TRAINING=NONE SEMANTICS=NONE)",
                               frozenset({1, 2}))


@pytest.fixture(scope="module")
def hacf():
    return HacfObservationLibrary(admit_native(native_build_dir() / "native" / "runtime" /
                                               "libelpis_hacf_observation.so"))


def node(n):
    return hashlib.sha256(b"fixture-node" + bytes([n])).digest()


def provenance(n):
    return hashlib.sha256(b"fixture-provenance" + bytes([n])).digest()


def _edge(hacf, nodes=(0, 1, 2), relation=RELATION, **kw):
    """A three-place fixture hyperedge; given out of canonical order (structure sorts it)."""
    parts = tuple(HacfParticipant(node(n), INCIDENCE_NAMESPACE | (3 - i)) for i, n in enumerate(nodes))
    return hacf.hyperedge(relation, parts, **kw)


def _refused(call, code):
    with pytest.raises(CompositionError) as info:
        call()
    assert info.value.code == code, (info.value.code, str(info.value))


def _tampered(value, struct, **fields):
    record = struct.from_buffer_copy(value.record)
    for name, raw in fields.items():
        if isinstance(raw, bytes):
            ctypes.memmove(ctypes.addressof(record) + getattr(struct, name).offset, raw, len(raw))
        else:
            setattr(record, name, raw)
    return replace(value, record=bytes(record))


def test_a_validated_hacf_assertion_reaches_the_committed_memory_and_survives_restart_exactly(
        hacf, library, tmp_path):   # noqa: F811
    edge = _edge(hacf)
    assert [p.incidence_role for p in edge.participants] == sorted(p.incidence_role for p in edge.participants)
    asserted, negated = hacf.assertion(edge, provenance(0xA0), 1), hacf.assertion(edge, provenance(0xA1), 2)
    observations = (hacf.observation(edge, asserted, "ASSERT", 30), hacf.observation(edge, negated, "NEGATE", 31))
    o = observations[0]
    # The map is lossless for the fields it accepts: nothing is invented, interpreted or dropped.
    assert (o.relation_type, o.polarity, o.tick, o.source_authority, o.provenance, o.structure_ref, o.payload) == \
        (RELATION, "ASSERT", 30, 1, provenance(0xA0), edge.identity, edge.payload_digest)
    assert o.participants == tuple((p.incidence_role, p.ordinal, p.node_identity) for p in edge.participants)

    store = _store(library, tmp_path / "ck")
    with fresh(library) as x, Runtime(_cfg(tmp_path, store)) as runtime:
        runtime.anchor_cognition(x)
        result = runtime.run_learn(x, _learn(), observations=observations, observation_authority=SOURCES)
        assert result.observed[0] == result.observed[1]          # one relation, two observations
        slot = x.slot(result.observed[0])
        assert (slot.assert_count, slot.negate_count, slot.uncertain_count, slot.last_polarity) == (1, 1, 0, "NEGATE")
        assert (slot.first_provenance, slot.last_provenance, slot.last_authority) == \
            (provenance(0xA0), provenance(0xA1), 2)
        assert slot.structure_ref == edge.identity and slot.relation_type == RELATION
        assert hacf.hyperedge_of(slot) == edge                   # the exact structure, identity recomputed
        target = slot.participants[2]
        cues = slot.participants[:2]
        conflict = x.lookup(RELATION, target[0], cues)
        assert (conflict.found, conflict.matches) == (False, 1)  # asserted and negated: reported, never resolved
        uncertain = hacf.observation(edge, hacf.assertion(edge, provenance(0xA2), 1), "UNCERTAIN", 32)
        again = hacf.observation(edge, hacf.assertion(edge, provenance(0xA3), 1), "ASSERT", 33)
        runtime.run_learn(x, _learn(), observations=(uncertain, again), observation_authority=SOURCES)
        answer = x.lookup(RELATION, target[0], cues)
        assert (answer.found, answer.answer, answer.uncertain) == (True, target[2], 1)
        assert runtime.publish_recoverable(x).acknowledged
        published, published_slot = x.snapshot(), x.slot(result.observed[0])
    with Runtime(_cfg(tmp_path, store)) as runtime:
        recovery = runtime.recover_k1()
        assert (recovery.disposition, recovery.envelope) == ("RESUMABLE", published)
        with CompositeState.restore(library, CONFIG, recovery.envelope) as y:
            restored = y.slot(result.observed[0])
            assert restored == published_slot and (restored.assert_count, restored.uncertain_count) == (2, 1)
            assert hacf.hyperedge_of(restored) == edge
            assert y.lookup(RELATION, target[0], cues) == replace(answer, state_digest=y.state_digest())


def test_malformed_or_undefined_hacf_records_are_refused_before_they_become_observations(hacf):
    edge = _edge(hacf)
    asserted = hacf.assertion(edge, provenance(0xA0), 1)
    other = _edge(hacf, nodes=(5, 6, 7))
    # A stored identity that structure does not recompute; a provenance changed after identification; an
    # assertion of another hyperedge.
    _refused(lambda: hacf.observation(_tampered(edge, _Hyperedge, hyperedge_identity=bytes(32)), asserted,
                                      "ASSERT", 1), "OBSERVATION_IDENTITY")
    _refused(lambda: hacf.observation(edge, _tampered(asserted, _Assertion, provenance_digest=provenance(0xEE)),
                                      "ASSERT", 1), "OBSERVATION_IDENTITY")
    _refused(lambda: hacf.observation(edge, hacf.assertion(other, provenance(0xA0), 1), "ASSERT", 1),
             "OBSERVATION_IDENTITY")
    # Undefined meaning is refused, never inferred: a hyperedge semantic flag, an assertion flag (HACF defines no
    # polarity), a query-local participant, more participants than one observation holds, a repeated (role,
    # ordinal); and a noncanonical order. Polarity is the caller's and must be one of the three.
    semantic = _edge(hacf, semantic_flags=1)
    _refused(lambda: hacf.observation(semantic, hacf.assertion(semantic, provenance(0xA0), 1), "ASSERT", 1),
             "OBSERVATION_UNSUPPORTED")
    flagged = hacf.assertion(edge, provenance(0xA0), 1, assertion_flags=1)
    _refused(lambda: hacf.observation(edge, flagged, "ASSERT", 1), "OBSERVATION_UNSUPPORTED")
    local = hacf.hyperedge(RELATION, (HacfParticipant(node(0), INCIDENCE_NAMESPACE | 1, query_local=True),
                                      HacfParticipant(node(1), INCIDENCE_NAMESPACE | 2)))
    _refused(lambda: hacf.observation(local, hacf.assertion(local, provenance(0xA0), 1), "ASSERT", 1),
             "OBSERVATION_UNSUPPORTED")
    wide = hacf.hyperedge(RELATION, tuple(HacfParticipant(node(n), INCIDENCE_NAMESPACE | (n + 1)) for n in range(9)))
    _refused(lambda: hacf.observation(wide, hacf.assertion(wide, provenance(0xA0), 1), "ASSERT", 1),
             "OBSERVATION_UNSUPPORTED")
    repeated = hacf.hyperedge(RELATION, (HacfParticipant(node(0), INCIDENCE_NAMESPACE | 1),
                                         HacfParticipant(node(1), INCIDENCE_NAMESPACE | 1)))
    _refused(lambda: hacf.observation(repeated, hacf.assertion(repeated, provenance(0xA0), 1), "ASSERT", 1),
             "OBSERVATION_UNSUPPORTED")
    # A self-consistent but noncanonical record: participants swapped, its identity recomputed by structure over
    # that order, and an assertion of exactly that identity. Only the canonical-order check can refuse it.
    raw = _Hyperedge.from_buffer_copy(edge.record)
    raw.participants[0], raw.participants[1] = _Participant.from_buffer_copy(bytes(raw.participants[1])), \
        _Participant.from_buffer_copy(bytes(raw.participants[0]))
    assert hacf._f["elpis_semantic_hyperedge_identity"](ctypes.byref(raw), ctypes.byref(raw.hyperedge_identity)) == 0
    swapped = HacfHyperedge._of(raw)
    assert swapped.identity != edge.identity
    _refused(lambda: hacf.observation(swapped, hacf.assertion(swapped, provenance(0xA0), 1), "ASSERT", 1),
             "OBSERVATION_NONCANONICAL")
    _refused(lambda: hacf.observation(edge, asserted, "TRUE", 1), "OBSERVATION_INVALID")
    # Structure's own validators: a hyperedge type outside its namespace, an assertion authority outside 0..3, and
    # the same authority forged into an identified record.
    _refused(lambda: _edge(hacf, relation=1), "OBSERVATION_STRUCTURE")
    _refused(lambda: hacf.assertion(edge, provenance(0xA0), 9), "OBSERVATION_STRUCTURE")
    _refused(lambda: hacf.observation(edge, _tampered(asserted, _Assertion, authority=9), "ASSERT", 1),
             "OBSERVATION_STRUCTURE")


def test_hacf_observations_need_explicit_authority_and_never_change_what_k1_learns(
        hacf, library, tmp_path):   # noqa: F811
    edge = _edge(hacf)
    o = hacf.observation(edge, hacf.assertion(edge, provenance(0xA0), 1), "ASSERT", 30)
    stranger = hacf.observation(edge, hacf.assertion(edge, provenance(0xA9), 3), "ASSERT", 31)   # valid, not admitted
    with fresh(library) as x, fresh(library) as twin, Runtime(_cfg(tmp_path, _store(library, tmp_path / "ck"))) \
            as runtime, Runtime(_cfg(tmp_path, _store(library, tmp_path / "ck-twin"), continuity="twin")) as other:
        runtime.anchor_cognition(x)
        other.anchor_cognition(twin)
        state = x.snapshot()
        _refused(lambda: runtime.run_learn(x, _learn(), observations=(o,)), "OBSERVATION_UNAUTHORIZED")
        _refused(lambda: runtime.run_learn(x, _learn(), observations=(o, stranger), observation_authority=SOURCES),
                 "OBSERVATION_AUTHORITY")
        # A failed turn installs nothing of the HACF observation: a stale observation after it aborts the turn.
        _refused(lambda: runtime.run_learn(x, _learn(), observations=(o, obs(1, 2, 1)),
                                           observation_authority=SOURCES), "ECS_REFUSED")
        with pytest.raises(CompositeError) as leased:
            x.transaction()                                       # no unmanaged mutation of a bound composite
        assert leased.value.code == "LEASED"
        assert x.snapshot() == state and runtime.fault is None
        # The same LEARN with and without the observation: identical K1 components, different memory components.
        observed = runtime.run_learn(x, _learn(), observations=(o,), observation_authority=SOURCES)
        plain = other.run_learn(twin, _learn())
        assert (observed.generation_after, observed.epoch_after) == (plain.generation_after, plain.epoch_after)
        k_x, k_twin = (library.envelope_identities(s.snapshot()) for s in (x, twin))
        assert k_x[1] == k_twin[1] and k_x[2] != k_twin[2]
        assert x.slot(observed.observed[0]).structure_ref == edge.identity


def test_the_adapter_runs_only_from_an_admitted_library():
    lib = ctypes.CDLL(str(native_build_dir() / "native" / "runtime" / "libelpis_hacf_observation.so"))
    _refused(lambda: HacfObservationLibrary(lib), "ECS_NATIVE_UNADMITTED")
