"""Typed HACF -> ECS memory observation transport (``libelpis_hacf_observation``, runtime subsystem; docs/ECS_MEMORY.md,
"HACF observations").

The native adapter maps one validated structure hyperedge (``elpis_semantic_hyperedge_v1``) and one assertion that
names it to one typed ECS memory observation, losslessly for the fields it accepts, and inverts the map from a
resident slot. It lives in the runtime subsystem because it joins two subsystems that may not depend on each other
(structure and ECS). This module only marshals: STRUCTURE OWNS IDENTITY AND VALIDATION. Every identity here is
computed by structure's own native functions (``elpis_semantic_*``, linked into the adapter library), the adapter
refuses whatever structure's validators refuse or whatever stored identity differs from the recomputed one, and
nothing in Python canonicalizes, hashes or interprets a record.

What is undefined in the repository is refused or supplied explicitly, never inferred: the observation polarity is
the caller's (HACF assertion flags have no named meaning and must be zero); a hyperedge semantic flag, a query-local
participant, more participants than one memory observation holds and a repeated (role, ordinal) are refused. A
mapped observation is evidence to record through an explicit :class:`~elpis.runtime.cognition.ObservationAuthority`
in a managed turn (``Runtime.run_observe``, or ``Runtime.run_learn``), never a truth, a learning instruction or an
authority, and no HACF -> meaning map is defined here or anywhere (docs/ECS_MEMORY.md, "Non-claims").

:func:`record_evidence` is the direct path for admitted typed structural evidence: every item is validated and mapped
by structure and the adapter first (a refusal touches no runtime state), then all of it is recorded in ONE
observe-only managed turn (RuntimeCore ABI v5: no K1 learning), all or nothing; the :class:`EvidenceReceipt` binds
each committed relation key to the hyperedge identity, assertion identity, provenance digest, source authority,
polarity and tick it records. The memory keeps the provenance digest exactly; resolving it to source coordinates
(documents, offsets) is structure's, outside the ECS state.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass

from elpis.ECS.composite import MemoryObservation, MemorySlot, POLARITY, _Observation, _Slot
from elpis.substrate.native_admission import AdmittedLibrary, admission_of

from .errors import CompositionError

__all__ = ("HYPEREDGE_NAMESPACE", "INCIDENCE_NAMESPACE", "EvidenceEntry", "EvidenceReceipt", "HacfAssertion",
           "HacfHyperedge", "HacfObservationLibrary", "HacfParticipant", "StructuralEvidence", "record_evidence")

LIBRARY_ID = "elpis_hacf_observation"
HYPEREDGE_NAMESPACE = 0x20000000    # SEMANTIC_HYPEREDGE_NAMESPACE: hyperedge type ids
INCIDENCE_NAMESPACE = 0x30000000    # SEMANTIC_INCIDENCE_NAMESPACE: incidence-role ids
_SEMANTIC_ABI = 1
_KIND_HYPEREDGE = 2                 # SEMANTIC_OBJECT_KIND_HYPEREDGE
_MAX_PARTICIPANTS = 64              # SEMANTIC_MAX_PARTICIPANTS (the memory holds at most 8; the adapter refuses more)
_QUERY = 0x01                       # SEMANTIC_PARTICIPANT_FLAG_QUERY
_CODES = {-1: "INVALID", -2: "STRUCTURE", -3: "IDENTITY", -4: "NONCANONICAL", -5: "UNSUPPORTED"}

_Digest = C.c_uint8 * 32


class _Participant(C.Structure):
    _fields_ = [("node_identity", _Digest), ("incidence_role", C.c_uint32), ("ordinal", C.c_uint32),
                ("participant_flags", C.c_uint32), ("reserved", C.c_uint8 * 32)]


class _Hyperedge(C.Structure):
    _fields_ = [("abi_version", C.c_uint32), ("hyperedge_type", C.c_uint32), ("semantic_flags", C.c_uint32),
                ("payload_digest", _Digest), ("participant_count", C.c_uint32),
                ("participants", _Participant * _MAX_PARTICIPANTS), ("hyperedge_identity", _Digest),
                ("reserved", C.c_uint8 * 32)]


class _Assertion(C.Structure):
    _fields_ = [("abi_version", C.c_uint32), ("asserted_object_kind", C.c_uint32), ("asserted_object_digest", _Digest),
                ("provenance_digest", _Digest), ("authority", C.c_uint32), ("assertion_flags", C.c_uint32),
                ("assertion_identity", _Digest), ("reserved", C.c_uint8 * 32)]


assert C.sizeof(_Participant) == 76 and C.sizeof(_Hyperedge) == 4976 and C.sizeof(_Assertion) == 144

_ABI = {
    "elpis_semantic_canonicalize_participants": ([C.POINTER(_Participant), C.c_uint32], C.c_int),
    "elpis_semantic_hyperedge_identity": ([C.POINTER(_Hyperedge), C.POINTER(_Digest)], C.c_int),
    "elpis_semantic_hyperedge_validate": ([C.POINTER(_Hyperedge)], C.c_int),
    "elpis_semantic_assertion_identity": ([C.POINTER(_Assertion), C.POINTER(_Digest)], C.c_int),
    "elpis_semantic_assertion_validate": ([C.POINTER(_Assertion)], C.c_int),
    "elpis_ecs_obs_hacf_observation": ([C.POINTER(_Hyperedge), C.POINTER(_Assertion), C.c_uint32, C.c_uint64,
                                        C.POINTER(_Observation)], C.c_int),
    "elpis_ecs_obs_hacf_hyperedge": ([C.POINTER(_Slot), C.POINTER(_Hyperedge)], C.c_int),
}


def _refuse(stage, rc):
    return CompositionError(f"OBSERVATION_{_CODES.get(int(rc), 'INVALID')}", f"{stage} refused ({int(rc)})")


def _digest(value, what):
    if type(value) is not bytes or len(value) != 32:
        raise CompositionError("OBSERVATION_INVALID", what + ": 32 bytes")
    return _Digest.from_buffer_copy(value)


def _u32(value, what):
    if type(value) is not int or not 0 <= value < 1 << 32:
        raise CompositionError("OBSERVATION_INVALID", what + ": a 32-bit unsigned integer")
    return value


@dataclass(frozen=True)
class HacfParticipant:
    """One participant of a hyperedge: a node identity in an incidence role (namespace-prefixed id) and an ordinal.
    ``query_local`` marks a query-local overlay participant (refused by the adapter: not persistent evidence)."""

    node_identity: bytes
    incidence_role: int
    ordinal: int = 0
    query_local: bool = False


@dataclass(frozen=True)
class HacfHyperedge:
    """A structure hyperedge exactly as structure identified it: ``record`` is the complete native
    ``elpis_semantic_hyperedge_v1`` (participants in structure's canonical order, identity computed by structure).
    The other fields are read from it."""

    record: bytes
    hyperedge_type: int
    participants: tuple
    payload_digest: bytes
    identity: bytes

    @classmethod
    def _of(cls, edge):
        parts = tuple(HacfParticipant(bytes(p.node_identity), int(p.incidence_role), int(p.ordinal),
                                      bool(p.participant_flags & _QUERY))
                      for p in edge.participants[:min(edge.participant_count, _MAX_PARTICIPANTS)])
        return cls(bytes(edge), int(edge.hyperedge_type), parts, bytes(edge.payload_digest),
                   bytes(edge.hyperedge_identity))

    def _native(self):
        if type(self.record) is not bytes or len(self.record) != C.sizeof(_Hyperedge):
            raise CompositionError("OBSERVATION_INVALID", "a complete hyperedge record is required")
        return _Hyperedge.from_buffer_copy(self.record)


@dataclass(frozen=True)
class HacfAssertion:
    """An assertion that names one hyperedge: ``record`` is the complete native ``elpis_semantic_assertion_v1``
    (identity computed by structure). ``authority`` is the asserting source's id (recorded by the memory as the
    observation's source authority, never interpreted)."""

    record: bytes
    asserted: bytes
    provenance_digest: bytes
    authority: int
    identity: bytes

    def _native(self):
        if type(self.record) is not bytes or len(self.record) != C.sizeof(_Assertion):
            raise CompositionError("OBSERVATION_INVALID", "a complete assertion record is required")
        return _Assertion.from_buffer_copy(self.record)


class HacfObservationLibrary:
    """The admitted ``libelpis_hacf_observation`` (sealed, pinned under :data:`LIBRARY_ID`): structure's identity
    and validation functions and the typed HACF -> memory observation adapter."""

    __slots__ = ("_f",)

    def __init__(self, admitted):
        if type(admitted) is not AdmittedLibrary or admission_of(admitted.lib) is not admitted or \
                admitted.library_id != LIBRARY_ID:
            raise CompositionError("ECS_NATIVE_UNADMITTED", "the HACF observation adapter runs only from an admitted, "
                                                            "pinned library")
        f = {}
        for name, (args, result) in _ABI.items():
            fn = getattr(admitted.lib, name)
            fn.argtypes, fn.restype = args, result
            f[name] = fn
        self._f = f

    def hyperedge(self, hyperedge_type, participants, *, payload_digest=bytes(32), semantic_flags=0):
        """A hyperedge identified by structure: participants put in structure's canonical order, its identity
        computed and the record validated by structure's own functions."""
        if type(participants) is not tuple or not 1 <= len(participants) <= _MAX_PARTICIPANTS or \
                not all(type(p) is HacfParticipant for p in participants):
            raise CompositionError("OBSERVATION_INVALID", f"participants: a tuple of 1..{_MAX_PARTICIPANTS} "
                                                          "HacfParticipant")
        edge = _Hyperedge()
        edge.abi_version, edge.hyperedge_type = _SEMANTIC_ABI, _u32(hyperedge_type, "hyperedge_type")
        edge.semantic_flags = _u32(semantic_flags, "semantic_flags")
        edge.payload_digest, edge.participant_count = _digest(payload_digest, "payload_digest"), len(participants)
        for i, p in enumerate(participants):
            if type(p.query_local) is not bool:
                raise CompositionError("OBSERVATION_INVALID", "query_local: bool")
            edge.participants[i] = _Participant(_digest(p.node_identity, "node_identity"),
                                                _u32(p.incidence_role, "incidence_role"), _u32(p.ordinal, "ordinal"),
                                                _QUERY if p.query_local else 0)
        for stage, call in (("structure canonicalization", lambda: self._f["elpis_semantic_canonicalize_participants"](
                                edge.participants, edge.participant_count)),
                            ("structure hyperedge identity", lambda: self._f["elpis_semantic_hyperedge_identity"](
                                C.byref(edge), C.byref(edge.hyperedge_identity))),
                            ("structure hyperedge validation", lambda: self._f["elpis_semantic_hyperedge_validate"](
                                C.byref(edge)))):
            code = call()
            if code != 0:
                raise CompositionError("OBSERVATION_STRUCTURE", f"{stage} refused ({code})")
        return HacfHyperedge._of(edge)

    def assertion(self, hyperedge, provenance_digest, authority, *, assertion_flags=0):
        """An assertion of ``hyperedge`` by source ``authority`` with ``provenance_digest``, identified and validated
        by structure."""
        if type(hyperedge) is not HacfHyperedge:
            raise CompositionError("OBSERVATION_INVALID", "an identified HacfHyperedge is required")
        a = _Assertion()
        a.abi_version, a.asserted_object_kind = _SEMANTIC_ABI, _KIND_HYPEREDGE
        a.asserted_object_digest = _digest(hyperedge.identity, "hyperedge identity")
        a.provenance_digest, a.authority = _digest(provenance_digest, "provenance_digest"), _u32(authority, "authority")
        a.assertion_flags = _u32(assertion_flags, "assertion_flags")
        for stage, call in (("structure assertion identity", lambda: self._f["elpis_semantic_assertion_identity"](
                                 C.byref(a), C.byref(a.assertion_identity))),
                            ("structure assertion validation", lambda: self._f["elpis_semantic_assertion_validate"](
                                 C.byref(a)))):
            code = call()
            if code != 0:
                raise CompositionError("OBSERVATION_STRUCTURE", f"{stage} refused ({code})")
        return HacfAssertion(bytes(a), bytes(a.asserted_object_digest), bytes(a.provenance_digest), int(a.authority),
                             bytes(a.assertion_identity))

    def observation(self, hyperedge, assertion, polarity, tick):
        """The adapter: the typed memory observation of ``hyperedge`` asserted by ``assertion``, with the caller's
        explicit ``polarity`` (``ASSERT``, ``NEGATE`` or ``UNCERTAIN``) and logical ``tick``. Refused
        ``OBSERVATION_STRUCTURE`` / ``_IDENTITY`` / ``_NONCANONICAL`` / ``_UNSUPPORTED`` / ``_INVALID`` exactly as the
        native adapter refuses."""
        if type(hyperedge) is not HacfHyperedge or type(assertion) is not HacfAssertion:
            raise CompositionError("OBSERVATION_INVALID", "a HacfHyperedge and a HacfAssertion are required")
        if polarity not in POLARITY:
            raise CompositionError("OBSERVATION_INVALID", "polarity: ASSERT, NEGATE or UNCERTAIN (the caller's)")
        if type(tick) is not int or not 0 <= tick < 1 << 64:
            raise CompositionError("OBSERVATION_INVALID", "tick: a 64-bit unsigned integer")
        edge, a, out = hyperedge._native(), assertion._native(), _Observation()
        rc = self._f["elpis_ecs_obs_hacf_observation"](C.byref(edge), C.byref(a), POLARITY[polarity], tick,
                                                       C.byref(out))
        if rc != 0:
            raise _refuse("the HACF observation adapter", rc)
        return MemoryObservation._from_native(out)

    def hyperedge_of(self, slot):
        """The inverse map: the hyperedge a resident memory slot was observed from, its identity recomputed by
        structure (equal to the slot's ``structure_ref`` exactly when it was observed through this adapter)."""
        if type(slot) is not MemorySlot or type(slot.record) is not _Slot:
            raise CompositionError("OBSERVATION_INVALID", "a resident MemorySlot read from a composite is required")
        edge = _Hyperedge()
        rc = self._f["elpis_ecs_obs_hacf_hyperedge"](C.byref(slot.record), C.byref(edge))
        if rc != 0:
            raise _refuse("the HACF observation inverse", rc)
        return HacfHyperedge._of(edge)


@dataclass(frozen=True)
class StructuralEvidence:
    """One admitted piece of typed structural evidence: a structure-identified ``hyperedge``, an ``assertion`` that
    names it (its ``authority`` is the asserting source), the caller's explicit ``polarity`` (``ASSERT``, ``NEGATE`` or
    ``UNCERTAIN``: HACF defines none, so it is never inferred) and a logical ``tick`` (the memory's clock: ticks of one
    relation must not go backwards)."""

    hyperedge: HacfHyperedge
    assertion: HacfAssertion
    polarity: str
    tick: int


@dataclass(frozen=True)
class EvidenceEntry:
    """What one committed observation records: the memory relation key and the structural identities behind it."""

    relation_key: bytes
    hyperedge_identity: bytes
    assertion_identity: bytes
    provenance_digest: bytes
    source_authority: int
    polarity: str
    tick: int


@dataclass(frozen=True)
class EvidenceReceipt:
    """A committed observe-only turn of structural evidence (VOLATILE until an explicit recoverable publication).
    ``entries`` follow the submitted order; ``generation_after == generation_before + 1``; ``k1_steps == 0``."""

    entries: tuple
    grant: str
    generation_before: int
    generation_after: int
    k1_steps: int
    durability: str


def record_evidence(runtime, substrate, adapter, evidence, observation_authority, *, budget=None) -> EvidenceReceipt:
    """Record admitted typed structural evidence into the memory of the ECS composite lineage ``substrate`` without
    any K1 learning.

    ``evidence`` is a non-empty tuple of :class:`StructuralEvidence`. Each item is mapped by the admitted ``adapter``
    (:class:`HacfObservationLibrary`: structure recomputes and checks every identity; a tampered, noncanonical,
    unsupported or unvalidated record is refused ``OBSERVATION_*``) before the runtime is touched; then
    ``runtime.run_observe`` records all of them in one observe-only managed turn under the explicit
    ``observation_authority`` (every assertion's source must be one it admits), all or nothing. Refusals install
    nothing. Evidence is storage: a recorded assertion is not true, not learned and not an authority; LOOKUP reports
    a conflict between sources instead of resolving it.
    """
    if type(adapter) is not HacfObservationLibrary:
        raise CompositionError("OBSERVATION_INVALID", "an admitted HacfObservationLibrary is required")
    if type(evidence) is not tuple or not evidence or not all(type(e) is StructuralEvidence for e in evidence):
        raise CompositionError("OBSERVATION", "evidence: a non-empty tuple of StructuralEvidence")
    observations = tuple(adapter.observation(e.hyperedge, e.assertion, e.polarity, e.tick) for e in evidence)
    result = runtime.run_observe(substrate, observations, observation_authority, budget=budget)
    if len(result.observed) != len(evidence):
        raise CompositionError("RUNTIME_INVALID", "the committed observations do not match the evidence")
    entries = tuple(EvidenceEntry(key, e.hyperedge.identity, e.assertion.identity, e.assertion.provenance_digest,
                                  e.assertion.authority, e.polarity, e.tick)
                    for key, e in zip(result.observed, evidence))
    return EvidenceReceipt(entries, result.grant, result.generation_before, result.generation_after, result.k1_steps,
                           result.durability)

