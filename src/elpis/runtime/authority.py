"""The session authority of the Python reference: the volatile ECS_C session RuntimeCore holds (ABI v8).

RuntimeCore embeds one volatile ECS_C session (native/ECS_C/session, docs/ECS_C_SESSION.md) per open: a fresh genesis
in memory, gone at close. Nothing is provisioned, published or recovered, and no file exists (docs/ELPIS_MISSION.md,
Amendment 3). This module holds, and owns no format, transition law or digest (the session's are native):

* **the view** RuntimeCore reports: :class:`ContinuitySnapshot` and :class:`EvolutionAuthority`, the values of
  RuntimeCore's ``elpis_continuity_snapshot`` and ``elpis_continuity_evolution``, and its refusals. RuntimeCore's ABI
  keeps those layouts, values and names, and so does this view: the record ``digest`` is the continuity register
  digest of the view, the evolution ``digest`` the ``elpis.continuity.evolution-authority.v2`` one, both computed
  natively;
* **the session's configuration** (:class:`SessionConfig`): its entity capacity, an optional nonce seed (reproducible
  sessions for differential runs), the founders' labels and the sources the operator founds at every open;
* **the operator's events** on an open session (:func:`source_found_event`, :func:`lifecycle_event`), sent through
  RuntimeCore (:meth:`elpis.runtime.core.RuntimeCore.authority_transition`).

ABI v7's durable ECS_C authority (record slots, witnesses, a deployment key), its provisioning and its own library are
retired: there is no ``AuthorityDirs``, ``AuthorityLibrary`` or ``provision_authority``.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass, field
import hashlib

from .errors import CompositionError

__all__ = ("ContinuityError", "ContinuitySnapshot", "EvolutionAuthority", "SessionConfig", "SourceGrant", "label")

_U64 = (1 << 64) - 1


# -- the view (RuntimeCore's elpis_continuity_snapshot / _evolution) ---------------------------------------------------

class ContinuityError(RuntimeError):
    """A session refusal with RuntimeCore's stable code (``CONTINUITY_*``, the names its ABI keeps)."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class EvolutionAuthority:
    """Evolution authority value: idle (``pending_assertion`` None) or reserved for one assertion.

    ``digest`` is the evolution-authority v2 digest when the value came from RuntimeCore.
    """

    revision: int
    head: str
    pending_assertion: str | None = None
    digest: str | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class ContinuitySnapshot:
    """The session's authority as RuntimeCore reports it (in memory). ``digest`` is the record digest of the view.

    ``state_format`` names what the anchored identity identifies: ``"K1"`` (an ELPISGK1 retained state) or
    ``"COMPOSITE"`` (an ELPISEC1 ECS composite; the field ``k1_state_digest`` keeps its historical name, as in the C
    ABI, and then holds the composite identity, never a K1 one)."""

    generation: int
    k1_state_digest: bytes | None
    evolution: EvolutionAuthority
    digest: str | None = field(default=None, compare=False, repr=False)
    state_format: str = "K1"

    @property
    def anchored(self) -> bool:
        return self.k1_state_digest is not None


class _Evolution(C.Structure):
    _fields_ = [("revision", C.c_uint64), ("head", C.c_uint8 * 32), ("pending", C.c_uint8),
                ("reserved", C.c_uint8 * 7), ("assertion", C.c_uint8 * 32)]


class _Snapshot(C.Structure):
    _fields_ = [("generation", C.c_uint64), ("anchored", C.c_uint8), ("state_format", C.c_uint8),
                ("reserved", C.c_uint8 * 6),
                ("k1_state_digest", C.c_uint8 * 32), ("evolution", _Evolution),
                ("evolution_digest", C.c_uint8 * 32), ("record_digest", C.c_uint8 * 32)]


_DIGEST = C.c_uint8 * 32


def _digest_arg(value):
    """32 raw bytes, or a 64-char lowercase hex string, as a C digest; anything else is NULL."""
    if type(value) is bytes and len(value) == 32:
        return _DIGEST.from_buffer_copy(value)
    if type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value):
        return _DIGEST.from_buffer_copy(bytes.fromhex(value))
    return None


def _hex_digest(value):
    return _digest_arg(value) if type(value) is str else None


def _evolution_arg(value):
    if type(value) is not EvolutionAuthority or type(value.revision) is not int or not 0 <= value.revision <= _U64:
        return None
    head = _digest_arg(value.head) if type(value.head) is str else None
    pending = value.pending_assertion
    assertion = _digest_arg(pending) if type(pending) is str else (_DIGEST() if pending is None else None)
    if head is None or assertion is None:
        return None
    return _Evolution(value.revision, head, int(pending is not None), (C.c_uint8 * 7)(), assertion)


_FORMATS = {0: "K1", 1: "COMPOSITE"}   # ELPIS_CONTINUITY_FORMAT_K1 / ELPIS_CONTINUITY_FORMAT_COMPOSITE


def _from_c(s: _Snapshot) -> ContinuitySnapshot:
    """The snapshot value RuntimeCore reported (format-aware: a lineage is K1 or ECS composite). An unknown format is
    refused."""
    state_format = _FORMATS.get(s.state_format)
    if state_format is None:
        raise ContinuityError("CONTINUITY_STATE_MISMATCH", f"unknown state format {s.state_format}")
    e = s.evolution
    authority = EvolutionAuthority(e.revision, bytes(e.head).hex(),
                                   bytes(e.assertion).hex() if e.pending else None,
                                   bytes(s.evolution_digest).hex())
    return ContinuitySnapshot(s.generation, bytes(s.k1_state_digest) if s.anchored else None, authority,
                              bytes(s.record_digest).hex(), state_format)


# -- the session's configuration and the operator's events ------------------------------------------------------------

MIN_CAPACITY, MAX_CAPACITY = 2, 4096
MAX_PER_TURN = 256
_EVENT_BYTES, _PAYLOAD_AT = 512, 80
_ENTITY_FOUND, _ENTITY_LIFECYCLE = 2, 3
SOURCE, FOUNDED, ACTIVE = 3, 1, 2   # ELPIS_ECS_C_SOURCE; ELPIS_ECS_C_FOUNDED, _ACTIVE
_LABEL_DOMAIN = b"elpis.host.label.v1\x00"


def label(text: str) -> bytes:
    """The 32-byte label of an operator name: SHA-256 under the native host's label domain, so the two operator
    interfaces name the same entity the same way."""
    if type(text) is not str or not text:
        raise CompositionError("AUTHORITY_INVALID", "a label is a non-empty string")
    return hashlib.sha256(_LABEL_DOMAIN + text.encode()).digest()


@dataclass(frozen=True)
class SourceGrant:
    """A source the operator founds (and activates): its label, the memory ``source_authority`` its typed observations
    record, its per-turn bound, its scope and version."""

    label: str
    source_authority: int
    per_turn: int
    scope_label: str
    version: int = 1
    activate: bool = True

    def __post_init__(self):
        if type(self.source_authority) is not int or not 0 <= self.source_authority < 1 << 32:
            raise CompositionError("AUTHORITY_INVALID", "source_authority: a 32-bit unsigned integer")
        if type(self.per_turn) is not int or not 1 <= self.per_turn <= MAX_PER_TURN:
            raise CompositionError("AUTHORITY_INVALID", f"per_turn: 1..{MAX_PER_TURN}")
        if type(self.version) is not int or not 1 <= self.version < 1 << 63:
            raise CompositionError("AUTHORITY_INVALID", "version: a positive 63-bit integer")


@dataclass(frozen=True)
class SessionConfig:
    """What every session of a runtime is created with: its entity ``capacity`` (the two founders and the sources),
    an optional 32-byte ``nonce_seed`` (None: the kernel draws a fresh nonce for every session; a seed derives each
    session's nonce from it and the session's ordinal, so two runtimes configured alike open identical sessions), the
    founders' labels, and the ``sources`` the runtime founds (and activates) at every open, in order: a new session
    starts at genesis, so the operator's sources are founded again each time."""

    capacity: int = 20
    nonce_seed: bytes | None = None
    sources: tuple = ()
    operator_label: str = "operator"
    runtime_label: str = "runtime"

    def __post_init__(self):
        if type(self.capacity) is not int or not MIN_CAPACITY <= self.capacity <= MAX_CAPACITY:
            raise CompositionError("AUTHORITY_INVALID", f"capacity: {MIN_CAPACITY}..{MAX_CAPACITY} entities")
        if self.nonce_seed is not None and (type(self.nonce_seed) is not bytes or len(self.nonce_seed) != 32
                                            or not any(self.nonce_seed)):
            raise CompositionError("AUTHORITY_INVALID", "nonce_seed: None or 32 bytes, not all zero")
        if type(self.sources) is not tuple or any(type(g) is not SourceGrant for g in self.sources):
            raise CompositionError("AUTHORITY_INVALID", "sources: a tuple of SourceGrant")
        if len(self.sources) + 2 > self.capacity:
            raise CompositionError("AUTHORITY_INVALID", "capacity: the founders and every source need a row")
        label(self.operator_label), label(self.runtime_label)


def source_found_event(sender: bytes, seq: int, receiver: bytes, grant: SourceGrant) -> bytes:
    """ENTITY_FOUND of a source (elpis/ecs_c_types.h; the session's event layout)."""
    e = bytearray(_EVENT_BYTES)
    e[0:2] = _ENTITY_FOUND.to_bytes(2, "little")
    e[8:40], e[40:48], e[48:80] = sender, seq.to_bytes(8, "little"), receiver
    p = _PAYLOAD_AT
    e[p] = SOURCE
    e[p + 4:p + 8] = grant.per_turn.to_bytes(4, "little")
    e[p + 8:p + 16] = grant.version.to_bytes(8, "little")
    e[p + 16:p + 48], e[p + 48:p + 80] = label(grant.label), label(grant.scope_label)
    e[p + 80:p + 84] = grant.source_authority.to_bytes(4, "little")
    return bytes(e)


def lifecycle_event(sender: bytes, seq: int, receiver: bytes, before: int, after: int) -> bytes:
    """ENTITY_LIFECYCLE (elpis/ecs_c_types.h; the session's event layout)."""
    e = bytearray(_EVENT_BYTES)
    e[0:2] = _ENTITY_LIFECYCLE.to_bytes(2, "little")
    e[8:40], e[40:48], e[48:80] = sender, seq.to_bytes(8, "little"), receiver
    e[_PAYLOAD_AT], e[_PAYLOAD_AT + 1] = before, after
    return bytes(e)
