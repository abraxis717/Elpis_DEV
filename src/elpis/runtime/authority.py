"""The durable authority of the offline Python reference: the ECS_C authority (docs/ECS_C_AUTHORITY.md).

RuntimeCore (ABI v7) publishes to exactly one kind of durable authority, the ECS_C authority; standalone continuity,
which the reference published to until then, retired into it (docs/NATIVE_SPLIT_CENSUS.md, section 9). This module
holds two things, and owns no format, transition law or digest (the authority's are native):

* **the durable view** RuntimeCore reports: :class:`ContinuitySnapshot` and :class:`EvolutionAuthority`, the values of
  RuntimeCore's ``elpis_continuity_snapshot`` and ``elpis_continuity_evolution``, and its refusals. RuntimeCore's ABI
  keeps those layouts, values and names, and so does this view: the record ``digest`` is the continuity register
  digest of the view, the evolution ``digest`` the ``elpis.continuity.evolution-authority.v2`` one, both computed
  natively;
* **the operator's side** of the authority (:class:`AuthorityLibrary`): provisioning, and founding the sources whose
  typed observations RuntimeCore admits, through the authority's own C ABI (``include/elpis/ecs_c_authority.h``),
  with no runtime open, over a deployment-pinned ``libelpis_ecs_c_authority.so`` admitted like every native library
  (``elpis.substrate.native_admission``). A held authority receives the same events through the open runtime
  (:meth:`elpis.runtime.core.RuntimeCore.authority_transition`).

An authority lives in one root directory: its state directory ``state/`` (the two record slots) and its separately
protected authority directory ``authority/`` (the deployment key and the two rollback-witness slots), siblings
(:class:`AuthorityDirs`). Provisioning creates the five files once, at their fixed sizes; nothing here creates,
grows or removes a file afterwards (``elpis.runtime.persistence``: ``ecs_c_authority_slots``).
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

from elpis.substrate.contracts import ContractError
from elpis.substrate.native_admission import AdmittedLibrary, admission_of, admit_library

from .errors import CompositionError

__all__ = (
    "AUTHORITY_LIBRARY_ID", "AuthorityDirs", "AuthorityLibrary", "ContinuityError", "ContinuityProcessDeath",
    "ContinuitySnapshot", "EvolutionAuthority", "SourceGrant", "label", "provision_authority",
)

_U64 = (1 << 64) - 1
_TESTING_PROCESS_DEATH = 255


# -- the durable view (RuntimeCore's elpis_continuity_snapshot / _evolution) ------------------------------------------

class ContinuityError(RuntimeError):
    """A durable-authority refusal with RuntimeCore's stable code (``CONTINUITY_*``, the names its ABI keeps)."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


class ContinuityProcessDeath(BaseException):
    """Testing library only: RuntimeCore simulated process death at a named step."""


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
    """One verified durable authority as RuntimeCore reports it. ``digest`` is the record digest of the view.

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


class _Counters(C.Structure):
    """``elpis_runtime_testing_io_counters``: the authority's I/O counters (testing library)."""
    _fields_ = [(name, C.c_uint64) for name in ("opens", "preads", "pread_bytes", "pwrites", "pwrite_bytes",
                                                "data_syncs", "dir_syncs", "renames", "unlinks")]


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
        raise ContinuityError("CONTINUITY_STATE_MISMATCH", f"unknown durable state format {s.state_format}")
    e = s.evolution
    authority = EvolutionAuthority(e.revision, bytes(e.head).hex(),
                                   bytes(e.assertion).hex() if e.pending else None,
                                   bytes(s.evolution_digest).hex())
    return ContinuitySnapshot(s.generation, bytes(s.k1_state_digest) if s.anchored else None, authority,
                              bytes(s.record_digest).hex(), state_format)


# -- the authority's files ------------------------------------------------------------------------------------------

STATE_FILES = ("authority.a", "authority.b")
KEY_FILE = "deployment.key"
WITNESS_FILES = ("witness.a", "witness.b")
KEY_BYTES, WITNESS_BYTES = 32, 240
MIN_CAPACITY, MAX_CAPACITY = 2, 4096
MAX_PER_TURN = 256


def record_bytes(capacity: int) -> int:
    """One record slot's size for an entity capacity (docs/ECS_C_AUTHORITY.md, "Persistence")."""
    return 1344 + 160 * capacity


@dataclass(frozen=True)
class AuthorityDirs:
    """The two directories of one authority under ``root``: ``state`` and the separately protected ``authority``."""

    root: Path

    def __post_init__(self):
        if not isinstance(self.root, Path) or not self.root.is_absolute():
            raise CompositionError("RUNTIME_PATH", "the authority root must be an absolute Path")

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def authority(self) -> Path:
        return self.root / "authority"

    def files(self) -> tuple[Path, ...]:
        """The five files a provisioned authority consists of, and nothing else."""
        return (tuple(self.state / n for n in STATE_FILES) + (self.authority / KEY_FILE,)
                + tuple(self.authority / n for n in WITNESS_FILES))

    def provisioned(self) -> bool:
        return all(p.is_file() for p in self.files())


# -- the authority's own C ABI (the operator's side) ------------------------------------------------------------------

AUTHORITY_LIBRARY_ID = "elpis_ecs_c_authority"
_ABI_VERSION = 1
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


class _Provisioned(C.Structure):
    _fields_ = [("capacity", C.c_uint32), ("reserved", C.c_uint32), ("record_bytes", C.c_uint64),
                ("lineage", _DIGEST), ("deployment", _DIGEST), ("record_digest", _DIGEST), ("state_root", _DIGEST)]


class _Status(C.Structure):
    _fields_ = [("generation", C.c_uint64), ("record_bytes", C.c_uint64), ("capacity", C.c_uint32),
                ("entity_count", C.c_uint32), ("mode", C.c_uint32), ("anchored", C.c_uint8),
                ("state_format", C.c_uint8), ("legacy_imported", C.c_uint8), ("legacy_format", C.c_uint8),
                ("lineage", _DIGEST), ("deployment", _DIGEST), ("record_digest", _DIGEST), ("state_root", _DIGEST),
                ("identity", _DIGEST), ("publications", C.c_uint64), ("last_publication_generation", C.c_uint64),
                ("evolution", _Evolution), ("evolution_digest", _DIGEST), ("receipt_head", _DIGEST),
                ("receipt_count", C.c_uint64), ("evidence_head", _DIGEST), ("evidence_count", C.c_uint64),
                ("reconcile_head", _DIGEST), ("reconcile_count", C.c_uint64), ("reclaimed_head", _DIGEST),
                ("reclaimed_count", C.c_uint64), ("next_founding_index", C.c_uint64),
                ("legacy_generation", C.c_uint64), ("legacy_digest", _DIGEST)]


class _Entity(C.Structure):
    _fields_ = [("id", _DIGEST), ("kind", C.c_uint8), ("lifecycle", C.c_uint8), ("reserved", C.c_uint8 * 2),
                ("per_turn_limit", C.c_uint32), ("founding_index", C.c_uint64), ("founded_generation", C.c_uint64),
                ("last_generation", C.c_uint64), ("sender_watermark", C.c_uint64), ("source_version", C.c_uint64),
                ("label", _DIGEST), ("scope", _DIGEST), ("source_authority", C.c_uint32), ("reserved2", C.c_uint32)]


assert C.sizeof(_Provisioned) == 144 and C.sizeof(_Status) == 528 and C.sizeof(_Entity) == 152


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


def source_found_event(sender: bytes, seq: int, receiver: bytes, grant: SourceGrant) -> bytes:
    """ENTITY_FOUND of a source (elpis/ecs_c_authority.h, "Events")."""
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
    """ENTITY_LIFECYCLE (elpis/ecs_c_authority.h, "Events")."""
    e = bytearray(_EVENT_BYTES)
    e[0:2] = _ENTITY_LIFECYCLE.to_bytes(2, "little")
    e[8:40], e[40:48], e[48:80] = sender, seq.to_bytes(8, "little"), receiver
    e[_PAYLOAD_AT], e[_PAYLOAD_AT + 1] = before, after
    return bytes(e)


class AuthorityLibrary:
    """The ECS_C authority's C ABI over an admitted ``libelpis_ecs_c_authority.so``: what the operator does with the
    runtime closed. It holds no authority state; every call is one call into the authority."""

    @classmethod
    def admit(cls, root, path, authority) -> "AuthorityLibrary":
        try:
            return cls(admit_library(root, path, authority, AUTHORITY_LIBRARY_ID))
        except ContractError as exc:
            raise CompositionError("RUNTIME_UNPINNED", str(exc)) from exc

    def __init__(self, admitted: AdmittedLibrary):
        if type(admitted) is not AdmittedLibrary or admission_of(admitted.lib) is not admitted \
                or admitted.library_id != AUTHORITY_LIBRARY_ID:
            raise CompositionError("RUNTIME_UNPINNED", "the ECS_C authority runs only from an admitted, pinned library")
        lib = admitted.lib
        lib.elpis_ecs_c_abi_version.restype = C.c_uint32
        if lib.elpis_ecs_c_abi_version() != _ABI_VERSION:
            raise CompositionError("RUNTIME_INVALID", "ECS_C authority ABI version mismatch")
        lib.elpis_ecs_c_code_name.restype, lib.elpis_ecs_c_code_name.argtypes = C.c_char_p, [C.c_int32]
        P, S = C.c_void_p, C.c_char_p
        for name, args in {"provision": [S, C.c_size_t, S, C.c_size_t, C.c_uint32, P, P, C.POINTER(_Provisioned)],
                           "open": [S, C.c_size_t, S, C.c_size_t, C.POINTER(C.c_void_p), P],
                           "read_status": [P, C.POINTER(_Status)],
                           "entity_at": [P, C.c_uint32, C.POINTER(_Entity)],
                           "entity_find": [P, P, C.POINTER(_Entity)],
                           "next_entity_id": [P, C.c_uint8, P, P],
                           "transition": [P, C.c_uint64, P, P, P]}.items():
            fn = getattr(lib, "elpis_ecs_c_" + name)
            fn.argtypes, fn.restype = args, C.c_int32
        lib.elpis_ecs_c_close.argtypes, lib.elpis_ecs_c_close.restype = [P], None
        self.admission = admitted
        self._lib = lib

    def check(self, rc: int, detail: str = "") -> None:
        if rc != 0:
            name = self._lib.elpis_ecs_c_code_name(rc)
            raise CompositionError(name.decode() if name else f"ECS_C_UNKNOWN_{rc}", detail)

    def provision(self, dirs: AuthorityDirs, capacity: int, operator: str = "operator",
                  runtime: str = "runtime") -> dict:
        """Provision a new authority (a fresh deployment key, lineage, genesis record and witness), once."""
        if type(capacity) is not int or not MIN_CAPACITY <= capacity <= MAX_CAPACITY:
            raise CompositionError("AUTHORITY_INVALID", f"capacity: {MIN_CAPACITY}..{MAX_CAPACITY} entities")
        state, authority = str(dirs.state).encode(), str(dirs.authority).encode()
        out = _Provisioned()
        self.check(self._lib.elpis_ecs_c_provision(state, len(state), authority, len(authority), capacity,
                                                    label(operator), label(runtime), C.byref(out)), "provision")
        return {"capacity": out.capacity, "record_bytes": out.record_bytes, "lineage": bytes(out.lineage).hex(),
                "record_digest": bytes(out.record_digest).hex()}

    def found_sources(self, dirs: AuthorityDirs, grants) -> list[dict]:
        """Found (and activate) each source of ``grants`` with the authority closed to every runtime: the operator's
        events, sent by the operator entity at its next sequences."""
        state, authority = str(dirs.state).encode(), str(dirs.authority).encode()
        handle = C.c_void_p()
        self.check(self._lib.elpis_ecs_c_open(state, len(state), authority, len(authority), C.byref(handle), None),
                   "open")
        try:
            operator = _Entity()
            self.check(self._lib.elpis_ecs_c_entity_at(handle, 0, C.byref(operator)))
            sender, founded = bytes(operator.id), []
            for grant in grants:
                if type(grant) is not SourceGrant:
                    raise CompositionError("AUTHORITY_INVALID", "a SourceGrant is required")
                receiver = _DIGEST()
                self.check(self._lib.elpis_ecs_c_next_entity_id(handle, SOURCE, _DIGEST.from_buffer_copy(
                    label(grant.label)), receiver))
                self._send(handle, source_found_event(sender, self._seq(handle, sender), bytes(receiver), grant))
                if grant.activate:
                    self._send(handle, lifecycle_event(sender, self._seq(handle, sender), bytes(receiver), FOUNDED,
                                                       ACTIVE))
                founded.append({"id": bytes(receiver).hex(), "source_authority": grant.source_authority,
                                "lifecycle": "ACTIVE" if grant.activate else "FOUNDED"})
            return founded
        finally:
            self._lib.elpis_ecs_c_close(handle)

    def _seq(self, handle, sender: bytes) -> int:
        e = _Entity()
        self.check(self._lib.elpis_ecs_c_entity_find(handle, _DIGEST.from_buffer_copy(sender), C.byref(e)))
        return e.sender_watermark + 1

    def _send(self, handle, event: bytes) -> None:
        s = _Status()
        self.check(self._lib.elpis_ecs_c_read_status(handle, C.byref(s)))
        self.check(self._lib.elpis_ecs_c_transition(handle, s.generation, s.state_root,
                                                    (C.c_uint8 * _EVENT_BYTES).from_buffer_copy(event), None))


def provision_authority(library: AuthorityLibrary, root: Path, *, capacity: int, sources=()) -> AuthorityDirs:
    """The operator's provisioning of one authority under ``root``: its two directories, its five fixed files and the
    ACTIVE sources ``sources`` names. Refused if anything exists there: an authority is never re-provisioned."""
    if type(library) is not AuthorityLibrary:
        raise TypeError("provision_authority takes an AuthorityLibrary")
    dirs = AuthorityDirs(root)
    if root.exists():
        raise CompositionError("AUTHORITY_EXISTS", f"{root} exists: an authority is never re-provisioned")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o700)
    library.provision(dirs, capacity)
    library.found_sources(dirs, tuple(sources))
    return dirs
