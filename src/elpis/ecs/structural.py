"""Frozen ECS Structural R0 mutation grammar (participation-mask ontology).

Structural R0 defines which structural edits to a frozen ECS reference state
are expressible at all: each exact nonzero frozen column of a 6 x N binary64
sidecar ``Theta`` carries one ACTIVE/DISABLED participation status, and the
only operations are ABSTAIN, DISABLE_COLUMN and RESTORE_COLUMN. Column slots
are operational gauge addresses, never intrinsic entities.

Input is canonical little-endian binary64 C-order (6, N) bytes. Endianness
conversion is deliberately outside this API. Signed zeros and subnormals are
retained verbatim; NaN and infinities are rejected by the finite-value contract.

The sealed authority bytes are embedded (``structural_authority``) and are
identified by logical anchors; filesystem locations never participate in
authority identity. ``R0``, the ontology name and every ``elpis.ecs.r0.*``
digest domain are persisted protocol identity and are intentionally unchanged.

Integration status: the grammar is executable and verified, but the ECS kernel
does not yet admit Structural R0 mutations as kernel transitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
import struct
from typing import Callable
from types import MappingProxyType

from .structural_authority import (
    HEADER_ANCHOR, MANIFEST_ANCHOR, authority_entry_anchor,
    read_authority_anchor,
)

ONTOLOGY = "MICROSCOPIC_COLUMN_PARTICIPATION_MASK_R0"
VERSION = "R0"
PRIMITIVE = "745fe2d762568dced27b9ff55e373ea40943f5104d24d995c30847052044ec90"
GAUGE = "BOUND_ARRAY_COLUMN_ORDER_G0"
WIDTHS = (36, 48, 72)
# Public release pins also reject a self-consistently rewritten manifest.
EXPECTED = MappingProxyType({
    "EDIT_IDENTITY_CONTRACT.json": "080582ee20d228af4351f923559d3e0fe0b563dd7a25db58a3dc68c6026367d9",
    "EQUIVALENCE_AND_GAUGE_ANALYSIS.json": "f6e4526b91904de77f9e51c0e6484d2547e9623089a88d081334822503f08f25",
    "FROZEN_WRITABLE_BOUNDARY.json": "fca454fa84c6da890fc60994f4beb42b7078258b08872842db077cd991c31240",
    "MATERIALIZATION_CONTRACT.json": "fce6ce9c1df090eb749ce8eb18328e449c6da6534052cbeb968f822fb18b270c",
    "MUTATION_GRAMMAR.json": "b6bc30c4a686e4273393cba36dba170678c69d7eb903803b0d91326a7d90df4c",
    "README.md": "edaf184158934a8653037b90e5f88b238e3f5e96cd827686ff0915efe8bfa743",
    "SELECTED_ONTOLOGY_SPEC.json": "f69ee197d5bff6641b3538d19a04dc27cb3aee3d4f29c3d0085c0e592423ed21",
})
HEADER_SHA256 = "5ea67cfabb46155e18cfae0e4f351c4650bf2082219741374fcec6b6a63f365d"
MANIFEST_SHA256 = "78dcc7ab307145450fd9155e0183f4cbdbae54a478a71ab16a2be17fc33f5af5"

class R0Error(ValueError):
    """Deterministic closed-contract rejection, with a stable machine code."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise R0Error(code)


def _json_digest(domain: str, payload: object) -> str:
    # Same domain + NUL + canonical JSON convention as c2r6p0.domain_digest.
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + encoded).hexdigest()


def _framed(domain: str, parts: tuple[bytes, ...]) -> bytes:
    """Domain + NUL, followed by uint64 LE lengths and exact byte payloads."""
    return domain.encode("ascii") + b"\0" + b"".join(
        struct.pack("<Q", len(part)) + part for part in parts)


AuthorityReader = Callable[[str], bytes]


@dataclass(frozen=True, slots=True)
class AuthorityBundle:
    '''Verified authority bytes identified only by logical anchors/content.'''

    header: bytes
    manifest: bytes
    entries: tuple[tuple[str, bytes], ...]

    def __post_init__(self) -> None:
        _verify_bundle(self)

    @property
    def content_identity(self) -> str:
        return _verify_bundle(self)


def materialize_authority(
    reader: AuthorityReader | None = None,
) -> AuthorityBundle:
    '''Snapshot exact sealed bytes from an authority-reader hook.'''
    hook = read_authority_anchor if reader is None else reader
    _require(callable(hook), "AUTHORITY_HOOK_REQUIRED")
    try:
        header = hook(HEADER_ANCHOR)
        manifest = hook(MANIFEST_ANCHOR)
        entries = tuple(
            (name, hook(authority_entry_anchor(name)))
            for name in sorted(EXPECTED)
        )
    except (LookupError, OSError, TypeError) as exc:
        raise R0Error("AUTHORITY_ANCHOR_UNREADABLE") from exc
    return AuthorityBundle(header, manifest, entries)


def _raw_sha256(raw: bytes) -> str:
    '''Raw-byte integrity only; never structured identity.'''
    return hashlib.sha256(raw).hexdigest()


def verify_authority(
    authority: AuthorityBundle | AuthorityReader | None = None,
) -> str:
    '''Verify exact Structural R0 authority bytes from logical anchors.'''
    bundle = authority if type(authority) is AuthorityBundle else materialize_authority(authority)
    return _verify_bundle(bundle)


def _verify_bundle(bundle: AuthorityBundle) -> str:
    _require(type(bundle) is AuthorityBundle, "AUTHORITY_BUNDLE_TYPE")
    _require(type(bundle.header) is bytes and type(bundle.manifest) is bytes,
             "AUTHORITY_ANCHOR_BYTES")
    _require(type(bundle.entries) is tuple, "AUTHORITY_ENTRIES_TYPE")
    for entry in bundle.entries:
        _require(type(entry) is tuple and len(entry) == 2,
                 "AUTHORITY_ENTRY_TYPE")
        _require(type(entry[0]) is str and type(entry[1]) is bytes,
                 "AUTHORITY_ANCHOR_BYTES")
    names = tuple(name for name, _ in bundle.entries)
    _require(names == tuple(sorted(EXPECTED)), "AUTHORITY_ANCHOR_SET")
    try:
        raw = bundle.manifest
        entries = {}
        for line in raw.decode("ascii").splitlines():
            match = re.fullmatch(r"([0-9a-f]{64})  ([A-Z][A-Z0-9_]*\.(?:json|md))", line)
            _require(match is not None, "AUTHORITY_MANIFEST_MALFORMED")
            digest, name = match.groups()
            _require(name not in entries, "AUTHORITY_MANIFEST_DUPLICATE")
            entries[name] = digest
        _require(entries == EXPECTED, "AUTHORITY_MANIFEST_ENTRIES")
        _require(_raw_sha256(raw) == MANIFEST_SHA256, "AUTHORITY_MANIFEST_DIGEST")
        _require(_raw_sha256(bundle.header) == HEADER_SHA256, "AUTHORITY_HEADER_DIGEST")
        bound = dict(bundle.entries)
        _require(set(bound) == set(EXPECTED), "AUTHORITY_ANCHOR_SET")
        for name, digest in sorted(entries.items()):
            _require(_raw_sha256(bound[name]) == digest, "AUTHORITY_DIGEST_MISMATCH")
    except (UnicodeError, ValueError, TypeError) as exc:
        if isinstance(exc, R0Error):
            raise
        raise R0Error("AUTHORITY_ANCHOR_UNREADABLE") from exc
    return _json_digest("elpis.ecs.r0.authority.v1", {
        "entries": entries, "manifest": MANIFEST_SHA256, "header": HEADER_SHA256})


class Status(str, Enum):
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    FROZEN_ZERO = "FROZEN_ZERO"


class Opcode(str, Enum):
    ABSTAIN = "ABSTAIN"
    DISABLE_COLUMN = "DISABLE_COLUMN"
    RESTORE_COLUMN = "RESTORE_COLUMN"


_CODES = {Status.ACTIVE: 1, Status.DISABLED: 0, Status.FROZEN_ZERO: 2}


def _prefix(width: int) -> tuple[bytes, ...]:
    return (ONTOLOGY.encode(), VERSION.encode(), bytes.fromhex(PRIMITIVE),
            struct.pack("<Q", 6), struct.pack("<Q", width))


@dataclass(frozen=True, slots=True)
class FrozenTheta:
    width: int
    data: bytes
    authority: AuthorityBundle | AuthorityReader | None = None
    encoding: str = "binary64-le-c-order"

    def __post_init__(self) -> None:
        _require(type(self.width) is int and self.width in WIDTHS, "WIDTH_SCOPE")
        _require(self.encoding == "binary64-le-c-order", "THETA_ENCODING")
        _require(type(self.data) in (bytes, bytearray, memoryview), "THETA_BYTES")
        object.__setattr__(self, "data", bytes(self.data))
        bundle = self.authority if type(self.authority) is AuthorityBundle else materialize_authority(self.authority)
        verify_authority(bundle)
        object.__setattr__(self, "authority", bundle)
        _require(len(self.data) == 6 * self.width * 8, "THETA_SHAPE")
        # Inspect exponent bits, never parse/repack a float.
        for (bits,) in struct.iter_unpack("<Q", self.data):
            _require((bits >> 52) & 0x7ff != 0x7ff, "THETA_NONFINITE")

    def column(self, slot: int) -> bytes:
        _require(type(slot) is int and 0 <= slot < self.width, "SLOT_BOUNDS")
        return b"".join(self.data[(row * self.width + slot) * 8:
                                  (row * self.width + slot + 1) * 8]
                        for row in range(6))

    def is_zero(self, slot: int) -> bool:
        return all(bits & 0x7fffffffffffffff == 0
                   for (bits,) in struct.iter_unpack("<Q", self.column(slot)))

    @property
    def digest(self) -> str:
        return hashlib.sha256(_framed("elpis.ecs.r0.theta.v1",
                                     _prefix(self.width) + (self.data,))).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class Candidate:
    theta: FrozenTheta
    mask: tuple[Status, ...]

    @classmethod
    def _from_validated_state(
        cls, theta: FrozenTheta, mask: tuple[Status, ...] | list[Status]
    ) -> Candidate:
        """Internal state constructor used only by bind/mutate/permute."""
        _require(type(theta) is FrozenTheta, "THETA_TYPE")
        verify_authority(theta.authority)
        _require(type(mask) in (tuple, list), "MASK_TYPE")
        normalized = tuple(mask)
        _require(len(normalized) == theta.width, "MASK_LENGTH")
        for slot, status in enumerate(normalized):
            _require(type(status) is Status, "MASK_STATUS")
            _require((status is Status.FROZEN_ZERO) == theta.is_zero(slot),
                     "FROZEN_ZERO_STATUS")
        self = object.__new__(cls)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "mask", normalized)
        return self

    @classmethod
    def bind(cls, theta: FrozenTheta) -> Candidate:
        _require(type(theta) is FrozenTheta, "THETA_TYPE")
        return cls._from_validated_state(
            theta, tuple(Status.FROZEN_ZERO if theta.is_zero(i)
                         else Status.ACTIVE for i in range(theta.width))
        )

    @property
    def width(self) -> int:
        return self.theta.width

    @property
    def canonical_state(self) -> bytes:
        records = tuple(sorted(self.theta.column(i) + bytes([_CODES[s]])
                               for i, s in enumerate(self.mask)))
        return _framed("elpis.ecs.r0.equivalence.v1", _prefix(self.width) + records)

    @property
    def equivalence_digest(self) -> str:
        return hashlib.sha256(self.canonical_state).hexdigest()

    @property
    def digest(self) -> str:
        return _json_digest("elpis.ecs.r0.candidate.v1", {
            "ontology_id": ONTOLOGY, "ontology_version": VERSION,
            "primitive_contract_digest": PRIMITIVE, "d": 6, "N": self.width,
            "ordered_frozen_sidecar_digest": self.theta.digest,
            "operational_gauge_id": GAUGE,
            "authority_digest": verify_authority(self.theta.authority),
            "mask": [s.value for s in self.mask],
            "state_equivalence_digest": self.equivalence_digest,
        })

    def materialize(self) -> bytes:
        verify_authority(self.theta.authority)
        return b"".join(
            b"\0" * 8 if self.mask[slot] is Status.DISABLED else
            self.theta.data[(row * self.width + slot) * 8:
                            (row * self.width + slot + 1) * 8]
            for row in range(6) for slot in range(self.width))

    def address(self, slot: int) -> EditAddress:
        self.theta.column(slot)  # strict int/bounds check, including bool rejection
        return EditAddress(ONTOLOGY, VERSION, PRIMITIVE, 6, self.width,
                           self.theta.digest, GAUGE, slot, self.mask[slot].value,
                           self.equivalence_digest)


@dataclass(frozen=True, slots=True)
class EditAddress:
    ontology_id: str
    ontology_version: str
    primitive_contract_digest: str
    d: int
    N: int
    ordered_frozen_sidecar_digest: str
    operational_gauge_id: str
    slot_index: int
    expected_pre_status: str
    pre_state_equivalence_digest: str


@dataclass(frozen=True, slots=True)
class MutationRequest:
    op: Opcode
    expected_candidate_digest: str
    address: EditAddress | None = None

    def __post_init__(self) -> None:
        _require(type(self.op) is Opcode, "OPCODE")
        _require(type(self.expected_candidate_digest) is str and
                 re.fullmatch("[0-9a-f]{64}", self.expected_candidate_digest) is not None,
                 "CANDIDATE_DIGEST_FORMAT")
        _require((self.address is None) if self.op is Opcode.ABSTAIN else
                 type(self.address) is EditAddress, "EDIT_ADDRESS")

    @classmethod
    def from_packet(cls, packet: dict) -> MutationRequest:
        _require(type(packet) is dict and set(packet) ==
                 {"op", "expected_candidate_digest", "address"}, "PACKET_FIELDS")
        try:
            _require(type(packet["op"]) is str, "OPCODE")
            op = Opcode(packet["op"])
        except ValueError as exc:
            raise R0Error("OPCODE") from exc
        address = packet["address"]
        if address is not None:
            _require(type(address) is dict and set(address) ==
                     set(EditAddress.__dataclass_fields__), "ADDRESS_FIELDS")
            address = EditAddress(**address)
        return cls(op, packet["expected_candidate_digest"], address)


@dataclass(frozen=True, slots=True)
class MutationResult:
    candidate: Candidate
    op: Opcode
    before_digest: str
    after_digest: str
    changed: bool


def mutate(candidate: Candidate, request: MutationRequest) -> MutationResult:
    _require(type(candidate) is Candidate, "CANDIDATE_TYPE")
    _require(type(request) is MutationRequest, "REQUEST_TYPE")
    before = candidate.digest
    _require(request.expected_candidate_digest == before, "STALE_CANDIDATE")
    if request.op is Opcode.ABSTAIN:
        return MutationResult(candidate, request.op, before, before, False)
    address = request.address
    _require(type(address) is EditAddress, "EDIT_ADDRESS")
    _require(type(address.d) is int and type(address.N) is int, "ADDRESS_DIMENSIONS")
    for name in EditAddress.__dataclass_fields__:
        if name not in ("d", "N", "slot_index"):
            _require(type(getattr(address, name)) is str, "ADDRESS_FIELD_TYPE")
    expected = candidate.address(address.slot_index)
    _require(address == expected, "STALE_ADDRESS")
    pre = Status.ACTIVE if request.op is Opcode.DISABLE_COLUMN else Status.DISABLED
    _require(candidate.mask[address.slot_index] is pre, "MUTATION_PRECONDITION")
    mask = list(candidate.mask)
    mask[address.slot_index] = Status.DISABLED if pre is Status.ACTIVE else Status.ACTIVE
    after = Candidate._from_validated_state(candidate.theta, tuple(mask))
    return MutationResult(after, request.op, before, after.digest, True)


def equivalent(left: Candidate, right: Candidate) -> bool:
    """Full canonical bytes, not hash equality, decide the R0 gauge relation."""
    return left.canonical_state == right.canonical_state


def permute(candidate: Candidate, order: tuple[int, ...]) -> tuple[Candidate, tuple[int, ...]]:
    """Return a rebound gauge plus old-slot -> new-slot operational addresses.

    No ordinary materialization or mutation ever sorts or rebinds slots.
    """
    _require(type(order) in (tuple, list) and len(order) == candidate.width and
             all(type(i) is int for i in order) and
             set(order) == set(range(candidate.width)), "PERMUTATION")
    data = b"".join(candidate.theta.data[(row * candidate.width + i) * 8:
                                         (row * candidate.width + i + 1) * 8]
                    for row in range(6) for i in order)
    theta = FrozenTheta(candidate.width, data, candidate.theta.authority)
    rebound = Candidate._from_validated_state(theta, tuple(candidate.mask[i] for i in order))
    inverse = tuple(order.index(i) for i in range(candidate.width))
    return rebound, inverse
