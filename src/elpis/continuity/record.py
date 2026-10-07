"""The fixed-size continuity record and its typed views (docs/CONTINUITY.md).

One record is exactly ``RECORD_SIZE`` bytes. It holds the current durable
authority, never history. Every field is fixed width, so the record cannot
grow with runtime lifetime.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

__all__ = (
    "RECORD_SIZE", "ZERO_DIGEST", "ContinuityError", "ContinuitySnapshot", "EvolutionAuthority",
    "decode_record", "encode_record",
)

MAGIC = b"ELPCONT\x01"
FORMAT_VERSION = 1
_DOMAIN = b"elpis.continuity.register.v1\x00"
_EVOLUTION_DOMAIN = b"elpis.continuity.evolution-authority.v1\x00"
# magic, version, reserved(6), generation, cognition state, reserved(7), k1 digest,
# evolution revision, evolution head.
_BODY = struct.Struct(">8sH6sQB7s32sQ32s")
RECORD_SIZE = _BODY.size + 32
ZERO_DIGEST = bytes(32)
_MAX_U63 = (1 << 63) - 1


class ContinuityError(RuntimeError):
    """Fail-closed continuity refusal with a stable machine code."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _checksum(body: bytes) -> bytes:
    return hashlib.sha256(_DOMAIN + body).digest()


@dataclass(frozen=True)
class EvolutionAuthority:
    """Current head of admitted evolution path transitions."""

    revision: int
    head: str  # 64-hex digest of the last admitted path-transition receipt, zeros at revision 0

    def __post_init__(self):
        if type(self.revision) is not int or not 0 <= self.revision <= _MAX_U63:
            raise ContinuityError("CONTINUITY_CORRUPT", "evolution revision")
        if (type(self.head) is not str or len(self.head) != 64
                or any(c not in "0123456789abcdef" for c in self.head)):
            raise ContinuityError("CONTINUITY_CORRUPT", "evolution head")
        if (self.revision == 0) != (self.head == "0" * 64):
            raise ContinuityError("CONTINUITY_CORRUPT", "evolution genesis")

    @property
    def digest(self) -> str:
        """Content identity the evolution gate binds (integrity, not authentication)."""
        return hashlib.sha256(_EVOLUTION_DOMAIN + struct.pack(">Q", self.revision)
                              + bytes.fromhex(self.head)).hexdigest()


@dataclass(frozen=True)
class ContinuitySnapshot:
    """One complete, verified continuity authority."""

    generation: int
    k1_state_digest: bytes | None  # None: no K1 lineage has been anchored
    evolution: EvolutionAuthority

    def __post_init__(self):
        if type(self.generation) is not int or not 1 <= self.generation <= _MAX_U63:
            raise ContinuityError("CONTINUITY_CORRUPT", "generation")
        if self.k1_state_digest is not None and (
                type(self.k1_state_digest) is not bytes or len(self.k1_state_digest) != 32):
            raise ContinuityError("CONTINUITY_CORRUPT", "K1 state digest")
        if type(self.evolution) is not EvolutionAuthority:
            raise ContinuityError("CONTINUITY_CORRUPT", "evolution authority")

    @property
    def anchored(self) -> bool:
        return self.k1_state_digest is not None

    @property
    def digest(self) -> str:
        """The record checksum: content identity of this whole authority."""
        return _checksum(_body(self)).hex()


def _body(snapshot: ContinuitySnapshot) -> bytes:
    anchored = snapshot.k1_state_digest is not None
    return _BODY.pack(MAGIC, FORMAT_VERSION, bytes(6), snapshot.generation, 1 if anchored else 0, bytes(7),
                      snapshot.k1_state_digest if anchored else ZERO_DIGEST,
                      snapshot.evolution.revision, bytes.fromhex(snapshot.evolution.head))


def encode_record(snapshot: ContinuitySnapshot) -> bytes:
    body = _body(snapshot)
    return body + _checksum(body)


def decode_record(raw: bytes) -> ContinuitySnapshot | None:
    """A verified snapshot, ``None`` for an empty (all-zero) slot; raises for anything else."""
    if type(raw) is not bytes or len(raw) != RECORD_SIZE:
        raise ContinuityError("CONTINUITY_CORRUPT", "record size")
    if raw == bytes(RECORD_SIZE):
        return None
    body, checksum = raw[:_BODY.size], raw[_BODY.size:]
    if _checksum(body) != checksum:
        raise ContinuityError("CONTINUITY_CORRUPT", "record checksum")
    magic, version, reserved, generation, cognition, pad, k1, revision, head = _BODY.unpack(body)
    if magic != MAGIC or version != FORMAT_VERSION or reserved != bytes(6) or pad != bytes(7):
        raise ContinuityError("CONTINUITY_CORRUPT", "record header")
    if cognition not in (0, 1) or (cognition == 0 and k1 != ZERO_DIGEST):
        raise ContinuityError("CONTINUITY_CORRUPT", "cognition field")
    return ContinuitySnapshot(generation, k1 if cognition else None, EvolutionAuthority(revision, head.hex()))
