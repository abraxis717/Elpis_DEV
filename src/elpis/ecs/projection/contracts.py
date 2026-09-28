"""Immutable query/result contracts; no alternate ECS state representation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

from elpis.identity import canonical_json_bytes, content_digest
from elpis.ecs.entity import TRANSITION_EVENT_KIND
from elpis.ecs.errors import EcsError
from elpis.ecs.limits import MAX_INT, SUPPORTED_SCHEDULER_PROTOCOLS

SCHEMA = "ecs.context-projection.v1"
EVENT_KINDS = frozenset(TRANSITION_EVENT_KIND.values()) | {
    "ENTITY_FOUNDED", "MESSAGE_ENQUEUED", "MESSAGE_PROCESSED",
}
MAX_RECORDS = 4096
MAX_RECORD_BYTES = 16 * 1024 * 1024
MAX_SELECTORS = 256


class ProjectionError(EcsError):
    """Deterministic query or source rejection; no partial result is returned."""


def require_int(value: Any, name: str, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ProjectionError(f"INVALID_{name}")


def require_digest(value: Any) -> None:
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ProjectionError("INVALID_DIGEST")


@dataclass(frozen=True)
class ProjectionRequest:
    """Conjunctive predicates; members within each selector are disjunctive.

    Empty selector tuples mean unrestricted. Clock bounds are inclusive.
    Message endpoint selectors apply only to MESSAGE_ENQUEUED facts.
    Patterns are an exact uppercase symbol or a prefix with one trailing '*'.
    The byte budget bounds the canonical records ARRAY, including provenance,
    brackets and commas. It does not include the fixed result/query envelope.
    """

    entity_ids: tuple[str, ...] = ()
    event_kinds: tuple[str, ...] = ()
    sender_ids: tuple[str, ...] = ()
    receiver_ids: tuple[str, ...] = ()
    clock_min: int = 0
    clock_max: int = MAX_INT
    kind_pattern: str = "*"
    max_records: int = 64
    max_record_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("entity_ids", "event_kinds", "sender_ids", "receiver_ids"):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) > MAX_SELECTORS:
                raise ProjectionError(f"INVALID_{name.upper()}")
            for value in values:
                if name == "event_kinds":
                    if type(value) is not str or value not in EVENT_KINDS:
                        raise ProjectionError("INVALID_EVENT_KIND")
                else:
                    require_digest(value)
            object.__setattr__(self, name, tuple(sorted(set(values))))
        require_int(self.clock_min, "CLOCK_MIN", 0, MAX_INT)
        require_int(self.clock_max, "CLOCK_MAX", self.clock_min, MAX_INT)
        require_int(self.max_records, "MAX_RECORDS", 0, MAX_RECORDS)
        require_int(self.max_record_bytes, "MAX_RECORD_BYTES", 2, MAX_RECORD_BYTES)
        pattern = self.kind_pattern
        if type(pattern) is not str or not 1 <= len(pattern) <= 64:
            raise ProjectionError("INVALID_KIND_PATTERN")
        stem = pattern[:-1] if pattern.endswith("*") else pattern
        if any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_" for c in stem):
            raise ProjectionError("INVALID_KIND_PATTERN")


@dataclass(frozen=True)
class HistoryBinding:
    """Content coordinates of one qualified history, NOT a trust credential.

    Integrators of the streaming hook must obtain these from a coherent,
    already replay/commit-validated source. Caller-chosen digests cannot prove
    semantic validity. The convenience project_history API derives this itself.
    """

    genesis_digest: str
    mailbox_capacity: int
    scheduler_protocol: str
    event_count: int
    head_event_digest: str
    final_state_root: str

    def __post_init__(self) -> None:
        for value in (self.genesis_digest, self.head_event_digest, self.final_state_root):
            require_digest(value)
        require_int(self.mailbox_capacity, "MAILBOX_CAPACITY", 1, MAX_INT)
        require_int(self.event_count, "EVENT_COUNT", 0, MAX_INT)
        if (type(self.scheduler_protocol) is not str
                or self.scheduler_protocol not in SUPPORTED_SCHEDULER_PROTOCOLS):
            raise ProjectionError("INVALID_SCHEDULER_PROTOCOL")


@dataclass(frozen=True)
class ProjectedEvent:
    """Exact canonical ECS bytes, with source event coordinates preserved."""

    event_index: int
    event_digest: str
    record_bytes: bytes

    def to_dict(self) -> dict:
        """Decode a fresh view; caller mutations cannot change this item."""
        return {
            "provenance": {"event_index": self.event_index, "event_digest": self.event_digest},
            "event": json.loads(self.record_bytes),
        }


@dataclass(frozen=True)
class ContextProjection:
    """Derived observation only; contains no execution capability."""

    source: HistoryBinding
    request: ProjectionRequest
    records: tuple[ProjectedEvent, ...]
    total_matches: int
    record_bytes_used: int
    budget_exhausted: tuple[str, ...]

    @property
    def truncated(self) -> bool:
        return self.total_matches > len(self.records)

    def to_dict(self) -> dict:
        """Return a fresh JSON-compatible view (digest excludes itself)."""
        return {
            "schema": SCHEMA,
            "source": asdict(self.source),
            "request": asdict(self.request),
            "records": [record.to_dict() for record in self.records],
            "budget": {
                "total_matches": self.total_matches,
                "emitted": len(self.records),
                "truncated": self.truncated,
                "record_bytes_used": self.record_bytes_used,
                "exhausted": list(self.budget_exhausted),
            },
        }

    def canonical_bytes(self) -> bytes:
        """Repository canonical UTF-8 JSON, reproducible across processes."""
        return canonical_json_bytes(self.to_dict())

    @property
    def projection_digest(self) -> str:
        """Cross-component content identity v1; not authentication."""
        return content_digest(SCHEMA, self.to_dict())
