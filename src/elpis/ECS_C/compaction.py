"""Elpis ECS — state-bearing compaction checkpoint (``ecs.compaction-checkpoint.v1``).

A compaction checkpoint is the verified BASE of a bounded history: it replaces
a retired event prefix by the exact kernel state that prefix produced, plus
the global coordinates needed to continue the same logical history.

It is deliberately distinct from :class:`elpis.ECS_C.persistence.Checkpoint`
(``ecs.checkpoint.v1``), which is a marker-only rollback floor that never
carries state. A compaction checkpoint binds:

  * schema/version, genesis label + genesis digest, scheduler protocol and
    mailbox configuration;
  * the global event count, logical clock, terminal event digest (the head
    the next surviving event links to), the history digest, and the terminal
    state root;
  * the complete ``ecs.state_root.v3`` record (next founding index, full
    registry/lifecycle/state versions, mailboxes with contents, sender
    watermarks, scheduler state) and the only registry field the root omits,
    each entity's ``causing_event_id`` provenance;
  * a small bounded opaque ``extension`` mapping owned by the integrating
    component (for example a fixed-size runtime continuity summary). ECS_C
    never interprets it.

It has a canonical content identity (``checkpoint_digest``, domain-separated
SHA-256 over the canonical body). Like every digest here this is content
identity / integrity evidence, NOT authentication.

Verification before use (``CompactionCheckpoint.from_bytes``) rebuilds a
KernelState from the record and requires the rebuilt canonical state root to
equal the recorded root EXACTLY (dict equality and digest). Genesis, entity
identities, state-version digests, envelope seals, mailbox capacities and
clock/count coordinates are all recomputed, never trusted. A count-0
checkpoint must be the genesis-empty state with the zero head.

The checkpoint is ECS_C mechanical state only. It is not cognitive-state
authority and carries no ECS_G/K1 state.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from . import canonical
from .bus import Mailbox, verify_envelope
from .entity import (
    ACTIVE,
    DORMANT,
    FOUNDED,
    TERMINATED,
    EntityRecord,
    EntityStateVersion,
    entity_id_from_founding,
    founding_record,
    initial_state_digest,
    state_digest,
)
from .errors import CorruptCompactionCheckpointError, EcsError
from .limits import MAX_INT, SUPPORTED_SCHEDULER_PROTOCOLS
from .persistence import (
    GENESIS_PREV_DIGEST,
    LENGTH_PREFIX,
    STATE_ROOT_SCHEMA,
    empty_state_root,
    genesis_descriptor_digest,
    state_root_digest,
)
from .replay import KernelState, _envelope_from_dict

__all__ = (
    "COMPACTION_CHECKPOINT_SCHEMA",
    "DOMAIN_COMPACTION_CHECKPOINT",
    "MAX_CHECKPOINT_BYTES",
    "MAX_EXTENSION_BYTES",
    "CompactionCheckpoint",
)

COMPACTION_CHECKPOINT_SCHEMA = "ecs.compaction-checkpoint.v1"
DOMAIN_COMPACTION_CHECKPOINT = "ecs.compaction-checkpoint.v1"

# Hard ceilings. A caller may impose a smaller per-history bound.
MAX_CHECKPOINT_BYTES = 1 << 20
MAX_EXTENSION_BYTES = 4096

_BODY_FIELDS = frozenset({
    "schema", "genesis_label", "genesis_digest", "scheduler_protocol",
    "mailbox_capacity", "event_count", "logical_clock", "head_event_digest",
    "history_digest", "state_root", "state_root_digest", "causing_event_ids",
    "extension",
})
_ROOT_FIELDS = frozenset({
    "schema", "history_digest", "mailbox_default_capacity", "genesis_digest",
    "logical_clock", "next_founding_index", "mailbox_capacity", "entities",
    "mailboxes", "watermarks", "scheduler_state",
})
_ENTITY_FIELDS = frozenset({
    "registry_key", "entity_id", "label", "founding_index", "founding_digest",
    "state_entity_id", "prev_state_digest", "lifecycle", "state_version",
    "state_digest", "payload",
})
_MAILBOX_FIELDS = frozenset({"mailbox_key", "receiver_entity_id", "capacity", "contents"})
_LIFECYCLES = frozenset({FOUNDED, ACTIVE, DORMANT, TERMINATED})


def _fail(code: str) -> None:
    raise CorruptCompactionCheckpointError(f"COMPACTION_CHECKPOINT_INVALID: {code}")


def _int(value: Any, code: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= MAX_INT:
        _fail(code)
    return value


def _hex(value: Any, code: str) -> str:
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        _fail(code)
    return value


def _rebuild_state(body: Mapping[str, Any]) -> KernelState:
    """Rebuild and fully re-verify the KernelState a checkpoint describes."""
    root = body["state_root"]
    if type(root) is not dict or set(root) != _ROOT_FIELDS:
        _fail("STATE_ROOT_FIELDS")
    if root["schema"] != STATE_ROOT_SCHEMA:
        _fail("STATE_ROOT_SCHEMA")
    genesis = body["genesis_digest"]
    capacity = body["mailbox_capacity"]
    if root["genesis_digest"] != genesis:
        _fail("GENESIS_MISMATCH")
    if root["mailbox_capacity"] != capacity or root["mailbox_default_capacity"] != capacity:
        _fail("MAILBOX_CAPACITY_MISMATCH")
    if root["logical_clock"] != body["logical_clock"] or root["history_digest"] != body["history_digest"]:
        _fail("ROOT_COORDINATES_MISMATCH")
    if root["scheduler_state"] != {}:
        _fail("SCHEDULER_STATE")
    next_founding = _int(root["next_founding_index"], "NEXT_FOUNDING_INDEX")

    state = KernelState(
        genesis_digest=genesis,
        history_digest=body["history_digest"],
        logical_clock=body["logical_clock"],
        next_founding_index=next_founding,
        mailbox_capacity=capacity,
        scheduler_protocol=body["scheduler_protocol"],
    )

    causing = body["causing_event_ids"]
    entities = root["entities"]
    if type(entities) is not list or type(causing) is not dict:
        _fail("ENTITIES")
    if len(entities) != next_founding or set(causing) != {
            e.get("entity_id") for e in entities if type(e) is dict}:
        _fail("ENTITY_SET")
    seen_indices = set()
    for item in entities:
        if type(item) is not dict or set(item) != _ENTITY_FIELDS:
            _fail("ENTITY_FIELDS")
        eid = _hex(item["entity_id"], "ENTITY_ID")
        if item["registry_key"] != eid or item["state_entity_id"] != eid or item["founding_digest"] != eid:
            _fail("ENTITY_KEY")
        index = _int(item["founding_index"], "FOUNDING_INDEX")
        if index >= next_founding or index in seen_indices:
            _fail("FOUNDING_INDEX")
        seen_indices.add(index)
        label = item["label"]
        try:
            expected = entity_id_from_founding(founding_record(index, label, genesis))
        except EcsError:
            _fail("FOUNDING_RECORD")
        if expected != eid:
            _fail("ENTITY_IDENTITY")
        lifecycle = item["lifecycle"]
        if lifecycle not in _LIFECYCLES:
            _fail("LIFECYCLE")
        version = _int(item["state_version"], "STATE_VERSION")
        payload = item["payload"]
        if type(payload) is not dict or not set(payload) <= {"delivered"}:
            _fail("ENTITY_PAYLOAD")
        if "delivered" in payload:
            _int(payload["delivered"], "DELIVERED")
        prev = _hex(item["prev_state_digest"], "PREV_STATE_DIGEST")
        digest = _hex(item["state_digest"], "STATE_DIGEST")
        if version == 0:
            if (lifecycle != FOUNDED or payload or prev != GENESIS_PREV_DIGEST
                    or digest != initial_state_digest(eid)):
                _fail("INITIAL_STATE_VERSION")
        elif digest != state_digest(eid, version, payload):
            _fail("STATE_VERSION_DIGEST")
        state.registry.add(EntityRecord(
            entity_id=eid,
            label=label,
            founding_index=index,
            founding_digest=eid,
            lifecycle=lifecycle,
            state=EntityStateVersion(
                entity_id=eid,
                version=version,
                prev_state_digest=prev,
                state_digest=digest,
                causing_event_id=_hex(causing[eid], "CAUSING_EVENT_ID"),
                payload=dict(payload),
            ),
        ))

    mailboxes = root["mailboxes"]
    if type(mailboxes) is not list:
        _fail("MAILBOXES")
    for item in mailboxes:
        if type(item) is not dict or set(item) != _MAILBOX_FIELDS:
            _fail("MAILBOX_FIELDS")
        receiver = _hex(item["receiver_entity_id"], "MAILBOX_RECEIVER")
        if item["mailbox_key"] != receiver or receiver not in state.registry:
            _fail("MAILBOX_RECEIVER")
        if item["capacity"] != capacity or receiver in state.mailboxes._boxes:
            _fail("MAILBOX_CAPACITY")
        contents = item["contents"]
        if type(contents) is not list or len(contents) > capacity:
            _fail("MAILBOX_CONTENTS")
        box: Mailbox = state.mailboxes.box(receiver)
        for raw in contents:
            try:
                env = _envelope_from_dict(raw)
                verify_envelope(env)
            except (EcsError, KeyError, TypeError, ValueError):
                _fail("ENVELOPE")
            if env.receiver_entity_id != receiver or env.sender_entity_id not in state.registry:
                _fail("ENVELOPE_ENDPOINTS")
            box.push(env)

    watermarks = root["watermarks"]
    if type(watermarks) is not dict:
        _fail("WATERMARKS")
    for sender, sequence in watermarks.items():
        if _hex(sender, "WATERMARK_SENDER") not in state.registry:
            _fail("WATERMARK_SENDER")
        state.watermarks._wm[sender] = _int(sequence, "WATERMARK_SEQUENCE", 1)

    # The decisive check: the rebuilt projection must reproduce the recorded
    # canonical root exactly (ordering, contents and digest).
    if state.state_root() != root or state.state_root_digest() != body["state_root_digest"]:
        _fail("STATE_ROOT_REBUILD_MISMATCH")
    return state


class CompactionCheckpoint:
    """Verified, immutable compaction base. Construct via from_state/from_bytes."""

    __slots__ = ("_body", "_digest", "_bytes")

    def __init__(self, body: Mapping[str, Any], *, _verified: bool = False) -> None:
        if not _verified:
            raise TypeError("use CompactionCheckpoint.from_state or from_bytes")
        self._body = body
        self._digest = canonical.domain_digest(DOMAIN_COMPACTION_CHECKPOINT, body)
        record = dict(body)
        record["checkpoint_digest"] = self._digest
        payload = canonical.canonical_bytes(record)
        self._bytes = LENGTH_PREFIX.pack(len(payload)) + payload

    # -- construction --------------------------------------------------------
    @classmethod
    def from_state(
        cls,
        state: KernelState,
        *,
        genesis_label: str,
        event_count: int,
        head_event_digest: str,
        extension: Mapping[str, Any] | None = None,
        max_bytes: int = MAX_CHECKPOINT_BYTES,
    ) -> "CompactionCheckpoint":
        """Capture a coherent transition-boundary state as a verified base."""
        if type(state) is not KernelState:
            raise TypeError("state must be a KernelState")
        body = {
            "schema": COMPACTION_CHECKPOINT_SCHEMA,
            "genesis_label": genesis_label,
            "genesis_digest": state.genesis_digest,
            "scheduler_protocol": state.scheduler_protocol,
            "mailbox_capacity": state.mailbox_capacity,
            "event_count": event_count,
            "logical_clock": state.logical_clock,
            "head_event_digest": head_event_digest,
            "history_digest": state.history_digest,
            "state_root": state.state_root(),
            "state_root_digest": state.state_root_digest(),
            "causing_event_ids": {
                eid: state.registry.get(eid).state.causing_event_id
                for eid in state.registry.ids_sorted()
            },
            "extension": dict(extension or {}),
        }
        # Round-trip through the exact verifier used on read: a checkpoint
        # that would be rejected on reopen is never written.
        record = dict(body)
        record["checkpoint_digest"] = canonical.domain_digest(DOMAIN_COMPACTION_CHECKPOINT, body)
        payload = canonical.canonical_bytes(record)
        return cls.from_bytes(
            LENGTH_PREFIX.pack(len(payload)) + payload,
            genesis_label=genesis_label,
            mailbox_capacity=state.mailbox_capacity,
            max_bytes=max_bytes,
        )

    @classmethod
    def from_bytes(
        cls,
        raw: bytes,
        *,
        genesis_label: str,
        mailbox_capacity: int,
        scheduler_protocol: str | None = None,
        max_bytes: int = MAX_CHECKPOINT_BYTES,
    ) -> "CompactionCheckpoint":
        """Parse and fully verify a framed checkpoint; fail closed on any doubt."""
        if type(raw) is not bytes:
            _fail("BYTES")
        if type(max_bytes) is not int or not 64 <= max_bytes <= MAX_CHECKPOINT_BYTES:
            _fail("MAX_BYTES")
        if len(raw) > max_bytes:
            _fail("TOO_LARGE")
        if len(raw) < 8:
            _fail("TRUNCATED")
        length = LENGTH_PREFIX.unpack_from(raw, 0)[0]
        if length != len(raw) - 8:
            _fail("FRAME_LENGTH")
        payload = raw[8:]

        def pairs(items):
            out = {}
            for key, value in items:
                if key in out:
                    raise ValueError("duplicate key")
                out[key] = value
            return out

        try:
            record = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs)
        except (UnicodeError, ValueError, RecursionError):
            _fail("JSON")
        if type(record) is not dict or canonical.canonical_bytes(record) != payload:
            _fail("NONCANONICAL")
        if set(record) != _BODY_FIELDS | {"checkpoint_digest"}:
            _fail("FIELDS")
        body = {k: v for k, v in record.items() if k != "checkpoint_digest"}
        if record["checkpoint_digest"] != canonical.domain_digest(DOMAIN_COMPACTION_CHECKPOINT, body):
            _fail("CHECKPOINT_DIGEST")
        if body["schema"] != COMPACTION_CHECKPOINT_SCHEMA:
            _fail("SCHEMA")
        if body["genesis_label"] != genesis_label:
            _fail("GENESIS_LABEL")
        scheduler = body["scheduler_protocol"]
        if scheduler not in SUPPORTED_SCHEDULER_PROTOCOLS:
            _fail("SCHEDULER_PROTOCOL")
        if scheduler_protocol is not None and scheduler != scheduler_protocol:
            _fail("SCHEDULER_PROTOCOL_MISMATCH")
        if body["genesis_digest"] != genesis_descriptor_digest(genesis_label, scheduler):
            _fail("GENESIS_DIGEST")
        if _int(body["mailbox_capacity"], "MAILBOX_CAPACITY", 1) != mailbox_capacity:
            _fail("MAILBOX_CAPACITY_CONFIG")
        count = _int(body["event_count"], "EVENT_COUNT")
        if _int(body["logical_clock"], "LOGICAL_CLOCK") != count:
            _fail("CLOCK_COUNT_MISMATCH")
        head = _hex(body["head_event_digest"], "HEAD_EVENT_DIGEST")
        history = _hex(body["history_digest"], "HISTORY_DIGEST")
        _hex(body["state_root_digest"], "STATE_ROOT_DIGEST")
        if (count == 0) != (head == GENESIS_PREV_DIGEST) or (count == 0) != (history == GENESIS_PREV_DIGEST):
            _fail("GENESIS_COORDINATES")
        extension = body["extension"]
        if type(extension) is not dict or len(canonical.canonical_bytes(extension)) > MAX_EXTENSION_BYTES:
            _fail("EXTENSION")
        if count == 0:
            empty = empty_state_root(body["genesis_digest"], body["mailbox_capacity"])
            if body["state_root"] != empty or body["state_root_digest"] != state_root_digest(empty):
                _fail("GENESIS_STATE")
        _rebuild_state(body)
        return cls(body, _verified=True)

    # -- views ---------------------------------------------------------------
    def to_bytes(self) -> bytes:
        return self._bytes

    def kernel_state(self) -> KernelState:
        """A FRESH KernelState for this base (callers may mutate it freely)."""
        return _rebuild_state(self._body)

    @property
    def checkpoint_digest(self) -> str:
        return self._digest

    @property
    def genesis_label(self) -> str:
        return self._body["genesis_label"]

    @property
    def genesis_digest(self) -> str:
        return self._body["genesis_digest"]

    @property
    def scheduler_protocol(self) -> str:
        return self._body["scheduler_protocol"]

    @property
    def mailbox_capacity(self) -> int:
        return self._body["mailbox_capacity"]

    @property
    def event_count(self) -> int:
        return self._body["event_count"]

    @property
    def logical_clock(self) -> int:
        return self._body["logical_clock"]

    @property
    def head_event_digest(self) -> str:
        return self._body["head_event_digest"]

    @property
    def history_digest(self) -> str:
        return self._body["history_digest"]

    @property
    def state_root_digest(self) -> str:
        return self._body["state_root_digest"]

    @property
    def extension(self) -> dict:
        return json.loads(canonical.canonical_bytes(self._body["extension"]))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CompactionCheckpoint) and other._bytes == self._bytes

    def __hash__(self) -> int:
        return hash(self._digest)

    def __repr__(self) -> str:
        return (f"CompactionCheckpoint(event_count={self.event_count}, "
                f"digest={self._digest[:16]}...)")
