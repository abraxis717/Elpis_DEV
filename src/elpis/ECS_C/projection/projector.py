"""Deterministic selection using the existing ECS validation contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
from typing import Any

from elpis.identity import canonical_json_bytes
from elpis.ECS_C import canonical
from elpis.ECS_C.bus import DEFAULT_MAILBOX_CAPACITY
from elpis.ECS_C.errors import EcsError
from elpis.ECS_C.limits import MAX_FRAME_BYTES
from elpis.ECS_C.persistence import GENESIS_PREV_DIGEST, verify_event_fields, verify_event_link
from elpis.ECS_C.replay import StreamReplay, initial_state, replay_from_events
from elpis.ECS_C.scheduler import SCHEDULER_V1

from .contracts import (
    ContextProjection, HistoryBinding, ProjectedEvent, ProjectionError,
    ProjectionRequest, RetainedHistoryBinding,
)


def _snapshot(event: Mapping[str, Any]) -> tuple[dict, bytes]:
    if type(event) is not dict:
        raise ProjectionError("EVENT_MUST_BE_ECS_DICT")
    raw = canonical.canonical_bytes(event)
    if len(raw) > MAX_FRAME_BYTES:
        raise ProjectionError("EVENT_TOO_LARGE")
    # Validation and emission operate on the SAME detached bytes.
    record = json.loads(raw)
    verify_event_fields(record, record.get("event_index"))
    return record, raw


def _matcher(request: ProjectionRequest):
    # Compile the finite pattern and membership sets once per projection.
    entity_ids = frozenset(request.entity_ids)
    kinds = frozenset(request.event_kinds)
    senders = frozenset(request.sender_ids)
    receivers = frozenset(request.receiver_ids)
    prefix = request.kind_pattern.endswith("*")
    symbol = request.kind_pattern[:-1] if prefix else request.kind_pattern

    def matches(event: dict) -> bool:
        kind = event["event_kind"]
        if entity_ids and event["entity_id"] not in entity_ids:
            return False
        if kinds and kind not in kinds:
            return False
        if not request.clock_min <= event["logical_clock"] <= request.clock_max:
            return False
        if not (kind.startswith(symbol) if prefix else kind == symbol):
            return False
        if senders or receivers:
            if kind != "MESSAGE_ENQUEUED":
                return False
            envelope = event["payload"]["envelope"]
            if senders and envelope["sender_entity_id"] not in senders:
                return False
            if receivers and envelope["receiver_entity_id"] not in receivers:
                return False
        return True

    return matches


def project_history(
    genesis_digest: str,
    events: Sequence[Mapping[str, Any]],
    request: ProjectionRequest,
    *,
    mailbox_capacity: int = DEFAULT_MAILBOX_CAPACITY,
    scheduler_protocol: str = SCHEDULER_V1,
) -> ContextProjection:
    """Qualify a materialized complete history, then project it read-only.

    Input order is irrelevant: sort by authoritative event_index, suppress
    byte-identical duplicates, reject conflicting identities. Existing ECS
    replay verifies ALL facts in detached state, including unselected events.
    This adapter is O(N) memory, as is the existing replay interface. It neither
    invokes live Kernel transitions nor writes to the durable log.
    """
    if type(request) is not ProjectionRequest:
        raise ProjectionError("INVALID_REQUEST")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes, bytearray)):
        raise ProjectionError("MATERIALIZED_SEQUENCE_REQUIRED")
    # Validate configuration even for empty histories, before any source reads.
    HistoryBinding(genesis_digest, mailbox_capacity, scheduler_protocol, 0,
                   GENESIS_PREV_DIGEST, GENESIS_PREV_DIGEST)
    try:
        unique: dict[int, tuple[dict, bytes]] = {}
        for event in events:
            record, raw = _snapshot(event)
            index = record["event_index"]
            previous = unique.get(index)
            if previous is not None and previous[1] != raw:
                raise ProjectionError("CONFLICTING_EVENT_INDEX")
            unique[index] = (record, raw)
        ordered = [unique[index][0] for index in sorted(unique)]
        state = replay_from_events(genesis_digest, ordered, mailbox_capacity,
                                   scheduler_protocol=scheduler_protocol)
        source = HistoryBinding(
            genesis_digest, mailbox_capacity, scheduler_protocol, len(ordered),
            ordered[-1]["event_digest"] if ordered else GENESIS_PREV_DIGEST,
            state.state_root_digest(),
        )
        return project_verified_events(iter(ordered), source, request)
    except ProjectionError:
        raise
    except EcsError as exc:
        raise ProjectionError(f"HISTORY_REJECTED: {exc}") from exc


def project_verified_events(
    events: Iterable[Mapping[str, Any]],
    source: HistoryBinding,
    request: ProjectionRequest,
) -> ContextProjection:
    """Integration hook for a coherent ALREADY-qualified ordered event stream.

    Recheck ECS schema, chain, digests, initial root and bound tail, but do NOT
    replay semantic transitions. The binding must come from a trusted ECS
    commit/replay owner; it is not a user-supplied authentication receipt.
    Arbitrary/unqualified histories must go through project_history.

    Single pass; O(output budget + one ECS frame) retained memory. Scan all
    records to produce exact total_matches and detect corrupt excluded tails.
    Exactly one physical record per authoritative index is required. Duplicates,
    gaps, disorder and records beyond the bound count all fail closed.
    Stop emitting at the first budget failure (canonical prefix), but continue
    validation/counting. This deliberately does not pack smaller later events.
    """
    if (type(source) not in (HistoryBinding, RetainedHistoryBinding)
            or type(request) is not ProjectionRequest):
        raise ProjectionError("INVALID_SOURCE_OR_REQUEST")
    if not isinstance(events, Iterable) or isinstance(events, (str, bytes, bytearray)):
        raise ProjectionError("EVENT_ITERABLE_REQUIRED")
    matches = _matcher(request)
    total = 0
    used = 2  # JSON array brackets, including the zero-record result.
    selected: list[ProjectedEvent] = []
    exhausted: tuple[str, ...] = ()
    try:
        if type(source) is RetainedHistoryBinding:
            # Retained window: the stream starts at the floor and links to the
            # verified checkpoint coordinates. A request that reaches into the
            # retired prefix is refused, never answered from the tail.
            if request.clock_min <= source.retention_floor:
                raise ProjectionError("BELOW_RETENTION_FLOOR")
            count = source.retention_floor
            previous_digest = source.floor_event_digest
            previous_root = source.floor_state_root
        else:
            count = 0
            previous_digest = GENESIS_PREV_DIGEST
            previous_root = initial_state(source.genesis_digest, source.mailbox_capacity,
                                          source.scheduler_protocol).state_root_digest()
        for candidate in events:
            if count >= source.event_count:
                raise ProjectionError("SOURCE_EVENT_COUNT_MISMATCH")
            record, raw = _snapshot(candidate)
            verify_event_link(record, count, previous_digest, previous_root)
            count += 1
            previous_digest = record["event_digest"]
            previous_root = record["after_state_root"]
            if not matches(record):
                continue
            total += 1
            if exhausted:
                continue
            item = ProjectedEvent(record["event_index"], record["event_digest"], raw)
            additional = len(canonical_json_bytes(item.to_dict())) + bool(selected)
            reasons = []
            if len(selected) >= request.max_records:
                reasons.append("records")
            if used + additional > request.max_record_bytes:
                reasons.append("bytes")
            if reasons:
                exhausted = tuple(reasons)
                continue
            selected.append(item)
            used += additional
        if count != source.event_count:
            raise ProjectionError("SOURCE_EVENT_COUNT_MISMATCH")
        if previous_digest != source.head_event_digest:
            raise ProjectionError("SOURCE_HEAD_MISMATCH")
        if previous_root != source.final_state_root:
            raise ProjectionError("SOURCE_ROOT_MISMATCH")
    except ProjectionError:
        raise
    except EcsError as exc:
        raise ProjectionError(f"HISTORY_REJECTED: {exc}") from exc
    return ContextProjection(source, request, tuple(selected), total, used, exhausted)


def project_retained_history(
    base,
    events: Sequence[Mapping[str, Any]],
    request: ProjectionRequest,
) -> ContextProjection:
    """Qualify a compacted history's retained window, then project it read-only.

    ``base`` is the verified :class:`elpis.ECS_C.compaction.CompactionCheckpoint`
    the window continues; ``events`` are the retained events in global order.
    Every retained event is semantically replayed from the base state (bounded
    by the window). With a zero floor this is exactly :func:`project_history`.
    Otherwise the result binds the floor (RetainedHistoryBinding, schema
    ``ecs.context-projection.retained.v1``) and requests reaching at or below
    the floor fail closed with ``BELOW_RETENTION_FLOOR``.
    """
    from elpis.ECS_C.compaction import CompactionCheckpoint

    if type(base) is not CompactionCheckpoint:
        raise ProjectionError("INVALID_RETAINED_BASE")
    if type(request) is not ProjectionRequest:
        raise ProjectionError("INVALID_REQUEST")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes, bytearray)):
        raise ProjectionError("MATERIALIZED_SEQUENCE_REQUIRED")
    if base.event_count == 0:
        return project_history(base.genesis_digest, events, request,
                               mailbox_capacity=base.mailbox_capacity,
                               scheduler_protocol=base.scheduler_protocol)
    if request.clock_min <= base.event_count:
        raise ProjectionError("BELOW_RETENTION_FLOOR")
    try:
        replay = StreamReplay(base.kernel_state(), base.event_count, base.head_event_digest)
        ordered = []
        for event in events:
            record, _raw = _snapshot(event)
            replay.apply(record)
            ordered.append(record)
        source = RetainedHistoryBinding(
            base.genesis_digest, base.mailbox_capacity, base.scheduler_protocol,
            replay.event_count, replay.head_event_digest, replay.state_root_digest,
            base.event_count, base.head_event_digest, base.state_root_digest,
        )
        return project_verified_events(iter(ordered), source, request)
    except ProjectionError:
        raise
    except EcsError as exc:
        raise ProjectionError(f"HISTORY_REJECTED: {exc}") from exc
