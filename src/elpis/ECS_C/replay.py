"""Elpis ECS — deterministic replay and crash recovery.

Replay reconstructs the ENTIRE kernel state from the committed event history
plus the trusted genesis authority. No hidden mutable logical state is needed:
the registry, mailboxes, watermarks, logical clock, and founding counter are
all projections of the event chain.

Required invariant (adjudication):
    initial durable state + ordered committed events -> exactly one
    reconstructed kernel state
    live final state digest == fresh-process replay final state digest

Replay verification
-------------------
For each event index ``i`` (zero-event genesis model):
  1. verify the exact expected logical clock: ``logical_clock == i + 1``
     (skipped, duplicated, regressed, boolean, or negative clocks are
     rejected);
  2. apply the event mutation to the reconstructed state;
  3. set the reconstructed state's logical clock to that event's clock;
  4. verify the reconstructed ``after_state_root`` equals the event's
     ``after_state_root``.

The logical clock IS part of the canonical state root. A transition
is EXPECTED to change ``before_state_root -> after_state_root``; the roots are
not required to remain equal across a transition.

The replay verifies the event chain (integrity/tamper evidence relative to the
trusted genesis head — NOT authentication) and checks that each event's
before/after state roots match the reconstructed state. A wrong genesis/
authority input fails closed (WrongAuthorityError).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import canonical
from .bus import (
    DEFAULT_MAILBOX_CAPACITY,
    Envelope,
    MailboxSet,
    SequenceWatermark,
    verify_envelope,
    check_receiver_deliverable,
)
from .entity import (
    ACTIVE,
    DORMANT,
    FOUNDED,
    TERMINATED,
    EntityRecord,
    EntityRegistry,
    EntityStateVersion,
    initial_state_digest,
    make_state_version,
    state_digest,
    founding_record, entity_id_from_founding, validate_transition,
)
from .errors import (
    BrokenChainError,
    CorruptEventError,
    EntityError,
    ReplayError,
    WrongAuthorityError,
)
from .persistence import (
    GENESIS_PREV_DIGEST,
    build_state_root,
    state_root_digest,
    verify_event_chain,
    event_intent_digest,
)

from .scheduler import (
    KNOWN_SCHEDULERS,
    ReadyItem,
    RANK_ENQUEUE,
    SCHEDULER_V1,
    order_ready,
)

# ---------------------------------------------------------------------------
# Kernel state (a pure projection of the event history)
# ---------------------------------------------------------------------------


@dataclass
class KernelState:
    """The full kernel state, reconstructable from events + genesis."""

    genesis_digest: str
    history_digest: str = GENESIS_PREV_DIGEST
    logical_clock: int = 0
    next_founding_index: int = 0
    mailbox_capacity: int = DEFAULT_MAILBOX_CAPACITY
    scheduler_protocol: str = SCHEDULER_V1
    registry: EntityRegistry = field(default_factory=EntityRegistry)
    mailboxes: MailboxSet = field(default_factory=MailboxSet)
    watermarks: SequenceWatermark = field(default_factory=SequenceWatermark)

    def __post_init__(self) -> None:
        # Ensure the MailboxSet uses the kernel's configured capacity. The
        # capacity is AUTHORITATIVE STATE: it changes future enqueue
        # acceptance and is bound into the state root. Boxes are created
        # lazily with this capacity, so it must be set before any box is
        # created.
        self.mailboxes.capacity = self.mailbox_capacity
        if self.scheduler_protocol not in KNOWN_SCHEDULERS:
            raise ReplayError("SCHEDULER_PROTOCOL_INVALID")

    def state_root(self) -> dict:
        """The canonical state root (integration v3).

        Includes ALL behavior-affecting state:
          * genesis_digest (trusted starting authority; constant across
            events);
          * logical_clock (the current commit counter; a transition is
            EXPECTED to change before_root -> after_root);
          * next_founding_index (determines the next entity identity);
          * mailbox_capacity (changes future enqueue acceptance; immutable
            for the life of one durable history);
          * entity registry/lifecycle + canonical states;
          * mailbox contents;
          * sender sequence/watermark state;
          * scheduler state (empty: the scheduler is a pure function of the
            mailboxes).
        """
        return build_state_root(
            genesis_digest=self.genesis_digest,
            logical_clock=self.logical_clock,
            entities=self.registry.state_root_projection(),
            mailboxes=self.mailboxes.as_sorted_list(),
            watermarks=self.watermarks.as_sorted_dict(),
            next_founding_index=self.next_founding_index,
            mailbox_capacity=self.mailbox_capacity,
            scheduler_state={},
            history_digest=self.history_digest,
            mailbox_default_capacity=self.mailboxes.capacity,
        )

    def state_root_digest(self) -> str:
        return state_root_digest(self.state_root())


def initial_state(
    genesis_digest: str,
    mailbox_capacity: int = DEFAULT_MAILBOX_CAPACITY,
    scheduler_protocol: str = SCHEDULER_V1,
) -> KernelState:
    return KernelState(
        genesis_digest=genesis_digest,
        mailbox_capacity=mailbox_capacity,
        scheduler_protocol=scheduler_protocol,
    )

# ---------------------------------------------------------------------------
# Event application (pure: state -> state)
# ---------------------------------------------------------------------------


def _envelope_from_dict(d: Mapping[str, Any]) -> Envelope:
    return Envelope(
        schema=d["schema"],
        message_id=d["message_id"],
        sender_entity_id=d["sender_entity_id"],
        receiver_entity_id=d["receiver_entity_id"],
        sequence=d["sequence"],
        payload=bytes.fromhex(d["payload_hex"]),
        payload_digest=d["payload_digest"],
        logical_clock=d["logical_clock"],
    )


def _apply_event(state: KernelState, ev: Mapping[str, Any]) -> None:
    """Apply one committed event to the state (in place).

    NOTE: this does NOT advance ``state.logical_clock``. The clock is set by
    the replay/commit driver (replay_from_events / Kernel._commit) AFTER the
    mutation, so that the before_root is computed from the pre-transition
    state and the after_root from the post-transition state with the new
    clock. See the module docstring.
    """
    kind = ev["event_kind"]
    entity_id = ev.get("entity_id")
    if entity_id is not None and not isinstance(entity_id, str):
        raise CorruptEventError("ENTITY_ID_NOT_STR")

    # All non-GENESIS kinds bind a string entity ID.
    if entity_id is None:
        raise CorruptEventError(f"ENTITY_ID_MISSING: kind={kind}")
    eid: str = entity_id

    if kind == "ENTITY_FOUNDED":
        payload = ev["payload"]
        expected = entity_id_from_founding(founding_record(
            state.next_founding_index, payload["label"], state.genesis_digest))
        if payload["founding_index"] != state.next_founding_index or eid != expected or payload["founding_digest"] != expected:
            raise ReplayError("FOUNDING_IDENTITY_MISMATCH")
        record = EntityRecord(
            entity_id=eid,
            label=payload["label"],
            founding_index=payload["founding_index"],
            founding_digest=payload["founding_digest"],
            lifecycle=FOUNDED,
            state=EntityStateVersion(
                entity_id=eid,
                version=0,
                prev_state_digest=GENESIS_PREV_DIGEST,
                state_digest=initial_state_digest(eid),
                causing_event_id=ev["event_digest"],
                payload={},
            ),
        )
        state.registry.add(record)
        state.next_founding_index += 1
        return

    if kind in (
        "ENTITY_ACTIVATED",
        "ENTITY_DORMANT",
        "ENTITY_REACTIVATED",
        "ENTITY_TERMINATED",
    ):
        rec = state.registry.get(eid)
        target = {
            "ENTITY_ACTIVATED": ACTIVE,
            "ENTITY_DORMANT": DORMANT,
            "ENTITY_REACTIVATED": ACTIVE,
            "ENTITY_TERMINATED": TERMINATED,
        }[kind]
        if ev["payload"] != {"from": rec.lifecycle, "to": target} or validate_transition(rec.lifecycle, target) != kind:
            raise ReplayError("LIFECYCLE_TRANSITION_MISMATCH")
        new_version = rec.state.version + 1
        new_state = make_state_version(
            entity_id=eid,
            version=new_version,
            prev_state_digest=rec.state.state_digest,
            causing_event_id=ev["event_digest"],
            payload=dict(rec.state.payload),
        )
        rec.lifecycle = target
        rec.state = new_state
        state.registry.replace(rec)
        return

    if kind == "MESSAGE_ENQUEUED":
        env = _envelope_from_dict(ev["payload"]["envelope"])
        verify_envelope(env)
        if eid != env.receiver_entity_id or state.registry.get(env.sender_entity_id).lifecycle != ACTIVE:
            raise ReplayError("ENQUEUE_ATTRIBUTION_OR_SENDER_STATE")
        check_receiver_deliverable(state.registry.get(eid).lifecycle)
        # The envelope's logical_clock is the COMMIT clock of the
        # enqueue event (pre-transition clock + 1). It must equal the event's
        # logical_clock; message identity (message_id) does NOT include the
        # clock, so this is a consistency check, not an identity input.
        if env.logical_clock != ev["logical_clock"]:
            raise ReplayError(
                f"ENVELOPE_CLOCK_MISMATCH: envelope clock={env.logical_clock} "
                f"event clock={ev['logical_clock']} (the envelope clock is the "
                "commit clock of the enqueue event)"
            )
        # Watermark advances at commit (enqueue), not at processing.
        state.watermarks.admit(env.sender_entity_id, env.sequence)
        state.mailboxes.box(env.receiver_entity_id).push(env)
        return

    if kind == "MESSAGE_PROCESSED":
        payload = ev["payload"]
        mid = payload["message_id"]
        receiver = payload["receiver_entity_id"]
        if eid != receiver or state.registry.get(receiver).lifecycle != ACTIVE:
            raise ReplayError("PROCESS_RECEIVER_NOT_ACTIVE_OR_MISMATCH")
        # Only the deterministic scheduler's head may be consumed.
        if state.scheduler_protocol == SCHEDULER_V1:
            ready_receivers = sorted(
                r
                for r, b in state.mailboxes._boxes.items()
                if len(b) and state.registry.get(r).lifecycle == ACTIVE
            )
            if not ready_receivers or receiver != ready_receivers[0]:
                raise ReplayError("PROCESS_SCHEDULER_ORDER")
        else:
            ready = []
            for candidate_receiver in sorted(state.mailboxes._boxes):
                candidate_box = state.mailboxes._boxes[candidate_receiver]
                if state.registry.get(candidate_receiver).lifecycle != ACTIVE:
                    continue
                for mailbox_index, envelope in enumerate(candidate_box._queue):
                    ready.append(
                        ReadyItem(
                            rank=RANK_ENQUEUE,
                            entity_id=candidate_receiver,
                            mailbox_index=mailbox_index,
                            message_id=envelope.message_id,
                            kind="PROCESS_MESSAGE",
                            ref=envelope.message_id,
                            ready_clock=envelope.logical_clock,
                        )
                    )
            ordered = order_ready(ready, protocol=state.scheduler_protocol)
            if (
                not ordered
                or receiver != ordered[0].entity_id
                or mid != ordered[0].message_id
            ):
                raise ReplayError("PROCESS_SCHEDULER_ORDER")

        box = state.mailboxes.box(receiver)
        head = box.peek()
        if head is None or head.message_id != mid:
            raise ReplayError(
                f"PROCESS_MISMATCH: expected head={None if head is None else head.message_id} "
                f"got={mid} (FIFO order violated)"
            )
        box.pop()
        # Processing is a state transition of the receiver: bump its version
        # and its mechanical delivery counter.
        rec = state.registry.get(receiver)
        new_payload = dict(rec.state.payload)
        new_payload["delivered"] = new_payload.get("delivered", 0) + 1
        new_state = make_state_version(
            entity_id=receiver,
            version=rec.state.version + 1,
            prev_state_digest=rec.state.state_digest,
            causing_event_id=ev["event_digest"],
            payload=new_payload,
        )
        rec.state = new_state
        state.registry.replace(rec)
        return

    raise ReplayError(f"UNKNOWN_EVENT_KIND: {kind}")


def replay_from_events(
    genesis_digest: str,
    events: Sequence[Mapping[str, Any]],
    mailbox_capacity: int = DEFAULT_MAILBOX_CAPACITY,
    scheduler_protocol: str = SCHEDULER_V1,
) -> KernelState:
    """Reconstruct the kernel state from the genesis authority + event chain.

    Fails closed on: wrong genesis, broken chain, clock-progression
    violation, state-root discontinuity, or an unknown event kind. Returns
    the unique reconstructed state.

    Per-event protocol (see module docstring):
      1. verify the exact expected logical clock (``i + 1``);
      2. apply the event mutation;
      3. set the reconstructed state's clock to the event's clock;
      4. verify the reconstructed after_state_root.
    """
    # Verify the chain (integrity/tamper evidence, not authentication). This
    # includes the exact clock-progression check.
    verify_event_chain(events)

    state = initial_state(
        genesis_digest, mailbox_capacity, scheduler_protocol
    )
    initial_root = state.state_root_digest()

    for i, ev in enumerate(events):
        if i == 0:
            # The first event's before_state_root must equal the initial root
            # (which binds genesis_digest AND mailbox_capacity). A capacity
            # mismatch therefore fails closed here.
            if ev["before_state_root"] != initial_root:
                raise WrongAuthorityError(
                    "WRONG_GENESIS_OR_CAPACITY: first event before_state_root "
                    "does not match the initial state root for this genesis "
                    "authority and mailbox capacity"
                )
        # Step 1: exact expected clock (already enforced by
        # verify_event_chain; re-asserted here for the per-event protocol).
        expected_clock = i + 1
        if ev["logical_clock"] != expected_clock:
            raise BrokenChainError(
                f"CLOCK_PROGRESSION_VIOLATION: index={i} "
                f"logical_clock={ev['logical_clock']} expected={expected_clock}"
            )
        if ev["before_state_root"] != state.state_root_digest():
            raise BrokenChainError("BEFORE_STATE_ROOT_MISMATCH")
        # Apply only validated mechanical transitions.
        _apply_event(state, ev)
        # Step 3: set the reconstructed state's clock to the event's clock.
        state.logical_clock = ev["logical_clock"]
        state.history_digest = event_intent_digest(ev)
        # Step 4: verify the reconstructed after_state_root.
        if state.state_root_digest() != ev["after_state_root"]:
            raise BrokenChainError(
                f"STATE_ROOT_MISMATCH: index={i} reconstructed="
                f"{state.state_root_digest()} event_after={ev['after_state_root']}"
            )
    return state


def replay_with_checkpoint(
    genesis_digest: str,
    events: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any] | None,
    mailbox_capacity: int = DEFAULT_MAILBOX_CAPACITY,
    scheduler_protocol: str = SCHEDULER_V1,
) -> KernelState:
    """Full replay plus a valid-local monotonic rollback-floor check.

    The checkpoint is never a state snapshot and never skips replay. Event
    history remains authoritative. A valid marker rejects complete-frame
    rollback behind its event index and divergence at that event. Appended
    valid history remains legal. Missing/corrupt marker data is represented as
    ``None`` by CheckpointStore and supplies no rollback anchor.
    """
    state = replay_from_events(genesis_digest, events, mailbox_capacity, scheduler_protocol=scheduler_protocol)
    if checkpoint is None:
        return state
    index = checkpoint.get("event_index")
    if type(index) is not int or index < 0:
        raise WrongAuthorityError("CHECKPOINT_INVALID: event_index")
    if index >= len(events):
        raise WrongAuthorityError(
            "HISTORY_ROLLBACK: checkpoint event index "
            f"{index} is beyond recovered history tail {len(events) - 1}"
        )
    observed_digest = events[index]["event_digest"]
    expected_digest = checkpoint.get("event_digest")
    if observed_digest != expected_digest:
        raise WrongAuthorityError(
            "HISTORY_DIVERGENCE: checkpoint event digest mismatch "
            f"at index {index} (history={observed_digest} checkpoint={expected_digest})"
        )
    if index == len(events) - 1:
        replayed_root = state.state_root_digest()
        expected_root = checkpoint.get("state_root_digest")
        if replayed_root != expected_root:
            raise WrongAuthorityError(
                "HISTORY_DIVERGENCE: checkpoint state root mismatch at tail "
                f"(replayed={replayed_root} checkpoint={expected_root})"
            )
    return state


class StreamReplay:
    """Incremental replay of a verified event stream onto a verified base state.

    ``state`` must be a KernelState whose coordinates are already verified:
    either the genesis-empty state (``base_count == 0``) or a state rebuilt
    from a verified compaction checkpoint whose ``logical_clock`` equals
    ``base_count``. Each ``apply(event)`` performs the same per-event protocol
    as :func:`replay_from_events` at GLOBAL index ``base_count + i``:

      1. structural link (schema, digests, global index, exact clock, prev
         digest, before root == reconstructed root);
      2. apply the event mutation;
      3. set the clock and history digest;
      4. verify the reconstructed after_state_root.

    Retained memory is the state plus the current head; events are not kept.
    """

    def __init__(self, state: KernelState, base_count: int, base_head: str) -> None:
        if type(base_count) is not int or base_count < 0 or state.logical_clock != base_count:
            raise ReplayError("STREAM_REPLAY_BASE_CLOCK_MISMATCH")
        if (base_count == 0) != (base_head == GENESIS_PREV_DIGEST):
            raise ReplayError("STREAM_REPLAY_BASE_HEAD_MISMATCH")
        self.state = state
        self.event_count = base_count
        self.head_event_digest = base_head
        self._root = state.state_root_digest()

    @property
    def state_root_digest(self) -> str:
        return self._root

    def apply(self, ev: Mapping[str, Any]) -> None:
        from .persistence import verify_event_link
        index = self.event_count
        if index == 0 and ev.get("before_state_root") != self._root:
            raise WrongAuthorityError(
                "WRONG_GENESIS_OR_CAPACITY: first event before_state_root "
                "does not match the initial state root for this genesis "
                "authority and mailbox capacity"
            )
        verify_event_link(ev, index, self.head_event_digest, self._root)
        _apply_event(self.state, ev)
        self.state.logical_clock = ev["logical_clock"]
        self.state.history_digest = event_intent_digest(ev)
        root = self.state.state_root_digest()
        if root != ev["after_state_root"]:
            raise BrokenChainError(
                f"STATE_ROOT_MISMATCH: index={index} reconstructed="
                f"{root} event_after={ev['after_state_root']}"
            )
        self._root = root
        self.event_count = index + 1
        self.head_event_digest = ev["event_digest"]
