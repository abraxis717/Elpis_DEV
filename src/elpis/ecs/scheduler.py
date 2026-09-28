"""Elpis ECS deterministic scheduler profiles.

The scheduler has NO semantic admission authority. It orders a finite ready set
under one explicitly selected protocol.

v1 — historical compatibility
------------------------------
Ordering:
    (rank, entity_id, mailbox_index, message_id)

This profile is retained for compatibility with pre-v2 histories.

v2 — committed-arrival fairness
-------------------------------
PROCESS_MESSAGE ordering:
    (rank, ready_clock, entity_id, mailbox_index, message_id)

``ready_clock`` is the committed logical clock of MESSAGE_ENQUEUED. It is
globally monotonic in one durable history and is not caller-selected.

LIFECYCLE ordering:
    (rank, founding_index, entity_id, mailbox_index, message_id)

``founding_index`` is kernel-assigned and monotonic.

F6A ratifies these pure ordering profiles. Kernel/replay integration and
genesis migration are performed in F6B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .errors import SchedulerError
from .limits import SCHEDULER_PROTOCOL_V1, SCHEDULER_PROTOCOL_V2

RANK_ENQUEUE = 0
RANK_ACTIVATE = 1

SCHEDULER_V1 = SCHEDULER_PROTOCOL_V1
SCHEDULER_V2 = SCHEDULER_PROTOCOL_V2
KNOWN_SCHEDULERS = frozenset({SCHEDULER_V1, SCHEDULER_V2})


@dataclass(frozen=True)
class ReadyItem:
    """One schedulable unit of work."""

    rank: int
    entity_id: str
    mailbox_index: int
    message_id: str
    kind: str
    ref: Any
    ready_clock: int | None = None
    founding_index: int | None = None

    def ordering_key(self, protocol: str = SCHEDULER_V1) -> tuple:
        if protocol == SCHEDULER_V1:
            return (
                self.rank,
                self.entity_id,
                self.mailbox_index,
                self.message_id,
            )

        if protocol != SCHEDULER_V2:
            raise SchedulerError(f"UNKNOWN_SCHEDULER_PROTOCOL:{protocol}")

        if self.rank == RANK_ENQUEUE:
            if (
                isinstance(self.ready_clock, bool)
                or not isinstance(self.ready_clock, int)
                or self.ready_clock < 1
            ):
                raise SchedulerError("SCHEDULER_V2_READY_CLOCK_REQUIRED")
            return (
                self.rank,
                self.ready_clock,
                self.entity_id,
                self.mailbox_index,
                self.message_id,
            )

        if self.rank == RANK_ACTIVATE:
            if (
                isinstance(self.founding_index, bool)
                or not isinstance(self.founding_index, int)
                or self.founding_index < 0
            ):
                raise SchedulerError("SCHEDULER_V2_FOUNDING_INDEX_REQUIRED")
            return (
                self.rank,
                self.founding_index,
                self.entity_id,
                self.mailbox_index,
                self.message_id,
            )

        raise SchedulerError(f"SCHEDULER_RANK_INVALID:{self.rank}")


def order_ready(
    items: Iterable[ReadyItem],
    protocol: str = SCHEDULER_V1,
) -> list[ReadyItem]:
    """Return the ready set in one fully specified deterministic order."""
    if protocol not in KNOWN_SCHEDULERS:
        raise SchedulerError(f"UNKNOWN_SCHEDULER_PROTOCOL:{protocol}")

    items = list(items)
    keys = [item.ordering_key(protocol) for item in items]
    if len(set(keys)) != len(keys):
        raise SchedulerError("AMBIGUOUS_READY_ORDER")
    return sorted(items, key=lambda item: item.ordering_key(protocol))


def run_deterministic(
    ready: Iterable[ReadyItem],
    step: Callable[[ReadyItem], None],
    max_steps: int | None = None,
    protocol: str = SCHEDULER_V1,
) -> int:
    """Execute the finite ready set in the selected deterministic order."""
    ordered = order_ready(ready, protocol=protocol)
    count = 0
    for item in ordered:
        if max_steps is not None and count >= max_steps:
            break
        step(item)
        count += 1
    return count
