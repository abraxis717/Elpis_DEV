"""Elpis continuity: the minimal durable runtime authority needed for safe restart.

Continuity records the current durable lineage/authority the runtime needs:
the expected K1 retained-state identity of the ECS lineage, and the current
head of admitted evolution path transitions. It is one fixed-size record in a
two-slot crash-safe register (docs/CONTINUITY.md).

It is not an ECS, not context memory, not an event history, not topology, not
a message bus and not an audit ledger. It imports no ECS, inference, runtime
or native code; the runtime composes it.
"""
from .record import RECORD_SIZE, ContinuityError, ContinuitySnapshot, EvolutionAuthority
from .store import SLOT_NAMES, ContinuityStore

__all__ = (
    "RECORD_SIZE", "SLOT_NAMES", "ContinuityError", "ContinuitySnapshot", "ContinuityStore",
    "EvolutionAuthority",
)
