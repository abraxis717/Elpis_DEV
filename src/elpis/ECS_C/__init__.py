"""Elpis ECS: persistent identity, durable history, replay and topology.

The ECS is the system's memory of *what happened*. It is a deterministic,
same-process kernel:

1. canonical serialization / domain-separated digests (``canonical``);
2. deterministic entity identity and lifecycle (``entity``);
3. canonical entity state with immutable logical versions;
4. local messaging with kernel-owned sender attribution through entity-bound
   invocation ports; the entity-facing API has no sender parameter
   (``bus``, ``port``);
5. bounded mailboxes with monotonic per-sender watermarks;
6. a deterministic scheduler with an explicit ordering tuple and no semantic
   authority (``scheduler``);
7. one authoritative durable event history: a recoverable length-framed
   append with complete-frame recovery, not a syscall-level atomic
   transaction (``persistence``);
8. deterministic replay and crash recovery; the logical clock is part of
   the canonical state root and is verified as exact progression (``replay``);
9. one process-local mutation lock serializing every state transition
   (``kernel``);
10. read-only projections of committed history: interaction topology
    (``topology``, ``topology_analysis``) and bounded context projection
    (``projection``);
11. the frozen Structural R0 mutation grammar (``structural``), whose
    admission into kernel transitions is not yet implemented.

Deferred and not implemented: cross-process transport and authority,
federation, execution of Structural R0 mutations through the kernel.

Digests provide content identity and integrity only. They are not
authentication and not proof of issuer. Entity existence, messaging,
scheduling, hashing, replay and persistence never create semantic authority.
"""

from __future__ import annotations

__all__ = [
    "canonical",
    "errors",
    "entity",
    "bus",
    "port",
    "scheduler",
    "persistence",
    "replay",
    "kernel",
    "topology",
    "topology_analysis",
]
