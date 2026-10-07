"""Elpis continuity: the minimal durable runtime authority needed for safe restart.

Continuity records the current durable lineage/authority the runtime needs:
the expected K1 retained-state identity of the ECS lineage, and the current
head of admitted evolution path transitions and one pending assertion identity.
It is one fixed-size record in a two-slot crash-safe register (docs/CONTINUITY.md).

The authority is implemented in Rust (native/continuity) behind a stable C ABI
(include/elpis/continuity.h). This package is a thin adapter over that ABI: it
holds no durable state, owns no format, transition law or digest, and imports
no ECS, inference or runtime code; the runtime composes it.
"""
from .adapter import (
    ContinuityError,
    ContinuityLibrary,
    ContinuityProcessDeath,
    ContinuitySnapshot,
    ContinuityStore,
    EvolutionAuthority,
)

__all__ = (
    "ContinuityError", "ContinuityLibrary", "ContinuityProcessDeath", "ContinuitySnapshot", "ContinuityStore",
    "EvolutionAuthority",
)
