"""The runtime composition: one composition root around RuntimeCore.

``Runtime`` (``composition``) wires the subsystems explicitly. Its mutable systems authority (lifecycle, fail-stop,
its volatile ECS_C session, the K1 lineage, the managed turn's native transaction and the evolution reservation) is
RuntimeCore's (native/runtime, Rust, ABI v8; ``core`` is the thin adapter). Nothing persists: each open is a new
session in memory (docs/ELPIS_MISSION.md, Amendment 3). The contextual (WARM) lane (``context``) keeps the session's
bounded active context as immutable versioned views; it is never persisted. There is exactly one runtime composition;
the beta's numbered runtime generations are retired.
"""
from .composition import CompositionError, ContextPreparation, Runtime, RuntimeConfig, RuntimeContinuity, SessionConfig
from .context import ContextLane, ContextView
from .core import VolatileCommit

__all__ = ("CompositionError", "ContextLane", "ContextPreparation", "ContextView", "Runtime", "RuntimeConfig",
           "RuntimeContinuity", "SessionConfig", "VolatileCommit")
