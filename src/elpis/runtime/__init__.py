"""The runtime composition: one composition root around RuntimeCore.

``Runtime`` (``composition``) wires the subsystems explicitly. Its mutable systems
authority (lifecycle, fail-stop, continuity, K1 lineage, the managed turn's native
transaction, the durability lane and evolution reservation) is RuntimeCore's (native/runtime, Rust;
``core`` is the thin adapter). The contextual (WARM) lane (``context``) keeps the session's bounded active
context as immutable versioned views; it is never persisted. There is exactly one runtime composition; the beta's
numbered runtime generations are retired.
"""
from .composition import CompositionError, ContextPreparation, Runtime, RuntimeConfig, RuntimeContinuity
from .context import ContextLane, ContextView
from .core import DurabilityStatus, DurableTicket, VolatileCommit

__all__ = ("CompositionError", "ContextLane", "ContextPreparation", "ContextView", "DurabilityStatus", "DurableTicket",
           "Runtime", "RuntimeConfig", "RuntimeContinuity", "VolatileCommit")
