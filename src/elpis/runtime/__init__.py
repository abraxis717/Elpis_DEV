"""The runtime composition: one composition root around RuntimeCore.

``Runtime`` (``composition``) wires the subsystems explicitly. Its mutable systems
authority (lifecycle, fail-stop, continuity, K1 lineage, the managed turn's native
transaction and evolution reservation) is RuntimeCore's (native/runtime, Rust;
``core`` is the thin adapter). There is exactly one runtime composition; the beta's
numbered runtime generations are retired.
"""
from .composition import CompositionError, ContextPreparation, Runtime, RuntimeConfig, RuntimeContinuity

__all__ = ("CompositionError", "ContextPreparation", "Runtime", "RuntimeConfig", "RuntimeContinuity")
