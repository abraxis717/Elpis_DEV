"""The runtime composition: one composition root around one continuity authority.

``Runtime`` (``composition``) wires the subsystems explicitly. It composes the
one ECS (``elpis.ECS``) for the canonical turn and ``elpis.continuity`` for the
minimal durable lineage/authority restart needs. There is exactly one runtime
composition; the beta's numbered runtime generations are retired.
"""
from .composition import CompositionError, ContextPreparation, Runtime, RuntimeConfig

__all__ = ("CompositionError", "ContextPreparation", "Runtime", "RuntimeConfig")
