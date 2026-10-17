"""Read-only, replay-backed projection over committed ECS history.

A projection selects a bounded, canonical view of committed events for a
consumer (for example the inference context). It has no mutation, validation,
model, network, tool or execution authority. ``HistoryBinding`` is identity
data, not authentication.
"""

from .contracts import (
    ContextProjection, HistoryBinding, ProjectionError, ProjectionRequest,
    RetainedHistoryBinding,
)
from .projector import project_history, project_retained_history, project_verified_events

__all__ = [
    "ContextProjection", "HistoryBinding", "ProjectionError", "ProjectionRequest",
    "RetainedHistoryBinding", "project_history", "project_retained_history",
    "project_verified_events",
]
