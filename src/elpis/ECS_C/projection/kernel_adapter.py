"""Read-only adapter from a live Elpis ECS Kernel to ContextProjection.

This adapter sits beside the projector core.
It uses only public Kernel interfaces and the replay-backed ``project_history``
path.  It does not construct or trust a caller-provided ``HistoryBinding``.
"""

from __future__ import annotations

from elpis.ECS_C.kernel import Kernel
from elpis.ECS_C.persistence import genesis_descriptor_digest

from .contracts import ContextProjection, ProjectionRequest
from .projector import project_history, project_retained_history


def project_kernel_history(
    kernel: Kernel,
    request: ProjectionRequest,
) -> ContextProjection:
    """Project one verified materialized snapshot of committed Kernel history.

    ``Kernel.events()`` returns a detached, durability-verified materialized
    history.  ``project_history`` then performs independent semantic replay.

    A transition that commits after ``Kernel.events()`` returns is not part of
    this projection; it belongs to a later history snapshot.  The returned
    projection therefore describes the exact committed prefix it consumed,
    rather than claiming to remain synchronized with later live Kernel state.

    The adapter has no mutation, validation, tool, model, network, or execution
    authority.
    """
    if not isinstance(kernel, Kernel):
        raise TypeError("kernel must be an elpis.ECS_C.kernel.Kernel")

    # These are public immutable/bound configuration surfaces.  Capture them
    # explicitly; never reach into Kernel private state.
    genesis_label = kernel.genesis_label
    mailbox_capacity = kernel.mailbox_capacity
    scheduler_protocol = kernel.scheduler_protocol

    if kernel.retention_floor:
        # Compacted history: only the retained window exists as events. The
        # projection is replay-qualified from the verified checkpoint base and
        # binds the floor; it never presents the window as complete history.
        return project_retained_history(
            kernel.retained_base,
            kernel.retained_events(),
            request,
        )

    # Public serialized read.  It fails closed if the Kernel is not open and
    # verifies the durable event log before returning detached records.
    events = kernel.events()

    genesis_digest = genesis_descriptor_digest(
        genesis_label,
        scheduler_protocol,
    )
    return project_history(
        genesis_digest,
        events,
        request,
        mailbox_capacity=mailbox_capacity,
        scheduler_protocol=scheduler_protocol,
    )
