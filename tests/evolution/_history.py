"""A small real ECS history and its context projection for evolution tests."""
from __future__ import annotations

from elpis.ECS_C.kernel import Kernel
from elpis.ECS_C.projection import ProjectionRequest, project_history


def ecs_projection(root):
    with Kernel(str(root)).open() as kernel:
        a = kernel.found_entity("population")
        b = kernel.found_entity("environment")
        kernel.run_until_quiescent()
        kernel.entity_port(b).propose(a, b"observation")
        kernel.run_until_quiescent()
        genesis = kernel.state.genesis_digest
        events = kernel.events()
    return project_history(genesis, events, ProjectionRequest())
