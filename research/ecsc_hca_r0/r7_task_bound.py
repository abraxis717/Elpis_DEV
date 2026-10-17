"""R7 DEV research-only: task-bound, fail-closed foreground readout.

R5 publication is version-bound but not task-bound: two different Task values
can consume the same ContextView coordinate. This facade makes the complete
readout key (published view coordinate, exact Task) and only exposes the
underlying R5 result if it is still the active publication.

WARM-only refresh is the only writer and may verify HACF bytes. Foreground
peek never does I/O, hashing, locking or an ECS_G encoding. A refresh revokes
old readout access before it starts; failure leaves the facade quarantined.
No semantic authority or current-file-freshness claim is implied.
"""
from __future__ import annotations

from threading import Lock

from elpis.runtime.context import ContextCoordinate

from .r5_warm_publication import PublishedExact, WarmExactPublication
from .task import Task


class TaskBoundPublication:
    """One-slot R5 facade that never answers another task's readout request."""

    def __init__(self):
        self._writer = Lock()
        self._inner = WarmExactPublication()
        self._issued: tuple[Task, PublishedExact] | None = None

    def refresh(self, root, manifest, view, authority, bindings, task: Task, **bounds) -> PublishedExact:
        """WARM ONLY: full source re-attestation, then one task-bound publication.

        Serializes the outer state update, including task binding. A failed
        refresh never leaves the previous readout accessible via this facade.
        """
        with self._writer:
            self._issued = None
            result = self._inner.refresh(root, manifest, view, authority, bindings, task, **bounds)
            if (type(task) is not Task or type(result) is not PublishedExact
                    or result.result.query != task or result.result.coordinate != result.coordinate):
                # Fail closed. This should be unreachable for the pinned R5
                # implementation, but must not publish if that contract drifts.
                raise RuntimeError("TASK_PUBLICATION_MISMATCH")
            self._issued = (task, result)
            return result

    def peek(self, coordinate: ContextCoordinate, task: Task) -> PublishedExact | None:
        """FOREGROUND: only a matching view AND exact task can use the result.

        Two reference reads may race with a refresh; mismatches refuse. A
        publication already read before revocation remains an immutable
        snapshot, as with the canonical ContextLane view contract.
        """
        if type(coordinate) is not ContextCoordinate or type(task) is not Task:
            return None
        issued = self._issued
        if issued is None or issued[0] != task or issued[1].coordinate != coordinate:
            return None
        current = self._inner.peek(coordinate)
        return current if current is issued[1] and current.result.query == task else None
