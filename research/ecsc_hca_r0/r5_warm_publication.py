"""ECS_C R5 research-only: fail-closed WARM attestation and lock-free context readout.

Only ``refresh`` may touch HACF or execute B0I, and it is a WARM-only call.
``peek`` is a single reference read and a coordinate comparison: no file I/O,
no cold materialization, no blocking on a WARM worker, no ECS_G input encoding.

An attested publication describes bytes as verified *at refresh time*, not
continuing filesystem validity. A detected failure withdraws the current
publication, but it cannot prevent filesystem changes between refreshes.
The supplied manifest, scope grants and semantic authority remain external
TEST_ONLY inputs; this is not semantic admission or scientific qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from elpis.runtime.context import ContextCoordinate, ContextView
from elpis.structure.retrieval.objects import CorpusManifest

from .baselines import Authority, Limits, require
from .r3_warm_exact import ExactWarmContext
from .r4_hacf_sources import HACFBinding, advance_verified_hacf, open_verified_hacf
from .task import PredictionResult, Task


@dataclass(frozen=True)
class PublishedExact:
    """Immutable pointer to one completed, provenance-bearing research readout."""

    coordinate: ContextCoordinate
    evidence_digest: str
    result: PredictionResult
    mode: str
    # Deliberately no source text, private mutable buffers or Python dicts.


class WarmExactPublication:
    """One session-local publication slot; no history, writer runs explicitly on WARM.

    A single WARM lock serializes refreshes; the foreground never acquires it.
    Source/policy errors quarantine an earlier publication rather than leaving
    an answer available after the publisher *knows* verification failed. The
    old ExactWarmContext is retained privately only for an exact same-authority
    retry (the failed attempt cannot mutate it).
    """

    def __init__(self, *, limits: Limits = Limits()):
        require(type(limits) is Limits, "INVALID_LIMIT")
        self._limits = limits
        self._writer = Lock()
        self._state: ExactWarmContext | None = None
        self._published: PublishedExact | None = None

    def peek(self, coordinate: ContextCoordinate) -> PublishedExact | None:
        """Nonblocking foreground lookup. Caller supplies its latest ContextLane coordinate.

        Does NOT reverify source bytes or claim to know source freshness. The
        calling foreground must not use this as a semantic or ECS_G authority.
        """
        if type(coordinate) is not ContextCoordinate:
            return None
        published = self._published  # atomic object-reference read in CPython
        return published if published is not None and published.coordinate == coordinate else None

    def refresh(self, root, manifest: CorpusManifest, view: ContextView,
                authority: Authority, bindings: tuple[HACFBinding, ...], task: Task,
                *, max_document_bytes: int = 1 << 20,
                max_total_document_bytes: int = 4 << 20) -> PublishedExact:
        """WARM ONLY: reverify sources and publish on all-or-nothing success.

        No silent switch of authority, limits or context lineage. Restart at a
        new authority epoch requires an explicitly new publisher. Refreshes
        for a skipped view fail rather than claiming that omitted versions
        were consumed. The result only represents the verified snapshot.
        """
        with self._writer:
            try:
                if self._state is None:
                    state, result, info = open_verified_hacf(
                        root, manifest, view, authority, bindings, task,
                        limits=self._limits, max_document_bytes=max_document_bytes,
                        max_total_document_bytes=max_total_document_bytes,
                    )
                else:
                    state, result, info = advance_verified_hacf(
                        self._state, root, manifest, view, authority, bindings, task,
                        max_document_bytes=max_document_bytes,
                        max_total_document_bytes=max_total_document_bytes,
                    )
                require(type(result) is PredictionResult and result.coordinate == view.coordinate
                        and result.evidence_digest == state.verified.graph.evidence_digest,
                        "RESULT_COORDINATE")
                issued = PublishedExact(view.coordinate, result.evidence_digest, result, info['lifetime'])
            except Exception:
                # Even a badly formed refresh may indicate that a prior result
                # is no longer safe to expose. Keep the old state *private* but
                # withdraw its readout; never publish a partial successor.
                self._published = None
                raise
            self._state = state
            self._published = issued  # one publication; no lock on the read path
            return issued
