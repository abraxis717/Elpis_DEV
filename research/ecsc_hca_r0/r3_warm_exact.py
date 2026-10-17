"""R3 DEV research-only WARM adapter for exact incremental ECS_C context.

Source authority: R2 source at Elpis_DEV cb530e7 (NOT canonical science).
The adapter consumes an immutable ContextView and caller-supplied TEST_ONLY
source/semantic grant authority. It never reads files or invokes HOT cognition.

Skipped views, a changed authority, or source-check changes without a published
successor REFUSE; a distinct ``open`` is required for a disclosed cold restart.
Positive salience is never a fact or an admitted semantic relation.
"""
from dataclasses import dataclass

from elpis.runtime.context import ContextView

from .baselines import Authority, Limits, Refusal, SourceCheck, require
from .r2_b0i import B0I
from .r2_common import VerifiedGraph
from .task import Task


@dataclass(frozen=True, init=False)
class ExactWarmContext:
    """Immutable per-scope WARM cache, bound to a verified published view.

    An object returned by ``open`` or ``advance`` is independently usable; the
    old instance is not modified if admission/readout fails. This is a caller-
    owned ephemeral cache, not continuity authority or a persistence store.
    """

    verified: VerifiedGraph
    task: Task
    checks: tuple[SourceCheck, ...]
    cache: B0I

    def __init__(self, *args, **kwargs):
        raise TypeError("Use ExactWarmContext.open or .advance")

    @classmethod
    def _issue(cls, verified, task, checks, cache):
        instance = object.__new__(cls)
        object.__setattr__(instance, "verified", verified)
        object.__setattr__(instance, "task", task)
        object.__setattr__(instance, "checks", checks)
        object.__setattr__(instance, "cache", cache)
        return instance

    @classmethod
    def open(cls, view: ContextView, authority: Authority, checks: tuple[SourceCheck, ...],
             task: Task, *, limits: Limits = Limits()):
        """Explicit cold construction. Called by WARM, never the HOT lane.

        The separate grants and checks are synthetic in R3; this does not
        provide real HACF source attestation or semantic qualification.
        """
        require(type(view) is ContextView and type(authority) is Authority
                and type(checks) is tuple and type(task) is Task and type(limits) is Limits,
                "INVALID_INPUT")
        verified = VerifiedGraph(view, authority, checks, limits)
        cache, result, info = B0I.start(verified, task)
        require(cache is not None, "CACHE_UNAVAILABLE")
        return cls._issue(verified, task, checks, cache), result, {
            **info, "lifetime": "EXPLICIT_COLD_OPEN",
            "graph_build_work": sum(n for _, n in verified.graph.build_operations),
            "view_version": view.version, "view_digest": view.digest,
        }

    def advance(self, view: ContextView, authority: Authority, checks: tuple[SourceCheck, ...],
                task: Task, *, limits: Limits | None = None):
        """Validate an identical or immediately succeeding view, then compute.

        No source checks may mutate under the SAME view coordinate. A change
        requires a separately published successor (and is then revalidated).
        An observer that skipped a version must disclose a cold open, never
        pretend that the cache consumed the missing event.
        """
        require(type(view) is ContextView and type(authority) is Authority
                and type(checks) is tuple and type(task) is Task, "INVALID_INPUT")
        limits = self.verified.graph.limits if limits is None else limits
        require(type(limits) is Limits, "INVALID_LIMIT")
        require(authority == self.verified.graph.authority and limits == self.verified.graph.limits,
                "AUTHORITY_CONTINUITY")
        require(task.scope == self.task.scope, "SCOPE_TRANSITION")
        old_view = self.verified.graph.view
        if view.digest == old_view.digest and view.version == old_view.version:
            require(view == old_view, "VIEW_MISMATCH")
            require(checks == self.checks, "SOURCE_REFRESH_REQUIRES_SUCCESSOR")
            verified = self.verified
            view_mode = "SAME_PUBLISHED_VIEW"
            graph_work = 0
        else:
            require(view.version == old_view.version + 1 and view.predecessor == old_view.digest,
                    "PREDECESSOR")
            verified = VerifiedGraph(view, authority, checks, limits, previous=self.verified)
            graph_work = sum(n for _, n in verified.graph.build_operations)
            view_mode = "VERIFIED_IMMEDIATE_SUCCESSOR"
        cache, result, info = self.cache.warm(verified, task)
        require(cache is not None, "CACHE_UNAVAILABLE")
        return type(self)._issue(verified, task, checks, cache), result, {
            **info, "lifetime": view_mode, "graph_build_work": graph_work,
            "view_version": view.version, "view_digest": view.digest,
        }
