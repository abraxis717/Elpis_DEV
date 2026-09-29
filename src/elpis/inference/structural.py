"""Structural address proposals: the edge contract inference accepts from slow lanes.

An :class:`AddressProposal` is an already validated, digest-pinned value: which
source and corpus it came from, the context snapshot and query overlay it is
bound to, a route key, candidate objects and provenance digests. Inference
never parses or validates slow-lane artifacts itself. The adapters that
produce proposals from ingress exports and retrieval bundles live in
:mod:`elpis.runtime.edges`. Confidence never grants authority.
"""
from dataclasses import dataclass
import math

from .contracts import ProposalOnly, digest_value, identity, require

@dataclass(frozen=True)
class RouteRule:
    key: str
    banks: tuple[str,...]=()
    expert_families: tuple[str,...]=()


@dataclass(frozen=True)
class AddressProposal(ProposalOnly):
    source: str
    regex_result: str
    corpus: str
    context_snapshot: str
    query_overlay: str
    route_key: str
    banks: tuple[str,...]
    objects: tuple[str,...]
    expert_families: tuple[str,...]
    confidence: float | None
    provenance: tuple[str,...]

    def __post_init__(self):
        for d in (self.source,self.regex_result,self.corpus,self.context_snapshot,self.query_overlay): digest_value(d)
        require(self.confidence is None or
                (type(self.confidence) in (int,float) and type(self.confidence) is not bool and
                 math.isfinite(self.confidence) and 0<=self.confidence<=1))
        require(bool(self.route_key))
        for values in (self.banks,self.objects,self.expert_families,self.provenance):
            require(type(values) is tuple and all(type(x) is str and bool(x) for x in values))

    @property
    def digest(self): return identity('structural-address',self)
