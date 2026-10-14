"""The contextual (WARM) lane: a bounded, session-local active context, published as immutable versioned views.

Execution lanes (docs/ARCHITECTURE.md, "Execution lanes"): the geometric (HOT) lane computes; the contextual (WARM)
lane maintains the active relational context; HACF addresses existing resources; durability preserves explicitly
acknowledged state. This module is the contextual lane's mechanical foundation: ownership, handoff and isolation.

* **References.** HACF supplies admitted, provenance-bearing references (:class:`ContextReference`: a verified HACF
  chunk identity, the pinned corpus manifest it belongs to and the digest of the proposal or proof that named it).
  :func:`references_from_resolved` builds them from the in-place resolver
  (:func:`elpis.structure.retrieval.objects.resolve_chunks`): resources stay where they are, their bytes are read
  and verified transiently, and no text is retained here. :class:`ResolveObservation` runs that resolution on the
  WARM worker, so cold materialization finishes before its result can be part of a published view.
* **Axioms and relations.** An axiom is one contextual variable with its references, a task relevance (caller
  metadata, never truth) and an epistemic status: ``PROVISIONAL`` (retrieved evidence: retrieval similarity,
  co-occurrence or interaction establish nothing) or ``BINDING`` (an externally binding instruction or constraint
  the operator supplied). A relation is a typed hyperedge over axioms: ``CANDIDATE`` (proposed, never
  authoritative: :func:`candidate_relations`) or ``ADMITTED`` (only what the structure layer's explicit semantic
  projection verified against its epoch's own chunks: :func:`admitted_relations` takes its typed result).
* **Bounded handoff.** :meth:`ContextLane.submit` never blocks and never waits for the worker. The queue holds at
  most ``capacity`` observations; overflow policy (:data:`OVERFLOW_POLICY`): an observation with the key of a
  pending one of the same kind of obligation replaces it (``COALESCED``); when full, an optional observation evicts
  the oldest pending optional one, or is itself dropped when only required ones are pending (``DROPPED``); a
  required observation evicts the oldest pending optional one, or is refused at once (``CONTEXT_FULL``): a
  required observation is never silently dropped. Every drop is counted and carried by the next view.
* **Views.** A WARM worker thread folds a batch of observations into a successor of the current view
  (:func:`successor`, pure and deterministic) under strict bounds, verifies it and publishes it by one reference
  swap. :meth:`ContextLane.view` never waits: it returns the complete current view, which never changes after
  publication (frozen; its digest covers every field). Each observation applies all-or-nothing and fails alone (a
  refused materialization, a relation over an axiom not held, binding capacity): it publishes nothing of itself and
  never costs the others. A batch-level fault publishes nothing at all. Either way the published view stays
  exactly as it was until a successor is complete.
* **Separate coordinates.** The view is session-local and never persisted. It is not part of the durable state: a
  durable publication neither includes nor waits for it, a view update writes nothing, and a restart starts from
  the empty view.

Not implemented, and not claimed: no contextual dynamics law (Hypergraph Cellular Automata are an unqualified
candidate mechanism; nothing here infers, scores, propagates or promotes a relationship); no encoding of a view into
the geometric lane's K1 stimulus (a cognitive operation records the coordinate of the view it consumed, nothing
more); no semantic truth from retrieval; no conversation history (the lane keeps a bounded set, not a transcript).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import re
import threading
from typing import Callable

from .errors import CompositionError

__all__ = (
    "ADMITTED", "BINDING", "CANDIDATE", "ContextAxiom", "ContextCoordinate", "ContextLane", "ContextLimits",
    "ContextReference", "ContextRelation", "ContextStatus", "ContextView", "EMPTY_VIEW", "OVERFLOW_POLICY",
    "Observation", "PROVISIONAL", "ResolveObservation", "admitted_relations", "candidate_relations",
    "references_from_resolved", "successor",
)

PROVISIONAL, BINDING = "PROVISIONAL", "BINDING"
CANDIDATE, ADMITTED = "CANDIDATE", "ADMITTED"
ZERO = "0" * 64
OVERFLOW_POLICY = "COALESCE_BY_KEY; EVICT_OLDEST_OPTIONAL; DROP_OPTIONAL_COUNTED; REFUSE_REQUIRED"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_DOMAIN = b"elpis.runtime.context-view.r0\0"
_U32 = 1 << 32


def _refuse(code: str, detail: str) -> CompositionError:
    return CompositionError(code, detail)


def _digest(value, what: str) -> str:
    if type(value) is not str or not _HEX.match(value):
        raise _refuse("CONTEXT_INVALID", f"{what} must be a lowercase SHA-256 hex digest")
    return value


def _bounded(value, what: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise _refuse("CONTEXT_INVALID", f"{what} must be an integer in [{low}, {high}]")
    return value


@dataclass(frozen=True, order=True)
class ContextReference:
    """One admitted, provenance-bearing HACF reference: a verified chunk of a pinned corpus, and who named it."""

    chunk_digest: str
    corpus_digest: str
    provenance_digest: str

    def __post_init__(self):
        _digest(self.chunk_digest, "chunk digest")
        _digest(self.corpus_digest, "corpus manifest digest")
        _digest(self.provenance_digest, "provenance digest")


@dataclass(frozen=True)
class ContextAxiom:
    """One active contextual variable. ``key`` is its identity; ``relevance`` is caller metadata, never truth."""

    key: str
    references: tuple[ContextReference, ...]
    status: str = PROVISIONAL
    relevance: int = 0

    def __post_init__(self):
        _digest(self.key, "axiom key")
        if type(self.references) is not tuple or not all(type(r) is ContextReference for r in self.references):
            raise _refuse("CONTEXT_INVALID", "references must be a tuple of ContextReference")
        if self.status not in (PROVISIONAL, BINDING):
            raise _refuse("CONTEXT_INVALID", "axiom status is PROVISIONAL or BINDING")
        if self.status == PROVISIONAL and not self.references:
            raise _refuse("CONTEXT_INVALID", "a provisional axiom needs at least one reference")
        if tuple(sorted(set(self.references))) != self.references:
            raise _refuse("CONTEXT_INVALID", "references must be sorted and distinct")
        _bounded(self.relevance, "relevance", 0, _U32 - 1)


@dataclass(frozen=True)
class ContextRelation:
    """One typed hyperedge over axioms (ordered roles). ``ADMITTED`` only from a verified relation proof."""

    relation_type: int
    members: tuple[str, ...]
    provenance_digest: str
    status: str = CANDIDATE

    def __post_init__(self):
        _bounded(self.relation_type, "relation type", 1, _U32 - 1)
        if type(self.members) is not tuple or len(self.members) < 2 or len(set(self.members)) != len(self.members):
            raise _refuse("CONTEXT_INVALID", "a relation has at least two distinct members")
        for member in self.members:
            _digest(member, "relation member")
        _digest(self.provenance_digest, "relation provenance")
        if self.status not in (CANDIDATE, ADMITTED):
            raise _refuse("CONTEXT_INVALID", "relation status is CANDIDATE or ADMITTED")

    @property
    def key(self) -> tuple[int, tuple[str, ...]]:
        return self.relation_type, self.members


@dataclass(frozen=True)
class ContextLimits:
    """Strict bounds of one lane: queue capacity, view size, relation arity and references per axiom."""

    capacity: int = 64
    max_axioms: int = 256
    max_relations: int = 512
    max_arity: int = 8
    max_references: int = 16
    batch: int = 16

    def __post_init__(self):
        _bounded(self.capacity, "capacity", 1, 1 << 16)
        _bounded(self.max_axioms, "max_axioms", 1, 1 << 16)
        _bounded(self.max_relations, "max_relations", 0, 1 << 16)
        _bounded(self.max_arity, "max_arity", 2, 64)
        _bounded(self.max_references, "max_references", 1, 1024)
        _bounded(self.batch, "batch", 1, 1 << 16)


def _field(h, text: str) -> None:
    raw = text.encode("ascii")
    h.update(len(raw).to_bytes(4, "big") + raw)


def _view_digest(version: int, predecessor: str, through: int, dropped: int, axioms, relations) -> str:
    h = hashlib.sha256(_DOMAIN)
    h.update(version.to_bytes(8, "big") + through.to_bytes(8, "big") + dropped.to_bytes(8, "big"))
    _field(h, predecessor)
    h.update(len(axioms).to_bytes(4, "big"))
    for a in axioms:
        _field(h, a.key)
        _field(h, a.status)
        h.update(a.relevance.to_bytes(4, "big") + len(a.references).to_bytes(4, "big"))
        for r in a.references:
            _field(h, r.chunk_digest)
            _field(h, r.corpus_digest)
            _field(h, r.provenance_digest)
    h.update(len(relations).to_bytes(4, "big"))
    for r in relations:
        h.update(r.relation_type.to_bytes(4, "big") + len(r.members).to_bytes(4, "big"))
        for m in r.members:
            _field(h, m)
        _field(h, r.provenance_digest)
        _field(h, r.status)
    return h.hexdigest()


@dataclass(frozen=True)
class ContextCoordinate:
    """Which published view a cognitive operation consumed (version and digest); never a durable coordinate."""

    version: int
    digest: str


@dataclass(frozen=True)
class ContextView:
    """One published active context: immutable, versioned, chained to its predecessor by digest.

    ``through`` is the sequence of the last observation folded into it; ``dropped`` counts the optional observations
    the overflow policy dropped or evicted before it was built (cumulative). Build views with :func:`successor`.
    """

    version: int
    predecessor: str
    through: int
    dropped: int
    axioms: tuple[ContextAxiom, ...]
    relations: tuple[ContextRelation, ...]
    digest: str

    @property
    def coordinate(self) -> ContextCoordinate:
        return ContextCoordinate(self.version, self.digest)

    def axiom(self, key: str) -> ContextAxiom | None:
        for a in self.axioms:
            if a.key == key:
                return a
        return None

    def verify(self, limits: ContextLimits | None = None) -> None:
        """Every invariant a published view holds; raises ``CONTEXT_INVALID`` otherwise."""
        keys = [a.key for a in self.axioms]
        if keys != sorted(set(keys)):
            raise _refuse("CONTEXT_INVALID", "axioms must be sorted by key and distinct")
        rkeys = [r.key for r in self.relations]
        if rkeys != sorted(set(rkeys)):
            raise _refuse("CONTEXT_INVALID", "relations must be sorted and distinct")
        present = set(keys)
        if any(m not in present for r in self.relations for m in r.members):
            raise _refuse("CONTEXT_INVALID", "a relation names an axiom the view does not hold")
        if limits is not None and (len(self.axioms) > limits.max_axioms or len(self.relations) > limits.max_relations
                                   or any(len(r.members) > limits.max_arity for r in self.relations)
                                   or any(len(a.references) > limits.max_references for a in self.axioms)):
            raise _refuse("CONTEXT_INVALID", "the view exceeds its limits")
        if _view_digest(self.version, self.predecessor, self.through, self.dropped, self.axioms,
                        self.relations) != self.digest:
            raise _refuse("CONTEXT_INVALID", "the view digest does not cover its contents")


def _seal(version, predecessor, through, dropped, axioms, relations) -> ContextView:
    return ContextView(version, predecessor, through, dropped, axioms, relations,
                       _view_digest(version, predecessor, through, dropped, axioms, relations))


EMPTY_VIEW = _seal(0, ZERO, 0, 0, (), ())


# -- observations: the unit of the WARM handoff ---------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """One update handed to the lane: assert an axiom, assert a relation, or retract an axiom (with its relations).

    ``required``: an externally binding instruction or constraint; it is never silently dropped (refused at submit
    when it cannot be queued). A ``BINDING`` axiom is always required.
    """

    axiom: ContextAxiom | None = None
    relation: ContextRelation | None = None
    retract: str | None = None
    required: bool = False

    def __post_init__(self):
        given = [x is not None for x in (self.axiom, self.relation, self.retract)]
        if sum(given) != 1:
            raise _refuse("CONTEXT_INVALID", "an observation carries exactly one axiom, relation or retraction")
        if self.axiom is not None and type(self.axiom) is not ContextAxiom:
            raise _refuse("CONTEXT_INVALID", "axiom must be a ContextAxiom")
        if self.relation is not None and type(self.relation) is not ContextRelation:
            raise _refuse("CONTEXT_INVALID", "relation must be a ContextRelation")
        if self.retract is not None:
            _digest(self.retract, "retracted axiom key")
        if type(self.required) is not bool:
            raise _refuse("CONTEXT_INVALID", "required is a bool")
        if self.axiom is not None and self.axiom.status == BINDING and not self.required:
            raise _refuse("CONTEXT_INVALID", "a binding axiom is a required observation")

    @property
    def key(self):
        if self.axiom is not None:
            return ("axiom", self.axiom.key)
        if self.relation is not None:
            return ("relation", self.relation.key)
        return ("axiom", self.retract)

    def materialize(self) -> tuple["Observation", ...]:
        return (self,)


@dataclass(frozen=True)
class ResolveObservation:
    """Resolve HACF chunk claims in place on the WARM worker and assert one provisional axiom per verified chunk.

    The resolution (``resolve_chunks``: pinned manifest, root-confined reads, digest and chunk-identity checks,
    explicit bounds) runs before the successor view is built; a refusal fails the batch and publishes nothing.
    Optional: retrieved evidence is never an obligation.
    """

    corpus_root: object
    manifest: object
    claims: tuple
    provenance_digest: str
    relevance: int = 0
    source_paths: dict | None = None
    max_objects: int = 16
    max_text_bytes: int = 1 << 16
    max_document_bytes: int = 1 << 22
    required: bool = False

    def __post_init__(self):
        from elpis.structure.retrieval.objects import ChunkClaim
        if type(self.claims) is not tuple or not all(type(c) is ChunkClaim for c in self.claims):
            raise _refuse("CONTEXT_INVALID", "claims must be a tuple of ChunkClaim")
        _digest(self.provenance_digest, "provenance digest")
        _bounded(self.relevance, "relevance", 0, _U32 - 1)
        if self.required:
            raise _refuse("CONTEXT_INVALID", "retrieved evidence is never a required observation")

    @property
    def key(self):
        return ("resolve", self.provenance_digest, tuple(c.chunk_digest for c in self.claims))

    def materialize(self) -> tuple[Observation, ...]:
        from elpis.structure.retrieval.errors import RetrievalError
        from elpis.structure.retrieval.objects import resolve_chunks
        try:
            resolved, _omitted = resolve_chunks(
                self.corpus_root, self.manifest, self.claims, max_objects=self.max_objects,
                max_text_bytes=self.max_text_bytes, max_document_bytes=self.max_document_bytes,
                source_paths=self.source_paths)
        except RetrievalError as exc:
            raise _refuse("CONTEXT_MATERIALIZATION", f"{exc.code}: {exc}") from exc
        refs = references_from_resolved(resolved, self.manifest.digest, self.provenance_digest)
        return tuple(Observation(axiom=ContextAxiom(r.chunk_digest, (r,), PROVISIONAL, self.relevance))
                     for r in refs)


def references_from_resolved(resolved, corpus_digest: str, provenance_digest: str) -> tuple[ContextReference, ...]:
    """Admitted references from chunks the in-place resolver verified (``ResolvedChunk``); no text is kept."""
    from elpis.structure.retrieval.objects import ResolvedChunk
    if type(resolved) is not tuple or not all(type(c) is ResolvedChunk for c in resolved):
        raise _refuse("CONTEXT_INVALID", "references come only from verified ResolvedChunk values")
    return tuple(ContextReference(c.chunk_digest, corpus_digest, provenance_digest) for c in resolved)


def _relations(edges, status: str) -> tuple[ContextRelation, ...]:
    def text(v):
        return v.decode("ascii") if isinstance(v, bytes) else v
    return tuple(ContextRelation(int(e.edge_type), (text(e.subject_chunk_digest), text(e.object_chunk_digest)),
                                 text(e.provenance_digest), status) for e in edges)


def candidate_relations(edges) -> tuple[ContextRelation, ...]:
    """``CANDIDATE`` relations from HACF context-graph edges (``ContextEdge``: subject chunk -> object chunk, typed,
    with provenance). Members are the chunk-keyed axioms. A candidate is never authoritative."""
    return _relations(edges, CANDIDATE)


def admitted_relations(admission) -> tuple[ContextRelation, ...]:
    """``ADMITTED`` relations: exactly the edges of a :class:`~elpis.structure.retrieval.hacf.SemanticAdmission`, the
    result of the structure layer's explicit Track C v1 projection, which verified each relation proof natively
    against its epoch's own chunks. This lane never runs that construction step itself, and admission is never a
    caller's flag: anything else is refused."""
    from elpis.structure.retrieval.hacf import SemanticAdmission
    if type(admission) is not SemanticAdmission:
        raise _refuse("CONTEXT_INVALID", "admitted relations come only from a verified SemanticAdmission")
    return _relations(admission.edges, ADMITTED)


# -- the successor law: pure, deterministic, bounded ----------------------------------------------------------------

def _apply(axioms: dict, relations: dict, o: Observation, limits: ContextLimits) -> dict:
    """Fold one observation into the working sets (in place); returns the relations (a retraction rebuilds them)."""
    held = axioms.get(o.axiom.key if o.axiom is not None else o.retract) if o.relation is None else None
    if held is not None and held.status == BINDING and not o.required:
        raise _refuse("CONTEXT_INVALID", "a binding axiom is replaced or retracted only by a required observation")
    if o.axiom is not None:
        a = o.axiom
        if len(a.references) > limits.max_references:
            raise _refuse("CONTEXT_INVALID", "an axiom exceeds max_references")
        if a.status == BINDING and held is None and \
                sum(1 for x in axioms.values() if x.status == BINDING) >= limits.max_axioms:
            raise _refuse("CONTEXT_BINDING_CAPACITY", "binding axioms would exceed the context's capacity")
        axioms[a.key] = a
    elif o.relation is not None:
        r = o.relation
        if len(r.members) > limits.max_arity:
            raise _refuse("CONTEXT_INVALID", "a relation exceeds max_arity")
        if any(m not in axioms for m in r.members):
            raise _refuse("CONTEXT_INVALID", "a relation names an axiom the context does not hold")
        relations[r.key] = r
    else:
        axioms.pop(o.retract, None)
        relations = {k: r for k, r in relations.items() if o.retract not in r.members}
    return relations


def successor(view: ContextView, observations, limits: ContextLimits, *, through: int, dropped: int,
              rejected: list | None = None) -> ContextView:
    """The view after folding ``observations`` (in order) into ``view``, within ``limits``.

    Each entry is an :class:`Observation` or a tuple of them (one materialized observation), applied all-or-nothing.
    Assertions replace an axiom or relation of the same key; a retraction removes an axiom and every relation over
    it; a relation whose members are not all held is refused; only a required observation replaces or retracts a
    binding axiom; a binding axiom beyond ``max_axioms`` binding axioms is refused (``CONTEXT_BINDING_CAPACITY``):
    binding axioms are never evicted. A refused entry raises, or, with a ``rejected`` list, is recorded there as
    ``(index, error)`` and skipped (the others still apply). Then, over ``max_axioms``, provisional axioms are evicted
    lowest relevance first (then key), with their relations; over ``max_relations``, candidate relations are evicted
    before admitted ones (then by key). Nothing is inferred: no relation is ever created here.
    """
    axioms = {a.key: a for a in view.axioms}
    relations = {r.key: r for r in view.relations}
    for index, entry in enumerate(observations):
        group = entry if type(entry) is tuple else (entry,)
        saved = (dict(axioms), dict(relations))
        try:
            for o in group:
                relations = _apply(axioms, relations, o, limits)
        except CompositionError as exc:
            if rejected is None:
                raise
            axioms, relations = saved
            rejected.append((index, exc))
    if len(axioms) > limits.max_axioms:
        order = sorted((a for a in axioms.values() if a.status == PROVISIONAL), key=lambda a: (a.relevance, a.key))
        for a in order[:len(axioms) - limits.max_axioms]:
            del axioms[a.key]
        relations = {k: r for k, r in relations.items() if all(m in axioms for m in r.members)}
    if len(relations) > limits.max_relations:
        order = sorted(relations.values(), key=lambda r: (r.status == ADMITTED, r.key))
        for r in order[:len(relations) - limits.max_relations]:
            del relations[r.key]
    out = _seal(view.version + 1, view.digest, through, dropped, tuple(axioms[k] for k in sorted(axioms)),
                tuple(relations[k] for k in sorted(relations)))
    out.verify(limits)
    return out


# -- the lane --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextStatus:
    """The lane as a status read sees it (never waits for the worker)."""

    version: int
    pending: int
    submitted: int
    coalesced: int
    dropped: int
    refused: int
    published: int
    failed: int
    failed_batches: int
    failed_required: int
    last_failure: str | None
    running: bool
    held: bool


class ContextLane:
    """The contextual (WARM) lane of one session: a bounded queue, one worker thread, one published view.

    ``view()`` and ``submit()`` never wait for the worker. ``wait(sequence, timeout)`` is the caller's explicit
    choice to wait for an observation's outcome (never on the HOT lane). The lane holds only the current view;
    a superseded view lives exactly as long as a reader holds it.
    """

    def __init__(self, limits: ContextLimits | None = None):
        self.limits = ContextLimits() if limits is None else limits
        if type(self.limits) is not ContextLimits:
            raise TypeError("limits must be ContextLimits")
        self._lock = threading.Lock()
        self._work = threading.Condition(self._lock)
        self._done = threading.Condition(self._lock)
        self._queue: deque = deque()            # (sequence, observation)
        self._inflight: frozenset = frozenset()  # the sequences of the batch the worker is building
        self._view = EMPTY_VIEW
        self._sequence = 0
        self._outcomes: dict[int, str] = {}     # bounded (oldest first out); a forgotten outcome reads UNKNOWN
        self._counts = dict(submitted=0, coalesced=0, dropped=0, refused=0, published=0, failed=0, failed_batches=0,
                            failed_required=0)
        self._last_failure: str | None = None
        self._held = False
        self._fail_next: Callable[[], None] | None = None
        self._closed = False
        self._running = False
        self._thread = threading.Thread(target=self._worker, name="elpis-context", daemon=True)
        self._thread.start()

    # -- foreground: never waits for the worker --------------------------------------------------------------------
    def view(self) -> ContextView:
        """The current published view (one reference read: complete, immutable, never partially updated)."""
        return self._view

    def submit(self, observation) -> tuple[int, str]:
        """Hand one observation to the lane: ``(sequence, "QUEUED" | "COALESCED" | "DROPPED")`` at once.

        A required observation that cannot be queued is refused (``CONTEXT_FULL``); it is never dropped.
        """
        if type(observation) not in (Observation, ResolveObservation):
            raise TypeError("submit takes an Observation or a ResolveObservation")
        key = observation.key
        with self._lock:
            if self._closed:
                raise _refuse("CONTEXT_CLOSED", "the context lane is closed")
            same = next((i for i, (_, p) in enumerate(self._queue)
                         if p.key == key and p.required == observation.required), None)
            victim = None
            if same is None and len(self._queue) >= self.limits.capacity:
                victim = next((i for i, (_, p) in enumerate(self._queue) if not p.required), None)
                if victim is None and observation.required:
                    self._counts["refused"] += 1
                    raise _refuse("CONTEXT_FULL", "every queued observation is required; nothing was dropped")
            self._sequence += 1
            seq = self._sequence
            self._counts["submitted"] += 1
            if same is not None:          # the newer observation of the same key replaces the pending one
                replaced, _ = self._queue[same]
                self._queue[same] = (seq, observation)
                self._counts["coalesced"] += 1
                self._outcome(replaced, "COALESCED")
                self._work.notify()
                return seq, "COALESCED"
            if len(self._queue) >= self.limits.capacity:
                self._counts["dropped"] += 1
                if victim is None:        # only required observations are pending: this optional one is dropped
                    self._outcome(seq, "DROPPED")
                    return seq, "DROPPED"
                evicted, _ = self._queue[victim]
                del self._queue[victim]
                self._outcome(evicted, "DROPPED")
            self._queue.append((seq, observation))
            self._work.notify()
            return seq, "QUEUED"

    def status(self) -> ContextStatus:
        with self._lock:
            return ContextStatus(self._view.version, len(self._queue), self._counts["submitted"],
                                 self._counts["coalesced"], self._counts["dropped"], self._counts["refused"],
                                 self._counts["published"], self._counts["failed"], self._counts["failed_batches"],
                                 self._counts["failed_required"], self._last_failure, self._running, self._held)

    # -- explicit waits (never on the HOT lane) ----------------------------------------------------------------------
    def wait(self, sequence: int, timeout: float | None = None) -> str:
        """The outcome of observation ``sequence``: ``APPLIED`` (a published view folded it), ``COALESCED`` (a newer
        one of its key replaced it), ``DROPPED``, ``FAILED`` (its batch published nothing), ``ABANDONED`` (closed
        first), ``PENDING`` after ``timeout``, or ``UNKNOWN`` (resolved too long ago for its outcome to be kept)."""
        with self._lock:
            if type(sequence) is not int or not 0 < sequence <= self._sequence:
                raise _refuse("CONTEXT_INVALID", "no such observation")

            def outcome():
                if sequence in self._outcomes:
                    return self._outcomes[sequence]
                if sequence in self._inflight or any(s == sequence for s, _ in self._queue):
                    return None
                return "UNKNOWN"
            self._done.wait_for(lambda: outcome() is not None, timeout)
            return outcome() or "PENDING"

    def close(self) -> None:
        """Stop the worker; observations still queued are abandoned (the published view stays readable)."""
        with self._lock:
            self._closed = True
            self._work.notify_all()
        self._thread.join()
        with self._lock:
            for seq, _ in self._queue:
                self._outcome(seq, "ABANDONED")
            self._queue.clear()
            self._done.notify_all()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- testing: stall the worker or fail its next batch ----------------------------------------------------------
    def testing_hold(self, hold: bool) -> None:
        with self._lock:
            self._held = bool(hold)
            self._work.notify_all()

    def testing_fail_next(self, fault: Callable[[], None]) -> None:
        """Run ``fault`` inside the next batch, after its successor was built and before publication."""
        with self._lock:
            self._fail_next = fault

    # -- the WARM worker ---------------------------------------------------------------------------------------------
    def _outcome(self, sequence: int, outcome: str) -> None:
        self._outcomes[sequence] = outcome
        while len(self._outcomes) > 4 * (self.limits.capacity + self.limits.batch):
            del self._outcomes[next(iter(self._outcomes))]
        self._done.notify_all()

    def _worker(self) -> None:
        while True:
            with self._lock:
                self._work.wait_for(lambda: self._closed or (self._queue and not self._held))
                if self._closed:
                    return
                batch = [self._queue.popleft() for _ in range(min(self.limits.batch, len(self._queue)))]
                self._inflight = frozenset(seq for seq, _ in batch)
                fault, self._fail_next = self._fail_next, None
                base, dropped = self._view, self._counts["dropped"]
                self._running = True
            # Cold materialization (in-place HACF resolution) happens here, before anything is published. Each
            # observation fails alone: its refusal publishes nothing of it and never costs the others.
            failed: dict[int, BaseException] = {}
            groups, owners = [], []
            for seq, obs in batch:
                try:
                    groups.append(tuple(obs.materialize()))
                    owners.append(seq)
                except Exception as exc:
                    failed[seq] = exc
            rejected: list = []
            view = None
            try:
                if groups:
                    view = successor(base, groups, self.limits, through=max(self._inflight), dropped=dropped,
                                     rejected=rejected)
                    if (view.axioms, view.relations, view.dropped) == (base.axioms, base.relations, base.dropped):
                        view = None   # nothing changed (all refused, or already held): no new version
                if fault is not None:
                    fault()
            except Exception as exc:   # a batch-level fault publishes nothing; the published view is untouched
                view, failed = None, {seq: exc for seq, _ in batch}
                with self._lock:
                    self._counts["failed_batches"] += 1
            for index, exc in rejected:
                failed[owners[index]] = exc
            with self._lock:
                if view is not None:
                    self._view = view      # the one publication: a reference swap
                    self._counts["published"] += 1
                self._running, self._inflight = False, frozenset()
                for seq, obs in batch:
                    if seq in failed:
                        self._counts["failed"] += 1
                        self._counts["failed_required"] += int(obs.required)
                        exc = failed[seq]
                        self._last_failure = getattr(exc, "code", type(exc).__name__)
                        self._outcome(seq, "FAILED")
                    else:
                        self._outcome(seq, "APPLIED")
            # Keep no superseded view alive between batches: a retired view lives only as long as a reader holds it.
            base = view = batch = groups = failed = rejected = None
