"""Gate C: the contextual (WARM) lane's isolation (elpis.runtime.context; docs/ARCHITECTURE.md "Execution lanes").

The lane keeps a bounded, session-local active context and publishes it as immutable versioned views. These laws are
mechanics only: ownership, handoff and isolation. Nothing here is a contextual dynamics law (no HCA), a semantic claim
or a cognition claim; the K1 fixture map is TEST_ONLY TRAINING=NONE SEMANTICS=NONE.

* readers (many threads) always see a complete, valid, never-changing view while successors are built and published;
* a stalled or failing WARM worker never blocks the foreground, including managed cognition that consumes the view;
* the handoff is strictly bounded, with the explicit overflow policy (coalesce, evict or drop optional, refuse
  required: never silently drop a required observation);
* cold materialization (in-place HACF resolution) runs on the worker and finishes before publication; a refused
  resolution (changed, absent or escaping source) publishes nothing and leaves the published view intact;
* superseded views are retired when no reader holds them, and memory stays bounded under any number of observations;
* the active view and the durable state are separate coordinates.
"""
from __future__ import annotations

import gc
import hashlib
import json
import threading
import weakref

import pytest

from elpis.runtime import Runtime
from elpis.runtime.cognition import LearnRequest, QueryRequest
from elpis.runtime.composition import CompositionError
from elpis.runtime.context import (
    ADMITTED, BINDING, CANDIDATE, EMPTY_VIEW, PROVISIONAL, ContextAxiom, ContextLane, ContextLimits,
    ContextReference, ContextRelation, ContextView, Observation, ResolveObservation, admitted_relations,
    candidate_relations, references_from_resolved, successor,
)
from elpis.structure.retrieval.objects import ChunkClaim, CorpusManifest, chunk_identity, normalize

from ._turn_fixtures import LEARN, ByteTokens, FixtureMap, admitted
from .test_codec_ecs_turn import _config, k1, world  # noqa: F401 (the k1 fixture)

PROFILE = "ab" * 32
PROVENANCE = "cd" * 32


def _hex(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _ref(label: str) -> ContextReference:
    return ContextReference(_hex("chunk " + label), _hex("corpus"), PROVENANCE)


def _axiom(label: str, relevance: int = 0) -> Observation:
    return Observation(axiom=ContextAxiom(_hex("chunk " + label), (_ref(label),), PROVISIONAL, relevance))


def _binding(label: str) -> Observation:
    return Observation(axiom=ContextAxiom(_hex("binding " + label), (), BINDING), required=True)


def _relation(a: str, b: str, kind: int = 7) -> Observation:
    return Observation(relation=ContextRelation(kind, (_hex("chunk " + a), _hex("chunk " + b)), PROVENANCE))


def _settle(lane: ContextLane, sequence: int) -> str:
    return lane.wait(sequence, 30)


# -- views ----------------------------------------------------------------------------------------------------------

def test_a_view_is_complete_immutable_versioned_and_chained():
    with ContextLane() as lane:
        first = lane.view()
        assert first is EMPTY_VIEW and first.version == 0
        seq, outcome = lane.submit(_axiom("a"))
        assert outcome == "QUEUED" and _settle(lane, seq) == "APPLIED"
        second = lane.view()
        assert (second.version, second.predecessor) == (1, first.digest) and second.axiom(_hex("chunk a"))
        for seq in (lane.submit(_axiom("b"))[0], lane.submit(_relation("a", "b"))[0]):
            assert _settle(lane, seq) == "APPLIED"
        third = lane.view()
        assert third.version > second.version and len(third.relations) == 1
        # Newly observed information never mutates a published view: the older ones still verify, unchanged.
        for view in (first, second, third):
            view.verify(lane.limits)
        assert [a.key for a in second.axioms] == [_hex("chunk a")]
        with pytest.raises(AttributeError):
            third.version = 99   # frozen
        forged = ContextView(third.version, third.predecessor, third.through, third.dropped, third.axioms[:1], (),
                             third.digest)
        with pytest.raises(CompositionError, match="CONTEXT_INVALID"):
            forged.verify()


def test_concurrent_readers_only_ever_see_complete_valid_views():
    limits = ContextLimits(capacity=32, max_axioms=64, max_relations=64, batch=4)
    errors, seen = [], []
    stop = threading.Event()

    def reader():
        last, versions = -1, set()
        while not stop.is_set():
            view = None
            try:
                view = lane.view()
                view.verify(limits)            # the digest covers every field: no partial mixture verifies
                assert view.version >= last    # publication is monotonic for every reader
                last = view.version
                versions.add(view.version)
            except Exception as exc:   # pragma: no cover (the assertion below reports it)
                errors.append((view, exc))
                return
        seen.append(versions)

    with ContextLane(limits) as lane:
        readers = [threading.Thread(target=reader) for _ in range(4)]
        for r in readers:
            r.start()
        last = 0
        for n in range(400):
            last = lane.submit(_axiom(f"x{n % 90}", n))[0]
            if n % 3 == 2:
                lane.submit(_relation(f"x{n % 90}", f"x{(n - 1) % 90}"))
            if n % 8 == 7:                           # pace the writer so the readers race many publications
                assert _settle(lane, last) in ("APPLIED", "COALESCED", "FAILED", "DROPPED")
        stop.set()
        for r in readers:
            r.join()
        final = lane.view()
    assert not errors, errors[:3]
    assert final.version > 10 and len(final.axioms) <= limits.max_axioms
    assert all(len(v) > 1 for v in seen)   # every reader observed several publications


# -- bounded handoff and a stalled worker ----------------------------------------------------------------------------

def test_the_handoff_is_bounded_with_an_explicit_overflow_policy():
    limits = ContextLimits(capacity=4, max_axioms=64, batch=4)
    with ContextLane(limits) as lane:
        lane.testing_hold(True)                       # the WARM consumer is stalled
        assert [lane.submit(_axiom(f"o{n}"))[1] for n in range(4)] == ["QUEUED"] * 4
        assert lane.status().pending == 4
        # Same key: the newer replaces the pending one (coalesced), never growing the queue.
        coalesced, outcome = lane.submit(_axiom("o1", 5))
        assert outcome == "COALESCED" and lane.status().pending == 4
        # Full: a required observation evicts the oldest optional one (counted) and is queued.
        required, outcome = lane.submit(_binding("b0"))
        assert outcome == "QUEUED" and lane.status().dropped == 1
        # Full: an optional observation evicts the oldest optional one.
        assert lane.submit(_axiom("o9"))[1] == "QUEUED" and lane.status().dropped == 2
        for n in range(1, 4):
            assert lane.submit(_binding(f"b{n}"))[1] == "QUEUED"
        assert lane.status().pending == 4 and lane.status().dropped == 5
        # Only required observations pending: an optional one is dropped (counted), a required one is refused.
        dropped, outcome = lane.submit(_axiom("late"))
        assert outcome == "DROPPED" and lane.wait(dropped, 0) == "DROPPED"
        with pytest.raises(CompositionError, match="CONTEXT_FULL"):
            lane.submit(_binding("refused"))
        status = lane.status()
        assert (status.pending, status.refused, status.dropped) == (4, 1, 6)
        lane.testing_hold(False)
        assert _settle(lane, required) == "APPLIED"
        view = lane.view()
        assert {a.key for a in view.axioms} == {_hex(f"binding b{n}") for n in range(4)}
        assert view.dropped == 6                      # the loss is carried by the next view, never hidden


def test_a_stalled_worker_never_blocks_managed_cognition_that_consumes_the_view(k1, tmp_path):   # noqa: F811
    learn = LearnRequest("hot", ByteTokens(), admitted(FixtureMap()), LEARN)
    query = QueryRequest("query", ByteTokens(), admitted(FixtureMap()))
    with ContextLane(ContextLimits(capacity=8)) as lane, world(k1, max_rows=8) as state, \
            Runtime(_config(tmp_path / "c")) as runtime:
        runtime.anchor_cognition(state)
        assert _settle(lane, lane.submit(_axiom("seed"))[0]) == "APPLIED"
        published = lane.view()
        lane.testing_hold(True)
        for n in range(120):
            lane.submit(_axiom(f"pending {n}"))      # never blocks: the stalled queue overflows by policy
            result = runtime.run_learn(state, learn, context=lane)
            assert result.context == published.coordinate, n   # the published view, read without waiting
            if n % 10 == 0:
                assert runtime.run_query(state, query, context=lane).context == published.coordinate
        assert lane.status().pending == 8 and lane.status().dropped == 120 - 8
        assert lane.view() is published
        lane.testing_hold(False)
        last = lane.submit(_axiom("after"))[0]
        assert _settle(lane, last) == "APPLIED"
        assert runtime.run_learn(state, learn, context=lane).context == lane.view().coordinate != published.coordinate
        # A pinned view is consumed exactly as given.
        assert runtime.run_query(state, query, context=published).context == published.coordinate


def test_a_failed_batch_publishes_nothing_and_the_published_view_stays_intact():
    with ContextLane() as lane:
        assert _settle(lane, lane.submit(_axiom("kept"))[0]) == "APPLIED"
        published = lane.view()

        def fault():
            raise CompositionError("CONTEXT_TESTING_FAULT", "injected after the successor was built")
        lane.testing_fail_next(fault)
        seq = lane.submit(_binding("lost"))[0]
        assert _settle(lane, seq) == "FAILED"
        status = lane.status()
        assert (status.failed_batches, status.failed_required, status.last_failure) == (1, 1, "CONTEXT_TESTING_FAULT")
        assert lane.view() is published               # a failed promotion corrupts and invalidates nothing
        published.verify()
        # A relation over an axiom the context does not hold fails alone: nothing is published.
        assert _settle(lane, lane.submit(_relation("kept", "absent"))[0]) == "FAILED"
        assert lane.view() is published and lane.status().failed == 2
        # ... and never costs an unrelated observation of the same batch.
        lane.testing_hold(True)
        bad = lane.submit(_relation("kept", "absent again"))[0]
        good = lane.submit(_axiom("next"))[0]
        lane.testing_hold(False)
        assert (_settle(lane, bad), _settle(lane, good)) == ("FAILED", "APPLIED")
        assert lane.view().predecessor == published.digest and lane.view().axiom(_hex("chunk next"))


def test_binding_axioms_are_never_evicted_downgraded_or_silently_dropped():
    limits = ContextLimits(capacity=16, max_axioms=4, batch=16)
    with ContextLane(limits) as lane:
        for n in range(3):
            lane.submit(_binding(f"b{n}"))
        last = 0
        for n in range(10):
            last = lane.submit(_axiom(f"p{n}", relevance=n))[0]
        assert _settle(lane, last) == "APPLIED"
        view = lane.view()
        kinds = sorted(a.status for a in view.axioms)
        assert kinds == [BINDING] * 3 + [PROVISIONAL] and view.axioms   # provisional ones evicted, binding kept
        assert view.axiom(_hex("chunk p9")) is not None                  # the most relevant provisional one stays
        # An optional observation may neither retract nor replace a binding axiom.
        key = _hex("binding b0")
        assert _settle(lane, lane.submit(Observation(retract=key))[0]) == "FAILED"
        assert _settle(lane, lane.submit(Observation(axiom=ContextAxiom(key, (_ref("x"),))))[0]) == "FAILED"
        assert lane.view().axiom(key).status == BINDING
        # More binding axioms than capacity: refused as a batch (counted, visible), never evicted.
        lane.testing_hold(True)                      # one batch of three
        for n in range(3, 6):
            last = lane.submit(_binding(f"b{n}"))[0]
        lane.testing_hold(False)
        assert _settle(lane, last) == "FAILED" and lane.status().last_failure == "CONTEXT_BINDING_CAPACITY"
        # The fourth binding axiom fits (a provisional one is evicted for it); the fifth and sixth are refused,
        # counted and visible to their submitters, never evicted or dropped silently.
        view = lane.view()
        assert lane.status().failed_required == 2 and view.axiom(_hex("binding b3")).status == BINDING
        assert view.axiom(_hex("binding b4")) is None and view.axiom(_hex("binding b5")) is None
        assert sorted(a.status for a in view.axioms) == [BINDING] * 4
        # A required retraction is honoured.
        assert _settle(lane, lane.submit(Observation(retract=key, required=True))[0]) == "APPLIED"
        assert lane.view().axiom(key) is None


# -- in-place HACF resolution on the WARM worker ---------------------------------------------------------------------

def _corpus(root, texts):
    """A pinned corpus manifest over documents that stay where they are (``root/source``), and one claim each."""
    (root / "source").mkdir(parents=True)
    documents, claims, paths = [], [], {}
    for label, text in texts.items():
        data = text.encode()
        (root / "source" / f"{label}.txt").write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        documents.append({"digest": digest, "size_bytes": len(data), "chunk_count": 1,
                          "chunk_profile_digest": PROFILE, "media_type": "text/plain"})
        norm = hashlib.sha256(normalize(data)).hexdigest()
        claims.append(ChunkClaim(chunk_identity(digest, 0, 0, len(data), norm, PROFILE), digest, 0, 0, len(data)))
        paths[digest] = f"source/{label}.txt"
    raw = json.dumps({"documents": documents}).encode()
    return CorpusManifest.verified(raw, expected_digest=hashlib.sha256(raw).hexdigest()), tuple(claims), paths


def test_cold_materialization_runs_on_the_worker_before_publication(tmp_path, monkeypatch):
    import elpis.structure.retrieval.objects as objects
    manifest, claims, paths = _corpus(tmp_path, {"arch": "Arch Linux primitive.", "rule": "A second resource."})
    before = sorted(p.name for p in tmp_path.rglob("*"))
    threads = []
    resolve = objects.resolve_chunks

    def traced(*args, **kwargs):
        threads.append(threading.current_thread().name)
        return resolve(*args, **kwargs)
    monkeypatch.setattr(objects, "resolve_chunks", traced)
    with ContextLane() as lane:
        observation = ResolveObservation(tmp_path, manifest, claims, PROVENANCE, relevance=3, source_paths=paths)
        assert _settle(lane, lane.submit(observation)[0]) == "APPLIED"
        view = lane.view()
    assert threads == ["elpis-context"]               # never the foreground thread
    assert [a.key for a in view.axioms] == sorted(c.chunk_digest for c in claims)
    for axiom in view.axioms:
        (ref,) = axiom.references
        assert (ref.corpus_digest, ref.provenance_digest, axiom.status) == (manifest.digest, PROVENANCE, PROVISIONAL)
        assert not hasattr(axiom, "text")             # the lane keeps references, never a copy of the resource
    # The resources stayed where they are: nothing was created, copied or moved.
    assert sorted(p.name for p in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("damage", ["changed", "absent", "symlink", "escape"])
def test_a_refused_resolution_publishes_nothing(tmp_path, damage):
    manifest, claims, paths = _corpus(tmp_path / "root", {"arch": "Arch Linux primitive."})
    source = tmp_path / "root" / "source" / "arch.txt"
    outside = tmp_path / "outside.txt"
    outside.write_bytes(source.read_bytes())
    if damage == "changed":
        source.write_bytes(b"Arch Linux primitivE.")
    elif damage == "absent":
        source.unlink()
    elif damage == "symlink":
        source.unlink()
        source.symlink_to(outside)
    else:
        paths = {d: "../outside.txt" for d in paths}
    with ContextLane() as lane:
        published = lane.view()
        if damage == "escape":
            with pytest.raises(CompositionError, match="CONTEXT_MATERIALIZATION"):
                ResolveObservation(tmp_path / "root", manifest, claims, PROVENANCE, source_paths=paths).materialize()
        seq = lane.submit(ResolveObservation(tmp_path / "root", manifest, claims, PROVENANCE, source_paths=paths))[0]
        assert _settle(lane, seq) == "FAILED"
        assert lane.status().last_failure == "CONTEXT_MATERIALIZATION"
        assert lane.view() is published


def test_references_and_relations_keep_their_epistemic_status():
    with pytest.raises(CompositionError, match="CONTEXT_INVALID"):
        references_from_resolved(({"chunk_digest": "0" * 64},), _hex("corpus"), PROVENANCE)   # unverified input
    with pytest.raises(CompositionError, match="CONTEXT_INVALID"):
        admitted_relations(object())                 # admission is the native verification, never a flag
    with pytest.raises(CompositionError, match="CONTEXT_INVALID"):
        admitted_relations(())                       # a bare tuple of edges is not an admission

    class Edge:   # the shape of a HACF ContextEdge
        subject_chunk_digest, object_chunk_digest = _hex("chunk a").encode(), _hex("chunk b").encode()
        provenance_digest, edge_type, authority = PROVENANCE.encode(), 0x53430103, 3
    (relation,) = candidate_relations((Edge(),))
    assert relation.status == CANDIDATE and relation.members == (_hex("chunk a"), _hex("chunk b"))
    assert ADMITTED != CANDIDATE
    # Retrieved evidence is never a required observation; a binding axiom always is.
    with pytest.raises(CompositionError, match="CONTEXT_INVALID"):
        ResolveObservation(".", None, (), PROVENANCE, required=True)
    with pytest.raises(CompositionError, match="CONTEXT_INVALID"):
        Observation(axiom=ContextAxiom(_hex("x"), (), BINDING))
    # A provisional axiom is grounded in at least one reference.
    with pytest.raises(CompositionError, match="CONTEXT_INVALID"):
        ContextAxiom(_hex("x"), ())


# -- lifetime, retirement and bounded memory -------------------------------------------------------------------------

def test_superseded_views_retire_when_unread_and_memory_stays_bounded():
    limits = ContextLimits(capacity=16, max_axioms=32, max_relations=16, batch=8)
    with ContextLane(limits) as lane:
        assert _settle(lane, lane.submit(_axiom("held"))[0]) == "APPLIED"
        held = lane.view()                           # a reader keeps this one
        assert _settle(lane, lane.submit(_axiom("dropped"))[0]) == "APPLIED"
        retired = weakref.ref(lane.view())
        last = 0
        for n in range(5000):
            last = lane.submit(_axiom(f"m{n}", n % 97))[0]
            if n % 5 == 0:
                lane.submit(_relation(f"m{n}", f"m{n - 1}"))
            status = lane.status()
            assert status.pending <= limits.capacity
        _settle(lane, last)
        gc.collect()
        assert retired() is None                     # no reader: the lane does not keep superseded views
        held.verify(limits)                          # a reader's view stays complete and valid
        view = lane.view()
        assert len(view.axioms) <= limits.max_axioms and len(view.relations) <= limits.max_relations
        assert len(lane._outcomes) <= 4 * (limits.capacity + limits.batch)
        assert lane.wait(1, 0) == "UNKNOWN"           # an outcome too old to keep is reported as unknown


def test_successor_is_pure_and_deterministic():
    limits = ContextLimits(max_axioms=8)
    observations = [_axiom("a", 2), _axiom("b", 1), _relation("a", "b")]
    one = successor(EMPTY_VIEW, observations, limits, through=3, dropped=0)
    two = successor(EMPTY_VIEW, list(observations), limits, through=3, dropped=0)
    assert one == two and one.digest == two.digest and EMPTY_VIEW.version == 0 and not EMPTY_VIEW.axioms


# -- separate coordinates ----------------------------------------------------------------------------------------------

def test_the_active_view_and_the_durable_state_are_separate_coordinates(k1, tmp_path):   # noqa: F811
    learn = LearnRequest("learn", ByteTokens(), admitted(FixtureMap()), LEARN)
    with world(k1, max_rows=8) as state:
        with ContextLane() as lane, Runtime(_config(tmp_path / "c", testing=True)) as runtime:
            runtime.anchor_cognition(state)
            assert _settle(lane, lane.submit(_axiom("a"))[0]) == "APPLIED"
            result = runtime.run_learn(state, learn, context=lane)
            durable = runtime.continuity.snapshot()
            io = runtime.continuity.testing_counters(reset=True)
            for n in range(20):                       # WARM updates write nothing durable
                lane.submit(_axiom(f"w{n}"))
            assert _settle(lane, lane.submit(_axiom("z"))[0]) == "APPLIED"
            assert runtime.continuity.snapshot() == durable
            assert all(v == 0 for v in runtime.continuity.testing_counters().values()), io
            ticket = runtime.publish_durable(state)   # durable publication neither includes nor waits for the view
            assert ticket.acknowledged and not hasattr(ticket, "context")
            assert result.context.version == 1 and lane.view().version > 1
        with ContextLane() as fresh:                  # a restart starts from the empty view
            assert fresh.view() is EMPTY_VIEW
