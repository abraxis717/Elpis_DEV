"""Gate F: HACF addresses existing resources in place; repeated epochs and contextual updates grow nothing.

Resources stay where they are (``documents_in_place`` reads them once through ``RootCapability`` for a bounded,
volatile retrieval epoch; resolution re-reads and re-verifies the originals with ``source_paths``). Repeated retrieval
epochs and contextual updates create no corpus file, no copied document blob, no SQLite/WAL file, no FMS cold
directory, no semantic artifact and no sidecar. Changed, absent, escaping and symlinked sources, invalid provenance and
exhausted budgets fail closed. Retrieval never waits for vector residency unless the epoch's owner chose to (bridge
ABI v3). The fixture embedding is deterministic and hash-derived: it is not evidence of semantic retrieval.
"""
from __future__ import annotations

import hashlib
import os

import pytest

from elpis.runtime.composition import CompositionError
from elpis.runtime.context import ContextLane, ContextReference, ResolveObservation
from elpis.structure.retrieval.errors import HybridRetrievalError
from elpis.structure.retrieval.hacf import RetrievalLibrary, build_corpus_and_index, hybrid_retrieve
from elpis.structure.retrieval.objects import (
    ChunkClaim, CorpusManifest, ObjectResolutionError, SourceBinding, documents_in_place, resolve_chunks,
)

from ..conftest import require_native_library
from ..structure.native_bridge_fixture import pin_bridge

# Single-paragraph sources: the native chunker keeps each whole document as chunk 0.
SOURCES = {
    "arch.md": "Arch Linux primitive: pacman installs packages from the synced repositories.",
    "build.md": "The build uses CMake and ctest; the runtime crate is built by cargo.",
    "rules.md": "Durable publication is explicit; ordinary inference never waits for it.",
}
BINDINGS = tuple(SourceBinding(name.split(".")[0], f"workspace/{name}", "elpis.docs", "reference")
                 for name in sorted(SOURCES))
LIMITS = dict(max_document_bytes=1 << 16, max_total_bytes=1 << 18)
PROVENANCE = "ef" * 32


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    path, root, authority = pin_bridge(require_native_library("elpis_retrieval_bridge"),
                                       tmp_path_factory.mktemp("hacf-in-place-lib"), "elpis_retrieval_bridge")
    return RetrievalLibrary(path, root=root, authority=authority)


@pytest.fixture
def workstation(tmp_path):
    """Existing resources on a workstation: the only files this test may ever see."""
    (tmp_path / "workspace").mkdir()
    for name, text in SOURCES.items():
        (tmp_path / "workspace" / name).write_text(text)
    return tmp_path


def _tree(root):
    """Every path under root with its size and inode: any created, copied or replaced file shows."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            st = os.lstat(path)
            out.append((os.path.relpath(path, root), st.st_size, st.st_ino))
    return sorted(out)


def _epoch(library, root, state):
    sources = documents_in_place(root, BINDINGS, **LIMITS)
    handle = build_corpus_and_index(library, state, list(sources.documents))
    manifest = CorpusManifest.verified(handle.corpus_manifest_json, expected_digest=handle.corpus_digest)
    return handle, manifest, sources


def _claims(handle, manifest, query):
    items = hybrid_retrieve(handle, query, lexical_limit=8, dense_limit=8, primary_limit=8,
                            total_limit=8)["data"]["items"]
    sizes = {d.digest: d.size_bytes for d in manifest.documents}
    return tuple(ChunkClaim(i["chunk_digest"], i["doc_digest"], 0, 0, sizes[i["doc_digest"]]) for i in items)


def test_repeated_epochs_and_contextual_updates_leave_resources_in_place_and_grow_nothing(library, workstation):
    state = workstation / "state"           # named, never created: the epoch is volatile
    before = _tree(workstation)
    with ContextLane() as lane:
        for epoch in range(12):
            handle, manifest, sources = _epoch(library, workstation, state)
            try:
                assert set(sources.source_paths.values()) == {b.path for b in BINDINGS}
                for query in ("pacman repositories", "cmake ctest cargo", "durable publication"):
                    claims = _claims(handle, manifest, query)
                    seq = lane.submit(ResolveObservation(workstation, manifest, claims, PROVENANCE, relevance=epoch,
                                                         source_paths=sources.source_paths))[0]
                    assert lane.wait(seq, 30) in ("APPLIED", "COALESCED"), (epoch, query)
            finally:
                handle.destroy()
            assert _tree(workstation) == before, epoch     # nothing created, copied, moved or replaced
        view = lane.view()
    assert not state.exists()
    assert len(view.axioms) == len(SOURCES)                # one axiom per verified chunk, keyed by its identity
    for axiom in view.axioms:
        (ref,) = axiom.references                          # coalesced across epochs: same corpus, same chunk
        assert ref.provenance_digest == PROVENANCE


def test_a_source_changed_after_the_epoch_was_built_is_refused_as_stale(library, workstation):
    handle, manifest, sources = _epoch(library, workstation, workstation / "state")
    try:
        claims = _claims(handle, manifest, "pacman repositories")
    finally:
        handle.destroy()
    (workstation / "workspace" / "arch.md").write_text(SOURCES["arch.md"].replace("pacman", "PACMAN"))
    with pytest.raises(ObjectResolutionError, match="INTEGRITY"):
        resolve_chunks(workstation, manifest, claims, max_objects=8, max_text_bytes=1 << 16,
                       max_document_bytes=1 << 16, source_paths=sources.source_paths)
    with ContextLane() as lane:
        published = lane.view()
        seq = lane.submit(ResolveObservation(workstation, manifest, claims, PROVENANCE,
                                             source_paths=sources.source_paths))[0]
        assert lane.wait(seq, 30) == "FAILED" and lane.view() is published


@pytest.mark.parametrize("damage", ["absent", "symlink", "traversal", "absolute", "fifo", "nul", "not-utf8",
                                    "document-budget", "total-budget", "duplicate"])
def test_in_place_sources_fail_closed(workstation, damage):
    bindings, limits = BINDINGS, dict(LIMITS)
    target = workstation / "workspace" / "arch.md"
    if damage == "absent":
        target.unlink()
    elif damage == "symlink":
        outside = workstation / "outside.md"
        outside.write_text(SOURCES["arch.md"])
        target.unlink()
        target.symlink_to(outside)
    elif damage == "fifo":
        target.unlink()
        os.mkfifo(target)
    elif damage == "nul":
        target.write_bytes(b"text\x00with a NUL")
    elif damage == "not-utf8":
        target.write_bytes(b"\xff\xfe not text")
    elif damage == "document-budget":
        limits["max_document_bytes"] = 16
    elif damage == "total-budget":
        limits["max_total_bytes"] = len(SOURCES["arch.md"]) + 4
    elif damage == "duplicate":
        (workstation / "workspace" / "build.md").write_text(SOURCES["arch.md"])
    if damage in ("traversal", "absolute"):
        with pytest.raises(ObjectResolutionError, match="INVALID"):
            SourceBinding("x", "../outside.md" if damage == "traversal" else "/etc/passwd", "elpis.docs", "reference")
        return
    with pytest.raises(ObjectResolutionError):
        documents_in_place(workstation, bindings, **limits)


def test_invalid_provenance_and_unverified_references_are_refused(workstation):
    with pytest.raises(CompositionError, match="CONTEXT_INVALID"):
        ContextReference("ab" * 32, "cd" * 32, "not a digest")
    with pytest.raises(CompositionError, match="CONTEXT_INVALID"):
        ResolveObservation(workstation, None, (), "ZZ" * 32)
    with pytest.raises(CompositionError, match="CONTEXT_INVALID"):
        ResolveObservation(workstation, None, ("not a claim",), PROVENANCE)


def test_retrieval_never_waits_for_residency_unless_the_owner_chose_to(library, workstation):
    state = workstation / "state"
    sources = documents_in_place(workstation, BINDINGS, **LIMITS)
    with build_corpus_and_index(library, state, list(sources.documents)) as default, \
            build_corpus_and_index(library, state, list(sources.documents), residency_wait=True) as waiting:
        assert default.residency_wait is False and waiting.residency_wait is True
        # Uncontended, the choice changes nothing: the same bundle, digest for digest.
        a = hybrid_retrieve(default, "pacman repositories")
        b = hybrid_retrieve(waiting, "pacman repositories")
        assert a["bundle_digest"] == b["bundle_digest"] and a["query_digest"] == b["query_digest"]
    with pytest.raises(HybridRetrievalError):
        build_corpus_and_index(library, state, list(sources.documents), residency_wait=1)
    assert not state.exists()


def test_the_in_place_epoch_is_the_same_epoch_as_one_built_from_the_same_text(library, workstation):
    """Reading in place changes where the bytes come from, never the epoch's identity."""
    sources = documents_in_place(workstation, BINDINGS, **LIMITS)
    texts = [(b.label, SOURCES[b.path.split("/")[1]], b.namespace, b.authority) for b in BINDINGS]
    with build_corpus_and_index(library, workstation / "s", list(sources.documents)) as a, \
            build_corpus_and_index(library, workstation / "s", texts) as b:
        assert (a.corpus_digest, a.shard_digest) == (b.corpus_digest, b.shard_digest)
    assert {hashlib.sha256(t.encode()).hexdigest() for _, t, _, _ in texts} == set(sources.source_paths)
