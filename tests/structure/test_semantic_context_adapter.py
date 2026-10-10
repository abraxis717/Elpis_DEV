"""Track C v1 through the structure layer: admitted semantic relation -> verified context edges -> graph epoch.

Scope ADMITTED_CLAIM_TO_CLAIM_PRIMARY_WITNESSES_ONLY. The proofs are real native admission records (exact ABI v1
images) emitted by the native fixture ``test_retrieval_semantic_projection --emit`` over DOCS; Python never
interprets them. Everything is verified natively against the epoch's own chunks, all-or-nothing, with no
inference and no persistence. The construction step is explicit: no runtime module invokes it.
"""
from __future__ import annotations

import ast
import struct
import subprocess
from pathlib import Path

import pytest

from elpis.structure.retrieval.errors import HybridRetrievalError
from elpis.structure.retrieval.hacf import (
    SEMANTIC_PROJECTION_SCOPE,
    ZERO_DIGEST,
    ContextEdge,
    RetrievalLibrary,
    SemanticEndpointWitness,
    SemanticRelationProof,
    build_corpus_and_index,
    build_semantic_context_epoch,
    hybrid_retrieve,
)

from ..conftest import native_build_dir, native_required, require_native_library
from .native_bridge_fixture import pin_bridge

# Must equal LABELS/TEXTS/NAMESPACES/AUTHORITIES in native/structure/bridge/tests/test_retrieval_semantic_projection.c.
DOCS = [("alpha", "Alpha original evidence chunk.", "elpis.docs", "reference"),
        ("beta", "Beta target evidence chunk.", "elpis.docs", "reference")]
SUPPORTS, CONTRADICTS = 0x53430103, 0x53430104
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    path, root, authority = pin_bridge(require_native_library("elpis_retrieval_bridge"),
                                       tmp_path_factory.mktemp("semantic-adapter-lib"), "elpis_retrieval_bridge")
    return RetrievalLibrary(path, root=root, authority=authority)


def _read_proofs(path: Path) -> list[SemanticRelationProof]:
    data = path.read_bytes()
    assert data[:8] == b"ELPTC1\0\0"
    (count,), offset = struct.unpack_from("<I", data, 8), 12
    proofs = []
    for _ in range(count):
        blobs = []
        for _ in range(15):
            (n,) = struct.unpack_from("<I", data, offset)
            blobs.append(data[offset + 4:offset + 4 + n])
            offset += 4 + n
        proofs.append(SemanticRelationProof(*blobs[:5], SemanticEndpointWitness(*blobs[5:10]),
                                            SemanticEndpointWitness(*blobs[10:15])))
    assert offset == len(data)
    return proofs


@pytest.fixture(scope="module")
def proofs(tmp_path_factory):
    emitters = sorted(p for p in native_build_dir().rglob("test_retrieval_semantic_projection") if p.is_file())
    if not emitters:
        message = f"test_retrieval_semantic_projection not built under {native_build_dir()}"
        if native_required():
            pytest.fail(message)
        pytest.skip(message + " (never an implicit PASS)")
    out = tmp_path_factory.mktemp("semantic-proofs") / "proofs.bin"
    subprocess.run([str(emitters[0]), "--emit", str(out)], check=True, timeout=60)
    return _read_proofs(out)


def _edge(e: ContextEdge):
    return (e.subject_chunk_digest, e.object_chunk_digest, e.provenance_digest, e.edge_type, e.authority)


def test_admitted_relations_project_to_verified_edges_between_actual_chunks(library, proofs, tmp_path):
    assert SEMANTIC_PROJECTION_SCOPE == "ADMITTED_CLAIM_TO_CLAIM_PRIMARY_WITNESSES_ONLY"
    with build_corpus_and_index(library, tmp_path / "epoch", DOCS) as epoch:
        edges = epoch.project_semantic_edges(proofs)
        again = epoch.project_semantic_edges(proofs)
        assert epoch.project_semantic_edges(()) == ()
        assert epoch.graph_edge_count == 0 and epoch.graph_snapshot_digest == ZERO_DIGEST  # this epoch unchanged
    assert [_edge(e) for e in edges] == [_edge(e) for e in again]
    assert [e.edge_type for e in edges] == [SUPPORTS, CONTRADICTS]
    assert all(e.subject_chunk_digest != e.object_chunk_digest and e.authority == 1 for e in edges)
    assert len({e.provenance_digest for e in edges}) == 2
    assert not (tmp_path / "epoch").exists()


def test_explicit_construction_builds_a_graph_epoch_for_bounded_retrieval(library, proofs, tmp_path):
    with build_semantic_context_epoch(library, tmp_path / "epoch", DOCS, proofs) as epoch:
        assert epoch.graph_edge_count == 2 and epoch.graph_snapshot_digest != ZERO_DIGEST
        result = hybrid_retrieve(epoch, "Alpha original evidence", primary_limit=1, total_limit=4)
        assert result["graph_snapshot_digest"] == epoch.graph_snapshot_digest
    assert not (tmp_path / "epoch").exists()


def _tamper(proof: SemanticRelationProof, field: str) -> SemanticRelationProof:
    data = bytearray(getattr(proof, field))
    data[-40] ^= 1
    return SemanticRelationProof(**{**proof.__dict__, field: bytes(data)})


def test_refusal_is_all_or_nothing_and_checks_the_epoch(library, proofs, tmp_path):
    with build_corpus_and_index(library, tmp_path / "epoch", DOCS) as epoch:
        with pytest.raises(HybridRetrievalError) as refused:
            epoch.project_semantic_edges([proofs[0], _tamper(proofs[1], "relation")])
        assert refused.value.code == "PROJECTION_REFUSED"
        with pytest.raises(HybridRetrievalError) as invalid:
            epoch.project_semantic_edges([SemanticRelationProof(**{**proofs[0].__dict__,
                                                                   "relation": proofs[0].relation[:-1]})])
        assert invalid.value.code == "PROJECTION_INVALID"
        with pytest.raises(HybridRetrievalError):
            epoch.project_semantic_edges([object()])
    other = [(label, "Unrelated " + text, ns, auth) for label, text, ns, auth in DOCS]
    with build_corpus_and_index(library, tmp_path / "other", other) as foreign:
        with pytest.raises(HybridRetrievalError) as foreign_refused:
            foreign.project_semantic_edges(proofs)
        assert foreign_refused.value.code == "PROJECTION_REFUSED"


def test_the_construction_step_is_never_reached_by_the_runtime():
    names = {"project_semantic_edges", "build_semantic_context_epoch", "SemanticRelationProof"}
    for path in (REPO / "src" / "elpis" / "runtime").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        assert not (used & names), (path.name, used & names)
