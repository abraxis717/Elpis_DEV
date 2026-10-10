"""HACF chunk resolution: verified document bytes, recomputed chunk identity, fail closed."""
from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from elpis.structure.retrieval.hacf import RetrievalLibrary, build_corpus_and_index, hybrid_retrieve
from elpis.structure.retrieval.objects import (
    ChunkClaim, CorpusManifest, ObjectResolutionError, chunk_identity, normalize, resolve_chunks,
)

from ..conftest import require_native_library
from .native_bridge_fixture import pin_bridge

DOCS = [
    # CRLF and trailing whitespace inside lines exercise normalization; no blank line, so the
    # native chunker keeps the whole document as chunk 0 (multi-chunk ranges are covered by
    # the integration suite with real ingress exports).
    ("alpha", "Alpha engine anchor keeps structural memory outside the model.  \r\nSecond line\tend", "elpis.docs",
     "canonical"),
    ("beta", "Beta notes about bounded context admission.", "elpis.docs", "reference"),
    ("gamma", "Gamma describes digest addressed retrieval of chunks.", "elpis.docs", "reference"),
]
LIMITS = dict(max_objects=8, max_text_bytes=1 << 16, max_document_bytes=1 << 20)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    # Only test fixtures may pin copied test ELFs from their own workspace.
    # Production requires independently supplied deployment authority.
    path, root, authority = pin_bridge(
        require_native_library("elpis_retrieval_bridge"),
        tmp_path_factory.mktemp("sealed-hacf-objects"),
        "elpis_retrieval_bridge",
    )
    library = RetrievalLibrary(path, root=root, authority=authority)
    state = tmp_path_factory.mktemp("hacf-objects")
    originals = state / "source"
    originals.mkdir()
    for label, text, _, _ in DOCS:
        (originals / f"{label}.txt").write_bytes(text.encode())
    handle = build_corpus_and_index(library, state, DOCS)
    # Retrieval cannot create a corpus, SQLite/WAL or FMS cold directory.
    assert not (state / "corpus").exists()
    assert not (state / "cold").exists()
    manifest = CorpusManifest.verified(handle.corpus_manifest_json, expected_digest=handle.corpus_digest)
    # Native chunk digests, observed through retrieval (single-chunk documents: ordinal 0, whole range).
    items = hybrid_retrieve(handle, "anchor context retrieval", lexical_limit=10, dense_limit=10,
                            primary_limit=10, total_limit=10)["data"]["items"]
    native = {item["doc_digest"]: item["chunk_digest"] for item in items}
    handle.destroy()
    return state, manifest, native


def _resolve(root, manifest, claims, **limits):
    # Bind native document identities to the original, authorized source files.
    paths = {hashlib.sha256(text.encode()).hexdigest(): f"source/{label}.txt"
             for label, text, _, _ in DOCS}
    return resolve_chunks(root, manifest, claims, source_paths=paths, **limits)


def _claims(manifest, native):
    by_doc = {d.digest: d for d in manifest.documents}
    return tuple(ChunkClaim(chunk, doc, 0, 0, by_doc[doc].size_bytes) for doc, chunk in sorted(native.items()))


def test_normalize_matches_hacf_rules():
    assert normalize(b"a  \r\nb\t\n\n\n") == b"a\nb"
    assert normalize(b"\n\n x \n") == b"\n\n x"
    assert normalize(b"") == b""


def test_chunks_resolve_to_verified_text_matching_native_identity(corpus):
    root, manifest, native = corpus
    assert len(native) == 3
    resolved, omitted = _resolve(root, manifest, _claims(manifest, native), **LIMITS)
    assert omitted == 0 and len(resolved) == 3
    for chunk in resolved:
        original = next(t for label, t, _, _ in DOCS
                        if hashlib.sha256(t.encode()).hexdigest() == chunk.doc_digest)
        assert chunk.text == normalize(original.encode())
        assert chunk.norm_digest == hashlib.sha256(chunk.text).hexdigest()
        assert chunk.chunk_digest == native[chunk.doc_digest]  # Python recomputation == native elpis-chunk-v1
    assert any(c.text != next(t.encode() for _, t, _, _ in DOCS
                              if hashlib.sha256(t.encode()).hexdigest() == c.doc_digest) for c in resolved)


def test_resolution_is_bounded_and_deterministic(corpus):
    root, manifest, native = corpus
    claims = _claims(manifest, native)
    first, omitted = _resolve(root, manifest, claims + claims, max_objects=2,
                                    max_text_bytes=1 << 16, max_document_bytes=1 << 20)
    assert omitted == 1 and [c.chunk_digest for c in first] == [c.chunk_digest for c in claims[:2]]
    sizes = [len(c.text) for c in _resolve(root, manifest, claims, **LIMITS)[0]]
    capped, omitted = _resolve(root, manifest, claims, max_objects=8, max_text_bytes=sizes[0],
                                     max_document_bytes=1 << 20)
    assert len(capped) == 1 and omitted == 2
    with pytest.raises(ObjectResolutionError, match="LIMIT"):
        _resolve(root, manifest, claims, max_objects=8, max_text_bytes=1 << 16, max_document_bytes=8)


def test_manifest_must_match_its_pin(corpus):
    _, manifest, _ = corpus
    with pytest.raises(ObjectResolutionError, match="INTEGRITY"):
        CorpusManifest.verified(json.dumps({"documents": []}), expected_digest=manifest.digest)


def test_claims_that_do_not_match_the_document_bytes_are_refused(corpus):
    root, manifest, native = corpus
    good = _claims(manifest, native)[0]
    doc = manifest.document(good.doc_digest)
    for bad, code in [
        (ChunkClaim("f" * 64, good.doc_digest, 0, 0, doc.size_bytes), "INTEGRITY"),   # identity mismatch
        (ChunkClaim(good.chunk_digest, good.doc_digest, 1, 0, doc.size_bytes), "INTEGRITY"),  # ordinal
        (ChunkClaim(good.chunk_digest, good.doc_digest, 0, 0, doc.size_bytes + 1), "INTEGRITY"),  # range
        (ChunkClaim(good.chunk_digest, "e" * 64, 0, 0, 1), "MISSING"),                 # not in manifest
    ]:
        with pytest.raises(ObjectResolutionError, match=code):
            _resolve(root, manifest, (bad,), **LIMITS)


def test_tampered_or_redirected_source_files_fail_closed(corpus, tmp_path):
    root, manifest, native = corpus
    claim = _claims(manifest, native)[0]
    copy = tmp_path / "corpus"
    shutil.copytree(root, copy)
    label = next(label for label, text, _, _ in DOCS
                 if hashlib.sha256(text.encode()).hexdigest() == claim.doc_digest)
    blob = copy / "source" / f"{label}.txt"
    data = bytearray(blob.read_bytes())
    data[0] ^= 1
    blob.write_bytes(bytes(data))
    with pytest.raises(ObjectResolutionError, match="INTEGRITY"):
        _resolve(copy, manifest, (claim,), **LIMITS)
    blob.unlink()
    elsewhere = tmp_path / "outside.blob"
    elsewhere.write_bytes(bytes(data))
    blob.symlink_to(elsewhere)  # openat2(RESOLVE_NO_SYMLINKS) must refuse the redirect
    with pytest.raises(ObjectResolutionError):
        _resolve(copy, manifest, (claim,), **LIMITS)
    with pytest.raises(ObjectResolutionError, match="MISSING"):
        _resolve(tmp_path / "absent", manifest, (claim,), **LIMITS)


def test_residual_gap_a_consistent_subrange_is_real_document_content(corpus):
    """Documented gap: segmentation is not re-derived. A self-consistent claim over a
    sub-range of a real document resolves, but only to that document's own verified bytes."""
    root, manifest, native = corpus
    good = _claims(manifest, native)[0]
    doc = manifest.document(good.doc_digest)
    resolved, _ = _resolve(root, manifest, (good,), **LIMITS)
    whole = resolved[0].text
    text = normalize(whole[:10])
    forged = chunk_identity(good.doc_digest, 0, 0, 10, hashlib.sha256(text).hexdigest(), doc.chunk_profile_digest)
    sub, _ = _resolve(root, manifest, (ChunkClaim(forged, good.doc_digest, 0, 0, 10),), **LIMITS)
    assert sub[0].text == text and whole.startswith(sub[0].text)
