"""Resolve digest-addressed HACF chunks into verified bytes.

HACF is Elpis's external context store. A chunk is named by its persisted
``elpis-chunk-v1`` identity: SHA-256 over the owning document digest, the
chunk ordinal, the byte range, the SHA-256 of the normalized chunk text and
the chunking-profile digest. Documents are content-addressed blobs
(``<corpus root>/corpus/<document digest>.blob``), and the corpus manifest
records each document's size, chunk count and chunking-profile digest.

Resolution trusts none of the text it is handed:

1. the corpus manifest must hash to the digest the caller pinned (the
   ingress result pins it);
2. the document blob is opened through :class:`elpis.substrate.boundary.RootCapability`
   beneath the corpus root (``openat2``, no symlinks, regular file only) at a
   path derived from the validated digest, bounded in size, and must hash to
   its document digest and match the manifest size;
3. the claimed range must lie inside the document and the ordinal below the
   manifest's chunk count;
4. the text is re-derived from the verified bytes with HACF's normalization,
   and the chunk identity recomputed from it must equal the requested digest.

The returned text is therefore verified document content bound to the chunk
digest. Residual gap: the native chunker is not exposed to Python, so the
claimed (ordinal, range) is shown to be consistent with the chunk identity
and the manifest, but is not re-derived as the chunker's own segmentation.

No network, no search, no traversal beyond the fixed blob path, and no
authority from retrieval scores.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
from contextlib import nullcontext

from elpis.substrate.boundary import RootCapability, read_exact
from elpis.substrate.contracts import ContractError

from .errors import RetrievalError

__all__ = (
    "ChunkClaim", "CorpusDocument", "CorpusManifest", "InPlaceDocuments", "ObjectResolutionError", "ResolvedChunk",
    "SourceBinding", "chunk_identity", "documents_in_place", "normalize", "resolve_chunks",
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
CHUNK_IDENTITY_TAG = b"elpis-chunk-v1"


class ObjectResolutionError(RetrievalError):
    """A chunk could not be resolved to verified bytes (fail closed)."""


def _digest(value, what):
    if type(value) is not str or not _DIGEST.match(value):
        raise ObjectResolutionError("INVALID", f"{what} must be a lowercase SHA-256 hex digest")
    return value


def _count(value, what, low=0):
    if type(value) is not int or value < low:
        raise ObjectResolutionError("INVALID", f"{what} must be an integer >= {low}")
    return value


def normalize(data: bytes) -> bytes:
    """HACF ``elpis_normalize``: strip trailing spaces, tabs and CR per line; drop trailing newlines."""
    lines = data.split(b"\n")
    out = b"\n".join(line.rstrip(b" \t\r") for line in lines)
    return out.rstrip(b"\n")


def chunk_identity(doc_digest: str, ordinal: int, byte_start: int, byte_end: int,
                   norm_digest: str, profile_digest: str) -> str:
    """The persisted ``elpis-chunk-v1`` chunk digest (HACF chunking, unchanged)."""
    def field(text):
        raw = text.encode("ascii")
        return len(raw).to_bytes(4, "big") + raw
    ident = (CHUNK_IDENTITY_TAG + field(doc_digest) + ordinal.to_bytes(4, "big")
             + byte_start.to_bytes(8, "big") + byte_end.to_bytes(8, "big")
             + field(norm_digest) + field(profile_digest))
    return hashlib.sha256(ident).hexdigest()


@dataclass(frozen=True)
class CorpusDocument:
    digest: str
    size_bytes: int
    chunk_count: int
    chunk_profile_digest: str
    media_type: str


@dataclass(frozen=True)
class CorpusManifest:
    """A corpus manifest verified against a pinned digest."""
    digest: str
    documents: tuple[CorpusDocument, ...]

    @classmethod
    def verified(cls, manifest_json, *, expected_digest: str) -> "CorpusManifest":
        _digest(expected_digest, "expected manifest digest")
        raw = manifest_json.encode("utf-8") if type(manifest_json) is str else manifest_json
        if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != expected_digest:
            raise ObjectResolutionError("INTEGRITY", "corpus manifest does not match the pinned digest")
        try:
            value = json.loads(raw)
        except ValueError as exc:
            raise ObjectResolutionError("INVALID", "corpus manifest is not JSON") from exc
        if type(value) is not dict or type(value.get("documents")) is not list:
            raise ObjectResolutionError("INVALID", "corpus manifest shape")
        documents = []
        for entry in value["documents"]:
            if type(entry) is not dict:
                raise ObjectResolutionError("INVALID", "corpus manifest document entry")
            documents.append(CorpusDocument(
                _digest(entry.get("digest"), "document digest"),
                _count(entry.get("size_bytes"), "document size"),
                _count(entry.get("chunk_count"), "document chunk count"),
                _digest(entry.get("chunk_profile_digest"), "chunk profile digest"),
                entry.get("media_type") if type(entry.get("media_type")) is str else "",
            ))
        if len({d.digest for d in documents}) != len(documents):
            raise ObjectResolutionError("INVALID", "duplicate document in corpus manifest")
        return cls(expected_digest, tuple(documents))

    def document(self, digest: str) -> CorpusDocument:
        for entry in self.documents:
            if entry.digest == digest:
                return entry
        raise ObjectResolutionError("MISSING", "document not in the pinned corpus manifest")


@dataclass(frozen=True)
class ChunkClaim:
    """Where a proposal says a chunk lives. Unverified until resolved."""
    chunk_digest: str
    doc_digest: str
    ordinal: int
    byte_start: int
    byte_end: int

    def __post_init__(self):
        _digest(self.chunk_digest, "chunk digest")
        _digest(self.doc_digest, "document digest")
        _count(self.ordinal, "chunk ordinal")
        _count(self.byte_start, "chunk start")
        _count(self.byte_end, "chunk end")
        if self.byte_end < self.byte_start:
            raise ObjectResolutionError("INVALID", "chunk range")


@dataclass(frozen=True)
class ResolvedChunk:
    """A chunk's normalized text, verified against its digest from the document bytes."""
    chunk_digest: str
    doc_digest: str
    ordinal: int
    byte_start: int
    byte_end: int
    norm_digest: str
    text: bytes


def _read_document(capability, document: CorpusDocument, max_document_bytes: int,
                   source_path: str) -> bytes:
    if document.size_bytes > max_document_bytes:
        raise ObjectResolutionError("LIMIT", "document exceeds the resolution budget")
    try:
        # Resolve the original authorized resource in place; never manufacture
        # a digest-named corpus blob. RootCapability enforces confined read-only
        # resolution and refuses symlink traversal.
        fd = capability.open_file(source_path)
    except ContractError as exc:
        raise ObjectResolutionError("MISSING" if exc.code.value == "MISSING" else "INTEGRITY",
                                    "source document could not be opened beneath the authorized root") from exc
    try:
        size = os.fstat(fd).st_size
        if size != document.size_bytes:
            raise ObjectResolutionError("INTEGRITY", "document blob size differs from the manifest")
        data = read_exact(fd, size, 0) if size else b""
    finally:
        os.close(fd)
    if hashlib.sha256(data).hexdigest() != document.digest:
        raise ObjectResolutionError("INTEGRITY", "document blob does not match its digest")
    return data


def resolve_chunks(corpus_root, manifest: CorpusManifest, claims, *, max_objects: int,
                   max_text_bytes: int, max_document_bytes: int, source_paths=None, source_handle=None):
    """Resolve claims in order into verified chunks under explicit bounds.

    Duplicate chunk digests are resolved once, at their first position. The
    result is the longest prefix of the distinct claims whose cumulative text
    fits ``max_text_bytes`` and ``max_objects``; the number of distinct claims
    left out is returned with it. Any verification failure raises.
    """
    if type(manifest) is not CorpusManifest:
        raise ObjectResolutionError("INVALID", "manifest must be a verified CorpusManifest")
    for value, what in ((max_objects, "max_objects"), (max_text_bytes, "max_text_bytes"),
                        (max_document_bytes, "max_document_bytes")):
        _count(value, what, 1)
    if type(claims) is not tuple or not all(type(c) is ChunkClaim for c in claims):
        raise ObjectResolutionError("INVALID", "claims must be a tuple of ChunkClaim")
    # Explicit caller-provided digest -> relative original path bindings.
    # Legacy blob mode stays available only to callers explicitly resolving
    # an already-existing imported corpus. Neither branch creates a file.
    if source_paths is not None:
        if type(source_paths) is not dict:
            raise ObjectResolutionError("INVALID", "source_paths must be an explicit dict")
        for digest, path in source_paths.items():
            _digest(digest, "source binding digest")
            if type(path) is not str or not path or path.startswith("/") or                any(part in ("", ".", "..") for part in path.split("/")):
                raise ObjectResolutionError("INVALID", "source binding must be a confined relative path")
    if source_handle is not None:
        if source_paths is not None or not source_handle._valid or source_handle.corpus_digest != manifest.digest:
            raise ObjectResolutionError("INTEGRITY", "volatile corpus owner or manifest mismatch")
    distinct, seen = [], set()
    for claim in claims:
        if claim.chunk_digest not in seen:
            seen.add(claim.chunk_digest)
            distinct.append(claim)
    resolved, used, documents = [], 0, {}
    if source_handle is None:
        try:
            capability = RootCapability(corpus_root)
        except ContractError as exc:
            raise ObjectResolutionError("MISSING", "corpus root could not be opened") from exc
    else:
        capability = None
    with (capability if capability is not None else nullcontext()):
        for claim in distinct:
            if len(resolved) == max_objects:
                break
            document = manifest.document(claim.doc_digest)
            if claim.ordinal >= document.chunk_count or claim.byte_end > document.size_bytes:
                raise ObjectResolutionError("INTEGRITY", "chunk claim outside its document")
            if claim.doc_digest not in documents:
                if source_paths is None:
                    path = f"corpus/{document.digest}.blob"  # historical operator-owned imported corpus
                else:
                    path = source_paths.get(document.digest)
                    if path is None:
                        raise ObjectResolutionError("MISSING", "no authorized source binding for document")
                if source_handle is not None:
                    if document.size_bytes > max_document_bytes:
                        raise ObjectResolutionError("LIMIT", "document exceeds the resolution budget")
                    try:
                        data = source_handle.read_document(document.digest, document.size_bytes)
                    except Exception as exc:
                        raise ObjectResolutionError("INTEGRITY", "volatile document failed verification") from exc
                    if hashlib.sha256(data).hexdigest() != document.digest:
                        raise ObjectResolutionError("INTEGRITY", "volatile document digest mismatch")
                    documents[claim.doc_digest] = data
                else:
                    documents[claim.doc_digest] = _read_document(capability, document, max_document_bytes, path)
            text = normalize(documents[claim.doc_digest][claim.byte_start:claim.byte_end])
            norm_digest = hashlib.sha256(text).hexdigest()
            if chunk_identity(claim.doc_digest, claim.ordinal, claim.byte_start, claim.byte_end,
                              norm_digest, document.chunk_profile_digest) != claim.chunk_digest:
                raise ObjectResolutionError("INTEGRITY", "chunk identity does not match the document bytes")
            if used + len(text) > max_text_bytes:
                break
            used += len(text)
            resolved.append(ResolvedChunk(claim.chunk_digest, claim.doc_digest, claim.ordinal,
                                          claim.byte_start, claim.byte_end, norm_digest, text))
    return tuple(resolved), len(distinct) - len(resolved)


# -- in-place sources for a bounded retrieval epoch ------------------------------------------------------------------

@dataclass(frozen=True)
class SourceBinding:
    """One existing resource to address in place: its label, its path relative to the authorized root, and the
    namespace and authority class the retrieval epoch records for it."""
    label: str
    path: str
    namespace: str
    authority: str

    def __post_init__(self):
        for value, what in ((self.label, "label"), (self.namespace, "namespace"), (self.authority, "authority")):
            if type(value) is not str or not value or "\x00" in value:
                raise ObjectResolutionError("INVALID", f"source {what}")
        if type(self.path) is not str or not self.path or self.path.startswith("/") or \
                any(part in ("", ".", "..") for part in self.path.split("/")) or "\x00" in self.path:
            raise ObjectResolutionError("INVALID", "source binding must be a confined relative path")


@dataclass(frozen=True)
class InPlaceDocuments:
    """What a bounded retrieval epoch needs from resources that stay where they are.

    ``documents`` are the transient ``(label, text, namespace, authority)`` tuples for
    :func:`elpis.structure.retrieval.hacf.build_corpus_and_index` (held only until the epoch is built);
    ``source_paths`` binds each document digest to its original relative path, so later resolution
    (:func:`resolve_chunks` with ``source_paths``) re-reads and re-verifies the original in place: a source changed
    since the epoch was built is refused (stale), never silently reused."""
    documents: tuple
    source_paths: dict


def documents_in_place(root, bindings, *, max_document_bytes: int, max_total_bytes: int) -> InPlaceDocuments:
    """Read existing resources once, in place, for a bounded retrieval epoch: nothing is copied or created.

    Each binding is opened beneath ``root`` through :class:`RootCapability` (``openat2``, no symlinks, regular file
    only, no traversal), bounded per document and in total, and must be strict UTF-8 text without NUL (the
    retrieval epoch's text boundary). Labels and paths are distinct. Any refusal raises and returns nothing.
    """
    _count(max_document_bytes, "max_document_bytes", 1)
    _count(max_total_bytes, "max_total_bytes", 1)
    if type(bindings) is not tuple or not bindings or not all(type(b) is SourceBinding for b in bindings):
        raise ObjectResolutionError("INVALID", "bindings must be a nonempty tuple of SourceBinding")
    if len({b.label for b in bindings}) != len(bindings) or len({b.path for b in bindings}) != len(bindings):
        raise ObjectResolutionError("INVALID", "duplicate source label or path")
    try:
        capability = RootCapability(root)
    except ContractError as exc:
        raise ObjectResolutionError("MISSING", "source root could not be opened") from exc
    documents, paths, total = [], {}, 0
    with capability:
        for binding in bindings:
            try:
                fd = capability.open_file(binding.path)
            except ContractError as exc:
                raise ObjectResolutionError("MISSING" if exc.code.value == "MISSING" else "INTEGRITY",
                                            "source could not be opened beneath the authorized root") from exc
            try:
                size = os.fstat(fd).st_size
                if size > max_document_bytes:
                    raise ObjectResolutionError("LIMIT", "source exceeds the per-document bound")
                total += size
                if total > max_total_bytes:
                    raise ObjectResolutionError("LIMIT", "sources exceed the epoch's total bound")
                data = read_exact(fd, size, 0) if size else b""
            finally:
                os.close(fd)
            if b"\x00" in data:
                raise ObjectResolutionError("INVALID", "source text contains NUL")
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ObjectResolutionError("INVALID", "source is not UTF-8 text") from exc
            digest = hashlib.sha256(data).hexdigest()
            if digest in paths:
                raise ObjectResolutionError("INVALID", "two sources with the same content")
            paths[digest] = binding.path
            documents.append((binding.label, text, binding.namespace, binding.authority))
    return InPlaceDocuments(tuple(documents), paths)
