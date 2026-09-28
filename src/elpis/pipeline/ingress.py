"""Bounded Regex -> HACF query ingress: a thin ctypes layer over the native bridge.

    text bytes -> streaming Regex lexer (bounded interval-specification grammar)
    -> read-only HACF lexical retrieval + one-hop context graph
    -> context proposal (``elpis.regex-hacf-context-proposal.r1``)
    -> atomic query-local proposal batch published as a semantic query overlay

Every result is a proposal. The semantic, admission, execution and runtime
authority flags are zero by construction. This binding re-reads them and
refuses any result that claims otherwise. Ambiguous lexical evidence is
rejected before any batch is built.

The library path is always supplied by the caller: there is no environment
variable lookup and no repository-relative fallback. The corpus root must hold
an existing HACF corpus. The bridge never creates one and never mutates it.
"""

from __future__ import annotations

import ctypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

BRIDGE_ABI_VERSION = 1
REGEX_ABI_VERSION = 1
QUERY_ABI_VERSION = 1
DEFAULT_CHUNK_BYTES = 4096
DEFAULT_CARRY_BYTES = 1024  # the bounded lexer's default whole-input profile
MIN_CARRY_BYTES = 256
PROPOSAL_SCHEMA = "elpis.regex-hacf-context-proposal.r1"

_HEX64 = re.compile(r"[0-9a-f]{64}")
_REGEX_ERRORS = {-1: "INVALID", -2: "PARSE", -3: "NOMEM", -4: "RANGE", -5: "STATE"}
_QUERY_ERRORS = {-1: "INVALID", -2: "REGEX", -3: "HACF", -4: "BATCH", -5: "NOMEM",
                 -6: "RANGE", -7: "CARDINALITY"}


class IngressError(Exception):
    """Fail-closed ingress rejection with a stable code."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}" if detail else code)


class _EvidenceView(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("evidence_id", ctypes.c_char * 65),
                ("pattern_id", ctypes.c_char * 96), ("lexical_anchor", ctypes.c_char * 128),
                ("reserved", ctypes.c_uint8 * 32)]


class _CandidateView(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("candidate_id", ctypes.c_char * 65),
                ("reserved", ctypes.c_uint8 * 32)]


class ContextEdge(ctypes.Structure):
    """One immutable context-graph edge between two HACF chunk digests."""

    _fields_ = [("subject_chunk_digest", ctypes.c_char * 65), ("object_chunk_digest", ctypes.c_char * 65),
                ("provenance_digest", ctypes.c_char * 65), ("edge_type", ctypes.c_uint32),
                ("authority", ctypes.c_uint32)]

    @classmethod
    def of(cls, subject: str, obj: str, provenance: str, edge_type: int, authority: int) -> "ContextEdge":
        for value in (subject, obj, provenance):
            _hex64(value, "edge digest")
        if type(edge_type) is not int or not 0 < edge_type < 2**32:
            raise IngressError("INVALID", "edge type")
        if type(authority) is not int or not 0 <= authority <= 3:
            raise IngressError("INVALID", "edge authority")
        return cls(subject.encode(), obj.encode(), provenance.encode(), edge_type, authority)


def _hex64(value: object, what: str) -> str:
    if type(value) is not str or not _HEX64.fullmatch(value):
        raise IngressError("IDENTITY", what)
    return value


def _sig(fn, argtypes, restype):
    fn.argtypes, fn.restype = argtypes, restype
    return fn


class IngressLibrary:
    """One explicitly loaded ``libelpis_ingress_bridge``."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.is_absolute() or not path.is_file():
            raise IngressError("LIB_NOT_FOUND", str(path))
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            raise IngressError("LIB_LOAD_FAILED", str(exc)) from exc
        self.path = path
        vp, sz, u32, u64, cp = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_char_p
        pvp = ctypes.POINTER(vp)
        r = "elpis_streaming_regex_"
        q = "elpis_regex_hacf_query_ingress_"
        self.bridge_abi = _sig(lib.elpis_ingress_bridge_abi_version, [], u32)
        self.regex_abi = _sig(getattr(lib, r + "abi_version_v1"), [], u32)
        self.query_abi = _sig(getattr(lib, q + "abi_version_v1"), [], u32)
        if (self.bridge_abi(), self.regex_abi(), self.query_abi()) != (
                BRIDGE_ABI_VERSION, REGEX_ABI_VERSION, QUERY_ABI_VERSION):
            raise IngressError("ABI_MISMATCH", str(path))

        self.parse_v1 = _sig(getattr(lib, r + "parse_bytes_v1"), [cp, sz, sz, sz, pvp], ctypes.c_int)
        self.stream_create = _sig(getattr(lib, r + "stream_create_v2"), [vp, pvp], ctypes.c_int)
        self.stream_feed = _sig(getattr(lib, r + "stream_feed_v2"), [vp, cp, sz], ctypes.c_int)
        self.stream_finalize = _sig(getattr(lib, r + "stream_finalize_v2"), [vp, pvp], ctypes.c_int)
        self.stream_destroy = _sig(getattr(lib, r + "stream_destroy_v2"), [vp], None)
        self.regex_destroy = _sig(getattr(lib, r + "result_destroy_v1"), [vp], None)
        self.regex_last_error = _sig(getattr(lib, r + "last_error_v1"), [], cp)
        self.regex_last_error_v2 = _sig(getattr(lib, r + "last_error_v2"), [], cp)
        for name in ("json", "ingress_json", "composition_json", "source_sha256"):
            setattr(self, "regex_" + name, _sig(getattr(lib, r + "result_" + name + "_v1"), [vp], cp))
        self.regex_source_bytes = _sig(getattr(lib, r + "result_source_bytes_v1"), [vp], u64)
        for name in ("evidence_count", "candidate_count", "ambiguity_count"):
            setattr(self, "regex_" + name, _sig(getattr(lib, r + "result_" + name + "_v1"), [vp], u32))
        self.regex_fail_closed = _sig(getattr(lib, r + "result_fail_closed_v1"), [vp], ctypes.c_int)
        self.regex_evidence_at = _sig(getattr(lib, r + "result_evidence_at_v1"),
                                      [vp, u32, ctypes.POINTER(_EvidenceView)], ctypes.c_int)
        self.regex_candidate_at = _sig(getattr(lib, r + "result_candidate_at_v1"),
                                       [vp, u32, ctypes.POINTER(_CandidateView)], ctypes.c_int)

        self.env_open = _sig(lib.elpis_ingress_env_open,
                             [cp, ctypes.POINTER(ContextEdge), u32, pvp, cp], ctypes.c_int)
        self.env_close = _sig(lib.elpis_ingress_env_close, [vp], None)
        self.env_run = _sig(lib.elpis_ingress_env_run, [vp, cp, sz, sz, pvp], ctypes.c_int)
        self.env_counts = _sig(lib.elpis_ingress_env_corpus_counts,
                               [vp, ctypes.POINTER(u64), ctypes.POINTER(u64)], ctypes.c_int)
        self.env_edge_count = _sig(lib.elpis_ingress_env_edge_count, [vp], u32)
        self.query_destroy = _sig(getattr(lib, q + "result_destroy_v1"), [vp], None)
        self.query_last_error = _sig(getattr(lib, q + "last_error_v1"), [], cp)
        for name in ("status", "source_sha256", "proposal_digest", "corpus_manifest_digest",
                     "context_graph_manifest_digest", "proposal_set_digest",
                     "query_local_segment_digest", "overlay_identity", "batch_receipt_identity",
                     "proposal_json"):
            setattr(self, "query_" + name, _sig(getattr(lib, q + "result_" + name + "_v1"), [vp], cp))
        self.query_candidate_count = _sig(getattr(lib, q + "result_candidate_count_v1"), [vp], u32)
        self.query_candidate_at = _sig(getattr(lib, q + "result_candidate_at_v1"),
                                       [vp, u32, ctypes.POINTER(_CandidateView)], ctypes.c_int)
        for name in ("fail_closed", "batch_published", "semantic_authority", "admission_authority",
                     "execution_authority", "runtime_admission"):
            setattr(self, "query_" + name, _sig(getattr(lib, q + "result_" + name + "_v1"), [vp], ctypes.c_int))


# ---------------------------------------------------------------------------
# Lexical ingress
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LexicalEvidence:
    evidence_id: str
    pattern_id: str
    lexical_anchor: str


@dataclass(frozen=True)
class LexicalResult:
    """Immutable copy of one finalized Regex result."""

    result_json: bytes
    ingress_json: bytes
    composition_json: bytes
    source_sha256: str
    source_bytes: int
    evidence: tuple[LexicalEvidence, ...]
    candidate_ids: tuple[str, ...]
    ambiguity_count: int
    fail_closed: bool


def _copy_lexical(lib: IngressLibrary, result: ctypes.c_void_p) -> LexicalResult:
    evidence = []
    for i in range(lib.regex_evidence_count(result)):
        view = _EvidenceView()
        if lib.regex_evidence_at(result, i, ctypes.byref(view)) != 0:
            raise IngressError("ABI", "evidence accessor")
        evidence.append(LexicalEvidence(_hex64(view.evidence_id.decode(), "evidence id"),
                                        view.pattern_id.decode(), view.lexical_anchor.decode()))
    candidates = []
    for i in range(lib.regex_candidate_count(result)):
        view = _CandidateView()
        if lib.regex_candidate_at(result, i, ctypes.byref(view)) != 0:
            raise IngressError("ABI", "candidate accessor")
        candidates.append(_hex64(view.candidate_id.decode(), "candidate id"))
    return LexicalResult(
        result_json=lib.regex_json(result), ingress_json=lib.regex_ingress_json(result),
        composition_json=lib.regex_composition_json(result),
        source_sha256=_hex64(lib.regex_source_sha256(result).decode(), "source digest"),
        source_bytes=int(lib.regex_source_bytes(result)), evidence=tuple(evidence),
        candidate_ids=tuple(candidates), ambiguity_count=int(lib.regex_ambiguity_count(result)),
        fail_closed=bool(lib.regex_fail_closed(result)))


def lex(library: IngressLibrary, data: bytes, *, chunk_size: int = DEFAULT_CHUNK_BYTES,
        carry_bytes: int = DEFAULT_CARRY_BYTES) -> LexicalResult:
    """Lex a whole bounded input. Input longer than ``carry_bytes`` is rejected
    (``RANGE``) and publishes nothing."""
    if type(data) is not bytes:
        raise IngressError("INVALID", "input must be bytes")
    if type(chunk_size) is not int or chunk_size < 1 or type(carry_bytes) is not int or carry_bytes < MIN_CARRY_BYTES:
        raise IngressError("INVALID", "stream bounds")
    result = ctypes.c_void_p()
    rc = library.parse_v1(data, len(data), chunk_size, carry_bytes, ctypes.byref(result))
    try:
        if rc != 0:
            detail = (library.regex_last_error() or b"").decode("utf-8", "replace")
            raise IngressError(_REGEX_ERRORS.get(rc, "NATIVE"), detail)
        return _copy_lexical(library, result)
    finally:
        library.regex_destroy(result)


def lex_stream(library: IngressLibrary, chunks: Iterable[bytes]) -> LexicalResult:
    """Lex an unbounded stream of byte chunks with the streaming lexer."""
    stream, result = ctypes.c_void_p(), ctypes.c_void_p()
    if library.stream_create(None, ctypes.byref(stream)) != 0:
        raise IngressError("NATIVE", "stream create")
    try:
        for chunk in chunks:
            if type(chunk) is not bytes:
                raise IngressError("INVALID", "chunk must be bytes")
            rc = library.stream_feed(stream, chunk, len(chunk))
            if rc != 0:
                raise IngressError(_REGEX_ERRORS.get(rc, "NATIVE"),
                                   (library.regex_last_error_v2() or b"").decode("utf-8", "replace"))
        rc = library.stream_finalize(stream, ctypes.byref(result))
        if rc != 0:
            raise IngressError(_REGEX_ERRORS.get(rc, "NATIVE"),
                               (library.regex_last_error_v2() or b"").decode("utf-8", "replace"))
        return _copy_lexical(library, result)
    finally:
        library.regex_destroy(result)
        library.stream_destroy(stream)


# ---------------------------------------------------------------------------
# Query ingress
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryIngressResult:
    """Immutable copy of one Regex -> HACF -> query-local composition.

    ``batch_published`` is True only for ``PUBLISHED_ATOMIC_QUERY_OVERLAY``;
    a fail-closed result carries no proposal set, segment, overlay or receipt.
    """

    status: str
    source_sha256: str
    proposal_digest: str
    corpus_manifest_digest: str
    context_graph_manifest_digest: str
    proposal_set_digest: str | None
    query_local_segment_digest: str | None
    overlay_identity: str | None
    batch_receipt_identity: str | None
    proposal_json: bytes
    candidate_ids: tuple[str, ...]
    fail_closed: bool
    batch_published: bool


def _optional_digest(value: bytes | None) -> str | None:
    if not value:
        return None
    return _hex64(value.decode(), "batch identity")


def _copy_query(lib: IngressLibrary, result: ctypes.c_void_p) -> QueryIngressResult:
    for flag in ("semantic_authority", "admission_authority", "execution_authority", "runtime_admission"):
        if getattr(lib, "query_" + flag)(result) != 0:
            raise IngressError("AUTHORITY_WIDENING", flag)
    candidates = []
    for i in range(lib.query_candidate_count(result)):
        view = _CandidateView()
        if lib.query_candidate_at(result, i, ctypes.byref(view)) != 0:
            raise IngressError("ABI", "candidate accessor")
        candidates.append(_hex64(view.candidate_id.decode(), "candidate id"))
    published = bool(lib.query_batch_published(result))
    fail_closed = bool(lib.query_fail_closed(result))
    batch = {name: _optional_digest(getattr(lib, "query_" + name)(result))
             for name in ("proposal_set_digest", "query_local_segment_digest", "overlay_identity",
                          "batch_receipt_identity")}
    if published == fail_closed or published != all(batch.values()):
        raise IngressError("ABI", "inconsistent publication state")
    proposal_json = lib.query_proposal_json(result)
    if not proposal_json:
        raise IngressError("ABI", "missing proposal export")
    return QueryIngressResult(
        status=lib.query_status(result).decode(),
        source_sha256=_hex64(lib.query_source_sha256(result).decode(), "source digest"),
        proposal_digest=_hex64(lib.query_proposal_digest(result).decode(), "proposal digest"),
        corpus_manifest_digest=_hex64(lib.query_corpus_manifest_digest(result).decode(), "corpus digest"),
        context_graph_manifest_digest=_hex64(lib.query_context_graph_manifest_digest(result).decode(),
                                             "context graph digest"),
        proposal_json=proposal_json, candidate_ids=tuple(candidates), fail_closed=fail_closed,
        batch_published=published, **batch)


class QueryIngress:
    """Owning handle: an existing HACF corpus plus one immutable context graph."""

    def __init__(self, library: IngressLibrary, corpus_root: str | Path,
                 edges: Iterable[ContextEdge] = ()) -> None:
        corpus_root = Path(corpus_root)
        if not corpus_root.is_absolute():
            raise IngressError("INVALID", "corpus root must be absolute")
        edge_list = list(edges)
        if not all(type(e) is ContextEdge for e in edge_list):
            raise IngressError("INVALID", "edges must be ContextEdge")
        array = (ContextEdge * len(edge_list))(*edge_list) if edge_list else None
        self.library = library
        self._env = ctypes.c_void_p()
        error = ctypes.create_string_buffer(256)
        rc = library.env_open(str(corpus_root).encode(), array, len(edge_list), ctypes.byref(self._env), error)
        if rc != 0:
            self._env = ctypes.c_void_p()
            raise IngressError("CORPUS" if rc == -2 else "GRAPH" if rc == -3 else "INVALID",
                               error.value.decode("utf-8", "replace"))

    def run(self, task: bytes, *, chunk_size: int = DEFAULT_CHUNK_BYTES) -> QueryIngressResult:
        if not self._env:
            raise IngressError("STATE", "ingress closed")
        if type(task) is not bytes or type(chunk_size) is not int or chunk_size < 1:
            raise IngressError("INVALID", "task bytes and chunk size")
        result = ctypes.c_void_p()
        rc = self.library.env_run(self._env, task, len(task), chunk_size, ctypes.byref(result))
        try:
            if rc != 0:
                detail = (self.library.query_last_error() or b"").decode("utf-8", "replace")
                raise IngressError(_QUERY_ERRORS.get(rc, "NATIVE"), detail)
            return _copy_query(self.library, result)
        finally:
            self.library.query_destroy(result)

    def corpus_counts(self) -> tuple[int, int]:
        docs, chunks = ctypes.c_uint64(), ctypes.c_uint64()
        if not self._env or self.library.env_counts(self._env, ctypes.byref(docs), ctypes.byref(chunks)) != 0:
            raise IngressError("CORPUS", "counts")
        return int(docs.value), int(chunks.value)

    @property
    def edge_count(self) -> int:
        return int(self.library.env_edge_count(self._env)) if self._env else 0

    def close(self) -> None:
        if self._env:
            self.library.env_close(self._env)
            self._env = ctypes.c_void_p()

    def __enter__(self) -> "QueryIngress":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
