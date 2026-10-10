"""HACF structural memory from Python: a thin ctypes layer over the native bridge.

Provides:
  - ``RetrievalLibrary(path, root=..., authority=...)``: deployment-pinned, substrate-sealed native bridge (ABI v2);
  - ``ContextEdge``: one explicit, provenance-bearing context-graph edge between two HACF chunks;
  - ``build_corpus_and_index`` -> ``HacfHandle`` (owning handle, deterministic cleanup), optionally with one
    immutable context graph built once from explicit edges;
  - ``hybrid_retrieve`` -> canonical bundle JSON plus identity digests;
  - ``bundle_from_json`` -> ``RetrievalBundle``.

The library path, trusted root and independent deployment authority are all required. There is no environment
variable lookup, path discovery or repository-relative fallback. All operations are
read-only after environment creation.

Context graph (native ``elpis_context_graph``). Edges are externally supplied, admitted structural facts
(subject chunk, object chunk, provenance digest, nonzero edge type, authority 0..3); the bridge never infers one
from embeddings, lexical overlap, rank, document adjacency or model output. Chunk digests are HACF's own
content-addressed identities (learned from HACF, e.g. a retrieval bundle over the same documents). The graph is
built once per environment, owned and destroyed with it, and its identity is invariant under edge order (exact
duplicates collapse). A malformed digest, self-edge or an endpoint that is not an admitted chunk of the
environment's corpus refuses construction. Retrieval mechanics and policy stay native (``elpis_hybrid_policy``):
Python only selects its graph fields. Without a graph the environment is exactly the lexical+dense one it was."""

from __future__ import annotations

import ctypes
import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Iterable

from .contracts import RetrievalBundle, RetrievalItem, _digest
from elpis.substrate.contracts import ContractError
from elpis.structure.native_loader import load_structure_bridge
from .errors import HybridRetrievalError, RetrievalLibraryError

ELPIS_EMBEDDING_DIM = 384
BUNDLE_JSON_CAP = 1 << 18  # 256 KiB buffer for bundle JSON
BRIDGE_ABI_VERSION = 2
ZERO_DIGEST = "0" * 64      # the native identity of "no context graph"
_POLICY_DEFAULT = 0xFFFFFFFF  # ELPIS_RETRIEVAL_POLICY_DEFAULT: keep the native default of that field
_HEX = frozenset("0123456789abcdef")


class ContextEdge(ctypes.Structure):
    """One explicit context-graph edge (``elpis_context_edge_input``): subject -> object, with provenance."""

    _fields_ = [("subject_chunk_digest", ctypes.c_char * 65), ("object_chunk_digest", ctypes.c_char * 65),
                ("provenance_digest", ctypes.c_char * 65), ("edge_type", ctypes.c_uint32),
                ("authority", ctypes.c_uint32)]

    @classmethod
    def of(cls, subject: str, obj: str, provenance: str, edge_type: int, authority: int) -> "ContextEdge":
        """Typed construction; the native graph law re-checks everything at environment construction."""
        for value in (subject, obj, provenance):
            if type(value) is not str or len(value) != 64 or not set(value) <= _HEX:
                raise HybridRetrievalError("GRAPH_EDGE_INVALID", "edge digests are 64 lowercase hex")
        if type(edge_type) is not int or not 0 < edge_type < 2 ** 32:
            raise HybridRetrievalError("GRAPH_EDGE_INVALID", "edge type is a nonzero u32")
        if type(authority) is not int or not 0 <= authority <= 3:
            raise HybridRetrievalError("GRAPH_EDGE_INVALID", "edge authority is 0..3")
        return cls(subject.encode(), obj.encode(), provenance.encode(), edge_type, authority)


class _GraphPolicy(ctypes.Structure):
    _fields_ = [("graph_seed_limit", ctypes.c_uint32), ("graph_neighbors_per_seed", ctypes.c_uint32),
                ("min_graph_authority", ctypes.c_uint32)]


class RetrievalLibrary:
    """One native retrieval bridge, sealed against deployment-pinned bytes."""

    def __init__(self, path: str | Path, *, root=None, authority=None) -> None:
        path = Path(path)
        if not path.is_absolute() or not path.is_file():
            raise RetrievalLibraryError("LIB_NOT_FOUND", f"retrieval bridge not found at {path}")
        try:
            lib = load_structure_bridge(root, path, authority, 'elpis_retrieval_bridge')
        except ContractError as e:
            raise RetrievalLibraryError("LIB_AUTHORITY", str(e)) from e
        except OSError as e:
            raise RetrievalLibraryError("LIB_LOAD_FAILED", str(e)) from e
        self.path = path
        self.lib = lib

        try:
            version = lib.elpis_retrieval_bridge_abi_version
        except AttributeError as e:
            raise RetrievalLibraryError("ABI_MISMATCH", "retrieval bridge predates ABI v2") from e
        version.restype, version.argtypes = ctypes.c_uint32, []
        if version() != BRIDGE_ABI_VERSION:
            raise RetrievalLibraryError("ABI_MISMATCH", f"retrieval bridge ABI {version()} != {BRIDGE_ABI_VERSION}")

        lib.elpis_retrieval_env_create.restype = ctypes.c_void_p
        lib.elpis_retrieval_env_create.argtypes = [
            ctypes.c_char_p,                   # state_root
            ctypes.POINTER(ctypes.c_char_p),   # labels
            ctypes.POINTER(ctypes.c_char_p),   # texts
            ctypes.POINTER(ctypes.c_char_p),   # namespaces
            ctypes.POINTER(ctypes.c_char_p),   # authorities
            ctypes.c_int,                      # n_docs
            ctypes.c_int,                      # with_graph
            ctypes.POINTER(ContextEdge),       # edges
            ctypes.c_uint32,                   # edge_count
            ctypes.c_char_p,                   # error_buf[256]
        ]
        lib.elpis_retrieval_env_borrow_corpus.restype = ctypes.c_void_p
        lib.elpis_retrieval_env_borrow_corpus.argtypes = [ctypes.c_void_p]
        lib.elpis_retrieval_env_copy_document.restype = ctypes.c_int
        lib.elpis_retrieval_env_copy_document.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        lib.elpis_retrieval_env_destroy.restype = None
        lib.elpis_retrieval_env_destroy.argtypes = [ctypes.c_void_p]
        lib.elpis_retrieval_env_embed.restype = ctypes.c_int
        lib.elpis_retrieval_env_embed.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ]
        lib.elpis_retrieval_env_retrieve.restype = ctypes.c_int
        lib.elpis_retrieval_env_retrieve.argtypes = [
            ctypes.c_void_p,                   # env
            ctypes.c_char_p,                   # query_text
            ctypes.POINTER(ctypes.c_float),    # query_vector
            ctypes.c_int,                      # query_dim
            ctypes.c_uint32,                   # lexical_limit
            ctypes.c_uint32,                   # dense_limit
            ctypes.c_uint32,                   # primary_limit
            ctypes.c_uint32,                   # total_limit
            ctypes.c_char_p,                   # namespace_filter (NULL = all)
            ctypes.c_char_p,                   # authority_filter (NULL = all)
            ctypes.POINTER(_GraphPolicy),      # graph_policy (NULL = native default / disabled)
            ctypes.c_char_p,                   # bundle_json_out
            ctypes.c_int,                      # bundle_json_cap
            ctypes.c_char_p,                   # bundle_digest_out[65]
            ctypes.c_char_p,                   # query_digest_out[65]
            ctypes.c_char_p,                   # corpus_manifest_digest_out[65]
            ctypes.c_char_p,                   # vindex_manifest_digest_out[65]
            ctypes.c_char_p,                   # graph_snapshot_digest_out[65]
            ctypes.c_char_p,                   # fusion_policy_digest_out[65]
            ctypes.c_char_p,                   # hacf_package_digest_out[65]
            ctypes.POINTER(ctypes.c_int),      # item_count_out
            ctypes.c_char_p,                   # error_buf[256]
        ]
        lib.elpis_retrieval_checked_manifest_copy.restype = ctypes.c_int
        lib.elpis_retrieval_checked_manifest_copy.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_char_p,
        ]
        for getter in ("corpus_digest", "shard_digest", "corpus_manifest", "vindex_manifest", "graph_digest"):
            fn = getattr(lib, "elpis_retrieval_env_" + getter)
            fn.restype = ctypes.c_char_p
            fn.argtypes = [ctypes.c_void_p]
        lib.elpis_retrieval_env_graph_edge_count.restype = ctypes.c_uint32
        lib.elpis_retrieval_env_graph_edge_count.argtypes = [ctypes.c_void_p]


class HacfHandle:
    """Owning handle to native HACF resources with deterministic cleanup."""

    def __init__(self, library: RetrievalLibrary) -> None:
        self.library = library
        self._ptr: ctypes.c_void_p = ctypes.c_void_p(0)
        self.corpus_manifest_json = ""
        self.corpus_digest = ""
        self.shard_digest = ""
        self.vindex_manifest_json = ""
        self.graph_snapshot_digest = ZERO_DIGEST   # the environment's context graph identity (zeros: none)
        self.graph_edge_count = 0
        self._borrows = 0
        self._guard = threading.RLock()

    @property
    def _valid(self) -> bool:
        return bool(self._ptr)

    def destroy(self) -> None:
        with self._guard:
            if self._borrows:
                raise HybridRetrievalError("HANDLE_BUSY", "live borrowed ingress still owns a reference")
            if self._valid:
                self.library.lib.elpis_retrieval_env_destroy(self._ptr)
                self._ptr = ctypes.c_void_p(0)

    def _borrow_corpus(self):
        if not self._valid:
            raise HybridRetrievalError("HANDLE_CLOSED", "retrieval epoch already destroyed")
        with self._guard:
            if not self._valid:
                raise HybridRetrievalError("HANDLE_CLOSED", "retrieval epoch already destroyed")
            self._borrows += 1
            result = self.library.lib.elpis_retrieval_env_borrow_corpus(self._ptr)
            if not result:
                self._borrows -= 1
                raise HybridRetrievalError("CORPUS_MISSING", "no native corpus in retrieval epoch")
            return result

    def _release_borrow(self):
        with self._guard:
            if self._borrows <= 0:
                raise HybridRetrievalError("HANDLE_STATE", "unbalanced corpus borrow")
            self._borrows -= 1

    def read_document(self, digest: str, expected_size: int) -> bytes:
        if not self._valid:
            raise HybridRetrievalError("HANDLE_CLOSED", "retrieval epoch already destroyed")
        if type(digest) is not str or len(digest) != 64 or not set(digest) <= _HEX:
            raise HybridRetrievalError("INVALID", "document digest")
        if type(expected_size) is not int or not 0 <= expected_size <= (1 << 20):
            raise HybridRetrievalError("LIMIT", "document byte budget")
        with self._guard:
            if not self._valid:
                raise HybridRetrievalError("HANDLE_CLOSED", "retrieval epoch already destroyed")
            buf = ctypes.create_string_buffer(expected_size or 1)
            actual = ctypes.c_size_t()
            rc = self.library.lib.elpis_retrieval_env_copy_document(
                self._ptr, digest.encode("ascii"), buf, expected_size, ctypes.byref(actual))
        if rc or actual.value != expected_size:
            raise HybridRetrievalError("INTEGRITY", "volatile document read failed")
        result = buf.raw[:expected_size]
        if hashlib.sha256(result).hexdigest() != digest:
            raise HybridRetrievalError("INTEGRITY", "volatile document digest mismatch")
        return result

    def __enter__(self) -> "HacfHandle":
        return self

    def __exit__(self, *args: Any) -> None:
        self.destroy()

    def __del__(self) -> None:
        # Lifetime safety net: an epoch dropped without destroy() must not keep its native corpus, vector index
        # and shard resident for the life of the process. A live borrow keeps this handle reachable through its
        # ingress (QueryIngress._owner), so none can exist here.
        try:
            self.destroy()
        except Exception:
            pass


def _text(value: bytes | None) -> str:
    return value.decode("utf-8") if value else ""


def build_corpus_and_index(
    library: RetrievalLibrary,
    state_root: str | Path,
    documents: list[tuple[str, str, str, str]],
    edges: Iterable[ContextEdge] | None = None,
) -> HacfHandle:
    """Build a complete HACF corpus + vector index from documents, and optionally one context graph.

    Args:
        library: explicitly loaded retrieval bridge.
        state_root: directory owned by this corpus (corpus state, cold storage).
        documents: ``(label, text, namespace, authority)`` tuples.
        edges: ``None`` for no context graph (lexical + dense retrieval); otherwise the explicit
            :class:`ContextEdge` facts of the environment's one immutable graph (an empty iterable is an
            empty graph). Every endpoint must be an admitted chunk of these documents.
    """
    edge_list = None if edges is None else list(edges)
    if edge_list is not None and not all(type(e) is ContextEdge for e in edge_list):
        raise HybridRetrievalError("GRAPH_EDGE_INVALID", "edges must be ContextEdge values")
    edge_array = (ContextEdge * len(edge_list))(*edge_list) if edge_list else None
    lib = library.lib
    sorted_docs = sorted(documents, key=lambda x: x[0])
    n = len(sorted_docs)
    labels = (ctypes.c_char_p * n)(*[l.encode("utf-8") for l, _, _, _ in sorted_docs])
    texts = (ctypes.c_char_p * n)(*[t.encode("utf-8") for _, t, _, _ in sorted_docs])
    namespaces = (ctypes.c_char_p * n)(*[ns.encode("utf-8") for _, _, ns, _ in sorted_docs])
    authorities = (ctypes.c_char_p * n)(*[a.encode("utf-8") for _, _, _, a in sorted_docs])

    err_buf = ctypes.create_string_buffer(256)
    ptr = lib.elpis_retrieval_env_create(
        str(state_root).encode("utf-8"),
        labels, texts, namespaces, authorities,
        n, int(edge_list is not None), edge_array, len(edge_list or ()), err_buf,
    )
    if not ptr:
        raise HybridRetrievalError("ENV_CREATE_FAILED", _text(err_buf.value) or "unknown")
    err = _text(err_buf.value)
    if err != "ok":
        lib.elpis_retrieval_env_destroy(ptr)
        raise HybridRetrievalError("ENV_CREATE_FAILED", err)

    handle = HacfHandle(library)
    handle._ptr = ctypes.c_void_p(ptr)
    handle.corpus_manifest_json = _text(lib.elpis_retrieval_env_corpus_manifest(ptr))
    handle.corpus_digest = _text(lib.elpis_retrieval_env_corpus_digest(ptr))
    handle.shard_digest = _text(lib.elpis_retrieval_env_shard_digest(ptr))
    handle.vindex_manifest_json = _text(lib.elpis_retrieval_env_vindex_manifest(ptr))
    handle.graph_snapshot_digest = _text(lib.elpis_retrieval_env_graph_digest(ptr))
    handle.graph_edge_count = int(lib.elpis_retrieval_env_graph_edge_count(ptr))
    return handle


def hybrid_retrieve(
    handle: HacfHandle,
    query_text: str,
    query_vector: list[float] | None = None,
    lexical_limit: int = 50,
    dense_limit: int = 50,
    primary_limit: int = 30,
    total_limit: int = 60,
    *,
    namespace_filter: str | None = None,
    authority_filter: str | None = None,
    graph_seed_limit: int | None = None,
    graph_neighbors_per_seed: int | None = None,
    min_graph_authority: int | None = None,
) -> dict[str, Any]:
    """Execute deterministic hybrid retrieval against the prepared corpus/index (and graph, if any).

    ``namespace_filter`` / ``authority_filter`` are exact native query filters (``None``: all). The graph
    arguments select fields of the native ``elpis_hybrid_policy``; ``None`` keeps the native default (seed limit
    bounded by ``primary_limit``). They are refused without a context graph, whose policy keeps them disabled.

    Returns bundle JSON, its digests (the graph snapshot and HACF package digests as the native bundle carries
    them), item count and the parsed bundle data.
    """
    if not handle._valid:
        raise HybridRetrievalError("NO_ENVIRONMENT", "Handle has no valid corpus/index")
    lib = handle.library.lib
    graph_fields = (graph_seed_limit, graph_neighbors_per_seed, min_graph_authority)
    graph_policy = None
    if any(v is not None for v in graph_fields):
        if not all(v is None or (type(v) is int and 0 <= v < _POLICY_DEFAULT) for v in graph_fields):
            raise HybridRetrievalError("POLICY_INVALID", "graph policy fields are u32 values")
        graph_policy = ctypes.byref(_GraphPolicy(*(_POLICY_DEFAULT if v is None else v for v in graph_fields)))
    filters = []
    for value in (namespace_filter, authority_filter):
        if value is not None and type(value) is not str:
            raise HybridRetrievalError("QUERY_INVALID", "filters are str or None")
        filters.append(None if value is None else value.encode("utf-8"))

    if query_vector is None:
        vec = (ctypes.c_float * ELPIS_EMBEDDING_DIM)()
        encoded = query_text.encode("utf-8")
        rc = lib.elpis_retrieval_env_embed(handle._ptr, encoded, len(encoded), vec, ELPIS_EMBEDDING_DIM)
        if rc != 0:
            raise HybridRetrievalError("EMBED_QUERY_FAILED", f"rc={rc}")
        query_vector = list(vec)

    vec_ptr = (ctypes.c_float * len(query_vector))(*query_vector)
    json_buf = ctypes.create_string_buffer(BUNDLE_JSON_CAP)
    bundle_digest_buf = ctypes.create_string_buffer(65)
    query_digest_buf = ctypes.create_string_buffer(65)
    corpus_manifest_digest_buf = ctypes.create_string_buffer(65)
    vindex_manifest_digest_buf = ctypes.create_string_buffer(65)
    graph_snapshot_digest_buf = ctypes.create_string_buffer(65)
    fusion_policy_digest_buf = ctypes.create_string_buffer(65)
    hacf_package_digest_buf = ctypes.create_string_buffer(65)
    item_count = ctypes.c_int(0)
    err_buf = ctypes.create_string_buffer(256)

    rc = lib.elpis_retrieval_env_retrieve(
        handle._ptr,
        query_text.encode("utf-8"),
        vec_ptr, len(query_vector),
        lexical_limit, dense_limit,
        primary_limit, total_limit,
        filters[0], filters[1], graph_policy,
        json_buf, BUNDLE_JSON_CAP,
        bundle_digest_buf,
        query_digest_buf,
        corpus_manifest_digest_buf,
        vindex_manifest_digest_buf,
        graph_snapshot_digest_buf,
        fusion_policy_digest_buf,
        hacf_package_digest_buf,
        ctypes.byref(item_count),
        err_buf,
    )
    if rc != 0:
        raise HybridRetrievalError(f"HACF_RETRIEVE_{rc}", _text(err_buf.value) or "unknown")

    json_str = _text(json_buf.value)
    data = json.loads(json_str) if json_str else {}
    graph_digest = _text(graph_snapshot_digest_buf.value)
    if data.get("graph_snapshot_digest") != graph_digest or graph_digest != handle.graph_snapshot_digest:
        raise HybridRetrievalError("IDENTITY_MISMATCH", "bundle graph identity is not the environment's graph")
    return {
        "bundle_json": json_str,
        "bundle_digest": _text(bundle_digest_buf.value),
        "item_count": item_count.value,
        "query_digest": _text(query_digest_buf.value),
        "corpus_manifest_digest": _text(corpus_manifest_digest_buf.value),
        "vector_index_manifest_digest": _text(vindex_manifest_digest_buf.value),
        "graph_snapshot_digest": graph_digest,
        "fusion_policy_digest": _text(fusion_policy_digest_buf.value),
        "hacf_package_digest": _text(hacf_package_digest_buf.value),
        "data": data,
    }


def get_vector_index_manifest(handle: HacfHandle) -> tuple[str, str]:
    """Vector index manifest JSON and its SHA-256."""
    j = _text(handle.library.lib.elpis_retrieval_env_vindex_manifest(handle._ptr))
    return j, (hashlib.sha256(j.encode("utf-8")).hexdigest() if j else "")


def bundle_from_json(bundle_json_str: str, metadata: dict[str, str]) -> RetrievalBundle:
    """Parse native bundle JSON into a RetrievalBundle (hex text/namespace decoded)."""
    data = json.loads(bundle_json_str)

    items: list[RetrievalItem] = []
    for item in data.get("items", []):
        raw_text = item.get("text") or ""
        if not raw_text and item.get("text_hex"):
            raw_text = bytes.fromhex(item["text_hex"]).decode("utf-8", errors="replace")

        raw_ns = item.get("namespace") or item.get("namespace_hex") or ""
        if raw_ns.startswith("0x"):
            raw_ns = raw_ns[2:]
        if item.get("namespace_hex") and not item.get("namespace"):
            try:
                raw_ns = bytes.fromhex(item["namespace_hex"]).decode("utf-8", errors="replace")
            except ValueError:
                pass

        items.append(RetrievalItem(
            chunk_digest=item.get("chunk_digest", ""),
            doc_digest=item.get("doc_digest", ""),
            namespace=raw_ns,
            authority=item.get("authority", ""),
            graph_parent_digest=item.get("graph_parent_digest", "0" * 64),
            text_digest=item.get("text_digest", ""),
            fusion_score_key=int(item.get("fusion_score_key", 0)),
            dense_score_key=int(item.get("dense_score_key", 0)),
            lexical_rank=int(item.get("lexical_rank", 0)),
            dense_rank=int(item.get("dense_rank", 0)),
            final_rank=int(item.get("final_rank", 0)),
            source_mask=int(item.get("source_mask", 0)),
            item_kind=int(item.get("item_kind", 1)),
            graph_hop=int(item.get("graph_hop", 0)),
            edge_type=int(item.get("edge_type", 0)),
            edge_authority=int(item.get("edge_authority", 0)),
            text=raw_text,
            text_bytes=int(item.get("text_bytes", len(raw_text.encode("utf-8")))),
        ))

    bundle_digest = metadata.get("bundle_digest", "") or _digest(data)
    return RetrievalBundle(
        schema=data.get("schema", "elpis.retrieval_bundle.v1"),
        query_digest=metadata.get("query_digest", ""),
        corpus_manifest_digest=metadata.get("corpus_manifest_digest", ""),
        vector_index_manifest_digest=metadata.get("vector_index_manifest_digest", ""),
        graph_snapshot_digest=metadata.get("graph_snapshot_digest", ""),
        fusion_policy_digest=metadata.get("fusion_policy_digest", ""),
        bundle_digest=bundle_digest,
        hacf_package_digest=metadata.get("hacf_package_digest", ""),
        corpus_epoch=int(metadata.get("corpus_epoch", 0)),
        vector_index_epoch=int(metadata.get("vector_index_epoch", 0)),
        items=tuple(items),
    )
