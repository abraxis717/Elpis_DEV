"""HACF structural memory from Python: a thin ctypes layer over the native bridge.

Provides:
  - ``RetrievalLibrary(path)``: explicitly loaded ``libelpis_retrieval_bridge``;
  - ``build_corpus_and_index`` -> ``HacfHandle`` (owning handle, deterministic cleanup);
  - ``hybrid_retrieve`` -> canonical bundle JSON plus identity digests;
  - ``bundle_from_json`` -> ``RetrievalBundle``.

The library path is always supplied by the caller. There is no environment
variable lookup and no repository-relative fallback. All operations are
read-only after corpus/index creation.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import RetrievalBundle, RetrievalItem, _digest
from .errors import HybridRetrievalError, RetrievalLibraryError

ELPIS_EMBEDDING_DIM = 384
BUNDLE_JSON_CAP = 1 << 18  # 256 KiB buffer for bundle JSON


class RetrievalLibrary:
    """One explicitly loaded native retrieval bridge."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.is_absolute() or not path.is_file():
            raise RetrievalLibraryError("LIB_NOT_FOUND", f"retrieval bridge not found at {path}")
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as e:
            raise RetrievalLibraryError("LIB_LOAD_FAILED", str(e)) from e
        self.path = path
        self.lib = lib

        lib.elpis_retrieval_env_create.restype = ctypes.c_void_p
        lib.elpis_retrieval_env_create.argtypes = [
            ctypes.c_char_p,                   # state_root
            ctypes.POINTER(ctypes.c_char_p),   # labels
            ctypes.POINTER(ctypes.c_char_p),   # texts
            ctypes.POINTER(ctypes.c_char_p),   # namespaces
            ctypes.POINTER(ctypes.c_char_p),   # authorities
            ctypes.c_int,                      # n_docs
            ctypes.c_char_p,                   # error_buf[256]
        ]
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
            ctypes.c_char_p,                   # bundle_json_out
            ctypes.c_int,                      # bundle_json_cap
            ctypes.c_char_p,                   # bundle_digest_out[65]
            ctypes.c_char_p,                   # query_digest_out[65]
            ctypes.c_char_p,                   # corpus_manifest_digest_out[65]
            ctypes.c_char_p,                   # vindex_manifest_digest_out[65]
            ctypes.c_char_p,                   # fusion_policy_digest_out[65]
            ctypes.POINTER(ctypes.c_int),      # item_count_out
            ctypes.c_char_p,                   # error_buf[256]
        ]
        lib.elpis_retrieval_checked_manifest_copy.restype = ctypes.c_int
        lib.elpis_retrieval_checked_manifest_copy.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_char_p,
        ]
        for getter in ("corpus_digest", "shard_digest", "corpus_manifest", "vindex_manifest"):
            fn = getattr(lib, "elpis_retrieval_env_" + getter)
            fn.restype = ctypes.c_char_p
            fn.argtypes = [ctypes.c_void_p]


class HacfHandle:
    """Owning handle to native HACF resources with deterministic cleanup."""

    def __init__(self, library: RetrievalLibrary) -> None:
        self.library = library
        self._ptr: ctypes.c_void_p = ctypes.c_void_p(0)
        self.corpus_manifest_json = ""
        self.corpus_digest = ""
        self.shard_digest = ""
        self.vindex_manifest_json = ""

    @property
    def _valid(self) -> bool:
        return bool(self._ptr)

    def destroy(self) -> None:
        if self._valid:
            self.library.lib.elpis_retrieval_env_destroy(self._ptr)
            self._ptr = ctypes.c_void_p(0)

    def __enter__(self) -> "HacfHandle":
        return self

    def __exit__(self, *args: Any) -> None:
        self.destroy()


def _text(value: bytes | None) -> str:
    return value.decode("utf-8") if value else ""


def build_corpus_and_index(
    library: RetrievalLibrary,
    state_root: str | Path,
    documents: list[tuple[str, str, str, str]],
) -> HacfHandle:
    """Build a complete HACF corpus + vector index from documents.

    Args:
        library: explicitly loaded retrieval bridge.
        state_root: directory owned by this corpus (corpus state, cold storage).
        documents: ``(label, text, namespace, authority)`` tuples.
    """
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
        n, err_buf,
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
    return handle


def hybrid_retrieve(
    handle: HacfHandle,
    query_text: str,
    query_vector: list[float] | None = None,
    lexical_limit: int = 50,
    dense_limit: int = 50,
    primary_limit: int = 30,
    total_limit: int = 60,
) -> dict[str, Any]:
    """Execute deterministic hybrid retrieval against the prepared corpus/index.

    Returns bundle JSON, its digests, item count and the parsed bundle data.
    """
    if not handle._valid:
        raise HybridRetrievalError("NO_ENVIRONMENT", "Handle has no valid corpus/index")
    lib = handle.library.lib

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
    fusion_policy_digest_buf = ctypes.create_string_buffer(65)
    item_count = ctypes.c_int(0)
    err_buf = ctypes.create_string_buffer(256)

    rc = lib.elpis_retrieval_env_retrieve(
        handle._ptr,
        query_text.encode("utf-8"),
        vec_ptr, len(query_vector),
        lexical_limit, dense_limit,
        primary_limit, total_limit,
        json_buf, BUNDLE_JSON_CAP,
        bundle_digest_buf,
        query_digest_buf,
        corpus_manifest_digest_buf,
        vindex_manifest_digest_buf,
        fusion_policy_digest_buf,
        ctypes.byref(item_count),
        err_buf,
    )
    if rc != 0:
        raise HybridRetrievalError(f"HACF_RETRIEVE_{rc}", _text(err_buf.value) or "unknown")

    json_str = _text(json_buf.value)
    return {
        "bundle_json": json_str,
        "bundle_digest": _text(bundle_digest_buf.value),
        "item_count": item_count.value,
        "query_digest": _text(query_digest_buf.value),
        "corpus_manifest_digest": _text(corpus_manifest_digest_buf.value),
        "vector_index_manifest_digest": _text(vindex_manifest_digest_buf.value),
        "graph_snapshot_digest": "",  # no context graph in this bridge
        "fusion_policy_digest": _text(fusion_policy_digest_buf.value),
        "hacf_package_digest": "",
        "data": json.loads(json_str) if json_str else {},
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
