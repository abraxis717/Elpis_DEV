"""Structural retrieval: query derivation, native hybrid retrieval, bundle
validation, budgets and evidence envelopes (HACF structural memory -> evidence)."""
from __future__ import annotations

import ctypes

import pytest

from elpis.structure.retrieval.budget import RetrievalBudget, check_budget
from elpis.structure.retrieval.contracts import (
    RetrievalBundle,
    RetrievalItem,
    _digest,
    _sha256_hex,
)
from elpis.structure.retrieval.errors import (
    BudgetOverflowError,
    BundleValidationError,
    HybridRetrievalError,
    QueryDerivationError,
    RetrievalLibraryError,
)
from elpis.structure.retrieval import hacf
from elpis.structure.retrieval.hacf import (
    RetrievalLibrary,
    build_corpus_and_index,
    bundle_from_json,
    hybrid_retrieve,
)
from elpis.structure.retrieval.evidence import build_evidence_envelope, evidence_envelope_digest
from elpis.structure.retrieval.query import derive_query
from elpis.structure.retrieval.validation import validate_bundle

from ..conftest import require_native_library

DOCS = [
    ("alpha", "alpha engine exact retrieval anchor", "elpis.docs", "canonical"),
    ("beta", "beta companion context bridge", "elpis.docs", "reference"),
    ("gamma", "gamma vector semantic neighbor", "elpis.code", "canonical"),
    ("delta", "delta unrelated background note", "elpis.notes", "advisory"),
]

BUDGET = RetrievalBudget()


@pytest.fixture(scope="module")
def library():
    return RetrievalLibrary(require_native_library("elpis_retrieval_bridge"))


@pytest.fixture(scope="module")
def handle(library, tmp_path_factory):
    h = build_corpus_and_index(library, tmp_path_factory.mktemp("hacf"), DOCS)
    yield h
    h.destroy()


def _item(text: str, **kw) -> RetrievalItem:
    """Build a RetrievalItem with all required fields, overriding via kw."""
    td = _sha256_hex(text.encode("utf-8")) if text else ""
    defaults = dict(
        chunk_digest="a" * 64,
        doc_digest="b" * 64,
        namespace="elpis.docs",
        authority="canonical",
        graph_parent_digest="0" * 64,
        text_digest=td,
        fusion_score_key=100,
        dense_score_key=50,
        lexical_rank=1,
        dense_rank=1,
        final_rank=0,
        source_mask=3,
        item_kind=1,
        graph_hop=0,
        edge_type=0,
        edge_authority=0,
        text=text,
        text_bytes=len(text.encode("utf-8")) if text else 0,
    )
    defaults.update(kw)
    return RetrievalItem(**defaults)


# ====================================================================
# Query derivation
# ====================================================================
class TestQueryDerivation:
    def test_basic_derivation(self):
        req = {
            "request_id": "retrieval_test_001",
            "prompt": "def solution(x): return x + 1",
            "domain": "python",
            "entrypoint": "solution",
        }
        q = derive_query(req)
        assert q.query_text != ""
        assert len(q.query_text.encode("utf-8")) <= 4096
        assert q.source_request_digest == _digest(req)

    def test_deterministic(self):
        req = {
            "request_id": "retrieval_det",
            "prompt": "test determinism",
            "domain": "python",
            "entrypoint": "main",
        }
        q1 = derive_query(req)
        q2 = derive_query(req)
        assert q1.query_digest == q2.query_digest

    def test_missing_field(self):
        with pytest.raises(QueryDerivationError, match="MISSING_FIELD"):
            derive_query({"request_id": "x"})

    def test_empty_after_normalization(self):
        req = {
            "request_id": "   ",
            "prompt": "   ",
            "domain": "   ",
            "entrypoint": "   ",
        }
        with pytest.raises(QueryDerivationError, match="EMPTY_QUERY"):
            derive_query(req)


# ====================================================================
# HACF retrieval (positive)
# ====================================================================
class TestHacfRetrieval:

    def test_retrieval_returns_bundle(self, handle):
        r = hybrid_retrieve(
            handle, "alpha engine", lexical_limit=50, dense_limit=50,
            primary_limit=30, total_limit=60,
        )
        assert r["item_count"] > 0
        assert r["bundle_json"] != ""
        assert r["query_digest"] != ""

    def test_bundle_schema(self, handle):
        r = hybrid_retrieve(handle, "alpha", lexical_limit=50, dense_limit=50)
        data = r["data"]
        assert data.get("schema") == "elpis.retrieval_bundle.v1"

    def test_bundle_has_items(self, handle):
        r = hybrid_retrieve(handle, "alpha", lexical_limit=50, dense_limit=50)
        assert len(r["data"].get("items", [])) > 0

    def test_bundle_items_have_text(self, handle):
        r = hybrid_retrieve(handle, "alpha", lexical_limit=50, dense_limit=50)
        for item in r["data"].get("items", []):
            assert item.get("chunk_digest"), "item missing chunk_digest"
            assert item.get("text_bytes", 0) > 0, "item has zero text_bytes"

    def test_deterministic_retrieval(self, handle):
        r1 = hybrid_retrieve(handle, "alpha", lexical_limit=50, dense_limit=50)
        r2 = hybrid_retrieve(handle, "alpha", lexical_limit=50, dense_limit=50)
        assert r1["bundle_digest"] == r2["bundle_digest"]

    def test_65_documents_refused_with_typed_limit(self, library, tmp_path):
        td = tmp_path / "limit"
        documents = [
            (
                f"doc-{i:02d}",
                f"bounded document {i}",
                "elpis.docs",
                "canonical",
            )
            for i in range(65)
        ]
        with pytest.raises(HybridRetrievalError, match="E_LIMIT"):
            build_corpus_and_index(library, td, documents)

    def test_manifest_copy_boundary_via_compiled_ctypes_helper(self, library):
        lib = library.lib

        capacity = 65536
        dst = ctypes.create_string_buffer(capacity)
        out_len = ctypes.c_size_t(0)
        err = ctypes.create_string_buffer(256)

        allowed = b"a" * 65535
        rc = lib.elpis_retrieval_checked_manifest_copy(
            allowed,
            dst,
            capacity,
            ctypes.byref(out_len),
            err,
        )
        assert rc == 0
        assert out_len.value == 65535
        assert dst.raw[:65535] == allowed
        assert dst.raw[65535] == 0

        refused = b"b" * 65536
        dst2 = ctypes.create_string_buffer(capacity)
        out_len2 = ctypes.c_size_t(999)
        err2 = ctypes.create_string_buffer(256)
        rc = lib.elpis_retrieval_checked_manifest_copy(
            refused,
            dst2,
            capacity,
            ctypes.byref(out_len2),
            err2,
        )
        assert rc == -2
        assert out_len2.value == 0
        assert b"E_LIMIT" in err2.value
        assert dst2.raw[0] == 0

    def test_retrieval_json_buffer_truncation_is_refused(self, handle):
        lib = handle.library.lib

        query = b"alpha"
        vec = (ctypes.c_float * hacf.ELPIS_EMBEDDING_DIM)()
        rc = lib.elpis_retrieval_env_embed(
            handle._ptr,
            query,
            len(query),
            vec,
            hacf.ELPIS_EMBEDDING_DIM,
        )
        assert rc == 0

        json_buf = ctypes.create_string_buffer(1)
        bundle_digest = ctypes.create_string_buffer(65)
        query_digest = ctypes.create_string_buffer(65)
        corpus_manifest_digest = ctypes.create_string_buffer(65)
        vindex_manifest_digest = ctypes.create_string_buffer(65)
        fusion_policy_digest = ctypes.create_string_buffer(65)
        item_count = ctypes.c_int(0)
        err = ctypes.create_string_buffer(256)

        rc = lib.elpis_retrieval_env_retrieve(
            handle._ptr,
            query,
            vec,
            hacf.ELPIS_EMBEDDING_DIM,
            50,
            50,
            30,
            60,
            json_buf,
            1,
            bundle_digest,
            query_digest,
            corpus_manifest_digest,
            vindex_manifest_digest,
            fusion_policy_digest,
            ctypes.byref(item_count),
            err,
        )

        assert rc == -2
        assert b"E_LIMIT" in err.value
        assert json_buf.raw == b"\x00"


# ====================================================================
# Bundle validation (positive + negative)
# ====================================================================
class TestBundleValidation:
    def _make_bundle(self, items):
        return RetrievalBundle(
            schema="elpis.retrieval_bundle.v1",
            query_digest="q" * 64,
            corpus_manifest_digest="c" * 64,
            items=tuple(items),
        )

    def test_valid_bundle(self):
        text = "alpha engine exact retrieval anchor"
        td = _sha256_hex(text.encode("utf-8"))
        items = [_item(text, chunk_digest="a" * 64, text_digest=td)]
        b = self._make_bundle(items)
        decision = validate_bundle(b, "q" * 64, "c" * 64, BUDGET)
        assert decision is not None
        assert not decision.exceeded

    def test_unknown_schema(self):
        b = RetrievalBundle(schema="bad.schema")
        with pytest.raises(BundleValidationError, match="UNKNOWN_SCHEMA"):
            validate_bundle(b, "q" * 64, "c" * 64)

    def test_query_digest_mismatch(self):
        b = self._make_bundle([])
        with pytest.raises(BundleValidationError, match="QUERY_DIGEST_MISMATCH"):
            validate_bundle(b, "x" * 64, "c" * 64)

    def test_corpus_digest_mismatch(self):
        b = self._make_bundle([])
        with pytest.raises(BundleValidationError, match="CORPUS_DIGEST_MISMATCH"):
            validate_bundle(b, "q" * 64, "x" * 64)

    def test_rank_order_mismatch(self):
        items = [_item("test", final_rank=5)]
        b = self._make_bundle(items)
        with pytest.raises(BundleValidationError, match="RANK_ORDER"):
            validate_bundle(b, "q" * 64, "c" * 64)

    def test_missing_chunk_digest(self):
        items = [_item("test", chunk_digest="")]
        b = self._make_bundle(items)
        with pytest.raises(BundleValidationError, match="MISSING_CHUNK_DIGEST"):
            validate_bundle(b, "q" * 64, "c" * 64)

    def test_missing_text(self):
        items = [_item("", chunk_digest="a" * 64)]
        b = self._make_bundle(items)
        with pytest.raises(BundleValidationError, match="MISSING_FROZEN_TEXT"):
            validate_bundle(b, "q" * 64, "c" * 64)

    def test_duplicate_chunk(self):
        text = "test"
        td = _sha256_hex(text.encode())
        items = [
            _item(text, chunk_digest="x" * 64, text_digest=td),
            _item(text, chunk_digest="x" * 64, final_rank=1, text_digest=td),
        ]
        b = self._make_bundle(items)
        with pytest.raises(BundleValidationError, match="DUPLICATE_CHUNK"):
            validate_bundle(b, "q" * 64, "c" * 64)

    def test_context_beyond_one_hop(self):
        items = [_item("test", graph_hop=2)]
        b = self._make_bundle(items)
        with pytest.raises(BundleValidationError, match="CONTEXT"):
            validate_bundle(b, "q" * 64, "c" * 64)


# ====================================================================
# Budget enforcement
# ====================================================================
class TestBudget:
    def test_within_budget(self):
        decision = check_budget(
            BUDGET,
            {"lexical": 10, "dense": 10, "fused": 20, "context": 0, "total": 20},
            500,
        )
        assert not decision.exceeded

    def test_overflow(self):
        with pytest.raises(BudgetOverflowError, match="BUDGET_EXCEEDED"):
            check_budget(
                BUDGET,
                {"lexical": 0, "dense": 0, "fused": 50, "context": 0, "total": 300},
                500,
            )


# ====================================================================
# Library loading is explicit
# ====================================================================
def test_library_path_must_be_explicit_and_absolute(tmp_path):
    with pytest.raises(RetrievalLibraryError, match="LIB_NOT_FOUND"):
        RetrievalLibrary("libelpis_retrieval_bridge.so")
    with pytest.raises(RetrievalLibraryError, match="LIB_NOT_FOUND"):
        RetrievalLibrary(tmp_path / "missing.so")


# ====================================================================
# Structural memory -> validated bundle -> evidence envelope
# ====================================================================
def test_retrieval_to_evidence_envelope_end_to_end(library, tmp_path):
    with build_corpus_and_index(library, tmp_path / "hacf", DOCS) as handle:
        request = {"request_id": "e2e", "prompt": "alpha engine anchor",
                   "domain": "elpis.docs", "entrypoint": "retrieve"}
        query = derive_query(request)
        result = hybrid_retrieve(handle, query.query_text)
        bundle = bundle_from_json(result["bundle_json"], result)
        decision = validate_bundle(bundle, result["query_digest"],
                                   result["corpus_manifest_digest"], BUDGET)
        assert decision is not None and not decision.exceeded
        envelope = build_evidence_envelope(_digest(request), query.query_digest, bundle,
                                           _digest(decision.to_canonical_dict()))
        assert envelope.evidence_references == tuple(i.chunk_digest for i in bundle.items)
        assert envelope.evidence_texts[0]
        # Same structural memory + same query -> identical observation identity.
        again = bundle_from_json(hybrid_retrieve(handle, query.query_text)["bundle_json"], result)
        envelope2 = build_evidence_envelope(_digest(request), query.query_digest, again,
                                            _digest(decision.to_canonical_dict()))
        assert evidence_envelope_digest(envelope) == evidence_envelope_digest(envelope2)


def test_evidence_envelope_refuses_empty_bundle():
    from elpis.structure.retrieval.errors import EvidenceError
    with pytest.raises(EvidenceError, match="EMPTY_BUNDLE"):
        build_evidence_envelope("a" * 64, "b" * 64, RetrievalBundle(), "c" * 64)
