"""HACF context-graph activation through the Python-facing retrieval bridge (native HACF, bridge ABI v2).

Every retrieval here runs the real native corpus, vector index, context graph and hybrid retriever. Edges are
explicit test facts between admitted chunks (their provenance digests name the fact); nothing is inferred from
text, embeddings or ranks. The graph-disabled environment is pinned to the identities the ABI v1 bridge produced
for the same documents and queries (captured from the v1 bridge on main at 9d9baab): enabling graph support
changed none of them.
"""
from __future__ import annotations

import ctypes
import hashlib
import json

import pytest

from elpis.structure.retrieval.errors import BundleValidationError, HybridRetrievalError, RetrievalLibraryError
from elpis.structure.retrieval.hacf import (
    ZERO_DIGEST,
    ContextEdge,
    RetrievalLibrary,
    build_corpus_and_index,
    bundle_from_json,
    hybrid_retrieve,
)
from elpis.structure.retrieval.validation import validate_bundle

from ..conftest import require_native_library

DOCS = [
    ("alpha", "alpha engine exact retrieval anchor", "elpis.docs", "canonical"),
    ("beta", "beta companion context bridge", "elpis.docs", "reference"),
    ("delta", "delta unrelated background note", "elpis.notes", "advisory"),
    ("epsilon", "epsilon isolated appendix record", "elpis.docs", "canonical"),
    ("gamma", "gamma vector semantic neighbor", "elpis.code", "canonical"),
    ("zeta", "zeta separate ledger entry", "elpis.notes", "reference"),
]
TEXT = {label: text for label, text, _, _ in DOCS}
LIMITS = {"default": (50, 50, 30, 60), "tight": (1, 1, 1, 4)}
GRAPH_BIT, LEXICAL_BIT, DENSE_BIT = 0x04, 0x01, 0x02

# Graph-disabled identities of the ABI v1 bridge (main 9d9baab), same documents and queries: the bundle digest
# is the SHA-256 of the canonical bundle JSON, so it pins every item, rank, mask and frozen text as well.
V1_CORPUS = "b58a0a8427f4b35e5240a1675767893a064e21eea753238127f11b95e95b8d4d"
V1_VECTOR_INDEX = "d7f4c974b2baacfc6f1c0fbce6eb47b39ba06dfa580f63f277426fd9a177d79f"
V1_POLICY = {"default": "c8252b0078a38a1bd21d5bafcdbc0f0990e0049fdc71707ec07a6affc8f9d22f",
             "tight": "7eb61820f6f762bd1adf4d502adf43d5cf4ffe66d72c3c57e9105d0e1a7fc53c"}
V1_QUERY = {"alpha": "c4901f1b159dd116b0b625333fa4d5c4985bebf7476ae408d20259069fd8ea91",
            "gamma vector": "ddd9c59a7b7a6a792562dd31964b1caf3ff4846c811eb38ef163dac723a73885",
            "zeta ledger": "e96ddc68138da39af3338dcef7e2b38e76f7aa39f25aa02f58027be27bab6195"}
V1_BUNDLE = {
    ("alpha", "default"): "620b066b92871b3a6eeff53222bc8167c33025157633dd12d01fd912bc4a5052",
    ("alpha", "tight"): "ef876e3794ef083d8da5c900788cb6b44925818ffb74aed0bc2d6409628582f4",
    ("gamma vector", "default"): "12f13f80effb634fa48b9a5f730a368bec28b9f46f74c2c1f49165dfe9286a08",
    ("gamma vector", "tight"): "5744c2083ed7215bbb573e0100bfa3b6069096e102ee3375231336e8c3d14929",
    ("zeta ledger", "default"): "b3e299e3b3f169829e521f57f494d5258b14134c19710aae8cdda195fb50feb6",
    ("zeta ledger", "tight"): "e1d904580218e706523900240f31c9cf2e06ad4e95965cd2a348095f19bb2d96",
}


@pytest.fixture(scope="module")
def library():
    return RetrievalLibrary(require_native_library("elpis_retrieval_bridge"))


@pytest.fixture(scope="module")
def plain(library, tmp_path_factory):
    with build_corpus_and_index(library, tmp_path_factory.mktemp("plain"), DOCS) as handle:
        yield handle


@pytest.fixture(scope="module")
def chunk(plain):
    """label -> HACF chunk digest, learned from HACF itself (manifest origin -> doc digest -> bundle chunk)."""
    labels = {d["digest"]: d["origin"] for d in json.loads(plain.corpus_manifest_json)["documents"]}
    items = hybrid_retrieve(plain, "alpha")["data"]["items"]
    out = {labels[i["doc_digest"]]: i["chunk_digest"] for i in items}
    assert sorted(out) == sorted(TEXT)
    return out


def _fact(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def _edge(chunk, subject, obj, edge_type=1, authority=2, provenance=None):
    return ContextEdge.of(chunk[subject], chunk[obj], provenance or _fact(subject, obj, edge_type), edge_type,
                          authority)


def _run(handle, query, limits="tight", **kw):
    lexical, dense, primary, total = limits if isinstance(limits, tuple) else LIMITS[limits]
    return hybrid_retrieve(handle, query, lexical_limit=lexical, dense_limit=dense, primary_limit=primary,
                           total_limit=total, **kw)


def _items(result):
    return result["data"]["items"]


def _text(item):
    return item["text"] if item.get("text") else bytes.fromhex(item["text_hex"]).decode()


def _context(result, chunk):
    by_digest = {v: k for k, v in chunk.items()}
    return [by_digest[i["chunk_digest"]] for i in _items(result) if i["item_kind"] == 2]


def _primaries(result, chunk):
    by_digest = {v: k for k, v in chunk.items()}
    return [by_digest[i["chunk_digest"]] for i in _items(result) if i["item_kind"] == 1]


def _graph(library, tmp_path, edges, docs=DOCS):
    return build_corpus_and_index(library, tmp_path, docs, edges)


# -- graph absent: the lexical + dense baseline is unchanged ----------------------------------------------------

@pytest.mark.parametrize("query,limits", sorted(V1_BUNDLE))
def test_graph_disabled_retrieval_equals_the_v1_bridge(plain, query, limits):
    r = _run(plain, query, limits)
    assert (r["bundle_digest"], r["query_digest"], r["fusion_policy_digest"]) == (
        V1_BUNDLE[query, limits], V1_QUERY[query], V1_POLICY[limits])
    assert hashlib.sha256(r["bundle_json"].encode()).hexdigest() == V1_BUNDLE[query, limits]
    assert (r["corpus_manifest_digest"], r["vector_index_manifest_digest"]) == (V1_CORPUS, V1_VECTOR_INDEX)
    # The no-graph identity is the native one the bundle itself carries (64 zeros), not a placeholder.
    assert r["graph_snapshot_digest"] == r["data"]["graph_snapshot_digest"] == ZERO_DIGEST
    assert plain.graph_snapshot_digest == ZERO_DIGEST and plain.graph_edge_count == 0
    assert len(r["hacf_package_digest"]) == 64 and r["hacf_package_digest"] != ZERO_DIGEST
    assert all(i["item_kind"] == 1 and i["graph_hop"] == 0 and not i["source_mask"] & GRAPH_BIT for i in _items(r))


def test_graph_policy_is_refused_without_a_graph(plain):
    with pytest.raises(HybridRetrievalError) as info:
        _run(plain, "alpha", graph_seed_limit=1)
    assert info.value.code == "HACF_RETRIEVE_-3" and "without a context graph" in info.value.detail


def test_an_empty_graph_has_an_identity_and_expands_nothing(library, tmp_path, plain):
    with _graph(library, tmp_path, ()) as empty:
        assert empty.graph_edge_count == 0 and empty.graph_snapshot_digest != ZERO_DIGEST
        r, base = _run(empty, "alpha"), _run(plain, "alpha")
        assert r["graph_snapshot_digest"] == empty.graph_snapshot_digest
        assert [i["chunk_digest"] for i in _items(r)] == [i["chunk_digest"] for i in _items(base)]
        assert r["query_digest"] == base["query_digest"] and r["bundle_digest"] != base["bundle_digest"]


# -- graph present: bounded one-hop context ---------------------------------------------------------------------

def _outside_pool(plain, chunk, query="alpha"):
    """Labels that are no fusion candidate of the tight query (neither its lexical nor its dense top hit)."""
    pool = set(_primaries(_run(plain, query, (1, 1, 2, 2)), chunk))
    return [label for label in sorted(TEXT) if label not in pool]


def test_one_hop_context_carries_parent_edge_and_graph_identity(library, tmp_path, plain, chunk):
    near, far = _outside_pool(plain, chunk)[:2]
    edges = [_edge(chunk, "alpha", near, 3, 2), _edge(chunk, near, far, 1, 3)]   # a two-hop chain
    with _graph(library, tmp_path, edges) as g:
        r, base = _run(g, "alpha"), _run(plain, "alpha")
        assert _primaries(r, chunk) == _primaries(base, chunk) == ["alpha"]
        assert _context(r, chunk) == [near]   # far is two hops away: never expanded
        ctx = _items(r)[1]
        assert (ctx["graph_parent_digest"], ctx["graph_hop"], ctx["edge_type"], ctx["edge_authority"]) == (
            chunk["alpha"], 1, 3, 2)
        assert ctx["source_mask"] == GRAPH_BIT and (ctx["lexical_rank"], ctx["dense_rank"]) == (0, 0)
        assert _text(ctx) == TEXT[near]   # frozen from the corpus, never from the edge
        assert ctx["final_rank"] == 1
        # The graph's identity is the bundle's, and it is bound into the bundle and package digests.
        assert r["graph_snapshot_digest"] == r["data"]["graph_snapshot_digest"] == g.graph_snapshot_digest
        assert g.graph_snapshot_digest != ZERO_DIGEST and g.graph_edge_count == 2
        assert r["bundle_digest"] != base["bundle_digest"] and r["hacf_package_digest"] != base["hacf_package_digest"]
        assert (r["query_digest"], r["corpus_manifest_digest"], r["vector_index_manifest_digest"]) == (
            base["query_digest"], base["corpus_manifest_digest"], base["vector_index_manifest_digest"])
        # The native default graph policy (seed limit bounded by the primary limit) is a policy identity.
        assert r["fusion_policy_digest"] != base["fusion_policy_digest"]
        # The Python RetrievalBundle and the validation gate carry the same identity.
        bundle = bundle_from_json(r["bundle_json"], r)
        assert bundle.graph_snapshot_digest == g.graph_snapshot_digest
        assert bundle.items[1].graph_parent_digest == chunk["alpha"] and bundle.items[1].text == TEXT[near]
        validate_bundle(bundle, r["query_digest"], r["corpus_manifest_digest"])


def _two_primaries(plain, chunk):
    limits = (2, 2, 2, 12)
    return limits, _primaries(_run(plain, "alpha", limits), chunk)


def test_seed_limit_bounds_which_primaries_expand(library, tmp_path, plain, chunk):
    limits, (first, second) = _two_primaries(plain, chunk)
    rest = [label for label in sorted(TEXT) if label not in (first, second)]
    edges = [_edge(chunk, first, rest[0]), _edge(chunk, second, rest[1])]
    with _graph(library, tmp_path, edges) as g:
        assert _context(_run(g, "alpha", limits), chunk) == [rest[0], rest[1]]   # seed order, then neighbors
        assert _context(_run(g, "alpha", limits, graph_seed_limit=1), chunk) == [rest[0]]
        assert _context(_run(g, "alpha", limits, graph_seed_limit=0), chunk) == []
        with pytest.raises(HybridRetrievalError):   # the native validator: seeds cannot exceed primaries
            _run(g, "alpha", limits, graph_seed_limit=3)


def test_neighbors_per_seed_and_total_limit_bound_the_context(library, tmp_path, chunk):
    edges = [_edge(chunk, "alpha", obj, 1) for obj in ("delta", "epsilon", "zeta")]
    order = sorted(("delta", "epsilon", "zeta"), key=lambda label: chunk[label])   # edge type, then object digest
    with _graph(library, tmp_path, edges) as g:
        assert _context(_run(g, "alpha", (1, 1, 1, 8)), chunk) == order[:2]       # native default: 2 per seed
        assert _context(_run(g, "alpha", (1, 1, 1, 8), graph_neighbors_per_seed=3), chunk) == order
        assert _context(_run(g, "alpha", (1, 1, 1, 8), graph_neighbors_per_seed=1), chunk) == order[:1]
        assert _context(_run(g, "alpha", (1, 1, 1, 2), graph_neighbors_per_seed=3), chunk) == order[:1]
        r = _run(g, "alpha", (1, 1, 1, 1), graph_neighbors_per_seed=3)
        assert len(_items(r)) == 1 and _context(r, chunk) == []


def test_edge_type_orders_neighbors_before_object_digest(library, tmp_path, chunk):
    edges = [_edge(chunk, "alpha", "delta", 9), _edge(chunk, "alpha", "zeta", 2), _edge(chunk, "alpha", "epsilon", 5)]
    with _graph(library, tmp_path, edges) as g:
        r = _run(g, "alpha", (1, 1, 1, 8), graph_neighbors_per_seed=3)
        assert _context(r, chunk) == ["zeta", "epsilon", "delta"]
        assert [i["edge_type"] for i in _items(r)[1:]] == [2, 5, 9]


def test_authority_floor_excludes_weaker_edges(library, tmp_path, chunk):
    edges = [_edge(chunk, "alpha", "delta", 1, 0), _edge(chunk, "alpha", "epsilon", 1, 1),
             _edge(chunk, "alpha", "zeta", 1, 3)]
    with _graph(library, tmp_path, edges) as g:
        for floor, expected in ((0, {"delta", "epsilon", "zeta"}), (1, {"epsilon", "zeta"}), (2, {"zeta"}),
                                (3, {"zeta"})):
            r = _run(g, "alpha", (1, 1, 1, 8), graph_neighbors_per_seed=3, min_graph_authority=floor)
            assert set(_context(r, chunk)) == expected
            assert all(i["edge_authority"] >= floor for i in _items(r)[1:])
        with pytest.raises(HybridRetrievalError):
            _run(g, "alpha", min_graph_authority=4)


def test_namespace_and_exact_authority_filters_apply_to_context(library, tmp_path, chunk):
    # epsilon: elpis.docs/canonical; zeta: elpis.notes/reference; beta: elpis.docs/reference.
    edges = [_edge(chunk, "alpha", obj) for obj in ("beta", "epsilon", "zeta")]
    with _graph(library, tmp_path, edges) as g:
        everything = set(_context(_run(g, "alpha", (1, 1, 1, 8), graph_neighbors_per_seed=3), chunk))
        assert everything == {"beta", "epsilon", "zeta"}
        docs = _run(g, "alpha", (1, 1, 1, 8), graph_neighbors_per_seed=3, namespace_filter="elpis.docs")
        assert set(_context(docs, chunk)) == {"beta", "epsilon"}
        assert all(i.namespace == "elpis.docs" for i in bundle_from_json(docs["bundle_json"], docs).items)
        canonical = _run(g, "alpha", (1, 1, 1, 8), graph_neighbors_per_seed=3, authority_filter="canonical")
        assert _context(canonical, chunk) == ["epsilon"] and all(i["authority"] == "canonical"
                                                                  for i in _items(canonical))
        # The filters are part of the query identity, as natively.
        assert len({docs["query_digest"], canonical["query_digest"], _run(g, "alpha")["query_digest"]}) == 3


def test_duplicates_collapse_and_a_selected_chunk_is_never_repeated(library, tmp_path, plain, chunk):
    one = [_edge(chunk, "alpha", "epsilon")]
    with _graph(library, tmp_path / "one", one) as single, \
            _graph(library, tmp_path / "dup", one * 3) as duplicated:
        assert duplicated.graph_edge_count == 1
        assert duplicated.graph_snapshot_digest == single.graph_snapshot_digest
        assert _run(duplicated, "alpha")["bundle_digest"] == _run(single, "alpha")["bundle_digest"]
    # Two facts for the same pair (different provenance) are two edges, but the chunk appears once.
    two = one + [_edge(chunk, "alpha", "epsilon", provenance=_fact("second", "fact"))]
    with _graph(library, tmp_path / "two", two) as g:
        assert g.graph_edge_count == 2
        r = _run(g, "alpha", (1, 1, 1, 8), graph_neighbors_per_seed=2)
        assert _context(r, chunk) == ["epsilon"]
    # A neighbor already selected as a primary is not repeated.
    limits, (first, second) = _two_primaries(plain, chunk)
    with _graph(library, tmp_path / "primary", [_edge(chunk, first, second)]) as g:
        r = _run(g, "alpha", limits)
        assert _context(r, chunk) == [] and len({i["chunk_digest"] for i in _items(r)}) == len(_items(r))


def test_a_neighbor_that_is_also_a_fusion_candidate_composes_its_source_mask(library, tmp_path, plain, chunk):
    limits = (2, 2, 1, 6)   # two candidates per source, one primary: the others stay unselected candidates
    base = _run(plain, "alpha", (2, 2, 6, 6))
    (primary,) = _primaries(_run(plain, "alpha", limits), chunk)
    candidate = _primaries(base, chunk)[1]   # the best unselected candidate
    ranked = next(i for i in _items(base) if i["chunk_digest"] == chunk[candidate])
    with _graph(library, tmp_path, [_edge(chunk, primary, candidate)]) as g:
        r = _run(g, "alpha", limits)
        ctx = _items(r)[1]
        assert _context(r, chunk) == [candidate]
        assert ctx["source_mask"] == GRAPH_BIT | ranked["source_mask"] and ranked["source_mask"] & (LEXICAL_BIT
                                                                                                  | DENSE_BIT)
        assert (ctx["lexical_rank"], ctx["dense_rank"], ctx["fusion_score_key"]) == (
            ranked["lexical_rank"], ranked["dense_rank"], ranked["fusion_score_key"])
        assert ctx["graph_parent_digest"] == chunk[primary]


# -- identity: determinism, permutation invariance, provenance binding ------------------------------------------

def test_retrieval_is_exactly_deterministic_and_order_invariant(library, tmp_path, chunk):
    edges = [_edge(chunk, "alpha", "epsilon", 1, 2), _edge(chunk, "alpha", "zeta", 4, 1),
             _edge(chunk, "gamma", "delta", 2, 3), _edge(chunk, "zeta", "beta", 1, 0)]
    results = []
    for n, (docs, order) in enumerate(((DOCS, edges), (DOCS[::-1], edges[::-1]),
                                       (DOCS[2:] + DOCS[:2], edges[1:] + edges[:1]))):
        with _graph(library, tmp_path / str(n), order, docs) as g:
            runs = [_run(g, q, (2, 2, 2, 10), graph_neighbors_per_seed=3) for q in ("alpha", "gamma vector") * 2]
            assert runs[0] == runs[2] and runs[1] == runs[3]   # repeat retrieval: byte-identical results
            results.append((g.graph_snapshot_digest, [(r["bundle_json"], r["bundle_digest"], r["hacf_package_digest"])
                                                      for r in runs]))
    assert results[0] == results[1] == results[2]


def test_edge_provenance_is_bound_into_graph_and_bundle_identity(library, tmp_path, chunk):
    with _graph(library, tmp_path / "a", [_edge(chunk, "alpha", "epsilon", provenance=_fact("a"))]) as a, \
            _graph(library, tmp_path / "b", [_edge(chunk, "alpha", "epsilon", provenance=_fact("b"))]) as b, \
            _graph(library, tmp_path / "c", [_edge(chunk, "alpha", "epsilon", authority=3, provenance=_fact("a"))]) as c:
        ra, rb, rc = (_run(h, "alpha") for h in (a, b, c))
        assert len({a.graph_snapshot_digest, b.graph_snapshot_digest, c.graph_snapshot_digest}) == 3
        assert _items(ra)[1]["chunk_digest"] == _items(rb)[1]["chunk_digest"] == chunk["epsilon"]
        # The same context item under a different provenance fact is a different, distinguishable bundle.
        assert len({ra["bundle_digest"], rb["bundle_digest"], rc["bundle_digest"]}) == 3
        assert _items(rc)[1]["edge_authority"] == 3


# -- integrity failures and lifecycle ----------------------------------------------------------------------------

def test_an_edge_endpoint_outside_the_corpus_refuses_construction(library, tmp_path, chunk):
    missing = _fact("not an admitted chunk")
    for n, edge in enumerate((ContextEdge.of(chunk["alpha"], missing, _fact(1), 1, 1),
                              ContextEdge.of(missing, chunk["alpha"], _fact(2), 1, 1))):
        with pytest.raises(HybridRetrievalError) as info:
            build_corpus_and_index(library, tmp_path / str(n), DOCS, [_edge(chunk, "alpha", "beta"), edge])
        assert info.value.code == "ENV_CREATE_FAILED" and "E_GRAPH" in info.value.detail
        assert "not an admitted corpus chunk" in info.value.detail


def test_malformed_and_self_edges_are_refused(library, tmp_path, chunk):
    for bad in (dict(subject=chunk["alpha"].upper()), dict(subject="ab"), dict(edge_type=0), dict(authority=4),
                dict(provenance=None)):
        args = dict(subject=chunk["alpha"], obj=chunk["beta"], provenance=_fact(0), edge_type=1, authority=1)
        args.update(bad)
        with pytest.raises(HybridRetrievalError) as info:
            ContextEdge.of(**args)
        assert info.value.code == "GRAPH_EDGE_INVALID"
    # The native graph law refuses the same values even when the Python check is bypassed.
    raw = [ContextEdge(chunk["alpha"].upper().encode(), chunk["beta"].encode(), _fact(0).encode(), 1, 1),
           ContextEdge(chunk["alpha"].encode(), chunk["beta"].encode(), _fact(0).encode(), 0, 1),
           ContextEdge(chunk["alpha"].encode(), chunk["beta"].encode(), _fact(0).encode(), 1, 7),
           ContextEdge(chunk["alpha"].encode(), chunk["alpha"].encode(), _fact(0).encode(), 1, 1)]
    for n, edge in enumerate(raw):
        with pytest.raises(HybridRetrievalError) as info:
            build_corpus_and_index(library, tmp_path / str(n), DOCS, [edge])
        assert "E_GRAPH: context graph rejected" in info.value.detail
    with pytest.raises(HybridRetrievalError):   # the self-edge, through the typed constructor
        build_corpus_and_index(library, tmp_path / "self", DOCS, [_edge(chunk, "alpha", "alpha")])
    with pytest.raises(HybridRetrievalError) as info:
        build_corpus_and_index(library, tmp_path / "type", DOCS, ["not an edge"])
    assert info.value.code == "GRAPH_EDGE_INVALID"


def test_the_environment_owns_and_destroys_its_graph(library, tmp_path, chunk):
    handle = _graph(library, tmp_path, [_edge(chunk, "alpha", "epsilon")])
    assert _context(_run(handle, "alpha"), chunk) == ["epsilon"]
    handle.destroy()
    handle.destroy()   # idempotent
    with pytest.raises(HybridRetrievalError) as info:
        _run(handle, "alpha")
    assert info.value.code == "NO_ENVIRONMENT"
    # Many graph environments built and destroyed in turn stay independent and identical.
    digests = set()
    for n in range(4):
        with _graph(library, tmp_path / f"cycle{n}", [_edge(chunk, "alpha", "epsilon")]) as h:
            digests.add((h.graph_snapshot_digest, _run(h, "alpha")["bundle_digest"]))
    assert len(digests) == 1


def test_a_library_without_the_v2_bridge_abi_is_refused():
    other = require_native_library("elpis_ingress_bridge")
    with pytest.raises(RetrievalLibraryError) as info:
        RetrievalLibrary(other)
    assert info.value.code == "ABI_MISMATCH"


# -- the validation gate binds context to the graph -------------------------------------------------------------

def test_validation_refuses_context_not_bound_to_a_graph_and_an_earlier_primary(library, tmp_path, chunk):
    from dataclasses import replace

    with _graph(library, tmp_path, [_edge(chunk, "alpha", "epsilon")]) as g:
        r = _run(g, "alpha")
    bundle = bundle_from_json(r["bundle_json"], r)
    q, c = r["query_digest"], r["corpus_manifest_digest"]
    validate_bundle(bundle, q, c)
    primary, context = bundle.items
    cases = {
        "GRAPH_PROVENANCE_UNBOUND": replace(bundle, graph_snapshot_digest=ZERO_DIGEST),
        "GRAPH_PARENT_UNBOUND": replace(bundle, items=(primary, replace(context, graph_parent_digest=_fact("x")))),
        "GRAPH_EDGE_UNBOUND": replace(bundle, items=(primary, replace(context, edge_type=0))),
        "GRAPH_SOURCE_UNBOUND": replace(bundle, items=(primary, replace(context, source_mask=LEXICAL_BIT))),
    }
    for code, forged in cases.items():
        with pytest.raises(BundleValidationError) as info:
            validate_bundle(forged, q, c)
        assert info.value.code == code
    # A context item ahead of its parent is unbound too.
    swapped = replace(bundle, items=(replace(context, final_rank=0), replace(primary, final_rank=1)))
    with pytest.raises(BundleValidationError) as info:
        validate_bundle(swapped, q, c)
    assert info.value.code == "GRAPH_PARENT_UNBOUND"
