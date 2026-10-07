"""Structural memory rendered through the codec: ingress -> adapter -> verified HACF objects -> admission.

Everything here runs over real native libraries: a real HACF corpus, the real
Regex/HACF ingress and verified document blobs. The runtime part is
``Runtime.admit_context``. Where a model consumes the admission, it is the
synthetic DSV4 fixture driven directly through the noncanonical principal
engine, as a mechanics check of bounded resident state and slow-lane
isolation; the runtime composes no model execution (docs/ELPIS_MISSION.md).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import fields
import hashlib
import inspect
import json
import shutil

import pytest

import elpis.identity
import elpis.pipeline.ingress as ingress_module
import elpis.runtime.composition as composition
import elpis.runtime.edges as edges
import elpis.structure.retrieval.hacf as hacf_module
import elpis.structure.retrieval.objects as objects_module
import elpis.substrate.boundary as boundary
import elpis.substrate.digests as digests
import elpis.inference.admission as admission_module
from elpis.inference.admission import ContextBudget, render_synthetic_nibble16
from elpis.inference.context import initial_snapshot
from elpis.inference.drivers.dsv4.fixtures import make_fixture
from elpis.inference.principal import PrincipalEngine, PrincipalRequest
from elpis.pipeline.ingress import QueryIngress
from elpis.runtime import Runtime, RuntimeConfig
from elpis.runtime.composition import ContextPreparation
from elpis.continuity import ContinuityStore
from elpis.structure.retrieval.hacf import build_corpus_and_index
from elpis.structure.retrieval.objects import ObjectResolutionError, normalize
from elpis.substrate.synthetic import SyntheticFileAssets

from ..inference.test_token_stream_kernel import Recorder
from .conftest import POSITIVE

NS = "elpis.docs"
CONTEXT = initial_snapshot().digest
BUDGET = ContextBudget(max_objects=6, max_bytes=1200, max_tokens=2400)
FILLER = "Filler paragraph {i} records structural bookkeeping and keeps the maximum end of each range, " \
         "with enough words to occupy a few hundred bytes of text so that the chunker has to pack units."


def long_document():
    paragraphs = [FILLER.format(i=i) + " " + FILLER.format(i=i) for i in range(10)]
    paragraphs[8] = "Late paragraph:  \r\ntouching endpoints may merge when the maximum end is kept.\t"
    return "\n\n".join(paragraphs)


def filler_document(index, paragraphs, repeats):
    return "\n\n".join(" ".join(FILLER.format(i=index * 1000 + k * 100 + r) for r in range(repeats))
                       for k in range(paragraphs))


SCALES = {0: (0, 0, 0), 1: (20, 1, 2), 2: (58, 2, 17)}  # (filler documents, paragraphs, repeats)


def corpus_documents(scale):
    """Grow HACF in documents, chunks and bytes.

    The only Python-reachable builder (the retrieval bridge) indexes at most 64
    documents and 128 chunks, so growth here stays inside that ceiling; the
    ingress and resolver themselves open corpora of any size.
    """
    docs = [
        ("spec", "Intervals whose touching endpoints may merge keep the maximum end.", NS, "canonical"),
        ("note", "Keep at least one merged range for every overlap.", NS, "reference"),
        ("long", long_document(), NS, "reference"),
    ]
    count, paragraphs, repeats = SCALES[scale]
    docs += [("filler-%02d" % i, filler_document(i, paragraphs, repeats), NS, "reference") for i in range(count)]
    return docs


def build(retrieval_library, root, extra=0):
    handle = build_corpus_and_index(retrieval_library, root, corpus_documents(extra))
    manifest = handle.corpus_manifest_json
    handle.destroy()
    return root / "corpus", manifest


@pytest.fixture(scope="module")
def substrate(retrieval_library, tmp_path_factory):
    return build(retrieval_library, tmp_path_factory.mktemp("context-substrate"))


@pytest.fixture
def model(native_workspace, fms_file_library):
    provider = SyntheticFileAssets(root=native_workspace, library=fms_file_library,
                                   warm_bytes=1 << 20, staging_bytes=1 << 16)
    target, resident, _ = make_fixture(provider, native_workspace / "dsv4")
    yield PrincipalEngine(target), resident
    provider.close()


def prepare(runtime, ingress_library, corpus_root, manifest, engine, budget=BUDGET):
    with QueryIngress(ingress_library, corpus_root) as ingress:
        return runtime.admit_context(ingress=ingress, task=POSITIVE, corpus_root=corpus_root,
                                     corpus_manifest=manifest, context_snapshot=CONTEXT,
                                     model=engine.target.model_identity, tokenizer=engine.target.config.tokenizer,
                                     budget=budget, max_document_bytes=1 << 20)


def test_ingress_to_admission_through_verified_hacf_objects(runtime, ingress_library, substrate, model):
    corpus_root, manifest = substrate
    engine, resident = model
    prepared = prepare(runtime, ingress_library, corpus_root, manifest, engine)
    admission = prepared.admission
    export = json.loads(prepared.ingress.proposal_json)
    hits = {h["chunk_digest"]: h for row in export["hacf"]["retrieval"] for h in row["hacf_primary_hits"]}
    assert admission.objects and all(o.object in hits for o in admission.objects)
    assert any(hits[o.object]["ordinal"] > 0 and hits[o.object]["byte_start"] > 0 for o in admission.objects), \
        "a later chunk of the multi-chunk document must be admitted"
    blobs = {d: (corpus_root / "corpus" / f"{d}.blob").read_bytes() for d in {h["doc_digest"] for h in hits.values()}}
    for obj in admission.objects:
        hit = hits[obj.object]
        text = normalize(blobs[hit["doc_digest"]][hit["byte_start"]:hit["byte_end"]])
        assert obj.content == hashlib.sha256(text).hexdigest() and obj.size == len(text)
        assert obj.tokens == render_synthetic_nibble16(text)
    assert admission.corpus == prepared.ingress.corpus_manifest_digest
    assert len(admission.objects) <= BUDGET.max_objects and len(admission.tokens) <= BUDGET.max_tokens

    state = engine.initial(CONTEXT)
    request = PrincipalRequest("turn-1", (1, 2), 12)
    sequence = engine.begin(state, request, admission, expected_state=state.digest, resident_experts=resident)
    emitted = list(sequence)
    result = engine.finalize(sequence)
    assert result.commit.outputs == tuple(emitted) and len(emitted) == 12
    assert result.commit.admission == admission.digest
    assert result.commit.prefill == len(admission.tokens) + len(request.prompt)
    # The runtime renders structural memory only; it runs no model and writes no continuity.
    assert runtime.continuity.snapshot().generation == 1
    assert engine.replay(state, request, admission, result.commit).commit == result.commit


TRAPPED = [
    (QueryIngress, "run"), (ingress_module, "lex"), (ingress_module, "lex_stream"),
    (hacf_module, "hybrid_retrieve"), (hacf_module, "build_corpus_and_index"),
    (edges, "from_regex_hacf"), (edges, "object_claims"), (edges, "from_retrieval_bundle"),
    (composition, "from_regex_hacf"), (composition, "object_claims"), (composition, "resolve_chunks"),
    (composition, "admit_context"), (objects_module, "resolve_chunks"), (objects_module, "_read_document"),
    (admission_module, "admit_context"), (ContinuityStore, "anchor_cognition"),
    (ContinuityStore, "commit_cognition_transition"), (ContinuityStore, "reserve_evolution_assertion"),
    (ContinuityStore, "commit_evolution_transition"), (boundary.RootCapability, "open_file"),
    (elpis.identity, "content_digest"), (digests, "content_digest"), (digests, "identity"), (digests, "raw_digest"),
]


def test_once_begun_the_token_loop_cannot_reach_any_slow_lane(runtime, ingress_library, substrate, model,
                                                              monkeypatch):
    corpus_root, manifest = substrate
    engine, resident = model
    admission = prepare(runtime, ingress_library, corpus_root, manifest, engine).admission
    state = engine.initial(CONTEXT)
    request = PrincipalRequest("r", (3,), 40)
    first = engine.begin(state, request, admission, expected_state=state.digest, resident_experts=resident)
    list(first)  # fault in every page this path touches
    sequence = engine.begin(state, request, admission, expected_state=state.digest, resident_experts=resident)

    hits = []
    for owner, name in TRAPPED:
        def trap(*args, _name=f"{getattr(owner, '__name__', owner)}.{name}", **kwargs):
            hits.append(_name)
            raise AssertionError(f"slow lane reached during an active sequence: {_name}")
        monkeypatch.setattr(owner, name, trap)
    recorder = Recorder()
    tokens = recorder.run(lambda: [sequence.next() for _ in range(40)])
    monkeypatch.undo()
    assert hits == [] and None not in tokens and recorder.calls == Counter(), (hits, recorder.calls)
    assert tuple(tokens) == engine.finalize(first).commit.outputs


def test_resident_model_state_stays_bounded_as_hacf_grows(retrieval_library, ingress_library, model,
                                                          tmp_path_factory, continuity_library):
    engine, resident = model
    shapes, peaks, admitted, sizes = set(), [], [], []
    for extra in (0, 1, 2):
        root = tmp_path_factory.mktemp("scale-%d" % extra)
        corpus_root, manifest = build(retrieval_library, root, extra)
        chunks = json.loads(manifest)["chunk_count"]
        sizes.append(sum(d["size_bytes"] for d in json.loads(manifest)["documents"]))
        with Runtime(RuntimeConfig(root / "continuity", continuity_library)) as runtime:
            admission = prepare(runtime, ingress_library, corpus_root, manifest, engine).admission
            state = engine.initial(CONTEXT)
            sequence = engine.begin(state, PrincipalRequest("r", (1,), 20), admission,
                                    expected_state=state.digest, resident_experts=resident)
            peak = len(sequence.working_state.keys)
            for _ in sequence:
                peak = max(peak, len(sequence.working_state.keys), len(sequence.working_state.values))
            result = engine.finalize(sequence)
        shapes.add(tuple((f.name, type(getattr(result.state, f.name)).__name__) for f in fields(result.state)))
        peaks.append(peak)
        admitted.append((chunks, len(admission.objects), len(admission.tokens)))
    assert admitted[-1][0] >= 10 * admitted[0][0] and sizes[-1] >= 100 * sizes[0]  # HACF really grew
    assert len(shapes) == 1 and max(peaks) <= engine.target.config.local_window
    assert all(objs <= BUDGET.max_objects and toks <= BUDGET.max_tokens for _, objs, toks in admitted)


_FORBIDDEN = ("key", "value", "kv", "latent", "hidden", "pool", "tensor", "logit", "vector")


@pytest.mark.parametrize("function", [composition.Runtime.admit_context, edges.object_claims,
                                      edges.from_regex_hacf, PrincipalEngine.begin])
def test_runtime_boundary_takes_no_kv_or_vectors(function):
    names = list(inspect.signature(function).parameters)
    assert not [n for n in names if any(bad in n.lower() for bad in _FORBIDDEN)], names


def test_preparation_record_carries_no_model_state():
    assert {f.name for f in fields(ContextPreparation)} == {"admission", "ingress"}


def test_tampered_blob_or_wrong_manifest_refuses_admission(ingress_library, substrate, model, tmp_path,
                                                           continuity_library):
    corpus_root, manifest = substrate
    engine, _ = model
    copy = tmp_path / "corpus"
    shutil.copytree(corpus_root, copy)
    for blob in (copy / "corpus").glob("*.blob"):
        data = bytearray(blob.read_bytes())
        data[-1] ^= 1
        blob.write_bytes(bytes(data))
    with Runtime(RuntimeConfig(tmp_path / "continuity", continuity_library)) as runtime:
        with pytest.raises(ObjectResolutionError, match="INTEGRITY"):
            prepare(runtime, ingress_library, copy, manifest, engine)
        with pytest.raises(ObjectResolutionError, match="INTEGRITY"):
            prepare(runtime, ingress_library, corpus_root, json.dumps({"documents": []}), engine)
        assert runtime.continuity.snapshot().generation == 1


def test_scaling_benchmark_reports_bounded_state_and_separate_costs(monkeypatch):
    from . import context_scaling_benchmark as bench
    monkeypatch.setattr(bench, "REPEATS", 1)
    rows = bench.run(scales=(0, 2))
    assert rows[1]["corpus"]["bytes"] >= 100 * rows[0]["corpus"]["bytes"]
    states = {r["resident_principal_state"]["committed_state_canonical_bytes"] for r in rows}
    peaks = {r["resident_principal_state"]["peak_attention_scratch_entries"] for r in rows}
    assert len(states) == 1 and max(peaks) <= 4
    for row in rows:
        assert set(row["latency_ms_median"]) == {"ingress", "resolve", "admit", "begin_prefill", "per_token",
                                                 "finalize"}
        assert row["admission"]["objects"] <= bench.BUDGET.max_objects
        assert row["legacy_kv_analytic"]["fits_legacy_max_tokens"] is False
