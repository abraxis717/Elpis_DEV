"""Mechanical benchmark: HACF context grows, resident principal model state does not.

    ELPIS_NATIVE_BUILD=$PWD/build python -m tests.integration.context_scaling_benchmark [--json out.json]

For each corpus scale it builds a real HACF corpus and runs the full
context-substrate path (ingress -> edge adapter -> verified object resolution
-> admission -> principal sequence), and reports:

* the addressable context: documents, chunks, bytes;
* latency of every boundary separately (median of repeated runs, warm
  pages): ingress, object resolution, admission, begin (validation, expert
  admission, prefill), per generated token, finalize;
* what was admitted under the fixed budget;
* resident principal model state: committed-state size (canonical bytes),
  peak attention-scratch entries and floats, the declared working set.

The claim is bounded resident model state, not free retrieval: resolution
reads and verifies whole document blobs, so its cost is reported, and it
grows with the size of the documents the admitted chunks come from.

The legacy contrast is ANALYTIC, not measured: holding the same history as a
compressed-KV pool would need history_tokens / compression pool entries
(2 x dimension floats each), beyond the fixture's max_tokens at every scale.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import statistics
import tempfile
import time

from elpis.identity import canonical_json_bytes
from elpis.inference.admission import ContextBudget, admit_context
from elpis.inference.context import initial_snapshot
from elpis.inference.drivers.dsv4.fixtures import make_fixture
from elpis.inference.principal import PrincipalEngine, PrincipalRequest
from elpis.pipeline.ingress import IngressLibrary, QueryIngress
from elpis.runtime import Runtime, RuntimeConfig
from elpis.runtime.edges import from_regex_hacf, object_claims
from elpis.structure.retrieval.hacf import RetrievalLibrary, build_corpus_and_index
from elpis.structure.retrieval.objects import CorpusManifest, resolve_chunks
from elpis.substrate.digests import raw_digest
from elpis.substrate.synthetic import SyntheticFileAssets

from tests.conftest import find_native_library
from tests.integration.conftest import POSITIVE
from tests.integration.test_context_substrate import SCALES, corpus_documents

BUDGET = ContextBudget(max_objects=6, max_bytes=1200, max_tokens=2400)
CONTEXT = initial_snapshot().digest
GENERATE = 32
REPEATS = 5


def _library(stem):
    found = find_native_library(stem)
    if found is None:
        raise SystemExit(f"lib{stem}.so not found; set ELPIS_NATIVE_BUILD to a native build directory")
    return found


def _median_ms(samples):
    return round(statistics.median(samples) * 1e3, 3)


def measure(scale, workspace, engine, resident, retrieval, ingress_library):
    root = workspace / f"scale-{scale}"
    handle = build_corpus_and_index(retrieval, root, corpus_documents(scale))
    manifest_json = handle.corpus_manifest_json
    handle.destroy()
    corpus_root = root / "corpus"
    manifest_doc = json.loads(manifest_json)
    timings = {k: [] for k in ("ingress", "resolve", "admit", "begin_prefill", "per_token", "finalize")}
    for _ in range(REPEATS):
        with QueryIngress(ingress_library, corpus_root) as ingress:
            t0 = time.perf_counter()
            result = ingress.run(POSITIVE)
            t1 = time.perf_counter()
        payload = result.proposal_json
        pin = raw_digest(payload)
        proposals = from_regex_hacf(payload, expected_payload=pin, expected_source=result.source_sha256,
                                    expected_corpus=result.corpus_manifest_digest, context_snapshot=CONTEXT,
                                    query_overlay=result.overlay_identity, rules=())
        claims = object_claims(payload, expected_payload=pin)
        t2 = time.perf_counter()
        manifest = CorpusManifest.verified(manifest_json, expected_digest=result.corpus_manifest_digest)
        resolved, omitted = resolve_chunks(corpus_root, manifest, claims, max_objects=BUDGET.max_objects,
                                           max_text_bytes=BUDGET.max_bytes, max_document_bytes=1 << 22)
        t3 = time.perf_counter()
        admission = admit_context(model=engine.target.model_identity, tokenizer=engine.target.config.tokenizer,
                                  context_snapshot=CONTEXT, corpus=result.corpus_manifest_digest,
                                  proposals=proposals, resolved=tuple((c.chunk_digest, c.text) for c in resolved),
                                  omitted=omitted, budget=BUDGET)
        admission.digest
        t4 = time.perf_counter()
        state = engine.initial(CONTEXT)
        sequence = engine.begin(state, PrincipalRequest("bench", (1,), GENERATE), admission,
                                expected_state=state.digest, resident_experts=resident)
        t5 = time.perf_counter()
        peak = len(sequence.working_state.keys)
        for _ in sequence:
            peak = max(peak, len(sequence.working_state.keys), len(sequence.working_state.values))
        t6 = time.perf_counter()
        final = engine.finalize(sequence)
        t7 = time.perf_counter()
        for key, value in (("ingress", t1 - t0), ("resolve", t3 - t2), ("admit", t4 - t3),
                           ("begin_prefill", t5 - t4), ("per_token", (t6 - t5) / GENERATE),
                           ("finalize", t7 - t6)):
            timings[key].append(value)
    config = engine.target.config
    history_tokens = 2 * sum(d["size_bytes"] for d in manifest_doc["documents"])
    return {
        "scale": scale,
        "corpus": {"documents": len(manifest_doc["documents"]), "chunks": manifest_doc["chunk_count"],
                   "bytes": sum(d["size_bytes"] for d in manifest_doc["documents"])},
        "latency_ms_median": {k: _median_ms(v) for k, v in timings.items()},
        "admission": {"objects": len(admission.objects), "omitted": admission.omitted,
                      "tokens": len(admission.tokens), "bytes": sum(o.size for o in admission.objects)},
        "resident_principal_state": {
            "committed_state_canonical_bytes": len(canonical_json_bytes(asdict(final.state))),
            "peak_attention_scratch_entries": peak,
            "peak_attention_scratch_floats": 2 * peak * config.dimension,
            "declared_working_set": list(final.commit.working_set),
        },
        "legacy_kv_analytic": {
            "history_tokens_synthetic": history_tokens,
            "pool_entries_to_hold_history": history_tokens // config.compression,
            "pool_floats_to_hold_history": 2 * config.dimension * (history_tokens // config.compression),
            "fits_legacy_max_tokens": history_tokens < config.max_tokens,
        },
    }


def run(scales=tuple(SCALES)):
    workspace = Path(tempfile.mkdtemp(prefix="elpis-context-bench-")).resolve()
    try:
        (workspace / "native").mkdir()
        fms = workspace / "native" / "libelpis_fms_file_assets.so"
        shutil.copyfile(_library("elpis_fms_file_assets"), fms)
        provider = SyntheticFileAssets(root=workspace, library=fms, warm_bytes=1 << 20, staging_bytes=1 << 16)
        try:
            target, resident, _ = make_fixture(provider, workspace / "dsv4")
            engine = PrincipalEngine(target)
            retrieval = RetrievalLibrary(_library("elpis_retrieval_bridge"))
            ingress_library = IngressLibrary(_library("elpis_ingress_bridge"))
            return [measure(s, workspace, engine, resident, retrieval, ingress_library) for s in scales]
        finally:
            provider.close()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="write the rows as JSON to this path")
    args = parser.parse_args()
    rows = run()
    header = ("scale", "docs", "chunks", "bytes", "ingress", "resolve", "admit", "begin+prefill", "per_tok",
              "finalize", "adm_obj", "adm_omit", "adm_tok", "state_B", "kv_peak", "legacy_pool_entries(analytic)")
    print(" | ".join(header))
    for r in rows:
        lat, cor, adm, st = r["latency_ms_median"], r["corpus"], r["admission"], r["resident_principal_state"]
        print(" | ".join(str(x) for x in (
            r["scale"], cor["documents"], cor["chunks"], cor["bytes"], lat["ingress"], lat["resolve"], lat["admit"],
            lat["begin_prefill"], lat["per_token"], lat["finalize"], adm["objects"], adm["omitted"], adm["tokens"],
            st["committed_state_canonical_bytes"], st["peak_attention_scratch_entries"],
            r["legacy_kv_analytic"]["pool_entries_to_hold_history"])))
    print("latencies are median milliseconds over", REPEATS, "warm runs; legacy column is analytic")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
