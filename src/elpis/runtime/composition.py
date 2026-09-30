"""The one runtime composition: explicit subsystem wiring around one receipt history.

``Runtime`` does not own models, corpora, canonical roots or targets. The
caller constructs each of them explicitly (explicit library paths, corpus
roots, file assets, publication ledgers) and passes them in. The runtime
runs one subsystem operation at a time through that subsystem's own
fail-closed entry point, and records the committed outcome in its history:

* ``run_ingress``: pipeline ingress on caller bytes. Only a published,
  zero-authority proposal batch is recorded; a fail-closed result is returned
  unrecorded.
* ``admit_retrieval``: structure retrieval-bundle validation against the
  expected query and corpus. Only a valid bundle is recorded.
* ``publish_canonical``: the canonical Grid81 publisher, with the caller's
  one-use promotion capability. The publication receipt is recorded, and an
  idempotent replay records nothing new.
* ``evolve``: the evolution path gate, bound to a projection of this runtime's
  own history taken at call time. An assertion built against an older history
  head is rejected before anything runs, and a transition receipt is recorded
  only for an admitted attempt.
* ``decode``: one inference decode transaction. Only a committed decode
  receipt is recorded; a typed failure leaves the history unchanged.
* ``admit_context``: the context-substrate path before a sequence. Pipeline
  ingress on the turn's bytes, then the edge adapter (address proposals and
  chunk claims from the pinned export), then HACF object resolution
  (verified document bytes, recomputed chunk identity, pinned corpus
  manifest), then inference context admission under an explicit budget. The
  ingress proposal and the admission are recorded.
* ``run_principal``: one principal sequence over an admission. ``begin``,
  then tokens handed to the caller's ``emit`` as they are produced, then
  ``finalize``. Nothing else runs while tokens stream; the commit is recorded
  after finalization.

Nothing is chained implicitly. No hidden fallback widens authority: every
check that refuses an operation is the owning subsystem's own check, and the
runtime adds a record only after that check passed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from elpis.ecs.projection.contracts import ContextProjection, ProjectionRequest
from elpis.inference.admission import ContextAdmission, ContextBudget, admit_context
from elpis.evolution.path_gate import EvolutionPathGate, GateExecuted, GateRejected
from elpis.pipeline.canonical.publisher import CanonicalPublicationReceipt, publish_candidate
from elpis.pipeline.ingress import QueryIngress, QueryIngressResult
from elpis.structure.retrieval.budget import RetrievalBudget
from elpis.structure.retrieval.contracts import RetrievalBundle
from elpis.structure.retrieval.objects import CorpusManifest, resolve_chunks
from elpis.structure.retrieval.validation import validate_bundle
from elpis.substrate.digests import raw_digest

from .edges import from_regex_hacf, object_claims
from .history import HistoryError, ReceiptHistory, ReceiptRecord, RecordedReceipt

if TYPE_CHECKING:
    from elpis.inference.principal import PrincipalEngine, PrincipalResult
    from elpis.inference.transaction import DecodeResult, InferenceEngine

__all__ = ("CompositionError", "ContextPreparation", "Runtime", "RuntimeConfig")


class CompositionError(RuntimeError):
    """A composed operation was refused by one of its stages (fail closed)."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class ContextPreparation:
    """What admit_context produced: the frozen admission and the records behind it."""

    admission: ContextAdmission
    ingress: QueryIngressResult
    ingress_record: RecordedReceipt
    admission_record: RecordedReceipt


@dataclass(frozen=True)
class RuntimeConfig:
    """Everything the runtime itself owns: the location of its durable history."""

    history_dir: Path

    def __post_init__(self):
        if not isinstance(self.history_dir, Path) or not self.history_dir.is_absolute():
            raise HistoryError("HISTORY_PATH", "history_dir must be an absolute Path")


class Runtime:
    def __init__(self, config: RuntimeConfig):
        if type(config) is not RuntimeConfig:
            raise TypeError("config must be a RuntimeConfig")
        self.config = config
        self.history = ReceiptHistory(config.history_dir)

    def open(self) -> "Runtime":
        self.history.open()
        return self

    def close(self) -> None:
        self.history.close()

    def __enter__(self) -> "Runtime":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- pipeline: bounded ingress ---------------------------------------------------
    def run_ingress(self, ingress: QueryIngress, task: bytes
                    ) -> tuple[QueryIngressResult, RecordedReceipt | None]:
        if type(ingress) is not QueryIngress or type(task) is not bytes:
            raise TypeError("run_ingress takes a QueryIngress handle and task bytes")
        result = ingress.run(task)
        if not result.batch_published:
            return result, None
        return result, self.history.record(ReceiptRecord.of(
            "pipeline", "ingress.proposal", result.proposal_digest,
            source=result.source_sha256,
            corpus=result.corpus_manifest_digest,
            context_graph=result.context_graph_manifest_digest,
            overlay=result.overlay_identity,
            batch_receipt=result.batch_receipt_identity,
        ))

    # -- structure: retrieval admission -------------------------------------------
    def admit_retrieval(self, bundle: RetrievalBundle, *, expected_query: str, expected_corpus: str,
                        budget: RetrievalBudget | None = None):
        if type(bundle) is not RetrievalBundle:
            raise TypeError("admit_retrieval takes a RetrievalBundle")
        decision = validate_bundle(bundle, expected_query, expected_corpus, budget or RetrievalBudget())
        return decision, self.history.record(ReceiptRecord.of(
            "structure", "retrieval.bundle", bundle.bundle_digest,
            query=expected_query, corpus=expected_corpus, items=str(len(bundle.items)),
        ))

    # -- pipeline: canonical publication --------------------------------------------
    def publish_canonical(self, *, project_root, candidate_root, ledger, promotion_capability,
                          lock_path=None) -> tuple[CanonicalPublicationReceipt, RecordedReceipt]:
        receipt = publish_candidate(project_root=project_root, candidate_root=candidate_root,
                                    ledger=ledger, promotion_capability=promotion_capability,
                                    lock_path=lock_path)
        return receipt, self.history.record(ReceiptRecord.of(
            "pipeline", "canonical.publication", receipt.publication_receipt_digest,
            artifact=receipt.artifact_digest,
            previous_canonical=receipt.previous_canonical_digest,
            resulting_canonical=receipt.resulting_canonical_digest,
            generation=str(receipt.generation_number),
            transaction=receipt.transaction_id,
        ))

    # -- evolution: gated attempts over this runtime's history -------------------------
    def history_projection(self, request: ProjectionRequest | None = None) -> ContextProjection:
        return self.history.projection(request)

    def evolve(self, gate: EvolutionPathGate, *, assertion, state, advance, advance_kwargs,
               projection_request: ProjectionRequest | None = None
               ) -> tuple[GateRejected | GateExecuted, RecordedReceipt | None]:
        if type(gate) is not EvolutionPathGate:
            raise TypeError("evolve takes an EvolutionPathGate")
        projection = self.history.projection(projection_request)
        result = gate.execute(assertion=assertion, state=state, projection=projection,
                              advance=advance, advance_kwargs=advance_kwargs)
        if not isinstance(result, GateExecuted):
            return result, None
        receipt = result.receipt
        return result, self.history.record(ReceiptRecord.of(
            "evolution", "evolution.path-transition", receipt.receipt_digest,
            assertion=receipt.path_assertion_digest,
            history_projection=projection.projection_digest,
            outcome=receipt.attempt_outcome,
        ))

    # -- inference: one decode transaction ---------------------------------------------
    def decode(self, engine: "InferenceEngine", state, request, *, expected_state: str,
               prefetch_enabled: bool = False) -> tuple["DecodeResult", RecordedReceipt | None]:
        from elpis.inference.transaction import InferenceEngine

        if type(engine) is not InferenceEngine:
            raise TypeError("decode takes an InferenceEngine")
        result = engine.execute(state, request, expected_state=expected_state,
                                prefetch_enabled=prefetch_enabled)
        receipt = result.receipt
        if receipt.terminal != "COMMITTED":
            return result, None
        return result, self.history.record(ReceiptRecord.of(
            "inference", "inference.decode", receipt.digest,
            request=receipt.request, model=receipt.model,
            input_state=receipt.input_state, output_state=receipt.output_state,
        ))

    # -- context substrate: ingress -> adapter -> HACF resolution -> admission ------------
    def admit_context(self, *, ingress: QueryIngress, task: bytes, corpus_root: Path, corpus_manifest,
                      context_snapshot: str, model: str, tokenizer: str, budget: ContextBudget,
                      max_document_bytes: int, rules: tuple = (), text_tokenizer=None) -> ContextPreparation:
        """Prepare the frozen context for the next sequence, at a turn boundary.

        ``corpus_manifest`` is the corpus manifest JSON the caller obtained from
        its HACF handle; it must hash to the digest the ingress result pins.
        Any refusal (ingress fail-closed, a tampered export, blob or manifest,
        an unproposed object) raises, and no admission is recorded.
        """
        if type(budget) is not ContextBudget:
            raise TypeError("admit_context takes a ContextBudget")
        result, ingress_record = self.run_ingress(ingress, task)
        if ingress_record is None:
            raise CompositionError("INGRESS_REFUSED", result.status)
        payload = result.proposal_json
        pin = raw_digest(payload)
        proposals = from_regex_hacf(payload, expected_payload=pin, expected_source=result.source_sha256,
                                    expected_corpus=result.corpus_manifest_digest,
                                    context_snapshot=context_snapshot, query_overlay=result.overlay_identity,
                                    rules=rules)
        claims = object_claims(payload, expected_payload=pin)
        manifest = CorpusManifest.verified(corpus_manifest, expected_digest=result.corpus_manifest_digest)
        resolved, omitted = resolve_chunks(corpus_root, manifest, claims, max_objects=budget.max_objects,
                                           max_text_bytes=budget.max_bytes,
                                           max_document_bytes=max_document_bytes)
        from elpis.inference.admission import SYNTHETIC_NIBBLE16
        from elpis.inference.text import DSV41_RENDERER
        admission = admit_context(model=model, tokenizer=tokenizer, context_snapshot=context_snapshot,
                                  corpus=result.corpus_manifest_digest, proposals=proposals,
                                  resolved=tuple((c.chunk_digest, c.text) for c in resolved),
                                  omitted=omitted, budget=budget, text_tokenizer=text_tokenizer,
                                  renderer=DSV41_RENDERER if text_tokenizer is not None else SYNTHETIC_NIBBLE16)
        admission_record = self.history.record(ReceiptRecord.of(
            "structure", "context.admission", admission.digest,
            ingress=result.proposal_digest, corpus=admission.corpus, objects=str(len(admission.objects)),
            omitted=str(admission.omitted), tokens=str(len(admission.tokens)),
        ))
        return ContextPreparation(admission, result, ingress_record, admission_record)

    # -- inference: one principal sequence ------------------------------------------------
    def run_text(self, engine, state, text, admission, *, tokenizer, request_id,
                 max_new_tokens, expected_state, emit=None, thinking=False, effort=None,
                 resident_experts=None):
        """Encode chat before begin; stream inert text; commit after finalization.

        Context comes from explicit bounded admission (including an explicitly
        empty admission). No retrieval is implicit in an arbitrary chat string.
        """
        from .text import run_text
        return run_text(self, engine, state, text, admission, tokenizer=tokenizer,
                        request_id=request_id, max_new_tokens=max_new_tokens,
                        expected_state=expected_state, emit=emit, thinking=thinking,
                        effort=effort, resident_experts=resident_experts)

    def run_principal(self, engine: "PrincipalEngine", state, request, admission: ContextAdmission, *,
                      expected_state: str, emit=None, resident_experts=None
                      ) -> tuple["PrincipalResult", RecordedReceipt | None]:
        """Stream one principal sequence to ``emit``; record its commit after finalization."""
        from elpis.inference.principal import PrincipalEngine

        if type(engine) is not PrincipalEngine:
            raise TypeError("run_principal takes a PrincipalEngine")
        if emit is not None and not callable(emit):
            raise TypeError("emit must be callable")
        sequence = engine.begin(state, request, admission, expected_state=expected_state,
                                resident_experts=resident_experts)
        for token in sequence:
            if emit is not None:
                emit(token)
        result = engine.finalize(sequence)
        if result.commit is None:
            return result, None
        commit = result.commit
        return result, self.history.record(ReceiptRecord.of(
            "inference", "principal.commit", commit.digest,
            admission=commit.admission, request=commit.request, state=commit.state,
            outputs=str(len(commit.outputs)), stop=commit.stop_reason,
        ))
