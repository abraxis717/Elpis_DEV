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

Nothing is chained implicitly. No hidden fallback widens authority: every
check that refuses an operation is the owning subsystem's own check, and the
runtime adds a record only after that check passed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from elpis.ecs.projection.contracts import ContextProjection, ProjectionRequest
from elpis.evolution.path_gate import EvolutionPathGate, GateExecuted, GateRejected
from elpis.inference.transaction import DecodeResult, InferenceEngine
from elpis.pipeline.canonical.publisher import CanonicalPublicationReceipt, publish_candidate
from elpis.pipeline.ingress import QueryIngress, QueryIngressResult
from elpis.structure.retrieval.budget import RetrievalBudget
from elpis.structure.retrieval.contracts import RetrievalBundle
from elpis.structure.retrieval.validation import validate_bundle

from .history import HistoryError, ReceiptHistory, ReceiptRecord, RecordedReceipt

__all__ = ("Runtime", "RuntimeConfig")


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
    def decode(self, engine: InferenceEngine, state, request, *, expected_state: str,
               prefetch_enabled: bool = False) -> tuple[DecodeResult, RecordedReceipt | None]:
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
