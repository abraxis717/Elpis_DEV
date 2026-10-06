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
* ``admit_context``: structural memory rendered through the DSV4 codec.
  Pipeline ingress on the turn's bytes, then the edge adapter (address
  proposals and chunk claims from the pinned export), then HACF object
  resolution (verified document bytes, recomputed chunk identity, pinned
  corpus manifest), then a budgeted, frozen token rendering. The ingress
  proposal and the admission are recorded. It is a communication operation:
  it does not make HACF a context window for a model.

The runtime composes no DSV model execution (docs/ELPIS_MISSION.md: DSV4
communicates, ECS computes). Model decode transactions and principal
sequences are noncanonical inference mechanics; the runtime neither runs nor
records them.

* ``run_turn``: the canonical cognitive turn, codec -> ECS -> codec
  (:mod:`elpis.runtime.cognition`), over a native K1 state (standalone or
  FMS-resident). Without a qualified ECS codec map it refuses with
  ``ECS_CODEC_UNQUALIFIED``: text generation is unavailable.

Nothing is chained implicitly. No hidden fallback widens authority: every
check that refuses an operation is the owning subsystem's own check, and the
runtime adds a record only after that check passed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from elpis.ECS_C.projection.contracts import ContextProjection, ProjectionRequest
from elpis.inference.admission import ContextAdmission, ContextBudget, admit_context
from elpis.evolution.path_gate import EvolutionPathGate, GateExecuted, GateRejected
from elpis.pipeline.canonical.publisher import CanonicalPublicationReceipt, publish_candidate
from elpis.pipeline.ingress import QueryIngress, QueryIngressResult
from elpis.structure.retrieval.budget import RetrievalBudget
from elpis.structure.retrieval.contracts import RetrievalBundle
from elpis.structure.retrieval.objects import CorpusManifest, resolve_chunks
from elpis.structure.retrieval.validation import validate_bundle
from elpis.substrate.digests import raw_digest

from elpis.ECS_G.k1 import K1Error

from .edges import from_regex_hacf, object_claims
from .history import HistoryError, ReceiptHistory, ReceiptRecord, RecordedReceipt, RuntimeHistoryPolicy

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
    """Runtime-owned bounded durable history and its explicit native ECS_C writer.

    ``history_policy`` is the finite storage policy of the history directory
    (segment, checkpoint and total directory bounds). It always has finite
    values; there is no unlimited setting.
    """

    history_dir: Path
    history_native_library: Path | None = None
    history_policy: RuntimeHistoryPolicy = RuntimeHistoryPolicy()

    def __post_init__(self):
        if type(self.history_policy) is not RuntimeHistoryPolicy:
            raise HistoryError("HISTORY_POLICY", "history_policy must be a RuntimeHistoryPolicy")
        if not isinstance(self.history_dir, Path) or not self.history_dir.is_absolute():
            raise HistoryError("HISTORY_PATH", "history_dir must be an absolute Path")
        if self.history_native_library is not None:
            if (
                not isinstance(self.history_native_library, Path)
                or not self.history_native_library.is_absolute()
                or not self.history_native_library.is_file()
            ):
                raise HistoryError(
                    "NATIVE_HISTORY_LIBRARY",
                    "history_native_library must be an existing absolute Path",
                )


class Runtime:
    def __init__(self, config: RuntimeConfig):
        if type(config) is not RuntimeConfig:
            raise TypeError("config must be a RuntimeConfig")
        self.config = config
        self.history = ReceiptHistory(
            config.history_dir,
            native_library=config.history_native_library,
            policy=config.history_policy,
        )
        self._turn_continuity_fault: str | None = None
        self._turn_substrate = None

    def open(self) -> "Runtime":
        self.history.open()
        self._turn_continuity_fault = None
        self._turn_substrate = None
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

    # -- structure -> codec: ingress -> adapter -> HACF resolution -> rendering ----------
    def admit_context(self, *, ingress: QueryIngress, task: bytes, corpus_root: Path, corpus_manifest,
                      context_snapshot: str, model: str, tokenizer: str, budget: ContextBudget,
                      max_document_bytes: int, rules: tuple = (), text_tokenizer=None) -> ContextPreparation:
        """Resolve proposed HACF objects and render them, frozen, through the DSV4 codec.

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


    def _cognition_chain_tip(self):
        """Tip of the durable K1 lineage from the history's fixed-size fold.

        The fold (carried across compaction in the checkpoint) applies the
        same lineage rules to every continuity receipt ever recorded; restart
        never re-reads retired receipts.
        """
        try:
            return self.history.cognition_tip()
        except HistoryError as exc:
            raise CompositionError(exc.code, str(exc)) from exc

    def anchor_cognition(self, substrate):
        if self._turn_continuity_fault is not None:
            raise CompositionError(
                self._turn_continuity_fault,
                "runtime cognition is fail-stopped pending restart/reconciliation",
            )

        if self._cognition_chain_tip() is not None:
            raise CompositionError(
                "COGNITION_ALREADY_ANCHORED",
                "durable cognition continuity already exists",
            )

        try:
            state = substrate.state_digest()
        except (AttributeError, TypeError) as exc:
            raise CompositionError(
                "ECS_STATE",
                "a native K1 state with state_digest() is required",
            ) from exc
        except K1Error as exc:
            raise CompositionError("ECS_REFUSED", str(exc)) from exc

        if type(state) is not bytes or len(state) != 32:
            raise CompositionError(
                "ECS_STATE",
                "K1 state_digest() must return 32 bytes",
            )

        state_hex = state.hex()

        record = ReceiptRecord.of(
            "ecs_g",
            "cognition.anchor",
            state_hex,
            mechanism="1",
            state=state_hex,
        )

        try:
            self.history.record(record)
        except HistoryError as exc:
            native_code = getattr(exc.__cause__, "code", None)

            if exc.code == "NATIVE_HISTORY_RECORD" and native_code in (-8, -10):
                code = "HISTORY_UNCERTAIN_BEFORE_ECS_COMMIT"
                self._turn_continuity_fault = code
            else:
                code = "HISTORY_REFUSED_BEFORE_ECS_COMMIT"

            raise CompositionError(code, str(exc)) from exc

        self._turn_substrate = substrate
        return record

    def _reconcile_cognition_substrate(self, substrate):
        if self._turn_substrate is not None:
            if substrate is not self._turn_substrate:
                raise CompositionError(
                    "COGNITION_SUBSTRATE_SWITCH",
                    "one open Runtime owns one K1 cognition lineage",
                )
            return

        expected = self._cognition_chain_tip()

        if expected is None:
            raise CompositionError(
                "COGNITION_UNANCHORED",
                "anchor_cognition(substrate) is required before the first managed K1 turn",
            )

        try:
            current = substrate.state_digest()
        except (AttributeError, TypeError) as exc:
            raise CompositionError(
                "ECS_STATE",
                "a native K1 state with state_digest() is required",
            ) from exc
        except K1Error as exc:
            raise CompositionError("ECS_REFUSED", str(exc)) from exc

        if type(current) is not bytes or len(current) != 32:
            raise CompositionError(
                "ECS_STATE",
                "K1 state_digest() must return 32 bytes",
            )

        if current.hex() != expected:
            self._turn_continuity_fault = "HISTORY_STATE_MISMATCH"
            raise CompositionError(
                self._turn_continuity_fault,
                "current K1 retained state does not match durable cognition continuity",
            )

        self._turn_substrate = substrate


    # -- cognition: DSV4 codec -> ECS -> DSV4 codec -> ECS_C continuity -----------------------
    def run_turn(self, substrate, text, *, tokenizer, codec_map=None, learning_rate=None, max_output_tokens=256):
        """Commit one canonical K1 turn, then record exactly one ``ecs_g / cognition.turn`` receipt."""
        if self._turn_continuity_fault is not None:
            raise CompositionError(
                self._turn_continuity_fault,
                "runtime cognition is fail-stopped pending restart/reconciliation",
            )

        from .cognition import _validate_turn_request, run_turn

        _validate_turn_request(
            substrate,
            text,
            codec_map=codec_map,
            learning_rate=learning_rate,
            max_output_tokens=max_output_tokens,
        )

        self._reconcile_cognition_substrate(substrate)

        result = run_turn(
            substrate,
            text,
            tokenizer=tokenizer,
            codec_map=codec_map,
            learning_rate=learning_rate,
            max_output_tokens=max_output_tokens,
        )

        continuity = result.continuity
        if continuity is None:
            self._turn_continuity_fault = "HISTORY_REFUSED_AFTER_ECS_COMMIT"
            raise CompositionError(
                self._turn_continuity_fault,
                "committed K1 turn has no continuity identity",
            )

        record = ReceiptRecord.of(
            "ecs_g",
            "cognition.turn",
            continuity.digest,
            codec=continuity.codec,
            epoch_after=str(continuity.epoch_after),
            epoch_before=str(continuity.epoch_before),
            generation_after=str(continuity.generation_after),
            generation_before=str(continuity.generation_before),
            input_tokens=continuity.input_tokens_digest,
            mechanism=continuity.mechanism,
            output_tokens=continuity.output_tokens_digest,
            readout=continuity.readout_digest,
            state_after=continuity.state_after_digest,
            state_before=continuity.state_before_digest,
            stimulus=continuity.stimulus_digest,
        )

        try:
            self.history.record(record)
        except HistoryError as exc:
            native_code = getattr(exc.__cause__, "code", None)
            if exc.code == "NATIVE_HISTORY_RECORD" and native_code in (-8, -10):
                code = "HISTORY_UNCERTAIN_AFTER_ECS_COMMIT"
            else:
                code = "HISTORY_REFUSED_AFTER_ECS_COMMIT"
            self._turn_continuity_fault = code
            raise CompositionError(code, str(exc)) from exc

        return result
