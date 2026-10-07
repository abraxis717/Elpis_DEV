"""The one runtime composition: explicit subsystem wiring around one continuity authority.

``Runtime`` does not own models, corpora, canonical roots or targets. The
caller constructs each of them explicitly (explicit library paths, corpus
roots, file assets, publication ledgers) and passes them in. The runtime runs
one subsystem operation at a time through that subsystem's own fail-closed
entry point and returns that subsystem's result:

* ``run_ingress``: pipeline ingress on caller bytes.
* ``admit_retrieval``: structure retrieval-bundle validation against the
  expected query and corpus.
* ``publish_canonical``: the canonical Grid81 publisher, with the caller's
  one-use promotion capability. The pipeline ledger owns publication
  durability and idempotent replay.
* ``admit_context``: structural memory rendered through the DSV4 codec.
  Pipeline ingress on the turn's bytes, then the edge adapter (address
  proposals and chunk claims from the pinned export), then HACF object
  resolution (verified document bytes, recomputed chunk identity, pinned
  corpus manifest), then a budgeted, frozen token rendering. It is a
  communication operation: it does not make HACF a context window for a model.
* ``evolve``: the evolution path gate, bound to the current evolution
  authority held by continuity (``evolution_authority``). The exact assertion
  is durably reserved before execution and finalized afterwards. Pending
  authority and stale assertions execute nothing.
* ``anchor_cognition`` / ``run_turn``: the canonical cognitive turn, codec ->
  ECS -> codec (:mod:`elpis.runtime.cognition`), over a native K1 state of the
  one ECS. The first managed lineage is anchored explicitly. Each committed
  turn publishes its new expected K1 retained-state identity to continuity.
  Without a qualified ECS codec map the turn refuses with
  ``ECS_CODEC_UNQUALIFIED``: text generation is unavailable.

There are no generic receipts and no history. ``elpis.continuity`` holds only
the current authority restart and evolution need (docs/CONTINUITY.md).

The runtime composes no DSV model execution (docs/ELPIS_MISSION.md: DSV4
communicates, ECS computes). Nothing is chained implicitly. No hidden
fallback widens authority: every check that refuses an operation is the owning
subsystem's own check.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from elpis.continuity import ContinuityError, ContinuityLibrary, ContinuitySnapshot, ContinuityStore
from elpis.ECS.k1 import K1Error
from elpis.evolution.path_gate import (
    EvolutionAuthorityBinding,
    EvolutionPathGate,
    GateExecuted,
    GateRejected,
)
from elpis.inference.admission import ContextAdmission, ContextBudget, admit_context
from elpis.pipeline.canonical.publisher import CanonicalPublicationReceipt, publish_candidate
from elpis.pipeline.ingress import QueryIngress, QueryIngressResult
from elpis.structure.retrieval.budget import RetrievalBudget
from elpis.structure.retrieval.contracts import RetrievalBundle
from elpis.structure.retrieval.objects import CorpusManifest, resolve_chunks
from elpis.structure.retrieval.validation import validate_bundle
from elpis.substrate.digests import raw_digest

from .edges import from_regex_hacf, object_claims

__all__ = ("CompositionError", "ContextPreparation", "Runtime", "RuntimeConfig")


class CompositionError(RuntimeError):
    """A composed operation was refused by one of its stages (fail closed)."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class ContextPreparation:
    """What admit_context produced: the frozen admission and the ingress behind it."""

    admission: ContextAdmission
    ingress: QueryIngressResult


@dataclass(frozen=True)
class RuntimeConfig:
    """The runtime's continuity directory and the continuity library (explicit absolute paths).

    ``continuity_library`` is the built ``libelpis_continuity.so`` (native/continuity), loaded by
    explicit path like every Elpis native library.
    """

    continuity_dir: Path
    continuity_library: Path

    def __post_init__(self):
        for value in (self.continuity_dir, self.continuity_library):
            if not isinstance(value, Path) or not value.is_absolute():
                raise CompositionError("CONTINUITY_PATH", "continuity paths must be absolute Paths")


def _k1_identity(substrate) -> bytes:
    try:
        digest = substrate.state_digest()
    except (AttributeError, TypeError) as exc:
        raise CompositionError("ECS_STATE", "a native K1 state with state_digest() is required") from exc
    except K1Error as exc:
        raise CompositionError("ECS_REFUSED", str(exc)) from exc
    if type(digest) is not bytes or len(digest) != 32:
        raise CompositionError("ECS_STATE", "K1 state_digest() must return 32 bytes")
    return digest


class Runtime:
    def __init__(self, config: RuntimeConfig):
        if type(config) is not RuntimeConfig:
            raise TypeError("config must be a RuntimeConfig")
        self.config = config
        try:
            self.continuity = ContinuityStore(ContinuityLibrary(config.continuity_library), config.continuity_dir)
        except ContinuityError as exc:
            raise CompositionError(exc.code, str(exc)) from exc
        # A continuity publication that did not become certain after an ECS
        # commit (or a lineage mismatch) fail-stops lineage-dependent work
        # until the runtime is reopened and reconciled.
        self._continuity_fault: str | None = None
        self._turn_substrate = None

    def open(self) -> "Runtime":
        try:
            self.continuity.open()
        except ContinuityError as exc:
            raise CompositionError(exc.code, str(exc)) from exc
        self._continuity_fault = None
        self._turn_substrate = None
        return self

    def close(self) -> None:
        self.continuity.close()

    def __enter__(self) -> "Runtime":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def _snapshot(self) -> ContinuitySnapshot:
        if self._continuity_fault is not None:
            raise CompositionError(self._continuity_fault,
                                   "runtime is fail-stopped pending restart/reconciliation")
        try:
            return self.continuity.snapshot()
        except ContinuityError as exc:
            raise CompositionError(exc.code, str(exc)) from exc

    # -- pipeline: bounded ingress ---------------------------------------------------
    def run_ingress(self, ingress: QueryIngress, task: bytes) -> QueryIngressResult:
        if type(ingress) is not QueryIngress or type(task) is not bytes:
            raise TypeError("run_ingress takes a QueryIngress handle and task bytes")
        return ingress.run(task)

    # -- structure: retrieval admission -------------------------------------------
    def admit_retrieval(self, bundle: RetrievalBundle, *, expected_query: str, expected_corpus: str,
                        budget: RetrievalBudget | None = None):
        if type(bundle) is not RetrievalBundle:
            raise TypeError("admit_retrieval takes a RetrievalBundle")
        return validate_bundle(bundle, expected_query, expected_corpus, budget or RetrievalBudget())

    # -- pipeline: canonical publication --------------------------------------------
    def publish_canonical(self, *, project_root, candidate_root, ledger, promotion_capability,
                          lock_path=None) -> CanonicalPublicationReceipt:
        return publish_candidate(project_root=project_root, candidate_root=candidate_root,
                                 ledger=ledger, promotion_capability=promotion_capability,
                                 lock_path=lock_path)

    # -- evolution: gated attempts bound to the current evolution authority -------------
    def evolution_authority(self) -> EvolutionAuthorityBinding:
        """The current evolution authority an assertion must be built against."""
        authority = self._snapshot().evolution
        if authority.pending_assertion is not None:
            raise CompositionError("CONTINUITY_EVOLUTION_PENDING", "explicit reconciliation required")
        return EvolutionAuthorityBinding(authority.revision, authority.digest, authority.head)

    def evolve(self, gate: EvolutionPathGate, *, assertion, state, advance, advance_kwargs
               ) -> GateRejected | GateExecuted:
        if type(gate) is not EvolutionPathGate:
            raise TypeError("evolve takes an EvolutionPathGate")
        current = self._snapshot().evolution
        if current.pending_assertion is not None:
            raise CompositionError("CONTINUITY_EVOLUTION_PENDING", "explicit reconciliation required")
        binding = EvolutionAuthorityBinding(current.revision, current.digest, current.head)
        reason = gate.reject_reason(assertion, state, binding)
        if reason is not None:
            return GateRejected(False, reason, 0)
        try:
            pending = self.continuity.reserve_evolution_assertion(current, assertion.digest).evolution
        except ContinuityError as exc:
            self._continuity_fault = exc.code
            raise CompositionError(exc.code, "evolution reservation failed; no attempt executed") from exc
        try:
            result = gate.execute(assertion=assertion, state=state, authority=binding,
                                  advance=advance, advance_kwargs=advance_kwargs)
            if not isinstance(result, GateExecuted):
                raise CompositionError("CONTINUITY_EVOLUTION_PENDING", "admission changed after reservation")
            self.continuity.commit_evolution_transition(pending, result.receipt.receipt_digest)
        except ContinuityError as exc:
            self._continuity_fault = exc.code
            raise CompositionError(exc.code, "evolution attempt executed; finalization not certain") from exc
        except BaseException:
            # Even an exception or an invalid result may follow an external
            # effect. Preserve the reservation; never infer that retry is safe.
            self._continuity_fault = "CONTINUITY_EVOLUTION_PENDING"
            raise
        return result

    # -- structure -> codec: ingress -> adapter -> HACF resolution -> rendering ----------
    def admit_context(self, *, ingress: QueryIngress, task: bytes, corpus_root: Path, corpus_manifest,
                      context_snapshot: str, model: str, tokenizer: str, budget: ContextBudget,
                      max_document_bytes: int, rules: tuple = (), text_tokenizer=None) -> ContextPreparation:
        """Resolve proposed HACF objects and render them, frozen, through the DSV4 codec.

        ``corpus_manifest`` is the corpus manifest JSON the caller obtained from
        its HACF handle; it must hash to the digest the ingress result pins.
        Any refusal (ingress fail-closed, a tampered export, blob or manifest,
        an unproposed object) raises and admits nothing.
        """
        if type(budget) is not ContextBudget:
            raise TypeError("admit_context takes a ContextBudget")
        result = self.run_ingress(ingress, task)
        if not result.batch_published:
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
        return ContextPreparation(admission, result)

    # -- cognition: K1 lineage ------------------------------------------------------------
    def anchor_cognition(self, substrate) -> ContinuitySnapshot:
        """Explicitly anchor the first managed K1 lineage at the substrate's retained state.

        Reads the K1 identity only; never mutates K1. Refused if a lineage is
        already anchored.
        """
        if self._snapshot().anchored:
            raise CompositionError("CONTINUITY_ALREADY_ANCHORED", "durable K1 lineage already exists")
        identity = _k1_identity(substrate)
        try:
            anchored = self.continuity.anchor_cognition(identity)
        except ContinuityError as exc:
            if exc.code == "CONTINUITY_PUBLICATION_UNCERTAIN":
                self._continuity_fault = exc.code
            raise CompositionError(exc.code, "no K1 mutation happened") from exc
        self._turn_substrate = substrate
        return anchored

    def _reconcile_cognition_substrate(self, substrate) -> None:
        if self._turn_substrate is not None:
            if substrate is not self._turn_substrate:
                raise CompositionError("COGNITION_SUBSTRATE_SWITCH", "one open Runtime owns one K1 lineage")
            return
        expected = self._snapshot().k1_state_digest
        if expected is None:
            raise CompositionError(
                "CONTINUITY_UNANCHORED",
                "anchor_cognition(substrate) is required before the first managed K1 turn",
            )
        if _k1_identity(substrate) != expected:
            self._continuity_fault = "CONTINUITY_STATE_MISMATCH"
            raise CompositionError(
                self._continuity_fault,
                "current K1 retained state does not match the durable expected identity",
            )
        self._turn_substrate = substrate

    # -- cognition: DSV4 codec -> ECS -> DSV4 codec -> continuity ---------------------------
    def run_turn(self, substrate, text, *, tokenizer, codec_map=None, learning_rate=None, max_output_tokens=256):
        """Commit one canonical K1 turn, then publish its new expected K1 identity to continuity."""
        if self._continuity_fault is not None:
            raise CompositionError(self._continuity_fault,
                                   "runtime is fail-stopped pending restart/reconciliation")

        from .cognition import _validate_turn_request, run_turn

        _validate_turn_request(substrate, text, codec_map=codec_map, learning_rate=learning_rate,
                               max_output_tokens=max_output_tokens)
        self._reconcile_cognition_substrate(substrate)

        result = run_turn(substrate, text, tokenizer=tokenizer, codec_map=codec_map,
                          learning_rate=learning_rate, max_output_tokens=max_output_tokens)

        # The K1 commit is final. Publication either makes its retained-state
        # identity the durable expectation or fail-stops the runtime; a K1
        # commit is never rolled back and no transition is ever synthesized.
        try:
            self.continuity.commit_cognition_transition(result.state_before_digest, result.state_after_digest)
        except ContinuityError as exc:
            self._continuity_fault = exc.code
            raise CompositionError(exc.code, "K1 turn committed; continuity publication failed") from exc
        return result
