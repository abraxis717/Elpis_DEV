"""The one runtime composition: explicit subsystem wiring around RuntimeCore.

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
  authority (``evolution_authority``). The exact assertion is durably reserved
  before execution and finalized afterwards. Pending authority and stale
  assertions execute nothing.
* ``anchor_cognition``: the explicit anchor of the first managed K1 lineage.
* ``run_query``: the canonical read-only QUERY, codec -> ``f_W(x)`` of the
  authoritative K1 state -> codec (:mod:`elpis.runtime.cognition`). No state
  change: nothing is begun, committed or published.
* ``run_learn``: the canonical LEARN under an explicit ``LearnAuthority``: the
  admitted experience schedule committed atomically, then its new expected K1
  retained-state identity published to continuity.
* ``run_turn``: the LEGACY learned turn (LEARN, then the decode of its S3
  readout), kept for replay; it needs the same explicit learning authority.
  Without a qualified ECS codec map every cognitive operation refuses with
  ``ECS_CODEC_UNQUALIFIED``: text generation is unavailable.

THE AUTHORITY IS RUNTIMECORE'S, NOT THIS MODULE'S. Every mutable systems decision
of an open runtime belongs to RuntimeCore (native/runtime, Rust; docs/RUNTIME_CORE.md):
its lifecycle and fail-stop disposition, the continuity store, the binding of the
K1 lineage to one native state and its verification against continuity, the
managed turn's native K1 transaction (begin, schedule, commit or abort) with its
continuity publication, and the evolution attempt's durable reservation and
finalization. This module is the compatibility facade: it validates requests at
the language boundary (token codec, ECS codec map, evolution gate), hands
RuntimeCore native-ready values and raises the code RuntimeCore returns. It holds
no fail-stop flag, no lineage state, no transaction and no authority value; the
only state it keeps is a reference that keeps the bound K1 state's owner alive
(lifetime, not authority).

There are no generic receipts and no history. Continuity holds only the current
authority restart and evolution need (docs/CONTINUITY.md).

The runtime composes no DSV model execution (docs/ELPIS_MISSION.md: DSV4
communicates, ECS computes). Nothing is chained implicitly. No hidden
fallback widens authority: every check that refuses an operation is the owning
subsystem's own check.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from elpis.continuity import ContinuityError, ContinuitySnapshot, EvolutionAuthority
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

from .core import RuntimeCore, RuntimeLibrary, describe
from .edges import from_regex_hacf, object_claims
from .errors import CompositionError

__all__ = ("CompositionError", "ContextPreparation", "Runtime", "RuntimeConfig", "RuntimeContinuity")


@dataclass(frozen=True)
class ContextPreparation:
    """What admit_context produced: the frozen admission and the ingress behind it."""

    admission: ContextAdmission
    ingress: QueryIngressResult


@dataclass(frozen=True)
class RuntimeConfig:
    """The runtime's continuity directory and the RuntimeCore library (explicit absolute paths), and the deployment
    pin of the codec admission authority.

    ``runtime_library`` is the built ``libelpis_runtime.so`` (native/runtime), loaded by explicit
    path like every Elpis native library. It embeds the continuity authority (native/continuity).

    ``codec_authority_sha256`` is the independent SHA-256 pin of the one codec authority catalog
    (``elpis.runtime.codec_authority``) whose admissions this runtime's cognitive operations accept. It is
    deployment configuration, never derived from a codec. Without it every managed cognitive operation that is
    given a codec refuses (``CODEC_AUTHORITY``).
    """

    continuity_dir: Path
    runtime_library: Path
    codec_authority_sha256: str | None = None

    def __post_init__(self):
        for value in (self.continuity_dir, self.runtime_library):
            if not isinstance(value, Path) or not value.is_absolute():
                raise CompositionError("RUNTIME_PATH", "runtime paths must be absolute Paths")
        pin = self.codec_authority_sha256
        if pin is not None and (type(pin) is not str or len(pin) != 64
                                or any(c not in "0123456789abcdef" for c in pin)):
            raise CompositionError("CODEC_AUTHORITY", "codec_authority_sha256: 64 lowercase hex characters")


class RuntimeContinuity:
    """The open runtime's continuity authority as RuntimeCore exposes it. It holds nothing.

    ``snapshot`` reads the current durable record (also while fail-stopped: it is what restart
    sees). ``commit_evolution_transition`` is the explicit reconciliation of a durable pending
    evolution authority with an externally established receipt. The testing library adds fault
    injection and the store's I/O counters. ``library`` is the record codec of the same library.
    Refusals raise :class:`~elpis.continuity.ContinuityError` with RuntimeCore's stable code.
    """

    def __init__(self, core: RuntimeCore):
        self._core = core
        self.library = core.library.continuity
        self.directory = core.directory

    @staticmethod
    def _call(fn, *args):
        try:
            return fn(*args)
        except CompositionError as exc:
            raise ContinuityError(exc.code, str(exc)) from exc

    def snapshot(self) -> ContinuitySnapshot:
        return self._call(self._core.snapshot)

    def commit_evolution_transition(self, expected: EvolutionAuthority, receipt_digest: str) -> ContinuitySnapshot:
        return self._call(self._core.evolution_reconcile, expected, receipt_digest)

    def testing_fault(self, publication: int, action: int, arg: int = 0) -> None:
        self._call(self._core.testing_fault, publication, action, arg)

    def testing_counters(self, reset: bool = False) -> dict[str, int]:
        return self._call(self._core.testing_io_counters, reset)


class Runtime:
    def __init__(self, config: RuntimeConfig):
        if type(config) is not RuntimeConfig:
            raise TypeError("config must be a RuntimeConfig")
        self.config = config
        self._core = RuntimeCore(RuntimeLibrary(config.runtime_library), config.continuity_dir)
        self.continuity = RuntimeContinuity(self._core)
        # Keeps the owner of the bound K1 state alive, so its identity cannot be reused while
        # RuntimeCore holds the binding (lifetime only: RuntimeCore decides the binding), and the bound
        # state itself, so close can give it back (release its managed lease).
        self._bound_owner = None
        self._bound_substrate = None

    def open(self) -> "Runtime":
        # An open runtime is refused and keeps its turn, so the owner is released only once RuntimeCore opened
        # (unbound) and can retain nothing of it.
        self._core.open()
        self._bound_owner = self._bound_substrate = None
        return self

    def close(self) -> None:
        """Give the bound K1 state back (end any open managed turn on it natively, release its managed lease), then
        close RuntimeCore. A state its owner already closed has nothing to give back."""
        substrate, self._bound_substrate = self._bound_substrate, None
        if substrate is not None:
            try:
                descriptor, _ = describe(substrate)
            except CompositionError:
                descriptor = None   # the owner closed the state: its lease went with it
            if descriptor is not None:
                for step in (self._core.turn_abort, self._core.release):
                    try:
                        step(descriptor)
                    except CompositionError:
                        pass        # no open turn; closed or fail-stopped runtime; a lease another owner took
        # RuntimeCore aborts an open managed turn natively before the bound owner is released.
        self._core.close()
        self._bound_owner = None

    def release(self, substrate) -> None:
        """Give the bound K1 state back: RuntimeCore releases its managed lease and unbinds it, so its owner may
        use it unmanaged (outside this lineage) again; the next cognitive operation binds and verifies again."""
        descriptor, _ = describe(substrate)
        self._core.release(descriptor)
        self._bound_owner = self._bound_substrate = None

    def __enter__(self) -> "Runtime":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def fault(self) -> str | None:
        """RuntimeCore's fail-stop disposition (None while live)."""
        return self._core.fault()

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
        authority = self._core.evolution_authority().evolution
        return EvolutionAuthorityBinding(authority.revision, authority.digest, authority.head)

    def evolve(self, gate: EvolutionPathGate, *, assertion, state, advance, advance_kwargs
               ) -> GateRejected | GateExecuted:
        """Validate (the gate), reserve (RuntimeCore), execute once (the gate), finalize (RuntimeCore).

        RuntimeCore refuses while fail-stopped or pending, durably reserves the exact assertion
        before anything executes and fail-stops on any reservation or finalization failure. An
        attempt that raises or returns an invalid result may already have had an external effect:
        RuntimeCore keeps its reservation pending and fail-stops; nothing is retried or inferred.
        """
        if type(gate) is not EvolutionPathGate:
            raise TypeError("evolve takes an EvolutionPathGate")
        current = self._core.evolution_authority().evolution
        binding = EvolutionAuthorityBinding(current.revision, current.digest, current.head)
        reason = gate.reject_reason(assertion, state, binding)
        if reason is not None:
            return GateRejected(False, reason, 0)
        self._core.evolution_reserve(current, assertion.digest)
        try:
            result = gate.execute(assertion=assertion, state=state, authority=binding,
                                  advance=advance, advance_kwargs=advance_kwargs)
            if not isinstance(result, GateExecuted):
                raise CompositionError("CONTINUITY_EVOLUTION_PENDING", "admission changed after reservation")
        except BaseException:
            self._core.evolution_abandon()
            raise
        self._core.evolution_finalize(result.receipt.receipt_digest)
        return result

    # -- structure -> codec: ingress -> adapter -> HACF resolution -> rendering ----------
    def admit_context(self, *, ingress: QueryIngress, task: bytes, corpus_root: Path, corpus_manifest,
                      context_snapshot: str, model: str, tokenizer: str, budget: ContextBudget,
                      max_document_bytes: int, rules: tuple = (), text_tokenizer=None,
                      hacf_handle=None) -> ContextPreparation:
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
        if hacf_handle is not None and ingress._owner is not hacf_handle:
            raise CompositionError("CORPUS_IDENTITY", "ingress is not borrowed from the same HACF epoch")
        if ingress._owner is not None and hacf_handle is None:
            raise CompositionError("CORPUS_IDENTITY", "borrowed ingress requires its source handle")
        resolved, omitted = resolve_chunks(corpus_root, manifest, claims, max_objects=budget.max_objects,
                                           max_text_bytes=budget.max_bytes,
                                           max_document_bytes=max_document_bytes, source_handle=hacf_handle)
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

        RuntimeCore reads the K1 identity only and never mutates K1. Refused if a lineage is
        already anchored.
        """
        descriptor, owner = describe(substrate)
        anchored = self._core.anchor(descriptor)
        self._bound_owner, self._bound_substrate = owner, substrate
        return anchored

    def _codec_pin(self) -> str:
        """The deployment pin every admitted codec's authority must match; ``""`` (matches nothing) when none is
        configured, so admission refuses (``CODEC_AUTHORITY``) after the fail-closed missing-codec check."""
        return self.config.codec_authority_sha256 or ""

    # -- cognition: QUERY (read-only) ---------------------------------------------------------------------------
    def run_query(self, substrate, request):
        """One managed QUERY (docs/COGNITION_R0.md): read-only.

        The boundary validates the :class:`~elpis.runtime.cognition.QueryRequest` and encodes the query rows.
        RuntimeCore answers ``f_W(x)`` from the lineage's authoritative K1 state in one native call that also
        identifies the state; the identity must be the durable expected one (a mismatch fail-stops and the answer is
        withheld). Nothing is begun, committed, published or written: W, epoch, H, a and continuity are unchanged.
        """
        self._core.require_live()

        from .cognition import QueryReadout, QueryResult, _decode_query, _encode_query, _validate_query_request
        from .fuel import admit_query

        codec_map, classification = _validate_query_request(substrate, request, self._codec_pin())
        descriptor, owner = describe(substrate)
        tokens, stimulus = _encode_query(substrate, request.text, request.tokenizer, codec_map)
        admit_query(stimulus, substrate.dim, substrate.width, request.budget, request.max_output_tokens)
        if self._bound_owner is None:
            self._bound_owner, self._bound_substrate = owner, substrate   # what this query binds stays alive
        values, digest = self._core.query(descriptor, stimulus, request.budget)   # RuntimeCore re-admits natively
        readout = QueryReadout(values, substrate.dim, substrate.width, digest)
        output, rendered = _decode_query(codec_map, readout, request.tokenizer, request.max_output_tokens)
        return QueryResult(tokens, output, rendered, readout, classification)

    # -- cognition: LEARN (explicit authority) -------------------------------------------------------------------
    def run_learn(self, substrate, request):
        """One managed LEARN under an explicit :class:`~elpis.runtime.cognition.LearnAuthority`.

        RuntimeCore verifies the K1 lineage, runs the admitted experience schedule on a native candidate, commits it
        atomically and publishes the new retained-state identity to continuity, fail-stopping when that publication
        is not certain. A refused LEARN installs nothing.
        """
        self._core.require_live()

        from .cognition import LearnResult, _encode, _validate_learn_request
        from .fuel import admit_learn

        codec_map, classification, authority = _validate_learn_request(substrate, request, self._codec_pin())
        descriptor, owner = describe(substrate)
        tokens, stimulus = _encode(substrate, request.text, request.tokenizer, codec_map)
        admit_learn(stimulus, substrate.dim, substrate.width, authority.budget)
        if self._bound_owner is None:
            self._bound_owner, self._bound_substrate = owner, substrate   # what this begin binds stays alive
        self._core.turn_begin(descriptor, stimulus, authority.learning_rate, authority.budget)
        committed, _ = self._core.turn_commit(descriptor)
        return LearnResult(tokens, stimulus.experiences, committed.commit.epoch_before, committed.commit.epoch_after,
                           classification, authority.grant, committed.state_before_digest,
                           committed.state_after_digest)

    # -- cognition: LEGACY learned turn (LEARN + S3 decode) --------------------------------------------------------
    def run_turn(self, substrate, text, *, tokenizer, codec=None, authority=None, max_output_tokens=256):
        """One managed LEGACY learned turn: a LEARN under explicit authority whose S3 readout is decoded.

        Kept for the replay of the original synthetic scaffold (``LEGACY_LEARNED_TURN``); it is never a query. The
        boundary validates the request, encodes the text and the stimulus and decodes the readout. RuntimeCore
        verifies the K1 lineage, runs the native K1 transaction on its candidate, commits it after the decode
        succeeded and publishes the committed identity to continuity, fail-stopping when that publication is not
        certain. A refused or aborted turn installs nothing.
        """
        self._core.require_live()

        from .cognition import Readout, TurnResult, _admit_turn, _decode, _encode, _validate_turn_request

        codec_map, classification, authority = _validate_turn_request(
            substrate, text, codec=codec, authority=authority, max_output_tokens=max_output_tokens,
            deployment_pin=self._codec_pin())
        descriptor, owner = describe(substrate)
        tokens, stimulus = _encode(substrate, text, tokenizer, codec_map)
        _admit_turn(substrate, stimulus, authority, max_output_tokens)
        if self._bound_owner is None:
            self._bound_owner, self._bound_substrate = owner, substrate   # what this begin binds stays alive
        begun = self._core.turn_begin(descriptor, stimulus, authority.learning_rate, authority.budget)
        try:
            readout = Readout(begun.s3, begun.epoch_after, substrate.dim, substrate.width)
            output, rendered = _decode(codec_map, readout, tokenizer, max_output_tokens)
        except BaseException:
            self._core.turn_abort(descriptor)
            raise
        committed, _ = self._core.turn_commit(descriptor)
        return TurnResult(tokens, output, rendered, readout, committed.commit.epoch_before,
                          committed.commit.epoch_after, classification, committed.state_before_digest,
                          committed.state_after_digest)
