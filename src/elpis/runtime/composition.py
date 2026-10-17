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
  authority (``evolution_authority``). The gate must be issued by the
  deployment-pinned evolution policy (``elpis.evolution.policy``;
  ``RuntimeConfig.evolution_policy_sha256``): a caller-built gate is refused.
  The exact assertion is durably reserved before execution and finalized
  afterwards. Pending authority and stale assertions execute nothing.
* ``anchor_cognition``: the explicit anchor of the first managed K1 lineage.
* ``run_query``: the canonical read-only QUERY, codec -> ``f_W(x)`` of the
  authoritative K1 state -> codec (:mod:`elpis.runtime.cognition`). No state
  change: nothing is begun, committed or published.
* ``run_learn``: the canonical LEARN under an explicit ``LearnAuthority``: the
  admitted experience schedule committed atomically into the live state. The
  commit is VOLATILE: nothing is checkpointed or published, and no I/O happens.
* ``run_query``, ``run_learn`` and ``run_turn`` take an optional ``context``: a
  contextual-lane :class:`~elpis.runtime.context.ContextLane` (its current
  published view, one reference read that never waits for the WARM worker) or
  a pinned :class:`~elpis.runtime.context.ContextView`. The result records the
  view's coordinate (``result.context``); no encoding of a view into the K1
  stimulus exists (docs/ARCHITECTURE.md, "Execution lanes").
* ``publish_recoverable`` / ``publish_identity``: the explicit durable
  publication of the live state, of one of two kinds never confused
  (docs/RUNTIME_CORE.md, "Recoverable and identity-only publication"):
  RuntimeCore captures it and its durability lane makes it durable off the HOT
  lane; the caller chooses whether to wait. ``publish_recoverable`` needs the
  checkpoint store and is ``ACKNOWLEDGED`` only with a read-back verified slot;
  ``publish_identity`` (no store) resolves ``IDENTITY_PUBLISHED``, never an
  acknowledgment. ``durability`` reads the lane's state; ``live_identity``
  reads the live state's identity on request.
* ``recover_k1``: K1 Recovery R0 (docs/K1_RECOVERY_R0.md), read-only: what
  restart may do with the lineage, and the complete authorized envelope when it
  is resumable. ``discard_k1_candidate`` / ``adopt_k1_candidate`` are the
  explicit operator reconciliation of an unresolved checkpoint candidate.
* ``run_turn``: the LEGACY learned turn (LEARN, then the decode of its S3
  readout), kept for replay; it needs the same explicit learning authority.
  Without a qualified ECS codec map every cognitive operation refuses with
  ``ECS_CODEC_UNQUALIFIED``: text generation is unavailable.

THE AUTHORITY IS RUNTIMECORE'S, NOT THIS MODULE'S. Every mutable systems decision
of an open runtime belongs to RuntimeCore (native/runtime, Rust; docs/RUNTIME_CORE.md):
its lifecycle and fail-stop disposition, the binding of the K1 lineage to one native
state (verified once against continuity) and its live coordinate, the managed
turn's native K1 transaction (begin, schedule, volatile commit or abort), the
durability lane (explicit durable publication) and the evolution attempt's durable
reservation and finalization. This module is the compatibility facade: it validates requests at
the language boundary (token codec, ECS codec map, evolution gate), hands
RuntimeCore native-ready values and raises the code RuntimeCore returns. It holds
no fail-stop flag, no lineage state, no transaction and no authority value; the
only state it keeps is a reference that keeps the bound K1 state's owner alive
(lifetime, not authority).

There are no generic receipts and no history. Continuity holds only the current
durable authority restart and evolution need (docs/CONTINUITY.md); the live state
may be ahead of it, and a crash loses at most the volatile work since the last
acknowledged durable publication.

The runtime composes no DSV model execution (docs/ELPIS_MISSION.md: DSV4
communicates, ECS computes). Nothing is chained implicitly. No hidden
fallback widens authority: every check that refuses an operation is the owning
subsystem's own check.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from elpis.continuity import ContinuityError, ContinuitySnapshot, EvolutionAuthority
from elpis.ECS.composite import CompositeState
from elpis.evolution.path_gate import (
    EvolutionAuthorityBinding,
    EvolutionPathGate,
    GateExecuted,
    GateRejected,
)
from elpis.evolution.policy import EvolutionPolicyError, policy_of
from elpis.inference.admission import ContextAdmission, ContextBudget, admit_context
from elpis.pipeline.canonical.publisher import CanonicalPublicationReceipt, publish_candidate
from elpis.pipeline.ingress import QueryIngress, QueryIngressResult
from elpis.structure.retrieval.budget import RetrievalBudget
from elpis.structure.retrieval.contracts import RetrievalBundle
from elpis.structure.retrieval.objects import CorpusManifest, resolve_chunks
from elpis.structure.retrieval.validation import validate_bundle
from elpis.substrate.authority import PinnedAuthority
from elpis.substrate.digests import raw_digest

from .core import DurabilityStatus, DurableTicket, K1Recovery, RuntimeCore, RuntimeLibrary, describe
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
    """The runtime's continuity directory, its RuntimeCore library and the deployment authorities it trusts.

    ``runtime_library`` is the built ``libelpis_runtime.so`` (native/runtime), named by an explicit absolute path.
    It is never loaded from that path: it is admitted (``elpis.substrate.native_admission``) beneath its directory
    as the trusted descriptor root, as the library ``runtime_library_id`` of the deployment-pinned
    ``native_authority`` (a ``PinnedAuthority`` whose own pin came from trusted configuration), verified and loaded
    from sealed bytes. Without a ``native_authority`` the runtime refuses to start (``RUNTIME_UNPINNED``). The
    library embeds the continuity authority (native/continuity).

    ``codec_authority_sha256`` is the independent SHA-256 pin of the one codec authority catalog
    (``elpis.runtime.codec_authority``) whose admissions this runtime's cognitive operations accept. It is
    deployment configuration, never derived from a codec. Without it every managed cognitive operation that is
    given a codec refuses (``CODEC_AUTHORITY``).

    ``evolution_policy_sha256`` is the independent SHA-256 pin of the one evolution policy
    (``elpis.evolution.policy``) whose gates ``Runtime.evolve`` accepts. Without it ``evolve`` refuses
    (``EVOLUTION_POLICY_UNPINNED``): no caller configures the scopes, budgets or contract of an attempt.

    ``k1_checkpoint_dir`` (optional) is an operator-provisioned K1 Recovery R0 store
    (:func:`elpis.runtime.recovery.provision_k1_checkpoint`). When set, every open attaches it (a missing or malformed
    store refuses the open), the anchor and every durable publication persist the complete envelope before continuity
    names it, and :meth:`Runtime.recover_k1` can resume the last acknowledged state after restart. A LEARN never
    touches it.
    """

    continuity_dir: Path
    runtime_library: Path
    codec_authority_sha256: str | None = None
    native_authority: PinnedAuthority | None = None
    runtime_library_id: str = "elpis_runtime"
    k1_checkpoint_dir: Path | None = None
    evolution_policy_sha256: str | None = None

    def __post_init__(self):
        paths = (self.continuity_dir, self.runtime_library)
        if self.k1_checkpoint_dir is not None:
            paths += (self.k1_checkpoint_dir,)
        for value in paths:
            if not isinstance(value, Path) or not value.is_absolute():
                raise CompositionError("RUNTIME_PATH", "runtime paths must be absolute Paths")
        for pin, code in ((self.codec_authority_sha256, "CODEC_AUTHORITY"),
                          (self.evolution_policy_sha256, "EVOLUTION_POLICY")):
            if pin is not None and (type(pin) is not str or len(pin) != 64
                                    or any(c not in "0123456789abcdef" for c in pin)):
                raise CompositionError(code, "authority pin: 64 lowercase hex characters")


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


def _consumed_context(context):
    """The coordinate of the published context view a cognitive operation consumes: one reference read of a
    :class:`~elpis.runtime.context.ContextLane` (it never waits for the lane's WARM worker) or a pinned
    :class:`~elpis.runtime.context.ContextView`; ``None`` without one."""
    if context is None:
        return None
    from .context import ContextLane, ContextView
    if type(context) is ContextLane:
        return context.view().coordinate
    if type(context) is ContextView:
        return context.coordinate
    raise TypeError("context is a ContextLane or a ContextView")


def _observation_budget(budget, observation_authority):
    """The native budget of a managed turn that records observations: its ``max_observations`` narrowed to the
    observation authority's bound, so RuntimeCore itself enforces what the authority admitted (ABI v5)."""
    if observation_authority is None:
        return budget
    from dataclasses import replace
    return replace(budget, max_observations=min(budget.max_observations, observation_authority.max_observations))


class Runtime:
    def __init__(self, config: RuntimeConfig):
        if type(config) is not RuntimeConfig:
            raise TypeError("config must be a RuntimeConfig")
        self.config = config
        if config.native_authority is None:
            raise CompositionError("RUNTIME_UNPINNED", "a deployment-pinned native authority is required")
        library = RuntimeLibrary.admit(config.runtime_library.parent, config.runtime_library.name,
                                       config.native_authority, config.runtime_library_id)
        self._core = RuntimeCore(library, config.continuity_dir)
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
        if self.config.k1_checkpoint_dir is not None:
            try:
                self._core.checkpoint_attach(self.config.k1_checkpoint_dir)
            except CompositionError:
                self._core.close()   # a configured store that cannot be attached refuses the open
                raise
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

        The gate must be issued by the evolution policy authority whose policy is the configured pin, and still carry
        exactly that policy's scopes, budgets and contract; anything else is refused before reservation
        (``EVOLUTION_POLICY_UNPINNED`` / ``EVOLUTION_POLICY_UNAUTHORIZED``). ``advance`` is a trusted operator
        callback (the policy's confinement is ``TRUSTED_OPERATOR_CALLBACKS``); Elpis does not sandbox it.

        RuntimeCore refuses while fail-stopped or pending, durably reserves the exact assertion
        before anything executes and fail-stops on any reservation or finalization failure. An
        attempt that raises or returns an invalid result may already have had an external effect:
        RuntimeCore keeps its reservation pending and fail-stops; nothing is retried or inferred.
        """
        if type(gate) is not EvolutionPathGate:
            raise TypeError("evolve takes an EvolutionPathGate")
        self._require_policy_gate(gate)
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

    def _require_policy_gate(self, gate: EvolutionPathGate) -> None:
        pin = self.config.evolution_policy_sha256
        if pin is None:
            raise CompositionError("EVOLUTION_POLICY_UNPINNED", "no evolution policy is configured")
        authority = policy_of(gate)
        if authority is None or authority.policy.digest != pin:
            raise CompositionError("EVOLUTION_POLICY_UNAUTHORIZED",
                                   "the gate was not issued by the configured evolution policy")
        try:
            authority.verify_gate(gate)
        except EvolutionPolicyError as exc:
            raise CompositionError(exc.code, str(exc)) from exc

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

    # -- cognition: K1 Recovery R0 -------------------------------------------------------------------------------
    def recover_k1(self) -> K1Recovery:
        """What restart may do with the K1 lineage (read-only; requires an attached checkpoint store).

        Continuity decides; the checkpoint slots are evidence. ``RESUMABLE`` carries the complete authorized
        ``(W, epoch, H, a)`` envelope: restore it (``K1State.restore``) and the next QUERY or LEARN binds it after
        verifying its identity against continuity. Nothing rolls forward or back implicitly: a newer unauthorized
        candidate is ``CANDIDATE_UNRESOLVED`` until an operator discards or adopts it.
        """
        return self._core.checkpoint_recover()

    def discard_k1_candidate(self, candidate: bytes) -> None:
        """Operator reconciliation: withdraw exactly the unresolved candidate ``candidate``; the lineage stays at
        continuity's authority (``CHECKPOINT_INVALID`` for any other identity)."""
        self._core.checkpoint_discard(candidate)

    def import_composite(self, substrate) -> ContinuitySnapshot:
        """Operator, explicit and LEGACY: move the open K1 lineage to the ECS composite ``substrate``, made by
        ``CompositeState.import_k1`` from exactly the authorized K1 state with an empty memory. Needs an attached
        checkpoint store provisioned for the composite envelope: the composite is captured and made durable first,
        then continuity publishes the import (a v3 record of format ``COMPOSITE``); the composite is then bound.
        Anything else (another K1 state, a non-empty memory) is ``CONTINUITY_STATE_MISMATCH`` and publishes
        nothing. A lineage never changes format implicitly."""
        if type(substrate) is not CompositeState:
            raise CompositionError("ECS_STATE", "the import target is an ECS CompositeState")
        descriptor, owner = describe(substrate)
        snapshot = self._core.import_composite(descriptor)
        self._bound_owner, self._bound_substrate = owner, substrate
        return snapshot

    def adopt_k1_candidate(self, candidate: bytes) -> ContinuitySnapshot:
        """Operator reconciliation: adopt exactly the unresolved candidate ``candidate`` after its complete envelope
        verified: continuity publishes authorized -> candidate. Refused while a K1 state is bound."""
        return self._core.checkpoint_adopt(candidate)

    def _codec_pin(self) -> str:
        """The deployment pin every admitted codec's authority must match; ``""`` (matches nothing) when none is
        configured, so admission refuses (``CODEC_AUTHORITY``) after the fail-closed missing-codec check."""
        return self.config.codec_authority_sha256 or ""

    # -- cognition: QUERY (read-only) ---------------------------------------------------------------------------
    def run_query(self, substrate, request, *, context=None):
        """One managed QUERY (docs/COGNITION_R0.md): read-only.

        The boundary validates the :class:`~elpis.runtime.cognition.QueryRequest` and encodes the query rows.
        RuntimeCore answers ``f_W(x)`` from the lineage's LIVE K1 state in one native call that also reports the
        state's generation; it must be the live one (otherwise the lineage is lost: fail-stop, the answer withheld).
        Nothing is begun, committed, published, hashed or written: W, epoch, H, a and continuity are unchanged. The
        answer reflects every managed LEARN before it, durable or not. ``context``: see the module docstring.
        """
        self._core.require_live()
        consumed = _consumed_context(context)

        from .cognition import QueryReadout, QueryResult, _decode_query, _encode_query, _validate_query_request
        from .fuel import admit_query

        codec_map, classification = _validate_query_request(substrate, request, self._codec_pin(), managed=True)
        descriptor, owner = describe(substrate)
        tokens, stimulus = _encode_query(substrate, request.text, request.tokenizer, codec_map)
        admit_query(stimulus, substrate.dim, substrate.width, request.budget, request.max_output_tokens)
        if self._bound_owner is None:
            self._bound_owner, self._bound_substrate = owner, substrate   # what this query binds stays alive
        values, generation = self._core.query(descriptor, stimulus, request.budget)   # RuntimeCore re-admits natively
        readout = QueryReadout(values, substrate.dim, substrate.width, None, generation)
        output, rendered = _decode_query(codec_map, readout, request.tokenizer, request.max_output_tokens)
        return QueryResult(tokens, output, rendered, readout, classification, context=consumed)

    # -- cognition: LEARN (explicit authority) -------------------------------------------------------------------
    def run_learn(self, substrate, request, *, context=None, observations=(), observation_authority=None):
        """One managed LEARN under an explicit :class:`~elpis.runtime.cognition.LearnAuthority`.

        RuntimeCore verifies the lineage (its live coordinate), runs the admitted experience schedule on a native
        candidate and commits it atomically into the live state. The commit is VOLATILE: nothing is checkpointed or
        published and no I/O happens (``result.durability == "VOLATILE"``; the result carries the live generations).
        Durability is :meth:`publish_recoverable` (or :meth:`publish_identity`). A refused LEARN installs nothing.
        ``context``: see the module docstring.

        ``observations`` (an ECS composite lineage only): typed
        :class:`~elpis.ECS.composite.MemoryObservation` values recorded into the memory candidate of the same turn,
        under an explicit :class:`~elpis.runtime.cognition.ObservationAuthority` (admitted before anything native is
        called). OBSERVE is exact evidence storage: it never changes the K1 schedule and is not learning or truth. The
        turn is all or nothing: if any observation is refused the whole turn is aborted (nothing of it is installed)
        and the refusal is raised; otherwise the K1 candidate and the memory candidate commit as one transition. No
        memory LEARN is reachable (unqualified). ``result.observed`` holds the committed relation keys in order.
        """
        self._core.require_live()
        consumed = _consumed_context(context)

        from .cognition import LearnResult, _admit_observations, _encode, _validate_learn_request
        from .fuel import admit_learn

        codec_map, classification, authority = _validate_learn_request(substrate, request, self._codec_pin(),
                                                                       managed=True)
        observed = _admit_observations(substrate, observations, observation_authority)
        descriptor, owner = describe(substrate)
        tokens, stimulus = _encode(substrate, request.text, request.tokenizer, codec_map)
        admit_learn(stimulus, substrate.dim, substrate.width, authority.budget)
        if self._bound_owner is None:
            self._bound_owner, self._bound_substrate = owner, substrate   # what this begin binds stays alive
        committed, keys = self._managed_learn(descriptor, stimulus, authority, observed, observation_authority)
        return LearnResult(tokens, stimulus.experiences, committed.epoch_before, committed.epoch_after,
                           classification, authority.grant, None, None,
                           generation_before=committed.generation_before, generation_after=committed.generation_after,
                           durability="VOLATILE", context=consumed, observed=tuple(keys))

    def _managed_learn(self, descriptor, stimulus, authority, observed, observation_authority):
        """Begin, run the admitted schedule, OBSERVE every admitted observation, commit: one volatile turn."""
        budget = _observation_budget(authority.budget, observation_authority if observed else None)
        self._core.turn_begin(descriptor, stimulus, authority.learning_rate, budget)
        keys = self._observe_all(descriptor, observed)
        return self._core.turn_commit(descriptor), keys

    # -- cognition: NUMERIC QUERY and LEARN (admitted numeric stimuli; no codec, no text) ---------------------
    def run_numeric_query(self, substrate, stimulus, *, budget=None, context=None):
        """One managed QUERY of an admitted numeric :class:`~elpis.runtime.cognition.QueryStimulus` (rows of the
        lineage's dimension): ``f_W(x)`` from the LIVE K1 state in one native ``leased_query`` that also reports its
        generation. Read-only exactly as :meth:`run_query`; it differs only in its edge: numbers in, numbers out. No
        codec and no text are involved, so nothing here is a language operation and the text turn still fails closed
        without a qualified codec (docs/ELPIS_MISSION.md)."""
        self._core.require_live()
        consumed = _consumed_context(context)
        from .cognition import NumericAnswer, QueryStimulus, _MANAGED_SUBSTRATES
        from .fuel import CEILING, CognitiveBudget, admit_query

        budget = CEILING if budget is None else budget
        if type(substrate) not in _MANAGED_SUBSTRATES:
            raise CompositionError("ECS_STATE", "a native K1 state or an ECS CompositeState is required")
        if type(stimulus) is not QueryStimulus or stimulus.dim != substrate.dim:
            raise CompositionError("STIMULUS", "a QueryStimulus of the lineage's dimension is required")
        if type(budget) is not CognitiveBudget:
            raise CompositionError("STIMULUS", "budget must be a CognitiveBudget")
        admit_query(stimulus, substrate.dim, substrate.width, budget, 0)
        descriptor, owner = describe(substrate)
        if self._bound_owner is None:
            self._bound_owner, self._bound_substrate = owner, substrate   # what this query binds stays alive
        values, generation = self._core.query(descriptor, stimulus, budget)   # RuntimeCore re-admits natively
        return NumericAnswer(tuple(values), stimulus.rows, substrate.dim, substrate.width, generation, consumed)

    def run_numeric_learn(self, substrate, stimulus, authority, *, observations=(), observation_authority=None,
                          context=None):
        """One managed LEARN of an admitted numeric :class:`~elpis.runtime.cognition.Stimulus` (an experience
        schedule of rows and targets) under an explicit :class:`~elpis.runtime.cognition.LearnAuthority`: the
        qualified K1 law runs natively on a candidate, committed atomically and VOLATILE, exactly as
        :meth:`run_learn`; typed ``observations`` ride the same turn under an explicit ``observation_authority`` (all
        or nothing). No codec and no text are involved."""
        self._core.require_live()
        consumed = _consumed_context(context)
        from .cognition import (LearnAuthority, NumericLearnResult, Stimulus, _MANAGED_SUBSTRATES,
                                _admit_observations)
        from .fuel import admit_learn

        if type(substrate) not in _MANAGED_SUBSTRATES:
            raise CompositionError("ECS_STATE", "a native K1 state or an ECS CompositeState is required")
        if authority is None:
            raise CompositionError("LEARN_UNAUTHORIZED", "LEARN requires an explicit LearnAuthority")
        if type(authority) is not LearnAuthority:
            raise CompositionError("LEARN_AUTHORITY", "authority must be a LearnAuthority")
        if type(stimulus) is not Stimulus or stimulus.dim != substrate.dim:
            raise CompositionError("STIMULUS", "a Stimulus of the lineage's dimension is required")
        observed = _admit_observations(substrate, observations, observation_authority)
        admit_learn(stimulus, substrate.dim, substrate.width, authority.budget)
        descriptor, owner = describe(substrate)
        if self._bound_owner is None:
            self._bound_owner, self._bound_substrate = owner, substrate   # what this begin binds stays alive
        committed, keys = self._managed_learn(descriptor, stimulus, authority, observed, observation_authority)
        return NumericLearnResult(stimulus.experiences, stimulus.rows, stimulus.steps, committed.epoch_before,
                                  committed.epoch_after, committed.generation_before, committed.generation_after,
                                  authority.grant, tuple(keys), context=consumed)

    # -- cognition: OBSERVE-ONLY (explicit authority; no K1 learning) -------------------------------------------
    def run_observe(self, substrate, observations, observation_authority, *, budget=None):
        """Record typed evidence into the memory of an ECS composite lineage WITHOUT any K1 learning (RuntimeCore
        ABI v5, the observe-only managed turn).

        ``observations``: a non-empty tuple of :class:`~elpis.ECS.composite.MemoryObservation` (for HACF evidence,
        :meth:`elpis.runtime.observation.HacfObservationLibrary.observation` maps a validated hyperedge and
        assertion to one), each from a source the explicit :class:`~elpis.runtime.cognition.ObservationAuthority`
        admits, within its bound; all of it is admitted before anything native is called. RuntimeCore verifies and
        binds the lineage, opens the composite transaction at the live generation and runs NO K1 schedule; the
        observations are recorded on the memory candidate and committed as one VOLATILE transition: the live
        generation advances by one, the K1 retained state stays byte-identical, nothing is published (durability is
        :meth:`publish_recoverable`). All or nothing: any refused observation aborts the whole turn (nothing is
        installed) and the refusal is raised. ``budget`` (a :class:`~elpis.runtime.fuel.CognitiveBudget`, default
        the ceiling) is admitted natively; its ``max_observations`` is narrowed to the authority's bound.
        Evidence is storage: an observation is not learning, truth or authority.
        """
        self._core.require_live()
        from .cognition import ObserveResult, _admit_observations
        from .fuel import CEILING, CognitiveBudget

        if budget is None:
            budget = CEILING
        if type(budget) is not CognitiveBudget:
            raise CompositionError("OBSERVATION", "budget must be a CognitiveBudget")
        if type(observations) is not tuple or not observations:
            raise CompositionError("OBSERVATION", "observations: a non-empty tuple of MemoryObservation")
        observed = _admit_observations(substrate, observations, observation_authority)
        descriptor, owner = describe(substrate)
        if self._bound_owner is None:
            self._bound_owner, self._bound_substrate = owner, substrate   # what this begin binds stays alive
        generation = self._core.turn_begin_observe(descriptor, _observation_budget(budget, observation_authority))
        keys = self._observe_all(descriptor, observed)
        committed = self._core.turn_commit(descriptor)
        if committed.generation_before != generation:
            raise CompositionError("RUNTIME_INVALID", "the observe-only commit does not continue the turn's generation")
        return ObserveResult(tuple(keys), observation_authority.grant, committed.generation_before,
                             committed.generation_after, committed.steps)

    def _observe_all(self, descriptor, observed) -> list:
        """OBSERVE every admitted observation on the open turn; all or nothing (any refusal aborts the turn)."""
        keys = []
        try:
            for observation in observed:
                keys.append(self._core.turn_observe(descriptor, observation))
        except BaseException:
            try:
                self._core.turn_abort(descriptor)   # all or nothing: no partial candidate is ever committed
            except CompositionError:
                pass   # a fail-stop already ended it natively, or close and destruction will (the turn law)
            raise
        return keys

    # -- durability: explicit, off the HOT lane -------------------------------------------------------------------
    def publish_recoverable(self, substrate, *, wait: bool = True, timeout: float | None = None) -> DurableTicket:
        """Request RECOVERABLE durable publication of the live state of ``substrate`` (binding it first if needed).

        The complete state becomes durable, so a checkpoint store must be attached (``k1_checkpoint_dir``): without
        one this is refused with ``RUNTIME_RECOVERY_UNAVAILABLE`` before anything is bound or captured; it is never
        silently downgraded to an identity-only publication. RuntimeCore captures the state's envelope under its
        lease (the one synchronous step) and hands it to its durability lane, which seals it, writes and syncs the
        checkpoint slot, reads it back and verifies it is exactly the captured envelope, then writes and syncs the
        continuity record, off the HOT lane. With ``wait`` (the caller's explicit choice) this returns the resolved
        :class:`~elpis.runtime.core.DurableTicket`; otherwise the PENDING one at once. ``ACKNOWLEDGED`` (always
        ``recoverable``) means restart resumes exactly this state (K1 Recovery R0). ``UNCERTAIN`` means the outcome is
        unknown until restart resolves it; it never means the state is not durable. A newer request supersedes a
        waiting one. Refused while the lane is faulted.
        """
        return self._publish(substrate, self._core.recoverable_request, wait, timeout)

    def publish_identity(self, substrate, *, wait: bool = True, timeout: float | None = None) -> DurableTicket:
        """Request IDENTITY-ONLY durable publication: continuity names the live state's identity; no store holds its
        bytes (the caller keeps them, and restart can rebind only a state the caller supplies).

        For runtimes without a checkpoint store; refused with ``RUNTIME_CHECKPOINT_ATTACHED`` while one is attached
        (every publication must then be recoverable). Resolves ``IDENTITY_PUBLISHED`` (never ``recoverable``), never
        ``ACKNOWLEDGED``: it is not a recoverable-state acknowledgment. Otherwise as :meth:`publish_recoverable`.
        """
        return self._publish(substrate, self._core.identity_request, wait, timeout)

    def _publish(self, substrate, request, wait: bool, timeout: float | None) -> DurableTicket:
        self._core.require_live()
        descriptor, owner = describe(substrate)
        if self._bound_owner is None:
            self._bound_owner, self._bound_substrate = owner, substrate   # what this request binds stays alive
        ticket = request(descriptor)
        if not wait:
            return ticket
        return self._core.durable_wait(ticket.sequence, timeout)

    def durability(self) -> DurabilityStatus:
        """The durability lane's state (read without waiting for any HOT operation)."""
        return self._core.durable_status()

    def live_identity(self, substrate) -> tuple[bytes, int]:
        """The bound state's retained-state identity and live generation, on request (cold path: it hashes)."""
        descriptor, owner = describe(substrate)
        if self._bound_owner is None:
            self._bound_owner, self._bound_substrate = owner, substrate
        return self._core.live_identity(descriptor)

    # -- cognition: LEGACY learned turn (LEARN + S3 decode) --------------------------------------------------------
    def run_turn(self, substrate, text, *, tokenizer, codec=None, authority=None, max_output_tokens=256,
                 context=None):
        """One managed LEGACY learned turn: a LEARN under explicit authority whose S3 readout is decoded.

        Kept for the replay of the original synthetic scaffold (``LEGACY_LEARNED_TURN``); it is never a query. The
        boundary validates the request, encodes the text and the stimulus and decodes the readout. RuntimeCore
        verifies the K1 lineage, runs the native K1 transaction on its candidate and commits it (volatile) after the
        decode succeeded. A refused or aborted turn installs nothing.
        """
        self._core.require_live()
        consumed = _consumed_context(context)

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
        committed = self._core.turn_commit(descriptor)
        return TurnResult(tokens, output, rendered, readout, committed.epoch_before, committed.epoch_after,
                          classification, generation_before=committed.generation_before,
                          generation_after=committed.generation_after, durability="VOLATILE", context=consumed)
