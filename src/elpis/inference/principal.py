"""Principal sequences: HACF-admitted context in, tokens out, no KV memory.

This is the principal inference path. Context lives outside the model, in
HACF. A turn is::

    admission = <runtime resolves and admits HACF objects>      # before begin
    sequence  = engine.begin(state, request, admission, expected_state=state.digest)
    for token in sequence:                                       # model arithmetic only
        emit(token)
    result    = engine.finalize(sequence)                        # commit boundary

**Committed state** (:class:`PrincipalState`) is bounded: model, numerical
profile, context snapshot, turn number and the previous commit's digest. It
holds no keys or values, no compressed pool, no global index and no token
history. It does not grow with the amount of context the system stores or has
ever attended to.

**Sequence state** is the driver's :class:`~elpis.inference.target.WindowState`:
at most ``local_window`` transient keys and values, the n-gram address tail,
the hidden vector and logits, plus the per-sequence trace and outputs, which
are bounded by the sequence's own token budget. It is dropped at
finalization. Nothing about it is durable.

``begin`` validates the committed state, the request and the admission once,
admits expert bytes, and runs the prefill: the admitted context tokens, then
the prompt, as ordinary model input. Each :meth:`PrincipalSequence.next` is
one model step. There is no latent input: G/X channels do not exist on this
path, and nothing (no retrieval, no world model, no steering) can reach the
model until the sequence is finalized and a new one begins.

The one admitted turn-level input is optional :class:`TurnConditioning`: a
bounded vector handed to ``begin`` and frozen for the whole sequence (the
target preprojects it once in ``window_initial``). Its digest is bound into
the commit. ``finalize`` returns a :class:`TurnObservation` of a committed
sequence's final distribution for whoever drives the next turn. Without
conditioning, commits and identities are exactly those of the unconditioned
path.

``finalize`` builds the :class:`PrincipalCommit` (admission, request,
outputs, stop reason and a digest over the per-step trace) and the next
committed state. :meth:`PrincipalEngine.replay` re-runs a commit from its
inputs and requires an identical commit.

What this path is not: it is not numerically equivalent to the legacy
transaction (:mod:`elpis.inference.transaction`), whose state carries a
growing compressed-KV pool with sparse global attention. That path, the
legacy-identical sequence path (:mod:`elpis.inference.sequence`) and their
persisted identities remain for replaying historical records.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .admission import ContextAdmission
from .conditioning import TurnConditioning, TurnObservation
from .contracts import Code, ContractError, digest_value, identity, integer, require
from .transaction import typed_failure

__all__ = ("PrincipalCommit", "PrincipalEngine", "PrincipalRequest", "PrincipalResult", "PrincipalSequence",
           "PrincipalState")


@dataclass(frozen=True)
class PrincipalState:
    """Committed principal state. Bounded; carries no model activations or K/V."""
    model: str
    numerical_profile: str
    context_snapshot: str
    turn: int
    predecessor: str | None

    def __post_init__(self):
        for value in (self.model, self.numerical_profile, self.context_snapshot):
            digest_value(value)
        integer(self.turn)
        if self.predecessor is not None:
            digest_value(self.predecessor)

    @property
    def digest(self):
        return identity("principal-state", self)


@dataclass(frozen=True)
class PrincipalRequest:
    request_id: str
    prompt: tuple[int, ...]
    max_new_tokens: int
    stop_tokens: tuple[int, ...] = ()

    def __post_init__(self):
        require(type(self.request_id) is str and bool(self.request_id), detail="request id")
        require(type(self.prompt) is tuple and type(self.stop_tokens) is tuple, detail="request tokens")
        for token in self.prompt + self.stop_tokens:
            integer(token)
        integer(self.max_new_tokens)

    @property
    def digest(self):
        return identity("principal-request", self)


@dataclass(frozen=True)
class PrincipalCommit:
    state: str            # digest of the committed state the sequence began from
    admission: str        # ContextAdmission digest
    request: str          # PrincipalRequest digest
    model: str
    numerical_profile: str
    prefill: int          # admitted context tokens + prompt tokens consumed
    outputs: tuple[int, ...]
    stop_reason: str
    trace: str            # digest over (token, address rows, expert route) of every step
    working_set: tuple[int, int]  # (local_window, dimension): the declared attention-scratch bound
    conditioning: str | None = None  # TurnConditioning digest, when the turn was conditioned

    def __post_init__(self):
        if self.conditioning is not None:
            digest_value(self.conditioning)

    @property
    def digest(self):
        if self.conditioning is None:
            # Unconditioned commits keep exactly their pre-conditioning identity.
            fields = {k: v for k, v in self.__dict__.items() if k != "conditioning"}
            return identity("principal-commit", fields)
        return identity("principal-commit", self)


@dataclass(frozen=True)
class PrincipalResult:
    state: PrincipalState
    commit: PrincipalCommit | None
    failure: str | None = None
    observation: TurnObservation | None = None  # only for a committed sequence


def observe_distribution(logits):
    """Token-identity-free summary of a next-token distribution (TurnObservation features).

    float64 softmax of the final logits; the six largest probabilities in
    descending order, the remaining tail mass, and entropy normalized by
    ln(vocab). Values only, never indices. ``-inf`` (masked) logits are
    allowed; NaN, ``+inf`` or an all-masked distribution are refused.
    """
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    require(values.size >= 2 and not bool(np.any(np.isnan(values) | (values == np.inf))) and
            bool(np.isfinite(np.max(values))), Code.ENCODING, "observation logits")
    p = np.exp(values - np.max(values))
    p /= np.sum(p)
    top = np.sort(p)[::-1][:6]
    top = np.concatenate((top, np.zeros(6 - top.size)))
    positive = p[p > 0]
    entropy = float(-np.sum(positive * np.log(positive)) / np.log(values.size))
    tail = max(0.0, 1.0 - float(np.sum(top)))
    return tuple(float(v) for v in top) + (tail, entropy)


class PrincipalSequence:
    """One active principal sequence. Ephemeral; not authority; no digest."""

    __slots__ = ("_engine", "_state", "_request", "_admission", "_experts", "_work", "_steps", "_outputs",
                 "_prefill", "_done", "_stop_reason", "_failure", "_conditioning", "_final_logits")

    def __init__(self, engine, state, request, admission, conditioning=None):
        self._engine, self._state, self._request, self._admission = engine, state, request, admission
        self._conditioning, self._final_logits = conditioning, None
        self._experts = self._work = None
        self._steps, self._outputs = [], []
        self._prefill = 0
        self._done, self._stop_reason, self._failure = False, None, None

    @property
    def done(self) -> bool:
        return self._done

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    @property
    def outputs(self) -> tuple[int, ...]:
        return tuple(self._outputs)

    @property
    def working_state(self):
        """The current bounded working state (read-only view, for inspection)."""
        return self._work

    def _fail(self, exc):
        """Typed failure: the sequence can never advance, so its working state is released now."""
        self._failure, self._done, self._stop_reason = exc.code.value, True, "FAILED"
        self._engine._release_work(self)

    def _abort(self):
        """Unexpected exception: mark failed and release before the caller sees the raise."""
        if self._failure is None:
            self._failure = Code.INVALID.value
        self._done, self._stop_reason = True, "FAILED"
        self._engine._release_work(self)

    def _advance(self, token):
        self._work = self._engine.target.window_step(self._work, token, experts=self._experts)
        self._steps.append(self._work.step)

    def next(self) -> int | None:
        """Generate one token and return it, or None once the sequence has ended."""
        if self._done:
            return None
        if len(self._outputs) == self._request.max_new_tokens:
            self._done, self._stop_reason = True, "COMPLETE"
            return None
        token = int(np.argmax(self._work.logits))
        try:
            self._advance(token)
        except ContractError as exc:
            self._fail(exc)
            return None
        except BaseException as exc:
            self._abort()
            raise typed_failure(exc) from exc
        self._outputs.append(token)
        if token in self._request.stop_tokens:
            self._done, self._stop_reason = True, "STOP_TOKEN"
        elif len(self._outputs) == self._request.max_new_tokens:
            self._done, self._stop_reason = True, "COMPLETE"
        return token

    def __iter__(self):
        while (token := self.next()) is not None:
            yield token

    def stop(self, reason: str = "YIELD") -> None:
        require(type(reason) is str and bool(reason), detail="stop reason")
        if not self._done:
            self._done, self._stop_reason = True, reason

    def close(self) -> None:
        """Explicit bounded cleanup for a sequence that will not be finalized.

        Idempotent. Ends the sequence (stop reason CLOSED if still active) and
        releases sequence-local driver state; finalize() remains possible.
        Cleanup never depends on garbage collection.
        """
        if not self._done:
            self._done, self._stop_reason = True, "CLOSED"
        self._engine._release_work(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class PrincipalEngine:
    def __init__(self, target):
        self.target = target

    def initial(self, context_snapshot: str) -> PrincipalState:
        digest_value(context_snapshot)
        return PrincipalState(self.target.model_identity, self.target.numerical_profile, context_snapshot, 0, None)

    def begin(self, state, request, admission, *, expected_state, experts=None,
              resident_experts=None, conditioning=None) -> PrincipalSequence:
        """Validate once, admit experts, run the prefill; return the streaming sequence.

        ``conditioning`` (optional :class:`TurnConditioning`) is frozen for the
        whole sequence; the target consumes it once, before the prefill.
        """
        sequence = PrincipalSequence(self, state, request, admission, conditioning)
        target, config = self.target, self.target.config
        try:
            require(type(state) is PrincipalState and type(request) is PrincipalRequest and
                    type(admission) is ContextAdmission, detail="principal inputs")
            require(conditioning is None or type(conditioning) is TurnConditioning, detail="turn conditioning")
            require(conditioning is None or getattr(target, "accepts_conditioning", False), Code.UNSUPPORTED,
                    "target does not admit turn conditioning")
            require(state.digest == expected_state, Code.STALE, "committed principal state")
            require(state.model == target.model_identity, Code.IDENTITY, "principal model")
            require(state.numerical_profile == target.numerical_profile, Code.UNSUPPORTED, "numerical profile")
            require(admission.model == target.model_identity and admission.tokenizer == config.tokenizer,
                    Code.IDENTITY, "admission model/tokenizer")
            require(admission.context_snapshot == state.context_snapshot, Code.STALE, "admission snapshot")
            prefill = admission.tokens + request.prompt
            require(bool(prefill), detail="a principal sequence needs input tokens")
            require(all(t < config.vocab for t in prefill + request.stop_tokens), detail="token vocabulary")
            require(len(prefill) + request.max_new_tokens <= config.max_tokens, Code.LIMIT,
                    "sequence token budget exceeds target capacity")
            sequence._experts = target.admit_stream(resident_experts=resident_experts) if experts is None else experts
            sequence._work = (target.window_initial() if conditioning is None else
                              target.window_initial(conditioning=conditioning))
            for token in prefill:
                sequence._advance(token)
            sequence._prefill = len(prefill)
        except ContractError as exc:
            sequence._fail(exc)
        except BaseException as exc:
            sequence._abort()
            raise typed_failure(exc) from exc
        return sequence

    def finalize(self, sequence: PrincipalSequence) -> PrincipalResult:
        """Commit boundary: trace digest, commit record, next committed state."""
        require(type(sequence) is PrincipalSequence and sequence._engine is self, Code.IDENTITY, "sequence engine")
        require(sequence.done, Code.INVALID, "sequence still active; stop() it first")
        state = sequence._state
        try:
            if sequence._failure is not None:
                return PrincipalResult(state, None, sequence._failure)
            trace = identity("principal-trace", tuple((s.token, s.rows, s.route) for s in sequence._steps))
            conditioning = sequence._conditioning
            commit = PrincipalCommit(state.digest, sequence._admission.digest, sequence._request.digest,
                                     self.target.model_identity, self.target.numerical_profile, sequence._prefill,
                                     tuple(sequence._outputs), sequence._stop_reason, trace,
                                     getattr(self.target, "principal_working_set",
                                             (self.target.config.local_window, self.target.config.dimension)),
                                     None if conditioning is None else conditioning.digest)
            logits = (sequence._final_logits if sequence._final_logits is not None else
                      getattr(sequence._work, "logits", None))
            observation = None
            if logits is not None:
                try:  # an unobservable distribution never blocks the commit itself
                    observation = TurnObservation(commit.digest, observe_distribution(logits),
                                                  len(sequence._outputs), sequence._stop_reason)
                except ContractError:
                    observation = None
            return PrincipalResult(PrincipalState(state.model, state.numerical_profile, state.context_snapshot,
                                                  state.turn + 1, commit.digest), commit, None, observation)
        finally:
            self._release_work(sequence)

    def _release_work(self, sequence):
        """Optional driver lifecycle hook; legacy fixture behavior/identities stay unchanged.

        Idempotent: the work handle is dropped after the first release.
        """
        work = sequence._work
        if work is not None and sequence._failure is None and sequence._final_logits is None:
            # Keep the final distribution for the commit-boundary observation (once per sequence).
            logits = getattr(work, "logits", None)
            sequence._final_logits = None if logits is None else np.array(logits, copy=True)
        release = getattr(self.target, "release_window", None)
        if release is not None and work is not None:
            sequence._work = None
            release(work)

    def replay(self, state, request, admission, commit: PrincipalCommit, *, conditioning=None) -> PrincipalResult:
        """Re-run a commit from its inputs (including its turn conditioning); the result must be identical."""
        require(type(commit) is PrincipalCommit, detail="commit")
        sequence = self.begin(state, request, admission, expected_state=state.digest, conditioning=conditioning)
        while not sequence.done:
            if commit.stop_reason == "YIELD" and len(sequence._outputs) == len(commit.outputs):
                sequence.stop("YIELD")
                break
            sequence.next()
        result = self.finalize(sequence)
        require(result.commit == commit, Code.IDENTITY, "principal replay mismatch")
        return result
