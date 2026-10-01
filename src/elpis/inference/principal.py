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
path, and nothing (no retrieval, no ECS, no steering) can reach the model
until the sequence is finalized and a new one begins.

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

    @property
    def digest(self):
        return identity("principal-commit", self)


@dataclass(frozen=True)
class PrincipalResult:
    state: PrincipalState
    commit: PrincipalCommit | None
    failure: str | None = None


class PrincipalSequence:
    """One active principal sequence. Ephemeral; not authority; no digest."""

    __slots__ = ("_engine", "_state", "_request", "_admission", "_experts", "_work", "_steps", "_outputs",
                 "_prefill", "_done", "_stop_reason", "_failure")

    def __init__(self, engine, state, request, admission):
        self._engine, self._state, self._request, self._admission = engine, state, request, admission
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
        self._failure, self._done, self._stop_reason = exc.code.value, True, "FAILED"

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
        except Exception as exc:
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


class PrincipalEngine:
    def __init__(self, target):
        self.target = target

    def initial(self, context_snapshot: str) -> PrincipalState:
        digest_value(context_snapshot)
        return PrincipalState(self.target.model_identity, self.target.numerical_profile, context_snapshot, 0, None)

    def begin(self, state, request, admission, *, expected_state, experts=None,
              resident_experts=None) -> PrincipalSequence:
        """Validate once, admit experts, run the prefill; return the streaming sequence."""
        sequence = PrincipalSequence(self, state, request, admission)
        target, config = self.target, self.target.config
        try:
            require(type(state) is PrincipalState and type(request) is PrincipalRequest and
                    type(admission) is ContextAdmission, detail="principal inputs")
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
            sequence._work = target.window_initial()
            for token in prefill:
                sequence._advance(token)
            sequence._prefill = len(prefill)
        except ContractError as exc:
            sequence._fail(exc)
        except Exception as exc:
            raise typed_failure(exc) from exc
        return sequence

    def finalize(self, sequence: PrincipalSequence) -> PrincipalResult:
        """Commit boundary: trace digest, commit record, next committed state."""
        require(type(sequence) is PrincipalSequence and sequence._engine is self, Code.IDENTITY, "sequence engine")
        require(sequence.done, Code.INVALID, "sequence still active; stop() it first")
        state = sequence._state
        if sequence._failure is not None:
            self._release_work(sequence)
            return PrincipalResult(state, None, sequence._failure)
        trace = identity("principal-trace", tuple((s.token, s.rows, s.route) for s in sequence._steps))
        commit = PrincipalCommit(state.digest, sequence._admission.digest, sequence._request.digest,
                                 self.target.model_identity, self.target.numerical_profile, sequence._prefill,
                                 tuple(sequence._outputs), sequence._stop_reason, trace,
                                 getattr(self.target, "principal_working_set",
                                         (self.target.config.local_window, self.target.config.dimension)))
        self._release_work(sequence)
        return PrincipalResult(PrincipalState(state.model, state.numerical_profile, state.context_snapshot,
                                              state.turn + 1, commit.digest), commit)

    def _release_work(self, sequence):
        """Optional driver lifecycle hook; legacy fixture behavior/identities stay unchanged."""
        release = getattr(self.target, "release_window", None)
        if release is not None:
            release(sequence._work)
            sequence._work = None

    def replay(self, state, request, admission, commit: PrincipalCommit) -> PrincipalResult:
        """Re-run a commit from its inputs; the result must be identical."""
        require(type(commit) is PrincipalCommit, detail="commit")
        sequence = self.begin(state, request, admission, expected_state=state.digest)
        while not sequence.done:
            if commit.stop_reason == "YIELD" and len(sequence._outputs) == len(commit.outputs):
                sequence.stop("YIELD")
                break
            sequence.next()
        result = self.finalize(sequence)
        require(result.commit == commit, Code.IDENTITY, "principal replay mismatch")
        return result
