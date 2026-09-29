"""Sequence transactions: validate once, stream tokens, commit at the boundary.

::

    sequence = engine.begin(committed, request, expected_state=committed.digest)
    for token in sequence:            # each token is externally visible at once
        emit(token)
    result = engine.finalize(sequence) # provenance, receipts, commit

Three kinds of state are kept apart:

* **Committed state** is the immutable :class:`DecodeState` the sequence
  started from. It is never modified: a failure at any point before
  finalization leaves it exactly as it was.
* **Sequence state** is the driver's stream state plus bounded per-step
  records. It is ephemeral, non-authoritative and not durable. It has no
  digest, and no identity is computed while tokens stream.
* **Finalization** turns the records into the canonical representation. The
  result is exactly the :class:`DecodeResult` that the legacy
  :meth:`InferenceEngine.execute` would have produced for the effective
  request, so legacy replay verifies it and no persisted identity changes
  meaning.

Everything a sequence depends on is fixed at :func:`begin`: the committed
state, the request (tokens or count, latents, structural proposals) and the
expert admission. A sequence has no way to accept new latents or proposals
while it runs; a later request, after finalization, is the only place new
inputs enter.

The synchronous token loop does model arithmetic only. The substrate still
verifies any cold page it faults in, since integrity is not traded for
speed; resident pages and admitted experts are not re-hashed.

Stopping early (a stop token, or :meth:`Sequence.stop` for an explicit yield)
commits the tokens produced so far as the effective request: the same request
with ``count`` (GREEDY) or ``tokens`` (PREFILL) cut to what was produced.
"""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter_ns

import numpy as np

from .contracts import Code, ContractError, integer, require
from .prefetch import plan_prefetch
from .transaction import DecodeResult, DecodeState, InferenceRequest, typed_failure

__all__ = ("Sequence", "begin", "finalize")


class Sequence:
    """One active token sequence over a committed state. Not authority, not durable."""

    __slots__ = ("_engine", "_committed", "_request", "_admission", "_stop_tokens", "_state",
                 "_records", "_metrics", "_planned", "_failure", "_done", "_stop_reason")

    def __init__(self, engine, committed, request, admission, stop_tokens, state, failure):
        self._engine = engine
        self._committed = committed
        self._request = request
        self._admission = admission
        self._stop_tokens = stop_tokens
        self._state = state
        self._records = []
        self._metrics = []
        self._failure = failure
        self._done = failure is not None
        self._stop_reason = "FAILED" if failure is not None else None
        self._planned = 0
        if failure is None:
            self._planned = len(request.tokens) if request.mode == "PREFILL" else request.count
            if self._planned == 0:
                self._done, self._stop_reason = True, "COMPLETE"

    @property
    def committed(self) -> DecodeState:
        return self._committed

    @property
    def request(self) -> InferenceRequest:
        return self._request

    @property
    def done(self) -> bool:
        return self._done

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    @property
    def tokens(self) -> tuple[int, ...]:
        """Tokens produced so far in this sequence (not yet committed)."""
        return tuple(record.step.token for record in self._records)

    def next(self) -> int | None:
        """Advance one token and return it, or None once the sequence has ended."""
        if self._done:
            return None
        position = len(self._records)
        request = self._request
        token = request.tokens[position] if request.mode == "PREFILL" else int(np.argmax(self._state.logits))
        target = self._engine.target
        try:
            self._state = target.stream_step(self._state, token, admission=self._admission,
                                             latents=request.latents)
        except ContractError as exc:
            self._failure, self._done, self._stop_reason = exc.code.value, True, "FAILED"
            return None
        except Exception as exc:
            raise typed_failure(exc) from exc
        self._records.append(self._state.record())
        self._metrics.append(dict(target.last_metrics))
        if token in self._stop_tokens:
            self._done, self._stop_reason = True, "STOP_TOKEN"
        elif position + 1 == self._planned:
            self._done, self._stop_reason = True, "COMPLETE"
        return token

    def __iter__(self):
        while (token := self.next()) is not None:
            yield token

    def stop(self, reason: str = "YIELD") -> None:
        """End the sequence now; finalization commits what was produced."""
        require(type(reason) is str and bool(reason), detail="stop reason")
        if not self._done:
            self._done, self._stop_reason = True, reason

    def effective_request(self) -> InferenceRequest:
        """The request the committed result corresponds to (cut to what was produced)."""
        produced = len(self._records)
        if self._failure is not None or produced == self._planned:
            return self._request
        if self._request.mode == "PREFILL":
            return replace(self._request, tokens=self._request.tokens[:produced])
        return replace(self._request, count=produced)


def begin(engine, committed, request, *, expected_state, admission=None, resident_experts=None,
          stop_tokens=()) -> Sequence:
    """Validate the committed input once and open a sequence over it.

    Validation is the legacy transaction's: the same checks in the same order,
    so a refused input yields the same failure receipt at :func:`finalize`.
    """
    try:
        require(type(stop_tokens) is tuple, detail="stop tokens")
        for token in stop_tokens:
            integer(token)
        base = committed.digest
        engine._validate_request_latents(request.latents)
        engine._validate_request_proposals(request.proposals)
        request.digest  # a non-canonical request fails here, before any step
        require(base == expected_state, Code.STALE, "runtime committed base")
        engine._validate(committed, request)
        if request.mode == "GREEDY":
            require(bool(committed.neural.logits), detail="greedy generation needs prefill")
        target = engine.target
        if admission is None:
            admission = target.admit_stream(resident_experts=resident_experts)
        state = target.resume_stream(committed.neural)
    except ContractError as exc:
        return Sequence(engine, committed, request, None, frozenset(), None, exc.code.value)
    except Exception as exc:
        raise typed_failure(exc) from exc
    return Sequence(engine, committed, request, admission, frozenset(stop_tokens), state, None)


def finalize(engine, sequence: Sequence) -> DecodeResult:
    """Commit boundary: build receipts and the committed state for what the sequence produced."""
    require(type(sequence) is Sequence and sequence._engine is engine, Code.IDENTITY, "sequence engine")
    require(sequence.done, Code.INVALID, "sequence still active; stop() it first")
    committed = sequence._committed
    telemetry = tuple(dict(target=metrics, prefetch=()) for metrics in sequence._metrics)
    if sequence._failure is not None:
        request = sequence._request
        return DecodeResult(
            committed,
            engine.receipt(request, committed, committed, failure=sequence._failure,
                           request_identity=engine._failure_request_identity(request)),
            telemetry,
        )
    request = sequence.effective_request()
    try:
        start = perf_counter_ns()
        pairs = engine.target.finalize_stream(committed.neural, tuple(sequence._records),
                                              latents=request.latents)
        state = committed
        for neural, receipt in pairs:
            plan = plan_prefetch(committed_state=receipt.input_state, context_snapshot=committed.context.digest,
                                 step=len(neural.tokens) - 1, proposals=request.proposals,
                                 catalog=engine.prefetch_catalog)
            state = DecodeState(
                neural=neural,
                context=committed.context,
                structural=request.proposals,
                prefetch=state.prefetch + (plan,),
                receipts=state.receipts + (receipt,),
                step_latents=state.step_latents + (request.latents,),
                step_proposals=state.step_proposals + (request.proposals,),
            )
        receipt = engine.receipt(request, committed, state,
                                 accepted=state.neural.tokens[len(committed.neural.tokens):])
    except ContractError:
        raise
    except Exception as exc:
        raise typed_failure(exc) from exc
    engine._remember_validated(state.digest)
    return DecodeResult(state, receipt, telemetry + (dict(commit_ns=perf_counter_ns() - start),))
