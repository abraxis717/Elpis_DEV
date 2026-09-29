"""Steered decoding: the decode transaction composed with future-only latent steering.

One InferenceEngine.execute call is one steering epoch (block). For epoch n:
  1. the pending proposal (derived from completed epoch n-1, if any)
       -> steering.apply_fast_steering -> request, application, control state
  2. InferenceEngine.execute(state, request, expected_state=...)   (the only
     model-execution path)
  3. committed DecodeResult -> bind_completed_epoch -> observe_epoch
     -> propose_fast_steering
  4. that proposal is stored in the returned session and is eligible from
     epoch n+1 only.

Invariants:
- Future-only: an epoch never receives or waits for a proposal derived from
  itself; epoch 0 of a session is always unsteered. With block size 1, token n
  may steer token n+1.
- Exactly one InferenceEngine.execute per epoch; no retry.
- All recurrence and control state lives in the immutable SteeredSessionState
  passed in and returned. The engine holds only the InferenceEngine reference.
- Requests are transformed only by steering.apply_fast_steering; the effect
  on inference exists only through the target's LatentInput path and is
  recorded by the transaction.
- TTL/hop/cycle/duplicate/active-control policy is the frozen steering policy.
- Committed decode state and receipts remain the sole authoritative lineage.
- An application attached to an epoch that is not committed is not adopted:
  the returned control state is the incoming one (NO_OP_RETAIN_ACTIVE_CONTROL).
- No evolution, ECS or tool dependency.

Scope: steering latents can be delivered into later epochs under the frozen
temporal/control contract. No quality, reasoning or fitness claim is made.

Legacy compatibility only. This engine applies steering between GREEDY blocks
of one trajectory, which the current product boundary no longer allows for new
writes: the sequence path (:mod:`elpis.inference.sequence`) refuses any input
change on a continuation, so a steering latent enters only on a PREFILL (turn
boundary). This module is kept to reproduce and verify the historical
elpis.r3sot/elpis.sot records; channel-X steering is pending architectural
debt.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from elpis.identity import content_digest
from elpis.inference.contracts import Code, ContractError, ProposalOnly, digest_value, integer
from elpis.inference.transaction import DecodeState, InferenceRequest, InferenceEngine, DecodeResult
from elpis.inference.steering import (
    InferenceEpochBinding,
    FastControlState,
    FastSteeringApplication,
    FastSteeringProposal,
    ObserverState,
    SteeringOutcome,
    apply_fast_steering,
    bind_completed_epoch,
    initial_fast_control_state,
    observe_epoch,
    propose_fast_steering,
)

__all__ = (
    'SteeredContractError',
    'SteeredSessionState',
    'SteeredEpochResult',
    'SteeredInferenceEngine',
    'initial_steered_session',
)

STEERING_CHANNEL = 'X'


class SteeredContractError(ContractError):
    """Malformed composition input (fail closed). steering ineligibility is never an error; it is a
    deterministic NO_OP application recorded by the steering lane."""

    def __init__(self, code, reason):
        self.reason = reason
        super().__init__(code, 'r3sot.' + reason)


def _check(condition, code, reason):
    if not condition:
        raise SteeredContractError(code, reason)


def _identity(kind, value):
    return content_digest('elpis.r3sot.' + kind + '.r0', value)


@dataclass(frozen=True)
class SteeredSessionState(ProposalOnly):
    """Explicit request-local steering recurrence/control state of one composed session."""
    request_id: str
    target_model: str
    next_epoch_index: int
    previous_binding: InferenceEpochBinding | None
    previous_observer: ObserverState | None
    pending_proposal: FastSteeringProposal | None
    control_state: FastControlState

    def __post_init__(self):
        _check(type(self.request_id) is str and bool(self.request_id), Code.INVALID, 'SESSION_REQUEST_ID')
        digest_value(self.target_model)
        integer(self.next_epoch_index)
        _check(type(self.control_state) is FastControlState and
               self.control_state.request_id == self.request_id, Code.IDENTITY, 'SESSION_CONTROL_STATE')
        records = (self.previous_binding, self.previous_observer, self.pending_proposal)
        if all(record is None for record in records):
            _check(self.next_epoch_index == 0 and
                   self.control_state == initial_fast_control_state(self.request_id),
                   Code.INVALID, 'SESSION_GENESIS')
            return
        binding, observer, pending = records
        _check(type(binding) is InferenceEpochBinding and type(observer) is ObserverState and
               type(pending) is FastSteeringProposal, Code.INVALID, 'SESSION_CHAIN')
        _check(binding.epoch_index == self.next_epoch_index - 1 and
               observer.epoch_index == binding.epoch_index and
               observer.epoch_binding_digest == binding.digest and
               observer.request_id == self.request_id and
               observer.target_model == self.target_model, Code.IDENTITY, 'SESSION_OBSERVER_LINEAGE')
        _check(pending.observer_state_digest == observer.digest and
               pending.request_id == self.request_id and
               pending.source_epoch_index == binding.epoch_index and
               pending.target_model == self.target_model and
               pending.active_control_predecessor_digest == self.control_state.active_control_digest and
               pending.previous_fast_application_digest == self.control_state.previous_fast_application_digest,
               Code.IDENTITY, 'SESSION_PENDING_LINEAGE')

    @property
    def digest(self):
        return _identity('session-state', self)


def initial_steered_session(request_id, *, target_model):
    """Genesis session: epoch 0, no predecessor, no pending proposal, genesis control."""
    return SteeredSessionState(request_id, target_model, 0, None, None, None,
                             initial_fast_control_state(request_id))


@dataclass(frozen=True)
class SteeredEpochResult:
    """Outcome of one composed epoch. runtime_result is the transaction's authoritative output; every steering
    record is proposal-only. provenance_digest excludes decode telemetry (wall-clock fields)."""
    epoch_index: int
    host_request: InferenceRequest
    executed_request: InferenceRequest
    runtime_result: DecodeResult
    application: FastSteeringApplication | None
    steering_committed: bool
    binding: InferenceEpochBinding
    observer: ObserverState
    proposal: FastSteeringProposal
    session: SteeredSessionState

    @property
    def executed_request_digest(self):
        return self.executed_request.digest

    @property
    def provenance_digest(self):
        return _identity('epoch-provenance', dict(
            epoch_index=self.epoch_index,
            host_request=self.host_request.digest,
            executed_request=self.executed_request.digest,
            state=self.runtime_result.state.digest,
            receipt=self.runtime_result.receipt.digest,
            application=None if self.application is None else self.application.digest,
            steering_committed=self.steering_committed,
            binding=self.binding.digest,
            observer=self.observer.digest,
            proposal=self.proposal.digest,
            session=self.session.digest,
        ))


class SteeredInferenceEngine:
    """Composition host over one InferenceEngine. Holds only the InferenceEngine reference."""
    __slots__ = ('_runtime',)

    def __init__(self, runtime):
        _check(type(runtime) is InferenceEngine, Code.INVALID, 'RUNTIME_TYPE')
        object.__setattr__(self, '_runtime', runtime)

    def __setattr__(self, name, value):
        raise AttributeError('SteeredInferenceEngine is immutable')

    @property
    def runtime(self):
        return self._runtime

    def initial_session(self, request_id):
        return initial_steered_session(request_id, target_model=self._runtime.target.model_identity)

    def execute_epoch(self, session, state, request, *, expected_state=None, prefetch_enabled=False):
        """One composed epoch: apply pending proposal (if any), one InferenceEngine.execute, then
        bind/observe/propose over the committed result. Returns SteeredEpochResult."""
        _check(type(session) is SteeredSessionState, Code.INVALID, 'SESSION_TYPE')
        _check(type(state) is DecodeState, Code.INVALID, 'STATE_TYPE')
        _check(type(request) is InferenceRequest, Code.INVALID, 'REQUEST_TYPE')
        _check(request.request_id == session.request_id, Code.IDENTITY, 'SESSION_REQUEST_MISMATCH')
        target = self._runtime.target
        _check(session.target_model == target.model_identity, Code.IDENTITY, 'SESSION_TARGET_MISMATCH')
        epoch = session.next_epoch_index
        projection = target.projections.get(STEERING_CHANNEL)
        control = session.control_state
        executed, application, attached = request, None, control
        if session.pending_proposal is not None:
            executed, application, attached = apply_fast_steering(
                session.pending_proposal, request, target_epoch_index=epoch, target_state=state,
                x_projection=projection, control_state=control)
        result = self._runtime.execute(
            state, executed, expected_state=state.digest if expected_state is None else expected_state,
            prefetch_enabled=prefetch_enabled)
        steering_committed = (application is not None and application.outcome is SteeringOutcome.APPLIED and
                              result.receipt.terminal == 'COMMITTED')
        control_after = attached if steering_committed else control
        binding = bind_completed_epoch(executed, result, epoch_index=epoch, predecessor=session.previous_binding)
        observer = observe_epoch(binding, executed, result, predecessor=session.previous_observer)
        proposal = propose_fast_steering(observer, target_model=target.model_identity, x_projection=projection,
                                         control_state=control_after)
        next_session = SteeredSessionState(session.request_id, session.target_model, epoch + 1, binding,
                                         observer, proposal, control_after)
        return SteeredEpochResult(epoch, request, executed, result, application, steering_committed, binding,
                                observer, proposal, next_session)

    def generate(self, session, state, request, *, block_size, prefetch_enabled=False):
        """Partition one GREEDY request into consecutive epochs of at most block_size tokens.
        State and session thread through committed results; stops at the first non-committed
        epoch (no retry). Returns a tuple of SteeredEpochResult."""
        _check(type(request) is InferenceRequest and request.mode == 'GREEDY', Code.INVALID, 'GENERATE_MODE')
        _check(type(block_size) is int and block_size >= 1, Code.INVALID, 'BLOCK_SIZE')
        remaining = request.count
        epochs = []
        while remaining > 0:
            block = replace(request, count=min(block_size, remaining))
            epoch = self.execute_epoch(session, state, block, prefetch_enabled=prefetch_enabled)
            epochs.append(epoch)
            if epoch.runtime_result.receipt.terminal != 'COMMITTED':
                break
            remaining -= block.count
            session, state = epoch.session, epoch.runtime_result.state
        return tuple(epochs)
