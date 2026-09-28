"""Read-only observation of completed decode epochs and future-only latent steering.

Placement: an asynchronous, request-local lane over COMPLETED decode epochs.
Decoding never waits for this module. Every entry point consumes only objects
that exist after a transaction terminated (the request and its DecodeResult or
DecodeReceipt) or immutable public target descriptors (model identity digest,
LatentProjection).

Topology:
  completed epoch n -> InferenceEpochBinding -> ObserverState
  -> FastSteeringProposal (not_before n+1, expires_after n+2)
  -> FastSteeringApplication + a later InferenceRequest carrying at most one
     LatentInput on channel X, gated by the host-owned FastControlState
     (stall / cycle / hop guard).
Application provenance is the unchanged transaction chain
InferenceRequest.latents -> StepReceipt.latents -> DecodeReceipt.target_steps
-> DecodeReceipt.digest. No second steering receipt exists.
global_event_fields() only exposes the values an ECS integration would bind;
this module never touches the ECS.

Excluded by construction: retroaction on the source epoch, token replacement,
logit overwrite, weight or context mutation, ECS access, model or tool
execution, semantic/admission/execution authority, hidden or global mutable
recurrence state. Every public record subclasses ProposalOnly. Malformed
programmer input raises SteeringContractError; ordinary ineligibility is a
deterministic NO_OP record.

Steering source: the DYN4 observation (current value and delta) only. The
CORE12 observation is observer/provenance state and contributes nothing to
steering. No quality, reasoning or causal-effect claim is made here (see
docs/NONCLAIMS.md).

Arithmetic: IEEE-754 binary64, fixed left-to-right accumulation, only
correctly rounded +, -, *, / and sqrt. _dyn4, _normalize_dyn4,
_canonical_steering4, _width_assignments and _lift_steering are pure and are
the replacement surface for a later native backend.

Fast-control guard: MAX_HOPS=8, CYCLE_WINDOW=4, TTL=2 epochs, genesis active
control "0"*64, failure default NO_OP_RETAIN_ACTIVE_CONTROL. The active-control
identity is the digest of the actually applied X LatentInput. Guard state is
host-owned (FastControlState) and passed explicitly in and out. Frozen
contract and guard digests below are persisted identities and are kept.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

from elpis.identity import CanonicalIdentityError, content_digest
from elpis.inference.context import Snapshot
from elpis.inference.contracts import Code, ContractError, ProposalOnly, digest_value, integer
from elpis.inference.target import LatentInput, LatentProjection, NeuralState, StepReceipt, Tensor
from elpis.inference.transaction import DecodeState, InferenceRequest, DecodeReceipt, DecodeResult

__all__ = (
    'CORE12_FEATURES', 'DYN4_FEATURES', 'SteeringOutcome', 'SteeringContractError',
    'InferenceEpochBinding', 'ObserverState', 'FastSteeringProposal',
    'FastSteeringApplication', 'bind_completed_epoch', 'observe_epoch',
    'propose_fast_steering', 'apply_fast_steering', 'global_event_fields',
    'width_transport_matrix', 'width_transport_digest',
    'frozen_steering_contract', 'frozen_steering_contract_digest',
    'FastGuardReason', 'FastControlState', 'initial_fast_control_state',
)

# Frozen steering and guard authority identities (persisted).
EPOCH_BINDING_CONTRACT_DIGEST = '63231a2c35fac3828d70f38ee3c2d0911f79b3fd56b010ef625a6748480f1236'
CORE12_CONTRACT_DIGEST = 'c8dc86d8890a5645d744d2206d2f847ec53f38f7daf8454f1d5f7ff43336a414'
CORE12_RECURRENCE_DIGEST = '7e95b3658ddf7ff07782b48420c187c3c0e0eced803ef3e6ab994b91ced2cee5'
DYN4_CONTRACT_DIGEST = 'aac52bb3f387cc47594abdc3d129c81c6d7cebb8a1f225ad302910ddb7a5419a'
DYN4_RECURRENCE_DIGEST = '9f490b171934cc6cb2983da0258f484768e9765d5f473bfedabed75fee77b5f5'
X_LATENT_CONTRACT_DIGEST = 'bb5d5cae93ee8cceac5aee8ac22dd185f56a102434591af72acd440a322aee79'
FAST_POLICY_DIGEST = '5a79c69fdec2fac63aded113d82292da0332240dcb114dc92751470e4b8549d4'
STALL_GUARD_DIGEST = '0f6a28db77c9fb85e03c50c36d8decdc6e710b633cad99b9ba754c218fa7ef8c'
FAST_STEERING_CONTRACT_DIGEST = '2533e28a485d2144d8229a46cdf10ff99430b8626c48001612672c003a992176'
PHASE4_MECHANICS_SOURCE_SHA256 = 'e35de2f58dac4a7f7ca3dee56c62649b64ef918de228dcbbf27802b15dc71460'
PHASE8_MECHANICS_SOURCE_SHA256 = '317394545b5ad24c20d5d794f6fec19425633b563a6fe7c5fa93a76413e1935f'

STEERING_OBJECT_LABEL = 'BB33_A4_D100'
STEERING_ADAPTER_DIGEST = '5ecf5a9eec6f3757b5aaef9ac9597761b221ab8c0d79e146915a02dd4bbb14ab'
STEERING_CONTRACT_DIGEST = 'ac72e897a7109aaa922b0ea9d599eedb626d93509dc7a496e300e593d92c0955'
ORIENTATION_DIGEST = '8cfa6780404d92df6d13cccc84f220afe176b94914472df77b3c3f7cb727dd18'
FIXED_GAIN = 0.09
ORIENTATION_8X4 = (
    (0.125, 0.125, 0.125, 0.125),
    (0.125, 0.125, 0.125, -0.125),
    (-0.125, -0.125, 0.125, 0.125),
    (0.125, 0.125, -0.125, -0.125),
    (-0.125, 0.125, -0.125, -0.125),
    (0.125, 0.125, -0.125, 0.125),
    (-0.125, 0.125, -0.125, 0.125),
    (0.125, -0.125, 0.125, 0.125),
)

WIDTH_RULE_LABEL = 'GENERAL_SUFFIX_BALANCED_REPEAT_ISOMETRY_R0'
WIDTH_RULE_AUTHORITY_DIGEST = 'daffd7eecd6505bfe8c2cbef292ca3a34b0b51d34d85726c1b2f7d72cb5fa4a7'
SOURCE_DIMENSION = 4
WIDTH_MIN = 4
WIDTH_MAX = 32  # scientific coverage bound; the rule's d>=4 domain is not generalized past 32
X_LATENT_TTL_EPOCHS = 2
MAX_HOPS = 8
CYCLE_WINDOW = 4
GENESIS_CONTROL_DIGEST = '0' * 64
FAST_FAILURE_DEFAULT = 'NO_OP_RETAIN_ACTIVE_CONTROL'

CORE12_FEATURES = (
    'epoch_committed_step_count',
    'epoch_accepted_prefix_count',
    'total_committed_token_count',
    'receipt_structural_count',
    'epoch_latent_input_count',
    'epoch_proposal_input_count',
    'context_canonical_count',
    'context_visible_count',
    'context_retired_count',
    'context_generation',
    'speculative_draft_present',
    'terminal_committed_flag',
)
DYN4_FEATURES = ('hidden_mean_square', 'logits_mean_square', 'logit_span', 'top1_top2_margin')
TERMINAL_COMMITTED_INDEX = 11
TERMINAL_STATUSES = ('COMMITTED', 'FAILED')


class SteeringContractError(ContractError):
    """Malformed or incoherent programmer input to the steering lane (fail closed)."""

    def __init__(self, code, reason):
        self.reason = reason
        super().__init__(code, 'sot.' + reason)


def _check(condition, code, reason):
    if not condition:
        raise SteeringContractError(code, reason)


def _is_digest(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _identity(kind, value):
    # New runtime identities; no digest compatibility with experiment artifacts is claimed.
    return content_digest('elpis.sot.' + kind + '.r0', value)


def _int_vector(value, size):
    return type(value) is tuple and len(value) == size and all(type(v) is int for v in value)


def _float_vector(value, size):
    return (type(value) is tuple and len(value) == size and
            all(type(v) is float and math.isfinite(v) for v in value))


def frozen_steering_contract():
    """Frozen steering contract body rebuilt from this module's constants."""
    return {
        'adapter_changed': False,
        'delta_contribution_gate': 0.05,
        'dyn4_orientation_digest': ORIENTATION_DIGEST,
        'dyn4_orientation_matrix': [list(row) for row in ORIENTATION_8X4],
        'fixed_gain': FIXED_GAIN,
        'gain_changed': False,
        'observer_definition_changed': False,
        'observer_specificity_gate': 0.05,
        'schema': 'elpis.branch43.phase30.frozen-steering.v0',
        'selected_adapter_digest': STEERING_ADAPTER_DIGEST,
        'selected_adapter_label': STEERING_OBJECT_LABEL,
        'thresholds_changed': False,
    }


# The ac72... value is historical provenance and remains
# externally visible. New production code does not recompute that
# unframed structured SHA-256. The reconstructed body is separately
# integrity-bound under Canonical Identity v1.
FROZEN_STEERING_CONTRACT_CANONICAL_IDENTITY = (
    'd9385850b2a5f128c18bdd6de5bee5f38fe79224a01db10501ebc973551b0022'
)


def frozen_steering_contract_digest():
    return STEERING_CONTRACT_DIGEST


if (content_digest('elpis.sot.frozen-steering-contract.r0',
                   frozen_steering_contract()) !=
        FROZEN_STEERING_CONTRACT_CANONICAL_IDENTITY):
    raise RuntimeError('steering constants diverge from the canonical frozen-contract binding')


# ------------------------------------------------------------ pure kernels
def _finite_floats(values, reason):
    _check(type(values) is tuple, Code.INVALID, reason)
    out = []
    for value in values:
        _check(type(value) in (int, float), Code.ENCODING, reason)
        try:
            number = float(value)
        except OverflowError:
            raise SteeringContractError(Code.ENCODING, reason) from None
        _check(math.isfinite(number), Code.ENCODING, reason)
        out.append(number)
    return tuple(out)


def _dyn4(hidden, logits):
    """DYN4_FEATURES aggregates of one committed NeuralState output (read-only)."""
    h = _finite_floats(hidden, 'DYN4_HIDDEN_NONFINITE')
    z = _finite_floats(logits, 'DYN4_LOGITS_NONFINITE')
    _check(len(h) >= 1, Code.INVALID, 'DYN4_HIDDEN_EMPTY')
    _check(len(z) >= 2, Code.UNSUPPORTED, 'DYN4_LOGITS_LENGTH')
    acc = 0.0
    for value in h:
        acc += value * value
    hidden_mean_square = acc / len(h)
    acc = 0.0
    for value in z:
        acc += value * value
    logits_mean_square = acc / len(z)
    ordered = sorted(z, reverse=True)
    out = (hidden_mean_square, logits_mean_square, ordered[0] - ordered[-1], ordered[0] - ordered[1])
    out = tuple(value + 0.0 for value in out)  # canonical +0.0
    _check(all(math.isfinite(v) and v >= 0.0 for v in out), Code.ENCODING, 'DYN4_DERIVED_INVALID')
    return out


def _normalize_dyn4(dyn4, dyn4_delta):
    current = tuple(x / (1.0 + x) for x in dyn4)
    delta = tuple(d / (1.0 + abs(d)) for d in dyn4_delta)
    return current + delta


def _canonical_steering4(r8):
    out = []
    for j in range(SOURCE_DIMENSION):
        acc = 0.0
        for i in range(8):
            acc += r8[i] * ORIENTATION_8X4[i][j]
        out.append(FIXED_GAIN * acc)
    return tuple(out)


def _width_assignments(width):
    _check(type(width) is int, Code.INVALID, 'WIDTH_TYPE')
    _check(WIDTH_MIN <= width <= WIDTH_MAX, Code.UNSUPPORTED, 'WIDTH_OUTSIDE_QUALIFIED_4_32')
    q, r = divmod(width, SOURCE_DIMENSION)
    return (0, 1, 2, 3) * q + tuple(range(SOURCE_DIMENSION - r, SOURCE_DIMENSION))


def _width_coefficients(assignments):
    counts = tuple(assignments.count(k) for k in range(SOURCE_DIMENSION))
    return tuple(1.0 / math.sqrt(counts[k]) for k in assignments)


def _lift_steering(u4, width):
    assignments = _width_assignments(width)
    coefficients = _width_coefficients(assignments)
    return tuple(coefficients[j] * u4[k] for j, k in enumerate(assignments))


def width_transport_matrix(width):
    """L_d (d x 4): exactly one nonzero per row; inspection only."""
    assignments = _width_assignments(width)
    coefficients = _width_coefficients(assignments)
    return tuple(tuple(coefficients[j] if k == column else 0.0 for column in range(SOURCE_DIMENSION))
                 for j, k in enumerate(assignments))


def width_transport_digest(width):
    assignments = _width_assignments(width)
    return _identity('width-transport', dict(
        rule=WIDTH_RULE_LABEL,
        rule_authority=WIDTH_RULE_AUTHORITY_DIGEST,
        source_dimension=SOURCE_DIMENSION,
        target_width=width,
        assignments=assignments,
        counts=tuple(assignments.count(k) for k in range(SOURCE_DIMENSION)),
        coefficient='1/sqrt(count[source])',
    ))


def _steering_profile():
    return dict(
        schema='elpis.sot.fast-steering-profile.r0',
        observer=dict(epoch_binding=EPOCH_BINDING_CONTRACT_DIGEST, core12=CORE12_CONTRACT_DIGEST,
                      core12_recurrence=CORE12_RECURRENCE_DIGEST, dyn4=DYN4_CONTRACT_DIGEST,
                      dyn4_recurrence=DYN4_RECURRENCE_DIGEST,
                      lineage='request_id+context_snapshot+target_model'),
        steering=dict(contract=STEERING_CONTRACT_DIGEST, object_label=STEERING_OBJECT_LABEL,
                      adapter=STEERING_ADAPTER_DIGEST, orientation=ORIENTATION_DIGEST,
                      orientation_matrix=ORIENTATION_8X4, gain=FIXED_GAIN,
                      source='DYN4_CURRENT_AND_DELTA', current_normalization='x/(1+x)',
                      delta_normalization='d/(1+abs(d))',
                      arithmetic='IEEE754_BINARY64_LEFT_TO_RIGHT', clipping='NONE'),
        transport=dict(rule=WIDTH_RULE_LABEL, authority=WIDTH_RULE_AUTHORITY_DIGEST,
                       source_dimension=SOURCE_DIMENSION, width_min=WIDTH_MIN, width_max=WIDTH_MAX),
        latent=dict(contract=X_LATENT_CONTRACT_DIGEST, policy=FAST_POLICY_DIGEST, channel='X',
                    ttl_epochs=X_LATENT_TTL_EPOCHS, host_x_collision='NO_OP',
                    placement='APPEND_AFTER_HOST_LATENTS'),
        fast_guard=dict(
            stall_guard=STALL_GUARD_DIGEST,
            fast_steering_contract=FAST_STEERING_CONTRACT_DIGEST,
            phase4_mechanics_source=PHASE4_MECHANICS_SOURCE_SHA256,
            phase8_mechanics_source=PHASE8_MECHANICS_SOURCE_SHA256,
            max_hops=MAX_HOPS,
            cycle_window=CYCLE_WINDOW,
            ttl_epochs=X_LATENT_TTL_EPOCHS,
            genesis_active_control=GENESIS_CONTROL_DIGEST,
            failure_default=FAST_FAILURE_DEFAULT,
            guard_order=('ACTIVE_CONTROL_PREDECESSOR_MISMATCH', 'HOP_BUDGET_EXHAUSTED',
                         'IDENTICAL_STATE_NOOP', 'CONTROL_CYCLE_DETECTED'),
            identical_pair='(observer_state_digest, active_control_digest)',
            active_control_identity='SoT X LatentInput.digest',  # persisted profile value
            historical_phase4_digest_equivalence=False,
        ),
        application_order=('PROPOSAL_DISPOSITION', 'TOO_EARLY', 'EXPIRED', 'REQUEST_LINEAGE',
                           'CONTEXT_LINEAGE', 'TARGET_MODEL', 'SOURCE_CONTINUITY', 'X_PROJECTION',
                           'DUPLICATE', 'HOST_X_COLLISION', 'FAST_GUARD', 'APPLY'),
    )


STEERING_PROFILE_DIGEST = _identity('fast-steering-profile', _steering_profile())


# ------------------------------------------------------------ epoch binding
@dataclass(frozen=True)
class InferenceEpochBinding(ProposalOnly):
    """One completed decode transaction (persisted schema elpis.branch43.inference-epoch-binding.v0)."""
    epoch_index: int
    predecessor_epoch_binding_digest: str | None
    runtime_receipt_digest: str
    runtime_request_digest: str
    input_decode_state_digest: str
    output_decode_state_digest: str
    context_snapshot_digest: str
    terminal_status: str

    def __post_init__(self):
        integer(self.epoch_index)
        if self.predecessor_epoch_binding_digest is not None:
            digest_value(self.predecessor_epoch_binding_digest)
        for value in (self.runtime_receipt_digest, self.runtime_request_digest,
                      self.input_decode_state_digest, self.output_decode_state_digest,
                      self.context_snapshot_digest):
            digest_value(value)
        _check(self.terminal_status in TERMINAL_STATUSES, Code.INVALID, 'TERMINAL_STATUS')

    @property
    def digest(self):
        return _identity('inference-epoch-binding', self)


def _validated_epoch(request, result):
    """Identity-level lineage of an existing (request, result); never re-executes decoding."""
    _check(type(request) is InferenceRequest, Code.INVALID, 'REQUEST_TYPE')
    _check(type(result) is DecodeResult, Code.INVALID, 'RESULT_TYPE')
    state, receipt = result.state, result.receipt
    _check(type(state) is DecodeState and type(state.neural) is NeuralState and
           type(state.context) is Snapshot, Code.INVALID, 'STATE_TYPE')
    _check(type(receipt) is DecodeReceipt, Code.INVALID, 'RECEIPT_TYPE')
    committed = receipt.terminal == 'COMMITTED'
    _check(receipt.terminal in TERMINAL_STATUSES and committed == (receipt.failure is None),
           Code.INVALID, 'TERMINAL_STATUS')
    _check(receipt.failure is None or (type(receipt.failure) is str and bool(receipt.failure)),
           Code.INVALID, 'FAILURE_CODE')
    for collection in (receipt.target_steps, receipt.accepted_prefix, receipt.structural,
                       state.receipts, state.step_latents, state.step_proposals, state.prefetch,
                       state.neural.tokens):
        _check(type(collection) is tuple, Code.INVALID, 'COLLECTION_TYPE')
    _check(all(type(step) is tuple for step in state.step_latents + state.step_proposals),
           Code.INVALID, 'STEP_INPUT_TYPE')
    _check(all(type(step) is StepReceipt for step in receipt.target_steps), Code.INVALID, 'STEP_RECEIPT_TYPE')
    _check(_is_digest(receipt.input_state), Code.INVALID, 'DIGEST_ENCODING')
    try:
        request_digest = request.digest
        receipt_digest = receipt.digest
        state_digest = state.digest
        context_digest = state.context.digest
        neural_digest = state.neural.digest
    except CanonicalIdentityError:
        raise SteeringContractError(Code.IDENTITY, 'NONCANONICAL_IDENTITY') from None
    _check(receipt.request == request_digest, Code.IDENTITY, 'REQUEST_IDENTITY_MISMATCH')
    _check(receipt.output_state == state_digest, Code.IDENTITY, 'OUTPUT_STATE_MISMATCH')
    _check(receipt.context_snapshot == context_digest == state.neural.context_snapshot,
           Code.STALE, 'CONTEXT_SNAPSHOT_MISMATCH')
    _check(receipt.model == state.neural.model, Code.IDENTITY, 'MODEL_MISMATCH')
    tokens, steps = len(state.neural.tokens), len(receipt.target_steps)
    _check(len(state.receipts) == tokens and len(state.step_latents) == tokens and
           len(state.step_proposals) == tokens and len(state.prefetch) == tokens and
           steps <= tokens and len(receipt.accepted_prefix) == steps,
           Code.STALE, 'COMMITTED_LINEAGE_CARDINALITY')
    if committed:
        _check(request.context_snapshot == context_digest, Code.STALE, 'CONTEXT_SNAPSHOT_MISMATCH')
    else:
        _check(steps == 0 and receipt.input_state == receipt.output_state,
               Code.IDENTITY, 'FAILED_RECEIPT_COMMITTED_EFFECT')
    _check(state.receipts[tokens - steps:] == receipt.target_steps and
           (steps == 0 or receipt.target_steps[-1].output_state == neural_digest),
           Code.IDENTITY, 'TARGET_STEP_SUFFIX_MISMATCH')
    return request_digest, receipt_digest, state_digest, context_digest, neural_digest


def bind_completed_epoch(request, result, *, epoch_index, predecessor=None):
    """Bind one completed decode transaction. Derived strictly after DecodeResult
    exists; nothing here participates in the bound receipt."""
    request_digest, receipt_digest, state_digest, context_digest, _ = _validated_epoch(request, result)
    predecessor_digest = None
    if predecessor is not None:
        _check(type(predecessor) is InferenceEpochBinding, Code.INVALID, 'PREDECESSOR_BINDING_TYPE')
        _check(type(epoch_index) is int and epoch_index > predecessor.epoch_index,
               Code.STALE, 'EPOCH_INDEX_NOT_MONOTONIC')
        predecessor_digest = predecessor.digest
    return InferenceEpochBinding(epoch_index, predecessor_digest, receipt_digest, request_digest,
                                 result.receipt.input_state, state_digest, context_digest,
                                 result.receipt.terminal)


# ------------------------------------------------------------ observer
def _core12(receipt, state):
    steps = len(receipt.target_steps)
    tokens = len(state.neural.tokens)
    start = tokens - steps  # suffix aligned exactly to DecodeReceipt.target_steps
    context = state.context
    return (
        steps,
        len(receipt.accepted_prefix),
        tokens,
        len(receipt.structural),
        sum(len(step) for step in state.step_latents[start:]),
        sum(len(step) for step in state.step_proposals[start:]),
        len(context.canonical),
        len(context.visible),
        len(context.retired),
        context.generation,
        0 if receipt.draft is None else 1,
        1 if receipt.terminal == 'COMMITTED' and receipt.failure is None else 0,
    )


def _dyn4_or_absent(neural):
    # No committed neural output yet: DYN4 is absent (None), never zero-encoded.
    _check(type(neural.hidden) is tuple and type(neural.logits) is tuple, Code.INVALID, 'NEURAL_OUTPUT_TYPE')
    if not neural.hidden and not neural.logits:
        return None
    return _dyn4(neural.hidden, neural.logits)


@dataclass(frozen=True)
class ObserverState(ProposalOnly):
    """Immutable CORE12_FEATURES + DYN4_FEATURES observation, recurrence depth 1.
    Raw hidden/logit vectors and token identities are never retained."""
    epoch_binding_digest: str
    request_id: str
    epoch_index: int
    context_snapshot_digest: str
    target_model: str
    source_neural_state_digest: str
    core12: tuple[int, ...]
    core12_delta: tuple[int, ...]
    reset_flag: int
    dyn4: tuple[float, ...] | None
    dyn4_delta: tuple[float, ...] | None
    predecessor_observer_digest: str | None
    terminal_status: str
    failure_code: str | None

    def __post_init__(self):
        for value in (self.epoch_binding_digest, self.context_snapshot_digest, self.target_model,
                      self.source_neural_state_digest):
            digest_value(value)
        if self.predecessor_observer_digest is not None:
            digest_value(self.predecessor_observer_digest)
        _check(type(self.request_id) is str and bool(self.request_id), Code.INVALID, 'REQUEST_ID')
        integer(self.epoch_index)
        _check(_int_vector(self.core12, 12) and all(v >= 0 for v in self.core12) and
               self.core12[10] in (0, 1) and self.core12[11] in (0, 1), Code.INVALID, 'CORE12')
        _check(_int_vector(self.core12_delta, 12), Code.INVALID, 'CORE12_DELTA')
        _check(type(self.reset_flag) is int and self.reset_flag in (0, 1), Code.INVALID, 'RESET_FLAG')
        _check(self.dyn4 is None or (_float_vector(self.dyn4, 4) and all(v >= 0.0 for v in self.dyn4)),
               Code.INVALID, 'DYN4')
        _check(self.dyn4_delta is None or (self.dyn4 is not None and _float_vector(self.dyn4_delta, 4)),
               Code.INVALID, 'DYN4_DELTA')
        if self.reset_flag == 1:
            _check(self.core12_delta == (0,) * 12 and
                   (self.dyn4_delta is None) == (self.dyn4 is None) and
                   (self.dyn4_delta is None or self.dyn4_delta == (0.0,) * 4),
                   Code.INVALID, 'RESET_DELTA_NONZERO')
        else:
            _check(self.predecessor_observer_digest is not None, Code.INVALID, 'NONRESET_WITHOUT_PREDECESSOR')
        _check(self.terminal_status in TERMINAL_STATUSES and
               (self.terminal_status == 'COMMITTED') == (self.failure_code is None) and
               self.core12[TERMINAL_COMMITTED_INDEX] == (1 if self.terminal_status == 'COMMITTED' else 0) and
               (self.failure_code is None or (type(self.failure_code) is str and bool(self.failure_code))),
               Code.INVALID, 'TERMINAL_METADATA')

    @property
    def digest(self):
        return _identity('observer-state', self)


def observe_epoch(binding, request, result, *, predecessor=None):
    """Read-only observation of one bound completed epoch. predecessor must be the
    observation of the immediately preceding bound epoch, or None (reset).
    Lineage key (request_id, context snapshot, target model): any change resets."""
    _check(type(binding) is InferenceEpochBinding, Code.INVALID, 'EPOCH_BINDING_TYPE')
    request_digest, receipt_digest, state_digest, context_digest, neural_digest = _validated_epoch(request, result)
    receipt, state = result.receipt, result.state
    _check((binding.runtime_request_digest, binding.runtime_receipt_digest,
            binding.input_decode_state_digest, binding.output_decode_state_digest,
            binding.context_snapshot_digest, binding.terminal_status) ==
           (request_digest, receipt_digest, receipt.input_state, state_digest, context_digest,
            receipt.terminal), Code.IDENTITY, 'EPOCH_BINDING_MISMATCH')
    core12 = _core12(receipt, state)
    dyn4 = _dyn4_or_absent(state.neural)
    reset_flag = 1
    core12_delta = (0,) * 12
    dyn4_delta = None if dyn4 is None else (0.0,) * 4
    predecessor_digest = None
    if predecessor is not None:
        _check(type(predecessor) is ObserverState, Code.INVALID, 'PREDECESSOR_OBSERVER_TYPE')
        _check(predecessor.epoch_binding_digest == binding.predecessor_epoch_binding_digest,
               Code.STALE, 'OBSERVER_PREDECESSOR_MISMATCH')
        _check(predecessor.epoch_index < binding.epoch_index, Code.STALE, 'OBSERVER_PREDECESSOR_NOT_EARLIER')
        predecessor_digest = predecessor.digest
        if ((predecessor.request_id, predecessor.context_snapshot_digest, predecessor.target_model) ==
                (request.request_id, context_digest, receipt.model)):
            reset_flag = 0
            core12_delta = tuple(a - b for a, b in zip(core12, predecessor.core12))
            dyn4_delta = (None if dyn4 is None or predecessor.dyn4 is None
                          else tuple(a - b for a, b in zip(dyn4, predecessor.dyn4)))
    return ObserverState(binding.digest, request.request_id, binding.epoch_index, context_digest,
                            receipt.model, neural_digest, core12, core12_delta, reset_flag, dyn4,
                            dyn4_delta, predecessor_digest, receipt.terminal, receipt.failure)


# ------------------------------------------------------------ fast steering
class SteeringOutcome(str, Enum):
    PROPOSED = 'PROPOSED'
    APPLIED = 'APPLIED'
    NO_OP_FAILED_SOURCE = 'NO_OP_FAILED_SOURCE'
    NO_OP_RESET = 'NO_OP_RESET'
    NO_OP_DYN4_ABSENT = 'NO_OP_DYN4_ABSENT'
    NO_OP_TARGET_MISMATCH = 'NO_OP_TARGET_MISMATCH'
    NO_OP_PROJECTION_MISMATCH = 'NO_OP_PROJECTION_MISMATCH'
    NO_OP_INVALID_WIDTH = 'NO_OP_INVALID_WIDTH'
    NO_OP_NONFINITE = 'NO_OP_NONFINITE'
    NO_OP_TOO_EARLY = 'NO_OP_TOO_EARLY'
    NO_OP_EXPIRED = 'NO_OP_EXPIRED'
    NO_OP_LINEAGE_MISMATCH = 'NO_OP_LINEAGE_MISMATCH'
    NO_OP_CONTEXT_MISMATCH = 'NO_OP_CONTEXT_MISMATCH'
    NO_OP_HOST_X_COLLISION = 'NO_OP_HOST_X_COLLISION'
    NO_OP_DUPLICATE = 'NO_OP_DUPLICATE'
    NO_OP_FAST_GUARD = 'NO_OP_FAST_GUARD'


class FastGuardReason(str, Enum):
    """Exact frozen guard rejection reasons."""
    ACTIVE_CONTROL_PREDECESSOR_MISMATCH = 'ACTIVE_CONTROL_PREDECESSOR_MISMATCH'
    HOP_BUDGET_EXHAUSTED = 'HOP_BUDGET_EXHAUSTED'
    IDENTICAL_STATE_NOOP = 'IDENTICAL_STATE_NOOP'
    CONTROL_CYCLE_DETECTED = 'CONTROL_CYCLE_DETECTED'


_PROPOSAL_DISPOSITIONS = frozenset((
    SteeringOutcome.PROPOSED, SteeringOutcome.NO_OP_FAILED_SOURCE, SteeringOutcome.NO_OP_RESET,
    SteeringOutcome.NO_OP_DYN4_ABSENT, SteeringOutcome.NO_OP_TARGET_MISMATCH,
    SteeringOutcome.NO_OP_PROJECTION_MISMATCH, SteeringOutcome.NO_OP_INVALID_WIDTH,
    SteeringOutcome.NO_OP_NONFINITE,
))


def _valid_x_projection(projection):
    return (type(projection) is LatentProjection and projection.channel == 'X' and
            type(projection.source_schema) is str and bool(projection.source_schema) and
            type(projection.weights) is Tensor and len(projection.weights.shape) == 2 and
            projection.version == 0)


@dataclass(frozen=True)
class FastControlState(ProposalOnly):
    """Host-owned, request-local fast-control chain for the frozen
    stall/cycle/hop guard. Active-control identity is the digest of the applied
    X LatentInput (a concrete specialization of the abstract active control;
    no equivalence with historical digests is claimed). Passed explicitly
    in and out; never cached, registered or persisted by this module."""
    request_id: str
    active_control_digest: str
    applied_control_history: tuple[str, ...]  # last min(hop_count, CYCLE_WINDOW) applied controls
    hop_count: int
    last_observer_state_digest: str | None  # identical-state pair, set on APPLY only
    last_active_control_digest: str | None
    previous_fast_application_digest: str | None
    last_applied_epoch_index: int | None

    def __post_init__(self):
        _check(type(self.request_id) is str and bool(self.request_id), Code.INVALID, 'REQUEST_ID')
        _check(_is_digest(self.active_control_digest), Code.INVALID, 'ACTIVE_CONTROL_DIGEST')
        _check(type(self.hop_count) is int and 0 <= self.hop_count <= MAX_HOPS, Code.INVALID, 'HOP_COUNT')
        history = self.applied_control_history
        _check(type(history) is tuple and all(_is_digest(d) for d in history) and
               len(history) == min(self.hop_count, CYCLE_WINDOW), Code.INVALID, 'CONTROL_HISTORY')
        optional = (self.last_observer_state_digest, self.last_active_control_digest,
                    self.previous_fast_application_digest)
        _check(all(d is None or _is_digest(d) for d in optional), Code.INVALID, 'CONTROL_DIGEST')
        _check(self.last_applied_epoch_index is None or
               (type(self.last_applied_epoch_index) is int and self.last_applied_epoch_index >= 0),
               Code.INVALID, 'LAST_APPLIED_EPOCH')
        started = self.hop_count > 0
        _check(all((value is not None) == started for value in optional + (self.last_applied_epoch_index,)),
               Code.INVALID, 'CONTROL_CHAIN')
        _check(history[-1] == self.active_control_digest if started else
               self.active_control_digest == GENESIS_CONTROL_DIGEST, Code.INVALID, 'ACTIVE_CONTROL')

    @property
    def digest(self):
        return _identity('fast-control-state', self)


def initial_fast_control_state(request_id):
    """Genesis control state for one request lineage (host creates it per request)."""
    return FastControlState(request_id, GENESIS_CONTROL_DIGEST, (), 0, None, None, None, None)


def _fast_guard(control_state, observer_state_digest, active_control_predecessor_digest,
                candidate_control_digest):
    """Frozen guard steps 4-7 over digest identities only (pure)."""
    if active_control_predecessor_digest != control_state.active_control_digest:
        return FastGuardReason.ACTIVE_CONTROL_PREDECESSOR_MISMATCH
    if control_state.hop_count >= MAX_HOPS:
        return FastGuardReason.HOP_BUDGET_EXHAUSTED
    if ((observer_state_digest, control_state.active_control_digest) ==
            (control_state.last_observer_state_digest, control_state.last_active_control_digest)):
        return FastGuardReason.IDENTICAL_STATE_NOOP
    if candidate_control_digest in control_state.applied_control_history[-CYCLE_WINDOW:]:
        return FastGuardReason.CONTROL_CYCLE_DETECTED
    return None


def _advance_fast_control(control_state, observer_state_digest, candidate_control_digest,
                          application_digest, target_epoch_index):
    """Guard step 8 (APPLY): exactly one hop; keep the last CYCLE_WINDOW controls."""
    return FastControlState(
        control_state.request_id, candidate_control_digest,
        (control_state.applied_control_history + (candidate_control_digest,))[-CYCLE_WINDOW:],
        control_state.hop_count + 1, observer_state_digest, control_state.active_control_digest,
        application_digest, target_epoch_index)


@dataclass(frozen=True)
class FastSteeringProposal(ProposalOnly):
    """Bounded future-only X proposal from one observation (PROPOSED or NO_OP_*)."""
    disposition: SteeringOutcome
    request_id: str
    source_epoch_index: int
    source_epoch_binding_digest: str
    observer_state_digest: str
    predecessor_observer_digest: str | None
    source_context_snapshot_digest: str
    source_neural_state_digest: str
    target_model: str
    x_projection_digest: str | None
    x_source_schema: str | None
    target_width: int | None
    width_transport_digest: str | None
    latent: LatentInput | None
    latent_digest: str | None
    proposed_active_control_digest: str | None
    active_control_predecessor_digest: str
    previous_fast_application_digest: str | None
    steering_contract_digest: str
    steering_profile_digest: str
    not_before_epoch: int
    expires_after_epoch: int

    def __post_init__(self):
        _check(type(self.disposition) is SteeringOutcome and self.disposition in _PROPOSAL_DISPOSITIONS,
               Code.INVALID, 'PROPOSAL_DISPOSITION')
        _check(type(self.request_id) is str and bool(self.request_id), Code.INVALID, 'REQUEST_ID')
        integer(self.source_epoch_index)
        for value in (self.source_epoch_binding_digest, self.observer_state_digest,
                      self.source_context_snapshot_digest, self.source_neural_state_digest,
                      self.target_model):
            digest_value(value)
        for value in (self.predecessor_observer_digest, self.previous_fast_application_digest):
            if value is not None:
                digest_value(value)
        _check(_is_digest(self.active_control_predecessor_digest), Code.INVALID, 'ACTIVE_CONTROL_PREDECESSOR')
        _check(self.steering_contract_digest == STEERING_CONTRACT_DIGEST and
               self.steering_profile_digest == STEERING_PROFILE_DIGEST, Code.IDENTITY, 'STEERING_PROFILE')
        _check(type(self.not_before_epoch) is int and type(self.expires_after_epoch) is int and
               self.not_before_epoch == self.source_epoch_index + 1 and
               self.expires_after_epoch == self.source_epoch_index + X_LATENT_TTL_EPOCHS,
               Code.INVALID, 'PROPOSAL_WINDOW')
        projection = (self.x_projection_digest, self.x_source_schema, self.target_width)
        bound = all(v is not None for v in projection)
        _check(bound or all(v is None for v in projection), Code.INVALID, 'PROPOSAL_PROJECTION')
        _check(not bound or (_is_digest(self.x_projection_digest) and type(self.x_source_schema) is str and
                             bool(self.x_source_schema) and type(self.target_width) is int and
                             self.target_width >= 1), Code.INVALID, 'PROPOSAL_PROJECTION')
        in_range = bound and WIDTH_MIN <= self.target_width <= WIDTH_MAX
        _check(self.width_transport_digest == (width_transport_digest(self.target_width) if in_range else None),
               Code.IDENTITY, 'WIDTH_TRANSPORT')
        if self.disposition is SteeringOutcome.PROPOSED:
            latent = self.latent
            _check(type(latent) is LatentInput and in_range and latent.channel == 'X' and
                   latent.source == self.observer_state_digest and
                   latent.source_schema == self.x_source_schema and
                   latent.projection == self.x_projection_digest and
                   latent.context_snapshot == self.source_context_snapshot_digest and
                   _float_vector(latent.values, self.target_width) and
                   self.latent_digest == latent.digest, Code.IDENTITY, 'PROPOSAL_LATENT')
            # Concrete specialization of the abstract active-control identity.
            _check(self.proposed_active_control_digest == self.latent_digest, Code.IDENTITY,
                   'PROPOSED_ACTIVE_CONTROL')
        else:
            _check(self.latent is None and self.latent_digest is None and
                   self.proposed_active_control_digest is None, Code.INVALID, 'NO_OP_PROPOSAL_LATENT')

    @property
    def digest(self):
        return _identity('fast-steering-proposal', self)


@dataclass(frozen=True)
class FastSteeringApplication(ProposalOnly):
    """Deterministic application outcome for one future invocation (APPLIED or
    NO_OP_*), bound to the explicit fast-control state it was evaluated against."""
    outcome: SteeringOutcome
    guard_reason: FastGuardReason | None
    proposal_digest: str
    observer_state_digest: str
    request_id: str
    source_epoch_index: int
    target_epoch_index: int
    control_state_digest: str
    active_control_before_digest: str
    active_control_after_digest: str
    hop_index: int | None
    target_state_digest: str
    input_request_digest: str
    output_request_digest: str
    applied_latent_digest: str | None
    predecessor_application_digest: str | None

    def __post_init__(self):
        _check(type(self.outcome) is SteeringOutcome and self.outcome is not SteeringOutcome.PROPOSED,
               Code.INVALID, 'APPLICATION_OUTCOME')
        _check((self.outcome is SteeringOutcome.NO_OP_FAST_GUARD) == (self.guard_reason is not None) and
               (self.guard_reason is None or type(self.guard_reason) is FastGuardReason),
               Code.INVALID, 'GUARD_REASON')
        for value in (self.proposal_digest, self.observer_state_digest, self.control_state_digest,
                      self.active_control_before_digest, self.active_control_after_digest,
                      self.target_state_digest, self.input_request_digest, self.output_request_digest):
            digest_value(value)
        if self.predecessor_application_digest is not None:
            digest_value(self.predecessor_application_digest)
        _check(type(self.request_id) is str and bool(self.request_id), Code.INVALID, 'REQUEST_ID')
        integer(self.source_epoch_index)
        integer(self.target_epoch_index)
        applied = self.outcome is SteeringOutcome.APPLIED
        _check(applied == (self.applied_latent_digest is not None) == (self.hop_index is not None),
               Code.INVALID, 'APPLICATION_LATENT')
        if applied:
            digest_value(self.applied_latent_digest)
            _check(type(self.hop_index) is int and 1 <= self.hop_index <= MAX_HOPS and
                   self.active_control_after_digest == self.applied_latent_digest and
                   self.active_control_after_digest != self.active_control_before_digest and
                   self.target_epoch_index > self.source_epoch_index and
                   self.output_request_digest != self.input_request_digest, Code.INVALID, 'APPLICATION_WINDOW')
        else:
            # Failure default NO_OP_RETAIN_ACTIVE_CONTROL: request and active control unchanged.
            _check(self.active_control_after_digest == self.active_control_before_digest and
                   self.output_request_digest == self.input_request_digest, Code.INVALID, 'NO_OP_STATE_CHANGED')

    @property
    def digest(self):
        return _identity('fast-steering-application', self)


def propose_fast_steering(observer, *, target_model, x_projection, control_state):
    """Derive at most one future-only X LatentInput from the DYN4 recurrence,
    bound to the host fast-control state active at derivation. The latent is
    independent of control_state; only provenance is bound. Eligibility order
    follows SOT_X_FAST_POLICY_R0; the stall/cycle/hop guard runs at application."""
    _check(type(observer) is ObserverState, Code.INVALID, 'OBSERVER_TYPE')
    _check(_is_digest(target_model), Code.INVALID, 'TARGET_MODEL')
    _check(type(control_state) is FastControlState, Code.INVALID, 'CONTROL_STATE_TYPE')
    _check(control_state.request_id == observer.request_id, Code.IDENTITY, 'CONTROL_STATE_REQUEST_MISMATCH')
    observer_digest = observer.digest
    projection_ok = _valid_x_projection(x_projection)
    x_digest = x_schema = width = transport = None
    if projection_ok:
        x_digest, x_schema, width = x_projection.digest, x_projection.source_schema, x_projection.weights.shape[0]
        if WIDTH_MIN <= width <= WIDTH_MAX:
            transport = width_transport_digest(width)
    latent = None
    if observer.core12[TERMINAL_COMMITTED_INDEX] != 1:
        disposition = SteeringOutcome.NO_OP_FAILED_SOURCE
    elif observer.reset_flag != 0:
        disposition = SteeringOutcome.NO_OP_RESET
    elif observer.dyn4 is None or observer.dyn4_delta is None:
        disposition = SteeringOutcome.NO_OP_DYN4_ABSENT
    elif target_model != observer.target_model:
        disposition = SteeringOutcome.NO_OP_TARGET_MISMATCH
    elif not projection_ok:
        disposition = SteeringOutcome.NO_OP_PROJECTION_MISMATCH
    elif transport is None:
        disposition = SteeringOutcome.NO_OP_INVALID_WIDTH
    else:
        values = _lift_steering(_canonical_steering4(_normalize_dyn4(observer.dyn4, observer.dyn4_delta)), width)
        if all(math.isfinite(v) for v in values):
            disposition = SteeringOutcome.PROPOSED
            latent = LatentInput('X', x_schema, observer_digest, observer.context_snapshot_digest, x_digest, values)
        else:
            disposition = SteeringOutcome.NO_OP_NONFINITE
    latent_digest = None if latent is None else latent.digest
    return FastSteeringProposal(
        disposition=disposition,
        request_id=observer.request_id,
        source_epoch_index=observer.epoch_index,
        source_epoch_binding_digest=observer.epoch_binding_digest,
        observer_state_digest=observer_digest,
        predecessor_observer_digest=observer.predecessor_observer_digest,
        source_context_snapshot_digest=observer.context_snapshot_digest,
        source_neural_state_digest=observer.source_neural_state_digest,
        target_model=target_model,
        x_projection_digest=x_digest,
        x_source_schema=x_schema,
        target_width=width,
        width_transport_digest=transport,
        latent=latent,
        latent_digest=latent_digest,
        proposed_active_control_digest=latent_digest,
        active_control_predecessor_digest=control_state.active_control_digest,
        previous_fast_application_digest=control_state.previous_fast_application_digest,
        steering_contract_digest=STEERING_CONTRACT_DIGEST,
        steering_profile_digest=STEERING_PROFILE_DIGEST,
        not_before_epoch=observer.epoch_index + 1,
        expires_after_epoch=observer.epoch_index + X_LATENT_TTL_EPOCHS,
    )


def _application_outcome(proposal, proposal_digest, request, target_epoch_index, target_state,
                         context_digest, neural_digest, x_projection, control_state):
    """(outcome, guard_reason): X-policy checks (guard steps 1-3 are TOO_EARLY /
    EXPIRED), then the frozen guard (steps 4-7), then APPLY (step 8)."""
    if proposal.disposition is not SteeringOutcome.PROPOSED:
        return proposal.disposition, None
    if target_epoch_index < proposal.not_before_epoch:  # same-epoch / not-yet-admissible
        return SteeringOutcome.NO_OP_TOO_EARLY, None
    if target_epoch_index > proposal.expires_after_epoch:
        return SteeringOutcome.NO_OP_EXPIRED, None
    if request.request_id != proposal.request_id:
        return SteeringOutcome.NO_OP_LINEAGE_MISMATCH, None
    if not (request.context_snapshot == context_digest == target_state.neural.context_snapshot ==
            proposal.latent.context_snapshot):
        return SteeringOutcome.NO_OP_CONTEXT_MISMATCH, None
    if target_state.neural.model != proposal.target_model:
        return SteeringOutcome.NO_OP_TARGET_MISMATCH, None
    source = proposal.source_neural_state_digest
    if source != neural_digest and not any(source in (step.input_state, step.output_state)
                                           for step in target_state.receipts):
        return SteeringOutcome.NO_OP_LINEAGE_MISMATCH, None  # target does not continue the source trajectory
    if not (_valid_x_projection(x_projection) and x_projection.digest == proposal.x_projection_digest and
            x_projection.source_schema == proposal.x_source_schema and
            x_projection.weights.shape[0] == proposal.target_width):
        return SteeringOutcome.NO_OP_PROJECTION_MISMATCH, None
    if control_state.active_control_digest == proposal.proposed_active_control_digest:
        return SteeringOutcome.NO_OP_DUPLICATE, None  # this exact packet is already the active control
    host_x = tuple(packet for packet in request.latents if packet.channel == 'X')
    if any(packet.digest == proposal.latent_digest for packet in host_x):
        return SteeringOutcome.NO_OP_DUPLICATE, None
    if host_x:
        return SteeringOutcome.NO_OP_HOST_X_COLLISION, None
    reason = _fast_guard(control_state, proposal.observer_state_digest,
                         proposal.active_control_predecessor_digest, proposal.proposed_active_control_digest)
    if reason is not None:
        return SteeringOutcome.NO_OP_FAST_GUARD, reason
    return SteeringOutcome.APPLIED, None


def apply_fast_steering(proposal, request, *, target_epoch_index, target_state, x_projection, control_state):
    """Attach the proposal's X LatentInput to a FUTURE request, or NO_OP.
    Returns (request_for_R3, application_record, control_state_out). Every NO_OP
    returns the host request and the incoming control state unchanged
    (NO_OP_RETAIN_ACTIVE_CONTROL); only APPLIED yields an advanced control state.
    target_state is the committed base of that invocation."""
    _check(type(proposal) is FastSteeringProposal, Code.INVALID, 'PROPOSAL_TYPE')
    _check(type(request) is InferenceRequest, Code.INVALID, 'REQUEST_TYPE')
    _check(all(type(packet) is LatentInput for packet in request.latents), Code.INVALID, 'REQUEST_LATENT_TYPE')
    _check(type(target_epoch_index) is int and target_epoch_index >= 0, Code.INVALID, 'TARGET_EPOCH_INDEX')
    _check(type(target_state) is DecodeState and type(target_state.neural) is NeuralState and
           type(target_state.context) is Snapshot and type(target_state.receipts) is tuple and
           all(type(step) is StepReceipt for step in target_state.receipts),
           Code.INVALID, 'TARGET_STATE_TYPE')
    _check(type(control_state) is FastControlState, Code.INVALID, 'CONTROL_STATE_TYPE')
    _check(control_state.request_id == proposal.request_id, Code.IDENTITY, 'CONTROL_STATE_REQUEST_MISMATCH')
    _check(control_state.last_applied_epoch_index is None or
           control_state.last_applied_epoch_index < target_epoch_index,
           Code.STALE, 'APPLICATION_PREDECESSOR_NOT_EARLIER')
    try:
        input_digest = request.digest
        state_digest = target_state.digest
        context_digest = target_state.context.digest
        neural_digest = target_state.neural.digest
    except CanonicalIdentityError:
        raise SteeringContractError(Code.IDENTITY, 'NONCANONICAL_IDENTITY') from None
    proposal_digest = proposal.digest
    outcome, guard_reason = _application_outcome(proposal, proposal_digest, request, target_epoch_index,
                                                 target_state, context_digest, neural_digest, x_projection,
                                                 control_state)
    applied = outcome is SteeringOutcome.APPLIED
    output = replace(request, latents=request.latents + (proposal.latent,)) if applied else request
    active_after = proposal.proposed_active_control_digest if applied else control_state.active_control_digest
    record = FastSteeringApplication(
        outcome=outcome,
        guard_reason=guard_reason,
        proposal_digest=proposal_digest,
        observer_state_digest=proposal.observer_state_digest,
        request_id=proposal.request_id,
        source_epoch_index=proposal.source_epoch_index,
        target_epoch_index=target_epoch_index,
        control_state_digest=control_state.digest,
        active_control_before_digest=control_state.active_control_digest,
        active_control_after_digest=active_after,
        hop_index=control_state.hop_count + 1 if applied else None,
        target_state_digest=state_digest,
        input_request_digest=input_digest,
        output_request_digest=output.digest,
        applied_latent_digest=proposal.latent_digest if applied else None,
        predecessor_application_digest=control_state.previous_fast_application_digest,
    )
    if not applied:
        return output, record, control_state
    return output, record, _advance_fast_control(control_state, proposal.observer_state_digest, active_after,
                                                 record.digest, target_epoch_index)


def global_event_fields(observer, proposal, application=None):
    """Values an ECS integration binds into a global steering event.
    Pure mapping: no ECS access."""
    _check(type(observer) is ObserverState and type(proposal) is FastSteeringProposal,
           Code.INVALID, 'GLOBAL_RECORD_TYPE')
    observer_digest = observer.digest
    _check(proposal.observer_state_digest == observer_digest, Code.IDENTITY, 'GLOBAL_PROPOSAL_OBSERVER_MISMATCH')
    proposal_digest = proposal.digest
    outcome_digest = None
    if application is not None:
        _check(type(application) is FastSteeringApplication and application.proposal_digest == proposal_digest,
               Code.IDENTITY, 'GLOBAL_APPLICATION_PROPOSAL_MISMATCH')
        outcome_digest = application.digest
    return dict(
        request_id=observer.request_id,
        source_epoch=observer.epoch_index,
        not_before_epoch=proposal.not_before_epoch,
        observer_digest=observer_digest,
        proposal_digest=proposal_digest,
        outcome_digest=outcome_digest,
        predecessor_observer_digest=observer.predecessor_observer_digest,
    )
