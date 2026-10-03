"""Steering qualification: observer, proposal, guard and application contracts plus
authority cross-checks. Native-backed tests run the real decode transaction over
the DSV4 compact-target fixture (see tests/inference/conftest.py).
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]

from elpis.identity import canonical_json_bytes, content_digest  # noqa: E402
from elpis.inference.context import (  # noqa: E402
    ContextItem, Lifetime, append_context, initial_snapshot, retire_context)
from elpis.inference.contracts import ContractError, ProposalOnly  # noqa: E402
from elpis.inference.drivers.dsv4.target import CompactTarget  # noqa: E402
from elpis.inference.target import LatentInput, LatentProjection, Tensor  # noqa: E402
from elpis.inference.structural import AddressProposal  # noqa: E402
from elpis.inference.transaction import InferenceRequest, InferenceEngine, DecodeResult  # noqa: E402
from elpis.inference.speculative import MarkovDrafter, run_speculative  # noqa: E402

import elpis.inference.steering as core  # noqa: E402
from elpis.inference.steering import (  # noqa: E402
    FastGuardReason, InferenceEpochBinding, SteeringContractError, FastControlState,
    FastSteeringApplication, FastSteeringProposal, ObserverState, apply_fast_steering,
    bind_completed_epoch, global_event_fields, initial_fast_control_state, observe_epoch,
    propose_fast_steering)
from elpis.inference.steering import SteeringOutcome as O  # noqa: E402

EXPECTED_CORE12 = (
    'epoch_committed_step_count', 'epoch_accepted_prefix_count', 'total_committed_token_count',
    'receipt_structural_count', 'epoch_latent_input_count', 'epoch_proposal_input_count',
    'context_canonical_count', 'context_visible_count', 'context_retired_count',
    'context_generation', 'speculative_draft_present', 'terminal_committed_flag')
EXPECTED_ORIENTATION = (
    (0.125, 0.125, 0.125, 0.125),
    (0.125, 0.125, 0.125, -0.125),
    (-0.125, -0.125, 0.125, 0.125),
    (0.125, 0.125, -0.125, -0.125),
    (-0.125, 0.125, -0.125, -0.125),
    (0.125, 0.125, -0.125, 0.125),
    (-0.125, 0.125, -0.125, 0.125),
    (0.125, -0.125, 0.125, 0.125),
)
AUTHORITY = ('semantic_authority', 'admission_authority', 'execution_authority',
             'mutation_authority', 'runtime_admission')
CORE_TREE = ast.parse(Path(core.__file__).read_text(encoding='utf-8'))
U4 = (0.3, -0.7, 0.11, 0.05)
R2 = 1.0 / math.sqrt(2.0)

# Recorded authority rule instances (frozen rule-instance digests), width -> (phase, digest).
AUTHORITY_INSTANCE_SCHEMAS = {
    '36': 'elpis.branch43.phase36.general-suffix-balanced-repeat-isometry.v0',
    '38': 'elpis.branch43.phase38.boundary-rule-instance.v0',
    '40': 'elpis.branch43.phase40.gap-rule-instance.v0',
    '42': 'elpis.branch43.phase42.extension-rule-instance.v0',
}
AUTHORITY_INSTANCE_DIGESTS = {
    5: ('38', '683602fea2a5bd84c9b7b4c9edeb69e363709aefb7fd86320f7b43b96a8334ff'),
    6: ('38', '98f7b3ca878e51fb33149864ef7e1a6e05a3b9149689f37bdf5886a439b3e0d2'),
    7: ('38', '7d202d59c37ea032a6b175d26a51062e40b7b3d2257bbf2d4a2e5b24bf2b1eb6'),
    9: ('36', '7b3bfa7c27d085f948746161d2d4fc7976e538869871df66c3662c90d90a3890'),
    10: ('36', 'a795364d6d4faf320fb2b556e79a6ae0d8c9e85eed27fe39803773fabade1ff0'),
    11: ('36', '57106faf059df8d850322fe9f0d38b824d9dafb34c77ba715cf499b1ec0b6457'),
    13: ('36', 'd147ae3e791db9141b0c39360842c2ebc53814bb675e9f361e93386f79cc260e'),
    14: ('36', '84869252ab6287a6fae1170b58b4a2ea65e801f81e75d6769e8a17fade45efab'),
    15: ('36', '184dd48e51fc765720120cc53c61686c591f93cf44fe4eb8b1f3bc4b79402049'),
    17: ('40', 'f849e31ee0655c4941895f89545fd81204733fa4e08ec5fa3e59434d06956d60'),
    18: ('40', '1cdf1e39a55e6da996261c1c068250de5d0f6e024a1d1dcd6ada9af7e446feda'),
    19: ('40', '9478eb15664129c9e021c8a28ff025c9c63f95221c3a8af33a1e7a22a136380e'),
    21: ('40', '83c8e9da9f7246a91dfeb70ebaad4967f89c6e37a34f2fd6f74ffdac698449ba'),
    22: ('40', '04b1e1c1e7a064d6d35635b9e17cd6503fb00b4da78ddc690703a102781b83aa'),
    23: ('40', 'c6ab4de7a2efb404732eca6e75c6b9635ad432be90f72e18c6f33b9cee075ffa'),
    25: ('42', '1e86a3a1252f29e4330bc49c19da234c3178eb5b4c6fbc0525054a3b4e02c8e3'),
    26: ('42', 'eb993890d137ad156b14d50b8c5914ff1f053810334c81d67d20c26ca34b2278'),
    27: ('42', '2899ecf1620f2600b0505aec35eda62f38d9967a7c9e28149c7db84eb490d7d4'),
    28: ('42', 'b435958b4b91795868e3db192f08e6de1b81b5f85dd1b02930da6145b981e47f'),
    29: ('42', 'b512afa951b531867e65287ada6f203e15134fb652a346b2336e21d36869fa37'),
    30: ('42', '96d395af564ee25a30708547254d01261f6b6a47feaed4c63d7278cc39f1aa14'),
    31: ('42', '83e33f2037a933d8cdbe519063a97ba848d47ea5990101fe388294df2c9a3ce5'),
    32: ('42', '974caee5497b32e99eaf20a6c7d3fe1eab73966d63adc9cfd0d55c665a1bb14d'),
}


# ------------------------------------------------------------ fixtures/helpers
@pytest.fixture
def make_rt(tmp_path, native_workspace, fms_file_library):
    from elpis.inference.drivers.dsv4.fixtures import make_fixture
    from elpis.substrate.synthetic import SyntheticFileAssets
    provider = SyntheticFileAssets(root=native_workspace, library=fms_file_library,
                                   warm_bytes=64, staging_bytes=128)

    def build(dimension=4):
        target, _, _ = make_fixture(provider, tmp_path / ('target%d' % dimension), dimension=dimension)
        return InferenceEngine(target)

    yield build
    provider.close()


@pytest.fixture
def rt(make_rt):
    return make_rt()


class Lane:
    """Host-owned epoch lane; every steering value is passed explicitly."""

    def __init__(self):
        self.binding = None
        self.observer = None
        self.observers = []
        self.index = -1

    def observe(self, request, result):
        self.index += 1
        binding = bind_completed_epoch(request, result, epoch_index=self.index, predecessor=self.binding)
        observer = observe_epoch(binding, request, result, predecessor=self.observer)
        self.binding, self.observer = binding, observer
        self.observers.append(observer)
        return binding, observer


def run(rt, state, request, expected=None):
    return rt.execute(state, request, expected_state=state.digest if expected is None else expected)


def prefill(request_id, ctx, tokens=(1, 2, 3), **kwargs):
    return InferenceRequest(request_id, ctx.digest, 'PREFILL', tokens, **kwargs)


def greedy(request_id, ctx, count=2, **kwargs):
    return InferenceRequest(request_id, ctx.digest, 'GREEDY', count=count, **kwargs)


def session(rt):
    """Epoch 0 prefill, epoch 1 greedy(2), proposal derived from epoch 1."""
    ctx = initial_snapshot()
    lane = Lane()
    s0 = rt.initial(ctx)
    r0 = prefill('req-A', ctx)
    res0 = run(rt, s0, r0)
    lane.observe(r0, res0)
    r1 = greedy('req-A', ctx)
    res1 = run(rt, res0.state, r1)
    lane.observe(r1, res1)
    xp = rt.target.projections['X']
    control = initial_fast_control_state('req-A')
    proposal = propose_fast_steering(lane.observer, target_model=rt.target.model_identity, x_projection=xp,
                                     control_state=control)
    return dict(ctx=ctx, lane=lane, s0=s0, r0=r0, res0=res0, r1=r1, res1=res1, xp=xp, proposal=proposal,
                control=control)


def apply_next(s, request, epoch_offset=1, **kwargs):
    p = s['proposal']
    kwargs.setdefault('target_state', s['res1'].state)
    kwargs.setdefault('x_projection', s['xp'])
    kwargs.setdefault('control_state', s['control'])
    return apply_fast_steering(p, request, target_epoch_index=p.source_epoch_index + epoch_offset, **kwargs)


def genesis(request_id='req-A'):
    return initial_fast_control_state(request_id)


def D(label):
    return hashlib.sha256(label.encode()).hexdigest()


def advanced_state(controls, request_id='req-A'):
    """Control state after successful APPLYs of the given controls (pure guard kernels)."""
    state = initial_fast_control_state(request_id)
    for k, control in enumerate(controls):
        state = core._advance_fast_control(state, D('observer-%d' % k), control, D('application-%d' % k), 0)
    return state


def synthetic_projection(width, channel='X', schema='executive-latent.r0'):
    data = (np.random.default_rng(width).normal(size=(width, 4)) * 0.1).astype('<f4')
    return LatentProjection(channel, schema, 'fixture-model', Tensor((width, 4), data.tobytes()))


def synthetic_observer(**overrides):
    values = dict(
        epoch_binding_digest='1' * 64, request_id='req-S', epoch_index=5,
        context_snapshot_digest='2' * 64, target_model='3' * 64, source_neural_state_digest='4' * 64,
        core12=(1, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1), core12_delta=(0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        reset_flag=0, dyn4=(1.0, 3.0, 0.0, 0.5), dyn4_delta=(0.0, -1.0, 2.0, 0.0),
        predecessor_observer_digest='5' * 64, terminal_status='COMMITTED', failure_code=None)
    values.update(overrides)
    return ObserverState(**values)


def make_drafter(rt):
    rng = np.random.default_rng(61)

    def tensor(shape):
        return Tensor(shape, (rng.normal(size=shape) * 0.1).astype('<f4').tobytes())

    return MarkovDrafter(target_model=rt.target.model_identity, hidden_projection=tensor((4, 16)),
                         token_embedding=tensor((16, 2)), markov_projection=tensor((2, 16)),
                         confidence_head=tensor((4, 1)))


def flatten(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from flatten(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten(item)
    else:
        yield value


# ------------------------------------------------------------ 1-5 CORE12 + recurrence
def test_01_core12_ordering_and_exact_counts(rt):
    assert core.CORE12_FEATURES == EXPECTED_CORE12
    base = initial_snapshot()
    ctx = append_context(base, ContextItem('stable', 'host', b's', Lifetime.STABLE), expected=base.digest)
    ctx = append_context(ctx, ContextItem('scratch', 'host', b'e', Lifetime.EPHEMERAL), expected=ctx.digest)
    ctx = retire_context(ctx, 'scratch', expected=ctx.digest, reason='qualification')
    gp = rt.target.projections['G']
    g = LatentInput('G', gp.source_schema, 'a' * 64, ctx.digest, gp.digest, (0.01, -0.02, 0.03, 0.0))
    address = AddressProposal('b' * 64, 'c' * 64, 'd' * 64, ctx.digest, 'e' * 64, 'route', (), (), (), None, ())
    lane = Lane()
    r0 = prefill('req-A', ctx, proposals=(address,), latents=(g,))
    res0 = run(rt, rt.initial(ctx), r0)
    _, o0 = lane.observe(r0, res0)
    assert o0.core12 == (3, 3, 3, 1, 3, 3, 2, 1, 1, 3, 0, 1)
    r1 = greedy('req-A', ctx)
    res1 = run(rt, res0.state, r1)
    _, o1 = lane.observe(r1, res1)
    assert o1.core12 == (2, 2, 5, 0, 0, 0, 2, 1, 1, 3, 0, 1)  # suffix-aligned: prefill inputs excluded
    r2 = greedy('req-A', ctx)
    res2 = run_speculative(rt, res1.state, r2, make_drafter(rt), expected_state=res1.state.digest, block_size=2)
    assert res2.receipt.terminal == 'COMMITTED' and res2.receipt.draft is not None
    _, o2 = lane.observe(r2, res2)
    assert o2.core12 == (2, 2, 7, 0, 0, 0, 2, 1, 1, 3, 1, 1)
    named = dict(zip(core.CORE12_FEATURES, o2.core12))
    assert named['speculative_draft_present'] == 1 and named['terminal_committed_flag'] == 1


def test_02_to_05_recurrence_reset_delta_request_and_context_lineage(rt):
    ctx = initial_snapshot()
    lane = Lane()
    r0 = prefill('req-A', ctx)
    res0 = run(rt, rt.initial(ctx), r0)
    _, o0 = lane.observe(r0, res0)
    assert o0.reset_flag == 1 and o0.predecessor_observer_digest is None                        # 2
    assert o0.core12_delta == (0,) * 12 and o0.dyn4_delta == (0.0,) * 4
    r1 = greedy('req-A', ctx)
    res1 = run(rt, res0.state, r1)
    _, o1 = lane.observe(r1, res1)
    assert o1.reset_flag == 0 and o1.predecessor_observer_digest == o0.digest                   # 3
    assert o1.core12_delta == tuple(a - b for a, b in zip(o1.core12, o0.core12)) != (0,) * 12
    r2 = greedy('req-B', ctx)
    res2 = run(rt, res1.state, r2)
    _, o2 = lane.observe(r2, res2)
    assert o2.reset_flag == 1 and o2.predecessor_observer_digest == o1.digest                   # 4
    assert o2.core12_delta == (0,) * 12 and o2.dyn4_delta == (0.0,) * 4
    ctx2 = append_context(ctx, ContextItem('late', 'host', b'x', Lifetime.DYNAMIC), expected=ctx.digest)
    r3 = prefill('req-B', ctx2)
    res3 = run(rt, rt.initial(ctx2), r3)
    _, o3 = lane.observe(r3, res3)
    assert o3.reset_flag == 1 and o3.predecessor_observer_digest == o2.digest                   # 5
    assert o3.context_snapshot_digest == ctx2.digest != o2.context_snapshot_digest
    assert o3.core12_delta == (0,) * 12 and o3.dyn4_delta == (0.0,) * 4
    r4 = greedy('req-B', ctx2)
    _, o4 = lane.observe(r4, run(rt, res3.state, r4))
    assert o4.reset_flag == 0


# ------------------------------------------------------------ 6-8 DYN4
def test_06_dyn4_exact_formulas_and_validation():
    assert core.DYN4_FEATURES == ('hidden_mean_square', 'logits_mean_square', 'logit_span', 'top1_top2_margin')
    assert core._dyn4((1.0, -2.0, 3.0), (0.5, -1.5, 2.0, 1.0)) == (14.0 / 3.0, 1.875, 3.5, 1.0)
    assert core._dyn4((0.0,), (2.0, 2.0, -1.0)) == (0.0, 3.0, 3.0, 0.0)
    for hidden, logits in (((), (1.0, 2.0)), ((1.0,), (1.0,)), ((math.nan,), (1.0, 2.0)),
                           ((1.0,), (math.inf, 0.0)), ((1e200,), (0.0, 1.0)), ([1.0], (0.0, 1.0)),
                           ((True,), (0.0, 1.0))):
        with pytest.raises(SteeringContractError):
            core._dyn4(hidden, logits)


def test_07_08_dyn4_on_committed_state_raw_vectors_absent_and_delta(rt):
    s = session(rt)
    o0, o1 = s['lane'].observers
    neural = s['res1'].state.neural
    assert o1.dyn4 == core._dyn4(neural.hidden, neural.logits)
    assert all(v >= 0.0 for v in o1.dyn4)
    assert o1.dyn4_delta == tuple(a - b for a, b in zip(o1.dyn4, o0.dyn4))                       # 8
    names = {f.name for f in dataclasses.fields(ObserverState)}
    assert not names & {'hidden', 'logits', 'tokens', 'token', 'argmax'}                          # 7
    leaves = list(flatten(dataclasses.asdict(o1)))
    assert sum(type(v) is float for v in leaves) == 8   # DYN4 current + delta only
    assert sum(type(v) is int for v in leaves) == 26    # epoch index, CORE12 q+delta, reset flag
    values = dataclasses.astuple(o1)
    assert neural.hidden not in values and neural.logits not in values and neural.tokens not in values


# ------------------------------------------------------------ 9-10 steering object
def test_09_frozen_orientation_matrix_exact_use():
    assert core.ORIENTATION_8X4 == EXPECTED_ORIENTATION
    assert core.frozen_steering_contract()['dyn4_orientation_matrix'] == [list(r) for r in EXPECTED_ORIENTATION]
    assert core.frozen_steering_contract_digest() == core.STEERING_CONTRACT_DIGEST == \
        'ac72e897a7109aaa922b0ea9d599eedb626d93509dc7a496e300e593d92c0955'
    assert core.ORIENTATION_DIGEST == '8cfa6780404d92df6d13cccc84f220afe176b94914472df77b3c3f7cb727dd18'
    assert (core.STEERING_OBJECT_LABEL, core.STEERING_ADAPTER_DIGEST) == (
        'BB33_A4_D100', '5ecf5a9eec6f3757b5aaef9ac9597761b221ab8c0d79e146915a02dd4bbb14ab')
    for i in range(8):
        basis = tuple(1.0 if k == i else 0.0 for k in range(8))
        assert core._canonical_steering4(basis) == tuple(0.09 * v for v in EXPECTED_ORIENTATION[i])


def test_10_fixed_gain_exact():
    assert core.FIXED_GAIN == 0.09
    column_sums = tuple(sum(row[j] for row in EXPECTED_ORIENTATION) for j in range(4))
    assert column_sums == (0.25, 0.5, 0.0, 0.25)
    assert core._canonical_steering4((1.0,) * 8) == tuple(0.09 * c for c in column_sums)
    r8 = core._normalize_dyn4((1.0, 3.0, 0.0, 0.5), (0.0, -1.0, 2.0, 0.0))
    assert r8 == (0.5, 0.75, 0.0, 0.5 / 1.5, 0.0, -0.5, 2.0 / 3.0, 0.0)
    reference = 0.09 * (np.asarray(r8) @ np.asarray(EXPECTED_ORIENTATION))
    assert np.max(np.abs(np.asarray(core._canonical_steering4(r8)) - reference)) <= 1e-17


# ------------------------------------------------------------ 11-19 width transport
@pytest.mark.parametrize('width,assignments', [
    (4, (0, 1, 2, 3)), (5, (0, 1, 2, 3, 3)), (6, (0, 1, 2, 3, 2, 3)),
    (7, (0, 1, 2, 3, 1, 2, 3)), (8, (0, 1, 2, 3, 0, 1, 2, 3))])
def test_11_to_15_width_assignments(width, assignments):
    assert core._width_assignments(width) == assignments


def test_11_width4_identity_transport():
    assert core._lift_steering(U4, 4) == U4
    assert core.width_transport_matrix(4) == tuple(tuple(1.0 if i == j else 0.0 for j in range(4)) for i in range(4))


def test_12_width5_suffix_transport():
    a, b, c, d = U4
    assert core._lift_steering(U4, 5) == (a, b, c, R2 * d, R2 * d)


def test_13_width6_duplicates_2_and_3_not_0_and_1():
    a, b, c, d = U4
    assert core._width_assignments(6)[4:] == (2, 3) != (0, 1)
    assert core._lift_steering(U4, 6) == (a, b, R2 * c, R2 * d, R2 * c, R2 * d)
    negative_predecessor = ((R2, 0.0, 0.0, 0.0), (0.0, R2, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0),
                            (0.0, 0.0, 0.0, 1.0), (R2, 0.0, 0.0, 0.0), (0.0, R2, 0.0, 0.0))
    assert core.width_transport_matrix(6) != negative_predecessor


def test_14_width7_suffix_transport():
    a, b, c, d = U4
    assert core._lift_steering(U4, 7) == (a, R2 * b, R2 * c, R2 * d, R2 * b, R2 * c, R2 * d)


def test_15_width8_balanced_repeat_transport():
    a, b, c, d = U4
    assert core._lift_steering(U4, 8) == (R2 * a, R2 * b, R2 * c, R2 * d) * 2


def test_16_l2_preserved_every_width_4_to_32():
    rng = np.random.default_rng(717)
    eps = sys.float_info.epsilon
    for width in range(4, 33):
        u4 = tuple(float(v) for v in rng.normal(size=4))
        ud = core._lift_steering(u4, width)
        n4 = math.sqrt(math.fsum(v * v for v in u4))
        nd = math.sqrt(math.fsum(v * v for v in ud))
        assert len(ud) == width and abs(nd - n4) <= 4 * eps * n4
        matrix = np.asarray(core.width_transport_matrix(width))
        assert matrix.shape == (width, 4) and all(np.count_nonzero(row) == 1 for row in matrix)
        assert np.max(np.abs(matrix.T @ matrix - np.eye(4))) <= 4 * eps


def test_17_repeated_identical_input_is_deterministic():
    first = propose_fast_steering(synthetic_observer(), target_model='3' * 64, x_projection=synthetic_projection(13), control_state=genesis('req-S'))
    second = propose_fast_steering(synthetic_observer(), target_model='3' * 64, x_projection=synthetic_projection(13), control_state=genesis('req-S'))
    assert first.disposition is O.PROPOSED
    assert first == second and first.digest == second.digest and first.latent.values == second.latent.values
    assert core._lift_steering(U4, 29) == core._lift_steering(U4, 29)
    assert core.width_transport_digest(29) == core.width_transport_digest(29)


@pytest.mark.parametrize('width', [3, 33])
def test_18_19_out_of_range_width_rejected_and_noop(width):
    with pytest.raises(SteeringContractError) as err:
        core._width_assignments(width)
    assert err.value.reason == 'WIDTH_OUTSIDE_QUALIFIED_4_32'
    with pytest.raises(SteeringContractError):
        core._lift_steering(U4, width)
    with pytest.raises(SteeringContractError):
        core.width_transport_digest(width)
    proposal = propose_fast_steering(synthetic_observer(), target_model='3' * 64,
                                     x_projection=synthetic_projection(width), control_state=genesis('req-S'))
    assert proposal.disposition is O.NO_OP_INVALID_WIDTH
    assert proposal.latent is None and proposal.latent_digest is None
    assert proposal.target_width == width and proposal.width_transport_digest is None


def test_width_rule_reproduces_recorded_authority_instances():
    assert core.WIDTH_RULE_AUTHORITY_DIGEST == 'daffd7eecd6505bfe8c2cbef292ca3a34b0b51d34d85726c1b2f7d72cb5fa4a7'
    for width, (phase, digest) in sorted(AUTHORITY_INSTANCE_DIGESTS.items()):
        assignments = list(core._width_assignments(width))
        matrix = [list(row) for row in core.width_transport_matrix(width)]
        q, r = divmod(width, 4)
        gram = 0.0
        for a in range(4):
            for b in range(4):
                acc = 0.0
                for row in matrix:
                    acc += row[a] * row[b]
                gram = max(gram, abs(acc - (1.0 if a == b else 0.0)))
        record = dict(
            counts_by_source_coordinate=[assignments.count(k) for k in range(4)],
            extra_copy_source_coordinates=list(range(4 - r, 4)),
            gram_max_abs_error_from_identity=gram, l2_norm_preserving=True, learned=False,
            matrix_target_by_source=matrix, rule_label=core.WIDTH_RULE_LABEL,
            schema=AUTHORITY_INSTANCE_SCHEMAS[phase], scheme_dependent=False,
            target_data_dependent=False, target_row_source_assignments=assignments,
            target_seed_dependent=False)
        if phase == '36':
            record.update(one_sparse_rows=True, quotient_q=q, remainder_r=r, source_dimension=4,
                          target_width=width)
        else:
            record.update(q=q, r=r, width=width)
        assert hashlib.sha256(canonical_json_bytes(record)).hexdigest() == digest, width


# ------------------------------------------------------------ 20-25 eligibility
def test_20_reset_state_cannot_steer(rt):
    s = session(rt)
    o0 = s['lane'].observers[0]
    assert o0.reset_flag == 1
    p0 = propose_fast_steering(o0, target_model=rt.target.model_identity, x_projection=s['xp'],
                               control_state=s['control'])
    assert p0.disposition is O.NO_OP_RESET and p0.latent is None
    request = greedy('req-A', s['ctx'])
    out, application, control = apply_fast_steering(p0, request, target_epoch_index=1,
                                                    target_state=s['res0'].state, x_projection=s['xp'],
                                                    control_state=s['control'])
    assert application.outcome is O.NO_OP_RESET and out is request and application.applied_latent_digest is None
    assert control is s['control']
    unit = propose_fast_steering(
        synthetic_observer(reset_flag=1, core12_delta=(0,) * 12, dyn4_delta=(0.0,) * 4),
        target_model='3' * 64, x_projection=synthetic_projection(4), control_state=genesis('req-S'))
    assert unit.disposition is O.NO_OP_RESET


def test_21_failed_source_receipt_cannot_steer(rt):
    s = session(rt)
    request = greedy('req-A', s['ctx'], count=1)
    failed = run(rt, s['res1'].state, request, expected='0' * 64)
    assert failed.receipt.terminal == 'FAILED' and failed.receipt.failure == 'STALE'
    assert failed.state == s['res1'].state and failed.receipt.target_steps == ()
    _, observer = s['lane'].observe(request, failed)
    assert observer.reset_flag == 0 and observer.terminal_status == 'FAILED' and observer.failure_code == 'STALE'
    assert observer.core12[:2] == (0, 0) and observer.core12[11] == 0
    assert observer.dyn4_delta == (0.0,) * 4  # genuine zero change of the preserved state
    proposal = propose_fast_steering(observer, target_model=rt.target.model_identity, x_projection=s['xp'],
                                     control_state=s['control'])
    assert proposal.disposition is O.NO_OP_FAILED_SOURCE and proposal.latent is None
    future = greedy('req-A', s['ctx'])
    out, application, control = apply_fast_steering(proposal, future, target_epoch_index=observer.epoch_index + 1,
                                                    target_state=failed.state, x_projection=s['xp'],
                                                    control_state=s['control'])
    assert application.outcome is O.NO_OP_FAILED_SOURCE and out is future and control is s['control']


def test_22_23_future_only_window_same_epoch_and_expiry(rt):
    s = session(rt)
    p = s['proposal']
    n = p.source_epoch_index
    assert p.disposition is O.PROPOSED and (p.not_before_epoch, p.expires_after_epoch) == (n + 1, n + 2)
    request = greedy('req-A', s['ctx'])
    for epoch, expected in ((n - 1, O.NO_OP_TOO_EARLY), (n, O.NO_OP_TOO_EARLY), (n + 1, O.APPLIED),
                            (n + 2, O.APPLIED), (n + 3, O.NO_OP_EXPIRED), (n + 9, O.NO_OP_EXPIRED)):
        out, application, _ = apply_fast_steering(p, request, target_epoch_index=epoch,
                                                  target_state=s['res1'].state, x_projection=s['xp'],
                                                  control_state=s['control'])
        assert application.outcome is expected
        assert (out is request) == (expected is not O.APPLIED)


def test_24_existing_host_x_latent_wins(rt):
    s = session(rt)
    xp, ctx = s['xp'], s['ctx']
    host_x = LatentInput('X', xp.source_schema, '9' * 64, ctx.digest, xp.digest, (0.02, 0.0, -0.01, 0.0))
    request = greedy('req-A', ctx, latents=(host_x,))
    out, application, retained = apply_next(s, request)
    assert application.outcome is O.NO_OP_HOST_X_COLLISION and out is request
    assert retained is s['control'] and retained.hop_count == 0  # host-X collision never advances hops
    assert application.applied_latent_digest is None
    result = run(rt, s['res1'].state, out)
    assert result.receipt.terminal == 'COMMITTED'
    assert all(step.latents[1:] == (host_x.digest,) for step in result.receipt.target_steps)


def test_25_host_g_and_r_latents_coexist(rt):
    s = session(rt)
    ctx, p = s['ctx'], s['proposal']
    gp, rp = rt.target.projections['G'], rt.target.projections['R']
    g = LatentInput('G', gp.source_schema, '6' * 64, ctx.digest, gp.digest, (0.01, 0.02, -0.01, 0.0))
    r = LatentInput('R', rp.source_schema, '7' * 64, ctx.digest, rp.digest, (0.1, 0.0, 0.0, -0.1))
    request = greedy('req-A', ctx, latents=(g, r))
    out, application, _ = apply_next(s, request)
    assert application.outcome is O.APPLIED and out.latents == (g, r, p.latent)
    assert (out.request_id, out.context_snapshot, out.mode, out.count) == (
        request.request_id, request.context_snapshot, request.mode, request.count)
    result = run(rt, s['res1'].state, out)
    assert result.receipt.terminal == 'COMMITTED'
    assert all(step.latents[1:] == (g.digest, r.digest, p.latent_digest) for step in result.receipt.target_steps)


# ------------------------------------------------------------ 26-30 packet + provenance
def test_26_to_29_packet_is_existing_x_latent_with_exact_bindings(rt):
    s = session(rt)
    p, xp, observer = s['proposal'], s['xp'], s['lane'].observer
    latent = p.latent
    assert type(latent) is LatentInput                                                   # 26
    assert latent.channel == 'X'                                                         # 27
    assert latent.source == observer.digest == p.observer_state_digest                   # 28
    assert latent.source_schema == xp.source_schema == p.x_source_schema                 # 29
    assert latent.projection == xp.digest == p.x_projection_digest
    assert latent.context_snapshot == s['ctx'].digest == s['res1'].state.context.digest
    assert len(latent.values) == xp.weights.shape[0] == p.target_width == 4
    u4 = core._canonical_steering4(core._normalize_dyn4(observer.dyn4, observer.dyn4_delta))
    assert latent.values == core._lift_steering(u4, 4) == u4
    assert p.latent_digest == latent.digest and p.width_transport_digest == core.width_transport_digest(4)
    for channel in ('M', 'G', 'R'):
        forged = dataclasses.replace(latent, channel=channel)
        with pytest.raises(SteeringContractError):
            dataclasses.replace(p, latent=forged, latent_digest=forged.digest)


def test_30_applied_steering_latent_recorded_in_later_step_receipts(rt):
    s = session(rt)
    p = s['proposal']
    request = greedy('req-A', s['ctx'])
    steered, application, _ = apply_next(s, request)
    assert application.outcome is O.APPLIED and application.applied_latent_digest == p.latent_digest
    result = run(rt, s['res1'].state, steered)
    assert result.receipt.terminal == 'COMMITTED' and len(result.receipt.target_steps) == 2
    assert result.receipt.request == steered.digest == application.output_request_digest
    assert all(step.latents[-1] == p.latent_digest for step in result.receipt.target_steps)
    control = run(rt, s['res1'].state, request)
    assert all(p.latent_digest not in step.latents for step in control.receipt.target_steps)
    assert control.receipt.digest != result.receipt.digest
    binding, observer = s['lane'].observe(steered, result)
    assert binding.runtime_receipt_digest == result.receipt.digest and observer.core12[4] == 2


def test_30b_width6_transport_through_real_runtime(make_rt):
    rt6 = make_rt(dimension=6)
    s = session(rt6)
    p, observer = s['proposal'], s['lane'].observer
    assert p.disposition is O.PROPOSED and p.target_width == 6
    a, b, c, d = core._canonical_steering4(core._normalize_dyn4(observer.dyn4, observer.dyn4_delta))
    assert p.latent.values == (a, b, R2 * c, R2 * d, R2 * c, R2 * d)
    steered, application, control = apply_next(s, greedy('req-A', s['ctx']))
    result = run(rt6, s['res1'].state, steered)
    assert application.outcome is O.APPLIED and result.receipt.terminal == 'COMMITTED'
    assert all(step.latents[-1] == p.latent_digest for step in result.receipt.target_steps)


# ------------------------------------------------------------ 31-34 replay, authority, isolation
def test_31_replay_and_reconstruction_identical(rt):
    def pipeline(runtime):
        s = session(runtime)
        steered, application, control = apply_next(s, greedy('req-A', s['ctx']))
        result = run(runtime, s['res1'].state, steered)
        assert runtime.replay(s['res1'].state, steered, result.receipt).receipt == result.receipt
        lane = s['lane']
        rebuilt = observe_epoch(lane.binding, s['r1'], s['res1'], predecessor=lane.observers[0])
        assert rebuilt == lane.observer
        return (tuple(o.digest for o in lane.observers), lane.binding.digest, s['proposal'].digest,
                application.digest, result.receipt.digest, control.digest)

    assert pipeline(rt) == pipeline(InferenceEngine(rt.target))


def test_32_every_steering_record_is_authority_zero(rt):
    s = session(rt)
    _, application, control = apply_next(s, greedy('req-A', s['ctx']))
    for record in (s['lane'].binding, s['lane'].observer, s['proposal'], application, s['proposal'].latent,
                   s['control'], control):
        assert isinstance(record, ProposalOnly)
        for flag in AUTHORITY:
            assert getattr(record, flag) is False
            with pytest.raises(AttributeError):
                setattr(record, flag, True)
    for cls in (InferenceEpochBinding, ObserverState, FastSteeringProposal, FastSteeringApplication,
                FastControlState):
        assert issubclass(cls, ProposalOnly)
        assert not {f.name for f in dataclasses.fields(cls)} & set(AUTHORITY)


def test_33_core_has_no_ecs_or_evolution_path():
    imports = set()
    for node in ast.walk(CORE_TREE):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)
    assert imports == {'__future__', 'dataclasses', 'enum', 'math', 'elpis.identity',
                       'elpis.inference.context', 'elpis.inference.contracts', 'elpis.inference.target',
                       'elpis.inference.transaction'}
    assert not any(tag in name.lower() for name in imports for tag in ('ecs', 'rrsi', 'darwin', 'fprm', 'anchor'))
    referenced = {n.id for n in ast.walk(CORE_TREE) if isinstance(n, ast.Name)}
    referenced |= {n.attr for n in ast.walk(CORE_TREE) if isinstance(n, ast.Attribute)}
    assert not referenced & {'EntityPort', 'Kernel', 'ECSContextProjector', 'Branch43GlobalEvent', 'send'}
    modules = [v for v in vars(core).values() if isinstance(v, type(sys))]
    assert not any(m.__name__.startswith(('elpis_ecs', 'ECS')) for m in modules)


def test_34_core_never_executes_model_or_runtime(rt, monkeypatch):
    s = session(rt)  # Engine work completes before the guard is installed
    calls = [node.func for node in ast.walk(CORE_TREE) if isinstance(node, ast.Call)]
    called = {f.attr for f in calls if isinstance(f, ast.Attribute)} | {f.id for f in calls if isinstance(f, ast.Name)}
    assert not called & {'step', 'execute', 'advance', 'replay', '_advance_validated', 'run_speculative',
                         'verify_draft', 'propose', 'send', 'commit'}

    def forbidden(*args, **kwargs):
        raise AssertionError('SoT invoked the model or runtime')

    for owner, name in ((CompactTarget, 'step'), (InferenceEngine, 'execute'), (InferenceEngine, 'advance'),
                        (InferenceEngine, 'replay'), (InferenceEngine, '_advance_validated'), (MarkovDrafter, 'propose')):
        monkeypatch.setattr(owner, name, forbidden)
    lane = Lane()
    lane.observe(s['r0'], s['res0'])
    lane.observe(s['r1'], s['res1'])
    proposal = propose_fast_steering(lane.observer, target_model=rt.target.model_identity, x_projection=s['xp'],
                                     control_state=genesis())
    _, application, control = apply_fast_steering(proposal, greedy('req-A', s['ctx']), target_epoch_index=2,
                                                  target_state=s['res1'].state, x_projection=s['xp'],
                                                  control_state=genesis())
    assert proposal == s['proposal'] and application.outcome is O.APPLIED and control.hop_count == 1
    _, guarded, retained = apply_fast_steering(proposal, greedy('req-A', s['ctx']), target_epoch_index=2,
                                               target_state=s['res1'].state, x_projection=s['xp'],
                                               control_state=advanced_state([D('other-control')]))
    assert guarded.guard_reason is FastGuardReason.ACTIVE_CONTROL_PREDECESSOR_MISMATCH and retained.hop_count == 1


# ------------------------------------------------------------ additional contract coverage
def test_epoch_binding_contract_fields_and_lineage_fail_closed(rt):
    s = session(rt)
    r0, res0, r1, res1 = s['r0'], s['res0'], s['r1'], s['res1']
    assert tuple(f.name for f in dataclasses.fields(InferenceEpochBinding)) == (
        'epoch_index', 'predecessor_epoch_binding_digest', 'runtime_receipt_digest', 'runtime_request_digest',
        'input_decode_state_digest', 'output_decode_state_digest', 'context_snapshot_digest', 'terminal_status')
    b0 = bind_completed_epoch(r0, res0, epoch_index=0)
    assert (b0.runtime_receipt_digest, b0.runtime_request_digest, b0.input_decode_state_digest,
            b0.output_decode_state_digest, b0.context_snapshot_digest, b0.terminal_status,
            b0.predecessor_epoch_binding_digest) == (
        res0.receipt.digest, r0.digest, s['s0'].digest, res0.state.digest, s['ctx'].digest, 'COMMITTED', None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        b0.epoch_index = 7
    b1 = bind_completed_epoch(r1, res1, epoch_index=1, predecessor=b0)
    assert b1.predecessor_epoch_binding_digest == b0.digest == s['lane'].observers[0].epoch_binding_digest
    assert b1.digest == s['lane'].binding.digest
    receipt = res1.receipt
    cases = (
        ((r1, res0), 'REQUEST_IDENTITY_MISMATCH'),
        ((r0, DecodeResult(res1.state, res0.receipt, ())), 'OUTPUT_STATE_MISMATCH'),
        ((r1, DecodeResult(res1.state, dataclasses.replace(
            receipt, target_steps=receipt.target_steps[:1], accepted_prefix=receipt.accepted_prefix[:1]), ())),
         'TARGET_STEP_SUFFIX_MISMATCH'),
        ((r1, DecodeResult(res1.state, dataclasses.replace(receipt, terminal='FAILED', failure='STALE'), ())),
         'FAILED_RECEIPT_COMMITTED_EFFECT'),
        ((r1, DecodeResult(res1.state, dataclasses.replace(receipt, failure='STALE'), ())), 'TERMINAL_STATUS'),
    )
    for (request, result), reason in cases:
        with pytest.raises(SteeringContractError) as err:
            bind_completed_epoch(request, result, epoch_index=3)
        assert err.value.reason == reason
    with pytest.raises(SteeringContractError) as err:
        bind_completed_epoch(r1, res1, epoch_index=0, predecessor=b0)
    assert err.value.reason == 'EPOCH_INDEX_NOT_MONOTONIC'


def test_observer_predecessor_wiring_fails_closed(rt):
    s = session(rt)
    lane = s['lane']
    o0, o1 = lane.observers
    b0 = bind_completed_epoch(s['r0'], s['res0'], epoch_index=0)
    with pytest.raises(SteeringContractError) as err:
        observe_epoch(b0, s['r0'], s['res0'], predecessor=o1)
    assert err.value.reason == 'OBSERVER_PREDECESSOR_MISMATCH'
    with pytest.raises(SteeringContractError) as err:
        observe_epoch(lane.binding, s['r0'], s['res0'], predecessor=o0)
    assert err.value.reason == 'EPOCH_BINDING_MISMATCH'
    missing = observe_epoch(lane.binding, s['r1'], s['res1'])  # missing predecessor output: reset
    assert missing.reset_flag == 1 and missing.core12_delta == (0,) * 12


def test_source_epoch_identity_invariant_to_sot(rt):
    ctx = initial_snapshot()
    s0 = rt.initial(ctx)
    request = prefill('req-A', ctx)
    disabled = run(rt, s0, request)
    lane = Lane()
    lane.observe(request, disabled)
    propose_fast_steering(lane.observer, target_model=rt.target.model_identity, x_projection=rt.target.projections['X'],
                          control_state=genesis())
    enabled = run(rt, s0, request)
    assert enabled.receipt == disabled.receipt and enabled.state == disabled.state
    assert enabled.receipt.digest == disabled.receipt.digest


def test_absent_dyn4_is_never_zero_encoded(rt):
    ctx = initial_snapshot()
    lane = Lane()
    s0 = rt.initial(ctx)
    bad = prefill('req-A', ctx, tokens=(1, 2))
    failed = run(rt, s0, bad, expected='0' * 64)
    assert failed.receipt.terminal == 'FAILED'
    _, o0 = lane.observe(bad, failed)
    assert o0.dyn4 is None and o0.dyn4_delta is None and o0.core12[11] == 0
    assert propose_fast_steering(o0, target_model=rt.target.model_identity,
                                 x_projection=rt.target.projections['X'], control_state=genesis()).disposition is O.NO_OP_FAILED_SOURCE
    good = prefill('req-A', ctx, tokens=(1, 2))
    _, o1 = lane.observe(good, run(rt, s0, good))
    assert o1.reset_flag == 0 and o1.dyn4 is not None and o1.dyn4_delta is None
    assert propose_fast_steering(o1, target_model=rt.target.model_identity,
                                 x_projection=rt.target.projections['X'], control_state=genesis()).disposition is O.NO_OP_DYN4_ABSENT


def test_application_lineage_context_projection_and_target_noops(rt):
    s = session(rt)
    ctx, xp, state = s['ctx'], s['xp'], s['res1'].state
    assert apply_next(s, greedy('req-OTHER', ctx))[1].outcome is O.NO_OP_LINEAGE_MISMATCH
    ctx2 = append_context(ctx, ContextItem('late', 'host', b'x', Lifetime.DYNAMIC), expected=ctx.digest)
    moved = apply_next(s, greedy('req-A', ctx2), target_state=rt.initial(ctx2))[1]
    assert moved.outcome is O.NO_OP_CONTEXT_MISMATCH
    unrelated = apply_next(s, greedy('req-A', ctx), target_state=rt.initial(ctx))[1]
    assert unrelated.outcome is O.NO_OP_LINEAGE_MISMATCH  # not a continuation of the source trajectory
    wrong_x = LatentProjection('X', xp.source_schema, xp.model, rt.target.projections['G'].weights)
    assert apply_next(s, greedy('req-A', ctx), x_projection=wrong_x)[1].outcome is O.NO_OP_PROJECTION_MISMATCH
    assert apply_next(s, greedy('req-A', ctx), target_state=state)[1].outcome is O.APPLIED
    observer = s['lane'].observer
    assert propose_fast_steering(observer, target_model=rt.target.model_identity, x_projection=rt.target.projections['G'],
                                 control_state=s['control']).disposition is O.NO_OP_PROJECTION_MISMATCH
    assert propose_fast_steering(observer, target_model='f' * 64, x_projection=xp,
                                 control_state=s['control']).disposition is O.NO_OP_TARGET_MISMATCH


def test_duplicate_application_noop(rt):
    s = session(rt)
    p = s['proposal']
    steered, a2, c2 = apply_next(s, greedy('req-A', s['ctx']))
    assert a2.outcome is O.APPLIED
    res2 = run(rt, s['res1'].state, steered)
    r3 = greedy('req-A', s['ctx'])
    out, a3, c3 = apply_next(s, r3, epoch_offset=2, target_state=res2.state, control_state=c2)
    assert a3.outcome is O.NO_OP_DUPLICATE and out is r3 and a3.predecessor_application_digest == a2.digest
    assert c3 is c2 and c3.hop_count == 1  # duplicate never advances the hop count
    assert apply_next(s, steered)[1].outcome is O.NO_OP_DUPLICATE  # packet already present
    with pytest.raises(SteeringContractError) as err:
        apply_next(s, r3, target_state=res2.state, control_state=c2)
    assert err.value.reason == 'APPLICATION_PREDECESSOR_NOT_EARLIER'
    assert p.latent_digest not in {x.digest for x in r3.latents}


def test_proposal_noop_dispositions_are_deterministic():
    xp, target = synthetic_projection(4), '3' * 64
    cases = (
        (synthetic_observer(core12=(1, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0), terminal_status='FAILED',
                            failure_code='STALE'), target, xp, O.NO_OP_FAILED_SOURCE),
        (synthetic_observer(reset_flag=1, core12_delta=(0,) * 12, dyn4_delta=(0.0,) * 4), target, xp, O.NO_OP_RESET),
        (synthetic_observer(dyn4_delta=None), target, xp, O.NO_OP_DYN4_ABSENT),
        (synthetic_observer(), 'f' * 64, xp, O.NO_OP_TARGET_MISMATCH),
        (synthetic_observer(), target, synthetic_projection(4, 'G', 'structural-latent.r0'), O.NO_OP_PROJECTION_MISMATCH),
        (synthetic_observer(), target, None, O.NO_OP_PROJECTION_MISMATCH),
        (synthetic_observer(), target, xp, O.PROPOSED),
    )
    for observer, target_model, projection, expected in cases:
        proposal = propose_fast_steering(observer, target_model=target_model, x_projection=projection,
                                         control_state=genesis('req-S'))
        assert proposal.disposition is expected
        assert (proposal.latent is None) == (expected is not O.PROPOSED)
        assert (proposal.not_before_epoch, proposal.expires_after_epoch) == (6, 7)


def test_record_invariants_fail_closed():
    for bad in (dict(reset_flag=1), dict(core12=(1,) * 11), dict(dyn4=(-1.0, 0.0, 0.0, 0.0)),
                dict(dyn4=(math.nan,) * 4), dict(predecessor_observer_digest=None),
                dict(terminal_status='FAILED'), dict(dyn4=None), dict(request_id=''),
                dict(core12=(1, 1, 4, 0, 0, 0, 0, 0, 0, 0, 2, 1))):
        with pytest.raises(ContractError):
            synthetic_observer(**bad)
    proposal = propose_fast_steering(synthetic_observer(), target_model='3' * 64, x_projection=synthetic_projection(4),
                                     control_state=genesis('req-S'))
    for bad in (dict(not_before_epoch=5), dict(expires_after_epoch=8), dict(steering_contract_digest='0' * 64),
                dict(latent=None), dict(disposition=O.APPLIED),
                dict(width_transport_digest=core.width_transport_digest(5)),
                dict(proposed_active_control_digest='a' * 64), dict(proposed_active_control_digest=None),
                dict(active_control_predecessor_digest='z' * 64), dict(previous_fast_application_digest='nope')):
        with pytest.raises(ContractError):
            dataclasses.replace(proposal, **bad)
    applied = FastSteeringApplication(O.APPLIED, None, D('p'), D('o'), 'req-S', 5, 6, D('c'), D('a'), D('x'), 1,
                                         D('t'), D('r'), D('r2'), D('x'), None)
    guarded = FastSteeringApplication(O.NO_OP_FAST_GUARD, FastGuardReason.HOP_BUDGET_EXHAUSTED, D('p'), D('o'),
                                         'req-S', 5, 6, D('c'), D('a'), D('a'), None, D('t'), D('r'), D('r'), None, None)
    for record, bad in ((applied, dict(hop_index=9)), (applied, dict(active_control_after_digest=D('y'))),
                        (applied, dict(active_control_before_digest=D('x'))),
                        (applied, dict(guard_reason=FastGuardReason.IDENTICAL_STATE_NOOP)),
                        (applied, dict(output_request_digest=D('r'))), (guarded, dict(guard_reason=None)),
                        (guarded, dict(active_control_after_digest=D('b'))),
                        (guarded, dict(output_request_digest=D('x'))), (guarded, dict(hop_index=1)),
                        (guarded, dict(outcome=O.NO_OP_EXPIRED))):
        with pytest.raises(ContractError):
            dataclasses.replace(record, **bad)


def test_global_event_fields_bind_branch44_shape(rt):
    s = session(rt)
    p, observer = s['proposal'], s['lane'].observer
    _, application, _ = apply_next(s, greedy('req-A', s['ctx']))
    fields = global_event_fields(observer, p, application)
    assert fields == dict(request_id='req-A', source_epoch=observer.epoch_index,
                          not_before_epoch=observer.epoch_index + 1, observer_digest=observer.digest,
                          proposal_digest=p.digest, outcome_digest=application.digest,
                          predecessor_observer_digest=observer.predecessor_observer_digest)
    assert fields['not_before_epoch'] >= fields['source_epoch'] + 1  # Branch43GlobalEvent future-only rule
    assert global_event_fields(observer, p)['outcome_digest'] is None
    with pytest.raises(SteeringContractError):
        global_event_fields(s['lane'].observers[0], p)


def test_no_hidden_mutable_module_state():
    assert not any(isinstance(node, (ast.Global, ast.Nonlocal)) for node in ast.walk(CORE_TREE))
    for node in CORE_TREE.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            assert not isinstance(node.value, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp))
    decorators = [d for node in ast.walk(CORE_TREE) if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                  for d in node.decorator_list]
    assert all('cache' not in ast.unparse(d) for d in decorators)


def test_frozen_constants_and_bounds():
    assert (core.WIDTH_MIN, core.WIDTH_MAX, core.X_LATENT_TTL_EPOCHS, core.SOURCE_DIMENSION) == (4, 32, 2, 4)
    for retired in ('BOUNDED_CLOSURE_CLAIM', 'NEGATIVE_CORE12_X_OUTCOME', 'CONTIGUOUS_WIDTH_CLOSURE_DIGEST'):
        assert not hasattr(core, retired)  # empirical claims are not carried as code
    assert 'unimplemented_policy_clauses' not in core._steering_profile()
    assert core._steering_profile()['fast_guard']['stall_guard'] == core.STALL_GUARD_DIGEST


# ------------------------------------------------------------ fast-control guard
NEW_STEERING_PROFILE_DIGEST = '39d34de2b4225c398a8f35415ec4bd3e9d7037a38cfce5f425e41d41b2e201d4'


def test_guard_01_genesis_control_state_exact():
    state = initial_fast_control_state('req-A')
    assert dataclasses.astuple(state) == ('req-A', '0' * 64, (), 0, None, None, None, None)
    assert state.active_control_digest == core.GENESIS_CONTROL_DIGEST == '0' * 64
    assert (core.MAX_HOPS, core.CYCLE_WINDOW, core.X_LATENT_TTL_EPOCHS) == (8, 4, 2)
    assert core.FAST_FAILURE_DEFAULT == 'NO_OP_RETAIN_ACTIVE_CONTROL'
    assert state.digest == initial_fast_control_state('req-A').digest != initial_fast_control_state('req-B').digest
    for bad in ('', None, 7):
        with pytest.raises(ContractError):
            initial_fast_control_state(bad)


def test_guard_02_control_state_authority_zero_and_fail_closed():
    state = initial_fast_control_state('req-A')
    assert isinstance(state, ProposalOnly) and issubclass(FastControlState, ProposalOnly)
    for flag in AUTHORITY:
        assert getattr(state, flag) is False
        with pytest.raises(AttributeError):
            setattr(state, flag, True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.hop_count = 1
    a, b = D('a'), D('b')
    valid = FastControlState('req-A', b, (a, b), 2, D('observer'), a, D('application'), 3)
    for bad in (dict(hop_count=9), dict(hop_count=-1), dict(hop_count=True), dict(request_id=''),
                dict(active_control_digest=a), dict(active_control_digest='Z' * 64),
                dict(applied_control_history=(a,)), dict(applied_control_history=(a, 'x' * 64)),
                dict(applied_control_history=[a, b]), dict(last_observer_state_digest=None),
                dict(previous_fast_application_digest=None), dict(last_applied_epoch_index=-1),
                dict(last_applied_epoch_index=None)):
        with pytest.raises(ContractError):
            dataclasses.replace(valid, **bad)
    with pytest.raises(ContractError):
        dataclasses.replace(state, active_control_digest=a)  # genesis active control is the zero digest


def test_guard_03_04_proposal_binds_predecessor_and_exact_control_identity(rt):
    s = session(rt)
    p = s['proposal']
    assert p.active_control_predecessor_digest == s['control'].active_control_digest == '0' * 64
    assert p.previous_fast_application_digest is None
    assert p.proposed_active_control_digest == p.latent.digest == p.latent_digest
    steered, first, advanced = apply_next(s, greedy('req-A', s['ctx']))
    result = run(rt, s['res1'].state, steered)
    s['lane'].observe(steered, result)
    p2 = propose_fast_steering(s['lane'].observer, target_model=rt.target.model_identity, x_projection=s['xp'],
                               control_state=advanced)
    assert p2.active_control_predecessor_digest == advanced.active_control_digest == p.latent_digest
    assert p2.previous_fast_application_digest == first.digest == advanced.previous_fast_application_digest
    assert p2.proposed_active_control_digest == p2.latent.digest != p.latent_digest
    early = propose_fast_steering(s['lane'].observers[0], target_model=rt.target.model_identity,
                                  x_projection=s['xp'], control_state=advanced)
    assert early.disposition is O.NO_OP_RESET and early.proposed_active_control_digest is None
    assert early.active_control_predecessor_digest == advanced.active_control_digest
    assert advanced == core._advance_fast_control(s['control'], p.observer_state_digest, p.latent_digest,
                                                  first.digest, p.not_before_epoch)


def test_guard_05_to_07_applied_advances_control_exactly_once(rt):
    s = session(rt)
    before, p = s['control'], s['proposal']
    _, application, after = apply_next(s, greedy('req-A', s['ctx']))
    assert application.outcome is O.APPLIED and application.guard_reason is None
    assert (after.active_control_digest == p.proposed_active_control_digest == application.applied_latent_digest
            == application.active_control_after_digest)
    assert application.active_control_before_digest == before.active_control_digest == '0' * 64
    assert after.hop_count == before.hop_count + 1 == application.hop_index == 1
    assert after.previous_fast_application_digest == application.digest
    assert application.control_state_digest == before.digest
    assert after.applied_control_history == (p.latent_digest,)
    assert (after.last_observer_state_digest, after.last_active_control_digest) == (p.observer_state_digest, '0' * 64)
    assert after.last_applied_epoch_index == p.not_before_epoch
    assert before == initial_fast_control_state('req-A')


def test_guard_08_09_active_predecessor_mismatch_retains_request_and_control(rt):
    s = session(rt)
    steered, _, advanced = apply_next(s, greedy('req-A', s['ctx']))
    result = run(rt, s['res1'].state, steered)
    s['lane'].observe(steered, result)
    stale = propose_fast_steering(s['lane'].observer, target_model=rt.target.model_identity, x_projection=s['xp'],
                                  control_state=s['control'])  # derived under genesis, applied after an APPLY
    request = greedy('req-A', s['ctx'])
    request_bytes, control_bytes = canonical_json_bytes(request), canonical_json_bytes(advanced)
    out, application, retained = apply_fast_steering(stale, request, target_epoch_index=stale.not_before_epoch,
                                                     target_state=result.state, x_projection=s['xp'],
                                                     control_state=advanced)
    assert (application.outcome, application.guard_reason) == (
        O.NO_OP_FAST_GUARD, FastGuardReason.ACTIVE_CONTROL_PREDECESSOR_MISMATCH)
    assert out is request and retained is advanced
    assert canonical_json_bytes(out) == request_bytes and canonical_json_bytes(retained) == control_bytes
    assert application.output_request_digest == application.input_request_digest == request.digest
    assert application.active_control_after_digest == application.active_control_before_digest
    assert application.hop_index is None and application.applied_latent_digest is None


def test_guard_10_11_identical_observer_active_pair_retains_active_control(rt):
    # Under the frozen pair-update rule this pair is unreachable from genesis through the public API (the active
    # control advances on every APPLY and the cycle window blocks returns); the frozen guard case is exercised with an
    # explicitly constructed, schema-valid control state.
    s = session(rt)
    observer, prior = s['lane'].observer, D('prior-control')
    stalled = FastControlState('req-A', prior, (prior,), 1, observer.digest, prior, D('prior-application'), 0)
    proposal = propose_fast_steering(observer, target_model=rt.target.model_identity, x_projection=s['xp'],
                                     control_state=stalled)
    request = greedy('req-A', s['ctx'])
    out, application, retained = apply_fast_steering(proposal, request, target_epoch_index=proposal.not_before_epoch,
                                                     target_state=s['res1'].state, x_projection=s['xp'],
                                                     control_state=stalled)
    assert (application.outcome, application.guard_reason) == (O.NO_OP_FAST_GUARD, FastGuardReason.IDENTICAL_STATE_NOOP)
    assert out is request and retained is stalled
    assert application.active_control_after_digest == application.active_control_before_digest == prior


def test_guard_12_to_15_cycles_2_3_4_rejected_length_5_outside_window(rt):
    s = session(rt)
    candidate = s['proposal'].latent_digest
    for length in (2, 3, 4, 5):
        state = advanced_state([candidate] + [D('control-%d' % k) for k in range(length - 1)])
        proposal = propose_fast_steering(s['lane'].observer, target_model=rt.target.model_identity,
                                         x_projection=s['xp'], control_state=state)
        assert proposal.proposed_active_control_digest == candidate
        request = greedy('req-A', s['ctx'])
        out, application, after = apply_fast_steering(proposal, request, target_epoch_index=proposal.not_before_epoch,
                                                       target_state=s['res1'].state, x_projection=s['xp'],
                                                       control_state=state)
        if length <= core.CYCLE_WINDOW:
            assert application.guard_reason is FastGuardReason.CONTROL_CYCLE_DETECTED, length
            assert out is request and after is state
        else:
            assert candidate not in state.applied_control_history
            assert application.outcome is O.APPLIED and after.hop_count == state.hop_count + 1


def test_guard_16_to_19_eight_applications_then_ninth_rejected(rt):
    s = session(rt)
    lane, ctx, xp = s['lane'], s['ctx'], s['xp']
    control, state = s['control'], s['res1'].state
    for hop in range(1, core.MAX_HOPS + 2):
        observer = lane.observer
        proposal = propose_fast_steering(observer, target_model=rt.target.model_identity, x_projection=xp,
                                         control_state=control)
        assert proposal.disposition is O.PROPOSED
        request = greedy('req-A', ctx, count=1)
        out, application, after = apply_fast_steering(proposal, request, target_epoch_index=observer.epoch_index + 1,
                                                      target_state=state, x_projection=xp, control_state=control)
        if hop <= core.MAX_HOPS:
            assert application.outcome is O.APPLIED and application.hop_index == hop == after.hop_count
            assert after.active_control_digest == proposal.latent_digest
            assert after.previous_fast_application_digest == application.digest
        else:
            assert (application.outcome, application.guard_reason) == (
                O.NO_OP_FAST_GUARD, FastGuardReason.HOP_BUDGET_EXHAUSTED)
            assert out is request
            assert after is control and after.hop_count == core.MAX_HOPS
            assert application.hop_index is None and application.applied_latent_digest is None
        result = run(rt, state, out)
        assert result.receipt.terminal == 'COMMITTED'
        assert (proposal.latent_digest in result.receipt.target_steps[0].latents) == (hop <= core.MAX_HOPS)
        lane.observe(out, result)
        control, state = after, result.state
    assert control.hop_count == core.MAX_HOPS and len(control.applied_control_history) == core.CYCLE_WINDOW


def test_guard_20_to_23_every_non_applied_outcome_retains_request_and_control(make_rt):
    rt = make_rt()
    s = session(rt)
    ctx, xp, state1, observer = s['ctx'], s['xp'], s['res1'].state, s['lane'].observer
    model = rt.target.model_identity
    base = advanced_state([D('seed-control')])  # non-genesis active control, hop 1

    def derived(control, source=observer):
        return propose_fast_steering(source, target_model=model, x_projection=xp, control_state=control)

    q = derived(base)
    n = q.source_epoch_index
    ctx2 = append_context(ctx, ContextItem('late', 'host', b'x', Lifetime.DYNAMIC), expected=ctx.digest)
    host_x = LatentInput('X', xp.source_schema, '9' * 64, ctx.digest, xp.digest, (0.02, 0.0, -0.01, 0.0))
    wrong_x = LatentProjection('X', xp.source_schema, xp.model, rt.target.projections['G'].weights)
    other_model_state = make_rt(dimension=6).initial(ctx)
    hop8 = advanced_state([D('hop-%d' % k) for k in range(core.MAX_HOPS)])
    stalled = FastControlState('req-A', D('stall'), (D('stall'),), 1, observer.digest, D('stall'), D('stall-app'), 0)
    cycle = advanced_state([q.latent_digest, D('cycle-1')])
    cases = (
        (derived(base, s['lane'].observers[0]), greedy('req-A', ctx), n + 1, state1, xp, base, O.NO_OP_RESET, None),
        (q, greedy('req-A', ctx), n, state1, xp, base, O.NO_OP_TOO_EARLY, None),
        (q, greedy('req-A', ctx), n + 3, state1, xp, base, O.NO_OP_EXPIRED, None),
        (q, greedy('req-OTHER', ctx), n + 1, state1, xp, base, O.NO_OP_LINEAGE_MISMATCH, None),
        (q, greedy('req-A', ctx2), n + 1, rt.initial(ctx2), xp, base, O.NO_OP_CONTEXT_MISMATCH, None),
        (q, greedy('req-A', ctx), n + 1, other_model_state, xp, base, O.NO_OP_TARGET_MISMATCH, None),
        (q, greedy('req-A', ctx), n + 1, rt.initial(ctx), xp, base, O.NO_OP_LINEAGE_MISMATCH, None),
        (q, greedy('req-A', ctx), n + 1, state1, wrong_x, base, O.NO_OP_PROJECTION_MISMATCH, None),
        (q, greedy('req-A', ctx, latents=(q.latent,)), n + 1, state1, xp, base, O.NO_OP_DUPLICATE, None),
        (q, greedy('req-A', ctx, latents=(host_x,)), n + 1, state1, xp, base, O.NO_OP_HOST_X_COLLISION, None),
        (s['proposal'], greedy('req-A', ctx), n + 1, state1, xp, base, O.NO_OP_FAST_GUARD,
         FastGuardReason.ACTIVE_CONTROL_PREDECESSOR_MISMATCH),
        (derived(hop8), greedy('req-A', ctx), n + 1, state1, xp, hop8, O.NO_OP_FAST_GUARD,
         FastGuardReason.HOP_BUDGET_EXHAUSTED),
        (derived(stalled), greedy('req-A', ctx), n + 1, state1, xp, stalled, O.NO_OP_FAST_GUARD,
         FastGuardReason.IDENTICAL_STATE_NOOP),
        (derived(cycle), greedy('req-A', ctx), n + 1, state1, xp, cycle, O.NO_OP_FAST_GUARD,
         FastGuardReason.CONTROL_CYCLE_DETECTED),
    )
    seen = set()
    for proposal, request, epoch, target_state, projection, control, outcome, reason in cases:
        control_bytes = canonical_json_bytes(control)
        out, application, retained = apply_fast_steering(proposal, request, target_epoch_index=epoch,
                                                         target_state=target_state, x_projection=projection,
                                                         control_state=control)
        assert (application.outcome, application.guard_reason) == (outcome, reason)
        assert out is request and retained is control and canonical_json_bytes(retained) == control_bytes
        assert (application.active_control_after_digest == application.active_control_before_digest ==
                control.active_control_digest)
        assert application.hop_index is None and application.applied_latent_digest is None
        assert application.output_request_digest == application.input_request_digest
        seen.add((outcome, reason))
    assert len(seen) == 13


def test_guard_24_deterministic_reconstruction(rt):
    def chain(runtime):
        s = session(runtime)
        steered, a1, c1 = apply_next(s, greedy('req-A', s['ctx']))
        result = run(runtime, s['res1'].state, steered)
        s['lane'].observe(steered, result)
        p2 = propose_fast_steering(s['lane'].observer, target_model=runtime.target.model_identity,
                                   x_projection=s['xp'], control_state=c1)
        steered2, a2, c2 = apply_fast_steering(p2, greedy('req-A', s['ctx']), target_epoch_index=p2.not_before_epoch,
                                               target_state=result.state, x_projection=s['xp'], control_state=c1)
        assert a2.outcome is O.APPLIED and c2.hop_count == 2
        return (s['proposal'].digest, a1.digest, c1.digest, p2.digest, a2.digest, c2.digest, steered2.digest)

    assert chain(rt) == chain(InferenceEngine(rt.target))


def _semantic_digest_probe():
    state = initial_fast_control_state('req-S')
    proposal = propose_fast_steering(synthetic_observer(), target_model='3' * 64, x_projection=synthetic_projection(9),
                                     control_state=state)
    values = [core.STEERING_PROFILE_DIGEST, state.digest, proposal.digest, core.width_transport_digest(9)]
    for k in range(core.MAX_HOPS + 1):
        candidate = D('probe-%d' % k)
        reason = core._fast_guard(state, D('probe-observer-%d' % k), state.active_control_digest, candidate)
        values.append('APPLY' if reason is None else reason.value)
        if reason is None:
            state = core._advance_fast_control(state, D('probe-observer-%d' % k), candidate, D('probe-app-%d' % k), k)
        values.append(state.digest)
    return hashlib.sha256('|'.join(values).encode()).hexdigest()


def test_guard_25_semantic_digests_invariant_under_pythonhashseed():
    expected = _semantic_digest_probe()
    code = ('import sys; sys.path[:0] = %r; from tests.inference import test_steering; '
            'print(test_steering._semantic_digest_probe())' % ([str(REPO / 'src'), str(REPO)],))
    for seed in ('0', '717', 'random'):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE='1')
        probe = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
        assert probe.returncode == 0, probe.stderr
        assert probe.stdout.strip() == expected, seed


def test_guard_26_to_28_guard_is_digest_only_no_model_no_ecs():
    functions = {node.name: node for node in ast.walk(CORE_TREE) if isinstance(node, ast.FunctionDef)}
    readers = {name for name, node in functions.items()
               if any(isinstance(n, ast.Attribute) and n.attr in ('hidden', 'logits') for n in ast.walk(node))}
    assert readers == {'_dyn4_or_absent'}
    allowed = {f.name for f in dataclasses.fields(FastControlState)} | {reason.name for reason in FastGuardReason}
    for name in ('_fast_guard', '_advance_fast_control'):
        attributes = {n.attr for n in ast.walk(functions[name]) if isinstance(n, ast.Attribute)}
        calls = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr
                 for n in ast.walk(functions[name]) if isinstance(n, ast.Call)}
        names = {n.id.lower() for n in ast.walk(functions[name]) if isinstance(n, ast.Name)}
        assert attributes <= allowed, attributes - allowed
        assert calls <= {'FastControlState'}
        assert not names & {'ecs', 'rrsi', 'kernel', 'entityport'}


def test_frozen_guard_compatibility_cases():
    """Semantic reproduction of the frozen guard cases over the concrete runtime identities; equality with
    historical qualification-local fixture digests is not claimed."""
    guard = core._fast_guard
    a, b = D('guard-a'), D('guard-b')
    stalled = FastControlState('req-G', a, (a,), 1, D('guard-observer'), a, D('guard-app'), 0)
    assert guard(stalled, D('guard-observer'), a, b) is FastGuardReason.IDENTICAL_STATE_NOOP
    assert guard(stalled, D('guard-observer-next'), a, b) is None
    for length, expected in ((2, FastGuardReason.CONTROL_CYCLE_DETECTED), (3, FastGuardReason.CONTROL_CYCLE_DETECTED),
                             (4, FastGuardReason.CONTROL_CYCLE_DETECTED), (5, None)):
        state = advanced_state([a] + [D('guard-cycle-%d' % k) for k in range(length - 1)], request_id='req-G')
        assert guard(state, D('guard-observer-next'), state.active_control_digest, a) is expected, length
    state = initial_fast_control_state('req-G')
    for hop in range(1, core.MAX_HOPS + 2):
        candidate = D('guard-hop-%d' % hop)
        reason = guard(state, D('guard-observer-%d' % hop), state.active_control_digest, candidate)
        if hop <= core.MAX_HOPS:
            assert reason is None
            state = core._advance_fast_control(state, D('guard-observer-%d' % hop), candidate,
                                               D('guard-app-%d' % hop), hop)
            assert state.hop_count == hop
        else:
            assert reason is FastGuardReason.HOP_BUDGET_EXHAUSTED
    c, d, e = D('guard-c'), D('guard-d'), D('guard-e')
    everything = FastControlState('req-G', e, (b, c, d, e), 8, D('guard-o'), e, D('guard-app'), 0)
    assert guard(everything, D('guard-o'), a, b) is FastGuardReason.ACTIVE_CONTROL_PREDECESSOR_MISMATCH
    assert guard(everything, D('guard-o'), e, b) is FastGuardReason.HOP_BUDGET_EXHAUSTED
    stalled_cycle = dataclasses.replace(everything, hop_count=4)
    assert guard(stalled_cycle, D('guard-o'), e, b) is FastGuardReason.IDENTICAL_STATE_NOOP
    assert guard(stalled_cycle, D('guard-o2'), e, b) is FastGuardReason.CONTROL_CYCLE_DETECTED
    assert [reason.value for reason in FastGuardReason] == [
        'ACTIVE_CONTROL_PREDECESSOR_MISMATCH', 'HOP_BUDGET_EXHAUSTED', 'IDENTICAL_STATE_NOOP', 'CONTROL_CYCLE_DETECTED']


def test_guard_profile_digest_changed_deterministic_and_scientific_object_unchanged():
    profile = core._steering_profile()
    assert 'unimplemented_policy_clauses' not in profile
    assert 'MAX_HOPS=8' in core.__doc__ and 'host-owned' in core.__doc__
    guard = profile['fast_guard']
    assert (guard['stall_guard'], guard['fast_steering_contract'], guard['phase4_mechanics_source'],
            guard['phase8_mechanics_source']) == (
        '0f6a28db77c9fb85e03c50c36d8decdc6e710b633cad99b9ba754c218fa7ef8c',
        '2533e28a485d2144d8229a46cdf10ff99430b8626c48001612672c003a992176',
        'e35de2f58dac4a7f7ca3dee56c62649b64ef918de228dcbbf27802b15dc71460',
        '317394545b5ad24c20d5d794f6fec19425633b563a6fe7c5fa93a76413e1935f')
    assert (guard['max_hops'], guard['cycle_window'], guard['ttl_epochs'], guard['genesis_active_control'],
            guard['failure_default'], guard['active_control_identity'],
            guard['historical_phase4_digest_equivalence']) == (
        8, 4, 2, '0' * 64, 'NO_OP_RETAIN_ACTIVE_CONTROL', 'SoT X LatentInput.digest', False)
    assert core.FAST_POLICY_DIGEST == '5a79c69fdec2fac63aded113d82292da0332240dcb114dc92751470e4b8549d4'
    assert core.STEERING_PROFILE_DIGEST == content_digest('elpis.sot.fast-steering-profile.r0', profile)
    assert core.STEERING_PROFILE_DIGEST == NEW_STEERING_PROFILE_DIGEST
    steering = profile['steering']
    assert (steering['contract'], steering['orientation'], steering['gain'], steering['orientation_matrix']) == (
        core.frozen_steering_contract_digest(), core.ORIENTATION_DIGEST, 0.09, EXPECTED_ORIENTATION)
    assert profile['transport'] == dict(rule=core.WIDTH_RULE_LABEL, authority=core.WIDTH_RULE_AUTHORITY_DIGEST,
                                        source_dimension=4, width_min=4, width_max=32)


def test_guard_request_lineage_control_state_fails_closed(rt):
    s = session(rt)
    foreign = initial_fast_control_state('req-B')
    with pytest.raises(SteeringContractError) as err:
        propose_fast_steering(s['lane'].observer, target_model=rt.target.model_identity, x_projection=s['xp'],
                              control_state=foreign)
    assert err.value.reason == 'CONTROL_STATE_REQUEST_MISMATCH'
    with pytest.raises(SteeringContractError) as err:
        apply_next(s, greedy('req-A', s['ctx']), control_state=foreign)
    assert err.value.reason == 'CONTROL_STATE_REQUEST_MISMATCH'
    with pytest.raises(SteeringContractError) as err:
        apply_next(s, greedy('req-A', s['ctx']), control_state=None)
    assert err.value.reason == 'CONTROL_STATE_TYPE'


# ------------------------------------------------------------ lane-separation architecture guards (branch-local)
# Subsystems the active inference path must never import.
SLOW_LANE_MODULES = ('elpis.evolution', 'elpis.ECS_C', 'elpis.pipeline', 'elpis.runtime', 'elpis.structure.grid81')
SLOW_LANE_PREFIXES = ()
SLOW_LANE_TAGS = ('darwin', 'evolution')
SLOW_LANE_ROOTS = tuple(str(REPO / 'src' / 'elpis' / name) for name in ('evolution', 'ecs', 'pipeline', 'runtime'))
INFERENCE_ROOT = REPO / 'src' / 'elpis' / 'inference'
STEERING_SOURCES = (INFERENCE_ROOT / 'steering.py', INFERENCE_ROOT / 'steered.py')
IMPORT_PROBE = """
import importlib, json, pkgutil, sys
sys.path[:0] = %r
import elpis.inference
names = sorted(m.name for m in pkgutil.walk_packages(elpis.inference.__path__, 'elpis.inference.'))
for name in names:
    importlib.import_module(name)
print(json.dumps({'entries': names, 'modules': sorted([n, getattr(m, '__file__', None) or ''] for n, m in sys.modules.items())}))
"""


def slow_lane_module(name):
    lowered = name.lower()
    return (any(name == m or name.startswith(m + '.') for m in SLOW_LANE_MODULES) or
            any(tag in lowered for tag in SLOW_LANE_TAGS))


def slow_lane_path(path):
    path = os.path.abspath(path)
    repo_local = path.startswith(str(REPO) + os.sep)
    return (any(path == root or path.startswith(root + os.sep) for root in SLOW_LANE_ROOTS) or
            (repo_local and any(tag in path.lower() for tag in SLOW_LANE_TAGS)))


def trace_calls(action):
    frames, c_calls = set(), set()

    def profiler(frame, event, arg):
        if event == 'call':
            frames.add((frame.f_code.co_filename, frame.f_code.co_name))
        elif event == 'c_call':
            c_calls.add((getattr(arg, '__module__', None) or '', getattr(arg, '__name__', None) or ''))

    set_all = getattr(threading, 'setprofile_all_threads', None)
    if set_all:
        set_all(profiler)
    sys.setprofile(profiler)
    threading.setprofile(profiler)
    try:
        result = action()
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
        if set_all:
            set_all(None)
    return result, frames, c_calls


def test_arch_01_active_inference_has_no_slow_lane_import_dependency():
    imported, decode_imports = {}, set()
    for root in (INFERENCE_ROOT,):
        for path in sorted(root.rglob('*.py')):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module] + [node.module + '.' + alias.name for alias in node.names]
                else:
                    continue
                for name in names:
                    imported.setdefault(name, set()).add(str(path.relative_to(REPO)))
                    if path not in STEERING_SOURCES:
                        decode_imports.add(name)
    assert imported
    assert not {name: files for name, files in imported.items() if slow_lane_module(name)}
    # Decoding never depends on steering.
    assert not any(name.startswith(('elpis.inference.steering', 'elpis.inference.steered')) for name in decode_imports)
    paths = [str(REPO / 'src')]
    probe = subprocess.run([sys.executable, '-c', IMPORT_PROBE % (paths,)], capture_output=True, text=True,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert probe.returncode == 0, probe.stderr
    report = json.loads(probe.stdout)
    assert 'elpis.inference.transaction' in report['entries'] and len(report['entries']) > 10
    assert not [(name, path) for name, path in report['modules']
                if slow_lane_module(name) or (path and slow_lane_path(path))]


def test_arch_02_to_05_token_path_zero_slow_lane_calls_and_no_global_wait(rt):
    loaded_before = {name for name in sys.modules if slow_lane_module(name)}

    def token_path():
        s = session(rt)  # prefill + decode, then steering observe/propose over completed epochs
        steered, application, control = apply_next(s, greedy('req-A', s['ctx']))
        steered_result = run(rt, s['res1'].state, steered)  # future decode block carrying the steering X latent
        speculative = run_speculative(rt, steered_result.state, greedy('req-A', s['ctx']), make_drafter(rt),
                                      expected_state=steered_result.state.digest, block_size=2)
        return application, steered_result, speculative

    (application, steered_result, speculative), frames, c_calls = trace_calls(token_path)
    files = {path for path, _ in frames}
    assert any(path.endswith(os.path.join('inference', 'transaction.py')) for path in files)
    assert any(path.endswith(os.path.join('drivers', 'dsv4', 'target.py')) for path in files)
    assert application.outcome is O.APPLIED
    assert steered_result.receipt.terminal == speculative.receipt.terminal == 'COMMITTED'
    assert not sorted(path for path in files if slow_lane_path(path))
    assert not sorted((module, name) for module, name in c_calls if module and slow_lane_module(module))
    assert {name for name in sys.modules if slow_lane_module(name)} == loaded_before  # nothing new loaded; clean-process closure is probed separately
    steering_frames = {name for path, name in frames if path.endswith(os.path.join('inference', 'steering.py'))}
    assert 'global_event_fields' not in steering_frames and 'propose_fast_steering' in steering_frames


def test_arch_06_fast_steering_affects_only_future_decode_after_source_completion(rt):
    s = session(rt)
    r1, res1, p = s['r1'], s['res1'], s['proposal']
    source_digests = (r1.digest, res1.receipt.digest, res1.state.digest)
    with pytest.raises(SteeringContractError) as err:  # nothing without a completed DecodeResult is observable
        bind_completed_epoch(r1, res1.state, epoch_index=9)
    assert err.value.reason == 'RESULT_TYPE'
    assert p.not_before_epoch == p.source_epoch_index + 1
    same, same_record, _ = apply_next(s, greedy('req-A', s['ctx']), epoch_offset=0)
    assert same_record.outcome is O.NO_OP_TOO_EARLY and p.latent_digest not in {x.digest for x in same.latents}
    steered, application, _ = apply_next(s, greedy('req-A', s['ctx']))
    future = run(rt, res1.state, steered)
    assert all(p.latent_digest not in step.latents for step in res1.receipt.target_steps)
    assert all(step.latents[-1] == p.latent_digest for step in future.receipt.target_steps)
    assert (r1.digest, res1.receipt.digest, res1.state.digest) == source_digests
    assert application.target_epoch_index > application.source_epoch_index


def test_arch_07_global_steering_output_cannot_enter_decode_directly(rt):
    s = session(rt)
    _, application, _ = apply_next(s, greedy('req-A', s['ctx']))
    fields = global_event_fields(s['lane'].observer, s['proposal'], application)
    assert all(value is None or type(value) in (str, int) for value in fields.values())  # identities only
    with pytest.raises(SteeringContractError):
        apply_fast_steering(fields, greedy('req-A', s['ctx']), target_epoch_index=2, target_state=s['res1'].state,
                            x_projection=s['xp'], control_state=s['control'])
    injected = InferenceRequest('req-A', s['ctx'].digest, 'GREEDY', count=1, latents=(fields,))
    try:
        rejected = rt.execute(s['res1'].state, injected, expected_state=s['res1'].state.digest)
    except ContractError:
        rejected = None
    assert rejected is None or (rejected.receipt.terminal == 'FAILED' and rejected.receipt.target_steps == ()
                                and rejected.state == s['res1'].state)
    names = {n.id for n in ast.walk(CORE_TREE) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(CORE_TREE) if isinstance(n, ast.Attribute)}
    assert not {name for name in names if slow_lane_module(name) or
                name.lower() in ('rrsi', 'ecs', 'entityport', 'kernel', 'propose_message')}
