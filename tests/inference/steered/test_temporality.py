import dataclasses

import pytest

from elpis.inference.context import initial_snapshot
from elpis.inference.contracts import ContractError
from elpis.inference.transaction import InferenceEngine
from elpis.inference.steered import SteeredInferenceEngine
from elpis.inference.steering import (SteeringOutcome as O, apply_fast_steering, bind_completed_epoch,
                            initial_fast_control_state, observe_epoch)

from ._helpers import greedy, prefill, scripted, two_epochs


def test_first_epoch_unsteered_without_fabricated_recurrence(rt):
    ctx = initial_snapshot()
    controller = SteeredInferenceEngine(rt)
    genesis = controller.initial_session('req-A')
    assert (genesis.next_epoch_index, genesis.previous_binding, genesis.previous_observer,
            genesis.pending_proposal) == (0, None, None, None)
    assert genesis.control_state == initial_fast_control_state('req-A')
    host = prefill('req-A', ctx)
    e0 = controller.execute_epoch(genesis, rt.initial(ctx), host)
    assert e0.application is None and e0.executed_request is host and not e0.steering_committed
    assert e0.observer.reset_flag == 1 and e0.observer.predecessor_observer_digest is None
    assert e0.proposal.disposition is O.NO_OP_RESET and e0.proposal.latent is None
    assert e0.session.control_state == genesis.control_state
    base = rt.initial(ctx)
    assert InferenceEngine(rt.target).execute(base, host, expected_state=base.digest).receipt == e0.runtime_result.receipt


def test_same_epoch_never_receives_its_own_proposal(rt):
    ctx, _, epochs = scripted(rt, greedy_tokens=5, block_size=1)
    projection = rt.target.projections['X']
    for e in epochs:
        p = e.proposal
        assert p.source_epoch_index == e.epoch_index and p.not_before_epoch == e.epoch_index + 1
        if e.application is not None:
            assert e.application.source_epoch_index == e.epoch_index - 1
        if p.latent is not None:
            assert p.latent not in e.executed_request.latents
            assert all(p.latent_digest not in step.latents for step in e.runtime_result.receipt.target_steps)
            _, own, _ = apply_fast_steering(p, greedy('req-A', ctx), target_epoch_index=e.epoch_index,
                                            target_state=e.runtime_result.state, x_projection=projection,
                                            control_state=initial_fast_control_state('req-A'))
            assert own.outcome is O.NO_OP_TOO_EARLY


def test_proposal_admissible_only_inside_ttl_window(rt):
    ctx, _, _, e1 = two_epochs(rt)
    p, n = e1.proposal, e1.proposal.source_epoch_index
    assert p.disposition is O.PROPOSED and (p.not_before_epoch, p.expires_after_epoch) == (n + 1, n + 2)
    for epoch, outcome in ((n - 1, O.NO_OP_TOO_EARLY), (n, O.NO_OP_TOO_EARLY), (n + 1, O.APPLIED),
                           (n + 2, O.APPLIED), (n + 3, O.NO_OP_EXPIRED)):
        host = greedy('req-A', ctx)
        out, application, _ = apply_fast_steering(p, host, target_epoch_index=epoch,
                                                  target_state=e1.runtime_result.state,
                                                  x_projection=rt.target.projections['X'],
                                                  control_state=e1.session.control_state)
        assert application.outcome is outcome and (out is host) == (outcome is not O.APPLIED)


def test_session_cannot_carry_stale_or_foreign_proposal(rt):
    _, _, e0, e1 = two_epochs(rt)
    s = e1.session
    for bad in (dict(next_epoch_index=s.next_epoch_index + 2), dict(pending_proposal=e0.proposal),
                dict(previous_observer=e0.observer), dict(previous_binding=e0.binding)):
        with pytest.raises(ContractError):
            dataclasses.replace(s, **bad)


def test_absent_dyn4_is_never_substituted(rt):
    ctx = initial_snapshot()
    controller = SteeredInferenceEngine(rt)
    s0 = rt.initial(ctx)
    failed = controller.execute_epoch(controller.initial_session('req-A'), s0, prefill('req-A', ctx),
                                      expected_state='0' * 64)
    assert failed.runtime_result.receipt.terminal == 'FAILED' and failed.runtime_result.state == s0
    assert failed.observer.dyn4 is None and failed.observer.dyn4_delta is None
    assert failed.proposal.disposition is O.NO_OP_FAILED_SOURCE
    first = controller.execute_epoch(failed.session, s0, prefill('req-A', ctx))
    assert first.application.outcome is O.NO_OP_FAILED_SOURCE and first.executed_request is first.host_request
    assert first.observer.reset_flag == 0 and first.observer.dyn4 is not None and first.observer.dyn4_delta is None
    assert first.proposal.disposition is O.NO_OP_DYN4_ABSENT
    second = controller.execute_epoch(first.session, first.runtime_result.state, greedy('req-A', ctx))
    assert second.application.outcome is O.NO_OP_DYN4_ABSENT and second.observer.dyn4_delta is not None
    assert second.proposal.disposition is O.PROPOSED
    third = controller.execute_epoch(second.session, second.runtime_result.state, greedy('req-A', ctx))
    assert third.application.outcome is O.APPLIED


def test_dyn4_recurrence_matches_direct_steering_primitives(rt):
    _, _, epochs = scripted(rt, greedy_tokens=4, block_size=1)
    binding = observer = None
    for e in epochs:
        binding = bind_completed_epoch(e.executed_request, e.runtime_result, epoch_index=e.epoch_index,
                                       predecessor=binding)
        observer = observe_epoch(binding, e.executed_request, e.runtime_result, predecessor=observer)
        assert binding == e.binding and observer == e.observer
    assert all(e.observer.dyn4 is not None for e in epochs)
