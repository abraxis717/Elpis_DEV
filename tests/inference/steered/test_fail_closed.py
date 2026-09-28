import dataclasses

import pytest

import elpis.inference.steering as steering_core
from elpis.inference.contracts import ContractError
from elpis.inference.target import LatentInput
from elpis.inference.transaction import InferenceEngine
from elpis.inference.steered import SteeredInferenceEngine
from elpis.inference.steering import FastGuardReason, SteeringOutcome as O, initial_fast_control_state, propose_fast_steering

from ._helpers import D, greedy, scripted, two_epochs


def test_r3_failure_on_steered_epoch_retains_control_and_state(rt, monkeypatch):
    ctx, controller, _, e1 = two_epochs(rt)
    calls, original = [], InferenceEngine.execute
    monkeypatch.setattr(InferenceEngine, 'execute', lambda self, *a, **k: calls.append(1) or original(self, *a, **k))
    base = e1.runtime_result.state
    failed = controller.execute_epoch(e1.session, base, greedy('req-A', ctx), expected_state='0' * 64)
    assert calls == [1]  # exactly one engine invocation, no retry
    assert failed.runtime_result.receipt.terminal == 'FAILED' and failed.runtime_result.state == base
    assert failed.application.outcome is O.APPLIED and not failed.steering_committed  # attached, not committed
    assert failed.session.control_state == e1.session.control_state  # NO_OP_RETAIN_ACTIVE_CONTROL
    assert failed.observer.terminal_status == 'FAILED' and failed.proposal.disposition is O.NO_OP_FAILED_SOURCE
    again = SteeredInferenceEngine(rt).execute_epoch(e1.session, base, greedy('req-A', ctx), expected_state='0' * 64)
    assert again.provenance_digest == failed.provenance_digest
    after = controller.execute_epoch(failed.session, base, greedy('req-A', ctx))
    assert after.application.outcome is O.NO_OP_FAILED_SOURCE and after.executed_request is after.host_request


def test_host_x_collision_host_packet_wins(rt):
    ctx, controller, _, e1 = two_epochs(rt)
    xp = rt.target.projections['X']
    host_x = LatentInput('X', xp.source_schema, D('host-x'), ctx.digest, xp.digest, (0.02, 0.0, -0.01, 0.0))
    host = greedy('req-A', ctx, latents=(host_x,))
    e2 = controller.execute_epoch(e1.session, e1.runtime_result.state, host)
    assert e2.application.outcome is O.NO_OP_HOST_X_COLLISION and e2.executed_request is host
    assert e2.runtime_result.receipt.terminal == 'COMMITTED'
    assert all(step.latents[1:] == (host_x.digest,) for step in e2.runtime_result.receipt.target_steps)
    assert e2.session.control_state == e1.session.control_state and not e2.steering_committed


def test_duplicate_and_cycle_guards_are_not_bypassed(rt):
    ctx, controller, _, e1 = two_epochs(rt)
    s, p = e1.session, e1.proposal
    assert p.disposition is O.PROPOSED and s.control_state.hop_count == 0

    def session_under(control):
        rederived = propose_fast_steering(s.previous_observer, target_model=s.target_model,
                                          x_projection=rt.target.projections['X'], control_state=control)
        assert rederived.latent_digest == p.latent_digest
        return dataclasses.replace(s, control_state=control, pending_proposal=rederived)

    duplicate = steering_core._advance_fast_control(s.control_state, D('prior-observer'), p.latent_digest,
                                               D('prior-application'), e1.epoch_index)
    cycle = steering_core._advance_fast_control(duplicate, D('prior-observer-2'), D('other-control'),
                                           D('prior-application-2'), e1.epoch_index)
    for control, outcome, reason in ((duplicate, O.NO_OP_DUPLICATE, None),
                                     (cycle, O.NO_OP_FAST_GUARD, FastGuardReason.CONTROL_CYCLE_DETECTED)):
        host = greedy('req-A', ctx)
        e2 = controller.execute_epoch(session_under(control), e1.runtime_result.state, host)
        assert (e2.application.outcome, e2.application.guard_reason) == (outcome, reason)
        assert e2.executed_request is host and e2.session.control_state == control and not e2.steering_committed


def test_hop_budget_is_not_bypassed(rt):
    _, _, epochs = scripted(rt, greedy_tokens=12, block_size=1)
    assert [e.application.outcome for e in epochs[1:10]] == [O.NO_OP_RESET] + [O.APPLIED] * 8
    assert all(e.application.guard_reason is FastGuardReason.HOP_BUDGET_EXHAUSTED for e in epochs[10:])
    assert all(e.executed_request is e.host_request and not e.steering_committed for e in epochs[10:])
    assert epochs[-1].session.control_state.hop_count == 8


def test_incoherent_session_fails_closed(rt):
    _, controller, _, e1 = two_epochs(rt)
    genesis = controller.initial_session('req-A')
    advanced = steering_core._advance_fast_control(genesis.control_state, D('o'), D('c'), D('a'), 0)
    for bad in (dict(request_id=''), dict(target_model='nope'), dict(next_epoch_index=3),
                dict(next_epoch_index=-1), dict(control_state=initial_fast_control_state('req-B')),
                dict(control_state=advanced)):
        with pytest.raises(ContractError):
            dataclasses.replace(genesis, **bad)
    for bad in (dict(pending_proposal=None), dict(control_state=advanced), dict(target_model=D('other-model'))):
        with pytest.raises(ContractError):
            dataclasses.replace(e1.session, **bad)
