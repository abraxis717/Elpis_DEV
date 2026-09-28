import pytest

from elpis.inference.transaction import InferenceEngine
from elpis.inference.steered import SteeredContractError
from elpis.inference.steering import SteeringOutcome as O

from ._helpers import greedy, prefill, scripted, two_epochs


def test_active_steering_witness_epoch_n_to_n_plus_1(rt):
    ctx, controller, e0, e1 = two_epochs(rt)
    assert e1.application.outcome is O.NO_OP_RESET  # epoch 0 was a reset observation
    p = e1.proposal
    assert p.disposition is O.PROPOSED and (p.source_epoch_index, p.not_before_epoch) == (1, 2)          # 1, 2
    host = greedy('req-A', ctx, count=2)
    host_digest = host.digest
    e2 = controller.execute_epoch(e1.session, e1.runtime_result.state, host)
    app, latent = e2.application, p.latent
    assert app.outcome is O.APPLIED and app.proposal_digest == p.digest and e2.steering_committed          # 3
    assert e2.executed_request.latents == host.latents + (latent,)                                         # 4
    committed = e2.runtime_result.state
    new = len(e2.runtime_result.receipt.target_steps)
    assert new == 2 and all(step[-1] == latent for step in committed.step_latents[-new:])                 # 5
    assert all(step.latents[-1] == app.applied_latent_digest == latent.digest
               for step in e2.runtime_result.receipt.target_steps)
    assert app.output_request_digest == e2.executed_request.digest == e2.runtime_result.receipt.request    # 6
    assert e2.host_request is host and host.latents == () and host.digest == host_digest                   # 7
    assert all(latent not in step for step in e1.runtime_result.state.step_latents)  # source epoch untouched
    direct = InferenceEngine(rt.target).execute(e0.runtime_result.state, e1.host_request,
                                          expected_state=e0.runtime_result.state.digest)
    assert direct.receipt == e1.runtime_result.receipt and direct.state == e1.runtime_result.state
    assert e2.session.control_state.active_control_digest == latent.digest


def test_steered_epoch_diverges_from_unsteered_control(rt):
    ctx, controller, _, e1 = two_epochs(rt)
    host = greedy('req-A', ctx, count=4)
    steered = controller.execute_epoch(e1.session, e1.runtime_result.state, host)
    control = InferenceEngine(rt.target).execute(e1.runtime_result.state, host,
                                           expected_state=e1.runtime_result.state.digest)
    assert steered.application.outcome is O.APPLIED
    assert steered.runtime_result.state.neural.hidden != control.state.neural.hidden
    assert steered.runtime_result.receipt.digest != control.receipt.digest


def test_block_size_one_token_n_steers_token_n_plus_1(rt):
    _, _, epochs = scripted(rt, greedy_tokens=6, block_size=1)
    assert [len(e.runtime_result.receipt.target_steps) for e in epochs] == [3, 1, 1, 1, 1, 1, 1]
    assert [e.application.outcome for e in epochs[1:]] == [O.NO_OP_RESET] + [O.APPLIED] * 5
    for previous, current in zip(epochs, epochs[1:]):
        assert current.application.proposal_digest == previous.proposal.digest
        assert current.application.source_epoch_index == previous.epoch_index == current.epoch_index - 1
        assert current.runtime_result.receipt.input_state == previous.runtime_result.state.digest
        if current.application.outcome is O.APPLIED:
            assert current.runtime_result.receipt.target_steps[0].latents[-1] == previous.proposal.latent_digest


def test_generate_block_partition_preserves_r3_lineage(rt):
    ctx, controller, epochs = scripted(rt, greedy_tokens=5, block_size=2)
    assert [e.host_request.count for e in epochs[1:]] == [2, 2, 1]
    assert [len(e.runtime_result.receipt.target_steps) for e in epochs[1:]] == [2, 2, 1]
    for previous, current in zip(epochs, epochs[1:]):
        before = previous.runtime_result.state.receipts
        assert current.runtime_result.receipt.input_state == previous.runtime_result.state.digest
        assert current.runtime_result.state.receipts[:len(before)] == before
    assert len(epochs[-1].runtime_result.state.neural.tokens) == 3 + 5
    last = epochs[-1]
    with pytest.raises(SteeredContractError):
        controller.generate(last.session, last.runtime_result.state, prefill('req-A', ctx), block_size=1)
    with pytest.raises(SteeredContractError):
        controller.generate(last.session, last.runtime_result.state, greedy('req-A', ctx), block_size=0)
