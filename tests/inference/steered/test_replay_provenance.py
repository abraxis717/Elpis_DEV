from elpis.inference.transaction import InferenceEngine
from elpis.inference.steered import SteeredInferenceEngine
from elpis.inference.steering import SteeringOutcome as O

from ._helpers import digest_vector, scripted


def test_replay_from_initial_state_reproduces_all_digests(rt):
    ctx, _, epochs = scripted(rt, greedy_tokens=6, block_size=2)
    replayer = SteeredInferenceEngine(InferenceEngine(rt.target))
    session, state, replayed = replayer.initial_session('req-A'), rt.initial(ctx), []
    for e in epochs:
        r = replayer.execute_epoch(session, state, e.host_request)
        replayed.append(r)
        session, state = r.session, r.runtime_result.state
    assert digest_vector(replayed) == digest_vector(epochs)
    assert any(e.application is not None and e.application.outcome is O.APPLIED for e in replayed)


def test_r3_replay_accepts_every_composed_epoch(rt):
    ctx, _, epochs = scripted(rt, greedy_tokens=4, block_size=1)
    base = rt.initial(ctx)
    for e in epochs:
        replayed = rt.replay(base, e.executed_request, e.runtime_result.receipt)
        assert replayed.receipt == e.runtime_result.receipt and replayed.state == e.runtime_result.state
        base = e.runtime_result.state


def test_application_provenance_matches_r3_commit(rt):
    _, _, epochs = scripted(rt, greedy_tokens=5, block_size=1)
    applied = [e for e in epochs if e.application is not None and e.application.outcome is O.APPLIED]
    assert len(applied) == 4
    for e in applied:
        result, app = e.runtime_result, e.application
        assert app.output_request_digest == e.executed_request_digest == result.receipt.request
        assert app.input_request_digest == e.host_request.digest
        assert app.target_state_digest == result.receipt.input_state
        new = len(result.receipt.target_steps)
        recorded = [packet.digest for step in result.state.step_latents[-new:] for packet in step
                    if packet.channel == 'X']
        assert recorded == [app.applied_latent_digest] * new
        assert all(step.latents[-1] == app.applied_latent_digest for step in result.receipt.target_steps)
        assert e.steering_committed and e.session.control_state.active_control_digest == app.applied_latent_digest
