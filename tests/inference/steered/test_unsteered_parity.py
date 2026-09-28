from elpis.inference.context import initial_snapshot
from elpis.inference.target import LatentInput
from elpis.inference.steered import SteeredInferenceEngine
from elpis.inference.steering import SteeringOutcome as O

from ._helpers import D, greedy, key_paths, prefill, strip_timing


def test_parity_with_direct_engine_when_no_proposal_is_eligible(make_rt):
    composed_rt, direct_rt = make_rt(), make_rt(name='direct')
    ctx = initial_snapshot()
    controller = SteeredInferenceEngine(composed_rt)
    xp = composed_rt.target.projections['X']
    host_x = LatentInput('X', xp.source_schema, D('host-x'), ctx.digest, xp.digest, (0.02, 0.0, -0.01, 0.0))
    hosts = (prefill('req-A', ctx), greedy('req-A', ctx, count=2), greedy('req-A', ctx, count=2, latents=(host_x,)))
    session, c_state, d_state, outcomes = controller.initial_session('req-A'), composed_rt.initial(ctx), \
        direct_rt.initial(ctx), []
    for host in hosts:
        composed = controller.execute_epoch(session, c_state, host)
        direct = direct_rt.execute(d_state, host, expected_state=d_state.digest)
        assert composed.executed_request is host
        assert composed.runtime_result.state == direct.state and composed.runtime_result.receipt == direct.receipt
        assert strip_timing(composed.runtime_result.telemetry) == strip_timing(direct.telemetry)
        assert key_paths(composed.runtime_result.telemetry) == key_paths(direct.telemetry)
        outcomes.append(None if composed.application is None else composed.application.outcome)
        session, c_state, d_state = composed.session, composed.runtime_result.state, direct.state
    assert outcomes == [None, O.NO_OP_RESET, O.NO_OP_HOST_X_COLLISION]


def test_wall_clock_timing_is_the_only_nondeterministic_engine_output(make_rt):
    a, b = make_rt(), make_rt(name='b')
    ctx = initial_snapshot()
    ra = a.execute(a.initial(ctx), prefill('req-A', ctx), expected_state=a.initial(ctx).digest)
    rb = b.execute(b.initial(ctx), prefill('req-A', ctx), expected_state=b.initial(ctx).digest)
    assert ra.state == rb.state and ra.receipt == rb.receipt
    assert strip_timing(ra.telemetry) == strip_timing(rb.telemetry) and key_paths(ra.telemetry) == key_paths(rb.telemetry)
    assert any(path.endswith('_ns') for path in key_paths(ra.telemetry))
