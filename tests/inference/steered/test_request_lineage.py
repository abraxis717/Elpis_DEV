import ast
import dataclasses

import pytest

from elpis.inference.context import ContextItem, Lifetime, append_context, initial_snapshot
from elpis.inference.contracts import ContractError
from elpis.inference.transaction import InferenceEngine
from elpis.inference.steered import SteeredContractError, SteeredSessionState, SteeredInferenceEngine
from elpis.inference.steering import SteeringOutcome as O, propose_fast_steering

from ._helpers import CONTROLLER, digest_vector, greedy, prefill, scripted, two_epochs


def test_request_id_mismatch_fails_closed_before_r3(rt, monkeypatch):
    ctx = initial_snapshot()
    controller = SteeredInferenceEngine(rt)
    calls, original = [], InferenceEngine.execute
    monkeypatch.setattr(InferenceEngine, 'execute', lambda self, *a, **k: calls.append(1) or original(self, *a, **k))
    session = controller.initial_session('req-A')
    with pytest.raises(SteeredContractError) as err:
        controller.execute_epoch(session, rt.initial(ctx), prefill('req-B', ctx))
    assert err.value.reason == 'SESSION_REQUEST_MISMATCH' and calls == []
    with pytest.raises(ContractError):
        SteeredSessionState('req-B', session.target_model, 0, None, None, None, session.control_state)


def test_session_is_bound_to_target_model(make_rt):
    rt, other = make_rt(), make_rt(dimension=6, name='other')
    ctx = initial_snapshot()
    session = SteeredInferenceEngine(rt).initial_session('req-A')
    with pytest.raises(SteeredContractError) as err:
        SteeredInferenceEngine(other).execute_epoch(session, other.initial(ctx), prefill('req-A', ctx))
    assert err.value.reason == 'SESSION_TARGET_MISMATCH'


def test_fresh_session_for_b_is_independent_of_a(make_rt):
    shared, clean = make_rt(), make_rt(name='clean')
    scripted(shared, request_id='req-A', greedy_tokens=4)
    _, _, after_a = scripted(shared, request_id='req-B', greedy_tokens=4)
    _, _, alone = scripted(clean, request_id='req-B', greedy_tokens=4)
    assert digest_vector(after_a) == digest_vector(alone)
    assert all(e.session.request_id == e.session.control_state.request_id == 'req-B' for e in after_a)


def test_context_change_inside_session_noops_and_resets(rt):
    ctx, controller, _, e1 = two_epochs(rt)
    assert e1.proposal.disposition is O.PROPOSED
    ctx2 = append_context(ctx, ContextItem('late', 'host', b'x', Lifetime.DYNAMIC), expected=ctx.digest)
    host = prefill('req-A', ctx2)
    moved = controller.execute_epoch(e1.session, rt.initial(ctx2), host)
    assert moved.application.outcome is O.NO_OP_CONTEXT_MISMATCH and moved.executed_request is host
    assert moved.runtime_result.receipt.terminal == 'COMMITTED'
    assert moved.observer.reset_flag == 1 and moved.proposal.disposition is O.NO_OP_RESET
    assert moved.session.control_state == e1.session.control_state


def test_projection_mismatch_pending_proposal_noops(rt):
    ctx, controller, _, e1 = two_epochs(rt)
    s = e1.session
    mismatched = propose_fast_steering(s.previous_observer, target_model=s.target_model,
                                       x_projection=rt.target.projections['G'], control_state=s.control_state)
    assert mismatched.disposition is O.NO_OP_PROJECTION_MISMATCH
    host = greedy('req-A', ctx)
    e2 = controller.execute_epoch(dataclasses.replace(s, pending_proposal=mismatched), e1.runtime_result.state, host)
    assert e2.application.outcome is O.NO_OP_PROJECTION_MISMATCH and e2.executed_request is host
    assert e2.session.control_state == s.control_state


def test_no_process_global_or_instance_steering_state(rt):
    tree = ast.parse(CONTROLLER.read_text(encoding='utf-8'))
    assert not any(isinstance(node, (ast.Global, ast.Nonlocal)) for node in ast.walk(tree))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            assert isinstance(node.value, (ast.Constant, ast.Tuple)), ast.unparse(node)
    decorators = [d for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                  for d in node.decorator_list]
    assert all('cache' not in ast.unparse(d) for d in decorators)
    controller = SteeredInferenceEngine(rt)
    assert SteeredInferenceEngine.__slots__ == ('_runtime',) and not hasattr(controller, '__dict__')
    with pytest.raises(AttributeError):
        controller.session = 'hidden'
