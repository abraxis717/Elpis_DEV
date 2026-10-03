from collections import Counter
from dataclasses import fields
import numpy as np
import pytest

from elpis.inference.admission import ContextBudget, admit_context
from elpis.inference.contracts import ContractError
from elpis.inference.principal import PrincipalEngine, PrincipalRequest, PrincipalState
from elpis.inference.text import DSV41_RENDERER, ChatMessage, V41Tokenizer
from ...inference.test_token_stream_kernel import Recorder
from ...integration.test_context_substrate import TRAPPED

CONTEXT = "c" * 64


def admission(target, tokenizer):
    return admit_context(model=target.model_identity, tokenizer=tokenizer.identity, context_snapshot=CONTEXT,
                         corpus="d" * 64, proposals=(), resolved=(), omitted=0,
                         budget=ContextBudget(4, 4096, 2048), renderer=DSV41_RENDERER, text_tokenizer=tokenizer)


def test_real_tower_generation_and_sequence_release(tower, v41):
    target, meta = tower
    assert meta["TRAINING"] == "NONE" and meta["classification"] == "PRODUCTION_SHAPED_FIXTURE"
    engine = PrincipalEngine(target)
    state = engine.initial(CONTEXT)
    prompt = v41.encode_chat((ChatMessage("user", "Hello, Elpis."),))
    request = PrincipalRequest("tower", prompt, 12, v41.stop_tokens)
    context = admission(target, v41)
    sequence = engine.begin(state, request, context, expected_state=state.digest)
    assert sequence.working_state is not None and not sequence.done
    work = sequence.working_state
    capacity = work.nbytes
    outputs = []
    for _ in range(12):
        expected = int(np.argmax(work.logits))
        token = sequence.next()
        if token is None:
            break
        assert token == expected
        outputs.append(token)
        assert work.nbytes == capacity == target.state_bytes
    assert work.attention[1].count > 0 and work.attention[4].count > 0
    assert work.selected[2] == work.selected[3] and work.selected[5] == work.selected[6]
    commit = engine.finalize(sequence)
    assert commit.commit.outputs == tuple(outputs)
    assert sequence.working_state is None and work.closed and work.nbytes == 0
    assert not {"attention", "logits", "history"} & {f.name for f in fields(PrincipalState)}
    assert engine.replay(state, request, context, commit.commit).commit == commit.commit
    with pytest.raises(ContractError):
        target.window_step(work, 0, experts=target.admit_stream())


def test_no_slow_lane_hashes_or_tokenizer_work_during_tower_decode(tower, v41, monkeypatch):
    target, _ = tower
    engine = PrincipalEngine(target)
    state = engine.initial(CONTEXT)
    request = PrincipalRequest("hot", v41.encode_chat((ChatMessage("user", "Test"),)), 16, v41.stop_tokens)
    context = admission(target, v41)
    sequence = engine.begin(state, request, context, expected_state=state.digest)
    decoder = v41.decoder()
    def fail(*args, **kwargs):
        raise AssertionError("slow lane during active tower sequence")
    for owner, name in TRAPPED + [(V41Tokenizer, "load"), (V41Tokenizer, "from_bytes"), (V41Tokenizer, "encode")]:
        monkeypatch.setattr(owner, name, fail)
    recorder = Recorder()
    pieces = recorder.run(lambda: [decoder.push(t) for t in sequence])
    assert recorder.calls == Counter(), recorder.calls
    monkeypatch.undo()
    decoder.finish()
    assert pieces and engine.finalize(sequence).commit
    assert target.store.high_water <= target.store.staging_budget and target.store.staged_bytes == 0


def test_two_sequences_have_isolated_attention_and_identical_replay(tower, v41):
    target, _ = tower
    first, second, oracle = (target.window_initial() for _ in range(3))
    experts = target.admit_stream()
    ids = v41.encode("independent sequences with shared immutable parameters")
    for i, token in enumerate(ids):
        target.window_step(first, token, experts=experts)
        target.window_step(second, (token + 13) % target.config.vocab, experts=experts)
        target.window_step(oracle, token, experts=experts)
        np.testing.assert_array_equal(first.logits, oracle.logits)
    assert not np.array_equal(first.logits, second.logits)
    for state in (first, second, oracle):
        target.release_window(state)


def _released(target, monkeypatch):
    seen = []
    original = target.release_window
    def release(state):
        seen.append(state)
        original(state)
    monkeypatch.setattr(target, "release_window", release)
    return seen


def _failing_step(target, monkeypatch, after, error):
    calls = [0]
    original = target.window_step
    def step(state, token, *, experts):
        calls[0] += 1
        if calls[0] > after:
            raise error
        return original(state, token, experts=experts)
    monkeypatch.setattr(target, "window_step", step)


def _begin(target, v41, new_tokens=8):
    engine = PrincipalEngine(target)
    state = engine.initial(CONTEXT)
    request = PrincipalRequest("life", v41.encode_chat((ChatMessage("user", "Lifecycle"),)), new_tokens, ())
    return engine, state, request, admission(target, v41)


def test_typed_failure_releases_tower_state_immediately(tower, v41, monkeypatch):
    from elpis.inference.contracts import Code
    target, _ = tower
    engine, state, request, context = _begin(target, v41)
    seen = _released(target, monkeypatch)
    sequence = engine.begin(state, request, context, expected_state=state.digest)
    work = sequence.working_state
    _failing_step(target, monkeypatch, 0, ContractError(Code.LIMIT, "injected"))
    assert sequence.next() is None and sequence.done and sequence.stop_reason == "FAILED"
    assert seen == [work] and work.closed and sequence.working_state is None
    result = engine.finalize(sequence)
    assert result.commit is None and result.failure == "LIMIT" and seen == [work]


@pytest.mark.parametrize("phase", ("prefill", "decode"))
def test_unexpected_exception_releases_before_raising(tower, v41, monkeypatch, phase):
    target, _ = tower
    engine, state, request, context = _begin(target, v41)
    seen = _released(target, monkeypatch)
    if phase == "prefill":
        _failing_step(target, monkeypatch, 3, RuntimeError("injected"))
        with pytest.raises(RuntimeError):
            engine.begin(state, request, context, expected_state=state.digest)
    else:
        sequence = engine.begin(state, request, context, expected_state=state.digest)
        _failing_step(target, monkeypatch, 0, RuntimeError("injected"))
        with pytest.raises(RuntimeError):
            sequence.next()
        assert sequence.done and sequence.stop_reason == "FAILED" and sequence.working_state is None
        assert engine.finalize(sequence).commit is None
    assert len(seen) == 1 and seen[0].closed and seen[0].nbytes == 0


def test_abandoned_sequence_has_explicit_bounded_cleanup(tower, v41, monkeypatch):
    target, _ = tower
    engine, state, request, context = _begin(target, v41)
    seen = _released(target, monkeypatch)
    with engine.begin(state, request, context, expected_state=state.digest) as sequence:
        work = sequence.working_state
        sequence.next()
    assert sequence.done and sequence.stop_reason == "CLOSED" and work.closed and seen == [work]
    sequence.close()
    result = engine.finalize(sequence)          # still a valid commit boundary, released once
    assert result.commit.stop_reason == "CLOSED" and seen == [work]
    assert not {"attention", "logits", "history", "layer_streams"} & {f.name for f in fields(type(result.state))}
