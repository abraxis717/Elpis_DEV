"""Real tokenizer/HACF/ECS wiring with an explicitly scripted emission double.

This is transport/commit qualification, NOT a trained model or verbal inference.
The existing synthetic neural suites separately qualify the arithmetic hot lane.
"""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from elpis.inference.admission import ContextBudget, admit_context
from elpis.inference.context import initial_snapshot
from elpis.inference.contracts import ContractError
from elpis.inference.principal import PrincipalEngine, PrincipalRequest
from elpis.inference.target import WindowStep
from elpis.inference.text import ChatMessage, DSV41_RENDERER, V41Tokenizer
from elpis.pipeline.ingress import QueryIngress
from elpis.runtime.history import ReceiptHistory

from ..inference.test_text import official, tokenizer
from ..inference.test_token_stream_kernel import Recorder
from .test_context_substrate import TRAPPED
from .conftest import POSITIVE

CONTEXT = initial_snapshot().digest


class TokenEmissionTestDouble:
    """No neural parameters. Forces known IDs solely to exercise output transport."""
    def __init__(self, tokenizer, prefill, output):
        self.config = SimpleNamespace(tokenizer=tokenizer.identity, vocab=tokenizer.vocab_size,
                                      max_tokens=4096, local_window=1, dimension=1)
        self.model_identity, self.numerical_profile = "a" * 64, "b" * 64
        self._prefill, self._output = prefill, output
        self._logits = []
        for token in output:
            logits = np.full(tokenizer.vocab_size, -1, dtype=np.float32)
            logits[token] = 1
            self._logits.append(logits)

    def admit_stream(self, *, resident_experts=None):
        return None

    def window_initial(self):
        return SimpleNamespace(position=0)

    def window_step(self, state, token, *, experts):
        position = state.position + 1
        index = min(max(0, position - self._prefill), len(self._output) - 1)
        return SimpleNamespace(position=position, logits=self._logits[index], step=WindowStep(token, (), ()))


def empty_admission(tokenizer, model="a" * 64):
    return admit_context(model=model, tokenizer=tokenizer.identity, context_snapshot=CONTEXT,
                         corpus="c" * 64, proposals=(), resolved=(), omitted=0,
                         budget=ContextBudget(4, 4096, 2048), renderer=DSV41_RENDERER,
                         text_tokenizer=tokenizer)


def test_text_stream_finalizes_then_records_and_replays(runtime, tokenizer):
    prompt = tokenizer.encode_chat((ChatMessage("user", "Hello, Elpis."),))
    # Forced text must never be presented as a neural generation transcript.
    output = tokenizer.encode("Transport probe: 你好 👋🏽") + tokenizer.stop_tokens
    target = TokenEmissionTestDouble(tokenizer, len(prompt), output)
    engine = PrincipalEngine(target)
    state = engine.initial(CONTEXT)
    admission = empty_admission(tokenizer)
    emitted = []
    def emit(text):
        assert runtime.history.records() == ()  # no receipt before sequence ends
        emitted.append(text)
    result, record = runtime.run_text(engine, state, "Hello, Elpis.", admission, tokenizer=tokenizer,
                                     request_id="text-probe", max_new_tokens=100,
                                     expected_state=state.digest, emit=emit)
    assert result.text == "".join(emitted) == "Transport probe: 你好 👋🏽"
    assert result.principal.commit.outputs == output
    assert result.principal.commit.stop_reason == "STOP_TOKEN"
    assert [r.record.kind for r in runtime.history.records()] == ["principal.commit", "text.output"]
    assert record.record.kind == "text.output"
    assert engine.replay(state, result.request, admission, result.principal.commit).commit == result.principal.commit
    assert all(ns >= 0 for _, ns in result.timings_ns)


def test_real_hacf_content_uses_admitted_tokenizer(runtime, tokenizer, ingress, corpus):
    prepared = runtime.admit_context(ingress=ingress, task=POSITIVE, corpus_root=corpus[1],
                                     corpus_manifest=corpus[0].corpus_manifest_json,
                                     context_snapshot=CONTEXT, model="a" * 64, tokenizer=tokenizer.identity,
                                     budget=ContextBudget(4, 4096, 2048), max_document_bytes=1 << 20,
                                     text_tokenizer=tokenizer)
    admission = prepared.admission
    assert admission.renderer == DSV41_RENDERER and admission.objects
    prompt = tokenizer.encode_chat((ChatMessage("user", "Summarize."),))
    output = tokenizer.encode("Probe.") + tokenizer.stop_tokens
    engine = PrincipalEngine(TokenEmissionTestDouble(tokenizer, len(prompt) + len(admission.tokens), output))
    state = engine.initial(CONTEXT)
    result, _ = runtime.run_text(engine, state, "Summarize.", admission, tokenizer=tokenizer,
                                request_id="hacf-probe", max_new_tokens=10, expected_state=state.digest)
    assert result.principal.commit.prefill == len(admission.tokens) + len(prompt)
    assert [r.record.kind for r in runtime.history.records()] == [
        "ingress.proposal", "context.admission", "principal.commit", "text.output"]


def test_text_hot_loop_traps_slow_lane_and_hashing(runtime, tokenizer, monkeypatch):
    prompt = tokenizer.encode_chat((ChatMessage("user", "Hello"),))
    output = tokenizer.encode("中文 streaming " * 20) + tokenizer.stop_tokens
    engine = PrincipalEngine(TokenEmissionTestDouble(tokenizer, len(prompt), output))
    state = engine.initial(CONTEXT)
    admission = empty_admission(tokenizer)
    sequence = engine.begin(state, PrincipalRequest("hot", prompt, 500, tokenizer.stop_tokens),
                            admission, expected_state=state.digest)
    decoder = tokenizer.decoder()
    def trap(*args, **kwargs):
        raise AssertionError("slow lane reached")
    for owner, name in TRAPPED + [(V41Tokenizer, "load"), (V41Tokenizer, "from_bytes"),
                                  (V41Tokenizer, "encode"), (Path, "read_bytes")]:
        monkeypatch.setattr(owner, name, trap)
    recorder = Recorder()
    pieces = recorder.run(lambda: [decoder.push(token) for token in sequence])
    assert recorder.calls == Counter()
    monkeypatch.undo()
    assert "".join(pieces) + decoder.finish() == "中文 streaming " * 20
    assert engine.finalize(sequence).commit.outputs == output


def test_mismatch_and_output_failure_do_not_record(runtime, tokenizer):
    prompt = tokenizer.encode_chat((ChatMessage("user", "Hi"),))
    target = TokenEmissionTestDouble(tokenizer, len(prompt), tokenizer.encode("probe") + tokenizer.stop_tokens)
    engine = PrincipalEngine(target)
    state = engine.initial(CONTEXT)
    admission = empty_admission(tokenizer)
    def call(emit=None):
        return runtime.run_text(engine, state, "Hi", admission, tokenizer=tokenizer,
                                request_id="failure", max_new_tokens=8, expected_state=state.digest, emit=emit)
    target.config.tokenizer = "wrong"
    with pytest.raises(ContractError, match="mismatch"):
        call()
    target.config.tokenizer = tokenizer.identity
    target.config.vocab -= 1
    with pytest.raises(ContractError, match="vocabulary"):
        call()
    target.config.vocab += 1
    def fail(piece):
        raise RuntimeError("emitter failed")
    with pytest.raises(RuntimeError, match="emitter failed"):
        call(fail)
    assert runtime.history.records() == ()
    assert state.turn == 0


def test_context_budget_and_tokenizer_binding(tokenizer):
    with pytest.raises(ContractError):
        empty_admission(tokenizer, model="not-a-digest")
    admission = empty_admission(tokenizer)
    assert admission.tokenizer == tokenizer.identity
    with pytest.raises(ContractError, match="admitted text tokenizer"):
        admit_context(model="a" * 64, tokenizer="b" * 64, context_snapshot=CONTEXT,
                      corpus="c" * 64, proposals=(), resolved=(), omitted=0,
                      budget=ContextBudget(1, 2, 3), renderer=DSV41_RENDERER, text_tokenizer=tokenizer)
