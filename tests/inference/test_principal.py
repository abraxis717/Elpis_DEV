"""Principal sequences: admitted context as tokens, bounded scratch, no K/V memory."""
from __future__ import annotations

from collections import Counter
from dataclasses import fields, replace
import inspect
import typing

import numpy as np
import pytest

from elpis.inference.admission import (
    AdmittedObject, ContextAdmission, ContextBudget, admit_context, render_synthetic_nibble16,
)
from elpis.inference.context import initial_snapshot
from elpis.inference.contracts import ContractError
from elpis.inference.drivers.dsv4.fixtures import make_fixture
from elpis.inference.experts import expert_kernel
from elpis.inference.principal import (
    PrincipalCommit, PrincipalEngine, PrincipalRequest, PrincipalState,
)
from elpis.inference.structural import AddressProposal
from elpis.inference.target import WindowState, softmax, vector
from elpis.substrate.synthetic import SyntheticFileAssets

from .test_token_stream_kernel import Recorder

CONTEXT = initial_snapshot().digest


@pytest.fixture
def warm_provider(native_workspace, fms_file_library):
    provider = SyntheticFileAssets(root=native_workspace, library=fms_file_library,
                                   warm_bytes=1 << 20, staging_bytes=1 << 16)
    yield provider
    provider.close()


@pytest.fixture
def fixture(warm_provider, native_workspace):
    target, resident, _ = make_fixture(warm_provider, native_workspace / "t")
    return target, resident


def _admission(target, texts, budget=ContextBudget(16, 1 << 16, 1 << 16)):
    objects = tuple(("%064x" % (i + 1)) for i in range(len(texts)))
    proposal = AddressProposal("1" * 64, "2" * 64, "3" * 64, CONTEXT, "4" * 64, "route", (), objects, (), None,
                               ("5" * 64,))
    return admit_context(model=target.model_identity, tokenizer=target.config.tokenizer,
                         context_snapshot=CONTEXT, corpus="3" * 64, proposals=(proposal,),
                         resolved=tuple(zip(objects, texts)), omitted=0, budget=budget)


def _run(engine, state, request, admission, **kwargs):
    sequence = engine.begin(state, request, admission, expected_state=state.digest, **kwargs)
    tokens = list(sequence)
    return tokens, engine.finalize(sequence), sequence


# --- oracle ------------------------------------------------------------------------------------------


def _reference(target, resident, tokens, generate):
    """Independent statement of the principal equations over the same fixture tensors."""
    c = target.config
    w = {k: t.array() for k, t in target.weights.items()}
    experts = {key: tuple(np.frombuffer(raw, dtype="<f4").reshape(t.shape)
                          for t, raw in zip(target.experts.manifest(*key).tensors, resident[key]))
               for key in resident}
    history, keys, values, logits_out, outputs = target.scheme.initial(), [], [], [], []
    p = target.scheme.parameters
    layer = p.layers.index(target.rows.table.bank.layer) if hasattr(p, "layers") else 0

    def step(token):
        nonlocal history, keys, values
        hashed = target.scheme.stream_hash(history, (token,))
        history = hashed.history
        from elpis.inference.contracts import RowIdentity
        rows = target.rows.lookup(tuple(RowIdentity(target.bank_identity, r) for r in hashed.rows[0][layer]))
        x = w["embedding"][token].copy() + np.asarray(vector(rows.mean(axis=0, dtype=np.float32)), "<f4") \
            @ target.projections["M"].weights.array()
        keys = (keys + [vector(x @ w["k"])])[-c.local_window:]
        values = (values + [vector(x @ w["v"])])[-c.local_window:]
        q = x @ w["q"]
        local = softmax(np.asarray(keys, "<f4") @ q / np.float32(c.dimension ** .5)) @ np.asarray(values, "<f4")
        hidden = np.tanh(x + local).astype("<f4")
        rl = hidden @ w["router"]
        idx = sorted(range(len(c.expert_ids)), key=lambda i: (-float(rl[i]), c.expert_ids[i]))[:c.active_experts]
        weights = tuple(float(v) for v in softmax(rl[idx]))
        mix = np.zeros_like(hidden)
        for expert, weight in tuple(zip((c.expert_ids[i] for i in idx), weights)) + tuple(
                (i, 1.0) for i in c.shared_experts):
            mix += np.float32(weight) * expert_kernel(hidden, experts[c.layer, expert])
        hidden = hidden + mix
        logits = hidden @ w["out"]
        logits_out.append(np.asarray(vector(logits), "<f4").tobytes())
        return logits

    for token in tokens:
        logits = step(token)
    for _ in range(generate):
        token = int(np.argmax(vector(logits)))
        outputs.append(token)
        logits = step(token)
    return outputs, logits_out


def test_principal_path_matches_its_independent_reference(fixture):
    target, resident = fixture
    admission = _admission(target, (b"bounded context from HACF", b"second object"))
    request = PrincipalRequest("r", (1, 4, 9, 2), 24)
    engine = PrincipalEngine(target)
    state = engine.initial(CONTEXT)
    sequence = engine.begin(state, request, admission, expected_state=state.digest, resident_experts=resident)
    tokens = list(sequence)
    expected_tokens, expected_logits = _reference(target, resident, admission.tokens + request.prompt, 24)
    assert tokens == expected_tokens
    assert np.asarray(sequence.working_state.logits, "<f4").tobytes() == expected_logits[-1]


def test_removed_term_is_exactly_the_compressed_global_kv(fixture):
    """Equivalence statement against the legacy kernel over the same tokens and no latents:
    keys, values, address rows and the memory term are identical at every step; hidden state
    and logits are identical exactly when the legacy global selection is empty."""
    target, resident = fixture
    admission = target.admit_stream(resident_experts=resident)
    stream, window = target.initial_stream(CONTEXT), target.window_initial()
    tokens = tuple((5 * i + 1) % 16 for i in range(20))
    differed = False
    for token in tokens:
        stream = target.stream_step(stream, token, admission=admission)
        window = target.window_step(window, token, experts=admission)
        assert stream.local_keys == window.keys and stream.local_values == window.values
        assert stream.step.rows == window.step.rows and stream.history == window.history
        same = stream.logits == window.logits and stream.hidden == window.hidden
        if not stream.step.selected:
            assert same
        differed |= not same
    assert differed and len(stream.global_pool) == len(tokens) // target.config.compression
    assert not hasattr(window, "global_pool") and not hasattr(window, "pending")


# --- bounded state ----------------------------------------------------------------------------------


def test_working_and_committed_state_are_bounded_independently_of_context(fixture):
    target, resident = fixture
    engine = PrincipalEngine(target)
    geometries = set()
    for size in (1, 40, 400):
        admission = _admission(target, tuple(bytes([65 + i % 26]) * 8 for i in range(size)),
                               budget=ContextBudget(512, 1 << 16, 4000))
        assert len(admission.tokens) <= 4000 and len(admission.objects) + admission.omitted == size
        state = engine.initial(CONTEXT)
        sequence = engine.begin(state, PrincipalRequest("r", (1,), 16), admission, expected_state=state.digest,
                                resident_experts=resident)
        peak = len(sequence.working_state.keys)
        for _ in sequence:
            peak = max(peak, len(sequence.working_state.keys), len(sequence.working_state.values))
        result = engine.finalize(sequence)
        assert peak <= target.config.local_window
        assert result.commit.working_set == (target.config.local_window, target.config.dimension)
        geometries.add(tuple((f.name, type(getattr(result.state, f.name)).__name__) for f in fields(result.state)))
    assert len(geometries) == 1  # committed state shape does not depend on admitted context size
    assert {f.name for f in fields(PrincipalState)} == {"model", "numerical_profile", "context_snapshot", "turn",
                                                        "predecessor"}
    assert {f.name for f in fields(WindowState)} == {"position", "history", "keys", "values", "hidden", "logits",
                                                     "step"}


# --- no K/V at the boundary ----------------------------------------------------------------------------

_FORBIDDEN = ("key", "value", "kv", "latent", "hidden", "pool", "pending", "global", "tensor", "logit")


def _flatten(annotation):
    origin = typing.get_origin(annotation)
    return [annotation] if origin is None else [annotation] + [a for arg in typing.get_args(annotation)
                                                               for a in _flatten(arg)]


@pytest.mark.parametrize("contract", [ContextAdmission, AdmittedObject, ContextBudget, PrincipalState,
                                      PrincipalRequest, PrincipalCommit])
def test_boundary_contracts_carry_no_kv_or_vectors(contract):
    hints = typing.get_type_hints(contract)
    for field in fields(contract):
        assert not any(bad in field.name.lower() for bad in _FORBIDDEN), (contract.__name__, field.name)
        assert float not in _flatten(hints[field.name]), (contract.__name__, field.name)


def test_begin_accepts_no_latents_and_admission_refuses_vectors(fixture):
    target, _ = fixture
    parameters = inspect.signature(PrincipalEngine.begin).parameters
    assert not any(bad in name for name in parameters for bad in ("latent", "key", "value", "kv"))
    good = _admission(target, (b"abc",))
    with pytest.raises(ContractError):
        AdmittedObject("1" * 64, "2" * 64, 3, (0.5, 1.5))
    with pytest.raises(ContractError):
        replace(good, objects=(AdmittedObject("1" * 64, "2" * 64, 3, (np.float32(1.0),)),))


# --- fast lane ------------------------------------------------------------------------------------------


def test_token_loop_does_no_identity_or_hashing_once_resident(fixture):
    target, resident = fixture
    engine = PrincipalEngine(target)
    admission = _admission(target, (b"warm the pages this path touches",))
    request = PrincipalRequest("r", (3, 1), 40)
    state = engine.initial(CONTEXT)
    _run(engine, state, request, admission, resident_experts=resident)
    sequence = engine.begin(state, request, admission, expected_state=state.digest, resident_experts=resident)
    recorder = Recorder()
    tokens = recorder.run(lambda: list(sequence))
    assert len(tokens) == 40 and recorder.calls == Counter(), recorder.calls


# --- commit, replay, stop, failures ----------------------------------------------------------------------


def test_commit_replays_and_chains_turns(fixture):
    target, resident = fixture
    engine = PrincipalEngine(target)
    state = engine.initial(CONTEXT)
    admission = _admission(target, (b"turn one",))
    request = PrincipalRequest("turn-1", (2, 7), 10)
    tokens, result, _ = _run(engine, state, request, admission)
    assert result.commit.outputs == tuple(tokens) and result.commit.prefill == len(admission.tokens) + 2
    assert engine.replay(state, request, admission, result.commit).commit == result.commit
    assert result.state.turn == 1 and result.state.predecessor == result.commit.digest
    tampered = replace(result.commit, outputs=result.commit.outputs[:-1] + ((result.commit.outputs[-1] + 1) % 16,))
    with pytest.raises(ContractError, match="replay mismatch"):
        engine.replay(state, request, admission, tampered)
    second, _, _ = _run(engine, result.state, PrincipalRequest("turn-2", (1,), 3), _admission(target, (b"two",)))
    assert len(second) == 3


def test_stop_token_and_yield(fixture):
    target, _ = fixture
    engine = PrincipalEngine(target)
    state = engine.initial(CONTEXT)
    admission = _admission(target, (b"stop here",))
    full, _, _ = _run(engine, state, PrincipalRequest("r", (1,), 12), admission)
    stopping = PrincipalRequest("r", (1,), 12, stop_tokens=(full[2],))
    tokens, result, sequence = _run(engine, state, stopping, admission)
    assert tokens == full[:full.index(full[2]) + 1] and result.commit.stop_reason == "STOP_TOKEN"
    sequence = engine.begin(state, PrincipalRequest("r", (1,), 12), admission, expected_state=state.digest)
    produced = [sequence.next() for _ in range(5)]
    sequence.stop()
    result = engine.finalize(sequence)
    assert result.commit.outputs == tuple(produced) and result.commit.stop_reason == "YIELD"
    assert engine.replay(state, PrincipalRequest("r", (1,), 12), admission, result.commit).commit == result.commit


def test_begin_fails_closed_without_touching_committed_state(fixture):
    target, _ = fixture
    engine = PrincipalEngine(target)
    state = engine.initial(CONTEXT)
    admission = _admission(target, (b"x",))
    request = PrincipalRequest("r", (1,), 4)
    cases = [
        dict(expected_state="0" * 64),                                                          # stale base
        dict(admission=replace(admission, context_snapshot="9" * 64)),                          # other snapshot
        dict(admission=replace(admission, model="8" * 64)),                                     # other model
        dict(request=PrincipalRequest("r", (99,), 4)),                                          # vocabulary
        dict(request=PrincipalRequest("r", (1,), target.config.max_tokens)),                    # capacity
    ]
    for case in cases:
        sequence = engine.begin(state, case.get("request", request), case.get("admission", admission),
                                expected_state=case.get("expected_state", state.digest))
        assert list(sequence) == [] and sequence.stop_reason == "FAILED"
        result = engine.finalize(sequence)
        assert result.state is state and result.commit is None and result.failure
    active = engine.begin(state, request, admission, expected_state=state.digest)
    with pytest.raises(ContractError, match="still active"):
        engine.finalize(active)


def test_admission_budget_order_and_provenance(fixture):
    target, _ = fixture
    texts = (b"aaaa", b"bbbbbbbb", b"cc")
    admission = _admission(target, texts, budget=ContextBudget(3, 12, 1 << 10))
    assert [o.size for o in admission.objects] == [4, 8] and admission.omitted == 1
    assert admission.tokens == render_synthetic_nibble16(b"aaaa") + render_synthetic_nibble16(b"bbbbbbbb")
    proposal = AddressProposal("1" * 64, "2" * 64, "3" * 64, CONTEXT, "4" * 64, "r", (), ("a" * 64,), (), None, ())
    base = dict(model=target.model_identity, tokenizer=target.config.tokenizer, context_snapshot=CONTEXT,
                corpus="3" * 64, proposals=(proposal,), omitted=0, budget=ContextBudget(4, 64, 64))
    with pytest.raises(ContractError, match="not proposed"):
        admit_context(resolved=(("b" * 64, b"x"),), **base)
    with pytest.raises(ContractError, match="STALE"):
        admit_context(resolved=(("a" * 64, b"x"),), **dict(base, context_snapshot="9" * 64))
    with pytest.raises(ContractError, match="UNSUPPORTED"):
        admit_context(resolved=(("a" * 64, b"x"),), **dict(base, tokenizer="production-bpe"))
