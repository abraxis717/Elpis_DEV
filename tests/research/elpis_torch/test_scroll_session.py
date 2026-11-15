"""Last-block CGC fast weights: cached keys, values and rolling memory stay valid. Full-stack ones do not."""
from __future__ import annotations

import copy
import io
import json
import math
import pickle
import statistics

import pytest

from .conftest import SKIP_REASON, make_model
from .test_scroll import _model, _tokens
from .test_session import _scenario

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch.lastblock import capture_pre, last_block_logits  # noqa: E402
from research.elpis_torch.model import parameter_digest  # noqa: E402
from research.elpis_torch.scroll_session import ScrollSession  # noqa: E402
from research.elpis_torch.session import (GeneratedTokens, SessionConfig, SessionError, Status,  # noqa: E402
                                          SupportSet, VolatileSession)
from research.elpis_torch.training import freeze  # noqa: E402

LAST = SessionConfig(lr=0.2, max_steps=16, fast_scope="last")
STREAM = [256] + [65 + (i * 7) % 5 for i in range(299)]            # a periodic stream: something to learn


def _frozen(**changes):
    return freeze(_model(sharpen=False, dtype="float32", **changes))


# -- the proof, numerically ----------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_logits_are_a_function_of_the_pre_channel_stream_and_the_last_core_weight(name):
    model = freeze(make_model(name, dtype="float64"))
    tokens = _tokens(2, 40)
    with torch.no_grad(), capture_pre(model) as captured:
        logits = model(tokens)
    assert float((last_block_logits(model, captured.pre) - logits).abs().max()) < 1e-12
    if name != "A":
        weight = model.blocks[-1].cgc.core.weight
        other = weight + 0.3 * torch.randn(weight.shape, generator=torch.Generator().manual_seed(0), dtype=weight.dtype)
        changed = copy.deepcopy(model)
        with torch.no_grad():
            changed.blocks[-1].cgc.core.weight.copy_(other)
            assert float((last_block_logits(model, captured.pre, other) - changed(tokens)).abs().max()) < 1e-12


def test_changing_the_last_blocks_cgc_changes_no_cached_key_or_value_but_changing_an_earlier_one_does():
    """The R0 KV cache, written under three sets of weights: bitwise identical for a last-block change."""
    model = freeze(make_model("B", dtype="float64"))
    tokens = _tokens(2, 40)

    def cache_under(block_index):
        other = copy.deepcopy(model)
        with torch.no_grad():
            other.blocks[block_index].cgc.core.weight.mul_(1.5)
            cache = other.new_cache(2)
            logits = other(tokens, cache)
        return cache, logits

    with torch.no_grad():
        base = model.new_cache(2)
        reference = model(tokens, base)
    last, last_logits = cache_under(-1)
    first, _ = cache_under(0)
    assert all(torch.equal(a, b) for a, b in zip(base.keys + base.values, last.keys + last.values))
    assert not torch.equal(last_logits, reference)                        # the weight change is real
    assert torch.equal(base.keys[0], first.keys[0])                       # layer 0 reads embeddings only
    assert not torch.equal(base.keys[1], first.keys[1]) and not torch.equal(base.values[1], first.values[1])


def test_the_rolling_memory_is_bitwise_what_a_model_with_the_adapted_weight_would_have_written():
    """Stream 300 tokens (2.3 horizons), adapt, keep streaming: compare with a model that had W + dW all along."""
    model = _frozen()
    with ScrollSession(model, LAST) as session:
        session.observe(STREAM[:200])
        before = [{k: v.clone() for k, v in layer.items()} for layer in session.state.layers]
        results = session.adapt_on_stream(steps=6)
        assert all(r.applied for r in results) and results[-1].loss_after < results[0].loss_before
        assert all(torch.equal(layer[k], kept[k]) for layer, kept in zip(session.state.layers, before) for k in kept)
        after_adapt = session.next_logits()
        session.observe(STREAM[200:300])
        continued = session.next_logits()
        adapted = copy.deepcopy(model)
        with torch.no_grad():
            adapted.blocks[-1].cgc.core.weight.add_(session._delta[session.fast_weight_names[0]])
        state = adapted.new_state(1)
        with torch.no_grad():                                            # the same chunk boundaries
            for start, stop in ((0, 32), (32, 64), (64, 96), (96, 128), (128, 160), (160, 192), (192, 200)):
                at_200 = adapted(torch.tensor([STREAM[start:stop]]), state)[0, -1]
            for start in range(200, 300, 32):
                at_300 = adapted(torch.tensor([STREAM[start:min(start + 32, 300)]]), state)[0, -1]
        assert torch.allclose(after_adapt, at_200, rtol=0, atol=1e-5)
        assert torch.allclose(continued, at_300, rtol=0, atol=1e-5)
        for layer, other in zip(session.state.layers, state.layers):
            assert all(torch.equal(layer[k][:, :, :300 % 8] if k[0] == "p" else layer[k],
                                   other[k][:, :, :300 % 8] if k[0] == "p" else other[k]) for k in layer)


# -- R0 session: full-stack against last-block ---------------------------------------------------------------

def test_r0_full_stack_updates_invalidate_the_cache_and_last_block_updates_keep_it(trained, splits):
    model = trained["B"][0]
    stream = splits.val.tokens[0].tolist()
    support = SupportSet.observed(splits.val.tokens[:8], source="held-out trajectories")
    outcome = {}
    for scope in ("all", "last"):
        with VolatileSession(model, SessionConfig(lr=0.1, max_steps=8, fast_scope=scope)) as session:
            session.observe(stream[:30])
            cache, version = session._cache, session._version
            assert all(r.applied for r in session.adapt_on_stream(steps=3))
            kept = session._cache is cache and session._version == version
            logits = session.next_logits()
            recomputed = session.forward(torch.tensor([stream[:30]]))[0, -1]
            assert torch.allclose(logits, recomputed, rtol=0, atol=2e-5)          # correct either way
            session.observe(stream[30:40])
            assert torch.allclose(session.next_logits(), session.forward(torch.tensor([stream[:40]]))[0, -1],
                                  rtol=0, atol=2e-5)
            names = session.fast_weight_names
            assert all(r.applied for r in session.adapt(support, steps=2))
            outcome[scope] = (kept, session.cache_rebuilds, len(names), session.volatile_bytes()["fast_weights"])
    assert outcome["all"][:3] == (False, 1, 2) and outcome["last"][:3] == (True, 0, 1)
    assert outcome["last"][3] * 2 == outcome["all"][3]
    assert names == (f"blocks.{model.config.n_layers - 1}.cgc.core.weight",)
    with pytest.raises(ValueError):
        SessionConfig(fast_scope="first")


def test_last_block_adaptation_learns_from_evidence_on_the_r0_task(trained, world):
    from research.elpis_torch.data import SINGLE, generate, rewire
    model = trained["B"][0]
    shifted = rewire(world, op=2, seed=0)
    evidence = generate(shifted, 8, 64, seed=100, p_double=0.0)
    held = generate(shifted, 64, 64, seed=101, p_double=0.0)
    hit = torch.zeros_like(held.kind, dtype=torch.bool)
    for state, op in shifted.changed_from(world):
        hit |= (held.state == state) & (held.op == op)
    changed = (hit & (held.kind == SINGLE))[:, 1:]
    support = SupportSet.observed(evidence.tokens, mask=evidence.target_mask(), source="shifted world")
    digest = parameter_digest(model)
    with VolatileSession(model, SessionConfig(lr=0.2, max_steps=16, fast_scope="last")) as session:
        before = session.loss(held.tokens, changed)
        assert all(r.applied for r in session.adapt(support, steps=16))
        assert session.loss(held.tokens, changed) < 0.8 * before
        session.reset()
        assert session.loss(held.tokens, changed) == before
    assert parameter_digest(model) == digest


# -- the scrolling session -----------------------------------------------------------------------------------

def test_a_scrolling_session_starts_at_the_checkpoint_and_refuses_what_it_cannot_honour():
    model = _frozen()
    with pytest.raises(SessionError):
        ScrollSession(model, SessionConfig(fast_scope="all"))            # would invalidate unrecomputable memory
    with pytest.raises(SessionError):
        ScrollSession(_model(), LAST)                                    # not frozen
    with pytest.raises(SessionError):
        ScrollSession(freeze(make_model("B")), LAST)                     # not a scrolling model
    with pytest.raises(SessionError):
        ScrollSession(model, LAST, learn_window=0)
    with ScrollSession(model, LAST) as session:
        assert session.fast_weight_names == ("blocks.1.cgc.core.weight",) and session.delta_norm() == 0.0
        nll = session.observe(STREAM[:70])
        assert math.isnan(nll[0]) and len(nll) == 70 and session.position == 70
        with torch.no_grad():
            reference = model.run(torch.tensor([STREAM[:70]]))[0][0, -1]
        assert torch.equal(session.next_logits(), reference)
        with pytest.raises(SessionError):
            session.observe([])


def test_prequential_loss_falls_with_learning_and_not_without():
    model = _frozen()
    tails = {}
    strong = SessionConfig(lr=5.0, max_steps=16, max_delta_ratio=5.0, fast_scope="last")
    for label, config in (("learning", strong), ("disabled", SessionConfig(learning_enabled=False, fast_scope="last"))):
        with ScrollSession(model, config, learn_window=128) as session:
            session.observe(STREAM[:150])
            results = session.adapt_on_stream(steps=16)
            tails[label] = statistics.mean(session.observe(STREAM[150:300]))
            tails[label + "_status"] = results[0].status
    assert tails["disabled_status"] is Status.REFUSED_LEARNING_DISABLED and tails["learning_status"] is Status.APPLIED
    # An UNTRAINED model: the last block's readout starts near zero, so its CGC has little leverage (measured
    # 5.56 -> 5.24 nats). The direction is what is asserted; how much a trained model gains is in EVIDENCE_R1.
    assert tails["learning"] < tails["disabled"] - 0.2


def test_only_observed_followers_in_the_bounded_window_teach():
    model = _frozen()
    with ScrollSession(model, LAST, learn_window=40) as session:
        session.observe(STREAM[:1])
        generated = session.generate(10)
        assert isinstance(generated, GeneratedTokens) and session.support_size() == 0
        assert [r.status for r in session.adapt_on_stream(steps=3)] == [Status.REFUSED_NO_OBSERVED_TARGET]
        with pytest.raises(SessionError):
            session.observe(generated)
        session.observe(STREAM[1:21])
        assert session.support_size() == 20                 # each observed token is the follower of one position
        session.generate(5)
        assert session.support_size() == 20 - 0 and session.position == 36
        size = session.volatile_bytes()
        session.observe(STREAM[21:200])                     # far more than the window holds
        assert session.support_size() == 39                 # the newest position has no follower yet
        assert session.volatile_bytes() == size             # nothing grew with the stream
        assert size["learn_window"] == 40 * (32 * 4 + 8 + 1)


def test_updates_are_bounded_refused_and_reset_exactly_as_in_r0():
    model = _frozen()
    with torch.no_grad():
        reference = model.run(torch.tensor([STREAM[:60]]))[0][0, -1]
    with ScrollSession(model, SessionConfig(lr=0.2, max_steps=2, fast_scope="last", grad_clip=0.01)) as session:
        session.observe(STREAM[:60])
        first, second, third = session.adapt_on_stream(steps=3)
        assert first.applied and first.clipped and first.delta_norm <= 0.2 * 0.01 * (1 + 1e-5)
        assert second.applied and third.status is Status.REFUSED_STEP_BUDGET and session.steps_used == 2
        assert not torch.equal(session.next_logits(), reference)
        session.reset()
        assert session.delta_norm() == 0.0 and session.position == 0 and session.support_size() == 0
        session.observe(STREAM[:60])
        assert torch.equal(session.next_logits(), reference)
    harsh = SessionConfig(lr=1e30, grad_clip=1e30, max_delta_ratio=1e30, fast_scope="last")
    with ScrollSession(model, harsh) as session:
        session.observe(STREAM[:60])
        assert session.adapt_on_stream()[0].status is Status.REFUSED_NONFINITE_CANDIDATE
        assert session.delta_norm() == 0.0 and torch.equal(session.next_logits(), reference)
    with ScrollSession(freeze(_model((1,), sharpen=False, dtype="float32", channel="ffn")), LAST) as session:
        session.observe(STREAM[:60])
        assert session.adapt_on_stream()[0].status is Status.REFUSED_NO_FAST_WEIGHTS


def test_sessions_are_isolated_unserializable_and_leave_the_checkpoint_alone():
    model = _frozen()
    digest, keys = parameter_digest(model), set(model.state_dict())
    one, two = ScrollSession(model, LAST), ScrollSession(model, LAST)
    one.observe(STREAM[:100])
    two.observe(STREAM[:100])
    untouched = two.next_logits()
    assert all(r.applied for r in one.adapt_on_stream(steps=4))
    assert torch.equal(two.next_logits(), untouched) and two.delta_norm() == 0.0
    assert not torch.equal(one.next_logits(), untouched)
    assert all(a is not b for a, b in zip(one.state.layers[0].values(), two.state.layers[0].values()))
    for attempt in (lambda: pickle.dumps(one), lambda: copy.deepcopy(one), lambda: torch.save(one, io.BytesIO())):
        with pytest.raises(SessionError):
            attempt()
    assert not {"save", "export", "state_dict", "checkpoint"} & set(dir(one))
    one.close()
    two.close()
    one.close()
    assert parameter_digest(model) == digest and set(model.state_dict()) == keys
    for call in (lambda: one.observe([1]), lambda: one.generate(1), one.adapt_on_stream, one.reset, one.next_logits):
        with pytest.raises(SessionError):
            call()


def test_a_scrolling_session_lifetime_writes_nothing(tmp_path):
    run = _scenario("--audit", "--scroll")
    assert run.returncode == 0, run.stderr
    record = json.loads(run.stdout.strip().splitlines()[-1])
    assert record["write_events"] == [] and record["applied_updates"] == 3 and record["position"] > 1000
    target = tmp_path / "leak"
    control = _scenario("--audit", "--scroll", "--negative-control", str(target))
    assert json.loads(control.stdout.strip().splitlines()[-1])["write_events"][0][1] == str(target)
