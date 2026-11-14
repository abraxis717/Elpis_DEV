"""Volatile CGC fast weights: evidence teaches them, bounds hold, nothing leaks, reset and restart are exact."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import pickle
import shutil
import statistics
import subprocess
import sys

import pytest

from .conftest import REPO, SKIP_REASON, make_model

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch.data import SINGLE, generate, rewire  # noqa: E402
from research.elpis_torch.model import (ContextLengthError, StaleCacheError, next_token_loss,  # noqa: E402
                                        parameter_digest)
from research.elpis_torch.session import (AdaptResult, BaseMutatedError, GeneratedTokens, Provenance,  # noqa: E402
                                          SessionConfig, SessionError, Status, SupportSet, VolatileSession)
from research.elpis_torch.training import freeze, load_checkpoint, save_checkpoint  # noqa: E402

SCENARIO = REPO / "tests" / "research" / "elpis_torch" / "zero_write_scenario.py"
LEARN = SessionConfig(lr=0.1, max_steps=16)


@pytest.fixture(scope="module")
def base(trained):
    return trained["B"][0]


@pytest.fixture(scope="module")
def shifted(world):
    """The session's world: operation 'c' rewired. The checkpoint was trained on the original world only."""
    other = rewire(world, op=2, seed=0)
    changed = other.changed_from(world)
    evidence = generate(other, 8, 64, seed=100, p_double=0.0)
    held_out = generate(other, 64, 64, seed=101, p_double=0.0)
    assert not {tuple(r.tolist()) for r in evidence.tokens} & {tuple(r.tolist()) for r in held_out.tokens}

    def split(batch):
        hit = torch.zeros_like(batch.kind, dtype=torch.bool)
        for state, op in changed:
            hit |= (batch.state == state) & (batch.op == op)
        hit &= batch.kind == SINGLE
        return hit[:, 1:], ((batch.kind == SINGLE) & ~hit)[:, 1:]

    return {"world": other, "changed": changed, "evidence": evidence, "held_out": held_out,
            "masks": split(held_out), "support": SupportSet.observed(evidence.tokens, mask=evidence.target_mask(),
                                                                      source="shifted-world evidence")}


def _snapshot(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _unchanged(model, snapshot):
    return all(torch.equal(p, snapshot[name]) for name, p in model.named_parameters())


# -- opening -----------------------------------------------------------------------------------------------

def test_a_session_opens_only_over_a_frozen_model_and_starts_at_the_checkpoint(base, splits):
    with pytest.raises(SessionError):
        VolatileSession(make_model("B"))                       # training mode, gradients enabled
    with pytest.raises(SessionError):
        VolatileSession(object())
    tokens = splits.val.tokens[:4]
    with VolatileSession(base, LEARN) as session:
        assert session.delta_norm() == 0.0 and session.steps_used == 0 and session.tokens == ()
        assert torch.equal(session.forward(tokens), base(tokens))
        assert len(session.fast_weight_names) == base.config.n_layers
        assert all(name.endswith("cgc.core.weight") for name in session.fast_weight_names)
        fast = sum(dict(base.named_parameters())[n].numel() for n in session.fast_weight_names)
        assert session.volatile_bytes()["fast_weights"] == 2 * 4 * fast      # delta and momentum, float32


@pytest.mark.parametrize("bad", [dict(lr=0.0), dict(lr=float("inf")), dict(grad_clip=-1.0), dict(max_delta_ratio=0.0),
                                 dict(momentum=1.0), dict(max_steps=-1), dict(max_steps=1.5)])
def test_unbounded_session_settings_are_refused(bad):
    with pytest.raises(ValueError):
        SessionConfig(**bad)


# -- the four demonstrations -------------------------------------------------------------------------------

def test_observed_evidence_changes_fast_weights_and_never_the_checkpoint(base, shifted):
    snapshot, digest = _snapshot(base), parameter_digest(base)
    with VolatileSession(base, LEARN) as session:
        results = session.adapt(shifted["support"], steps=4)
        assert [r.status for r in results] == [Status.APPLIED] * 4
        assert results[-1].delta_norm > 0 and session.delta_norm() == results[-1].delta_norm
        assert results[-1].loss_after < results[0].loss_before and session.steps_used == 4
        assert _unchanged(base, snapshot) and parameter_digest(base) == digest
        session.verify_base_unchanged()
    assert _unchanged(base, snapshot) and parameter_digest(base) == digest
    assert not any(p.requires_grad for p in base.parameters()) and all(p.grad is None for p in base.parameters())


def test_adaptation_improves_a_held_out_continuation_and_the_disabled_baseline_does_not(base, shifted):
    held, (changed, unchanged) = shifted["held_out"].tokens, shifted["masks"]
    with VolatileSession(base, LEARN) as learner, \
            VolatileSession(base, SessionConfig(learning_enabled=False)) as frozen:
        before = learner.loss(held, changed)
        assert before == frozen.loss(held, changed) > math.log(6)            # confidently wrong: the world moved
        results = learner.adapt(shifted["support"], steps=16)
        assert all(r.applied for r in results)
        refused = frozen.adapt(shifted["support"], steps=16)
        assert [r.status for r in refused] == [Status.REFUSED_LEARNING_DISABLED]
        after = learner.loss(held, changed)
        # Held-out trajectories of the same shifted world, none of them in the evidence.
        assert after < 0.5 * before
        assert frozen.loss(held, changed) == before and frozen.delta_norm() == 0.0
        # Interference is real and measured, not assumed away: transitions the world did NOT change get worse
        # (about 0.00 -> 0.3 nats in this small model), far less than the changed ones improve.
        assert frozen.loss(held, unchanged) < 0.05 < learner.loss(held, unchanged) < 1.0
        assert learner.loss(held, shifted["masks"][0] | shifted["masks"][1]) < frozen.loss(
            held, shifted["masks"][0] | shifted["masks"][1])


def test_a_stream_learns_from_its_observed_past_and_predicts_its_own_future_better(base, shifted):
    """Prequential: each continuation token is scored by the prediction made before it was seen."""
    learned, disabled = [], []
    for seed in range(16):
        batch = generate(shifted["world"], 1, 64, seed=1000 + seed, p_double=0.0)
        tokens, cut = batch.tokens[0].tolist(), 44
        scored = [i for i in range(cut, 64) if int(batch.kind[0, i]) == SINGLE
                  and (int(batch.state[0, i]), int(batch.op[0, i])) in shifted["changed"]]
        for config, sink in ((LEARN, learned), (SessionConfig(learning_enabled=False), disabled)):
            with VolatileSession(base, config) as session:       # a new session per stream: no carry-over
                assert math.isnan(session.observe(tokens[:cut])[0])
                session.adapt_on_stream(steps=8)
                nll = session.observe(tokens[cut:])
                sink += [nll[i - cut] for i in scored]
    assert len(learned) == len(disabled) >= 20
    assert statistics.mean(learned) < 0.75 * statistics.mean(disabled)


def test_reset_restores_the_initial_behavior_exactly(base, shifted, splits):
    tokens = splits.val.tokens[:4]
    reference = base(tokens)
    config = SessionConfig(lr=0.1, max_steps=3, momentum=0.9)
    with VolatileSession(base, config) as session:
        session.observe(shifted["evidence"].tokens[0, :20].tolist())
        assert all(r.applied for r in session.adapt(shifted["support"], steps=3))
        assert session.adapt(shifted["support"])[0].status is Status.REFUSED_STEP_BUDGET
        assert not torch.equal(session.forward(tokens), reference)
        session.reset()
        assert torch.equal(session.forward(tokens), reference)
        assert session.delta_norm() == 0.0 and session.steps_used == 0 and session.tokens == ()
        assert all(float(v.abs().sum()) == 0.0 for v in session._velocity.values())     # optimizer state is gone
        assert session.volatile_bytes()["kv_cache"] == 0
        first = session.adapt(shifted["support"])[0]                   # the budget is restored too
        assert first.applied and first.steps_used == 1
    with VolatileSession(base, config) as fresh:       # a reset session is indistinguishable from a new one
        again = fresh.adapt(shifted["support"])[0]
    assert (again.loss_before, again.loss_after, again.delta_norm) == (first.loss_before, first.loss_after,
                                                                       first.delta_norm)


def test_restarting_from_the_checkpoint_reproduces_parameters_and_predictions(base, shifted, splits, tmp_path):
    path = tmp_path / "base.checkpoint"
    save_checkpoint(base, path)                                      # explicit, offline, before any session
    file_digest, stat = hashlib.sha256(path.read_bytes()).hexdigest(), path.stat()
    tokens = splits.val.tokens[:4]
    first = load_checkpoint(path)
    initial_digest, initial_logits = parameter_digest(first), first(tokens)
    session = VolatileSession(first, LEARN)
    session.observe(shifted["evidence"].tokens[0].tolist())
    assert all(r.applied for r in session.adapt(shifted["support"], steps=6))
    session.close()
    after = path.stat()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == file_digest
    assert (after.st_mtime_ns, after.st_size, after.st_ino) == (stat.st_mtime_ns, stat.st_size, stat.st_ino)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["base.checkpoint"]
    restarted = load_checkpoint(path)
    assert parameter_digest(restarted) == initial_digest == parameter_digest(first) == parameter_digest(base)
    assert torch.equal(restarted(tokens), initial_logits)
    with VolatileSession(restarted, LEARN) as new_session:
        assert new_session.delta_norm() == 0.0 and torch.equal(new_session.forward(tokens), initial_logits)


# -- isolation ---------------------------------------------------------------------------------------------

def test_sessions_over_one_model_share_nothing_mutable(base, shifted, world, splits):
    tokens, reference = splits.val.tokens[:4], base(splits.val.tokens[:4])
    with VolatileSession(base, LEARN) as one, VolatileSession(base, LEARN) as two:
        one.adapt(shifted["support"], steps=4)
        assert torch.equal(two.forward(tokens), reference) and two.delta_norm() == 0.0 and two.steps_used == 0
        assert torch.equal(base(tokens), reference)                  # the shared model itself is untouched
        ones = one.forward(tokens)
        assert not torch.equal(ones, reference)
        other = generate(rewire(world, op=0, seed=3), 8, 64, seed=7, p_double=0.0)
        two.adapt(SupportSet.observed(other.tokens, mask=other.target_mask(), source="another world"), steps=4)
        assert torch.equal(one.forward(tokens), ones)                 # two's learning did not reach one
        assert not torch.equal(two.forward(tokens), ones)
        assert all(a is not b for a, b in zip(one._delta.values(), two._delta.values()))
        with pytest.raises(StaleCacheError):                          # nor can one's cache serve two
            two.forward(tokens[:, :1], one.new_cache(4))
    assert torch.equal(base(tokens), reference)


def test_without_cgc_there_is_nothing_to_adapt(trained, shifted, splits):
    model = trained["A"][0]
    with VolatileSession(model, LEARN) as session:
        assert session.fast_weight_names == ()
        assert session.adapt(shifted["support"])[0].status is Status.REFUSED_NO_FAST_WEIGHTS
        assert torch.equal(session.forward(splits.val.tokens[:2]), model(splits.val.tokens[:2]))
        session.observe([256, 65, 97, 61])
        assert len(session.generate(3)) == 3


def test_the_augmented_model_adapts_only_its_cgc_cores(trained, shifted):
    model = trained["C"][0]
    with VolatileSession(model, SessionConfig(lr=1.0, max_steps=32)) as session:
        assert len(session.fast_weight_names) == model.config.n_layers
        before = session.loss(shifted["held_out"].tokens, shifted["masks"][0])
        assert all(r.applied for r in session.adapt(shifted["support"], steps=16))
        assert session.loss(shifted["held_out"].tokens, shifted["masks"][0]) < before


# -- bounds, refusal and rollback --------------------------------------------------------------------------

def test_every_update_is_clipped_projected_and_budgeted(base, shifted):
    norms = {n: float(p.norm()) for n, p in base.named_parameters() if n.endswith("cgc.core.weight")}
    with VolatileSession(base, SessionConfig(lr=0.5, grad_clip=0.01, max_delta_ratio=10.0, max_steps=2)) as session:
        first = session.adapt(shifted["support"])[0]
        assert first.applied and first.clipped and not first.projected
        assert first.delta_norm <= 0.5 * 0.01 * (1 + 1e-5)             # one step moves at most lr * grad_clip
        second, third = session.adapt(shifted["support"], steps=2)
        assert second.applied and third.status is Status.REFUSED_STEP_BUDGET and session.steps_used == 2
    with VolatileSession(base, SessionConfig(lr=5.0, grad_clip=1e9, max_delta_ratio=0.01, max_steps=8)) as session:
        results = session.adapt(shifted["support"], steps=8)
        assert all(r.applied for r in results) and any(r.projected for r in results) and not results[0].clipped
        for name, delta in session._delta.items():
            assert float(delta.norm()) <= 0.01 * norms[name] * (1 + 1e-5)


def _state(session):
    return ({n: d.clone() for n, d in session._delta.items()}, session.steps_used, session._version,
            session._cache, session.tokens)


def _same(session, state):
    deltas, steps, version, cache, tokens = state
    return (all(torch.equal(session._delta[n], d) for n, d in deltas.items()) and session.steps_used == steps
            and session._version == version and session._cache is cache and session.tokens == tokens)


def test_a_non_finite_support_loss_is_refused_and_changes_nothing(shifted):
    model = make_model("B")
    with torch.no_grad():
        model.blocks[1].cgc.core.weight[0, 0, 0] = float("nan")
    with VolatileSession(freeze(model), LEARN) as session:
        state = _state(session)
        result = session.adapt(shifted["support"], steps=3)
        assert [r.status for r in result] == [Status.REFUSED_NONFINITE_LOSS] and _same(session, state)


def test_a_candidate_that_breaks_the_model_is_rolled_back(shifted, splits):
    """Unbounded (faithful) cubic cells and an absurd step: the candidate overflows and is never committed."""
    model = freeze(make_model("B", cgc_preact_bound=None))
    tokens = splits.val.tokens[:2]
    huge = dict(grad_clip=1e30, max_delta_ratio=1e30)
    with VolatileSession(model, SessionConfig(lr=1e17, **huge)) as session:
        session.observe(shifted["evidence"].tokens[0, :12].tolist())
        state, reference = _state(session), session.forward(tokens)
        result = session.adapt(shifted["support"], steps=3)
        # The candidate weights are finite, but the cubic of their pre-activations is not.
        assert [r.status for r in result] == [Status.ROLLED_BACK_NONFINITE_AFTER]
        assert math.isfinite(result[0].loss_before) and not math.isfinite(result[0].loss_after)
        assert _same(session, state) and torch.equal(session.forward(tokens), reference)
        assert bool(torch.isfinite(session.next_logits()).all())
    with VolatileSession(model, SessionConfig(lr=1e30, **huge)) as session:
        # Here the candidate's own norm overflows: refused, never rescaled into something that looks valid.
        state = _state(session)
        assert [r.status for r in session.adapt(shifted["support"], steps=3)] == [Status.REFUSED_NONFINITE_CANDIDATE]
        assert _same(session, state)
    with VolatileSession(freeze(make_model("B")), SessionConfig(lr=1e17, **huge)) as session:
        # The bounded cells survive the same step: the pre-activation bound, not luck, keeps the forward finite.
        result = session.adapt(shifted["support"])[0]
        assert result.applied and math.isfinite(result.loss_after)


def test_a_step_that_raises_the_support_loss_can_be_rolled_back(base, shifted):
    harsh = dict(lr=50.0, grad_clip=1e9, max_delta_ratio=5.0)
    with VolatileSession(base, SessionConfig(**harsh, reject_loss_increase=True)) as session:
        state = _state(session)
        result = session.adapt(shifted["support"])[0]
        assert result.status is Status.ROLLED_BACK_LOSS_INCREASED and result.loss_after > result.loss_before
        assert _same(session, state)
    with VolatileSession(base, SessionConfig(**harsh)) as session:    # without the option the same step commits
        assert session.adapt(shifted["support"])[0].applied


def test_non_finite_gradients_and_candidates_are_refused_under_fault_injection(base, shifted, monkeypatch):
    with VolatileSession(base, SessionConfig(lr=0.2, momentum=0.5)) as session:
        state = _state(session)
        for name in session._velocity:                                 # poison the optimizer state
            session._velocity[name].fill_(float("inf"))
        assert session.adapt(shifted["support"])[0].status is Status.REFUSED_NONFINITE_CANDIDATE
        assert _same(session, state)
        session.reset()
        state, real = _state(session), torch.autograd.grad
        monkeypatch.setattr(torch.autograd, "grad",
                            lambda *a, **k: tuple(g * float("nan") for g in real(*a, **k)))
        assert session.adapt(shifted["support"])[0].status is Status.REFUSED_NONFINITE_GRADIENT
        monkeypatch.undo()
        assert _same(session, state)
        assert session.adapt(shifted["support"])[0].applied            # and the session is still usable


# -- provenance: what may be a target -----------------------------------------------------------------------

def test_generated_tokens_are_context_and_never_targets(base, shifted):
    observed = shifted["evidence"].tokens[0, :17].tolist()
    with VolatileSession(base, LEARN) as session:
        session.observe(observed[:1])
        generated = session.generate(6)
        assert isinstance(generated, GeneratedTokens) and len(generated) == 6
        assert session.provenance == (Provenance.OBSERVED,) + (Provenance.GENERATED,) * 6
        # Only generated tokens follow the first one: there is no observed target at all.
        assert [r.status for r in session.adapt_on_stream(steps=3)] == [Status.REFUSED_NO_OBSERVED_TARGET]
        assert session.delta_norm() == 0.0
        with pytest.raises(SessionError):
            session.observe(generated)
        with pytest.raises(SessionError):
            SupportSet.observed(generated, source="laundered")
    with VolatileSession(base, LEARN) as stream, VolatileSession(base, LEARN) as masked, \
            VolatileSession(base, LEARN) as naive:
        stream.observe(observed)
        generated = list(stream.generate(8))
        stream.observe(shifted["evidence"].tokens[0, 17:20].tolist())
        sequence = torch.tensor([list(stream.tokens)])
        only_observed = torch.tensor([[p is Provenance.OBSERVED for p in stream.provenance[1:]]])
        assert int(only_observed.sum()) == 16 + 3 and sequence.shape[1] == 17 + 8 + 3
        a = stream.adapt_on_stream()[0]
        b = masked.adapt(SupportSet.observed(sequence, mask=only_observed, source="same stream, observed targets"))[0]
        c = naive.adapt(SupportSet.observed(sequence, source="same stream, every token a target"))[0]
        # The stream update is exactly the update that scores observed targets only ...
        assert a.applied and (a.loss_before, a.grad_norm) == (b.loss_before, b.grad_norm)
        assert all(torch.equal(stream._delta[n], masked._delta[n]) for n in stream._delta)
        # ... and not the one that would also have learned from the session's own samples.
        assert c.loss_before != a.loss_before
        assert not all(torch.equal(stream._delta[n], naive._delta[n]) for n in stream._delta)


def test_a_support_set_must_be_declared_and_well_formed():
    tokens = torch.zeros(2, 5, dtype=torch.long)
    for bad in (dict(source=""), dict(source="x", mask=torch.ones(2, 5, dtype=torch.bool)),
                dict(source="x", mask=torch.ones(2, 4))):
        with pytest.raises(SessionError):
            SupportSet.observed(tokens, **bad)
    with pytest.raises(SessionError):
        SupportSet.observed(torch.zeros(5, dtype=torch.long), source="x")
    with pytest.raises(TypeError):
        SupportSet.observed(tokens)


# -- attention-cache coherence -------------------------------------------------------------------------------

def test_an_update_invalidates_the_cache_and_the_stream_is_recomputed_under_the_new_weights(base, shifted):
    tokens = shifted["evidence"].tokens[0].tolist()
    with VolatileSession(base, LEARN) as session:
        session.observe(tokens[:30])
        stale, user_cache = session._cache, session.new_cache(1)
        session.forward(torch.tensor([tokens[:30]]), user_cache)
        assert stale.length == 30 and session.cache_rebuilds == 0
        stale_logits = session.next_logits()
        assert all(r.applied for r in session.adapt_on_stream(steps=4))
        assert session._cache is None                                  # dropped at commit
        # The next read rebuilds the cache from the stream's tokens under the new weights ...
        fresh = session.next_logits()
        full = session.forward(torch.tensor([tokens[:30]]))[0, -1]
        assert session.cache_rebuilds == 1 and session._cache is not stale and session._cache.length == 30
        assert torch.allclose(fresh, full, rtol=0, atol=2e-5) and not torch.allclose(fresh, stale_logits, atol=1e-3)
        # ... and incremental decoding continues in parity with a from-scratch forward pass.
        session.observe(tokens[30:40])
        session.generate(5)
        assert session.cache_rebuilds == 1
        again = session.forward(torch.tensor([list(session.tokens)]))[0, -1]
        assert torch.allclose(session.next_logits(), again, rtol=0, atol=2e-5)
        # Both caches written before the update are refused outright.
        for old in (stale, user_cache):
            with pytest.raises(StaleCacheError):
                session.forward(torch.tensor([tokens[30:31]]), old)
        # Negative control: forge the tag and the stale cache silently gives the wrong answer. This is the
        # failure the tag exists to prevent.
        stale.weights_tag = session._tag
        forged = session.forward(torch.tensor([tokens[30:31]]), stale)[0, -1]
        correct = session.forward(torch.tensor([tokens[:31]]))[0, -1]
        assert float((forged - correct).abs().max()) > 1e-2


def test_a_refused_update_keeps_the_cache_and_a_reset_drops_it(base, shifted):
    tokens = shifted["evidence"].tokens[0].tolist()
    with VolatileSession(base, SessionConfig(lr=0.2, max_steps=0)) as session:
        session.observe(tokens[:10])
        cache = session._cache
        assert session.adapt_on_stream()[0].status is Status.REFUSED_STEP_BUDGET
        assert session._cache is cache and session.cache_rebuilds == 0
        session.reset()
        with pytest.raises(StaleCacheError):
            session.forward(torch.tensor([tokens[10:11]]), cache)
        with pytest.raises(SessionError):
            session.next_logits()                                      # the stream is empty again


def test_the_stream_is_bounded_by_the_context_length(base, shifted):
    tokens = shifted["evidence"].tokens[0].tolist()
    with VolatileSession(base, LEARN) as session:
        session.observe(tokens[:60])
        with pytest.raises(ContextLengthError):
            session.observe(tokens[:5])
        with pytest.raises(ContextLengthError):
            session.generate(5)
        assert len(session.tokens) == 60                               # refused whole: nothing was appended
        assert len(session.generate(4)) == 4 and len(session.tokens) == 64
        assert session.volatile_bytes()["kv_cache"] == 2 * 2 * 1 * 16 * 64 * 4


# -- nothing is exported -------------------------------------------------------------------------------------

def test_a_session_cannot_be_serialized_copied_or_found_in_the_model(base, shifted):
    keys, state = set(base.state_dict()), {k: v.clone() for k, v in base.state_dict().items()}
    session = VolatileSession(base, LEARN)
    session.adapt(shifted["support"], steps=2)
    for attempt in (lambda: pickle.dumps(session), lambda: copy.deepcopy(session), lambda: copy.copy(session),
                    lambda: torch.save(session, io.BytesIO())):
        with pytest.raises(SessionError):
            attempt()
    assert not {"save", "export", "state_dict", "load_state_dict", "checkpoint"} & set(dir(session))
    assert set(base.state_dict()) == keys and all(torch.equal(base.state_dict()[k], v) for k, v in state.items())
    assert not list(base.buffers(recurse=True)) or all(not b.requires_grad for b in base.buffers())
    session.close()
    assert session._delta == {} and session._cache is None and session.tokens == ()


def test_a_closed_session_refuses_everything_and_close_is_idempotent(base, shifted):
    session = VolatileSession(base, LEARN)
    session.close()
    session.close()
    tokens = torch.zeros(1, 4, dtype=torch.long)
    for call in (lambda: session.forward(tokens), lambda: session.loss(tokens), lambda: session.observe([1]),
                 lambda: session.generate(1), lambda: session.adapt(shifted["support"]), session.adapt_on_stream,
                 session.reset, session.next_logits, lambda: session.new_cache(1)):
        with pytest.raises(SessionError):
            call()
    with pytest.raises(RuntimeError, match="boom"):
        with VolatileSession(base, LEARN) as failing:
            failing.adapt(shifted["support"])
            raise RuntimeError("boom")
    assert failing._closed and failing._delta == {}


def test_closing_detects_a_checkpoint_that_was_mutated_under_the_session():
    model = freeze(make_model("B"))
    session = VolatileSession(model, LEARN)
    with torch.no_grad():
        model.blocks[0].cgc.core.weight[0, 0, 0] += 1.0
    with pytest.raises(BaseMutatedError):
        session.verify_base_unchanged()
    with pytest.raises(BaseMutatedError):
        session.close()


def test_adapt_results_are_plain_records():
    result = AdaptResult(Status.REFUSED_STEP_BUDGET)
    assert not result.applied and math.isnan(result.loss_before) and "budget" in result.status.value
    assert AdaptResult(Status.APPLIED).applied and {s.name for s in Status if "REFUSED" in s.name or "ROLLED" in s.name} \
        == {s.name for s in Status} - {"APPLIED"}


# -- zero writes ---------------------------------------------------------------------------------------------

def _scenario(*args, prefix=()):
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1",
           "HOME": os.environ.get("HOME", "/")}
    return subprocess.run([*prefix, sys.executable, "-B", str(SCENARIO), *args], capture_output=True, text=True,
                          env=env, cwd=str(REPO), timeout=300)


def test_a_session_lifetime_performs_no_python_level_filesystem_write(tmp_path):
    run = _scenario("--audit")
    assert run.returncode == 0, run.stderr
    record = json.loads(run.stdout.strip().splitlines()[-1])
    assert record == {"applied_updates": 4, "write_events": []}
    target = tmp_path / "leak"
    control = _scenario("--audit", "--negative-control", str(target))
    events = json.loads(control.stdout.strip().splitlines()[-1])["write_events"]
    assert events and events[0][1] == str(target) and target.read_text() == "not volatile"   # the hook can see


WRITE_CALLS = ("creat(", "rename", "unlink", "mkdir", "truncate", "link", "symlink", "mknod", "fsync", "fdatasync",
               "msync", "sendfile", "copy_file_range", "fallocate", "memfd_create")
WRITE_OPEN = ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND")
TRACED = ("openat,open,creat,write,pwrite64,writev,pwritev,rename,renameat,renameat2,unlink,unlinkat,mkdir,mkdirat,"
          "truncate,ftruncate,link,linkat,symlink,symlinkat,mknod,mknodat,fsync,fdatasync,msync,sendfile,"
          "copy_file_range,fallocate,memfd_create")


def _call(line: str) -> str:
    """The system call of one ``strace -f`` line. The pid column is padded, so split on any run of blanks."""
    parts = line.split(None, 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else line


def test_the_trace_parser_reads_lines_of_any_pid_width():
    """A 4-digit pid is followed by two blanks. Misreading that would make the zero-write gate vacuous."""
    for pid in ("7", "3076", "13227", "1234567"):
        pad = " " * max(1, 6 - len(pid))
        trace = "\n".join([f'{pid}{pad}write(2</dev/pts/0>, "ELPIS_SESSION_BEGIN\\n", 20) = 20',
                           f'{pid}{pad}openat(AT_FDCWD</r>, "/x/leak", O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC, 0666) = 3',
                           f'{pid}{pad}openat(AT_FDCWD</r>, "/x/read", O_RDONLY|O_CLOEXEC) = 4</x/read>',
                           f'{pid}{pad}write(3</x/leak>, "not volatile", 12) = 12',
                           f'{pid}{pad}unlinkat(AT_FDCWD</r>, "/x/leak", 0) = 0',
                           f'{pid}{pad}write(2</dev/pts/0>, "ELPIS_SESSION_END\\n", 18) = 18'])
        seen = [_call(line).split("(", 1)[0] for line in _write_class(trace)]
        assert seen == ["openat", "write", "unlinkat"], (pid, seen)


def _write_class(trace: str) -> list[str]:
    """Write-class system calls between the session markers (the markers themselves go to stderr, fd 2)."""
    lines = trace.splitlines()
    begin = next(i for i, line in enumerate(lines) if "ELPIS_SESSION_BEGIN" in line)
    end = next(i for i, line in enumerate(lines) if "ELPIS_SESSION_END" in line)
    out = []
    for line in lines[begin + 1:end]:
        call = _call(line)
        if call.startswith(("openat(", "open(")):
            if any(flag in call for flag in WRITE_OPEN):
                out.append(line)
        elif call.startswith(("write(", "pwrite64(", "writev(", "pwritev(")) or call.startswith(WRITE_CALLS):
            out.append(line)
    return out


def test_a_session_lifetime_issues_no_write_class_system_call(tmp_path):
    strace = shutil.which("strace")
    usable = strace and subprocess.run([strace, "-o", os.devnull, "true"], capture_output=True).returncode == 0
    if not usable:
        if os.environ.get("ELPIS_REQUIRE_TORCH") == "1":
            pytest.fail("strace is required for the system-call zero-write gate (never an implicit PASS)")
        pytest.skip("strace is not available or ptrace is not permitted here; not a PASS")
    trace = tmp_path / "session.trace"
    prefix = (strace, "-f", "-y", "-o", str(trace), "-e", f"trace={TRACED}")
    run = _scenario(prefix=prefix)
    assert run.returncode == 0, run.stderr
    assert _write_class(trace.read_text()) == []
    target = tmp_path / "leak"
    control = _scenario("--negative-control", str(target), prefix=prefix)
    assert control.returncode == 0, control.stderr
    seen = _write_class(trace.read_text())
    assert any(str(target) in line and "O_WRONLY" in line for line in seen)                  # the trace can see
    assert any(_call(line).startswith("write(") and str(target) in line for line in seen)
