"""Turn conditioning in, turn observation out, on the principal DSV4.1 path.

Admitted conditioning is frozen per sequence, preprojected once by the
target-bound projection, and added to every token embedding: on the host for
NumPy and DSV41NativeBackend, on the provider for YTS-R0 (sent once at
STREAM_OPEN). Native and provider stay bitwise identical, NumPy within the
established tolerance. Without conditioning nothing changes.
"""
from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

from elpis.identity import content_digest
from elpis.inference.conditioning import OBSERVATION_FEATURES, TurnConditioning, TurnObservation
from elpis.inference.contracts import Code, ContractError, identity
from elpis.inference.drivers.dsv41 import stream_protocol as P
from elpis.inference.drivers.dsv41.fixtures import fixture_conditioning_projection
from elpis.inference.drivers.dsv41.parameters import ConditioningProjection
from elpis.inference.principal import PrincipalCommit, PrincipalEngine, PrincipalRequest, observe_distribution
from elpis.inference.target import Tensor

from . import provider_harness as H
from .test_provider_stream import assert_same_run, balanced, run
from .test_tower import CONTEXT, admission

WIDTH = 83  # S3 width for a d=6 world model; inference only sees a bounded vector


def _conditioning(seed=0, provenance="e" * 64):
    return TurnConditioning(tuple(((i * 37 + seed * 11) % 23 - 11) / 7.0 for i in range(WIDTH)), provenance)


@pytest.fixture
def rig(native_workspace, v41):
    rig = H.make_rig(native_workspace, v41)
    yield rig
    rig.close()


# ---------------------------------------------------------------------------
# contracts (no native code)
# ---------------------------------------------------------------------------

def test_turn_conditioning_contract():
    c = _conditioning()
    assert c.digest == _conditioning().digest != _conditioning(1).digest
    assert _conditioning(provenance="f" * 64).digest != c.digest
    for bad in ((), [1.0], (1,), (float("nan"),), (float("inf"),), tuple(0.0 for _ in range(4097))):
        with pytest.raises(ContractError):
            TurnConditioning(bad, "e" * 64)
    with pytest.raises(ContractError):
        TurnConditioning((1.0,), "not-a-digest")
    with pytest.raises(Exception):
        c.values = (0.0,)  # frozen


def test_turn_observation_is_token_identity_free_and_bounded():
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 3, 257).astype(np.float32)
    features = observe_distribution(logits)
    assert len(features) == OBSERVATION_FEATURES and all(type(v) is float for v in features)
    assert features == observe_distribution(logits.copy())  # deterministic
    # Values only, never indices: a vocabulary permutation changes nothing beyond summation order.
    np.testing.assert_allclose(observe_distribution(logits[rng.permutation(logits.size)]), features,
                               rtol=0, atol=1e-12)
    assert list(features[:6]) == sorted(features[:6], reverse=True)
    assert abs(sum(features[:7]) - 1.0) < 1e-12 and 0.0 <= features[7] <= 1.0
    uniform = observe_distribution(np.zeros(64, dtype=np.float32))
    assert abs(uniform[7] - 1.0) < 1e-12
    masked = logits.copy()
    masked[::2] = -np.inf  # masked vocabulary entries are allowed
    assert observe_distribution(masked)[0] > features[0]
    for bad in ([0.0, np.nan], [0.0, np.inf], [-np.inf, -np.inf]):
        with pytest.raises(ContractError):
            observe_distribution(np.array(bad))
    with pytest.raises(ContractError):
        TurnObservation("a" * 64, features[:7], 1, "COMPLETE")


def test_unconditioned_commit_keeps_its_pre_integration_identity():
    names = [f.name for f in fields(PrincipalCommit)]
    assert names[-1] == "conditioning"
    values = dict(state="a" * 64, admission="b" * 64, request="c" * 64, model="d" * 64, numerical_profile="e" * 64,
                  prefill=3, outputs=(1, 2), stop_reason="COMPLETE", trace="f" * 64, working_set=(8, 16))
    commit = PrincipalCommit(**values)
    # Exactly the digest the ten-field record had before the conditioning field existed.
    assert commit.digest == identity("principal-commit", values)
    conditioned = PrincipalCommit(**values, conditioning=_conditioning().digest)
    assert conditioned.digest != commit.digest
    with pytest.raises(ContractError):
        PrincipalCommit(**values, conditioning="nope")


# ---------------------------------------------------------------------------
# DSV4.1: NumPy / native / provider
# ---------------------------------------------------------------------------

def test_projection_is_explicit_target_bound_and_validated(rig):
    projection = fixture_conditioning_projection(rig.config, WIDTH)
    plain = rig.native_target(rig.fms(), "plain")
    conditioned = rig.native_target(rig.fms(), "conditioned", conditioning_projection=projection)
    assert not plain.accepts_conditioning and conditioned.accepts_conditioning
    # Only a conditioned target changes identity; the plain identity formula is untouched.
    assert plain.model_identity == content_digest("elpis.inference.dsv41.target.v1", dict(
        manifest=plain.store.manifest.digest, architecture="elpis.inference.dsv41.tower-spec.v1"))
    assert conditioned.model_identity != plain.model_identity
    other = fixture_conditioning_projection(rig.config, WIDTH, seed=1)
    assert rig.native_target(rig.fms(), "other", conditioning_projection=other).model_identity != \
        conditioned.model_identity
    bad_shape = ConditioningProjection(rig.config.model, Tensor((rig.config.dimension + 1, WIDTH),
                                                                 bytes(4 * (rig.config.dimension + 1) * WIDTH)))
    bad_model = ConditioningProjection("0" * 64, projection.tensor)
    for bad in (bad_shape, bad_model):
        with pytest.raises(ContractError) as info:
            rig.native_target(rig.fms(), "bad", conditioning_projection=bad)
        assert info.value.code == Code.IDENTITY
    vector = conditioned.conditioning_vector(_conditioning())
    assert vector.dtype == np.float32 and vector.shape == (rig.config.dimension,) and not vector.flags.writeable
    with pytest.raises(ContractError):
        conditioned.conditioning_vector(TurnConditioning((1.0,) * (WIDTH - 1), "e" * 64))


def test_conditioned_tower_numpy_native_provider_parity(rig):
    """Conditioned runs: provider bitwise == native, NumPy within tolerance; conditioning is really applied."""
    projection = fixture_conditioning_projection(rig.config, WIDTH)
    conditioning = _conditioning()
    tokens = rig.tokens(24)
    native_target = rig.native_target(rig.fms(), "native", conditioning_projection=projection)
    numpy_target = rig.numpy_target(rig.fms(), "numpy", conditioning_projection=projection)
    provider = rig.provider()
    provider_target = rig.provider_target(rig.fms(), "provider", provider, conditioning_projection=projection)
    assert provider.features & P.FEATURE_CONDITIONING

    _, native = run(native_target, tokens, native_target.window_initial(conditioning=conditioning))
    _, numpy_ref = run(numpy_target, tokens, numpy_target.window_initial(conditioning=conditioning))
    state, ours = run(provider_target, tokens, provider_target.window_initial(conditioning=conditioning))
    assert_same_run(native, ours)
    for position, (ref, mine) in enumerate(zip(numpy_ref, ours)):
        np.testing.assert_allclose(mine["logits"], ref["logits"], rtol=H.RTOL, atol=H.ATOL, err_msg=str(position))
        np.testing.assert_allclose(mine["streams"], ref["streams"], rtol=H.RTOL, atol=H.ATOL, err_msg=str(position))
        assert int(np.argmax(mine["logits"])) == int(np.argmax(ref["logits"]))
    provider_target.release_window(state)

    # The same targets without conditioning are bitwise the plain (projection-free) targets.
    plain_target = rig.native_target(rig.fms(), "plain")
    _, plain = run(plain_target, tokens)
    _, unconditioned = run(native_target, tokens)
    assert_same_run(plain, unconditioned)
    state, provider_plain = run(provider_target, tokens)
    assert_same_run(plain, provider_plain)
    provider_target.release_window(state)
    assert not all(H.bitwise_equal(a["logits"], b["logits"]) for a, b in zip(plain, native))
    provider.close()
    assert balanced(provider) and all(v == 0 for v in rig.live().values())


def test_conditioning_travels_once_at_stream_open_never_per_token(rig, monkeypatch):
    projection = fixture_conditioning_projection(rig.config, WIDTH)
    provider = rig.provider()
    target = rig.provider_target(rig.fms(), "provider", provider, conditioning_projection=projection)
    traffic = []
    exchange = provider.exchange

    def recording(kind, **kwargs):
        traffic.append((kind, len(kwargs.get("body", b"")) + kwargs.get("tail", 0)))
        return exchange(kind, **kwargs)

    monkeypatch.setattr(provider, "exchange", recording)
    tokens = rig.tokens(12)
    runs = {}
    for name, conditioning in (("plain", None), ("conditioned", _conditioning())):
        traffic.clear()
        state = target.window_initial(conditioning=conditioning)
        if conditioning is not None:
            assert not state.conditioning.flags.writeable
            with pytest.raises(ValueError):
                state.conditioning[0] = 0.0
        state, snaps = run(target, tokens, state)
        target.release_window(state)
        runs[name] = (list(traffic), snaps)
    (plain, _), (conditioned, _) = runs["plain"], runs["conditioned"]
    D = rig.config.dimension
    assert plain[0] == (P.STREAM_OPEN, 8) and conditioned[0] == (P.STREAM_OPEN, 16 + 4 * D)
    assert plain[1:] == conditioned[1:]  # every token transaction is byte-for-byte the same size
    assert sum(kind == P.STREAM_OPEN for kind, _ in conditioned) == 1
    provider.close()


def test_principal_conditioned_commit_replay_and_observation(rig):
    v41 = rig.v41
    from elpis.inference.text import ChatMessage
    projection = fixture_conditioning_projection(rig.config, WIDTH)
    native_target = rig.native_target(rig.fms(), "native", conditioning_projection=projection)
    provider = rig.provider()
    provider_target = rig.provider_target(rig.fms(), "provider", provider, conditioning_projection=projection)
    prompt = v41.encode_chat((ChatMessage("user", "Hello, Elpis."),))

    def generate(target, conditioning):
        engine = PrincipalEngine(target)
        state = engine.initial(CONTEXT)
        request = PrincipalRequest("conditioned", prompt, 12, v41.stop_tokens)
        context = admission(target, v41)
        sequence = engine.begin(state, request, context, expected_state=state.digest, conditioning=conditioning)
        list(sequence)
        return engine, state, request, context, engine.finalize(sequence)

    conditioning = _conditioning()
    *_, native = generate(native_target, conditioning)
    engine, state, request, context, ours = generate(provider_target, conditioning)
    *_, plain = generate(provider_target, None)
    assert ours.commit.outputs == native.commit.outputs and ours.commit.trace == native.commit.trace
    assert ours.observation.features == native.observation.features
    assert ours.commit.conditioning == native.commit.conditioning == conditioning.digest
    assert plain.commit.conditioning is None and plain.commit.digest != ours.commit.digest
    obs = ours.observation
    assert type(obs) is TurnObservation and obs.commit == ours.commit.digest
    assert obs.outputs == len(ours.commit.outputs) and obs.stop_reason == ours.commit.stop_reason
    assert engine.replay(state, request, context, ours.commit, conditioning=conditioning) == ours
    for wrong in (None, _conditioning(1)):
        with pytest.raises(ContractError) as info:
            engine.replay(state, request, context, ours.commit, conditioning=wrong)
        assert info.value.code == Code.IDENTITY
    assert provider.stream is None
    provider.close()
    assert balanced(provider) and all(v == 0 for v in rig.live().values())


def test_failed_or_refused_conditioning_yields_no_commit_and_no_observation(rig):
    v41 = rig.v41
    plain_target = rig.native_target(rig.fms(), "plain")
    engine = PrincipalEngine(plain_target)
    state = engine.initial(CONTEXT)
    request = PrincipalRequest("refused", v41.encode("Hi"), 4)
    sequence = engine.begin(state, request, admission(plain_target, v41), expected_state=state.digest,
                            conditioning=_conditioning())
    assert sequence.done and sequence.stop_reason == "FAILED"
    result = engine.finalize(sequence)
    assert result.commit is None and result.observation is None and result.failure == Code.UNSUPPORTED.value
    projection = fixture_conditioning_projection(rig.config, WIDTH)
    target = rig.native_target(rig.fms(), "conditioned", conditioning_projection=projection)
    engine = PrincipalEngine(target)
    state = engine.initial(CONTEXT)
    narrow = TurnConditioning((0.5,) * (WIDTH - 1), "e" * 64)
    sequence = engine.begin(state, request, admission(target, v41), expected_state=state.digest, conditioning=narrow)
    result = engine.finalize(sequence)
    assert result.commit is None and result.observation is None and result.failure == Code.ENCODING.value
    # An unconditioned run of a closed (not exhausted) sequence still commits with an observation.
    sequence = engine.begin(state, request, admission(target, v41), expected_state=state.digest)
    sequence.next()
    sequence.close()
    result = engine.finalize(sequence)
    assert result.commit is not None and result.observation is not None
    assert result.observation.commit == result.commit.digest


def test_provider_without_conditioning_grant_is_refused_cold(rig, monkeypatch):
    """A provider that does not grant FEATURE_CONDITIONING never admits a conditioned target."""
    from elpis.inference.drivers.dsv41 import provider_stream as PS
    projection = fixture_conditioning_projection(rig.config, WIDTH)
    provider = rig.provider()
    decode = PS.P.decode_admit_ack

    def strip(body, requested):
        ack = decode(body, requested)
        return replace(ack, features=ack.features & ~P.FEATURE_CONDITIONING)

    monkeypatch.setattr(PS.P, "decode_admit_ack", strip)
    with pytest.raises(ContractError) as info:
        rig.provider_target(rig.fms(), "provider", provider, conditioning_projection=projection)
    assert info.value.code == Code.UNSUPPORTED
    monkeypatch.undo()
    assert provider.state != "QUARANTINED"
    provider.close()
    assert all(v == 0 for v in rig.live().values())
