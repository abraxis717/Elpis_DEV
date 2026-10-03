"""ECS_G world model across principal DSV4.1 turns, composed by the runtime.

turn N commits -> runtime derives the observation drive -> ECS_G steps W_N -> W_N+1
-> S3(W_N+1) is admitted as frozen conditioning -> turn N+1 begins. Nothing
touches ECS_G while tokens stream; a failed or uncommitted turn changes
nothing. Qualified on the tokenizer-free synthetic DSV4.1 fixture and, when the
pinned tokenizer is configured, on the production-shaped fixture; NumPy, the
sealed native backend and the YTS-R0 provider agree.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from elpis.ECS_G.native import WorldState
from elpis.inference.admission import ContextAdmission, ContextBudget
from elpis.inference.conditioning import OBSERVATION_FEATURES, TurnConditioning
from elpis.inference.drivers.dsv41.fixtures import fixture_conditioning_projection
from elpis.inference.principal import PrincipalEngine, PrincipalRequest, PrincipalSequence
from elpis.inference.text import DSV41_RENDERER
from elpis.runtime import Runtime, RuntimeConfig
from elpis.runtime.composition import CompositionError
from elpis.runtime.world_model import DriveMap, WorldModelError, WorldModelLoop, load_world_library

from ..ECS_G._math_oracle import gd_step, project_s3
from ..inference.dsv41 import provider_harness as H

DIM, WIDTH, ROWS, RATE = 6, 36, 8, 0.002
CONTEXT = "c" * 64
S3_WIDTH = 83


def drive_map(seed=2026):
    rng = np.random.default_rng(seed)
    f = OBSERVATION_FEATURES

    def floats(scale, count):
        return tuple(float(v) for v in rng.normal(0.0, scale, count))

    return DriveMap(DIM, ROWS, floats(0.5, ROWS * DIM * f), floats(0.3, ROWS * DIM),
                    floats(0.5, ROWS * f), floats(0.2, ROWS))


def initial_w(seed=36):
    return np.random.default_rng(seed).normal(0.0, 0.18, size=(DIM, WIDTH))


@pytest.fixture(params=("synthetic", "production"))
def rig(native_workspace, request, monkeypatch):
    monkeypatch.setattr(H, "LIBRARIES", H.LIBRARIES + ("elpis_ecsg_math",))
    if request.param == "synthetic":
        from ..inference.dsv41.clock_harness import make_rig
        r = make_rig(native_workspace)
    else:
        path = os.environ.get("ELPIS_V41_TOKENIZER")
        if not path:
            pytest.skip("ELPIS_V41_TOKENIZER required for the production-shaped fixture")
        from elpis.inference.text import RECIPE_SHA256, V41Tokenizer
        from elpis.substrate.boundary import RootCapability
        with RootCapability(Path(path).parent) as root:
            v41 = V41Tokenizer.load(root, Path(path).name, expected_sha256=RECIPE_SHA256)
        r = H.make_rig(native_workspace, v41)
    r.ecsg = load_world_library(r.workspace, r.paths["elpis_ecsg_math"], authority=r.authority,
                                library_id="elpis_ecsg_math")
    yield r
    r.close()
    for fms in getattr(r, "file_assets", ()):
        fms.close()


@pytest.fixture
def runtime(tmp_path_factory):
    with Runtime(RuntimeConfig(tmp_path_factory.mktemp("runtime") / "history")) as rt:
        yield rt


def new_world(rig, w=None):
    w = initial_w() if w is None else w
    return WorldModelLoop(WorldState.create(rig.ecsg, DIM, WIDTH, w.reshape(-1).tolist()), drive_map(),
                          learning_rate=RATE)


def conditioned(rig, name, kind="native", provider=None):
    projection = fixture_conditioning_projection(rig.config, S3_WIDTH)
    if kind == "provider":
        return rig.provider_target(rig.fms(), name, provider, conditioning_projection=projection)
    make = rig.native_target if kind == "native" else rig.numpy_target
    return make(rig.fms(), name, conditioning_projection=projection)


def admission(target):
    return ContextAdmission(CONTEXT, target.model_identity, target.config.tokenizer, DSV41_RENDERER,
                            "d" * 64, (), (), 0, ContextBudget(4, 4096, 2048))


def request(rig, turn, count=6):
    return PrincipalRequest(f"turn-{turn}", tuple(rig.tokens(3 + turn)), count)


def turns(runtime, engine, rig, world, count, *, start=None, emit=None):
    state = start or engine.initial(CONTEXT)
    out = []
    for turn in range(count):
        before = world.turn_conditioning() if world is not None else None
        result, record = runtime.run_principal(engine, state, request(rig, turn), admission(engine.target),
                                               expected_state=state.digest, emit=emit, world=world)
        out.append((before, result, record))
        state = result.state
    return out


def oracle_s3(w):
    return np.concatenate(project_s3(w))


# ---------------------------------------------------------------------------

def test_committed_turn_advances_world_exactly_once_and_only_the_next_turn_sees_it(rig, runtime):
    target = conditioned(rig, "native")
    engine = PrincipalEngine(target)
    world = new_world(rig)
    w = initial_w()
    np.testing.assert_allclose(world.s3(), oracle_s3(w), rtol=1e-12, atol=1e-12)
    history = turns(runtime, engine, rig, world, 3)
    for turn, (before, result, record) in enumerate(history):
        commit = result.commit
        # Turn N was conditioned by S3(W_N), computed before it began; never by itself.
        assert commit.conditioning == before.digest
        np.testing.assert_allclose(before.values, oracle_s3(w), rtol=1e-12, atol=1e-12)
        assert dict(record.record.bindings)["conditioning"] == before.digest
        assert result.observation.commit == commit.digest
        x, y = world.drive_map.drive(result.observation)
        w = gd_step(w, np.asarray(x), np.asarray(y), RATE)  # exactly one step per committed turn
    assert world.epoch == world.transitions == 3 and world.applied == history[-1][1].commit.digest
    np.testing.assert_allclose(world.s3(), oracle_s3(w), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(world.turn_conditioning().values, oracle_s3(w), rtol=1e-11, atol=1e-12)
    digests = [before.digest for before, _, _ in history] + [world.turn_conditioning().digest]
    assert len(set(digests)) == 4
    # The conditioning reaches the model: unconditioned turn 0 differs from conditioned turn 0.
    plain = turns(runtime, engine, rig, None, 1)[0][1]
    assert plain.commit.conditioning is None and plain.commit.digest != history[0][1].commit.digest
    assert [r.record.kind for r in runtime.history.records()] == ["principal.commit"] * 4


def test_failed_or_uncommitted_turn_never_mutates_the_world(rig, runtime):
    target = conditioned(rig, "native")
    engine = PrincipalEngine(target)
    world = new_world(rig)
    snapshot, conditioning = world.checkpoint(), world.turn_conditioning()
    state = engine.initial(CONTEXT)
    # Stale committed state: begin fails typed, nothing commits, nothing is recorded.
    result, record = runtime.run_principal(engine, state, request(rig, 0), admission(target),
                                           expected_state="0" * 64, world=world)
    assert result.commit is None and record is None
    # Emission failure mid-stream: the sequence never finalizes.
    def explode(token):
        raise RuntimeError("client went away")
    with pytest.raises(RuntimeError):
        runtime.run_principal(engine, state, request(rig, 0), admission(target), expected_state=state.digest,
                              emit=explode, world=world)
    # A target that cannot admit conditioning: typed failure, world untouched.
    plain_engine = PrincipalEngine(rig.native_target(rig.fms(), "plain"))
    plain_state = plain_engine.initial(CONTEXT)
    result, record = runtime.run_principal(plain_engine, plain_state, request(rig, 0), admission(plain_engine.target),
                                           expected_state=plain_state.digest, world=world)
    assert result.commit is None and record is None and result.failure == "UNSUPPORTED"
    assert world.checkpoint() == snapshot and world.turn_conditioning() == conditioning
    assert world.epoch == world.transitions == 0 and runtime.history.records() == ()
    # The loop itself refuses anything but a fresh commit conditioned by its current state.
    with pytest.raises(WorldModelError) as info:
        world.advance(result)
    assert info.value.code == "UNCOMMITTED"
    sequence = engine.begin(state, request(rig, 0), admission(target), expected_state=state.digest)
    list(sequence)
    unconditioned = engine.finalize(sequence)
    with pytest.raises(WorldModelError) as info:
        world.advance(unconditioned)
    assert info.value.code == "STALE"
    (_, done, _), = turns(runtime, engine, rig, world, 1)
    with pytest.raises(WorldModelError) as info:
        world.advance(done)
    assert info.value.code == "REPLAYED"
    assert world.epoch == 1


def test_no_ecs_g_call_while_tokens_stream_and_conditioning_is_frozen(rig, runtime, monkeypatch):
    target = conditioned(rig, "native")
    engine = PrincipalEngine(target)
    world = new_world(rig)
    streaming, calls, seen = [False], [], []
    for name in ("s3", "step", "snapshot", "w", "close"):
        original = getattr(WorldState, name)

        def guarded(self, *args, __original=original, __name=name, **kwargs):
            assert not streaming[0], "ECS_G called while tokens stream: " + __name
            calls.append(__name)
            return __original(self, *args, **kwargs)
        monkeypatch.setattr(WorldState, name, guarded)
    begin, finalize = engine.begin, engine.finalize

    def tracked_begin(*args, **kwargs):
        sequence = begin(*args, **kwargs)
        assert type(sequence) is PrincipalSequence
        seen.append(sequence)
        streaming[0] = True
        return sequence

    def tracked_finalize(sequence):
        streaming[0] = False
        return finalize(sequence)

    frozen = []

    def emit(token):
        vector = seen[-1].working_state.conditioning
        assert not vector.flags.writeable
        frozen.append(vector.tobytes())

    monkeypatch.setattr(engine, "begin", tracked_begin)
    monkeypatch.setattr(engine, "finalize", tracked_finalize)
    calls.clear()
    history = turns(runtime, engine, rig, world, 2, emit=emit)
    assert len(frozen) == sum(len(r.commit.outputs) for _, r, _ in history) > 0
    first = len(history[0][1].commit.outputs)
    assert len(set(frozen[:first])) == 1 and len(set(frozen[first:])) == 1 and frozen[0] != frozen[-1]
    # Exactly one step per turn, each followed by the re-projection for the next turn.
    assert calls.count("step") == 2 and calls.count("s3") == 2 and calls.count("snapshot") == 2


def test_replay_and_determinism_across_independent_runtimes(rig, tmp_path):
    target = conditioned(rig, "native")
    engine = PrincipalEngine(target)
    runs = []
    for name in ("a", "b"):
        with Runtime(RuntimeConfig(tmp_path / name)) as rt:
            world = new_world(rig)
            history = turns(rt, engine, rig, world, 3)
            runs.append(([(b.digest, r.commit.digest, r.observation) for b, r, _ in history],
                         world.checkpoint(), world.turn_conditioning()))
    assert runs[0] == runs[1]
    # Each commit replays from its inputs plus the same frozen conditioning; never without it.
    state = engine.initial(CONTEXT)
    world = new_world(rig)
    with Runtime(RuntimeConfig(tmp_path / "c")) as rt:
        history = turns(rt, engine, rig, world, 2)
    for turn, (before, result, _) in enumerate(history):
        replayed = engine.replay(state, request(rig, turn), admission(target), result.commit, conditioning=before)
        assert replayed == result
        state = result.state


def test_snapshot_restore_preserves_continuation(rig, tmp_path):
    target = conditioned(rig, "native")
    engine = PrincipalEngine(target)
    with Runtime(RuntimeConfig(tmp_path / "live")) as rt:
        world = new_world(rig)
        (_, first, _), = turns(rt, engine, rig, world, 1)
        snapshot, applied = world.checkpoint()
        restored = WorldModelLoop.restore(rig.ecsg, snapshot, drive_map(), learning_rate=RATE, applied=applied)
        assert restored.turn_conditioning() == world.turn_conditioning() and restored.epoch == 1
        continued = turns(rt, engine, rig, world, 2, start=first.state)
    with Runtime(RuntimeConfig(tmp_path / "restored")) as rt:
        resumed = turns(rt, engine, rig, restored, 2, start=first.state)
    assert [(b, r.commit, r.observation) for b, r, _ in continued] == \
        [(b, r.commit, r.observation) for b, r, _ in resumed]
    assert restored.checkpoint() == world.checkpoint() and restored.epoch == world.epoch == 3


def test_absent_world_preserves_pre_integration_behavior(rig, runtime):
    target = rig.native_target(rig.fms(), "plain")
    engine = PrincipalEngine(target)
    state = engine.initial(CONTEXT)
    sequence = engine.begin(state, request(rig, 0), admission(target), expected_state=state.digest)
    list(sequence)
    expected = engine.finalize(sequence)
    result, record = runtime.run_principal(engine, state, request(rig, 0), admission(target),
                                           expected_state=state.digest)
    assert result == expected and result.commit.conditioning is None
    assert sorted(dict(record.record.bindings)) == ["admission", "outputs", "request", "state", "stop"]


def test_world_turns_agree_across_numpy_native_and_provider(rig, tmp_path):
    """Same world, same turns: native and YTS-R0 provider agree bitwise; NumPy agrees on every commit."""
    provider = rig.provider()
    outcomes = {}
    for kind in ("native", "provider", "numpy"):
        target = conditioned(rig, kind, kind, provider if kind == "provider" else None)
        engine = PrincipalEngine(target)
        world = new_world(rig)
        with Runtime(RuntimeConfig(tmp_path / kind)) as rt:
            history = turns(rt, engine, rig, world, 3)
        outcomes[kind] = dict(outputs=[r.commit.outputs for _, r, _ in history],
                              trace=[r.commit.trace for _, r, _ in history],
                              observation=[r.observation.features for _, r, _ in history],
                              conditioning=[b.values for b, _, _ in history],
                              world=world.s3())
    assert outcomes["provider"] == outcomes["native"]
    assert outcomes["numpy"]["outputs"] == outcomes["native"]["outputs"]
    np.testing.assert_allclose(outcomes["numpy"]["observation"], outcomes["native"]["observation"],
                               rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(outcomes["numpy"]["world"], outcomes["native"]["world"], rtol=1e-6, atol=1e-9)
    provider.close()


def test_world_refusal_after_commit_leaves_w_unchanged(rig, runtime, monkeypatch):
    target = conditioned(rig, "native")
    engine = PrincipalEngine(target)
    world = new_world(rig)
    snapshot = world.checkpoint()
    from elpis.ECS_G.native import ECSGError

    def refuse(self, *args, **kwargs):
        raise ECSGError("NONFINITE", "gradient step")
    monkeypatch.setattr(WorldState, "step", refuse)
    state = engine.initial(CONTEXT)
    with pytest.raises(CompositionError) as info:
        runtime.run_principal(engine, state, request(rig, 0), admission(target), expected_state=state.digest,
                              world=world)
    assert info.value.code == "WORLD_MODEL"
    assert [r.record.kind for r in runtime.history.records()] == ["principal.commit"]
    assert world.checkpoint() == snapshot and world.epoch == world.transitions == 0


def test_conditioning_is_only_admitted_through_the_generic_contract():
    """Inference never names ECS_G; ECS_G never names inference or runtime (static)."""
    import ast
    src = Path(__file__).resolve().parents[2] / "src" / "elpis"
    for package, forbidden in (("inference", ("elpis.ECS_G",)),
                               ("ECS_G", ("elpis.inference", "elpis.runtime", "elpis.ECS_C"))):
        for path in (src / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                         [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else [])
                assert not any(n == f or n.startswith(f + ".") for n in names for f in forbidden), (path, names)
    assert TurnConditioning.__module__ == "elpis.inference.conditioning"
