"""The canonical turn: DSV4 codec -> ECS -> DSV4 codec, failing closed (docs/ELPIS_MISSION.md).

No ECS<->DSV semantic codec is qualified, so the canonical turn refuses
without one. To exercise the interface mechanics only, these tests supply a
deterministic fixture map labelled TRAINING=NONE SEMANTICS=NONE and a
byte-level stand-in for the token boundary. Neither is a codec, and nothing
here is a claim about cognition: the tests prove the direction of the
dataflow, atomicity, determinism, and that an ECS state transition needs no
DSV model at all.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from elpis.ECS_G.native import ECSGLibrary, WorldState
from elpis.runtime.cognition import CODEC_UNQUALIFIED, Readout, run_turn
from elpis.runtime.composition import CompositionError

from ..ECS_G._math_oracle import gd_step, project_s3
from ..ECS_G.test_math_r0 import REPO, _library_path
from ._turn_fixtures import FIXTURE, ByteTokens, FixtureMap

DIM, WIDTH, RATE = 6, 36, 0.002


@pytest.fixture(scope="module")
def api():
    return ECSGLibrary(ctypes.CDLL(str(_library_path())))


def initial_w(seed=36):
    return np.random.default_rng(seed).normal(0.0, 0.18, size=(DIM, WIDTH))


def world(api, w=None):
    w = initial_w() if w is None else w
    return WorldState.create(api, DIM, WIDTH, w.reshape(-1).tolist())


def test_canonical_turn_fails_closed_without_a_qualified_codec(api, runtime):
    with world(api) as state:
        before = state.snapshot()
        for call in (lambda: run_turn(state, "Hello, Elpis.", tokenizer=ByteTokens(), learning_rate=RATE),
                     lambda: runtime.run_turn(state, "Hello, Elpis.", tokenizer=ByteTokens(), learning_rate=RATE)):
            with pytest.raises(CompositionError) as info:
                call()
            assert info.value.code == CODEC_UNQUALIFIED == "ECS_CODEC_UNQUALIFIED"
            assert "text generation unavailable" in str(info.value)
        assert state.snapshot() == before and state.epoch == 0
    assert runtime.history.records() == ()


def test_turn_runs_codec_then_ecs_then_codec(api, runtime):
    fixture = FixtureMap(steps=3)
    with world(api) as state:
        result = runtime.run_turn(state, "Hello, Elpis.", tokenizer=ByteTokens(), codec_map=fixture,
                                  learning_rate=RATE)
        assert [name for name, _ in fixture.calls] == ["encode", "decode"]
        assert fixture.calls[0][1] == result.input_tokens == tuple(b"Hello, Elpis.")
        # Decode sees the readout of the state the stimulus produced, never the prior state.
        readout = fixture.calls[1][1]
        assert type(readout) is Readout and readout == result.readout
        assert (result.epoch_before, readout.epoch, result.epoch_after, state.epoch) == (0, 3, 3, 3)
        w = initial_w()
        for x, y in fixture.encode(result.input_tokens).drives:
            w = gd_step(w, np.asarray(x), np.asarray(y), RATE)
        np.testing.assert_allclose(readout.s3, np.concatenate(project_s3(w)), rtol=1e-11, atol=1e-12)
        assert readout.s3 == state.s3() and len(readout.s3) == 83
        assert result.output_tokens == tuple(b"ok") and result.text == "ok"
        assert result.codec == FIXTURE and "SEMANTICS=NONE" in result.codec


@pytest.mark.parametrize("case", ["ecs_refused", "decode_out_of_vocab", "decode_over_limit",
                                  "not_a_stimulus", "unclassified_map", "bad_rate"])
def test_a_refused_turn_leaves_ecs_untouched(api, case):
    fixture = FixtureMap(poison_step=1 if case == "ecs_refused" else None,
                         reply=(300,) if case == "decode_out_of_vocab" else b"x" * 9)
    if case == "not_a_stimulus":
        fixture.encode = lambda tokens: ((), ())
    if case == "unclassified_map":
        fixture.classification = ""
    kwargs = dict(tokenizer=ByteTokens(), codec_map=fixture, learning_rate=RATE,
                  max_output_tokens=4 if case == "decode_over_limit" else 256)
    if case == "bad_rate":
        kwargs["learning_rate"] = float("nan")
    expected = {"ecs_refused": "ECS_REFUSED", "decode_out_of_vocab": "DECODE", "decode_over_limit": "DECODE",
                "not_a_stimulus": "STIMULUS", "unclassified_map": "CODEC_MAP", "bad_rate": "LEARNING_RATE"}[case]
    with world(api) as state:
        before = state.snapshot()
        with pytest.raises(CompositionError) as info:
            run_turn(state, "perturb", **kwargs)
        assert info.value.code == expected
        assert state.snapshot() == before and state.epoch == 0


def test_turns_are_deterministic_and_snapshots_continue(api):
    texts = ("first", "second turn", "third")
    with world(api) as a, world(api) as b:
        ra = [run_turn(a, t, tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=RATE) for t in texts]
        rb = [run_turn(b, t, tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=RATE) for t in texts]
        assert ra == rb and a.snapshot() == b.snapshot()
    with world(api) as live:
        run_turn(live, texts[0], tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=RATE)
        with WorldState.restore(api, live.snapshot()) as resumed:
            for t in texts[1:]:
                x = run_turn(live, t, tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=RATE)
                y = run_turn(resumed, t, tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=RATE)
                assert x == y
            assert live.snapshot() == resumed.snapshot() and live.epoch == 6


_NO_DSV_PROBE = r"""
import ctypes, json, sys
sys.path.insert(0, sys.argv[1])
from elpis.ECS_G.native import ECSGLibrary, WorldState
from elpis.runtime.cognition import run_turn
from tests.integration._turn_fixtures import ByteTokens, FixtureMap
api = ECSGLibrary(ctypes.CDLL(sys.argv[2]))
with WorldState.create(api, 6, 36, [0.01 * (i % 13 - 6) for i in range(216)]) as state:
    result = run_turn(state, "an ECS transition", tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=0.002)
    epoch = state.epoch
print(json.dumps({"epoch": epoch, "text": result.text, "codec": sys.argv[3:],
                  "modules": sorted(m for m in sys.modules if m.startswith(("elpis", "research", "numpy")))}))
"""


def test_an_ecs_transition_needs_no_dsv_model():
    """Cognitive state exists and moves with no DSV tower, parameter bank or model machinery loaded."""
    codec = ("elpis.inference.contracts", "elpis.inference.text", "elpis.inference.admission",
             "elpis.inference.structural")
    result = subprocess.run([sys.executable, "-c", _NO_DSV_PROBE, str(REPO), str(_library_path()), *codec],
                            capture_output=True, text=True, cwd=REPO,
                            env={"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["epoch"] == 2 and report["text"] == "ok"
    loaded = report["modules"]
    model = [m for m in loaded if m.startswith("elpis.inference.") and not m.startswith(codec)]
    assert not model, model  # only the codec modules the runtime composition imports
    assert not [m for m in loaded if m.startswith(("research", "numpy"))], loaded


def test_canonical_turn_with_the_admitted_v41_codec(api):
    """With the real digest-bound tokenizer: still fails closed without a map; mechanics with the fixture."""
    path = os.environ.get("ELPIS_V41_TOKENIZER")
    if not path:
        pytest.skip("ELPIS_V41_TOKENIZER required for the real token boundary")
    from elpis.inference.text import RECIPE_SHA256, V41Tokenizer
    from elpis.substrate.boundary import RootCapability
    with RootCapability(Path(path).parent) as root:
        v41 = V41Tokenizer.load(root, Path(path).name, expected_sha256=RECIPE_SHA256)
    with world(api) as state:
        with pytest.raises(CompositionError) as info:
            run_turn(state, "Hello, Elpis.", tokenizer=v41, learning_rate=RATE)
        assert info.value.code == CODEC_UNQUALIFIED and state.epoch == 0
        fixture = FixtureMap(reply=v41.encode("probe"))
        result = run_turn(state, "Hello, Elpis.", tokenizer=v41, codec_map=fixture, learning_rate=RATE)
        assert result.input_tokens == v41.encode("Hello, Elpis.") and result.text == "probe"
        assert state.epoch == 2


def test_a_turn_whose_state_moved_meanwhile_is_refused_not_half_installed(api):
    """The trial is adopted in one commit only if the authoritative state is still the one it forked."""
    with world(api) as state:
        fixture = FixtureMap()
        interloper = FixtureMap(steps=1)
        decode = fixture.decode

        def decode_while_state_moves(readout):
            x, y = interloper.encode(tuple(b"interloper")).drives[0]
            state.step(x, y, RATE)  # someone else commits a transition during the turn
            return decode(readout)
        fixture.decode = decode_while_state_moves
        with pytest.raises(CompositionError) as info:
            run_turn(state, "perturb", tokenizer=ByteTokens(), codec_map=fixture, learning_rate=RATE)
        assert info.value.code == "ECS_STALE"
        assert state.epoch == 1  # only the interloper's step; the turn installed nothing
