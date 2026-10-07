"""The canonical turn: DSV4 codec -> native K1 ECS -> DSV4 codec, failing closed (docs/ELPIS_MISSION.md).

No ECS<->DSV semantic codec is qualified, so the canonical turn refuses without one. To exercise the interface
mechanics only, these tests supply a deterministic fixture map labelled TRAINING=NONE SEMANTICS=NONE and a
byte-level stand-in for the token boundary. Neither is a codec, and nothing here is a claim about cognition: the
tests prove the direction of the dataflow, that every experience is the qualified K1 transition (learn, then
consolidate) executed natively on a candidate of the complete (W, epoch, H, a) state, atomicity, determinism, the
equality of the standalone and FMS-resident paths, and that an ECS state transition needs no DSV model at all.
"""
from __future__ import annotations

from array import array
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from elpis.ECS.k1 import K1FMSRuntime, K1Library, K1State
from elpis.ECS.native import ECSGLibrary, Executor
from elpis.runtime.cognition import CODEC_UNQUALIFIED, Readout, Stimulus, run_turn
from elpis.runtime.composition import CompositionError
from elpis.substrate.residency import Context

from ..ECS.test_math_r0 import REPO, _library_path
from ._turn_fixtures import FIXTURE, ByteTokens, FixtureMap

DIM, WIDTH, RATE = 6, 36, 0.002


def _beside(name):
    path = Path(_library_path()).with_name(name)
    if not path.is_file():
        raise AssertionError(f"{name} not built beside the ECS library: {path}")
    return path


@pytest.fixture(scope="module")
def api():
    return ECSGLibrary(ctypes.CDLL(str(_library_path())))


@pytest.fixture(scope="module")
def k1():
    return K1Library(ctypes.CDLL(str(_beside("libelpis_ecsg_k1.so"))))


@pytest.fixture(scope="module")
def adapter():
    return ctypes.CDLL(str(_beside("libelpis_ecsg_k1_fms.so")))


def initial_w(seed=36):
    return np.random.default_rng(seed).normal(0.0, 0.18, size=(DIM, WIDTH))


def world(k1, w=None, max_rows=64):
    w = initial_w() if w is None else w
    return K1State.create(k1, DIM, WIDTH, memoryview(np.ascontiguousarray(w).reshape(-1)), max_rows=max_rows)


class _Resident:
    """One FMS-resident K1 state over a fresh, test-owned FMS context."""

    def __init__(self, k1, adapter, tmp_path, w=None, max_rows=64):
        image = k1.envelope_bytes(DIM, WIDTH) - 32
        w = initial_w() if w is None else w
        self.ctx = Context(adapter, warm_bytes=8 * image, cold_bytes=10 ** 6, max_objects=4,
                           cold_root=tmp_path / "cold")
        self.runtime = K1FMSRuntime(k1, self.ctx, adapter, max_states=2)
        sid = self.runtime.register(b"\x07" * 32, DIM, WIDTH, memoryview(np.ascontiguousarray(w).reshape(-1)),
                                    max_rows=max_rows)
        self.state = self.runtime.state(sid)

    def __enter__(self):
        return self.state

    def __exit__(self, *exc):
        self.runtime.close_state(self.state.id)
        self.runtime.close()
        self.ctx.close()


def explicit(k1, stimulus, w=None):
    """The same experiences through the existing K1 transaction operations, one experience at a time."""
    state = world(k1, w)
    x = np.frombuffer(stimulus.x, dtype=np.float64).reshape(-1, DIM)
    y = np.frombuffer(stimulus.y, dtype=np.float64)
    off = 0
    with state.transaction() as txn:
        for rows, steps in zip(stimulus.schedule[0::2], stimulus.schedule[1::2]):
            txn.learn(memoryview(x[off:off + rows].reshape(-1)), memoryview(y[off:off + rows]), RATE, steps)
            txn.consolidate(memoryview(x[off:off + rows].reshape(-1)))
            off += rows
        txn.commit()
    return state


def test_canonical_turn_fails_closed_without_a_qualified_codec(k1, runtime):
    with world(k1) as state:
        before = state.snapshot()
        for call in (lambda: run_turn(state, "Hello, Elpis.", tokenizer=ByteTokens(), learning_rate=RATE),
                     lambda: runtime.run_turn(state, "Hello, Elpis.", tokenizer=ByteTokens(), learning_rate=RATE)):
            with pytest.raises(CompositionError) as info:
                call()
            assert info.value.code == CODEC_UNQUALIFIED == "ECS_CODEC_UNQUALIFIED"
            assert "text generation unavailable" in str(info.value)
        assert state.snapshot() == before and state.epoch == 0
    assert not runtime.continuity.snapshot().anchored


def test_the_executor_is_no_longer_the_canonical_turn_substrate(api):
    with Executor.create(api, DIM, WIDTH, initial_w().reshape(-1).tolist()) as executor:
        with pytest.raises(CompositionError) as info:
            run_turn(executor, "x", tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=RATE)
        assert info.value.code == "ECS_STATE" and executor.epoch == 0


def test_turn_runs_codec_then_native_k1_experiences_then_codec(api, k1, runtime):
    fixture = FixtureMap(experiences=3, steps=4)
    with world(k1) as state:
        runtime.anchor_cognition(state)
        result = runtime.run_turn(state, "Hello, Elpis.", tokenizer=ByteTokens(), codec_map=fixture,
                                  learning_rate=RATE)
        assert [name for name, _ in fixture.calls] == ["encode", "decode"]
        assert fixture.calls[0][1] == result.input_tokens == tuple(b"Hello, Elpis.")
        # Decode sees the readout of the candidate the stimulus produced, never the prior state.
        readout = fixture.calls[1][1]
        assert type(readout) is Readout and readout == result.readout
        assert (result.epoch_before, readout.epoch, result.epoch_after, state.epoch) == (0, 12, 12, 12)
        # Every experience was the K1 transition: the same as the existing operations applied in order, bitwise.
        stimulus = fixture.encode(result.input_tokens)
        with explicit(k1, stimulus) as reference:
            assert state.snapshot() == reference.snapshot()
        # The readout is the native S3 of the committed W (bitwise the Runtime R1 projection of that W).
        with Executor.create(api, DIM, WIDTH, state.w()) as same_w:
            assert readout.s3 == same_w.s3() and len(readout.s3) == 83
        # The retained state moved: H accumulated every experience; a is S3 of W after the last experience.
        assert any(state.h_packed()) and state.a() == readout.s3
        assert state.provenance == "COMPLETE"
        assert result.output_tokens == tuple(b"ok") and result.text == "ok"
        assert result.codec == FIXTURE and "SEMANTICS=NONE" in result.codec


def test_one_experience_with_empty_retained_state_is_runtime_r1_g1(api, k1):
    fixture = FixtureMap(experiences=1, steps=9)
    with world(k1) as state:
        run_turn(state, "one experience", tokenizer=ByteTokens(), codec_map=fixture, learning_rate=RATE)
        stimulus = fixture.encode(tuple(b"one experience"))
        with Executor.create(api, DIM, WIDTH, initial_w().reshape(-1).tolist(), max_rows=64) as g1:
            g1.learn(memoryview(stimulus.x), memoryview(stimulus.y), RATE, 9)
            assert state.w() == tuple(g1.w())   # H = 0 during the only learning: bitwise Runtime R1 G1


@pytest.mark.parametrize("case", ["ecs_refused_middle_experience", "decode_out_of_vocab", "decode_over_limit",
                                  "not_a_stimulus", "unclassified_map", "bad_rate", "wrong_dim"])
def test_a_refused_turn_leaves_the_complete_k1_state_untouched(k1, case):
    fixture = FixtureMap(experiences=3, poison_experience=1 if case == "ecs_refused_middle_experience" else None,
                         reply=(300,) if case == "decode_out_of_vocab" else b"x" * 9)
    if case == "not_a_stimulus":
        fixture.encode = lambda tokens: ((), ())
    if case == "unclassified_map":
        fixture.classification = ""
    if case == "wrong_dim":
        fixture.encode = lambda tokens: Stimulus(array("d", [0.1] * 5), array("d", [0.0]), array("Q", [1, 1]), dim=5)
    kwargs = dict(tokenizer=ByteTokens(), codec_map=fixture, learning_rate=RATE,
                  max_output_tokens=4 if case == "decode_over_limit" else 256)
    if case == "bad_rate":
        kwargs["learning_rate"] = float("nan")
    expected = {"ecs_refused_middle_experience": "ECS_REFUSED", "decode_out_of_vocab": "DECODE",
                "decode_over_limit": "DECODE", "not_a_stimulus": "STIMULUS", "unclassified_map": "CODEC_MAP",
                "bad_rate": "LEARNING_RATE", "wrong_dim": "STIMULUS"}[case]
    with world(k1) as state:
        run_turn(state, "warm", tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=RATE)  # H, a != 0
        before, epoch = state.snapshot(), state.epoch
        with pytest.raises(CompositionError) as info:
            run_turn(state, "perturb", **kwargs)
        assert info.value.code == expected
        assert state.snapshot() == before and state.epoch == epoch   # W, epoch, H, a byte-for-byte
        run_turn(state, "after", tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=RATE)  # not stuck


def test_stimulus_admission_is_bounded_and_explicit():
    ok = dict(dim=DIM)
    for x, y, schedule in ((array("d", [0.0] * 6), array("d", [0.0]), array("Q", [1, 0])),       # zero steps
                           (array("d", [0.0] * 6), array("d", [0.0]), array("Q", [0, 1])),       # zero rows
                           (array("d", [0.0] * 6), array("d", [0.0]), array("Q", [2, 1])),       # rows mismatch
                           (array("d", [0.0] * 6), array("d", [0.0]), array("Q", [1])),          # odd schedule
                           (array("d", [0.0] * 6), array("d", [0.0]), array("Q", [1, 1] * 65)),  # > 64 experiences
                           (array("f", [0.0] * 6), array("d", [0.0]), array("Q", [1, 1])),       # not binary64
                           (array("d", [0.0] * 6), array("d", [0.0]), array("i", [1, 1]))):      # not uint64
        with pytest.raises(CompositionError) as info:
            Stimulus(x, y, schedule, **ok)
        assert info.value.code == "STIMULUS"
    src = array("d", [0.5] * 6)
    s = Stimulus(src, array("d", [0.1]), array("Q", [1, 3]), dim=DIM)
    src[0] = 99.0   # admission took an owned copy
    assert s.x[0] == 0.5 and (s.experiences, s.rows, s.steps, s.max_experience_rows) == (1, 1, 3, 1)


def test_turns_are_deterministic_and_retained_snapshots_continue(k1):
    texts = ("first", "second turn", "third")
    with world(k1) as a, world(k1) as b:
        ra = [run_turn(a, t, tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=RATE) for t in texts]
        rb = [run_turn(b, t, tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=RATE) for t in texts]
        assert ra == rb and a.snapshot() == b.snapshot()
    with world(k1) as live:
        run_turn(live, texts[0], tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=RATE)
        with K1State.restore(k1, live.snapshot()) as resumed:
            for t in texts[1:]:
                x = run_turn(live, t, tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=RATE)
                y = run_turn(resumed, t, tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=RATE)
                assert x == y
            assert live.snapshot() == resumed.snapshot() and live.epoch == 18


def test_fms_resident_turns_equal_standalone_turns_bitwise(k1, adapter, tmp_path):
    texts = ("first", "a second, longer turn", "third")
    with world(k1) as standalone, _Resident(k1, adapter, tmp_path) as resident:
        for i, text in enumerate(texts):
            fixture = dict(experiences=1 + i, steps=2 + i)
            a = run_turn(standalone, text, tokenizer=ByteTokens(), codec_map=FixtureMap(**fixture), learning_rate=RATE)
            b = run_turn(resident, text, tokenizer=ByteTokens(), codec_map=FixtureMap(**fixture), learning_rate=RATE)
            assert a == b
            assert standalone.snapshot() == resident.snapshot()
        before = resident.snapshot()
        with pytest.raises(CompositionError) as info:
            run_turn(resident, "poison", tokenizer=ByteTokens(), learning_rate=RATE,
                     codec_map=FixtureMap(experiences=3, poison_experience=1))
        assert info.value.code == "ECS_REFUSED" and resident.snapshot() == before
        with pytest.raises(CompositionError) as info:
            run_turn(resident, "decode", tokenizer=ByteTokens(), learning_rate=RATE, codec_map=FixtureMap(reply=(999,)))
        assert info.value.code == "DECODE" and resident.snapshot() == before
        info = resident._r.inspect(resident.id)
        assert info["transaction_open"] == 0 and info["lease_count"] == 0   # no pin left behind


def test_fms_rows_beyond_capacity_grow_on_the_cold_path(k1, adapter, tmp_path):
    with _Resident(k1, adapter, tmp_path, max_rows=4) as resident, world(k1, max_rows=4) as standalone:
        for state in (resident, standalone):
            run_turn(state, "rows", tokenizer=ByteTokens(), codec_map=FixtureMap(rows=12), learning_rate=RATE)
            assert state.max_rows >= 12
        assert resident.snapshot() == standalone.snapshot()


_NO_DSV_PROBE = r"""
import ctypes, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from elpis.ECS.k1 import K1Library, K1State
from elpis.runtime.cognition import run_turn
from tests.integration._turn_fixtures import ByteTokens, FixtureMap
k1 = K1Library(ctypes.CDLL(str(Path(sys.argv[2]).with_name("libelpis_ecsg_k1.so"))))
with K1State.create(k1, 6, 36, [0.01 * (i % 13 - 6) for i in range(216)]) as state:
    result = run_turn(state, "an ECS transition", tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=0.002)
    epoch = state.epoch
print(json.dumps({"epoch": epoch, "text": result.text, "codec": sys.argv[3:],
                  "modules": sorted(m for m in sys.modules if m.startswith(("elpis", "research", "numpy", "torch")))}))
"""


def test_an_ecs_transition_needs_no_dsv_model():
    """Cognitive state exists and moves with no DSV tower, parameter bank, model machinery or NumPy loaded."""
    codec = ("elpis.inference.contracts", "elpis.inference.text", "elpis.inference.admission",
             "elpis.inference.structural")
    result = subprocess.run([sys.executable, "-c", _NO_DSV_PROBE, str(REPO), str(_library_path()), *codec],
                            capture_output=True, text=True, cwd=REPO,
                            env={"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["epoch"] == 6 and report["text"] == "ok"
    loaded = report["modules"]
    model = [m for m in loaded if m.startswith("elpis.inference.") and not m.startswith(codec)]
    assert not model, model  # only the codec modules the runtime composition imports
    assert not [m for m in loaded if m.startswith(("research", "numpy", "torch"))], loaded


def test_canonical_turn_with_the_admitted_v41_codec(k1):
    """With the real digest-bound tokenizer: still fails closed without a map; mechanics with the fixture."""
    path = os.environ.get("ELPIS_V41_TOKENIZER")
    if not path:
        pytest.skip("ELPIS_V41_TOKENIZER required for the real token boundary")
    from elpis.inference.text import RECIPE_SHA256, V41Tokenizer
    from elpis.substrate.boundary import RootCapability
    with RootCapability(Path(path).parent) as root:
        v41 = V41Tokenizer.load(root, Path(path).name, expected_sha256=RECIPE_SHA256)
    with world(k1) as state:
        with pytest.raises(CompositionError) as info:
            run_turn(state, "Hello, Elpis.", tokenizer=v41, learning_rate=RATE)
        assert info.value.code == CODEC_UNQUALIFIED and state.epoch == 0
        fixture = FixtureMap(reply=v41.encode("probe"))
        result = run_turn(state, "Hello, Elpis.", tokenizer=v41, codec_map=fixture, learning_rate=RATE)
        assert result.input_tokens == v41.encode("Hello, Elpis.") and result.text == "probe"
        assert state.epoch == 6


def test_a_turn_whose_state_moved_meanwhile_is_refused_not_half_installed(k1):
    """The candidate commits natively only if the authoritative state is still the one the turn began from."""
    with world(k1) as state:
        fixture = FixtureMap()
        interloper = FixtureMap(experiences=1).encode(tuple(b"interloper"))
        decode = fixture.decode

        def decode_while_state_moves(readout):
            state.learn(interloper.x, interloper.y, RATE, 1)  # someone else commits a transition during the turn
            return decode(readout)
        fixture.decode = decode_while_state_moves
        with pytest.raises(CompositionError) as info:
            run_turn(state, "perturb", tokenizer=ByteTokens(), codec_map=fixture, learning_rate=RATE)
        assert info.value.code == "ECS_STALE"
        assert state.epoch == 1 and not any(state.h_packed())  # only the interloper's step; the turn installed nothing


class _Counting:
    """Counts native crossings per bound symbol."""

    def __init__(self, namespace):
        self.calls = {}
        for name, fn in vars(namespace).copy().items():
            setattr(namespace, name, self._wrap(name, fn))

    def _wrap(self, name, fn):
        def call(*args):
            self.calls[name] = self.calls.get(name, 0) + 1
            return fn(*args)
        return call


@pytest.mark.parametrize("experiences,steps", [(1, 1), (1, 500), (64, 1), (8, 60)])
def test_one_schedule_crossing_per_turn_whatever_experiences_or_steps(experiences, steps, tmp_path):
    k1 = K1Library(ctypes.CDLL(str(_beside("libelpis_ecsg_k1.so"))))
    adapter = ctypes.CDLL(str(_beside("libelpis_ecsg_k1_fms.so")))
    counter = _Counting(k1._k)
    fixture = dict(experiences=experiences, steps=steps)
    with world(k1) as state:
        run_turn(state, "warm", tokenizer=ByteTokens(), codec_map=FixtureMap(**fixture), learning_rate=RATE)
        heap = state.stats()["heap_allocations"]
        counter.calls.clear()
        run_turn(state, "turn", tokenizer=ByteTokens(), codec_map=FixtureMap(**fixture), learning_rate=RATE)
        assert counter.calls == {"max_rows": 1, "txn_begin": 1, "txn_run_schedule": 1, "txn_commit_identity": 1}, counter.calls
        assert state.stats()["heap_allocations"] == heap   # no allocation on the prepared hot path
    with _Resident(k1, adapter, tmp_path) as resident:
        fcounter = _Counting(resident._r._f)
        run_turn(resident, "warm", tokenizer=ByteTokens(), codec_map=FixtureMap(**fixture), learning_rate=RATE)
        heap = resident._r.k1_stats(resident.id)["heap_allocations"]
        fcounter.calls.clear()
        run_turn(resident, "turn", tokenizer=ByteTokens(), codec_map=FixtureMap(**fixture), learning_rate=RATE)
        # no restore, no snapshot, no import, no copy of W/H/a: begin, one schedule, commit
        assert fcounter.calls == {"txn_begin": 1, "txn_run_schedule": 1, "txn_commit_identity": 1}, fcounter.calls
        assert resident._r.k1_stats(resident.id)["heap_allocations"] == heap   # no allocation on the warm path



def test_canonical_turn_exposes_the_commit_bound_retained_state_identities(k1):
    with world(k1) as state:
        before = state.snapshot()
        result = run_turn(state, "continuity", tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=RATE)
        after = state.snapshot()
        # The exact K1 retained-state identities (ELPISGK1 trailers) of the committed transition.
        assert result.state_before_digest == before[-32:]
        assert result.state_after_digest == after[-32:] == state.state_digest()


# -- runtime lineage through continuity (docs/CONTINUITY.md) -------------------------------------

def _config(path):
    from elpis.runtime import RuntimeConfig
    return RuntimeConfig(path)


def test_runtime_turn_publishes_exactly_the_new_expected_k1_identity(k1, tmp_path):
    from elpis.runtime import Runtime

    with world(k1) as state:
        with Runtime(_config(tmp_path / "c")) as runtime:
            anchored = runtime.anchor_cognition(state)
            assert anchored.k1_state_digest == state.state_digest()
            result = runtime.run_turn(state, "one", tokenizer=ByteTokens(), codec_map=FixtureMap(),
                                      learning_rate=RATE)
            snap = runtime.continuity.snapshot()
            assert snap.k1_state_digest == result.state_after_digest == state.state_digest()
            assert snap.generation == anchored.generation + 1
            # Nothing else is durable: no generation, epochs or token identities.
            assert snap.evolution == anchored.evolution


def test_runtime_refuses_unanchored_cognition_before_k1_mutation(k1, tmp_path):
    from elpis.runtime import Runtime

    with world(k1) as state:
        before = state.snapshot()
        with Runtime(_config(tmp_path / "unanchored")) as runtime:
            with pytest.raises(CompositionError) as info:
                runtime.run_turn(state, "must anchor first", tokenizer=ByteTokens(), codec_map=FixtureMap(),
                                 learning_rate=RATE)
            assert info.value.code == "CONTINUITY_UNANCHORED"
            assert state.snapshot() == before
            assert not runtime.continuity.snapshot().anchored  # no implicit anchor, ever


def test_explicit_anchor_is_durable_once_and_does_not_mutate_k1(k1, tmp_path):
    from elpis.runtime import Runtime

    with world(k1) as state:
        before = state.snapshot()
        with Runtime(_config(tmp_path / "anchor")) as runtime:
            runtime.anchor_cognition(state)
            assert state.snapshot() == before
            with pytest.raises(CompositionError) as info:
                runtime.anchor_cognition(state)
            assert info.value.code == "CONTINUITY_ALREADY_ANCHORED"
        with Runtime(_config(tmp_path / "anchor")) as restarted:
            assert restarted.continuity.snapshot().k1_state_digest == before[-32:]
            with pytest.raises(CompositionError) as info:
                restarted.anchor_cognition(state)
            assert info.value.code == "CONTINUITY_ALREADY_ANCHORED"


def test_matched_restart_resumes_without_a_new_anchor(k1, tmp_path):
    from elpis.runtime import Runtime

    with world(k1) as state:
        with Runtime(_config(tmp_path / "c")) as first:
            first.anchor_cognition(state)
            first.run_turn(state, "before restart", tokenizer=ByteTokens(), codec_map=FixtureMap(),
                           learning_rate=RATE)
            generation = first.continuity.snapshot().generation
        with Runtime(_config(tmp_path / "c")) as restarted:
            result = restarted.run_turn(state, "resume", tokenizer=ByteTokens(), codec_map=FixtureMap(),
                                        learning_rate=RATE)
            snap = restarted.continuity.snapshot()
            assert snap.k1_state_digest == result.state_after_digest
            assert snap.generation == generation + 1  # one transition, no anchor


def test_mismatched_restart_fails_before_k1_mutation(k1, tmp_path):
    from elpis.runtime import Runtime

    with world(k1) as anchored, world(k1) as other:
        with Runtime(_config(tmp_path / "c")) as first:
            first.anchor_cognition(anchored)
        run_turn(other, "out of band", tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=RATE)
        before = other.snapshot()
        with Runtime(_config(tmp_path / "c")) as restarted:
            durable = restarted.continuity.snapshot()
            with pytest.raises(CompositionError) as info:
                restarted.run_turn(other, "must refuse", tokenizer=ByteTokens(), codec_map=FixtureMap(),
                                   learning_rate=RATE)
            assert info.value.code == "CONTINUITY_STATE_MISMATCH"
            assert other.snapshot() == before
            # Fail-stopped, and nothing synthesized.
            assert restarted.continuity.snapshot() == durable
            with pytest.raises(CompositionError) as info:
                restarted.run_turn(anchored, "still stopped", tokenizer=ByteTokens(), codec_map=FixtureMap(),
                                   learning_rate=RATE)
            assert info.value.code == "CONTINUITY_STATE_MISMATCH"


@pytest.mark.parametrize("failure", ["write", "sync"])
def test_k1_commit_then_failed_publication_keeps_k1_and_fail_stops(k1, tmp_path, monkeypatch, failure):
    """The crash law: the K1 commit stands; restart sees old authority and refuses the moved state."""
    from elpis.continuity import store as continuity_store
    from elpis.runtime import Runtime

    with world(k1) as state:
        with Runtime(_config(tmp_path / "c")) as runtime:
            anchored = runtime.anchor_cognition(state)

            def fail(*_args):
                raise OSError(5, "EIO")

            if failure == "write":
                monkeypatch.setattr(continuity_store.os, "pwrite", fail)
                expected = "CONTINUITY_PUBLICATION_REFUSED"
            else:
                monkeypatch.setattr(continuity_store.os, "fdatasync", fail, raising=False)
                monkeypatch.setattr(continuity_store.os, "fsync", fail)
                expected = "CONTINUITY_PUBLICATION_UNCERTAIN"
            with pytest.raises(CompositionError) as info:
                runtime.run_turn(state, "commit then fail", tokenizer=ByteTokens(), codec_map=FixtureMap(),
                                 learning_rate=RATE)
            monkeypatch.undo()
            assert info.value.code == expected
            committed = state.snapshot()
            # K1 committed and is never rolled back.
            assert state.epoch > 0 and committed[-32:] != anchored.k1_state_digest
            with pytest.raises(CompositionError) as info:  # fail-stopped: no further K1 mutation
                runtime.run_turn(state, "again", tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=RATE)
            assert info.value.code == expected
            assert state.snapshot() == committed
        with Runtime(_config(tmp_path / "c")) as restarted:
            durable = restarted.continuity.snapshot().k1_state_digest
            if durable == anchored.k1_state_digest:
                # Old durable authority: the moved K1 state is a mismatch, refused before mutation.
                with pytest.raises(CompositionError) as info:
                    restarted.run_turn(state, "after restart", tokenizer=ByteTokens(), codec_map=FixtureMap(),
                                       learning_rate=RATE)
                assert info.value.code == "CONTINUITY_STATE_MISMATCH"
                assert state.snapshot() == committed
            else:
                # New durable authority (the synced-unknown write did land): resume.
                assert durable == committed[-32:]
                restarted.run_turn(state, "after restart", tokenizer=ByteTokens(), codec_map=FixtureMap(),
                                   learning_rate=RATE)


def test_open_runtime_refuses_switching_k1_lineage_handles(k1, tmp_path):
    from elpis.runtime import Runtime

    with world(k1) as first_state, world(k1) as second_state:
        with Runtime(_config(tmp_path / "switch")) as runtime:
            runtime.anchor_cognition(first_state)
            first_before = first_state.snapshot()
            second_before = second_state.snapshot()
            with pytest.raises(CompositionError) as info:
                runtime.run_turn(second_state, "wrong lineage", tokenizer=ByteTokens(), codec_map=FixtureMap(),
                                 learning_rate=RATE)
            assert info.value.code == "COGNITION_SUBSTRATE_SWITCH"
            assert first_state.snapshot() == first_before
            assert second_state.snapshot() == second_before
