"""ECS_G K1 cognition continuity across ECS_C history compaction (real native K1).

The anchor and early turn receipts are retired below the retention floor;
restart must still reconcile the K1 lineage from the fixed-size continuity
summary carried in the compaction checkpoint, without re-reading retired
receipts, without a new anchor, and with every safety property intact:
a mismatching K1 state fails before any K1 mutation, no receipt is
synthesized, and anchoring again is refused.
"""
from __future__ import annotations

import ctypes

import pytest

from elpis.ECS_G.k1 import K1Library
from elpis.runtime.cognition import run_turn
from elpis.runtime.composition import CompositionError
from elpis.runtime.history import ReceiptRecord, RuntimeHistoryPolicy

from ._turn_fixtures import ByteTokens, FixtureMap
from .test_codec_ecs_turn import RATE, _beside, world


@pytest.fixture(scope="module")
def k1():
    return K1Library(ctypes.CDLL(str(_beside("libelpis_ecsg_k1.so"))))

KiB = 1024
_probe = RuntimeHistoryPolicy(max_segment_bytes=64 * KiB, max_segment_events=64,
                              max_checkpoint_bytes=16 * KiB, max_directory_bytes=1 << 40)
SMALL = RuntimeHistoryPolicy(max_segment_bytes=64 * KiB, max_segment_events=64,
                             max_checkpoint_bytes=16 * KiB,
                             max_directory_bytes=_probe.required_directory_bytes)


def _filler(runtime, start, until_generation):
    n = start
    while runtime.history.generation < until_generation:
        runtime.history.record(ReceiptRecord.of("pipeline", "filler", format(n, "064x"), pad="f" * 200))
        n += 1
    return n


def _config(tmp_path, history_library):
    from elpis.runtime import RuntimeConfig
    return RuntimeConfig(tmp_path / "history", history_native_library=history_library,
                         history_policy=SMALL)


def test_k1_lineage_survives_compaction_and_restart(k1, history_library, tmp_path):
    from elpis.runtime import Runtime

    config = _config(tmp_path, history_library)
    with world(k1) as state:
        with Runtime(config) as runtime:
            runtime.anchor_cognition(state)
            first = runtime.run_turn(state, "before compaction", tokenizer=ByteTokens(),
                                     codec_map=FixtureMap(), learning_rate=RATE)
            n = _filler(runtime, 0, 3)
            # The anchor and first turn are retired below the floor.
            assert runtime.history.retention_floor > 0
            kinds = [item.record.kind for item in runtime.history.retained_records()]
            assert "cognition.anchor" not in kinds and "cognition.turn" not in kinds
            # The open runtime keeps going on the same lineage.
            second = runtime.run_turn(state, "after compaction", tokenizer=ByteTokens(),
                                      codec_map=FixtureMap(), learning_rate=RATE)
            assert second.continuity.state_before_digest == first.continuity.state_after_digest
            n = _filler(runtime, n, 5)
            tip = runtime.history.cognition_tip()
            assert tip == state.state_digest().hex()
            receipts = runtime.history.receipt_count

        with Runtime(config) as restarted:
            assert restarted.history.cognition_tip() == tip
            assert restarted.history.receipt_count == receipts
            assert restarted.history.continuity.records == 3  # anchor + two turns
            # Restart reconciles from the summary: no new anchor, turn continues.
            third = restarted.run_turn(state, "after restart", tokenizer=ByteTokens(),
                                       codec_map=FixtureMap(), learning_rate=RATE)
            assert third.continuity.state_before_digest == tip
            with pytest.raises(CompositionError) as info:
                restarted.anchor_cognition(state)
            assert info.value.code == "COGNITION_ALREADY_ANCHORED"
            anchors = [item for item in restarted.history.retained_records()
                       if item.record.kind == "cognition.anchor"]
            assert anchors == []


def test_k1_mismatch_after_compaction_fails_before_mutation(k1, history_library, tmp_path):
    from elpis.runtime import Runtime

    config = _config(tmp_path, history_library)
    with world(k1) as anchored, world(k1) as other:
        with Runtime(config) as runtime:
            runtime.anchor_cognition(anchored)
            _filler(runtime, 0, 3)
            receipts = runtime.history.receipt_count

        run_turn(other, "out of band", tokenizer=ByteTokens(), codec_map=FixtureMap(), learning_rate=RATE)
        before = other.snapshot()

        with Runtime(config) as restarted:
            with pytest.raises(CompositionError) as info:
                restarted.run_turn(other, "must refuse", tokenizer=ByteTokens(),
                                   codec_map=FixtureMap(), learning_rate=RATE)
            assert info.value.code == "HISTORY_STATE_MISMATCH"
            assert other.snapshot() == before
            # Nothing synthesized: no receipt was added by the refusal.
            assert restarted.history.receipt_count == receipts


def test_unanchored_history_after_compaction_still_requires_explicit_anchor(k1, history_library, tmp_path):
    from elpis.runtime import Runtime

    config = _config(tmp_path, history_library)
    with world(k1) as state:
        with Runtime(config) as runtime:
            _filler(runtime, 0, 3)
        before = state.snapshot()
        with Runtime(config) as restarted:
            with pytest.raises(CompositionError) as info:
                restarted.run_turn(state, "unanchored", tokenizer=ByteTokens(),
                                   codec_map=FixtureMap(), learning_rate=RATE)
            assert info.value.code == "COGNITION_UNANCHORED"
            assert state.snapshot() == before
            assert restarted.history.cognition_tip() is None
