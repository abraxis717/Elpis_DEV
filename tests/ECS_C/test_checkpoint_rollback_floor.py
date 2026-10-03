from __future__ import annotations
from pathlib import Path
import pytest
from elpis.ECS_C.errors import WrongAuthorityError
from elpis.ECS_C.kernel import Kernel
from elpis.ECS_C.persistence import LENGTH_PREFIX
from elpis.ECS_C.replay import replay_with_checkpoint


def _history(directory: str):
    k = Kernel(directory).open()
    a = k.found_entity("alpha")
    b = k.found_entity("beta")
    k.run_until_quiescent()
    k.entity_port(a).propose(b, b"one")
    k.run_until_quiescent()
    cp = k.checkpoint()
    events = k.events()
    root = k.state_root_digest()
    genesis = k._genesis_digest
    k.close()
    return events, cp, root, genesis


def _frame_boundaries(path: Path):
    data = path.read_bytes()
    offsets = [0]
    pos = 0
    while pos < len(data):
        assert len(data) - pos >= LENGTH_PREFIX.size
        length = LENGTH_PREFIX.unpack(data[pos:pos + LENGTH_PREFIX.size])[0]
        pos += LENGTH_PREFIX.size + length
        assert pos <= len(data)
        offsets.append(pos)
    return offsets


def test_complete_frame_truncation_behind_checkpoint_is_rejected(tmp_path):
    directory = str(tmp_path)
    events, cp, _root, _genesis = _history(directory)
    assert cp.event_index == len(events) - 1
    path = Path(directory) / "events.log"
    boundaries = _frame_boundaries(path)
    assert len(boundaries) >= 3
    path.write_bytes(path.read_bytes()[:boundaries[-3]])
    with pytest.raises(WrongAuthorityError, match="HISTORY_ROLLBACK"):
        Kernel(directory).open()


def test_appended_history_after_checkpoint_is_legal(tmp_path):
    directory = str(tmp_path)
    _events, cp, _root, _genesis = _history(directory)
    k = Kernel(directory).open()
    k.found_entity("gamma")
    post_root = k.state_root_digest()
    assert cp.event_index < len(k.events()) - 1
    k.close()
    reopened = Kernel(directory).open()
    assert reopened.state_root_digest() == post_root
    reopened.close()


def test_checkpoint_event_divergence_is_rejected(tmp_path):
    events, cp, _root, genesis = _history(str(tmp_path))
    marker = cp.to_dict()
    marker["event_digest"] = "0" * 64
    with pytest.raises(WrongAuthorityError, match="HISTORY_DIVERGENCE"):
        replay_with_checkpoint(genesis, events, marker)


def test_tail_state_root_divergence_is_rejected(tmp_path):
    events, cp, _root, genesis = _history(str(tmp_path))
    marker = cp.to_dict()
    marker["state_root_digest"] = "f" * 64
    with pytest.raises(WrongAuthorityError, match="HISTORY_DIVERGENCE"):
        replay_with_checkpoint(genesis, events, marker)


def test_no_checkpoint_history_still_replays(tmp_path):
    directory = str(tmp_path)
    k = Kernel(directory).open()
    k.found_entity("alpha")
    root = k.state_root_digest()
    k.close()
    checkpoint = Path(directory) / "checkpoint.bin"
    if checkpoint.exists():
        checkpoint.unlink()
    reopened = Kernel(directory).open()
    assert reopened.state_root_digest() == root
    reopened.close()


def test_deleted_marker_is_explicitly_outside_rollback_floor(tmp_path):
    directory = str(tmp_path)
    _events, _cp, root, _genesis = _history(directory)
    checkpoint = Path(directory) / "checkpoint.bin"
    assert checkpoint.exists()
    checkpoint.unlink()
    reopened = Kernel(directory).open()
    assert reopened.state_root_digest() == root
    reopened.close()
