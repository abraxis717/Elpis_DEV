from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import elpis.ECS_G.k1 as k1_module
from elpis.ECS_G.k1 import CommitIdentity, K1Library, K1Transaction


def _native_library() -> Path:
    root = Path(os.environ["ELPIS_NATIVE_BUILD"]).resolve()
    matches = sorted(p for p in root.rglob("libelpis_ecsg_k1.so") if p.is_file())
    assert len(matches) == 1
    return matches[0]


class _State:
    def __init__(self, table, handle):
        self._k = table
        self._handle = handle

    def _live(self):
        return self._handle


def _snapshot(table, handle) -> bytes:
    size = int(table.snapshot_size(handle))
    out = (C.c_uint8 * size)()
    assert table.snapshot_write(handle, out, size) == 0
    return bytes(out)


def test_commit_identity_matches_native_envelope_trailer_without_extra_state_call():
    library = K1Library(C.CDLL(str(_native_library())))
    table = library._k
    dim, width = 6, 2
    initial = (C.c_double * (dim * width))(
        *[0.01 * (i + 1) for i in range(dim * width)]
    )
    handle = C.c_void_p()
    assert table.create(dim, width, 4, initial, C.byref(handle)) == 0

    try:
        before = _snapshot(table, handle)
        token = C.c_uint64()
        assert table.txn_begin(handle, C.byref(token)) == 0

        result = K1Transaction(_State(table, handle), token.value).commit_identity()
        after = _snapshot(table, handle)

        assert type(result) is CommitIdentity
        assert result.state_before_digest == before[-32:]
        assert result.state_after_digest == after[-32:]

        # Generation is transition metadata, not a retained-state byte.
        assert before == after
        assert result.state_before_digest == result.state_after_digest
        assert result.commit.generation_before == 0
        assert result.commit.generation_after == 1
        assert result.commit.epoch_before == 0
        assert result.commit.epoch_after == 0

        assert "txn_commit_identity" in k1_module._K1_ABI
        assert "txn_commit_identity" in k1_module._FMS_ABI
        assert hasattr(k1_module._FMSTransaction, "commit_identity")
    finally:
        assert table.destroy(C.byref(handle)) == 0
