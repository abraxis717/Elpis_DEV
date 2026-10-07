from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

from elpis.ECS_G.k1 import K1Library, K1State


def _library() -> K1Library:
    root = Path(os.environ["ELPIS_NATIVE_BUILD"]).resolve()
    matches = sorted(
        p
        for p in root.rglob("libelpis_ecsg_k1.so")
        if p.is_file()
    )
    assert len(matches) == 1
    return K1Library(C.CDLL(str(matches[0])))


def test_state_digest_is_exact_snapshot_trailer():
    k1 = _library()

    with K1State.create(
        k1,
        6,
        2,
        [0.01 * (i + 1) for i in range(12)],
        max_rows=4,
    ) as state:
        digest = state.state_digest()
        snapshot = state.snapshot()

        assert type(digest) is bytes
        assert len(digest) == 32
        assert digest == snapshot[-32:]


def test_restore_preserves_exact_state_digest():
    k1 = _library()

    with K1State.create(
        k1,
        6,
        2,
        [0.01 * (i + 1) for i in range(12)],
        max_rows=4,
    ) as state:
        snapshot = state.snapshot()
        expected = state.state_digest()

        with K1State.restore(k1, snapshot) as restored:
            assert restored.state_digest() == expected
            assert restored.snapshot()[-32:] == expected
