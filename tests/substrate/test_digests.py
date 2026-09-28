from __future__ import annotations

import array

import pytest

import elpis.substrate.digests as file_assets
from elpis.substrate.digests import identity


@pytest.mark.parametrize("size", [0, 1, 2, 15, 16, 17, 65535, 65536, 65537, 200003])
def test_streaming_raw_digest_is_exact_legacy_identity(size):
    data = bytes((index * 37 + 11) & 0xFF for index in range(size))
    expected = identity("raw-bytes", data)
    assert file_assets.raw_digest(data) == expected
    assert file_assets.raw_digest(bytearray(data)) == expected
    assert file_assets.raw_digest(memoryview(data)) == expected


def test_streaming_raw_digest_preserves_nonbyte_contiguous_memoryview_identity():
    values = array.array("H", [0, 1, 255, 256, 65535])
    view = memoryview(values)
    assert file_assets.raw_digest(view) == identity("raw-bytes", bytes(view))


def test_streaming_raw_digest_hexlify_is_chunk_bounded(monkeypatch):
    real = file_assets.binascii.hexlify
    seen = []

    def observed(value):
        seen.append(len(value))
        return real(value)

    monkeypatch.setattr(file_assets.binascii, "hexlify", observed)
    size = file_assets._RAW_DIGEST_CHUNK * 3 + 17
    data = b"x" * size
    assert file_assets.raw_digest(data) == identity("raw-bytes", data)
    assert seen
    assert max(seen) <= file_assets._RAW_DIGEST_CHUNK
    assert len(seen) == 4
