"""ctypes view of the research observer library ``libelpis_ebo_r0`` (RESEARCH_ONLY).

This module loads only the observer library. It imports nothing from ``elpis.runtime``, holds no ECS handle and
cannot reach LEARN or continuity: it is handed the caller-owned results of an accepted QUERY and returns a record.
"""
from __future__ import annotations

import ctypes as C
from pathlib import Path

from . import law as L

_D = C.POINTER(C.c_double)
_U8 = C.POINTER(C.c_uint8)


class ObserverError(Exception):
    def __init__(self, code: int, what: str):
        super().__init__(f"{what}: status {code}")
        self.code = code


class Observer:
    """One loaded observer library and one admitted law."""

    def __init__(self, library: Path, blob: bytes):
        lib = C.CDLL(str(library))
        lib.ebo_law_bytes.restype = C.c_size_t
        lib.ebo_ring_bytes.restype = C.c_size_t
        lib.ebo_law_init.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t]
        lib.ebo_law_init.restype = C.c_int
        lib.ebo_encode.argtypes = [C.c_void_p, C.c_size_t, C.c_size_t, _D, _D, _U8, _U8]
        lib.ebo_encode.restype = C.c_int
        lib.ebo_record_check.argtypes = [C.c_void_p, _U8]
        lib.ebo_record_check.restype = C.c_int
        self._lib = lib
        self._law = (C.c_uint8 * lib.ebo_law_bytes())()
        rc = lib.ebo_law_init(self._law, bytes(blob), len(blob))
        if rc != 0:
            raise ObserverError(rc, "law refused")
        self.law = L.parse_law(blob)   # the reference view of the same law (identical admission rules)
        self._record = (C.c_uint8 * L.RECORD_BYTES)()

    def encode(self, dim: int, rows: int, x, values, digest: bytes) -> bytes:
        """x and values are caller-owned binary64 buffers (array('d') / memoryview / ctypes); never written."""
        xb = (C.c_double * (rows * dim)).from_buffer_copy(memoryview(x).cast("B")[: rows * dim * 8])
        vb = (C.c_double * rows).from_buffer_copy(memoryview(values).cast("B")[: rows * 8])
        db = (C.c_uint8 * 32).from_buffer_copy(bytes(digest))
        rc = self._lib.ebo_encode(self._law, dim, rows, xb, vb, db, self._record)
        if rc != 0:
            raise ObserverError(rc, "observation refused")
        return bytes(self._record)

    def check(self, record: bytes) -> int:
        return int(self._lib.ebo_record_check(self._law, (C.c_uint8 * L.RECORD_BYTES).from_buffer_copy(record)))
