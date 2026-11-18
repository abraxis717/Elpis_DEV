"""R7: immutable, row-addressable PLE tables for inference. Read only: never trained, never written, never cached.

One contract, two backends. ``read(addresses)`` returns ``[..., dim]`` at the stored precision, from an exact copy of
a trained table (``ple.PLE.attach_store`` checks shape and addressing):

    TensorStore   an immutable in-memory copy (the reference)
    FileStore     a file with a fixed header and row-major rows, opened read only; each call reads exactly the
                  distinct rows it needs with positioned reads (no memory mapping, no cache, no full load)

File format, version 1 (little-endian):

    offset 0              8 bytes     magic b"ELPISPLE"
    offset 8              4 bytes     n, the header's length
    offset 12             n bytes     header, UTF-8 JSON: version, dtype ("float32" | "bfloat16"), rows, dim,
                                      row_bytes, data_offset, addressing_digest, content_sha256
    data_offset (64-aligned)          rows x dim values, row-major

The file is written only by the experiment harness, an explicit offline operation (``experiments/run_r7.py``).
Counters: calls, rows requested, distinct rows read, positioned reads and logical bytes read, kept in memory and
reset on request. Nothing persists between calls. Page-cache hits are possible, and ``FileStore`` reports no
physical device timing.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import torch

MAGIC = b"ELPISPLE"
FORMAT_VERSION = 1
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def _counters() -> dict:
    return {"calls": 0, "rows_requested": 0, "rows_read": 0, "reads": 0, "bytes_read": 0}


class TensorStore:
    """An immutable copy of a table (``dtype``: the stored precision; float32 is the numerical reference)."""

    def __init__(self, table: torch.Tensor, *, addressing_digest: str, dtype: torch.dtype = torch.float32):
        if table.ndim != 2:
            raise ValueError("a table is [rows, dim]")
        if dtype not in DTYPES.values():
            raise ValueError(f"stored precision must be one of {sorted(DTYPES)}")
        self._table = table.detach().to(device="cpu", dtype=dtype).clone()
        self._table.requires_grad_(False)
        self.rows, self.dim = self._table.shape
        self.dtype, self.element_size = dtype, self._table.element_size()
        self.row_bytes = self.dim * self.element_size
        self.addressing_digest = addressing_digest
        self.counters = _counters()

    def read(self, addresses: torch.Tensor) -> torch.Tensor:
        flat = _validated(addresses, self.rows)
        _count(self.counters, flat.numel(), torch.unique(flat).numel(), 1, self.row_bytes)
        return self._table.index_select(0, flat).view(*addresses.shape, self.dim)

    def reset_counters(self) -> None:
        self.counters = _counters()


class FileStore:
    """A table in a read-only file (format above). Holds an open read-only descriptor and the header."""

    def __init__(self, path):
        if sys.byteorder != "little":
            raise RuntimeError("the store format is little-endian")
        self.path = os.fspath(path)
        self._fd = os.open(self.path, os.O_RDONLY)
        try:
            head = os.pread(self._fd, 12, 0)
            if len(head) != 12 or head[:8] != MAGIC:
                raise ValueError("not an Elpis PLE store")
            size = int.from_bytes(head[8:12], "little")
            header = json.loads(os.pread(self._fd, size, 12).decode("utf-8"))
            if header.get("version") != FORMAT_VERSION or header.get("dtype") not in DTYPES:
                raise ValueError("unsupported store version or dtype")
            self.dtype = DTYPES[header["dtype"]]
            self.element_size = torch.empty(0, dtype=self.dtype).element_size()
            self.rows, self.dim = int(header["rows"]), int(header["dim"])
            self.row_bytes, self.data_offset = int(header["row_bytes"]), int(header["data_offset"])
            if self.row_bytes != self.dim * self.element_size or self.data_offset < 12 + size or self.data_offset % 64:
                raise ValueError("inconsistent store header")
            if os.fstat(self._fd).st_size != self.data_offset + self.rows * self.row_bytes:
                raise ValueError("the store's size does not match its header (truncated or extended)")
            self.addressing_digest = header["addressing_digest"]
            self.content_sha256 = header["content_sha256"]
        except Exception:
            os.close(self._fd)
            raise
        self.counters = _counters()

    def read(self, addresses: torch.Tensor) -> torch.Tensor:
        """The rows of ``addresses`` (any shape), read from the file: one positioned read per run of consecutive
        distinct rows. Memory: one buffer of the distinct rows of this call."""
        flat = _validated(addresses, self.rows)
        unique, inverse = torch.unique(flat, return_inverse=True)
        wanted = unique.tolist()
        buffer = bytearray(len(wanted) * self.row_bytes)
        reads, at = 0, 0
        while at < len(wanted):
            end = at + 1
            while end < len(wanted) and wanted[end] == wanted[end - 1] + 1:
                end += 1
            length = (end - at) * self.row_bytes
            data = os.pread(self._fd, length, self.data_offset + wanted[at] * self.row_bytes)
            if len(data) != length:
                raise OSError("short read from the store")
            buffer[at * self.row_bytes:at * self.row_bytes + length] = data
            reads, at = reads + 1, end
        _count(self.counters, flat.numel(), len(wanted), reads, self.row_bytes)
        table = torch.frombuffer(buffer, dtype=self.dtype).view(len(wanted), self.dim) if wanted else \
            torch.empty(0, self.dim, dtype=self.dtype)
        return table.index_select(0, inverse).view(*addresses.shape, self.dim)

    def verify(self) -> bool:
        """EXPLICIT, OFFLINE: read every row once and compare with the recorded content digest."""
        digest, at, total = hashlib.sha256(), self.data_offset, self.rows * self.row_bytes
        while total:
            data = os.pread(self._fd, min(total, 1 << 20), at)
            if not data:
                return False
            digest.update(data)
            at, total = at + len(data), total - len(data)
        return digest.hexdigest() == self.content_sha256

    def reset_counters(self) -> None:
        self.counters = _counters()

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _validated(addresses: torch.Tensor, rows: int) -> torch.Tensor:
    if addresses.dtype != torch.long:
        raise ValueError("addresses must be int64")
    flat = addresses.reshape(-1).cpu()
    if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= rows):
        raise IndexError(f"addresses must be in [0, {rows})")
    return flat


def _count(counters: dict, requested: int, distinct: int, reads: int, row_bytes: int) -> None:
    counters["calls"] += 1
    counters["rows_requested"] += requested
    counters["rows_read"] += distinct
    counters["reads"] += reads
    counters["bytes_read"] += distinct * row_bytes
