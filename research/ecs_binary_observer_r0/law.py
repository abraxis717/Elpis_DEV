"""EBO1 encoding law: the pure-Python reference of research/ecs_binary_observer_r0/native/ebo_r0.c.

Same binary64 operations in the same order, so a record from the native observer and from this reference are
bitwise identical (tests/research/ecs_binary_observer_r0 checks it on every laboratory record). The reference is the
specification; the native library is the prototype being measured.

Also here: the science-only matched-budget comparators (RAW16, RAW4). They are not encoders of the record.
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

RECORD_BYTES = 64
FORMAT_VERSION = 1
LAW_VERSION = 1
WINDOW_ROWS = 64
PAYLOAD_BITS = 256
MAX_ROWS = 4096
MAX_DIM = 64
GRAY4_THRESHOLDS = 15
SPROJ_TERMS = 4
FLAG_TRUNCATED, FLAG_PADDED = 1, 2
GRAY4, SPROJ = 1, 2
ENCODERS = {"GRAY4": GRAY4, "SPROJ": SPROJ}

OK, INVALID, NONFINITE, SHAPE, VERSION, LAW = 0, -1, -2, -3, -4, -5

_M64 = (1 << 64) - 1
_H64_K = 0x9E3779B97F4A7C15
_H64_LANE = (0x243F6A8885A308D3, 0x13198A2E03707344, 0xA4093822299F31D0, 0x082EFA98EC4E6C89)


class Refused(Exception):
    def __init__(self, code: int, what: str):
        super().__init__(f"{what} ({code})")
        self.code = code


@dataclass(frozen=True)
class Law:
    encoder: int
    tag: bytes
    blob: bytes
    thresholds: tuple = ()
    mu: float = 0.0
    inv_scale: float = 1.0
    index: tuple = ()      # 256 x 4
    sign: tuple = ()       # 256 x 4, +1 / -1
    tau: tuple = ()        # 256

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.blob).hexdigest()


def blob_bytes(encoder: int) -> int:
    return {GRAY4: 8 + 8 * GRAY4_THRESHOLDS, SPROJ: 8 + 16 + 2 * PAYLOAD_BITS * SPROJ_TERMS + 8 * PAYLOAD_BITS}.get(encoder, 0)


def gray4_blob(thresholds) -> bytes:
    values = [float(t) for t in thresholds]
    if len(values) != GRAY4_THRESHOLDS:
        raise Refused(INVALID, "GRAY4 needs 15 thresholds")
    return b"EBOL" + struct.pack("<HBB", LAW_VERSION, GRAY4, 0) + struct.pack("<15d", *values)


def sproj_blob(mu, inv_scale, index, sign, tau) -> bytes:
    flat_index = [int(i) for row in index for i in row]
    flat_sign = [0x01 if int(s) > 0 else 0xFF for row in sign for s in row]
    if len(flat_index) != PAYLOAD_BITS * SPROJ_TERMS or len(flat_sign) != PAYLOAD_BITS * SPROJ_TERMS or len(tau) != PAYLOAD_BITS:
        raise Refused(INVALID, "SPROJ table shape")
    return (b"EBOL" + struct.pack("<HBB", LAW_VERSION, SPROJ, 0) + struct.pack("<2d", float(mu), float(inv_scale))
            + bytes(flat_index) + bytes(flat_sign) + struct.pack("<256d", *[float(t) for t in tau]))


def parse_law(blob: bytes) -> Law:
    """Admit a canonical blob exactly as ebo_law_init does (same refusal codes)."""
    blob = bytes(blob)
    if len(blob) < 8:
        raise Refused(INVALID, "law blob too short")
    if blob[:4] != b"EBOL":
        raise Refused(LAW, "law magic")
    if blob[4] | (blob[5] << 8) != LAW_VERSION:
        raise Refused(VERSION, "law version")
    encoder = blob[6]
    if blob_bytes(encoder) == 0 or blob[7] != 0:
        raise Refused(LAW, "law encoder")
    if len(blob) != blob_bytes(encoder):
        raise Refused(INVALID, "law size")
    tag = hashlib.sha256(blob).digest()[:4]
    if encoder == GRAY4:
        t = struct.unpack_from("<15d", blob, 8)
        for i, v in enumerate(t):
            if not math.isfinite(v) or (i > 0 and not v > t[i - 1]):
                raise Refused(LAW, "GRAY4 thresholds")
        return Law(GRAY4, tag, blob, thresholds=t)
    mu, inv_scale = struct.unpack_from("<2d", blob, 8)
    if not (math.isfinite(mu) and math.isfinite(inv_scale) and inv_scale > 0.0):
        raise Refused(LAW, "SPROJ normalization")
    n = PAYLOAD_BITS * SPROJ_TERMS
    raw_index, raw_sign = blob[24:24 + n], blob[24 + n:24 + 2 * n]
    if any(i >= WINDOW_ROWS for i in raw_index) or any(s not in (0x01, 0xFF) for s in raw_sign):
        raise Refused(LAW, "SPROJ table")
    tau = struct.unpack_from("<256d", blob, 24 + 2 * n)
    if not all(math.isfinite(v) for v in tau):
        raise Refused(LAW, "SPROJ tau")
    index = tuple(tuple(raw_index[k * SPROJ_TERMS:(k + 1) * SPROJ_TERMS]) for k in range(PAYLOAD_BITS))
    sign = tuple(tuple(1 if s == 0x01 else -1 for s in raw_sign[k * SPROJ_TERMS:(k + 1) * SPROJ_TERMS])
                 for k in range(PAYLOAD_BITS))
    return Law(SPROJ, tag, blob, mu=mu, inv_scale=inv_scale, index=index, sign=sign, tau=tau)


def _bits(v: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", v))[0]


def _rotl(v: int, r: int) -> int:
    return ((v << r) | (v >> (64 - r))) & _M64


def _mix(z: int) -> int:
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def input_identity(dim: int, rows: int, x, window: int) -> int:
    lane = list(_H64_LANE)
    for j in range(window * dim):
        w = _bits(x[j])
        lane[j & 3] = _rotl(((lane[j & 3] ^ w) * _H64_K) & _M64, 31)
    h = _mix(dim ^ (rows << 8) ^ (window << 24))
    for v in lane:
        h = _mix(h ^ v)
    return h


def gray4_level(law: Law, v: float) -> int:
    return sum(1 for t in law.thresholds if v >= t)


def _gray4_payload(law: Law, values, window: int) -> bytes:
    out = bytearray(32)
    for r in range(window):
        b = gray4_level(law, values[r])
        b ^= b >> 1
        out[r >> 1] |= b << (4 * (r & 1))
    return bytes(out)


def sproj_sums(law: Law, values, window: int) -> list:
    u = [(values[r] - law.mu) * law.inv_scale if r < window else 0.0 for r in range(WINDOW_ROWS)]
    sums = []
    for k in range(PAYLOAD_BITS):
        s = 0.0
        for i in range(SPROJ_TERMS):
            t = u[law.index[k][i]]
            s = s + (t if law.sign[k][i] > 0 else -t)
        sums.append(s)
    return sums


def _sproj_payload(law: Law, values, window: int) -> bytes:
    out = bytearray(32)
    for k, s in enumerate(sproj_sums(law, values, window)):
        out[k >> 3] |= int(s >= law.tau[k]) << (k & 7)
    return bytes(out)


def encode(law: Law, dim: int, rows: int, x, values, digest: bytes) -> bytes:
    """One record, or Refused with the native refusal code. Reads only the window rows of x and values."""
    if law is None or x is None or values is None or digest is None:
        raise Refused(INVALID, "argument")
    if not 1 <= dim <= MAX_DIM or not 1 <= rows <= MAX_ROWS:
        raise Refused(SHAPE, "shape")
    if len(digest) != 32:
        raise Refused(INVALID, "digest")
    window = min(rows, WINDOW_ROWS)
    if not all(math.isfinite(values[r]) for r in range(window)):
        raise Refused(NONFINITE, "answer")
    if not all(math.isfinite(x[j]) for j in range(window * dim)):
        raise Refused(NONFINITE, "query row")
    ident = input_identity(dim, rows, x, window)
    flags = (FLAG_TRUNCATED if rows > WINDOW_ROWS else 0) | (FLAG_PADDED if rows < WINDOW_ROWS else 0)
    head = (b"EBOR" + struct.pack("<HBBHBB", FORMAT_VERSION, law.encoder, flags, rows, dim, window) + law.tag
            + bytes(digest[:8]) + struct.pack("<Q", ident))
    payload = _gray4_payload(law, values, window) if law.encoder == GRAY4 else _sproj_payload(law, values, window)
    return head + payload


@dataclass(frozen=True)
class Record:
    encoder: int
    flags: int
    rows: int
    dim: int
    window: int
    law_tag: bytes
    state_identity: bytes
    input_identity: int
    payload: bytes


def parse_record(record: bytes) -> Record:
    if len(record) != RECORD_BYTES or record[:4] != b"EBOR":
        raise Refused(INVALID, "record")
    version, encoder, flags, rows, dim, window = struct.unpack_from("<HBBHBB", record, 4)
    if version != FORMAT_VERSION:
        raise Refused(VERSION, "record version")
    return Record(encoder, flags, rows, dim, window, record[12:16], record[16:24],
                  struct.unpack_from("<Q", record, 24)[0], record[32:64])


def check_record(law: Law, record: bytes) -> Record:
    rec = parse_record(record)
    if rec.encoder != law.encoder or rec.law_tag != law.tag:
        raise Refused(LAW, "record bound to another law")
    window = min(rec.rows, WINDOW_ROWS)
    flags = (FLAG_TRUNCATED if rec.rows > WINDOW_ROWS else 0) | (FLAG_PADDED if rec.rows < WINDOW_ROWS else 0)
    if not 1 <= rec.rows <= MAX_ROWS or not 1 <= rec.dim <= MAX_DIM or rec.window != window or rec.flags != flags:
        raise Refused(INVALID, "record header")
    if law.encoder == GRAY4 and any((rec.payload[r >> 1] >> (4 * (r & 1))) & 0xF for r in range(window, WINDOW_ROWS)):
        raise Refused(INVALID, "record padding")
    return rec


# --- decoding (consumers and the laboratory) -------------------------------------------------------------------

_UNGRAY = {g ^ (g >> 1): g for g in range(16)}


def payload_bits(payload: bytes) -> list:
    return [(payload[k >> 3] >> (k & 7)) & 1 for k in range(PAYLOAD_BITS)]


def gray4_levels(payload: bytes, window: int) -> list:
    return [_UNGRAY[(payload[r >> 1] >> (4 * (r & 1))) & 0xF] for r in range(window)]


def gray4_midpoints(law: Law) -> tuple:
    """A fixed value per level: the midpoint of its interval; the open outer levels extend by the adjacent width."""
    t = law.thresholds
    lo = t[0] - (t[1] - t[0]) / 2.0
    hi = t[-1] + (t[-1] - t[-2]) / 2.0
    return (lo,) + tuple((t[k] + t[k + 1]) / 2.0 for k in range(GRAY4_THRESHOLDS - 1)) + (hi,)


# --- matched-budget comparators (science only; 256 bits each) ----------------------------------------------------

def raw16(values) -> bytes:
    """The first 16 answers as IEEE binary16 (round to nearest even; saturated at +-65504). Not an encoder."""
    out = []
    for v in list(values[:16]) + [0.0] * max(0, 16 - len(values)):
        out.append(struct.pack("<e", max(-65504.0, min(65504.0, float(v)))))
    return b"".join(out)


def raw4(values) -> bytes:
    """The first 4 answers as binary64. Not an encoder."""
    vals = list(values[:4]) + [0.0] * max(0, 4 - len(values))
    return struct.pack("<4d", *[float(v) for v in vals])
