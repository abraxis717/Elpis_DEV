"""Counter-based random streams (splitmix64 finaliser over a Weyl sequence).

Every random array of the benchmark is a *stream* addressed by a tuple of labels. The stream key is
the first eight bytes (little endian) of SHA-256 of ``ROOT + '/'.join(labels)``; element ``i`` is the
splitmix64 finaliser of ``key + (i + 1) * GOLDEN (mod 2**64)``. Integer arithmetic only, so uniforms
are bitwise identical on every platform and numpy version; element ``i`` never depends on how many
elements were requested before it, so an episode's data does not depend on batching.
"""
from __future__ import annotations

import hashlib

import numpy as np

ROOT = "elpis/cgc_lao/solver_lab/v1/"
GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31, _S11 = np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11)
_INV53 = 1.0 / 9007199254740992.0  # 2**-53


def derive_key(*labels: object) -> int:
    """The uint64 stream key of a label path (labels are joined with '/')."""
    text = ROOT + "/".join(str(label) for label in labels)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def _finalise(z: np.ndarray) -> np.ndarray:
    z = (z ^ (z >> _S30)) * _M1
    z = (z ^ (z >> _S27)) * _M2
    return z ^ (z >> _S31)


def bits(key: int, count: int, offset: int = 0) -> np.ndarray:
    """``count`` uint64 words of stream ``key`` starting at element ``offset``."""
    if count < 0 or offset < 0:
        raise ValueError("RNG_RANGE")
    idx = np.arange(offset + 1, offset + 1 + count, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return _finalise(np.uint64(key) + idx * GOLDEN)


def uniform(key: int, count: int, offset: int = 0) -> np.ndarray:
    """Uniforms in [0, 1) with 53 random bits each."""
    return (bits(key, count, offset) >> _S11).astype(np.float64) * _INV53


def normal(key: int, count: int, offset: int = 0) -> np.ndarray:
    """Standard normals by Box-Muller (cosine branch); element i consumes uniforms 2i and 2i+1."""
    u = uniform(key, 2 * count, 2 * offset)
    u1 = 1.0 - u[0::2]  # (0, 1]
    u2 = u[1::2]
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
