"""Numpy float32 port of the checked-in Q4E5 CGC reference (``research/cgc_lao/reference``).

``cgc_forward`` is ``FrozenCGC3.forward``: ``z = einsum('bd,odn->bon', features, w)``,
``0.5 * (z + z^2 + z^3).sum(-1)``. ``cgc_step`` is ``ProbeEarlyEpisode.observe``: residual features,
``delta = 0.25 * tanh(cgc(features))``, ``state <- clip(state - clip(delta), +-1.5)``. The optional
``attenuation`` multiplies the clipped proposal (``STATIC_ATTENUATION``); ``attenuation = 1`` is the
reference. Weights are a frozen, read-only float32 [3, 5, 8] array; nothing here can write to them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .env import K, sigmoid

HORIZON = 8
STEP_CAP = np.float32(0.25)
STATE_CAP = np.float32(1.5)
WEIGHT_SHAPE = (3, 5, 8)


def frozen_weights(values) -> np.ndarray:
    """Validate and freeze a [3, 5, 8] float32 coefficient array (a read-only copy)."""
    w = np.array(values, dtype=np.float32).reshape(WEIGHT_SHAPE)
    if not np.isfinite(w).all():
        raise ValueError("CGC_WEIGHT_CONTRACT")
    w.setflags(write=False)
    return w


def weights_digest(w: np.ndarray) -> str:
    """SHA-256 of the little-endian float32 bytes in C order."""
    return hashlib.sha256(np.ascontiguousarray(w, dtype="<f4").tobytes()).hexdigest()


def load_weights(path: Path) -> np.ndarray:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    w = frozen_weights(np.array([float.fromhex(h) for h in record["weights_float32_hex"]], dtype=np.float32))
    if weights_digest(w) != record["weights_sha256"]:
        raise ValueError("CGC_WEIGHT_DIGEST_MISMATCH")
    return w


def cgc_forward(features: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``FrozenCGC3.forward`` in float32: features [B, 5] -> [B, 3]."""
    if features.ndim != 2 or features.shape[1] != 5 or features.dtype != np.float32 or w.dtype != np.float32:
        raise ValueError("CGC_FEATURE_CONTRACT")
    z = np.einsum("bd,odn->bon", features, w)
    return np.float32(0.5) * (z + z * z + z * z * z).sum(-1)


def cgc_step(t: int, base: np.ndarray, state: np.ndarray, action: np.ndarray, reward: np.ndarray,
             w: np.ndarray, attenuation: float = 1.0):
    """One ``ProbeEarlyEpisode.observe`` update. Returns (next_state, proposal delta)."""
    E = state.shape[0]
    idx = np.arange(E)
    pred = sigmoid(base + state)
    onehot = np.eye(K, dtype=np.float32)[action]
    residual = pred[idx, action] - reward
    grad = np.float32(3) * onehot * residual[:, None]
    surprise = (residual * residual)[:, None]
    clock = np.full_like(surprise, np.float32((t + 1) / HORIZON))
    features = np.concatenate((np.tanh(grad), np.tanh(surprise), clock), -1)
    delta = STEP_CAP * np.tanh(cgc_forward(features, w))
    applied = np.clip(delta, -STEP_CAP, STEP_CAP)
    if attenuation != 1.0:
        applied = np.float32(attenuation) * applied
    return np.clip(state - applied, -STATE_CAP, STATE_CAP).astype(np.float32), delta
