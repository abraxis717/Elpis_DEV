"""Gauge actions on the whole substrate (frames and relation states), numpy only.

Relation gauge  g_r in SO(2)xSO(2):  X_r -> g_r X_r,      U[r,p] -> g_r U[r,p]
Participant gauge h_p:                U[r,p] -> U[r,p] h_p^T   (links invariant)

Frames and gauges are [..., 8] block arrays (two row-major 2x2 blocks).
"""
from __future__ import annotations

import numpy as np

from .registry import build_registry, rotation_blocks


def bmul(a, b):
    """Blockwise 2x2 products of [...,8] block arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    for o in (0, 4):
        a0, a1, a2, a3 = (a[..., o + i] for i in range(4))
        b0, b1, b2, b3 = (b[..., o + i] for i in range(4))
        out[..., o + 0] = a0 * b0 + a1 * b2
        out[..., o + 1] = a0 * b1 + a1 * b3
        out[..., o + 2] = a2 * b0 + a3 * b2
        out[..., o + 3] = a2 * b1 + a3 * b3
    return out


def btrans(a):
    a = np.asarray(a, dtype=np.float64)
    return a[..., [0, 2, 1, 3, 4, 6, 5, 7]]


def bapply(g, x):
    """Block-diagonal 4x4 (as [...,8]) times 4-vectors [...,4]."""
    g = np.asarray(g, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return np.stack([g[..., 0] * x[..., 0] + g[..., 1] * x[..., 1],
                     g[..., 2] * x[..., 0] + g[..., 3] * x[..., 1],
                     g[..., 4] * x[..., 2] + g[..., 5] * x[..., 3],
                     g[..., 6] * x[..., 2] + g[..., 7] * x[..., 3]], axis=-1)


def random_so2x2(rng, n):
    return rotation_blocks(rng.uniform(-np.pi, np.pi, n), rng.uniform(-np.pi, np.pi, n), exact_libm=False)


def gauge_registry(reg, g_rel=None, h_part=None):
    """A new registry with frames g_r U[r,p] h_p^T (same IDs, same incidence)."""
    fr = reg.frames
    if g_rel is not None:
        fr = bmul(np.asarray(g_rel)[reg.inc_relation], fr)
    if h_part is not None:
        fr = bmul(fr, btrans(np.asarray(h_part)[reg.members]))
    return build_registry(reg.participant_ids, reg.relation_ids, reg.rel_offsets,
                          reg.participant_ids[reg.members], fr, reg.limits)


def gauge_rows(state, g_rel):
    """g_r applied to every materialized relation row: rows [n,8] -> [n,8]."""
    rows = np.nonzero(state.row_ref > 0)[0]
    g = np.asarray(g_rel)[state.row_rel[rows]]
    out = state.pool.copy()
    out[rows, :4] = bapply(g, state.pool[rows, :4])
    out[rows, 4:] = bapply(g, state.pool[rows, 4:])
    return out
