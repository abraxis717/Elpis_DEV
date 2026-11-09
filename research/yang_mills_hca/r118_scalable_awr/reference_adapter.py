"""R118-A exact reference adapter: the frozen historical operator over substrate windows.

For each instance this builds the historical link dictionary exactly as
``portable_mechanics.link_builder`` does (dst ascending, src ascending,
shared participant ascending; ``U[dst,p] @ U[src,p].T`` with torch float64
4x4 matmul from losslessly stored frames) and executes the reconstructed
``r64.HCA`` through ``r62_source.graph_step_split``. The controller is the
published algebra (``contract.correct_instance_exact`` -> ``reference.py``).

It is deliberately slow (one historical call per instance) and is the oracle
against which faster backends are qualified. Given historical frames and
states, it reproduces historical rollouts bitwise (qualify_c checks this).
"""
from __future__ import annotations

import math

import numpy as np

from . import contract as C
from .registry import blocks_to_matrix


def window_adj(T, registry, slot_rel) -> dict:
    links = registry.window_links(slot_rel)
    adj = {}
    for d in range(C.SLOTS):
        mine = [lk for lk in links if lk[0] == d]
        if not mine:
            continue
        idx = T.tensor([lk[1] for lk in mine], dtype=T.long)
        ops = T.stack([
            T.from_numpy(blocks_to_matrix(registry.frames[lk[3]])) @ T.from_numpy(blocks_to_matrix(registry.frames[lk[4]])).T
            for lk in mine
        ])
        adj[d] = (idx, ops)
    return adj


def _adj_key(adj) -> bytes:
    parts = []
    for d in sorted(adj):
        idx, ops = adj[d]
        parts.append(bytes([d]) + idx.numpy().tobytes() + ops.numpy().tobytes())
    return b"|".join(parts)


def oracle_candidates(registry, slot_rel, slot_row, cur: dict, checkpoint: str, controller: str):
    """Candidates {row: [8-vectors in instance order]}, or None if any value is nonfinite.

    Instances whose windows have byte-identical link tensors are executed as one
    historical batch (first-appearance order); torch rounding depends on batch
    shape, so this reproduces historical batched rollouts bitwise.
    """
    from .frozen_oracle import get_oracle

    o = get_oracle()
    T = o.torch
    model = o.models()[checkpoint]
    rel_rows = np.asarray(slot_rel)
    row_rows = np.asarray(slot_row)
    groups: dict[bytes, tuple] = {}
    with T.no_grad():
        for i, rel4 in enumerate(rel_rows):
            adj = window_adj(T, registry, rel4)
            key = _adj_key(adj)
            if key not in groups:
                groups[key] = (adj, [])
            groups[key][1].append(i)
        preds = {}
        for adj, members in groups.values():
            a = T.zeros((len(members), 4, 4), dtype=T.float64)
            b = T.zeros((len(members), 4, 4), dtype=T.float64)
            for j, i in enumerate(members):
                for d in range(C.SLOTS):
                    if row_rows[i, d] >= 0:
                        v = cur[int(row_rows[i, d])]
                        a[j, d] = T.from_numpy(np.array(v[:4]))
                        b[j, d] = T.from_numpy(np.array(v[4:]))
            pa, pb = o.step((a, b), adj, model)
            for j, i in enumerate(members):
                preds[i] = (a[j], b[j], pa[j], pb[j])
    out: dict[int, list] = {}
    for i in range(len(rel_rows)):
        a, b, pa, pb = preds[i]
        s32 = C.to_state32(a.tolist(), b.tolist())
        p32 = C.to_state32(pa.tolist(), pb.tolist())
        try:
            q32 = C.correct_instance_exact(s32, p32, controller, checkpoint)
        except ValueError:
            return None
        qa, qb = C.from_state32(q32)
        for d in range(C.SLOTS):
            if row_rows[i, d] >= 0:
                out.setdefault(int(row_rows[i, d]), []).append(np.array(qa[d] + qb[d], dtype=np.float64))
    return out


def reconcile_mean(cands: dict) -> dict:
    """Order-independent mean: per coordinate, ascending-value sequential sum / count."""
    out = {}
    for row, vecs in cands.items():
        m = len(vecs)
        if m == 1:
            out[row] = vecs[0].copy()
            continue
        res = np.empty(C.SLOT_COORDS, dtype=np.float64)
        for k in range(C.SLOT_COORDS):
            vals = sorted(float(v[k]) for v in vecs)
            acc = vals[0]
            for x in vals[1:]:
                acc += x
            res[k] = acc / m
        if not all(map(math.isfinite, res)):
            raise ValueError("NONFINITE_RECONCILIATION")
        out[row] = res
    return out
