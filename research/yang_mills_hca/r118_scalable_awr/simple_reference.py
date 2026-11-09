"""R118-B intentionally simple exact reference of the AWR semantics. Small graphs only.

Pure Python: dictionaries, explicit loops, full 4x4 matrices as nested lists,
``math`` for exp/tanh/sqrt. No numpy, no torch, no index arithmetic. It shares
nothing with the substrate except the controller algebra (contract.py, which
defers to the published reference.py). It is the independent check for
indexing, isolation, transport and reconciliation in the optimized path.

Data model:
    relations : {relation_id: {participant_id: frame8}}
    instances : {instance_id: [relation_id or None] * 4}
    state     : {relation_id: [A0..A3, B0..B3]}   (relations bound by any instance)
"""
from __future__ import annotations

import math

from . import contract as C


def _mat(blocks):
    b = [float(x) for x in blocks]
    return [[b[0], b[1], 0.0, 0.0], [b[2], b[3], 0.0, 0.0], [0.0, 0.0, b[4], b[5]], [0.0, 0.0, b[6], b[7]]]


def _acc(terms):
    """Plain left-to-right float sum. (Builtin ``sum`` is compensated since Python 3.12.)"""
    acc = 0.0
    for t in terms:
        acc += t
    return acc


def _mul_t(U, V):
    """U @ V^T for 4x4 nested lists."""
    return [[_acc(U[i][k] * V[j][k] for k in range(4)) for j in range(4)] for i in range(4)]


def _apply(L, x):
    return [_acc(L[i][j] * x[j] for j in range(4)) for i in range(4)]


def _dot(x, y):
    return _acc(x[k] * y[k] for k in range(4))


def window_links(relations, slots):
    links = []
    for d in range(4):
        if slots[d] is None:
            continue
        md = relations[slots[d]]
        for s in range(4):
            if s == d or slots[s] is None:
                continue
            ms = relations[slots[s]]
            for p in sorted(set(md) & set(ms)):
                links.append((d, s, _mul_t(_mat(md[p]), _mat(ms[p]))))
    return links


def hca_window(W, bias, slots, relations, state):
    """Frozen per-slot law over one window; returns prediction {slot: 8-list} and state32 lists."""
    A = [[0.0] * 4 for _ in range(4)]
    B = [[0.0] * 4 for _ in range(4)]
    for d in range(4):
        if slots[d] is not None:
            A[d] = [float(v) for v in state[slots[d]][:4]]
            B[d] = [float(v) for v in state[slots[d]][4:]]
    links = window_links(relations, slots)
    pred = {}
    for d in range(4):
        if slots[d] is None:
            continue
        mine = [(s, L) for (dd, s, L) in links if dd == d]
        sal = [0.0] * 4
        col = [0.0] * 4
        if mine:
            w = [_apply(L, A[s]) for s, L in mine]
            wc = [_apply(L, B[s]) for s, L in mine]
            x = [8.0 * math.sqrt(_dot(v, v)) for v in w]
            mx = max(x)
            ex = [math.exp(v - mx) for v in x]
            tot = 0.0
            for v in ex:
                tot += v
            for e, v in enumerate(w):
                pe = ex[e] / tot
                for k in range(4):
                    sal[k] += pe * v[k]
            for k in range(4):
                acc = 0.0
                for v in wc:
                    acc += v[k]
                col[k] = acc / len(wc)
        a, b = A[d], B[d]
        feats = [_dot(a, a), _dot(b, b), _dot(sal, sal), _dot(col, col), _dot(a, sal), _dot(a, col),
                 _dot(b, sal), _dot(b, col), _dot(sal, col), 0.0]
        f = [v / (1.0 + abs(v)) for v in feats]
        kk = []
        for q in range(8):
            z = 0.0
            for j in range(10):
                z += W[q][j] * f[j]
            z = z + bias[q]
            kk.append(1.0 / (1.0 + math.exp(-z)) if q in (0, 4) else math.tanh(z))
        ua = [kk[0] * a[k] + kk[1] * sal[k] + kk[2] * col[k] + kk[3] * b[k] for k in range(4)]
        ub = [kk[4] * b[k] + kk[5] * sal[k] + kk[6] * col[k] + kk[7] * a[k] for k in range(4)]
        for v in (ua, ub):
            scale = 4.0 / (math.sqrt(_dot(v, v)) + 1e-30)
            if scale > 1.0:
                scale = 1.0
            for k in range(4):
                v[k] = v[k] * scale
        pred[d] = ua + ub
    s32 = [A[d][k] for d in range(4) for k in range(4)] + [B[d][k] for d in range(4) for k in range(4)]
    p32 = [0.0] * 32
    for d, v in pred.items():
        p32[d * 4:(d + 1) * 4] = v[:4]
        p32[16 + d * 4:16 + (d + 1) * 4] = v[4:]
    return s32, p32


def step(W, bias, relations, instances, state, active, checkpoint, controller, passes=1, mode="MEAN"):
    """Synchronous passes over the active instances; returns the new state dict."""
    state = {r: list(v) for r, v in state.items()}
    for _ in range(passes):
        cands = {}
        for iid in sorted(active):
            slots = instances[iid]
            s32, p32 = hca_window(W, bias, slots, relations, state)
            q32 = C.correct_instance_exact(s32, p32, controller, checkpoint)
            for d in range(4):
                if slots[d] is not None:
                    cands.setdefault(slots[d], []).append(list(q32[d * 4:(d + 1) * 4]) + list(q32[16 + d * 4:16 + (d + 1) * 4]))
        if mode == "EXCLUSIVE" and any(len(v) > 1 for v in cands.values()):
            raise ValueError("OVERLAP_CONFLICT")
        for r, vecs in cands.items():
            new = []
            for k in range(8):
                vals = sorted(v[k] for v in vecs)
                acc = vals[0]
                for x in vals[1:]:
                    acc += x
                new.append(acc / len(vecs))
            state[r] = new
    return state
