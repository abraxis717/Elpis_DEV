"""R119 forensics: per-site census of subnormal floating-point operations in the exact path.

An instrumented pure-Python replica of the R118 native kernel (csrc/awr_kernel.c, the
exact path: no FTZ). Every arithmetic operation is executed with Python floats (IEEE-754
binary64, round-to-nearest, no fused multiply-add: identical to the C kernel compiled with
``-ffp-contract=off``; exp/tanh/sqrt come from the same glibc), in the kernel's order, and
classified per code site:

* ``ops``          operations executed;
* ``sub_result``   result subnormal (0 < |r| < 2^-1022);
* ``sub_operand``  at least one operand subnormal;
* ``underflow0``   multiply/divide of nonzero operands whose result rounded to exactly 0.

The replica is only evidence if it is bit-identical to the C kernel: ``census_step``
returns the reconciled rows and the caller compares them bit for bit with the native step.
"""
from __future__ import annotations

import math
import struct

import numpy as np

TINY = 2.2250738585072014e-308  # 2^-1022, smallest positive normal
SITES = ("link_op", "transport", "link_norm", "softmax", "collective", "invariants", "squash", "gates",
         "activation", "update", "clamp", "ctrl_diff", "ctrl_norm", "ctrl_dot", "ctrl_radial", "ctrl_out",
         "reconcile")


def is_sub(x: float) -> bool:
    return x != 0.0 and abs(x) < TINY


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


PENALIZED_OPS = ("mul", "div", "sqrt", "tanh", "add")


class Census:
    """Per site: ops, subnormal results, subnormal operands, underflow-to-zero, and per op
    type the number of *penalized* events -- mul/div/sqrt/tanh with a subnormal result, or a
    subnormal operand and a nonzero result (the classes the R119 microbenchmark measured at
    ~45-52 ns on this CPU; add/sub/exp and underflow to exact zero carry no penalty)."""

    def __init__(self):
        self.c = {s: [0, 0, 0, 0] for s in SITES}
        self.pen = {s: {o: [0, 0] for o in PENALIZED_OPS} for s in SITES}

    def _rec(self, site, r, *operands, mul=False, op=None):
        c = self.c[site]
        c[0] += 1
        sr = is_sub(r)
        so = any(is_sub(o) for o in operands)
        if sr:
            c[1] += 1
        if so:
            c[2] += 1
        if mul and r == 0.0 and all(o != 0.0 for o in operands):
            c[3] += 1
        if op is not None:
            e = self.pen[site][op]
            e[0] += 1
            if op == "add":
                # add/sub: output-denormal assist only when every operand is normal or zero
                if sr and not so:
                    e[1] += 1
            elif sr or (so and r != 0.0):
                e[1] += 1
        return r

    def mul(self, a, b, s):
        return self._rec(s, a * b, a, b, mul=True, op="mul")

    def div(self, a, b, s):
        return self._rec(s, a / b, a, b, mul=True, op="div")

    def add(self, a, b, s):
        return self._rec(s, a + b, a, b, op="add")

    def sub(self, a, b, s):
        return self._rec(s, a - b, a, b, op="add")

    def sqrt(self, a, s):
        return self._rec(s, math.sqrt(a), a, op="sqrt")

    def exp(self, a, s):
        return self._rec(s, math.exp(a), a)

    def tanh(self, a, s):
        return self._rec(s, math.tanh(a), a, op="tanh")

    def as_dict(self):
        out = {}
        for s, v in self.c.items():
            out[s] = dict(zip(("ops", "sub_result", "sub_operand", "underflow0"), v))
            out[s]["penalized"] = {o: {"ops": n, "penalized": k} for o, (n, k) in self.pen[s].items() if n}
        return out


def _dot4(C, x, y, s):
    acc = C.mul(x[0], y[0], s)
    for k in (1, 2, 3):
        acc = C.add(acc, C.mul(x[k], y[k], s), s)
    return acc


def _fsum(C, v, s):
    """Instrumented port of the kernel's math.fsum port (CPython algorithm)."""
    p = []
    for x in v:
        i = 0
        for j in range(len(p)):
            y = p[j]
            if abs(x) < abs(y):
                x, y = y, x
            hi = C.add(x, y, s)
            yr = C.sub(hi, x, s)
            lo = C.sub(y, yr, s)
            if lo != 0.0:
                p[i] = lo
                i += 1
            x = hi
        del p[i:]
        if x != 0.0:
            p.append(x)
    hi = 0.0
    lo = 0.0
    n = len(p)
    if n:
        n -= 1
        hi = p[n]
        while n > 0:
            x = hi
            n -= 1
            y = p[n]
            hi = C.add(x, y, s)
            yr = C.sub(hi, x, s)
            lo = C.sub(y, yr, s)
            if lo != 0.0:
                break
        if n > 0 and ((lo < 0.0 and p[n - 1] < 0.0) or (lo > 0.0 and p[n - 1] > 0.0)):
            y = C.mul(lo, 2.0, s)
            x = C.add(hi, y, s)
            yr = C.sub(x, hi, s)
            if y == yr:
                hi = x
    return hi


def instance_candidates(C, reg, rel4, rows4, cur, W, bias, gains):
    """One instance (kernel ``instance_pass``): returns {slot: 8 floats} for bound slots."""
    A = [[0.0] * 4 for _ in range(4)]
    B = [[0.0] * 4 for _ in range(4)]
    for d in range(4):
        if rows4[d] >= 0:
            v = cur[rows4[d]]
            A[d] = list(v[:4])
            B[d] = list(v[4:])
    S = [[0.0] * 4 for _ in range(4)]
    Cc = [[0.0] * 4 for _ in range(4)]
    off, mem, fr = reg.rel_offsets, reg.members, reg.frames
    for d in range(4):
        if rel4[d] < 0:
            continue
        w, wc, x = [], [], []
        d0, d1 = int(off[rel4[d]]), int(off[rel4[d] + 1])
        for s in range(4):
            if s == d or rel4[s] < 0:
                continue
            i, j, j1 = d0, int(off[rel4[s]]), int(off[rel4[s] + 1])
            while i < d1 and j < j1:
                if mem[i] < mem[j]:
                    i += 1
                elif mem[i] > mem[j]:
                    j += 1
                else:
                    Ud, Us = fr[i].tolist(), fr[j].tolist()
                    L = []
                    for b in (0, 4):
                        dd_, ss_ = Ud[b:b + 4], Us[b:b + 4]
                        L += [C.add(C.mul(dd_[0], ss_[0], "link_op"), C.mul(dd_[1], ss_[1], "link_op"), "link_op"),
                              C.add(C.mul(dd_[0], ss_[2], "link_op"), C.mul(dd_[1], ss_[3], "link_op"), "link_op"),
                              C.add(C.mul(dd_[2], ss_[0], "link_op"), C.mul(dd_[3], ss_[1], "link_op"), "link_op"),
                              C.add(C.mul(dd_[2], ss_[2], "link_op"), C.mul(dd_[3], ss_[3], "link_op"), "link_op")]

                    def ap(v):
                        t = "transport"
                        return [C.add(C.mul(L[0], v[0], t), C.mul(L[1], v[1], t), t),
                                C.add(C.mul(L[2], v[0], t), C.mul(L[3], v[1], t), t),
                                C.add(C.mul(L[4], v[2], t), C.mul(L[5], v[3], t), t),
                                C.add(C.mul(L[6], v[2], t), C.mul(L[7], v[3], t), t)]
                    we, wce = ap(A[s]), ap(B[s])
                    w.append(we)
                    wc.append(wce)
                    x.append(C.mul(8.0, C.sqrt(_dot4(C, we, we, "link_norm"), "link_norm"), "link_norm"))
                    i += 1
                    j += 1
        ne = len(w)
        if ne == 0:
            continue
        mx = x[0]
        for e in range(1, ne):
            if x[e] > mx:
                mx = x[e]
        tot = 0.0
        for e in range(ne):
            x[e] = C.exp(C.sub(x[e], mx, "softmax"), "softmax")
            tot = C.add(tot, x[e], "softmax")
        for e in range(ne):
            pe = C.div(x[e], tot, "softmax")
            for k in range(4):
                S[d][k] = C.add(S[d][k], C.mul(pe, w[e][k], "softmax"), "softmax")
        for k in range(4):
            acc = 0.0
            for e in range(ne):
                acc = C.add(acc, wc[e][k], "collective")
            Cc[d][k] = C.div(acc, float(ne), "collective")
    P = [[0.0] * 8 for _ in range(4)]
    for d in range(4):
        if rel4[d] < 0:
            continue
        a, b, s, c = A[d], B[d], S[d], Cc[d]
        raw = [_dot4(C, a, a, "invariants"), _dot4(C, b, b, "invariants"), _dot4(C, s, s, "invariants"),
               _dot4(C, c, c, "invariants"), _dot4(C, a, s, "invariants"), _dot4(C, a, c, "invariants"),
               _dot4(C, b, s, "invariants"), _dot4(C, b, c, "invariants"), _dot4(C, s, c, "invariants"), 0.0]
        f = [C.div(v, C.add(1.0, abs(v), "squash"), "squash") for v in raw]
        kk = []
        for q in range(8):
            z = 0.0
            for j in range(10):
                z = C.add(z, C.mul(W[q][j], f[j], "gates"), "gates")
            z = C.add(z, bias[q], "gates")
            if q in (0, 4):
                kk.append(C.div(1.0, C.add(1.0, C.exp(-z, "activation"), "activation"), "activation"))
            else:
                kk.append(C.tanh(z, "activation"))
        ua, ub = [], []
        for k in range(4):
            u = "update"
            ua.append(C.add(C.add(C.add(C.mul(kk[0], a[k], u), C.mul(kk[1], s[k], u), u), C.mul(kk[2], c[k], u), u),
                            C.mul(kk[3], b[k], u), u))
            ub.append(C.add(C.add(C.add(C.mul(kk[4], b[k], u), C.mul(kk[5], s[k], u), u), C.mul(kk[6], c[k], u), u),
                            C.mul(kk[7], a[k], u), u))
        for v in (ua, ub):
            n = C.sqrt(_dot4(C, v, v, "clamp"), "clamp")
            scale = C.div(4.0, C.add(n, 1e-30, "clamp"), "clamp")
            if scale > 1.0:
                scale = 1.0
            for k in range(4):
                v[k] = C.mul(v[k], scale, "clamp")
        P[d] = ua + ub
    if gains is not None:
        alpha, beta = gains
        s32 = [A[d][k] for d in range(4) for k in range(4)] + [B[d][k] for d in range(4) for k in range(4)]
        p32 = [P[d][k] for d in range(4) for k in range(4)] + [P[d][4 + k] for d in range(4) for k in range(4)]
        dd = [C.sub(p32[i], s32[i], "ctrl_diff") for i in range(32)]
        norm = _fsum(C, [C.mul(s32[i], s32[i], "ctrl_norm") for i in range(32)], "ctrl_norm")
        dt = _fsum(C, [C.mul(s32[i], dd[i], "ctrl_dot") for i in range(32)], "ctrl_dot")
        den = 1e-24 if 1e-24 > norm else norm
        for i in range(32):
            r = C.div(C.mul(s32[i], dt, "ctrl_radial"), den, "ctrl_radial")
            o = "ctrl_out"
            p32[i] = C.add(C.add(p32[i], C.mul(alpha, r, o), o), C.mul(beta, C.sub(dd[i], r, o), o), o)
        for d in range(4):
            P[d] = p32[d * 4:(d + 1) * 4] + p32[16 + d * 4:16 + (d + 1) * 4]
    return {d: P[d] for d in range(4) if rel4[d] >= 0}


def census_step(reg, slot_rel, slot_row, pool, W, bias, gains):
    """One synchronous pass over the given instances; returns (new rows dict, Census)."""
    C = Census()
    rows = sorted({int(r) for r in np.asarray(slot_row).reshape(-1) if r >= 0})
    cur = {r: pool[r].tolist() for r in rows}
    cands = {}
    for rel4, row4 in zip(np.asarray(slot_rel).tolist(), np.asarray(slot_row).tolist()):
        out = instance_candidates(C, reg, rel4, row4, cur, W, bias, gains)
        for d, v in out.items():
            cands.setdefault(row4[d], []).append(v)
    new = {}
    for r, vecs in cands.items():
        m = len(vecs)
        if m == 1:
            new[r] = list(vecs[0])
            continue
        res = []
        for k in range(8):
            vals = sorted(v[k] for v in vecs)
            acc = vals[0]
            for t in vals[1:]:
                acc = C.add(acc, t, "reconcile")
            res.append(C.div(acc, float(m), "reconcile"))
        new[r] = res
    return new, C


# ---------------------------------------------------------------------------------
def trajectory_census(checkpoints=(1000, 1700, 1750, 1800, 2000, 2300, 2400, 2500, 2600, 2700, 2800, 3000,
                                   3500, 4000), timing_reps=20, max_instances=None):
    """Regenerate the R118 failure trajectory (SMALL, R104 on 54007/STATE8, all instances,
    exact path) and census every checkpoint; verify the replica bit-for-bit against the
    native step from the same snapshot, and time that native step on the same states."""
    import time

    from research.yang_mills_hca.r118_scalable_awr import bench as B
    from research.yang_mills_hca.r118_scalable_awr import contract as CT
    from research.yang_mills_hca.r118_scalable_awr import engine as E

    proto, _ = B.protocol()
    _, st, _, _ = B.build(proto["registries"]["SMALL"])
    ck, ctrl = "54007/STATE8", "R104"
    eng = E.Engine(st, ck, controller=ctrl, budgets=B.HOT)
    W, bias = eng.W.tolist(), eng.b.tolist()
    gains = CT.controller_gains(ctrl, ck)
    allh = np.nonzero(st.inst_live)[0]
    sel = allh if max_instances is None else allh[:max_instances]
    rows = np.nonzero(st.row_ref > 0)[0]
    out = []
    step = 0
    for target in checkpoints:
        while step < target:
            assert eng.step(allh).exact
            step += 1
        snap = st.pool.copy()
        new, C = census_step(st.registry, st.slot_rel[sel], st.slot_row[sel], snap, W, bias, gains)
        st.pool[:] = snap
        assert eng.step(sel).exact
        native = {r: st.pool[r].tolist() for r in new}
        mismatched = sum(bits(a) != bits(b) for r in new for a, b in zip(new[r], native[r]))
        lat, lat_all = [], []
        for _ in range(timing_reps):
            st.pool[:] = snap
            t0 = time.perf_counter_ns()
            eng.step(sel)
            lat.append((time.perf_counter_ns() - t0) / 1e6)
            st.pool[:] = snap
            t0 = time.perf_counter_ns()
            eng.step(allh)
            lat_all.append((time.perf_counter_ns() - t0) / 1e6)
        st.pool[:] = snap
        v = np.abs(snap[rows])
        nz = v[v > 0]
        out.append({
            "step": target, "instances": int(sel.size),
            "state_abs_max": float(v.max()), "state_abs_median_nonzero": float(np.median(nz)) if nz.size else 0.0,
            "state_log10_p05_p95": [float(np.log10(np.percentile(nz, 5))), float(np.log10(np.percentile(nz, 95)))]
            if nz.size else None,
            "state_subnormal_coords": int(np.sum((v > 0) & (v < TINY))), "state_zero_coords": int(np.sum(v == 0)),
            "replica_vs_native_bit_mismatches": int(mismatched), "replica_coords": int(8 * len(new)),
            "native_step_ms_median_census_subset": float(np.median(lat)),
            "native_step_ms_median_all_instances": float(np.median(lat_all)),
            "native_step_ms_p95_all_instances": float(np.percentile(lat_all, 95)), "all_instances": int(allh.size),
            "census": C.as_dict(),
        })
    return out
