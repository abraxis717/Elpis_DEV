"""Kernel-level differential testing: frozen R118 kernel vs R119 kernel, bit for bit.

One synchronous pass of candidates (no reconciliation, no commit) is computed for the
same windows and compact states by

* R118 ``awr_candidates`` (the published exact path),
* R119 ``awr119_candidates`` with guard mask 0 (R119 build of the R118 arithmetic),
* R119 ``awr119_candidates`` with all guards,

and compared as uint64 bit patterns (so -0.0 != +0.0 and every ulp counts). States are
generated across magnitude regimes, with mixed scales inside an instance, exact zeros,
negative zeros, subnormals and values placed exactly at every guard threshold.
"""
from __future__ import annotations

import ctypes

import numpy as np

from research.yang_mills_hca.r118_scalable_awr import engine as E118
from research.yang_mills_hca.r118_scalable_awr import native as N118
from research.yang_mills_hca.r118_scalable_awr.contract import controller_gains

from . import guards
from . import native_r119 as N

TINY = 2.2250738585072014e-308
SCALES = (0.0, -5.0, -9.4, -9.6, -12.0, -18.0, -20.0, -22.0, -24.0, -26.0, -60.0, -100.0, -105.0, -108.0, -130.0, -149.0, -152.0, -155.0, -158.0, -161.0, -165.0,
          -200.0, -280.0, -300.0, -306.0, -307.5, -308.0, -310.0, -315.0, -320.0, -323.0)


def random_states(rng, nrows, mode):
    """[nrows, 8] float64 states for a regime ``mode``."""
    if mode == "uniform_scale":
        e = rng.choice(SCALES)
        return rng.normal(0.0, 1.0, (nrows, 8)) * 10.0 ** e
    if mode == "mixed_rows":
        e = rng.choice(SCALES, size=(nrows, 1))
        return rng.normal(0.0, 1.0, (nrows, 8)) * 10.0 ** e
    if mode == "mixed_coords":
        e = rng.uniform(-323.0, 0.5, (nrows, 8))
        return rng.normal(0.0, 1.0, (nrows, 8)) * 10.0 ** e
    if mode == "zeros_and_signs":
        x = rng.normal(0.0, 1.0, (nrows, 8)) * 10.0 ** rng.choice(SCALES, size=(nrows, 1))
        z = rng.random((nrows, 8))
        x[z < 0.2] = 0.0
        x[(z >= 0.2) & (z < 0.3)] = -0.0
        return x
    if mode == "subnormal":
        return rng.integers(-2000, 2000, (nrows, 8)).astype(np.float64) * 2.0 ** -1074
    raise ValueError(mode)


def threshold_states(rng, nrows, theta):
    """States whose slot max-abs sits on the guard thresholds (just below, at, just above)."""
    targets = [theta, 2.0 ** -60, 1.9375, 2.0 ** -60 / 16.0, 2.0, 4.0 / np.sqrt(8.0)]
    x = rng.normal(0.0, 1.0, (nrows, 8))
    for i in range(nrows):
        t = targets[i % len(targets)]
        t = (np.nextafter(t, 0.0), t, np.nextafter(t, np.inf))[(i // len(targets)) % 3]
        x[i] *= t / np.abs(x[i]).max()
    return x


class Differential:
    def __init__(self, registry, checkpoint, controller):
        self.reg = registry
        self.W, self.b, _ = E118.load_weights(checkpoint)
        gains = controller_gains(controller, checkpoint)
        self.correct = int(gains is not None)
        self.alpha, self.beta = gains if gains is not None else (0.0, 0.0)
        self.theta = guards.gate_theta(self.W.tolist(), self.b.tolist())
        self.k118 = N118.build().lib
        self.k119, _ = N.kernel()
        self.gc = np.zeros(8, dtype=np.int64)

    def _run(self, fn, n, slot_rel, slot_cmp, cin, extra):
        cand = np.zeros(n * 32, dtype=np.float64)
        links = np.zeros(1, dtype=np.int64)
        reg = self.reg
        rc = fn(n, slot_rel.ctypes.data, slot_cmp.ctypes.data, cin.ctypes.data, cand.ctypes.data,
                reg.rel_offsets.ctypes.data, reg.members.ctypes.data, reg.frames.ctypes.data, self.W.ctypes.data,
                self.b.ctypes.data, self.correct, self.alpha, self.beta, links.ctypes.data, *extra)
        return rc, cand

    def compare(self, slot_rel, slot_cmp, cin):
        """Returns (rc, mismatched_coordinates_r118_vs_guarded, mismatched_r118_vs_unguarded)."""
        slot_rel = np.ascontiguousarray(slot_rel, dtype=np.int32).reshape(-1)
        slot_cmp = np.ascontiguousarray(slot_cmp, dtype=np.int32).reshape(-1)
        cin = np.ascontiguousarray(cin, dtype=np.float64)
        n = slot_rel.size // 4
        rc0, ref = self._run(self.k118.awr_candidates, n, slot_rel, slot_cmp, cin, ())
        rc1, unguarded = self._run(self.k119.awr119_candidates, n, slot_rel, slot_cmp, cin,
                                   (ctypes.c_uint32(0), self.theta, self.gc.ctypes.data))
        rc2, guarded = self._run(self.k119.awr119_candidates, n, slot_rel, slot_cmp, cin,
                                 (ctypes.c_uint32(15), self.theta, self.gc.ctypes.data))
        if not (rc0 == rc1 == rc2):
            return ("RC_MISMATCH", rc0, rc1, rc2), -1, -1
        mb = int(np.sum(ref.view(np.uint64) != guarded.view(np.uint64)))
        mu = int(np.sum(ref.view(np.uint64) != unguarded.view(np.uint64)))
        return rc0, mb, mu


def windows_for(registry, n, rng):
    """n windows of distinct relations (1..4 bound slots) and a compact row per bound slot."""
    from research.yang_mills_hca.r118_scalable_awr.fixtures import extract_window

    slot_rel = np.full((n, 4), -1, dtype=np.int32)
    slot_cmp = np.full((n, 4), -1, dtype=np.int32)
    row_of = {}
    for i in range(n):
        w = extract_window(registry, int(rng.integers(registry.num_relations)), max_incidences=1 << 16)
        w = w[:int(rng.integers(1, len(w) + 1))] if rng.random() < 0.15 else w
        for s, r in zip(rng.permutation(4)[:len(w)], w):
            slot_rel[i, s] = r
            if r not in row_of:
                row_of[r] = len(row_of)
            slot_cmp[i, s] = row_of[r]
    return slot_rel, slot_cmp, len(row_of)
