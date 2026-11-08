"""Explicit-loop scalar reference of the v2 reliability-mixture learner for ONE episode (Python ``math`` only).

It is the line-by-line template of the native C11 component (``v2/native/src/cgc_lao_rmf.c``): every sum is an
explicit left-to-right loop starting from 0.0, every transcendental is the C library's (``math.exp``,
``math.log``, ``math.sqrt``, ``math.pow``), and branches mirror the C code. On one machine the two are
therefore bitwise identical. The vectorised filter (numpy pairwise sums, numpy's own exp) agrees within
floating-point tolerance. Hypotheses: [(q or [q nodes], rho), ...]; prior weights; PROBE_EARLY (0) or
greedy-with-deadline (1) acquisition; posterior expected-utility terminal decision with STOP.
"""
from __future__ import annotations

import math

from cgc_lao_lab.filters import LW_FLOOR, NODES, V_FLOOR, WEIGHTS

GH_X = tuple(float(x) for x in NODES)
GH_W = tuple(float(w) for w in WEIGHTS)
K = 3
STOP = 3
STATE_CAP = 1.5
MAX_HYP, MAX_NODES, MAX_READS = 8, 4, 2


def sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class ScalarRMF2:
    def __init__(self, hypotheses, prior, bias_var=1.0, stop_payoff=0.6, horizon=8, acquisition=0):
        if not (1 <= len(hypotheses) <= MAX_HYP) or len(prior) != len(hypotheses):
            raise ValueError("SCALAR2_CONFIG")
        self.q, self.nw, self.rho = [], [], []
        for q, rho in hypotheses:
            nodes = [float(q)] if isinstance(q, (int, float)) else [float(v) for v in q]
            if not (1 <= len(nodes) <= MAX_NODES) or not all(0.0 <= v < 0.5 for v in nodes) or not 0.0 <= rho <= 1.0:
                raise ValueError("SCALAR2_CONFIG")
            self.q.append(nodes)
            self.nw.append(1.0 / len(nodes))
            self.rho.append(float(rho))
        total = 0.0
        for p in prior:
            if not (p > 0.0 and math.isfinite(p)):
                raise ValueError("SCALAR2_CONFIG")
            total += float(p)
        self.log_prior = [math.log(float(p) / total) for p in prior]
        self.M = len(hypotheses)
        self.bias_var = float(bias_var)
        self.stop_payoff = float(stop_payoff)
        self.horizon = int(horizon)
        self.acquisition = int(acquisition)
        self.reset()

    def reset(self) -> None:
        self.lw = list(self.log_prior)
        self.mu = [[0.0] * K for _ in range(self.M)]
        self.v = [[self.bias_var] * K for _ in range(self.M)]
        self.counts = [0] * K
        self.t = 0

    def state_scalars(self) -> int:
        return self.M * (1 + 2 * K) + K

    def weights(self):
        top = self.lw[0]
        for m in range(1, self.M):
            if self.lw[m] > top:
                top = self.lw[m]
        w = [math.exp(self.lw[m] - top) for m in range(self.M)]
        s = 0.0
        for m in range(self.M):
            s += w[m]
        return [w[m] / s for m in range(self.M)]

    def _success(self, m: int, a: int, z: float) -> float:
        sd = math.sqrt(self.v[m][a])
        acc = 0.0
        for g in range(16):
            acc += GH_W[g] * sigmoid(z + (self.mu[m][a] + sd * GH_X[g]))
        return acc

    def predictive(self, z_row):
        w = self.weights()
        out = []
        for a in range(K):
            acc = 0.0
            for m in range(self.M):
                acc += self._success(m, a, z_row[a]) * w[m]
            out.append(acc)
        return out

    @staticmethod
    def _first_argmax(values, mask=None):
        best, best_v = -1, -math.inf
        for a in range(K):
            v = values[a] if (mask is None or mask[a]) else -1e9
            if best < 0 or v > best_v:
                best, best_v = a, v
        return best

    def select(self, z_row) -> int:
        if self.t >= self.horizon:
            raise ValueError("SCALAR2_HORIZON")
        pred = self.predictive(z_row)
        unseen = [self.counts[a] == 0 for a in range(K)]
        need = 0
        for a in range(K):
            need += 1 if unseen[a] else 0
        best = self._first_argmax(pred)
        unseen_best = self._first_argmax(pred, unseen)
        force = need >= self.horizon - self.t
        probe = self.acquisition == 0 and self.t in (1, 2) and need > 0
        return unseen_best if (force or probe) else best

    def _evidence(self, m: int, reads):
        """(A1, A0): probability of the readings given clean outcome 1 / 0 under hypothesis m."""
        out = []
        for c in (1, 0):
            acc = 0.0
            for q in self.q[m]:
                if len(reads) == 1:
                    f = float(reads[0] ^ c)
                    p = math.pow(q, f) * math.pow(1.0 - q, 1.0 - f)
                else:
                    f0, f1 = float(reads[0] ^ c), float(reads[1] ^ c)
                    s = f0 + f1
                    indep = math.pow(q, s) * math.pow(1.0 - q, 2.0 - s)
                    shared = 0.0 if s == 1.0 else math.pow(q, f0) * math.pow(1.0 - q, 1.0 - f0)
                    p = (1.0 - self.rho[m]) * indep + self.rho[m] * shared
                acc += p * self.nw[m]
            out.append(acc)
        return out[0], out[1]

    def update(self, a: int, z: float, reads, advance: bool = True) -> None:
        if not (0 <= a < K) or not (1 <= len(reads) <= MAX_READS) or any(r not in (0, 1) for r in reads):
            raise ValueError("SCALAR2_INPUT")
        if advance and self.t >= self.horizon:
            raise ValueError("SCALAR2_HORIZON")
        new_lw = [0.0] * self.M
        for m in range(self.M):
            a1, a0 = self._evidence(m, reads)
            mu, sd = self.mu[m][a], math.sqrt(self.v[m][a])
            L = m1 = m2 = 0.0
            for g in range(16):
                b = mu + sd * GH_X[g]
                p = sigmoid(z + b)
                lik = (p * a1 + (1.0 - p) * a0) * GH_W[g]
                L += lik
                m1 += lik * b
                m2 += lik * b * b
            if L > 1e-300:
                m1 = m1 / L
                m2 = m2 / L
                var = m2 - m1 * m1
                self.mu[m][a] = m1
                self.v[m][a] = var if var > V_FLOOR else V_FLOOR
                new_lw[m] = self.lw[m] + math.log(L)
            else:
                new_lw[m] = self.lw[m] + LW_FLOOR
        top = new_lw[0]
        for m in range(1, self.M):
            if new_lw[m] > top:
                top = new_lw[m]
        for m in range(self.M):
            x = new_lw[m] - top
            self.lw[m] = x if x > LW_FLOOR else LW_FLOOR
        if advance:
            self.counts[a] += 1
            self.t += 1

    def state_correction(self):
        w = self.weights()
        out = []
        for a in range(K):
            acc = 0.0
            for m in range(self.M):
                acc += w[m] * self.mu[m][a]
            out.append(STATE_CAP if acc > STATE_CAP else (-STATE_CAP if acc < -STATE_CAP else acc))
        return out

    def decide(self, z_row) -> int:
        pred = self.predictive(z_row)
        best = self._first_argmax(pred)
        return STOP if pred[best] < self.stop_payoff else best
