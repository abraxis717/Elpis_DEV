"""Dependency-free scalar reliability-mixture filter for ONE episode (Python ``math`` only).

The same equations as ``filters.ReliabilityMixtureFilter`` and ``agents.BayesAgent`` (PROBE_EARLY or
greedy-with-deadline acquisition, Bayes terminal decision), written as fixed loops over fixed-size lists:
M hypotheses x 3 experts x 16 quadrature nodes. It is the template for a native port (no allocation
after construction, no recursion, a fixed number of operations per call) and an independent parity
check of the vectorised implementation.
"""
from __future__ import annotations

import math

from .filters import LW_FLOOR, NODES, V_FLOOR, WEIGHTS

_NODES = tuple(float(x) for x in NODES)
_WEIGHTS = tuple(float(x) for x in WEIGHTS)
K = 3
STOP = 3


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class ScalarRMF:
    def __init__(self, qs, prior, bias_var=1.0, kinds=None, stop_payoff=0.6, horizon=8, acquisition="probe"):
        self.qs = tuple(float(q) for q in qs)
        total = float(sum(prior))
        self.log_prior = tuple(math.log(p / total) for p in prior)
        self.event = tuple(k == "event" for k in (kinds or ("channel",) * len(self.qs)))
        self.bias_var = float(bias_var)
        self.stop_payoff = float(stop_payoff)
        self.horizon = int(horizon)
        self.acquisition = acquisition
        self.M = len(self.qs)
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
        top = max(self.lw)
        w = [math.exp(x - top) for x in self.lw]
        s = sum(w)
        return [x / s for x in w]

    def _expected_success(self, m: int, a: int, z: float) -> float:
        sd = math.sqrt(self.v[m][a])
        return sum(w * _sigmoid(z + self.mu[m][a] + sd * x) for x, w in zip(_NODES, _WEIGHTS))

    def predictive(self, z_row):
        w = self.weights()
        return [sum(w[m] * self._expected_success(m, a, z_row[a]) for m in range(self.M)) for a in range(K)]

    def select(self, z_row) -> int:
        pred = self.predictive(z_row)
        unseen = [c == 0 for c in self.counts]
        need = sum(unseen)
        best = max(range(K), key=lambda a: (pred[a], -a))
        unseen_best = max(range(K), key=lambda a: (pred[a] if unseen[a] else -1e9, -a))
        force = need >= self.horizon - self.t
        probe = self.acquisition == "probe" and self.t in (1, 2) and need > 0
        return unseen_best if (force or probe) else best

    def update(self, a: int, z: float, reads) -> None:
        n1 = sum(reads)
        n0 = len(reads) - n1
        agree = n1 == 0 or n0 == 0
        new_lw = []
        for m, q in enumerate(self.qs):
            if self.event[m]:
                a1 = (1.0 - q if n1 else q) if agree else 0.0
                a0 = (q if n1 else 1.0 - q) if agree else 0.0
            else:
                a1 = (1.0 - q) ** n1 * q ** n0
                a0 = q ** n1 * (1.0 - q) ** n0
            mu, sd = self.mu[m][a], math.sqrt(self.v[m][a])
            L = m1 = m2 = 0.0
            for x, w in zip(_NODES, _WEIGHTS):
                b = mu + sd * x
                p = _sigmoid(z + b)
                lik = (p * a1 + (1.0 - p) * a0) * w
                L += lik
                m1 += lik * b
                m2 += lik * b * b
            if L > 1e-300:
                m1 /= L
                m2 /= L
                self.mu[m][a] = m1
                self.v[m][a] = max(m2 - m1 * m1, V_FLOOR)
                new_lw.append(self.lw[m] + math.log(L))
            else:
                new_lw.append(self.lw[m] + LW_FLOOR)
        top = max(new_lw)
        self.lw = [max(x - top, LW_FLOOR) for x in new_lw]
        self.counts[a] += 1
        self.t += 1

    def decide(self, z_row) -> int:
        pred = self.predictive(z_row)
        best = max(range(K), key=lambda a: (pred[a], -a))
        return STOP if pred[best] < self.stop_payoff else best
