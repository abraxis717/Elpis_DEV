"""Explicit-loop scalar reference of the relational learner for ONE episode (Python ``math`` only): the template of
a native port, and the learner the research host drives. Same arithmetic order as learners.RelationalFilter
(left-to-right sums, scalar2's evidence and moment matching), so the two agree within floating-point tolerance and
the scalar reference is bitwise reproducible on one machine.

State (fixed capacity): mu, v, lam [M][R_MAX], counts [R_MAX], and a provenance-indexed event buffer of at most
T = 8 events (12 readings). Invalidation: replay of retained events (strategy B) or slot reset (strategy A).
"""
from __future__ import annotations

import math

from cgc_lao_lab.filters import LW_FLOOR, NODES, V_FLOOR, WEIGHTS

GH_X = tuple(float(x) for x in NODES)
GH_W = tuple(float(w) for w in WEIGHTS)
R_MAX, M_MAX, BUFFER = 8, 4, 8


def sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class ScalarRelational:
    def __init__(self, hypotheses=((0.0, 0.0), (0.4, 0.0), (0.48, 0.0)), prior=(1.0, 1.0, 1.0), slots: int = R_MAX,
                 bias_var: float = 1.0):
        if not (1 <= len(hypotheses) <= M_MAX) or len(prior) != len(hypotheses) or not (1 <= slots <= R_MAX):
            raise ValueError("SCALAR_REL_CONFIG")
        for q, rho in hypotheses:
            if not (0.0 <= q < 0.5 and 0.0 <= rho <= 1.0):
                raise ValueError("SCALAR_REL_CONFIG")
        total = 0.0
        for p in prior:
            if not (p > 0.0 and math.isfinite(p)):
                raise ValueError("SCALAR_REL_CONFIG")
            total += float(p)
        self.q = [float(h[0]) for h in hypotheses]
        self.rho = [float(h[1]) for h in hypotheses]
        self.log_prior = [math.log(float(p) / total) for p in prior]
        self.M, self.R, self.bias_var = len(hypotheses), int(slots), float(bias_var)
        self.reset()

    # -- lifecycle ------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self._reset_state()
        self.buffer: list = []             # retained events: (step, slot, z, reads, binding)

    def _reset_state(self) -> None:
        self.mu = [[0.0] * self.R for _ in range(self.M)]
        self.v = [[self.bias_var] * self.R for _ in range(self.M)]
        self.lam = [[0.0] * self.R for _ in range(self.M)]
        self.counts = [0] * self.R

    def zeroize(self) -> None:
        for rows in (self.mu, self.v, self.lam):
            for row in rows:
                row[:] = [0.0] * len(row)
        self.counts[:] = [0] * len(self.counts)
        self.buffer = []

    def state_scalars(self) -> int:
        return 3 * self.M * self.R + self.R

    def state(self) -> tuple:
        return (tuple(tuple(r) for r in self.mu), tuple(tuple(r) for r in self.v),
                tuple(tuple(r) for r in self.lam), tuple(self.counts))

    # -- the RMF3 law on one slot ---------------------------------------------------------------------------------
    def _evidence(self, m: int, reads):
        q = self.q[m]
        out = []
        for c in (1, 0):
            if len(reads) == 1:
                f = float(reads[0] ^ c)
                p = math.pow(q, f) * math.pow(1.0 - q, 1.0 - f)
            else:
                f0, f1 = float(reads[0] ^ c), float(reads[1] ^ c)
                s = f0 + f1
                indep = math.pow(q, s) * math.pow(1.0 - q, 2.0 - s)
                shared = 0.0 if s == 1.0 else math.pow(q, f0) * math.pow(1.0 - q, 1.0 - f0)
                p = (1.0 - self.rho[m]) * indep + self.rho[m] * shared
            out.append(p)
        return out[0], out[1]

    def update(self, r: int, z: float, reads, step: int | None = None, binding=None, record: bool = True) -> None:
        if not (0 <= r < self.R) or not (1 <= len(reads) <= 2) or any(y not in (0, 1) for y in reads):
            raise ValueError("SCALAR_REL_INPUT")
        if record and len(self.buffer) >= BUFFER:
            raise OverflowError("SCALAR_REL_BUFFER")
        for m in range(self.M):
            a1, a0 = self._evidence(m, reads)
            mu, sd = self.mu[m][r], math.sqrt(self.v[m][r])
            L = m1 = m2 = 0.0
            for g in range(16):
                b = mu + sd * GH_X[g]
                p = sigmoid(z + b)
                lik = (p * a1 + (1.0 - p) * a0) * GH_W[g]
                L += lik
                m1 += lik * b
                m2 += lik * b * b
            if L > 1e-300:
                m1, m2 = m1 / L, m2 / L
                var = m2 - m1 * m1
                self.mu[m][r] = m1
                self.v[m][r] = var if var > V_FLOOR else V_FLOOR
                self.lam[m][r] += math.log(L)
            else:
                self.lam[m][r] += LW_FLOOR
        self.counts[r] += 1
        if record:
            self.buffer.append((step, r, float(z), tuple(int(y) for y in reads), binding))

    # -- reads ----------------------------------------------------------------------------------------------------
    def log_weights(self):
        out = []
        for m in range(self.M):
            s = sorted(self.lam[m])
            acc = s[0]
            for x in s[1:]:
                acc += x
            out.append(self.log_prior[m] + acc)
        return out

    def weights(self):
        lw = self.log_weights()
        top = max(lw)
        w = [math.exp(x - top) for x in lw]
        s = 0.0
        for x in w:
            s += x
        return [x / s for x in w]

    def mean(self, m: int, r: int, z: float) -> float:
        sd = math.sqrt(self.v[m][r])
        acc = 0.0
        for g in range(16):
            acc += GH_W[g] * sigmoid(z + (self.mu[m][r] + sd * GH_X[g]))
        return acc

    def predictive(self, z_row) -> list:
        w = self.weights()
        out = []
        for r in range(self.R):
            acc = 0.0
            for m in range(self.M):
                acc += w[m] * self.mean(m, r, z_row[r])
            out.append(acc)
        return out

    def expected_utility(self, alternative, z_row) -> float:
        """sum_h w_h (1 - prod_paths (1 - prod_relations E_h[p])) for an alternative = (path, ...)."""
        w = self.weights()
        total = 0.0
        for m in range(self.M):
            miss = 1.0
            for path in alternative:
                prod = 1.0
                for r in path:
                    prod *= self.mean(m, r, z_row[r])
                miss *= 1.0 - prod
            total += w[m] * (1.0 - miss)
        return total

    def decide(self, alternatives, stop: float, z_row) -> int:
        best, best_v = -1, -math.inf
        for a, alt in enumerate(alternatives):
            v = self.expected_utility(alt, z_row)
            if best < 0 or v > best_v:
                best, best_v = a, v
        return -1 if best_v < stop else best

    # -- invalidation ---------------------------------------------------------------------------------------------
    def rebuild(self, apply) -> None:
        """Strategy B: the state of a fresh learner fed exactly the retained events for which apply(event) holds,
        in their original order (the same arithmetic, so bitwise equal to that learner)."""
        self._reset_state()
        for step, r, z, reads, binding in self.buffer:
            if apply((step, r, z, reads, binding)):
                self.update(r, z, reads, step, binding, record=False)

    def drop(self, predicate) -> None:
        self.buffer = [e for e in self.buffer if not predicate(e)]

    def reset_slot(self, r: int) -> None:
        """Strategy A: return slot r to its prior and remove its evidence term, without any replay (exact: slots are
        independent under every hypothesis, so sum_r lam[h][r] loses exactly slot r's marginal likelihood). Its
        buffered events are discarded for good: the slot needs fresh evidence."""
        for m in range(self.M):
            self.mu[m][r], self.v[m][r], self.lam[m][r] = 0.0, self.bias_var, 0.0
        self.counts[r] = 0
        self.drop(lambda e: e[1] == r)
