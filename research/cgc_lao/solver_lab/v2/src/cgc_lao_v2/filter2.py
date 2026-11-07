"""Generalised reliability-mixture filter (v2): hypotheses over (corruption rate nodes, reading correlation).

A hypothesis ``m`` is a set of corruption-rate nodes ``q_mj`` with weights ``w_mj`` (one node: a point
hypothesis; several: a band averaged per event) and a reading correlation ``rho_m``. Given the selected
event's clean outcome ``c``, the readings' flips ``f_r = y_r XOR c`` have probability

    R = 1:  q^f (1-q)^(1-f)
    R = 2:  (1 - rho) q^(f0+f1) (1-q)^(2-f0-f1) + rho [f0 == f1] q^f0 (1-q)^(1-f0)

(with probability rho both readings share one flip; otherwise they flip independently). rho = 0 is v1's
``channel`` kind and rho = 1 its ``event`` kind; with those hypotheses this filter reproduces v1 bitwise.
Everything else is the v1 assumed-density filter: per (hypothesis, expert) Gaussian bias belief, 16-node
Gauss-Hermite moment matching, Bayes-rule mixture weights. State per episode: M log-weights + 2*M*K moments.
"""
from __future__ import annotations

import numpy as np

from cgc_lao_lab.filters import LW_FLOOR, NODES, V_FLOOR, WEIGHTS, _entropy
from cgc_lao_lab.env import K, sigmoid

MAX_NODES = 4


def normalise_hypotheses(hypotheses):
    """[(q or [q nodes], rho), ...] -> (nodes [M, J], node weights [M, J], rho [M]); equal node weights."""
    nodes, rhos = [], []
    for q, rho in hypotheses:
        qs = [float(q)] if np.isscalar(q) else [float(v) for v in q]
        if not (1 <= len(qs) <= MAX_NODES) or not all(0.0 <= v < 0.5 for v in qs) or not (0.0 <= rho <= 1.0):
            raise ValueError("RMF2_HYPOTHESIS_RANGE")
        nodes.append(qs)
        rhos.append(float(rho))
    J = max(len(n) for n in nodes)
    qn = np.zeros((len(nodes), J))
    wn = np.zeros((len(nodes), J))
    for m, n in enumerate(nodes):
        qn[m, :len(n)] = n
        qn[m, len(n):] = n[-1]
        wn[m, :len(n)] = 1.0 / len(n)
    return qn, wn, np.array(rhos)


class ReliabilityMixtureFilter2:
    def __init__(self, hypotheses, prior, bias_var: float = 1.0):
        self.qn, self.wn, self.rho = normalise_hypotheses(hypotheses)
        prior = np.asarray(prior, dtype=np.float64)
        if prior.shape != (len(self.rho),) or not (prior > 0).all() or bias_var <= 0:
            raise ValueError("RMF2_PRIOR")
        self.prior = prior / prior.sum()
        self.M = len(self.rho)
        self.bias_var = float(bias_var)
        self.qbar = (self.qn * self.wn).sum(1)               # mean corruption rate per hypothesis
        self.qs = self.qbar                                   # v1-compatible attribute (diagnostics)
        for arr in (self.qn, self.wn, self.rho, self.prior, self.qbar):
            arr.setflags(write=False)

    def reset(self, E: int, q_override: np.ndarray | None = None) -> None:
        self.E = E
        self.lw = np.tile(np.log(self.prior), (E, 1))
        self.mu = np.zeros((E, self.M, K))
        self.v = np.full((E, self.M, K), self.bias_var)
        if q_override is None:
            self.q_nodes = np.broadcast_to(self.qn, (E,) + self.qn.shape)
        else:   # privileged oracle only: per-episode point hypotheses
            self.q_nodes = np.asarray(q_override, float).reshape(E, self.M, 1) * np.ones((1, 1, self.qn.shape[1]))

    def state_scalars(self) -> int:
        return self.M * (1 + 2 * K)

    def volatile_state(self) -> dict:
        return {"lw": self.lw, "mu": self.mu, "v": self.v}

    def _evidence(self, reads: np.ndarray):
        """A1, A0 [E, M, 1]: probability of the readings given clean outcome 1 / 0, per hypothesis."""
        R = reads.shape[1]
        q = self.q_nodes[:, :, :, None]                       # [E, M, J, 1]
        w = self.wn[None, :, :]                               # [1, M, J]
        out = []
        for c in (1, 0):
            f = (reads ^ c).astype(np.float64)                # flips implied by clean outcome c, [E, R]
            if R == 1:
                f0 = f[:, None, None, 0:1]
                p = q ** f0 * (1.0 - q) ** (1.0 - f0)
            else:
                s = (f[:, 0] + f[:, 1])[:, None, None, None]
                indep = q ** s * (1.0 - q) ** (2.0 - s)
                f0 = f[:, None, None, 0:1]
                shared = np.where(s == 1.0, 0.0, q ** f0 * (1.0 - q) ** (1.0 - f0))
                rho = self.rho[None, :, None, None]
                p = (1.0 - rho) * indep + rho * shared
            out.append((p[..., 0] * w).sum(-1)[:, :, None])   # band average -> [E, M, 1]
        return out[0], out[1]

    def update(self, action: np.ndarray, z: np.ndarray, reads: np.ndarray) -> None:
        E = self.E
        idx = np.arange(E)
        reads = np.asarray(reads)
        if reads.shape[0] != E or reads.ndim != 2 or not (1 <= reads.shape[1] <= 2) or not np.isin(reads, (0, 1)).all():
            raise ValueError("RMF2_READS_CONTRACT")
        a1, a0 = self._evidence(reads.astype(np.uint8))
        mu = self.mu[idx, :, action]
        v = self.v[idx, :, action]
        b = mu[..., None] + np.sqrt(v)[..., None] * NODES
        p = sigmoid(z[:, None, None] + b)
        lik = (p * a1 + (1.0 - p) * a0) * WEIGHTS
        L = lik.sum(-1)
        ok = L > 1e-300
        Ls = np.where(ok, L, 1.0)
        m1 = (lik * b).sum(-1) / Ls
        m2 = (lik * b * b).sum(-1) / Ls
        self.mu[idx, :, action] = np.where(ok, m1, mu)
        self.v[idx, :, action] = np.where(ok, np.maximum(m2 - m1 * m1, V_FLOOR), v)
        lw = self.lw + np.where(ok, np.log(Ls), LW_FLOOR)
        self.lw = np.maximum(lw - lw.max(1, keepdims=True), LW_FLOOR)

    def weights(self) -> np.ndarray:
        w = np.exp(self.lw - self.lw.max(1, keepdims=True))
        return w / w.sum(1, keepdims=True)

    def component_moments(self, z: np.ndarray):
        b = self.mu[:, None, :, :, None] + np.sqrt(self.v)[:, None, :, :, None] * NODES
        p = sigmoid(z[:, :, None, :, None] + b)
        return (p * WEIGHTS).sum(-1), (p * p * WEIGHTS).sum(-1)

    def predictive(self, z: np.ndarray) -> np.ndarray:
        shape = z.shape
        zz = z.reshape(shape[0], -1, K)
        mean, _ = self.component_moments(zz)
        return (mean * self.weights()[:, None, :, None]).sum(2).reshape(shape)

    def point_bias(self) -> np.ndarray:
        return (self.weights()[:, :, None] * self.mu).sum(1)

    def reliability_information(self, z: np.ndarray) -> np.ndarray:
        mean, _ = self.component_moments(z[:, None, :])
        q = (self.q_nodes * self.wn).sum(-1)[:, :, None]
        py = q + (1.0 - 2.0 * q) * mean[:, 0]
        w = self.weights()[:, :, None]
        return _entropy((w * py).sum(1)) - (w * _entropy(py)).sum(1)

    def gain_ratio(self) -> np.ndarray:
        r = ((1.0 - 2.0 * self.q_nodes) ** 2 * self.wn).sum(-1)
        return (self.weights() * r).sum(1)
