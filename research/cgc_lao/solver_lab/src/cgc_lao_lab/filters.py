"""Bounded reliability-mixture filter (RMF): a finite mixture over feedback-reliability hypotheses,
each carrying a Gaussian belief over every expert's hidden bias, updated by assumed-density filtering.

Model (per hypothesis ``m``): ``b_a ~ N(0, bias_var)`` independently; the selected event is clean with
probability ``sigmoid(z + b_a)``; each reading of it is flipped with probability ``q_m``, independently
across readings (``kind='channel'``) or jointly for all readings (``kind='event'``).

Exact Bayes under this model factorises over experts given the hypothesis, so the posterior is the
mixture weights plus, per (hypothesis, expert), a one-dimensional bias posterior. ADF keeps that
posterior Gaussian by matching its first two moments after every observation (16-node Gauss-Hermite
quadrature); the mixture weights are updated with the same quadrature's predictive likelihood.
Volatile state per episode: ``M`` log-weights + ``2*M*K`` moments. Work per update: ``O(M*16)``.
"""
from __future__ import annotations

import numpy as np

from .env import K, sigmoid

# 16-node probabilists' Gauss-Hermite rule, normalised to the standard normal (fixed literals: the
# quadrature never depends on a platform eigen-solver). Exact for polynomials up to degree 31.
NODES = np.array((
    -6.630878198393129, -5.472225705949343, -4.492955302520011, -3.6008736241715487,
    -2.7602450476307014, -1.9519803457163334, -1.1638291005549648, -0.3867606045005574,
    0.3867606045005574, 1.1638291005549648, 1.9519803457163334, 2.7602450476307014,
    3.6008736241715487, 4.492955302520011, 5.472225705949343, 6.630878198393129))
WEIGHTS = np.array((
    1.4978147231618412e-10, 1.309473216286817e-07, 1.530003216248732e-05, 0.0005259849265739087,
    0.007266937601184749, 0.04728475235401406, 0.1583383727509497, 0.286568521238012,
    0.286568521238012, 0.1583383727509497, 0.04728475235401406, 0.007266937601184749,
    0.0005259849265739087, 1.530003216248732e-05, 1.309473216286817e-07, 1.4978147231618412e-10))
NODES.setflags(write=False)
WEIGHTS.setflags(write=False)

LW_FLOOR = -1.0e4     # log-weight of an eliminated hypothesis (exp underflows to exactly 0)
V_FLOOR = 1.0e-6      # smallest bias variance kept by moment matching
KINDS = ("channel", "event")


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


class ReliabilityMixtureFilter:
    """Fixed-size Bayesian state over (reliability hypothesis, expert biases) for a batch of episodes."""

    def __init__(self, qs, prior, bias_var: float = 1.0, kinds=None):
        qs = np.asarray(qs, dtype=np.float64)
        prior = np.asarray(prior, dtype=np.float64)
        if qs.ndim != 1 or prior.shape != qs.shape or len(qs) < 1:
            raise ValueError("RMF_HYPOTHESIS_SHAPE")
        if not ((qs >= 0) & (qs < 0.5)).all() or not (prior > 0).all() or bias_var <= 0:
            raise ValueError("RMF_HYPOTHESIS_RANGE")
        kinds = tuple(kinds) if kinds is not None else ("channel",) * len(qs)
        if len(kinds) != len(qs) or any(k not in KINDS for k in kinds):
            raise ValueError("RMF_KINDS")
        self.qs = qs
        self.prior = prior / prior.sum()
        self.kinds = kinds
        self.event = np.array([k == "event" for k in kinds])
        self.bias_var = float(bias_var)
        self.M = len(qs)
        self.E = 0

    # -- volatile state -------------------------------------------------------------------------------
    def reset(self, E: int, q_override: np.ndarray | None = None) -> None:
        """Fresh state for ``E`` episodes. ``q_override`` ([E, M]) is for privileged oracles only."""
        self.E = E
        self.lw = np.tile(np.log(self.prior), (E, 1))
        self.mu = np.zeros((E, self.M, K))
        self.v = np.full((E, self.M, K), self.bias_var)
        self.q = np.tile(self.qs, (E, 1)) if q_override is None else np.asarray(q_override, float).reshape(E, self.M)

    def state_scalars(self) -> int:
        return self.M * (1 + 2 * K)

    def volatile_state(self) -> dict:
        return {"lw": self.lw, "mu": self.mu, "v": self.v}

    # -- update ---------------------------------------------------------------------------------------
    def update(self, action: np.ndarray, z: np.ndarray, reads: np.ndarray) -> None:
        """Incorporate the readings ``reads[E, R]`` of expert ``action[E]`` at public logit ``z[E]``."""
        E = self.E
        idx = np.arange(E)
        reads = np.asarray(reads)
        if reads.shape[0] != E or reads.ndim != 2 or not np.isin(reads, (0, 1)).all():
            raise ValueError("RMF_READS_CONTRACT")
        R = reads.shape[1]
        n1 = reads.sum(-1).astype(np.float64)[:, None, None]
        n0 = R - n1
        q = self.q[:, :, None]
        a1 = (1.0 - q) ** n1 * q ** n0          # P(readings | clean event = 1), channel model
        a0 = q ** n1 * (1.0 - q) ** n0          # P(readings | clean event = 0)
        if self.event.any():
            agree = (n1 == 0) | (n1 == R)
            ones = n1 == R
            a1e = np.where(agree, np.where(ones, 1.0 - q, q), 0.0)
            a0e = np.where(agree, np.where(ones, q, 1.0 - q), 0.0)
            ev = self.event[None, :, None]
            a1 = np.where(ev, a1e, a1)
            a0 = np.where(ev, a0e, a0)
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

    # -- read-outs ------------------------------------------------------------------------------------
    def weights(self) -> np.ndarray:
        w = np.exp(self.lw - self.lw.max(1, keepdims=True))
        return w / w.sum(1, keepdims=True)

    def component_moments(self, z: np.ndarray):
        """Per hypothesis, E and E[.^2] of the clean success probability at logits ``z[E, N, K]``."""
        b = self.mu[:, None, :, :, None] + np.sqrt(self.v)[:, None, :, :, None] * NODES
        p = sigmoid(z[:, :, None, :, None] + b)                      # [E, N, M, K, G]
        return (p * WEIGHTS).sum(-1), (p * p * WEIGHTS).sum(-1)       # [E, N, M, K] each

    def predictive(self, z: np.ndarray) -> np.ndarray:
        """Posterior mean clean success probability at public logits ``z[E, ..., K]``."""
        shape = z.shape
        zz = z.reshape(shape[0], -1, K)
        mean, _ = self.component_moments(zz)
        return (mean * self.weights()[:, None, :, None]).sum(2).reshape(shape)

    def point_bias(self) -> np.ndarray:
        return (self.weights()[:, :, None] * self.mu).sum(1)

    def reliability_information(self, z: np.ndarray) -> np.ndarray:
        """Mutual information (nats) between one reading of each expert at logits ``z[E, K]`` and the
        hypothesis index, under the current posterior."""
        mean, _ = self.component_moments(z[:, None, :])
        pc = mean[:, 0]                                               # [E, M, K]
        q = self.q[:, :, None]
        py = q + (1.0 - 2.0 * q) * pc
        w = self.weights()[:, :, None]
        return _entropy((w * py).sum(1)) - (w * _entropy(py)).sum(1)

    def gain_ratio(self) -> np.ndarray:
        """Posterior mean of the Fisher-information ratio ``(1 - 2q)^2`` (1 = clean, 0 = pure noise)."""
        return (self.weights() * (1.0 - 2.0 * self.q) ** 2).sum(1)
