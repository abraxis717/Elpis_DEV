"""Nonreciprocal associative network of Aguilera & De Martino, arXiv:2609.07341. RESEARCH_ONLY.

What the paper states (cited by equation or section):

* ``N`` binary spins updated in discrete time. With probability ``Delta``
  a spin is refreshed by Glauber dynamics at inverse temperature ``beta``,
  otherwise it is held (Eqs. 1, 3).
* Couplings ``J_ij = (1/N) sum_v xi_{i,v}^T A_v xi_{j,v}`` with ``J_ii = 0``
  (Eq. 2); block ``v = 0`` is the target attractor and the other blocks are
  quenched disorder. The load is ``alpha = P / N`` (text after Eq. 4).
* At ``alpha = 0`` the overlaps follow
  ``m_{t+1} = (1 - Delta) m_t + Delta 2^{-M} sum_sigma sigma tanh(beta sigma^T A m_t)``
  (End Matter, "Phase diagram of two-pattern cycles"). For ``A = Omega_phi``
  the Hopf onset is
  ``beta_c(phi) = Delta^{-1} (-(1-Delta) cos phi + sqrt((1-Delta)^2 cos^2 phi + Delta (2 - Delta)))``,
  with oscillation frequency
  ``omega = arctan(Delta beta sin phi / ((1 - Delta) + Delta beta cos phi))`` (same paragraph).
* Uniformly distributed eigenphases of the disorder blocks cancel the
  retarded kernel (``K = 0``, Eq. 23) and raise retrieval capacity relative to
  coherently aligned eigenphases (Fig. 1; "Encoding cycles with isotropic
  eigenvalues"; Discussion). The paper's microscopic checks use
  ``N = 50,000`` to ``500,000`` and ``Delta = 0.1`` (End Matter, "Numerical setup").

The laboratory's own derivation of the Hopf line: linearizing the ``alpha = 0``
map at ``m = 0`` gives ``(1 - Delta) I + Delta beta A``, because
``2^{-M} sum_sigma sigma sigma^T = I``. For ``A = Omega_phi`` the eigenvalues are
``(1 - Delta) + Delta beta e^{+-i phi}``, whose modulus is 1 exactly at
``beta_c(phi)`` and whose argument is ``omega``.
:func:`linearization_check` verifies this numerically.

The microscopic simulation here is small (``N`` in the low thousands).
Finite-size effects are large at that scale, so anything measured here is a
finite-N descriptive comparison, not a capacity estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from .systems import RecurrentSystem


def rotation(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, s], [-s, c]])


def hopf_beta(phi: float, delta: float) -> float:
    """``beta_c(phi)`` from the End Matter of arXiv:2609.07341."""
    c = np.cos(phi)
    return float((-(1 - delta) * c + np.sqrt((1 - delta) ** 2 * c * c + delta * (2 - delta))) / delta)


def linear_frequency(phi: float, beta: float, delta: float) -> float:
    return float(np.arctan2(delta * beta * np.sin(phi), (1 - delta) + delta * beta * np.cos(phi)))


def _sigmas(M: int) -> np.ndarray:
    return np.array(list(product((-1.0, 1.0), repeat=M)))


@dataclass(eq=False)
class OverlapMeanField(RecurrentSystem):
    """The ``alpha = 0`` overlap map as a deterministic recurrent system."""

    A: np.ndarray
    beta: float
    delta: float
    dimension: int = field(init=False)

    def __post_init__(self):
        self.A = np.array(self.A, dtype=np.float64)
        self.dimension = self.A.shape[0]
        self._sig = _sigmas(self.dimension)

    def initial_state(self, rng):
        return rng.uniform(-1.0, 1.0, self.dimension)

    def step(self, m, u=None):
        self._input(u)
        fields = self.beta * (self._sig @ (self.A @ m))
        return (1 - self.delta) * m + self.delta * (self._sig.T @ np.tanh(fields)) / self._sig.shape[0]

    def jacobian(self, m, u=None):
        fields = self.beta * (self._sig @ (self.A @ m))
        w = 1.0 - np.tanh(fields) ** 2
        inner = (self._sig.T * w) @ self._sig / self._sig.shape[0]
        return (1 - self.delta) * np.eye(self.dimension) + self.delta * self.beta * inner @ self.A

    def parameters(self):
        return {"A": self.A.copy(), "beta": self.beta, "delta": self.delta}


def linearization_check(phi: float, delta: float) -> dict:
    """Spectral radius and eigen-argument of the ``alpha = 0`` linearization at ``beta_c(phi)``."""
    beta = hopf_beta(phi, delta)
    sys = OverlapMeanField(rotation(phi), beta, delta)
    eig = np.linalg.eigvals(sys.jacobian(np.zeros(2)))
    return {"phi": phi, "beta_c": beta, "spectral_radius": float(np.max(np.abs(eig))),
            "argument": float(np.max(np.abs(np.angle(eig)))), "omega_formula": abs(linear_frequency(phi, beta, delta))}


# ---------------------------------------------------------------------------
# Microscopic network (small N; seeded stochastic Glauber dynamics)
# ---------------------------------------------------------------------------


def couplings(rng: np.random.Generator, n: int, target_phi: float, blocks: int, disorder: str,
              coherent_phi: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``J = (xi A xi^T + xi_hat A_hat xi_hat^T) / N`` with zero diagonal (Eq. 4), and the target patterns.

    ``disorder`` is ``"coherent"`` (every block rotates by the target angle) or
    ``"uniform"`` (block angles i.i.d. uniform on [0, 2 pi)). ``blocks`` is the
    number of 2x2 disorder blocks, so ``P = 2 * blocks`` and ``alpha = P / N``.
    """
    xi = rng.choice((-1.0, 1.0), size=(n, 2))
    J = xi @ rotation(target_phi) @ xi.T
    if blocks:
        xh = rng.choice((-1.0, 1.0), size=(n, 2 * blocks))
        if disorder == "coherent":
            angles = np.full(blocks, target_phi if coherent_phi is None else coherent_phi)
        elif disorder == "uniform":
            angles = rng.uniform(0.0, 2.0 * np.pi, blocks)
        else:
            raise ValueError("disorder must be coherent or uniform")
        Ah = np.zeros((2 * blocks, 2 * blocks))
        for k, a in enumerate(angles):
            Ah[2 * k: 2 * k + 2, 2 * k: 2 * k + 2] = rotation(a)
        J += xh @ Ah @ xh.T
    J /= n
    np.fill_diagonal(J, 0.0)
    return J, xi


def glauber_overlaps(rng: np.random.Generator, J: np.ndarray, xi: np.ndarray, x0: np.ndarray, *, beta: float,
                     delta: float, steps: int) -> np.ndarray:
    """Target overlaps ``m_t = xi^T x_t / N`` along one seeded Glauber trajectory (Eq. 3)."""
    n = J.shape[0]
    x = np.array(x0, dtype=np.float64)
    out = np.empty((steps + 1, xi.shape[1]))
    out[0] = xi.T @ x / n
    for t in range(steps):
        h = J @ x
        update = rng.random(n) < delta
        up = rng.random(n) < 0.5 * (1.0 + np.tanh(beta * h))
        x = np.where(update, np.where(up, 1.0, -1.0), x)
        out[t + 1] = xi.T @ x / n
    return out


def retrieval_norm(overlaps: np.ndarray, last: int) -> float:
    """``sqrt(<||m_t||^2>)`` over the last ``last`` steps (End Matter, capacity calculation)."""
    tail = overlaps[-last:]
    return float(np.sqrt(np.mean(np.sum(tail * tail, axis=1))))
