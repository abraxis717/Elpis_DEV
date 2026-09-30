"""Generic recurrent-system interface and the synthetic families. RESEARCH_ONLY.

Every system is a deterministic map ``x_{t+1} = F(x_t, u_t)`` with explicit
parameters. Randomness enters only through a caller-supplied
``numpy.random.Generator`` (see :func:`spec.world_rng`); nothing here touches a
global RNG.

Families:

* :class:`TanhRNN`: ``x_{t+1} = tanh(g J x_t + b + B u_t)`` with the
  nonreciprocal coupling ``J = (S + gamma A) / sqrt(1 + gamma^2)``, see
  :func:`make_tanh_rnn`. A synthetic probe, NOT the production architecture.
* :class:`LinearSystem`: ``x_{t+1} = M x_t`` (stable/unstable fixtures).
* :class:`CyclicShift`: exact period-p permutation of coordinates.
* :class:`LogisticMap`: coordinatewise ``r x (1 - x)``; a detector fixture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


class RecurrentSystem:
    """Interface: deterministic initial state, step, Jacobian, parameters, clone."""

    dimension: int
    input_dimension: int = 0

    def initial_state(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def step(self, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    def jacobian(self, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        """``dF/dx`` at ``x``; systems without one raise ``NotImplementedError``."""
        raise NotImplementedError

    def parameters(self) -> dict:
        """Explicit parameters as copies (mutating the result never changes the system)."""
        raise NotImplementedError

    def clone(self) -> "RecurrentSystem":
        """An exact, independent copy (bitwise-equal parameters, no shared buffers)."""
        return type(self)(**{k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
                             for k, v in self.parameters().items()})

    def _input(self, u):
        if self.input_dimension == 0:
            if u is not None and np.size(u) != 0:
                raise ValueError("this system takes no input")
            return None
        if u is None:
            return np.zeros(self.input_dimension)
        u = np.asarray(u, dtype=np.float64)
        if u.shape != (self.input_dimension,):
            raise ValueError("input dimension")
        return u


def rollout(system: RecurrentSystem, x0: np.ndarray, steps: int, inputs: np.ndarray | None = None) -> np.ndarray:
    """States ``x_0 .. x_steps`` as a ``(steps + 1, d)`` array."""
    x = np.array(x0, dtype=np.float64, copy=True)
    out = np.empty((steps + 1, x.size))
    out[0] = x
    for t in range(steps):
        x = system.step(x, None if inputs is None else inputs[t])
        out[t + 1] = x
    return out


def run(system: RecurrentSystem, x0: np.ndarray, steps: int) -> np.ndarray:
    """Final state after ``steps`` steps (no trajectory kept)."""
    x = np.array(x0, dtype=np.float64, copy=True)
    for _ in range(steps):
        x = system.step(x)
    return x


def perturb(x: np.ndarray, rng: np.random.Generator, magnitude: float) -> np.ndarray:
    """``x + magnitude * v`` for a uniformly random unit direction ``v``."""
    v = rng.standard_normal(x.size)
    return x + magnitude * v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# tanh recurrent network with nonreciprocal coupling
# ---------------------------------------------------------------------------


def nonreciprocal_coupling(rng: np.random.Generator, d: int, gamma: float) -> np.ndarray:
    """``J = (S + gamma A) / sqrt(1 + gamma^2)``.

    ``S = (G1 + G1^T) / sqrt(2d)`` and ``A = (G2 - G2^T) / sqrt(2d)`` with
    ``G1, G2`` i.i.d. standard normal. Every off-diagonal entry of ``J`` then
    has variance ``1/d`` for every gamma, and the correlation between ``J_ij``
    and ``J_ji`` is ``tau = (1 - gamma^2) / (1 + gamma^2)``: gamma = 0 is
    symmetric (reciprocal), gamma = 1 is uncorrelated, gamma -> inf is
    antisymmetric. Diagonal entries come from ``S`` only (variance
    ``2 / (d (1 + gamma^2))``).
    """
    g1 = rng.standard_normal((d, d))
    g2 = rng.standard_normal((d, d))
    s = (g1 + g1.T) / math.sqrt(2 * d)
    a = (g2 - g2.T) / math.sqrt(2 * d)
    return (s + gamma * a) / math.sqrt(1.0 + gamma * gamma)


@dataclass(eq=False)
class TanhRNN(RecurrentSystem):
    """``x_{t+1} = tanh(gain * J @ x + b + B @ u)``. A probe, not the production architecture."""

    J: np.ndarray
    b: np.ndarray
    B: np.ndarray
    gain: float
    gamma: float
    init_scale: float = 1.0
    dimension: int = field(init=False)
    input_dimension: int = field(init=False)

    def __post_init__(self):
        self.J = np.array(self.J, dtype=np.float64)
        self.b = np.array(self.b, dtype=np.float64)
        self.B = np.array(self.B, dtype=np.float64).reshape(self.J.shape[0], -1)
        self.dimension = self.J.shape[0]
        self.input_dimension = self.B.shape[1]
        for arr in (self.J, self.b, self.B):
            arr.setflags(write=False)

    def initial_state(self, rng):
        return self.init_scale * rng.standard_normal(self.dimension)

    def _pre(self, x, u):
        pre = self.gain * (self.J @ x) + self.b
        u = self._input(u)
        if u is not None:
            pre = pre + self.B @ u
        return pre

    def step(self, x, u=None):
        return np.tanh(self._pre(x, u))

    def jacobian(self, x, u=None):
        y = np.tanh(self._pre(x, u))
        return (1.0 - y * y)[:, None] * (self.gain * self.J)

    def parameters(self):
        return {"J": self.J.copy(), "b": self.b.copy(), "B": self.B.copy(), "gain": self.gain,
                "gamma": self.gamma, "init_scale": self.init_scale}


def make_tanh_rnn(rng: np.random.Generator, dimension: int, gain: float, gamma: float,
                  bias_scale: float = 0.0, input_dimension: int = 0, init_scale: float = 1.0) -> TanhRNN:
    J = nonreciprocal_coupling(rng, dimension, gamma)
    b = bias_scale * rng.standard_normal(dimension)
    B = rng.standard_normal((dimension, input_dimension)) / math.sqrt(max(input_dimension, 1))
    return TanhRNN(J, b, B, float(gain), float(gamma), float(init_scale))


# ---------------------------------------------------------------------------
# Fixtures with known behaviour
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class LinearSystem(RecurrentSystem):
    """``x_{t+1} = M x_t``: perturbations decay iff the spectral radius is below one."""

    M: np.ndarray
    dimension: int = field(init=False)

    def __post_init__(self):
        self.M = np.array(self.M, dtype=np.float64)
        self.M.setflags(write=False)
        self.dimension = self.M.shape[0]

    def initial_state(self, rng):
        return rng.standard_normal(self.dimension)

    def step(self, x, u=None):
        self._input(u)
        return self.M @ x

    def jacobian(self, x, u=None):
        return self.M.copy()

    def parameters(self):
        return {"M": self.M.copy()}


@dataclass(eq=False)
class CyclicShift(RecurrentSystem):
    """Cyclic coordinate shift: every state with distinct entries has exact period ``dimension``."""

    period: int
    dimension: int = field(init=False)

    def __post_init__(self):
        if type(self.period) is not int or self.period < 1:
            raise ValueError("period")
        self.dimension = self.period

    def initial_state(self, rng):
        return rng.standard_normal(self.dimension)

    def step(self, x, u=None):
        self._input(u)
        return np.roll(x, 1)

    def jacobian(self, x, u=None):
        return np.roll(np.eye(self.dimension), 1, axis=0)

    def parameters(self):
        return {"period": self.period}


@dataclass(eq=False)
class LogisticMap(RecurrentSystem):
    """Uncoupled logistic maps ``x -> r x (1 - x)`` on (0, 1): a diagnostic fixture only."""

    r: float
    size: int = 1
    dimension: int = field(init=False)

    def __post_init__(self):
        self.dimension = self.size

    def initial_state(self, rng):
        return rng.uniform(0.2, 0.8, self.dimension)

    def step(self, x, u=None):
        self._input(u)
        return self.r * x * (1.0 - x)

    def jacobian(self, x, u=None):
        return np.diag(self.r * (1.0 - 2.0 * x))

    def parameters(self):
        return {"r": self.r, "size": self.size}
