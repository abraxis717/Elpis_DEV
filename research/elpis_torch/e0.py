"""The CGC / ECS_G E0 cubic cell: a float64 reference, implemented separately from the pinned upstream file.

    phi(u)  = (u + u^2 + u^3) / 2
    f_W(x)  = sum_i phi(x^T w_i)            W in R^(d x N), x in R^d, f scalar
    phi'(u) = 1/2 + u + (3/2) u^2

This module is the FAITHFUL E0 mathematics and nothing else: one scalar cell, float64, no normalization, no
bound, no projection. It is the oracle every tensorized operator in ``cgc.py`` is tested against. It is itself
validated (tests/research/elpis_torch/test_e0_reference.py) against the pinned upstream source
(``upstream/cgc_ecsg_e0.py``, bitwise), exact rational arithmetic, the S3 moment identity, autograd and finite
differences, and the native ECS_G kernel in its qualified regime.

It is not a qualified ECS_G kernel: the qualified kernel is native/ECS_G (d=6, N in {36,48,72}).
"""
from __future__ import annotations

import torch

DTYPE = torch.float64


class E0Error(ValueError):
    """An input outside the E0 reference contract."""


def _check(w: torch.Tensor, x: torch.Tensor | None = None) -> None:
    if not isinstance(w, torch.Tensor) or w.ndim != 2 or w.dtype != DTYPE:
        raise E0Error("W must be a float64 tensor of shape [d, N]")
    if w.shape[0] < 1 or w.shape[1] < 1:
        raise E0Error("d and N must be positive")
    if not bool(torch.isfinite(w).all()):
        raise E0Error("W must be finite")
    if x is not None:
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.dtype != DTYPE or x.shape[1] != w.shape[0]:
            raise E0Error("X must be a float64 tensor of shape [R, d]")
        if not bool(torch.isfinite(x).all()):
            raise E0Error("X must be finite")


def phi(u: torch.Tensor) -> torch.Tensor:
    """The cubic primitive, in the upstream operation order."""
    return 0.5 * u + 0.5 * u * u + 0.5 * u * u * u


def phi_prime(u: torch.Tensor) -> torch.Tensor:
    """Its derivative. ``phi' >= 1/3`` everywhere (minimum at u = -1/3): phi is strictly increasing."""
    return 0.5 + u + 1.5 * u * u


def forward(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``f_W(x_r) = sum_i phi(x_r^T w_i)`` for every row of ``x``: ``[R, d] -> [R]``."""
    _check(w, x)
    return phi(x @ w).sum(dim=1)


def mse_loss(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    prediction = forward(w, x)
    if y.shape != prediction.shape or y.dtype != DTYPE:
        raise E0Error("y must be a float64 tensor of shape [R]")
    return ((prediction - y) ** 2).mean()


def analytic_grad(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``grad_W mean_r (f_W(x_r) - y_r)^2 = (2/R) X^T [e * phi'(XW)]`` (the G1 recurrence's gradient)."""
    error = forward(w, x) - y
    return (2.0 / x.shape[0]) * x.T @ (error[:, None] * phi_prime(x @ w))


def gd_step(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor, learning_rate: float) -> torch.Tensor:
    """One explicit gradient step ``W' = W - eta grad W`` (differentiable; never in place)."""
    return w - learning_rate * analytic_grad(w, x, y)


def moments(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The raw-sum moments ``mu = sum w_i``, ``M = sum w_i w_i^T``, ``T3 = sum w_i^(x3)`` (unpacked S3)."""
    _check(w)
    return w.sum(dim=1), w @ w.T, torch.einsum("ai,bi,ci->abc", w, w, w)


def forward_from_moments(mu: torch.Tensor, m: torch.Tensor, t3: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``f = (x.mu + x^T M x + T3[x,x,x]) / 2``: E0 is a cubic polynomial of ``x`` fixed by three moments."""
    linear = x @ mu
    quadratic = torch.einsum("ra,ab,rb->r", x, m, x)
    cubic = torch.einsum("abc,ra,rb,rc->r", t3, x, x, x)
    return 0.5 * (linear + quadratic + cubic)
