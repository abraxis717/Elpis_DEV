"""Batched CGC operators for transformer hidden states ``[batch, sequence, hidden]``.

Three layers, from faithful to architectural:

``CGCCore`` (faithful E0, vector valued)
    ``C`` independent E0 cells that share one input. The weight is ``[d, C, N]``; ``weight[:, c, :]`` is the
    ``[d, N]`` microscopic weight of cell ``c``, exactly the E0 layout, and

        y_c(x) = sum_i phi(x^T w_{c,i})                       x: [..., d]  ->  y: [..., C]

    With ``preact_bound=None`` this IS E0, channel by channel (tested to float64 round-off against ``e0.py``).

``preact_bound`` (extension, not E0)
    A soft clamp of the pre-activation, ``u -> b tanh(u / b)``, applied before phi. It caps every unit at
    ``phi(b)`` whatever the weights are, which bounds the cubic's growth in a stack. It deviates from E0 by at
    most ``|u|^3 / (3 b^2)`` in the pre-activation and is exactly E0 in the limit ``b -> inf``.

``CGCTransform`` (architecture around the core)
    ``hidden -> hidden``: an optional linear bottleneck ``P_in`` (hidden -> d), the core, and a linear readout
    ``P_out`` (C -> hidden). With the bottleneck and no bound, cell ``c`` is still exactly E0 on the full hidden
    input with the rank-constrained weight ``W_c = P_in^T core.weight[:, c, :]`` (``effective_e0_weight``): the
    bottleneck is a factorization of ``W``, not a different nonlinearity. ``P_out`` is new: E0 has no readout.

None of these tensorized forms is a qualified ECS_G kernel.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def phi(u: torch.Tensor) -> torch.Tensor:
    """``(u + u^2 + u^3) / 2`` in Horner form (three multiplications per unit)."""
    return 0.5 * u * (1.0 + u * (1.0 + u))


def phi_prime(u: torch.Tensor) -> torch.Tensor:
    return 0.5 + u * (1.0 + 1.5 * u)


def soft_bound(u: torch.Tensor, bound: float | None) -> torch.Tensor:
    return u if bound is None else bound * torch.tanh(u / bound)


class CGCCore(nn.Module):
    """``C`` independent E0 cells over one shared input: ``[..., d] -> [..., C]``."""

    def __init__(self, in_dim: int, channels: int, units: int, *, preact_bound: float | None = None,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        for name, value in (("in_dim", in_dim), ("channels", channels), ("units", units)):
            if type(value) is not int or value < 1:
                raise ValueError(f"CGCCore {name} must be a positive integer")
        if preact_bound is not None and not (isinstance(preact_bound, (int, float)) and math.isfinite(preact_bound)
                                             and preact_bound > 0):
            raise ValueError("CGCCore preact_bound must be None or a positive finite number")
        self.in_dim, self.channels, self.units = in_dim, channels, units
        self.preact_bound = None if preact_bound is None else float(preact_bound)
        self.weight = nn.Parameter(torch.zeros(in_dim, channels, units, dtype=dtype))

    def preactivations(self, x: torch.Tensor) -> torch.Tensor:
        """``u_{c,i} = x^T w_{c,i}`` (after the optional bound): ``[..., C, N]``."""
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"CGCCore expects [..., {self.in_dim}], got {tuple(x.shape)}")
        u = x @ self.weight.reshape(self.in_dim, self.channels * self.units)
        return soft_bound(u, self.preact_bound).unflatten(-1, (self.channels, self.units))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return phi(self.preactivations(x)).sum(dim=-1)

    def output_ceiling(self) -> float:
        """The largest ``|y_c|`` any input and any weights can produce (infinite when unbounded)."""
        b = self.preact_bound
        return math.inf if b is None else self.units * 0.5 * (b + b * b + b * b * b)

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor) -> dict:
        """Raw pre-activation scale and how much of it the bound is compressing."""
        raw = x @ self.weight.reshape(self.in_dim, self.channels * self.units)
        b = self.preact_bound
        return {"preact_rms": float(raw.pow(2).mean().sqrt()), "preact_abs_max": float(raw.abs().max()),
                "saturated_fraction": 0.0 if b is None else float((raw.abs() > b).double().mean())}


class CGCTransform(nn.Module):
    """``hidden -> hidden`` through E0 cells: ``P_out(core(P_in x))`` (``P_in`` omitted in the dense variant)."""

    def __init__(self, hidden: int, channels: int, units: int, *, bottleneck: int | None = None,
                 preact_bound: float | None = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        if bottleneck is not None and (type(bottleneck) is not int or bottleneck < 1):
            raise ValueError("CGCTransform bottleneck must be None or a positive integer")
        self.hidden, self.bottleneck = hidden, bottleneck
        self.proj_in = None if bottleneck is None else nn.Linear(hidden, bottleneck, bias=False, dtype=dtype)
        self.core = CGCCore(bottleneck or hidden, channels, units, preact_bound=preact_bound, dtype=dtype)
        self.proj_out = nn.Linear(channels, hidden, bias=False, dtype=dtype)

    def cells(self, x: torch.Tensor) -> torch.Tensor:
        """The E0 cell outputs before the readout: ``[..., hidden] -> [..., C]``."""
        return self.core(x if self.proj_in is None else self.proj_in(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.cells(x))

    def effective_e0_weight(self, channel: int) -> torch.Tensor:
        """The ``[hidden, N]`` E0 weight of one cell on the full hidden input (``P_in^T W_c`` when factorized)."""
        w = self.core.weight[:, channel, :]
        return w if self.proj_in is None else self.proj_in.weight.T @ w

    @staticmethod
    def parameter_count(hidden: int, channels: int, units: int, bottleneck: int | None) -> int:
        d = bottleneck or hidden
        return (hidden * bottleneck if bottleneck else 0) + d * channels * units + channels * hidden

    @staticmethod
    def multiply_accumulates(hidden: int, channels: int, units: int, bottleneck: int | None) -> int:
        """Multiply-accumulates per token: every weight once, plus three multiplications per cubic unit."""
        return CGCTransform.parameter_count(hidden, channels, units, bottleneck) + 3 * channels * units
