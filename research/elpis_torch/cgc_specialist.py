"""R11: the microscopic, expert-local CGC specialist profile, its matched control, its accounting and diagnostics.

The expert keeps its anatomy, ``Expert(u) = SwiGLU(u) + g * CGCTransform(u)`` (``sparse_channel.ChannelExpert``).
R11 only bounds the CGC branch to R97W's microscopic geometry and names two mathematical profiles:

    bottleneck d = 5, cells C = 3, units N = 8           core: d * C * N = 120 parameters (R97W's E0 budget)
    P_in: hidden -> 5 (5H)    P_out: 3 -> hidden (3H)    gate: 1             adapter: 8H + 121 parameters

``E0_REFERENCE``        ``preact_bound=None``: every cell is exactly E0 (``e0.py``), factorized through ``P_in``.
``BOUNDED_SPECIALIST``  ``preact_bound=3``: ``u -> 3 tanh(u/3)`` before phi. E0-DERIVED, NOT E0.

``P_out`` and the gate are architecture around E0, not E0. Nothing here gives CGC memory, attention, retrieval or
meta-learning: any adaptation is an explicit procedure of the caller (R11 Stage A/B harness, offline research only).

``GeluSpecialist`` is the budget-matched ordinary nonlinear specialist (a CONTROL arm): the same ``P_in`` and
``P_out``, and R97W's 120-parameter GELU network (5 -> 13 -> 3, with biases) in place of the cubic core.
"""
from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn
from torch.nn import functional as F

from .cgc import CGCTransform
from .config import ConfigError, ModelConfig

MICRO_BOTTLENECK, MICRO_CHANNELS, MICRO_UNITS = 5, 3, 8
MICRO_CORE_PARAMETERS = MICRO_BOTTLENECK * MICRO_CHANNELS * MICRO_UNITS           # 120
GELU_WIDTH = 13                                                                    # 13*5 + 13 + 3*13 + 3 = 120
PROFILES = {"E0_REFERENCE": None, "BOUNDED_SPECIALIST": 3.0}
BUDGET_FRACTION = 0.05          # R11 engineering budget: adapter <= 5% of the expert's SwiGLU parameters


class GeluSpecialist(nn.Module):
    """``P_out(W2 gelu(W1 P_in u + b1) + b2)``: R97W's 120-parameter GELU network between the CGC's projections."""

    def __init__(self, hidden: int, bottleneck: int, width: int, outputs: int, *, dtype=torch.float32):
        super().__init__()
        self.proj_in = nn.Linear(hidden, bottleneck, bias=False, dtype=dtype)
        self.hidden = nn.Linear(bottleneck, width, dtype=dtype)
        self.out = nn.Linear(width, outputs, dtype=dtype)
        self.proj_out = nn.Linear(outputs, hidden, bias=False, dtype=dtype)

    def core_parameters(self) -> list:
        return [self.hidden.weight, self.hidden.bias, self.out.weight, self.out.bias]

    def cells(self, u: torch.Tensor) -> torch.Tensor:
        return self.out(F.gelu(self.hidden(self.proj_in(u))))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.cells(u))


# -- the profile ---------------------------------------------------------------------------------------------------

def profile_bound(profile: str) -> float | None:
    if profile not in PROFILES:
        raise ConfigError(f"profile in {tuple(PROFILES)}")
    return PROFILES[profile]


def micro_sparse(base, *, profile: str, gate_mode: str = "learned", kind: str = "ffn+cgc"):
    """``base`` (a ``SparseConfig``) with every routed and shared expert's specialist bounded to the micro geometry.

    Expert count, width, routing and layers are unchanged: only the specialist branch shrinks. ``kind="ffn+mlp"``
    builds the budget-matched GELU control in the same slot."""
    return dataclasses.replace(base, kind=kind, cgc_bottleneck=MICRO_BOTTLENECK, cgc_channels=MICRO_CHANNELS,
                               cgc_units=MICRO_UNITS, cgc_preact_bound=profile_bound(profile),
                               cgc_gate_mode=gate_mode, mlp_width=GELU_WIDTH if kind == "ffn+mlp" else None)


def micro_model(m: ModelConfig, *, profile: str) -> ModelConfig:
    """``m`` with its dense ``ffn+cgc``/``cgc`` channels bounded to the micro geometry (a NEW configuration)."""
    return m.replace(cgc_bottleneck=MICRO_BOTTLENECK, cgc_channels=MICRO_CHANNELS, cgc_units=MICRO_UNITS,
                     cgc_preact_bound=profile_bound(profile))


def adapter_parameters(hidden: int, *, bottleneck: int = MICRO_BOTTLENECK, channels: int = MICRO_CHANNELS,
                       units: int = MICRO_UNITS, gate: bool = True) -> dict:
    core = bottleneck * channels * units
    out = {"proj_in": hidden * bottleneck, "core": core, "proj_out": channels * hidden, "gate": int(gate)}
    out["total"] = sum(out.values())
    return out


def specialist_budget(hidden: int, ffn_hidden: int, adapter_total: int, fraction: float = BUDGET_FRACTION) -> dict:
    swiglu = 3 * hidden * ffn_hidden
    return {"swiglu_parameters": swiglu, "adapter_parameters": adapter_total, "budget_fraction": fraction,
            "budget_parameters": fraction * swiglu, "adapter_fraction": adapter_total / swiglu,
            "within_budget": adapter_total <= fraction * swiglu}


def operator_costs(hidden: int, bottleneck: int, channels: int, units: int, bounded: bool) -> dict:
    """Per token: weight multiply-accumulates, and the elementwise work a MAC count does not see."""
    cubic_units = channels * units
    return {"proj_in_macs": hidden * bottleneck, "core_macs": bottleneck * cubic_units,
            "proj_out_macs": channels * hidden,
            "weight_macs": hidden * bottleneck + bottleneck * cubic_units + channels * hidden,
            "cubic_horner_multiplies": 3 * cubic_units, "unit_sum_adds": cubic_units,
            "tanh_evaluations": cubic_units if bounded else 0, "gate_multiplies": hidden}


# -- accounting of whole models (meta tensors: nothing is allocated) -----------------------------------------------

def _specialist_role(name: str) -> str | None:
    if ".cgc." in name or name.endswith("cgc_gate"):
        return "gate" if name.endswith("cgc_gate") else ("proj_in" if ".proj_in." in name else
                                                          "proj_out" if ".proj_out." in name else "core")
    return None


def cgc_census(model) -> dict:
    """Every CGC instance of ``model`` by placement (dense channel, routed expert, shared expert), with exact
    per-component parameter counts read from the tensors, resident and active per token."""
    instances = {"dense": 0, "routed": 0, "shared": 0}
    by_component = {"proj_in": 0, "core": 0, "proj_out": 0, "gate": 0}
    by_placement = {"dense": 0, "routed": 0, "shared": 0}
    active = 0
    for index, block in enumerate(model.blocks):
        channel = getattr(block, "sparse", None)
        if channel is None:
            if getattr(block, "cgc", None) is not None:
                instances["dense"] += 1
                n = _module_cgc_parameters(block)
                active += n
                by_placement["dense"] += n
            continue
        top_k = channel.config.top_k
        for expert in channel.experts:
            if expert.cgc is not None:
                instances["routed"] += 1
                by_placement["routed"] += _module_cgc_parameters(expert)
        if channel.experts[0].cgc is not None:
            active += top_k * _module_cgc_parameters(channel.experts[0])
        if channel.shared is not None and channel.shared.cgc is not None:
            instances["shared"] += 1
            n = _module_cgc_parameters(channel.shared)
            by_placement["shared"] += n
            active += n
    for name, p in model.named_parameters():
        role = _specialist_role(name)
        if role is not None:
            by_component[role] += p.numel()
    total = sum(p.numel() for p in model.parameters())
    resident = sum(by_component.values())
    return {"instances": instances, "instances_total": sum(instances.values()), "by_component": by_component,
            "by_placement": by_placement, "resident_cgc_parameters": resident,
            "active_cgc_parameters_per_token": active, "model_total_parameters": total,
            "cgc_share_of_model": resident / total}


def _module_cgc_parameters(module) -> int:
    n = 0
    if getattr(module, "cgc", None) is not None:
        n += sum(p.numel() for p in module.cgc.parameters())
    if getattr(module, "cgc_gate", None) is not None:
        n += module.cgc_gate.numel()
    return n


# -- diagnostics of one expert --------------------------------------------------------------------------------------

@torch.no_grad()
def branch_diagnostics(expert, u: torch.Tensor) -> dict:
    """What the specialist branch actually contributes on inputs ``u``: gate, output RMS against the FFN's, the
    effective residual contribution, pre-activation scale and saturation, and finiteness."""
    ffn = expert.ffn(u)
    out = {"ffn_rms": float(ffn.pow(2).mean().sqrt())}
    if expert.cgc is not None:
        raw = expert.cgc(u)
        code = u if expert.cgc.proj_in is None else expert.cgc.proj_in(u)
        cells = expert.cgc.core(code)
        out.update(expert.cgc.core.diagnostics(code))
        out["bound"] = expert.cgc.core.preact_bound
        out["cell_rms"] = float(cells.pow(2).mean().sqrt())
        out["cell_abs_max"] = float(cells.abs().max())
        gate = 1.0 if expert.cgc_gate is None else float(expert.cgc_gate)
        out["gate"] = gate
        out["gate_mode"] = "fixed" if expert.cgc_gate is None else "learned"
    elif expert.mlp is not None:
        raw = expert.mlp(u)
        cells = expert.mlp.cells(u)
        out["cell_rms"] = float(cells.pow(2).mean().sqrt())
        out["cell_abs_max"] = float(cells.abs().max())
        gate = 1.0
        out["gate"], out["gate_mode"] = gate, "fixed"
    else:
        return out
    branch = gate * raw
    out["branch_unscaled_rms"] = float(raw.pow(2).mean().sqrt())
    out["branch_rms"] = float(branch.pow(2).mean().sqrt())
    out["branch_to_ffn_rms"] = out["branch_rms"] / max(out["ffn_rms"], 1e-30)
    out["nonfinite"] = int((~torch.isfinite(branch)).sum())
    return out


def gradient_norms(expert) -> dict:
    """L2 norms of the gradients present now, by component of the expert."""
    groups = {}
    for name, p in expert.named_parameters():
        if p.grad is None:
            continue
        key = ("gate" if name == "cgc_gate" else "ffn" if name.startswith("ffn.") else
               "proj_in" if ".proj_in." in name else "proj_out" if ".proj_out." in name else "core")
        groups[key] = groups.get(key, 0.0) + float(p.grad.detach().double().pow(2).sum())
    return {k: math.sqrt(v) for k, v in groups.items()}


def specialist_core_names(expert) -> list:
    """Parameter names of the specialist's core (the 120 parameters an explicit inner loop may adapt)."""
    if expert.cgc is not None:
        return ["cgc.core.weight"]
    if expert.mlp is not None:
        return ["mlp.hidden.weight", "mlp.hidden.bias", "mlp.out.weight", "mlp.out.bias"]
    return []


def check_adapter(expert, hidden: int, ffn_hidden: int) -> dict:
    """The exact adapter accounting of one built expert, checked against the formula and the budget."""
    if expert.cgc is not None:
        t = expert.cgc
        counted = {"proj_in": 0 if t.proj_in is None else t.proj_in.weight.numel(), "core": t.core.weight.numel(),
                   "proj_out": t.proj_out.weight.numel(), "gate": 0 if expert.cgc_gate is None else 1}
        formula = adapter_parameters(hidden, bottleneck=t.core.in_dim, channels=t.core.channels,
                                     units=t.core.units, gate=expert.cgc_gate is not None)
    elif expert.mlp is not None:
        t = expert.mlp
        counted = {"proj_in": t.proj_in.weight.numel(), "core": sum(p.numel() for p in t.core_parameters()),
                   "proj_out": t.proj_out.weight.numel(), "gate": 0}
        formula = None
    else:
        raise ConfigError("the expert has no specialist branch")
    counted["total"] = sum(counted.values())
    if formula is not None and {k: formula[k] for k in counted} != counted:
        raise AssertionError(f"adapter accounting mismatch: {counted} against {formula}")
    return {"counted": counted, "budget": specialist_budget(hidden, ffn_hidden, counted["total"]),
            "cgc_formula_parameter_count": None if expert.cgc is None else CGCTransform.parameter_count(
                hidden, expert.cgc.core.channels, expert.cgc.core.units, expert.cgc.bottleneck)}
