"""R6: a sparse, token-routed channel of FFN + CGC experts for selected ERSA blocks.

ERSA (and the optional window attention) stays the block's token mixer and the only owner of volatile state. This
module replaces the block's CHANNEL sublayer, nothing else:

    u      = RMSNorm(x)                                   (the block's channel norm)
    p      = softmax(W_r u)                               router in float32 (or wider)
    S      = top-K experts of p (ties: lower expert index first; deterministic)
    w_i    = p_i / sum_{j in S} p_j  (i in S)             normalized routing weights (or p_i unnormalized)
    out    = shared(u) + sum_{i in S} w_i expert_i(u)     shared expert optional
    expert_i(u) = FFN_i(u) + g_i CGC_i(u)                 own SwiGLU, own CGC, own learnable gate g_i

Every token is processed by exactly its K selected experts (no capacity limit, no dropping). The grouped dispatch
evaluates only selected experts: tokens are sorted by expert once, each expert runs on its contiguous segment,
and weighted outputs are scattered back. The reference dispatch evaluates every expert on every token and masks;
it exists only to test the grouped path.

Batch variance (R6 closeout): an expert's GEMMs run on the rows routed to it, so their row count depends on every
token's route in the call, future tokens included, and the BLAS may round a row differently for a different row
count (MKL double precision on AVX2 kernels with several threads does; ``run_r6 causality`` measures it). Routes
and values never cross tokens, so the grouped path is causal to rounding, not bit for bit: the tests check the
fixed-shape reference dispatch bit for bit, the grouped path within 1e-12 in float64, and a planted leak.

Per call, ``self.report`` holds the auxiliary losses (load balance, router z-loss; differentiable, with their
configured weights) and detached routing statistics. It is overwritten by every call (bounded, never persisted).
Token routing never reads labels: the router sees the hidden state only.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .cgc import CGCTransform
from .config import ConfigError
from .model import SwiGLU

KINDS = ("ffn+cgc", "ffn")
DISPATCH = ("grouped", "reference")


@dataclass(frozen=True)
class SparseConfig:
    experts: int = 8
    top_k: int = 2
    layers: tuple = ()                    # block indices whose channel is sparse (others keep the dense channel)
    shared: bool = False                  # an always-active shared expert (same anatomy)
    kind: str = "ffn+cgc"                 # routed expert anatomy: "ffn+cgc" or "ffn"
    ffn_hidden: int | None = None         # per expert; None: the model's ffn_width
    cgc_channels: int | None = None       # per expert; None: the model's CGC geometry
    cgc_bottleneck: int | None = None
    cgc_units: int | None = None
    gate_init: float = 0.1                # g_i at initialization (0.0: exactly the FFN; then g_i learns first)
    normalize: bool = True                # renormalize the selected probabilities to sum to 1
    balance_weight: float = 0.0           # weight of the load-balance loss in ``aux_loss`` (always reported)
    z_weight: float = 0.0                 # weight of the router z-loss in ``aux_loss`` (always reported)
    dispatch: str = "grouped"

    def __post_init__(self):
        for name in ("experts", "top_k"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ConfigError(f"{name} must be a positive integer")
        if self.top_k > self.experts:
            raise ConfigError("top_k cannot exceed the number of experts")
        if self.kind not in KINDS or self.dispatch not in DISPATCH:
            raise ConfigError(f"kind in {KINDS}, dispatch in {DISPATCH}")
        layers = tuple(self.layers)
        if len(set(layers)) != len(layers) or any(type(i) is not int or i < 0 for i in layers):
            raise ConfigError("layers must be distinct non-negative block indices")
        object.__setattr__(self, "layers", tuple(sorted(layers)))
        for name in ("ffn_hidden", "cgc_channels", "cgc_bottleneck", "cgc_units"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ConfigError(f"{name} must be None or a positive integer")
        for name in ("gate_init", "balance_weight", "z_weight"):
            if not math.isfinite(float(getattr(self, name))) or (name != "gate_init" and getattr(self, name) < 0):
                raise ConfigError(f"{name} must be finite (weights non-negative)")

    def replace(self, **changes) -> "SparseConfig":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class ChannelExpert(nn.Module):
    """``FFN(u) + g * CGC(u)`` with its own parameters (``kind="ffn"``: the FFN alone)."""

    def __init__(self, m, s: SparseConfig, dtype):
        super().__init__()
        self.ffn = SwiGLU(m.d_model, s.ffn_hidden or m.ffn_width, dtype)
        self.cgc = self.cgc_gate = None
        if s.kind == "ffn+cgc":
            self.cgc = CGCTransform(m.d_model, s.cgc_channels or m.cgc_cells, s.cgc_units or m.cgc_units,
                                    bottleneck=s.cgc_bottleneck or m.cgc_bottleneck, preact_bound=m.cgc_preact_bound,
                                    dtype=dtype)
            self.cgc_gate = nn.Parameter(torch.full((1,), float(s.gate_init), dtype=dtype))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        out = self.ffn(u)
        return out if self.cgc is None else out + self.cgc_gate * self.cgc(u)

    def macs(self) -> int:
        """Multiply-accumulates per token (weights only; elementwise work not counted)."""
        return sum(p.numel() for name, p in self.named_parameters() if p.ndim >= 2)


class TokenRouter(nn.Module):
    """Per-token top-K over experts from the hidden state (float32 logits, deterministic ties)."""

    def __init__(self, d_model: int, experts: int, top_k: int, normalize: bool, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(experts, d_model, dtype=dtype))
        self.top_k, self.normalize = top_k, normalize

    def forward(self, u: torch.Tensor):
        """``u [N, d]`` -> ``(logits [N, E], probs [N, E], index [N, K], weight [N, K])`` in float32, or in the
        input's precision when it is wider (a float64 model routes in float64)."""
        precision = torch.promote_types(u.dtype, torch.float32)
        logits = F.linear(u.to(precision), self.weight.to(precision))
        probs = torch.softmax(logits, dim=-1)
        order = torch.sort(probs, dim=-1, descending=True, stable=True).indices     # ties: lower index first
        index = order[:, :self.top_k]
        weight = torch.gather(probs, 1, index)
        if self.normalize:
            weight = weight / weight.sum(-1, keepdim=True)
        return logits, probs, index, weight


class SparseChannel(nn.Module):
    """The routed channel of one block (see the module docstring)."""

    def __init__(self, m, s: SparseConfig, dtype):
        super().__init__()
        self.config = s
        self.router = TokenRouter(m.d_model, s.experts, s.top_k, s.normalize, dtype)
        self.experts = nn.ModuleList(ChannelExpert(m, s, dtype) for _ in range(s.experts))
        self.shared = ChannelExpert(m, s, dtype) if s.shared else None
        self.routed_enabled = True        # CONTROL: False drops every routed contribution (shared expert only)
        self.report = {}

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        shape = u.shape
        flat = u.reshape(-1, shape[-1])
        out = self.shared(flat) if self.shared is not None else torch.zeros_like(flat)
        if not self.routed_enabled:
            self.report = {"routed_enabled": False, "tokens": flat.shape[0]}
            return out.reshape(shape)
        logits, probs, index, weight = self.router(flat)
        weight = weight.to(flat.dtype)
        if self.config.dispatch == "reference":
            out = out + self._reference(flat, index, weight)
        else:
            out = self._grouped(flat, index, weight, out)
        self.report = self._report(logits, probs, index)
        return out.reshape(shape)

    def _grouped(self, flat, index, weight, out):
        """Evaluate each expert only on its own tokens: one stable sort by expert, contiguous segments."""
        k = index.shape[1]
        assignment = index.reshape(-1)
        order = torch.argsort(assignment, stable=True)
        counts = torch.bincount(assignment, minlength=len(self.experts)).tolist()
        token = order // k
        scale = weight.reshape(-1)[order]
        start = 0
        for expert, count in zip(self.experts, counts):
            if count:
                rows = token[start:start + count]
                out = out.index_add(0, rows, expert(flat[rows]) * scale[start:start + count, None])
            start += count
        return out

    def _reference(self, flat, index, weight):
        """REFERENCE ONLY: every expert on every token, masked by the routing (tests the grouped path)."""
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            mask = (index == e).to(flat.dtype)
            coefficient = (mask * weight).sum(-1, keepdim=True)
            out = out + coefficient * expert(flat)
        return out

    def _report(self, logits, probs, index) -> dict:
        n, experts = probs.shape
        k = index.shape[1]
        assignment = F.one_hot(index, experts).sum(1).float()              # [N, E], one per selected expert
        fraction = assignment.mean(0) / k                                    # share of the N*K assignments
        balance = experts * (fraction.detach() * probs.mean(0)).sum()        # 1.0 when perfectly uniform
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        aux = self.config.balance_weight * balance + self.config.z_weight * z
        with torch.no_grad():
            counts = assignment.sum(0)
            entropy = -(probs * torch.log(probs.clamp_min(1e-30))).sum(-1).mean()
            co = assignment.t() @ assignment
            mean = counts.mean().clamp_min(1e-12)
            per_expert = self.experts[0].macs()
            router = experts * self.router.weight.shape[1]
            shared = self.shared.macs() if self.shared is not None else 0
            stats = {"tokens": n, "counts": counts.long().tolist(), "frequency": (counts / max(1, n * k)).tolist(),
                     "load_imbalance_max_over_mean": float(counts.max() / mean),
                     "load_cv": float(counts.std(unbiased=False) / mean), "router_entropy": float(entropy),
                     "router_entropy_max": math.log(experts), "top1_probability": float(probs.max(-1).values.mean()),
                     "inactive_fraction": float((counts == 0).float().mean()),
                     "coactivation": co.long().tolist(),
                     "active_macs_per_token": per_expert * k + shared + router,
                     "macs_this_call": int(counts.sum()) * per_expert + n * (shared + router)}
        return {"aux_loss": aux, "balance_loss": balance, "z_loss": z, "stats": stats}


def channel_reports(model) -> list:
    """The per-call reports of every sparse channel of ``model`` (in block order)."""
    return [block.sparse.report for block in model.blocks if getattr(block, "sparse", None) is not None]


def aux_loss(model) -> torch.Tensor | None:
    """Sum of the configured auxiliary losses of the last call (``None`` without sparse channels)."""
    terms = [r["aux_loss"] for r in channel_reports(model) if "aux_loss" in r]
    return sum(terms) if terms else None


def active_parameters(model) -> dict:
    """Parameters per token: total, resident (all, which must be in memory to serve any route) and active
    (dense parts + K routed experts + the shared expert + the router per sparse block)."""
    total = sum(p.numel() for p in model.parameters())
    inactive = 0
    for block in model.blocks:
        channel = getattr(block, "sparse", None)
        if channel is None:
            continue
        per_expert = sum(p.numel() for p in channel.experts[0].parameters())
        inactive += per_expert * (len(channel.experts) - channel.config.top_k)
    ple = getattr(model, "ple", None)
    if ple is not None:                      # R7: a token reads `lookups` rows of the table, not the table
        inactive += ple.table.weight.numel() - ple.config.lookups * ple.config.head_dim
    return {"total": total, "resident": total, "active_per_token": total - inactive}


@torch.no_grad()
def upcycle(dense, sparse: SparseConfig):
    """A sparse copy of a dense ``ErsaLM``-family model whose function equals the dense model's, by construction.

    Without a shared expert, every routed expert of a sparse block is a copy of that block's dense channel; the
    selected weights are renormalized to sum to 1, so the routed sum equals the dense channel up to the rounding
    of that sum (exact algebraically). With ``shared=True`` the shared expert is the copy and every routed expert
    starts with a zero output (its FFN output projection and CGC gate are zero), so the output is bitwise the dense
    channel's until training moves them. A dense ``cgc`` channel becomes ``FFN(u) = 0`` plus gate 1 times the CGC;
    a dense ``ffn`` channel becomes the FFN with a zero CGC gate."""
    if not sparse.normalize and not sparse.shared:
        raise ConfigError("cloned routed experts reproduce the dense channel only with normalized routing weights")
    m = dense.config
    ersa = dataclasses.replace(dense.ersa, sparse=sparse)
    if hasattr(dense, "evidence"):
        new = type(dense)(ersa, dense.evidence)
    elif hasattr(dense, "history"):
        new = type(dense)(ersa, dense.history)
    else:
        new = type(dense)(ersa)
    new = new.to(dtype=dense.embed.weight.dtype)
    source = dense.state_dict()
    target = new.state_dict()
    for name in target:
        if name in source and ".sparse." not in name:
            target[name].copy_(source[name])
    for index in sparse.layers:
        old, channel = dense.blocks[index], new.blocks[index].sparse
        clones = [channel.shared] if sparse.shared else list(channel.experts)
        for expert in clones:
            _copy_dense_channel(old, expert, m, sparse)
        if sparse.shared:
            for expert in channel.experts:
                expert.ffn.out.weight.zero_()
                if expert.cgc_gate is not None:
                    expert.cgc_gate.zero_()
    return new


def _copy_dense_channel(block, expert: ChannelExpert, m, sparse: SparseConfig) -> None:
    if block.ffn is not None:
        if expert.ffn.gate.weight.shape != block.ffn.gate.weight.shape:
            raise ConfigError("the expert FFN width must equal the dense FFN width to upcycle")
        expert.ffn.load_state_dict(block.ffn.state_dict())
    else:
        expert.ffn.out.weight.zero_()                       # a dense CGC-only channel: the FFN contributes 0
    if block.cgc is not None:
        if expert.cgc is None:
            raise ConfigError("a dense channel with CGC needs ffn+cgc experts to upcycle")
        expert.cgc.load_state_dict(block.cgc.state_dict())
        expert.cgc_gate.fill_(1.0 if block.cgc_gate is None else float(block.cgc_gate))
    elif expert.cgc_gate is not None:
        expert.cgc_gate.zero_()
