"""R13 P2: the shared trainable neural kernels of the cognitive substrate. RESEARCH_ONLY.

No per-vertex modules: every vertex and arc uses the same parameters (144002 vertices are addressable; only the
K resident ones carry rows).

    query ingress      q = tanh(P_query pre)                       the final block's pre-channel stream -> a row
    evidence ingress   e = tanh(W_e [attention pool of the block's encoded tokens])   an admitted block -> a row
    message            m(u -> v) = W_2 gelu(W_1 h_u + P_out[port(u -> v)] + P_in[port(v -> u)] + b_1) + b_2
    aggregation        a_v = sum of m(u -> v) over admitted arcs into v, in arc-identity order (deterministic)
    update             z = sigmoid(W_z [h_v; a_v]),  c = tanh(W_c [h_v; a_v]),  h_v' = h_v + z (c - h_v)
    readout            r = tanh(W_r mean(h over the readout vertices))
    injection          delta = g P_out r,  g a scalar that starts at exactly zero

The message depends only on the SOURCE's old row and the arc's two port ordinals: exactly what Geodome's
transport callback receives (arc, source, destination, source row). The update depends on the destination's old
row and its inbox: exactly what Geodome's update callback receives. That makes the tensor executor and the
Geodome scheduler comparable on matched semantics (``reference.py``). A port is the position of a neighbour in
the index's neighbour list (0..5): an identity ordinal of the frozen index, not a learned or geometric direction.

Every operation is a PyTorch tensor operation: no host extraction, NumPy, callbacks or ctypes on this path.
Initialization is deterministic (seeded per parameter name).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..model import _seeded_normal

PORTS = 6                                   # degree <= 6 in the frequency-120 index (12 vertices have 5)


@dataclass(frozen=True)
class CognitiveConfig:
    width: int = 32                         # D: latent row width
    hidden: int = 64                        # message MLP width
    capacity: int = 64                      # K: resident vertices (sparse residency), at most 4096
    max_rounds: int = 16                    # per reason() call, at most 256 (Geodome)
    max_arcs: int = 1024                    # admitted arcs per plan, at most 24576 (Geodome)
    max_visits: int = 16384                 # arc visits per reason() call
    max_evidence: int = 8                   # admitted evidence items per observation
    evidence_heads: int = 2                 # attention-pooling queries over an encoded block
    seed: int = 13
    gate_init: float = 0.0                  # 0: the refined logits start as the backbone's, exactly
    residency: str = "sparse"               # "dense" only for the residency parity gate

    def __post_init__(self):
        for name in ("width", "hidden", "capacity", "max_rounds", "max_arcs", "max_visits", "max_evidence",
                     "evidence_heads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.capacity > 4096 or self.max_rounds > 256 or self.max_arcs > 24576:
            raise ValueError("capacity <= 4096, max_rounds <= 256 and max_arcs <= 24576 (Geodome's caps)")
        if self.residency not in ("sparse", "dense"):
            raise ValueError("residency must be 'sparse' or 'dense'")
        if not math.isfinite(float(self.gate_init)):
            raise ValueError("gate_init must be finite")


class CognitiveKernels(nn.Module):
    """The trainable parameters of the substrate: ingress, message, update, readout and injection."""

    def __init__(self, config: CognitiveConfig, d_model: int, encoder_width: int, dtype=torch.float32):
        super().__init__()
        c, w = config, encoder_width
        self.config = config
        self.query_in = nn.Linear(d_model, c.width, dtype=dtype)                           # P_query
        self.evidence_query = nn.Parameter(torch.zeros(c.evidence_heads, w, dtype=dtype))
        self.evidence_in = nn.Linear(c.evidence_heads * w, c.width, dtype=dtype)
        self.message_in = nn.Linear(c.width, c.hidden, dtype=dtype)
        self.port_out = nn.Parameter(torch.zeros(PORTS, c.hidden, dtype=dtype))
        self.port_in = nn.Parameter(torch.zeros(PORTS, c.hidden, dtype=dtype))
        self.message_out = nn.Linear(c.hidden, c.width, dtype=dtype)
        self.update_gate = nn.Linear(2 * c.width, c.width, dtype=dtype)
        self.update_cand = nn.Linear(2 * c.width, c.width, dtype=dtype)
        self.readout_proj = nn.Linear(c.width, c.width, dtype=dtype)
        self.inject = nn.Linear(c.width, d_model, bias=False, dtype=dtype)                 # P_out
        self.gate = nn.Parameter(torch.full((1,), float(c.gate_init), dtype=dtype))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        seed = self.config.seed
        for name, p in self.named_parameters():
            if name == "gate":
                p.fill_(float(self.config.gate_init))
            elif name.endswith(".bias"):
                p.zero_()
            elif name in ("port_out", "port_in"):
                p.copy_(_seeded_normal(seed, "r13:" + name, p.shape, 0.5, p.dtype))
            elif name == "evidence_query":
                p.copy_(_seeded_normal(seed, "r13:" + name, p.shape, p.shape[1] ** -0.5, p.dtype))
            else:                                                       # a Linear weight [out, in]
                p.copy_(_seeded_normal(seed, "r13:" + name, p.shape, p.shape[1] ** -0.5, p.dtype))

    # -- ingress ----------------------------------------------------------------------------------------------
    def query(self, pre: torch.Tensor) -> torch.Tensor:
        """``pre [..., d_model]`` -> ``[..., D]``."""
        return torch.tanh(self.query_in(pre))

    def evidence(self, encoded: torch.Tensor) -> torch.Tensor:
        """``encoded [n, s + 1, w]`` (the causal encoder's states; position 0 is the token before the block) ->
        ``[n, D]``. Attention pooling over the block's own ``s`` tokens."""
        tokens = encoded[:, 1:]
        scores = torch.einsum("hw,nsw->nhs", self.evidence_query, tokens) * tokens.shape[-1] ** -0.5
        pooled = torch.einsum("nhs,nsw->nhw", torch.softmax(scores, dim=-1), tokens)
        return torch.tanh(self.evidence_in(pooled.flatten(1)))

    # -- distributed computation --------------------------------------------------------------------------------
    def message(self, source_rows: torch.Tensor, ports: torch.Tensor, back_ports: torch.Tensor) -> torch.Tensor:
        """``source_rows [E, D]`` and the arcs' port ordinals ``[E]`` -> messages ``[E, D]``."""
        hidden = self.message_in(source_rows) + self.port_out[ports] + self.port_in[back_ports]
        return self.message_out(F.gelu(hidden))

    def update(self, rows: torch.Tensor, inbox: torch.Tensor) -> torch.Tensor:
        """``rows [N, D]`` (old) and summed messages ``[N, D]`` -> new rows ``[N, D]``."""
        both = torch.cat((rows, inbox), dim=-1)
        z = torch.sigmoid(self.update_gate(both))
        return rows + z * (torch.tanh(self.update_cand(both)) - rows)

    # -- readout and injection ----------------------------------------------------------------------------------
    def readout(self, rows: torch.Tensor) -> torch.Tensor:
        """``rows [R, D]`` (the readout vertices) -> the latent ``[D]``."""
        return torch.tanh(self.readout_proj(rows.mean(dim=0)))

    def injection(self, latent: torch.Tensor) -> torch.Tensor:
        """``latent [D]`` -> the residual added to the final pre-channel stream ``[d_model]``."""
        return self.gate * self.inject(latent)
