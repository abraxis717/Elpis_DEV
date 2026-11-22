"""R13 P5: the competence study's arms, computed on one cached observation. RESEARCH_ONLY.

ARMS (all but NATIVE are trained, with final-answer-only supervision, from the same seeded initialization):
    NATIVE              the frozen backbone's own prediction at the query (the bypass logits); not trained
    GEODOME             the substrate on a physical region (a Geodome-validated plan); deployable evidence
    EQUAL_BUDGET        the same ingress, readout and injection; the graph and its rounds replaced by ``DirectReader``
                        (one attention read of the evidence rows by the query row, then an MLP); parameter-matched
    COMPUTE_MATCHED     the same kernels and rounds on a fully mixed random graph of the region's degree sequence
    FLAT_RECURRENT      the same kernels and rounds on a hub that reads the evidence leaves (no topology, no ports)
    RELABEL_ISOMORPHIC  the same kernels and rounds on the region under a declared vertex permutation
    REWIRED             the same kernels and rounds on the region rewired by degree-preserving swaps
    DIAGNOSTIC_ORACLE   GEODOME with the episode's FACT blocks as evidence (gold; DIAGNOSTIC ONLY, never deployable)

Every graph arm uses the bridge's own pure computation: ``CognitiveSession.compute`` (ingress, synchronous rounds,
readout) and ``CognitiveSession.refine`` (injection through the backbone's actual final channel over the chunk's own
rows). A ``View`` is one episode as the frozen backbone saw it. It is computed once per seed from a reset session,
shared by every arm, and carries no graph: the observation, the deployable evidence (the backbone's own top
fine-ranked candidates, exactly reread) and the oracle evidence. An arm starts every episode from rows of zero, so no
state and no autograd graph crosses episodes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..model import _seeded_normal
from .activation import run_synchronous
from .binding import AdmittedEvidence, CognitiveSession, Observation
from .kernels import CognitiveConfig, CognitiveKernels

ARMS = ("NATIVE", "GEODOME", "EQUAL_BUDGET", "COMPUTE_MATCHED", "FLAT_RECURRENT", "RELABEL_ISOMORPHIC", "REWIRED",
        "DIAGNOSTIC_ORACLE")
TRAINABLE = ARMS[1:]
DEPLOYABLE = tuple(a for a in ARMS if a != "DIAGNOSTIC_ORACLE")
CORE = ("message_in", "message_out", "port_out", "port_in", "update_gate", "update_cand")


@dataclass(frozen=True)
class View:
    """One episode as the frozen backbone saw it (no graph)."""
    observation: Observation
    deployable: AdmittedEvidence | None      # the backbone's own fine-ranked candidates, exact reread
    oracle: AdmittedEvidence                 # the episode's FACT blocks, seeded order (DIAGNOSTIC ONLY)
    answer: int
    hops: int
    candidates: tuple                        # the deployable candidates' block numbers, rank order
    complete: bool                           # every required fact is among the deployable candidates


class DirectReader(nn.Module):
    """EQUAL_BUDGET's core: the query row reads the evidence rows once (scaled dot-product attention; values are the
    evidence rows), then an MLP of [query; read]. No graph, no rounds."""

    def __init__(self, width: int, hidden: int, dtype, seed: int):
        super().__init__()
        self.q = nn.Linear(width, width, dtype=dtype)
        self.k = nn.Linear(width, width, dtype=dtype)
        self.mlp_in = nn.Linear(2 * width, hidden, dtype=dtype)
        self.mlp_out = nn.Linear(hidden, width, dtype=dtype)
        with torch.no_grad():
            for name, p in self.named_parameters():
                p.copy_(torch.zeros_like(p) if name.endswith(".bias") else
                        _seeded_normal(seed, "r13p5-reader:" + name, p.shape, p.shape[1] ** -0.5, p.dtype))

    def forward(self, query: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        """``query [1, D]``, ``evidence [n, D]`` -> ``[1, D]``."""
        if evidence.shape[0]:
            scores = self.q(query) @ self.k(evidence).T * query.shape[-1] ** -0.5
            read = torch.softmax(scores, dim=-1) @ evidence
        else:
            read = torch.zeros_like(query)
        return torch.tanh(self.mlp_out(F.gelu(self.mlp_in(torch.cat((query, read), dim=-1)))))


def core_parameters(width: int, hidden: int) -> int:
    """The kernels' message, port and update parameters (what EQUAL_BUDGET replaces)."""
    return (width * hidden + hidden) + (hidden * width + width) + 2 * 6 * hidden + 2 * (2 * width * width + width)


def reader_hidden(width: int, hidden: int) -> int:
    """The DirectReader width whose parameters come closest to ``core_parameters``."""
    fixed = 2 * (width * width + width) + width                       # q, k, and mlp_out's bias
    return max(1, round((core_parameters(width, hidden) - fixed) / (2 * width + 1 + width)))


class Arm(nn.Module):
    """One arm's trainable parameters and its computation on a ``View``."""

    def __init__(self, arm: str, model, config: CognitiveConfig, plans: dict, *, reader_width: int | None = None):
        super().__init__()
        if arm not in TRAINABLE:
            raise ValueError(f"unknown trainable arm {arm!r}")
        self.arm = arm
        self.kernels = CognitiveKernels(config, model.config.d_model, model.encoder.width,
                                        dtype=model.embed.weight.dtype)
        self.reader = DirectReader(config.width, reader_width or reader_hidden(config.width, config.hidden),
                                   model.embed.weight.dtype, config.seed) if arm == "EQUAL_BUDGET" else None
        self.session = CognitiveSession(model, self.kernels)         # compute / refine only; it never observes
        self.plans = plans                                            # variant -> region -> evidence count -> plan

    def trainable(self) -> list:
        """Every parameter the arm trains. EQUAL_BUDGET has no message, port or update kernels: they are excluded."""
        if self.arm != "EQUAL_BUDGET":
            return list(self.kernels.parameters())
        kept = [p for name, p in self.kernels.named_parameters() if not name.startswith(CORE)]
        return kept + list(self.reader.parameters())

    def evidence_for(self, view: View, source):
        if source == "deployable":
            return view.deployable
        if source == "oracle":
            return view.oracle
        return source                                                 # an explicit AdmittedEvidence, or None

    def forward(self, view: View, region: int, *, source="deployable", variant: str = "main") -> torch.Tensor:
        """The refined final-position logits ``[1, vocab]`` for one episode on one region."""
        evidence = self.evidence_for(view, source)
        count = 0 if evidence is None else len(evidence.addresses)
        if self.arm == "EQUAL_BUDGET":
            k = self.kernels
            query = k.query(view.observation.pre)
            rows = k.evidence(evidence.encoded) if count else query.new_zeros(0, query.shape[-1])
            latent = k.readout(self.reader(query, rows))
            return self.session.refine(latent, view.observation)
        compiled = self.plans[variant][region][count]
        rows = torch.zeros(compiled.incoming.numel(), self.kernels.config.width, dtype=self.kernels.gate.dtype)
        _, latent = self.session.compute(compiled, rows, view.observation, evidence)
        return self.session.refine(latent, view.observation)

    def erased(self, view: View, region: int) -> torch.Tensor:
        """Intermediate-state erasure: half the rounds, then every row but the query and evidence rows set to zero,
        then the other half (no ingress again)."""
        evidence = view.deployable
        count = 0 if evidence is None else len(evidence.addresses)
        half = self.plans["half"][region][count]
        rows = torch.zeros(half.incoming.numel(), self.kernels.config.width, dtype=self.kernels.gate.dtype)
        result, _ = self.session.compute(half, rows, view.observation, evidence)
        keep = torch.zeros(half.incoming.numel(), 1, dtype=torch.bool)
        keep[torch.cat((half.query, half.evidence))] = True
        second = run_synchronous(torch.where(keep, result.rows, torch.zeros_like(result.rows)), half, self.kernels)
        latent = self.kernels.readout(second.rows.index_select(0, half.readout))
        return self.session.refine(latent, view.observation)

    def disabled(self, view: View) -> torch.Tensor:
        """The injection disabled: the backbone's own tail on the observation's rows (the bypass, bit for bit)."""
        return self.session.final_logits(view.observation.rows)


def parameter_count(params) -> int:
    return sum(p.numel() for p in params)


def gradient_norms(arm: Arm) -> dict:
    """The L2 norm of the gradient of each parameter group (0.0 where no gradient arrived)."""
    groups = {"ingress": ("query_in", "evidence_query", "evidence_in"), "message": ("message_in", "message_out",
                                                                                 "port_out", "port_in"),
              "update": ("update_gate", "update_cand"), "readout": ("readout_proj",), "inject": ("inject",),
              "gate": ("gate",), "reader": ()}
    out = {}
    named = dict(arm.kernels.named_parameters())
    if arm.reader is not None:
        named.update({"reader." + n: p for n, p in arm.reader.named_parameters()})
    for group, prefixes in groups.items():
        squares = 0.0
        for name, p in named.items():
            hit = name.startswith("reader.") if group == "reader" else name.startswith(prefixes)
            if hit and p.grad is not None:
                squares += float(p.grad.detach().pow(2).sum())
        out[group] = math.sqrt(squares)
    return out
