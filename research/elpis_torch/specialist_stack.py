"""R11 Stage C: frozen-weight transfer into the ACTUAL ERSA stack. RESEARCH_ONLY; no inner loop, no weight update
at inference.

The episodes are R11's R97W-form tasks (``specialist_tasks.teacher``), presented IN CONTEXT: a sequence holds ``k``
support tokens ``[x, y, 0]`` and then 48 query tokens ``[x, 0, 1]``; the model predicts ``y`` at the queries. The
stack is the repository's: the ``ErsaBlock`` modules of an ``ErsaLM`` (built and initialized by ``ErsaLM`` itself;
its embedding and head are not used) between a linear input map (9 -> d) and a linear readout (d -> 3) -- a
disclosed harness difference from the byte language model. ERSA owns every bit of recurrent state; the expert
channel (FFN + micro specialist) is an operator in each block, after that block's ERSA (``CGC_BOUNDING_R11.md`` §6).
"""
from __future__ import annotations

import dataclasses
import math
import statistics

import torch
from torch import nn

from . import specialist_tasks as st
from .cgc_specialist import micro_sparse
from .config import ModelConfig
from .ersa_model import ErsaConfig, ErsaLM
from .model import RMSNorm
from .sparse_channel import SparseConfig

TOKEN = st.INPUTS + st.OUTPUTS + 1          # [x (5), y (3), is_query (1)]
K_GRID = (0, 1, 2, 4, 8, 16, 32)
STAGE_C = {"d_model": 64, "layers": 2, "ersa_heads": 4, "key_dim": 16, "value_dim": 16, "conv_kernel": 4,
           "ffn_hidden": 86, "scan_block": 16, "queries": 48, "support_max": 32}
ARMS_C = {"G0": {"kind": "ffn", "gate_mode": "learned"},
          "G1": {"kind": "ffn+cgc", "gate_mode": "learned"},
          "G2": {"kind": "ffn+cgc", "gate_mode": "fixed"},
          "G3": {"kind": "ffn+mlp", "gate_mode": "fixed"}}


class RegressionStack(nn.Module):
    """``in_proj`` -> the ErsaLM's blocks (ERSA mixer, then the expert channel) -> RMSNorm -> ``readout``."""

    def __init__(self, arm: str, profile: str, seed: int, experts: int = 1, namespace: str = "R11C"):
        super().__init__()
        spec = ARMS_C[arm]
        c = STAGE_C
        m = ModelConfig(vocab_size=2, d_model=c["d_model"], n_layers=c["layers"], n_heads=4, max_seq_len=8,
                        channel="ffn", ffn_hidden=c["ffn_hidden"], cgc_bottleneck=5, cgc_channels=3, cgc_units=8,
                        dtype="float32", seed=seed)
        base = SparseConfig(experts=experts, top_k=1, layers=tuple(range(c["layers"])), ffn_hidden=c["ffn_hidden"],
                            gate_init=0.1)
        sparse = base.replace(kind="ffn") if spec["kind"] == "ffn" else micro_sparse(
            base, profile=profile, gate_mode=spec["gate_mode"], kind=spec["kind"])
        lm = ErsaLM(ErsaConfig(m, ersa_heads=c["ersa_heads"], key_dim=c["key_dim"], value_dim=c["value_dim"],
                               conv_kernel=c["conv_kernel"], chunk=128, ersa_kernel="chunked",
                               scan_block=c["scan_block"], sparse=sparse))
        self.blocks = lm.blocks                       # the repository's blocks, with ErsaLM's own initialization
        for index, block in enumerate(self.blocks):   # the A1 branch rule of Stage A/B, per expert
            for e, expert in enumerate(block.sparse.experts):
                st.init_branch_like_stage_a(expert, seed * 100 + index * 10 + e, namespace)
                if expert.cgc is not None:
                    expert.cgc.float()
                if expert.mlp is not None:
                    expert.mlp.float()
        g = torch.Generator().manual_seed(st.seed_of(namespace, "io", seed))
        self.in_proj = nn.Linear(TOKEN, c["d_model"], bias=False)
        self.readout = nn.Linear(c["d_model"], st.OUTPUTS, bias=True)
        with torch.no_grad():
            self.in_proj.weight.copy_(torch.randn(c["d_model"], TOKEN, generator=g) / math.sqrt(TOKEN))
            self.readout.weight.copy_(torch.randn(st.OUTPUTS, c["d_model"], generator=g) / math.sqrt(c["d_model"]))
            self.readout.bias.zero_()
        self.final_norm = RMSNorm(c["d_model"], m.norm_eps, torch.float32)
        self.arm = arm

    def new_state(self, batch: int):
        return [b.ersa.new_state(batch) for b in self.blocks], [b.ersa.new_tail(batch) for b in self.blocks]

    def forward(self, tokens: torch.Tensor, state, *, ablate_blocks=(), reset_after: int | None = None):
        """``tokens [B, T, 9]`` -> predictions ``[B, T, 3]``. ``ablate_blocks``: blocks whose specialist branch is
        removed (frozen ablation). ``reset_after``: ERSA states and tails are zeroed after that many tokens (the
        support prefix), so the queries see no in-context information."""
        cells, tails = state
        if reset_after is not None:
            first = self.forward(tokens[:, :reset_after], state, ablate_blocks=ablate_blocks) if reset_after else None
            cells, tails = self.new_state(tokens.shape[0])
            rest = self.forward(tokens[:, reset_after:], (cells, tails), ablate_blocks=ablate_blocks)
            return rest if first is None else torch.cat([first, rest], dim=1)
        x = self.in_proj(tokens)
        for index, block in enumerate(self.blocks):
            u = block.mix_norm(x)
            mixed, cells[index], tails[index] = block.ersa(u, cells[index], tails[index])
            pre = x + mixed
            h = block.channel_norm(pre)
            if index in ablate_blocks:
                x = pre + _channel_without_specialist(block.sparse, h)
            else:
                x = pre + block.channel(h)
        return self.readout(self.final_norm(x))


def _channel_without_specialist(channel, h):
    """The routed channel with every expert's specialist branch removed (FFN only), same routing."""
    flat = h.reshape(-1, h.shape[-1])
    _, _, index, weight = channel.router(flat)
    out = torch.zeros_like(flat)
    for e, expert in enumerate(channel.experts):
        coefficient = ((index == e).to(flat.dtype) * weight.to(flat.dtype)).sum(-1, keepdim=True)
        if bool(coefficient.any()):
            out = out + coefficient * expert.ffn(flat)
    return out.reshape(h.shape)


def episode_tokens(namespace: str, family: str, seed: int, split: str, index: int, k: int):
    """``k`` support tokens then the 48 query tokens of one episode; targets at the queries."""
    e = st.teacher(namespace, family, seed, split, index)
    sx, sy = e.support[0][:k].float(), e.support[1][:k].float()
    qx, qy = e.query[0].float(), e.query[1].float()
    support = torch.cat([sx, sy, torch.zeros(k, 1)], dim=-1)
    query = torch.cat([qx, torch.zeros(qx.shape[0], st.OUTPUTS), torch.ones(qx.shape[0], 1)], dim=-1)
    return torch.cat([support, query]), qy


def batch_tokens(namespace, families, seed, split, start, size, k):
    rows, targets = [], []
    for b in range(size):
        family = families[b % len(families)]
        tokens, y = episode_tokens(namespace, family, seed, split, start + b, k)
        rows.append(tokens)
        targets.append(y)
    return torch.stack(rows), torch.stack(targets)


def query_mse(model, tokens, targets, **kw) -> torch.Tensor:
    q = targets.shape[1]
    pred = model(tokens, model.new_state(tokens.shape[0]), **kw)[:, -q:]
    return (pred - targets).pow(2).mean(dim=(1, 2))


def train_stack(model, namespace, families, seed, plan, *, dev_families=None) -> dict:
    """End-to-end training (no inner loop anywhere); k drawn per step from the K grid; per-arm checkpoint selection
    on DEV episodes at k = 32 (the same rule for every arm)."""
    opt = torch.optim.Adam(model.parameters(), lr=plan["adam_lr"])
    steps = plan["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / steps)))
    rng = torch.Generator().manual_seed(st.seed_of(namespace, "k-schedule", seed))
    best = {"score": math.inf, "step": None, "state": None}
    trace = []
    for step in range(steps + 1):
        if step % plan["select_every"] == 0 or step == steps:
            with torch.no_grad():
                scores = []
                for family in (dev_families or families):
                    tokens, targets = batch_tokens(namespace, [family], seed, "DEV", 0, plan["dev_episodes"], 32)
                    scores.append(float(query_mse(model, tokens, targets).mean()))
                score = statistics.fmean(scores)
            trace.append({"step": step, "dev_k32": score})
            if math.isfinite(score) and score < best["score"]:
                best.update(score=score, step=step, state={k: v.detach().clone() for k, v in model.state_dict().items()})
        if step == steps:
            break
        k = K_GRID[int(torch.randint(len(K_GRID), (1,), generator=rng))]
        tokens, targets = batch_tokens(namespace, families, seed, "TRAIN", step * plan["batch"], plan["batch"], k)
        loss = query_mse(model, tokens, targets).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.load_state_dict(best["state"])
    return {"selected_step": best["step"], "selected_dev_score": best["score"], "trace": trace}


@torch.no_grad()
def evaluate_stack(model, namespace, family, seed, episodes) -> dict:
    """Per test episode: query MSE at every k; at k = 32 also the frozen specialist ablations (all blocks, block 0
    only, block 1 only) and the ERSA reset after the support prefix."""
    rows = []
    for index in range(episodes):
        row = {"episode": index, "curve": {}}
        for k in K_GRID:
            tokens, targets = batch_tokens(namespace, [family], seed, "TEST", index, 1, k)
            row["curve"][str(k)] = [float(query_mse(model, tokens, targets)[0])]
        tokens, targets = batch_tokens(namespace, [family], seed, "TEST", index, 1, 32)
        row["ablate_all"] = float(query_mse(model, tokens, targets, ablate_blocks=(0, 1))[0])
        row["ablate_block0"] = float(query_mse(model, tokens, targets, ablate_blocks=(0,))[0])
        row["ablate_block1"] = float(query_mse(model, tokens, targets, ablate_blocks=(1,))[0])
        row["ersa_reset_after_support"] = float(query_mse(model, tokens, targets, reset_after=32)[0])
        row["null"] = float(targets.pow(2).mean())
        rows.append(row)
    return rows


@torch.no_grad()
def routing_purity(model, namespace, seed, episodes) -> dict:
    """Two-expert banks trained on mixed families: the share of each family's query tokens routed to that family's
    majority expert, per block (0.5 = no family separation)."""
    out = {}
    for b, block in enumerate(model.blocks):
        counts = {}
        for family in st.FAMILIES:
            tally = torch.zeros(len(block.sparse.experts))
            for index in range(episodes):
                tokens, targets = batch_tokens(namespace, [family], seed, "TEST", index, 1, 32)
                captured = {}
                handle = block.sparse.register_forward_hook(lambda mod, inp, outp: captured.update(
                    index=mod.report.get("stats", {}).get("counts")))
                model(tokens, model.new_state(1))
                handle.remove()
                tally += torch.tensor(captured["index"], dtype=torch.float)
            counts[family] = tally
        majority = {f: int(torch.argmax(c)) for f, c in counts.items()}
        purity = {f: float(c[majority[f]] / c.sum()) for f, c in counts.items()}
        out[str(b)] = {"counts": {f: c.tolist() for f, c in counts.items()}, "majority_expert": majority,
                       "purity": purity, "families_separated": majority["smooth"] != majority["conditional"]}
    return out
