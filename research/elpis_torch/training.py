"""Offline training and evaluation. The only code in this package that changes model weights in place.

Training is an OFFLINE operation, separate from any inference session: it mutates the parameters of the model
object it is handed, in memory, and writes nothing. A persistent artifact exists only when the caller invokes
``save_checkpoint`` with an explicit path (docs/ELPIS_MISSION.md, Amendment 3: "An offline process that
produces persistent model artifacts needs its own explicit authorization"). Sessions never call it.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from .config import ModelConfig
from .data import DOUBLE, HELDOUT, KIND_NAMES, SINGLE, Batch
from .model import ElpisTorchLM, next_token_loss, parameter_digest

CHECKPOINT_FORMAT = "elpis-torch-r0.checkpoint.v1"
GROUPS = ("cgc.core", "cgc.proj", "cgc_gate", "ffn", "attn", "norm", "embed")


def group_of(name: str) -> str:
    """The accounting group of a parameter name."""
    if "cgc.core" in name:
        return "cgc.core"
    if "cgc.proj" in name:
        return "cgc.proj"
    if "cgc_gate" in name:
        return "cgc_gate"
    if ".ffn." in name:
        return "ffn"
    if ".attn." in name:
        return "attn"
    if "norm" in name:
        return "norm"
    return "embed"


@dataclass(frozen=True)
class TrainConfig:
    steps: int
    batch_size: int = 64
    lr: float = 3e-3
    min_lr_fraction: float = 0.1
    warmup_steps: int = 50
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0
    eval_every: int = 0          # 0: evaluate only at the end

    def __post_init__(self):
        if self.steps < 1 or self.batch_size < 1 or self.warmup_steps < 0 or self.eval_every < 0:
            raise ValueError("steps and batch_size must be positive; warmup_steps and eval_every non-negative")
        for name in ("lr", "grad_clip"):
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) > 0):
                raise ValueError(f"{name} must be positive and finite")
        if not (0 <= self.min_lr_fraction <= 1) or self.weight_decay < 0:
            raise ValueError("min_lr_fraction must be in [0, 1] and weight_decay non-negative")

    def lr_at(self, step: int) -> float:
        """Linear warmup, then cosine decay to ``min_lr_fraction * lr``."""
        if step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.steps - self.warmup_steps)
        floor = self.min_lr_fraction
        return self.lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


@dataclass
class TrainReport:
    status: str = "COMPLETED"                      # COMPLETED | DIVERGED
    steps_done: int = 0
    tokens_seen: int = 0
    first_loss: float = math.nan
    final_loss: float = math.nan                   # mean training loss over the last 20 steps
    history: list = field(default_factory=list)    # evaluation records
    grad_norm: dict = field(default_factory=dict)  # per group: min / max of the pre-clip gradient norm
    update_norm: dict = field(default_factory=dict)  # per group: ||theta_final - theta_initial|| / ||theta_initial||
    anomalies: list = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@torch.no_grad()
def evaluate(model: ElpisTorchLM, batch: Batch, *, chunk: int = 256) -> dict:
    """Teacher-forced loss over all targets, and loss / accuracy at the state targets of each kind."""
    model.eval()
    total, count = 0.0, 0
    per = {kind: [0.0, 0, 0] for kind in (SINGLE, DOUBLE, HELDOUT)}   # loss sum, correct, targets
    for start in range(0, len(batch), chunk):
        tokens, kind = batch.tokens[start:start + chunk], batch.kind[start:start + chunk, 1:]
        logits = model(tokens[:, :-1])
        log_probs = torch.log_softmax(logits.double(), dim=-1)
        nll = -log_probs.gather(-1, tokens[:, 1:, None])[..., 0]
        total, count = total + float(nll.sum()), count + nll.numel()
        correct = logits.argmax(dim=-1) == tokens[:, 1:]
        for k, acc in per.items():
            mask = kind == k
            acc[0] += float(nll[mask].sum())
            acc[1] += int(correct[mask].sum())
            acc[2] += int(mask.sum())
    out = {"loss": total / count, "targets": count}
    for k, (loss_sum, right, n) in per.items():
        if n:
            out[KIND_NAMES[k]] = {"loss": loss_sum / n, "accuracy": right / n, "targets": n}
    return out


@torch.no_grad()
def rollout(model: ElpisTorchLM, batch: Batch) -> dict:
    """Free-running generation of every state through the KV cache.

    Operations and '=' are read from the reference sequence; every state after '=' is the model's own greedy
    token, fed back as context. A wrong state therefore corrupts the context of everything after it. A
    trajectory counts as exact only when all its states are right.
    """
    model.eval()
    tokens, is_state = batch.tokens, batch.kind != 0
    cache = model.new_cache(tokens.shape[0])
    current = tokens[:, :1]
    right = torch.zeros(tokens.shape, dtype=torch.bool)
    for position in range(tokens.shape[1] - 1):
        predicted = model(current, cache)[:, -1].argmax(dim=-1)
        target_is_state = is_state[:, position + 1]
        right[:, position + 1] = (predicted == tokens[:, position + 1]) & target_is_state
        current = torch.where(target_is_state, predicted, tokens[:, position + 1])[:, None]
    states = int(is_state.sum())
    exact = (right | ~is_state).all(dim=1)
    return {"state_accuracy": int(right.sum()) / states, "exact_trajectories": float(exact.double().mean()),
            "states": states, "trajectories": tokens.shape[0]}


def train(model: ElpisTorchLM, data: Batch, config: TrainConfig, *, val: Batch | None = None) -> TrainReport:
    """Teacher-forced next-token training with AdamW. Deterministic for a fixed seed and thread count.

    A non-finite loss or gradient stops training before the step is applied (status ``DIVERGED``): the
    parameters keep their last finite values and the anomaly is recorded, never hidden.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    named = dict(model.named_parameters())
    initial = {name: p.detach().clone() for name, p in named.items()}
    decay = [p for p in named.values() if p.ndim >= 2]
    plain = [p for p in named.values() if p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": config.weight_decay},
                                   {"params": plain, "weight_decay": 0.0}], lr=config.lr, betas=(0.9, 0.95))
    sampler = torch.Generator().manual_seed(config.seed)
    report = TrainReport()
    recent: list[float] = []
    started = time.perf_counter()
    for step in range(config.steps):
        model.train()
        index = torch.randint(len(data), (config.batch_size,), generator=sampler)
        tokens = data.tokens[index]
        loss = next_token_loss(model(tokens[:, :-1]), tokens)
        optimizer.zero_grad(set_to_none=True)
        if not math.isfinite(float(loss.detach())):
            report.status = "DIVERGED"
            report.anomalies.append(f"step {step}: non-finite training loss")
            break
        loss.backward()
        norms: dict = {}
        for name, p in named.items():
            norms[group_of(name)] = norms.get(group_of(name), 0.0) + float(p.grad.pow(2).sum())
        if not all(math.isfinite(v) for v in norms.values()):
            report.status = "DIVERGED"
            report.anomalies.append(f"step {step}: non-finite gradient")
            break
        for group, squared in norms.items():
            low, high = report.grad_norm.get(group, (math.inf, 0.0))
            report.grad_norm[group] = (min(low, math.sqrt(squared)), max(high, math.sqrt(squared)))
        torch.nn.utils.clip_grad_norm_(named.values(), config.grad_clip)
        for group in optimizer.param_groups:
            group["lr"] = config.lr_at(step)
        optimizer.step()
        value = float(loss.detach())
        if step == 0:
            report.first_loss = value
        recent = (recent + [value])[-20:]
        report.steps_done = step + 1
        report.tokens_seen += tokens.numel()
        if val is not None and config.eval_every and (step + 1) % config.eval_every == 0:
            report.history.append({"step": step + 1, "train_loss": sum(recent) / len(recent),
                                   "val": evaluate(model, val)})
    report.seconds = time.perf_counter() - started
    report.final_loss = sum(recent) / len(recent) if recent else math.nan
    if val is not None and (not report.history or report.history[-1]["step"] != report.steps_done):
        report.history.append({"step": report.steps_done, "train_loss": report.final_loss,
                               "val": evaluate(model, val)})
    moved: dict = {}
    for name, p in named.items():
        delta, base = moved.setdefault(group_of(name), [0.0, 0.0])
        moved[group_of(name)] = [delta + float((p.detach() - initial[name]).pow(2).sum()),
                                 base + float(initial[name].pow(2).sum())]
    report.update_norm = {g: math.sqrt(d) / math.sqrt(b) if b else math.sqrt(d) for g, (d, b) in moved.items()}
    report.grad_norm = {g: {"min": lo, "max": hi} for g, (lo, hi) in report.grad_norm.items()}
    model.eval()
    return report


def freeze(model: ElpisTorchLM) -> ElpisTorchLM:
    """Inference form: evaluation mode, no parameter requires a gradient, no training gradient is left behind."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return model


def save_checkpoint(model: ElpisTorchLM, path) -> str:
    """EXPLICIT OFFLINE ARTIFACT PRODUCTION: write the model's configuration and weights to ``path``.

    Never called by a session or by generation. Returns the parameter digest of what was written.
    """
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({"format": CHECKPOINT_FORMAT, "config": model.config.to_dict(), "state_dict": state}, path)
    return parameter_digest(model)


def load_checkpoint(path) -> ElpisTorchLM:
    """Read a checkpoint (read-only: tensors only, no pickled code) into a frozen model."""
    record = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(record, dict) or record.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("not an elpis-torch-r0 checkpoint")
    model = ElpisTorchLM(ModelConfig.from_dict(record["config"]))
    model.load_state_dict(record["state_dict"], strict=True)
    return freeze(model)
