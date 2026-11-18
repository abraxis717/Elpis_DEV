"""R7: parametric lookup embeddings (PLE): rows addressed by hashed n-grams of the token ids, fused by a context gate.

ADMISSION CANDIDATE (research only; ELPIS_TORCH_R7). The mechanism follows the Engram paper and its demonstration
code (DeepSeek-AI, arXiv:2601.07372; github.com/deepseek-ai/Engram at fb7f84a, Apache-2.0). It is written here from
their description, not copied; every divergence is listed in docs/research/elpis_torch/DESIGN_R7.md.

    address    a[t, n, h] = offset[n, h] + (XOR_k  x'[t - k] * m[n, h, k]) mod P[n, h]     k = 0 .. n - 1
               x'[t - k] is the token k positions back, or SENTINEL (= the vocabulary size, never a token) before
               the stream start and before the latest boundary token (BOS) inside the n-gram
    rows       e[t] = concatenation of E[a[t, n, h]] over orders n and heads h              a gather, no arithmetic
    fusion     k = W_K e,  v = W_V e,  alpha = sigmoid(RMSNorm(x) . RMSNorm(k) / sqrt(d))   one scalar per position
               x <- x + alpha * v, before the configured block's mixer

Addresses are pure integer functions of token ids: they are computed once per call and reused. The only state is
the last ``max_order - 1`` token ids of each stream, which is bounded, volatile, reset with the session and never
serialized. Each (order, head) owns a prime-sized row range of one table; every multiplier is odd and bounded so
that no product of a token id overflows int64. Trainable tables use sparse gradients (lazy updates: a row not
addressed in a step is not touched by that step); an immutable store (``ple_store``) can replace them for inference.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
from dataclasses import dataclass

import torch
from torch import nn

from .config import ConfigError, ModelConfig
from .model import RMSNorm

ADDRESSING_VERSION = "elpis-ple-addressing/1"
ADDRESSINGS = ("hash", "random")
INT64_LIMIT = 1 << 63


@dataclass(frozen=True)
class PLEConfig:
    orders: tuple = (2, 3)        # n-gram orders; each >= 2 (the token embedding already is the unigram)
    heads: int = 4                # hash heads per order, each with its own prime-sized row range
    rows: int = 4096              # rows per head before rounding up to distinct primes
    head_dim: int = 16            # width of one row
    layer: int = 0                # the block whose input the rows are fused into (before its mixer)
    seed: int = 0                 # addressing seed (the multipliers); independent of the weights' seed
    boundary: int | None = None   # a token id that cuts the history (BOS: no n-gram crosses a document start)
    lr_multiplier: float = 5.0    # tables: lazy Adam at this multiple of the model's learning rate, no weight decay
    addressing: str = "hash"      # "hash" | "random" (CONTROL: content-independent rows, redrawn on every call)

    def __post_init__(self):
        orders = tuple(self.orders)
        if not orders or len(set(orders)) != len(orders) or any(type(n) is not int or not 2 <= n <= 8 for n in orders):
            raise ConfigError("orders must be distinct integers in 2 .. 8")
        object.__setattr__(self, "orders", tuple(sorted(orders)))
        for name in ("heads", "head_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ConfigError(f"{name} must be a positive integer")
        if type(self.rows) is not int or self.rows < 2:
            raise ConfigError("rows must be an integer >= 2")
        if type(self.layer) is not int or self.layer < 0:
            raise ConfigError("layer must be a non-negative integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ConfigError("seed must be a non-negative integer")
        if self.boundary is not None and (type(self.boundary) is not int or self.boundary < 0):
            raise ConfigError("boundary must be None or a token id")
        if not (isinstance(self.lr_multiplier, (int, float)) and math.isfinite(self.lr_multiplier)
                and self.lr_multiplier > 0):
            raise ConfigError("lr_multiplier must be a positive number")
        if self.addressing not in ADDRESSINGS:
            raise ConfigError(f"addressing must be one of {ADDRESSINGS}")

    @property
    def max_order(self) -> int:
        return self.orders[-1]

    @property
    def history(self) -> int:
        """Token ids kept per stream between calls (the bounded suffix that n-gram addressing needs)."""
        return self.max_order - 1

    @property
    def lookups(self) -> int:
        """Rows read per token."""
        return len(self.orders) * self.heads

    @property
    def width(self) -> int:
        return self.lookups * self.head_dim

    def replace(self, **changes) -> "PLEConfig":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(fields: dict) -> "PLEConfig":
        return PLEConfig(**{**fields, "orders": tuple(fields["orders"])})


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def table_sizes(config: PLEConfig) -> list[int]:
    """Distinct primes >= ``rows``, one per (order, head) in order-major order (Engram: a prime-sized range each)."""
    sizes, candidate = [], config.rows
    for _ in range(config.lookups):
        while not _is_prime(candidate):
            candidate += 1
        sizes.append(candidate)
        candidate += 1
    return sizes


def _multiplier(config: PLEConfig, order: int, head: int, k: int, bound: int) -> int:
    """An odd integer in [1, bound) from SHA-256 of a versioned description: stable on every platform."""
    text = f"{ADDRESSING_VERSION}|seed={config.seed}|order={order}|head={head}|k={k}"
    raw = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    return (raw % (bound // 2)) * 2 + 1


class NgramAddresser:
    """Global row ids ``[B, T, lookups]`` of every position's suffix n-grams; no parameters.

    ``hash`` addressing is a pure function of the token ids at and before each position (and of the stream's
    history suffix for the first positions of a call). ``random`` addressing is the CONTROL that removes content:
    each call draws rows uniformly in every head's range from a generator seeded by the configuration."""

    def __init__(self, config: PLEConfig, vocab_size: int):
        if type(vocab_size) is not int or vocab_size < 2:
            raise ConfigError("vocab_size must be an integer >= 2")
        if config.boundary is not None and config.boundary >= vocab_size:
            raise ConfigError("the boundary token must be a token id of the vocabulary")
        self.config, self.vocab_size = config, vocab_size
        self.sentinel = vocab_size                               # never a token id: "no token here"
        self.bound = INT64_LIMIT // (self.sentinel + 1)          # x' * m < (sentinel + 1) * bound <= 2^63
        if self.bound < 4:
            raise ConfigError("the vocabulary is too large for int64 multiplicative hashing")
        self.sizes = table_sizes(config)
        self.offsets = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self.total_rows = sum(self.sizes)
        self.multipliers = [[[_multiplier(config, n, h, k, self.bound) for k in range(n)]
                             for h in range(config.heads)] for n in config.orders]
        self._random = torch.Generator().manual_seed(config.seed)

    def digest(self) -> str:
        """SHA-256 of everything the addresses depend on (checkpoints and stores carry it; a mismatch is refused)."""
        c = self.config
        text = repr((ADDRESSING_VERSION, c.orders, c.heads, c.boundary, c.addressing, self.vocab_size, self.sizes,
                     self.multipliers))
        return hashlib.sha256(text.encode()).hexdigest()

    def new_history(self, batch: int, device=None) -> torch.Tensor:
        return torch.full((batch, self.config.history), self.sentinel, dtype=torch.long, device=device)

    def next_history(self, tokens: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """The last ``history`` raw token ids after ``tokens`` (boundaries are re-applied when addressing)."""
        return torch.cat((history, tokens), dim=1)[:, -self.config.history:]

    def grams(self, tokens: torch.Tensor, history: torch.Tensor) -> dict:
        """Per order ``n``: the n-gram ``[B, T, n]`` ending at each position (x'[t - k] at index k)."""
        self._check(tokens, history)
        c, keep = self.config, history.shape[1]
        stream = torch.cat((history, tokens), dim=1)
        length = tokens.shape[1]
        if c.boundary is not None:
            position = torch.arange(stream.shape[1], device=stream.device).expand_as(stream)
            latest = torch.where(stream == c.boundary, position, torch.full_like(position, -1)).cummax(dim=1).values
            latest = latest[:, keep:]                                    # the latest boundary at or before t
        out = {}
        for n in c.orders:
            parts = []
            for k in range(n):
                part = stream[:, keep - k:keep - k + length]
                if c.boundary is not None and k:
                    before = torch.arange(keep - k, keep - k + length, device=stream.device)[None] < latest
                    part = torch.where(before, torch.full_like(part, self.sentinel), part)
                parts.append(part)
            out[n] = torch.stack(parts, dim=-1)
        return out

    def rows_of(self, order_index: int, gram: torch.Tensor) -> torch.Tensor:
        """Global rows ``[..., heads]`` of the n-grams ``gram [..., n]`` of order ``orders[order_index]``."""
        c = self.config
        out = []
        for h in range(c.heads):
            m, slot = self.multipliers[order_index][h], order_index * c.heads + h
            mix = gram[..., 0] * m[0]
            for k in range(1, gram.shape[-1]):
                mix = torch.bitwise_xor(mix, gram[..., k] * m[k])
            out.append(self.offsets[slot] + torch.remainder(mix, self.sizes[slot]))
        return torch.stack(out, dim=-1)

    def addresses(self, tokens: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        c = self.config
        if c.addressing == "random":
            self._check(tokens, history)
            draws = [torch.randint(0, size, tokens.shape, generator=self._random) + offset
                     for size, offset in zip(self.sizes, self.offsets)]
            return torch.stack(draws, dim=-1).to(tokens.device)
        grams = self.grams(tokens, history)
        return torch.cat([self.rows_of(i, grams[n]) for i, n in enumerate(c.orders)], dim=-1)

    def _check(self, tokens: torch.Tensor, history: torch.Tensor) -> None:
        if tokens.dtype != torch.long or history.dtype != torch.long or tokens.ndim != 2 or history.ndim != 2:
            raise ValueError("tokens and history must be int64 tensors [batch, T] and [batch, history]")
        if history.shape != (tokens.shape[0], self.config.history):
            raise ValueError(f"the history must be [batch, {self.config.history}]")
        if tokens.numel() and (int(tokens.min()) < 0 or int(tokens.max()) >= self.vocab_size):
            raise ValueError(f"token ids must be in [0, {self.vocab_size})")
        if history.numel() and (int(history.min()) < 0 or int(history.max()) > self.sentinel):
            raise ValueError("the history holds an id outside the vocabulary and the sentinel")


class PLE(nn.Module):
    """The table, the gate and the value projection of one fusion point (``PLEConfig.layer``)."""

    def __init__(self, m: ModelConfig, p: PLEConfig, dtype):
        super().__init__()
        self.config = p
        self.addresser = NgramAddresser(p, m.vocab_size)
        self.table = nn.Embedding(self.addresser.total_rows, p.head_dim, sparse=True, dtype=dtype)
        self.key = nn.Linear(p.width, m.d_model, bias=False, dtype=dtype)
        self.value = nn.Linear(p.width, m.d_model, bias=False, dtype=dtype)
        self.query_norm = RMSNorm(m.d_model, m.norm_eps, dtype)
        self.key_norm = RMSNorm(m.d_model, m.norm_eps, dtype)
        self.enabled = True       # CONTROL: False removes the contribution exactly (the same model without PLE)
        self.store = None         # an immutable inference store (ple_store) that replaces the trainable rows
        self.report, self.last_gate = {}, None

    def attach_store(self, store) -> None:
        """Serve rows from an immutable store (inference only). Shape, dtype family and addressing must match."""
        if store is not None:
            if (store.rows, store.dim) != (self.addresser.total_rows, self.config.head_dim):
                raise ValueError("the store's table shape does not match this PLE")
            if store.addressing_digest != self.addresser.digest():
                raise ValueError("the store was built for different addressing")
        self.store = store

    def rows(self, addresses: torch.Tensor) -> torch.Tensor:
        if self.store is None:
            return self.table(addresses)
        if torch.is_grad_enabled() and self.training:
            raise RuntimeError("an immutable PLE store cannot be trained; detach it (attach_store(None)) to train")
        return self.store.read(addresses).to(device=self.value.weight.device, dtype=self.value.weight.dtype)

    def forward(self, x: torch.Tensor, addresses: torch.Tensor) -> torch.Tensor:
        e = self.rows(addresses).flatten(-2)                                       # [B, T, lookups * head_dim]
        score = (self.query_norm(x) * self.key_norm(self.key(e))).sum(-1, keepdim=True) / math.sqrt(x.shape[-1])
        gate = torch.sigmoid(score)
        self.last_gate = gate.detach()                  # this call's gates only (overwritten; for diagnostics)
        with torch.no_grad():
            g = gate.detach().float()
            self.report = {"gate_mean": float(g.mean()), "gate_min": float(g.min()), "gate_max": float(g.max()),
                           "gate_below_0.05": float((g < 0.05).float().mean()),
                           "gate_above_0.95": float((g > 0.95).float().mean()), "positions": g.numel()}
        return gate * self.value(e)

    def lookup_bytes_per_token(self) -> int:
        """Logical bytes of the rows one token reads (the store's precision when a store is attached)."""
        width = self.store.element_size if self.store is not None else self.table.weight.element_size()
        return self.config.lookups * self.config.head_dim * width


def ple_parameters(p: PLEConfig, d_model: int, vocab_size: int) -> dict:
    """Parameter counts: the table (resident; ``lookups`` rows read per token) and the fusion (always active)."""
    rows = sum(table_sizes(p))
    fusion = 2 * p.width * d_model + 2 * d_model
    return {"table": rows * p.head_dim, "table_rows": rows, "fusion": fusion,
            "active_per_token": p.lookups * p.head_dim + fusion, "vocab_size": vocab_size}
