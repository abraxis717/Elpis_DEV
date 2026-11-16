"""The decoder-only transformer: embeddings, pre-norm blocks, rotary causal attention, a language-model head.

Block topology (identical attention in every configuration; only the channel transformation differs):

    x = x + Attention(RMSNorm(x))
    x = x + Channel(RMSNorm(x))        Channel = SwiGLU | CGCTransform | SwiGLU + gate * CGCTransform

Two execution modes compute the same function (tests/research/elpis_torch/test_model.py):

    batched   ``forward(tokens)``: all positions at once under a causal mask.
    cached    ``forward(tokens, cache)``: new positions only, attending to the keys and values a ``KVCache``
              holds for earlier positions. The cache is explicit, caller owned and never implicit module state.

A cache is valid only for the weights that wrote it. Every cache carries the ``weights_tag`` it was written
under, and a forward pass under any other tag refuses with ``StaleCacheError`` (the session relies on this).
"""
from __future__ import annotations

import ctypes
import hashlib
import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from .cgc import CGCTransform
from .config import ModelConfig

BASE_WEIGHTS = "base"


class ContextLengthError(RuntimeError):
    """The sequence would exceed ``max_seq_len``. Refused whole; nothing is truncated silently."""


class StaleCacheError(RuntimeError):
    """A cache written under other weights (or for another batch) was presented."""


@dataclass
class KVCache:
    """Per-layer rotated keys and values of the positions processed so far: ``[batch, kv_heads, length, head_dim]``."""
    batch_size: int
    capacity: int
    weights_tag: object = BASE_WEIGHTS
    length: int = 0
    keys: list = field(default_factory=list)
    values: list = field(default_factory=list)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (*self.keys, *self.values))

    def clear(self) -> None:
        self.length, self.keys, self.values = 0, [], []


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class Rotary(nn.Module):
    """Rotary position embedding: pairs ``(x_2i, x_2i+1)`` rotated by ``position * theta^(-2i/head_dim)``."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float, dtype: torch.dtype):
        super().__init__()
        frequency = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
        angle = torch.arange(max_seq_len, dtype=torch.float64)[:, None] * frequency[None, :]
        self.register_buffer("cos", angle.cos().to(dtype), persistent=False)
        self.register_buffer("sin", angle.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, start: int) -> torch.Tensor:
        """``x``: ``[batch, heads, T, head_dim]`` holding absolute positions ``start .. start + T - 1``."""
        length = x.shape[2]
        cos, sin = self.cos[start:start + length], self.sin[start:start + length]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    """Multi-head or grouped-query causal attention with rotary positions."""

    def __init__(self, c: ModelConfig, dtype: torch.dtype):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = c.n_heads, c.kv_heads, c.head_dim
        self.q = nn.Linear(c.d_model, self.heads * self.head_dim, bias=False, dtype=dtype)
        self.k = nn.Linear(c.d_model, self.kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.v = nn.Linear(c.d_model, self.kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.out = nn.Linear(self.heads * self.head_dim, c.d_model, bias=False, dtype=dtype)

    def forward(self, x, rotary: Rotary, start: int, past: tuple | None):
        batch, length, _ = x.shape
        q = rotary(self.q(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2), start)
        k = rotary(self.k(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2), start)
        v = self.v(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        if past is not None:
            k, v = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)
        present = (k, v)
        group = self.heads // self.kv_heads
        if group > 1:   # each key/value head serves `group` consecutive query heads
            k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        scores = (q @ k.transpose(-1, -2)) * (self.head_dim ** -0.5)       # [batch, heads, T, start + T]
        # Query at absolute position start + i sees keys at absolute positions <= start + i, never later ones.
        visible = (torch.arange(start + length, device=x.device)[None, :]
                   <= (start + torch.arange(length, device=x.device))[:, None])
        weights = torch.softmax(scores.masked_fill(~visible, float("-inf")), dim=-1)
        mixed = (weights @ v).transpose(1, 2).reshape(batch, length, self.heads * self.head_dim)
        return self.out(mixed), present


class SwiGLU(nn.Module):
    """``W_out (silu(W_gate x) * W_up x)``: the conventional baseline channel transformation."""

    def __init__(self, dim: int, hidden: int, dtype: torch.dtype):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False, dtype=dtype)
        self.up = nn.Linear(dim, hidden, bias=False, dtype=dtype)
        self.out = nn.Linear(hidden, dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c: ModelConfig, dtype: torch.dtype):
        super().__init__()
        self.attn_norm = RMSNorm(c.d_model, c.norm_eps, dtype)
        self.attn = CausalSelfAttention(c, dtype)
        self.channel_norm = RMSNorm(c.d_model, c.norm_eps, dtype)
        self.ffn = SwiGLU(c.d_model, c.ffn_width, dtype) if c.has_ffn else None
        self.cgc = CGCTransform(c.d_model, c.cgc_cells, c.cgc_units, bottleneck=c.cgc_bottleneck,
                                preact_bound=c.cgc_preact_bound, dtype=dtype) if c.has_cgc else None
        # C only. A scalar: at 0 the block is exactly the FFN block.
        self.cgc_gate = nn.Parameter(torch.zeros(1, dtype=dtype)) if c.channel == "ffn+cgc" else None

    def channel(self, x: torch.Tensor) -> torch.Tensor:
        if self.cgc is None:
            return self.ffn(x)
        if self.ffn is None:
            return self.cgc(x)
        return self.ffn(x) + self.cgc_gate * self.cgc(x)

    def forward(self, x, rotary: Rotary, start: int, past: tuple | None):
        attended, present = self.attn(self.attn_norm(x), rotary, start, past)
        x = x + attended
        return x + self.channel(self.channel_norm(x)), present


def _seeded_normal(seed: int, name: str, shape, std: float, dtype: torch.dtype) -> torch.Tensor:
    """A normal tensor that depends only on (seed, parameter name): never on construction or call order.

    Drawn in float64 and then cast, so a float32 and a float64 model of one seed hold the same weights up to
    rounding, and two configurations that share a parameter name and shape (the attention and FFN of A and C)
    hold identical values.
    """
    digest = hashlib.sha256(f"elpis-torch-r0:{seed}:{name}".encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") >> 1)
    return (torch.randn(tuple(shape), generator=generator, dtype=torch.float64) * std).to(dtype)


class ElpisTorchLM(nn.Module):
    """Token embeddings -> ``n_layers`` blocks -> RMSNorm -> vocabulary logits."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError("ElpisTorchLM expects a ModelConfig")
        self.config = c = config
        dtype = getattr(torch, c.dtype)
        self.embed = nn.Embedding(c.vocab_size, c.d_model, dtype=dtype)
        self.rotary = Rotary(c.head_dim, c.max_seq_len, c.rope_theta, dtype)
        self.blocks = nn.ModuleList(Block(c, dtype) for _ in range(c.n_layers))
        self.final_norm = RMSNorm(c.d_model, c.norm_eps, dtype)
        self.head = None if c.tie_embeddings else nn.Linear(c.d_model, c.vocab_size, bias=False, dtype=dtype)
        self.reset_parameters()

    # -- initialization ------------------------------------------------------------------------------------

    def _init_std(self, name: str) -> float | None:
        """The initial standard deviation of a weight, by role; ``None`` for constants (norms, gates)."""
        c = self.config
        residual = c.init_std / math.sqrt(2 * c.n_layers)     # every write into the residual stream
        if name.endswith(("attn.out.weight", "ffn.out.weight", "cgc.proj_out.weight")):
            return residual
        if name.endswith("cgc.proj_in.weight"):
            return c.d_model ** -0.5                          # unit-RMS input -> unit-variance bottleneck code
        if name.endswith("cgc.core.weight"):
            # Its input has unit RMS in both variants, so pre-activations start at std = cgc_init_gain.
            return c.cgc_init_gain * (c.cgc_bottleneck or c.d_model) ** -0.5
        if name.endswith(("norm.weight", "cgc_gate")):
            return None
        return c.init_std

    @torch.no_grad()
    def reset_parameters(self) -> None:
        c = self.config
        for name, parameter in self.named_parameters():
            std = self._init_std(name)
            if std is not None:
                parameter.copy_(_seeded_normal(c.seed, name, parameter.shape, std, parameter.dtype))
            elif name.endswith("cgc_gate"):
                parameter.fill_(c.cgc_gate_init)
            else:
                parameter.fill_(1.0)

    # -- execution -----------------------------------------------------------------------------------------

    def new_cache(self, batch_size: int, weights_tag: object = BASE_WEIGHTS) -> KVCache:
        return KVCache(batch_size=batch_size, capacity=self.config.max_seq_len, weights_tag=weights_tag)

    def forward(self, tokens: torch.Tensor, cache: KVCache | None = None, *,
                weights_tag: object = BASE_WEIGHTS) -> torch.Tensor:
        """Logits ``[batch, T, vocab]`` for ``tokens`` ``[batch, T]``.

        With a cache, ``tokens`` are the positions after the ``cache.length`` already processed, and the cache
        is extended in place only after every layer has succeeded.
        """
        c = self.config
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be an int64 tensor of shape [batch, T]")
        batch, length = tokens.shape
        if length < 1:
            raise ValueError("tokens must hold at least one position")
        if int(tokens.min()) < 0 or int(tokens.max()) >= c.vocab_size:
            raise ValueError(f"token ids must be in [0, {c.vocab_size})")
        start = 0
        if cache is not None:
            if cache.weights_tag != weights_tag:
                raise StaleCacheError("the cache was written under other weights; rebuild it")
            if cache.batch_size != batch or cache.capacity != c.max_seq_len:
                raise StaleCacheError("the cache does not belong to this model and batch")
            if cache.length and len(cache.keys) != c.n_layers:
                raise StaleCacheError("the cache does not hold this model's layers")
            start = cache.length
        if start + length > c.max_seq_len:
            raise ContextLengthError(f"{start} + {length} positions exceed max_seq_len {c.max_seq_len}")
        x = self.embed(tokens)
        keys, values = [], []
        for index, block in enumerate(self.blocks):
            past = (cache.keys[index], cache.values[index]) if start else None
            x, (k, v) = block(x, self.rotary, start, past)
            keys.append(k)
            values.append(v)
        x = self.final_norm(x)
        logits = F.linear(x, self.embed.weight) if self.head is None else self.head(x)
        if cache is not None:
            # Detached: a cache is inference state, never a path for gradients into earlier calls.
            cache.keys, cache.values = [k.detach() for k in keys], [v.detach() for v in values]
            cache.length = start + length
        return logits

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int, *, temperature: float = 0.0,
                 top_k: int | None = None, generator: torch.Generator | None = None,
                 use_cache: bool = True) -> torch.Tensor:
        """Autoregressive continuation: ``[batch, T] -> [batch, max_new_tokens]`` of new token ids.

        ``temperature == 0`` is greedy. ``use_cache=False`` recomputes the whole prefix every step: the slow
        reference the cached path is tested against.
        """
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if tokens.shape[1] + max_new_tokens > self.config.max_seq_len:
            raise ContextLengthError("prompt plus continuation exceeds max_seq_len")
        cache = self.new_cache(tokens.shape[0]) if use_cache else None
        context, new, feed = tokens, [], tokens
        for _ in range(max_new_tokens):
            logits = self(feed, cache)[:, -1] if use_cache else self(context)[:, -1]
            token = sample_next(logits, temperature=temperature, top_k=top_k, generator=generator)
            new.append(token)
            feed = token[:, None]
            context = torch.cat((context, feed), dim=1)
        return torch.stack(new, dim=1)


def sample_next(logits: torch.Tensor, *, temperature: float = 0.0, top_k: int | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """One token per row of ``logits`` ``[batch, vocab]``. Greedy at temperature 0 (ties: lowest id)."""
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("non-finite logits")
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    scaled = logits / temperature
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        threshold = scaled.topk(min(top_k, scaled.shape[-1]), dim=-1).values[:, -1:]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
    return torch.multinomial(torch.softmax(scaled, dim=-1), 1, generator=generator)[:, 0]


def next_token_loss(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Teacher-forced cross-entropy: ``logits[:, t]`` (from ``tokens[:, :t+1]``) against ``tokens[:, t+1]``.

    ``logits`` is ``[batch, T-1, vocab]`` for ``tokens`` ``[batch, T]``. ``mask`` ``[batch, T-1]`` selects the
    targets that count; the mean is over selected targets only.
    """
    targets = tokens[:, 1:]
    if logits.shape[:2] != targets.shape:
        raise ValueError("logits must cover exactly the positions that have a next token")
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    if mask is None:
        return losses.mean()
    weights = mask.reshape(-1).to(losses.dtype)
    if float(weights.sum()) == 0.0:
        raise ValueError("the target mask selects no position")
    return (losses * weights).sum() / weights.sum()


def parameter_digest(model: nn.Module) -> str:
    """SHA-256 over every parameter's name, shape, dtype and bytes: the model's weight identity."""
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(f"{name}|{tuple(parameter.shape)}|{parameter.dtype}|".encode())
        data = parameter.detach().cpu().contiguous()
        digest.update(ctypes.string_at(data.data_ptr(), data.numel() * data.element_size()))
    return digest.hexdigest()
