"""R4 comparator: a minimal Spark-style hybrid attention decoder, for context-COST comparisons only.

Written here from the published configuration of XHToken Spark-X2.5-4B / X2.6-4B (``config.json``, model card;
Apache-2.0). No Spark source code, weights or data were used or are needed. It is untrained and claims nothing
about Spark's quality.

What it reproduces from the published configuration:

* ``layer_types``: three sliding-window layers per full-attention layer.
* sliding layers: causal attention over the last ``window`` tokens (512 in Spark), full rotary positions
  (theta 1e4 in Spark); full layers: causal attention over the whole history, PARTIAL rotary (a fraction of each
  head, 0.25 in Spark, theta 5e6).
* grouped-query attention (16 query / 4 KV heads, head dim 256 in Spark) and a head-wise sigmoid gate on the
  attention output (``headwise_attn_output_gate``).
* a GELU MLP, RMSNorm, tied embeddings.

Cache: each sliding layer keeps a ring of ``window`` rotated keys and values; each full layer keeps EVERY key and
value of the stream in a buffer preallocated to ``capacity`` tokens. That buffer is the memory the comparison is
about: it grows linearly with the context and cannot scroll beyond ``capacity`` (a refusal, not a silent drop).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

SPARK_4B = dict(layers=36, sliding=27, full=9, window=512, query_heads=16, kv_heads=4, head_dim=256, hidden=2560,
                vocab=131072, horizon=1 << 20, partial_rotary=0.25)


def spark_kv_bytes(*, tokens: int, layers_full: int = 9, layers_sliding: int = 27, kv_heads: int = 4,
                   head_dim: int = 256, window: int = 512, scalar_bytes: int = 2) -> dict:
    """KV-cache bytes of a hybrid sliding/full stack at ``tokens`` of context (keys and values, one stream)."""
    per_token = 2 * kv_heads * head_dim * scalar_bytes
    full = layers_full * per_token * tokens
    sliding = layers_sliding * per_token * min(window, tokens)
    return {"full_layers": full, "sliding_layers": sliding, "total": full + sliding,
            "per_token_full_layers": layers_full * per_token}


@dataclass(frozen=True)
class SparkStyleConfig:
    vocab: int = 593
    hidden: int = 64
    layers: int = 4                    # pattern: sliding, sliding, sliding, full, ...
    query_heads: int = 4
    kv_heads: int = 1
    head_dim: int = 16
    window: int = 512
    partial_rotary: float = 0.25
    theta_sliding: float = 1e4
    theta_full: float = 5e6
    mlp: int = 256
    capacity: int = 1 << 16            # tokens a full layer can hold (preallocated)
    seed: int = 0

    @property
    def layer_types(self) -> tuple:
        return tuple("full" if (i + 1) % 4 == 0 else "sliding" for i in range(self.layers))


def _rotate(x: torch.Tensor, positions: torch.Tensor, theta: float, fraction: float) -> torch.Tensor:
    """Rotary on the first ``fraction`` of the head dimension (pairs), the rest unchanged."""
    dim = x.shape[-1]
    rotary = int(dim * fraction) // 2 * 2
    if rotary == 0:
        return x
    frequency = theta ** (-torch.arange(0, rotary, 2, dtype=torch.float64, device=x.device) / rotary)
    angle = positions.to(torch.float64)[:, None] * frequency[None]
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    head, tail = x[..., :rotary], x[..., rotary:]
    even, odd = head[..., 0::2], head[..., 1::2]
    turned = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
    return torch.cat((turned, tail), dim=-1)


class _Layer(nn.Module):
    def __init__(self, c: SparkStyleConfig, kind: str):
        super().__init__()
        self.c, self.kind = c, kind
        self.norm1, self.norm2 = nn.RMSNorm(c.hidden), nn.RMSNorm(c.hidden)
        self.q = nn.Linear(c.hidden, c.query_heads * c.head_dim, bias=False)
        self.kv = nn.Linear(c.hidden, 2 * c.kv_heads * c.head_dim, bias=False)
        self.gate = nn.Linear(c.hidden, c.query_heads, bias=False)        # head-wise sigmoid output gate
        self.o = nn.Linear(c.query_heads * c.head_dim, c.hidden, bias=False)
        self.up, self.down = nn.Linear(c.hidden, c.mlp, bias=False), nn.Linear(c.mlp, c.hidden, bias=False)

    def forward(self, x, positions, cache):
        c, (b, t, _) = self.c, x.shape
        h = self.norm1(x)
        theta, fraction = (c.theta_full, c.partial_rotary) if self.kind == "full" else (c.theta_sliding, 1.0)
        q = _rotate(self.q(h).view(b, t, c.query_heads, c.head_dim).transpose(1, 2), positions, theta, fraction)
        k, v = self.kv(h).view(b, t, 2, c.kv_heads, c.head_dim).permute(2, 0, 3, 1, 4)
        k = _rotate(k, positions, theta, fraction)
        keys, values, key_pos = cache.extend(self.kind, k, v, positions)
        group = c.query_heads // c.kv_heads
        keys, values = keys.repeat_interleave(group, 1), values.repeat_interleave(group, 1)
        visible = key_pos[None, :] <= positions[:, None]
        if self.kind == "sliding":
            visible &= key_pos[None, :] > positions[:, None] - c.window
        scores = (q @ keys.transpose(-1, -2)) / math.sqrt(c.head_dim)
        out = torch.softmax(scores.masked_fill(~visible, float("-inf")), dim=-1) @ values         # [b, h, t, e]
        out = out * torch.sigmoid(self.gate(h)).transpose(1, 2)[..., None]
        x = x + self.o(out.transpose(1, 2).reshape(b, t, -1))
        return x + self.down(F.gelu(self.up(self.norm2(x))))


class _LayerCache:
    """Preallocated per-layer KV: a ring of ``window`` for sliding layers, ``capacity`` slots for full layers."""

    def __init__(self, c: SparkStyleConfig, kind: str, batch: int, dtype):
        size = c.window if kind == "sliding" else c.capacity
        self.size, self.kind = size, kind
        self.k = torch.zeros(batch, c.kv_heads, size, c.head_dim, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.pos = torch.full((size,), -1, dtype=torch.long)


class SparkStyleCache:
    def __init__(self, model: "SparkStyleLM", batch: int):
        c = model.c
        dtype = next(model.parameters()).dtype
        self.layers = [_LayerCache(c, kind, batch, dtype) for kind in c.layer_types]
        self.position, self._current = 0, 0

    def extend(self, kind, k, v, positions):
        layer = self.layers[self._current]
        self._current += 1
        if kind == "full" and int(positions[-1]) >= layer.size:
            raise MemoryError(f"a full-attention layer holds {layer.size} tokens; the stream would exceed it")
        slots = positions % layer.size
        old_k, old_v, old_pos = layer.k.clone(), layer.v.clone(), layer.pos.clone()
        layer.k[:, :, slots], layer.v[:, :, slots], layer.pos[slots] = k, v, positions
        held = layer.pos >= 0
        if kind == "sliding":
            # Keys of this chunk plus the window before it (the ring may have overwritten some of those).
            keep = old_pos >= 0
            keys = torch.cat((old_k[:, :, keep], k), dim=2)
            values = torch.cat((old_v[:, :, keep], v), dim=2)
            return keys, values, torch.cat((old_pos[keep], positions))
        return layer.k[:, :, held], layer.v[:, :, held], layer.pos[held]

    def nbytes(self) -> dict:
        out = {"sliding": 0, "full": 0}
        for layer in self.layers:
            out[layer.kind] += (layer.k.numel() + layer.v.numel()) * layer.k.element_size()
        out["total"] = out["sliding"] + out["full"]
        return out


class SparkStyleLM(nn.Module):
    def __init__(self, c: SparkStyleConfig):
        super().__init__()
        self.c = c
        torch.manual_seed(c.seed)
        self.embed = nn.Embedding(c.vocab, c.hidden)
        self.layers = nn.ModuleList(_Layer(c, kind) for kind in c.layer_types)
        self.norm = nn.RMSNorm(c.hidden)

    def new_cache(self, batch: int) -> SparkStyleCache:
        return SparkStyleCache(self, batch)

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, cache: SparkStyleCache, *, logit_positions=None):
        positions = cache.position + torch.arange(tokens.shape[1])
        cache._current = 0
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x, positions, cache)
        cache.position += tokens.shape[1]
        if logit_positions is not None:
            x = x[:, logit_positions]
        return F.linear(self.norm(x), self.embed.weight)
