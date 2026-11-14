"""R1: a scrolling-context transformer. Bounded local attention plus sparse access to a long rolling history.

RESEARCH_ONLY. Lossy by construction: old context is reachable only through compressed blocks and a few
selected blocks per query. Nothing here is equivalent to attention over the full context, and no such claim
is made (docs/research/elpis_torch/DESIGN_R1.md).

Four coordinate systems, never mixed:

    absolute stream position   p, an unbounded integer. Token p is the p-th token of the stream.
    resident window slot       p mod W: where token p's local key and value live while p is among the last W.
    memory address             a block index b = p // s (compressed tier, slot b mod M) or a token position
                               (fine tier and token history, slot p mod (H + s)).
    logical horizon            H tokens: what a query can reach in any form. Older blocks are overwritten.

One attention, one softmax per query, over four groups of keys:

    local       the last W tokens, raw, rotary positions RELATIVE to the window (never the absolute p, so the
                result does not depend on where in the stream the window sits);
    sinks       the first `sinks` tokens of the stream, pinned, at a fixed relative offset behind the window;
    compressed  one pooled key/value per block of s tokens, for blocks entirely older than the window;
    fine        the raw tokens of the `fine_blocks` best blocks.

Which old blocks a query reads is decided by the attention logits themselves (top-k of the query's own logits
against the compressed keys). There is no retrieval controller, ranker or second model. Old keys carry no
rotary phase; a learned per-head slope on log-distance is their only position signal.

Visibility is defined per query, not per chunk: a block b is reachable through memory iff it ends at or before
p - W (so nothing is both in the window and in memory, and nothing falls in between) and b >= p // s - M (the
horizon is measured from the query). The chunk size is
therefore an execution detail: any chunking gives the same logits (tested).

Every buffer is allocated once, at ``new_state``. A forward pass never concatenates onto history, never
builds a [T, history] dense matrix over raw tokens, and refuses a chunk it cannot hold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .cgc import CGCTransform
from .config import ConfigError, ModelConfig
from .model import ElpisTorchLM, RMSNorm, Rotary, SwiGLU, _seeded_normal


class ChunkError(RuntimeError):
    """A forward call was given more tokens than one chunk may hold. Refused whole; split the input."""


class EvictedError(LookupError):
    """An address older than the horizon (or not yet written) was requested."""


@dataclass(frozen=True)
class ScrollConfig:
    """``model`` gives the widths (its ``max_seq_len`` is unused here). The rest shapes the context."""
    model: ModelConfig
    window: int = 128              # W: raw tokens every query sees
    block: int = 16                # s: tokens per compressed block
    chunk: int = 64                # C: most tokens one forward call may process (W >= C + s)
    horizon: int = 1 << 20         # H: tokens reachable through memory; a multiple of s
    global_layers: tuple = ()      # layers that read the memory; () is the local-only topology
    global_topk: int = 8           # old blocks a query reads (compressed and fine together)
    fine_blocks: int = 2           # of those, how many are read as raw tokens (0: no fine tier)
    sinks: int = 0                 # first tokens of the stream kept visible forever

    def __post_init__(self):
        if not isinstance(self.model, ModelConfig):
            raise ConfigError("ScrollConfig.model must be a ModelConfig")
        for name in ("window", "block", "chunk", "horizon", "global_topk"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ConfigError(f"{name} must be a positive integer")
        for name in ("fine_blocks", "sinks"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ConfigError(f"{name} must be a non-negative integer")
        if self.window < self.chunk + self.block:
            raise ConfigError("window must be at least chunk + block, or tokens could fall between window and memory")
        if self.horizon % self.block or self.horizon < self.window:
            raise ConfigError("horizon must be a multiple of block and at least window")
        layers = tuple(self.global_layers)
        if any(type(i) is not int or not 0 <= i < self.model.n_layers for i in layers) or len(set(layers)) != len(layers):
            raise ConfigError("global_layers must be distinct layer indices")
        if self.fine_blocks > self.global_topk:
            raise ConfigError("fine_blocks cannot exceed global_topk")
        object.__setattr__(self, "global_layers", layers)

    @property
    def slots(self) -> int:
        """M: compressed blocks held."""
        return self.horizon // self.block

    @property
    def fine_capacity(self) -> int:
        """Raw-token slots of the fine tier and of the token history: the horizon plus one block of slack."""
        return self.horizon + self.block

    @property
    def topology(self) -> str:
        return "sparse_global" if self.global_layers else "local"


class StreamState:
    """All volatile state of ``batch`` parallel streams. Preallocated; its size never changes."""

    def __init__(self, config: ScrollConfig, batch: int, origin: int, dtype: torch.dtype):
        c, m = config, config.model
        if origin < 0 or origin % c.block:
            raise ValueError("the stream origin must be a non-negative multiple of the block size")
        self.config, self.batch, self.origin, self.position = c, batch, origin, origin
        shape = (batch, m.kv_heads)
        self.tokens = torch.zeros(batch, c.fine_capacity, dtype=torch.int32)
        self.layers = []
        for index in range(m.n_layers):
            layer = {"lk": torch.zeros(*shape, c.window, m.head_dim, dtype=dtype),
                     "lv": torch.zeros(*shape, c.window, m.head_dim, dtype=dtype)}
            if c.sinks:
                layer["sk"] = torch.zeros(*shape, c.sinks, m.head_dim, dtype=dtype)
                layer["sv"] = torch.zeros(*shape, c.sinks, m.head_dim, dtype=dtype)
            if index in c.global_layers:
                layer["pk"] = torch.zeros(*shape, c.block, m.head_dim, dtype=dtype)     # the open block
                layer["pv"] = torch.zeros(*shape, c.block, m.head_dim, dtype=dtype)
                layer["pg"] = torch.zeros(*shape, c.block, dtype=dtype)
                layer["ck"] = torch.zeros(*shape, c.slots, m.head_dim, dtype=dtype)
                layer["cv"] = torch.zeros(*shape, c.slots, m.head_dim, dtype=dtype)
                if c.fine_blocks:
                    layer["fk"] = torch.zeros(*shape, c.fine_capacity, m.head_dim, dtype=dtype)
                    layer["fv"] = torch.zeros(*shape, c.fine_capacity, m.head_dim, dtype=dtype)
            self.layers.append(layer)

    TIERS = {"lk": "window", "lv": "window", "sk": "sinks", "sv": "sinks", "pk": "open_block", "pv": "open_block",
             "pg": "open_block", "ck": "compressed", "cv": "compressed", "fk": "fine", "fv": "fine"}

    def nbytes(self) -> dict:
        """Bytes held, by tier. Constant for the life of the state."""
        out = {"token_history": self.tokens.numel() * self.tokens.element_size()}
        for layer in self.layers:
            for name, tensor in layer.items():
                out[self.TIERS[name]] = out.get(self.TIERS[name], 0) + tensor.numel() * tensor.element_size()
        out["total"] = sum(out.values())
        return out

    def coordinates(self) -> dict:
        """Where the stream is, in each coordinate system."""
        c, seen = self.config, self.position - self.origin
        blocks_done = self.position // c.block - self.origin // c.block
        oldest_block = self.position // c.block - min(blocks_done, c.slots)
        return {"position": self.position, "tokens_seen": seen,
                "window": (max(self.origin, self.position - c.window), self.position),
                "blocks_resident": min(blocks_done, c.slots), "oldest_resident_block": oldest_block,
                "oldest_reachable_position": max(self.origin, oldest_block * c.block) if blocks_done else
                max(self.origin, self.position - c.window),
                "oldest_addressable_token": max(self.origin, self.position - c.fine_capacity),
                "evicted_blocks": max(0, blocks_done - c.slots)}

    def token_at(self, position: int) -> torch.Tensor:
        """The token ids at an absolute position, for every stream. Refuses what the history no longer holds."""
        if not max(self.origin, self.position - self.config.fine_capacity) <= position < self.position:
            raise EvictedError(f"position {position} is not in the resident history")
        return self.tokens[:, position % self.config.fine_capacity].long()


def _put(buffer: torch.Tensor, dim: int, index: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Write ``value`` at ``index`` along ``dim``. In place for inference; out of place when a graph is attached."""
    if value.requires_grad or buffer.requires_grad:
        return buffer.index_copy(dim, index, value)
    return buffer.index_copy_(dim, index, value.detach())


class ScrollAttention(nn.Module):
    def __init__(self, c: ScrollConfig, is_global: bool, dtype: torch.dtype):
        super().__init__()
        m = c.model
        self.c, self.is_global = c, is_global
        self.heads, self.kv_heads, self.head_dim = m.n_heads, m.kv_heads, m.head_dim
        self.group = self.heads // self.kv_heads
        self.q = nn.Linear(m.d_model, self.heads * self.head_dim, bias=False, dtype=dtype)
        self.k = nn.Linear(m.d_model, self.kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.v = nn.Linear(m.d_model, self.kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.out = nn.Linear(self.heads * self.head_dim, m.d_model, bias=False, dtype=dtype)
        if is_global:
            # Memory has its own query and key maps (content addressing without rotary phase) and a gate that
            # decides how a block's tokens are pooled into its compressed key and value.
            self.gq = nn.Linear(m.d_model, self.heads * self.head_dim, bias=False, dtype=dtype)
            self.gk = nn.Linear(m.d_model, self.kv_heads * self.head_dim, bias=False, dtype=dtype)
            self.gate = nn.Linear(m.d_model, self.kv_heads, bias=False, dtype=dtype)
            self.recency = nn.Parameter(torch.zeros(self.heads, dtype=dtype))     # slope on log-distance, >= 0

    def _split(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], heads, self.head_dim).transpose(1, 2)

    def _wide(self, x: torch.Tensor) -> torch.Tensor:
        """Key/value heads -> query heads (each kv head serves ``group`` consecutive query heads)."""
        return x if self.group == 1 else x.repeat_interleave(self.group, dim=1)

    def forward(self, x: torch.Tensor, rotary: Rotary, layer: dict, state: StreamState):
        c, e, scale = self.c, self.head_dim, self.head_dim ** -0.5
        batch, length, _ = x.shape
        p0, origin, W = state.position, state.origin, c.window
        seen = p0 - origin
        positions = p0 + torch.arange(length)                                   # absolute, int64
        q_u, k_u, v = self._split(self.q(x), self.heads), self._split(self.k(x), self.kv_heads), \
            self._split(self.v(x), self.kv_heads)

        # -- local: the resident window, in chronological order, then this chunk -----------------------------
        held = min(seen, W)
        ring = torch.arange(p0 - held, p0)
        keys_u = torch.cat((layer["lk"][:, :, ring % W], k_u), dim=2)
        values = torch.cat((layer["lv"][:, :, ring % W], v), dim=2)
        # Rotary positions are relative to the window (W - held .. W + length - 1), never absolute.
        q_r, keys_r = rotary(q_u, W), rotary(keys_u, W - held)
        logits = [(q_r @ self._wide(keys_r).transpose(-1, -2)) * scale]
        key_positions = torch.cat((ring, positions))
        visible = (key_positions[None, :] <= positions[:, None]) & (key_positions[None, :] > positions[:, None] - W)
        logits[0] = logits[0].masked_fill(~visible, float("-inf"))
        parts = [self._wide(values)]

        # -- sinks: pinned first tokens, at a constant relative offset behind the window ---------------------
        if c.sinks:
            offsets = W + c.sinks - torch.arange(c.sinks)                       # distance of sink j from the query
            cos, sin = rotary.cos[offsets], rotary.sin[offsets]                 # [sinks, e/2]
            even, odd = q_u[..., None, 0::2], q_u[..., None, 1::2]              # [B, h, T, 1, e/2]
            q_s = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
            sink_logits = (q_s * self._wide(layer["sk"])[:, :, None]).sum(-1) * scale     # [B, h, T, sinks]
            behind = (origin + torch.arange(c.sinks))[None, :] <= positions[:, None] - W
            logits.append(sink_logits.masked_fill(~behind, float("-inf")))

        # -- memory: blocks that ended at or before p - W -----------------------------------------------------
        selected = None
        blocks_done = p0 // c.block - origin // c.block
        if self.is_global and blocks_done:
            gq = self._split(self.gq(x), self.heads)
            slope = F.softplus(self.recency)[None, :, None, None]
            used = min(blocks_done, c.slots)
            blocks = torch.arange(p0 // c.block - used, p0 // c.block)          # resident, oldest first
            kv = (batch, self.kv_heads)
            ck = layer["ck"][:, :, blocks % c.slots]                            # [B, g, used, e]
            distance = positions[:, None] - (blocks * c.block + c.block - 1)[None, :]       # [T, used], exact ints
            # Reachable: entirely behind the window, and within the horizon OF THIS QUERY (not of the chunk).
            eligible = (distance >= W) & (blocks[None, :] >= (positions // c.block - c.slots)[:, None])
            grouped = gq.reshape(*kv, self.group, length, e) @ ck[:, :, None].transpose(-1, -2)
            block_logits = grouped.reshape(batch, self.heads, length, used) * scale \
                - slope * torch.log(distance.clamp(min=1).to(x.dtype))[None, None]
            # The query's own logits choose what it reads: the best blocks by its strongest head.
            score = block_logits.max(dim=1).values.masked_fill(~eligible[None], float("-inf"))
            top = score.topk(min(c.global_topk, used), dim=-1)                   # [B, T, K]
            valid, fine = top.values > float("-inf"), min(c.fine_blocks, top.indices.shape[-1])
            selected = blocks[top.indices].masked_fill(~valid, -1)
            coarse = top.indices[..., fine:]
            if coarse.shape[-1]:
                picked = coarse[:, None].expand(batch, self.heads, length, coarse.shape[-1])
                logits.append(block_logits.gather(-1, picked).masked_fill(~valid[:, None, :, fine:], float("-inf")))
                index = coarse[:, None, :, :, None].expand(*kv, length, coarse.shape[-1], e)
                cv = layer["cv"][:, :, blocks % c.slots][:, :, None].expand(*kv, length, used, e)
                parts.append(("each", self._wide(cv.gather(3, index))))
            if fine:
                tokens = (blocks[top.indices[..., :fine]][..., None] * c.block + torch.arange(c.block)).flatten(-2)
                index = (tokens % c.fine_capacity)[:, None, :, :, None].expand(*kv, length, tokens.shape[-1], e)

                def read(buffer):                                               # [B, h, T, fine * s, e]
                    source = buffer[:, :, None].expand(*kv, length, c.fine_capacity, e)
                    return self._wide(source.gather(3, index))

                gap = (positions[None, :, None] - tokens).clamp(min=1).to(x.dtype)
                fine_logits = (gq[:, :, :, None] * read(layer["fk"])).sum(-1) * scale - slope * torch.log(gap)[:, None]
                ok = valid[..., :fine].repeat_interleave(c.block, dim=-1)
                logits.append(fine_logits.masked_fill(~ok[:, None], float("-inf")))
                parts.append(("each", read(layer["fv"])))

        # -- one softmax over everything the query can see ---------------------------------------------------
        weights = torch.softmax(torch.cat(logits, dim=-1), dim=-1)
        width = keys_u.shape[2]
        mixed = weights[..., :width] @ parts[0]
        at = width
        if c.sinks:
            mixed = mixed + weights[..., at:at + c.sinks] @ self._wide(layer["sv"])
            at += c.sinks
        for _, value in parts[1:]:
            mixed = mixed + (weights[..., at:at + value.shape[3], None] * value).sum(3)
            at += value.shape[3]
        attended = self.out(mixed.transpose(1, 2).reshape(batch, length, self.heads * e))
        pending = {"k": k_u, "v": v}
        if self.is_global:
            pending["gk"] = self._split(self.gk(x), self.kv_heads)
            pending["gate"] = self.gate(x).transpose(1, 2)                      # [B, g, T]
        return attended, pending, selected

    def commit(self, layer: dict, state: StreamState, pending: dict) -> None:
        """Write this chunk into the rings. Overwriting the oldest slot is the eviction; nothing else evicts."""
        c, p0, length = self.c, state.position, pending["k"].shape[2]
        positions = p0 + torch.arange(length)
        layer["lk"] = _put(layer["lk"], 2, positions % c.window, pending["k"])
        layer["lv"] = _put(layer["lv"], 2, positions % c.window, pending["v"])
        if c.sinks and p0 - state.origin < c.sinks:
            take = min(length, c.sinks - (p0 - state.origin))
            index = torch.arange(p0 - state.origin, p0 - state.origin + take)
            layer["sk"] = _put(layer["sk"], 2, index, pending["k"][:, :, :take])
            layer["sv"] = _put(layer["sv"], 2, index, pending["v"][:, :, :take])
        if not self.is_global:
            return
        if c.fine_blocks:
            layer["fk"] = _put(layer["fk"], 2, positions % c.fine_capacity, pending["gk"])
            layer["fv"] = _put(layer["fv"], 2, positions % c.fine_capacity, pending["v"])
        opened = p0 % c.block                                                    # tokens already in the open block
        keys = torch.cat((layer["pk"][:, :, :opened], pending["gk"]), dim=2)
        values = torch.cat((layer["pv"][:, :, :opened], pending["v"]), dim=2)
        gates = torch.cat((layer["pg"][:, :, :opened], pending["gate"]), dim=2)
        complete = keys.shape[2] // c.block
        if complete:
            span = complete * c.block
            shape = (keys.shape[0], keys.shape[1], complete, c.block)
            pool = torch.softmax(gates[:, :, :span].reshape(shape), dim=-1)[..., None]
            first = (p0 - opened) // c.block
            index = (first + torch.arange(complete)) % c.slots
            layer["ck"] = _put(layer["ck"], 2, index, (pool * keys[:, :, :span].reshape(*shape, -1)).sum(3))
            layer["cv"] = _put(layer["cv"], 2, index, (pool * values[:, :, :span].reshape(*shape, -1)).sum(3))
        rest = keys.shape[2] - complete * c.block
        index = torch.arange(rest)
        layer["pk"] = _put(layer["pk"], 2, index, keys[:, :, complete * c.block:])
        layer["pv"] = _put(layer["pv"], 2, index, values[:, :, complete * c.block:])
        layer["pg"] = _put(layer["pg"], 2, index, gates[:, :, complete * c.block:])


class ScrollBlock(nn.Module):
    """The R0 block with the scrolling attention: same norms, same channel transformation, same names."""

    def __init__(self, c: ScrollConfig, index: int, dtype: torch.dtype):
        super().__init__()
        m = c.model
        self.attn_norm = RMSNorm(m.d_model, m.norm_eps, dtype)
        self.attn = ScrollAttention(c, index in c.global_layers, dtype)
        self.channel_norm = RMSNorm(m.d_model, m.norm_eps, dtype)
        self.ffn = SwiGLU(m.d_model, m.ffn_width, dtype) if m.has_ffn else None
        self.cgc = CGCTransform(m.d_model, m.cgc_cells, m.cgc_units, bottleneck=m.cgc_bottleneck,
                                preact_bound=m.cgc_preact_bound, dtype=dtype) if m.has_cgc else None
        self.cgc_gate = nn.Parameter(torch.zeros(1, dtype=dtype)) if m.channel == "ffn+cgc" else None

    def channel(self, x: torch.Tensor) -> torch.Tensor:
        if self.cgc is None:
            return self.ffn(x)
        if self.ffn is None:
            return self.cgc(x)
        return self.ffn(x) + self.cgc_gate * self.cgc(x)


class ScrollLM(nn.Module):
    """Embeddings -> scrolling blocks -> RMSNorm -> logits. State lives in a ``StreamState``, never here."""

    RECENCY_INIT = -2.0          # softplus(-2) = 0.127: a mild initial preference for recent memory

    def __init__(self, config: ScrollConfig):
        super().__init__()
        if not isinstance(config, ScrollConfig):
            raise TypeError("ScrollLM expects a ScrollConfig")
        self.scroll, self.config = config, config.model
        m, dtype = config.model, getattr(torch, config.model.dtype)
        self.embed = nn.Embedding(m.vocab_size, m.d_model, dtype=dtype)
        self.rotary = Rotary(m.head_dim, config.window + config.chunk + config.sinks + 1, m.rope_theta, dtype)
        self.blocks = nn.ModuleList(ScrollBlock(config, i, dtype) for i in range(m.n_layers))
        self.final_norm = RMSNorm(m.d_model, m.norm_eps, dtype)
        self.head = None if m.tie_embeddings else nn.Linear(m.d_model, m.vocab_size, bias=False, dtype=dtype)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """The R0 rule: every weight a function of (seed, parameter name) only. Shared names get R0's values."""
        for name, parameter in self.named_parameters():
            if name.endswith("attn.recency"):
                parameter.fill_(self.RECENCY_INIT)
                continue
            std = ElpisTorchLM._init_std(self, name)
            if std is not None:
                parameter.copy_(_seeded_normal(self.config.seed, name, parameter.shape, std, parameter.dtype))
            elif name.endswith("cgc_gate"):
                parameter.fill_(self.config.cgc_gate_init)
            else:
                parameter.fill_(1.0)

    def new_state(self, batch: int, *, origin: int = 0) -> StreamState:
        return StreamState(self.scroll, batch, origin, getattr(torch, self.config.dtype))

    def forward(self, tokens: torch.Tensor, state: StreamState, *, return_pre: bool = False,
                return_selected: bool = False):
        """Logits ``[batch, T, vocab]`` for the next ``T <= chunk`` tokens of the streams in ``state``.

        ``state`` advances only after every layer has succeeded. ``return_pre`` also returns the last block's
        stream after attention and before its channel transformation (what a last-block fast weight acts on).
        """
        c = self.scroll
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be an int64 tensor of shape [batch, T]")
        batch, length = tokens.shape
        if batch != state.batch or state.config != c:
            raise ValueError("the state does not belong to this model and batch")
        if not 1 <= length <= c.chunk:
            raise ChunkError(f"{length} tokens do not fit one chunk of {c.chunk}; split the input")
        if int(tokens.min()) < 0 or int(tokens.max()) >= self.config.vocab_size:
            raise ValueError(f"token ids must be in [0, {self.config.vocab_size})")
        x = self.embed(tokens)
        staged, chosen, pre = [], {}, None
        for index, block in enumerate(self.blocks):
            attended, pending, selected = block.attn(block.attn_norm(x), self.rotary, state.layers[index], state)
            staged.append(pending)
            if selected is not None:
                chosen[index] = selected
            pre = x + attended
            x = pre + block.channel(block.channel_norm(pre))
        x = self.final_norm(x)
        logits = F.linear(x, self.embed.weight) if self.head is None else self.head(x)
        for index, block in enumerate(self.blocks):
            block.attn.commit(state.layers[index], state, staged[index])
        positions = state.position + torch.arange(length)
        state.tokens.index_copy_(1, positions % c.fine_capacity, tokens.to(torch.int32))
        state.position += length
        out = (logits,)
        if return_pre:
            out += (pre,)
        if return_selected:
            out += (chosen,)
        return out[0] if len(out) == 1 else out

    def run(self, tokens: torch.Tensor, state: StreamState | None = None, *, chunk: int | None = None):
        """Feed a whole ``[batch, T]`` sequence in chunks. Returns ``(logits [batch, T, vocab], state)``."""
        state = self.new_state(tokens.shape[0]) if state is None else state
        step = chunk or self.scroll.chunk
        parts = [self(tokens[:, at:at + step], state) for at in range(0, tokens.shape[1], step)]
        return torch.cat(parts, dim=1), state
