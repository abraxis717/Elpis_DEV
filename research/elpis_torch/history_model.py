"""R3: ERSA + CGC with learned historical addressing and selected-source rereading.

Three functions, kept apart (DESIGN_R3.md):

    working memory      ERSA (and optional window attention) carry the ongoing computation in bounded state.
    addressing          a learned query from the model's own hidden state scores a bounded index of completed
                        blocks; the top-``K`` blocks are selected. The index holds one small learned key per
                        block (``per_block`` vectors of ``index_dim``), not the block's contents.
    readout             the selected blocks are REREAD from the exact token history, re-encoded by a small block
                        encoder, and attended by the query. Source identity is explicit: every historical key
                        carries its position inside its block and its distance bucket from the query, and it
                        enters only through this path (never through ERSA, the window or the token stream).

Modes of the same weights (``forward(..., mode=...)``):

    "off"              recurrent only: no history is read.
    "learned"          autonomous: the model selects blocks itself. No label enters.
    "oracle"           DIAGNOSIS ONLY: at positions where ``address`` gives a block, that block is read (the
                       address, never the answer); elsewhere nothing is read.
    "oracle_override"  DIAGNOSIS ONLY: the oracle address where given, the learned selection elsewhere.

Gradients. Selection is a hard top-``K``. The readout's logits include ``log softmax`` of the selected blocks'
index scores, so the language-model loss reaches the scores of the blocks that were selected (and through them
the query and key maps). A block that was not selected receives a gradient only from the optional supervised
addressing loss (``addressing_loss``: training-only labels from the task generator, never an input of the
model). Index keys written to the ring are detached: across chunks, stored keys receive no gradient; blocks
completed inside the current chunk (all of them, when a training sequence is one chunk) do.

Causality. A block is eligible for the query at position ``p`` iff it is complete at ``p`` (its last token is
at or before ``p``) and it is among the ``horizon / block`` most recent complete blocks of ``p``. The encoder
sees a block and ``overlap`` tokens before it, all at or before the block's end: never the future.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .cgc import CGCTransform
from .config import ConfigError
from .ersa_model import ErsaConfig, ErsaLM, ErsaState
from .context_runtime import exact_topk, ring_segments, tiled_select
from .history import BlockIndex, TokenHistory
from .model import RMSNorm, SwiGLU, _seeded_normal

MODES = ("off", "learned", "oracle", "oracle_override")


@dataclass(frozen=True)
class HistoryConfig:
    block: int = 32                  # s: tokens per addressable block
    horizon: int = 1 << 20           # tokens addressable through the index (a multiple of block)
    topk: int = 2                    # K: blocks read per query
    index_dim: int = 32              # width of one index key
    per_block: int = 2               # learned index keys per block (a block's score is their max)
    read_heads: int = 2              # readout query heads
    read_kv_heads: int = 1           # readout key/value heads
    read_head_dim: int = 32
    overlap: int = 8                 # tokens before a block that its encoder sees (context, never read)
    layers: tuple = (-1,)            # blocks that carry a history read (negative: from the end)
    encoder_heads: int = 2
    encoder_dim: int | None = None   # width of the block encoder; None = d_model (a narrower one is cheaper)
    mode: str = "learned"            # default mode of forward()

    def __post_init__(self):
        for name in ("block", "horizon", "topk", "index_dim", "per_block", "read_heads", "read_kv_heads",
                     "read_head_dim", "encoder_heads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ConfigError(f"{name} must be a positive integer")
        if type(self.overlap) is not int or self.overlap < 1:
            raise ConfigError("overlap must be a positive integer (the readout keys need each token's predecessor)")
        if self.encoder_dim is not None and (type(self.encoder_dim) is not int or self.encoder_dim < 2
                                             or self.encoder_dim % 2 or self.encoder_dim % self.encoder_heads):
            raise ConfigError("encoder_dim must be None or an even positive multiple of encoder_heads")
        if self.horizon % self.block:
            raise ConfigError("horizon must be a multiple of block")
        if self.read_heads % self.read_kv_heads:
            raise ConfigError("read_heads must be a multiple of read_kv_heads")
        if self.mode not in MODES:
            raise ConfigError(f"mode must be one of {MODES}")
        layers = tuple(self.layers)
        if not layers or any(type(i) is not int for i in layers):
            raise ConfigError("layers must be a non-empty tuple of block indices")
        object.__setattr__(self, "layers", layers)

    @property
    def slots(self) -> int:
        """Index records held: the blocks of one horizon."""
        return self.horizon // self.block

    @property
    def history_capacity(self) -> int:
        """Token slots: the horizon, one block of slack and the encoder's overlap."""
        return self.horizon + self.block + self.overlap

    @property
    def span(self) -> int:
        return self.overlap + self.block


def log_distance(distance: torch.Tensor, dtype) -> torch.Tensor:
    """``log(1 + distance)``: the only distance signal, so that it extrapolates beyond the trained lengths."""
    return torch.log1p(distance.clamp(min=0).to(dtype))


class _Channel(nn.Module):
    """The model's own channel kind (CGC, FFN or both) for the block encoder."""

    def __init__(self, m, dtype):
        super().__init__()
        self.ffn = SwiGLU(m.d_model, m.ffn_width, dtype) if m.has_ffn else None
        self.cgc = CGCTransform(m.d_model, m.cgc_cells, m.cgc_units, bottleneck=m.cgc_bottleneck,
                                preact_bound=m.cgc_preact_bound, dtype=dtype) if m.has_cgc else None
        self.cgc_gate = nn.Parameter(torch.zeros(1, dtype=dtype)) if m.channel == "ffn+cgc" else None

    def forward(self, x):
        if self.cgc is None:
            return self.ffn(x)
        if self.ffn is None:
            return self.cgc(x)
        return self.ffn(x) + self.cgc_gate * self.cgc(x)


class BlockEncoder(nn.Module):
    """Re-encodes a block from its token ids alone: shared embeddings + a learned position inside the span, one
    bidirectional attention layer over the span (all of it is in the past of any query that may read it) and
    one channel layer of the model's kind. Output: the block's ``s`` token states (the overlap is context)."""

    def __init__(self, m, h: HistoryConfig, dtype):
        super().__init__()
        self.heads, self.span, self.block = h.encoder_heads, h.span, h.block
        width = h.encoder_dim or m.d_model
        if width % self.heads:
            raise ConfigError("the encoder width must be divisible by encoder_heads")
        # A narrower encoder projects the shared embeddings down and uses a channel of the same kind at its width.
        self.down = nn.Linear(m.d_model, width, bias=False, dtype=dtype) if width != m.d_model else None
        if width != m.d_model:
            m = m.replace(d_model=width, n_heads=1, n_kv_heads=None, cgc_bottleneck=max(2, width // 2),
                          cgc_channels=2 * width, ffn_hidden=round(8 * width / 3))
        self.width = width
        self.position = nn.Parameter(torch.zeros(h.span, width, dtype=dtype))
        self.attn_norm = RMSNorm(width, m.norm_eps, dtype)
        self.qkv = nn.Linear(width, 3 * width, bias=False, dtype=dtype)
        self.attn_out = nn.Linear(width, width, bias=False, dtype=dtype)
        self.channel_norm = RMSNorm(width, m.norm_eps, dtype)
        self.channel = _Channel(m, dtype)
        self.out_norm = RMSNorm(width, m.norm_eps, dtype)

    def forward(self, embedded: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``embedded [N, span, d_model]``, ``valid [N, span]`` (False before the stream origin) -> ``[N, s + 1, width]``:
        the block's ``s`` token states preceded by the state of the token before the block."""
        if self.down is not None:
            embedded = self.down(embedded)
        n, span, d = embedded.shape
        x = embedded + self.position
        q, k, v = self.qkv(self.attn_norm(x)).view(n, span, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        mask = valid[:, None, None, :]
        scores = (q @ k.transpose(-1, -2)) * (d // self.heads) ** -0.5
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        x = x + self.attn_out((weights @ v).transpose(1, 2).reshape(n, span, d))
        x = x + self.channel(self.channel_norm(x))
        out = self.out_norm(x[:, -(self.block + 1):])           # the block and the token before it
        return out * valid[:, -(self.block + 1):, None]          # nothing before the stream origin


class HistoryReader(nn.Module):
    """Learned addressing (query, block keys, recency) and the readout attention for one block of the model."""

    def __init__(self, m, h: HistoryConfig, dtype):
        super().__init__()
        self.h = h
        d, e, width = m.d_model, h.read_head_dim, h.encoder_dim or m.d_model
        self.pool = nn.Linear(width, h.per_block, bias=False, dtype=dtype)        # which tokens make each key
        self.index_key = nn.Linear(width, h.index_dim, bias=False, dtype=dtype)
        self.index_query = nn.Linear(d, h.index_dim, bias=False, dtype=dtype)
        self.recency = nn.Parameter(torch.full((1,), -2.0, dtype=dtype))     # index: -softplus(.) log(1 + d)
        self.select_scale = nn.Parameter(torch.ones(1, dtype=dtype))          # weight of the selection prior
        self.index_scale = nn.Parameter(torch.full((1,), 0.5 * math.log(h.index_dim), dtype=dtype))   # log temp
        self.q = nn.Linear(d, h.read_heads * e, bias=False, dtype=dtype)
        self.k = nn.Linear(width, h.read_kv_heads * e, bias=False, dtype=dtype)
        self.k_prev = nn.Linear(width, h.read_kv_heads * e, bias=False, dtype=dtype)     # the previous token
        self.v = nn.Linear(width, h.read_kv_heads * e, bias=False, dtype=dtype)
        self.key_position = nn.Parameter(torch.zeros(h.block, h.read_kv_heads * e, dtype=dtype))
        self.key_rank = nn.Parameter(torch.zeros(h.topk, h.read_kv_heads * e, dtype=dtype))   # newest first
        self.read_recency = nn.Parameter(torch.full((h.read_heads,), -2.0, dtype=dtype))   # readout slope
        self.null = nn.Parameter(torch.zeros(h.read_heads, dtype=dtype))       # "read nothing" logit per head
        self.out = nn.Linear(h.read_heads * e, d, bias=False, dtype=dtype)

    def block_keys(self, states: torch.Tensor) -> torch.Tensor:
        """``[..., s + 1, w]`` encoded block -> ``[..., per_block, index_dim]`` index keys (the block's own tokens)."""
        states = states[..., 1:, :]
        weights = torch.softmax(self.pool(states), dim=-2)                       # [..., s, per_block]
        pooled = weights.transpose(-1, -2) @ states                              # [..., per_block, d]
        return F.normalize(self.index_key(pooled), dim=-1)

    def scores(self, u: torch.Tensor, keys: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        """``u [B, T, d]``, ``keys [B, N, per_block, index_dim]``, ``distance [T, N]`` -> ``[B, T, N]``."""
        query = F.normalize(self.index_query(u), dim=-1)
        match = torch.einsum("btd,bnkd->btnk", query, keys.to(query.dtype)).amax(dim=-1)
        slope = F.softplus(self.recency)
        return match * self.index_scale.exp() - slope * log_distance(distance, match.dtype)[None]

    def project(self, states: torch.Tensor):
        """Readout keys and values, once per block: ``[B, n, s + 1, w]`` -> two ``[B, n, s, g e]``.

        A token's key also carries its PREDECESSOR's state, so a query that matches what came before a token reads
        that token (``KEY_k`` matched, ``VAL_v`` read): the induction pattern, built into the readout's keys."""
        return (self.k(states[..., 1:, :]) + self.k_prev(states[..., :-1, :]) + self.key_position,
                self.v(states[..., 1:, :]))

    def read(self, u, keys, values, chosen_scores, valid, distance):
        """Readout over the selected blocks.

        ``keys``/``values [B, T, K, s, g e]`` of the selected blocks (from ``project``); ``chosen_scores
        [B, T, K]`` their index scores; ``valid [B, T, K]``; ``distance [B, T, K]`` from the query to each block's
        end. Returns the mixer term ``[B, T, d]`` and the readout weights ``[B, T, heads, K * s + 1]`` (last: null).
        """
        h, e = self.h, self.h.read_head_dim
        batch, length, picks, s, _ = keys.shape
        group = h.read_heads // h.read_kv_heads
        q = self.q(u).view(batch, length, h.read_kv_heads, group, e)
        k = keys.view(batch, length, picks, s, h.read_kv_heads, e)
        v = values.view(batch, length, picks, s, h.read_kv_heads, e)
        # Source identity, length-invariant: each selected block's recency rank among the selection (newest 0)
        # as a key embedding, and a per-head slope on log-distance. Both are per block, not per token.
        far = torch.where(valid, distance, torch.full_like(distance, 1 << 62))
        rank = torch.argsort(torch.argsort(far, dim=-1), dim=-1)
        where = self.key_rank[rank].view(batch, length, picks, h.read_kv_heads, e)
        logits = torch.einsum("btgqe,btksge->btgqks", q, k)
        logits = (logits + torch.einsum("btgqe,btkge->btgqk", q, where)[..., None]) * e ** -0.5
        slope = F.softplus(self.read_recency).view(1, 1, h.read_kv_heads, group, 1, 1)
        logits = logits - slope * log_distance(distance, logits.dtype)[:, :, None, None, :, None]
        prior = torch.log_softmax(chosen_scores.masked_fill(~valid, float("-inf")), dim=-1)
        prior = torch.where(valid, prior, torch.zeros_like(prior))
        logits = logits + self.select_scale * prior[:, :, None, None, :, None]
        logits = logits.masked_fill(~valid[:, :, None, None, :, None], float("-inf"))
        logits = logits.reshape(batch, length, h.read_heads, picks * s)
        null = self.null.view(1, 1, h.read_heads, 1).expand(batch, length, h.read_heads, 1)
        weights = torch.softmax(torch.cat((logits, null.to(logits.dtype)), dim=-1), dim=-1)
        taken = weights[..., :-1].view(batch, length, h.read_kv_heads, group, picks, s)
        mixed = torch.einsum("btgqks,btksge->btgqe", taken, v)
        return self.out(mixed.reshape(batch, length, h.read_heads * e)), weights


class HistoryState(ErsaState):
    """ERSA state plus the exact token history and the bounded block index. Fixed size from creation."""

    def __init__(self, model: "ErsaHistoryLM", batch: int, origin: int = 0, *, history_device=None,
                 index_device=None):
        super().__init__(model, batch)
        h = model.history
        if origin < 0 or origin % h.block:
            raise ValueError("the stream origin must be a non-negative multiple of the history block")
        self.origin = self.position = origin
        if model.ersa.attention == "window":
            self.window.origin = self.window.position = origin
        # R4 placement: the token ring and the index may live off the model's device (e.g. in CPU RAM).
        self.tokens = TokenHistory(batch, h.history_capacity, origin=origin,
                                   device=self.device if history_device is None else history_device)
        self.indexes = {layer: BlockIndex(batch, h.slots, h.per_block, h.index_dim, dtype=self.cells[0].dtype,
                                          device=self.device if index_device is None else index_device)
                        for layer in model.read_layers}
        self.runtime = None             # R4: None is the R3 reference path (context_runtime.RuntimeOptions)

    def reset(self) -> None:
        """R4: an empty stream in the same buffers."""
        super().reset()
        self.position = self.origin
        if self.window is not None:
            self.window.origin = self.window.position = self.origin
        self.tokens.reset()
        for index in self.indexes.values():
            index.reset()

    def nbytes(self) -> dict:
        out = super().nbytes()
        out.pop("total")
        out["history_tokens"] = self.tokens.nbytes()
        out["block_index"] = sum(index.nbytes() for index in self.indexes.values())
        out["total"] = sum(out.values())
        return out


class ErsaHistoryLM(ErsaLM):
    """``ErsaLM`` whose ``history.layers`` blocks add a learned historical read beside ERSA."""

    def __init__(self, config: ErsaConfig, history: HistoryConfig):
        if not isinstance(history, HistoryConfig):
            raise TypeError("ErsaHistoryLM expects a HistoryConfig")
        self.history = history
        super().__init__(config)
        m, dtype = config.model, getattr(torch, config.model.dtype)
        layers = sorted({i % m.n_layers for i in history.layers})
        if len(layers) != len(history.layers) or any(not -m.n_layers <= i < m.n_layers for i in history.layers):
            raise ConfigError("history.layers must name distinct blocks of the model")
        self.read_layers = tuple(layers)
        self.encoder = BlockEncoder(m, history, dtype)
        self.readers = nn.ModuleDict({str(i): HistoryReader(m, history, dtype) for i in layers})
        self._init_history()

    @torch.no_grad()
    def _init_history(self) -> None:
        m = self.config
        residual = m.init_std / (2 * m.n_layers) ** 0.5
        for name, parameter in self.named_parameters():
            if not name.startswith(("encoder.", "readers.")):
                continue
            if name.endswith("recency"):
                parameter.fill_(-2.0)
            elif name.endswith("null"):
                parameter.zero_()
            elif name.endswith("select_scale"):
                parameter.fill_(1.0)
            elif name.endswith("index_scale"):
                parameter.fill_(0.5 * math.log(self.history.index_dim))       # exp(.) = sqrt(index_dim)
            elif name.endswith("norm.weight"):
                parameter.fill_(1.0)
            elif name.endswith("cgc_gate"):
                parameter.fill_(m.cgc_gate_init)
            elif name.startswith("readers.") and name.endswith(".out.weight"):
                parameter.copy_(_seeded_normal(m.seed, name, parameter.shape, residual, parameter.dtype))
            elif name.endswith("cgc.core.weight"):
                std = m.cgc_init_gain * (m.cgc_bottleneck or m.d_model) ** -0.5
                parameter.copy_(_seeded_normal(m.seed, name, parameter.shape, std, parameter.dtype))
            elif name.endswith("cgc.proj_in.weight"):
                parameter.copy_(_seeded_normal(m.seed, name, parameter.shape, m.d_model ** -0.5, parameter.dtype))
            elif name.endswith(("cgc.proj_out.weight", "ffn.out.weight", "attn_out.weight")):
                parameter.copy_(_seeded_normal(m.seed, name, parameter.shape, residual, parameter.dtype))
            else:
                parameter.copy_(_seeded_normal(m.seed, name, parameter.shape, m.init_std, parameter.dtype))

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, "encoder"):
            self._init_history()

    def new_state(self, batch: int, *, origin: int = 0, history_device=None, index_device=None) -> HistoryState:
        return HistoryState(self, batch, origin, history_device=history_device, index_device=index_device)

    # -- encoding blocks from token ids --------------------------------------------------------------------

    def _encode(self, windows: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``windows [B, n, span]`` token ids (positions before the origin masked) -> ``[B, n, s + 1, width]``."""
        batch, count, span = windows.shape
        if count == 0:
            return torch.zeros(batch, 0, self.history.block + 1, self.encoder.width,
                               dtype=self.embed.weight.dtype, device=windows.device)
        embedded = self.embed(windows.reshape(batch * count, span))
        out = self.encoder(embedded, valid.reshape(batch * count, span))
        return out.view(batch, count, self.history.block + 1, -1)

    def _windows_from(self, starts: torch.Tensor, source, origin: int):
        """Token windows ``[B, n, span]`` starting at absolute positions ``starts [n]``, and their validity."""
        span = self.history.span
        positions = starts[:, None] + torch.arange(span, device=starts.device)          # [n, span]
        valid = positions >= origin
        tokens = source(positions.clamp(min=origin))
        return tokens, valid[None].expand(tokens.shape[0], -1, -1)

    # -- the chunk -------------------------------------------------------------------------------------------

    def _chunk_context(self, tokens, state, extra):
        mode = extra.pop("mode", None) or self.history.mode
        address = extra.pop("address", None)
        diagnose = extra.pop("diagnostics", False)
        if extra:
            raise TypeError(f"unexpected arguments {sorted(extra)}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode in ("oracle", "oracle_override") and address is None:
            raise ValueError("an oracle mode needs the evaluation fixture's address tensor")
        if mode in ("off", "learned") and address is not None:
            raise ValueError("an address is accepted only by the oracle diagnosis modes")
        if not isinstance(state, HistoryState):
            raise ValueError("ErsaHistoryLM runs on a HistoryState (model.new_state)")
        h, device = self.history, tokens.device
        batch, length = tokens.shape
        p0, s = state.position, h.block
        positions = p0 + torch.arange(length, device=device)
        # Blocks that complete inside this chunk: encoded now, from token ids (history before p0, chunk after).
        first_new, end_new = p0 // s, (p0 + length) // s
        new_ids = torch.arange(first_new, end_new, device=device)
        held_tokens = state.tokens

        def source(where):                                       # [n, span] absolute positions -> [B, n, span]
            out = tokens[:, (where - p0).clamp(min=0, max=length - 1)]
            old = where < p0
            if bool(old.any()):                                  # earlier positions come from the exact history
                safe = torch.where(old, where, torch.full_like(where, int(where[old].min())))
                out = torch.where(old[None], held_tokens.gather(safe).to(out.device), out)
            return out

        windows, valid = self._windows_from(new_ids * s - h.overlap, source, state.origin)
        new_states = self._encode(windows, valid) if mode != "off" else None
        context = {"mode": mode, "address": address, "positions": positions, "new_ids": new_ids, "first_new": first_new,
                   "new_states": new_states, "new_keys": {}, "diagnostics": {} if diagnose else None,
                   "source": source}
        return context

    def _extra_mix(self, index, u, state, context):
        if context is None or index not in self.read_layers:
            return None
        h, reader, mode = self.history, self.readers[str(index)], context["mode"]
        positions, new_ids = context["positions"], context["new_ids"]
        batch, length, _ = u.shape
        s, device = h.block, u.device
        if mode == "off":
            return None
        new_keys = reader.block_keys(context["new_states"]) if new_ids.numel() else \
            u.new_zeros(batch, 0, h.per_block, h.index_dim)
        context["new_keys"][index] = new_keys
        runtime = getattr(state, "runtime", None)
        if runtime is not None and runtime.scorer == "tiled":
            if mode != "learned":
                raise ValueError("the tiled scorer serves the learned mode; the oracle diagnosis modes use the "
                                 "reference path")
            return self._tiled_read(index, u, state, context, new_keys, runtime)
        ring = state.indexes[index]
        held = ring.ids >= 0                                    # empty slots are never scored
        ring_ids, ring_keys = (ring.ids, ring.keys) if bool(held.all()) else (ring.ids[held], ring.keys[:, held])
        ring_ids, ring_keys = ring_ids.to(device), ring_keys.to(device)
        if runtime is not None and runtime.key_grad_blocks and torch.is_grad_enabled() and ring_ids.numel():
            # R4 replay: the keys of recent resident blocks are recomputed from their original tokens WITH
            # gradient (same encoder, same pooling, same values for unchanged weights) instead of the stored,
            # detached ones. Scoring mathematics and eligibility are unchanged.
            span = runtime.key_grad_blocks
            replay = (ring_ids >= context["first_new"] - span) if span > 0 else torch.ones_like(ring_ids, dtype=torch.bool)
            replay = replay.nonzero(as_tuple=True)[0]
            if replay.numel():
                windows, ok = self._windows_from(ring_ids[replay] * s - h.overlap, context["source"], state.origin)
                fresh = reader.block_keys(self._encode(windows, ok))
                ring_keys = torch.index_copy(ring_keys.detach().to(fresh.dtype), 1, replay, fresh)
                runtime.count("replayed_key_blocks", int(replay.numel()))
        table_ids = torch.cat((ring_ids, new_ids))                                       # [N]
        keys = torch.cat((ring_keys.to(new_keys.dtype), new_keys), dim=1)                # [B, N, per, dim]
        resident = ring_ids.shape[0]
        ends = (table_ids + 1) * s - 1
        distance = positions[:, None] - ends[None, :]                                     # [T, N]
        complete = (positions + 1) // s                                                   # blocks complete at p
        eligible = (table_ids[None, :] >= 0) & (distance >= 0) & (table_ids[None, :] >= complete[:, None] - h.slots)
        scores = reader.scores(u, keys, distance).masked_fill(~eligible[None], float("-inf"))   # [B, T, N]

        address = context["address"]
        if not table_ids.numel():                               # nothing complete yet: the read is empty
            if mode in ("oracle", "oracle_override") and bool((address >= 0).any()):
                raise ValueError("an oracle address is not an eligible resident block for its query")
            if context["diagnostics"] is not None:
                none = torch.full((batch, length), -1, dtype=torch.long, device=device)
                context["diagnostics"][index] = {
                    "selected": torch.full((batch, length, h.topk), -1, dtype=torch.long, device=device),
                    "read_position": none, "null_weight": torch.ones(batch, length, dtype=u.dtype, device=device),
                    "scores": scores, "table_ids": table_ids}
            return torch.zeros_like(u)
        if mode in ("learned", "oracle_override"):
            # Deterministic exact top-K (R4): the R3 ranking, with exact ties resolved newer-block-first.
            picked, chosen = exact_topk(scores, table_ids, h.topk)
            valid = picked > float("-inf")
            if runtime is not None:
                runtime.count("records_scored", int(table_ids.numel()) * length)
        if mode in ("oracle", "oracle_override"):
            address = address.to(device)
            if address.shape != (batch, length):
                raise ValueError("address must be [batch, T] absolute block numbers (-1: none)")
            given = address >= 0
            match = table_ids[None, None, :] == address[..., None]                        # [B, T, N]
            found = match.any(-1)
            target = match.float().argmax(-1)
            ok = torch.gather(eligible.expand(batch, -1, -1), 2, target[..., None])[..., 0]
            if bool((given & ~(found & ok)).any()):
                raise ValueError("an oracle address is not an eligible resident block for its query")
            oracle_choice = torch.full((batch, length, h.topk), 0, dtype=torch.long, device=device)
            oracle_choice[..., 0] = target
            oracle_valid = torch.zeros(batch, length, h.topk, dtype=torch.bool, device=device)
            oracle_valid[..., 0] = given
            if mode == "oracle":
                chosen, valid = oracle_choice, oracle_valid
            else:
                chosen = torch.where(given[..., None], oracle_choice, chosen)
                valid = torch.where(given[..., None], oracle_valid, valid)
            picked = torch.gather(scores, 2, chosen)
            picked = torch.where(valid, picked, torch.zeros_like(picked))
        # -- reread: resident selected blocks are re-encoded from the token history; new ones are at hand -------
        flat = chosen[valid]
        reread = torch.unique(flat[flat < resident])                                      # table indices
        if reread.numel():
            windows, ok = self._windows_from(ring_ids[reread] * s - h.overlap, context["source"], state.origin)
            old_states = self._encode(windows, ok)
            if runtime is not None:
                runtime.count("blocks_reencoded", int(reread.numel()))
        else:
            old_states = u.new_zeros(batch, 0, s + 1, self.encoder.width)
        compact = torch.full((table_ids.shape[0],), 0, dtype=torch.long, device=device)
        compact[reread] = torch.arange(reread.numel(), device=device)
        compact[resident:] = reread.numel() + torch.arange(new_ids.numel(), device=device)
        # A zero block at the end stands in for invalid picks (they are masked out of the readout).
        states = torch.cat((old_states, context["new_states"], old_states.new_zeros(batch, 1, s + 1, self.encoder.width)),
                           dim=1)
        where = torch.where(valid, compact[chosen], torch.full_like(chosen, states.shape[1] - 1))   # [B, T, K]
        rows = torch.arange(batch, device=device)[:, None, None]
        block_keys, block_values = reader.project(states)                                  # [B, U + n + 1, s, g e]
        block_distance = torch.gather(distance[None].expand(batch, -1, -1), 2, chosen)
        out, weights = reader.read(u, block_keys[rows, where], block_values[rows, where], picked, valid,
                                   block_distance)
        if context["diagnostics"] is not None:
            selected = torch.where(valid, table_ids[chosen], torch.full_like(chosen, -1))
            mass = weights[..., :-1].mean(dim=2)                                            # [B, T, K * s]
            best = mass.argmax(-1)
            pick, offset = best // s, best % s
            source_block = torch.gather(selected, 2, pick[..., None])[..., 0]
            read_position = torch.where(source_block >= 0, source_block * s + offset, torch.full_like(best, -1))
            null = weights[..., -1].mean(dim=2)
            context["diagnostics"][index] = {"selected": selected, "read_position": read_position,
                                             "null_weight": null, "scores": scores, "table_ids": table_ids}
        return out

    def _tiled_read(self, index, u, state, context, new_keys, runtime):
        """R4 inference path: the R3 selection evaluated by ``tiled_select`` (exact, bounded scratch), resident
        selected blocks taken from the block cache or re-encoded from the token history, the R3 readout."""
        h, reader = self.history, self.readers[str(index)]
        s, device, batch = h.block, u.device, u.shape[0]
        ring, positions, new_ids = state.indexes[index], context["positions"], context["new_ids"]
        segments = ring_segments(ring) + [(new_keys, new_ids, context["first_new"])]
        chosen, picked = tiled_select(reader, u, segments, positions, block=s, slots=h.slots, k=h.topk,
                                      tile=runtime.tile, tile_elements=runtime.tile_elements,
                                      counters=runtime.counters)
        valid = chosen >= 0
        unique = torch.unique(chosen[valid])                                               # ascending block ids
        first_new = context["first_new"]
        resident = unique[unique < first_new]
        cache = runtime.cache if (runtime.cache is not None and not torch.is_grad_enabled()) else None
        if runtime.cache is not None and cache is None:
            runtime.cache.stats["bypassed"] += 1
        parts = []
        if resident.numel():
            ids = resident.tolist()
            found = {}
            if cache is not None:
                cache.check(self)
                found = {b: e for b in ids for e in [cache.get(b)] if e is not None}
            missing = [b for b in ids if b not in found]
            if missing:
                starts = torch.tensor(missing, dtype=torch.long, device=device) * s - h.overlap
                windows, ok = self._windows_from(starts, context["source"], state.origin)
                encoded = self._encode(windows, ok)
                runtime.count("blocks_reencoded", len(missing))
                for column, block in enumerate(missing):
                    found[block] = encoded[:, column]
                    if cache is not None:
                        cache.put(block, encoded[:, column])
            parts.append(torch.stack([found[b] for b in ids], dim=1))
        fresh = unique[unique >= first_new]
        if fresh.numel():
            parts.append(context["new_states"][:, fresh - first_new])
        parts.append(u.new_zeros(batch, 1, s + 1, self.encoder.width))                     # invalid picks
        states = torch.cat(parts, dim=1)
        where = torch.where(valid, torch.searchsorted(unique, chosen.clamp(min=0)), torch.full_like(chosen,
                                                                                                    unique.numel()))
        rows = torch.arange(batch, device=device)[:, None, None]
        block_keys, block_values = reader.project(states)
        distance = positions[None, :, None] - ((chosen + 1) * s - 1)
        picked_scores = torch.where(valid, picked, torch.zeros_like(picked))
        out, weights = reader.read(u, block_keys[rows, where], block_values[rows, where], picked_scores, valid,
                                   distance)
        if context["diagnostics"] is not None:
            mass = weights[..., :-1].mean(dim=2)
            best = mass.argmax(-1)
            pick, offset = best // s, best % s
            source_block = torch.gather(chosen, 2, pick[..., None])[..., 0]
            read_position = torch.where(source_block >= 0, source_block * s + offset, torch.full_like(best, -1))
            context["diagnostics"][index] = {"selected": chosen, "read_position": read_position,
                                             "null_weight": weights[..., -1].mean(dim=2), "scores": None,
                                             "table_ids": None}
        return out

    def _commit_context(self, tokens, state, context) -> None:
        if context is None:
            return
        new_ids = context["new_ids"]
        runtime = getattr(state, "runtime", None)
        cache = runtime.cache if runtime is not None else None
        if cache is not None and new_ids.numel() and context["new_states"] is not None \
                and not torch.is_grad_enabled():
            cache.check(self)
            for column, block in enumerate(new_ids.tolist()):          # fresh blocks are likely to be read next
                cache.put(block, context["new_states"][:, column])
        if cache is not None and new_ids.numel():
            cache.purge_below(int(new_ids[-1]) + 1 - self.history.slots)   # outside every future query's horizon
        if new_ids.numel():
            for index in self.read_layers:
                keys = context["new_keys"].get(index)
                if keys is None:                                  # mode "off": keys still have to be kept current
                    states = context["new_states"]
                    if states is None:
                        windows, valid = self._windows_from(new_ids * self.history.block - self.history.overlap,
                                                            context["source"], state.origin)
                        states = self._encode(windows, valid)
                        context["new_states"] = states
                    keys = self.readers[str(index)].block_keys(states)
                state.indexes[index].write(int(new_ids[0]), keys)
        state.tokens.append(tokens)


def addressing_loss(diagnostics: dict, gold_block: torch.Tensor) -> torch.Tensor:
    """TRAINING ONLY: cross-entropy of each read layer's index scores against the generator's source block.

    ``gold_block [B, T]`` absolute block numbers, ``-1`` where the position has no labelled source. Positions
    whose source block is not eligible (still open, or outside the horizon) are skipped. Never an input of the
    model; evaluation passes no labels.
    """
    total, count = None, 0
    for record in diagnostics.values():
        scores, table = record["scores"], record["table_ids"]
        if scores is None:
            raise ValueError("the addressing loss needs the reference scorer's full score table")
        match = (table[None, None, :] == gold_block[..., None]) & (gold_block[..., None] >= 0)
        target = match.float().argmax(-1)
        usable = match.any(-1)
        usable = usable & torch.isfinite(torch.gather(scores, 2, target[..., None])[..., 0])
        if not bool(usable.any()):
            continue
        loss = F.cross_entropy(scores[usable], target[usable])
        total = loss if total is None else total + loss
        count += 1
    if total is None:
        return torch.zeros((), dtype=torch.float32)
    return total / count
