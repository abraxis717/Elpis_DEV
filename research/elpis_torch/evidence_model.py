"""R5: learned coarse-to-fine evidence retrieval over a bounded 1,048,576-token exact history, beside an
UNCHANGED ERSA/CGC core (``ersa_model.ErsaLM``, used through its extension hooks; no core code is modified).

    exact history   ``EvidenceRing``: a FIFO ring of exactly ``capacity`` token ids with session / epoch / slot /
                    generation identity; stale addresses are refused (``StaleAddressError``).
    coarse index    each completed block gets ``per_block`` learned descriptors (unit vectors of ``key_dim``),
                    computed ONCE from its exact tokens by a small causal block encoder and stored in the ring's
                    bounded index (detached; bfloat16 on CUDA by default). A query from the core's hidden state
                    scans every live block (one vectorized product) and keeps the ``coarse_top`` best, with the
                    deterministic rule of R4 (score descending, ties to the newer block).
    exact reread    the candidates' ORIGINAL token ids are reread from the ring (validated addresses, positions
                    preserved) and re-encoded. No per-token key/value cache exists.
    fine reranker   token-level matching of a second query against the reread candidates, plus a learned
                    supersession feature (how strongly a NEWER candidate also matches) and the coarse score: the
                    top ``evidence`` blocks are selected (``fine_top`` are exposed with absolute ranges).
    injection       a deterministic, bounded working context: the ``evidence`` selected blocks (ranked by
                    position) plus the block containing the query (its exact tokens up to the query: recent local
                    context), attended by the query state with a null option. The result is added to the LAST
                    block's mixer output.

Placement. The read sits in the last block, after that block's ERSA update, so it changes only the logits of
its own position: no recurrent state and no ring depends on it. Computing it only where logits are requested
(``logit_positions``) is therefore EXACTLY the dense computation restricted to those positions (tested).

Modes: ``off`` (core only), ``learned`` (autonomous; labels, if passed, are used for metrics only), ``oracle``
(DIAGNOSIS: the gold block is forced into the selected evidence). Reranking ablations: ``learned``; ``coarse``
(the coarse order picks the evidence, reread kept); ``all`` (every coarse candidate is reread and injected).
Losses (training only): ``coarse_source_loss`` and ``fine_evidence_loss`` from the generator's evidence
positions, beside the answer loss computed by the caller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import ConfigError
from .context_runtime import exact_topk
from .ersa_model import ErsaConfig, ErsaLM, ErsaState
from .evidence_ring import EvidenceRing
from .history_model import _Channel
from .model import RMSNorm, _seeded_normal

MODES = ("off", "learned", "oracle")
RERANK = ("learned", "coarse", "all")
FILL = -1.0e4                       # finite "masked" value where a gradient may pass (avoids NaN in backward)


@dataclass(frozen=True)
class EvidenceConfig:
    block: int = 32                 # tokens per addressable block
    capacity: int = 1 << 20         # exact ring tokens; live blocks <= capacity // block
    per_block: int = 4              # coarse descriptors per block (a block's coarse match is their max)
    key_dim: int = 64
    key_dtype: str = "auto"         # stored descriptors: "auto" = bfloat16 on CUDA, float32 elsewhere
    coarse_top: int = 64
    evidence: int = 4               # reranked blocks injected into the working context
    fine_top: int = 8               # reranked blocks exposed (ranges, scores, hit/miss)
    overlap: int = 8                # left context the block encoder sees (never injected)
    encoder_dim: int = 64
    encoder_heads: int = 2
    encoder_layers: int = 2
    fine_dim: int = 32
    read_heads: int = 2
    read_head_dim: int = 32
    query_tile: int = 128           # most queries scored together
    tile_elements: int = 1 << 22    # bound of a tile's [batch, queries, blocks, per_block] coarse scratch
    mode: str = "learned"
    rerank: str = "learned"

    def __post_init__(self):
        for name in ("block", "capacity", "per_block", "key_dim", "coarse_top", "evidence", "fine_top", "overlap",
                     "encoder_dim", "encoder_heads", "encoder_layers", "fine_dim", "read_heads", "read_head_dim",
                     "query_tile", "tile_elements"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ConfigError(f"{name} must be a positive integer")
        if self.capacity % self.block:
            raise ConfigError("capacity must be a multiple of block")
        if not self.evidence <= self.fine_top <= self.coarse_top:
            raise ConfigError("evidence <= fine_top <= coarse_top is required")
        if self.encoder_dim % self.encoder_heads or (self.encoder_dim // self.encoder_heads) % 2:
            raise ConfigError("encoder_dim must be an even multiple of encoder_heads")
        if self.key_dtype not in ("auto", "float32", "bfloat16", "float16"):
            raise ConfigError("key_dtype must be auto, float32, bfloat16 or float16")
        if self.mode not in MODES or self.rerank not in RERANK:
            raise ConfigError(f"mode in {MODES}, rerank in {RERANK}")

    @property
    def slots(self) -> int:
        return self.capacity // self.block

    @property
    def span(self) -> int:
        return self.overlap + self.block

    def replace(self, **changes) -> "EvidenceConfig":
        import dataclasses
        return dataclasses.replace(self, **changes)


def _gather_rows(x: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``x [B, U, ...]``, ``rows [B, *idx]`` -> ``[B, *idx, ...]``."""
    batch = torch.arange(x.shape[0], device=x.device).view(-1, *([1] * (rows.ndim - 1)))
    return x[batch, rows]


class _EncoderLayer(nn.Module):
    def __init__(self, m, width: int, heads: int, dtype):
        super().__init__()
        self.heads = heads
        self.attn_norm = RMSNorm(width, m.norm_eps, dtype)
        self.qkv = nn.Linear(width, 3 * width, bias=False, dtype=dtype)
        self.attn_out = nn.Linear(width, width, bias=False, dtype=dtype)
        self.channel_norm = RMSNorm(width, m.norm_eps, dtype)
        self.channel = _Channel(m, dtype)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        n, span, d = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(n, span, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        scores = (q @ k.transpose(-1, -2)) * (d // self.heads) ** -0.5
        weights = torch.softmax(scores.masked_fill(~allowed[:, None], float("-inf")), dim=-1)
        x = x + self.attn_out((weights @ v).transpose(1, 2).reshape(n, span, d))
        return x + self.channel(self.channel_norm(x))


class CausalBlockEncoder(nn.Module):
    """Encodes a block from its exact token ids: the core's token embeddings (read, as in R3) plus a learned
    zero-initialized correction ``delta`` (so the evidence path can adapt while the core is frozen), a learned
    position inside the span, ``encoder_layers`` CAUSAL attention + channel layers. A token's state depends only on
    the tokens at or before it inside the span, so a partial block's states equal the complete block's states
    on the shared prefix (the local slot and the later index record agree). Output: ``[N, s + 1, width]``, the
    block's tokens preceded by the token before the block."""

    def __init__(self, m, e: EvidenceConfig, dtype):
        super().__init__()
        width = e.encoder_dim
        if width != m.d_model:
            raise ConfigError("encoder_dim must equal d_model (the encoder reads the core's embeddings)")
        self.block, self.span, self.width = e.block, e.span, width
        self.delta = nn.Embedding(m.vocab_size, width, dtype=dtype)
        # causal n-gram mixing: each token's input also carries its three predecessors (statements are 2-4 tokens)
        self.shift = nn.ModuleList(nn.Linear(width, width, bias=False, dtype=dtype) for _ in range(3))
        self.position = nn.Parameter(torch.zeros(e.span, width, dtype=dtype))
        self.layers = nn.ModuleList(_EncoderLayer(m, width, e.encoder_heads, dtype) for _ in range(e.encoder_layers))
        self.out_norm = RMSNorm(width, m.norm_eps, dtype)
        self.register_buffer("causal", torch.tril(torch.ones(e.span, e.span, dtype=torch.bool)), persistent=False)

    def forward(self, embedded: torch.Tensor, windows: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``embedded [N, span, d]`` (the core's embeddings of ``windows [N, span]``), ``valid [N, span]`` ->
        ``[N, s + 1, width]`` (invalid positions zero)."""
        x = (embedded + self.delta(windows)) * valid[..., None]
        mixed = x
        for back, linear in enumerate(self.shift, start=1):
            mixed = mixed + linear(F.pad(x[:, :-back], (0, 0, back, 0)))
        x = mixed + self.position
        eye = torch.eye(self.span, dtype=torch.bool, device=windows.device)
        allowed = (self.causal[None] & valid[:, None, :]) | eye[None]       # a token always sees itself
        for layer in self.layers:
            x = layer(x, allowed)
        tail = valid[:, -(self.block + 1):, None]
        return self.out_norm(x[:, -(self.block + 1):]) * tail


class EvidenceHead(nn.Module):
    """Coarse descriptors and query, the fine reranker and the readout of the injected working context."""

    def __init__(self, m, e: EvidenceConfig, dtype):
        super().__init__()
        d, w, f, r = m.d_model, e.encoder_dim, e.fine_dim, e.read_head_dim
        self.e = e
        # coarse
        self.pool = nn.Linear(w, e.per_block, bias=False, dtype=dtype)
        self.descriptor = nn.Linear(w, e.key_dim, bias=False, dtype=dtype)
        self.coarse_q = nn.Linear(d, e.key_dim, bias=False, dtype=dtype)
        self.coarse_scale = nn.Parameter(torch.full((1,), math.log(10.0), dtype=dtype))
        self.coarse_recency = nn.Parameter(torch.full((1,), -4.0, dtype=dtype))
        # fine
        self.fine_q = nn.Linear(d, f, bias=False, dtype=dtype)
        self.fine_k = nn.Linear(w, f, bias=False, dtype=dtype)
        self.fine_prev = nn.Linear(w, f, bias=False, dtype=dtype)
        self.fine_position = nn.Parameter(torch.zeros(e.block, f, dtype=dtype))
        self.fine_scale = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.coarse_weight = nn.Parameter(torch.full((1,), 0.1, dtype=dtype))
        self.sup_threshold = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.sup_sharpness = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.sup_weight = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.local_bias = nn.Parameter(torch.zeros(1, dtype=dtype))
        # readout
        self.heads, self.head_dim = e.read_heads, r
        self.q = nn.Linear(d, e.read_heads * r, bias=False, dtype=dtype)
        self.k = nn.Linear(w, r, bias=False, dtype=dtype)
        self.k_prev = nn.Linear(w, r, bias=False, dtype=dtype)
        self.v = nn.Linear(w, r, bias=False, dtype=dtype)
        self.key_position = nn.Parameter(torch.zeros(e.block, r, dtype=dtype))
        self.key_rank = nn.Parameter(torch.zeros(e.evidence + 1, r, dtype=dtype))     # newest 0 (the local block)
        self.select_scale = nn.Parameter(torch.ones(1, dtype=dtype))
        self.null = nn.Parameter(torch.zeros(e.read_heads, dtype=dtype))
        self.out = nn.Linear(e.read_heads * r, d, bias=False, dtype=dtype)

    # coarse ---------------------------------------------------------------------------------------------------
    def describe(self, states: torch.Tensor) -> torch.Tensor:
        """``[..., s + 1, w]`` encoded block -> ``[..., per_block, key_dim]`` unit descriptors."""
        states = states[..., 1:, :]
        weights = torch.softmax(self.pool(states), dim=-2)
        return F.normalize(self.descriptor(weights.transpose(-1, -2) @ states), dim=-1)

    def coarse_query(self, uq: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.coarse_q(uq), dim=-1)

    @staticmethod
    def match(cq: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """``cq [B, Q, D]``, ``keys [B, N, P, D]`` -> ``[B, Q, N]`` (max over a block's descriptors)."""
        if keys.shape[1] == 0:
            return cq.new_zeros(cq.shape[0], cq.shape[1], 0)
        return torch.einsum("bqd,bnpd->bqnp", cq, keys.to(cq.dtype)).amax(-1)

    def coarse_score(self, match: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        slope = F.softplus(self.coarse_recency)
        return match * self.coarse_scale.exp() - slope * torch.log1p(distance.clamp(min=0).to(match.dtype))

    # fine and readout keys (once per encoded block) ----------------------------------------------------------
    def fine_keys(self, states: torch.Tensor) -> torch.Tensor:
        return self.fine_k(states[..., 1:, :]) + self.fine_prev(states[..., :-1, :]) + self.fine_position

    def read_keys(self, states: torch.Tensor):
        return (self.k(states[..., 1:, :]) + self.k_prev(states[..., :-1, :]) + self.key_position,
                self.v(states[..., 1:, :]))

    def read(self, uq, rk_all, rv_all, rows, tok_valid, block_valid, prior, rank):
        """Readout over the working context. ``rows [B, Q, M]`` index the encoded blocks ``rk_all/rv_all
        [B, U, s, r]``; ``tok_valid [B, Q, M, s]``; ``block_valid``, ``prior`` (block scores) and ``rank``
        (newest 0) ``[B, Q, M]``. Returns ``([B, Q, d], weights [B, Q, heads, M * s + 1])`` (last: null)."""
        heads, r = self.heads, self.head_dim
        batch, count, _ = uq.shape
        m, s = rows.shape[-1], rk_all.shape[2]
        q = self.q(uq).view(batch, count, heads, r)
        logits = torch.gather(torch.einsum("bqhr,busr->bqhus", q, rk_all), 3,
                              rows[:, :, None, :, None].expand(batch, count, heads, m, s))
        by_rank = torch.einsum("bqhr,kr->bqhk", q, self.key_rank)                      # small; gathered by rank
        by_rank = torch.gather(by_rank, 3, rank.clamp(max=self.key_rank.shape[0] - 1)[:, :, None, :]
                               .expand(batch, count, heads, m))
        logits = (logits + by_rank[..., None]) * r ** -0.5
        prior = torch.log_softmax(prior.masked_fill(~block_valid, FILL), dim=-1)
        prior = torch.where(block_valid, prior, torch.zeros_like(prior))
        logits = logits + self.select_scale * prior[:, :, None, :, None]
        allowed = (tok_valid & block_valid[..., None])[:, :, None]
        logits = logits.masked_fill(~allowed, float("-inf")).reshape(batch, count, heads, m * s)
        null = self.null.view(1, 1, heads, 1).expand(batch, count, heads, 1).to(logits.dtype)
        weights = torch.softmax(torch.cat((logits, null), dim=-1), dim=-1)
        taken = weights[..., :-1].view(batch, count, heads, m, s)
        # scatter the attention onto the encoded blocks (a gather in backward), then one product with the values
        spread = taken.new_zeros(batch, count, heads, rk_all.shape[1], s).scatter_add(
            3, rows[:, :, None, :, None].expand(batch, count, heads, m, s), taken)
        mixed = torch.einsum("bqhus,busr->bqhr", spread, rv_all)
        return self.out(mixed.reshape(batch, count, heads * r)), weights


class EvidenceState(ErsaState):
    """ERSA state plus the exact ring and its index. Fixed size from creation."""

    def __init__(self, model: "EvidenceLM", batch: int, origin: int = 0, *, ring_device=None):
        super().__init__(model, batch)
        e = model.evidence
        if origin < 0 or origin % e.block:
            raise ValueError("the stream origin must be a non-negative multiple of the evidence block")
        self.origin = self.position = origin
        if self.window is not None:
            self.window.origin = self.window.position = origin
        device = self.device if ring_device is None else torch.device(ring_device)
        self.ring = EvidenceRing(batch, capacity=e.capacity, block=e.block, per_block=e.per_block, dim=e.key_dim,
                                 origin=origin, overlap=e.overlap, key_dtype=model.key_dtype(device), device=device)
        self.shuffle = None             # ABLATION ONLY: a seed; descriptors are attributed to permuted blocks
        self.timing = None              # a dict to accumulate per-stage seconds (measurement only)

    def reset(self) -> None:
        super().reset()
        self.position = self.origin
        if self.window is not None:
            self.window.origin = self.window.position = self.origin
        self.ring.reset()

    def nbytes(self) -> dict:
        out = super().nbytes()
        out.pop("total")
        out.update(self.ring.nbytes())
        out["total"] = sum(out.values())
        return out


class EvidenceLM(ErsaLM):
    """``ErsaLM`` (unchanged) plus the R5 evidence path in its last block."""

    def __init__(self, config: ErsaConfig, evidence: EvidenceConfig):
        if not isinstance(evidence, EvidenceConfig):
            raise TypeError("EvidenceLM expects an EvidenceConfig")
        self.evidence = evidence
        super().__init__(config)
        m, dtype = config.model, getattr(torch, config.model.dtype)
        self.read_layer = m.n_layers - 1
        self.encoder = CausalBlockEncoder(m, evidence, dtype)
        self.lookup = EvidenceHead(m, evidence, dtype)          # (``head`` is the core's output head)
        self._init_evidence()

    @torch.no_grad()
    def _init_evidence(self) -> None:
        m = self.config
        residual = m.init_std / (2 * m.n_layers) ** 0.5
        fixed = {"coarse_scale": math.log(10.0), "coarse_recency": -4.0, "coarse_weight": 0.1, "select_scale": 1.0,
                 "sup_threshold": 2.0, "sup_weight": -3.0}         # supersession starts weak (softplus(-3) ~ 0.05)
        for name, parameter in self.named_parameters():
            if not name.startswith(("encoder.", "lookup.")):
                continue
            short = name.rsplit(".", 1)[-1]
            if short in fixed:
                parameter.fill_(fixed[short])
            elif short in ("fine_scale", "sup_sharpness", "local_bias", "null") or name.endswith("delta.weight"):
                parameter.zero_()
            elif ".shift." in name:                                  # predecessors start as decaying copies
                back = int(name.split(".shift.")[1].split(".")[0]) + 1
                parameter.copy_(torch.eye(parameter.shape[0], dtype=parameter.dtype) * 0.5 ** back)
            elif name.endswith("norm.weight"):
                parameter.fill_(1.0)
            elif name.endswith("cgc_gate"):
                parameter.fill_(m.cgc_gate_init)
            elif name.endswith(("lookup.out.weight", "attn_out.weight", "cgc.proj_out.weight", "ffn.out.weight")):
                parameter.copy_(_seeded_normal(m.seed, name, parameter.shape, residual, parameter.dtype))
            elif name.endswith("cgc.core.weight"):
                std = m.cgc_init_gain * (m.cgc_bottleneck or m.d_model) ** -0.5
                parameter.copy_(_seeded_normal(m.seed, name, parameter.shape, std, parameter.dtype))
            elif name.endswith("cgc.proj_in.weight"):
                parameter.copy_(_seeded_normal(m.seed, name, parameter.shape, m.d_model ** -0.5, parameter.dtype))
            else:
                parameter.copy_(_seeded_normal(m.seed, name, parameter.shape, m.init_std, parameter.dtype))

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, "encoder"):
            self._init_evidence()

    def key_dtype(self, device) -> torch.dtype:
        name = self.evidence.key_dtype
        if name == "auto":
            return torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
        return getattr(torch, name)

    def new_state(self, batch: int, *, origin: int = 0, ring_device=None) -> EvidenceState:
        return EvidenceState(self, batch, origin, ring_device=ring_device)

    def forward(self, tokens, state, *, return_pre=False, logit_positions=None, **extra):
        """``ErsaLM.forward``; the evidence read is computed exactly where logits are requested."""
        extra["_queries"] = logit_positions
        return super().forward(tokens, state, return_pre=return_pre, logit_positions=logit_positions, **extra)

    # -- blocks from exact token ids ---------------------------------------------------------------------------
    def _encode(self, windows: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``windows [B, n, span]`` (``valid [n, span]`` or ``[B, n, span]``) -> ``[B, n, s + 1, width]``."""
        batch, count, span = windows.shape
        if count == 0:
            return torch.zeros(batch, 0, self.evidence.block + 1, self.encoder.width,
                               dtype=self.embed.weight.dtype, device=windows.device)
        valid = valid.expand(batch, count, span) if valid.ndim == 2 else valid
        flat = windows.reshape(batch * count, span)
        out = self.encoder(self.embed(flat), flat, valid.reshape(batch * count, span))
        return out.view(batch, count, self.evidence.block + 1, -1)

    # -- the chunk ---------------------------------------------------------------------------------------------
    def _chunk_context(self, tokens, state, extra):
        queries = extra.pop("_queries", None)
        mode = extra.pop("mode", None) or self.evidence.mode
        rerank = extra.pop("rerank", None) or self.evidence.rerank
        coarse_top = extra.pop("coarse_top", None) or self.evidence.coarse_top
        labels = extra.pop("evidence", None)
        stale = extra.pop("stale", None)
        teacher = bool(extra.pop("teacher", False))
        diagnose = extra.pop("diagnostics", False)          # True: losses and metrics; "losses": losses only
        if extra:
            raise TypeError(f"unexpected arguments {sorted(extra)}")
        if mode not in MODES or rerank not in RERANK:
            raise ValueError(f"mode must be one of {MODES}, rerank one of {RERANK}")
        if not isinstance(state, EvidenceState):
            raise ValueError("EvidenceLM runs on an EvidenceState (model.new_state)")
        if (mode == "oracle" or teacher) and labels is None:
            raise ValueError("the oracle diagnosis mode and teacher insertion need evidence labels")
        if teacher and not torch.is_grad_enabled():
            raise ValueError("teacher insertion is a training device; evaluation never inserts the gold")
        e, device = self.evidence, tokens.device
        batch, length = tokens.shape
        p0, s = state.position, e.block
        if queries is None:
            q_index = torch.arange(length, device=device)
        else:
            q_index = torch.as_tensor(queries, dtype=torch.long, device=device).reshape(-1)
            if q_index.numel() and (int(q_index.min()) < 0 or int(q_index.max()) >= length
                                    or bool((q_index[1:] <= q_index[:-1]).any())):
                raise ValueError("logit_positions must be increasing chunk indices")
        if labels is not None and tuple(labels.shape) != (batch, q_index.numel()):
            raise ValueError("evidence labels must be [batch, queries] absolute positions (-1: none)")
        ring = state.ring
        first, last = p0 // s, (p0 + length - 1) // s
        complete = (p0 + length) // s - first                     # touched blocks completed by this chunk

        def source(where):                                       # [n, span] absolute -> ([B, n, span], valid)
            inside = (where >= p0) & (where < p0 + length)
            before = (where < p0) & (where >= ring.oldest)
            out = tokens[:, (where - p0).clamp(min=0, max=length - 1)]
            if bool(before.any()):
                safe = torch.where(before, where, torch.full_like(where, int(where[before].min())))
                out = torch.where(before[None], ring.gather(safe).to(device), out)
            return out, inside | before

        touched = torch.arange(first, last + 1, device=device)
        starts = touched * s - e.overlap
        windows, valid = source(starts[:, None] + torch.arange(e.span, device=device))
        states = self._encode(windows, valid) if mode != "off" else None
        return {"mode": mode, "rerank": rerank, "coarse_top": coarse_top, "labels": labels, "stale": stale,
                "teacher": teacher, "q_index": q_index, "q_pos": p0 + q_index, "first": first,
                "complete": complete, "states": states, "windows": windows, "valid": valid, "new_desc": None,
                "losses": {}, "diagnostics": {} if diagnose else None, "full": diagnose is True}

    def _extra_mix(self, index, u, state, context):
        if context is None or index != self.read_layer or context["mode"] == "off":
            return None
        e, states, complete = self.evidence, context["states"], context["complete"]
        context["new_desc"] = self.lookup.describe(states[:, :complete]) if complete else \
            u.new_zeros(u.shape[0], 0, e.per_block, e.key_dim)
        q_index = context["q_index"]
        if q_index.numel() == 0:
            return None
        read = self._retrieve(u, state, context)
        return u.new_zeros(u.shape).index_copy(1, q_index, read.to(u.dtype))

    def _commit_context(self, tokens, state, context) -> None:
        if context is None:
            return
        state.ring.append(tokens)
        complete = context["complete"]
        if complete:
            desc = context["new_desc"]
            if desc is None:                                     # mode "off": the index is still kept current
                states = self._encode(context["windows"][:, :complete], context["valid"][:complete])
                desc = self.lookup.describe(states)
            state.ring.write(context["first"], desc)

    # -- coarse scan, exact reread, rerank, injection -----------------------------------------------------------
    def _retrieve(self, u, state, context) -> torch.Tensor:
        e, head, ring = self.evidence, self.lookup, state.ring
        s, device = e.block, u.device
        batch = u.shape[0]
        first, states = context["first"], context["states"]
        q_index, q_pos = context["q_index"], context["q_pos"]
        lo, hi = ring.live()
        resident = torch.arange(lo, hi, device=device)                         # written, fully held blocks
        first_slot, count_live = lo % ring.slots, hi - lo
        if first_slot + count_live <= ring.slots:                              # contiguous: a view, no copy
            res_keys = ring.keys[:, first_slot:first_slot + count_live]
        else:                                                                  # wrapped: two views joined
            res_keys = torch.cat((ring.keys[:, first_slot:], ring.keys[:, :first_slot + count_live - ring.slots]), 1)
        res_keys = res_keys.to(device=device, dtype=u.dtype)
        if torch.is_grad_enabled() and res_keys.untyped_storage().data_ptr() == ring.keys.untyped_storage().data_ptr():
            res_keys = res_keys.clone()        # saved for backward: the commit overwrites the ring in place
        if state.shuffle is not None and resident.numel() > 1:                 # ABLATION: identity destroyed
            perm = torch.randperm(resident.numel(), generator=torch.Generator().manual_seed(state.shuffle + lo))
            res_keys = res_keys[:, perm.to(device)]
        new_ids = first + torch.arange(context["complete"], device=device)
        table = torch.cat((resident, new_ids, torch.full((1,), -1, dtype=torch.long, device=device)))   # + sentinel
        ends = (table + 1) * s - 1
        # per-chunk projections of the chunk's own blocks (resident candidates are added per tile)
        states = context["states"]
        context["chunk_keys"] = (head.fine_keys(states), *head.read_keys(states))
        tile = max(1, min(e.query_tile, e.tile_elements // max(1, batch * table.numel() * e.per_block)))
        reads, diags = [], []
        for start in range(0, q_index.numel(), tile):
            sl = slice(start, start + tile)
            out, diag = self._retrieve_tile(u[:, q_index[sl]], q_pos[sl], state, context, table, ends, res_keys,
                                            resident.numel(), sl)
            reads.append(out)
            diags.append(diag)
        if context["diagnostics"] is not None:
            if diags and diags[0] is not None:
                context["diagnostics"].update({k: torch.cat([d[k] for d in diags], dim=1) for k in diags[0]})
            if context["losses"]:
                context["diagnostics"]["losses"] = collect_losses(context["losses"])
        return torch.cat(reads, dim=1)

    def _retrieve_tile(self, uq, pos, state, context, table, ends, res_keys, n_res, sl):
        e, head, ring = self.evidence, self.lookup, state.ring
        s, device = e.block, uq.device
        batch, count = uq.shape[0], pos.numel()
        first, states = context["first"], context["states"]
        mode, rerank = context["mode"], context["rerank"]
        k_top = min(context["coarse_top"], table.numel())
        local = pos // s                                                        # the query's own block
        clock = _Clock(state.timing)
        # 1. coarse scan over every live block (vectorized), deterministic top-K
        cq = head.coarse_query(uq)
        match = torch.cat((head.match(cq, res_keys), head.match(cq, context["new_desc"]),
                           cq.new_zeros(batch, count, 1)), dim=2)
        coarse = head.coarse_score(match, pos[:, None] - ends[None, :])
        start = (table * s - e.overlap).clamp(min=ring.origin)                     # a live block's whole span
        eligible = (table[None] >= 0) & (table[None] < local[:, None]) & (start[None] >= pos[:, None] + 1 - e.capacity)
        coarse = coarse.masked_fill(~eligible[None], float("-inf"))
        values, picks = exact_topk(coarse.detach(), table, k_top)
        cand = torch.where(torch.isfinite(values), table[picks], torch.full_like(picks, -1))   # [B, Q, K]
        cvals = torch.gather(coarse, 2, picks)
        clock.lap("coarse_scan")
        # labels (training and evaluation metrics; never a selection input except teacher / oracle)
        labels = context["labels"]
        gold = hist = gcol = None
        if labels is not None:
            gpos = labels[:, sl].to(device)
            gold = torch.where(gpos >= 0, gpos // s, torch.full_like(gpos, -1))
            in_res = (gold >= table[0] if n_res else torch.zeros_like(gold, dtype=torch.bool)) & (gold < first)
            gcol = torch.where(in_res, gold - (table[0] if n_res else 0), n_res + gold - first)
            ok_col = (gcol >= 0) & (gcol < table.numel())
            gcol = torch.where(ok_col, gcol, torch.zeros_like(gcol))
            hist = (gold >= 0) & ok_col & (gold < local[None]) & torch.gather(
                eligible[None].expand(batch, -1, -1), 2, gcol[..., None])[..., 0] & (table[gcol] == gold)
            hit_coarse = (cand == gold[..., None]).any(-1) & hist
            if context["teacher"] or mode == "oracle":                          # gold into the candidates
                insert = hist & ~hit_coarse
                if bool(insert.any()):
                    cand = cand.clone()
                    cand[..., -1] = torch.where(insert, gold, cand[..., -1])
                    gscore = torch.gather(coarse, 2, gcol[..., None])[..., 0]
                    cvals = torch.cat((cvals[..., :-1], torch.where(insert, gscore, cvals[..., -1])[..., None]), -1)
        valid_c = cand >= 0
        # 2. exact reread of resident candidates (validated identity), encoding; chunk blocks are at hand
        flat = cand[valid_c]
        needed = torch.unique(flat)
        old = needed[needed < first]
        if old.numel():
            pairs = valid_c & (cand < first)
            ring._check(cand[pairs], pos[None, :, None].expand_as(cand)[pairs])            # every (block, query)
            tokens, ok = ring.reread(old.to(ring.device), e.overlap)
            old_states = self._encode(tokens.to(device), ok.to(device))
        else:
            old_states = states.new_zeros(batch, 0, s + 1, states.shape[-1])
        block_ids = torch.cat((old, first + torch.arange(states.shape[1], device=device)))
        enc = torch.cat((old_states, states), dim=1)                                        # [B, U, s + 1, w]
        clock.lap("exact_reread")
        rows = torch.searchsorted(block_ids, cand.clamp(min=0))
        rows = torch.where(valid_c, rows, torch.zeros_like(rows))
        lrow = (old.numel() + local - first)[None].expand(batch, -1)                        # [B, Q]
        tok_pos = local[:, None] * s + torch.arange(s, device=device)                       # local tokens <= query
        local_tok = (tok_pos <= pos[:, None])[None].expand(batch, -1, -1)
        # 3. fine reranking: token-level match, supersession by newer matching candidates, coarse score
        fq = head.fine_q(uq)
        chunk_fk, chunk_rk, chunk_rv = context["chunk_keys"]
        if old.numel():
            old_fk, (old_rk, old_rv) = head.fine_keys(old_states), head.read_keys(old_states)
            fk = torch.cat((old_fk, chunk_fk), dim=1)
            rk, rv = torch.cat((old_rk, chunk_rk), dim=1), torch.cat((old_rv, chunk_rv), dim=1)
        else:
            fk, rk, rv = chunk_fk, chunk_rk, chunk_rv                                       # [B, U, s, .]
        tl = torch.einsum("bqf,butf->bqut", fq, fk) * (head.fine_scale.exp() * fq.shape[-1] ** -0.5)
        cand_tl = torch.gather(tl, 2, rows[..., None].expand(-1, -1, -1, s))
        # log-mean-exp over a block's tokens: a soft maximum that is ~0 when nothing matches
        m_c = torch.logsumexp(cand_tl.masked_fill(~valid_c[..., None], FILL), dim=-1) - math.log(s)
        loc_tl = torch.gather(tl, 2, lrow[..., None, None].expand(-1, -1, 1, s))[:, :, 0]
        m_l = torch.logsumexp(loc_tl.masked_fill(~local_tok, FILL), dim=-1) - torch.log(local_tok.sum(-1).to(tl.dtype))
        ids_all = torch.cat((cand, local[None, :, None].expand(batch, -1, 1)), dim=-1)
        m_all = torch.cat((m_c, m_l[..., None]), dim=-1)
        ok_all = torch.cat((valid_c, torch.ones_like(m_l[..., None], dtype=torch.bool)), dim=-1)
        strength = torch.sigmoid(head.sup_sharpness.exp() * (m_all - head.sup_threshold)) * ok_all
        newer = ids_all[..., None, :] > cand[..., :, None]                                  # [B, Q, K, K + 1]
        superseded = (newer * strength[..., None, :]).sum(-1)
        if rerank == "learned":
            fine = m_c + head.coarse_weight * cvals.masked_fill(~valid_c, 0.0) - F.softplus(head.sup_weight) * superseded
        else:
            fine = cvals
        fine = fine.masked_fill(~valid_c, float("-inf"))
        width = k_top if rerank == "all" else min(e.evidence, k_top)
        sel_vals, sel_pos = exact_topk(fine.detach(), cand.clamp(min=0), width)
        sel = torch.where(torch.isfinite(sel_vals), torch.gather(cand, 2, sel_pos), torch.full_like(sel_pos, -1))
        sel_vals = torch.gather(fine, 2, sel_pos)
        if mode == "oracle" and gold is not None:                                           # DIAGNOSIS ONLY
            missing = hist & ~(sel == gold[..., None]).any(-1)
            if bool(missing.any()):
                gpick = (cand == gold[..., None]).float().argmax(-1)
                sel = sel.clone()
                sel[..., -1] = torch.where(missing, gold, sel[..., -1])
                sel_pos = sel_pos.clone()
                sel_pos[..., -1] = torch.where(missing, gpick, sel_pos[..., -1])
                sel_vals = torch.gather(fine.masked_fill(~valid_c, FILL), 2, sel_pos)
        clock.lap("rerank")
        # 4. the working context: selected evidence (by position) + the local block; readout; injection
        sel_ok = sel >= 0
        blocks = torch.cat((sel, local[None, :, None].expand(batch, -1, 1)), dim=-1)
        brow = torch.cat((torch.where(sel_ok, torch.searchsorted(block_ids, sel.clamp(min=0)),
                                      torch.zeros_like(sel)), lrow[..., None]), dim=-1)
        bvalid = torch.cat((sel_ok, torch.ones_like(sel_ok[..., :1])), dim=-1)
        prior = torch.cat((sel_vals.masked_fill(~sel_ok, FILL), (m_l + head.local_bias)[..., None]), dim=-1)
        far = torch.where(bvalid, blocks, torch.full_like(blocks, -(1 << 62)))
        rank = torch.argsort(torch.argsort(far, dim=-1, descending=True, stable=True), dim=-1)
        tok_ok = torch.cat((torch.ones_like(sel_ok[..., None].expand(-1, -1, -1, s)), local_tok[:, :, None]), dim=2)
        out, weights = head.read(uq, rk, rv, brow, tok_ok, bvalid, prior, rank)
        clock.lap("readout", count)
        # 5. training losses
        if labels is not None and torch.is_grad_enabled():
            self._losses(context, coarse, cand, fine, gold, hist, gcol, old, old_states, cq, pos, ends, n_res,
                         table, first, sl)
        if not context["full"]:
            return out, None
        return out, self._diagnose(coarse, table, cand, fine, sel, weights, blocks, gold, hist, gcol, values,
                                   pos, local)

    def _losses(self, context, coarse, cand, fine, gold, hist, gcol, old, old_states, cq, pos, ends, n_res, table,
                first, sl):
        """coarse_source_loss over every eligible block (superseded evidence of the same query excluded: the
        coarse stage is not asked to order revisions); fine_evidence_loss over the candidates."""
        head, s = self.lookup, self.evidence.block
        if old.numel():                                       # replay: candidate descriptors WITH gradient
            replay = head.describe(old_states)
            rmatch = head.match(cq, replay)
            cols = old - (table[0] if n_res else 0)
            inside = (cols >= 0) & (cols < n_res)
            if bool(inside.any()):
                rscore = head.coarse_score(rmatch[..., inside], pos[:, None] - ends[cols[inside]][None, :])
                rscore = rscore.masked_fill(~torch.isfinite(coarse[..., cols[inside]]), float("-inf"))
                coarse = coarse.index_copy(2, cols[inside], rscore)
        target = coarse
        stale = context["stale"]
        if stale is not None:
            sb = stale[:, sl].to(coarse.device)
            sb = torch.where(sb >= 0, sb // s, torch.full_like(sb, -1))
            drop = (table[None, None, None, :] == sb[..., None]) & (sb[..., None] != gold[..., None, None])
            target = coarse.masked_fill(drop.any(2), float("-inf"))
        if bool(hist.any()):
            loss = F.cross_entropy(target[hist], gcol[hist])
            context["losses"].setdefault("coarse_source_loss", []).append((loss, int(hist.sum())))
            present = hist & (cand == gold[..., None]).any(-1)
            if bool(present.any()):
                where = (cand == gold[..., None]).float().argmax(-1)
                fl = F.cross_entropy(fine[present], where[present])
                context["losses"].setdefault("fine_evidence_loss", []).append((fl, int(present.sum())))

    @torch.no_grad()
    def _diagnose(self, coarse, table, cand, fine, sel, weights, blocks, gold, hist, gcol, values, pos, local):
        e, s = self.evidence, self.evidence.block
        batch, count = cand.shape[:2]
        top_vals, top_pos = exact_topk(fine, cand.clamp(min=0), min(e.fine_top, cand.shape[-1]))
        fine_top = torch.where(torch.isfinite(top_vals), torch.gather(cand, 2, top_pos), torch.full_like(top_pos, -1))
        mass = weights[..., :-1].mean(dim=2)
        best = mass.argmax(-1)
        pick, offset = best // s, best % s
        read_block = torch.gather(blocks, 2, pick[..., None])[..., 0]
        pad = lambda t, width, value: F.pad(t, (0, max(0, width - t.shape[-1])), value=value)   # noqa: E731
        fine_top, top_vals = pad(fine_top, e.fine_top, -1), pad(top_vals, e.fine_top, float("-inf"))
        sel = pad(sel, e.evidence, -1)                         # fixed widths, whatever the table size
        out = {"local_block": local[None].expand(batch, -1), "fine_top": fine_top, "fine_scores": top_vals,
               "selected": sel, "read_block": read_block,
               "read_position": torch.where(read_block >= 0, read_block * s + offset, torch.full_like(best, -1)),
               "null_weight": weights[..., -1].mean(dim=2), "candidates": (cand >= 0).sum(-1),
               "eligible": torch.isfinite(coarse).sum(-1)}
        if gold is not None:
            gscore = torch.gather(coarse, 2, gcol[..., None])[..., 0]
            better = (coarse > gscore[..., None]) | ((coarse == gscore[..., None]) & (table[None, None] > gold[..., None]))
            out["coarse_rank"] = torch.where(hist, better.sum(-1) + 1, torch.full_like(gold, -1))
            in_cand = (cand == gold[..., None])
            gfine = torch.where(in_cand.any(-1), torch.gather(fine, 2, in_cand.float().argmax(-1)[..., None])[..., 0],
                                torch.full_like(gscore, float("-inf")))
            fbetter = (fine > gfine[..., None]) | ((fine == gfine[..., None]) & (cand > gold[..., None]))
            out["fine_rank"] = torch.where(hist & in_cand.any(-1), fbetter.sum(-1) + 1,
                                           torch.where(hist, torch.full_like(gold, cand.shape[-1] + 1),
                                                       torch.full_like(gold, -1)))
            out["gold_block"] = gold
            out["historical"] = hist
        return out


class _Clock:
    """Per-stage wall time into ``state.timing`` (measurement only; a no-op when timing is off)."""

    def __init__(self, sink):
        import time
        self.sink, self.now = sink, time.perf_counter
        self.last = self.now() if sink is not None else 0.0

    def lap(self, name: str, queries: int = 0) -> None:
        if self.sink is None:
            return
        now = self.now()
        self.sink[name] = self.sink.get(name, 0.0) + now - self.last
        if queries:
            self.sink["queries"] = self.sink.get("queries", 0) + queries
        self.last = now


def collect_losses(context_losses: dict) -> dict:
    """Probe-weighted means of the per-tile losses."""
    out = {}
    for name, parts in context_losses.items():
        total = sum(n for _, n in parts)
        out[name] = sum(loss * n for loss, n in parts) / max(total, 1)
        out[name + "_n"] = total
    return out
