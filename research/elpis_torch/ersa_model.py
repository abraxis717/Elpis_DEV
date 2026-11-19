"""R2: an autoregressive transformer whose token mixer is ERSA, with CGC as its channel transformation.

Block (pre-norm, the same channel sublayer as R0 and R1):

    u = RMSNorm(x)
    x = x + ERSA(u)  [+ Attention(u)]         recurrent associative mixing, optionally plus a bounded window
    x = x + Channel(RMSNorm(x))               FFN | CGC | FFN + gate * CGC

``attention`` selects what accompanies ERSA:

    "none"    ERSA is the only token mixer. No key/value cache exists at all.
    "window"  plus causal attention over the last ``attn_window`` tokens (R1's ring buffer, local topology).

The window is the only attention there is: a "small number of grouped-query heads" is this with few heads and
a longer window. Nothing attends over the whole history, so nothing grows with the stream.

State (``ErsaState``): per layer one ERSA matrix per head, the convolution tail if there is one, and (with a
window) R1's preallocated ring. Its size is fixed at ``new_state`` and reported by ``nbytes``.

The last block's CGC acts after the last mixer, so, exactly as in R1, changing its weight changes no ERSA state
and no window cache (``lastblock.last_block_logits`` applies unchanged).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .cgc import CGCTransform
from .config import ConfigError, ModelConfig
from .ersa import ERSAMixer
from .model import ElpisTorchLM, RMSNorm, Rotary, SwiGLU, _seeded_normal, sample_next
from .ple import PLE, PLEConfig
from .scroll import ChunkError, ScrollAttention, ScrollConfig, StreamState
from .sparse_channel import SparseChannel, SparseConfig


@dataclass(frozen=True)
class ErsaConfig:
    """``model`` gives the widths and the channel transformation (its attention fields are unused here)."""
    model: ModelConfig
    ersa_heads: int = 4
    key_dim: int = 16
    value_dim: int = 16
    conv_kernel: int = 0               # 0: no local convolution
    tie_erase: bool = True
    attention: str = "none"            # "none" | "window"
    attn_heads: int = 2
    attn_kv_heads: int = 1
    attn_window: int = 16
    chunk: int = 64                    # most tokens one forward call may process
    attn_layers: tuple | None = None   # R3: layers that carry the window attention; None = every layer
    ersa_kernel: str = "serial"        # R3: "serial" (the R2 reference loop) or "chunked" (exact, block-parallel)
    scan_block: int = 64               # R3: positions per block of the chunked kernel
    sparse: SparseConfig | None = None # R6: blocks whose channel is a routed bank of FFN + CGC experts
    ple: PLEConfig | None = None       # R7: hashed n-gram lookup rows fused before one block's mixer

    def __post_init__(self):
        if not isinstance(self.model, ModelConfig):
            raise ConfigError("ErsaConfig.model must be a ModelConfig")
        for name in ("ersa_heads", "key_dim", "value_dim", "attn_heads", "attn_kv_heads", "attn_window", "chunk"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ConfigError(f"{name} must be a positive integer")
        if type(self.conv_kernel) is not int or self.conv_kernel < 0 or self.conv_kernel == 1:
            raise ConfigError("conv_kernel must be 0 or at least 2")
        if self.attention not in ("none", "window"):
            raise ConfigError("attention must be 'none' or 'window'")
        if self.ersa_kernel not in ("serial", "chunked"):
            raise ConfigError("ersa_kernel must be 'serial' or 'chunked'")
        if type(self.scan_block) is not int or self.scan_block < 1:
            raise ConfigError("scan_block must be a positive integer")
        if self.attn_layers is not None:
            layers = tuple(self.attn_layers)
            if (not layers or len(set(layers)) != len(layers)
                    or any(type(i) is not int or not 0 <= i < self.model.n_layers for i in layers)):
                raise ConfigError("attn_layers must be distinct layer indices (or None for every layer)")
            if self.attention != "window":
                raise ConfigError("attn_layers needs attention='window'")
            object.__setattr__(self, "attn_layers", tuple(sorted(layers)))
        if self.attention == "window":
            self.window_config()            # validates heads, head dimension and window >= chunk + 1
        if self.sparse is not None:
            if not isinstance(self.sparse, SparseConfig):
                raise ConfigError("sparse must be a SparseConfig (or None for the dense channel everywhere)")
            if any(i >= self.model.n_layers for i in self.sparse.layers):
                raise ConfigError("sparse.layers must name blocks of the model")
        if self.ple is not None:
            if not isinstance(self.ple, PLEConfig):
                raise ConfigError("ple must be a PLEConfig (or None for no lookup rows)")
            if self.ple.layer >= self.model.n_layers:
                raise ConfigError("ple.layer must name a block of the model")

    @property
    def window_layers(self) -> tuple:
        """The layers that carry window attention (empty without attention)."""
        if self.attention != "window":
            return ()
        return tuple(range(self.model.n_layers)) if self.attn_layers is None else self.attn_layers

    def window_config(self) -> ScrollConfig:
        """R1's local topology with this model's attention geometry; one ring per attention-carrying layer."""
        layers = self.model.n_layers if self.attn_layers is None else len(self.attn_layers)
        widths = self.model.replace(n_heads=self.attn_heads, n_kv_heads=self.attn_kv_heads, n_layers=layers)
        return ScrollConfig(widths, window=self.attn_window, block=1, chunk=self.chunk, horizon=self.attn_window)


class ErsaState:
    """All volatile state of ``batch`` streams. Fixed size; nothing in it grows with the stream."""

    def __init__(self, model: "ErsaLM", batch: int):
        self.batch, self.position = batch, 0
        self.cells = [block.ersa.new_state(batch) for block in model.blocks]      # on the model's device
        self.tails = [block.ersa.new_tail(batch) for block in model.blocks]
        self.device = self.cells[0].device
        self.window = None
        if model.ersa.attention == "window":
            self.window = StreamState(model.ersa.window_config(), batch, 0, self.cells[0].dtype, self.device)
        self.window_slots = {layer: slot for slot, layer in enumerate(model.ersa.window_layers)}
        self.ple_history = None             # R7: the last max_order - 1 token ids (n-gram addressing only)
        if model.ple is not None:
            self.ple_history = model.ple.addresser.new_history(batch, self.device)
        self._ple_sentinel = None if model.ple is None else model.ple.addresser.sentinel

    def reset(self) -> None:
        """R4: an empty stream in the same buffers. Tensors that carry a training graph are replaced by zeros."""
        self.cells = [c.detach().zero_() for c in self.cells]
        self.tails = [None if t is None else t.detach().zero_() for t in self.tails]
        if self.window is not None:
            self.window.reset()
        if self.ple_history is not None:
            self.ple_history.fill_(self._ple_sentinel)
        self.position = 0

    def nbytes(self) -> dict:
        size = lambda t: t.numel() * t.element_size()   # noqa: E731
        out = {"ersa_cells": sum(size(c) for c in self.cells),
               "conv_tail": sum(size(t) for t in self.tails if t is not None)}
        if self.window is not None:
            ring = self.window.nbytes()
            out["attention_window"] = ring["window"]
            out["token_history"] = ring["token_history"]
        if self.ple_history is not None:
            out["ple_history"] = size(self.ple_history)
        out = {k: v for k, v in out.items() if v}
        out["total"] = sum(out.values())
        return out


class ErsaBlock(nn.Module):
    def __init__(self, c: ErsaConfig, dtype: torch.dtype, index: int = 0):
        super().__init__()
        m = c.model
        self.mix_norm = RMSNorm(m.d_model, m.norm_eps, dtype)
        self.ersa = ERSAMixer(m.d_model, heads=c.ersa_heads, key_dim=c.key_dim, value_dim=c.value_dim,
                              conv_kernel=c.conv_kernel, tie_erase=c.tie_erase, dtype=dtype,
                              kernel=c.ersa_kernel, scan_block=c.scan_block)
        self.attn = ScrollAttention(c.window_config(), False, dtype) if index in c.window_layers else None
        self.channel_norm = RMSNorm(m.d_model, m.norm_eps, dtype)
        self.ffn = SwiGLU(m.d_model, m.ffn_width, dtype) if m.has_ffn else None
        self.cgc = CGCTransform(m.d_model, m.cgc_cells, m.cgc_units, bottleneck=m.cgc_bottleneck,
                                preact_bound=m.cgc_preact_bound, dtype=dtype) if m.has_cgc else None
        self.cgc_gate = nn.Parameter(torch.zeros(1, dtype=dtype)) if m.channel == "ffn+cgc" else None
        self.sparse = None
        if c.sparse is not None and index in c.sparse.layers:      # R6: the routed channel replaces the dense one
            self.ffn = self.cgc = self.cgc_gate = None
            self.sparse = SparseChannel(m, c.sparse, dtype)

    def channel(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparse is not None:
            return self.sparse(x)
        if self.cgc is None:
            return self.ffn(x)
        if self.ffn is None:
            return self.cgc(x)
        return self.ffn(x) + self.cgc_gate * self.cgc(x)


class ErsaLM(nn.Module):
    """Embeddings -> ERSA blocks -> RMSNorm -> logits."""

    FORGET_BIAS, WRITE_BIAS = -1.5, 0.0          # hard sigmoid: retain starts at 0.875, write at 0.5

    def __init__(self, config: ErsaConfig):
        super().__init__()
        if not isinstance(config, ErsaConfig):
            raise TypeError("ErsaLM expects an ErsaConfig")
        self.ersa, self.config = config, config.model
        m, dtype = config.model, getattr(torch, config.model.dtype)
        self.embed = nn.Embedding(m.vocab_size, m.d_model, dtype=dtype)
        self.blocks = nn.ModuleList(ErsaBlock(config, dtype, index) for index in range(m.n_layers))
        self.ple = None if config.ple is None else PLE(m, config.ple, dtype)
        self.rotary = None
        if config.attention == "window":
            scroll = config.window_config()
            self.rotary = Rotary(scroll.model.head_dim, scroll.window + scroll.chunk + 1, m.rope_theta, dtype)
        self.final_norm = RMSNorm(m.d_model, m.norm_eps, dtype)
        self.head = None if m.tie_embeddings else nn.Linear(m.d_model, m.vocab_size, bias=False, dtype=dtype)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """R0's rule (a function of seed and parameter name), plus ERSA's gates and convolution."""
        m = self.config
        residual = m.init_std / (2 * m.n_layers) ** 0.5
        for name, parameter in self.named_parameters():
            if name.endswith("ersa.out.weight"):
                parameter.copy_(_seeded_normal(m.seed, name, parameter.shape, residual, parameter.dtype))
            elif name.endswith("ersa.conv.weight"):
                parameter.zero_()
                parameter[:, 0, -1] = 1.0                       # starts as the identity on the current token
            elif name.endswith("ersa.gates.bias"):
                heads = self.ersa.ersa_heads
                parameter.zero_()
                parameter.view(heads, -1)[:, 0] = self.FORGET_BIAS
                parameter.view(heads, -1)[:, 1:] = self.WRITE_BIAS
            elif name.endswith("ersa.og.bias"):
                parameter.fill_(2.0)                            # output gate starts mostly open
            elif name == "ple.table.weight":                    # unit-RMS rows, like an embedding's
                parameter.copy_(_seeded_normal(m.seed, name, parameter.shape, 1.0, parameter.dtype))
            elif name == "ple.value.weight":                    # writes into the residual stream
                parameter.copy_(_seeded_normal(m.seed, name, parameter.shape, residual, parameter.dtype))
            else:
                std = ElpisTorchLM._init_std(self, name)
                if std is not None:
                    parameter.copy_(_seeded_normal(m.seed, name, parameter.shape, std, parameter.dtype))
                elif name.endswith("cgc_gate"):
                    sparse = ".sparse." in name
                    parameter.fill_(self.ersa.sparse.gate_init if sparse else m.cgc_gate_init)
                else:
                    parameter.fill_(1.0)

    def new_state(self, batch: int) -> ErsaState:
        return ErsaState(self, batch)

    def forward(self, tokens: torch.Tensor, state: ErsaState, *, return_pre: bool = False,
                logit_positions: torch.Tensor | None = None, **extra):
        """Logits ``[batch, T, vocab]`` for the next ``T <= chunk`` tokens. ``state`` advances at the end.

        ``logit_positions`` (R4): indices into the chunk; only those positions get logits ``[batch, n, vocab]``
        (and ``pre`` is restricted alike). The state advance is unchanged."""
        c = self.ersa
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be an int64 tensor of shape [batch, T]")
        batch, length = tokens.shape
        if batch != state.batch:
            raise ValueError("the state does not belong to this batch")
        if not 1 <= length <= c.chunk:
            raise ChunkError(f"{length} tokens do not fit one chunk of {c.chunk}; split the input")
        if int(tokens.min()) < 0 or int(tokens.max()) >= self.config.vocab_size:
            raise ValueError(f"token ids must be in [0, {self.config.vocab_size})")
        x = self.embed(tokens)
        context = self._chunk_context(tokens, state, extra)
        addresses = None
        if self.ple is not None and self.ple.enabled:          # R7: once per call, from token ids alone
            addresses = self.ple.addresser.addresses(tokens, state.ple_history)
        cells, tails, staged, pre = [], [], [], None
        for index, block in enumerate(self.blocks):
            if addresses is not None and index == c.ple.layer:
                x = x + self.ple(x, addresses)
            u = block.mix_norm(x)
            mixed, new_cells, new_tail = block.ersa(u, state.cells[index], state.tails[index])
            cells.append(new_cells)
            tails.append(new_tail)
            if block.attn is not None:
                ring = state.window.layers[state.window_slots[index]]
                attended, pending, _ = block.attn(u, self.rotary, ring, state.window)
                mixed = mixed + attended
                staged.append((index, pending))
            read = self._extra_mix(index, u, state, context)
            if read is not None:
                mixed = mixed + read
            pre = x + mixed
            x = pre + block.channel(block.channel_norm(pre))
        if logit_positions is not None:          # R4: the head only where the caller needs it ([B, T, V] is avoided)
            x = x[:, logit_positions]
            pre = pre[:, logit_positions]
        x = self.final_norm(x)
        logits = F.linear(x, self.embed.weight) if self.head is None else self.head(x)
        graph = torch.is_grad_enabled() and any(t.requires_grad for t in cells)
        for index, block in enumerate(self.blocks):
            if graph:
                state.cells[index] = cells[index]              # training: the recurrence carries the graph
            else:
                state.cells[index].copy_(cells[index])         # inference: the same buffer, overwritten
            if tails[index] is not None:
                state.tails[index] = tails[index] if graph else state.tails[index].copy_(tails[index])
        for index, pending in staged:
            self.blocks[index].attn.commit(state.window.layers[state.window_slots[index]], state.window, pending)
        if state.window is not None:
            positions = state.window.position + torch.arange(length, device=tokens.device)
            state.window.tokens.index_copy_(1, positions % state.window.config.fine_capacity, tokens.to(torch.int32))
            state.window.position += length
        self._commit_context(tokens, state, context)
        if state.ple_history is not None:
            state.ple_history.copy_(self.ple.addresser.next_history(tokens, state.ple_history))
        state.position += length
        if context is not None and context.get("diagnostics") is not None:
            return (logits, pre, context["diagnostics"]) if return_pre else (logits, context["diagnostics"])
        return (logits, pre) if return_pre else logits

    # -- extension points (R3): a subclass adds a mixer term per block and commits its own state ----------------

    def _chunk_context(self, tokens, state, extra):
        """Per-chunk context for ``_extra_mix``; ``None`` here: ERSA [+ window] only."""
        if extra:
            raise TypeError(f"unexpected arguments {sorted(extra)}")
        return None

    def _extra_mix(self, index, u, state, context):
        return None

    def _commit_context(self, tokens, state, context) -> None:
        return None

    def run(self, tokens: torch.Tensor, state: ErsaState | None = None, *, chunk: int | None = None):
        """Feed a whole ``[batch, T]`` sequence in chunks: ``(logits [batch, T, vocab], state)``."""
        state = self.new_state(tokens.shape[0]) if state is None else state
        step = chunk or self.ersa.chunk
        parts = [self(tokens[:, at:at + step], state) for at in range(0, tokens.shape[1], step)]
        return torch.cat(parts, dim=1), state

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int, *, temperature: float = 0.0,
                 top_k: int | None = None, generator: torch.Generator | None = None,
                 state: ErsaState | None = None) -> torch.Tensor:
        """Autoregressive continuation, one token per step through the recurrent state. No length limit."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        logits, state = self.run(tokens, state)
        new = []
        for _ in range(max_new_tokens):
            token = sample_next(logits[:, -1], temperature=temperature, top_k=top_k, generator=generator)
            new.append(token)
            logits = self(token[:, None], state)
        return torch.stack(new, dim=1)
