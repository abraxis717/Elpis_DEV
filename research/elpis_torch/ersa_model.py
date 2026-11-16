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
from .scroll import ChunkError, ScrollAttention, ScrollConfig, StreamState


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
        if self.attention == "window":
            self.window_config()            # validates heads, head dimension and window >= chunk + 1

    def window_config(self) -> ScrollConfig:
        """R1's local topology with this model's attention geometry."""
        widths = self.model.replace(n_heads=self.attn_heads, n_kv_heads=self.attn_kv_heads)
        return ScrollConfig(widths, window=self.attn_window, block=1, chunk=self.chunk, horizon=self.attn_window)


class ErsaState:
    """All volatile state of ``batch`` streams. Fixed size; nothing in it grows with the stream."""

    def __init__(self, model: "ErsaLM", batch: int):
        self.batch, self.position = batch, 0
        self.cells = [block.ersa.new_state(batch) for block in model.blocks]
        self.tails = [block.ersa.new_tail(batch) for block in model.blocks]
        self.window = None
        if model.ersa.attention == "window":
            self.window = StreamState(model.ersa.window_config(), batch, 0, self.cells[0].dtype)

    def nbytes(self) -> dict:
        size = lambda t: t.numel() * t.element_size()   # noqa: E731
        out = {"ersa_cells": sum(size(c) for c in self.cells),
               "conv_tail": sum(size(t) for t in self.tails if t is not None)}
        if self.window is not None:
            ring = self.window.nbytes()
            out["attention_window"] = ring["window"]
            out["token_history"] = ring["token_history"]
        out = {k: v for k, v in out.items() if v}
        out["total"] = sum(out.values())
        return out


class ErsaBlock(nn.Module):
    def __init__(self, c: ErsaConfig, dtype: torch.dtype):
        super().__init__()
        m = c.model
        self.mix_norm = RMSNorm(m.d_model, m.norm_eps, dtype)
        self.ersa = ERSAMixer(m.d_model, heads=c.ersa_heads, key_dim=c.key_dim, value_dim=c.value_dim,
                              conv_kernel=c.conv_kernel, tie_erase=c.tie_erase, dtype=dtype)
        self.attn = ScrollAttention(c.window_config(), False, dtype) if c.attention == "window" else None
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
        self.blocks = nn.ModuleList(ErsaBlock(config, dtype) for _ in range(m.n_layers))
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
            else:
                std = ElpisTorchLM._init_std(self, name)
                if std is not None:
                    parameter.copy_(_seeded_normal(m.seed, name, parameter.shape, std, parameter.dtype))
                elif name.endswith("cgc_gate"):
                    parameter.fill_(m.cgc_gate_init)
                else:
                    parameter.fill_(1.0)

    def new_state(self, batch: int) -> ErsaState:
        return ErsaState(self, batch)

    def forward(self, tokens: torch.Tensor, state: ErsaState, *, return_pre: bool = False):
        """Logits ``[batch, T, vocab]`` for the next ``T <= chunk`` tokens. ``state`` advances at the end."""
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
        cells, tails, staged, pre = [], [], [], None
        for index, block in enumerate(self.blocks):
            u = block.mix_norm(x)
            mixed, new_cells, new_tail = block.ersa(u, state.cells[index], state.tails[index])
            cells.append(new_cells)
            tails.append(new_tail)
            if block.attn is not None:
                attended, pending, _ = block.attn(u, self.rotary, state.window.layers[index], state.window)
                mixed = mixed + attended
                staged.append(pending)
            pre = x + mixed
            x = pre + block.channel(block.channel_norm(pre))
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
            if block.attn is not None:
                block.attn.commit(state.window.layers[index], state.window, staged[index])
        if state.window is not None:
            positions = state.window.position + torch.arange(length)
            state.window.tokens.index_copy_(1, positions % state.window.config.fine_capacity, tokens.to(torch.int32))
            state.window.position += length
        state.position += length
        return (logits, pre) if return_pre else logits

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
