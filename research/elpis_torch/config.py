"""Model configuration: every dimension explicit, every combination validated before a tensor exists."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass

CHANNELS = ("ffn", "cgc", "ffn+cgc")
DTYPES = ("float32", "float64")


class ConfigError(ValueError):
    """An invalid or inconsistent model configuration."""


def _positive(name: str, value) -> None:
    if type(value) is not int or value < 1:
        raise ConfigError(f"{name} must be a positive integer, got {value!r}")


def _real(name: str, value, *, positive: bool = True) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ConfigError(f"{name} must be a {'positive ' if positive else ''}finite number, got {value!r}")


@dataclass(frozen=True)
class ModelConfig:
    """A decoder-only transformer. ``channel`` selects the block's nonlinear channel transformation:

    ``"ffn"``      A: SwiGLU feed-forward network, no CGC.
    ``"cgc"``      B: the CGC transform is the block's only channel nonlinearity.
    ``"ffn+cgc"``  C: the SwiGLU network plus a parallel CGC transform behind a learnable scalar gate.
    """
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    max_seq_len: int
    n_kv_heads: int | None = None        # grouped-query attention; None or n_heads = ordinary multi-head
    channel: str = "ffn"
    ffn_hidden: int | None = None        # SwiGLU hidden width; None = round(8/3 d_model)
    cgc_channels: int | None = None      # C: E0 cells per block; None = d_model
    cgc_units: int = 4                   # N: cubic units per cell
    cgc_bottleneck: int | None = None    # d: None = dense (cells read the full hidden state)
    cgc_preact_bound: float | None = 3.0  # None = faithful, unbounded E0 pre-activations
    cgc_init_gain: float = 0.5           # initial pre-activation standard deviation
    cgc_gate_init: float = 0.0           # C only: 0 makes the block exactly the FFN block at initialization
    tie_embeddings: bool = True
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    init_std: float = 0.02
    dtype: str = "float32"
    seed: int = 0

    def __post_init__(self):
        for name in ("vocab_size", "d_model", "n_layers", "n_heads", "max_seq_len", "cgc_units"):
            _positive(name, getattr(self, name))
        for name in ("n_kv_heads", "ffn_hidden", "cgc_channels", "cgc_bottleneck"):
            if getattr(self, name) is not None:
                _positive(name, getattr(self, name))
        if self.vocab_size < 2:
            raise ConfigError("vocab_size must be at least 2")
        if self.d_model % self.n_heads:
            raise ConfigError(f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}")
        if (self.d_model // self.n_heads) % 2:
            raise ConfigError(f"head dimension {self.d_model // self.n_heads} must be even for rotary embeddings")
        if self.n_heads % self.kv_heads:
            raise ConfigError(f"n_heads {self.n_heads} is not a multiple of n_kv_heads {self.kv_heads}")
        if self.channel not in CHANNELS:
            raise ConfigError(f"channel must be one of {CHANNELS}, got {self.channel!r}")
        if self.dtype not in DTYPES:
            raise ConfigError(f"dtype must be one of {DTYPES}, got {self.dtype!r}")
        if self.cgc_preact_bound is not None:
            _real("cgc_preact_bound", self.cgc_preact_bound)
        for name in ("cgc_init_gain", "norm_eps", "rope_theta", "init_std"):
            _real(name, getattr(self, name))
        _real("cgc_gate_init", self.cgc_gate_init, positive=False)
        if type(self.tie_embeddings) is not bool:
            raise ConfigError("tie_embeddings must be a bool")
        if type(self.seed) is not int or self.seed < 0:
            raise ConfigError("seed must be a non-negative integer")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_heads(self) -> int:
        return self.n_heads if self.n_kv_heads is None else self.n_kv_heads

    @property
    def ffn_width(self) -> int:
        return self.ffn_hidden if self.ffn_hidden is not None else round(8 * self.d_model / 3)

    @property
    def cgc_cells(self) -> int:
        return self.cgc_channels if self.cgc_channels is not None else self.d_model

    @property
    def has_ffn(self) -> bool:
        return self.channel in ("ffn", "ffn+cgc")

    @property
    def has_cgc(self) -> bool:
        return self.channel in ("cgc", "ffn+cgc")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "ModelConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        if not isinstance(value, dict) or set(value) != names:
            raise ConfigError("configuration record does not match this ModelConfig schema")
        return cls(**value)

    def replace(self, **changes) -> "ModelConfig":
        return dataclasses.replace(self, **changes)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()
