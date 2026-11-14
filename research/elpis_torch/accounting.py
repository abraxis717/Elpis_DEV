"""Parameter, theoretical-compute and memory accounting. Formulas only: nothing here measures a machine.

Three different quantities, never interchangeable:

    parameters            trainable scalars.
    multiply-accumulates  theoretical scalar multiply(-add)s to produce one token's logits at context length T
                          in cached (incremental) execution. Additions, normalizations' divisions/square roots,
                          exponentials, tanh and softmax are NOT counted; they are listed as ``transcendentals``.
    activation memory     scalars the forward pass holds for a training batch (autograd keeps them all).

Two models with equal parameter counts are parameter-matched. They are compute-matched only if their
multiply-accumulate counts are also equal, and even then wall-clock and memory can differ: CGC's widest
activation is ``C * N`` per token, SwiGLU's is ``ffn_hidden``.
"""
from __future__ import annotations

from .cgc import CGCTransform
from .config import ModelConfig
from .training import GROUPS, group_of


def parameter_counts(model) -> dict:
    """Trainable parameters in total and per accounting group."""
    groups = {g: 0 for g in GROUPS}
    for name, p in model.named_parameters():
        groups[group_of(name)] += p.numel()
    return {"total": sum(groups.values()), **{g: n for g, n in groups.items() if n}}


def channel_parameters(c: ModelConfig) -> dict:
    """Parameters of ONE block's channel transformation, by formula."""
    ffn = 3 * c.d_model * c.ffn_width if c.has_ffn else 0
    cgc = CGCTransform.parameter_count(c.d_model, c.cgc_cells, c.cgc_units, c.cgc_bottleneck) if c.has_cgc else 0
    return {"ffn": ffn, "cgc": cgc, "gate": 1 if c.channel == "ffn+cgc" else 0}


def compute_per_token(c: ModelConfig, context: int) -> dict:
    """Theoretical multiply-accumulates for one new token attending to ``context`` positions (itself included)."""
    h, hd = c.d_model, c.head_dim
    attention_projections = h * c.n_heads * hd + 2 * h * c.kv_heads * hd + c.n_heads * hd * h
    rotary = 2 * (c.n_heads + c.kv_heads) * hd                       # two products per rotated scalar
    attention_mixing = 2 * c.n_heads * hd * context                  # scores, then the weighted value sum
    ffn = 3 * h * c.ffn_width + c.ffn_width if c.has_ffn else 0
    cgc = (CGCTransform.multiply_accumulates(h, c.cgc_cells, c.cgc_units, c.cgc_bottleneck)
           + (h if c.channel == "ffn+cgc" else 0)) if c.has_cgc else 0
    norms = 2 * 2 * h                                                # square and scale, two norms per block
    block = attention_projections + rotary + attention_mixing + ffn + cgc + norms
    head = h * c.vocab_size + 2 * h
    transcendentals = c.n_layers * (c.n_heads * context                                  # softmax exponentials
                                    + (c.ffn_width if c.has_ffn else 0)                  # SiLU sigmoids
                                    + (c.cgc_cells * c.cgc_units if c.has_cgc and c.cgc_preact_bound else 0))
    return {"context": context, "total": c.n_layers * block + head, "per_block": block,
            "per_block_attention": attention_projections + rotary + attention_mixing,
            "per_block_channel": ffn + cgc, "head": head, "transcendentals": transcendentals}


def memory_estimate(c: ModelConfig, *, batch: int, length: int) -> dict:
    """Scalars held by one training forward pass (retained for backward), and their bytes at the model dtype.

    Counts the residual stream, attention scores and weights, the channel transformation's intermediates and
    the logits. An estimate of activation memory only: parameters, gradients and optimizer state add
    ``~4 * parameters`` scalars under AdamW; allocator overhead is not modelled.
    """
    h, tokens = c.d_model, batch * length
    attention = 2 * batch * c.n_heads * length * length + tokens * (2 * c.n_heads + 2 * c.kv_heads) * c.head_dim
    ffn = 3 * tokens * c.ffn_width if c.has_ffn else 0
    cells = c.cgc_cells * c.cgc_units
    cgc = tokens * ((c.cgc_bottleneck or 0) + (3 if c.cgc_preact_bound else 2) * cells + c.cgc_cells) if c.has_cgc else 0
    residual = 4 * tokens * h
    scalars = c.n_layers * (attention + ffn + cgc + residual) + tokens * (h + c.vocab_size)
    width = 8 if c.dtype == "float64" else 4
    return {"batch": batch, "length": length, "activation_scalars": scalars, "activation_bytes": scalars * width,
            "widest_channel_activation_per_token": max(c.ffn_width if c.has_ffn else 0, cells if c.has_cgc else 0),
            "kv_cache_bytes_at_full_context": 2 * c.n_layers * c.kv_heads * c.head_dim * c.max_seq_len * width}


def session_memory(c: ModelConfig) -> dict:
    """Volatile bytes one session adds: the fast-weight deltas (plus equal momentum) and one stream cache."""
    width = 8 if c.dtype == "float64" else 4
    fast = c.n_layers * (c.cgc_bottleneck or c.d_model) * c.cgc_cells * c.cgc_units if c.has_cgc else 0
    return {"fast_weight_scalars": fast, "fast_weight_bytes": fast * width,
            "kv_cache_bytes": 2 * c.n_layers * c.kv_heads * c.head_dim * c.max_seq_len * width}
