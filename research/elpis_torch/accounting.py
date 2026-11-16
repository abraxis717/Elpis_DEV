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


# -- R1: scrolling context -------------------------------------------------------------------------------------

def scroll_state_bytes(scroll, *, batch: int = 1, scalar_bytes: int | None = None) -> dict:
    """Bytes of a ``StreamState`` by tier, from the configuration alone (equals ``StreamState.nbytes()``).

    None of it depends on how many tokens have been streamed.
    """
    m = scroll.model
    width = scalar_bytes or (8 if m.dtype == "float64" else 4)
    kv = batch * m.kv_heads * m.head_dim * width            # one key or one value, all heads
    layers, memory = m.n_layers, len(scroll.global_layers)
    out = {"token_history": batch * scroll.fine_capacity * 4,
           "window": layers * 2 * kv * scroll.window,
           "compressed": memory * 2 * kv * scroll.slots,
           "open_block": memory * (2 * kv * scroll.block + batch * m.kv_heads * scroll.block * width)}
    if scroll.sinks:
        out["sinks"] = layers * 2 * kv * scroll.sinks
    if scroll.fine_blocks and memory:
        out["fine"] = memory * 2 * kv * scroll.fine_capacity
    out = {tier: size for tier, size in out.items() if size}
    out["total"] = sum(out.values())
    return out


def scroll_compute_per_token(scroll, *, resident_blocks: int | None = None) -> dict:
    """Theoretical multiply-accumulates of the ATTENTION MIXING for one token, per layer kind.

    Local layers cost ``O(window)``, independent of the stream length. Memory layers add a score against every
    resident compressed block (``O(horizon / block)``: linear in the horizon, with a small constant) and a
    constant number of selected keys. Nothing is quadratic in the stream length.
    """
    m = scroll.model
    blocks = scroll.slots if resident_blocks is None else min(resident_blocks, scroll.slots)
    per_key = 2 * m.n_heads * m.head_dim
    local = per_key * (scroll.window + scroll.sinks)
    coarse = scroll.global_topk - scroll.fine_blocks
    return {"local_layer": local, "memory_scoring": m.n_heads * m.head_dim * blocks,
            "memory_selected": per_key * (coarse + scroll.fine_blocks * scroll.block),
            "dense_reference_at_horizon": per_key * scroll.horizon}


def deployment_projection(*, d_model: int = 2048, n_layers: int = 24, n_heads: int = 16, n_kv_heads: int = 4,
                          vocab: int = 100_000, window: int = 8192, block: int = 64, horizon: int = 1 << 20,
                          memory_layers: int = 6, chunk: int = 512, scalar_bytes: int = 2, gpu_bytes: int = 24 << 30,
                          fine_on_gpu: bool = False, separate_memory_keys: bool = False) -> dict:
    """PROJECTED (never measured) placement of one 1M-token stream of a deployment-scale model on a 24 GB GPU.

    Three classes of memory: GPU-resident state, CPU RAM, and immutable persistent resources (the checkpoint,
    read-only). ``fine_on_gpu`` moves the raw-token tier of the memory layers onto the GPU.
    """
    head_dim = d_model // n_heads
    kv = n_kv_heads * head_dim * scalar_bytes
    attention = d_model * head_dim * (2 * n_heads + 2 * n_kv_heads)
    parameters = n_layers * (attention + 8 * d_model * d_model + 2 * d_model) \
        + memory_layers * d_model * (head_dim * (n_heads + n_kv_heads) * separate_memory_keys + n_kv_heads) \
        + vocab * d_model
    fine = memory_layers * 2 * kv * (horizon + block)
    gpu = {"weights": parameters * scalar_bytes, "window": n_layers * 2 * kv * window,
           "compressed": memory_layers * 2 * kv * (horizon // block),
           "chunk_activations_estimate": 64 * chunk * d_model * scalar_bytes,
           "block_scores_per_chunk": n_heads * chunk * (horizon // block) * scalar_bytes}
    cpu = {"token_history": (horizon + block) * 4}
    (gpu if fine_on_gpu else cpu)["fine"] = fine
    gpu["total"], cpu["total"] = sum(gpu.values()), sum(cpu.values())
    return {"assumptions": {"d_model": d_model, "n_layers": n_layers, "n_heads": n_heads, "n_kv_heads": n_kv_heads,
                            "window": window, "block": block, "horizon": horizon, "memory_layers": memory_layers,
                            "chunk": chunk, "scalar_bytes": scalar_bytes, "parameters": parameters,
                            "fine_on_gpu": fine_on_gpu},
            "gpu_resident": gpu, "cpu_ram": cpu, "immutable_persistent": {"checkpoint": parameters * scalar_bytes},
            "gpu_budget": gpu_bytes, "fits_gpu": gpu["total"] <= gpu_bytes,
            "dense_kv_cache_at_horizon_for_comparison": n_layers * 2 * kv * horizon,
            "status": "PROJECTED"}


# -- R2: ERSA context state ----------------------------------------------------------------------------------------

def ersa_context_projection(*, d_model: int = 2048, n_layers: int = 24, ersa_heads: int = 16, key_dim: int = 128,
                            value_dim: int = 128, conv_kernel: int = 4, attn_window: int = 4096,
                            attn_kv_heads: int = 4, attn_head_dim: int = 128, index_layers: int = 0,
                            index_block: int = 64, horizon: int = 1 << 20, chunk: int = 512, scalar_bytes: int = 2,
                            target_bytes: int = 1 << 30, ceiling_bytes: int = 2 << 30) -> dict:
    """PROJECTED (never measured) active context state of one 1,048,576-token stream of a deployment-scale
    ERSA model, against the 1 GiB target and the 2 GiB ceiling. Model weights are not context and are excluded.

    GPU: the ERSA cells, the convolution tails, the attention window's keys and values, an optional compressed
    long-range index (R1's compressed tier, no raw-token tier) and per-chunk scratch. CPU: the token history.
    """
    kv = attn_kv_heads * attn_head_dim * scalar_bytes
    gpu = {"ersa_cells": n_layers * ersa_heads * key_dim * value_dim * scalar_bytes,
           "conv_tails": n_layers * max(0, conv_kernel - 1) * d_model * scalar_bytes,
           "attention_window": n_layers * 2 * kv * attn_window if attn_window else 0,
           "long_range_index": index_layers * 2 * kv * (horizon // index_block),
           "scratch_per_chunk_estimate": 64 * chunk * d_model * scalar_bytes}
    gpu = {name: size for name, size in gpu.items() if size}
    gpu["total"] = sum(gpu.values())
    cpu = {"token_history": horizon * 4}
    cpu["total"] = sum(cpu.values())
    return {"status": "PROJECTED", "gpu_resident": gpu, "cpu_ram": cpu, "combined": gpu["total"] + cpu["total"],
            "within_target": gpu["total"] <= target_bytes, "within_ceiling": gpu["total"] <= ceiling_bytes,
            "assumptions": {"d_model": d_model, "n_layers": n_layers, "ersa_heads": ersa_heads, "key_dim": key_dim,
                            "value_dim": value_dim, "conv_kernel": conv_kernel, "attn_window": attn_window,
                            "attn_kv_heads": attn_kv_heads, "attn_head_dim": attn_head_dim,
                            "index_layers": index_layers, "index_block": index_block, "horizon": horizon,
                            "chunk": chunk, "scalar_bytes": scalar_bytes},
            "exact_recall_claim": "none: the recurrent state holds on the order of key_dim associations per head"}
