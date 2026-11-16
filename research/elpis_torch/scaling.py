"""R3 P4: the 500M and 1B configurations, counted by instantiation, and their context-memory and compute
accounting for a 1,048,576-token rolling horizon.

Three kinds of number, never mixed:

    INSTANTIATED   the model (or its per-stream state) is constructed on PyTorch's ``meta`` device and its
                   tensors are counted. Shapes are exact; no memory is allocated and nothing is measured.
    MEASURED       bytes actually allocated on this CPU (``measure_small``): a small model's state run to length.
    PROJECTED      formulas for quantities that depend on execution (scratch buffers, compute, training memory).

Untrained weights: a 500M configuration built here is an architecture, not a language model.
"""
from __future__ import annotations

import torch

from .config import ModelConfig
from .ersa_model import ErsaConfig
from .history_model import ErsaHistoryLM, HistoryConfig

HORIZON = 1 << 20
CONFIGURATIONS = {
    # name: widths. Attention (a 4,096-token window, 2 KV heads) every fourth block; history reads in two blocks.
    "r3_500m": dict(d_model=1280, n_layers=24, ersa_heads=10, key_dim=128, cgc_bottleneck=512, cgc_channels=2560),
    "r3_1b": dict(d_model=2048, n_layers=20, ersa_heads=16, key_dim=128, cgc_bottleneck=768, cgc_channels=4096),
}
VOCAB = 65_536          # a future subword tokenizer with byte fallback (tokenizer.SubwordTokenizerProtocol)


def configuration(name: str, *, horizon: int = HORIZON, chunk: int = 512):
    w = CONFIGURATIONS[name]
    d, layers = w["d_model"], w["n_layers"]
    model = ModelConfig(vocab_size=VOCAB, d_model=d, n_layers=layers, n_heads=d // 128, n_kv_heads=2,
                        max_seq_len=64, channel="cgc", cgc_bottleneck=w["cgc_bottleneck"],
                        cgc_channels=w["cgc_channels"], cgc_units=4)
    ersa = ErsaConfig(model, ersa_heads=w["ersa_heads"], key_dim=w["key_dim"], value_dim=w["key_dim"],
                      conv_kernel=4, attention="window", attn_heads=d // 128, attn_kv_heads=2, attn_window=4096,
                      attn_layers=tuple(range(3, layers, 4)), chunk=chunk, ersa_kernel="chunked", scan_block=64)
    history = HistoryConfig(block=64, horizon=horizon, topk=2, index_dim=64, per_block=2, read_heads=8,
                            read_kv_heads=1, read_head_dim=128, overlap=16, layers=(layers // 2 - 1, layers - 1),
                            encoder_heads=4, encoder_dim=256)
    return ersa, history


def instantiate(name: str, *, horizon: int = HORIZON):
    """The model and one stream's state on the meta device (exact shapes, no memory)."""
    ersa, history = configuration(name, horizon=horizon)
    with torch.device("meta"):
        model = ErsaHistoryLM(ersa, history)
        state = model.new_state(1)
    return model, state


def parameter_breakdown(model) -> dict:
    groups: dict = {}
    for name, p in model.named_parameters():
        if name.startswith("embed"):
            key = "embedding (tied head)"
        elif name.startswith(("encoder.", "readers.")):
            key = "history encoder" if name.startswith("encoder.") else "history readers"
        elif ".ersa." in name:
            key = "ersa mixers"
        elif ".attn." in name:
            key = "window attention"
        elif ".cgc." in name:
            key = "cgc channels"
        else:
            key = "norms and gates"
        groups[key] = groups.get(key, 0) + p.numel()
    groups["total"] = sum(groups.values())
    return groups


def context_memory(name: str, *, horizon: int = HORIZON, chunk: int = 512, scalar_bytes: int = 2) -> dict:
    """One stream's context memory at ``horizon`` tokens. State tensors are INSTANTIATED (their dtype here is
    float32 = the accumulation precision of ERSA); scratch is PROJECTED at ``scalar_bytes`` per activation."""
    ersa, history = configuration(name, horizon=horizon, chunk=chunk)
    model, state = instantiate(name, horizon=horizon)
    m = ersa.model
    held = state.nbytes()
    instantiated = {
        "ersa_cells_fp32": held["ersa_cells"],
        "conv_tails": held["conv_tail"],
        # the window ring is allocated in the model dtype (float32 here); deployment keeps it at scalar_bytes
        "attention_window_kv": held["attention_window"] * scalar_bytes // 4,
        "attention_window_token_ring": held.get("token_history", 0),
        "token_history_int32": held["history_tokens"],
        "block_index": held["block_index"] * scalar_bytes // 4 if scalar_bytes < 4 else held["block_index"],
    }
    reads, slots, span = len(history.layers), history.slots, history.span
    projected = {
        "index_scores_per_chunk_fp32": reads * chunk * slots * 4,
        "reread_encoded_blocks_per_chunk": reads * chunk * history.topk * history.block * history.encoder_dim
        * scalar_bytes,
        "reread_readout_kv_per_chunk": reads * chunk * history.topk * history.block * 2 * history.read_kv_heads
        * history.read_head_dim * scalar_bytes,
        "encoder_activations_per_chunk": chunk * history.topk * span * history.encoder_dim * 8 * scalar_bytes,
        "chunk_activations": chunk * m.d_model * 16 * scalar_bytes,
        "fast_weights_last_block_cgc_with_momentum_fp32": 2 * m.cgc_bottleneck * m.cgc_cells * m.cgc_units * 4,
    }
    gpu = dict(instantiated)
    gpu.pop("token_history_int32")
    gpu.update(projected)
    gpu_total = sum(gpu.values())
    cpu = {"token_history_int32": instantiated["token_history_int32"]}
    return {"horizon": horizon, "chunk": chunk, "activation_bytes": scalar_bytes,
            "instantiated_state": instantiated, "projected_scratch": projected,
            "placement": {"gpu": gpu, "gpu_total": gpu_total, "cpu": cpu, "cpu_total": sum(cpu.values()),
                          "combined_total": gpu_total + sum(cpu.values())},
            "gpu_target_1GiB": gpu_total <= 1 << 30, "gpu_ceiling_2GiB": gpu_total <= 2 << 30,
            "dense_kv_cache_same_attention_geometry_at_horizon": len(ersa.window_layers) * 2 * ersa.attn_kv_heads
            * 128 * horizon * scalar_bytes,
            "dense_kv_cache_every_layer_at_horizon": m.n_layers * 2 * ersa.attn_kv_heads * 128 * horizon * scalar_bytes,
            "status": "INSTANTIATED state + PROJECTED scratch; nothing measured on a GPU"}


def compute_per_token(name: str, *, horizon: int = HORIZON, chunk: int = 512) -> dict:
    """PROJECTED multiply-accumulates for one decoded token at a full horizon."""
    ersa, history = configuration(name, horizon=horizon, chunk=chunk)
    model, _ = instantiate(name, horizon=horizon)
    m = ersa.model
    counts = parameter_breakdown(model)
    dense = counts["total"] - counts["history encoder"]            # every weight once (the head is tied)
    ersa_state = m.n_layers * ersa.ersa_heads * ersa.key_dim * ersa.value_dim * 4
    window = len(ersa.window_layers) * 2 * ersa.attn_heads * 128 * ersa.attn_window
    scoring = len(history.layers) * history.slots * history.per_block * history.index_dim
    encoder = counts["history encoder"]
    reread_uncached = len(history.layers) * history.topk * history.span * encoder
    readout = len(history.layers) * 2 * history.read_heads * history.read_head_dim * history.topk * history.block
    return {"weights": dense, "ersa_state_update_and_read": ersa_state, "window_attention": window,
            "index_scoring": scoring, "readout_attention": readout,
            "reread_encoder_uncached": reread_uncached,
            "total_uncached": dense + ersa_state + window + scoring + readout + reread_uncached,
            "total_if_every_selected_block_is_already_encoded": dense + ersa_state + window + scoring + readout,
            "note": "decode re-encodes each selected block (K x span tokens through the encoder) unless a bounded "
                    "cache of encoded blocks is added; that cache is not implemented. Prefill shares rereads "
                    "across the chunk. Scoring is linear in horizon/block.",
            "status": "PROJECTED"}


def training_memory(name: str, *, batch: int = 1, length: int = 4096, horizon: int = 4 * 4096) -> dict:
    """PROJECTED training memory: fp32 master weights + grads + AdamW moments, and a bf16 activation estimate."""
    ersa, history = configuration(name, horizon=horizon, chunk=min(length, 4095))
    model, _ = instantiate(name, horizon=horizon)
    p = sum(t.numel() for t in model.parameters())
    m = ersa.model
    tokens = batch * length
    per_token_layer = m.d_model * 14 + m.cgc_cells * m.cgc_units * 3 + ersa.ersa_heads * ersa.key_dim * 6
    activations = tokens * m.n_layers * per_token_layer * 2
    reads = tokens * len(history.layers) * history.topk * history.block * history.read_kv_heads \
        * history.read_head_dim * 2 * 2
    return {"parameters": p, "weights_fp32": 4 * p, "gradients_fp32": 4 * p, "adamw_moments_fp32": 8 * p,
            "activations_bf16_estimate": activations, "history_readout_gathers_bf16": reads,
            "total": 16 * p + activations + reads, "fits_24GiB": 16 * p + activations + reads <= 24 << 30,
            "batch": batch, "length": length,
            "note": "without activation checkpointing; chunked ERSA keeps per-block intermediates", "status": "PROJECTED"}


def measure_small(length: int = 1 << 20, *, chunk: int = 512) -> dict:
    """MEASURED on this CPU: the R3 experiment model's history state streamed to ``length`` tokens. Its size is
    allocated once; the record compares bytes before and after, and to the formula."""
    from .experiments.run_r3 import build
    model = build("history", "facts", 0, chunk=chunk, horizon=length).eval()
    state = model.new_state(1)
    before = state.nbytes()
    tokens = torch.randint(1, 30, (1, chunk), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        for _ in range(length // chunk):
            model(tokens, state)
    after = state.nbytes()
    h = model.history
    formula = {"history_tokens": h.history_capacity * 4,
               "block_index": h.slots * (h.per_block * h.index_dim * 4 + 8) * len(model.read_layers)}
    return {"tokens": state.position, "bytes_before": before, "bytes_after": after, "constant": before == after,
            "formula": formula, "matches_formula": all(after[k] == v for k, v in formula.items()),
            "status": "MEASURED"}


def report(*, measure: bool = True) -> dict:
    out = {"configurations": {}, "status": "INSTANTIATED / PROJECTED; no GPU, no trained large model"}
    for name in CONFIGURATIONS:
        model, _ = instantiate(name)
        ersa, history = configuration(name)
        out["configurations"][name] = {
            "parameters": parameter_breakdown(model),
            "weights_bytes": {"fp32": 4 * sum(p.numel() for p in model.parameters()),
                              "bf16": 2 * sum(p.numel() for p in model.parameters())},
            "topology": {"d_model": ersa.model.d_model, "layers": ersa.model.n_layers,
                         "ersa_heads": ersa.ersa_heads, "ersa_key_value_dim": ersa.key_dim,
                         "cgc": [ersa.model.cgc_bottleneck, ersa.model.cgc_cells, ersa.model.cgc_units],
                         "attention_layers": list(ersa.window_layers), "attention_query_heads": ersa.attn_heads,
                         "attention_kv_heads": ersa.attn_kv_heads, "attention_window": ersa.attn_window,
                         "history_read_layers": list(history.layers), "history_block": history.block,
                         "history_topk": history.topk, "index_dim": history.index_dim,
                         "index_keys_per_block": history.per_block, "vocab": VOCAB},
            "context_memory_1M": context_memory(name),
            "compute_per_token_1M": compute_per_token(name),
            "training_memory_4k": training_memory(name),
        }
    if measure:
        out["measured_small_model_state_at_1M"] = measure_small()
    return out


# -- R4: exact fixed state and bounded scratch of a streaming session ---------------------------------------------

def _width(dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def fixed_state_bytes(model, batch: int = 1, *, state_dtype=None) -> dict:
    """EXACT bytes of a fresh state of ``model`` (equal to ``model.new_state(batch).nbytes()``; tested), from the
    configuration alone. ``state_dtype`` overrides the parameter dtype for the floating state tiers."""
    c = model.ersa
    m = c.model
    w = _width(state_dtype or getattr(torch, m.dtype))
    out = {"ersa_cells": m.n_layers * batch * c.ersa_heads * c.value_dim * c.key_dim * w}
    if c.conv_kernel:
        out["conv_tail"] = m.n_layers * batch * (c.conv_kernel - 1) * m.d_model * w
    if c.attention == "window":
        head_dim = m.d_model // c.attn_heads
        out["attention_window"] = len(c.window_layers) * 2 * batch * c.attn_kv_heads * c.attn_window * head_dim * w
        out["token_history"] = batch * (c.attn_window + 1) * 4            # the window ring's own token slots
    h = getattr(model, "history", None)
    if h is not None:
        reads = len(model.read_layers)
        out["history_tokens"] = batch * h.history_capacity * 4
        out["block_index"] = reads * (batch * h.slots * h.per_block * h.index_dim * w + h.slots * 8)
    out["total"] = sum(out.values())
    return out


def session_scratch_bytes(model, *, batch: int = 1, chunk: int, scorer: str = "tiled", tile: int = 4096,
                          activation_bytes: int | None = None) -> dict:
    """PROJECTED upper bound of the per-chunk working memory of one forward call (not measured).

    Selection: the reference scorer holds the full ``[B, T, horizon/block (+ new)]`` score table with its distance,
    eligibility and pooled-match intermediates; the tiled scorer holds one tile and the running top-K. Reread:
    encoded selected blocks and their readout keys/values. New blocks: encoder activations. Logits: the head is
    applied to the positions asked for only (one per stream assumed)."""
    c, m = model.ersa, model.ersa.model
    w = activation_bytes or _width(getattr(torch, m.dtype))
    d, T, out = m.d_model, chunk, {}
    out["chunk_activations"] = batch * T * d * 16 * w + batch * T * c.ersa_heads * (c.key_dim + c.value_dim) * 4 * w
    out["logits"] = batch * m.vocab_size * w
    h = getattr(model, "history", None)
    if h is not None:
        reads = len(model.read_layers)
        records = h.slots + T // h.block + 1
        n = records if scorer == "reference" else min(tile, records)
        per_tile = batch * T * n * (h.per_block + 2) * w + T * n * (8 + w + 1)
        if scorer == "tiled":
            per_tile += batch * T * (h.topk + n) * (w + 8) * 2
        out["selection"] = reads * per_tile
        width = h.encoder_dim or d
        blocks = min(T * h.topk, h.slots) + T // h.block + 2
        kv = 2 * h.read_kv_heads * h.read_head_dim
        out["reread_states"] = reads * batch * blocks * (h.block + 1) * width * w
        out["readout"] = reads * batch * (blocks * h.block * kv + T * h.topk * h.block * kv * 2) * w
        out["encoder_activations"] = batch * (T // h.block + 2 + min(T * h.topk, h.slots)) * h.span * width * 8 * w
    out["total"] = sum(out.values())
    return out
