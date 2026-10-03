"""Donor SWA + shared compressed sparse attention (including candidate stage).

All mutable storage belongs to one sequence. No module-global shared-attention
slots, no prefix reconstruction, no cache persists in the committed state.
"""
from dataclasses import dataclass
import numpy as np
from .numerics import F32, linear, quant_dequant, rms, rotate, softmax


@dataclass
class AttentionState:
    window: np.ndarray
    compressed: np.ndarray | None
    index_keys: np.ndarray | None
    pending: np.ndarray | None
    scores: np.ndarray | None
    count: int = 0

    @classmethod
    def create(cls, c, layer):
        owner = layer in c.kv_sources
        ratio = c.compress_ratios[layer]
        capacity = c.max_tokens // ratio if owner else 0
        return cls(np.zeros((c.local_window, c.head_dim), dtype=F32),
                   np.zeros((capacity, c.head_dim), dtype=F32) if owner else None,
                   np.zeros((capacity, c.index_dim), dtype=F32) if owner else None,
                   np.zeros((ratio, c.head_dim), dtype=F32) if owner and ratio > 1 else None,
                   np.zeros((ratio, c.head_dim), dtype=F32) if owner and ratio > 1 else None)

    @property
    def nbytes(self):
        return sum(x.nbytes for x in (self.window, self.compressed, self.index_keys, self.pending, self.scores)
                   if x is not None)


@dataclass
class SharedAttention:
    owner: AttentionState | None = None
    selected: np.ndarray | None = None
    candidates: np.ndarray | None = None


def candidate_mask(logits, top_blocks, block_size):
    """Block maximum with newest block pinned. Ties: lower block position."""
    width = len(logits)
    if not width:
        return np.zeros(0, dtype=bool)
    scores = np.full((width + block_size - 1) // block_size, -np.inf, dtype=F32)
    for block in range(len(scores)):
        scores[block] = np.max(logits[block*block_size:min((block+1)*block_size, width)])
    scores[(width - 1) // block_size] = np.inf
    top = np.argsort(-scores, kind="stable")[:min(top_blocks, len(scores))]
    keep = np.zeros(len(scores), dtype=bool)
    keep[top] = scores[top] > -np.inf
    return np.repeat(keep, block_size)[:width]


def select_positions(scores, topk):
    return np.sort(np.argsort(-scores, kind="stable")[:min(topk, len(scores))])


def sparse_attention(query, kv, sink):
    """A zero-value learned sink participates only in the denominator."""
    scores = np.einsum("hd,nd->hn", query, kv, optimize=False, dtype=F32) * F32(query.shape[-1] ** -.5)
    maximum = np.maximum(np.max(scores, axis=-1), sink)
    weights = np.exp(scores - maximum[:, None])
    denominator = np.sum(weights, axis=-1, dtype=F32) + np.exp(sink - maximum)
    return np.einsum("hn,nd->hd", weights / denominator[:, None], kv, optimize=False, dtype=F32)


class Attention:
    def __init__(self, c, layer, weights, freqs):
        self.config, self.layer, self.w, self.freqs = c, layer, weights, freqs
        self.kv_owner = layer in c.kv_sources
        self.index_source = layer in c.index_sources
        self.ratio = c.compress_ratios[layer]

    def _compress(self, x, state, position):
        c, w, ratio = self.config, self.w, self.ratio
        kv = linear(x, w["compressor.wkv"])
        if ratio > 1:
            slot = position % ratio
            state.pending[slot] = kv
            state.scores[slot] = linear(x, w["compressor.wgate"])
            if (position + 1) % ratio:
                return None
            kv = np.sum(state.pending * softmax(state.scores, axis=0), axis=0, dtype=F32)
        return rms(kv, w["compressor.norm"], c.norm_eps)

    def apply(self, x, state, shared, position):
        c, w = self.config, self.w
        freq = self.freqs[position]
        qr = rms(linear(x, w["wq_a"]), w["q_norm"], c.norm_eps)
        query = rotate(linear(qr, w["wq_b"]).reshape(c.heads, c.head_dim), freq)
        local = quant_dequant(rotate(rms(linear(x, w["wkv"]), w["kv_norm"], c.norm_eps), freq), "local")
        state.window[position % c.local_window] = local
        # Chronological ring order matches donor decode; no history-sized copy.
        start = max(0, position + 1 - c.local_window)
        slots = np.arange(start, position + 1) % c.local_window
        kv = state.window[slots]
        selected = np.zeros(0, dtype=np.int64)
        if self.ratio:
            n = (position + 1) // self.ratio
            latent = None
            if self.kv_owner:
                shared.owner = state
                latent = self._compress(x, state, position)
                if latent is not None:
                    key = rms(linear(latent, w["indexer.wk"]), w["indexer.k_norm"], c.norm_eps)
                    group_freq = self.freqs[position + 1 - self.ratio]
                    state.index_keys[n - 1] = quant_dequant(rotate(key, group_freq), "index")
            owner = shared.owner
            if self.index_source:
                if n:
                    iq = linear(qr, w["indexer.wq_b"]).reshape(c.index_heads, c.index_dim)
                    iq = quant_dequant(rotate(iq, freq), "index")
                    iw = linear(x, w["indexer.weights_proj"]) * F32(c.index_dim ** -.5 * c.index_heads ** -.5)
                    score = np.einsum("hd,nd->hn", iq, owner.index_keys[:n], dtype=F32, optimize=False)
                    score = np.sum(np.maximum(score, F32(0)) * iw[:, None], axis=0, dtype=F32)
                    if self.layer == c.candidate_source:
                        shared.candidates = candidate_mask(score, c.candidate_blocks, c.candidate_block_size)
                    elif 0 <= c.candidate_source < self.layer:
                        score = np.where(shared.candidates, score, -np.inf)
                    selected = select_positions(score, c.index_topk)
                shared.selected = selected
            else:
                selected = shared.selected
            # Indexer consumes pre-RoPE latent before the compressed cache write.
            if latent is not None:
                state.compressed[n - 1] = quant_dequant(rotate(latent, group_freq), "compressed")
            if self.kv_owner:
                state.count = n
            if len(selected):
                kv = np.concatenate((kv, owner.compressed[selected]), axis=0)
        out = rotate(sparse_attention(query, kv, w["attn_sink"]), freq, inverse=True)
        grouped = out.reshape(c.o_groups, -1)
        projection = w["wo_a"].reshape(c.o_groups, c.o_rank, -1)
        out = np.einsum("gi,gri->gr", grouped, projection, dtype=F32, optimize=False)
        return linear(out.reshape(-1), w["wo_b"]), tuple(int(i) for i in selected)
