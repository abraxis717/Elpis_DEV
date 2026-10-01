"""Parametric text-backbone contract; source equations pinned in the tower spec."""
from dataclasses import dataclass
import math

from elpis.identity import content_digest
from ...contracts import Code, digest_value, integer, require

PROFILE = "ELPIS_DSV41_F32_DONOR_CACHE_V1"


@dataclass(frozen=True)
class TowerConfig:
    model: str
    tokenizer: str
    vocab: int
    dimension: int
    layers: int
    heads: int
    head_dim: int
    rope_dim: int
    q_rank: int
    o_groups: int
    o_rank: int
    local_window: int
    max_tokens: int
    compress_ratios: tuple[int, ...]
    kv_sources: tuple[int, ...]
    index_sources: tuple[int, ...]
    index_heads: int
    index_dim: int
    index_topk: int
    expert_count: int
    active_experts: int
    expert_dim: int
    engram_layers: tuple[int, ...]
    engram_order: int
    engram_heads: int
    engram_dim: int
    engram_bucket: int
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    norm_eps: float = 1e-20
    score_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.0
    swiglu_limit: float = 0.0
    shared_experts: int = 1
    rope_theta: float = 10000.0
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_factor: float = 40.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    candidate_source: int = -1
    candidate_blocks: int = 0
    candidate_block_size: int = 0
    numerical_profile: str = PROFILE

    def __post_init__(self):
        require(type(self.model) is str and bool(self.model), detail="tower model label")
        digest_value(self.tokenizer)
        for name in ("vocab", "dimension", "layers", "heads", "head_dim", "rope_dim", "q_rank",
                     "o_groups", "o_rank", "local_window", "max_tokens", "index_heads", "index_dim",
                     "index_topk", "expert_count", "active_experts", "expert_dim", "engram_heads",
                     "engram_dim", "engram_bucket", "hc_mult", "hc_sinkhorn_iters"):
            integer(getattr(self, name), 1)
        integer(self.engram_order, 2, 32)
        integer(self.original_seq_len)
        require(self.head_dim % 32 == self.index_dim % 32 == self.engram_dim % 32 == 0,
                detail="donor quantization groups")
        require(self.rope_dim % 2 == 0 and self.rope_dim <= min(self.head_dim, self.index_dim),
                detail="rotary tail")
        require(self.heads % self.o_groups == 0 and self.active_experts <= self.expert_count,
                detail="head/expert geometry")
        require(self.shared_experts == 1, Code.UNSUPPORTED, "donor has exactly one shared expert")
        require(type(self.norm_topk_prob) is bool and self.score_func in ("softmax", "sigmoid", "sqrtsoftplus"))
        require(self.numerical_profile == PROFILE, Code.UNSUPPORTED, "tower numerical profile")
        for name in ("hc_eps", "norm_eps", "gate_temp", "route_scale", "rope_theta", "compress_rope_theta",
                     "rope_factor", "beta_fast", "beta_slow"):
            x = getattr(self, name)
            require(type(x) in (float, int) and math.isfinite(x) and x > 0, detail=name)
        require(self.rope_theta > 1 and self.compress_rope_theta > 1, detail="rotary base")
        require(type(self.swiglu_limit) in (float, int) and math.isfinite(self.swiglu_limit)
                and self.swiglu_limit >= 0, detail="SwiGLU clamp")
        require(type(self.compress_ratios) is tuple and len(self.compress_ratios) == self.layers,
                detail="layer ratios")
        for ratio in self.compress_ratios:
            integer(ratio, 0, self.max_tokens)
        for entries in (self.kv_sources, self.index_sources, self.engram_layers):
            require(type(entries) is tuple and tuple(sorted(set(entries))) == entries, detail="ordered layer IDs")
            for layer in entries:
                integer(layer, 0, self.layers - 1)
        require(set(self.kv_sources) <= set(self.index_sources), detail="KV sources must publish index keys")
        kv = index = None
        for layer, ratio in enumerate(self.compress_ratios):
            if layer in self.kv_sources:
                kv = layer
            if layer in self.index_sources:
                index = layer
            if ratio:
                require(kv is not None and index is not None and index >= kv and
                        self.compress_ratios[kv] == self.compress_ratios[index] == ratio,
                        detail="causal shared-attention source/ratio")
            else:
                require(layer not in self.kv_sources + self.index_sources, detail="local layer cannot own compression")
        integer(self.candidate_source, -1, self.layers - 1)
        integer(self.candidate_blocks)
        integer(self.candidate_block_size)
        if self.candidate_source >= 0:
            require(self.candidate_source in self.index_sources and self.candidate_blocks > 0
                    and self.candidate_block_size > 0, detail="candidate source")
            ratio = self.compress_ratios[self.candidate_source]
            require(all(self.compress_ratios[i] == ratio for i in self.index_sources if i > self.candidate_source),
                    detail="candidate source/consumer ratio")
        else:
            require(self.candidate_blocks == self.candidate_block_size == 0, detail="disabled candidate geometry")

    @property
    def digest(self):
        return content_digest("elpis.inference.dsv41.config.v1", self)

    @property
    def hash_columns(self):
        return (self.engram_order - 1) * self.engram_heads

    @property
    def attention_capacity(self):
        return self.layers * self.local_window + sum(self.max_tokens // self.compress_ratios[i]
                                                    for i in self.kv_sources)


def tensor_shapes(c: TowerConfig):
    """Exact role set, donor [out,in] layout. No shape inference from candidate bytes."""
    d, hc = c.dimension, c.hc_mult
    shapes = {"embed": (c.vocab, d), "head": (c.vocab, d), "norm": (d,)}
    for layer in range(c.layers):
        prefix = f"layers.{layer}."
        local = {"attn_norm": (d,), "ffn_norm": (d,),
                 "attn.wq_a": (c.q_rank, d), "attn.q_norm": (c.q_rank,),
                 "attn.wq_b": (c.heads * c.head_dim, c.q_rank), "attn.wkv": (c.head_dim, d),
                 "attn.kv_norm": (c.head_dim,), "attn.attn_sink": (c.heads,),
                 "attn.wo_a": (c.o_groups * c.o_rank, c.heads * c.head_dim // c.o_groups),
                 "attn.wo_b": (d, c.o_groups * c.o_rank),
                 "ffn.gate.weight": (c.expert_count, d), "ffn.gate.bias": (c.expert_count,)}
        for sub in ("attn", "ffn"):
            local.update({f"hc_{sub}_fn": ((2 + hc) * hc, hc * d),
                          f"hc_{sub}_base": ((2 + hc) * hc,), f"hc_{sub}_scale": (3,)})
        if layer in c.kv_sources:
            local.update({"attn.compressor.wkv": (c.head_dim, d), "attn.compressor.norm": (c.head_dim,),
                          "attn.indexer.wk": (c.index_dim, c.head_dim), "attn.indexer.k_norm": (c.index_dim,)})
            if c.compress_ratios[layer] > 1:
                local["attn.compressor.wgate"] = (c.head_dim, d)
        if layer in c.index_sources:
            local.update({"attn.indexer.wq_b": (c.index_heads * c.index_dim, c.q_rank),
                          "attn.indexer.weights_proj": (c.index_heads, d)})
        if layer in c.engram_layers:
            local.update({"engram.wkv": ((hc + 1) * d, c.hash_columns * c.engram_dim),
                          "engram.q_weight": (hc, d), "engram.k_weight": (hc, d)})
        for expert in (*map(str, range(c.expert_count)), "shared"):
            for role, shape in (("w1", (c.expert_dim, d)), ("w3", (c.expert_dim, d)), ("w2", (d, c.expert_dim))):
                local[f"ffn.experts.{expert}.{role}"] = shape
        shapes.update((prefix + name, shape) for name, shape in local.items())
    return shapes
