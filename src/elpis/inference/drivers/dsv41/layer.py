"""Engram before the block, delayed mHC pre-mixes, attention then MoE."""
from time import perf_counter_ns
from .attention import Attention
from .engram import EngramLayer
from .moe import MoE
from .numerics import hc_mixes, hc_pre, hc_post, rms


class Layer:
    def __init__(self, c, layer, store, rows, freqs):
        self.config = c
        prefix = f"layers.{layer}."
        self.w = {name[len(prefix):]: value for name, value in store.dense.items() if name.startswith(prefix)}
        self.attention = Attention(c, layer, {k[5:]: v for k, v in self.w.items() if k.startswith("attn.")}, freqs)
        self.moe = MoE(c, layer, store)
        self.engram = None if rows is None else EngramLayer(c, rows, {
            k[7:]: v for k, v in self.w.items() if k.startswith("engram.")})

    def apply(self, stream, pre_mix, attention_state, shared, position, row_ids, metrics):
        c, w = self.config, self.w
        if self.engram is not None:
            start = perf_counter_ns()
            stream = self.engram.apply(stream, row_ids)
            metrics["engram_ns"] += perf_counter_ns() - start
        start = perf_counter_ns()
        ap, apost, ac = hc_mixes(stream, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"], c)
        x = rms(hc_pre(stream, pre_mix), w["attn_norm"], c.norm_eps)
        metrics["mhc_norm_ns"] += perf_counter_ns() - start
        start = perf_counter_ns()
        out, selected = self.attention.apply(x, attention_state, shared, position)
        metrics["attention_ns"] += perf_counter_ns() - start
        start = perf_counter_ns()
        stream = hc_post(out, stream, apost, ac)
        fp, fpost, fc = hc_mixes(stream, w["hc_ffn_fn"], w["hc_ffn_scale"], w["hc_ffn_base"], c)
        x = rms(hc_pre(stream, ap), w["ffn_norm"], c.norm_eps)
        metrics["mhc_norm_ns"] += perf_counter_ns() - start
        start = perf_counter_ns()
        out, route = self.moe.apply(x)
        metrics["moe_ns"] += perf_counter_ns() - start
        start = perf_counter_ns()
        stream = hc_post(out, stream, fpost, fc)
        metrics["mhc_norm_ns"] += perf_counter_ns() - start
        return stream, fp, route, selected
