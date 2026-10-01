"""Pinned donor CPU differential harness; never imported by Elpis runtime.

Execute unchanged high-level definitions via AST extraction. GPU/storage leaves
are independently expressed Torch CPU F32 equivalents of the declared profile.
This is not GPU/BF16-bitwise parity. Unspecified top-k ties use lower indices.
"""
import ast
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from types import ModuleType
from typing import Literal
from unittest.mock import patch
import sys
import numpy as np
import pytest


def read_pinned(relative):
    root = os.environ.get("ELPIS_TOWER_DONORS")
    if not root:
        pytest.skip("ELPIS_TOWER_DONORS required for pinned donor differentials")
    path = Path(__file__).parents[3] / "docs/inference/dsv41_donor_authority.json"
    authority = json.loads(path.resolve().read_text())
    data = (Path(root) / relative).read_bytes()
    assert hashlib.sha256(data).hexdigest() == authority["files"][relative]["sha256"]
    return data.decode()


def load_definitions(relative, names, namespace, module_name):
    tree = ast.parse(read_pinned(relative), filename=relative)
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in tree.body} == set(names)
    module = ModuleType(module_name)
    module.__dict__.update(namespace)
    sys.modules[module_name] = module
    exec(compile(tree, relative, "exec"), module.__dict__)
    return module


def load_engram():
    torch = pytest.importorskip("torch")
    from sympy import isprime
    return load_definitions("DeepSeek4.1/00_V41_REFERENCE/inference/engram.py",
                            ("find_next_prime", "build_compressed_token_map", "compute_hash_multipliers",
                             "EngramLayout", "NgramHashState"),
                            dict(torch=torch, nn=torch.nn, np=np, isprime=isprime, dataclass=dataclass),
                            "elpis_test_donor_engram")


def torch_cache_quant(x, mode):
    import torch
    block = 16 if mode == "compressed" else 32
    groups = x.float().unflatten(-1, (-1, block))
    amax = groups.abs().amax(-1, keepdim=True)
    if mode == "local":
        scale = (amax.clamp_min(1e-4) / 448).log2().ceil().exp2()
        out = (groups / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
    else:
        scale = ((amax.clamp_min(6 * 2.0 ** -126) / 6).log2().ceil().exp2() if mode == "index" else
                 (amax.clamp_min(6 * 2.0 ** -9) / 6).to(torch.float8_e4m3fn).float())
        z = (groups / scale).clamp(-6, 6)
        levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
        distances = (z.abs().unsqueeze(-1) - levels).abs()
        tied = distances == distances.amin(-1, keepdim=True)
        priority = torch.tensor([0, 9, 2, 11, 4, 13, 6, 15])
        pick = torch.where(tied, priority, 99).argmin(-1)
        out = torch.copysign(levels[pick], z) * scale
    return out.flatten(-2).to(x.dtype)


def torch_hc(mixes, scale, base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    import torch
    h = hc_mult
    pre = torch.sigmoid(mixes[..., :h] * scale[0] + base[:h]) + eps
    post = 2 * torch.sigmoid(mixes[..., h:2*h] * scale[1] + base[h:2*h])
    comb = (mixes[..., 2*h:] * scale[2] + base[2*h:]).unflatten(-1, (h, h)).softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def torch_sparse(q, kv, sink, indices, scale):
    import torch
    output = torch.zeros_like(q)
    for b in range(q.shape[0]):
        for t in range(q.shape[1]):
            ids = indices[b, t]
            values = kv[b, ids[ids >= 0]]
            scores = torch.einsum("hd,nd->hn", q[b, t], values) * scale
            prob = torch.cat((scores, sink[:, None]), -1).softmax(-1)[..., :-1]
            output[b, t] = torch.einsum("hn,nd->hd", prob, values)
    return output


def load_model():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    engram = load_engram()
    def act_quant(x, block_size=32, scale_fmt=None, scale_dtype=None, inplace=False):
        assert inplace and block_size == 32
        x.copy_(torch_cache_quant(x, "local"))
        return x
    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=None):
        assert inplace
        x.copy_(torch_cache_quant(x, "compressed" if block_size == 16 else "index"))
        return x
    namespace = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, dist=torch.distributed,
                     np=np, math=math, dataclass=dataclass, contextmanager=contextmanager, lru_cache=lru_cache,
                     Literal=Literal, world_size=1, rank=0, default_dtype=torch.float32,
                     fp8_block_size=32, fp4_block_size=32, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu,
                     EngramLayout=engram.EngramLayout, NgramHashState=engram.NgramHashState,
                     act_quant=act_quant, fp4_act_quant=fp4_act_quant,
                     sparse_attn=torch_sparse, hc_split_sinkhorn=torch_hc)
    names = ("set_dtype", "ModelArgs", "ParallelEmbedding", "linear", "Linear", "ColumnParallelLinear",
             "RowParallelLinear", "RMSNorm", "ParallelEngramEmbedding", "Engram", "precompute_freqs_cis",
             "apply_rotary_emb", "get_window_topk_idxs", "Compressor", "Indexer", "select_candidate_blocks",
             "Attention", "Gate", "Expert", "MoE", "Block", "ParallelHead", "make_identity_pre_mix",
             "SharedAttentionRuntime", "Transformer", "sample")
    module = load_definitions("DeepSeek4.1/00_V41_REFERENCE/inference/model.py", names, namespace,
                              "elpis_test_donor_model")
    module.shared_attn = module.SharedAttentionRuntime()
    return module


def donor_args(module, c, p):
    return module.ModelArgs(
        max_batch_size=1, max_seq_len=c.max_tokens, temperature=0, dtype="bf16", expert_dtype=None,
        vocab_size=c.vocab, dim=c.dimension, n_layers=c.layers, n_mtp_layers=0, n_heads=c.heads,
        moe_inter_dim=c.expert_dim, n_routed_experts=c.expert_count, n_shared_experts=1,
        n_activated_experts=c.active_experts, score_func=c.score_func, gate_temp=c.gate_temp,
        norm_topk_prob=c.norm_topk_prob, route_scale=c.route_scale, swiglu_limit=c.swiglu_limit,
        q_lora_rank=c.q_rank, head_dim=c.head_dim, rope_head_dim=c.rope_dim, norm_eps=c.norm_eps,
        o_groups=c.o_groups, o_lora_rank=c.o_rank, window_size=c.local_window,
        compress_ratios=c.compress_ratios, kv_source_layers=c.kv_sources, index_source_layers=c.index_sources,
        compress_rope_theta=c.compress_rope_theta, original_seq_len=c.original_seq_len,
        rope_theta=c.rope_theta, rope_factor=c.rope_factor, beta_fast=c.beta_fast, beta_slow=c.beta_slow,
        index_n_heads=c.index_heads, index_head_dim=c.index_dim, index_topk=c.index_topk,
        candidate_source_layer=c.candidate_source, candidate_topk_blocks=c.candidate_blocks,
        candidate_block_size=c.candidate_block_size, hc_mult=c.hc_mult,
        hc_sinkhorn_iters=c.hc_sinkhorn_iters, hc_eps=c.hc_eps, engram_layer_ids=c.engram_layers,
        engram_num_embeddings=p.table_rows, engram_max_ngram_size=c.engram_order,
        engram_vocab_size=c.engram_bucket, engram_n_heads=c.engram_heads, engram_head_dim=c.engram_dim,
        engram_compressed_vocab_size=p.compressed_vocab)


class TokenizerView:
    def __init__(self, tokenizer):
        self.backend_tokenizer = tokenizer._encoder
        self.size = tokenizer.vocab_size
    def __len__(self):
        return self.size


@contextmanager
def stable_topk():
    import torch
    TopK = namedtuple("TopK", "values indices")
    def topk(self, k, dim=-1, largest=True, sorted=True):
        ids = torch.argsort(self, dim=dim, descending=largest, stable=True).narrow(dim, 0, k)
        return TopK(self.gather(dim, ids), ids)
    with patch.object(torch.Tensor, "topk", topk):
        yield


@contextmanager
def owner_binds_index_keys(module):
    """Test-harness correction for one pinned-donor decode defect (nothing else changes).

    SharedAttentionRuntime documents that "every source writes before its consumers read",
    but Indexer.forward rebinds `shared_attn.index_k` only when the owner just completed a
    group. During decode, a compress_ratio > 1 owner whose group is still filling therefore
    scores against whatever owner published last -- in the production layout, layer 20's
    ratio-1 keys from the previous token -- and the donor's own prefill and decode paths
    disagree at every group-starting position
    (test_donor_decode_index_slot_defect_against_its_own_prefill). Here an owner binds its
    own key cache every step, exactly as its prefill path does.
    """
    original = module.Indexer.forward
    def forward(self, x, qr, latent, start_pos, offset):
        if self.owns_k:
            module.shared_attn.index_k = self.k_cache
        return original(self, x, qr, latent, start_pos, offset)
    module.Indexer.forward = forward
    try:
        yield
    finally:
        module.Indexer.forward = original


def build_oracle(target, tokenizer):
    import torch
    from elpis.inference.contracts import RowIdentity
    module = load_model()
    c, p = target.config, target.scheme.parameters
    args = donor_args(module, c, p)
    model = module.Transformer(args, TokenizerView(tokenizer)).float()
    donor_weights = dict(model.named_parameters())
    for role, binding in target.store.bindings.items():
        name = role.replace(".ffn.experts.shared.", ".ffn.shared_experts.")
        if not name.endswith((".weight", ".bias", "_fn", "_base", "_scale", "attn_sink", "q_weight", "k_weight")):
            name += ".weight"
        value = np.frombuffer(target.store._read(binding), dtype="<f4").reshape(binding.shape)
        with torch.no_grad():
            donor_weights[name].copy_(torch.from_numpy(value.copy()))
    for layer, rows in target.rows.items():
        values = []
        for start in range(0, rows.table.bank.rows, c.hash_columns):
            request = tuple(RowIdentity(rows.bank_identity, i)
                            for i in range(start, min(start+c.hash_columns, rows.table.bank.rows)))
            values.append(rows.lookup(request))
        model.layers[layer].engram.embed = torch.nn.Embedding.from_pretrained(
            torch.tensor(np.concatenate(values)), freeze=True)
    return model, module
