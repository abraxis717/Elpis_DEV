"""PRODUCTION_SHAPED_FIXTURE, TRAINING=NONE. Explicit test-only materialization.

Every tensor is deterministic untrained data. The topology and full tokenizer
vocabulary are real; no forced logits, scripted output or language claim.
"""
from dataclasses import replace
import numpy as np

from ...associative import AddressScheme
from ...contracts import Bank
from ...rows import RowEngine, RowTable, row_representation
from ...target import Tensor
from elpis.substrate.file_assets import bounded_path, inspect_asset
from elpis.substrate.digests import raw_digest
from .config import TowerConfig, tensor_shapes
from .engram import build_parameters
from .parameters import FileTensor, TensorBinding, ParameterManifest
from .target import DSV41Target


def fixture_config(tokenizer, *, seed=1041, max_tokens=128):
    return TowerConfig(
        model=f"PRODUCTION_SHAPED_FIXTURE-DSV41-seed{seed}", tokenizer=tokenizer.identity,
        vocab=tokenizer.vocab_size, dimension=8, layers=7, heads=2, head_dim=32, rope_dim=8,
        q_rank=8, o_groups=2, o_rank=4, local_window=4, max_tokens=max_tokens,
        compress_ratios=(0, 2, 2, 2, 1, 1, 1), kv_sources=(1, 4), index_sources=(1, 2, 4, 5),
        index_heads=2, index_dim=32, index_topk=2, expert_count=4, active_experts=2, expert_dim=12,
        engram_layers=(1, 5), engram_order=4, engram_heads=2, engram_dim=32, engram_bucket=17,
        candidate_source=4, candidate_blocks=2, candidate_block_size=2,
        original_seq_len=32, rope_factor=4.0, swiglu_limit=10.0, route_scale=1.5)


def make_fixture(provider, directory, tokenizer, *, seed=1041, config=None, parameters=None, native_backend=None):
    """Create only fixture-owned raw files beneath the provider root."""
    directory = bounded_path(provider.root, directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = config or fixture_config(tokenizer, seed=seed)
    parameters = parameters or build_parameters(tokenizer, config)
    rng = np.random.default_rng(seed)
    resident, expert_specs, chunks = [], [], []
    offset = 0
    for role, shape in sorted(tensor_shapes(config).items()):
        if role.endswith(("norm", "q_weight", "k_weight")):
            array = (1 + rng.normal(0, .04, shape)).astype("<f4")
        elif role.endswith("_scale"):
            array = np.array([.08, .09, .07], dtype="<f4")
        else:
            array = rng.normal(0, .12, shape).astype("<f4")
        tensor = Tensor(shape, array.tobytes())
        if ".ffn.experts." in role:
            expert_specs.append((role, tensor, offset))
            chunks.append(tensor.data)
            offset += len(tensor.data)
        else:
            resident.append(TensorBinding(config.model, role, shape, tensor))
    path = directory / "fixture-experts.dat"
    path.write_bytes(b"".join(chunks))
    artifact = inspect_asset(provider.root, path, 4096)
    asset = provider.register(path, artifact, expected_manifest=artifact.digest)
    bindings = resident + [TensorBinding(config.model, role, t.shape, FileTensor(asset, off, raw_digest(t.data)))
                           for role, t, off in expert_specs]
    rows = {}
    for index, layer in enumerate(config.engram_layers):
        n = parameters.table_rows[index]
        # Actual donor row representation: E4M3 data then E8M0 scales, BF16 decode.
        codes = rng.integers(0, 96, (n, config.engram_dim), dtype=np.uint8)
        codes |= rng.integers(0, 2, codes.shape, dtype=np.uint8) * 128
        scales = np.full((n, config.engram_dim // 32), 119, dtype=np.uint8)
        raw = np.concatenate((codes, scales), axis=1).tobytes()
        path = directory / f"fixture-engram-{layer}.dat"
        path.write_bytes(raw)
        artifact = inspect_asset(provider.root, path, 4096)
        row_asset = provider.register(path, artifact, expected_manifest=artifact.digest)
        codec = "DS4_E4M3_E8M0_BF16"
        bank = Bank(f"fixture-engram-{layer}", config.model, config.tokenizer, parameters.schema,
                    parameters.digest, layer, n, config.engram_dim, artifact.content,
                    row_representation(row_asset, 0, n, config.engram_dim, codec))
        rows[layer] = RowEngine(provider, RowTable(bank, row_asset, 0, codec), expected_bank=bank.digest,
                                workers=1, max_rows=config.hash_columns)
    manifest = ParameterManifest(config.model, config.tokenizer, config.digest, parameters.digest,
                                 tuple((i, rows[i].table.bank.digest) for i in config.engram_layers), tuple(bindings))
    target = DSV41Target(config, manifest, tokenizer, parameters, rows, provider, expected_manifest=manifest.digest,
                         resident_budget=64 << 20, staging_budget=1 << 16, state_budget=16 << 20,
                         native_backend=native_backend)
    metadata = dict(classification="PRODUCTION_SHAPED_FIXTURE", TRAINING="NONE", seed=seed,
                    model=target.model_identity, tokenizer=tokenizer.identity, config=config.digest,
                    manifest=manifest.digest, address=parameters.digest)
    return target, metadata
