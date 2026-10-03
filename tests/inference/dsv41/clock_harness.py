"""Tokenizer-independent SYNTHETIC clock fixture, never donor qualification.

Uses the canonical TowerConfig, TensorStore, RowEngine, Layer, DSV41Target
recurrence methods and real provider admission. Only cold tokenizer intake is
absent: raw token IDs and explicit synthetic address parameters are the input.
This permits clock/protocol qualification when the separately pinned text
environment is unavailable, without changing its version gate or oracle.
"""
from hashlib import sha256
import json
import shutil
from pathlib import Path
from types import SimpleNamespace, MappingProxyType

import numpy as np

from elpis.identity import content_digest
from elpis.inference.associative import AddressScheme, DSV41Parameters
from elpis.inference.contracts import Bank
from elpis.inference.rows import RowEngine, RowTable, row_representation
from elpis.inference.target import Tensor
from elpis.inference.drivers.dsv41.config import tensor_shapes
from elpis.inference.drivers.dsv41.fixtures import fixture_config
from elpis.inference.drivers.dsv41.layer import Layer
from elpis.inference.drivers.dsv41.numerics import rope_frequencies
from elpis.inference.drivers.dsv41.parameters import TensorStore, TensorBinding, FileTensor, ParameterManifest
from elpis.inference.drivers.dsv41.target import DSV41Target
from elpis.inference.drivers.dsv41.native_backend import DSV41NativeBackend
from elpis.substrate.authority import PinnedAuthority
from elpis.substrate.boundary import RootCapability, load_native
from elpis.substrate.file_assets import inspect_asset
from elpis.substrate.digests import raw_digest
from . import provider_harness as H


class RawTokenTower:
    """Exact checked-in recurrence functions on explicitly synthetic cold data."""
    window_initial = DSV41Target.window_initial
    window_step = DSV41Target.window_step
    _provider_step = DSV41Target._provider_step
    release_window = DSV41Target.release_window
    admit_stream = DSV41Target.admit_stream
    accepts_conditioning = DSV41Target.accepts_conditioning
    conditioning_vector = DSV41Target.conditioning_vector
    conditioning_projection = _conditioning_weights = None


def target(rig, fms, name, native_backend=None, provider_stream=None, row_mutator=None, conditioning_projection=None):
    c, p = rig.config, rig.parameters
    directory = rig.workspace / name
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1041)
    bindings, chunks, specs = [], [], []
    offset = 0
    for role, shape in sorted(tensor_shapes(c).items()):
        if role.endswith(("norm", "q_weight", "k_weight")):
            a = (1 + rng.normal(0, .04, shape)).astype("<f4")
        elif role.endswith("_scale"):
            a = np.array([.08, .09, .07], "<f4")
        else:
            a = rng.normal(0, .12, shape).astype("<f4")
        t = Tensor(shape, a.tobytes())
        if ".ffn.experts." in role:
            specs.append((role, t, offset)); chunks.append(t.data); offset += len(t.data)
        else:
            bindings.append(TensorBinding(c.model, role, shape, t))
    path = directory / "fixture-experts.dat"
    path.write_bytes(b"".join(chunks))
    asset = inspect_asset(fms.root, path, 4096)
    aid = fms.register(path, asset, expected_manifest=asset.digest)
    bindings += [TensorBinding(c.model, role, t.shape, FileTensor(aid, off, raw_digest(t.data))) for role, t, off in specs]
    rows = {}
    for i, layer in enumerate(c.engram_layers):
        n = p.table_rows[i]
        codes = rng.integers(0, 96, (n, c.engram_dim), dtype=np.uint8)
        codes |= rng.integers(0, 2, codes.shape, dtype=np.uint8) * 128
        scales = np.full((n, c.engram_dim // 32), 119, dtype=np.uint8)
        if row_mutator is not None:  # adversarial stored codes; admission still verifies the bytes
            codes, scales = row_mutator(codes, scales)
        path = directory / f"rows-{layer}.dat"
        path.write_bytes(np.concatenate((codes, scales), axis=1).tobytes())
        asset = inspect_asset(fms.root, path, 4096)
        aid = fms.register(path, asset, expected_manifest=asset.digest)
        codec = "DS4_E4M3_E8M0_BF16"
        b = Bank(f"synthetic-clock-{layer}", c.model, c.tokenizer, p.schema, p.digest, layer, n, c.engram_dim,
                 asset.content, row_representation(aid, 0, n, c.engram_dim, codec))
        rows[layer] = RowEngine(fms, RowTable(b, aid, 0, codec), expected_bank=b.digest, max_rows=c.hash_columns)
    manifest = ParameterManifest(c.model, c.tokenizer, c.digest, p.digest,
                                tuple((i, rows[i].table.bank.digest) for i in c.engram_layers), tuple(bindings))
    t = RawTokenTower()
    t.config, t.scheme, t.rows = c, AddressScheme(p, expected_digest=p.digest, tokenizer=p.tokenizer, scheme=p.schema), rows
    for row in rows.values():
        t.scheme.validate_bank(row.table.bank)
    t.store = TensorStore(c, manifest, fms, expected_manifest=manifest.digest, resident_budget=64 << 20, staging_budget=1 << 16)
    t.native_backend, t.provider_stream = native_backend, provider_stream
    frequencies = {b: rope_frequencies(c, b) for b in (False, True)}
    t.layers = tuple(Layer(c, i, t.store, rows.get(i), frequencies[bool(c.compress_ratios[i])],
                           native_backend=native_backend) for i in range(c.layers))
    t.model_identity = content_digest("synthetic-clock-model", manifest.digest)
    if conditioning_projection is not None:
        t.conditioning_projection = conditioning_projection
        t._conditioning_weights = conditioning_projection.tensor.array()
        t.model_identity = content_digest("synthetic-clock-model", [manifest.digest, conditioning_projection.digest])
    t.numerical_profile = content_digest("synthetic-clock-numerics", c.numerical_profile)
    if provider_stream:
        provider_stream.admit(t, frequencies)
        t.numerical_profile = content_digest("synthetic-clock-provider", provider_stream.profile)
    t._row_index = {l: i for i, l in enumerate(c.engram_layers)}
    t._initial_pre = np.zeros(c.hc_mult, dtype="<f4"); t._initial_pre[0] = 1
    t.principal_working_set, t.last_metrics = (c.attention_capacity, max(c.head_dim, c.index_dim)), {}
    return t


class Rig(H.Rig):
    def fms(self, **kwargs):
        fms = super().fms(**kwargs)
        self.file_assets.append(fms)
        return fms

    def close(self):
        super().close()
        for fms in self.file_assets:
            fms.close()
        self.file_assets.clear()

    def native_target(self, fms, name, **kwargs):
        return target(self, fms, name, native_backend=self.native, **kwargs)

    def numpy_target(self, fms, name, **kwargs):
        return target(self, fms, name, **kwargs)

    def provider_target(self, fms, name, provider, **kwargs):
        return target(self, fms, name, provider_stream=provider, **kwargs)

    def tokens(self, count=24):
        return tuple((i * 7 + 3) % 32 for i in range(count))


def make_rig(workspace):
    paths, entries = {}, []
    (workspace / "native").mkdir()
    for stem in H.LIBRARIES:
        source = H.require_native_library(stem)
        dest = workspace / "native" / source.name
        shutil.copyfile(source, dest)
        paths[stem] = Path("native") / source.name
        data = dest.read_bytes()
        entries.append(dict(library_id=stem, size=len(data), sha256=sha256(data).hexdigest()))
    doc = json.dumps(dict(schema="elpis.inference-authority.v1", source="native clock synthetic test",
                         provenance="synthetic-test", assets=[], libraries=entries)).encode()
    authority = PinnedAuthority(doc, expected_sha256=sha256(doc).hexdigest())
    native = DSV41NativeBackend(workspace, paths["elpis_dsv41_native"], authority=authority, library_id="elpis_dsv41_native")
    with RootCapability(workspace) as root:
        ref = load_native(root, paths["elpis_dsv41_stream_reference_provider"], authority.libraries["elpis_dsv41_stream_reference_provider"])
    import ctypes as C
    ref.elpis_dsv41_reference_provider_configure.argtypes = [C.POINTER(H.RefConfig)]
    ref.elpis_dsv41_reference_provider_configure.restype = None
    ref.elpis_dsv41_reference_provider_counters.argtypes = [C.POINTER(H.RefCounters)]
    ref.elpis_dsv41_reference_provider_counters.restype = None
    tok = SimpleNamespace(identity="a" * 64, vocab_size=32)
    c = fixture_config(tok, max_tokens=32)
    primes = ((17, 19, 23, 29, 31, 37), (41, 43, 47, 53, 59, 61))
    offsets = tuple(tuple(sum(group[:i]) for i in range(6)) for group in primes)
    p = DSV41Parameters(tok.identity, tuple(i // 2 for i in range(32)), 16, 5, c.engram_layers, 4, 2, 32,
                        ((11, 13, 17, 19), (23, 29, 31, 37)), primes, offsets, tuple(map(sum, primes)))
    rig = Rig(workspace, authority, paths, tok, c, p, native, ref, [])
    rig.file_assets = []
    rig.configure()
    return rig
