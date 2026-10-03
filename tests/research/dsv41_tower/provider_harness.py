"""Shared YTS-R0 qualification harness (test-only).

Builds matching DSV4.1 fixture towers on the NumPy reference, the in-process
DSV41NativeBackend and the YTS-R0 provider stream backed by the test-only
reference provider, all from the same pinned synthetic-test authority.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np

from research.dsv41_tower import fixtures as fixtures_mod
from research.dsv41_tower.fixtures import fixture_config, make_fixture
from research.dsv41_tower.native_backend import DSV41NativeBackend
from research.dsv41_tower.provider_stream import DSV41StreamProvider
from elpis.substrate.authority import PinnedAuthority
from elpis.substrate.boundary import RootCapability, load_native
from elpis.substrate.synthetic import SyntheticFileAssets

from ...conftest import require_native_library

LIBRARIES = ("elpis_execution", "elpis_dsv41_stream_reference_provider", "elpis_dsv41_native",
             "elpis_fms_file_assets")
RTOL, ATOL = 1e-4, 1e-5
TEXT = "Hello, Elpis. Multi layer attention and memory exercise."

# Reference-provider test controls (dsv41_reference_provider.c).
MODE_SYNC, MODE_DEVICE = 0, 1
FAULT_NONE, FAULT_SUBMIT_REJECT, FAULT_POLL_FAIL, FAULT_POLL_TIMEOUT = 0, 1, 2, 3
FAULT_BAD_ECHO, FAULT_BAD_NEED, FAULT_NONFINITE, FAULT_BAD_CURSOR, FAULT_SLOW = 4, 5, 6, 7, 8


class RefConfig(C.Structure):
    _fields_ = [("mode", C.c_int32), ("notify", C.c_int32), ("device_delay_us", C.c_uint32),
                ("fault_kind", C.c_uint32), ("fault_message", C.c_int64)]


class RefCounters(C.Structure):
    _fields_ = [(n, C.c_int64) for n in ("contexts", "models", "streams", "slots", "cache_entries", "cache_bytes",
                                         "tokens", "device_threads")] + \
               [(n, C.c_uint64) for n in ("messages", "bytes_in", "bytes_out", "aborts", "cache_hits",
                                          "supplied_experts", "slot_high_water", "faults_fired", "cache_evictions",
                                          "observed", "observe_ns", "observe_max_ns")]


LIVE = ("contexts", "models", "streams", "slots", "cache_entries", "cache_bytes", "tokens", "device_threads")


@dataclass
class Rig:
    workspace: Path
    authority: PinnedAuthority
    paths: dict
    v41: object
    config: object
    parameters: object
    native: DSV41NativeBackend
    ref: object  # reference provider CDLL (same sealed instance the provider attaches)
    created: list

    def fms(self, **kwargs):
        options = dict(warm_bytes=1 << 20, staging_bytes=1 << 16)
        options.update(kwargs)
        return SyntheticFileAssets(root=self.workspace, library=self.paths["elpis_fms_file_assets"], **options)

    def configure(self, mode=MODE_SYNC, notify=0, delay_us=0, fault_kind=FAULT_NONE, fault_message=-1):
        self.ref.elpis_dsv41_reference_provider_configure(C.byref(RefConfig(mode, notify, delay_us, fault_kind,
                                                                             fault_message)))

    def counters(self):
        out = RefCounters()
        self.ref.elpis_dsv41_reference_provider_counters(C.byref(out))
        return {name: getattr(out, name) for name, _ in RefCounters._fields_}

    def live(self):
        counters = self.counters()
        return {name: counters[name] for name in LIVE}

    def provider(self, **kwargs):
        options = dict(observe_layer_streams=True)
        options.update(kwargs)
        provider = DSV41StreamProvider(self.workspace, execution_library=self.paths["elpis_execution"],
                                   provider_library=self.paths["elpis_dsv41_stream_reference_provider"],
                                   authority=self.authority, execution_id="elpis_execution",
                                   provider_id="elpis_dsv41_stream_reference_provider", **options)
        self.created.append(provider)
        return provider

    def close(self):
        """Teardown safety net so one failing test cannot leak provider state into the next."""
        for provider in self.created:
            provider.close()
        self.created.clear()
        self.configure()

    def native_target(self, fms, name, **kwargs):
        target, _ = make_fixture(fms, self.workspace / name, self.v41, config=self.config,
                                 parameters=self.parameters, native_backend=self.native, **kwargs)
        return target

    def numpy_target(self, fms, name, **kwargs):
        target, _ = make_fixture(fms, self.workspace / name, self.v41, config=self.config,
                                 parameters=self.parameters, **kwargs)
        return target

    def provider_target(self, fms, name, provider, **kwargs):
        target, _ = make_fixture(fms, self.workspace / name, self.v41, config=self.config,
                                 parameters=self.parameters, provider_stream=provider, **kwargs)
        return target

    def tokens(self, count=24):
        tokens = (self.v41.encode(TEXT) * 2)[:count]
        assert len(tokens) == count
        return tokens


def make_rig(workspace, v41, *, max_tokens=32):
    paths, entries = {}, []
    (workspace / "native").mkdir(parents=True, exist_ok=True)
    for stem in LIBRARIES:
        source = require_native_library(stem)
        target = workspace / "native" / source.name
        if not target.exists():
            shutil.copyfile(source, target)
        paths[stem] = Path("native") / source.name
        data = target.read_bytes()
        entries.append(dict(library_id=stem, size=len(data), sha256=sha256(data).hexdigest()))
    document = json.dumps(dict(schema="elpis.inference-authority.v1", source="YTS-R0 qualification",
                               provenance="synthetic-test", assets=[], libraries=entries),
                          sort_keys=True, separators=(",", ":")).encode()
    authority = PinnedAuthority(document, expected_sha256=sha256(document).hexdigest())
    native = DSV41NativeBackend(workspace, paths["elpis_dsv41_native"], authority=authority,
                                library_id="elpis_dsv41_native")
    with RootCapability(workspace) as boundary:
        ref = load_native(boundary, paths["elpis_dsv41_stream_reference_provider"],
                          authority.libraries["elpis_dsv41_stream_reference_provider"])
    ref.elpis_dsv41_reference_provider_configure.argtypes = [C.POINTER(RefConfig)]
    ref.elpis_dsv41_reference_provider_configure.restype = None
    ref.elpis_dsv41_reference_provider_counters.argtypes = [C.POINTER(RefCounters)]
    ref.elpis_dsv41_reference_provider_counters.restype = None
    config = fixture_config(v41, max_tokens=max_tokens)
    parameters = fixtures_mod.build_parameters(v41, config)
    rig = Rig(workspace, authority, paths, v41, config, parameters, native, ref, [])
    rig.configure()
    return rig


def bits(array):
    return np.ascontiguousarray(array).view(np.uint32)


def bitwise_equal(a, b):
    return a.shape == b.shape and np.array_equal(bits(a), bits(b))


def variant_target(rig, fms, name, *, mutate_role, provider_stream=None, native_backend=None):
    """The standard fixture with exactly one expert tensor changed (cache-isolation model B).

    Mirrors fixtures.make_fixture draw for draw; only ``mutate_role`` is negated before
    it is written, so every other tensor is bit-identical to the standard fixture.
    """
    from elpis.inference.contracts import Bank
    from research.dsv41_tower.config import tensor_shapes
    from research.dsv41_tower.parameters import FileTensor, ParameterManifest, TensorBinding
    from research.dsv41_tower.target import DSV41Target
    from elpis.inference.rows import RowEngine, RowTable, row_representation
    from elpis.inference.target import Tensor
    from elpis.substrate.digests import raw_digest
    from elpis.substrate.file_assets import bounded_path, inspect_asset

    config, parameters, tokenizer = rig.config, rig.parameters, rig.v41
    directory = bounded_path(fms.root, rig.workspace / name)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1041)
    resident, expert_specs, chunks, offset = [], [], [], 0
    for role, shape in sorted(tensor_shapes(config).items()):
        if role.endswith(("norm", "q_weight", "k_weight")):
            array = (1 + rng.normal(0, .04, shape)).astype("<f4")
        elif role.endswith("_scale"):
            array = np.array([.08, .09, .07], dtype="<f4")
        else:
            array = rng.normal(0, .12, shape).astype("<f4")
        if role == mutate_role:
            assert ".ffn.experts." in role
            array = -array
        tensor = Tensor(shape, array.tobytes())
        if ".ffn.experts." in role:
            expert_specs.append((role, tensor, offset))
            chunks.append(tensor.data)
            offset += len(tensor.data)
        else:
            resident.append(TensorBinding(config.model, role, shape, tensor))
    path = directory / "fixture-experts.dat"
    path.write_bytes(b"".join(chunks))
    artifact = inspect_asset(fms.root, path, 4096)
    asset = fms.register(path, artifact, expected_manifest=artifact.digest)
    bindings = resident + [TensorBinding(config.model, role, t.shape, FileTensor(asset, off, raw_digest(t.data)))
                           for role, t, off in expert_specs]
    rows = {}
    for index, layer in enumerate(config.engram_layers):
        n = parameters.table_rows[index]
        codes = rng.integers(0, 96, (n, config.engram_dim), dtype=np.uint8)
        codes |= rng.integers(0, 2, codes.shape, dtype=np.uint8) * 128
        scales = np.full((n, config.engram_dim // 32), 119, dtype=np.uint8)
        path = directory / f"fixture-engram-{layer}.dat"
        path.write_bytes(np.concatenate((codes, scales), axis=1).tobytes())
        artifact = inspect_asset(fms.root, path, 4096)
        row_asset = fms.register(path, artifact, expected_manifest=artifact.digest)
        codec = "DS4_E4M3_E8M0_BF16"
        bank = Bank(f"fixture-engram-{layer}", config.model, config.tokenizer, parameters.schema,
                    parameters.digest, layer, n, config.engram_dim, artifact.content,
                    row_representation(row_asset, 0, n, config.engram_dim, codec))
        rows[layer] = RowEngine(fms, RowTable(bank, row_asset, 0, codec), expected_bank=bank.digest,
                                workers=1, max_rows=config.hash_columns)
    manifest = ParameterManifest(config.model, config.tokenizer, config.digest, parameters.digest,
                                 tuple((i, rows[i].table.bank.digest) for i in config.engram_layers), tuple(bindings))
    return DSV41Target(config, manifest, tokenizer, parameters, rows, fms, expected_manifest=manifest.digest,
                       resident_budget=64 << 20, staging_budget=1 << 16, state_budget=16 << 20,
                       native_backend=native_backend, provider_stream=provider_stream)


def flip_expert_bytes(path, offset, length):
    """Corrupt an admitted backing file in place (FMS must refuse it on the next page load)."""
    data = bytearray(path.read_bytes())
    for i in range(offset, offset + length):
        data[i] ^= 0x01
    path.write_bytes(bytes(data))
