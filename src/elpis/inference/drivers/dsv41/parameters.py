"""Serialization-neutral pinned tensor contract. File bytes come only through FMS.

Dense tensors are retained read-only; routed/shared experts can stay file-backed.
Content verification and identity construction happen during intake, never in
materialize(). No file discovery, registration or parameter generation here.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from threading import RLock
from time import perf_counter_ns
import numpy as np

from elpis.identity import content_digest
from elpis.substrate.digests import raw_digest
from ...contracts import Code, digest_value, integer, require
from ...target import Tensor
from .config import TowerConfig, tensor_shapes


@dataclass(frozen=True)
class FileTensor:
    asset: str
    offset: int
    content: str

    def __post_init__(self):
        digest_value(self.asset)
        digest_value(self.content)
        integer(self.offset)


@dataclass(frozen=True)
class TensorBinding:
    model: str
    role: str
    shape: tuple[int, ...]
    source: Tensor | FileTensor
    dtype: str = "F32_LE"
    layout: str = "ROW_MAJOR_OUT_IN"

    def __post_init__(self):
        require(type(self.model) is str and bool(self.model) and type(self.role) is str and bool(self.role))
        require(type(self.shape) is tuple and bool(self.shape), detail="tensor shape")
        for dim in self.shape:
            integer(dim, 1)
        require((self.dtype, self.layout) == ("F32_LE", "ROW_MAJOR_OUT_IN"),
                Code.UNSUPPORTED, "tower tensor representation")
        require(type(self.source) in (Tensor, FileTensor), detail="explicit tensor source")
        if type(self.source) is Tensor:
            require(self.source.shape == self.shape, detail="resident tensor geometry")

    @property
    def size(self):
        count = 4
        for dim in self.shape:
            count *= dim
        return count

    @property
    def digest(self):
        source = self.source.digest if type(self.source) is Tensor else self.source
        return content_digest("elpis.inference.dsv41.tensor-binding.v1", dict(
            model=self.model, role=self.role, shape=self.shape, source=source, dtype=self.dtype, layout=self.layout))


@dataclass(frozen=True)
class ParameterManifest:
    model: str
    tokenizer: str
    config: str
    address: str
    row_banks: tuple[tuple[int, str], ...]
    tensors: tuple[TensorBinding, ...]

    def __post_init__(self):
        require(type(self.model) is str and bool(self.model))
        for value in (self.tokenizer, self.config, self.address):
            digest_value(value)
        require(type(self.tensors) is tuple and all(type(t) is TensorBinding for t in self.tensors))
        require(len({t.role for t in self.tensors}) == len(self.tensors), detail="duplicate tensor role")
        require(all(t.model == self.model for t in self.tensors), Code.IDENTITY, "tensor model")
        require(type(self.row_banks) is tuple, detail="row bank bindings")
        for entry in self.row_banks:
            require(type(entry) is tuple and len(entry) == 2, detail="row bank binding")
            integer(entry[0])
            digest_value(entry[1])
        require(len({i for i, _ in self.row_banks}) == len(self.row_banks), detail="duplicate row bank")

    @property
    def digest(self):
        return content_digest("elpis.inference.dsv41.parameter-manifest.v1", dict(
            model=self.model, tokenizer=self.tokenizer, config=self.config, address=self.address,
            rows=self.row_banks, tensors=tuple((t.role, t.digest) for t in sorted(self.tensors, key=lambda t: t.role))))


class TensorStore:
    def __init__(self, config: TowerConfig, manifest: ParameterManifest, provider, *, expected_manifest,
                 resident_budget, staging_budget):
        integer(resident_budget, 1)
        integer(staging_budget, 1)
        require(type(manifest) is ParameterManifest and manifest.digest == expected_manifest,
                Code.IDENTITY, "tower parameter manifest pin")
        require((manifest.model, manifest.tokenizer, manifest.config) ==
                (config.model, config.tokenizer, config.digest), Code.IDENTITY, "tower manifest binding")
        expected = tensor_shapes(config)
        bindings = {b.role: b for b in manifest.tensors}
        require(set(bindings) == set(expected), Code.MISSING, "complete exact tower tensor roles")
        for name, binding in bindings.items():
            require(binding.shape == expected[name], Code.ENCODING, "tensor geometry: " + name)
            if type(binding.source) is FileTensor:
                source = binding.source
                require(provider is not None and source.asset in provider._assets, Code.MISSING, "tensor file asset")
                require(source.offset + binding.size <= provider._assets[source.asset][1].size,
                        Code.ENCODING, "tensor asset range")
        self.provider = provider
        self.bindings = MappingProxyType(bindings)
        self.manifest = manifest
        self.staging_budget = staging_budget
        self._lock = RLock()  # per-store bounded staging, no thread pool or global lock
        self.high_water = 0
        self.staged_bytes = 0
        self.last_materialize_ns = 0
        self.last_bytes = 0
        self.resident_bytes = sum(b.size for n, b in bindings.items()
                                  if ".ffn.experts." not in n or type(b.source) is Tensor)
        require(self.resident_bytes <= resident_budget, Code.LIMIT, "dense/resident parameter budget")
        expert_size = 3 * config.dimension * config.expert_dim * 4
        # Reserve for raw copies plus decoded arrays; one expert at a time.
        require(expert_size * 3 <= staging_budget, Code.LIMIT, "one-expert staging budget")
        dense = {}
        for name, binding in sorted(bindings.items()):
            # Verify every file range once, without retaining every expert.
            raw = self._read(binding)
            if type(binding.source) is FileTensor:
                require(raw_digest(raw) == binding.source.content, Code.INTEGRITY, "tensor content: " + name)
            array = np.frombuffer(raw, dtype="<f4").reshape(binding.shape)
            require(np.all(np.isfinite(array)), Code.ENCODING, "nonfinite tensor: " + name)
            if ".ffn.experts." not in name or type(binding.source) is Tensor:
                dense[name] = array
        self.dense = MappingProxyType(dense)

    def _read(self, binding):
        if type(binding.source) is Tensor:
            return binding.source.data
        source = binding.source
        page = self.provider._assets[source.asset][1].page_size
        data = bytearray()
        offset = source.offset
        while len(data) < binding.size:
            count = min(binding.size - len(data), page - offset % page)
            with self.provider.acquire(source.asset, offset, count) as lease:
                data.extend(lease.read())
            offset += count
        return bytes(data)

    @contextmanager
    def expert(self, roles):
        with self._lock:
            reservation = sum(self.bindings[r].size for r in roles) * 3
            require(self.staged_bytes + reservation <= self.staging_budget, Code.LIMIT, "expert staging in use")
            self.staged_bytes += reservation
            self.high_water = max(self.high_water, self.staged_bytes)
            start = perf_counter_ns()
            try:
                arrays = tuple(self.dense[r] if r in self.dense else
                               np.frombuffer(self._read(self.bindings[r]), dtype="<f4").reshape(self.bindings[r].shape)
                               for r in roles)
                self.last_materialize_ns += perf_counter_ns() - start
                self.last_bytes += sum(self.bindings[r].size for r in roles if r not in self.dense)
                yield arrays
            finally:
                self.staged_bytes -= reservation


class TowerAdmission:
    __slots__ = ("target",)

    def __init__(self, target):
        self.target = target
