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
                require(provider is not None, Code.MISSING, "tensor file asset")
                asset = provider.manifest(binding.source.asset)
                require(binding.source.offset + binding.size <= asset.size, Code.ENCODING, "tensor asset range")
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
        # One bounded, reusable staging scratch sized to the largest file-backed expert (its
        # w1/w3/w2 together). Experts are copied into it once per use; nothing else is
        # materialized and no expert becomes resident.
        groups = {}
        for name, binding in bindings.items():
            if ".ffn.experts." in name and type(binding.source) is FileTensor:
                key = name.rsplit(".", 1)[0]
                groups[key] = groups.get(key, 0) + binding.size
        scratch = max(groups.values(), default=0)
        require(scratch <= staging_budget, Code.LIMIT, "one-expert staging budget")
        self._scratch = np.empty(scratch, dtype=np.uint8)
        self._scratch_busy = False
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

    def _copy(self, asset, offset, size, out):
        """Copy `size` bytes of `asset` at `offset` into writable `out` via FMS range leases.

        A lease spans at most half the provider's native budget (headroom for other
        leases), so a contiguous run normally costs one lease; FMS still verifies every
        page digest on load.
        """
        page = self.provider.manifest(asset).page_size
        pages = max(1, self.provider.warm_budget // (2 * page))
        view = memoryview(out)
        done = 0
        while done < size:
            count = min(size - done, (offset // page + pages) * page - offset)
            with self.provider.acquire(asset, offset, count) as lease:
                lease.readinto(view[done:done + count])
            done += count
            offset += count

    def _fill(self, binding, out):
        self._copy(binding.source.asset, binding.source.offset, binding.size, out)

    def _read(self, binding):
        """Immutable bytes of one tensor (admission verification, oracle harness)."""
        if type(binding.source) is Tensor:
            return binding.source.data
        out = bytearray(binding.size)
        self._fill(binding, out)
        return bytes(out)

    @contextmanager
    def expert(self, roles):
        """Read-only views of one expert's tensors, valid only inside the with-block.

        File-backed tensors are copied once into the store's reusable scratch; the views
        alias it and must not be retained after exit.
        """
        with self._lock:
            require(not self._scratch_busy, Code.BUSY, "expert staging in use")
            reservation = sum(self.bindings[r].size for r in roles if r not in self.dense)
            require(self.staged_bytes + reservation <= self.staging_budget, Code.LIMIT, "expert staging in use")
            self._scratch_busy = True
            self.staged_bytes += reservation
            self.high_water = max(self.high_water, self.staged_bytes)
            start = perf_counter_ns()
            try:
                # Byte-contiguous tensors of one asset (e.g. an expert's w1/w2/w3) are copied
                # with one range lease; each role is then a read-only view at its offset.
                files = sorted((r for r in roles if r not in self.dense),
                               key=lambda r: (self.bindings[r].source.asset, self.bindings[r].source.offset))
                views, at, run = {}, 0, []
                def flush():
                    nonlocal at
                    first, last = self.bindings[run[0]].source, self.bindings[run[-1]]
                    size = last.source.offset + last.size - first.offset
                    region = self._scratch[at:at + size]
                    self._copy(first.asset, first.offset, size, region)
                    for r in run:
                        b = self.bindings[r]
                        rel = b.source.offset - first.offset
                        view = region[rel:rel + b.size].view("<f4").reshape(b.shape)
                        view.flags.writeable = False
                        views[r] = view
                    at += size
                for r in files:
                    b = self.bindings[r]
                    if run:
                        p = self.bindings[run[-1]]
                        if p.source.asset == b.source.asset and p.source.offset + p.size == b.source.offset:
                            run.append(r)
                            continue
                        flush()
                    run = [r]
                if run:
                    flush()
                arrays = tuple(self.dense[r] if r in self.dense else views[r] for r in roles)
                self.last_materialize_ns += perf_counter_ns() - start
                self.last_bytes += reservation
                yield arrays
            finally:
                self.staged_bytes -= reservation
                self._scratch_busy = False


class TowerAdmission:
    __slots__ = ("target",)

    def __init__(self, target):
        self.target = target
