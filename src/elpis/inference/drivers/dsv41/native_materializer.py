"""DSV4.1 Native Materializer R1 control plane (cold admission only).

Builds the sealed production Elpis host service behind the unchanged clock
table ``elpis_dsv41_materializer_v1``. Python's role ends at ``seal``:

* authority: the admitted ``FMSFileAssets`` lends each already-authorized,
  already-open descriptor with its pinned page map and identity stamp
  (``transfer_asset``); the native service duplicates it and never resolves a
  path. Row banks and expert tensor bindings come from the admitted
  ``RowEngine``/``TensorStore`` records.
* runtime: page loads, page verification, FMS residency/leases, the
  FP8/E8M0/BF16 row decoder and canonical ``w1 || w3 || w2`` staging are
  native. No Python callable is installed anywhere in the table.

The accelerator provider receives neither this object nor its descriptors,
page maps or FMS authority; the clock passes it decoded rows and expert bytes.
"""
from __future__ import annotations

import ctypes as C

from elpis.substrate.file_assets import FMSFileAssets
from ...contracts import Code, ContractError, require
from ...rows import RowEngine
from .native_clock import CODES, NativeMaterializer, _load

U32, U64, I32, I64, VP = C.c_uint32, C.c_uint64, C.c_int32, C.c_int64, C.c_void_p
CODECS = {"F32_LE": 1, "DS4_E4M3_E8M0_BF16": 2}


class FileServiceConfig(C.Structure):
    _fields_ = [("abi_version", U32), ("reserved", U32), ("warm_bytes", U64), ("staging_bytes", U64),
                ("map_bytes", U64), ("max_pages", U32), ("max_assets", U32), ("max_ranges", U32),
                ("absent_policy", U32)]


class Stamp(C.Structure):
    _fields_ = [("dev", U64), ("ino", U64), ("size", U64), ("mtime_ns", I64), ("ctime_ns", I64)]


class FileAsset(C.Structure):
    _fields_ = [("abi_version", U32), ("page_size", U32), ("fd", I32), ("reserved", U32), ("size", U64),
                ("page_count", U64), ("stamp", Stamp), ("page_digests", VP)]


class Config(C.Structure):
    _fields_ = [("abi_version", U32), ("reserved", U32), ("file", FileServiceConfig), ("layers", U32),
                ("expert_count", U32), ("image_bytes", U64), ("staging_bytes", U64), ("max_banks", U32),
                ("max_rows", U32), ("max_dimension", U32), ("reserved2", U32)]


class BankV1(C.Structure):
    _fields_ = [("layer", U32), ("dimension", U32), ("codec", U32), ("asset", U32), ("rows", U64),
                ("offset", U64), ("bank", C.c_uint8 * 32), ("max_rows", U32), ("reserved", U32),
                ("max_output_bytes", U64)]


class ExpertV1(C.Structure):
    _fields_ = [("layer", U32), ("expert", U32), ("assets", U32 * 3), ("reserved", U32),
                ("offsets", U64 * 3), ("sizes", U64 * 3), ("digests", C.c_uint8 * 96)]


FILE_STATS = ("warm_budget", "staging_budget", "max_pages", "map_bytes", "assets", "resident_bytes",
              "resident_high_water", "resident_pages", "pinned_bytes", "hits", "misses", "reads", "pread_bytes",
              "semantic_bytes", "evictions", "interrupted_reads", "lease_acquires", "lease_releases",
              "range_acquires", "range_releases", "live_ranges", "forced_releases", "integrity_failures",
              "io_failures", "staging_high_water", "read_ns", "integrity_ns", "staging_ns")
STATS = ("row_calls", "row_requests", "unique_rows", "row_bytes", "row_ns", "expert_calls", "expert_bytes",
         "expert_chunks", "expert_ns", "span_acquires", "span_releases", "live_spans", "stale_releases",
         "forced_releases", "quiesces", "refusals", "staging_budget", "staging_high_water", "samples")


class Stats(C.Structure):
    _fields_ = [("file", type("FileStats", (C.Structure,), {"_fields_": [(n, U64) for n in FILE_STATS]}))] + [
        (n, U64) for n in STATS]


def _check(rc):
    require(rc == 0, CODES.get(rc, Code.INVALID), "native materializer status " + str(rc))


class FileMaterializer:
    """Cold admission of the production file-backed materializer for one target.

    ``files`` must be the FMSFileAssets that admitted every row table and
    file-backed expert tensor of ``target``. Budgets default to that provider's
    WARM/page/staging budgets and the TensorStore expert staging budget, so the
    native service is exactly as bounded as the Python oracle. Close is
    explicit (never garbage-collection driven); the owner keeps this object
    alive for as long as any clock may call it.
    """
    SYMBOLS = {
        "abi_version": ([], U32), "create": ([C.POINTER(Config), C.POINTER(U64)], C.c_int),
        "admit_asset": ([U64, C.POINTER(FileAsset), C.POINTER(U32)], C.c_int),
        "add_bank": ([U64, C.POINTER(BankV1)], C.c_int), "add_expert": ([U64, C.POINTER(ExpertV1)], C.c_int),
        "seal": ([U64], C.c_int), "stats": ([U64, C.POINTER(Stats)], C.c_int),
        "pages": ([U64, C.POINTER(U32), C.POINTER(U64), C.c_size_t, C.POINTER(C.c_size_t)], C.c_int),
        "samples": ([U64, C.POINTER(U64), C.POINTER(U64), C.POINTER(U32), C.c_size_t, C.POINTER(C.c_size_t)], C.c_int),
        "evict": ([U64], C.c_int), "quiesce": ([U64], C.c_int), "destroy": ([U64], C.c_int)}

    def __init__(self, target, files, root, library, *, authority, library_id, warm_bytes=None,
                 max_pages=None, page_staging_bytes=None, max_ranges=4, map_bytes=64 << 20):
        c, store = target.config, target.store
        require(isinstance(files, FMSFileAssets) and store.provider is files, Code.IDENTITY,
                "materializer file assets must be the target's admitting provider")
        require(all(type(e) is RowEngine and e.provider is files for e in target.rows.values()),
                Code.IDENTITY, "Engram tables admitted by the same file assets")
        self.target, self.files = target, files
        self._lib = _load(root, library, authority, library_id)
        for name, (args, result) in self.SYMBOLS.items():
            fn = getattr(self._lib, "elpis_dsv41_materializer_" + name)
            fn.argtypes, fn.restype = args, result
        require(self._lib.elpis_dsv41_materializer_abi_version() == 1, Code.UNSUPPORTED, "materializer ABI")
        image_bytes = 3 * c.expert_dim * c.dimension * 4
        engines = [target.rows[layer] for layer in c.engram_layers]
        x = Config(abi_version=1, layers=c.layers, expert_count=c.expert_count, image_bytes=image_bytes,
                   staging_bytes=store.staging_budget, max_banks=max(1, len(engines)),
                   max_rows=max([e.max_rows for e in engines] + [1]),
                   max_dimension=max([e.table.bank.dimension for e in engines] + [1]))
        x.file = FileServiceConfig(1, 0, warm_bytes or files.warm_budget,
                                   page_staging_bytes or files.staging_budget, map_bytes,
                                   max_pages or files.max_pages, 1024, max_ranges,
                                   int(files.hot_absent_policy == "REJECT"))
        self.config = x
        self._handle, self.closed = U64(), False
        _check(self._lib.elpis_dsv41_materializer_create(C.byref(x), C.byref(self._handle)))
        try:
            self.assets = {}
            for engine in engines:
                self._asset(engine.table.asset)
            experts = []
            for layer in range(c.layers):
                for e in range(c.expert_count + 1):
                    name = "shared" if e == c.expert_count else str(e)
                    roles = tuple(f"layers.{layer}.ffn.experts.{name}.{w}" for w in ("w1", "w3", "w2"))
                    image = store.expert_image(roles)
                    if image.resident:
                        continue  # resident experts are never materialized
                    require(image.image_bytes == image_bytes, Code.ENCODING, "canonical expert image geometry")
                    sources = [store.bindings[r].source for r in roles]
                    experts.append(ExpertV1(layer, e, (U32 * 3)(*(self._asset(s.asset) for s in sources)), 0,
                                            (U64 * 3)(*(s.offset for s in sources)), (U64 * 3)(*image.sizes),
                                            (C.c_uint8 * 96)(*b"".join(image.digests))))
            for layer, engine in zip(c.engram_layers, engines):
                t = engine.table
                require(t.codec in CODECS, Code.UNSUPPORTED, "row codec")
                b = BankV1(layer, t.bank.dimension, CODECS[t.codec], self.assets[t.asset], t.bank.rows, t.offset,
                           (C.c_uint8 * 32)(*bytes.fromhex(t.bank.digest)), engine.max_rows, 0, engine.max_output_bytes)
                _check(self._lib.elpis_dsv41_materializer_add_bank(self._handle, C.byref(b)))
            for e in experts:
                _check(self._lib.elpis_dsv41_materializer_add_expert(self._handle, C.byref(e)))
            _check(self._lib.elpis_dsv41_materializer_seal(self._handle))
            self.materializer = NativeMaterializer(root, library, authority=authority, library_id=library_id,
                                                   bind_symbol="elpis_dsv41_materializer_bind",
                                                   context=self._handle.value, owner=self)
        except BaseException:
            self._lib.elpis_dsv41_materializer_destroy(self._handle)
            self.closed = True
            raise

    def _asset(self, asset):
        if asset not in self.assets:
            def receive(fd, manifest, stamp):
                raw = b"".join(bytes.fromhex(p) for p in manifest.pages)
                digests = C.create_string_buffer(raw, len(raw))
                record = FileAsset(1, manifest.page_size, fd, 0, manifest.size, len(manifest.pages),
                                   Stamp(*stamp), C.cast(digests, VP))
                index = U32()
                _check(self._lib.elpis_dsv41_materializer_admit_asset(self._handle, C.byref(record), C.byref(index)))
                return index.value
            self.assets[asset] = self.files.transfer_asset(asset, receive)
        return self.assets[asset]

    def _live(self):
        require(not self.closed, Code.CLOSED, "native materializer closed")

    def stats(self):
        self._live()
        s = Stats()
        _check(self._lib.elpis_dsv41_materializer_stats(self._handle, C.byref(s)))
        out = {n: getattr(s, n) for n in STATS}
        out.update({"file_" + n: getattr(s.file, n) for n in FILE_STATS})
        out["buffered_page_cache_accounted"] = False
        return out

    def pages(self):
        """Resident pages oldest-first as (asset id, page index)."""
        self._live()
        count = C.c_size_t()
        _check(self._lib.elpis_dsv41_materializer_pages(self._handle, None, None, 0, C.byref(count)))
        n = count.value
        assets, pages = (U32 * max(1, n))(), (U64 * max(1, n))()
        _check(self._lib.elpis_dsv41_materializer_pages(self._handle, assets, pages, n, C.byref(count)))
        names = {v: k for k, v in self.assets.items()}
        return [(names[assets[i]], pages[i]) for i in range(min(n, count.value))]

    def samples(self):
        """Recent acquisitions: (kind, latency_ns, resident_bytes); kind 1 rows, 2 expert."""
        self._live()
        lat, res, kinds, count = (U64 * 4096)(), (U64 * 4096)(), (U32 * 4096)(), C.c_size_t()
        _check(self._lib.elpis_dsv41_materializer_samples(self._handle, lat, res, kinds, 4096, C.byref(count)))
        return [(kinds[i], lat[i], res[i]) for i in range(count.value)]

    def evict(self):
        self._live()
        _check(self._lib.elpis_dsv41_materializer_evict(self._handle))

    def quiesce(self):
        self._live()
        _check(self._lib.elpis_dsv41_materializer_quiesce(self._handle))

    def close(self):
        """Destroy the native service. BUSY while a call or borrowed span is outstanding."""
        if self.closed:
            return
        rc = self._lib.elpis_dsv41_materializer_destroy(self._handle)
        if rc != 0:
            raise ContractError(CODES.get(rc, Code.INVALID), "native materializer destroy")
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
