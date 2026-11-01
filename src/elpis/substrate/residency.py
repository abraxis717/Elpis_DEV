"""Lifecycle binding for a generic owned mutable FMS context (FMS ABI v3: volatile).

The context is RAM-only: WARM residency in process memory (the RAM-only POSIX PAL offers no HOT device), no COLD
tier, no storage domain, no scratch root and no file. An object that does not fit the WARM budget is refused
(FMS_E_LIMIT), never spilled. Ownership transfers to a native consumer; the Context handle is cleared after a
successful transfer.

Until FMS ABI v3 the POSIX PAL owned an explicit writable scratch root for COLD residency (``cold_root`` and
``cold_bytes``). That writer is retired, not configurable: there is no such parameter, and passing one is a
``TypeError``. Immutable COLD assets are read, never written, through ``elpis.substrate.file_assets``.
"""
import ctypes as C


class Config(C.Structure):
    _fields_ = [
        ("tier_budget", C.c_uint64 * 3),
        ("domain_ceiling", C.c_uint64 * 3),
        ("high_wm", C.c_float),
        ("low_wm", C.c_float),
        ("move_rate_bps", C.c_uint64),
        ("cooldown_ns", C.c_uint64),
        ("min_residency_ns", C.c_uint64),
        ("fence_timeout_ns", C.c_uint64),
        ("max_objects", C.c_uint32),
        ("hot_absent_policy", C.c_uint8),
        ("cold_absent_policy", C.c_uint8),
    ]


class Context:
    """A volatile RAM-only FMS context, owned until transferred or closed."""

    def __init__(self, library, *, warm_bytes, max_objects):
        if not all(type(v) is int and v > 0 for v in (warm_bytes, max_objects)):
            raise ValueError("a positive WARM budget and capacity are required")
        library.elpis_fms_create_ram_only.argtypes = [C.POINTER(Config)]
        library.elpis_fms_create_ram_only.restype = C.c_void_p
        library.fms_destroy.argtypes = [C.c_void_p]
        library.fms_destroy.restype = None
        # HOT, WARM, COLD budgets and RAM, DEVICE, STORAGE ceilings: no COLD budget and no storage ceiling (fms_create
        # refuses either). HOT is absent from the RAM-only PAL; both absent-tier policies are REJECT (1), so a HOT
        # request is refused rather than folded down, and a COLD request is refused whatever the policy says.
        cfg = Config((0, warm_bytes, 0), (warm_bytes, 1, 0),
                     0.90, 0.70, 0, 0, 0, 0, max_objects, 1, 1)
        self.library = library
        self.handle = C.c_void_p(library.elpis_fms_create_ram_only(C.byref(cfg)))
        if not self.handle.value:
            raise ValueError("FMS context creation refused")

    def close(self):
        if self.handle.value:
            self.library.fms_destroy(self.handle)
            self.handle.value = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
