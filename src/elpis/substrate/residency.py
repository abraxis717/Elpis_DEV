"""Lifecycle binding for a generic owned mutable FMS context.

The POSIX PAL owns an explicit scratch root. This storage is independent of
admitted immutable file assets. Ownership transfers to a native consumer; the
Context handle is cleared after a successful transfer.
"""
import ctypes as C
import os


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
    """Explicit POSIX FMS context, owned until transferred or closed."""

    def __init__(self, library, *, warm_bytes, cold_bytes, max_objects, cold_root):
        if not all(type(v) is int and v > 0 for v in (warm_bytes, cold_bytes, max_objects)):
            raise ValueError("positive FMS budgets and capacity required")
        root = os.fsencode(cold_root)
        if not os.path.isabs(root) or b"\0" in root:
            raise ValueError("explicit absolute scratch root required")
        library.elpis_fms_create_posix.argtypes = [C.POINTER(Config), C.c_char_p]
        library.elpis_fms_create_posix.restype = C.c_void_p
        library.fms_destroy.argtypes = [C.c_void_p]
        library.fms_destroy.restype = None
        cfg = Config((0, warm_bytes, cold_bytes), (warm_bytes, 1, cold_bytes),
                     0.90, 0.70, 0, 0, 0, 0, max_objects, 1, 1)
        self.library = library
        self.handle = C.c_void_p(library.elpis_fms_create_posix(C.byref(cfg), root))
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
