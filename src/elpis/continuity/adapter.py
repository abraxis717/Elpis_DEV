"""Thin Python adapter over the Rust continuity authority (``include/elpis/continuity.h``).

The authority — record format, checksum and digest domains, slot selection, locking, publication and
every state transition — lives in ``libelpis_continuity.so`` (native/continuity). This module only
marshals values across the C ABI and raises the stable code the library returns. It holds no durable
state, decides no transition and computes no digest. Malformed Python inputs are passed as NULL so that
the library refuses them with its own code, in its own order.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass, field
import os
from pathlib import Path

__all__ = (
    "ContinuityError", "ContinuityLibrary", "ContinuityProcessDeath", "ContinuitySnapshot", "ContinuityStore",
    "EvolutionAuthority",
)

_ABI_VERSION = 1
_TESTING_PROCESS_DEATH = 255
_U64 = (1 << 64) - 1


class ContinuityError(RuntimeError):
    """Fail-closed continuity refusal with the library's stable code."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


class ContinuityProcessDeath(BaseException):
    """Testing library only: the library simulated process death at a named step."""


@dataclass(frozen=True)
class EvolutionAuthority:
    """Evolution authority value: idle (``pending_assertion`` None) or reserved for one assertion.

    ``digest`` is the library's evolution-authority v2 digest when the value came from the library.
    """

    revision: int
    head: str
    pending_assertion: str | None = None
    digest: str | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class ContinuitySnapshot:
    """One complete authority as returned by the library. ``digest`` is the record checksum."""

    generation: int
    k1_state_digest: bytes | None
    evolution: EvolutionAuthority
    digest: str | None = field(default=None, compare=False, repr=False)

    @property
    def anchored(self) -> bool:
        return self.k1_state_digest is not None


class _Evolution(C.Structure):
    _fields_ = [("revision", C.c_uint64), ("head", C.c_uint8 * 32), ("pending", C.c_uint8),
                ("reserved", C.c_uint8 * 7), ("assertion", C.c_uint8 * 32)]


class _Snapshot(C.Structure):
    _fields_ = [("generation", C.c_uint64), ("anchored", C.c_uint8), ("reserved", C.c_uint8 * 7),
                ("k1_state_digest", C.c_uint8 * 32), ("evolution", _Evolution),
                ("evolution_digest", C.c_uint8 * 32), ("record_digest", C.c_uint8 * 32)]


class _Counters(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in ("opens", "preads", "pread_bytes", "pwrites", "pwrite_bytes",
                                                "data_syncs", "dir_syncs", "renames", "unlinks")]


_DIGEST = C.c_uint8 * 32


def _digest_arg(value):
    """32 raw bytes, or a 64-char lowercase hex string, as a C digest; anything else is NULL."""
    if type(value) is bytes and len(value) == 32:
        return _DIGEST.from_buffer_copy(value)
    if type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value):
        return _DIGEST.from_buffer_copy(bytes.fromhex(value))
    return None


def _evolution_arg(value):
    if type(value) is not EvolutionAuthority or type(value.revision) is not int or not 0 <= value.revision <= _U64:
        return None
    head = _digest_arg(value.head) if type(value.head) is str else None
    pending = value.pending_assertion
    assertion = _digest_arg(pending) if type(pending) is str else (_DIGEST() if pending is None else None)
    if head is None or assertion is None:
        return None
    return _Evolution(value.revision, head, int(pending is not None), (C.c_uint8 * 7)(), assertion)


def _snapshot_arg(value):
    evolution = _evolution_arg(getattr(value, "evolution", None))
    k1 = getattr(value, "k1_state_digest", None)
    k1_arg = _DIGEST() if k1 is None else _digest_arg(k1) if type(k1) is bytes else None
    generation = getattr(value, "generation", None)
    if (type(value) is not ContinuitySnapshot or evolution is None or k1_arg is None
            or type(generation) is not int or not 0 <= generation <= _U64):
        return None
    return _Snapshot(generation, int(k1 is not None), (C.c_uint8 * 7)(), k1_arg, evolution)


def _from_c(s: _Snapshot) -> ContinuitySnapshot:
    e = s.evolution
    authority = EvolutionAuthority(e.revision, bytes(e.head).hex(),
                                   bytes(e.assertion).hex() if e.pending else None,
                                   bytes(s.evolution_digest).hex())
    return ContinuitySnapshot(s.generation, bytes(s.k1_state_digest) if s.anchored else None, authority,
                              bytes(s.record_digest).hex())


class ContinuityLibrary:
    """The loaded continuity library (explicit path, like every Elpis native library)."""

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.is_absolute() or not path.is_file():
            raise ContinuityError("CONTINUITY_PATH", f"continuity library not found: {path}")
        lib = C.CDLL(str(path))
        lib.elpis_continuity_abi_version.restype = C.c_uint32
        if lib.elpis_continuity_abi_version() != _ABI_VERSION:
            raise ContinuityError("CONTINUITY_INVALID", "continuity ABI version mismatch")
        lib.elpis_continuity_record_size.restype = C.c_size_t
        lib.elpis_continuity_code_name.restype = C.c_char_p
        lib.elpis_continuity_code_name.argtypes = [C.c_int]
        sig = {
            "evolution_digest": [C.POINTER(_Evolution), C.c_void_p],
            "record_encode": [C.POINTER(_Snapshot), C.c_void_p],
            "record_decode": [C.c_char_p, C.c_size_t, C.POINTER(_Snapshot), C.POINTER(C.c_int)],
            "store_create": [C.c_char_p, C.c_size_t, C.POINTER(C.c_void_p)],
            "store_open": [C.c_void_p, C.POINTER(_Snapshot)],
            "store_snapshot": [C.c_void_p, C.POINTER(_Snapshot)],
            "anchor_cognition": [C.c_void_p, C.c_void_p, C.POINTER(_Snapshot)],
            "commit_cognition": [C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(_Snapshot)],
            "reserve_evolution": [C.c_void_p, C.POINTER(_Evolution), C.c_void_p, C.POINTER(_Snapshot)],
            "finalize_evolution": [C.c_void_p, C.POINTER(_Evolution), C.c_void_p, C.POINTER(_Snapshot)],
        }
        for name, args in sig.items():
            fn = getattr(lib, "elpis_continuity_" + name)
            fn.argtypes, fn.restype = args, C.c_int
        for name in ("store_close", "store_destroy"):
            getattr(lib, "elpis_continuity_" + name).restype = None
        lib.elpis_continuity_store_close.argtypes = [C.c_void_p]
        lib.elpis_continuity_store_destroy.argtypes = [C.POINTER(C.c_void_p)]
        self.testing = hasattr(lib, "elpis_continuity_testing_fault")
        if self.testing:
            lib.elpis_continuity_testing_fault.argtypes = [C.c_void_p, C.c_uint64, C.c_uint32, C.c_uint64]
            lib.elpis_continuity_testing_counters.argtypes = [C.c_void_p, C.POINTER(_Counters), C.c_int]
        self.path = path
        self.record_size = lib.elpis_continuity_record_size()
        self._lib = lib

    def check(self, rc: int) -> None:
        if rc == 0:
            return
        if rc == _TESTING_PROCESS_DEATH:
            raise ContinuityProcessDeath("simulated process death")
        name = self._lib.elpis_continuity_code_name(rc)
        raise ContinuityError(name.decode() if name else f"CONTINUITY_UNKNOWN_{rc}")

    def encode_record(self, snapshot: ContinuitySnapshot) -> bytes:
        arg = _snapshot_arg(snapshot)
        if arg is None:
            raise ContinuityError("CONTINUITY_CORRUPT", "not a well-formed continuity snapshot")
        out = (C.c_uint8 * self.record_size)()
        self.check(self._lib.elpis_continuity_record_encode(C.byref(arg), out))
        return bytes(out)

    def decode_record(self, raw: bytes) -> ContinuitySnapshot | None:
        if type(raw) is not bytes:
            raise ContinuityError("CONTINUITY_CORRUPT", "record bytes required")
        out, empty = _Snapshot(), C.c_int(0)
        self.check(self._lib.elpis_continuity_record_decode(raw, len(raw), C.byref(out), C.byref(empty)))
        return None if empty.value else _from_c(out)

    def evolution_digest(self, authority: EvolutionAuthority) -> str:
        arg = _evolution_arg(authority)
        if arg is None:
            raise ContinuityError("CONTINUITY_CORRUPT", "not a well-formed evolution authority")
        out = _DIGEST()
        self.check(self._lib.elpis_continuity_evolution_digest(C.byref(arg), out))
        return bytes(out).hex()


class ContinuityStore:
    """One continuity directory, owned through the library. Single owner; calls must not overlap."""

    def __init__(self, library: ContinuityLibrary, directory: str | Path):
        if type(library) is not ContinuityLibrary:
            raise TypeError("ContinuityStore takes a ContinuityLibrary")
        self.library = library
        self.directory = Path(directory)
        raw = os.fsencode(str(self.directory))
        handle = C.c_void_p()
        library.check(library._lib.elpis_continuity_store_create(raw, len(raw), C.byref(handle)))
        self._handle = handle

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None and handle.value:
            self.library._lib.elpis_continuity_store_destroy(C.byref(handle))

    def _call(self, name: str, *args) -> ContinuitySnapshot:
        out = _Snapshot()
        self.library.check(getattr(self.library._lib, "elpis_continuity_" + name)(self._handle, *args, C.byref(out)))
        return _from_c(out)

    # -- lifecycle -------------------------------------------------------------------------------------
    def open(self) -> "ContinuityStore":
        self._call("store_open")
        return self

    def close(self) -> None:
        self.library._lib.elpis_continuity_store_close(self._handle)

    def __enter__(self) -> "ContinuityStore":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- reads and the only transitions ------------------------------------------------------------------
    def snapshot(self) -> ContinuitySnapshot:
        return self._call("store_snapshot")

    def anchor_cognition(self, k1_state_digest: bytes) -> ContinuitySnapshot:
        return self._call("anchor_cognition", _bytes_digest(k1_state_digest))

    def commit_cognition_transition(self, before: bytes, after: bytes) -> ContinuitySnapshot:
        return self._call("commit_cognition", _bytes_digest(before), _bytes_digest(after))

    def reserve_evolution_assertion(self, expected: EvolutionAuthority, assertion_digest: str) -> ContinuitySnapshot:
        return self._call("reserve_evolution", _ref(_evolution_arg(expected)), _hex_digest(assertion_digest))

    def commit_evolution_transition(self, expected: EvolutionAuthority, receipt_digest: str) -> ContinuitySnapshot:
        return self._call("finalize_evolution", _ref(_evolution_arg(expected)), _hex_digest(receipt_digest))

    # -- testing library only ------------------------------------------------------------------------------
    def testing_fault(self, publication: int, action: int, arg: int = 0) -> None:
        self._require_testing()
        self.library.check(self.library._lib.elpis_continuity_testing_fault(self._handle, publication, action, arg))

    def testing_counters(self, reset: bool = False) -> dict[str, int]:
        self._require_testing()
        out = _Counters()
        self.library.check(self.library._lib.elpis_continuity_testing_counters(self._handle, C.byref(out),
                                                                                int(reset)))
        return {name: getattr(out, name) for name, _ in _Counters._fields_}

    def _require_testing(self) -> None:
        if not self.library.testing:
            raise ContinuityError("CONTINUITY_INVALID", "fault injection exists only in the testing library")


def _ref(value):
    return C.byref(value) if value is not None else None


def _bytes_digest(value):
    return _digest_arg(value) if type(value) is bytes else None


def _hex_digest(value):
    return _digest_arg(value) if type(value) is str else None
