"""Differential authority for the native ECS_C receipt codec.

The Python ReceiptRecord byte form remains the reference in this stage.
The native implementation must reproduce it exactly, including ensure_ascii.
"""

from __future__ import annotations

import ctypes as C

import pytest

from elpis.ECS_C import canonical
from elpis.runtime.history import ReceiptRecord
from tests.conftest import require_native_library


class Binding(C.Structure):
    _fields_ = [
        ("name", C.c_char_p),
        ("name_len", C.c_size_t),
        ("value", C.c_char_p),
        ("value_len", C.c_size_t),
    ]


@pytest.fixture(scope="module")
def lib():
    dll = C.CDLL(str(require_native_library("elpis_ecsc_history")))

    dll.elpis_ecsc_history_abi_version.restype = C.c_uint32

    common = [
        C.c_char_p, C.c_size_t,
        C.c_char_p, C.c_size_t,
        C.c_char_p, C.c_size_t,
        C.POINTER(Binding), C.c_size_t,
    ]

    dll.elpis_ecsc_receipt_payload_size.argtypes = [
        *common,
        C.POINTER(C.c_size_t),
    ]
    dll.elpis_ecsc_receipt_payload_size.restype = C.c_int

    dll.elpis_ecsc_receipt_payload_write.argtypes = [
        *common,
        C.POINTER(C.c_uint8), C.c_size_t,
        C.POINTER(C.c_size_t),
    ]
    dll.elpis_ecsc_receipt_payload_write.restype = C.c_int

    dll.elpis_ecsc_digest_bytes.argtypes = [
        C.c_void_p, C.c_size_t, C.POINTER(C.c_char)
    ]
    dll.elpis_ecsc_digest_bytes.restype = C.c_int

    return dll


def _native_payload(lib, record: ReceiptRecord) -> bytes:
    subsystem = record.subsystem.encode("ascii")
    kind = record.kind.encode("ascii")
    digest = record.digest.encode("ascii")

    keepalive: list[bytes] = []
    raw_bindings = []

    for name, value in record.bindings:
        nb = name.encode("ascii")
        vb = value.encode("utf-8")
        keepalive.extend((nb, vb))
        raw_bindings.append(Binding(nb, len(nb), vb, len(vb)))

    if raw_bindings:
        array_type = Binding * len(raw_bindings)
        binding_array = array_type(*raw_bindings)
        binding_ptr = C.cast(binding_array, C.POINTER(Binding))
    else:
        binding_array = None
        binding_ptr = C.POINTER(Binding)()

    size = C.c_size_t()

    rc = lib.elpis_ecsc_receipt_payload_size(
        subsystem, len(subsystem),
        kind, len(kind),
        digest, len(digest),
        binding_ptr, len(raw_bindings),
        C.byref(size),
    )
    assert rc == 0

    out = (C.c_uint8 * size.value)()
    written = C.c_size_t()

    rc = lib.elpis_ecsc_receipt_payload_write(
        subsystem, len(subsystem),
        kind, len(kind),
        digest, len(digest),
        binding_ptr, len(raw_bindings),
        out, size.value,
        C.byref(written),
    )
    assert rc == 0
    assert written.value == size.value

    return bytes(out)


@pytest.mark.parametrize(
    "record",
    [
        ReceiptRecord.of(
            "pipeline",
            "ingress.proposal",
            "0" * 64,
        ),
        ReceiptRecord.of(
            "inference",
            "codec.fixture",
            "1" * 64,
            alpha='quote"slash\\',
            zeta="é😀",
        ),
        ReceiptRecord.of(
            "structure",
            "context.admission",
            "abcdef" * 10 + "abcd",
            corpus="2" * 64,
            objects="16",
            omitted="0",
            tokens="4096",
        ),
        ReceiptRecord.of(
            "evolution",
            "evolution.path-transition",
            "f" * 64,
            assertion="3" * 64,
            history_projection="4" * 64,
            outcome="OUTCOME_A",
        ),
    ],
)
def test_native_receipt_payload_is_exact_python_authority(lib, record):
    assert lib.elpis_ecsc_history_abi_version() == 1
    assert _native_payload(lib, record) == record.payload()


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"abc",
        bytes(range(256)),
        b"\x00\x00\xff\x10receipt",
    ],
)
def test_native_digest_bytes_is_exact_python_authority(lib, raw):
    if raw:
        src = (C.c_uint8 * len(raw)).from_buffer_copy(raw)
        ptr = C.cast(src, C.c_void_p)
    else:
        src = None
        ptr = C.c_void_p()

    out = (C.c_char * 65)()

    assert lib.elpis_ecsc_digest_bytes(ptr, len(raw), out) == 0
    native = bytes(out).split(b"\x00", 1)[0].decode("ascii")

    assert native == canonical.digest_bytes(raw)


def test_native_refuses_unsorted_binding_names(lib):
    digest = b"a" * 64
    b1n, b1v = b"z", b"1"
    b2n, b2v = b"a", b"2"

    arr = (Binding * 2)(
        Binding(b1n, 1, b1v, 1),
        Binding(b2n, 1, b2v, 1),
    )

    size = C.c_size_t(999)

    rc = lib.elpis_ecsc_receipt_payload_size(
        b"ecs_g", 5,
        b"cognition.turn", 14,
        digest, 64,
        arr, 2,
        C.byref(size),
    )

    assert rc == -1
    assert size.value == 0
