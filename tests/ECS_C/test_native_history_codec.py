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


def _bind_phase_b(lib):
    lib.elpis_ecsc_message_id.argtypes = [
        C.c_char_p,
        C.c_char_p,
        C.c_uint64,
        C.c_void_p,
        C.c_size_t,
        C.POINTER(C.c_char),
        C.POINTER(C.c_char),
    ]
    lib.elpis_ecsc_message_id.restype = C.c_int

    lib.elpis_ecsc_envelope_size.argtypes = [
        C.c_char_p,
        C.c_char_p,
        C.c_uint64,
        C.c_void_p,
        C.c_size_t,
        C.c_uint64,
        C.POINTER(C.c_size_t),
    ]
    lib.elpis_ecsc_envelope_size.restype = C.c_int

    lib.elpis_ecsc_envelope_write.argtypes = [
        C.c_char_p,
        C.c_char_p,
        C.c_uint64,
        C.c_void_p,
        C.c_size_t,
        C.c_uint64,
        C.POINTER(C.c_uint8),
        C.c_size_t,
        C.POINTER(C.c_size_t),
        C.POINTER(C.c_char),
        C.POINTER(C.c_char),
    ]
    lib.elpis_ecsc_envelope_write.restype = C.c_int

    lib.elpis_ecsc_event_frame_size.argtypes = [
        C.c_size_t,
        C.POINTER(C.c_size_t),
    ]
    lib.elpis_ecsc_event_frame_size.restype = C.c_int

    lib.elpis_ecsc_event_frame_write.argtypes = [
        C.c_void_p,
        C.c_size_t,
        C.POINTER(C.c_uint8),
        C.c_size_t,
        C.POINTER(C.c_size_t),
    ]
    lib.elpis_ecsc_event_frame_write.restype = C.c_int


def _payload_ptr(raw: bytes):
    buf = (C.c_uint8 * len(raw)).from_buffer_copy(raw)
    return buf, C.cast(buf, C.c_void_p)


@pytest.mark.parametrize(
    ("sequence", "logical_clock", "payload"),
    [
        (1, 1, b"x"),
        (7, 11, b'{"x":1}'),
        (2**31 + 17, 2**32 + 9, bytes(range(256))),
        ((1 << 63) - 1, (1 << 63) - 1, b"\x00\xffreceipt"),
    ],
)
def test_native_message_and_envelope_match_python_authority(
    lib, sequence, logical_clock, payload
):
    from elpis.ECS_C.bus import message_id, payload_digest, seal_envelope
    from elpis.ECS_C.entity import entity_id_from_founding, founding_record

    _bind_phase_b(lib)

    genesis = "a" * 64

    sender = entity_id_from_founding(
        founding_record(0, "sender", genesis)
    )
    receiver = entity_id_from_founding(
        founding_record(1, "receiver", genesis)
    )

    payload_buf, payload_ptr = _payload_ptr(payload)

    native_mid = (C.c_char * 65)()
    native_pd = (C.c_char * 65)()

    assert (
        lib.elpis_ecsc_message_id(
            sender.encode("ascii"),
            receiver.encode("ascii"),
            sequence,
            payload_ptr,
            len(payload),
            native_mid,
            native_pd,
        )
        == 0
    )

    mid = bytes(native_mid).split(b"\x00", 1)[0].decode("ascii")
    pd = bytes(native_pd).split(b"\x00", 1)[0].decode("ascii")

    assert pd == payload_digest(payload)
    assert mid == message_id(sender, sequence, receiver, payload)

    expected = seal_envelope(
        sender,
        receiver,
        sequence,
        payload,
        logical_clock,
    ).canonical_bytes()

    native_size = C.c_size_t()

    assert (
        lib.elpis_ecsc_envelope_size(
            sender.encode("ascii"),
            receiver.encode("ascii"),
            sequence,
            payload_ptr,
            len(payload),
            logical_clock,
            C.byref(native_size),
        )
        == 0
    )

    assert native_size.value == len(expected)

    out = (C.c_uint8 * native_size.value)()
    written = C.c_size_t()

    assert (
        lib.elpis_ecsc_envelope_write(
            sender.encode("ascii"),
            receiver.encode("ascii"),
            sequence,
            payload_ptr,
            len(payload),
            logical_clock,
            out,
            len(out),
            C.byref(written),
            native_mid,
            native_pd,
        )
        == 0
    )

    assert written.value == len(expected)
    assert bytes(out) == expected


@pytest.mark.parametrize(
    "canonical_event",
    [
        b"{}",
        b'{"a":1}',
        canonical.canonical_bytes(
            {
                "schema": canonical.EVENT_SCHEMA,
                "event_index": 0,
                "logical_clock": 1,
            }
        ),
    ],
)
def test_native_event_frame_matches_python_length_prefix(
    lib, canonical_event
):
    import struct

    _bind_phase_b(lib)

    source = (C.c_uint8 * len(canonical_event)).from_buffer_copy(
        canonical_event
    )

    size = C.c_size_t()

    assert (
        lib.elpis_ecsc_event_frame_size(
            len(canonical_event),
            C.byref(size),
        )
        == 0
    )

    assert size.value == 8 + len(canonical_event)

    out = (C.c_uint8 * size.value)()
    written = C.c_size_t()

    assert (
        lib.elpis_ecsc_event_frame_write(
            C.cast(source, C.c_void_p),
            len(canonical_event),
            out,
            len(out),
            C.byref(written),
        )
        == 0
    )

    expected = struct.pack(">Q", len(canonical_event)) + canonical_event

    assert written.value == len(expected)
    assert bytes(out) == expected


def test_native_message_rejects_noncanonical_entity_id(lib):
    _bind_phase_b(lib)

    payload = b"x"
    payload_buf, payload_ptr = _payload_ptr(payload)

    out_mid = (C.c_char * 65)()
    out_pd = (C.c_char * 65)()

    rc = lib.elpis_ecsc_message_id(
        b"NOT_AN_ENTITY_ID",
        b"2" * 64,
        1,
        payload_ptr,
        len(payload),
        out_mid,
        out_pd,
    )

    assert rc == -1
