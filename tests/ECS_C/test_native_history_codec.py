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


def _bind_phase_c(lib):
    lib.elpis_ecsc_enqueue_event_size.argtypes = [
        C.c_char_p,
        C.c_char_p,
        C.c_uint64,
        C.c_void_p,
        C.c_size_t,
        C.c_uint64,
        C.c_char_p,
        C.c_char_p,
        C.c_char_p,
        C.POINTER(C.c_size_t),
    ]
    lib.elpis_ecsc_enqueue_event_size.restype = C.c_int

    lib.elpis_ecsc_enqueue_event_write.argtypes = [
        C.c_char_p,
        C.c_char_p,
        C.c_uint64,
        C.c_void_p,
        C.c_size_t,
        C.c_uint64,
        C.c_char_p,
        C.c_char_p,
        C.c_char_p,
        C.POINTER(C.c_uint8),
        C.c_size_t,
        C.POINTER(C.c_size_t),
        C.POINTER(C.c_char),
        C.POINTER(C.c_char),
        C.POINTER(C.c_char),
    ]
    lib.elpis_ecsc_enqueue_event_write.restype = C.c_int

    lib.elpis_ecsc_processed_event_size.argtypes = [
        C.c_char_p,
        C.c_char_p,
        C.c_uint64,
        C.c_char_p,
        C.c_char_p,
        C.c_char_p,
        C.POINTER(C.c_size_t),
    ]
    lib.elpis_ecsc_processed_event_size.restype = C.c_int

    lib.elpis_ecsc_processed_event_write.argtypes = [
        C.c_char_p,
        C.c_char_p,
        C.c_uint64,
        C.c_char_p,
        C.c_char_p,
        C.c_char_p,
        C.POINTER(C.c_uint8),
        C.c_size_t,
        C.POINTER(C.c_size_t),
        C.POINTER(C.c_char),
        C.POINTER(C.c_char),
    ]
    lib.elpis_ecsc_processed_event_write.restype = C.c_int


def _native_enqueue_event(
    lib,
    *,
    sender,
    receiver,
    sequence,
    payload,
    event_index,
    before,
    after,
    previous,
):
    _bind_phase_c(lib)

    payload_buf, payload_ptr = _payload_ptr(payload)

    size = C.c_size_t()

    assert (
        lib.elpis_ecsc_enqueue_event_size(
            sender.encode("ascii"),
            receiver.encode("ascii"),
            sequence,
            payload_ptr,
            len(payload),
            event_index,
            before.encode("ascii"),
            after.encode("ascii"),
            previous.encode("ascii"),
            C.byref(size),
        )
        == 0
    )

    out = (C.c_uint8 * size.value)()
    written = C.c_size_t()
    event_digest = (C.c_char * 65)()
    intent_digest = (C.c_char * 65)()
    message_id = (C.c_char * 65)()

    assert (
        lib.elpis_ecsc_enqueue_event_write(
            sender.encode("ascii"),
            receiver.encode("ascii"),
            sequence,
            payload_ptr,
            len(payload),
            event_index,
            before.encode("ascii"),
            after.encode("ascii"),
            previous.encode("ascii"),
            out,
            len(out),
            C.byref(written),
            event_digest,
            intent_digest,
            message_id,
        )
        == 0
    )

    return (
        bytes(out[: written.value]),
        bytes(event_digest).split(b"\x00", 1)[0].decode("ascii"),
        bytes(intent_digest).split(b"\x00", 1)[0].decode("ascii"),
        bytes(message_id).split(b"\x00", 1)[0].decode("ascii"),
    )


def _native_processed_event(
    lib,
    *,
    receiver,
    message_id,
    event_index,
    before,
    after,
    previous,
):
    _bind_phase_c(lib)

    size = C.c_size_t()

    assert (
        lib.elpis_ecsc_processed_event_size(
            receiver.encode("ascii"),
            message_id.encode("ascii"),
            event_index,
            before.encode("ascii"),
            after.encode("ascii"),
            previous.encode("ascii"),
            C.byref(size),
        )
        == 0
    )

    out = (C.c_uint8 * size.value)()
    written = C.c_size_t()
    event_digest = (C.c_char * 65)()
    intent_digest = (C.c_char * 65)()

    assert (
        lib.elpis_ecsc_processed_event_write(
            receiver.encode("ascii"),
            message_id.encode("ascii"),
            event_index,
            before.encode("ascii"),
            after.encode("ascii"),
            previous.encode("ascii"),
            out,
            len(out),
            C.byref(written),
            event_digest,
            intent_digest,
        )
        == 0
    )

    return (
        bytes(out[: written.value]),
        bytes(event_digest).split(b"\x00", 1)[0].decode("ascii"),
        bytes(intent_digest).split(b"\x00", 1)[0].decode("ascii"),
    )


@pytest.mark.parametrize(
    ("sequence", "payload", "event_index"),
    [
        (1, b"x", 0),
        (7, b'{"receipt":"alpha"}', 10),
        (91, bytes(range(256)), 1234),
    ],
)
def test_native_enqueue_event_is_exact_python_authority(
    lib, sequence, payload, event_index
):
    from elpis.ECS_C.bus import seal_envelope
    from elpis.ECS_C.entity import (
        entity_id_from_founding,
        founding_record,
    )
    from elpis.ECS_C.persistence import (
        build_event,
        event_intent_digest,
    )

    genesis = "9" * 64

    sender = entity_id_from_founding(
        founding_record(0, "sender", genesis)
    )
    receiver = entity_id_from_founding(
        founding_record(1, "receiver", genesis)
    )

    before = "1" * 64
    after = "2" * 64
    previous = "3" * 64
    clock = event_index + 1

    env = seal_envelope(
        sender,
        receiver,
        sequence,
        payload,
        clock,
    )

    expected = build_event(
        event_index=event_index,
        logical_clock=clock,
        transaction_id=f"ENQ:{env.message_id}",
        event_kind="MESSAGE_ENQUEUED",
        entity_id=receiver,
        payload={"envelope": env.to_dict()},
        before_state_root=before,
        after_state_root=after,
        prev_event_digest=previous,
    )

    (
        native_bytes,
        native_event_digest,
        native_intent_digest,
        native_message_id,
    ) = _native_enqueue_event(
        lib,
        sender=sender,
        receiver=receiver,
        sequence=sequence,
        payload=payload,
        event_index=event_index,
        before=before,
        after=after,
        previous=previous,
    )

    assert native_message_id == env.message_id
    assert native_event_digest == expected["event_digest"]
    assert native_intent_digest == event_intent_digest(expected)
    assert native_bytes == canonical.canonical_bytes(expected)


@pytest.mark.parametrize("event_index", [1, 11, 999])
def test_native_processed_event_is_exact_python_authority(
    lib, event_index
):
    from elpis.ECS_C.entity import (
        entity_id_from_founding,
        founding_record,
    )
    from elpis.ECS_C.persistence import (
        build_event,
        event_intent_digest,
    )

    genesis = "8" * 64
    receiver = entity_id_from_founding(
        founding_record(1, "receiver", genesis)
    )

    message_id = "a" * 64
    before = "4" * 64
    after = "5" * 64
    previous = "6" * 64
    clock = event_index + 1

    expected = build_event(
        event_index=event_index,
        logical_clock=clock,
        transaction_id=f"PROC:{message_id}",
        event_kind="MESSAGE_PROCESSED",
        entity_id=receiver,
        payload={
            "message_id": message_id,
            "receiver_entity_id": receiver,
        },
        before_state_root=before,
        after_state_root=after,
        prev_event_digest=previous,
    )

    (
        native_bytes,
        native_event_digest,
        native_intent_digest,
    ) = _native_processed_event(
        lib,
        receiver=receiver,
        message_id=message_id,
        event_index=event_index,
        before=before,
        after=after,
        previous=previous,
    )

    assert native_event_digest == expected["event_digest"]
    assert native_intent_digest == event_intent_digest(expected)
    assert native_bytes == canonical.canonical_bytes(expected)


def test_native_committed_event_refuses_index_overflow(lib):
    _bind_phase_c(lib)

    size = C.c_size_t(999)

    rc = lib.elpis_ecsc_processed_event_size(
        b"1" * 64,
        b"2" * 64,
        (1 << 63) - 1,
        b"3" * 64,
        b"4" * 64,
        b"5" * 64,
        C.byref(size),
    )

    assert rc == -1
    assert size.value == 0


def test_processed_event_allows_prev_digest_output_buffer_reuse(lib):
    """Natural native chaining may reuse last event digest as next prev/output."""

    _bind_phase_c(lib)

    receiver = b"2" * 64
    message_id = b"a" * 64
    before = b"3" * 64
    after = b"4" * 64

    # Same storage is deliberately used as both prev_event_digest input and
    # out_event_digest output.
    chained_digest = (C.c_char * 65)()
    chained_digest.value = b"5" * 64

    size = C.c_size_t()

    assert (
        lib.elpis_ecsc_processed_event_size(
            receiver,
            message_id,
            7,
            before,
            after,
            C.cast(chained_digest, C.c_char_p),
            C.byref(size),
        )
        == 0
    )

    out = (C.c_uint8 * size.value)()
    written = C.c_size_t()
    intent = (C.c_char * 65)()

    assert (
        lib.elpis_ecsc_processed_event_write(
            receiver,
            message_id,
            7,
            before,
            after,
            C.cast(chained_digest, C.c_char_p),
            out,
            len(out),
            C.byref(written),
            chained_digest,
            intent,
        )
        == 0
    )

    assert written.value == size.value
    assert len(bytes(chained_digest).split(b"\x00", 1)[0]) == 64
    assert len(bytes(intent).split(b"\x00", 1)[0]) == 64


def _bind_phase_d(lib):
    lib.elpis_ecsc_genesis_digest.argtypes = [
        C.c_char_p,
        C.c_size_t,
        C.c_uint32,
        C.POINTER(C.c_char),
    ]
    lib.elpis_ecsc_genesis_digest.restype = C.c_int

    lib.elpis_ecsc_entity_id.argtypes = [
        C.c_uint64,
        C.c_char_p,
        C.c_size_t,
        C.c_char_p,
        C.POINTER(C.c_char),
    ]
    lib.elpis_ecsc_entity_id.restype = C.c_int

    lib.elpis_ecsc_initial_state_digest.argtypes = [
        C.c_char_p,
        C.POINTER(C.c_char),
    ]
    lib.elpis_ecsc_initial_state_digest.restype = C.c_int


def _char65_text(buf):
    return bytes(buf).split(b"\x00", 1)[0].decode("ascii")


@pytest.mark.parametrize(
    ("scheduler_code", "scheduler"),
    [
        (
            1,
            "active-mailbox-fifo/"
            "entity-id-before-founded-activation.v1",
        ),
        (
            2,
            "active-mailbox-global-arrival/"
            "founding-index-activation.v2",
        ),
    ],
)
@pytest.mark.parametrize(
    "label",
    [
        "elpis.runtime.history.v1",
        "history-test",
        "hist-é-😀",
    ],
)
def test_native_genesis_identity_is_exact_python_authority(
    lib, scheduler_code, scheduler, label
):
    from elpis.ECS_C.persistence import (
        genesis_descriptor_digest,
    )

    _bind_phase_d(lib)

    raw = label.encode("utf-8")
    out = (C.c_char * 65)()

    assert (
        lib.elpis_ecsc_genesis_digest(
            raw,
            len(raw),
            scheduler_code,
            out,
        )
        == 0
    )

    assert _char65_text(out) == genesis_descriptor_digest(
        label,
        scheduler,
    )


@pytest.mark.parametrize(
    ("index", "label"),
    [
        (0, "history"),
        (1, "pipeline"),
        (2, "structure"),
        (3, "evolution"),
        (4, "inference"),
        (5, "ecs_g"),
        (9, "role-é-😀"),
        ((1 << 63) - 1, "max-index"),
    ],
)
def test_native_entity_identity_is_exact_python_authority(
    lib, index, label
):
    from elpis.ECS_C.entity import (
        entity_id_from_founding,
        founding_record,
        initial_state_digest,
    )
    from elpis.ECS_C.persistence import (
        genesis_descriptor_digest,
    )
    from elpis.ECS_C.scheduler import SCHEDULER_V2

    _bind_phase_d(lib)

    genesis = genesis_descriptor_digest(
        "elpis.runtime.history.v1",
        SCHEDULER_V2,
    )

    label_raw = label.encode("utf-8")

    out_entity = (C.c_char * 65)()

    assert (
        lib.elpis_ecsc_entity_id(
            index,
            label_raw,
            len(label_raw),
            genesis.encode("ascii"),
            out_entity,
        )
        == 0
    )

    native_entity = _char65_text(out_entity)

    expected_entity = entity_id_from_founding(
        founding_record(
            index,
            label,
            genesis,
        )
    )

    assert native_entity == expected_entity

    out_state = (C.c_char * 65)()

    assert (
        lib.elpis_ecsc_initial_state_digest(
            native_entity.encode("ascii"),
            out_state,
        )
        == 0
    )

    assert _char65_text(out_state) == initial_state_digest(
        expected_entity
    )


def test_native_runtime_role_identity_set_matches_python(
    lib,
):
    from elpis.ECS_C.entity import (
        entity_id_from_founding,
        founding_record,
    )
    from elpis.ECS_C.persistence import (
        genesis_descriptor_digest,
    )
    from elpis.ECS_C.scheduler import SCHEDULER_V2

    _bind_phase_d(lib)

    roles = (
        "history",
        "pipeline",
        "structure",
        "evolution",
        "inference",
        "ecs_g",
    )

    genesis = genesis_descriptor_digest(
        "elpis.runtime.history.v1",
        SCHEDULER_V2,
    )

    native = []

    for index, role in enumerate(roles):
        role_raw = role.encode("ascii")
        out = (C.c_char * 65)()

        assert (
            lib.elpis_ecsc_entity_id(
                index,
                role_raw,
                len(role_raw),
                genesis.encode("ascii"),
                out,
            )
            == 0
        )

        native.append(_char65_text(out))

    python = [
        entity_id_from_founding(
            founding_record(
                index,
                role,
                genesis,
            )
        )
        for index, role in enumerate(roles)
    ]

    assert native == python
    assert len(set(native)) == len(roles)


class StateEntity(C.Structure):
    _fields_ = [
        ("registry_key", C.c_char_p),
        ("entity_id", C.c_char_p),
        ("label", C.c_char_p),
        ("label_len", C.c_size_t),
        ("founding_index", C.c_uint64),
        ("founding_digest", C.c_char_p),
        ("state_entity_id", C.c_char_p),
        ("prev_state_digest", C.c_char_p),
        ("lifecycle", C.c_uint32),
        ("state_version", C.c_uint64),
        ("state_digest", C.c_char_p),
        ("has_delivered", C.c_uint32),
        ("delivered", C.c_uint64),
    ]


class StateEnvelope(C.Structure):
    _fields_ = [
        ("logical_clock", C.c_uint64),
        ("message_id", C.c_char_p),
        ("payload_digest", C.c_char_p),
        ("payload", C.c_void_p),
        ("payload_size", C.c_size_t),
        ("receiver_entity_id", C.c_char_p),
        ("sender_entity_id", C.c_char_p),
        ("sequence", C.c_uint64),
    ]


class StateMailbox(C.Structure):
    _fields_ = [
        ("mailbox_key", C.c_char_p),
        ("receiver_entity_id", C.c_char_p),
        ("capacity", C.c_uint64),
        ("contents", C.POINTER(StateEnvelope)),
        ("content_count", C.c_size_t),
    ]


class StateWatermark(C.Structure):
    _fields_ = [
        ("sender_entity_id", C.c_char_p),
        ("sequence", C.c_uint64),
    ]


class StateRoot(C.Structure):
    _fields_ = [
        ("genesis_digest", C.c_char_p),
        ("history_digest", C.c_char_p),
        ("logical_clock", C.c_uint64),
        ("mailbox_capacity", C.c_uint64),
        ("mailbox_default_capacity", C.c_uint64),
        ("next_founding_index", C.c_uint64),
        ("entities", C.POINTER(StateEntity)),
        ("entity_count", C.c_size_t),
        ("mailboxes", C.POINTER(StateMailbox)),
        ("mailbox_count", C.c_size_t),
        ("watermarks", C.POINTER(StateWatermark)),
        ("watermark_count", C.c_size_t),
    ]


_LIFECYCLE_CODE = {
    "FOUNDED": 1,
    "ACTIVE": 2,
    "DORMANT": 3,
    "TERMINATED": 4,
}


def _bind_phase_e(lib):
    lib.elpis_ecsc_state_root_digest.argtypes = [
        C.POINTER(StateRoot),
        C.POINTER(C.c_char),
    ]
    lib.elpis_ecsc_state_root_digest.restype = C.c_int


def _native_root_from_python_root(lib, root):
    _bind_phase_e(lib)

    keepalive = []

    def keep_bytes(value, encoding="ascii"):
        raw = value.encode(encoding)
        keepalive.append(raw)
        return raw

    entity_values = []

    for entity in root["entities"]:
        payload = entity["payload"]

        assert set(payload).issubset({"delivered"})

        entity_values.append(
            StateEntity(
                keep_bytes(entity["registry_key"]),
                keep_bytes(entity["entity_id"]),
                keep_bytes(entity["label"], "utf-8"),
                len(entity["label"].encode("utf-8")),
                entity["founding_index"],
                keep_bytes(entity["founding_digest"]),
                keep_bytes(entity["state_entity_id"]),
                keep_bytes(entity["prev_state_digest"]),
                _LIFECYCLE_CODE[entity["lifecycle"]],
                entity["state_version"],
                keep_bytes(entity["state_digest"]),
                1 if "delivered" in payload else 0,
                payload.get("delivered", 0),
            )
        )

    if entity_values:
        EntityArray = StateEntity * len(entity_values)
        entities = EntityArray(*entity_values)
        keepalive.append(entities)
        entity_ptr = C.cast(
            entities,
            C.POINTER(StateEntity),
        )
    else:
        entity_ptr = C.POINTER(StateEntity)()

    mailbox_values = []

    for mailbox in root["mailboxes"]:
        env_values = []

        for env in mailbox["contents"]:
            payload = bytes.fromhex(env["payload_hex"])
            payload_buf = (
                C.c_uint8 * len(payload)
            ).from_buffer_copy(payload)

            keepalive.append(payload_buf)

            env_values.append(
                StateEnvelope(
                    env["logical_clock"],
                    keep_bytes(env["message_id"]),
                    keep_bytes(env["payload_digest"]),
                    C.cast(payload_buf, C.c_void_p),
                    len(payload),
                    keep_bytes(
                        env["receiver_entity_id"]
                    ),
                    keep_bytes(
                        env["sender_entity_id"]
                    ),
                    env["sequence"],
                )
            )

        if env_values:
            EnvelopeArray = (
                StateEnvelope * len(env_values)
            )
            envelopes = EnvelopeArray(*env_values)
            keepalive.append(envelopes)

            env_ptr = C.cast(
                envelopes,
                C.POINTER(StateEnvelope),
            )
        else:
            env_ptr = C.POINTER(StateEnvelope)()

        mailbox_values.append(
            StateMailbox(
                keep_bytes(mailbox["mailbox_key"]),
                keep_bytes(
                    mailbox["receiver_entity_id"]
                ),
                mailbox["capacity"],
                env_ptr,
                len(env_values),
            )
        )

    if mailbox_values:
        MailboxArray = StateMailbox * len(
            mailbox_values
        )
        mailboxes = MailboxArray(*mailbox_values)
        keepalive.append(mailboxes)

        mailbox_ptr = C.cast(
            mailboxes,
            C.POINTER(StateMailbox),
        )
    else:
        mailbox_ptr = C.POINTER(StateMailbox)()

    watermark_values = [
        StateWatermark(
            keep_bytes(sender),
            sequence,
        )
        for sender, sequence
        in sorted(root["watermarks"].items())
    ]

    if watermark_values:
        WatermarkArray = StateWatermark * len(
            watermark_values
        )
        watermarks = WatermarkArray(
            *watermark_values
        )
        keepalive.append(watermarks)

        watermark_ptr = C.cast(
            watermarks,
            C.POINTER(StateWatermark),
        )
    else:
        watermark_ptr = (
            C.POINTER(StateWatermark)()
        )

    native_root = StateRoot(
        keep_bytes(root["genesis_digest"]),
        keep_bytes(root["history_digest"]),
        root["logical_clock"],
        root["mailbox_capacity"],
        root["mailbox_default_capacity"],
        root["next_founding_index"],
        entity_ptr,
        len(entity_values),
        mailbox_ptr,
        len(mailbox_values),
        watermark_ptr,
        len(watermark_values),
    )

    out = (C.c_char * 65)()

    rc = lib.elpis_ecsc_state_root_digest(
        C.byref(native_root),
        out,
    )

    assert rc == 0

    return _char65_text(out)


def _assert_native_root_matches(lib, state):
    from elpis.ECS_C.persistence import (
        state_root_digest,
    )

    root = state.state_root()

    assert (
        _native_root_from_python_root(
            lib,
            root,
        )
        == state_root_digest(root)
    )


def test_native_state_root_matches_empty_v2_state(
    lib,
):
    from elpis.ECS_C.persistence import (
        genesis_descriptor_digest,
    )
    from elpis.ECS_C.replay import initial_state
    from elpis.ECS_C.scheduler import SCHEDULER_V2

    genesis = genesis_descriptor_digest(
        "elpis.runtime.history.v1",
        SCHEDULER_V2,
    )

    state = initial_state(
        genesis,
        mailbox_capacity=16,
        scheduler_protocol=SCHEDULER_V2,
    )

    _assert_native_root_matches(
        lib,
        state,
    )


def test_native_state_root_matches_runtime_history_evolution(
    lib,
    tmp_path,
):
    from elpis.ECS_C.kernel import Kernel
    from elpis.runtime.history import (
        HISTORY_GENESIS_LABEL,
        ROLES,
        ReceiptRecord,
    )

    storage = (
        tmp_path /
        "native-state-root-differential"
    )

    kernel = Kernel(
        str(storage),
        genesis_label=HISTORY_GENESIS_LABEL,
    )

    kernel.open()

    try:
        # Empty v2 history.
        _assert_native_root_matches(
            lib,
            kernel.state,
        )

        # Existing five-role runtime authority.
        ids = {}

        for role in ROLES:
            ids[role] = kernel.found_entity(role)

        kernel.run_until_quiescent()

        _assert_native_root_matches(
            lib,
            kernel.state,
        )

        # Exact deterministic legacy extension shape:
        # append one normal ECS founding transition for ecs_g,
        # then its normal activation transition.
        ids["ecs_g"] = kernel.found_entity("ecs_g")

        _assert_native_root_matches(
            lib,
            kernel.state,
        )

        kernel.run_until_quiescent()

        _assert_native_root_matches(
            lib,
            kernel.state,
        )

        # Runtime receipt payload is legal protocol bytes even though
        # ReceiptRecord cannot yet admit ecs_g as a recorder.
        receipt_payload = (
            b'{"bindings":{"epoch_after":"1"},'
            b'"digest":"'
            + b"a" * 64
            + b'","kind":"cognition.turn",'
            b'"schema":"elpis.runtime.receipt-record.v1",'
            b'"subsystem":"ecs_g"}'
        )

        port = kernel.entity_port(
            ids["ecs_g"]
        )

        message_id = port.propose(
            ids["history"],
            receipt_payload,
        )

        assert len(message_id) == 64

        # MESSAGE_ENQUEUED:
        # watermark + one queued envelope.
        _assert_native_root_matches(
            lib,
            kernel.state,
        )

        # MESSAGE_PROCESSED:
        # queue remains materialized but empty;
        # history state gains delivered=1.
        assert kernel.step() == 1

        _assert_native_root_matches(
            lib,
            kernel.state,
        )

        state = kernel.state

        history = state.registry.get(
            ids["history"]
        )

        assert history.state.payload == {
            "delivered": 1
        }

        assert (
            state.watermarks.get(ids["ecs_g"])
            == 1
        )

        assert (
            state.mailboxes.box(
                ids["history"]
            ).peek()
            is None
        )

    finally:
        kernel.close()


def test_native_state_root_rejects_reordered_projection(
    lib,
):
    from elpis.ECS_C.persistence import (
        genesis_descriptor_digest,
    )
    from elpis.ECS_C.replay import initial_state
    from elpis.ECS_C.scheduler import SCHEDULER_V2

    _bind_phase_e(lib)

    genesis = genesis_descriptor_digest(
        "elpis.runtime.history.v1",
        SCHEDULER_V2,
    )

    state = initial_state(
        genesis,
        scheduler_protocol=SCHEDULER_V2,
    )

    root = state.state_root()

    # Empty projection has no reorder surface. Add explicit
    # fake structurally-valid entities and deliberately reverse
    # their registry-key order.
    ids = [
        "1" * 64,
        "2" * 64,
    ]

    entities = []

    for index, entity_id in enumerate(ids):
        entities.append(
            {
                "registry_key": entity_id,
                "entity_id": entity_id,
                "label": f"r{index}",
                "founding_index": index,
                "founding_digest": entity_id,
                "state_entity_id": entity_id,
                "prev_state_digest": "0" * 64,
                "lifecycle": "ACTIVE",
                "state_version": 1,
                "state_digest": (
                    "a" if index == 0 else "b"
                ) * 64,
                "payload": {},
            }
        )

    root["entities"] = list(
        reversed(entities)
    )
    root["next_founding_index"] = 2

    keepalive = []

    def kb(value):
        raw = value.encode("ascii")
        keepalive.append(raw)
        return raw

    values = []

    for entity in root["entities"]:
        values.append(
            StateEntity(
                kb(entity["registry_key"]),
                kb(entity["entity_id"]),
                kb(entity["label"]),
                len(entity["label"]),
                entity["founding_index"],
                kb(entity["founding_digest"]),
                kb(entity["state_entity_id"]),
                kb(entity["prev_state_digest"]),
                2,
                entity["state_version"],
                kb(entity["state_digest"]),
                0,
                0,
            )
        )

    Array = StateEntity * 2
    array = Array(*values)
    keepalive.append(array)

    native_root = StateRoot(
        kb(root["genesis_digest"]),
        kb(root["history_digest"]),
        root["logical_clock"],
        root["mailbox_capacity"],
        root["mailbox_default_capacity"],
        root["next_founding_index"],
        C.cast(
            array,
            C.POINTER(StateEntity),
        ),
        2,
        C.POINTER(StateMailbox)(),
        0,
        C.POINTER(StateWatermark)(),
        0,
    )

    out = (C.c_char * 65)()

    assert (
        lib.elpis_ecsc_state_root_digest(
            C.byref(native_root),
            out,
        )
        == -1
    )


class NativeLogScan(C.Structure):
    _fields_ = [
        ("total_size", C.c_uint64),
        ("complete_prefix", C.c_uint64),
        ("frame_count", C.c_uint64),
        ("has_incomplete_tail", C.c_uint32),
    ]


def _bind_phase_f1(lib):
    lib.elpis_ecsc_log_open.argtypes = [
        C.c_char_p,
        C.c_size_t,
        C.POINTER(C.c_void_p),
    ]
    lib.elpis_ecsc_log_open.restype = C.c_int

    lib.elpis_ecsc_log_close.argtypes = [
        C.c_void_p,
    ]
    lib.elpis_ecsc_log_close.restype = None

    lib.elpis_ecsc_log_recover_scan.argtypes = [
        C.c_void_p,
        C.POINTER(NativeLogScan),
    ]
    lib.elpis_ecsc_log_recover_scan.restype = C.c_int

    lib.elpis_ecsc_log_finish_recovery.argtypes = [
        C.c_void_p,
        C.c_uint64,
    ]
    lib.elpis_ecsc_log_finish_recovery.restype = C.c_int

    lib.elpis_ecsc_log_append_event_bytes.argtypes = [
        C.c_void_p,
        C.c_void_p,
        C.c_size_t,
    ]
    lib.elpis_ecsc_log_append_event_bytes.restype = C.c_int

    lib.elpis_ecsc_log_size.argtypes = [
        C.c_void_p,
    ]
    lib.elpis_ecsc_log_size.restype = C.c_uint64

    lib.elpis_ecsc_log_frame_count.argtypes = [
        C.c_void_p,
    ]
    lib.elpis_ecsc_log_frame_count.restype = C.c_uint64


def _native_log_open(lib, path):
    _bind_phase_f1(lib)

    raw = str(path).encode("utf-8")
    handle = C.c_void_p()

    rc = lib.elpis_ecsc_log_open(
        raw,
        len(raw),
        C.byref(handle),
    )

    return rc, handle


def test_native_log_exact_append_and_crash_tail_recovery(
    lib,
    tmp_path,
):
    import struct

    _bind_phase_f1(lib)

    path = tmp_path / "events.log"

    rc, log = _native_log_open(lib, path)

    assert rc == 0
    assert log.value is not None

    try:
        second_rc, second = _native_log_open(
            lib,
            path,
        )

        assert second_rc == -5
        assert second.value is None

        event = canonical.canonical_bytes(
            {
                "a": 1,
                "z": "receipt",
            }
        )

        event_buf = (
            C.c_uint8 * len(event)
        ).from_buffer_copy(event)

        # Recovery must be explicitly completed.
        assert (
            lib.elpis_ecsc_log_append_event_bytes(
                log,
                C.cast(event_buf, C.c_void_p),
                len(event),
            )
            == -9
        )

        scan = NativeLogScan()

        assert (
            lib.elpis_ecsc_log_recover_scan(
                log,
                C.byref(scan),
            )
            == 0
        )

        assert scan.total_size == 0
        assert scan.complete_prefix == 0
        assert scan.frame_count == 0
        assert scan.has_incomplete_tail == 0

        assert (
            lib.elpis_ecsc_log_finish_recovery(
                log,
                0,
            )
            == 0
        )

        assert (
            lib.elpis_ecsc_log_append_event_bytes(
                log,
                C.cast(event_buf, C.c_void_p),
                len(event),
            )
            == 0
        )

        expected = (
            struct.pack(">Q", len(event))
            + event
        )

        assert path.read_bytes() == expected

        assert (
            lib.elpis_ecsc_log_size(log)
            == len(expected)
        )

        assert (
            lib.elpis_ecsc_log_frame_count(log)
            == 1
        )

    finally:
        lib.elpis_ecsc_log_close(log)

    # Crash debris: partial next frame prefix.
    with path.open("ab") as fh:
        fh.write(b"\x00\x00\x00")
        fh.flush()

        import os
        os.fsync(fh.fileno())

    rc, log = _native_log_open(lib, path)
    assert rc == 0

    try:
        scan = NativeLogScan()

        assert (
            lib.elpis_ecsc_log_recover_scan(
                log,
                C.byref(scan),
            )
            == 0
        )

        assert scan.frame_count == 1
        assert scan.has_incomplete_tail == 1

        expected_prefix = len(expected)

        assert (
            scan.complete_prefix
            == expected_prefix
        )

        assert (
            scan.total_size
            == expected_prefix + 3
        )

        # Complete frames may not be erased as "recovery".
        assert (
            lib.elpis_ecsc_log_finish_recovery(
                log,
                0,
            )
            == -6
        )

        # Caller says every complete frame was semantically validated.
        assert (
            lib.elpis_ecsc_log_finish_recovery(
                log,
                expected_prefix,
            )
            == 0
        )

    finally:
        lib.elpis_ecsc_log_close(log)

    assert path.read_bytes() == expected


def test_native_log_recovery_refuses_impossible_frame_length(
    lib,
    tmp_path,
):
    import struct

    path = tmp_path / "events.log"

    # Complete length prefix: zero is never legal.
    path.write_bytes(struct.pack(">Q", 0))

    rc, log = _native_log_open(lib, path)
    assert rc == 0

    try:
        scan = NativeLogScan()

        assert (
            lib.elpis_ecsc_log_recover_scan(
                log,
                C.byref(scan),
            )
            == -6
        )
    finally:
        lib.elpis_ecsc_log_close(log)


def test_native_log_recovery_refuses_oversize_frame_length(
    lib,
    tmp_path,
):
    import struct

    path = tmp_path / "events.log"

    path.write_bytes(
        struct.pack(">Q", 262145)
    )

    rc, log = _native_log_open(lib, path)
    assert rc == 0

    try:
        scan = NativeLogScan()

        assert (
            lib.elpis_ecsc_log_recover_scan(
                log,
                C.byref(scan),
            )
            == -6
        )
    finally:
        lib.elpis_ecsc_log_close(log)


def test_native_log_accepts_incomplete_payload_only_as_recovery_tail(
    lib,
    tmp_path,
):
    import struct

    path = tmp_path / "events.log"

    path.write_bytes(
        struct.pack(">Q", 10)
        + b"abc"
    )

    rc, log = _native_log_open(lib, path)
    assert rc == 0

    try:
        scan = NativeLogScan()

        assert (
            lib.elpis_ecsc_log_recover_scan(
                log,
                C.byref(scan),
            )
            == 0
        )

        assert scan.complete_prefix == 0
        assert scan.frame_count == 0
        assert scan.has_incomplete_tail == 1

        assert (
            lib.elpis_ecsc_log_finish_recovery(
                log,
                0,
            )
            == 0
        )

    finally:
        lib.elpis_ecsc_log_close(log)

    assert path.read_bytes() == b""
