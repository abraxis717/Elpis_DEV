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
