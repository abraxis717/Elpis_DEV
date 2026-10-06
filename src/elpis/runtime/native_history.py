"""Narrow ctypes binding for native ECS_C runtime-history ownership.

This module contains no history semantics of its own.  The state-view ctypes
layout is copied from the qualified native-history differential authority and
is checked again when a session opens because native ECS_C recomputes the
supplied state root before accepting ownership.
"""
from __future__ import annotations

import ctypes as C
from pathlib import Path


class NativeHistoryError(RuntimeError):
    def __init__(self, code: int, operation: str):
        self.code = int(code)
        self.operation = operation
        super().__init__(f"{operation}: native ECS_C rc={code}")


class StateEntity(C.Structure):
    _fields_ = [('registry_key', C.c_char_p), ('entity_id', C.c_char_p), ('label', C.c_char_p), ('label_len', C.c_size_t), ('founding_index', C.c_uint64), ('founding_digest', C.c_char_p), ('state_entity_id', C.c_char_p), ('prev_state_digest', C.c_char_p), ('lifecycle', C.c_uint32), ('state_version', C.c_uint64), ('state_digest', C.c_char_p), ('has_delivered', C.c_uint32), ('delivered', C.c_uint64)]

class StateEnvelope(C.Structure):
    _fields_ = [('logical_clock', C.c_uint64), ('message_id', C.c_char_p), ('payload_digest', C.c_char_p), ('payload', C.c_void_p), ('payload_size', C.c_size_t), ('receiver_entity_id', C.c_char_p), ('sender_entity_id', C.c_char_p), ('sequence', C.c_uint64)]

class StateMailbox(C.Structure):
    _fields_ = [('mailbox_key', C.c_char_p), ('receiver_entity_id', C.c_char_p), ('capacity', C.c_uint64), ('contents', C.POINTER(StateEnvelope)), ('content_count', C.c_size_t)]

class StateWatermark(C.Structure):
    _fields_ = [('sender_entity_id', C.c_char_p), ('sequence', C.c_uint64)]

class StateRoot(C.Structure):
    _fields_ = [('genesis_digest', C.c_char_p), ('history_digest', C.c_char_p), ('logical_clock', C.c_uint64), ('mailbox_capacity', C.c_uint64), ('mailbox_default_capacity', C.c_uint64), ('next_founding_index', C.c_uint64), ('entities', C.POINTER(StateEntity)), ('entity_count', C.c_size_t), ('mailboxes', C.POINTER(StateMailbox)), ('mailbox_count', C.c_size_t), ('watermarks', C.POINTER(StateWatermark)), ('watermark_count', C.c_size_t)]

_LIFECYCLE_CODE = {'FOUNDED': 1, 'ACTIVE': 2, 'DORMANT': 3, 'TERMINATED': 4}

class RuntimeRecordPlanResult(C.Structure):
    _fields_ = [('sequence', C.c_uint64), ('enqueue_event_index', C.c_uint64), ('processed_event_index', C.c_uint64), ('final_logical_clock', C.c_uint64), ('final_history_state_version', C.c_uint64), ('final_delivered', C.c_uint64), ('enqueue_event_size', C.c_size_t), ('processed_event_size', C.c_size_t), ('message_id', C.c_char * 65), ('enqueue_state_root', C.c_char * 65), ('final_state_root', C.c_char * 65), ('final_history_digest', C.c_char * 65), ('enqueue_event_digest', C.c_char * 65), ('processed_event_digest', C.c_char * 65), ('final_history_state_digest', C.c_char * 65)]

def _state_root_ctypes_view(root):
    keepalive = []

    def keep_bytes(value, encoding='ascii'):
        raw = value.encode(encoding)
        keepalive.append(raw)
        return raw
    entity_values = []
    for entity in root['entities']:
        payload = entity['payload']
        assert set(payload).issubset({'delivered'})
        entity_values.append(StateEntity(keep_bytes(entity['registry_key']), keep_bytes(entity['entity_id']), keep_bytes(entity['label'], 'utf-8'), len(entity['label'].encode('utf-8')), entity['founding_index'], keep_bytes(entity['founding_digest']), keep_bytes(entity['state_entity_id']), keep_bytes(entity['prev_state_digest']), _LIFECYCLE_CODE[entity['lifecycle']], entity['state_version'], keep_bytes(entity['state_digest']), 1 if 'delivered' in payload else 0, payload.get('delivered', 0)))
    if entity_values:
        EntityArray = StateEntity * len(entity_values)
        entities = EntityArray(*entity_values)
        keepalive.append(entities)
        entity_ptr = C.cast(entities, C.POINTER(StateEntity))
    else:
        entity_ptr = C.POINTER(StateEntity)()
    mailbox_values = []
    for mailbox in root['mailboxes']:
        env_values = []
        for env in mailbox['contents']:
            payload = bytes.fromhex(env['payload_hex'])
            payload_buf = (C.c_uint8 * len(payload)).from_buffer_copy(payload)
            keepalive.append(payload_buf)
            env_values.append(StateEnvelope(env['logical_clock'], keep_bytes(env['message_id']), keep_bytes(env['payload_digest']), C.cast(payload_buf, C.c_void_p), len(payload), keep_bytes(env['receiver_entity_id']), keep_bytes(env['sender_entity_id']), env['sequence']))
        if env_values:
            EnvelopeArray = StateEnvelope * len(env_values)
            envelopes = EnvelopeArray(*env_values)
            keepalive.append(envelopes)
            env_ptr = C.cast(envelopes, C.POINTER(StateEnvelope))
        else:
            env_ptr = C.POINTER(StateEnvelope)()
        mailbox_values.append(StateMailbox(keep_bytes(mailbox['mailbox_key']), keep_bytes(mailbox['receiver_entity_id']), mailbox['capacity'], env_ptr, len(env_values)))
    if mailbox_values:
        MailboxArray = StateMailbox * len(mailbox_values)
        mailboxes = MailboxArray(*mailbox_values)
        keepalive.append(mailboxes)
        mailbox_ptr = C.cast(mailboxes, C.POINTER(StateMailbox))
    else:
        mailbox_ptr = C.POINTER(StateMailbox)()
    watermark_values = [StateWatermark(keep_bytes(sender), sequence) for sender, sequence in sorted(root['watermarks'].items())]
    if watermark_values:
        WatermarkArray = StateWatermark * len(watermark_values)
        watermarks = WatermarkArray(*watermark_values)
        keepalive.append(watermarks)
        watermark_ptr = C.cast(watermarks, C.POINTER(StateWatermark))
    else:
        watermark_ptr = C.POINTER(StateWatermark)()
    native = StateRoot(keep_bytes(root['genesis_digest']), keep_bytes(root['history_digest']), root['logical_clock'], root['mailbox_capacity'], root['mailbox_default_capacity'], root['next_founding_index'], entity_ptr, len(entity_values), mailbox_ptr, len(mailbox_values), watermark_ptr, len(watermark_values))
    keepalive.append(native)
    return (native, keepalive)


def _text(value) -> str:
    return bytes(value).split(b"\x00", 1)[0].decode("ascii")


class NativeHistorySession:
    """Owning Python handle over one ``elpis_ecsc_runtime_session``."""

    _ERRORS = {
        -1: "INVALID",
        -2: "CAPACITY",
        -3: "ENCODING",
        -4: "IO",
        -5: "LOCKED",
        -6: "CORRUPT",
        -7: "APPEND_ROLLED_BACK",
        -8: "APPEND_UNCERTAIN",
        -9: "NOT_READY",
        -10: "PARTIAL_COMMIT",
    }

    def __init__(
        self,
        library_path: str | Path,
        log_path: str | Path,
        state,
        event_digest: str,
        event_count: int,
    ) -> None:
        library_path = Path(library_path)
        log_path = Path(log_path)

        if not library_path.is_absolute() or not library_path.is_file():
            raise NativeHistoryError(-1, "LIBRARY_PATH")

        if not log_path.is_absolute():
            raise NativeHistoryError(-1, "LOG_PATH")

        try:
            self._lib = C.CDLL(str(library_path))
        except OSError as exc:
            raise NativeHistoryError(-4, "LIBRARY_LOAD") from exc

        self.library_path = library_path
        self.log_path = log_path
        self._handle = C.c_void_p()

        self._bind()

        root, keepalive = _state_root_ctypes_view(
            state.state_root()
        )

        path = str(log_path).encode("utf-8")

        rc = self._lib.elpis_ecsc_runtime_session_open(
            path,
            len(path),
            C.byref(root),
            event_digest.encode("ascii"),
            event_count,
            C.byref(self._handle),
        )

        # Keep ctypes projection alive through the C call.  Native session
        # copies the validated seed, so it is not retained afterwards.
        _ = keepalive

        if rc != 0 or not self._handle:
            self._handle = C.c_void_p()
            raise NativeHistoryError(
                rc,
                "SESSION_OPEN:"
                + self._ERRORS.get(rc, "NATIVE"),
            )

    def _bind(self) -> None:
        vp = C.c_void_p

        self._lib.elpis_ecsc_runtime_session_open.argtypes = [
            C.c_char_p,
            C.c_size_t,
            C.POINTER(StateRoot),
            C.c_char_p,
            C.c_uint64,
            C.POINTER(vp),
        ]
        self._lib.elpis_ecsc_runtime_session_open.restype = C.c_int

        self._lib.elpis_ecsc_runtime_session_close.argtypes = [
            vp,
        ]
        self._lib.elpis_ecsc_runtime_session_close.restype = None

        self._lib.elpis_ecsc_runtime_session_record.argtypes = [
            vp,
            C.c_char_p,
            C.c_char_p,
            vp,
            C.c_size_t,
            C.POINTER(RuntimeRecordPlanResult),
        ]
        self._lib.elpis_ecsc_runtime_session_record.restype = C.c_int

        self._lib.elpis_ecsc_runtime_session_state_root_digest.argtypes = [
            vp,
            C.POINTER(C.c_char),
        ]
        self._lib.elpis_ecsc_runtime_session_state_root_digest.restype = C.c_int

        self._lib.elpis_ecsc_runtime_session_event_count.argtypes = [
            vp,
        ]
        self._lib.elpis_ecsc_runtime_session_event_count.restype = C.c_uint64

    @property
    def open(self) -> bool:
        return bool(self._handle)

    @property
    def event_count(self) -> int:
        if not self._handle:
            raise NativeHistoryError(-9, "EVENT_COUNT:CLOSED")
        return int(
            self._lib.elpis_ecsc_runtime_session_event_count(
                self._handle
            )
        )

    @property
    def state_root(self) -> str:
        if not self._handle:
            raise NativeHistoryError(-9, "STATE_ROOT:CLOSED")

        out = (C.c_char * 65)()

        rc = (
            self._lib
            .elpis_ecsc_runtime_session_state_root_digest(
                self._handle,
                out,
            )
        )

        if rc != 0:
            raise NativeHistoryError(
                rc,
                "STATE_ROOT:"
                + self._ERRORS.get(rc, "NATIVE"),
            )

        return _text(out)

    def record(
        self,
        sender_entity_id: str,
        history_entity_id: str,
        payload: bytes,
    ) -> RuntimeRecordPlanResult:
        if not self._handle:
            raise NativeHistoryError(-9, "RECORD:CLOSED")

        if type(payload) is not bytes or not payload:
            raise NativeHistoryError(-1, "RECORD:PAYLOAD")

        buf = (
            C.c_uint8 * len(payload)
        ).from_buffer_copy(payload)

        result = RuntimeRecordPlanResult()

        rc = self._lib.elpis_ecsc_runtime_session_record(
            self._handle,
            sender_entity_id.encode("ascii"),
            history_entity_id.encode("ascii"),
            C.cast(buf, C.c_void_p),
            len(payload),
            C.byref(result),
        )

        if rc != 0:
            raise NativeHistoryError(
                rc,
                "RECORD:"
                + self._ERRORS.get(rc, "NATIVE"),
            )

        return result

    def close(self) -> None:
        if self._handle:
            self._lib.elpis_ecsc_runtime_session_close(
                self._handle
            )
            self._handle = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
