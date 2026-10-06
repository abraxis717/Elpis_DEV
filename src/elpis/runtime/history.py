"""One durable receipt history for the composed system, over one ECS kernel.

The history is an ordinary ECS kernel history. At genesis it founds, in a
fixed order, one ``history`` entity and one recorder entity per subsystem
(``pipeline``, ``structure``, ``evolution``, ``inference``, ``ecs_g``).
A receipt is recorded as one message from the subsystem's recorder to the history entity,
so the sender is kernel-attributed and cannot be chosen by the record.

Recording is not admission. A record carries the digest of something its
owning subsystem already committed (or, for ingress, of a zero-authority
proposal the subsystem already published). Recording grants that thing no
authority, and the history never executes anything.

Records are idempotent: recording a record equal to one already in the history
returns the existing entry and appends nothing, so a replayed commit (for
example an ``ALREADY_COMMITTED`` publication) is recorded once.

Opening validates the whole event chain through the kernel (replay with
state-root verification). A history whose founding entities differ from the
fixed founding order is refused.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from elpis.ECS_C.entity import entity_id_from_founding, founding_record
from elpis.ECS_C.kernel import Kernel
from elpis.ECS_C.projection.contracts import ContextProjection, ProjectionRequest
from elpis.ECS_C.projection.kernel_adapter import project_kernel_history
from elpis.runtime.native_history import NativeHistoryError, NativeHistorySession

__all__ = (
    "HISTORY_GENESIS_LABEL", "RECORD_SCHEMA", "RECORDERS", "ROLES",
    "HistoryError", "ReceiptHistory", "ReceiptRecord", "RecordedReceipt",
)

HISTORY_GENESIS_LABEL = "elpis.runtime.history.v1"
RECORD_SCHEMA = "elpis.runtime.receipt-record.v1"

# Historical runtime-history topology admitted before ECS_G turn continuity.
# It remains an opening/migration authority only; new/opened histories converge
# to ROLES exactly.
_LEGACY_ROLES = (
    "history",
    "pipeline",
    "structure",
    "evolution",
    "inference",
)

# Canonical runtime-history topology.
ROLES = _LEGACY_ROLES + ("ecs_g",)
RECORDERS = ROLES[1:]

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_MAX_BINDINGS = 16
_MAX_VALUE_BYTES = 256


class HistoryError(RuntimeError):
    """Fail-closed history refusal with a stable machine code."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


@dataclass(frozen=True)
class ReceiptRecord:
    """What the history records: one subsystem digest plus named bindings."""

    subsystem: str
    kind: str
    digest: str
    bindings: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if self.subsystem not in RECORDERS:
            raise HistoryError("RECORD_SUBSYSTEM", repr(self.subsystem))
        if type(self.kind) is not str or not _NAME.match(self.kind):
            raise HistoryError("RECORD_KIND", repr(self.kind))
        if type(self.digest) is not str or not _DIGEST.match(self.digest):
            raise HistoryError("RECORD_DIGEST", repr(self.digest))
        if type(self.bindings) is not tuple or len(self.bindings) > _MAX_BINDINGS:
            raise HistoryError("RECORD_BINDINGS", "bounded tuple of (name, value) pairs")
        names = []
        for pair in self.bindings:
            if (type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str
                    or not _NAME.match(pair[0]) or type(pair[1]) is not str
                    or len(pair[1].encode("utf-8")) > _MAX_VALUE_BYTES):
                raise HistoryError("RECORD_BINDINGS", repr(pair))
            names.append(pair[0])
        if names != sorted(set(names)):
            raise HistoryError("RECORD_BINDINGS", "names must be unique and sorted")

    @classmethod
    def of(cls, subsystem: str, kind: str, digest: str, **bindings: str) -> "ReceiptRecord":
        return cls(subsystem, kind, digest, tuple(sorted(bindings.items())))

    def payload(self) -> bytes:
        return _canonical({"schema": RECORD_SCHEMA, "subsystem": self.subsystem, "kind": self.kind,
                           "digest": self.digest, "bindings": dict(self.bindings)})

    @classmethod
    def from_payload(cls, payload: bytes) -> "ReceiptRecord":
        try:
            value = json.loads(payload.decode("ascii"))
        except (UnicodeError, ValueError) as exc:
            raise HistoryError("RECORD_PAYLOAD", "not canonical JSON") from exc
        if (type(value) is not dict or set(value) != {"schema", "subsystem", "kind", "digest", "bindings"}
                or value["schema"] != RECORD_SCHEMA or type(value["bindings"]) is not dict):
            raise HistoryError("RECORD_PAYLOAD", "unexpected record shape")
        record = cls(value["subsystem"], value["kind"], value["digest"],
                     tuple(sorted(value["bindings"].items())))
        if record.payload() != payload:
            raise HistoryError("RECORD_PAYLOAD", "non-canonical encoding")
        return record


@dataclass(frozen=True)
class RecordedReceipt:
    """A record as it stands in the history: its message and enqueue event."""

    record: ReceiptRecord
    message_id: str
    event_index: int
    event_digest: str


class ReceiptHistory:
    """Owning handle on the durable history; not thread-shared beyond the kernel lock."""

    def __init__(
        self,
        storage_dir: str | Path,
        *,
        native_library: str | Path | None = None,
    ):
        storage_dir = Path(storage_dir)
        if not storage_dir.is_absolute():
            raise HistoryError("HISTORY_PATH", "storage directory must be absolute")
        self.storage_dir = storage_dir
        self._kernel = Kernel(str(storage_dir), genesis_label=HISTORY_GENESIS_LABEL)
        self._ids: dict[str, str] = {}
        self._ports = {}
        self._records: list[RecordedReceipt] = []
        self._record_index: dict[ReceiptRecord, RecordedReceipt] = {}
        self._native_library = (
            Path(native_library)
            if native_library is not None
            else None
        )
        self._native: NativeHistorySession | None = None

    def _recover_native_nonquiescent_prefix(self, kernel) -> bool:
        if self._native_library is None:
            return False

        before = tuple(kernel.events())

        if not before or before[-1]["event_kind"] != "MESSAGE_ENQUEUED":
            return False

        if len(before) >= 2 and before[-2]["event_kind"] == "MESSAGE_ENQUEUED":
            raise HistoryError(
                "NATIVE_HISTORY_RECOVERY",
                "multiple unresolved enqueue events are outside the runtime recovery contract",
            )

        try:
            kernel.run_until_quiescent()
        except Exception as exc:
            raise HistoryError(
                "NATIVE_HISTORY_RECOVERY",
                str(exc),
            ) from exc

        after = tuple(kernel.events())

        if (
            len(after) != len(before) + 1
            or after[:len(before)] != before
            or after[-1]["event_kind"] != "MESSAGE_PROCESSED"
        ):
            raise HistoryError(
                "NATIVE_HISTORY_RECOVERY",
                "canonical replay did not resolve exactly one enqueue-only prefix",
            )

        return True

    def _native_handoff(self) -> None:
        if self._native_library is None:
            return
        if self._native is not None:
            raise HistoryError("NATIVE_HISTORY_STATE", "already open")

        kernel = self._kernel
        state = kernel.state
        events = kernel.events()

        if not events:
            raise HistoryError(
                "NATIVE_HISTORY_STATE",
                "history has no committed head",
            )

        head = events[-1]["event_digest"]
        event_count = len(events)
        log_path = kernel.log_path

        # The Python kernel must relinquish flock ownership before the native
        # durable owner opens exactly the same events.log.
        kernel.close()
        self._ports = {}

        try:
            self._native = NativeHistorySession(
                self._native_library,
                log_path,
                state,
                head,
                event_count,
            )
        except NativeHistoryError as exc:
            raise HistoryError(
                "NATIVE_HISTORY_OPEN",
                str(exc),
            ) from exc

    def _python_read_window(self):
        """Temporarily return log ownership to canonical Python replay.

        This is a read-side/cold operation used by ``projection()``.  The
        committed record hot path never enters it.
        """
        if self._native is None:
            return None

        self._native.close()
        self._native = None

        try:
            self._kernel.open()
        except BaseException:
            raise

        return True

    def _close_python_read_window(self, active) -> None:
        if not active:
            return

        try:
            if self._native_library is None:
                return

            state = self._kernel.state
            events = self._kernel.events()

            if not events:
                raise HistoryError(
                    "NATIVE_HISTORY_STATE",
                    "history lost committed head",
                )

            head = events[-1]["event_digest"]
            count = len(events)
            log_path = self._kernel.log_path

        finally:
            self._kernel.close()

        try:
            self._native = NativeHistorySession(
                self._native_library,
                log_path,
                state,
                head,
                count,
            )
        except NativeHistoryError as exc:
            raise HistoryError(
                "NATIVE_HISTORY_REOPEN",
                str(exc),
            ) from exc

    # -- lifecycle ---------------------------------------------------------------
    def open(self) -> "ReceiptHistory":
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        kernel = self._kernel.open()
        try:
            genesis = kernel.state.genesis_digest

            expected = [
                entity_id_from_founding(
                    founding_record(i, role, genesis)
                )
                for i, role in enumerate(ROLES)
            ]

            expected_legacy = expected[:len(_LEGACY_ROLES)]
            existing = kernel.entity_ids()

            if not existing:
                # New history: materialize the current six-role authority.
                for role in ROLES:
                    kernel.found_entity(role)
                kernel.run_until_quiescent()

            elif sorted(existing) == sorted(expected_legacy):
                # Deterministic one-way compatibility migration.
                #
                # A valid legacy history is already fully replay-validated by
                # Kernel.open(). Add only the one missing fixed recorder using
                # normal ECS_C transitions. Activate it explicitly so migration
                # is exactly FOUND + ACTIVATE and cannot accidentally consume an
                # unrelated ready item.
                ecs_g = kernel.found_entity("ecs_g")

                if ecs_g != expected[-1]:
                    raise HistoryError(
                        "FOREIGN_HISTORY",
                        "ecs_g founding identity differs from runtime authority",
                    )

                kernel.activate(ecs_g)

            # Each expected ID is derived from (founding index, role, genesis).
            # Set equality therefore proves both the founding order and labels.
            # Any six-role near-match, extra role, missing role or different
            # founding order is foreign and is never repaired.
            if sorted(kernel.entity_ids()) != sorted(expected):
                raise HistoryError(
                    "FOREIGN_HISTORY",
                    "founding entities differ from the runtime roles",
                )

            self._ids = dict(zip(ROLES, expected))
            self._ports = {
                role: kernel.entity_port(self._ids[role])
                for role in RECORDERS
            }

            # Explicit native-writer reopen may encounter the exact
            # enqueue-only durable prefix left by PARTIAL_COMMIT or a
            # resolved APPEND_UNCERTAIN outcome. Canonical Python replay
            # finishes that already-durable message before writer handoff.
            self._recover_native_nonquiescent_prefix(kernel)

            self._records = self._read_records()
            self._record_index = {
                item.record: item
                for item in self._records
            }
            self._native_handoff()
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        if self._native is not None:
            self._native.close()
            self._native = None
        self._kernel.close()
        self._ports = {}

    def __enter__(self) -> "ReceiptHistory":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- reads -------------------------------------------------------------------
    @property
    def state_root(self) -> str:
        if self._native is not None:
            try:
                return self._native.state_root
            except NativeHistoryError as exc:
                raise HistoryError(
                    "NATIVE_HISTORY_STATE_ROOT",
                    str(exc),
                ) from exc
        return self._kernel.state_root_digest()

    @property
    def history_entity(self) -> str:
        return self._ids["history"]

    def records(self) -> tuple[RecordedReceipt, ...]:
        return tuple(self._records)

    def projection(self, request: ProjectionRequest | None = None) -> ContextProjection:
        """Read-only projection of the committed history (records by default)."""
        if request is None:
            request = ProjectionRequest(receiver_ids=(self.history_entity,),
                                        event_kinds=("MESSAGE_ENQUEUED",))

        read_window = self._python_read_window()

        try:
            return project_kernel_history(
                self._kernel,
                request,
            )
        finally:
            self._close_python_read_window(
                read_window
            )

    def _read_records(self) -> list[RecordedReceipt]:
        senders = {entity: role for role, entity in self._ids.items() if role in RECORDERS}
        records = []
        for event in self._kernel.events():
            if event["event_kind"] != "MESSAGE_ENQUEUED":
                continue
            envelope = event["payload"]["envelope"]
            if envelope["receiver_entity_id"] != self.history_entity:
                raise HistoryError("FOREIGN_MESSAGE", "message not addressed to the history entity")
            role = senders.get(envelope["sender_entity_id"])
            if role is None:
                raise HistoryError("FOREIGN_MESSAGE", "sender is not a recorder entity")
            record = ReceiptRecord.from_payload(bytes.fromhex(envelope["payload_hex"]))
            if record.subsystem != role:
                raise HistoryError("RECORD_SENDER_MISMATCH", f"{record.subsystem} recorded by {role}")
            records.append(RecordedReceipt(record, envelope["message_id"], event["event_index"],
                                           event["event_digest"]))
        return records

    # -- the one write -----------------------------------------------------------
    def record(self, record: ReceiptRecord) -> RecordedReceipt:
        if type(record) is not ReceiptRecord:
            raise HistoryError(
                "RECORD_TYPE",
                type(record).__name__,
            )

        existing = self._record_index.get(record)

        if existing is not None:
            return existing

        if self._native is not None:
            try:
                result = self._native.record(
                    self._ids[record.subsystem],
                    self.history_entity,
                    record.payload(),
                )
            except NativeHistoryError as exc:
                raise HistoryError(
                    "NATIVE_HISTORY_RECORD",
                    str(exc),
                ) from exc

            recorded = RecordedReceipt(
                record=record,
                message_id=bytes(
                    result.message_id
                ).split(b"\\x00", 1)[0].decode("ascii"),
                event_index=int(
                    result.enqueue_event_index
                ),
                event_digest=bytes(
                    result.enqueue_event_digest
                ).split(b"\\x00", 1)[0].decode("ascii"),
            )

            self._records.append(recorded)
            self._record_index[record] = recorded

            return recorded

        # Once runtime history ownership has been handed to the native
        # session there is deliberately no Python write fallback.  If the
        # native owner is absent, this object is not an authorized writer.
        raise HistoryError("HISTORY_CLOSED")
