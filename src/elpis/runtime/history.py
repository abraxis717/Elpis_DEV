"""One durable receipt history for the composed system, over one ECS kernel.

The history is an ordinary ECS kernel history. At genesis it founds, in a
fixed order, one ``history`` entity and one recorder entity per subsystem
(``pipeline``, ``structure``, ``evolution``, ``inference``). A receipt is
recorded as one message from the subsystem's recorder to the history entity,
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

from elpis.ecs.entity import entity_id_from_founding, founding_record
from elpis.ecs.kernel import Kernel
from elpis.ecs.projection.contracts import ContextProjection, ProjectionRequest
from elpis.ecs.projection.kernel_adapter import project_kernel_history

__all__ = (
    "HISTORY_GENESIS_LABEL", "RECORD_SCHEMA", "RECORDERS", "ROLES",
    "HistoryError", "ReceiptHistory", "ReceiptRecord", "RecordedReceipt",
)

HISTORY_GENESIS_LABEL = "elpis.runtime.history.v1"
RECORD_SCHEMA = "elpis.runtime.receipt-record.v1"
ROLES = ("history", "pipeline", "structure", "evolution", "inference")
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

    def __init__(self, storage_dir: str | Path):
        storage_dir = Path(storage_dir)
        if not storage_dir.is_absolute():
            raise HistoryError("HISTORY_PATH", "storage directory must be absolute")
        self.storage_dir = storage_dir
        self._kernel = Kernel(str(storage_dir), genesis_label=HISTORY_GENESIS_LABEL)
        self._ids: dict[str, str] = {}
        self._ports = {}
        self._records: list[RecordedReceipt] = []

    # -- lifecycle ---------------------------------------------------------------
    def open(self) -> "ReceiptHistory":
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        kernel = self._kernel.open()
        try:
            genesis = kernel.state.genesis_digest
            expected = [entity_id_from_founding(founding_record(i, role, genesis))
                        for i, role in enumerate(ROLES)]
            if not kernel.entity_ids():
                for role in ROLES:
                    kernel.found_entity(role)
                kernel.run_until_quiescent()
            # Each expected id is derived from (founding index, role, genesis), so set
            # equality proves both the founding order and the labels.
            if sorted(kernel.entity_ids()) != sorted(expected):
                raise HistoryError("FOREIGN_HISTORY", "founding entities differ from the runtime roles")
            self._ids = dict(zip(ROLES, expected))
            self._ports = {role: kernel.entity_port(self._ids[role]) for role in RECORDERS}
            self._records = self._read_records()
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        self._kernel.close()
        self._ports = {}

    def __enter__(self) -> "ReceiptHistory":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- reads -------------------------------------------------------------------
    @property
    def state_root(self) -> str:
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
        return project_kernel_history(self._kernel, request)

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
            raise HistoryError("RECORD_TYPE", type(record).__name__)
        for existing in self._records:
            if existing.record == record:
                return existing
        if not self._ports:
            raise HistoryError("HISTORY_CLOSED")
        message_id = self._ports[record.subsystem].propose(self.history_entity, record.payload())
        self._kernel.run_until_quiescent()
        self._records = self._read_records()
        recorded = self._records[-1]
        if recorded.message_id != message_id or recorded.record != record:
            raise HistoryError("RECORD_NOT_DURABLE", message_id)
        return recorded
