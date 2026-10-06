"""One durable, BOUNDED receipt history for the composed system, over ECS_C.

The history is an ordinary ECS kernel history. At genesis it founds, in a
fixed order, one ``history`` entity and one recorder entity per subsystem
(``pipeline``, ``structure``, ``evolution``, ``inference``, ``ecs_g``).
A receipt is recorded as one message from the subsystem's recorder to the
history entity, so the sender is kernel-attributed and cannot be chosen by the
record.

Recording is not admission. A record carries the digest of something its
owning subsystem already committed (or, for ingress, of a zero-authority
proposal the subsystem already published). Recording grants that thing no
authority, and the history never executes anything.

Bounded storage
---------------
The history lives in one directory owned through a finite
:class:`RuntimeHistoryPolicy` (no unlimited setting exists). Its durable form
is one published GENERATION (see :mod:`elpis.ECS_C.generations`): a verified
state-bearing compaction checkpoint (the retired prefix) plus one active
segment holding the events after it, with their original global indices.
When a receipt would not fit the active segment, native ECS_C refuses it
before any mutation (``SEGMENT_FULL``); the history then compacts on the cold
path (checkpoint the current state, publish a new generation crash-safely,
drop the old one) and retries that exact receipt once. Nothing is archived:
the directory size is bounded by ``policy.max_directory_bytes``.

Retention
---------
``retention_floor`` is the global index of the first retained event. Only the
retained window (the active segment) exists as events. ``records()`` returns
the complete record list only while the floor is 0 and otherwise refuses with
``HISTORY_BELOW_RETENTION_FLOOR``; ``retained_records()`` returns the window.
Exact duplicate detection (``record`` returning the existing entry and
appending nothing) is guaranteed ONLY within the retained window. A record
equal to one that was retired below the floor is recorded again.

Opening rebuilds the base state from the verified checkpoint and replays only
the bounded segment. The ECS_G cognition lineage is carried across
compaction as a fixed-size :class:`~elpis.runtime.continuity.CognitionContinuity`
summary in the checkpoint. A history whose founding entities differ from the
fixed founding order is refused.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re

from elpis.ECS_C.bus import DEFAULT_MAILBOX_CAPACITY
from elpis.ECS_C.compaction import MAX_CHECKPOINT_BYTES, CompactionCheckpoint
from elpis.ECS_C.entity import entity_id_from_founding, founding_record
from elpis.ECS_C.errors import (
    CorruptCompactionCheckpointError,
    EcsError,
    GenerationError,
    RetentionFloorError,
    StorageCapacityError,
)
from elpis.ECS_C.generations import (
    MAX_MANIFEST_BYTES,
    DirectoryLock,
    GenerationManifest,
    GenerationStore,
)
from elpis.ECS_C.kernel import Kernel
from elpis.ECS_C.persistence import GENESIS_PREV_DIGEST, genesis_descriptor_digest
from elpis.ECS_C.projection.contracts import ContextProjection, ProjectionError, ProjectionRequest
from elpis.ECS_C.projection.kernel_adapter import project_kernel_history
from elpis.ECS_C.replay import initial_state
from elpis.ECS_C.scheduler import SCHEDULER_V2
from elpis.runtime.continuity import CognitionContinuity, ContinuityError
from elpis.runtime.native_history import (
    SEGMENT_FULL,
    SEGMENT_MISMATCH,
    NativeHistoryError,
    NativeHistorySession,
)

__all__ = (
    "HISTORY_GENESIS_LABEL", "RECORD_SCHEMA", "RECORDERS", "ROLES",
    "HistoryError", "ReceiptHistory", "ReceiptRecord", "RecordedReceipt",
    "RuntimeHistoryPolicy",
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




_KiB = 1024
_MiB = 1024 * _KiB

# Cold-path appends made at open (role founding/upgrade, enqueue-only
# recovery) must always fit: open compacts first when less than this headroom
# remains in the active segment.
OPEN_RESERVE_BYTES = 32 * _KiB
OPEN_RESERVE_FRAMES = 16

_HISTORY_SUMMARY_SCHEMA = "elpis.runtime.history-summary.v1"
_LEGACY_LOG = "events.log"
_LEGACY_MARKERS = ("checkpoint.bin", "checkpoint.bin.tmp")


@dataclass(frozen=True)
class RuntimeHistoryPolicy:
    """Finite storage policy of one runtime history. There is no unlimited value.

    * ``max_segment_bytes`` / ``max_segment_events``: hard bounds of the
      active segment, enforced natively BEFORE every append (planned sizes).
      They also bound the retained window and the resident record index.
    * ``max_checkpoint_bytes``: hard bound of one compaction checkpoint.
    * ``max_directory_bytes``: hard bound of every byte in the directory,
      including the compaction headroom (one in-flight checkpoint and one
      in-flight manifest). It must be at least ``required_directory_bytes``.
    """

    max_segment_bytes: int = 8 * _MiB
    max_segment_events: int = 32768
    max_checkpoint_bytes: int = 64 * _KiB
    max_directory_bytes: int = 9 * _MiB

    MIN_SEGMENT_BYTES = 64 * _KiB
    MIN_SEGMENT_EVENTS = 64
    MIN_CHECKPOINT_BYTES = 16 * _KiB
    MAX_BOUND = 1 << 48

    def __post_init__(self):
        for name, low, high in (
            ("max_segment_bytes", self.MIN_SEGMENT_BYTES, self.MAX_BOUND),
            ("max_segment_events", self.MIN_SEGMENT_EVENTS, self.MAX_BOUND),
            ("max_checkpoint_bytes", self.MIN_CHECKPOINT_BYTES, MAX_CHECKPOINT_BYTES),
            ("max_directory_bytes", 1, self.MAX_BOUND),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise HistoryError("HISTORY_POLICY", f"{name} must be a finite int in [{low}, {high}]")
        if self.max_directory_bytes < self.required_directory_bytes:
            raise HistoryError(
                "HISTORY_POLICY",
                f"max_directory_bytes must cover one segment plus compaction headroom "
                f"({self.required_directory_bytes} bytes)",
            )

    @property
    def required_directory_bytes(self) -> int:
        """Worst-case directory bytes: segment + two checkpoints + two manifests."""
        return self.max_segment_bytes + 2 * self.max_checkpoint_bytes + 2 * MAX_MANIFEST_BYTES


def _summary(continuity: CognitionContinuity, receipts_retired: int) -> dict:
    return {"schema": _HISTORY_SUMMARY_SCHEMA, "cognition": continuity.to_dict(),
            "receipts_retired": receipts_retired}


def _parse_summary(extension) -> tuple[CognitionContinuity, int]:
    try:
        if (type(extension) is not dict or set(extension) != {"schema", "cognition", "receipts_retired"}
                or extension["schema"] != _HISTORY_SUMMARY_SCHEMA):
            raise ContinuityError("CONTINUITY_SUMMARY_INVALID", "summary shape")
        retired = extension["receipts_retired"]
        if type(retired) is not int or retired < 0:
            raise ContinuityError("CONTINUITY_SUMMARY_INVALID", "receipts_retired")
        return CognitionContinuity.from_dict(extension["cognition"]), retired
    except ContinuityError as exc:
        raise HistoryError("HISTORY_CHECKPOINT_INVALID", str(exc)) from exc


_STORAGE_ERRORS = (StorageCapacityError, CorruptCompactionCheckpointError,
                   RetentionFloorError, GenerationError)


def _translate(exc: BaseException, default: str) -> "HistoryError":
    """Map ECS_C storage failures to the runtime's stable history codes."""
    text = str(exc)
    if isinstance(exc, HistoryError):
        return exc
    if isinstance(exc, StorageCapacityError):
        code = "HISTORY_STORAGE_CAPACITY"
    elif isinstance(exc, CorruptCompactionCheckpointError):
        code = "HISTORY_CHECKPOINT_INVALID"
    elif isinstance(exc, RetentionFloorError):
        code = "HISTORY_BELOW_RETENTION_FLOOR"
    elif isinstance(exc, GenerationError):
        if text.startswith("STORE_LOCKED"):
            code = "HISTORY_LOCKED"
        elif text.startswith("HISTORY_COMPACTION_FAILED"):
            code = "HISTORY_COMPACTION_FAILED"
        else:
            code = "HISTORY_GENERATION_MISMATCH"
    else:
        code = default
    return HistoryError(code, text)


class ReceiptHistory:
    """Owning handle on the bounded durable history (single owner per directory)."""

    def __init__(
        self,
        storage_dir: str | Path,
        *,
        native_library: str | Path | None = None,
        policy: RuntimeHistoryPolicy | None = None,
    ):
        storage_dir = Path(storage_dir)
        if not storage_dir.is_absolute():
            raise HistoryError("HISTORY_PATH", "storage directory must be absolute")
        if policy is None:
            policy = RuntimeHistoryPolicy()
        if type(policy) is not RuntimeHistoryPolicy:
            raise HistoryError("HISTORY_POLICY", "policy must be a RuntimeHistoryPolicy")
        self.storage_dir = storage_dir
        self.policy = policy
        self._lock = DirectoryLock(storage_dir)
        self._store = GenerationStore(
            storage_dir,
            genesis_label=HISTORY_GENESIS_LABEL,
            mailbox_capacity=DEFAULT_MAILBOX_CAPACITY,
            max_checkpoint_bytes=policy.max_checkpoint_bytes,
            max_directory_bytes=policy.max_directory_bytes,
        )
        self._manifest: GenerationManifest | None = None
        self._base: CompactionCheckpoint | None = None
        # The Python ECS_C kernel of the CURRENT generation: the canonical
        # cold-path replay/verification authority. It is closed whenever the
        # native session owns the segment.
        self._kernel: Kernel | None = None
        self._ids: dict[str, str] = {}
        self._ports = {}
        # Retained window only (bounded by the segment policy).
        self._records: list[RecordedReceipt] = []
        self._record_index: dict[ReceiptRecord, RecordedReceipt] = {}
        self._continuity = CognitionContinuity()
        self._receipts_retired = 0
        self._native_library = (
            Path(native_library)
            if native_library is not None
            else None
        )
        self._native: NativeHistorySession | None = None

    # -- generation plumbing (cold path) ------------------------------------------
    def _generation_kernel(self, manifest: GenerationManifest, base: CompactionCheckpoint) -> Kernel:
        return Kernel(
            str(self.storage_dir),
            genesis_label=HISTORY_GENESIS_LABEL,
            log_path=str(self._store.segment_path(manifest)),
            base=base,
            max_log_bytes=self.policy.max_segment_bytes,
        )

    def _expected_ids(self, genesis: str) -> list[str]:
        return [entity_id_from_founding(founding_record(i, role, genesis)) for i, role in enumerate(ROLES)]

    def _publish(self, checkpoint: CompactionCheckpoint, *, link_segment_from: Path | None = None,
                 exclude_legacy: bool = False) -> GenerationManifest:
        try:
            return self._store.publish(checkpoint, previous=self._manifest,
                                       link_segment_from=link_segment_from,
                                       exclude_legacy=exclude_legacy)
        except (EcsError, OSError) as exc:
            raise _translate(exc, "HISTORY_COMPACTION_FAILED") from exc

    def _initialize(self) -> GenerationManifest:
        """Publish generation 1 of a new history: the genesis-empty base."""
        genesis = genesis_descriptor_digest(HISTORY_GENESIS_LABEL, SCHEDULER_V2)
        state = initial_state(genesis, DEFAULT_MAILBOX_CAPACITY, SCHEDULER_V2)
        checkpoint = CompactionCheckpoint.from_state(
            state,
            genesis_label=HISTORY_GENESIS_LABEL,
            event_count=0,
            head_event_digest=GENESIS_PREV_DIGEST,
            extension=_summary(CognitionContinuity(), 0),
            max_bytes=self.policy.max_checkpoint_bytes,
        )
        return self._publish(checkpoint)

    def _legacy_entities_admissible(self, kernel: Kernel) -> None:
        genesis = kernel.state.genesis_digest
        expected = self._expected_ids(genesis)
        existing = sorted(kernel.entity_ids())
        if existing not in ([], sorted(expected[:len(_LEGACY_ROLES)]), sorted(expected)):
            raise HistoryError("FOREIGN_HISTORY", "founding entities differ from the runtime roles")

    def _migrate_legacy(self) -> GenerationManifest:
        """Deterministic one-time migration of the pre-compaction single-log layout.

        The legacy ``events.log`` (+ ``checkpoint.bin`` marker) is fully
        validated once by the original whole-log authority. Its bytes are
        never reinterpreted: either the log becomes generation 1's segment
        verbatim (hard link; base = genesis, floor 0), or, when it exceeds the
        segment policy, its validated head state becomes generation 1's
        checkpoint with an empty segment. Only after the manifest is published
        are the legacy names removed; the legacy log is never replayed again.
        """
        legacy_log = self.storage_dir / _LEGACY_LOG
        kernel = Kernel(str(self.storage_dir), genesis_label=HISTORY_GENESIS_LABEL)
        try:
            kernel.open()
        except EcsError as exc:
            raise HistoryError("HISTORY_LEGACY_MIGRATION_FAILED", str(exc)) from exc
        try:
            self._legacy_entities_admissible(kernel)
            snapshot = kernel.snapshot()
            count = snapshot["event_count"]
            size = legacy_log.stat().st_size
            fits = (size <= self.policy.max_segment_bytes - OPEN_RESERVE_BYTES
                    and count <= self.policy.max_segment_events - OPEN_RESERVE_FRAMES)
            if count == 0:
                link, checkpoint = None, None
            elif fits:
                genesis = kernel.state.genesis_digest
                checkpoint = CompactionCheckpoint.from_state(
                    initial_state(genesis, DEFAULT_MAILBOX_CAPACITY, kernel.scheduler_protocol),
                    genesis_label=HISTORY_GENESIS_LABEL,
                    event_count=0,
                    head_event_digest=GENESIS_PREV_DIGEST,
                    extension=_summary(CognitionContinuity(), 0),
                    max_bytes=self.policy.max_checkpoint_bytes,
                )
                link = legacy_log
            else:
                if kernel.ready_items():
                    raise HistoryError(
                        "HISTORY_LEGACY_MIGRATION_FAILED",
                        "a non-quiescent legacy history larger than the segment policy cannot be compacted",
                    )
                continuity, receipts = CognitionContinuity(), 0
                ids = {entity: role for role, entity in zip(ROLES, self._expected_ids(kernel.state.genesis_digest))}
                for event in kernel.events():
                    record = self._record_from_event(event, ids)
                    if record is not None:
                        receipts += 1
                        continuity = continuity.fold(record.record)
                checkpoint = CompactionCheckpoint.from_state(
                    kernel.state,
                    genesis_label=HISTORY_GENESIS_LABEL,
                    event_count=count,
                    head_event_digest=snapshot["event_digest"],
                    extension=_summary(continuity, receipts),
                    max_bytes=self.policy.max_checkpoint_bytes,
                )
                link = None
        except HistoryError:
            raise
        except EcsError as exc:
            raise HistoryError("HISTORY_LEGACY_MIGRATION_FAILED", str(exc)) from exc
        finally:
            kernel.close()

        try:
            if checkpoint is None:
                manifest = self._initialize()
            else:
                manifest = self._store.publish(checkpoint, previous=None,
                                               link_segment_from=link, exclude_legacy=True)
        except (EcsError, OSError) as exc:
            raise HistoryError("HISTORY_LEGACY_MIGRATION_FAILED", str(exc)) from exc
        self._remove_legacy(manifest)
        return manifest

    def _remove_legacy(self, manifest: GenerationManifest) -> None:
        """Remove legacy names beside a PUBLISHED generation (never replayed)."""
        removed = False
        for name in _LEGACY_MARKERS:
            path = self.storage_dir / name
            if path.exists():
                path.unlink()
                removed = True
        legacy_log = self.storage_dir / _LEGACY_LOG
        if legacy_log.exists():
            segment = self._store.segment_path(manifest)
            same = segment.exists() and os.path.samefile(legacy_log, segment)
            if not same and not (manifest.generation == 1 and manifest.base_event_count > 0):
                raise HistoryError(
                    "HISTORY_LEGACY_MIGRATION_FAILED",
                    "a legacy events.log exists beside a published generation it was not migrated into",
                )
            legacy_log.unlink()
            removed = True
        if removed:
            fd = os.open(self.storage_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def _establish_roles(self, kernel: Kernel) -> None:
        genesis = kernel.state.genesis_digest
        expected = self._expected_ids(genesis)
        expected_legacy = expected[:len(_LEGACY_ROLES)]
        existing = kernel.entity_ids()

        if not existing:
            # New history: materialize the current six-role authority.
            for role in ROLES:
                kernel.found_entity(role)
            kernel.run_until_quiescent()

        elif sorted(existing) == sorted(expected_legacy):
            # Deterministic one-way compatibility migration of the role set:
            # add only the one missing fixed recorder using normal ECS_C
            # transitions, exactly FOUND + ACTIVATE.
            ecs_g = kernel.found_entity("ecs_g")
            if ecs_g != expected[-1]:
                raise HistoryError(
                    "FOREIGN_HISTORY",
                    "ecs_g founding identity differs from runtime authority",
                )
            kernel.activate(ecs_g)

        # Each expected ID is derived from (founding index, role, genesis).
        # Set equality therefore proves both the founding order and labels.
        if sorted(kernel.entity_ids()) != sorted(expected):
            raise HistoryError(
                "FOREIGN_HISTORY",
                "founding entities differ from the runtime roles",
            )
        self._ids = dict(zip(ROLES, expected))

    def _recover_native_nonquiescent_prefix(self, kernel) -> bool:
        if self._native_library is None:
            return False

        before = tuple(kernel.retained_events())

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

        after = tuple(kernel.retained_events())

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

    def _record_from_event(self, event, senders) -> RecordedReceipt | None:
        if event["event_kind"] != "MESSAGE_ENQUEUED":
            return None
        envelope = event["payload"]["envelope"]
        history = next(entity for entity, role in senders.items() if role == "history")
        if envelope["receiver_entity_id"] != history:
            raise HistoryError("FOREIGN_MESSAGE", "message not addressed to the history entity")
        role = senders.get(envelope["sender_entity_id"])
        if role is None or role == "history":
            raise HistoryError("FOREIGN_MESSAGE", "sender is not a recorder entity")
        record = ReceiptRecord.from_payload(bytes.fromhex(envelope["payload_hex"]))
        if record.subsystem != role:
            raise HistoryError("RECORD_SENDER_MISMATCH", f"{record.subsystem} recorded by {role}")
        return RecordedReceipt(record, envelope["message_id"], event["event_index"], event["event_digest"])

    def _load_retained(self, kernel: Kernel) -> None:
        """Rebuild the retained window and fold the lineage over it (streamed)."""
        continuity, retired = _parse_summary(self._base.extension)
        senders = {entity: role for role, entity in self._ids.items()}
        records: list[RecordedReceipt] = []

        def visit(event):
            item = self._record_from_event(event, senders)
            if item is not None:
                records.append(item)

        kernel.scan_retained_events(visit)
        for item in records:
            continuity = continuity.fold(item.record)
        self._records = records
        self._record_index = {item.record: item for item in records}
        self._continuity = continuity
        self._receipts_retired = retired

    def _needs_open_compaction(self, kernel: Kernel) -> bool:
        """Compact at open only when open itself could not proceed otherwise.

        Normal reopen has no side effects: a nearly full segment is compacted
        lazily by the first receipt that does not fit (native SEGMENT_FULL).
        Open compacts only if the segment already exceeds the policy (the
        policy was lowered) or if the one-time legacy role upgrade must append
        and the segment lacks the open reserve.
        """
        snapshot = kernel.snapshot()
        frames = snapshot["event_count"] - kernel.retention_floor
        size = kernel.segment_bytes()
        if size > self.policy.max_segment_bytes or frames > self.policy.max_segment_events:
            return True
        expected = self._expected_ids(kernel.state.genesis_digest)
        upgrade = sorted(kernel.entity_ids()) == sorted(expected[:len(_LEGACY_ROLES)])
        low = (size > self.policy.max_segment_bytes - OPEN_RESERVE_BYTES
               or frames > self.policy.max_segment_events - OPEN_RESERVE_FRAMES)
        return upgrade and low

    def _compact(self) -> None:
        """Checkpoint the open kernel's quiescent state and switch generations.

        Precondition: the Python kernel of the current generation is open and
        the native session is closed. On success the new generation's kernel
        is open (empty segment). On failure the history is left closed; the
        previous generation stays authoritative unless publication completed.
        """
        kernel = self._kernel
        if kernel.ready_items():
            raise HistoryError("HISTORY_COMPACTION_REQUIRED",
                               "segment is full and the history is not at a quiescent boundary")
        snapshot = kernel.snapshot()
        retained = len(self._records)
        try:
            checkpoint = CompactionCheckpoint.from_state(
                kernel.state,
                genesis_label=HISTORY_GENESIS_LABEL,
                event_count=snapshot["event_count"],
                head_event_digest=snapshot["event_digest"],
                extension=_summary(self._continuity, self._receipts_retired + retained),
                max_bytes=self.policy.max_checkpoint_bytes,
            )
        except EcsError as exc:
            raise _translate(exc, "HISTORY_COMPACTION_FAILED") from exc
        kernel.close()
        self._ports = {}
        manifest = self._publish(checkpoint)
        self._manifest, self._base = manifest, checkpoint
        self._records, self._record_index = [], {}
        self._receipts_retired += retained
        self._kernel = self._generation_kernel(manifest, checkpoint)
        try:
            self._kernel.open()
        except _STORAGE_ERRORS as exc:
            raise _translate(exc, "HISTORY_GENERATION_MISMATCH") from exc

    def _activate(self, manifest: GenerationManifest) -> None:
        """Open one published generation: bounded replay, roles, recovery, handoff."""
        try:
            base = self._store.load_checkpoint(manifest)
        except (EcsError, OSError) as exc:
            raise _translate(exc, "HISTORY_GENERATION_MISMATCH") from exc
        self._manifest, self._base = manifest, base
        _parse_summary(base.extension)
        kernel = self._generation_kernel(manifest, base)
        self._kernel = kernel
        try:
            kernel.open()
        except _STORAGE_ERRORS as exc:
            raise _translate(exc, "HISTORY_GENERATION_MISMATCH") from exc

        # Explicit native-writer reopen may encounter the exact enqueue-only
        # durable prefix left by PARTIAL_COMMIT or a resolved APPEND_UNCERTAIN
        # outcome. Canonical Python replay finishes that already-durable
        # message (it always fits: native admits an enqueue only together with
        # room for its processed event).
        self._recover_native_nonquiescent_prefix(kernel)

        if self._needs_open_compaction(kernel):
            if kernel.ready_items():
                raise HistoryError("HISTORY_COMPACTION_REQUIRED",
                                   "segment is over policy at a non-quiescent boundary")
            # Retired-to-be records must enter the lineage summary first.
            self._establish_roles_if_complete(kernel)
            self._load_retained(kernel)
            self._compact()
            kernel = self._kernel

        try:
            self._establish_roles(kernel)
        except _STORAGE_ERRORS as exc:
            raise _translate(exc, "HISTORY_STORAGE_CAPACITY") from exc
        self._load_retained(kernel)

    def _establish_roles_if_complete(self, kernel: Kernel) -> None:
        expected = self._expected_ids(kernel.state.genesis_digest)
        existing = sorted(kernel.entity_ids())
        if existing not in (sorted(expected), sorted(expected[:len(_LEGACY_ROLES)])):
            raise HistoryError("FOREIGN_HISTORY", "founding entities differ from the runtime roles")
        self._ids = dict(zip(ROLES, expected))

    def _native_handoff(self) -> None:
        if self._native_library is None:
            return
        if self._native is not None:
            raise HistoryError("NATIVE_HISTORY_STATE", "already open")

        kernel = self._kernel
        state = kernel.state
        snapshot = kernel.snapshot()
        if snapshot["event_count"] == 0:
            raise HistoryError("NATIVE_HISTORY_STATE", "history has no committed head")
        segment = Path(kernel.log_path)

        # The Python kernel must relinquish the segment flock before the native
        # durable owner opens exactly the same segment.
        kernel.close()
        self._ports = {}

        try:
            self._native = NativeHistorySession(
                self._native_library,
                segment,
                state,
                snapshot["event_digest"],
                snapshot["event_count"],
                segment_base=self._base.event_count,
                max_segment_bytes=self.policy.max_segment_bytes,
                max_segment_frames=self.policy.max_segment_events,
            )
        except NativeHistoryError as exc:
            if exc.code == SEGMENT_MISMATCH:
                code = "HISTORY_NATIVE_SEGMENT_MISMATCH"
            elif exc.code == SEGMENT_FULL:
                code = "HISTORY_COMPACTION_REQUIRED"
            else:
                code = "NATIVE_HISTORY_OPEN"
            raise HistoryError(code, str(exc)) from exc

    def _python_read_window(self):
        """Temporarily return segment ownership to canonical Python replay.

        This is a read-side/cold operation used by ``projection()`` and
        compaction. The committed record hot path never enters it. The replay
        is bounded: checkpoint base plus the active segment only.
        """
        if self._native is None:
            return None

        self._native.close()
        self._native = None
        try:
            self._kernel.open()
        except _STORAGE_ERRORS as exc:
            raise _translate(exc, "HISTORY_GENERATION_MISMATCH") from exc
        return True

    def _close_python_read_window(self, active) -> None:
        if not active:
            return
        self._native_handoff()

    # -- lifecycle ---------------------------------------------------------------
    def open(self) -> "ReceiptHistory":
        created = not self.storage_dir.exists()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        if created:
            for directory in (self.storage_dir.parent, self.storage_dir):
                fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        try:
            try:
                self._lock.acquire()
                manifest = self._store.read_manifest()
                entries = self._store.entries()
            except (EcsError, OSError) as exc:
                raise _translate(exc, "HISTORY_GENERATION_MISMATCH") from exc
            if manifest is None:
                try:
                    self._store.remove_unpublished(None)
                except (EcsError, OSError) as exc:
                    raise _translate(exc, "HISTORY_GENERATION_MISMATCH") from exc
                if _LEGACY_LOG in entries["legacy"]:
                    manifest = self._migrate_legacy()
                else:
                    for name in _LEGACY_MARKERS:
                        (self.storage_dir / name).unlink(missing_ok=True)
                    manifest = self._initialize()
            else:
                if entries["legacy"]:
                    self._remove_legacy(manifest)
                try:
                    self._store.remove_unpublished(manifest)
                except (EcsError, OSError) as exc:
                    raise _translate(exc, "HISTORY_GENERATION_MISMATCH") from exc
            self._activate(manifest)
            if self._native_library is not None:
                self._native_handoff()
            else:
                self._ports = {
                    role: self._kernel.entity_port(self._ids[role])
                    for role in RECORDERS
                }
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        if self._native is not None:
            self._native.close()
            self._native = None
        if self._kernel is not None:
            self._kernel.close()
        self._ports = {}
        self._lock.release()

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
        if self._kernel is None:
            raise HistoryError("HISTORY_CLOSED")
        return self._kernel.state_root_digest()

    @property
    def history_entity(self) -> str:
        return self._ids["history"]

    @property
    def retention_floor(self) -> int:
        """Global index of the first retained event (0 until the first compaction)."""
        if self._base is None:
            raise HistoryError("HISTORY_CLOSED")
        return self._base.event_count

    @property
    def generation(self) -> int:
        if self._manifest is None:
            raise HistoryError("HISTORY_CLOSED")
        return self._manifest.generation

    @property
    def segment_path(self) -> Path:
        if self._manifest is None:
            raise HistoryError("HISTORY_CLOSED")
        return self._store.segment_path(self._manifest)

    @property
    def event_count(self) -> int:
        """GLOBAL committed event count (== logical clock)."""
        if self._native is not None:
            return self._native.event_count
        if self._kernel is None:
            raise HistoryError("HISTORY_CLOSED")
        return self._kernel.snapshot()["event_count"]

    def storage_bytes(self) -> int:
        """Logical bytes currently held by the history directory."""
        return self._store.usage_bytes()

    @property
    def receipt_count(self) -> int:
        """Lifetime receipts (retired below the floor + retained)."""
        return self._receipts_retired + len(self._records)

    @property
    def continuity(self) -> CognitionContinuity:
        return self._continuity

    def cognition_tip(self) -> str | None:
        """Tip of the durable ECS_G K1 lineage (None if unanchored)."""
        try:
            return self._continuity.require_tip()
        except ContinuityError as exc:
            raise HistoryError(exc.code, str(exc)) from exc

    def records(self) -> tuple[RecordedReceipt, ...]:
        """The COMPLETE record list; refused once a prefix has been retired."""
        if self.retention_floor:
            raise HistoryError(
                "HISTORY_BELOW_RETENTION_FLOOR",
                f"records() requires the complete history; retention_floor={self.retention_floor}; "
                "use retained_records()",
            )
        return tuple(self._records)

    def retained_records(self) -> tuple[RecordedReceipt, ...]:
        """Records at event indices >= retention_floor (the retained window)."""
        return tuple(self._records)

    def projection(self, request: ProjectionRequest | None = None) -> ContextProjection:
        """Read-only projection of committed history (records by default).

        With a retention floor the default request selects the retained
        window (``clock_min = floor + 1``) and the result binds the floor; an
        explicit request reaching at or below the floor fails closed.
        """
        floor = self.retention_floor
        if request is None:
            request = ProjectionRequest(receiver_ids=(self.history_entity,),
                                        event_kinds=("MESSAGE_ENQUEUED",),
                                        clock_min=floor + 1 if floor else 0)
        elif type(request) is ProjectionRequest and floor and request.clock_min <= floor:
            raise HistoryError(
                "HISTORY_BELOW_RETENTION_FLOOR",
                f"projection clock_min={request.clock_min} reaches the retired prefix; "
                f"retention_floor={floor} (first retained clock {floor + 1})",
            )

        read_window = self._python_read_window()

        try:
            return project_kernel_history(
                self._kernel,
                request,
            )
        except ProjectionError as exc:
            if "BELOW_RETENTION_FLOOR" in str(exc):
                raise HistoryError("HISTORY_BELOW_RETENTION_FLOOR", str(exc)) from exc
            raise
        finally:
            self._close_python_read_window(
                read_window
            )

    # -- the one write -----------------------------------------------------------
    def _compact_and_reopen(self) -> None:
        """Cold path after a non-mutating SEGMENT_FULL: compact, then re-handoff."""
        self._native.close()
        self._native = None
        try:
            self._kernel.open()
        except _STORAGE_ERRORS as exc:
            raise _translate(exc, "HISTORY_COMPACTION_FAILED") from exc
        self._compact()
        self._native_handoff()

    def _native_record(self, record: ReceiptRecord):
        return self._native.record(
            self._ids[record.subsystem],
            self.history_entity,
            record.payload(),
        )

    def record(self, record: ReceiptRecord) -> RecordedReceipt:
        if type(record) is not ReceiptRecord:
            raise HistoryError(
                "RECORD_TYPE",
                type(record).__name__,
            )

        # Exact duplicate detection within the retained window only.
        existing = self._record_index.get(record)

        if existing is not None:
            return existing

        if self._native is not None:
            try:
                result = self._native_record(record)
            except NativeHistoryError as exc:
                if exc.code != SEGMENT_FULL:
                    raise HistoryError(
                        "NATIVE_HISTORY_RECORD",
                        str(exc),
                    ) from exc
                # Non-mutating capacity disposition: nothing was appended.
                # Compact on the cold path and retry this exact receipt once.
                try:
                    self._compact_and_reopen()
                except HistoryError:
                    raise
                except Exception as compaction:
                    # Any other cold-path failure is still a stable, fail-stop
                    # history refusal: the receipt was not appended.
                    raise HistoryError("HISTORY_COMPACTION_FAILED", str(compaction)) from compaction
                try:
                    result = self._native_record(record)
                except NativeHistoryError as retry:
                    if retry.code == SEGMENT_FULL:
                        raise HistoryError(
                            "HISTORY_STORAGE_CAPACITY",
                            "receipt does not fit an empty segment under the policy",
                        ) from retry
                    raise HistoryError(
                        "NATIVE_HISTORY_RECORD",
                        str(retry),
                    ) from retry

            recorded = RecordedReceipt(
                record=record,
                message_id=bytes(result.message_id).split(b"\x00", 1)[0].decode("ascii"),
                event_index=int(result.enqueue_event_index),
                event_digest=bytes(result.enqueue_event_digest).split(b"\x00", 1)[0].decode("ascii"),
            )

            self._records.append(recorded)
            self._record_index[record] = recorded
            self._continuity = self._continuity.fold(record)

            return recorded

        # Once runtime history ownership has been handed to the native
        # session there is deliberately no Python write fallback.  If the
        # native owner is absent, this object is not an authorized writer.
        raise HistoryError("HISTORY_CLOSED")
