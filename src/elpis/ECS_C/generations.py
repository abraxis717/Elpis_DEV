"""Elpis ECS — crash-safe compacted-history generations in one bounded directory.

Layout (one exclusively owned directory)::

    LOCK                 flock target for the owner's whole lifetime (0 bytes)
    MANIFEST             the ONLY publication pointer (framed canonical JSON)
    g<N>.ckpt            immutable verified compaction checkpoint of generation N
    g<N>.seg             append-only active event segment of generation N
    MANIFEST.tmp         transient (publication in progress)
    g<N>.ckpt.tmp        transient (checkpoint being written)

Every artifact of a published generation is immutable except the active
segment, which only grows by verified framed appends (and is bounded by the
caller's policy). A generation is the pair (checkpoint, segment); the
segment's first frame continues the checkpoint's global coordinates.

Generation switch law (``publish``), N -> N+1, each step named for the crash
matrix (``_crash_point``):

  1. ``checkpoint.write``    write ``g<N+1>.ckpt.tmp`` (O_EXCL)
  2. ``checkpoint.fsync``    fsync it
  3. ``checkpoint.rename``   rename to ``g<N+1>.ckpt``
  4. ``segment.create``      create empty ``g<N+1>.seg`` (O_EXCL), or hard-link
                             a migrated log to it
  5. ``segment.fsync``       fsync the segment
  6. ``artifacts.dirsync``   fsync the directory (artifacts durable)
  7. ``manifest.write``      write + fsync ``MANIFEST.tmp``
  8. ``manifest.replace``    atomically rename ``MANIFEST.tmp`` -> ``MANIFEST``
  9. ``manifest.dirsync``    fsync the directory (publication durable)
  then ``cleanup``           remove generation N's files, fsync the directory

A process crash before step 8 reopens generation N (the new files are
unpublished orphans and are removed at the next open). A crash at or after
step 8 reopens generation N+1 (generation N's files are orphans). Generation
N is never removed before step 9 has succeeded, so a failed compaction never
deletes the last known-good generation. Power-loss ordering between steps 8
and 9 may expose either generation; both remain complete until cleanup.

Directory bound: at most one published generation plus one in-flight
checkpoint and one in-flight manifest exist at any time, and every write is
admitted against ``max_directory_bytes`` BEFORE it happens (StorageCapacityError,
nothing written). Sizes are logical file bytes; filesystem block/metadata
overhead is outside this bound.

This module is ECS_C storage mechanics only; it never interprets events.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import canonical
from .compaction import CompactionCheckpoint
from .errors import GenerationError, StorageCapacityError
from .limits import MAX_INT, SUPPORTED_SCHEDULER_PROTOCOLS
from .persistence import LENGTH_PREFIX, _fsync, _sync_directory, _write_all

__all__ = (
    "LOCK_NAME", "MANIFEST_NAME", "MANIFEST_SCHEMA", "MAX_MANIFEST_BYTES",
    "LEGACY_NAMES", "DirectoryLock", "GenerationManifest", "GenerationStore",
    "checkpoint_name", "segment_name",
)

LOCK_NAME = "LOCK"
MANIFEST_NAME = "MANIFEST"
MANIFEST_TMP_NAME = "MANIFEST.tmp"
MANIFEST_SCHEMA = "ecs.generation-manifest.v1"
DOMAIN_MANIFEST = "ecs.generation-manifest.v1"
MAX_MANIFEST_BYTES = 4096

# Pre-compaction single-log layout. Recognized so the owner can migrate it;
# never interpreted by this module.
LEGACY_NAMES = frozenset({"events.log", "checkpoint.bin", "checkpoint.bin.tmp"})

_ARTIFACT = re.compile(r"g([0-9]{16})\.(ckpt|seg|ckpt\.tmp)\Z")
_MANIFEST_FIELDS = frozenset({
    "schema", "generation", "checkpoint_file", "checkpoint_digest",
    "segment_file", "base_event_count", "base_head_event_digest",
    "base_state_root_digest", "genesis_label", "scheduler_protocol",
    "mailbox_capacity",
})


def checkpoint_name(generation: int) -> str:
    return f"g{generation:016d}.ckpt"


def segment_name(generation: int) -> str:
    return f"g{generation:016d}.seg"


def _crash_point(step: str) -> None:
    """Test seam: the crash matrix replaces this to stop at a named step.

    Production behaviour is a no-op. A simulated crash must raise a
    BaseException that is not an Exception so no in-process cleanup runs,
    exactly like a killed process.
    """


def _gen_error(code: str, detail: str = "") -> GenerationError:
    return GenerationError(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class GenerationManifest:
    generation: int
    checkpoint_digest: str
    base_event_count: int
    base_head_event_digest: str
    base_state_root_digest: str
    genesis_label: str
    scheduler_protocol: str
    mailbox_capacity: int

    @property
    def checkpoint_file(self) -> str:
        return checkpoint_name(self.generation)

    @property
    def segment_file(self) -> str:
        return segment_name(self.generation)

    def _body(self) -> dict:
        return {
            "schema": MANIFEST_SCHEMA,
            "generation": self.generation,
            "checkpoint_file": self.checkpoint_file,
            "checkpoint_digest": self.checkpoint_digest,
            "segment_file": self.segment_file,
            "base_event_count": self.base_event_count,
            "base_head_event_digest": self.base_head_event_digest,
            "base_state_root_digest": self.base_state_root_digest,
            "genesis_label": self.genesis_label,
            "scheduler_protocol": self.scheduler_protocol,
            "mailbox_capacity": self.mailbox_capacity,
        }

    def to_bytes(self) -> bytes:
        body = self._body()
        record = dict(body, manifest_digest=canonical.domain_digest(DOMAIN_MANIFEST, body))
        payload = canonical.canonical_bytes(record)
        raw = LENGTH_PREFIX.pack(len(payload)) + payload
        if len(raw) > MAX_MANIFEST_BYTES:
            raise _gen_error("MANIFEST_TOO_LARGE")
        return raw

    @classmethod
    def from_bytes(cls, raw: bytes) -> "GenerationManifest":
        if len(raw) > MAX_MANIFEST_BYTES or len(raw) < 8:
            raise _gen_error("MANIFEST_INVALID", "size")
        if LENGTH_PREFIX.unpack_from(raw, 0)[0] != len(raw) - 8:
            raise _gen_error("MANIFEST_INVALID", "frame")
        payload = raw[8:]
        try:
            record = json.loads(payload.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise _gen_error("MANIFEST_INVALID", "json") from exc
        if type(record) is not dict or canonical.canonical_bytes(record) != payload:
            raise _gen_error("MANIFEST_INVALID", "noncanonical")
        if set(record) != _MANIFEST_FIELDS | {"manifest_digest"}:
            raise _gen_error("MANIFEST_INVALID", "fields")
        body = {k: v for k, v in record.items() if k != "manifest_digest"}
        if record["manifest_digest"] != canonical.domain_digest(DOMAIN_MANIFEST, body):
            raise _gen_error("MANIFEST_INVALID", "digest")
        if body["schema"] != MANIFEST_SCHEMA:
            raise _gen_error("MANIFEST_INVALID", "schema")
        generation = body["generation"]
        count = body["base_event_count"]
        capacity = body["mailbox_capacity"]
        for value, low in ((generation, 1), (count, 0), (capacity, 1)):
            if type(value) is not int or not low <= value <= MAX_INT:
                raise _gen_error("MANIFEST_INVALID", "integer field")
        for key in ("checkpoint_digest", "base_head_event_digest", "base_state_root_digest"):
            value = body[key]
            if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise _gen_error("MANIFEST_INVALID", key)
        if body["scheduler_protocol"] not in SUPPORTED_SCHEDULER_PROTOCOLS or type(body["genesis_label"]) is not str:
            raise _gen_error("MANIFEST_INVALID", "configuration")
        manifest = cls(
            generation=generation,
            checkpoint_digest=body["checkpoint_digest"],
            base_event_count=count,
            base_head_event_digest=body["base_head_event_digest"],
            base_state_root_digest=body["base_state_root_digest"],
            genesis_label=body["genesis_label"],
            scheduler_protocol=body["scheduler_protocol"],
            mailbox_capacity=capacity,
        )
        if body["checkpoint_file"] != manifest.checkpoint_file or body["segment_file"] != manifest.segment_file:
            raise _gen_error("MANIFEST_INVALID", "artifact names")
        return manifest

    @classmethod
    def for_checkpoint(cls, generation: int, checkpoint: CompactionCheckpoint) -> "GenerationManifest":
        return cls(
            generation=generation,
            checkpoint_digest=checkpoint.checkpoint_digest,
            base_event_count=checkpoint.event_count,
            base_head_event_digest=checkpoint.head_event_digest,
            base_state_root_digest=checkpoint.state_root_digest,
            genesis_label=checkpoint.genesis_label,
            scheduler_protocol=checkpoint.scheduler_protocol,
            mailbox_capacity=checkpoint.mailbox_capacity,
        )


class DirectoryLock:
    """Exclusive non-blocking flock on ``<dir>/LOCK`` for an owner's lifetime."""

    def __init__(self, directory: Path) -> None:
        self.path = Path(directory) / LOCK_NAME
        self._fd: int | None = None

    def acquire(self) -> None:
        if self._fd is not None:
            return
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise _gen_error("STORE_LOCKED", "another owner holds this history") from exc
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd

    def release(self) -> None:
        if self._fd is not None:
            fd, self._fd = self._fd, None
            os.close(fd)

    @property
    def held(self) -> bool:
        return self._fd is not None


def _write_new_file(path: Path, data: bytes, *, step_written: str, step_synced: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _write_all(fd, data)
        _crash_point(step_written)
        _fsync(fd)
        _crash_point(step_synced)
    finally:
        os.close(fd)


class GenerationStore:
    """Owner of one compacted-history directory (caller holds DirectoryLock)."""

    def __init__(
        self,
        directory: Path,
        *,
        genesis_label: str,
        mailbox_capacity: int,
        max_checkpoint_bytes: int,
        max_directory_bytes: int,
    ) -> None:
        self.directory = Path(directory)
        self.genesis_label = genesis_label
        self.mailbox_capacity = mailbox_capacity
        self.max_checkpoint_bytes = max_checkpoint_bytes
        self.max_directory_bytes = max_directory_bytes

    # -- inspection ----------------------------------------------------------
    def entries(self) -> dict[str, Any]:
        """Classify every directory entry; unknown entries are refused."""
        found: dict[str, Any] = {"manifest": False, "manifest_tmp": False, "legacy": set(),
                                 "artifacts": {}, "lock": False}
        for entry in os.scandir(self.directory):
            name = entry.name
            if not entry.is_file(follow_symlinks=False):
                raise _gen_error("FOREIGN_ENTRY", name)
            if name == LOCK_NAME:
                found["lock"] = True
            elif name == MANIFEST_NAME:
                found["manifest"] = True
            elif name == MANIFEST_TMP_NAME:
                found["manifest_tmp"] = True
            elif name in LEGACY_NAMES:
                found["legacy"].add(name)
            else:
                match = _ARTIFACT.match(name)
                if match is None:
                    raise _gen_error("FOREIGN_ENTRY", name)
                found["artifacts"][name] = int(match.group(1))
        return found

    def usage_bytes(self, *, exclude_legacy: bool = False) -> int:
        """Logical bytes of every regular file, counting hard links once.

        ``exclude_legacy`` omits a pre-existing legacy layout that is being
        migrated (it is removed right after publication); every other byte
        counts.
        """
        seen = set()
        total = 0
        for entry in os.scandir(self.directory):
            if exclude_legacy and entry.name in LEGACY_NAMES:
                continue
            st = entry.stat(follow_symlinks=False)
            key = (st.st_dev, st.st_ino)
            if key not in seen:
                seen.add(key)
                total += st.st_size
        return total

    def admit(self, planned_bytes: int, what: str, *, exclude_legacy: bool = False) -> None:
        """Refuse (before writing anything) a write that would exceed the budget."""
        if self.usage_bytes(exclude_legacy=exclude_legacy) + planned_bytes > self.max_directory_bytes:
            raise StorageCapacityError(
                f"HISTORY_STORAGE_CAPACITY: {what} needs {planned_bytes} bytes; "
                f"directory budget {self.max_directory_bytes} would be exceeded"
            )

    def read_manifest(self) -> GenerationManifest | None:
        path = self.directory / MANIFEST_NAME
        try:
            with open(path, "rb") as fh:
                raw = fh.read(MAX_MANIFEST_BYTES + 1)
        except FileNotFoundError:
            return None
        manifest = GenerationManifest.from_bytes(raw)
        if (manifest.genesis_label != self.genesis_label
                or manifest.mailbox_capacity != self.mailbox_capacity):
            raise _gen_error("HISTORY_GENERATION_MISMATCH", "manifest configuration")
        return manifest

    def load_checkpoint(self, manifest: GenerationManifest) -> CompactionCheckpoint:
        """Load and verify the generation's checkpoint against its manifest."""
        path = self.directory / manifest.checkpoint_file
        try:
            with open(path, "rb") as fh:
                raw = fh.read(self.max_checkpoint_bytes + 1)
        except FileNotFoundError as exc:
            raise _gen_error("HISTORY_GENERATION_MISMATCH", "checkpoint missing") from exc
        checkpoint = CompactionCheckpoint.from_bytes(
            raw,
            genesis_label=self.genesis_label,
            mailbox_capacity=self.mailbox_capacity,
            scheduler_protocol=manifest.scheduler_protocol,
            max_bytes=self.max_checkpoint_bytes,
        )
        if GenerationManifest.for_checkpoint(manifest.generation, checkpoint) != manifest:
            raise _gen_error("HISTORY_GENERATION_MISMATCH", "manifest does not bind this checkpoint")
        if not (self.directory / manifest.segment_file).is_file():
            raise _gen_error("HISTORY_GENERATION_MISMATCH", "segment missing")
        return checkpoint

    def segment_path(self, manifest: GenerationManifest) -> Path:
        return self.directory / manifest.segment_file

    # -- recovery --------------------------------------------------------------
    def remove_unpublished(self, manifest: GenerationManifest | None) -> None:
        """Remove crash debris. Never touches the published generation.

        With a manifest: every transient file and every artifact of any other
        generation (pre-publication orphans and post-publication leftovers).
        Without a manifest no generation was ever published; artifacts are
        removed only if every segment is empty (nothing could have been
        recorded through an unpublished generation). Anything else fails
        closed rather than guess.
        """
        entries = self.entries()
        doomed = []
        if entries["manifest_tmp"]:
            doomed.append(MANIFEST_TMP_NAME)
        for name, generation in entries["artifacts"].items():
            if name.endswith(".tmp"):
                doomed.append(name)
            elif manifest is None or generation != manifest.generation:
                if (manifest is None and name.endswith(".seg")
                        and (self.directory / name).stat().st_size
                        and not self._is_legacy_link(self.directory / name)):
                    raise _gen_error("HISTORY_GENERATION_MISMATCH",
                                     f"non-empty segment {name} without a published manifest")
                doomed.append(name)
        for name in sorted(doomed):
            os.unlink(self.directory / name)
        if doomed:
            _sync_directory(str(self.directory))

    def _is_legacy_link(self, path: Path) -> bool:
        # An unpublished migration hard-links the intact legacy log; removing
        # that name loses nothing because the legacy log itself remains.
        legacy = self.directory / "events.log"
        try:
            return legacy.is_file() and os.path.samefile(legacy, path)
        except OSError:
            return False

    # -- the switch ------------------------------------------------------------
    def publish(
        self,
        checkpoint: CompactionCheckpoint,
        *,
        previous: GenerationManifest | None,
        link_segment_from: Path | None = None,
        exclude_legacy: bool = False,
    ) -> GenerationManifest:
        """Publish generation previous+1 = (checkpoint, new segment). See module law."""
        generation = 1 if previous is None else previous.generation + 1
        ckpt_bytes = checkpoint.to_bytes()
        if len(ckpt_bytes) > self.max_checkpoint_bytes:
            raise StorageCapacityError("HISTORY_STORAGE_CAPACITY: checkpoint exceeds its bound")
        manifest = GenerationManifest.for_checkpoint(generation, checkpoint)
        manifest_bytes = manifest.to_bytes()
        # The whole switch is admitted up front: new checkpoint + transient
        # manifest. A new segment is empty; a migrated segment is a hard link
        # (no new data blocks).
        self.admit(len(ckpt_bytes) + len(manifest_bytes), "generation switch",
                   exclude_legacy=exclude_legacy)

        directory = self.directory
        ckpt_tmp = directory / (checkpoint_name(generation) + ".tmp")
        ckpt_path = directory / checkpoint_name(generation)
        seg_path = directory / segment_name(generation)
        manifest_tmp = directory / MANIFEST_TMP_NAME
        published = False
        try:
            _write_new_file(ckpt_tmp, ckpt_bytes,
                            step_written="checkpoint.write", step_synced="checkpoint.fsync")
            os.rename(ckpt_tmp, ckpt_path)
            _crash_point("checkpoint.rename")
            if link_segment_from is None:
                fd = os.open(seg_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            else:
                os.link(link_segment_from, seg_path)
                fd = os.open(seg_path, os.O_RDONLY)
            try:
                _crash_point("segment.create")
                _fsync(fd)
                _crash_point("segment.fsync")
            finally:
                os.close(fd)
            _sync_directory(str(directory))
            _crash_point("artifacts.dirsync")
            _write_new_file(manifest_tmp, manifest_bytes,
                            step_written="manifest.write.data", step_synced="manifest.write")
            os.replace(manifest_tmp, directory / MANIFEST_NAME)
            published = True
            _crash_point("manifest.replace")
            _sync_directory(str(directory))
            _crash_point("manifest.dirsync")
        except Exception as exc:
            if not published:
                # Unpublished: the previous generation is untouched and stays
                # authoritative. Remove only the new generation's debris.
                for path in (manifest_tmp, seg_path, ckpt_path, ckpt_tmp):
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass
            raise _gen_error("HISTORY_COMPACTION_FAILED",
                             "published" if published else "unpublished; previous generation kept") from exc

        if previous is not None:
            # Publication is durable: the previous generation is now garbage.
            try:
                os.unlink(directory / previous.checkpoint_file)
                _crash_point("cleanup.checkpoint")
                os.unlink(directory / previous.segment_file)
                _crash_point("cleanup.segment")
                _sync_directory(str(directory))
            except Exception as exc:
                raise _gen_error("HISTORY_COMPACTION_FAILED", "published; cleanup pending") from exc
        return manifest

