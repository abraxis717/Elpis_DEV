"""Two-slot crash-safe continuity register (docs/CONTINUITY.md).

The store keeps exactly two fixed-size slot files and the one current
snapshot in memory. Publication writes the complete next record into the
slot that does not hold the current authority and then ``fdatasync``s it.
Restart reads both slots and selects the valid record with the higher
generation. Nothing grows, nothing is replayed, nothing is compacted.
"""
from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path

from .record import (
    RECORD_SIZE,
    ContinuityError,
    ContinuitySnapshot,
    EvolutionAuthority,
    decode_record,
    encode_record,
)

__all__ = ("SLOT_NAMES", "ContinuityStore")

SLOT_NAMES = ("continuity.a", "continuity.b")
_TMP_SUFFIX = ".tmp"
# Names of the retired receipt-history layout. They are refused, never read.
_LEGACY_NAMES = frozenset({"MANIFEST", "MANIFEST.tmp", "LOCK", "events.log", "checkpoint.bin",
                           "checkpoint.bin.tmp"})
_GENESIS = ContinuitySnapshot(1, None, EvolutionAuthority(0, "0" * 64))


def _crash_point(step: str) -> None:
    """Test seam for the crash matrix. Production behaviour is a no-op.

    A simulated process death raises a ``BaseException`` that is not an
    ``Exception``, so no in-process recovery runs.
    """


def _sync_fd(fd: int) -> None:
    sync = getattr(os, "fdatasync", os.fsync)
    while True:
        try:
            sync(fd)
            return
        except InterruptedError:
            continue


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        _sync_fd(fd)
    finally:
        os.close(fd)


def _pwrite_all(fd: int, data: bytes, offset: int) -> None:
    view = memoryview(data)
    while view:
        try:
            n = os.pwrite(fd, view, offset)
        except InterruptedError:
            continue
        if n <= 0:
            raise OSError(errno.EIO, "short continuity write")
        view = view[n:]
        offset += n


def _pread_exact(fd: int, size: int) -> bytes:
    data = b""
    while len(data) < size:
        try:
            chunk = os.pread(fd, size - len(data), len(data))
        except InterruptedError:
            continue
        if not chunk:
            break
        data += chunk
    return data


class ContinuityStore:
    """Exclusive owner of one continuity directory. Single writer, same process."""

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        if not directory.is_absolute():
            raise ContinuityError("CONTINUITY_PATH", "continuity directory must be absolute")
        self.directory = directory
        self._fds: tuple[int, int] | None = None
        self._lock: int | None = None
        self._current: ContinuitySnapshot | None = None
        self._slot = 0  # index of the slot holding the current authority
        self._poisoned: str | None = None

    # -- lifecycle -------------------------------------------------------------------
    def open(self) -> "ContinuityStore":
        if self._fds is not None:
            raise ContinuityError("CONTINUITY_OPEN", "already open")
        self._poisoned = None
        created = not self.directory.exists()
        self.directory.mkdir(parents=True, exist_ok=True)
        if created:
            _sync_dir(self.directory.parent)
        lock = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            # The directory itself is the lock: no lock file, and nothing is
            # inspected or initialized before exclusive ownership.
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ContinuityError("CONTINUITY_LOCKED", "another owner holds this continuity") from exc
            self._admit_directory()
            fds = []
            try:
                records = []
                for name in SLOT_NAMES:
                    fd = os.open(self.directory / name, os.O_RDWR)
                    fds.append(fd)
                    if os.fstat(fd).st_size != RECORD_SIZE:
                        raise ContinuityError("CONTINUITY_CORRUPT", "slot size")
                    records.append(_pread_exact(fd, RECORD_SIZE))
                current, slot = self._resolve(records)
            except BaseException:
                for fd in fds:
                    os.close(fd)
                raise
        except BaseException:
            os.close(lock)
            raise
        self._lock, self._fds = lock, (fds[0], fds[1])
        self._current, self._slot = current, slot
        return self

    def close(self) -> None:
        if self._fds is not None:
            fds, self._fds = self._fds, None
            for fd in fds:
                os.close(fd)
        if self._lock is not None:
            lock, self._lock = self._lock, None
            os.close(lock)
        self._current = None

    def __enter__(self) -> "ContinuityStore":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def _admit_directory(self) -> None:
        names = {entry.name for entry in os.scandir(self.directory)}
        legacy = sorted(n for n in names if n in _LEGACY_NAMES or (n.startswith("g") and n.endswith((".ckpt", ".seg"))))
        if legacy:
            raise ContinuityError(
                "CONTINUITY_LEGACY_STORAGE",
                f"retired receipt-history layout present ({', '.join(legacy[:4])}); see docs/CONTINUITY.md",
            )
        allowed = set(SLOT_NAMES) | {n + _TMP_SUFFIX for n in SLOT_NAMES}
        foreign = sorted(names - allowed)
        if foreign:
            raise ContinuityError("CONTINUITY_CORRUPT", f"foreign entries: {', '.join(foreign[:4])}")
        if SLOT_NAMES[0] in names:
            if SLOT_NAMES[1] not in names:
                raise ContinuityError("CONTINUITY_CORRUPT", "second slot missing")
            for name in SLOT_NAMES:  # debris of an initialization that already completed
                if name + _TMP_SUFFIX in names:
                    os.unlink(self.directory / (name + _TMP_SUFFIX))
            return
        # No continuity.a: initialization never reached its publication point.
        # Its debris is a renamed but still empty continuity.b at most. A
        # continuity.b that carries a record was published to, so its partner
        # was deleted: fail closed rather than re-initialize over authority.
        if SLOT_NAMES[1] in names and (self.directory / SLOT_NAMES[1]).read_bytes() != bytes(RECORD_SIZE):
            raise ContinuityError("CONTINUITY_CORRUPT", "first slot missing beside a published record")
        self._initialize(names)

    def _initialize(self, names: set[str]) -> None:
        """First open of an empty directory: publish generation 1 (unanchored)."""
        for name in names:  # an initialization that never reached its publication point
            os.unlink(self.directory / name)
        payloads = ((SLOT_NAMES[1], bytes(RECORD_SIZE)), (SLOT_NAMES[0], encode_record(_GENESIS)))
        for name, payload in payloads:
            fd = os.open(self.directory / (name + _TMP_SUFFIX), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                _pwrite_all(fd, payload, 0)
                _sync_fd(fd)
            finally:
                os.close(fd)
            _crash_point(f"init.{name}.written")
        for name, _ in payloads:  # b first: the existence of continuity.a is the publication point
            os.rename(self.directory / (name + _TMP_SUFFIX), self.directory / name)
            _crash_point(f"init.{name}.renamed")
        _sync_dir(self.directory)
        _crash_point("init.dirsync")

    @staticmethod
    def _resolve(records: list[bytes]) -> tuple[ContinuitySnapshot, int]:
        decoded: list[ContinuitySnapshot | None] = []
        for raw in records:
            try:
                decoded.append(decode_record(raw))
            except ContinuityError:
                decoded.append(None)  # torn or stale slot: the other one must carry authority
        candidates = [(snap.generation, i) for i, snap in enumerate(decoded) if snap is not None]
        if not candidates:
            raise ContinuityError("CONTINUITY_CORRUPT", "no valid continuity record")
        if len(candidates) == 2 and candidates[0][0] == candidates[1][0]:
            raise ContinuityError("CONTINUITY_CORRUPT", "ambiguous continuity generations")
        generation, slot = max(candidates)
        return decoded[slot], slot

    # -- reads --------------------------------------------------------------------------
    def _require(self) -> ContinuitySnapshot:
        if self._poisoned is not None:
            raise ContinuityError(self._poisoned, "the continuity store must be reopened")
        if self._current is None or self._fds is None:
            raise ContinuityError("CONTINUITY_CLOSED")
        return self._current

    def snapshot(self) -> ContinuitySnapshot:
        return self._require()

    # -- the only transitions ------------------------------------------------------------
    def _publish(self, nxt: ContinuitySnapshot) -> ContinuitySnapshot:
        record = encode_record(nxt)
        target = 1 - self._slot
        fd = self._fds[target]
        _crash_point("publish.begin")
        try:
            _pwrite_all(fd, record, 0)
        except OSError as exc:
            # The current authority's slot is untouched and stays in force.
            raise ContinuityError("CONTINUITY_PUBLICATION_REFUSED", str(exc)) from exc
        _crash_point("publish.written")
        try:
            _sync_fd(fd)
        except OSError as exc:
            # Durable outcome unknown: reopen resolves to exactly one complete record.
            self._poisoned = "CONTINUITY_PUBLICATION_UNCERTAIN"
            self.close()
            raise ContinuityError("CONTINUITY_PUBLICATION_UNCERTAIN", str(exc)) from exc
        _crash_point("publish.synced")
        self._current, self._slot = nxt, target
        return nxt

    def anchor_cognition(self, k1_state_digest: bytes) -> ContinuitySnapshot:
        """Explicitly anchor the first K1 lineage at a retained-state identity."""
        current = self._require()
        if type(k1_state_digest) is not bytes or len(k1_state_digest) != 32:
            raise ContinuityError("CONTINUITY_INVALID", "a 32-byte K1 retained-state digest is required")
        if current.anchored:
            raise ContinuityError("CONTINUITY_ALREADY_ANCHORED", "a K1 lineage is already anchored")
        return self._publish(ContinuitySnapshot(current.generation + 1, k1_state_digest, current.evolution))

    def commit_cognition_transition(self, before: bytes, after: bytes) -> ContinuitySnapshot:
        """Publish ``after`` as the expected K1 identity of a committed turn from ``before``."""
        current = self._require()
        for value in (before, after):
            if type(value) is not bytes or len(value) != 32:
                raise ContinuityError("CONTINUITY_INVALID", "32-byte K1 retained-state digests are required")
        if not current.anchored:
            raise ContinuityError("CONTINUITY_UNANCHORED", "no K1 lineage is anchored")
        if current.k1_state_digest != before:
            raise ContinuityError("CONTINUITY_STATE_MISMATCH", "transition does not start from the expected K1 identity")
        return self._publish(ContinuitySnapshot(current.generation + 1, after, current.evolution))

    def commit_evolution_transition(self, expected: EvolutionAuthority, receipt_digest: str) -> ContinuitySnapshot:
        """Advance the evolution authority from ``expected`` to the admitted receipt."""
        current = self._require()
        if type(expected) is not EvolutionAuthority or expected != current.evolution:
            raise ContinuityError("CONTINUITY_AUTHORITY_MISMATCH", "evolution authority moved")
        if receipt_digest == "0" * 64:
            raise ContinuityError("CONTINUITY_INVALID", "a path-transition receipt digest is required")
        try:
            nxt = EvolutionAuthority(current.evolution.revision + 1, receipt_digest)
        except ContinuityError as exc:
            raise ContinuityError("CONTINUITY_INVALID", "a 64-hex path-transition receipt digest is required") from exc
        return self._publish(ContinuitySnapshot(current.generation + 1, current.k1_state_digest, nxt))
