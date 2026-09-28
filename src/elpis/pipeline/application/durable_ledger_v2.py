"""Explicit artifact-bound schema-v2 primitive; no v1 migration or publication.

The physical tables/indexes retain the predecessor's authenticated storage
layout. Only its storage declarations/inspector are reused, never its ledger
class, entry hashing, genesis or schema markers. Requiring a real artifact
digest makes the inherited partial uniqueness index cover every valid v2 entry.

Use one connection per thread/process on storage honoring SQLite locks/fsync.
Entries detect corruption relative to trusted database/head authority; they are
not signatures and cannot defeat coherent database/head rewrites or rollback.
Opening an existing database first checks its file markers without writing.
There is no migration, export registration, publisher adoption or default change.
"""
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
import sqlite3

from .canonical import canonical_digest
from .durable_ledger import _SCHEMA_SQL as _STORAGE_SCHEMA_SQL, _verify_schema


SCHEMA_ID = "elpis.grid81.durable-application-ledger.v2"
SCHEMA_VERSION = 2
APPLICATION_ID = 0x454C5032  # ELP2; distinct from v1 ELPL.
ENTRY_DOMAIN = "elpis.grid81.application-ledger.entry.v2"
GENESIS_DOMAIN = "elpis.grid81.application-ledger.genesis.v2"
GENESIS_HEAD = canonical_digest({
    "domain": GENESIS_DOMAIN, "schema": SCHEMA_ID,
    "sequence": 0, "previous_head": "", "entries": [],
})
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _require_digest(name, value):
    # Canonical SHA-256 output convention, stricter than int(value, 16): no
    # sign, whitespace, Unicode digits, case aliases, subclasses or sentinel.
    if type(value) is not str:
        raise TypeError(f"{name} must be a lowercase SHA-256 hex string")
    if _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")


def entry_digest_v2(sequence: int, previous_head: str, receipt_digest: str,
                    artifact_digest: str) -> str:
    """Canonical, domain-separated identity of all logical v2 entry inputs."""
    if type(sequence) is not int or not 0 < sequence <= 2**63 - 1:
        raise ValueError("sequence must be a positive SQLite signed-64-bit integer")
    for name, value in (("previous_head", previous_head), ("receipt_digest", receipt_digest),
                        ("artifact_digest", artifact_digest)):
        _require_digest(name, value)
    return canonical_digest({
        "domain": ENTRY_DOMAIN, "schema": SCHEMA_ID, "sequence": sequence,
        "previous_head": previous_head, "receipt_digest": receipt_digest,
        "artifact_digest": artifact_digest,
    })


@dataclass(frozen=True, slots=True)
class LedgerEntryV2:
    """Immutable entry; artifact_digest is identity, not optional metadata."""
    sequence: int
    previous_head: str
    receipt_digest: str
    artifact_digest: str
    entry_digest: str

    def __post_init__(self):
        _require_digest("entry_digest", self.entry_digest)
        expected = entry_digest_v2(self.sequence, self.previous_head,
                                   self.receipt_digest, self.artifact_digest)
        if self.entry_digest != expected:
            raise ValueError("v2 entry digest mismatch")


class DurableApplicationLedgerV2:
    """Persistent CAS/replay exclusion with mandatory artifact-bound identity.

    No inheritance from v1: existing publisher isinstance admission cannot
    accidentally admit this primitive. Failed append rolls back both tables.
    Opening/reopening performs complete schema, SQLite integrity, chain and
    association verification. The owner then binds data_version, schema
    versions, connection change count and the verified tail sequence/head.
    Ordinary owner-only append validates only that snapshot and the new
    entry/association. Another connection's commit or an out-of-band mutation
    forces full revalidation before ordinary CAS semantics continue.
    SQLite contention uses the predecessor's 30-second timeout and FULL sync.
    """

    def __init__(self, path: str | os.PathLike[str]):
        raw_path = os.fspath(path)
        if not isinstance(raw_path, str) or raw_path in ("", ":memory:"):
            raise ValueError("A persistent database file path is required")
        database_path = Path(raw_path).resolve()
        if any(parent.name == "Grid81" and parent.parent.name == "Canonical"
               for parent in (database_path, *database_path.parents)):
            raise ValueError("Ledger database cannot reside inside Canonical/Grid81")
        self._pid = os.getpid()
        self._database_path = database_path
        self._connection = None
        self._storage_identity = None
        self._verified_data_version = None
        self._verified_schema_version = None
        self._verified_temp_schema_version = None
        self._verified_total_changes = 0
        self._verified_count = 0
        self._verified_head = GENESIS_HEAD
        if database_path.exists() and not self._has_v2_header(database_path):
            # Unknown/v1 files receive only read-only SQLite admission. Explicit
            # v2 headers instead permit SQLite's ordinary hot-journal recovery
            # before full verification; a read-only connection cannot recover.
            with closing(sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True,
                                         timeout=30.0, isolation_level=None)) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA temp_store = MEMORY")
                connection.execute("BEGIN")
                if not self._uninitialized(connection):
                    self._require_valid(connection)
                connection.rollback()
        self._connection = sqlite3.connect(str(database_path), timeout=30.0,
                                           isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        try:
            info = database_path.stat()
            self._storage_identity = (info.st_dev, info.st_ino)
            self._connection.execute("PRAGMA synchronous = FULL")
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA temp_store = MEMORY")
            with self._transaction(write=True) as connection:
                # Repeat admission under the write lock: preflight is not a
                # trust receipt, and another process may have initialized it.
                blank = self._uninitialized(connection)
                if not blank:
                    self._require_valid(connection)
                # New SQLite files default to DELETE. Never convert an existing
                # database's journal mode, including unsupported v1/WAL files.
                if connection.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
                    raise RuntimeError("SQLite rollback journaling is required")
                if blank:
                    for statement in _STORAGE_SCHEMA_SQL.values():
                        connection.execute(statement)
                    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                    connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
                self._require_valid(connection)
                self._capture_verified_owner_snapshot(connection)
        except BaseException:
            self._connection.close()
            self._connection = None
            raise

    @staticmethod
    def _has_v2_header(path):
        """Read-only format gate, not a trust receipt for schema or chain validity.

        SQLite's 100-byte file header stores user_version at offset 60 and
        application_id at offset 68, both big-endian 32-bit integers. A valid
        historical v1 file never has both v2 markers. Coherent header/database
        rewrites remain outside the external-authentication nonclaim.
        """
        with path.open("rb") as stream:
            header = stream.read(100)
        return (len(header) == 100 and header[:16] == b"SQLite format 3\0" and
                int.from_bytes(header[60:64], "big") == SCHEMA_VERSION and
                int.from_bytes(header[68:72], "big") == APPLICATION_ID)

    @staticmethod
    def _uninitialized(connection):
        return (connection.execute("PRAGMA user_version").fetchone()[0] == 0 and
                connection.execute("PRAGMA application_id").fetchone()[0] == 0 and
                connection.execute("SELECT 1 FROM main.sqlite_master LIMIT 1").fetchone() is None)

    def _check_open(self):
        if os.getpid() != self._pid:
            raise RuntimeError("Open a new DurableApplicationLedgerV2 in each process")
        if self._connection is None:
            raise RuntimeError("Ledger is closed")
        try:
            info = self._database_path.stat()
        except FileNotFoundError as error:
            raise RuntimeError("Ledger database was replaced or removed") from error
        if (info.st_dev, info.st_ino) != self._storage_identity:
            raise RuntimeError("Ledger database was replaced")

    @property
    def storage_identity(self) -> tuple[int, int]:
        """Host-local inode identity only, not portable semantic authority."""
        self._check_open()
        return self._storage_identity

    @contextmanager
    def _transaction(self, *, write=False):
        self._check_open()
        connection = self._connection
        connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield connection
            self._check_open()
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    @staticmethod
    def _tail_state(connection):
        row = connection.execute(
            "SELECT sequence, entry_digest FROM ledger_entries "
            "ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return 0, GENESIS_HEAD
        return row["sequence"], row["entry_digest"]

    def _capture_verified_owner_snapshot(self, connection):
        count, head = self._tail_state(connection)
        self._verified_count = count
        self._verified_head = head
        self._verified_data_version = connection.execute(
            "PRAGMA data_version"
        ).fetchone()[0]
        self._verified_schema_version = connection.execute(
            "PRAGMA schema_version"
        ).fetchone()[0]
        self._verified_temp_schema_version = connection.execute(
            "PRAGMA temp.schema_version"
        ).fetchone()[0]
        self._verified_total_changes = connection.total_changes

    def _require_incremental_owner_snapshot(self, connection):
        if self._verified_data_version is None:
            raise RuntimeError("Ledger owner snapshot is not initialized")

        current = (
            connection.execute("PRAGMA data_version").fetchone()[0],
            connection.execute("PRAGMA schema_version").fetchone()[0],
            connection.execute("PRAGMA temp.schema_version").fetchone()[0],
            connection.total_changes,
        )
        expected = (
            self._verified_data_version,
            self._verified_schema_version,
            self._verified_temp_schema_version,
            self._verified_total_changes,
        )

        if current != expected:
            self._require_valid(connection)
            self._capture_verified_owner_snapshot(connection)

        count, head = self._tail_state(connection)
        if count != self._verified_count or head != self._verified_head:
            self._require_valid(connection)
            self._capture_verified_owner_snapshot(connection)
            count, head = self._tail_state(connection)
        return count, head

    @staticmethod
    def _verify_new_entry_association(connection, entry):
        row = connection.execute(
            "SELECT * FROM ledger_entries WHERE sequence = ?",
            (entry.sequence,),
        ).fetchone()
        if row is None:
            raise RuntimeError("New ledger entry readback missing")
        if (
            row["sequence"] != entry.sequence
            or row["previous_head"] != entry.previous_head
            or row["receipt_digest"] != entry.receipt_digest
            or row["artifact_digest"] != entry.artifact_digest
        ):
            raise RuntimeError("New ledger entry readback mismatch")
        expected_digest = entry_digest_v2(
            row["sequence"], row["previous_head"],
            row["receipt_digest"], row["artifact_digest"],
        )
        if row["entry_digest"] != expected_digest:
            raise RuntimeError(f"digest_mismatch_at_sequence:{entry.sequence}")
        if row["entry_digest"] != entry.entry_digest:
            raise RuntimeError("New ledger entry digest readback mismatch")

        association = connection.execute(
            "SELECT artifact_digest, sequence, receipt_digest "
            "FROM applied_artifacts WHERE artifact_digest = ?",
            (entry.artifact_digest,),
        ).fetchone()
        expected_association = (
            entry.artifact_digest, entry.sequence, entry.receipt_digest,
        )
        if association is None or tuple(association) != expected_association:
            raise RuntimeError("New artifact association readback mismatch")

    @staticmethod
    def _head(connection):
        row = connection.execute(
            "SELECT entry_digest FROM ledger_entries ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else GENESIS_HEAD

    @property
    def head(self) -> str:
        self._check_open()
        return self._head(self._connection)

    @property
    def is_empty(self) -> bool:
        self._check_open()
        return self._connection.execute("SELECT 1 FROM ledger_entries LIMIT 1").fetchone() is None

    def has_artifact(self, artifact_digest: str) -> bool:
        """Return whether this exact artifact digest has been durably applied."""
        self._check_open()
        _require_digest("artifact_digest", artifact_digest)
        return self._connection.execute(
            "SELECT 1 FROM applied_artifacts WHERE artifact_digest = ?", (artifact_digest,)
        ).fetchone() is not None

    def has_receipt(self, receipt_digest: str) -> bool:
        """Return whether this exact receipt digest is present in the v2 ledger."""
        self._check_open()
        _require_digest("receipt_digest", receipt_digest)
        return self._connection.execute(
            "SELECT 1 FROM ledger_entries WHERE receipt_digest = ?", (receipt_digest,)
        ).fetchone() is not None

    def append(self, previous_head: str, receipt_digest: str,
               artifact_digest: str) -> LedgerEntryV2:
        for name, value in (
            ("previous_head", previous_head),
            ("receipt_digest", receipt_digest),
            ("artifact_digest", artifact_digest),
        ):
            _require_digest(name, value)

        committed_entry = None
        with self._transaction(write=True) as connection:
            count, head = self._require_incremental_owner_snapshot(connection)
            if previous_head != head:
                raise ValueError(
                    f"Stale ledger head: expected {head[:16]}..., "
                    f"got {previous_head[:16]}..."
                )
            if connection.execute(
                "SELECT 1 FROM applied_artifacts WHERE artifact_digest = ?",
                (artifact_digest,),
            ).fetchone():
                raise ValueError("Duplicate artifact: already applied")
            if connection.execute(
                "SELECT 1 FROM ledger_entries WHERE receipt_digest = ?",
                (receipt_digest,),
            ).fetchone():
                raise ValueError("Duplicate receipt: already applied")

            sequence = count + 1
            entry = LedgerEntryV2(
                sequence,
                head,
                receipt_digest,
                artifact_digest,
                entry_digest_v2(sequence, head, receipt_digest, artifact_digest),
            )
            connection.execute(
                "INSERT INTO ledger_entries VALUES (?, ?, ?, ?, ?)",
                (
                    sequence,
                    head,
                    receipt_digest,
                    entry.entry_digest,
                    artifact_digest,
                ),
            )
            connection.execute(
                "INSERT INTO applied_artifacts VALUES (?, ?, ?)",
                (artifact_digest, sequence, receipt_digest),
            )
            self._verify_new_entry_association(connection, entry)

            # Advance only owner-known state while the write lock is still held.
            # PRAGMA data_version is deliberately NOT refreshed here: SQLite does
            # not advance it for this connection's own commit, and retaining the
            # previously validated external baseline guarantees that any foreign
            # commit after our commit remains detectable on the next operation.
            self._verified_count = entry.sequence
            self._verified_head = entry.entry_digest
            self._verified_total_changes = connection.total_changes
            committed_entry = entry

        return committed_entry

    @staticmethod
    def _verify(connection) -> tuple[bool, str]:
        if (connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION or
                connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID):
            return False, "unsupported_ledger_schema"
        schema_ok, reason = _verify_schema(connection)
        if not schema_ok:
            return False, reason
        if [row[0] for row in connection.execute("PRAGMA integrity_check")] != ["ok"]:
            return False, "database_integrity_failure"
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            return False, "artifact_association_mismatch"
        previous_head = GENESIS_HEAD
        receipts, artifacts, expected_associations = set(), set(), set()
        count = 0
        for count, row in enumerate(connection.execute(
                "SELECT * FROM ledger_entries ORDER BY sequence"), start=1):
            if type(row["sequence"]) is not int or row["sequence"] != count:
                return False, f"invalid_sequence:{row['sequence']}"
            try:
                for column in ("previous_head", "receipt_digest", "entry_digest", "artifact_digest"):
                    _require_digest(column, row[column])
            except (TypeError, ValueError):
                return False, f"invalid_entry_digest_format_at_sequence:{count}"
            if row["previous_head"] != previous_head:
                return False, f"chain_break_at_sequence:{count}"
            expected = entry_digest_v2(count, previous_head, row["receipt_digest"], row["artifact_digest"])
            if row["entry_digest"] != expected:
                return False, f"digest_mismatch_at_sequence:{count}"
            if row["receipt_digest"] in receipts:
                return False, f"duplicate_receipt_at_sequence:{count}"
            if row["artifact_digest"] in artifacts:
                return False, f"duplicate_artifact_at_sequence:{count}"
            receipts.add(row["receipt_digest"])
            artifacts.add(row["artifact_digest"])
            expected_associations.add((row["artifact_digest"], count, row["receipt_digest"]))
            previous_head = expected
        associations = [tuple(row) for row in connection.execute(
            "SELECT artifact_digest, sequence, receipt_digest FROM applied_artifacts")]
        if len(associations) != len(expected_associations) or set(associations) != expected_associations:
            return False, "artifact_association_mismatch"
        return (True, "valid") if count else (True, "empty")

    @classmethod
    def _require_valid(cls, connection):
        ok, reason = cls._verify(connection)
        if not ok:
            raise RuntimeError(f"Invalid durable v2 ledger: {reason}")

    def verify_chain(self) -> tuple[bool, str]:
        """Verify schema, integrity, complete chain and associations in one snapshot."""
        self._check_open()
        try:
            with self._transaction() as connection:
                return self._verify(connection)
        except sqlite3.DatabaseError as error:
            return False, f"database_error:{error}"

    def to_dict(self) -> dict:
        """Validated v2 evidence, including each entry's artifact digest."""
        with self._transaction() as connection:
            self._require_valid(connection)
            entries = [asdict(LedgerEntryV2(**dict(row))) for row in connection.execute(
                "SELECT * FROM ledger_entries ORDER BY sequence")]
            return dict(schema=SCHEMA_ID, entries=entries, head=self._head(connection), count=len(entries))

    def close(self):
        if self._connection is not None:
            if os.getpid() != self._pid:
                raise RuntimeError("Open a new DurableApplicationLedgerV2 in each process")
            self._connection.close()
            self._connection = None

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
