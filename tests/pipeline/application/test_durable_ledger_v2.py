"""Executable v2 qualification. Run with --basetemp inside workspace .qualification_tmp.

Fault injection is confined to the successor module; v1 behavior is never patched.
Spawned writers use pipes and persistent files, with no timing-based race sleeps.
"""
from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import json
import os
from pathlib import Path
import select
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest

import elpis
from elpis.pipeline.application import DurableApplicationLedger as V1
from elpis.pipeline.application.canonical import canonical_digest
from elpis.pipeline.application.ledger import ledger_head_digest
from elpis.pipeline.application import durable_ledger_v2 as v2

V2 = v2.DurableApplicationLedgerV2
# Child processes import exactly the package root this process imported.
_COMPONENT_SRC = Path(elpis.__file__).resolve().parents[1]


def _child_env():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = str(_COMPONENT_SRC)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def digest(label):
    return canonical_digest({"fixture": label})


def byte_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    connection = sqlite3.connect(path)
    try:
        return tuple(connection.execute(f"SELECT * FROM {table} ORDER BY sequence").fetchall()
                     for table in ("ledger_entries", "applied_artifacts"))
    finally:
        connection.close()


def append(ledger, number):
    return ledger.append(ledger.head, digest(f"receipt{number}"), digest(f"artifact{number}"))


def test_identity_genesis_persistence_and_immutable_entry(tmp_path):
    path = tmp_path / "v2.sqlite"
    with V2(path) as ledger:
        assert ledger.head == v2.GENESIS_HEAD != ledger_head_digest([])
        assert ledger.head == canonical_digest({
            "domain": v2.GENESIS_DOMAIN, "schema": v2.SCHEMA_ID,
            "sequence": 0, "previous_head": "", "entries": [],
        })
        assert ledger.is_empty and ledger.verify_chain() == (True, "empty")
        assert not isinstance(ledger, V1)  # No accidental publisher admission.
        assert ledger._connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert ledger._connection.execute("PRAGMA application_id").fetchone()[0] == v2.APPLICATION_ID
        entries = []
        for index in range(8):
            entry = append(ledger, index)
            assert entry.entry_digest == canonical_digest({
                "domain": v2.ENTRY_DOMAIN, "schema": v2.SCHEMA_ID,
                "sequence": index + 1, "previous_head": entry.previous_head,
                "receipt_digest": digest(f"receipt{index}"),
                "artifact_digest": digest(f"artifact{index}"),
            })
            assert dict(ledger._connection.execute(
                "SELECT * FROM ledger_entries WHERE sequence = ?", (index + 1,)).fetchone()) == asdict(entry)
            assert entry.previous_head == (entries[-1]["entry_digest"] if entries else v2.GENESIS_HEAD)
            entries.append(asdict(entry))
            assert ledger.verify_chain() == (True, "valid")
            assert ledger.has_artifact(entry.artifact_digest)
            assert ledger.has_receipt(entry.receipt_digest)
        with pytest.raises(FrozenInstanceError):
            entry.artifact_digest = digest("forged")
        with pytest.raises(ValueError, match="digest mismatch"):
            replace(entry, artifact_digest=digest("forged"))
        expected = dict(schema=v2.SCHEMA_ID, entries=entries, head=entry.entry_digest, count=8)
        assert ledger.to_dict() == expected
        detached = ledger.to_dict()
        detached["entries"][0]["artifact_digest"] = "forged"
        assert ledger.to_dict() == expected
    with V2(path) as reopened:
        assert reopened.head == expected["head"]
        assert reopened.to_dict() == expected
        assert reopened.verify_chain() == (True, "valid")


def test_only_artifact_changes_new_entry_identity_in_equivalent_starting_ledgers(tmp_path):
    entries = []
    historical_heads = []
    for index in (0, 1):
        with V2(tmp_path / f"v2-{index}.sqlite") as ledger:
            entries.append(ledger.append(ledger.head, digest("same-receipt"), digest(f"artifact{index}")))
        with V1(tmp_path / f"v1-{index}.sqlite") as historical:
            historical_heads.append(historical.append(historical.head, digest("same-receipt"),
                                                      digest(f"artifact{index}")).entry_digest)
    left, right = entries
    assert (left.sequence, left.previous_head, left.receipt_digest) == (
        right.sequence, right.previous_head, right.receipt_digest)
    assert left.artifact_digest != right.artifact_digest
    assert left.entry_digest != right.entry_digest
    assert historical_heads[0] == historical_heads[1]  # Frozen v1 interpretation.


@pytest.mark.parametrize("field", ["previous_head", "receipt_digest", "artifact_digest"])
@pytest.mark.parametrize("value", ["", "a" * 63, "a" * 65, "A" * 64, "g" * 64,
                                  " " + "a" * 63, "+" + "a" * 63, "１" * 64,
                                  "a" * 63 + "\n", None, 1, True, b"a" * 64])
def test_precise_digest_contract_rejects_ambiguity_atomically(tmp_path, field, value):
    path = tmp_path / "v2.sqlite"
    with V2(path) as ledger:
        arguments = dict(previous_head=ledger.head, receipt_digest=digest("receipt"),
                         artifact_digest=digest("artifact"))
        arguments[field] = value
        before = rows(path), ledger.head, byte_digest(path)
        with pytest.raises((ValueError, TypeError)):
            ledger.append(**arguments)
        assert (rows(path), ledger.head, byte_digest(path)) == before
        assert ledger.verify_chain() == (True, "empty")


def test_no_artifact_sentinel_or_digest_subclass(tmp_path):
    class DigestSubclass(str):
        pass
    with V2(tmp_path / "v2.sqlite") as ledger:
        with pytest.raises(TypeError):
            ledger.append(ledger.head, digest("receipt"))
        with pytest.raises(TypeError):
            ledger.append(ledger.head, digest("receipt"), DigestSubclass(digest("artifact")))
        with pytest.raises(ValueError):
            ledger.has_artifact("")
        with pytest.raises(ValueError):
            ledger.has_receipt("")


def test_v2_receipt_and_artifact_queries_are_semantically_distinct(tmp_path):
    with V2(tmp_path / "v2.sqlite") as ledger:
        receipt = digest("receipt-query")
        artifact = digest("artifact-query")
        ledger.append(ledger.head, receipt, artifact)

        assert receipt != artifact
        assert ledger.has_artifact(artifact)
        assert not ledger.has_artifact(receipt)
        assert ledger.has_receipt(receipt)
        assert not ledger.has_receipt(artifact)


@pytest.mark.parametrize("sequence", [0, -1, True, 1.0, "1", 2**63])
def test_entry_sequence_contract(sequence):
    with pytest.raises(ValueError, match="sequence"):
        v2.entry_digest_v2(sequence, v2.GENESIS_HEAD, digest("receipt"), digest("artifact"))


@pytest.mark.parametrize("seeded", [False, True])
@pytest.mark.parametrize("mode", ["DELETE", "WAL"])
def test_v2_rejects_v1_without_byte_or_logical_mutation(tmp_path, seeded, mode):
    path = tmp_path / "historical.sqlite"
    with V1(path) as historical:
        if seeded:
            historical.append(historical.head, "historical receipt", "historical artifact")
        head = historical.head
        verification = historical.verify_chain()
        evidence = historical.to_dict()
    # Hold a WAL writer open to ensure v2 rejection cannot checkpoint or switch
    # the historical journal mode. Include committed WAL bytes in the comparison.
    keeper = sqlite3.connect(path)
    try:
        assert keeper.execute(f"PRAGMA journal_mode = {mode}").fetchone()[0] == mode.lower()
        keeper.execute("PRAGMA user_version = 1")
        keeper.commit()
        wal = Path(str(path) + "-wal")
        before = byte_digest(path), rows(path), wal.read_bytes() if wal.exists() else None
        with pytest.raises(RuntimeError, match="unsupported_ledger_schema"):
            V2(path)
        after = byte_digest(path), rows(path), wal.read_bytes() if wal.exists() else None
        assert after == before
        assert keeper.execute("PRAGMA journal_mode").fetchone()[0] == mode.lower()
    finally:
        keeper.close()
    with V1(path) as historical:
        assert historical.head == head
        assert historical.verify_chain() == verification
        assert historical.to_dict() == evidence


@pytest.mark.parametrize("seeded", [False, True])
def test_existing_v1_reader_rejects_v2_without_reinterpretation(tmp_path, seeded):
    path = tmp_path / "v2.sqlite"
    with V2(path) as ledger:
        if seeded:
            append(ledger, 1)
        expected = ledger.to_dict()
    before = byte_digest(path), rows(path)
    with pytest.raises(RuntimeError, match="unsupported_ledger_schema"):
        V1(path)
    assert (byte_digest(path), rows(path)) == before
    with V2(path) as ledger:
        assert ledger.to_dict() == expected
        assert ledger.verify_chain()[0]


def test_retagging_v1_entries_does_not_turn_them_into_v2_hashes(tmp_path):
    path = tmp_path / "retagged.sqlite"
    with V1(path) as ledger:
        ledger.append(ledger.head, digest("receipt"), digest("artifact"))
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA user_version = {v2.SCHEMA_VERSION}")
        connection.execute(f"PRAGMA application_id = {v2.APPLICATION_ID}")
    before = byte_digest(path)
    with pytest.raises(RuntimeError, match="chain_break"):
        V2(path)
    assert byte_digest(path) == before


def test_schema_is_checked_through_read_only_connection_before_write_open(tmp_path, monkeypatch):
    path = tmp_path / "v1.sqlite"
    with V1(path):
        pass
    real_connect = sqlite3.connect
    calls = []
    def guarded(database, **kwargs):
        calls.append((database, kwargs))
        assert kwargs.get("uri") is True and database.endswith("?mode=ro")
        return real_connect(database, **kwargs)
    monkeypatch.setattr(v2, "sqlite3", SimpleNamespace(connect=guarded, Row=sqlite3.Row))
    with pytest.raises(RuntimeError, match="unsupported_ledger_schema"):
        V2(path)
    assert len(calls) == 1


def assert_corrupt_rejected(path, ledger, reason):
    before = rows(path), byte_digest(path)
    assert ledger.verify_chain() == (False, reason)
    statements = []
    ledger._connection.set_trace_callback(statements.append)
    with pytest.raises(RuntimeError, match=reason):
        ledger.append(digest("head"), digest("unused-receipt"), digest("unused-artifact"))
    assert statements[0] == "BEGIN IMMEDIATE"
    assert not any(statement.startswith("INSERT") for statement in statements)
    assert not ledger._connection.in_transaction
    assert (rows(path), byte_digest(path)) == before
    ledger.close()
    with pytest.raises(RuntimeError, match=reason):
        V2(path)
    assert (rows(path), byte_digest(path)) == before


@pytest.mark.parametrize("field,reason", [
    ("artifact_digest", "digest_mismatch_at_sequence:1"),
    ("receipt_digest", "digest_mismatch_at_sequence:1"),
    ("previous_head", "chain_break_at_sequence:1"),
    ("entry_digest", "digest_mismatch_at_sequence:1"),
    ("sequence", "invalid_sequence:2"),
])
def test_tampered_entry_detected_even_with_matching_artifact_associations(tmp_path, field, reason):
    path = tmp_path / "v2.sqlite"
    with V2(path) as ledger:
        append(ledger, 1)
        append(ledger, 2)
        with sqlite3.connect(path) as connection:
            value = 7 if field == "sequence" else digest("tampered")
            connection.execute(f"UPDATE ledger_entries SET {field} = ? WHERE sequence = 1", (value,))
            if field in ("sequence", "receipt_digest", "artifact_digest"):
                connection.execute(f"UPDATE applied_artifacts SET {field} = ? WHERE sequence = 1", (value,))
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert_corrupt_rejected(path, ledger, reason)


@pytest.mark.parametrize("attack", ["link", "missing_association", "wrong_artifact",
                                    "wrong_receipt", "orphan", "entry_artifact_only"])
def test_link_and_artifact_association_tampering(tmp_path, attack):
    path = tmp_path / "v2.sqlite"
    with V2(path) as ledger:
        append(ledger, 1)
        append(ledger, 2)
        with sqlite3.connect(path) as connection:
            sql, args = {
                "link": ("UPDATE ledger_entries SET previous_head = ? WHERE sequence = 2", (v2.GENESIS_HEAD,)),
                "missing_association": ("DELETE FROM applied_artifacts WHERE sequence = 1", ()),
                "wrong_artifact": ("UPDATE applied_artifacts SET artifact_digest = ? WHERE sequence = 1", (digest("bad"),)),
                "wrong_receipt": ("UPDATE applied_artifacts SET receipt_digest = ? WHERE sequence = 1", (digest("bad"),)),
                "orphan": ("INSERT INTO applied_artifacts VALUES (?, 99, ?)", (digest("orphan"), digest("receipt"))),
                "entry_artifact_only": ("UPDATE ledger_entries SET artifact_digest = ? WHERE sequence = 1", (digest("bad"),)),
            }[attack]
            connection.execute(sql, args)
        reason = "chain_break_at_sequence:2" if attack == "link" else "artifact_association_mismatch"
        assert_corrupt_rejected(path, ledger, reason)


@pytest.mark.parametrize("field,value", [("artifact_digest", ""), ("receipt_digest", "A" * 64),
                                        ("artifact_digest", b"a" * 64)])
def test_persisted_noncanonical_digest_is_not_accepted(tmp_path, field, value):
    path = tmp_path / "v2.sqlite"
    with V2(path) as ledger:
        append(ledger, 1)
        with sqlite3.connect(path) as connection:
            connection.execute(f"UPDATE ledger_entries SET {field} = ?", (value,))
            if value == "":
                connection.execute("DELETE FROM applied_artifacts")
            else:
                connection.execute(f"UPDATE applied_artifacts SET {field} = ?", (value,))
        assert_corrupt_rejected(path, ledger, "invalid_entry_digest_format_at_sequence:1")


def test_database_integrity_check_detects_corrupt_index_root(tmp_path):
    path = tmp_path / "v2.sqlite"
    with V2(path) as ledger:
        append(ledger, 1)
        append(ledger, 2)
        with sqlite3.connect(path) as connection:
            version = connection.execute("PRAGMA schema_version").fetchone()[0]
            # Keep SQL declarations intact but swap artifact/receipt index
            # b-trees. Structural introspection alone cannot see
            # the wrong keys; SQLite's integrity check must reject the contents.
            root = connection.execute("SELECT rootpage FROM sqlite_master WHERE name = ?",
                                      ("sqlite_autoindex_ledger_entries_1",)).fetchone()[0]
            artifact_root = connection.execute("SELECT rootpage FROM sqlite_master WHERE name = ?",
                                               ("unique_applied_artifact",)).fetchone()[0]
            connection.execute("PRAGMA writable_schema = ON")
            connection.execute("UPDATE sqlite_master SET rootpage = CASE name WHEN ? THEN ? ELSE ? END "
                               "WHERE name IN (?, ?)",
                               ("unique_applied_artifact", root, artifact_root,
                                "unique_applied_artifact", "sqlite_autoindex_ledger_entries_1"))
            connection.execute("PRAGMA writable_schema = OFF")
            connection.execute(f"PRAGMA schema_version = {version + 1}")
        assert_corrupt_rejected(path, ledger, "database_integrity_failure")


def test_sqlite_itself_enforces_replay_uniqueness(tmp_path):
    with V2(tmp_path / "v2.sqlite") as ledger:
        entry = append(ledger, 1)
        for receipt, artifact in ((entry.receipt_digest, digest("new-artifact")),
                                  (digest("new-receipt"), entry.artifact_digest)):
            with pytest.raises(sqlite3.IntegrityError):
                ledger._connection.execute("INSERT INTO ledger_entries VALUES (?, ?, ?, ?, ?)",
                    (2, entry.entry_digest, receipt,
                     v2.entry_digest_v2(2, entry.entry_digest, receipt, artifact), artifact))
        assert ledger.to_dict()["count"] == 1
        assert ledger.verify_chain() == (True, "valid")


def test_stale_cas_duplicate_receipt_and_artifact_leave_no_rows(tmp_path):
    path = tmp_path / "v2.sqlite"
    with V2(path) as first, V2(path) as second:
        stale = second.head
        committed = append(first, 1)
        before = rows(path), second.head, byte_digest(path)
        for head, receipt, artifact, reason in (
            (stale, digest("receipt2"), digest("artifact2"), "Stale ledger head"),
            (second.head, committed.receipt_digest, digest("artifact2"), "Duplicate receipt"),
            (second.head, digest("receipt2"), committed.artifact_digest, "Duplicate artifact"),
        ):
            with pytest.raises(ValueError, match=reason):
                second.append(head, receipt, artifact)
            assert (rows(path), second.head, byte_digest(path)) == before
        append(second, 2)
        assert first.verify_chain() == (True, "valid")


@pytest.mark.parametrize("fault", ["before_association", "after_association", "post_verify", "commit"])
def test_insert_verification_and_commit_failure_roll_back(tmp_path, monkeypatch, fault):
    class FaultConnection(sqlite3.Connection):
        armed = False
        reached = False

        def execute(self, sql, parameters=()):
            if self.armed and sql.startswith("INSERT INTO applied_artifacts"):
                assert self.in_transaction
                self.reached = True
                if fault == "before_association":
                    raise sqlite3.IntegrityError("injected before association")
                result = super().execute(sql, parameters)
                if fault == "after_association":
                    raise sqlite3.IntegrityError("injected after association")
                if fault == "post_verify":
                    super().execute("UPDATE ledger_entries SET entry_digest = ? WHERE sequence = 2",
                                    (digest("corruption"),))
                return result
            return super().execute(sql, parameters)

        def commit(self):
            if self.armed and fault == "commit":
                self.reached = True
                raise sqlite3.OperationalError("injected commit failure")
            return super().commit()

    def connect(*args, **kwargs):
        return sqlite3.connect(*args, **kwargs, factory=FaultConnection)
    monkeypatch.setattr(v2, "sqlite3", SimpleNamespace(
        connect=connect, Row=sqlite3.Row, DatabaseError=sqlite3.DatabaseError))
    path = tmp_path / "v2.sqlite"
    with V2(path) as ledger:
        append(ledger, 1)
        before = rows(path), ledger.head, byte_digest(path)
        ledger._connection.armed = True
        with pytest.raises((sqlite3.DatabaseError, RuntimeError), match="injected|digest_mismatch"):
            append(ledger, 2)
        assert ledger._connection.reached
        assert not ledger._connection.in_transaction
        assert (rows(path), ledger.head, byte_digest(path)) == before
        ledger._connection.armed = False
        append(ledger, 2)
    with V2(path) as ledger:
        assert ledger.to_dict()["count"] == 2
        assert ledger.verify_chain() == (True, "valid")


def recreate_table(connection, table, old, new):
    statement = connection.execute("SELECT sql FROM sqlite_master WHERE name = ?", (table,)).fetchone()[0]
    assert old in statement
    indexes = [row[0] for row in connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND tbl_name = ? AND sql IS NOT NULL", (table,))]
    columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
    records = connection.execute(f'SELECT * FROM "{table}"').fetchall()
    connection.execute(f'DROP TABLE "{table}"')
    connection.execute(statement.replace(old, new, 1))
    connection.executemany(f'INSERT INTO "{table}" ({", ".join(columns)}) VALUES ({", ".join("?" for _ in columns)})', records)
    for statement in indexes:
        connection.execute(statement)


@pytest.mark.parametrize("mutation,reason", [
    (("PRAGMA user_version = 1",), "unsupported_ledger_schema"),
    (("PRAGMA user_version = 99",), "unsupported_ledger_schema"),
    (("PRAGMA application_id = 0",), "unsupported_ledger_schema"),
    (("DROP INDEX unique_applied_artifact",), "schema_mismatch:objects"),
    (("CREATE TABLE extra (value TEXT)",), "schema_mismatch:objects"),
    (("CREATE VIEW extra AS SELECT * FROM ledger_entries",), "schema_mismatch:objects"),
    (("CREATE TRIGGER extra AFTER INSERT ON ledger_entries BEGIN SELECT 1; END",), "schema_mismatch:objects"),
    (("CREATE INDEX extra ON ledger_entries(previous_head)",), "schema_mismatch:objects"),
    (("ledger_entries", "previous_head TEXT NOT NULL", "extra TEXT, previous_head TEXT NOT NULL"), "schema_mismatch:columns"),
    (("ledger_entries", "entry_digest TEXT NOT NULL", "entry_digest BLOB NOT NULL"), "schema_mismatch:columns"),
    (("ledger_entries", "entry_digest TEXT NOT NULL", "entry_digest TEXT"), "schema_mismatch:columns"),
    (("ledger_entries", "sequence INTEGER PRIMARY KEY", "sequence INTEGER"), "schema_mismatch:columns"),
    (("ledger_entries", "receipt_digest TEXT NOT NULL UNIQUE", "receipt_digest TEXT NOT NULL"), "schema_mismatch:indexes"),
    (("ledger_entries", "UNIQUE(sequence, receipt_digest, artifact_digest)", "CHECK(sequence > 0)"), "schema_mismatch:indexes"),
    (("ledger_entries", "CHECK(sequence > 0)", "CHECK(sequence >= 0)"), "schema_mismatch:declarations"),
    (("ledger_entries", "TEXT NOT NULL UNIQUE", "TEXT NOT NULL UNIQUE ON CONFLICT REPLACE"), "schema_mismatch:declarations"),
    (("applied_artifacts", "TEXT PRIMARY KEY NOT NULL", "TEXT NOT NULL"), "schema_mismatch:columns"),
    (("applied_artifacts", "sequence INTEGER NOT NULL", "sequence INT NOT NULL"), "schema_mismatch:columns"),
    (("applied_artifacts", "sequence INTEGER NOT NULL UNIQUE", "sequence INTEGER NOT NULL"), "schema_mismatch:indexes"),
    (("applied_artifacts", "receipt_digest TEXT NOT NULL UNIQUE", "receipt_digest TEXT NOT NULL"), "schema_mismatch:indexes"),
    (("applied_artifacts", "CHECK(artifact_digest != '')", "CHECK(1)"), "schema_mismatch:declarations"),
    (("applied_artifacts", "REFERENCES ledger_entries(", "REFERENCES wrong_parent("), "schema_mismatch:foreign_keys"),
    (("applied_artifacts", "REFERENCES ledger_entries(\n                    sequence, receipt_digest, artifact_digest)",
      "REFERENCES ledger_entries(sequence, receipt_digest, artifact_digest) DEFERRABLE INITIALLY DEFERRED"), "schema_mismatch:declarations"),
])
def test_schema_authority_fails_closed_without_mutation(tmp_path, mutation, reason):
    path = tmp_path / "v2.sqlite"
    with V2(path) as ledger:
        append(ledger, 1)
        with sqlite3.connect(path) as connection:
            if len(mutation) == 1:
                connection.execute(mutation[0])
            else:
                recreate_table(connection, *mutation)
        assert_corrupt_rejected(path, ledger, reason)


@pytest.mark.parametrize("replacement,reason", [
    ("CREATE INDEX unique_applied_artifact ON ledger_entries(artifact_digest) WHERE artifact_digest != ''", "indexes"),
    ("CREATE UNIQUE INDEX unique_applied_artifact ON ledger_entries(artifact_digest)", "indexes"),
    ("CREATE UNIQUE INDEX unique_applied_artifact ON ledger_entries(receipt_digest) WHERE artifact_digest != ''", "indexes"),
    ("CREATE UNIQUE INDEX unique_applied_artifact ON ledger_entries(artifact_digest COLLATE NOCASE) WHERE artifact_digest != ''", "indexes"),
    ("CREATE UNIQUE INDEX unique_applied_artifact ON ledger_entries(artifact_digest DESC) WHERE artifact_digest != ''", "indexes"),
    ("CREATE UNIQUE INDEX unique_applied_artifact ON ledger_entries(artifact_digest) WHERE artifact_digest = ''", "declarations"),
    ("CREATE UNIQUE INDEX unique_applied_artifact ON ledger_entries(artifact_digest) WHERE artifact_digest != '' AND sequence > 100", "declarations"),
])
def test_index_definition_attacks(tmp_path, replacement, reason):
    path = tmp_path / "v2.sqlite"
    with V2(path) as ledger:
        append(ledger, 1)
        with sqlite3.connect(path) as connection:
            connection.execute("DROP INDEX unique_applied_artifact")
            connection.execute(replacement)
        assert_corrupt_rejected(path, ledger, f"schema_mismatch:{reason}")


def test_schema_formatting_is_not_identity_but_temp_shadowing_is_rejected(tmp_path):
    path = tmp_path / "v2.sqlite"
    with V2(path) as ledger:
        append(ledger, 1)
        expected = ledger.to_dict()
        with sqlite3.connect(path) as connection:
            recreate_table(connection, "applied_artifacts", "CREATE TABLE applied_artifacts",
                           'create /* formatting */ table "applied_artifacts"')
        assert ledger.verify_chain() == (True, "valid")
        assert ledger.to_dict() == expected
        ledger._connection.execute("CREATE TEMP TABLE ledger_entries (sequence INTEGER)")
        assert ledger.verify_chain() == (False, "schema_mismatch:objects")
        with pytest.raises(RuntimeError, match="schema_mismatch:objects"):
            ledger.append(expected["head"], digest("receipt2"), digest("artifact2"))
    with V2(path) as ledger:
        assert ledger.to_dict() == expected


def test_durability_foreign_keys_and_lifecycle(tmp_path):
    path = tmp_path / "v2.sqlite"
    with pytest.raises(LookupError):
        with V2(path) as ledger:
            assert ledger._connection.execute("PRAGMA synchronous").fetchone()[0] == 2
            assert ledger._connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
            assert ledger._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            with pytest.raises(sqlite3.IntegrityError):
                ledger._connection.execute("INSERT INTO applied_artifacts VALUES (?, 42, ?)",
                                           (digest("orphan"), digest("receipt")))
            append(ledger, 1)
            raise LookupError("caller failure after successful durable commit")
    ledger.close()
    with pytest.raises(RuntimeError, match="closed"):
        ledger.to_dict()
    with V2(path) as ledger:
        assert ledger.verify_chain() == (True, "valid")


@pytest.mark.parametrize("path", ["", ":memory:"])
def test_nonpersistent_paths_rejected(path):
    with pytest.raises(ValueError, match="persistent"):
        V2(path)


def test_canonical_location_and_unrelated_database_rejected(tmp_path):
    with pytest.raises(ValueError, match="Canonical/Grid81"):
        V2(tmp_path / "Canonical" / "Grid81" / "ledger.sqlite")
    path = tmp_path / "unrelated.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
    before = byte_digest(path)
    with pytest.raises(RuntimeError, match="unsupported_ledger_schema"):
        V2(path)
    assert byte_digest(path) == before


def test_storage_replacement_and_removed_storage_rejected(tmp_path):
    path = tmp_path / "v2.sqlite"
    with V2(path) as ledger:
        append(ledger, 1)
        original_identity = ledger.storage_identity
        path.rename(tmp_path / "original.sqlite")
        with pytest.raises(RuntimeError, match="removed"):
            ledger.verify_chain()
        with V2(path) as replacement:
            assert replacement.storage_identity != original_identity
            for action in (lambda: ledger.head, lambda: ledger.storage_identity,
                           ledger.verify_chain, ledger.to_dict, lambda: append(ledger, 2)):
                with pytest.raises(RuntimeError, match="replaced"):
                    action()
            assert replacement.is_empty
    with V2(tmp_path / "original.sqlite") as original:
        assert original.to_dict()["count"] == 1


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX process ownership check")
def test_inherited_connection_cannot_be_used_in_child(tmp_path):
    with V2(tmp_path / "v2.sqlite") as ledger:
        pid = os.fork()
        if pid == 0:
            try:
                ledger.head
            except RuntimeError as error:
                os._exit(0 if "each process" in str(error) else 3)
            os._exit(2)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        append(ledger, 1)


def _race_worker(path, identity):
    with V2(path) as ledger:
        previous = ledger.head
        print(json.dumps({"pid": os.getpid(), "head": previous}), flush=True)
        assert sys.stdin.readline() == "go\n"
        try:
            entry = ledger.append(previous, digest(f"receipt{identity}"), digest(f"artifact{identity}"))
        except ValueError as error:
            result = dict(outcome="rejected", reason=str(error))
        else:
            result = dict(outcome="committed", entry=asdict(entry))
        result.update(evidence=ledger.to_dict(), verification=ledger.verify_chain())
        print(json.dumps(result), flush=True)


def _crash_worker(path, point):
    class CrashLedger(V2):
        armed = False

        def _verify_new_entry_association(self, connection, entry):
            super()._verify_new_entry_association(connection, entry)
            if self.armed and point == "before_commit" and entry.sequence == 2:
                os._exit(23)

    with CrashLedger(path) as ledger:
        # Force dirty pages to spill before process death: the surviving journal
        # must be recovered, not simply ignored because all writes stayed in RAM.
        ledger._connection.execute("PRAGMA cache_size = 1")
        ledger._connection.execute("PRAGMA cache_spill = ON")
        CrashLedger.armed = True
        append(ledger, 2)
        os._exit(24)  # Successful commit, before connection close.


@pytest.mark.parametrize("point", ["before_commit", "after_commit"])
def test_process_death_recovers_or_retains_exact_durable_commit(tmp_path, point):
    path = tmp_path / "v2.sqlite"
    with V2(path) as ledger:
        append(ledger, 1)
        before = ledger.to_dict()
    cp = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                         "crash", str(path), point], capture_output=True, text=True,
                        timeout=40, env=_child_env())
    assert cp.returncode == (23 if point == "before_commit" else 24), cp.stderr
    if point == "before_commit":
        assert Path(str(path) + "-journal").exists()
        # Demonstrate that read-only SQLite cannot recover this actual fixture.
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                connection.execute("SELECT * FROM ledger_entries").fetchall()
        finally:
            connection.close()
    with V2(path) as ledger:
        assert ledger.verify_chain() == (True, "valid")
        if point == "before_commit":
            assert ledger.to_dict() == before
            append(ledger, 2)
        else:
            assert ledger.to_dict()["count"] == 2
            assert ledger.head == v2.entry_digest_v2(2, before["head"], digest("receipt2"), digest("artifact2"))
            with pytest.raises(ValueError, match="Duplicate artifact"):
                ledger.append(ledger.head, digest("new-receipt"), digest("artifact2"))


@pytest.mark.parametrize("seeded", [False, True])
def test_independent_process_cas_contention(tmp_path, seeded):
    path = tmp_path / "v2.sqlite"
    if seeded:
        with V2(path) as ledger:
            append(ledger, "seed")
    processes = []
    try:
        for identity in range(3):
            processes.append(subprocess.Popen(
                [sys.executable, "-B", str(Path(__file__).resolve()), "race", str(path), str(identity)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=_child_env()))
        ready = []
        for process in processes:
            assert select.select([process.stdout], [], [], 30)[0], "writer did not reach barrier"
            ready.append(json.loads(process.stdout.readline()))
        assert len({item["pid"] for item in ready} | {os.getpid()}) == 4
        assert len({item["head"] for item in ready}) == 1
        for process in processes:
            process.stdin.write("go\n")
            process.stdin.flush()
        results = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=40)
            assert process.returncode == 0, stderr
            results.append(json.loads(stdout))
        winners = [result for result in results if result["outcome"] == "committed"]
        losers = [result for result in results if result["outcome"] == "rejected"]
        assert len(winners) == 1 and len(losers) == 2
        assert all(result["reason"].startswith("Stale ledger head:") for result in losers)
        assert all(result["verification"] == [True, "valid"] for result in results)
        assert all(result["evidence"] == winners[0]["evidence"] for result in results)
        with V2(path) as ledger:
            assert ledger.to_dict() == winners[0]["evidence"]
            assert ledger.to_dict()["count"] == 1 + int(seeded)
            winner = winners[0]["entry"]
            assert ledger.head == v2.entry_digest_v2(winner["sequence"], winner["previous_head"],
                                                     winner["receipt_digest"], winner["artifact_digest"])
            for index, result in enumerate(results):
                if result["outcome"] == "rejected":
                    append(ledger, index)  # Loser left no receipt/artifact reservation.
            assert ledger.verify_chain() == (True, "valid")
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)


if __name__ == "__main__":
    if sys.argv[1] == "race":
        _race_worker(sys.argv[2], sys.argv[3])
    else:
        assert sys.argv[1] == "crash"
        _crash_worker(sys.argv[2], sys.argv[3])
