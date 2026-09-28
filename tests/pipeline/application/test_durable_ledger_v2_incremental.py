from contextlib import contextmanager
import sqlite3
import threading

import pytest

from elpis.pipeline.application import durable_ledger_v2 as v2
from elpis.pipeline.application.canonical import canonical_digest


V2 = v2.DurableApplicationLedgerV2


def digest(label):
    return canonical_digest({"fixture": label})


def append(ledger, number):
    return ledger.append(
        ledger.head,
        digest(f"receipt-{number}"),
        digest(f"artifact-{number}"),
    )


def test_steady_state_append_avoids_full_verifier(tmp_path):
    path = tmp_path / "incremental.sqlite"
    with V2(path) as ledger:
        statements = []
        ledger._connection.set_trace_callback(statements.append)
        for index in range(12):
            append(ledger, index)
        upper = [statement.upper() for statement in statements]
        assert not any("PRAGMA INTEGRITY_CHECK" in statement for statement in upper)
        assert not any(
            "SELECT * FROM LEDGER_ENTRIES ORDER BY SEQUENCE" in statement
            for statement in upper
        )
        assert ledger.verify_chain() == (True, "valid")


def test_owner_snapshot_advances_after_own_commits(tmp_path):
    path = tmp_path / "owner.sqlite"
    with V2(path) as ledger:
        initial_version = ledger._verified_data_version
        first = append(ledger, 1)
        assert ledger._verified_count == 1
        assert ledger._verified_head == first.entry_digest
        assert ledger._connection.execute(
            "PRAGMA data_version"
        ).fetchone()[0] == initial_version

        second = append(ledger, 2)
        assert ledger._verified_count == 2
        assert ledger._verified_head == second.entry_digest
        assert ledger.verify_chain() == (True, "valid")


def test_valid_external_commit_revalidates_then_preserves_cas(tmp_path):
    path = tmp_path / "two-owners.sqlite"
    first = V2(path)
    second = V2(path)
    try:
        stale = first.head
        append(second, 1)
        statements = []
        first._connection.set_trace_callback(statements.append)
        with pytest.raises(ValueError, match="Stale ledger head"):
            first.append(
                stale,
                digest("first-receipt"),
                digest("first-artifact"),
            )
        assert any(
            "PRAGMA INTEGRITY_CHECK" in statement.upper()
            for statement in statements
        )
        append(first, 2)
        assert first.verify_chain() == (True, "valid")
    finally:
        first.close()
        second.close()


def test_external_corruption_revalidates_before_insert(tmp_path):
    path = tmp_path / "corrupt.sqlite"
    with V2(path) as ledger:
        append(ledger, 1)
        with sqlite3.connect(path) as attacker:
            attacker.execute(
                "UPDATE ledger_entries SET entry_digest = ? "
                "WHERE sequence = 1",
                (digest("tampered-entry"),),
            )

        statements = []
        ledger._connection.set_trace_callback(statements.append)
        with pytest.raises(RuntimeError, match="digest_mismatch_at_sequence:1"):
            ledger.append(
                ledger.head,
                digest("next-receipt"),
                digest("next-artifact"),
            )
        assert not any(
            statement.lstrip().upper().startswith("INSERT")
            for statement in statements
        )


def test_explicit_verify_chain_remains_full_validation(tmp_path):
    path = tmp_path / "verify.sqlite"
    with V2(path) as ledger:
        append(ledger, 1)
        statements = []
        ledger._connection.set_trace_callback(statements.append)
        assert ledger.verify_chain() == (True, "valid")
        assert any(
            "PRAGMA INTEGRITY_CHECK" in statement.upper()
            for statement in statements
        )

class _PostCommitRaceLedger(V2):
    # Expose a deterministic post-commit/pre-return race window for regression.
    def arm_post_commit_race(self, lock_held, attacker_done):
        self._race_lock_held = lock_held
        self._race_attacker_done = attacker_done
        self._race_armed = True

    @contextmanager
    def _transaction(self, *, write=False):
        with super()._transaction(write=write) as connection:
            if write and getattr(self, "_race_armed", False):
                self._race_lock_held.set()
            yield connection
        if write and getattr(self, "_race_armed", False):
            if not self._race_attacker_done.wait(timeout=10):
                raise RuntimeError("attacker did not commit in post-commit window")
            self._race_armed = False


def test_post_commit_foreign_corruption_cannot_be_absorbed_into_verified_snapshot(tmp_path):
    path = tmp_path / "post-commit-race.sqlite"
    ledger = _PostCommitRaceLedger(path)
    attacker_errors = []
    try:
        append(ledger, 1)

        lock_held = threading.Event()
        attacker_done = threading.Event()
        ledger.arm_post_commit_race(lock_held, attacker_done)

        def attack_after_owner_commit():
            try:
                if not lock_held.wait(timeout=10):
                    raise RuntimeError("owner write lock was never observed")
                with sqlite3.connect(path, timeout=10.0, isolation_level=None) as attacker:
                    attacker.execute("BEGIN IMMEDIATE")
                    attacker.execute(
                        "UPDATE ledger_entries SET receipt_digest = ? WHERE sequence = 1",
                        (digest("post-commit-tamper"),),
                    )
                    attacker.commit()
            except BaseException as exc:
                attacker_errors.append(exc)
            finally:
                attacker_done.set()

        thread = threading.Thread(target=attack_after_owner_commit, daemon=True)
        thread.start()

        second = append(ledger, 2)
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert attacker_errors == []
        assert ledger.head == second.entry_digest

        with pytest.raises(RuntimeError, match="Invalid durable v2 ledger:"):
            ledger.append(
                ledger.head,
                digest("third-receipt"),
                digest("third-artifact"),
            )
        assert ledger._connection.execute(
            "SELECT COUNT(*) FROM ledger_entries"
        ).fetchone()[0] == 2
    finally:
        ledger.close()
