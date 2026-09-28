"""Canonical writer end to end: authority -> candidate -> atomic publication -> recorded once."""
from __future__ import annotations

import pytest

from elpis.pipeline.application import DurableApplicationLedger
from elpis.pipeline.canonical.candidate import construct_candidate
from elpis.pipeline.canonical.publisher import PublicationError
from elpis.structure.grid81.canonical import load_current_grid81

from tests.pipeline.test_canonical_writer_chain import _artifact, _copy_root, _issue


def test_publication_is_recorded_and_replay_records_nothing_new(runtime, tmp_path):
    current = _copy_root(tmp_path, "live")
    candidate = tmp_path / "candidate"
    artifact = _artifact()
    before = load_current_grid81(current)
    with DurableApplicationLedger(tmp_path / "publication-ledger.sqlite3") as ledger:
        capability = _issue(current, ledger, artifact, "operator-A")
        constructed = construct_candidate(project_root=current, candidate_root=candidate,
                                          promotion_capability=capability, structural_artifact=artifact)
        assert runtime.history.records() == ()  # candidate construction is not a commit
        kwargs = dict(project_root=current, candidate_root=candidate, ledger=ledger,
                      promotion_capability=capability, lock_path=current / "Canonical")
        receipt, recorded = runtime.publish_canonical(**kwargs)
        after = load_current_grid81(current)
        assert receipt.status == "COMMITTED"
        assert after.canonical_digest == constructed.candidate_canonical_digest
        bindings = dict(recorded.record.bindings)
        assert recorded.record.digest == receipt.publication_receipt_digest
        assert bindings["previous_canonical"] == before.canonical_digest
        assert bindings["resulting_canonical"] == after.canonical_digest
        assert bindings["generation"] == str(after.generation_number) == "2"

        replay, again = runtime.publish_canonical(**kwargs)
        assert replay.status == "ALREADY_COMMITTED"
        assert again == recorded and runtime.history.records() == (recorded,)


def test_refused_publication_leaves_canonical_state_and_history_unchanged(runtime, tmp_path):
    current = _copy_root(tmp_path, "live")
    other = _copy_root(tmp_path, "other")
    artifact = _artifact()
    before = load_current_grid81(current)
    with DurableApplicationLedger(tmp_path / "ledger.sqlite3") as ledger, \
            DurableApplicationLedger(tmp_path / "other-ledger.sqlite3") as other_ledger:
        capability = _issue(current, ledger, artifact, "operator-A")
        construct_candidate(project_root=current, candidate_root=tmp_path / "candidate",
                            promotion_capability=capability, structural_artifact=artifact)
        foreign = _issue(other, other_ledger, artifact, "operator-B")
        with pytest.raises(PublicationError):
            runtime.publish_canonical(project_root=current, candidate_root=tmp_path / "candidate",
                                      ledger=ledger, promotion_capability=foreign,
                                      lock_path=current / "Canonical")
    assert load_current_grid81(current).canonical_digest == before.canonical_digest
    assert runtime.history.records() == ()
