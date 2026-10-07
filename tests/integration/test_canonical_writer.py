"""Canonical writer end to end: authority -> candidate -> atomic publication owned by the pipeline ledger."""
from __future__ import annotations

import pytest

from elpis.pipeline.application import DurableApplicationLedger
from elpis.pipeline.canonical.candidate import construct_candidate
from elpis.pipeline.canonical.publisher import PublicationError
from elpis.structure.grid81.canonical import load_current_grid81

from tests.pipeline.test_canonical_writer_chain import _artifact, _copy_root, _issue


def test_publication_is_owned_by_the_pipeline_ledger_and_replay_is_idempotent(runtime, tmp_path):
    current = _copy_root(tmp_path, "live")
    candidate = tmp_path / "candidate"
    artifact = _artifact()
    before = load_current_grid81(current)
    with DurableApplicationLedger(tmp_path / "publication-ledger.sqlite3") as ledger:
        capability = _issue(current, ledger, artifact, "operator-A")
        constructed = construct_candidate(project_root=current, candidate_root=candidate,
                                          promotion_capability=capability, structural_artifact=artifact)
        authority = runtime.continuity.snapshot()
        kwargs = dict(project_root=current, candidate_root=candidate, ledger=ledger,
                      promotion_capability=capability, lock_path=current / "Canonical")
        receipt = runtime.publish_canonical(**kwargs)
        after = load_current_grid81(current)
        assert receipt.status == "COMMITTED"
        assert after.canonical_digest == constructed.candidate_canonical_digest
        assert receipt.previous_canonical_digest == before.canonical_digest
        assert receipt.resulting_canonical_digest == after.canonical_digest
        assert after.generation_number == 2

        replay = runtime.publish_canonical(**kwargs)
        assert replay.status == "ALREADY_COMMITTED"
        assert replay.publication_receipt_digest == receipt.publication_receipt_digest
        # Publication durability is the ledger's; the runtime duplicates none of it.
        assert runtime.continuity.snapshot() == authority


def test_refused_publication_leaves_canonical_state_and_continuity_unchanged(runtime, tmp_path):
    current = _copy_root(tmp_path, "live")
    other = _copy_root(tmp_path, "other")
    artifact = _artifact()
    before = load_current_grid81(current)
    authority = runtime.continuity.snapshot()
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
    assert runtime.continuity.snapshot() == authority
