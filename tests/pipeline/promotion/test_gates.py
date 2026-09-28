"""Tests for promotion gates."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from elpis.pipeline.promotion.source_binding import build_source_chain
from elpis.pipeline.promotion.gates import (
    evaluate_gates,
    GATE_DEFINITIONS,
    first_failure,
    _check_replay_protection,
)
from elpis.pipeline.promotion.canonical import PhaseEvidence, SourceChain
from elpis.pipeline.promotion.decision import make_decision, DECISION_NOT_READY

def _h(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _empty_chain(tmp_path):
    """A structurally complete source chain whose evidence directories are empty."""
    phases = {}
    for phase_id, key in (("G5.3B.1", "g53b1"), ("G5.3C", "g53c"), ("G5.3D", "g53d")):
        directory = tmp_path / key
        directory.mkdir()
        phases[key] = PhaseEvidence(
            phase_id=phase_id,
            source_directory=str(directory),
            manifest_path=str(directory / "RAW_EVIDENCE_MANIFEST.json"),
            manifest_digest=_h(f"{phase_id}:manifest"),
            disposition="PASS",
            evidence_files=(("evidence.json", _h(f"{phase_id}:evidence"), 1),),
        )
    return SourceChain(**phases)


def test_gate_count(tmp_path):
    results = evaluate_gates(_empty_chain(tmp_path))
    assert len(results) == 19


def test_absent_evidence_fails_closed(tmp_path):
    chain = _empty_chain(tmp_path)
    results = evaluate_gates(chain)
    assert first_failure(results) is not None
    assert not any(r.passed for r in results[:1])
    assert make_decision(results, chain).decision == DECISION_NOT_READY


def test_no_unestablished_phase_disposition_gate():
    assert all(gate_id != "GATE_ALL_PHASE_DISPOSITIONS_PRESENT" for gate_id, _, _ in GATE_DEFINITIONS)


def test_gate_order_deterministic(tmp_path):
    chain = _empty_chain(tmp_path)
    r1 = evaluate_gates(chain)
    r2 = evaluate_gates(chain)
    for i in range(len(r1)):
        assert r1[i].gate_id == r2[i].gate_id
        assert r1[i].gate_ordinal == r2[i].gate_ordinal


def test_gate_digests_deterministic(tmp_path):
    chain = _empty_chain(tmp_path)
    r1 = evaluate_gates(chain)
    r2 = evaluate_gates(chain)
    for i in range(len(r1)):
        assert r1[i].digest == r2[i].digest


def test_established_gate_set_excludes_unstructured_disposition():
    assert len(GATE_DEFINITIONS) == 19
    assert all(
        gate_id != "GATE_ALL_PHASE_DISPOSITIONS_PRESENT"
        for gate_id, _, _ in GATE_DEFINITIONS
    )

def test_missing_phase_rejected(tmp_path):
    config = {key: str(tmp_path / key) for key in ("g53b1_directory", "g53c_directory", "g53d_directory")}
    with pytest.raises(FileNotFoundError):
        build_source_chain(config)


def test_unresolved_source_directory_rejected():
    config = {
        "g53b1_directory": "${ELPIS_TEST_UNSET_EVIDENCE_ROOT}/g53b1",
        "g53c_directory": "/unused",
        "g53d_directory": "/unused",
    }
    with pytest.raises(ValueError, match="SOURCE_DIRECTORY_UNRESOLVED"):
        build_source_chain(config)


def test_gate_ids_unique():
    ids = [gid for gid, _, _ in GATE_DEFINITIONS]
    assert len(ids) == len(set(ids))


def test_gate_ordinal_ascending():
    ordinals = [ord_ for _, ord_, _ in GATE_DEFINITIONS]
    for i in range(len(ordinals) - 1):
        assert ordinals[i] < ordinals[i + 1]

def _replay_chain(tmp_path, payload):
    source = tmp_path / "g53b1"
    source.mkdir()
    (source / "G53B_REPLAY_AUDIT.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return SimpleNamespace(
        g53b1=SimpleNamespace(source_directory=str(source))
    )


def test_replay_protection_missing_status_fields_fails_closed(tmp_path):
    chain = _replay_chain(tmp_path, {"audit": "present"})
    assert _check_replay_protection(chain) is False


def test_replay_protection_explicit_legacy_true_is_accepted(tmp_path):
    chain = _replay_chain(tmp_path, {"replay_protection": True})
    assert _check_replay_protection(chain) is True


def test_replay_protection_conflicting_status_fields_fail_closed(tmp_path):
    chain = _replay_chain(
        tmp_path,
        {
            "replay_protection_qualified": True,
            "replay_protection": False,
        },
    )
    assert _check_replay_protection(chain) is False

def test_replay_protection_non_mapping_payloads_fail_closed(tmp_path):
    payloads = ([], "qualified", 1, True, None)
    for index, payload in enumerate(payloads):
        case = tmp_path / f"case-{index}"
        case.mkdir()
        chain = _replay_chain(case, payload)
        assert _check_replay_protection(chain) is False

