"""Authority regression tests using existing producer contracts, without reports."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from elpis.pipeline.consumption.artifact import create_structural_influence_artifact
from elpis.pipeline.consumption.policy import create_compiler_contract
from elpis.pipeline.application.artifact import create_application_receipt
from elpis.pipeline.promotion.canonical import PhaseEvidence, SourceChain
from elpis.pipeline.promotion import gates
from elpis.pipeline.promotion.source_binding import census_g53b1

ARTIFACTS = "G53B_STRUCTURAL_INFLUENCE_ARTIFACT_INVENTORY.jsonl"
RECEIPTS = "G53C_APPLICATION_RECEIPTS.jsonl"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def phase(directory, phase_id, records):
    directory.mkdir(exist_ok=True)
    entries = {}
    for name, rows in records.items():
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
        (directory / name).write_bytes(payload)
        entries[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
    manifest_name = "G53B_RAW_EVIDENCE_MANIFEST.json" if phase_id == "G5.3B.1" else "RAW_EVIDENCE_MANIFEST.json"
    manifest = directory / manifest_name
    manifest.write_text(json.dumps({"evidence_files": entries}, sort_keys=True))
    return PhaseEvidence(
        phase_id, str(directory), str(manifest),
        hashlib.sha256(manifest.read_bytes()).hexdigest(), "",
        tuple((name, meta["sha256"], meta["size"]) for name, meta in sorted(entries.items())),
    )


@pytest.fixture
def authority(tmp_path):
    artifacts = []
    receipts = []
    for index in range(2):
        capability = {key: digest([index, key]) for key in (
            "capability_digest", "capability_semantic_digest", "source_request_digest",
            "source_adjudication_record_digest", "source_proposal_set_digest",
        )}
        capability["authorized_proposal_digests"] = [digest(["proposal", index])]
        contract = create_compiler_contract()
        request = {
            "consumer_contract_digest": contract["compiler_contract_digest"],
            "consumption_request_digest": digest(["request", index]),
            "logical_tick": index,
        }
        artifact = create_structural_influence_artifact(capability, {}, request, contract)
        artifacts.append(artifact)
        receipts.append(create_application_receipt(
            artifact["artifact_digest"], capability["capability_digest"],
            "APPLICATION_ACCEPTED", digest(["state", index]), digest(["state", index + 1]),
            digest(["ledger", index]), digest(["ledger", index + 1]), "STRUCTURAL_INFLUENCE_COMPILER_V1",
        ))

    def seal():
        b = phase(tmp_path / "b", "G5.3B.1", {ARTIFACTS: artifacts})
        c = phase(tmp_path / "c", "G5.3C", {RECEIPTS: receipts})
        c = replace(c,
                    artifact_digest=":".join(sorted(r["artifact_digest"] for r in receipts)),
                    capability_digest=":".join(sorted(r["capability_digest"] for r in receipts)))
        d = phase(tmp_path / "d", "G5.3D", {})
        return SourceChain(b, c, d)
    return artifacts, receipts, seal


def result(chain, gate_id):
    return next(r for r in gates.evaluate_gates(chain) if r.gate_id == gate_id)


def test_positive_identity_qualification(authority):
    _, _, seal = authority
    chain = seal()
    assert gates._check_artifact_identity(chain)
    assert gates._check_capability_identity(chain)
    assert gates._check_compiler_identity(chain)
    assert gates._check_no_executable_authority(chain)


@pytest.mark.parametrize("value", [
    "", "FAILED", "G53B_FAILED", "G53C_FAILED", "G53D_FAILED", "arbitrary",
    "PASS", "QUALIFIED", "SEALED", "CONSUMPTION_ACCEPTED", "APPLICATION_ACCEPTED",
])
def test_phase_disposition_strings_are_not_promotion_authority(authority, value):
    _, _, seal = authority
    baseline = seal()
    expected = [(r.gate_id, r.passed, r.rejection_code) for r in gates.evaluate_gates(baseline)]
    changed = replace(baseline, **{key: replace(getattr(baseline, key), disposition=value) for key in ("g53b1", "g53c", "g53d")})
    observed = [(r.gate_id, r.passed, r.rejection_code) for r in gates.evaluate_gates(changed)]
    assert all(gate_id != "GATE_ALL_PHASE_DISPOSITIONS_PRESENT" for gate_id, _, _ in observed)
    assert observed == expected


@pytest.mark.parametrize("field,gate_id,code", [
    ("artifact_digest", "GATE_ARTIFACT_IDENTITY_CONTINUITY", "ARTIFACT_IDENTITY_DISCONTINUITY"),
    ("capability_digest", "GATE_CAPABILITY_IDENTITY_CONTINUITY", "CAPABILITY_IDENTITY_DISCONTINUITY"),
])
def test_resealed_downstream_mismatch_rejected(authority, field, gate_id, code):
    _, receipts, seal = authority
    receipts[-1][field] = "a" * 64
    # Recompute the receipt digest, manifest and aggregate reference so the
    # mismatch is caught by identity comparison, not a stale hash or summary.
    receipts[-1]["receipt_digest"] = digest(
        {k: v for k, v in receipts[-1].items() if k != "receipt_digest"})
    failure = result(seal(), gate_id)
    assert failure.passed is False
    assert failure.rejection_code == code


@pytest.mark.parametrize("field", ["compiler_contract_digest", "consumer_contract_digest"])
def test_rehashed_compiler_reference_mismatch_rejected(authority, field):
    artifacts, receipts, seal = authority
    artifacts[-1][field] = "b" * 64
    artifacts[-1]["artifact_digest"] = digest(
        {k: v for k, v in artifacts[-1].items() if k != "artifact_digest"})
    receipts[-1]["artifact_digest"] = artifacts[-1]["artifact_digest"]
    chain = seal()
    assert gates._check_artifact_identity(chain)
    assert gates._check_capability_identity(chain)
    failure = result(chain, "GATE_COMPILER_IDENTITY_CONTINUITY")
    assert failure.passed is False
    assert failure.rejection_code == "COMPILER_IDENTITY_DISCONTINUITY"


def test_upstream_content_is_recomputed(authority):
    artifacts, _, seal = authority
    artifacts[-1]["logical_tick"] += 1
    assert not result(seal(), "GATE_ARTIFACT_IDENTITY_CONTINUITY").passed


@pytest.mark.parametrize("field", ["artifact_digest", "capability_digest"])
def test_caller_summary_is_compared(authority, field):
    _, _, seal = authority
    chain = seal()
    chain = replace(chain, g53c=replace(chain.g53c, **{field: "c" * 64}))
    check = gates._check_artifact_identity if field == "artifact_digest" else gates._check_capability_identity
    assert not check(chain)


@pytest.mark.parametrize("empty", ["artifacts", "receipts"])
def test_empty_identity_inventory_cannot_pass(authority, empty):
    artifacts, receipts, seal = authority
    (artifacts if empty == "artifacts" else receipts).clear()
    chain = seal()
    for gate_id in ("GATE_ARTIFACT_IDENTITY_CONTINUITY", "GATE_CAPABILITY_IDENTITY_CONTINUITY",
                    "GATE_COMPILER_IDENTITY_CONTINUITY"):
        assert not result(chain, gate_id).passed


def test_manifest_census_binding_is_checked(authority):
    _, _, seal = authority
    chain = seal()
    Path(chain.g53b1.manifest_path).write_text("{}")
    assert not result(chain, "GATE_ARTIFACT_IDENTITY_CONTINUITY").passed


def test_human_markdown_cannot_supply_or_change_disposition(authority):
    _, _, seal = authority
    chain = seal()
    directory = Path(chain.g53b1.source_directory)
    report = directory / "G53B1_FINAL_REPORT.md"
    report.write_text("G53B_FAILED\n")
    before = census_g53b1(str(directory))
    report.write_text("G53B_QUALIFIED\n")
    after = census_g53b1(str(directory))
    assert before == after
    assert after.disposition == ""
    before_vector = [(r.gate_id, r.passed, r.rejection_code) for r in gates.evaluate_gates(chain)]
    after_vector = [(r.gate_id, r.passed, r.rejection_code) for r in gates.evaluate_gates(replace(chain, g53b1=after))]
    assert before_vector == after_vector
    assert all(gate_id != "GATE_ALL_PHASE_DISPOSITIONS_PRESENT" for gate_id, _, _ in after_vector)


def test_gate_labels_and_rejection_order(authority):
    _, _, seal = authority
    results = gates.evaluate_gates(seal())
    assert results[0].gate_id == "GATE_SOURCE_MANIFEST_CLOSURE"
    assert results[1].gate_id == "GATE_SOURCE_HASH_SIZE_VALIDITY"
    assert results[2].gate_id == "GATE_ARTIFACT_IDENTITY_CONTINUITY"
    assert all(r.gate_id != "GATE_ALL_PHASE_DISPOSITIONS_PRESENT" for r in results)
    assert [r.gate_ordinal for r in results] == list(range(1, 20))
