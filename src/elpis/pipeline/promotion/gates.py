"""Promotion gates — deterministic, ordered, immutable checks."""

import hashlib
import json
import os

from .import_boundary import check_import_boundary
from .source_binding import _hex64

from .canonical import GateResult, REJECTION_PRECEDENCE, SourceChain, _sha256_str, _canonical_json


# Gate definitions in fixed order
GATE_DEFINITIONS = [
    ("GATE_SOURCE_MANIFEST_CLOSURE", 1, "1.0.0"),
    ("GATE_SOURCE_HASH_SIZE_VALIDITY", 2, "1.0.0"),
    ("GATE_ARTIFACT_IDENTITY_CONTINUITY", 3, "1.0.0"),
    ("GATE_CAPABILITY_IDENTITY_CONTINUITY", 4, "1.0.0"),
    ("GATE_COMPILER_IDENTITY_CONTINUITY", 5, "1.0.0"),
    ("GATE_SHADOW_APPLICATION_ACCEPTED", 6, "1.0.0"),
    ("GATE_RECEIPT_INTEGRITY", 7, "1.0.0"),
    ("GATE_SHADOW_STATE_TRANSITION_INTEGRITY", 8, "1.0.0"),
    ("GATE_LEDGER_HEAD_CONTINUITY", 9, "1.0.0"),
    ("GATE_REPLAY_PROTECTION_QUALIFIED", 10, "1.0.0"),
    ("GATE_MUTATION_EXACTNESS_QUALIFIED", 11, "1.0.0"),
    ("GATE_ATOMICITY_QUALIFIED", 12, "1.0.0"),
    ("GATE_CANONICAL_NONMUTATION_QUALIFIED", 13, "1.0.0"),
    ("GATE_AUTHORITY_BOUNDARY_QUALIFIED", 14, "1.0.0"),
    ("GATE_THREE_SEED_DETERMINISM_QUALIFIED", 15, "1.0.0"),
    ("GATE_G53D_BUNDLE_CONSISTENCY", 16, "1.0.0"),
    ("GATE_CAPABILITY_CANONICALLY_UNCONSUMED", 17, "1.0.0"),
    ("GATE_SOURCE_REPORTS_UNCHANGED", 18, "1.0.0"),
    ("GATE_NO_EXECUTABLE_AUTHORITY", 19, "1.0.0"),
]


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _make_gate_result(gate_id: str, ordinal: int, version: str,
                      passed: bool, rejection_code: str | None,
                      evidence: tuple, observed: str | None = None,
                      expected: str | None = None) -> GateResult:
    return GateResult(
        gate_id=gate_id,
        gate_ordinal=ordinal,
        gate_version=version,
        passed=passed,
        rejection_code=rejection_code if not passed else None,
        evidence_bindings=evidence,
        observed_value=observed,
        expected_value=expected,
    )


def _all_phases_have_manifests(chain: SourceChain) -> bool:
    for phase in [chain.g53b1, chain.g53c, chain.g53d]:
        if not os.path.exists(phase.manifest_path):
            return False
    return True


def _verify_file_hashes(directory: str, evidence_files: tuple) -> bool:
    for fname, expected_hash, expected_size in evidence_files:
        path = os.path.join(directory, fname)
        if not os.path.exists(path):
            return False
        actual_hash = _file_sha256(path)
        actual_size = os.path.getsize(path)
        if actual_hash != expected_hash:
            return False
        if actual_size != expected_size:
            return False
    return True


def _bound_jsonl(phase, filename: str) -> list:
    """Read structured evidence only after checking its census hash and size."""
    entries = tuple(e for e in phase.evidence_files if e[0] == filename)
    if _file_sha256(phase.manifest_path) != phase.manifest_digest:
        raise ValueError("STRUCTURED_IDENTITY_MANIFEST_CHANGED")
    meta = _read_json(phase.manifest_path)["evidence_files"][filename]
    if entries != ((filename, meta["sha256"], meta["size"]),):
        raise ValueError("STRUCTURED_IDENTITY_MANIFEST_BINDING_MISMATCH")
    if len(entries) != 1 or not _verify_file_hashes(phase.source_directory, entries):
        raise ValueError("STRUCTURED_IDENTITY_EVIDENCE_MISSING_OR_CHANGED")
    with open(os.path.join(phase.source_directory, filename)) as source:
        records = [json.loads(line) for line in source if line.strip()]
    if not records or any(type(record) is not dict for record in records):
        raise ValueError("STRUCTURED_IDENTITY_EVIDENCE_EMPTY_OR_INVALID")
    return records


def _identity_bindings(chain: SourceChain) -> tuple:
    """Join downstream receipts to independently rehashed upstream artifacts.

    Artifact identity follows consumption_compiler.artifact's full-record
    digest contract (all fields except artifact_digest). These are evidence
    bindings relative to the supplied manifests, not an external seal pin.
    """
    artifacts = _bound_jsonl(
        chain.g53b1, "G53B_STRUCTURAL_INFLUENCE_ARTIFACT_INVENTORY.jsonl")
    receipts = _bound_jsonl(chain.g53c, "G53C_APPLICATION_RECEIPTS.jsonl")
    by_digest = {}
    for artifact in artifacts:
        digest = artifact.get("artifact_digest")
        actual = _sha256_str(_canonical_json(
            {k: v for k, v in artifact.items() if k != "artifact_digest"}))
        if (artifact.get("schema_version") != "structural-influence-artifact.v1"
                or digest != actual or digest in by_digest):
            raise ValueError("UPSTREAM_ARTIFACT_IDENTITY_INVALID")
        by_digest[digest] = artifact
    return receipts, by_digest


def _check_artifact_identity(chain: SourceChain) -> bool:
    """Every application artifact reference must resolve to a rehashed consumption artifact."""
    receipts, artifacts = _identity_bindings(chain)
    return (all(r.get("artifact_digest") in artifacts for r in receipts)
            and chain.g53c.artifact_digest == ":".join(
                sorted(r["artifact_digest"] for r in receipts)))


def _check_capability_identity(chain: SourceChain) -> bool:
    """Each receipt capability must equal its upstream artifact's source binding."""
    receipts, artifacts = _identity_bindings(chain)
    return (all(_hex64(r.get("capability_digest")) and r.get("capability_digest") ==
                artifacts[r["artifact_digest"]]["source_capability_digest"]
                for r in receipts)
            and chain.g53c.capability_digest == ":".join(
                sorted(r["capability_digest"] for r in receipts)))


def _check_compiler_identity(chain: SourceChain) -> bool:
    """Check compiler *contract* identity along the receipt/artifact join.

    The authoritative constructor defines StructuralInfluenceCompilerContractV1.
    Compare both compiler and consumer references on each downstream-referenced
    upstream artifact. This does not establish compiler source-byte identity:
    the legacy upstream-identity/audit files have no repository-defined binding
    contract and cannot establish that broader claim.
    """
    from elpis.pipeline.consumption.policy import create_compiler_contract

    contract = create_compiler_contract()
    expected = _sha256_str(_canonical_json(
        {k: v for k, v in contract.items() if k != "compiler_contract_digest"}))
    receipts, artifacts = _identity_bindings(chain)
    return all(
        artifacts[r["artifact_digest"]].get("compiler_contract_digest") == expected
        and artifacts[r["artifact_digest"]].get("consumer_contract_digest") == expected
        for r in receipts
    )


def _check_shadow_application_accepted(chain: SourceChain) -> bool:
    """All application receipts must show APPLICATION_ACCEPTED."""
    receipts_path = os.path.join(
        chain.g53c.source_directory,
        "G53C_APPLICATION_RECEIPTS.jsonl",
    )
    if not os.path.exists(receipts_path):
        return False
    with open(receipts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            receipt = json.loads(line)
            if receipt.get("application_outcome") != "APPLICATION_ACCEPTED":
                return False
    return True


def _check_receipt_integrity(chain: SourceChain) -> bool:
    """Receipt chain digest must match the independent aggregate identity."""
    if not chain.g53c.receipt_chain_digest:
        return False
    receipts_path = os.path.join(
        chain.g53c.source_directory,
        "G53C_APPLICATION_RECEIPTS.jsonl",
    )
    receipts = []
    with open(receipts_path) as f:
        for line in f:
            line = line.strip()
            if line:
                receipts.append(json.loads(line))
    # Recompute chain digest
    computed = hashlib.sha256(
        json.dumps(receipts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return computed == chain.g53c.receipt_chain_digest


def _check_shadow_state_transition(chain: SourceChain) -> bool:
    """Each receipt's resulting_state_digest is fixture-local (independent state spaces).
    We verify that each receipt has both previous_state and resulting_state fields
    present and non-empty, confirming the state transition was recorded."""
    receipts_path = os.path.join(
        chain.g53c.source_directory,
        "G53C_APPLICATION_RECEIPTS.jsonl",
    )
    receipts = []
    with open(receipts_path) as f:
        for line in f:
            line = line.strip()
            if line:
                receipts.append(json.loads(line))
    for r in receipts:
        if not r.get("previous_state_digest") or not r.get("resulting_state_digest"):
            return False
        if r["previous_state_digest"] == r["resulting_state_digest"]:
            return False
    return True


def _check_ledger_head_continuity(chain: SourceChain) -> bool:
    """Ledger heads must form a continuous chain across receipts."""
    receipts_path = os.path.join(
        chain.g53c.source_directory,
        "G53C_APPLICATION_RECEIPTS.jsonl",
    )
    receipts = []
    with open(receipts_path) as f:
        for line in f:
            line = line.strip()
            if line:
                receipts.append(json.loads(line))
    for i in range(len(receipts) - 1):
        current_result = receipts[i]["resulting_ledger_head"]
        next_previous = receipts[i + 1]["previous_ledger_head"]
        if current_result != next_previous:
            return False
    return True


def _check_replay_protection(chain: SourceChain) -> bool:
    """consumption replay audit must pass."""
    replay_path = os.path.join(
        chain.g53b1.source_directory,
        "G53B_REPLAY_AUDIT.json",
    )
    if not os.path.exists(replay_path):
        return False
    audit = _read_json(replay_path)
    if not isinstance(audit, dict):
        return False
    fields = [
        audit[name]
        for name in ("replay_protection_qualified", "replay_protection")
        if name in audit
    ]
    # At least one recognized status must be explicitly present. If both legacy
    # and qualified statuses are present they must agree on True. Missing,
    # false, conflicting, or non-boolean evidence fails closed.
    return bool(fields) and all(type(value) is bool and value is True for value in fields)


def _check_mutation_exactness(chain: SourceChain) -> bool:
    """consumption mutation results must show exact_match for all mutations."""
    mutation_path = os.path.join(
        chain.g53b1.source_directory,
        "G53B_MUTATION_RESULTS.json",
    )
    if not os.path.exists(mutation_path):
        return False
    results = _read_json(mutation_path)
    # Actual field names: total_mutations, caught, exact_match
    exact = results.get("exact_match", 0)
    caught = results.get("caught", 0)
    total = results.get("total_mutations", 0)
    return exact == caught and caught == total and total > 0


def _check_atomicity(chain: SourceChain) -> bool:
    """Both consumption and application atomicity audits must pass."""
    g53b_atomic = os.path.join(
        chain.g53b1.source_directory,
        "G53B_ATOMICITY_AUDIT.json",
    )
    g53c_atomic = os.path.join(
        chain.g53c.source_directory,
        "G53C_ATOMICITY_AUDIT.json",
    )
    if not os.path.exists(g53b_atomic) or not os.path.exists(g53c_atomic):
        return False
    b_audit = _read_json(g53b_atomic)
    c_audit = _read_json(g53c_atomic)
    # consumption uses accepted_atomicity_ok, application uses atomicity_verified
    b_pass = b_audit.get("accepted_atomicity_ok", False) or b_audit.get("atomicity_verified", False)
    c_pass = c_audit.get("atomicity_verified", False)
    return b_pass and c_pass


def _check_canonical_nonmutation(chain: SourceChain) -> bool:
    """Both application and application-evidence canonical nonmutation must be proven."""
    g53c_nm = os.path.join(
        chain.g53c.source_directory,
        "G53C_CANONICAL_NONMUTATION_AUDIT.json",
    )
    g53d_nm = os.path.join(
        chain.g53d.source_directory,
        "G53D_CANONICAL_NONMUTATION_AUDIT.json",
    )
    if not os.path.exists(g53c_nm) or not os.path.exists(g53d_nm):
        return False
    c_audit = _read_json(g53c_nm)
    d_audit = _read_json(g53d_nm)
    # application uses no_canonical_write, application-evidence uses canonical_state_mutated
    c_pass = c_audit.get("no_canonical_write", False)
    c_pass = c_pass and c_audit.get("canonical_consumption_count", -1) == 0
    d_pass = d_audit.get("canonical_state_mutated", True) == False
    return c_pass and d_pass


def _check_authority_boundary(chain: SourceChain) -> bool:
    """All phases must report no authority violations."""
    g53b_auth = os.path.join(
        chain.g53b1.source_directory,
        "G53B_AUTHORITY_BOUNDARY_AUDIT.json",
    )
    g53c_auth = os.path.join(
        chain.g53c.source_directory,
        "G53C_AUTHORITY_AUDIT.json",
    )
    g53d_auth = os.path.join(
        chain.g53d.source_directory,
        "G53D_AUTHORITY_AUDIT.json",
    )
    if not all(os.path.exists(p) for p in [g53b_auth, g53c_auth, g53d_auth]):
        return False
    for path in [g53b_auth, g53c_auth, g53d_auth]:
        audit = _read_json(path)
        # consumption: authority_violations top-level
        # application: authority_violations top-level
        # application-evidence: source_authority_audit.authority_violations
        violations = audit.get("authority_violations", -1)
        if violations != 0:
            source_audit = audit.get("source_authority_audit", {})
            inner_violations = source_audit.get("authority_violations", -1)
            if inner_violations != 0:
                return False
    return True


def _check_three_seed_determinism(chain: SourceChain) -> bool:
    """consumption and application three-seed determinism must pass."""
    g53b_det = os.path.join(
        chain.g53b1.source_directory,
        "G53B_FULL_THREE_SEED_DETERMINISM.json",
    )
    g53c_det = os.path.join(
        chain.g53c.source_directory,
        "G53C_THREE_SEED_DETERMINISM.json",
    )
    if not os.path.exists(g53b_det) or not os.path.exists(g53c_det):
        return False
    b_det = _read_json(g53b_det)
    c_det = _read_json(g53c_det)
    # consumption uses all_seeds_match, application uses deterministic
    b_pass = b_det.get("all_seeds_match", False) or b_det.get("deterministic", False)
    c_pass = c_det.get("deterministic", False) or c_det.get("three_seed_byte_identity", False)
    return b_pass and c_pass


def _check_g53d_bundle_consistency(chain: SourceChain) -> bool:
    """application-evidence bundle digest must be present and match post-qualification."""
    if not chain.g53d.bundle_digest:
        return False
    post_qual = os.path.join(
        chain.g53d.source_directory,
        "G53D_POST_QUALIFICATION_VERIFICATION.json",
    )
    if not os.path.exists(post_qual):
        return False
    pq = _read_json(post_qual)
    return pq.get("bundle_digest", "") == chain.g53d.bundle_digest


def _check_capability_canonically_unconsumed(chain: SourceChain) -> bool:
    """Canonical lifecycle must be GRANTED_UNCONSUMED."""
    return chain.g53c.lifecycle_state == "GRANTED_UNCONSUMED"


def _check_source_reports_unchanged(chain: SourceChain) -> bool:
    """Verify all evidence file hashes match their manifest entries."""
    for phase in [chain.g53b1, chain.g53c, chain.g53d]:
        valid = _verify_file_hashes(phase.source_directory, phase.evidence_files)
        if not valid:
            return False
    return True


def _check_no_executable_authority(chain: SourceChain) -> bool:
    """Check statically resolvable forbidden imports in every package Python file.

    This is a bounded import policy, not arbitrary dynamic-code detection or
    proof of absence of every possible executable authority.
    """
    pkg_dir = os.path.dirname(__file__)
    forbidden = {"torch", "subprocess", "socket", "urllib", "requests", "http.client"}
    return check_import_boundary(pkg_dir, forbidden)[0]


# Gate function dispatch table — deterministic order
_GATE_FUNCTIONS = [
    (0, _all_phases_have_manifests, REJECTION_PRECEDENCE[0]),
    (1, lambda c: _verify_file_hashes(c.g53b1.source_directory, c.g53b1.evidence_files) and
                  _verify_file_hashes(c.g53c.source_directory, c.g53c.evidence_files) and
                  _verify_file_hashes(c.g53d.source_directory, c.g53d.evidence_files),
     REJECTION_PRECEDENCE[1]),
    (2, _check_artifact_identity, REJECTION_PRECEDENCE[4]),
    (3, _check_capability_identity, REJECTION_PRECEDENCE[5]),
    (4, _check_compiler_identity, REJECTION_PRECEDENCE[6]),
    (5, _check_shadow_application_accepted, REJECTION_PRECEDENCE[7]),
    (6, _check_receipt_integrity, REJECTION_PRECEDENCE[8]),
    (7, _check_shadow_state_transition, REJECTION_PRECEDENCE[9]),
    (8, _check_ledger_head_continuity, REJECTION_PRECEDENCE[10]),
    (9, _check_replay_protection, REJECTION_PRECEDENCE[11]),
    (10, _check_mutation_exactness, REJECTION_PRECEDENCE[12]),
    (11, _check_atomicity, REJECTION_PRECEDENCE[13]),
    (12, _check_canonical_nonmutation, REJECTION_PRECEDENCE[14]),
    (13, _check_authority_boundary, REJECTION_PRECEDENCE[15]),
    (14, _check_three_seed_determinism, REJECTION_PRECEDENCE[16]),
    (15, _check_g53d_bundle_consistency, REJECTION_PRECEDENCE[17]),
    (16, _check_capability_canonically_unconsumed, REJECTION_PRECEDENCE[18]),
    (17, _check_source_reports_unchanged, REJECTION_PRECEDENCE[19]),
    (18, _check_no_executable_authority, REJECTION_PRECEDENCE[20]),
]


def evaluate_gates(chain: SourceChain) -> list:
    """Evaluate all gates in deterministic order. Returns list of GateResult."""
    results = []
    for idx, (func_idx, func, rejection_code) in enumerate(_GATE_FUNCTIONS):
        gate_id, ordinal, version = GATE_DEFINITIONS[idx]
        try:
            passed = func(chain)
        except Exception as exc:
            passed = False
            detail = str(exc) if isinstance(exc, ValueError) else ""
        else:
            detail = ""

        observed = "PASS" if passed else (detail or "FAIL")
        expected = "PASS"

        results.append(
            _make_gate_result(
                gate_id=gate_id,
                ordinal=ordinal,
                version=version,
                passed=passed,
                rejection_code=rejection_code,
                evidence=(
                    (chain.g53b1.phase_id, chain.g53b1.manifest_digest),
                    (chain.g53c.phase_id, chain.g53c.manifest_digest),
                    (chain.g53d.phase_id, chain.g53d.manifest_digest),
                ),
                observed=observed,
                expected=expected,
            )
        )
    return results


def first_failure(results: list) -> str | None:
    """Return the rejection code of the first failing gate, or None."""
    for r in results:
        if not r.passed:
            return r.rejection_code
    return None
