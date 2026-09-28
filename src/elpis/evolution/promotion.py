"""Evaluation-gated promotion of candidate workspaces.

A candidate child workspace is promoted over its parent (the incumbent) only
when its evaluation evidence is bound to the exact parent, candidate, path
receipt, evaluation contract and all four data partitions (EVOLVE,
CALIBRATION, HELD_OUT, OOD), every gate passes, the held-out improvement
exceeds the fixed noise envelope and the OOD delta stays above the regression
floor. Otherwise the incumbent is retained. The selected child is then
materialized atomically from the parent plus explicit edits, and only if the
result matches the selected manifest byte for byte.

Workspaces are plain directory trees. Symlinks are refused everywhere, so a
candidate can never import content from outside its tree. Record schemas keep
their persisted ``elpis.rsi.*`` identifiers; they make no claim of
self-improvement.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile

from .digests import domain_digest, require_digest, sha256_file


NOISE_ENVELOPE = 3
OOD_REGRESSION_FLOOR = -2
RESOURCE_COST_LIMIT = 120
GENESIS_DIGEST = "0" * 64
REQUIRED_PARTITIONS = frozenset({"EVOLVE", "CALIBRATION", "HELD_OUT", "OOD"})


@dataclass(frozen=True)
class HarnessManifest:
    generation: int
    parent_harness_manifest_digest: str | None
    candidate_manifest_digest: str
    path_transition_receipt_digest: str
    editable_surface_manifest_digest: str
    component_content_digests: tuple[tuple[str, str], ...]
    configuration_digest: str
    tool_contract_digest: str
    prompt_or_policy_surface_digest: str
    build_or_materialization_digest: str

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("invalid generation")
        if self.generation == 0:
            if self.parent_harness_manifest_digest is not None:
                raise ValueError("founder cannot have parent")
        else:
            require_digest(self.parent_harness_manifest_digest)
        for digest in (
            self.candidate_manifest_digest,
            self.path_transition_receipt_digest,
            self.editable_surface_manifest_digest,
            self.configuration_digest,
            self.tool_contract_digest,
            self.prompt_or_policy_surface_digest,
            self.build_or_materialization_digest,
        ):
            require_digest(digest)
        if not self.component_content_digests:
            raise ValueError("empty component content map")
        names = [x[0] for x in self.component_content_digests]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("component paths must be sorted unique")
        for name, digest in self.component_content_digests:
            if not name or name.startswith("/") or ".." in Path(name).parts:
                raise ValueError("unsafe component path")
            require_digest(digest)

    def payload(self) -> dict:
        return {
            "schema": "elpis.rsi.harness-manifest.v0",
            "generation": self.generation,
            "parent_harness_manifest_digest_or_null": self.parent_harness_manifest_digest,
            "candidate_manifest_digest": self.candidate_manifest_digest,
            "path_transition_receipt_digest": self.path_transition_receipt_digest,
            "editable_surface_manifest_digest": self.editable_surface_manifest_digest,
            "component_content_digests": [
                [name, digest] for name, digest in self.component_content_digests
            ],
            "configuration_digest": self.configuration_digest,
            "tool_contract_digest": self.tool_contract_digest,
            "prompt_or_policy_surface_digest": self.prompt_or_policy_surface_digest,
            "build_or_materialization_digest": self.build_or_materialization_digest,
        }

    @property
    def digest(self) -> str:
        return domain_digest("elpis.rsi.harness-manifest.v0", self.payload())


@dataclass(frozen=True)
class EvaluationEvidence:
    candidate_harness_manifest_digest: str
    parent_harness_manifest_digest: str
    evaluation_contract_digest: str
    path_transition_receipt_digest: str
    partition_manifest_digests: tuple[tuple[str, str], ...]
    correctness_pass: bool
    leakage_pass: bool
    resource_pass: bool
    source_scope_pass: bool
    resource_cost: int
    held_out_delta: int
    ood_delta: int
    noise_envelope: int

    def __post_init__(self) -> None:
        for digest in (
            self.candidate_harness_manifest_digest,
            self.parent_harness_manifest_digest,
            self.evaluation_contract_digest,
            self.path_transition_receipt_digest,
        ):
            require_digest(digest)
        if type(self.resource_cost) is not int or self.resource_cost < 0:
            raise ValueError("invalid resource cost")
        if any(
            type(value) is not int
            for value in (self.held_out_delta, self.ood_delta, self.noise_envelope)
        ):
            raise ValueError("metric deltas must be integers")
        if self.noise_envelope < 0:
            raise ValueError("negative noise envelope")
        names = tuple(x[0] for x in self.partition_manifest_digests)
        if tuple(sorted(names)) != names or len(names) != len(set(names)):
            raise ValueError("partitions must be sorted unique")
        for _, digest in self.partition_manifest_digests:
            require_digest(digest)

    @property
    def partitions(self) -> dict[str, str]:
        return {key: value for key, value in self.partition_manifest_digests}

    def payload(self) -> dict:
        return {
            "schema": "elpis.rsi.evaluation-evidence.v0",
            "candidate_harness_manifest_digest": self.candidate_harness_manifest_digest,
            "parent_harness_manifest_digest": self.parent_harness_manifest_digest,
            "evaluation_contract_digest": self.evaluation_contract_digest,
            "path_transition_receipt_digest": self.path_transition_receipt_digest,
            "partition_manifest_digests": [
                [name, digest] for name, digest in self.partition_manifest_digests
            ],
            "correctness_pass": self.correctness_pass,
            "leakage_pass": self.leakage_pass,
            "resource_pass": self.resource_pass,
            "source_scope_pass": self.source_scope_pass,
            "resource_cost": self.resource_cost,
            "held_out_delta": self.held_out_delta,
            "ood_delta": self.ood_delta,
            "noise_envelope": self.noise_envelope,
        }

    @property
    def digest(self) -> str:
        return domain_digest("elpis.rsi.evaluation-evidence.v0", self.payload())


@dataclass(frozen=True)
class CandidateRecord:
    manifest: HarnessManifest
    evidence: EvaluationEvidence
    edit_count: int


@dataclass(frozen=True)
class SelectionReceipt:
    parent_harness_manifest_digest: str
    selected_harness_manifest_digest: str
    selected_candidate_manifest_digest_or_null: str | None
    selected_evaluation_evidence_digest_or_null: str | None
    disposition: str
    eligible_candidate_manifest_digests: tuple[str, ...]
    rejected: tuple[tuple[str, str], ...]

    def payload(self) -> dict:
        return {
            "schema": "elpis.rsi.selection-receipt.v0",
            "parent_harness_manifest_digest": self.parent_harness_manifest_digest,
            "selected_harness_manifest_digest": self.selected_harness_manifest_digest,
            "selected_candidate_manifest_digest_or_null": self.selected_candidate_manifest_digest_or_null,
            "selected_evaluation_evidence_digest_or_null": self.selected_evaluation_evidence_digest_or_null,
            "disposition": self.disposition,
            "eligible_candidate_manifest_digests": list(
                self.eligible_candidate_manifest_digests
            ),
            "rejected": [[candidate, reason] for candidate, reason in self.rejected],
        }

    @property
    def digest(self) -> str:
        return domain_digest("elpis.rsi.selection-receipt.v0", self.payload())


def eligibility(
    parent: HarnessManifest,
    candidate: CandidateRecord,
    evaluation_contract_digest: str,
    partition_digests: dict[str, str],
) -> tuple[bool, str]:
    manifest = candidate.manifest
    evidence = candidate.evidence
    if manifest.parent_harness_manifest_digest != parent.digest:
        return False, "STALE_PARENT_BINDING"
    if evidence.parent_harness_manifest_digest != parent.digest:
        return False, "STALE_EVALUATION_PARENT_BINDING"
    if evidence.candidate_harness_manifest_digest != manifest.digest:
        return False, "STALE_CANDIDATE_BINDING"
    if evidence.path_transition_receipt_digest != manifest.path_transition_receipt_digest:
        return False, "PATH_RECEIPT_MISMATCH"
    if evidence.evaluation_contract_digest != evaluation_contract_digest:
        return False, "EVALUATION_CONTRACT_MISMATCH"
    if set(evidence.partitions) != REQUIRED_PARTITIONS:
        return False, "MISSING_REQUIRED_PARTITION"
    if any(
        evidence.partitions[name] != partition_digests[name]
        for name in REQUIRED_PARTITIONS
    ):
        return False, "PARTITION_MANIFEST_MISMATCH"
    if not evidence.correctness_pass:
        return False, "CORRECTNESS_GATE_FAIL"
    if not evidence.leakage_pass:
        return False, "LEAKAGE_GATE_FAIL"
    if not evidence.resource_pass or evidence.resource_cost > RESOURCE_COST_LIMIT:
        return False, "RESOURCE_GATE_FAIL"
    if not evidence.source_scope_pass:
        return False, "SOURCE_SCOPE_GATE_FAIL"
    if evidence.noise_envelope != NOISE_ENVELOPE:
        return False, "NOISE_ENVELOPE_MISMATCH"
    if evidence.held_out_delta <= evidence.noise_envelope:
        return False, "WITHIN_NOISE_OR_NONIMPROVING"
    if evidence.ood_delta < OOD_REGRESSION_FLOOR:
        return False, "OOD_CATASTROPHIC_REGRESSION"
    return True, "ELIGIBLE"


def select(
    parent: HarnessManifest,
    candidates: list[CandidateRecord] | tuple[CandidateRecord, ...],
    evaluation_contract_digest: str,
    partition_digests: dict[str, str],
) -> SelectionReceipt:
    eligible: list[CandidateRecord] = []
    rejected: list[tuple[str, str]] = []
    for candidate in candidates:
        ok, reason = eligibility(
            parent, candidate, evaluation_contract_digest, partition_digests
        )
        if ok:
            eligible.append(candidate)
        else:
            rejected.append((candidate.manifest.candidate_manifest_digest, reason))
    eligible.sort(
        key=lambda candidate: (
            -candidate.evidence.held_out_delta,
            candidate.evidence.resource_cost,
            candidate.edit_count,
            candidate.manifest.candidate_manifest_digest,
        )
    )
    if not eligible:
        return SelectionReceipt(
            parent_harness_manifest_digest=parent.digest,
            selected_harness_manifest_digest=parent.digest,
            selected_candidate_manifest_digest_or_null=None,
            selected_evaluation_evidence_digest_or_null=None,
            disposition="RETAIN_INCUMBENT",
            eligible_candidate_manifest_digests=(),
            rejected=tuple(sorted(rejected)),
        )
    winner = eligible[0]
    return SelectionReceipt(
        parent_harness_manifest_digest=parent.digest,
        selected_harness_manifest_digest=winner.manifest.digest,
        selected_candidate_manifest_digest_or_null=winner.manifest.candidate_manifest_digest,
        selected_evaluation_evidence_digest_or_null=winner.evidence.digest,
        disposition="SELECT_CHALLENGER",
        eligible_candidate_manifest_digests=tuple(
            candidate.manifest.candidate_manifest_digest for candidate in eligible
        ),
        rejected=tuple(sorted(rejected)),
    )


def digest_tree(root: str | Path) -> tuple[tuple[str, str], ...]:
    """Sorted (relative path, SHA-256) of every regular file; symlinks are refused."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("workspace root must be a real directory")
    out: list[tuple[str, str]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("symlink in workspace: " + str(path.relative_to(root)))
        for name in filenames:
            path = Path(directory) / name
            if not path.is_file():
                raise ValueError("non-regular file in workspace: " + str(path.relative_to(root)))
            out.append((path.relative_to(root).as_posix(), sha256_file(path)))
    return tuple(sorted(out))


def content_map_digest(entries: tuple[tuple[str, str], ...]) -> str:
    return domain_digest(
        "elpis.rsi.component-content-map.v0",
        [[name, digest] for name, digest in entries],
    )


def build_manifest_from_tree(
    root: str | Path,
    *,
    generation: int,
    parent_digest: str | None,
    candidate_manifest_digest: str,
    path_receipt_digest: str,
    editable_surface_digest: str,
) -> HarnessManifest:
    entries = digest_tree(root)
    build_digest = content_map_digest(entries)
    return HarnessManifest(
        generation=generation,
        parent_harness_manifest_digest=parent_digest,
        candidate_manifest_digest=candidate_manifest_digest,
        path_transition_receipt_digest=path_receipt_digest,
        editable_surface_manifest_digest=editable_surface_digest,
        component_content_digests=entries,
        configuration_digest=domain_digest("cfg", {"entries": entries}),
        tool_contract_digest=domain_digest("tools", {"entries": entries}),
        prompt_or_policy_surface_digest=domain_digest("policy", {"entries": entries}),
        build_or_materialization_digest=build_digest,
    )


def atomic_materialize(
    parent_dir: str | Path,
    dest_dir: str | Path,
    edits: list[dict],
    expected_manifest: HarnessManifest,
    fail_after_edit: bool = False,
) -> dict[str, str]:
    parent_dir = Path(parent_dir)
    dest_dir = Path(dest_dir)
    if dest_dir.exists() or dest_dir.is_symlink():
        raise ValueError("destination already exists")
    digest_tree(parent_dir)  # refuses symlinks and special files before any copy
    stage = Path(tempfile.mkdtemp(prefix=".rsi-child.", dir=str(dest_dir.parent)))
    child = stage / "child"
    try:
        shutil.copytree(parent_dir, child, symlinks=True)
        for index, edit in enumerate(edits):
            op = edit["op"]
            rel = Path(edit["path"])
            if rel.is_absolute() or ".." in rel.parts or not rel.parts:
                raise ValueError("unsafe edit path")
            path = child / rel
            if op == "write":
                if type(edit.get("data")) is not bytes:
                    raise ValueError("write edit requires bytes")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(edit["data"])
            elif op == "delete":
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            else:
                raise ValueError("unknown edit op")
            if fail_after_edit and index == 0:
                raise RuntimeError("INJECTED_MATERIALIZATION_FAILURE")
        actual_entries = digest_tree(child)
        if actual_entries != expected_manifest.component_content_digests:
            raise ValueError("materialized content does not match selected manifest")
        if (
            content_map_digest(actual_entries)
            != expected_manifest.build_or_materialization_digest
        ):
            raise ValueError("materialization digest mismatch")
        os.replace(child, dest_dir)
        return {
            "status": "MATERIALIZED",
            "child_digest": expected_manifest.digest,
            "content_map_digest": content_map_digest(actual_entries),
        }
    finally:
        shutil.rmtree(stage, ignore_errors=True)
