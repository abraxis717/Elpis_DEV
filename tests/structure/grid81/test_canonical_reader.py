"""Canonical reader and runtime reduction over the historical Grid81 fixture.

Every adversarial scenario proves both that the reader rejects the malformed
canonical state and that the runtime reduction yields no object.
"""
from __future__ import annotations

import json
import pathlib
import shutil

import pytest

from elpis.structure.grid81.canonical import (
    CanonicalReadError,
    load_current_grid81,
    load_grid81_runtime_state,
)

FIXTURE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "grid81"
GRID = pathlib.PurePosixPath("Canonical/Grid81")

# Pinned digests of the fixture generation; identical to the donor reader and
# reducer at the migration basis commit.
FIXTURE_CANONICAL_DIGEST = "54e757477205f691717f0ca6365cb8a512155a69d949e106c3c0dfdf10b1b308"
FIXTURE_RUNTIME_PROJECTION = "dadfd4156cdb2653c99d10b296dec864ad1912c21e336237c5caab70aadc7c8c"


def _copy(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "root"
    shutil.copytree(FIXTURE_ROOT / GRID, root / GRID)
    return root


def _edit(root, rel, mutate):
    path = root / GRID / rel
    path.write_text(json.dumps(mutate(json.loads(path.read_text())), indent=2))


def _unlink(rel):
    return lambda root: (root / GRID / rel).unlink()


def _write(rel, text):
    return lambda root: (root / GRID / rel).write_text(text)


def _patch(rel, **fields):
    return lambda root: _edit(root, rel, lambda d: {**d, **fields})


def _symlink(rel):
    def apply(root):
        path = root / GRID / rel
        real = path.with_suffix(".json.real")
        shutil.move(str(path), str(real))
        path.symlink_to(real)
    return apply


def _manifest_audit_hash(root):
    _edit(root, ".transaction_manifest.json", lambda d: {**d, "artifact_inventory": [
        {**a, "sha256": "0" * 64} if a.get("artifact_role") == "authority_audit" else a
        for a in d.get("artifact_inventory", [])
    ]})


def _generation_capability(root):
    _edit(root, "generations/000001.json", lambda d: {
        **d, "authority_record": {**d.get("authority_record", {}), "capability_id": "0" * 64}})


def _capability_not_consumed(root):
    _edit(root, ".consumed_capability.json", lambda d: {
        **d, "lifecycle": {**d.get("lifecycle", {}), "consumed": False}})


def _remove_grid(root):
    shutil.rmtree(root / GRID)


def _empty_root(root):
    shutil.rmtree(root / "Canonical")


ZERO = "0" * 64
SCENARIOS = {
    "head_deleted": _unlink("HEAD.json"),
    "head_malformed_json": _write("HEAD.json", "{invalid json"),
    "head_wrong_generation_hash": _patch("HEAD.json", generation_file_sha256=ZERO),
    "head_wrong_semantic_digest": _patch("HEAD.json", generation_semantic_digest=ZERO),
    "generation_deleted": _unlink("generations/000001.json"),
    "generation_malformed_json": _write("generations/000001.json", "NOT_JSON"),
    "generation_wrong_transaction_id": _patch("generations/000001.json", transaction_id=ZERO),
    "generation_wrong_capability_id": _generation_capability,
    "head_symlink": _symlink("HEAD.json"),
    "generation_symlink": _symlink("generations/000001.json"),
    "manifest_symlink": _symlink(".transaction_manifest.json"),
    "manifest_deleted": _unlink(".transaction_manifest.json"),
    "manifest_wrong_transaction_id": _patch(".transaction_manifest.json", transaction_id=ZERO),
    "manifest_wrong_capability_id": _patch(".transaction_manifest.json", capability_id=ZERO),
    "manifest_wrong_hash_authority_audit": _manifest_audit_hash,
    "consumed_capability_deleted": _unlink(".consumed_capability.json"),
    "capability_not_consumed": _capability_not_consumed,
    "consumption_receipt_deleted": _unlink(".consumption_receipt.json"),
    "receipt_not_committed": _patch(".consumption_receipt.json", commit_status="UNCOMMITTED"),
    "authority_audit_deleted": _unlink(".authority_audit.json"),
    "source_nonmutation_deleted": _unlink(".source_nonmutation_audit.json"),
    "canonical_dir_deleted": _remove_grid,
    "unexpected_file_in_canonical": _write(".sneaky_backdoor.json", "{}"),
    "head_generation_mismatch": _patch("HEAD.json", generation=999),
    "head_txn_vs_generation_txn_mismatch": _patch("HEAD.json", transaction_id=ZERO),
    "head_capability_vs_generation_capability_mismatch": _patch("HEAD.json", capability_id=ZERO),
    "generation_content_modified": _patch("generations/000001.json", generation_number=999),
    "empty_project_root": _empty_root,
}


def test_fixture_reads_and_reduces(tmp_path):
    root = _copy(tmp_path)
    state = load_current_grid81(root)
    runtime = load_grid81_runtime_state(root)
    assert state.canonical_digest == FIXTURE_CANONICAL_DIGEST
    assert runtime.canonical_digest == FIXTURE_CANONICAL_DIGEST
    assert runtime.generation_number == state.generation_number == 1
    assert runtime.runtime_projection_digest == FIXTURE_RUNTIME_PROJECTION


def test_scenario_matrix_is_complete():
    assert len(SCENARIOS) == 28


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_adversarial_state_rejected_and_runtime_gets_nothing(tmp_path, name):
    root = _copy(tmp_path)
    SCENARIOS[name](root)
    with pytest.raises(CanonicalReadError):
        load_current_grid81(root)
    runtime = None
    with pytest.raises(CanonicalReadError):
        runtime = load_grid81_runtime_state(root)
    assert runtime is None


def test_supplied_root_is_never_imported_from(tmp_path):
    import sys

    root = _copy(tmp_path)
    before = list(sys.path)
    load_grid81_runtime_state(root)
    assert sys.path == before
    assert str(root) not in sys.path
