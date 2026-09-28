"""Deterministic synthetic Grid81 rows run through the real structural stages.

    typed projection -> structural groups

The rows are tiny synthetic data, not a learned-model corpus. Token 0 is a void
cell and token 6 marks an expansion locus, per the typed-projection contract.
"""
from __future__ import annotations

import hashlib

from elpis.structure.grid81.typed.compiler import compile_inventories
from elpis.structure.grid81.groups.compiler import compile_structural_groups
from elpis.structure.grid81.groups.source_join import join_inventory_records

SOURCE_MANIFEST_SHA = hashlib.sha256(b"elpis.test.synthetic-grid81-chain.v1").hexdigest()


def _grid(seed: int) -> list[int]:
    return [((i * 7 + seed) % 5) + 1 for i in range(81)]  # tokens 1..5: no void, no expansion


def synthetic_rows() -> list[tuple[dict, str]]:
    rows = []
    # EDIT: exactly one writable cell differs from the canonical target.
    base = _grid(1)
    target = list(base)
    target[40] = 9
    mask = [1 if i == 40 else 0 for i in range(81)]
    rows.append(({
        "case_id": "synthetic-edit-001", "abi_version": "t00.v1",
        "provenance_digest": hashlib.sha256(b"synthetic-edit-001").hexdigest(),
        "input_grid": base, "input_mask": mask, "canonical_target_grid": target,
        "expansion_targets": [], "quiescence_target": True, "rationale_codes": ["SINGLE_WRITABLE_DELTA"],
    }, "train"))
    # NOOP with void cells and one expansion locus: not quiescent.
    grid = _grid(2)
    grid[0] = 0
    grid[80] = 6
    rows.append(({
        "case_id": "synthetic-noop-002", "abi_version": "t00.v1",
        "provenance_digest": hashlib.sha256(b"synthetic-noop-002").hexdigest(),
        "input_grid": grid, "input_mask": [0] * 81, "canonical_target_grid": list(grid),
        "expansion_targets": [{"cell": 80}], "quiescence_target": False, "rationale_codes": [],
    }, "validation"))
    # Quiescent NOOP.
    grid = _grid(3)
    rows.append(({
        "case_id": "synthetic-noop-003", "abi_version": "t00.v1",
        "provenance_digest": hashlib.sha256(b"synthetic-noop-003").hexdigest(),
        "input_grid": grid, "input_mask": [0] * 81, "canonical_target_grid": list(grid),
        "expansion_targets": [], "quiescence_target": True, "rationale_codes": ["QUIESCENT"],
    }, "test"))
    return rows


def typed_inventories():
    inventories, stats = compile_inventories(synthetic_rows())
    assert not stats["errors"], stats["errors"]
    return inventories


def structural_groups():
    inv = typed_inventories()
    joined, audit = join_inventory_records(
        inv["identity"], inv["transition"], inv["expansion"], inv["quiescence"], inv["rationale"],
        expected_row_count=len(synthetic_rows()))
    assert audit["status"] == "SOURCE_JOIN_VERIFIED", audit
    return compile_structural_groups(joined, SOURCE_MANIFEST_SHA)
