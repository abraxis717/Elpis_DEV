"""Atomic materialization of a selected child workspace: OFFLINE ARTIFACT PRODUCTION ONLY.

Moved out of :mod:`elpis.evolution.promotion` unchanged so that the promotion law (records, evidence, selection,
digests), which the runtime's evolution gate imports, carries no writer (docs/VOLATILE_RUNTIME_CENSUS.md,
``evolution_child_materialization``). It is reached only through
:meth:`elpis.evolution.policy.EvolutionPolicyAuthority.materialize` with an operator-approved promotion grant, or
directly by an offline caller: one new child workspace at a destination that must not exist. No Runtime operation
calls it, and no module a composer imports imports this one.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

from .promotion import HarnessManifest, content_map_digest, digest_tree


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
