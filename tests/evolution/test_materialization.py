from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from elpis.evolution.promotion import (
    atomic_materialize,
    build_manifest_from_tree,
    digest_tree,
    domain_digest,
)


def prepare_expected(parent: Path, expected: Path, edits: list[dict]):
    shutil.copytree(parent, expected)
    for edit in edits:
        path = expected / edit["path"]
        if edit["op"] == "write":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(edit["data"])
        elif edit["op"] == "delete":
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()


def manifest_for(root: Path):
    return build_manifest_from_tree(
        root,
        generation=1,
        parent_digest=domain_digest("parent", {"x": 1}),
        candidate_manifest_digest=domain_digest("candidate", {"x": 1}),
        path_receipt_digest=domain_digest("path", {"x": 1}),
        editable_surface_digest=domain_digest("surface", {"x": 1}),
    )


def test_successful_child_materialization_is_exact(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "a.txt").write_text("A")
    edits = [{"op": "write", "path": "b.txt", "data": b"B"}]
    expected = tmp_path / "expected"
    prepare_expected(parent, expected, edits)
    manifest = manifest_for(expected)
    dest = tmp_path / "child"
    receipt = atomic_materialize(parent, dest, edits, manifest)
    assert receipt["status"] == "MATERIALIZED"
    assert receipt["child_digest"] == manifest.digest
    assert digest_tree(dest) == manifest.component_content_digests


def test_injected_failure_leaves_parent_and_destination_unchanged(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "a.txt").write_text("A")
    before = digest_tree(parent)
    edits = [{"op": "write", "path": "b.txt", "data": b"B"}]
    expected = tmp_path / "expected"
    prepare_expected(parent, expected, edits)
    manifest = manifest_for(expected)
    dest = tmp_path / "child"
    with pytest.raises(RuntimeError, match="INJECTED_MATERIALIZATION_FAILURE"):
        atomic_materialize(parent, dest, edits, manifest, fail_after_edit=True)
    assert digest_tree(parent) == before
    assert not dest.exists()


def test_content_mismatch_rejected_without_activation(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "a.txt").write_text("A")
    edits = [{"op": "write", "path": "b.txt", "data": b"B"}]
    expected = tmp_path / "expected"
    prepare_expected(parent, expected, edits)
    (expected / "extra.txt").write_text("unexpected")
    manifest = manifest_for(expected)
    dest = tmp_path / "child"
    with pytest.raises(ValueError, match="materialized content does not match selected manifest"):
        atomic_materialize(parent, dest, edits, manifest)
    assert not dest.exists()


def test_symlinked_parent_content_is_refused(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "a.txt").write_bytes(b"a")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"secret")
    (parent / "leak.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        digest_tree(parent)
    with pytest.raises(ValueError, match="symlink"):
        atomic_materialize(parent, tmp_path / "child", [], expected_manifest=None)
    assert not (tmp_path / "child").exists()


def test_symlinked_directory_is_refused(tmp_path):
    parent = tmp_path / "parent"
    (parent / "sub").mkdir(parents=True)
    (parent / "sub" / "x.txt").write_bytes(b"x")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "y.txt").write_bytes(b"y")
    (parent / "linked").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        digest_tree(parent)


def test_edit_paths_must_stay_inside_the_child(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "a.txt").write_bytes(b"a")
    for bad in ("../escape.txt", "/abs.txt", ""):
        with pytest.raises(ValueError):
            atomic_materialize(parent, tmp_path / "child", [{"op": "write", "path": bad, "data": b"x"}],
                               expected_manifest=None)
    assert not (tmp_path / "child").exists() and not (tmp_path / "escape.txt").exists()
