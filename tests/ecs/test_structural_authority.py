"""Structural R0 authority is identified by logical anchors, never by paths."""
import ast
from pathlib import Path

import pytest

from elpis.ecs import structural
from elpis.ecs.structural_authority import (
    HEADER_ANCHOR, MANIFEST_ANCHOR, authority_entry_anchor, read_authority_anchor,
)

ANCHORS = (HEADER_ANCHOR, MANIFEST_ANCHOR,
           *(authority_entry_anchor(name) for name in structural.EXPECTED))
PACKAGE = Path(structural.__file__).resolve().parent


def test_embedded_payload_matches_sealed_digests():
    # The donor's sealed SHA256SUMS digests are pinned in EXPECTED; the
    # embedded bytes must verify against them with no on-disk copy involved.
    identity = structural.verify_authority()
    assert identity == structural.materialize_authority().content_identity
    assert len(identity) == 64


@pytest.mark.parametrize("anchor", ANCHORS)
@pytest.mark.parametrize("fault", ("missing", "mutable", "wrong_type", "tampered"))
def test_every_anchor_fails_closed(anchor, fault):
    def reader(requested):
        raw = read_authority_anchor(requested)
        if requested != anchor:
            return raw
        if fault == "missing": raise KeyError(requested)
        if fault == "mutable": return bytearray(raw)
        if fault == "wrong_type": return raw.decode()
        return raw + b"\n"
    with pytest.raises(structural.R0Error):
        structural.materialize_authority(reader)


def test_core_has_no_filesystem_authority_mechanism():
    for name in ("structural.py", "structural_authority.py"):
        tree = ast.parse((PACKAGE / name).read_text())
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                       for alias in node.names)
        assert not imports.intersection({"os", "pathlib"})
        assert not {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}.intersection(
            {"__file__", "REPO_ROOT", "authority_root"})
    assert "authority_root" not in structural.FrozenTheta.__dataclass_fields__
