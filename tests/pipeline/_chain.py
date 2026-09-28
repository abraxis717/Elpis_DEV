"""Deterministic in-memory structural chain used as a test fixture.

The synthetic rows of ``tests.structure.grid81._rows`` run through the real
stages:

    typed projection (structure) -> structural groups (structure)
    -> adjudication (pipeline) -> capability review requests
"""
from __future__ import annotations

from elpis.pipeline.adjudication.compiler import compile_row
from elpis.pipeline.adjudication.source_join import build_row_map
from tests.structure.grid81._rows import (
    SOURCE_MANIFEST_SHA, structural_groups, synthetic_rows, typed_inventories,
)

__all__ = [
    "SOURCE_MANIFEST_SHA", "structural_groups", "synthetic_rows", "typed_inventories",
    "adjudications", "review_requests",
]


def adjudications():
    groups = structural_groups()
    rows = build_row_map(groups)
    return [compile_row(src, rows, SOURCE_MANIFEST_SHA) for src in sorted(rows)]


def review_requests():
    return [result["review_request"] for result in adjudications()]
