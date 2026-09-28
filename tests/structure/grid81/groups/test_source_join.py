"""Typed-inventory join fails closed on count, duplicate and orphan defects."""
import pytest

from elpis.structure.grid81.groups.source_join import join_inventory_records
from tests.structure.grid81 import _rows


def _views():
    inv = _rows.typed_inventories()
    return [inv["identity"], inv["transition"], inv["expansion"], inv["quiescence"], inv["rationale"]]


def test_join_verified_with_explicit_count():
    joined, audit = join_inventory_records(*_views(), expected_row_count=len(_rows.synthetic_rows()))
    assert audit["status"] == "SOURCE_JOIN_VERIFIED"
    assert [r["source_row_digest"] for r in joined] == sorted(r["source_row_digest"] for r in joined)


def test_count_mismatch_fails():
    _, audit = join_inventory_records(*_views(), expected_row_count=len(_rows.synthetic_rows()) + 1)
    assert audit["status"] == "SOURCE_JOIN_FAILED"


@pytest.mark.parametrize("count", [0, -1, True, 3.0, None])
def test_count_must_be_positive_integer(count):
    with pytest.raises(ValueError):
        join_inventory_records(*_views(), expected_row_count=count)


def test_duplicate_view_fails():
    views = _views()
    views[1] = views[1] + views[1][:1]
    _, audit = join_inventory_records(*views, expected_row_count=len(_rows.synthetic_rows()))
    assert audit["status"] == "SOURCE_JOIN_FAILED"
    assert audit["no_duplicates"] is False


def test_orphan_view_fails():
    views = _views()
    views[2] = views[2] + [dict(views[2][0], source_row_digest="e" * 64)]
    joined, audit = join_inventory_records(*views, expected_row_count=len(_rows.synthetic_rows()))
    assert audit["status"] == "SOURCE_JOIN_FAILED"
    assert audit["no_orphan_views"] is False
    assert joined == []


def test_missing_view_row_fails():
    views = _views()
    views[4] = views[4][1:]
    joined, audit = join_inventory_records(*views, expected_row_count=len(_rows.synthetic_rows()))
    assert audit["status"] == "SOURCE_JOIN_FAILED"
    assert joined == []
