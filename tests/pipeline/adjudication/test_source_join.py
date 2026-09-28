"""Adjudication source join fails closed on incomplete or orphaned inputs."""
import copy

import pytest

from elpis.pipeline.adjudication.errors import (
    ProposalSetDuplicate, ProposalSetIncomplete, SourceJoinMissingRow,
)
from elpis.pipeline.adjudication.source_join import (
    build_row_map, join_source_row, verify_source_join,
)
from tests.pipeline import _chain


def test_verified_join_over_chain():
    groups = _chain.structural_groups()
    rows = build_row_map(groups)
    audit = verify_source_join(rows, groups, expected_row_count=len(_chain.synthetic_rows()))
    assert audit["status"] == "ADJUDICATION_SOURCE_JOIN_VERIFIED"


def test_row_count_is_pinned_by_caller():
    groups = _chain.structural_groups()
    rows = build_row_map(groups)
    audit = verify_source_join(rows, groups, expected_row_count=len(rows) + 1)
    assert audit["status"] == "SOURCE_JOIN_FAILED"
    with pytest.raises(ValueError):
        verify_source_join(rows, groups, expected_row_count=0)


def test_proposal_without_evidence_rejected():
    groups = copy.deepcopy(_chain.structural_groups())
    groups["proposals"][0]["evidence_digest"] = "0" * 64
    with pytest.raises(SourceJoinMissingRow):
        build_row_map(groups)


def test_evidence_for_unknown_row_rejected():
    groups = copy.deepcopy(_chain.structural_groups())
    groups["row_index"] = groups["row_index"][1:]
    with pytest.raises(SourceJoinMissingRow):
        build_row_map(groups)


def test_missing_row_rejected():
    rows = build_row_map(_chain.structural_groups())
    with pytest.raises(SourceJoinMissingRow):
        join_source_row("f" * 64, rows)


def test_incomplete_proposal_set_rejected():
    rows = build_row_map(_chain.structural_groups())
    src = sorted(rows)[0]
    rows[src]["proposals"] = rows[src]["proposals"][:4]
    with pytest.raises(ProposalSetIncomplete):
        join_source_row(src, rows)


def test_duplicate_proposal_rejected():
    rows = build_row_map(_chain.structural_groups())
    src = sorted(rows)[0]
    rows[src]["proposals"] = rows[src]["proposals"][:4] + rows[src]["proposals"][:1]
    with pytest.raises(ProposalSetDuplicate):
        join_source_row(src, rows)


def test_inadmissible_proposal_rejected():
    rows = build_row_map(_chain.structural_groups())
    src = sorted(rows)[0]
    rows[src]["proposals"][0] = dict(rows[src]["proposals"][0], admissible_for_adjudication=False)
    with pytest.raises(ProposalSetIncomplete):
        join_source_row(src, rows)
