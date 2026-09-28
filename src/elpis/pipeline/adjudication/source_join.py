"""Structural-group inventories -> per-row adjudication inputs (joined by source_row_digest)."""

from .errors import SourceJoinMissingRow, ProposalSetIncomplete, ProposalSetDuplicate


def build_row_map(inventories):
    """Build per-row data structures from inventories.

    Returns dict keyed by source_row_digest with:
      proposals: list of proposal records
      evidence: list of evidence records
      ordering: ordering record
      conflicts: list of conflict records
      row_index: row index record
    """
    # Build from row index as the primary key
    rows = {}
    for ri in inventories["row_index"]:
        src = ri["source_row_digest"]
        if src not in rows:
            rows[src] = {
                "proposals": [],
                "evidence": [],
                "ordering": None,
                "conflicts": [],
                "row_index": ri,
            }

    # Map proposals to rows via evidence_digest -> source_row_digest
    # First build evidence lookup: canonical_payload_digest -> source_row_digest
    evidence_map = {}
    for e in inventories["evidence"]:
        evidence_map[e["canonical_payload_digest"]] = e

    # Now map proposals; a proposal whose evidence or row is absent fails closed.
    for p in inventories["proposals"]:
        ev = evidence_map.get(p["evidence_digest"])
        if ev is None:
            raise SourceJoinMissingRow(f"Proposal {p['proposal_digest']} has no evidence record")
        src = ev["source_row_digest"]
        if src not in rows:
            raise SourceJoinMissingRow(f"Evidence {p['evidence_digest']} binds unknown row {src}")
        rows[src]["proposals"].append(p)
        rows[src]["evidence"].append(ev)

    # Map orderings
    for o in inventories["orderings"]:
        src = o["source_row_digest"]
        if src in rows:
            rows[src]["ordering"] = o

    # Map conflicts
    for c in inventories["conflicts"]:
        src = c["source_row_digest"]
        if src in rows:
            rows[src]["conflicts"].append(c)

    return rows


def join_source_row(source_row_digest, rows):
    """Join data for a single source row. Returns row data dict."""
    if source_row_digest not in rows:
        raise SourceJoinMissingRow(f"Row {source_row_digest} not found")

    row = rows[source_row_digest]

    # Verify proposal count
    if len(row["proposals"]) != 5:
        raise ProposalSetIncomplete(
            f"Row {source_row_digest} has {len(row['proposals'])} proposals, expected 5"
        )

    # Check for duplicate proposals
    proposal_digests = [p["proposal_digest"] for p in row["proposals"]]
    if len(set(proposal_digests)) != len(proposal_digests):
        raise ProposalSetDuplicate(
            f"Row {source_row_digest} has duplicate proposal digests"
        )

    # All proposals must be admissible
    for p in row["proposals"]:
        if not p.get("admissible_for_adjudication", False):
            raise ProposalSetIncomplete(
                f"Row {source_row_digest} has inadmissible proposal {p['proposal_digest']}"
            )

    return row


def verify_source_join(rows, inventories, *, expected_row_count):
    """Verify source join completeness against an explicit row count.

    Every row carries exactly one proposal and one evidence record per
    structural group, so proposal and evidence totals are pinned to
    ``5 * expected_row_count``. Returns the audit report.
    """
    if isinstance(expected_row_count, bool) or not isinstance(expected_row_count, int) or expected_row_count < 1:
        raise ValueError("expected_row_count must be a positive integer")
    expected_records = 5 * expected_row_count
    row_count = len(rows)
    total_proposals = sum(len(r["proposals"]) for r in rows.values())
    total_evidence = sum(len(r["evidence"]) for r in rows.values())
    total_conflicts = sum(len(r["conflicts"]) for r in rows.values())

    # Verify counts
    all_checks = []

    check = {"check": "row_count", "expected": expected_row_count, "actual": row_count, "pass": row_count == expected_row_count}
    all_checks.append(check)

    check = {"check": "proposal_count", "expected": expected_records, "actual": total_proposals, "pass": total_proposals == expected_records}
    all_checks.append(check)

    check = {"check": "evidence_count", "expected": expected_records, "actual": total_evidence, "pass": total_evidence == expected_records}
    all_checks.append(check)

    orderings_present = sum(1 for r in rows.values() if r["ordering"])
    check = {"check": "orderings_present", "expected": expected_row_count, "actual": orderings_present,
             "pass": orderings_present == expected_row_count}
    all_checks.append(check)

    # Check 5 proposals per row
    rows_with_5 = sum(1 for r in rows.values() if len(r["proposals"]) == 5)
    check = {"check": "five_proposals_per_row", "expected": expected_row_count, "actual": rows_with_5, "pass": rows_with_5 == expected_row_count}
    all_checks.append(check)

    # Check all proposals admissible
    all_admissible = all(
        p["admissible_for_adjudication"]
        for r in rows.values()
        for p in r["proposals"]
    )
    check = {"check": "all_proposals_admissible", "pass": all_admissible}
    all_checks.append(check)

    # Verify evidence binding
    evidence_map = {e["canonical_payload_digest"]: e for e in inventories["evidence"]}
    evidence_bindings_ok = True
    for r in rows.values():
        for p in r["proposals"]:
            ev = evidence_map.get(p["evidence_digest"])
            if not ev or ev["source_row_digest"] != r["row_index"]["source_row_digest"]:
                evidence_bindings_ok = False
                break
    check = {"check": "evidence_bindings_valid", "pass": evidence_bindings_ok}
    all_checks.append(check)

    all_pass = all(c["pass"] for c in all_checks)

    return {
        "row_count": row_count,
        "proposal_count": total_proposals,
        "evidence_count": total_evidence,
        "conflict_count": total_conflicts,
        "checks": all_checks,
        "all_pass": all_pass,
        "status": "ADJUDICATION_SOURCE_JOIN_VERIFIED" if all_pass else "SOURCE_JOIN_FAILED",
    }
