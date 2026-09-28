"""Structural-group projection orchestration.

typed rows (joined) -> evidence -> proposals -> ordering -> conflicts -> row index
"""

from typing import Any, Dict, List

from elpis.structure.grid81.groups.canonical import canonical_json
from elpis.structure.grid81.groups.evidence import compile_all_evidence
from elpis.structure.grid81.groups.proposal import compile_all_proposals
from elpis.structure.grid81.groups.ordering import compile_all_orderings
from elpis.structure.grid81.groups.conflicts import compile_all_conflicts


def compile_row_index(
    joined_rows: List[Dict[str, Any]],
    evidence_records: List[Dict[str, Any]],
    proposals: List[Dict[str, Any]],
    orderings: List[Dict[str, Any]],
    conflicts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build the G50B_ROW_COMPILATION_INDEX.jsonl."""
    # Index conflicts by source_row_digest
    conflict_by_row = {}
    for c in conflicts:
        srd = c['source_row_digest']
        if srd not in conflict_by_row:
            conflict_by_row[srd] = []
        conflict_by_row[srd].append(c)

    index_rows = []
    ev_idx = 0
    prop_idx = 0
    for i, row in enumerate(joined_rows):
        srd = row['source_row_digest']
        row_evidence = evidence_records[ev_idx:ev_idx + 5]
        row_proposals = proposals[prop_idx:prop_idx + 5]

        evidence_digests = [e['canonical_payload_digest'] for e in row_evidence]
        proposal_digests = [p['proposal_digest'] for p in row_proposals]
        relevant_groups = [e['group_id'] for e in row_evidence if e['group_relevant']]

        row_conflicts = conflict_by_row.get(srd, [])
        conflict_digests = [c['canonical_conflict_digest'] for c in row_conflicts]

        ordering = orderings[i]

        index_rows.append({
            'source_row_digest': srd,
            'source_split': row['identity'].get('source_split', 'unknown'),
            'evidence_digests': evidence_digests,
            'proposal_digests': proposal_digests,
            'ordering_digest': ordering['ordering_digest'],
            'conflict_digests': conflict_digests,
            'relevant_group_ids': relevant_groups,
            'evidence_count': len(row_evidence),
            'proposal_count': len(row_proposals),
            'conflict_count': len(row_conflicts),
        })

        ev_idx += 5
        prop_idx += 5

    # Sort by source_row_digest
    index_rows.sort(key=lambda r: r['source_row_digest'])
    return index_rows


GROUP_IDS = ('TRANSITION_EDIT', 'TRANSITION_NOOP', 'EXPANSION_DECOMPOSITION', 'QUIESCENCE', 'RATIONALE_DIAGNOSTIC')


def compile_structural_groups(
    joined_rows: List[Dict[str, Any]],
    source_manifest_sha: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Compile joined typed rows into structural-group inventories in memory.

    Steps: evidence -> proposals -> ordering -> conflicts -> row index. The
    returned inventories use the canonical sort order of the persisted form
    and are exactly what the adjudication stage consumes
    (``elpis.pipeline.adjudication.source_join.build_row_map``).
    """
    evidence_records = compile_all_evidence(joined_rows, source_manifest_sha)
    proposals = compile_all_proposals(evidence_records)
    orderings = compile_all_orderings(joined_rows, evidence_records, proposals)
    conflicts = compile_all_conflicts(joined_rows, evidence_records, proposals)
    row_index = compile_row_index(joined_rows, evidence_records, proposals, orderings, conflicts)

    evidence_sorted = sorted(evidence_records, key=lambda e: (e['source_row_digest'], e['group_id']))
    evidence_by_digest = {e['canonical_payload_digest']: e for e in evidence_records}
    proposals_with_row = []
    for p in proposals:
        e = evidence_by_digest.get(p['evidence_digest'])
        srd = e['source_row_digest'] if e else ''
        proposals_with_row.append((srd, p['group_id'], canonical_json(p), p))
    proposals_with_row.sort(key=lambda item: item[:3])
    proposals_sorted = [p for _, _, _, p in proposals_with_row]
    orderings_sorted = sorted(orderings, key=lambda o: o['source_row_digest'])
    conflicts_sorted = sorted(conflicts, key=lambda c: (
        c['source_row_digest'], c['conflict_kind'], c['group_ids'], c['proposal_digests'],
    ))
    return {
        'evidence': evidence_sorted,
        'proposals': proposals_sorted,
        'orderings': orderings_sorted,
        'conflicts': conflicts_sorted,
        'row_index': row_index,
    }
