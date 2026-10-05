"""Retention R2 current-runtime regression contract (corrective RR2-CR0, after RET2F; the evidence is unchanged).

The v1 current-runtime clause demanded that every per-mechanism count recomputed on an arbitrary host equal the
primary QUAL host's counts. RET2E's own numerical-robustness gate (M) had already falsified that invariance for
two secondary rows: the forced-kernel children differed from the primary in exactly those rows, so gate M failed
and NUMERICALLY_FRAGILE was recorded true. A heterogeneous CI fleet then exposed the contradiction (main
af5b4c0, run #90). This module separates the question the regression answers from the one gate M answered:

* gate M (recorded, unchanged, FAILED): did every registered count stay invariant across the registered
  robustness kernels during QUAL? No.
* this regression (never skipped, runs on the host it is given, natural BLAS kernel): does the closed R2
  scientific result still hold on this runtime, with the recorded numerical fragility preserved as a finding?

Exact on this runtime: validity, mechanics, gates A-L, disposition, outcome, the frozen selected candidate, the
set of count rows, and every count row the QUAL record did not itself show to be kernel-sensitive (the selected
candidate, M0, M1, C1R, K1 and every gating ablation among them). Exempt from primary-host equality, and only
from that: the rows the QUAL record shows differing between its primary run and a robustness child. They are
derived from the immutable record, never named here, and may never include a decision-bearing row. Their
deviations are reported, not hidden, and cannot touch gate M, NUMERICALLY_FRAGILE, the selection, the
disposition or eligibility (none of which this regression computes or relaxes).

Nothing here depends on the identity of the CPU or the BLAS kernel; the recorded kernel names delimit historical
bitwise replay only (test_evidence.py).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LAB = REPO / "research" / "ecs_retention_r2"
QUAL_FILE = LAB / "evidence" / "qual" / "ecsg-retention-r2.v1.qual.json"
FROZEN_FILE = LAB / "frozen" / "ecsg-retention-r2.v1.frozen.json"
# The write-once pins of test_evidence.py, repeated so this contract never reads a different record.
QUAL_SHA256 = "e1be3bf6a572a06a0d99994fc6dff758883d286c8da125aa713d95525cb56869"
FROZEN_SHA256 = "03fc60974ef7bbfae20e3d15a214796214c29171ddd3cca47037e448840824da"

FRAGILE_LABEL = "KNOWN_R2_NUMERICALLY_FRAGILE_SECONDARY_ROWS"
VERDICT_KEYS = ("validity", "mechanics", "gates", "disposition", "outcome")
RECORDED_RESULT = {"disposition": "PARTIAL_REDUCTION", "outcome": "OUTCOME_C"}
# Rows that carry the recorded decision; the record may never make one of them exempt.
CONTROL_ROWS = ("M0", "M1")
GATING_ABLATIONS = ("ablation:state_removed", "ablation:extended_state_transplant_at_B",
                    "ablation:consolidation_reset_at_B")


class ContractError(AssertionError):
    pass


def _record(path: Path, sha256: str) -> dict:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ContractError(f"{path.name} is not the write-once record this contract is bound to")
    return json.loads(data)["body"]


def qual_body() -> dict:
    return _record(QUAL_FILE, QUAL_SHA256)


def frozen_body() -> dict:
    return _record(FROZEN_FILE, FROZEN_SHA256)


def selected_candidate(qual: dict) -> str:
    return qual["choices"]["selected"]["candidate"]


def decision_rows(qual: dict) -> tuple[str, ...]:
    return (selected_candidate(qual),) + CONTROL_ROWS + GATING_ABLATIONS


def recorded_fragile_rows(qual: dict) -> frozenset[str]:
    """The count rows the QUAL record itself shows differing between its primary run and a robustness child."""
    primary = qual["pre_robustness_verdict"]["counts"]
    rows: set[str] = set()
    for child in qual["robustness"].values():
        counts = child["comparable"]["counts"]
        rows |= set(primary) ^ set(counts)
        rows |= {row for row in primary if row in counts and counts[row] != primary[row]}
    return frozenset(rows)


def recorded_preconditions(qual: dict, frozen: dict) -> list[str]:
    """The closed R2 result the regression is bound to; any failure means the record is not the R2 record."""
    problems = []
    if qual["choices"] != frozen["choices"]:
        problems.append("QUAL choices differ from the frozen choices")
    for key, value in RECORDED_RESULT.items():
        if qual[key] != value or qual["pre_robustness_verdict"][key] != value:
            problems.append(f"recorded {key} is not {value}")
    if qual["gates"].get("M_numerical_robustness") is not False:
        problems.append("recorded gate M is not FAILED")
    if qual["qualifiers"].get("NUMERICALLY_FRAGILE") is not True:
        problems.append("recorded NUMERICALLY_FRAGILE is not true")
    if "M_numerical_robustness" in qual["pre_robustness_verdict"]["gates"]:
        problems.append("the pre-robustness verdict unexpectedly carries gate M")
    fragile = recorded_fragile_rows(qual)
    if not fragile:
        problems.append("the record shows no kernel-sensitive row; gate M could not have failed on counts")
    if fragile & set(decision_rows(qual)):
        problems.append(f"the record shows decision-bearing rows as kernel-sensitive: {sorted(fragile & set(decision_rows(qual)))}")
    return problems


def check(current: dict, selected: str, qual: dict, frozen: dict) -> dict:
    """Compare one current-runtime comparable verdict with the recorded one.

    Returns {"violations": [...], "fragile_deviations": {row: {key: [recorded, current]}}}. The regression
    passes only with no violation; fragile deviations are reported as the recorded finding.
    """
    violations = recorded_preconditions(qual, frozen)
    recorded = qual["pre_robustness_verdict"]
    if selected != selected_candidate(qual) or selected != frozen["choices"]["selected"]["candidate"]:
        violations.append(f"selected candidate {selected!r} is not the frozen {selected_candidate(qual)!r}")
    for key in VERDICT_KEYS:
        if current.get(key) != recorded[key]:
            violations.append(f"{key} differs from the recorded verdict")
    for key, value in RECORDED_RESULT.items():
        if current.get(key) != value:
            violations.append(f"current {key} is not {value}")

    fragile = recorded_fragile_rows(qual)
    counts, recorded_counts = current.get("counts", {}), recorded["counts"]
    if set(counts) != set(recorded_counts):
        violations.append(f"count rows differ: missing {sorted(set(recorded_counts) - set(counts))}, "
                          f"unexpected {sorted(set(counts) - set(recorded_counts))}")
    worlds = qual["world_count"]
    deviations = {}
    for row, want in recorded_counts.items():
        got = counts.get(row)
        if got is None:
            continue
        if row not in fragile:
            if got != want:
                violations.append(f"count row {row} differs and the QUAL record never showed it kernel-sensitive")
            continue
        if set(got) != set(want) or not all(isinstance(v, int) and 0 <= v <= worlds for v in got.values()):
            violations.append(f"{FRAGILE_LABEL} row {row} is malformed")
            continue
        diff = {k: [want[k], got[k]] for k in want if got[k] != want[k]}
        if diff:
            deviations[row] = diff

    sel = counts.get(selected, {})
    if sel and not (sel.get("SEQUENCE_HELD") == worlds and sel.get("CATASTROPHIC_any") == 0):
        violations.append(f"selected candidate {selected} no longer holds the sequence in every world")
    if counts.get("ablation:state_removed") != counts.get("M0"):
        violations.append("state_removed no longer reproduces M0 (recorded causal claim)")
    if counts.get("M1", {}).get("SEQUENCE_HELD") != worlds:
        violations.append("M1 (rehearsal) no longer holds the sequence in every world")
    return {"violations": violations, "fragile_deviations": deviations}
