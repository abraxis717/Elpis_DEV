"""Corrective RR2-CR0: the current-runtime regression contract, tested against the immutable R2 record.

No NumPy, no native library, no laboratory run: synthetic current-runtime verdicts are derived from the recorded
QUAL verdict (and from its own forced-kernel robustness children, the record's evidence of kernel sensitivity)
and fed to the contract. The R2 evidence, specification and CANDIDATES.md stay pinned by test_evidence.py and
test_preregistration.py; historical replay stays in test_evidence.py, untouched.
"""
from __future__ import annotations

import copy
import hashlib
import re

import pytest

from . import _runtime_contract as C

QUAL = C.qual_body()
FROZEN = C.frozen_body()
RECORDED = QUAL["pre_robustness_verdict"]
SELECTED = C.selected_candidate(QUAL)
FRAGILE = C.recorded_fragile_rows(QUAL)


def _current(**changes) -> dict:
    out = copy.deepcopy({k: RECORDED[k] for k in C.VERDICT_KEYS + ("counts",)})
    out.update(changes)
    return out


def _check(current, selected=SELECTED):
    return C.check(current, selected, QUAL, FROZEN)


# --- the record the contract is bound to ----------------------------------------------------------------------


def test_contract_reads_the_pinned_write_once_records():
    for path, pin in ((C.QUAL_FILE, C.QUAL_SHA256), (C.FROZEN_FILE, C.FROZEN_SHA256)):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin, path.name


def test_closed_r2_result_is_unchanged_gate_m_failed_and_fragility_recorded():
    assert C.recorded_preconditions(QUAL, FROZEN) == []
    assert (QUAL["disposition"], QUAL["outcome"]) == ("PARTIAL_REDUCTION", "OUTCOME_C")
    assert QUAL["gates"]["M_numerical_robustness"] is False
    assert sorted(k for k, v in QUAL["gates"].items() if not v) == [
        "G_state_causality", "L_native_feasibility", "M_numerical_robustness"]
    assert QUAL["qualifiers"]["NUMERICALLY_FRAGILE"] is True
    assert SELECTED == "K3" and FROZEN["choices"] == QUAL["choices"]
    assert not any(child["identical"] for child in QUAL["robustness"].values())


def test_exempt_rows_are_exactly_the_rows_the_record_showed_kernel_sensitive():
    # Derived from the record (primary run vs. its robustness children), not chosen after observing CI.
    assert FRAGILE == {"K2", "ablation:uninformed_statistics"}
    assert not FRAGILE & set(C.decision_rows(QUAL))
    for child in QUAL["robustness"].values():
        counts = child["comparable"]["counts"]
        assert all(counts[row] == RECORDED["counts"][row] for row in RECORDED["counts"] if row not in FRAGILE)
        assert {k: child["comparable"][k] for k in C.VERDICT_KEYS} == {k: RECORDED[k] for k in C.VERDICT_KEYS}


# --- passes ---------------------------------------------------------------------------------------------------


def test_recorded_verdict_passes_with_no_deviation():
    assert _check(_current()) == {"violations": [], "fragile_deviations": {}}


@pytest.mark.parametrize("child", sorted(QUAL["robustness"]))
def test_another_kernel_passes_when_only_recorded_fragile_rows_vary(child):
    """Each robustness child of the record is a real different-kernel verdict; the contract must accept it and
    report exactly the recorded fragile rows. (The CI hosts of runs #88-#90 produced the same rows.)"""
    comparable = QUAL["robustness"][child]["comparable"]
    result = _check(copy.deepcopy({k: comparable[k] for k in C.VERDICT_KEYS + ("counts",)}))
    assert result["violations"] == []
    assert set(result["fragile_deviations"]) == FRAGILE


def test_other_values_of_fragile_rows_pass_but_are_reported():
    current = _current()
    current["counts"]["K2"]["SEQUENCE_HELD"] -= 1
    current["counts"]["ablation:uninformed_statistics"]["HELD_B"] -= 2
    result = _check(current)
    assert result["violations"] == []
    assert set(result["fragile_deviations"]) == FRAGILE


# --- failures -------------------------------------------------------------------------------------------------


def _fails(current, selected=SELECTED) -> list:
    violations = _check(current, selected)["violations"]
    assert violations
    return violations


@pytest.mark.parametrize("key", ["SEQUENCE_HELD", "CATASTROPHIC_any", "HELD_D", "RETAINED_A@D", "LEARNED_C",
                                 "refused_worlds"])
def test_any_change_to_the_selected_candidate_fails(key):
    current = _current()
    current["counts"][SELECTED][key] += -1 if current["counts"][SELECTED][key] else 1
    _fails(current)


def test_a_different_selected_candidate_fails():
    _fails(_current(), selected="K1")


@pytest.mark.parametrize("row", ["M0", "M1"])
@pytest.mark.parametrize("key", ["SEQUENCE_HELD", "CATASTROPHIC_any", "RETAINED_A@B"])
def test_control_drift_fails(row, key):
    current = _current()
    current["counts"][row][key] += -1 if current["counts"][row][key] else 1
    _fails(current)


def test_state_removed_no_longer_reproducing_m0_fails():
    current = _current()
    current["counts"]["ablation:state_removed"]["SEQUENCE_HELD"] += 1
    assert any("state_removed" in v for v in _fails(current))


@pytest.mark.parametrize("key", C.VERDICT_KEYS)
def test_verdict_change_fails(key):
    current = _current()
    value = current[key]
    if isinstance(value, dict):
        name = sorted(value)[0]
        value[name] = not value[name]
    else:
        current[key] = {"disposition": "REPLAY_ONLY", "outcome": "OUTCOME_B"}[key]
    _fails(current)


@pytest.mark.parametrize("gate", sorted(RECORDED["gates"]))
def test_every_gate_a_to_l_is_exact(gate):
    current = _current()
    current["gates"][gate] = not current["gates"][gate]
    _fails(current)


def test_gate_m_cannot_enter_through_the_current_verdict():
    current = _current()
    current["gates"]["M_numerical_robustness"] = True
    _fails(current)


@pytest.mark.parametrize("row", sorted(set(RECORDED["counts"]) - FRAGILE))
def test_an_unrecorded_row_becoming_kernel_sensitive_fails(row):
    current = _current()
    key = sorted(current["counts"][row])[0]
    current["counts"][row][key] += -1 if current["counts"][row][key] else 1
    assert any(row in v for v in _fails(current))


def test_missing_extra_or_malformed_rows_fail():
    current = _current()
    del current["counts"]["ablation:uninformed_statistics"]
    _fails(current)
    current = _current()
    current["counts"]["ablation:new"] = {"SEQUENCE_HELD": 0}
    _fails(current)
    current = _current()
    current["counts"]["K2"]["SEQUENCE_HELD"] = QUAL["world_count"] + 1
    _fails(current)
    current = _current()
    del current["counts"]["K2"]["HELD_D"]
    _fails(current)


def test_contract_refuses_a_record_that_would_exempt_a_decision_row():
    qual = copy.deepcopy(QUAL)
    qual["robustness"]["Prescott"]["comparable"]["counts"][SELECTED]["SEQUENCE_HELD"] -= 1
    assert any("decision-bearing" in p for p in C.recorded_preconditions(qual, FROZEN))
    assert C.check(_current(), SELECTED, qual, FROZEN)["violations"]


def test_contract_refuses_a_record_where_gate_m_passed_or_fragility_was_cleared():
    for mutate in (lambda q: q["gates"].__setitem__("M_numerical_robustness", True),
                   lambda q: q["qualifiers"].__setitem__("NUMERICALLY_FRAGILE", False)):
        qual = copy.deepcopy(QUAL)
        mutate(qual)
        assert C.check(_current(), SELECTED, qual, FROZEN)["violations"]


# --- portability ----------------------------------------------------------------------------------------------


def test_regression_is_not_conditional_on_the_cpu_or_blas_kernel():
    """The kernel identity delimits historical replay only. The contract and the regression neither branch on
    it nor force one (OPENBLAS_CORETYPE)."""
    here = C.REPO / "tests" / "research" / "ecs_retention_r2"
    kernels = re.compile(r"(?i)\b(zen|skylakex|cooperlake|prescott|haswell|sandybridge)\b|OPENBLAS_CORETYPE")
    for name in ("_runtime_contract.py", "test_runtime_regression.py"):
        code = "\n".join(line for line in (here / name).read_text(encoding="utf-8").splitlines()
                         if not line.lstrip().startswith("#"))
        code = re.sub(r'"""(?s:.*?)"""', "", code)
        assert not kernels.search(code), name


def test_correction_is_documented_without_altering_the_historical_claim():
    results = " ".join((C.REPO / "docs" / "research" / "ECS_RETENTION_R2_RESULTS.md").read_text(encoding="utf-8").split())
    for needle in ("Corrective note RR2-CR0", "TEST-CONTRACT / EVIDENCE-REPRODUCTION-CONTRACT defect",
                   "the original clause proved too strong for heterogeneous hosts", C.FRAGILE_LABEL,
                   "Gate M stays **FAILED**", "`NUMERICALLY_FRAGILE` stays true", "NO_CANONICAL_PROMOTION",
                   "**Disposition: `PARTIAL_REDUCTION` (`OUTCOME_C`).**", "The v1 clause demanded every count."):
        assert needle in results, needle
    policy = " ".join((C.REPO / "docs" / "CI_POLICY.md").read_text(encoding="utf-8").split())
    assert "Pull request #24" in policy and "`af5b4c0`" in policy and "History was not rewritten" in policy
