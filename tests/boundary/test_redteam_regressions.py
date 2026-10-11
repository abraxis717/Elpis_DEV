"""The post-red-team cognitive boundary, as one index of findings and the regressions that hold them.

Each red-team finding the post-red-team roster closed is listed with the law that closes it and the tests that
reproduce the attack and prove its refusal. This gate fails when a listed regression disappears or is renamed, so a
finding cannot silently lose its test. ``tests/boundary/test_redteam_mutations.py`` (STRESS lane) proves the Python
guards are load-bearing: removing each one makes its regression fail (mutation adequacy). Native guards (K1 lease,
RuntimeCore fuel, checkpoint protocol) are held by the Rust and C suites (``ctest -L runtime``/``-L ECS``).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

FINDINGS: dict[str, tuple[str, tuple[str, ...]]] = {
    "RT-01 QUERY could learn": (
        "QUERY is read-only: one native query_identity, no transaction, no publication; LEARN needs authority", (
            "tests/integration/test_query_learn.py::test_a_query_answers_f_w_of_the_authoritative_state_and_changes_nothing",
            "tests/integration/test_query_learn.py::test_a_managed_query_changes_no_state_and_no_continuity_byte",
            "tests/integration/test_query_learn.py::test_learn_without_explicit_authority_is_refused_before_any_state_is_touched",
        )),
    "RT-02 a codec authorized itself": (
        "classification is metadata; admission is an independently pinned catalog with separate capabilities", (
            "tests/integration/test_codec_authority.py::test_forged_classification_is_refused",
            "tests/integration/test_codec_authority.py::test_an_unknown_codec_identity_is_refused",
            "tests/integration/test_codec_authority.py::test_the_right_identity_with_the_wrong_bytes_is_refused",
            "tests/integration/test_codec_authority.py::test_the_right_bytes_under_the_wrong_authority_are_refused",
            "tests/integration/test_codec_authority.py::test_a_query_only_codec_never_learns_and_a_learn_only_codec_never_answers",
            "tests/integration/test_codec_authority.py::test_a_revoked_codec_is_refused",
        )),
    "RT-03 unbounded cognitive work": (
        "total integer fuel admitted before reserve, transaction or mutation; no deadline claimed", (
            "tests/integration/test_cognitive_fuel.py::test_the_per_field_maximum_schedule_is_refused_before_anything_native",
            "tests/integration/test_cognitive_fuel.py::test_each_total_binds_at_its_exact_integer_boundary",
            "tests/integration/test_cognitive_fuel.py::test_managed_oversized_learn_is_refused_before_reserve_begin_schedule_commit_and_publication",
        )),
    "RT-04 unmanaged mutation of a bound K1 state": (
        "managed lease: refused (LEASED) or fail-stop, never silently incorporated", (
            "tests/integration/test_managed_ownership.py::test_every_unmanaged_mutation_of_a_bound_state_is_refused_and_the_lineage_continues",
            "tests/integration/test_managed_ownership.py::test_a_second_runtime_taking_the_state_makes_the_first_fail_stop_before_any_mutation",
        )),
    "RT-05 native code by pathname": (
        "sealed, pinned admission of RuntimeCore, K1, K1 FMS and continuity, with dependency verification", (
            "tests/integration/test_native_admission.py::test_a_modified_library_is_refused",
            "tests/integration/test_native_admission.py::test_symlinks_and_escapes_are_refused",
            "tests/integration/test_native_admission.py::test_the_managed_runtime_refuses_k1_code_that_was_not_admitted",
            "tests/integration/test_native_admission.py::test_a_dependency_bound_to_unpinned_bytes_refuses_the_admission",
        )),
    "RT-06 an unrecoverable K1 state": (
        "K1 Recovery R0: two fixed slots; checkpoint bytes never authorize themselves; explicit dispositions", (
            "tests/integration/test_k1_recovery.py::test_checkpoint_bytes_never_authorize_themselves",
            "tests/integration/test_k1_recovery.py::test_a_crash_inside_a_checkpointed_learn_needs_explicit_reconciliation",
            "tests/integration/test_k1_recovery.py::test_anchor_and_every_learn_keep_the_complete_authorized_state_resumable_in_two_fixed_slots",
        )),
    "RT-07 evolution authorized and evaluated itself": (
        "pinned evolution policy: issued gates, independent evaluation, operator-approved promotion", (
            "tests/integration/test_evolution.py::test_only_the_pinned_policy_gate_reaches_reservation",
            "tests/evolution/test_policy.py::test_self_reported_or_altered_evidence_is_never_weighed",
            "tests/evolution/test_policy.py::test_a_candidate_cannot_edit_or_carry_its_evaluator",
            "tests/evolution/test_policy.py::test_an_assertion_cannot_widen_its_own_scope_budget_or_contract",
            "tests/evolution/test_policy.py::test_nothing_is_promoted_without_an_issued_selection_and_an_operator_grant",
        )),
    "RT-08 HACF evidence laundered into ECS": (
        "RESEARCH_ONLY bridge: validated structure only, identified maps, read-only QUERY, no writeback", (
            "tests/research/hacf_ecs_bridge_r0/test_bridge.py::test_every_degraded_evidence_condition_is_refused_or_recorded",
            "tests/research/hacf_ecs_bridge_r0/test_bridge.py::test_the_map_is_identified_by_measurement_not_by_its_own_claims",
            "tests/research/hacf_ecs_bridge_r0/test_bridge.py::test_the_bridge_has_no_learn_writeback_or_authority_path",
        )),
    "RT-09 fitness reported by the candidate": (
        "RESEARCH_ONLY environment: the evaluator recomputes fitness from replayed trajectories", (
            "tests/research/evolution_fitness_r0/test_fitness.py::test_the_evaluator_recomputes_fitness_and_never_reads_a_self_report",
            "tests/research/evolution_fitness_r0/test_fitness.py::test_leakage_malformed_and_baseline_candidates_fail_their_gates",
        )),
    "RT-10 ever-expanding persistence": (
        "every writer classified; the autonomous path leaves only fixed slots", (
            "tests/integration/test_autonomous_no_growth.py::test_ordinary_autonomous_path_leaves_only_the_two_continuity_slots",
            "tests/integration/test_autonomous_no_growth.py::test_a_provisioned_k1_checkpoint_stays_two_fixed_slots_on_the_autonomous_path",
        )),
}


def _defined(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


@pytest.mark.parametrize("finding", sorted(FINDINGS))
def test_every_finding_keeps_its_regressions(finding):
    law, nodes = FINDINGS[finding]
    assert law.strip() and nodes
    for node in nodes:
        path, name = node.split("::")
        assert name in _defined(REPO / path), f"{finding}: regression {node} is gone"


def test_every_mutant_names_a_listed_regression():
    from .test_redteam_mutations import MUTANTS
    listed = {node for _, nodes in FINDINGS.values() for node in nodes}
    for mutant in MUTANTS:
        assert mutant.killer in listed, (mutant.id, mutant.killer)
        assert mutant.finding in FINDINGS, mutant.id
