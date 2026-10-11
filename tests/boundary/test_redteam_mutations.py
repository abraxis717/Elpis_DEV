"""Mutation adequacy of the red-team regressions (STRESS lane; tests/lanes.py).

Each mutant removes one Python guard that closes a red-team finding (tests/boundary/test_redteam_regressions.py).
The repository's ``src``, ``tests`` and the two research laboratories are copied into a sandbox once; each mutant is
applied there (the exact original snippet must occur exactly once, so a mutant cannot silently drift from the code),
its killing regression is run in a fresh interpreter whose ``PYTHONPATH`` puts the sandbox first, and the regression
must fail. The unmutated sandbox must pass every killer first, with native libraries required, so a kill is never an
environment failure. Native guards are held by the Rust and C suites, not mutated here (rebuild cost).
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Mutant:
    id: str
    finding: str
    path: str
    original: str
    mutated: str
    killer: str


MUTANTS = (
    Mutant("M-evolve-unpinned-gate", "RT-07 evolution authorized and evaluated itself",
           "src/elpis/runtime/composition.py", "        self._require_policy_gate(gate)\n", "",
           "tests/integration/test_evolution.py::test_only_the_pinned_policy_gate_reaches_reservation"),
    Mutant("M-select-self-reported-evidence", "RT-07 evolution authorized and evaluated itself",
           "src/elpis/evolution/policy.py",
           "            if issued != (record.manifest.digest, record.edit_count):", "            if False:",
           "tests/evolution/test_policy.py::test_self_reported_or_altered_evidence_is_never_weighed"),
    Mutant("M-evaluator-in-candidate", "RT-07 evolution authorized and evaluated itself",
           "src/elpis/evolution/policy.py",
           '            _refuse("EVALUATOR_IN_CANDIDATE", "a candidate never carries or edits its evaluator")',
           "            pass",
           "tests/evolution/test_policy.py::test_a_candidate_cannot_edit_or_carry_its_evaluator"),
    Mutant("M-assertion-own-edit-budget", "RT-07 evolution authorized and evaluated itself",
           "src/elpis/evolution/path_gate.py", '            return "EDIT_BUDGET_NOT_AUTHORIZED"', "            pass",
           "tests/evolution/test_policy.py::test_an_assertion_cannot_widen_its_own_scope_budget_or_contract"),
    Mutant("M-promotion-without-grant", "RT-07 evolution authorized and evaluated itself",
           "src/elpis/evolution/policy.py",
           "        if type(grant) is not PromotionGrant or self._grants.get(grant.selection_receipt_digest) != grant:",
           "        if type(grant) is not PromotionGrant:",
           "tests/evolution/test_policy.py::test_nothing_is_promoted_without_an_issued_selection_and_an_operator_grant"),
    Mutant("M-codec-capability", "RT-02 a codec authorized itself", "src/elpis/runtime/codec_authority.py",
           "        if type(needed) is not frozenset or not needed or not needed <= entry.capabilities:",
           "        if type(needed) is not frozenset or not needed:",
           "tests/integration/test_codec_authority.py::"
           "test_a_query_only_codec_never_learns_and_a_learn_only_codec_never_answers"),
    Mutant("M-codec-classification", "RT-02 a codec authorized itself", "src/elpis/runtime/codec_authority.py",
           '        if getattr(codec, "classification", None) != entry.classification:', "        if False:",
           "tests/integration/test_codec_authority.py::test_forged_classification_is_refused"),
    Mutant("M-unadmitted-k1", "RT-05 native code by pathname", "src/elpis/runtime/core.py",
           "    if admitted is None or admitted.library_id != K1_LIBRARY_IDS[kind]:", "    if False:",
           "tests/integration/test_native_admission.py::test_the_managed_runtime_refuses_k1_code_that_was_not_admitted"),
    Mutant("M-bridge-absent-evidence", "RT-08 HACF evidence laundered into ECS",
           "research/hacf_ecs_bridge_r0/bridge.py", "    if not packet.items:", "    if False:",
           "tests/research/hacf_ecs_bridge_r0/test_bridge.py::test_every_degraded_evidence_condition_is_refused_or_recorded"),
    Mutant("M-bridge-unidentified-map", "RT-08 HACF evidence laundered into ECS",
           "research/hacf_ecs_bridge_r0/bridge.py", "    if map_identity(observation_map) != pin:", "    if False:",
           "tests/research/hacf_ecs_bridge_r0/test_bridge.py::test_the_map_is_identified_by_measurement_not_by_its_own_claims"),
    Mutant("M-fitness-leakage", "RT-09 fitness reported by the candidate",
           "research/evolution_fitness_r0/evaluator.py", "    leaked = _leaks(Path(candidate_root))",
           "    leaked = False",
           "tests/research/evolution_fitness_r0/test_fitness.py::"
           "test_leakage_malformed_and_baseline_candidates_fail_their_gates"),
    Mutant("M-fitness-self-report", "RT-09 fitness reported by the candidate",
           "research/evolution_fitness_r0/evaluator.py",
           '                           held_out_delta=held_score - fitness(parent, "HELD_OUT"),',
           '                           held_out_delta=int((Path(candidate_root) / "agent" / "fitness.txt").read_text()'
           '.split("=")[-1]) if (Path(candidate_root) / "agent" / "fitness.txt").exists() else '
           'held_score - fitness(parent, "HELD_OUT"),',
           "tests/research/evolution_fitness_r0/test_fitness.py::"
           "test_the_evaluator_recomputes_fitness_and_never_reads_a_self_report"),
)

_COPY = ("src", "tests", "research/hacf_ecs_bridge_r0", "research/evolution_fitness_r0", "research/__init__.py",
         "pyproject.toml", "ELPIS_SYSTEM.json")


def test_every_mutant_still_matches_the_code_exactly_once():
    for mutant in MUTANTS:
        assert (REPO / mutant.path).read_text(encoding="utf-8").count(mutant.original) == 1, mutant.id
    assert len({m.id for m in MUTANTS}) == len(MUTANTS)


def _env(sandbox: Path) -> dict:
    from ..conftest import native_build_dir
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST")}
    env.update(PYTHONPATH=str(sandbox / "src"), PYTHONDONTWRITEBYTECODE="1", ELPIS_REQUIRE_NATIVE="1",
               ELPIS_NATIVE_BUILD=str(native_build_dir()))
    return env


def _pytest(sandbox: Path, *nodes: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *nodes],
                          cwd=sandbox, env=_env(sandbox), capture_output=True, text=True, timeout=900)


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    from ..conftest import require_runtime_library
    require_runtime_library()            # the killers drive real native libraries
    root = tmp_path_factory.mktemp("mutation") / "repo"
    for name in _COPY:
        source, target = REPO / name, root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copyfile(source, target)
    probe = subprocess.run([sys.executable, "-c", "import elpis; print(elpis.__file__)"], cwd=root, env=_env(root),
                           capture_output=True, text=True, timeout=120)
    assert probe.stdout.strip().startswith(str(root / "src")), probe   # the sandbox's code is the code under test
    baseline = _pytest(root, *sorted({m.killer for m in MUTANTS}))
    assert baseline.returncode == 0, ("the unmutated sandbox must pass every killer", baseline.stdout[-4000:])
    return root


@pytest.mark.parametrize("mutant", MUTANTS, ids=[m.id for m in MUTANTS])
def test_the_regression_kills_the_mutant(sandbox, mutant):
    path = sandbox / mutant.path
    original = path.read_text(encoding="utf-8")
    assert original.count(mutant.original) == 1, mutant.id
    path.write_text(original.replace(mutant.original, mutant.mutated), encoding="utf-8")
    try:
        result = _pytest(sandbox, mutant.killer)
    finally:
        path.write_text(original, encoding="utf-8")
    assert result.returncode == 1, (mutant.id, "SURVIVED" if result.returncode == 0 else "ERROR",
                                    result.stdout[-3000:], result.stderr[-2000:])
