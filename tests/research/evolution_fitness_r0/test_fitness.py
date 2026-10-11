"""Evolution Fitness R0 infrastructure (research/evolution_fitness_r0). RESEARCH_ONLY; NO_CLAIM.

The environment is deterministic and replay-identified; its partitions are frozen and disjoint; the evaluator
recomputes fitness from replayed trajectories and never reads what a candidate says about itself; and an evolution
policy that pins it (elpis.evolution.policy) selects, rejects and promotes through it. Tiny corridors only.
"""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from elpis.evolution.policy import EvolutionPolicy, EvolutionPolicyAuthority, implementation_identity
from elpis.evolution.promotion import build_manifest_from_tree

from research.evolution_fitness_r0 import partitions as P
from research.evolution_fitness_r0.environment import (
    OBSERVATIONS,
    Action,
    EnvironmentSpec,
    World,
    replay,
    run,
    verify,
)
from research.evolution_fitness_r0.evaluator import FitnessRefusal, evaluate, fitness, policy_spec

REPO = Path(__file__).resolve().parents[3]
LAB = REPO / "research" / "evolution_fitness_r0"


def _table(choose):
    return {o: choose(int(o.split(":")[0])) for o in OBSERVATIONS}


SEEK = _table(lambda d: Action.RIGHT if d > 0 else Action.LEFT if d < 0 else Action.STAY)
RIGHTWARD = _table(lambda d: Action.RIGHT)
HESITANT = {**SEEK, "-1:3": Action.STAY}   # beats every negative baseline, yet never reaches a far-left goal


def test_the_environment_is_deterministic_and_replay_identified():
    world = World(P.IN_DISTRIBUTION, "dev-0")
    trajectory = run(world, SEEK.__getitem__)
    assert trajectory.steps[-1].next_state.done and verify(trajectory)
    assert replay(world, [s.action for s in trajectory.steps]) == trajectory
    assert run(world, SEEK.__getitem__).identity == trajectory.identity
    tampered = replace(trajectory, steps=trajectory.steps[:-1] + (replace(trajectory.steps[-1], feedback=99),))
    assert not verify(tampered)
    actions = [s.action for s in trajectory.steps]
    with pytest.raises(ValueError):
        replay(world, actions[:-1])
    with pytest.raises(ValueError):
        replay(world, actions + [Action.STAY])
    assert run(World(P.IN_DISTRIBUTION, "dev-1"), SEEK.__getitem__).identity != trajectory.identity
    assert run(World(EnvironmentSpec(10), "dev-0"), SEEK.__getitem__).identity != trajectory.identity
    # The episode score is the environment's: goal +10, every other step -1.
    assert trajectory.score == 10 - (len(trajectory.steps) - 1)


def test_partitions_are_frozen_disjoint_and_the_ood_partition_is_out_of_distribution():
    frozen = json.loads((LAB / "frozen" / "partitions.json").read_text())
    assert {n: v["digest"] for n, v in frozen["partitions"].items()} == {n: P.partition_digest(n) for n in P.PARTITIONS}
    assert {n: list(w) for n, (_, w) in P.PARTITIONS.items()} == {n: v["worlds"] for n, v in frozen["partitions"].items()}
    worlds = [w for _, ws in P.PARTITIONS.values() for w in ws]
    assert len(worlds) == len(set(worlds))
    assert P.PARTITIONS["OOD"][0] != P.PARTITIONS["HELD_OUT"][0] == P.PARTITIONS["CALIBRATION"][0]
    assert set(P.promotion_partitions()) == {"EVOLVE", "CALIBRATION", "HELD_OUT", "OOD"}
    assert len(set(P.promotion_partitions().values())) == 4


def test_negative_baselines_frame_the_measure():
    seek = fitness(SEEK, "HELD_OUT")
    assert all(fitness(b, "HELD_OUT") < seek for b in P.NEGATIVE_BASELINES.values())


def _workspace(root: Path, table, extra=None) -> Path:
    (root / "agent").mkdir(parents=True)
    (root / "agent" / "policy.json").write_bytes(P.policy_document(table))
    for name, data in (extra or {}).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    return root


def test_the_evaluator_recomputes_fitness_and_never_reads_a_self_report(tmp_path):
    parent = _workspace(tmp_path / "parent", RIGHTWARD)
    claim = {"agent/fitness.txt": b"held_out_delta=1000000000"}
    better = _workspace(tmp_path / "better", SEEK, claim)
    report = evaluate(parent_root=parent, candidate_root=better, partitions=P.promotion_partitions())
    assert report.held_out_delta == fitness(SEEK, "HELD_OUT") - fitness(RIGHTWARD, "HELD_OUT") > 0
    assert report.correctness_pass and report.leakage_pass
    boaster = _workspace(tmp_path / "boaster", RIGHTWARD, claim)
    assert evaluate(parent_root=parent, candidate_root=boaster, partitions=P.promotion_partitions()).held_out_delta == 0


def test_leakage_malformed_and_baseline_candidates_fail_their_gates(tmp_path):
    parent = _workspace(tmp_path / "parent", RIGHTWARD)
    leak = _workspace(tmp_path / "leak", SEEK, {"agent/notes.txt": b"tuned on held-7 and ood-2"})
    assert not evaluate(parent_root=parent, candidate_root=leak, partitions=P.promotion_partitions()).leakage_pass
    digest_leak = _workspace(tmp_path / "digest", SEEK, {"agent/x": P.partition_digest("HELD_OUT").encode()})
    assert not evaluate(parent_root=parent, candidate_root=digest_leak,
                        partitions=P.promotion_partitions()).leakage_pass
    malformed = _workspace(tmp_path / "malformed", SEEK)
    (malformed / "agent" / "policy.json").write_text('{"schema": "x", "table": {}}')
    assert not evaluate(parent_root=parent, candidate_root=malformed, partitions=P.promotion_partitions()).correctness_pass
    away = _workspace(tmp_path / "away", P.NEGATIVE_BASELINES["AWAY"])
    assert not evaluate(parent_root=parent, candidate_root=away, partitions=P.promotion_partitions()).correctness_pass


def test_the_evaluator_measures_only_its_frozen_partitions(tmp_path):
    parent = _workspace(tmp_path / "parent", RIGHTWARD)
    child = _workspace(tmp_path / "child", SEEK)
    other = {**P.promotion_partitions(), "HELD_OUT": hashlib.sha256(b"easier worlds").hexdigest()}
    with pytest.raises(FitnessRefusal) as info:
        evaluate(parent_root=parent, candidate_root=child, partitions=other)
    assert info.value.code == "PARTITION_MISMATCH"
    shutil.rmtree(parent / "agent")
    with pytest.raises(FitnessRefusal) as info:
        evaluate(parent_root=parent, candidate_root=child, partitions=P.promotion_partitions())
    assert info.value.code == "INCUMBENT_INVALID"


@pytest.fixture
def authority():
    raw = json.dumps(policy_spec(), sort_keys=True, separators=(",", ":")).encode()
    return EvolutionPolicyAuthority(EvolutionPolicy(raw, expected_sha256=hashlib.sha256(raw).hexdigest()), evaluate)


def _manifest(root, generation, parent=None, label="x"):
    return build_manifest_from_tree(root, generation=generation, parent_digest=parent,
                                    candidate_manifest_digest=hashlib.sha256(label.encode()).hexdigest(),
                                    path_receipt_digest=hashlib.sha256(("path-" + label).encode()).hexdigest(),
                                    editable_surface_digest=hashlib.sha256(b"agent").hexdigest())


def test_bound_into_the_policy_authority_selection_follows_recomputed_fitness(tmp_path, authority):
    assert (authority.policy.evaluator, authority.policy.evaluator_sha256) == implementation_identity(evaluate)
    assert max(fitness(b, "HELD_OUT") for b in P.NEGATIVE_BASELINES.values()) < fitness(HESITANT, "HELD_OUT") < \
        fitness(SEEK, "HELD_OUT")
    parent_root = _workspace(tmp_path / "parent", HESITANT)
    parent = _manifest(parent_root, 0, label="founder")
    children = {
        "seek": _workspace(tmp_path / "seek", SEEK),
        "same": _workspace(tmp_path / "same", HESITANT, {"agent/fitness.txt": b"10**9"}),
        "leak": _workspace(tmp_path / "leak", SEEK, {"agent/notes.txt": b"held-0"}),
        "scope": _workspace(tmp_path / "scope", SEEK, {"tools/helper.txt": b"outside the agent scope"}),
    }
    records = {name: authority.evaluate(parent=parent, candidate=_manifest(root, 1, parent.digest, name),
                                        parent_root=parent_root, candidate_root=root)
               for name, root in children.items()}
    receipt = authority.select(parent, list(records.values()))
    assert receipt.disposition == "SELECT_CHALLENGER"
    assert receipt.selected_harness_manifest_digest == records["seek"].manifest.digest
    reasons = dict(receipt.rejected)
    digest = {name: r.manifest.candidate_manifest_digest for name, r in records.items()}
    assert reasons[digest["same"]] == "WITHIN_NOISE_OR_NONIMPROVING"
    assert reasons[digest["leak"]] == "LEAKAGE_GATE_FAIL"
    assert reasons[digest["scope"]] == "SOURCE_SCOPE_GATE_FAIL"
    grant = authority.approve(receipt, operator_approval_digest=hashlib.sha256(b"operator").hexdigest())
    out = authority.materialize(grant, parent_dir=parent_root, dest_dir=tmp_path / "promoted",
                                edits=[{"op": "write", "path": "agent/policy.json", "data": P.policy_document(SEEK)}],
                                expected_manifest=records["seek"].manifest)
    assert out["status"] == "MATERIALIZED" and P.load_policy(tmp_path / "promoted") == P.load_policy(children["seek"])


def test_the_lab_writes_nothing_and_executes_nothing_from_a_workspace():
    for path in sorted(LAB.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert not names & {"write_bytes", "write_text", "mkdir", "unlink", "rename", "exec", "eval", "import_module",
                            "__import__", "run_learn", "Runtime"}, path.name
    for path in (REPO / "src").rglob("*.py"):
        assert "evolution_fitness_r0" not in path.read_text(encoding="utf-8"), path
