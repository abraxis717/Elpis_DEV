"""Retention R3 preregistration guard (RET3A). No laboratory, no worlds, no numerics.

* the R3 specification and PREREGISTRATION.md are write-once (pinned) and only RET3A commits ever touched them;
* inheritance from closed R2 is exact where R3 says it inherits (regime, task family, classes, K1 law, controls);
  the hidden gain is fixed (no grid); K1 is the sole eligible mechanism; K2 and K3 are not run;
* fresh worlds: R3's seed, name and world ids differ from R2's;
* the causality design reads the reset challenge at the first post-interference boundary (C), never at D;
* R3 is not weaker than R2 (classes, validity, joint quality, ablation thresholds, world count, budgets);
* gate L covers every decision-bearing quantity; the native budget matches the analytic K1 counts;
* RET3 commits never touch R0, R1, R2 or Cognitive/Runtime R1 authority; RET3A commits carry no implementation
  or result; records appear only after their predecessor step (early stop: RET3F may follow RET3C);
* until terminal evidence exists, no authority pointer claims an R3 result; no canonical ECS code names a
  retention mechanism before an R3 OUTCOME_A record exists.
"""
from __future__ import annotations

import hashlib
from itertools import combinations_with_replacement
import json
from pathlib import Path
import re
import subprocess

import pytest

from .._k1_promotion import admitted

REPO = Path(__file__).resolve().parents[3]
LAB = "research/ecs_retention_r3"
ROOT = REPO / LAB
SPEC_PATH = ROOT / "specs" / "ecsg-retention-r3.v1.spec.json"
PREREG_PATH = ROOT / "PREREGISTRATION.md"
R2_SPEC_PATH = REPO / "research" / "ecs_retention_r2" / "specs" / "ecsg-retention-r2.v1.spec.json"
RESULTS = "docs/research/ECS_RETENTION_R3_RESULTS.md"
BASE_MAIN = "1b5474bdd033aa8a32cb5ca3aa4a06d98bb337bb"

PINNED_SHA256 = {
    "specs/ecsg-retention-r3.v1.spec.json": "3900d6a9e995885ac888b1ebda5d52bb13df415b20879dab82f401b3910c9e33",
    "PREREGISTRATION.md": "46ee92c38d179e253c392c4908b4dec43ab6e2005982eaae7ddd6d5564e2582c",
}
STEPS = ("RET3A", "RET3B", "RET3C", "RET3D", "RET3E", "RET3F")
RECORD_LAW = (
    (f"{LAB}/evidence/dev", "RET3C", "RET3B"),
    (f"{LAB}/frozen", "RET3D", "RET3C"),
    (f"{LAB}/evidence/qual", "RET3E", "RET3D"),
    (RESULTS, "RET3F", "RET3E"),
)
EARLY_STOPS = ("TASK_INVALID_ON_DEV", "K1_STOPPED_ON_DEV")
RESULT_TOKENS = ("OUTCOME_A", "OUTCOME_B", "OUTCOME_C", "OUTCOME_D", "OUTCOME_V", "RETENTION_SUPPORTED",
                 "PARTIAL_REDUCTION", "REPLAY_ONLY", "NO_MATERIAL_IMPROVEMENT", "TASK_INVALID", "K1_STOPPED",
                 "K1_PROCEEDS", "qualified K1")
CLOSED = (re.compile(r"^research/ecs_(retention_r0|retention_r1|retention_r2|cognition_r0|runtime_r1)/"),
          re.compile(r"^tests/research/ecs_(retention_r0|retention_r1|retention_r2|cognition_r0|runtime_r1)/"),
          re.compile(r"^docs/research/(ECS_RETENTION_R[012]|COGNITION_R0)_RESULTS\.md$"))
HISTORY_INCOMPLETE = "RETENTION_R3_AUTHORITY_HISTORY_INCOMPLETE"
CHRONOLOGY_VIOLATED = "RETENTION_R3_CHRONOLOGY_VIOLATED"
PREMATURE_RESULT = "RETENTION_R3_PREMATURE_RESULT"

SPEC = json.loads(SPEC_PATH.read_text(encoding="ascii"))
R2 = json.loads(R2_SPEC_PATH.read_text(encoding="ascii"))
PREREG = PREREG_PATH.read_text(encoding="utf-8")


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def _tag(subject: str) -> str:
    return subject.split(" ", 1)[0]


def _is_step(tag: str, step: str) -> bool:
    return tag == step or tag.startswith(step + ".")


def _history(path: str) -> list[tuple[str, str]]:
    inside = _git("rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        pytest.fail(f"{HISTORY_INCOMPLETE}: {REPO} is not a git checkout; the chronology gate needs the history")
    log = _git("log", "--no-merges", "--format=%H %s", "--", path)
    if log.returncode != 0:
        pytest.fail(f"{HISTORY_INCOMPLETE}: git log failed for {path}: {log.stderr.strip()}")
    return [(line[:40], _tag(line[41:])) for line in log.stdout.splitlines() if line]


def _shallow_hint() -> str:
    shallow = _git("rev-parse", "--is-shallow-repository").stdout.strip() == "true"
    return " in a shallow checkout; check out full history (actions/checkout fetch-depth: 0)" if shallow else ""


def _exists(location: str) -> bool:
    path = REPO / location
    return path.is_dir() and any(path.glob("*.json"))


def _dev_stopped() -> bool:
    dev = REPO / LAB / "evidence" / "dev"
    records = sorted(dev.glob("*.json")) if dev.is_dir() else []
    stopped = any(json.loads(r.read_bytes())["body"].get("disposition") in EARLY_STOPS for r in records)
    if stopped:
        assert not _exists(f"{LAB}/frozen") and not _exists(f"{LAB}/evidence/qual"), (
            f"{CHRONOLOGY_VIOLATED}: frozen or QUAL evidence exists after an early stop on DEV")
    return stopped


def _qual_outcome() -> str | None:
    qual = REPO / LAB / "evidence" / "qual"
    records = sorted(qual.glob("*.json")) if qual.is_dir() else []
    return json.loads(records[0].read_bytes())["body"].get("outcome") if records else None


# --- the preregistration itself --------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PINNED_SHA256))
def test_preregistration_is_write_once(name):
    data = (ROOT / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == PINNED_SHA256[name], (
        f"{name} changed after RET3A; a changed experiment is a new version, never an edit")
    data.decode("ascii" if name.endswith(".json") else "utf-8")


def test_specification_is_complete():
    for key in ("question", "lineage", "regime", "task", "splits", "mechanisms", "state_semantics", "causality",
                "thresholds", "classes", "dev_rules", "validity", "mechanics_checks", "pass_rule", "native_budget",
                "evidence_contract", "numerical_binding", "promotion", "chronology", "predictions", "nonclaims"):
        assert key in SPEC, key
    assert SPEC["name"] == "ecsg-retention-r3" and SPEC["version"] == 1 and SPEC["authored_at"] == "RET3A"
    assert SPEC["base"]["main"] == BASE_MAIN


# --- inheritance from closed R2, and only where R3 says so --------------------------------------------------------


def test_inheritance_from_closed_r2_is_exact():
    assert SPEC["regime"] == R2["regime"]
    assert SPEC["classes"] == R2["classes"]
    assert SPEC["state_classes"] == R2["state_classes"]
    task = dict(SPEC["task"])
    for key in ("hidden_gain", "hidden_gain_origin", "role", "mismatched_plane"):
        task.pop(key)
    r2_task = {k: v for k, v in R2["task"].items() if k not in ("hidden_gain_grid", "role", "mismatched_plane")}
    assert task == r2_task
    assert SPEC["task"]["hidden_gain"] == 2.0 == R2["task"]["hidden_gain_grid"][0]
    assert "hidden_gain_grid" not in SPEC["task"], "R3 fixes the gain; no grid, no search"
    for name in ("M0", "M1", "O", "WSTAR"):
        assert SPEC["mechanisms"][name] == R2["mechanisms"][name], name
    for name in ("C1R", "K1"):
        for key in ("definition", "consolidation_inputs", "hyperparameters", "state"):
            assert SPEC["mechanisms"][name][key] == R2["mechanisms"][name][key], (name, key)
    assert SPEC["lineage"]["r2_spec_sha256"] == hashlib.sha256(R2_SPEC_PATH.read_bytes()).hexdigest()


def test_k1_is_the_sole_eligible_mechanism_and_k2_k3_are_not_run():
    mechs = SPEC["mechanisms"]
    assert set(mechs) == {"M0", "M1", "O", "WSTAR", "C1R", "K1"}
    eligible = [k for k, v in mechs.items() if v.get("eligible_for_selection")]
    assert eligible == ["K1"] and mechs["K1"]["class"] == "SOLE_ELIGIBLE_MECHANISM"
    assert "selects nothing" in mechs["K1"]["selection"]
    assert "candidate" not in SPEC["dev_rules"] and "K2" not in json.dumps(SPEC["pass_rule"])
    assert "K3" not in json.dumps(SPEC["pass_rule"]) and "K3" not in json.dumps(SPEC["mechanics_checks"])


def test_fresh_disjoint_worlds():
    assert SPEC["seed"] != R2["seed"] and SPEC["name"] != R2["name"]
    assert "r3dev-0000..r3dev-0007" in SPEC["splits"]["ids"] and "r3qual-0000..r3qual-0031" in SPEC["splits"]["ids"]
    assert SPEC["splits"]["dev_worlds"] == 8 and SPEC["splits"]["qual_worlds"] == 32


# --- the repaired causality design ------------------------------------------------------------------------------


def test_reset_challenge_is_read_at_the_first_post_interference_boundary():
    reset = SPEC["causality"]["reset_challenge"]
    steps = " | ".join(reset["procedure"])
    assert "RESET: H <- 0, a <- 0; keep W_B and epoch_B bitwise" in steps
    assert "IMMEDIATELY evaluate A and B" in steps and steps.rstrip().endswith(
        "STOP: no D, no further consolidation is evaluated in this branch")
    assert "before D" in reset["comparison"]
    gate_f = SPEC["pass_rule"]["F_declared_state_causality"]
    assert "every reset record ends at C" in gate_f and "at the C boundary" in gate_f
    th = SPEC["thresholds"]
    assert (th["reset_min_factor"], th["reset_min_absolute_nmse"], th["reset_min_degraded_fraction"],
            th["reset_min_median_ratio"]) == (10.0, 0.05, 0.75, 10.0)
    assert "e_reset >= 10 x e_uninterrupted and e_reset >= 0.05" == reset["RESET_DEGRADED_world"]


def test_every_required_causal_test_is_registered():
    c = SPEC["causality"]
    assert set(c) == {"state_removed", "full_state_transplant_at_B", "w_only_negative_control_at_B",
                      "reset_challenge"}
    assert "UNCONSOLIDATED" in c["w_only_negative_control_at_B"] and "ELPISG01" in c["w_only_negative_control_at_B"]
    assert "never counted as a retained-state transplant" in c["w_only_negative_control_at_B"]
    assert "(W, epoch, H, a)" in c["full_state_transplant_at_B"]
    assert SPEC["state_semantics"]["authoritative_state"] == ["W", "epoch", "H", "a"]
    for check in ("query_ignores_consolidation_state", "consolidation_state_shapes_learning", "reset_mechanics"):
        assert check in SPEC["mechanics_checks"], check


# --- not weaker than R2 -----------------------------------------------------------------------------------------


def test_r3_is_not_weaker_than_r2():
    th, t2 = SPEC["thresholds"], R2["thresholds"]
    for key in ("class_nmse_absolute", "class_nmse_relative", "witness_max_nmse", "capacity_max_ceiling_nmse",
                "novelty_min_nmse", "baseline_max_sequence_held_fraction", "baseline_min_median_final_earlier_nmse",
                "baseline_min_median_A_end_ratio", "qual_validity_min_fraction", "first_acquisition_median_nmse",
                "joint_quality_median_final_mean_nmse", "ablation_max_sequence_held_fraction",
                "ablation_min_median_factor", "partial_reduction_max_median_factor", "feature_identity_relative",
                "pte_s3_before_max", "pte_s3_after_min"):
        assert th[key] == t2[key], key
    assert th["native_hot_path_max_g1_multiple"] <= t2["native_hot_path_max_g1_multiple"]
    assert th["native_consolidation_max_ops"] <= t2["native_cold_path_max_ops_per_consolidation"]
    assert SPEC["splits"]["qual_worlds"] >= R2["splits"]["qual_worlds"]
    assert SPEC["validity"]["V2_forgetting"] == R2["validity"]["V2_forgetting"]
    assert SPEC["validity"]["V5_rehearsal_feasibility"] == R2["validity"]["V5_rehearsal_feasibility"]
    gates = list(SPEC["pass_rule"])[:12]
    assert [g[0] for g in gates] == list("ABCDEFGHIJKL")


def test_gate_l_covers_every_decision_bearing_quantity():
    record = SPEC["pass_rule"]["decision_record"]
    text = record["definition"]
    for item in ("validity", "mechanics", "gates A-K", "disposition", "outcome", "LEARNED_s", "RETAINED_t_at_s",
                 "CATASTROPHIC_t_at_s", "HELD_s", "SEQUENCE_HELD", "state_removed SEQUENCE_HELD", "RESET_DEGRADED",
                 "transplant bitwise", "W-only control pass", "M0 SEQUENCE_HELD", "M1 SEQUENCE_HELD"):
        assert item in text, item
    assert "any K1 decision-bearing kernel sensitivity fails R3" in record["principle"]
    assert "Prescott" in SPEC["pass_rule"]["L_numerical_robustness"] and "Haswell" in SPEC["pass_rule"][
        "L_numerical_robustness"]
    portability = SPEC["evidence_contract"]["portability"]
    for forbidden in ("CPU-model branches", "OPENBLAS_CORETYPE in ordinary CI", "per-host expected outputs",
                      "post-hoc tolerances"):
        assert forbidden in portability, forbidden


def test_native_budget_matches_the_analytic_k1_counts():
    reg = SPEC["regime"]
    d, n, rows = reg["dim"], reg["width"], reg["train_rows"]
    P = len(list(combinations_with_replacement(range(d), 2)))
    T = len(list(combinations_with_replacement(range(d), 3)))
    F = d + P + T
    assert F == reg["features"] == 83
    g1 = 4 * rows * n * d + 12 * rows * n
    hot = n * (d + 2 * P + 3 * T) + 2 * F * F + n * (d + 2 * d * d + 9 * T)
    consolidation = rows * F + rows * F * (F + 1) + F * (F + 1) // 2 + n * (d + 2 * P + 3 * T)
    budget = SPEC["native_budget"]
    assert (budget["g1_ops_per_step_at_R64"], budget["k1_hot_path_extra_ops_per_step"]) == (g1, hot) == (82944, 42506)
    assert consolidation == 462782 and "462,782" in PREREG
    assert 8 * (F * (F + 1) // 2 + F) == budget["k1_persistent_extra_bytes"] == SPEC["thresholds"][
        "native_persistent_extra_bytes"] == 28552
    assert hot <= SPEC["thresholds"]["native_hot_path_max_g1_multiple"] * g1
    assert consolidation <= SPEC["thresholds"]["native_consolidation_max_ops"]


# --- chronology from git history -------------------------------------------------------------------------------


def test_preregistration_was_only_ever_touched_by_ret3a():
    for name in PINNED_SHA256:
        path = f"{LAB}/{name}"
        commits = _history(path)
        assert commits, f"{HISTORY_INCOMPLETE}: no visible commit touches {path}{_shallow_hint()}"
        assert all(_is_step(tag, "RET3A") for _, tag in commits), (
            f"{CHRONOLOGY_VIOLATED}: {path} touched by {[t for _, t in commits]}; only RET3A commits may")


def _ret3_commits() -> list[tuple[str, str]]:
    log = _git("log", "--no-merges", "--format=%H %s", f"{BASE_MAIN}..HEAD")
    return [(line[:40], _tag(line[41:])) for line in log.stdout.splitlines()
            if line and any(_is_step(_tag(line[41:]), s) for s in STEPS)]


def test_ret3_commits_never_touch_closed_authority():
    for sha, tag in _ret3_commits():
        files = _git("show", "--no-renames", "--name-only", "--format=", sha).stdout.split()
        offenders = [f for f in files if any(p.search(f) for p in CLOSED)]
        assert not offenders, f"{CHRONOLOGY_VIOLATED}: {tag} commit {sha[:12]} touches closed authority {offenders}"


def test_ret3a_commits_contain_no_implementation_or_results():
    ret3a = [sha for sha, tag in _history(LAB) if _is_step(tag, "RET3A")]
    assert ret3a, f"{HISTORY_INCOMPLETE}: no RET3A commit touches {LAB}{_shallow_hint()}"
    forbidden = (re.compile(r"^(src|native)/"), re.compile(r"^research/.*\.py$"),
                 re.compile(rf"^{LAB}/(evidence|frozen)/"), re.compile(rf"^{re.escape(RESULTS)}$"))
    for sha in ret3a:
        files = _git("show", "--no-renames", "--name-only", "--format=", sha).stdout.split()
        offenders = [f for f in files if any(p.search(f) for p in forbidden)]
        assert not offenders, f"{CHRONOLOGY_VIOLATED}: RET3A commit {sha[:12]} carries {offenders}"


def _records(location: str) -> list[str]:
    path = REPO / location
    if path.is_file():
        return [location]
    return sorted(f"{location}/{p.name}" for p in path.glob("*.json")) if path.is_dir() else []


def test_records_appear_only_after_their_chronological_step():
    lab_history = _history(LAB) + _history(RESULTS)
    order = {sha: i for i, (sha, _) in enumerate(_history("."))}
    ranks = [STEPS.index(next(s for s in STEPS if _is_step(tag, s)))
             for _, tag in sorted(lab_history, key=lambda c: order.get(c[0], -1))
             if any(_is_step(tag, s) for s in STEPS)]
    assert ranks == sorted(ranks, reverse=True), f"{CHRONOLOGY_VIOLATED}: RET3 steps out of order {ranks}"
    for location, step, predecessor in RECORD_LAW:
        for record in _records(location):
            adds = _git("log", "--no-merges", "--diff-filter=A", "--format=%H %s", "--", record).stdout.splitlines()
            assert adds, f"{HISTORY_INCOMPLETE}: no visible commit adds {record}{_shallow_hint()}"
            assert len(adds) == 1 and _is_step(_tag(adds[0][41:]), step), (
                f"{PREMATURE_RESULT}: {record} added by {[a[41:] for a in adds]}; expected one {step} commit")
            if location == RESULTS and _dev_stopped():
                predecessor = "RET3C"
            added_at = order[adds[0][:40]]
            before = [order[sha] for sha, tag in lab_history if _is_step(tag, predecessor)]
            assert before and min(before) > added_at, (
                f"{PREMATURE_RESULT}: {record} ({step}) has no {predecessor} commit before it")


# --- no result before its time, no promotion before OUTCOME_A ----------------------------------------------------


def _pointers() -> dict:
    system = json.loads((REPO / "ELPIS_SYSTEM.json").read_text(encoding="utf-8"))
    component = next(c for c in system["research"]["components"] if c["path"] == LAB)
    ecsg = next(s for s in system["subsystems"] if s["id"] == "ECS")
    interface = next(i for i in ecsg["incomplete_interfaces"] if i.startswith("Retention R3"))
    cognition = (REPO / "docs" / "COGNITION_R0.md").read_text(encoding="utf-8")
    return {"component": component["classification"], "ECS interface": interface,
            "COGNITION_R0": cognition.split("**Retention R3**", 1)[1].split("\n\n", 1)[0]}


def test_no_result_is_claimed_before_terminal_evidence():
    if _exists(f"{LAB}/evidence/qual") or _dev_stopped():
        pytest.skip("terminal evidence exists; result claims are governed by the RET3F interpretation tests")
    assert not (REPO / RESULTS).exists(), f"{PREMATURE_RESULT}: {RESULTS} exists without terminal evidence"
    for where, text in _pointers().items():
        assert "preregistered" in text.lower(), where
        found = [token for token in RESULT_TOKENS if token.lower() in text.lower()]
        assert not found, f"{PREMATURE_RESULT}: {where} claims {found} before terminal evidence exists"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "`PREREGISTERED`" in readme and not re.search(r"(?m)^## Result", readme)


def test_no_canonical_retention_mechanism_before_an_r3_outcome_a_record():
    """Before OUTCOME_A no canonical file names a retention mechanism; after it, only the K1 milestone's files do,
    and only once the native milestone is QUALIFIED (tests/research/_k1_promotion.py)."""
    vocabulary = re.compile(r"(?i)consolidat|retention|laplace|fibre|recondition|rehears")
    sources = sorted((REPO / "src" / "elpis" / "ECS").rglob("*.py")) + sorted(
        p for p in (REPO / "native" / "ECS").rglob("*") if p.suffix in (".c", ".h"))
    exempt = admitted(REPO)
    assert not exempt or _qual_outcome() == "OUTCOME_A"
    offenders = [str(p.relative_to(REPO)) for p in sources if vocabulary.search(p.read_text(encoding="utf-8"))
                 and str(p.relative_to(REPO)) not in exempt]
    assert not offenders, offenders
