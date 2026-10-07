"""Retention R2 preregistration guard (RET2A). No laboratory, no worlds, no numerics.

* the R2 specification and CANDIDATES.md are write-once (pinned) and only RET2A commits ever touched them;
* experimental isolation: R2's mechanisms, state, runtime, FMS, binding and evidence sections are R1's verbatim,
  and R1 v1 stays closed (its authority files are pinned by its own guard);
* the task is the registered aliasing family; the DEV rule is control-only with V1-V6; R2 is not weaker than R1;
* RET2A commits contain no implementation or result and touch no R0/R1 evidence;
* a DEV, frozen, QUAL or results record exists only if one commit of its step added it after its predecessor
  (the registered early stop TASK_INVALID_ON_DEV lets RET2F follow RET2C directly);
* until terminal evidence exists, no authority pointer claims an R2 result; no canonical ECS code names a
  retention mechanism.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

import pytest

from .._k1_promotion import admitted, field_admitted

REPO = Path(__file__).resolve().parents[3]
LAB = "research/ecs_retention_r2"
ROOT = REPO / LAB
SPEC_PATH = ROOT / "specs" / "ecsg-retention-r2.v1.spec.json"
CANDIDATES_PATH = ROOT / "CANDIDATES.md"
R1_SPEC_PATH = REPO / "research" / "ecs_retention_r1" / "specs" / "ecsg-retention-r1.v1.spec.json"
RESULTS = "docs/research/ECS_RETENTION_R2_RESULTS.md"
BASE_MAIN = "fb1a96db6eb10f02f6b87cb5dfd2f490ab1ba0c5"

PINNED_SHA256 = {
    "specs/ecsg-retention-r2.v1.spec.json": "b9930ab6580c11f690319eae2fd888ee1139287b40e4c8097766424a0e96f1df",
    "CANDIDATES.md": "9fde482ad54f9e32e2781db636fbdc8f317545cbb73ddfeafb5845168fd6eaff",
}
STEPS = ("RET2A", "RET2B", "RET2C", "RET2D", "RET2E", "RET2F")
RECORD_LAW = (
    (f"{LAB}/evidence/dev", "RET2C", "RET2B"),
    (f"{LAB}/frozen", "RET2D", "RET2C"),
    (f"{LAB}/evidence/qual", "RET2E", "RET2D"),
    (RESULTS, "RET2F", "RET2E"),
)
RESULT_TOKENS = ("OUTCOME_A", "OUTCOME_B", "OUTCOME_C", "OUTCOME_D", "OUTCOME_V", "RETENTION_SUPPORTED",
                 "PARTIAL_REDUCTION", "REPLAY_ONLY", "NO_MATERIAL_IMPROVEMENT", "TASK_INVALID", "selected candidate is",
                 "winner")
HISTORY_INCOMPLETE = "RETENTION_R2_AUTHORITY_HISTORY_INCOMPLETE"
CHRONOLOGY_VIOLATED = "RETENTION_R2_CHRONOLOGY_VIOLATED"
PREMATURE_RESULT = "RETENTION_R2_PREMATURE_RESULT"
REUSED = ("mechanisms", "state_classes", "state_accounting", "runtime_feasibility", "fms_implications", "metrics",
          "numerical_binding", "promotion_eligibility")

SPEC = json.loads(SPEC_PATH.read_text(encoding="ascii"))
R1 = json.loads(R1_SPEC_PATH.read_text(encoding="ascii"))
CANDIDATES = CANDIDATES_PATH.read_text(encoding="utf-8")


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
    stopped = any(json.loads(r.read_bytes())["body"].get("disposition") == "TASK_INVALID_ON_DEV" for r in records)
    if stopped:
        assert not _exists(f"{LAB}/frozen") and not _exists(f"{LAB}/evidence/qual"), (
            f"{CHRONOLOGY_VIOLATED}: frozen or QUAL evidence exists after TASK_INVALID_ON_DEV")
    return stopped


# --- the preregistration itself --------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PINNED_SHA256))
def test_preregistration_is_write_once(name):
    data = (ROOT / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == PINNED_SHA256[name], (
        f"{name} changed after RET2A; a changed experiment is a new version, never an edit")
    data.decode("ascii" if name.endswith(".json") else "utf-8")


def test_specification_is_complete():
    required = {"name", "version", "authored_at", "labels", "lineage", "question", "base", "starting_point",
                "analysis", "chronology", "regime", "task", "seed", "splits", "state_classes", "shared_procedure",
                "mechanisms", "state_accounting", "runtime_feasibility", "fms_implications", "excluded_families",
                "classes", "thresholds", "metrics", "dev_rules", "validity", "pass_rule", "mechanics_checks",
                "ablations", "secondary", "predictions", "numerical_binding", "evidence_contract",
                "promotion_eligibility", "nonclaims"}
    assert required <= set(SPEC), sorted(required - set(SPEC))
    assert SPEC["name"] == "ecsg-retention-r2" and SPEC["version"] == 1 and SPEC["authored_at"] == "RET2A"
    assert SPEC["base"]["main"] == BASE_MAIN and SPEC["seed"] != R1["seed"]
    assert tuple(SPEC["chronology"]["order"]) == STEPS
    for label in ("RESEARCH_ONLY", "NO_RUNTIME_AUTHORITY", "NO_LANGUAGE_CLAIM", "SEMANTICS=NONE", "PREREGISTERED"):
        assert label in SPEC["labels"], label
    assert set(SPEC["pass_rule"]) == set(R1["pass_rule"])


def test_experimental_isolation_from_r1():
    """The mechanisms and everything that depends only on them are R1's, verbatim; only the task changes."""
    for key in REUSED:
        assert SPEC[key] == R1[key] or (key == "mechanisms" and _same_mechanisms()), key
    assert SPEC["regime"] == R1["regime"]
    reworded = {"capacity_expansion"}   # restated for the single-task design; every other exclusion verbatim
    assert set(SPEC["excluded_families"]) == set(R1["excluded_families"])
    assert all(SPEC["excluded_families"][k] == R1["excluded_families"][k] for k in R1["excluded_families"]
               if k not in reworded)
    for key in ("A_first_acquisition", "B_stage_acquisition", "C_sequence_retention", "D_no_catastrophe",
                "E_joint_quality", "F_joint_state", "G_state_causality", "H_no_external_answer_store",
                "J_control_honesty", "K_capacity_accounting", "L_native_feasibility"):
        assert SPEC["pass_rule"][key] == R1["pass_rule"][key], key
    for key in ("LEARNED_s", "RETAINED_t_at_s", "HELD_at_s", "SEQUENCE_HELD", "CATASTROPHIC_t_at_s"):
        assert SPEC["classes"][key] == R1["classes"][key], key


def _same_mechanisms():
    a, b = dict(SPEC["mechanisms"]), dict(R1["mechanisms"])
    a["WSTAR"], b["WSTAR"] = dict(a["WSTAR"], definition=""), dict(b["WSTAR"], definition="")
    return a == b


def test_task_is_the_registered_aliasing_family():
    task = SPEC["task"]
    assert task["aliasing_angle_degrees"] == 45.0 and task["input_scale"] == 1.0
    assert task["hidden_gain_grid"] == [2.0, 2.5, 1.5]
    assert task["planes"]["A"] == "[q1, q2, q3]"
    for t, k in (("B", "1"), ("C", "2"), ("D", "3")):
        assert f"sin(theta) q{int(k) + 3}" in task["planes"][t] and f"cos(theta) q{k}" in task["planes"][t]
    assert "W* = [T* | 0]" in task["witness"] and "g q4, g q5, g q6" in task["teacher"]
    assert "aliased axis of B" in task["pte"]
    assert "Haar" in task["basis"]
    for needle in ("## 2. Why R1's task could not forget", "Lemma", "## 3. The primary task family",
                   "Design analysis disclosed", "No candidate, reference or consolidation mechanism was run"):
        assert needle in " ".join(CANDIDATES.split()), needle


def test_dev_rule_is_control_only_with_six_validity_conditions():
    rule = SPEC["dev_rules"]["task"]
    assert "controls only" in rule and "no candidate, no reference" in rule
    for v in ("V1", "V2", "V3", "V4", "V5", "V6"):
        assert v in rule, v
    assert "final_earlier_mean_nmse >= 0.5" in rule and "A_end_ratio >= 10" in rule
    assert "nmse_s(O_prev) >= 0.5" in rule and "M1 SEQUENCE_HELD in every DEV world" in rule
    assert "TASK_INVALID_ON_DEV" in rule
    assert "DEV chooses only the hidden gain and the selected candidate" in SPEC["dev_rules"]["nothing_else"]


def test_r2_is_not_weaker_than_r1():
    t, r = SPEC["thresholds"], R1["thresholds"]
    for key in ("class_nmse_absolute", "class_nmse_relative", "first_acquisition_median_nmse",
                "joint_quality_median_final_mean_nmse", "baseline_max_sequence_held_fraction",
                "ablation_max_sequence_held_fraction", "reset_max_both_retained_fraction"):
        assert t[key] <= r[key], key
    for key in ("candidate_world_fraction", "baseline_min_median_A_end_ratio", "ablation_min_median_factor",
                "novelty_min_nmse"):
        assert t[key] >= r[key], key
    assert t["novelty_min_nmse"] == 0.5 > r["novelty_min_nmse"]
    assert t["baseline_min_median_final_earlier_nmse"] == 0.5 and t["capacity_max_ceiling_nmse"] == 1e-6
    assert SPEC["splits"]["qual_worlds"] >= R1["splits"]["qual_worlds"]
    assert SPEC["splits"]["dev_worlds"] >= R1["splits"]["dev_worlds"]


# --- chronology from git history -------------------------------------------------------------------------------


def test_preregistration_was_only_ever_touched_by_ret2a():
    for name in PINNED_SHA256:
        path = f"{LAB}/{name}"
        commits = _history(path)
        assert commits, f"{HISTORY_INCOMPLETE}: no visible commit touches {path}{_shallow_hint()}"
        assert all(_is_step(tag, "RET2A") for _, tag in commits), (
            f"{CHRONOLOGY_VIOLATED}: {path} touched by {[t for _, t in commits]}; only RET2A commits may")


def test_ret2a_commits_contain_no_implementation_or_results():
    ret2a = [sha for sha, tag in _history(LAB) if _is_step(tag, "RET2A")]
    assert ret2a, f"{HISTORY_INCOMPLETE}: no RET2A commit touches {LAB}{_shallow_hint()}"
    forbidden = (re.compile(r"^(src|native)/"), re.compile(r"^research/.*\.py$"),
                 re.compile(rf"^{LAB}/(evidence|frozen)/"),
                 re.compile(r"^research/ecs_(retention_r0|retention_r1|cognition_r0)/"),
                 re.compile(rf"^{re.escape(RESULTS)}$"))
    for sha in ret2a:
        files = _git("show", "--no-renames", "--name-only", "--format=", sha).stdout.split()
        offenders = [f for f in files if any(p.search(f) for p in forbidden)]
        assert not offenders, f"{CHRONOLOGY_VIOLATED}: RET2A commit {sha[:12]} carries {offenders}"


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
    assert ranks == sorted(ranks, reverse=True), f"{CHRONOLOGY_VIOLATED}: RET2 steps out of order {ranks}"
    for location, step, predecessor in RECORD_LAW:
        for record in _records(location):
            adds = _git("log", "--no-merges", "--diff-filter=A", "--format=%H %s", "--", record).stdout.splitlines()
            assert adds, f"{HISTORY_INCOMPLETE}: no visible commit adds {record}{_shallow_hint()}"
            assert len(adds) == 1 and _is_step(_tag(adds[0][41:]), step), (
                f"{PREMATURE_RESULT}: {record} added by {[a[41:] for a in adds]}; expected one {step} commit")
            if location == RESULTS and _dev_stopped():
                predecessor = "RET2C"
            added_at = order[adds[0][:40]]
            before = [order[sha] for sha, tag in lab_history if _is_step(tag, predecessor)]
            assert before and min(before) > added_at, (
                f"{PREMATURE_RESULT}: {record} ({step}) has no {predecessor} commit before it")


# --- no result before its time, no promotion ------------------------------------------------------------------


def test_no_result_is_claimed_before_terminal_evidence():
    if _exists(f"{LAB}/evidence/qual") or _dev_stopped():
        pytest.skip("terminal evidence exists; result claims are governed by the RET2F interpretation tests")
    assert not (REPO / RESULTS).exists(), f"{PREMATURE_RESULT}: {RESULTS} exists without terminal evidence"
    system = json.loads((REPO / "ELPIS_SYSTEM.json").read_text(encoding="utf-8"))
    component = next(c for c in system["research"]["components"] if c["path"] == LAB)
    ecsg = next(s for s in system["subsystems"] if s["id"] == "ECS")
    interface = next(i for i in ecsg["incomplete_interfaces"] if "Retention R2" in i)
    cognition = (REPO / "docs" / "COGNITION_R0.md").read_text(encoding="utf-8")
    pointers = {"component": component["classification"],
                "ECS interface": interface.split("Retention R2", 1)[1],
                "COGNITION_R0": cognition.split("Retention R2", 1)[1].split("\n\n", 1)[0]}
    assert component["classification"].startswith("PREREGISTERED")
    for where, text in pointers.items():
        assert "preregistered" in text.lower(), where
        found = [token for token in RESULT_TOKENS if token.lower() in text.lower()]
        assert not found, f"{PREMATURE_RESULT}: {where} claims {found} before terminal evidence exists"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "`PREREGISTERED`" in readme and not re.search(r"(?m)^## Result", readme)


def test_no_canonical_retention_mechanism_is_promoted():
    vocabulary = re.compile(r"(?i)consolidat|retention|laplace|fibre|recondition|rehears")
    sources = sorted((REPO / "src" / "elpis" / "ECS").rglob("*.py")) + sorted(
        p for p in (REPO / "native" / "ECS").rglob("*") if p.suffix in (".c", ".h"))
    exempt = admitted(REPO)          # tests/research/_k1_promotion.py: the K1 milestone, only after R3 OUTCOME_A
    offenders = [str(p.relative_to(REPO)) for p in sources if vocabulary.search(p.read_text(encoding="utf-8"))
                 and str(p.relative_to(REPO)) not in exempt]
    assert not offenders, offenders
