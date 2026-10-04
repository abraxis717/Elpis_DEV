"""Retention R1 preregistration guard (RET1A). No laboratory, no worlds, no numerics.

The preregistration exists, is complete and bounded, and is not weaker than Retention R0. It was written
before any laboratory code or result, and no result can be claimed before its chronological step:

* the specification and CANDIDATES.md are write-once (pinned SHA-256; a change is a new version), and only
  RET1A commits ever touched them;
* RET1A commits contain no implementation (no Python under research/, nothing under src/ or native/) and
  touch no Retention R0 or Cognitive R0 evidence;
* a later record (DEV, frozen, QUAL, results) may exist only if exactly one commit of its own step added it,
  after a commit of the step before it;
* until QUAL evidence exists, no authority pointer claims a result, and no canonical ECS_G code names a
  retention mechanism.

The history checks need full git history (CI checks out fetch-depth 0). A shallow or missing history fails
by name, never with a bare lookup error.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "research" / "ecs_retention_r1"
LAB = "research/ecs_retention_r1"
SPEC_PATH = ROOT / "specs" / "ecsg-retention-r1.v1.spec.json"
CANDIDATES_PATH = ROOT / "CANDIDATES.md"
R0_SPEC_PATH = REPO / "research" / "ecs_retention_r0" / "specs" / "ecsg-retention-r0.v1.spec.json"
RESULTS = "docs/research/ECS_RETENTION_R1_RESULTS.md"
BASE_MAIN = "8a1a9c72866e5c7eab12a8c02e55fa9fc45dae8b"

# Write-once preregistration (RET1A). A changed experiment is a new version, never an edit of these bytes.
PINNED_SHA256 = {
    "specs/ecsg-retention-r1.v1.spec.json": "79a863ce70eb85ed713a45e1afe7cd7a58f9c467da85dacbce499ca9b27a09c5",
    "CANDIDATES.md": "8c29e4bab21fad69add1c6db2236d4ecbf7ef81459dcc73292aa85989eea88f9",
}
STEPS = ("RET1A", "RET1B", "RET1C", "RET1D", "RET1E", "RET1F")
# Each result-bearing record: where it lives, the step that must add it, the step that must precede it.
RECORD_LAW = (
    (f"{LAB}/evidence/dev", "RET1C", "RET1B"),
    (f"{LAB}/frozen", "RET1D", "RET1C"),
    (f"{LAB}/evidence/qual", "RET1E", "RET1D"),
    (RESULTS, "RET1F", "RET1E"),
)
RESULT_TOKENS = ("OUTCOME_A", "OUTCOME_B", "OUTCOME_C", "OUTCOME_D", "OUTCOME_V", "RETENTION_SUPPORTED",
                 "PARTIAL_REDUCTION", "REPLAY_ONLY", "NO_MATERIAL_IMPROVEMENT", "TASK_INVALID", "selected candidate is",
                 "winner")
HISTORY_INCOMPLETE = "RETENTION_R1_AUTHORITY_HISTORY_INCOMPLETE"
CHRONOLOGY_VIOLATED = "RETENTION_R1_CHRONOLOGY_VIOLATED"
PREMATURE_RESULT = "RETENTION_R1_PREMATURE_RESULT"

SPEC = json.loads(SPEC_PATH.read_text(encoding="ascii"))
R0_SPEC = json.loads(R0_SPEC_PATH.read_text(encoding="ascii"))
CANDIDATES = CANDIDATES_PATH.read_text(encoding="utf-8")
SELECTABLE = ("K1", "K2", "K3")


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def _tag(subject: str) -> str:
    return subject.split(" ", 1)[0]


def _is_step(tag: str, step: str) -> bool:
    return tag == step or tag.startswith(step + ".")


def _history(path: str) -> list[tuple[str, str]]:
    """(sha, tag) of the non-merge commits touching ``path``, newest first; fails by name without history."""
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


def _qual_exists() -> bool:
    qual = REPO / LAB / "evidence" / "qual"
    return qual.is_dir() and any(qual.glob("*.json"))


# --- the preregistration itself --------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PINNED_SHA256))
def test_preregistration_is_write_once(name):
    data = (ROOT / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == PINNED_SHA256[name], (
        f"{name} changed after RET1A; a changed experiment is a new version (v2), never an edit")
    data.decode("ascii" if name.endswith(".json") else "utf-8")


def test_specification_is_complete():
    required = {"name", "version", "authored_at", "labels", "question", "base", "starting_point", "analysis",
                "chronology", "regime", "arms", "seed", "splits", "state_classes", "shared_procedure", "mechanisms",
                "state_accounting", "runtime_feasibility", "fms_implications", "excluded_families", "classes",
                "thresholds", "metrics", "dev_rules", "validity", "pass_rule", "mechanics_checks", "ablations",
                "secondary", "predictions", "numerical_binding", "evidence_contract", "promotion_eligibility",
                "nonclaims"}
    assert required <= set(SPEC), sorted(required - set(SPEC))
    assert SPEC["name"] == "ecsg-retention-r1" and SPEC["version"] == 1 and SPEC["authored_at"] == "RET1A"
    assert SPEC["base"]["main"] == BASE_MAIN
    for label in ("RESEARCH_ONLY", "NO_RUNTIME_AUTHORITY", "NO_LANGUAGE_CLAIM", "SEMANTICS=NONE", "PREREGISTERED"):
        assert label in SPEC["labels"], label
    assert tuple(SPEC["chronology"]["order"]) == STEPS
    assert set(SPEC["pass_rule"]) == {"A_first_acquisition", "B_stage_acquisition", "C_sequence_retention",
                                      "D_no_catastrophe", "E_joint_quality", "F_joint_state", "G_state_causality",
                                      "H_no_external_answer_store", "I_determinism", "J_control_honesty",
                                      "K_capacity_accounting", "L_native_feasibility", "M_numerical_robustness",
                                      "disposition", "qualifiers"}
    for outcome in ("MECHANICS_FAIL", "TASK_INVALID_UNDER_QUAL", "OUTCOME_A", "OUTCOME_B", "OUTCOME_C", "OUTCOME_D"):
        assert outcome in SPEC["pass_rule"]["disposition"], outcome
    # The primary requirement is the four-experience sequence, decided at RET1A.
    assert SPEC["regime"]["order"] == ["A", "B", "C", "D"]
    assert "every stage s" in SPEC["classes"]["SEQUENCE_HELD"]


def test_arms_separate_capacity_from_stability_plasticity():
    s, r = SPEC["arms"]["S"], SPEC["arms"]["R"]
    assert s["role"] == "PRIMARY_GATING" and r["role"] == "SECONDARY_DESCRIPTIVE"
    coords = {t: set(c) for t, c in s["active_coordinates"].items()}
    assert set(coords) == {"A", "B", "C", "D"} and all(len(c) == 3 for c in coords.values())
    tasks = sorted(coords)
    for i, a in enumerate(tasks):
        for b in tasks[i + 1:]:
            assert len(coords[a] & coords[b]) == 1, (a, b)   # interference through exactly one shared row
        assert set(s["complement_coordinates"][a]) == set(range(SPEC["regime"]["dim"])) - coords[a]
    assert "W* = [T* | 0]" in s["witness"]
    # Arm R is the R0 v1 family at R0's frozen offset.
    assert r["tasks"] == {t: R0_SPEC["tasks"][t] for t in "ABCD"} and r["offset"] == 1.0
    assert r["input_scale"] == R0_SPEC["regime"]["input_scale"]


def test_candidate_set_is_small_and_untuned():
    mechanisms = SPEC["mechanisms"]
    assert tuple(sorted(k for k, m in mechanisms.items() if m["eligible_for_selection"])) == SELECTABLE
    for name, cls in (("M0", "NEGATIVE_BASELINE"), ("M1", "EXTERNAL_MEMORY_CONTROL"), ("O", "EXPERIMENT_ONLY"),
                      ("WSTAR", "EXPERIMENT_ONLY"), ("C1R", "REFERENCE")):
        assert mechanisms[name]["class"] == cls and mechanisms[name]["eligible_for_selection"] is False, name
    # The reference is R0's selected configuration, unchanged.
    r0_choice = {"family": "C1", "lambda": 4.0}
    assert mechanisms["C1R"]["hyperparameters"] == {"lambda": r0_choice["lambda"]}
    assert 4.0 in R0_SPEC["mechanisms"]["C1"]["grid"]["lambda"]
    for name in SELECTABLE:
        m = mechanisms[name]
        assert m["class"] == "CANDIDATE"
        assert "grid" not in json.dumps(m), f"{name} carries a tuning grid"
        assert all(isinstance(v, (int, float)) for v in m["hyperparameters"].values()), name
        assert "not tuned" in m["hyperparameter_origin"], name
        assert f"### {name}:" in CANDIDATES, name
    assert mechanisms["K1"]["hyperparameters"]["lambda"] == 1.0
    # DEV's only free choices: the arm-S input scale (controls only) and one of three candidates.
    assert len(SPEC["arms"]["S"]["input_scale_grid"]) == 3
    assert "controls only" in SPEC["dev_rules"]["task"] and "no candidate" in SPEC["dev_rules"]["task"]
    assert "DEV chooses only the arm-S input scale and the selected candidate" in SPEC["dev_rules"]["nothing_else"]
    assert "never selected" in SPEC["dev_rules"]["candidate"]


def test_every_mechanism_declares_its_state_runtime_and_fms_form():
    regime = SPEC["regime"]
    d = regime["dim"]
    F = d + d * (d + 1) // 2 + d * (d + 1) * (d + 2) // 6
    assert F == regime["features"] == 83
    packed = 8 * F * (F + 1) // 2
    expected = {"K1": packed + 8 * F, "K2": packed, "K3": packed + 8 * F, "C1R": packed + 8 * F, "M0": 0, "M1": 0}
    for name, nbytes in expected.items():
        assert SPEC["mechanisms"][name]["state"]["extra_persistent_bytes"] == nbytes, name
        assert "W" in SPEC["mechanisms"][name]["state"]["cognitive"], name
    accounting, runtime, fms = (SPEC["state_accounting"], SPEC["runtime_feasibility"], SPEC["fms_implications"])
    for name in ("K1", "K2", "K3", "C1R", "M0", "M1"):
        assert set(accounting["required_fields_per_mechanism"]) <= set(accounting["per_mechanism"][name]), name
    for name in ("K1", "K2", "K3", "C1R"):
        assert set(runtime["required_fields"]) <= set(runtime["per_mechanism"][name]), name
        assert set(fms["required_fields"]) <= set(fms["per_mechanism"][name]), name
        assert runtime["per_mechanism"][name]["state_growth"] == "none", name
        assert SPEC["mechanisms"][name]["consolidation_inputs"] == ["W", "experienced_inputs"], name
        assert SPEC["mechanisms"][name]["state"]["external_memory"] == "none", name
        assert accounting["per_mechanism"][name]["query_time_dependencies"] == "W", name
    assert "growing" in SPEC["mechanisms"]["M1"]["state"]["external_memory"]
    for needle in ("## 10. State accounting", "## 11. Native-runtime feasibility", "## 12. FMS implications",
                   "PYTHON MAY NOT EXECUTE THE ECS HOT PATH"):
        assert needle in CANDIDATES, needle


def test_every_named_family_is_assessed_before_dev():
    excluded = " ".join(SPEC["excluded_families"])
    for family in ("selective", "slow_fast", "low_rank", "gradient_compatibility", "recruitable",
                   "sparse_entity_local", "topology", "capacity_expansion"):
        assert family in excluded, family
    assert all(reason.strip() for reason in SPEC["excluded_families"].values())
    assert "## 9. Families assessed and excluded before DEV" in CANDIDATES


def test_gates_are_not_weaker_than_retention_r0():
    r0_metrics, r0_rule = R0_SPEC["metrics"], R0_SPEC["pass_rule"]
    # The R0 values this comparison relies on, read from the R0 specification itself.
    assert "nmse_A_after_B <= 0.5 and nmse_A_after_B <= 0.5 * nmse_A_init" in r0_metrics["RETAINED_A"]
    assert "nmse_B_after <= 0.5 and nmse_B_after <= 0.5 * nmse_B_before" in r0_metrics["LEARNED_B"]
    assert "every QUAL world RETAINED_A" in r0_rule["C_retention"] and "every QUAL world LEARNED_B" in \
        r0_rule["B_acquisition"]
    assert "<= 0.25" in r0_rule["A_acquisition"]
    assert "at least 75%" in r0_rule["H_negative_baseline"] and ">= 10" in r0_rule["H_negative_baseline"]
    assert "at least 75%" in r0_rule["E_state_causality"] and "0.5 x" in r0_rule["E_state_causality"]
    t = SPEC["thresholds"]
    assert t["class_nmse_absolute"] <= 0.5 and t["class_nmse_relative"] <= 0.5
    assert "<= 0.5" in SPEC["classes"]["RETAINED_t_at_s"] and "0.5 * nmse_t(W0)" in SPEC["classes"]["RETAINED_t_at_s"]
    assert "0.5 * nmse_s(W_prev)" in SPEC["classes"]["LEARNED_s"]
    assert t["candidate_world_fraction"] == 1.0
    assert t["first_acquisition_median_nmse"] <= 0.25 and t["joint_quality_median_final_mean_nmse"] <= 0.25
    assert t["baseline_max_sequence_held_fraction"] <= 0.25 and t["baseline_min_median_A_end_ratio"] >= 10.0
    assert t["ablation_max_sequence_held_fraction"] <= 0.25 and t["ablation_min_median_factor"] >= 2.0
    assert SPEC["splits"]["qual_worlds"] >= R0_SPEC["splits"]["qual_worlds"]
    assert SPEC["splits"]["dev_worlds"] >= R0_SPEC["splits"]["dev_worlds"]
    assert SPEC["regime"]["steps_per_experience"] == R0_SPEC["regime"]["steps"]
    assert SPEC["regime"]["learning_rate"] == R0_SPEC["regime"]["learning_rate"]
    for gate, needle in (("B_stage_acquisition", "every QUAL world"), ("C_sequence_retention", "every QUAL world"),
                         ("D_no_catastrophe", "no CATASTROPHIC"), ("G_state_causality", "bitwise"),
                         ("M_numerical_robustness", "Prescott")):
        assert needle in SPEC["pass_rule"][gate], gate
    assert "TASK_INVALID" in SPEC["validity"]["role"]


def test_numerical_binding_and_evidence_contract_are_complete():
    fields = " ".join(SPEC["numerical_binding"]["required_fields"])
    for needle in ("python_version", "numpy_version", "numpy_build_configuration", "blas_implementation",
                   "effective_blas_kernel_core", "effective_blas_thread_count", "compiler_identity_and_version",
                   "cmake_build_type_and_c_flags", "native_library_path_and_sha256", "machine_architecture", "cpu_model",
                   "bound_source_sha256", "laboratory_source_digest", "pass_rule_digest"):
        assert needle in fields, needle
    contract = SPEC["evidence_contract"]
    assert set(contract) == {"1_evidence_integrity", "2_implementation_correctness", "3_historical_replay",
                             "4_current_runtime_regression"}
    assert "never a scientific gate" in contract["3_historical_replay"]
    assert "never skipped" in contract["4_current_runtime_regression"]
    assert "must be 1" in " ".join(SPEC["numerical_binding"]["rules"])


# --- chronology from git history -------------------------------------------------------------------------------


def test_preregistration_was_only_ever_touched_by_ret1a():
    for name in PINNED_SHA256:
        path = f"{LAB}/{name}"
        commits = _history(path)
        assert commits, f"{HISTORY_INCOMPLETE}: no visible commit touches {path}{_shallow_hint()}"
        assert all(_is_step(tag, "RET1A") for _, tag in commits), (
            f"{CHRONOLOGY_VIOLATED}: {path} touched by {[t for _, t in commits]}; only RET1A commits may")


def test_ret1a_commits_contain_no_implementation_or_results():
    ret1a = [sha for sha, tag in _history(LAB) if _is_step(tag, "RET1A")]
    assert ret1a, f"{HISTORY_INCOMPLETE}: no RET1A commit touches {LAB}{_shallow_hint()}"
    forbidden = (re.compile(r"^(src|native)/"), re.compile(r"^research/.*\.py$"),
                 re.compile(rf"^{LAB}/(evidence|frozen)/"), re.compile(r"^research/ecs_(retention_r0|cognition_r0)/"),
                 re.compile(rf"^{re.escape(RESULTS)}$"))
    for sha in ret1a:
        files = _git("show", "--no-renames", "--name-only", "--format=", sha).stdout.split()
        offenders = [f for f in files if any(p.search(f) for p in forbidden)]
        assert not offenders, f"{CHRONOLOGY_VIOLATED}: RET1A commit {sha[:12]} carries {offenders}"


def _records(location: str) -> list[str]:
    path = REPO / location
    if path.is_file():
        return [location]
    return sorted(f"{location}/{p.name}" for p in path.glob("*.json")) if path.is_dir() else []


def test_records_appear_only_after_their_chronological_step():
    """A DEV, frozen, QUAL or results record exists only if one commit of its step added it, after its predecessor."""
    lab_history = _history(LAB) + _history(RESULTS)
    order = {sha: i for i, (sha, _) in enumerate(_history("."))}   # newest first over the whole history
    ranks = [STEPS.index(next(s for s in STEPS if _is_step(tag, s)))
             for _, tag in sorted(lab_history, key=lambda c: order.get(c[0], -1))
             if any(_is_step(tag, s) for s in STEPS)]
    assert ranks == sorted(ranks, reverse=True), f"{CHRONOLOGY_VIOLATED}: RET1 steps out of order {ranks}"
    for location, step, predecessor in RECORD_LAW:
        for record in _records(location):
            adds = _git("log", "--no-merges", "--diff-filter=A", "--format=%H %s", "--", record).stdout.splitlines()
            assert adds, f"{HISTORY_INCOMPLETE}: no visible commit adds {record}{_shallow_hint()}"
            assert len(adds) == 1 and _is_step(_tag(adds[0][41:]), step), (
                f"{PREMATURE_RESULT}: {record} added by {[a[41:] for a in adds]}; expected one {step} commit")
            added_at = order[adds[0][:40]]
            before = [order[sha] for sha, tag in lab_history if _is_step(tag, predecessor)]
            assert before and min(before) > added_at, (
                f"{PREMATURE_RESULT}: {record} ({step}) has no {predecessor} commit before it")


# --- no result before its time, no promotion ------------------------------------------------------------------


def _after(text: str, marker: str) -> str:
    assert marker in text, marker
    return text.split(marker, 1)[1].split("\n\n", 1)[0]


def test_no_result_is_claimed_before_qual_evidence():
    if _qual_exists():
        pytest.skip("QUAL evidence exists; result claims are governed by the RET1F interpretation tests")
    assert not (REPO / RESULTS).exists(), f"{PREMATURE_RESULT}: {RESULTS} exists without QUAL evidence"
    assert not set(SPEC) & {"choices", "selected", "results", "disposition_recorded", "outcome"}
    system = json.loads((REPO / "ELPIS_SYSTEM.json").read_text(encoding="utf-8"))
    component = next(c for c in system["research"]["components"] if c["path"] == LAB)
    ecsg = next(s for s in system["subsystems"] if s["id"] == "ECS_G")
    interface = next(i for i in ecsg["incomplete_interfaces"] if "Retention R1" in i)
    # The authority pointers to R1: each says preregistered and names no outcome.
    pointers = {
        "ELPIS_SYSTEM.json research component": component["classification"],
        "ELPIS_SYSTEM.json ECS_G interface": interface.split("Retention R1", 1)[1],
        "docs/COGNITION_R0.md": _after((REPO / "docs" / "COGNITION_R0.md").read_text(encoding="utf-8"),
                                       "Retention R1"),
    }
    assert component["classification"].startswith("PREREGISTERED")
    for where, text in pointers.items():
        assert "preregistered" in text.lower(), where
        found = [token for token in RESULT_TOKENS if token.lower() in text.lower()]
        assert not found, f"{PREMATURE_RESULT}: {where} claims {found} before QUAL evidence exists"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "`PREREGISTERED`" in readme
    assert not re.search(r"(?m)^## Result", readme), f"{PREMATURE_RESULT}: {LAB}/README.md has a Result section"


def test_no_canonical_retention_mechanism_is_promoted():
    """Canonical ECS_G learning is unchanged: no consolidation, protection or reconditioning in canonical code."""
    vocabulary = re.compile(r"(?i)consolidat|retention|laplace|fibre|recondition|rehears")
    sources = sorted((REPO / "src" / "elpis" / "ECS_G").rglob("*.py")) + sorted(
        p for p in (REPO / "native" / "ECS_G").rglob("*") if p.suffix in (".c", ".h"))
    assert sources
    offenders = [str(p.relative_to(REPO)) for p in sources if vocabulary.search(p.read_text(encoding="utf-8"))]
    assert not offenders, offenders
    system = json.loads((REPO / "ELPIS_SYSTEM.json").read_text(encoding="utf-8"))
    ecsg = next(s for s in system["subsystems"] if s["id"] == "ECS_G")
    for key in ("purpose", "mutation_authority", "runtime_participation"):
        assert not vocabulary.search(ecsg[key]), key
