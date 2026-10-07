"""Retention R2 evidence guards: write-once bytes, binding, recorded verdicts recomputed from recorded rows,
chronology of DEV -> freeze -> QUAL, and bitwise historical replay only under the recorded binding.

The never-skipped current-runtime regression of QUAL is test_runtime_regression.py (added with QUAL evidence).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

from research.ecs_retention_r2 import experiment as X
from research.ecs_retention_r2 import protocol as P
from research.ecs_retention_r2 import run as R

from ...ECS.test_math_r0 import REPO, _library_path

# SHA-256 of each write-once record as written.
EVIDENCE_SHA256 = {
    "evidence/qual/ecsg-retention-r2.v1.qual.json": "e1be3bf6a572a06a0d99994fc6dff758883d286c8da125aa713d95525cb56869",
    "frozen/ecsg-retention-r2.v1.frozen.json": "03fc60974ef7bbfae20e3d15a214796214c29171ddd3cca47037e448840824da",
    "evidence/dev/ecsg-retention-r2.v1.dev.json": "beb9c18abd15bd06bd81d30ee9d58ee2e4b1b14df9029c577fadbe015647a221",
}
DEV = P.load(R.DEV_PATH, "dev")
RET2B_COMMIT = "5145f286cc5a1a57ec4ed86b63a37db2daf65aaa"


@pytest.mark.parametrize("name", sorted(EVIDENCE_SHA256))
def test_write_once_evidence_bytes_are_unchanged(name):
    data = (R.ROOT / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == EVIDENCE_SHA256[name], f"{name} changed after it was written"


def test_dev_record_is_bound_to_the_clean_ret2b_laboratory():
    body = DEV["body"]
    impl = body["implementation"]
    assert impl["base_commit"] == RET2B_COMMIT and not impl["dirty"]
    assert impl["lab_source_digest"] == P.source_digest_at(RET2B_COMMIT)
    assert all(body[k] == v for k, v in P.spec_digests(P.load_spec()).items())
    prof = impl["numerical_profile"]
    assert prof["blas"]["threads"] == 1 and prof["blas"]["core"] != "unknown"
    for key in ("python", "numpy", "numpy_build", "cpu", "machine", "os", "libc", "thread_environment"):
        assert prof[key], key


def test_dev_verdicts_and_selection_follow_from_the_recorded_rows():
    body = DEV["body"]
    spec = P.load_spec()
    assert [r["hidden_gain"] for r in body["task_rule"]] == spec["task"]["hidden_gain_grid"][:len(body["task_rule"])]
    for rule in body["task_rule"]:
        assert X.task_validity_dev(rule["worlds"], spec["thresholds"]) == rule["validity"]
        for row in rule["worlds"].values():
            assert set(row) == {"M0", "M1", "ceilings", "witness", "W_A_refused"}   # controls only
    assert body["disposition"] == "VALID_TASK_ON_DEV" and body["task_rule"][-1]["validity"]["valid"]
    assert not any(r["validity"]["valid"] for r in body["task_rule"][:-1])
    assert body["choices"]["hidden_gain"] == body["task_rule"][-1]["hidden_gain"]
    selected, ranking = X.select(body["candidate_table"])
    assert selected == body["choices"]["selected"] and ranking == body["ranking"]
    assert all(set(row["P"]) == {"M0", "M1", "C1R", "K1", "K2", "K3"} for row in body["candidate_table"].values())


_REPLAY = """
import ctypes, json, sys
import research.ecs_retention_r2
from elpis.ECS.native import ECSGLibrary
from research.ecs_retention_r2 import experiment as X, protocol as P, run as R
from tests.research._renamed_sources import apply_renamed_lab_binding
apply_renamed_lab_binding(P)
library = sys.argv[1]
dev = P.load(R.DEV_PATH, "dev")["body"]
stale = P.binding_mismatch(dev["implementation"], P.implementation(library))
rows = None
if not stale:
    lab = X.Lab(ECSGLibrary(ctypes.CDLL(library)), P.load_spec())
    gain = dev["choices"]["hidden_gain"]
    rows = {w: P.plain(X.dev_controls_world(lab, w, gain)) for w in ("dev-0000", "dev-0005")}
sys.stdout.write(json.dumps({"stale": stale, "rows": rows}))
"""


def test_dev_controls_reproduce_exactly_under_the_recorded_binding():
    """HISTORICAL REPLAY (bitwise) of two DEV worlds' control rows; only under the full recorded binding."""
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", _REPLAY, str(_library_path())], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True)
    result = json.loads(out.stdout)
    if result["stale"]:
        pytest.skip(f"HISTORICAL_REPLAY_ENVIRONMENT_MISMATCH: bitwise DEV replay is demanded only under the "
                    f"recorded binding: {result['stale']}")
    for wid, row in result["rows"].items():
        assert row == DEV["body"]["task_rule"][-1]["worlds"][wid], wid


FROZEN = P.load(R.FROZEN_PATH, "frozen")


def test_frozen_authority_binds_dev_and_the_laboratory():
    body = FROZEN["body"]
    spec = P.load_spec()
    assert all(body[k] == v for k, v in P.spec_digests(spec).items())
    assert body["dev_evidence_digest"] == DEV["digest"] and body["choices"] == DEV["body"]["choices"]
    assert body["lab_commit"] == RET2B_COMMIT and body["lab_source_digest"] == P.source_digest_at(RET2B_COMMIT)
    assert body["lab_source_digest"] == P.source_digest()          # no laboratory change after the freeze
    assert P.binding_mismatch(DEV["body"]["implementation"], body["implementation"]) == []
    assert body["pass_rule_text"] == spec["pass_rule"] and body["qual_world_ids"] == list(P.world_ids(spec, "QUAL"))
    assert body["robustness_cores"] == ["Prescott", "Haswell"]


QUAL = P.load(R.QUAL_PATH, "qual")


def test_qual_is_bound_to_the_freeze_and_ran_on_disjoint_worlds():
    q, frozen = QUAL["body"], FROZEN["body"]
    assert q["frozen_digest"] == FROZEN["digest"] and q["choices"] == frozen["choices"]
    assert q["world_ids"] == frozen["qual_world_ids"] and q["world_count"] == 24 == len(q["per_world"])
    assert not set(q["world_ids"]) & set(DEV["body"]["world_ids"])
    assert not q["implementation"]["dirty"] and P.binding_mismatch(frozen["implementation"], q["implementation"]) == []
    assert q["implementation"]["numerical_profile"]["blas"]["threads"] == 1
    assert set(q["robustness"]) == {"Prescott", "Haswell"}
    assert all(v["core_in_effect"] == core and v["threads"] == 1 for core, v in q["robustness"].items())


def test_recorded_verdict_follows_from_the_recorded_rows_and_the_registered_rule():
    q = QUAL["body"]
    spec = P.load_spec()
    verdict = X.gates(q["per_world"], spec, q["choices"], q["determinism"], q["transplant_probe"])
    for key in ("validity", "mechanics", "gates", "disposition", "outcome", "counts"):
        assert P.plain(verdict[key]) == q["pre_robustness_verdict"][key], key
    robust_ok = all(v["identical"] for v in q["robustness"].values())
    assert robust_ok == all(v["comparable"] == q["pre_robustness_verdict"] for v in q["robustness"].values())
    assert q["gates"] == dict(q["pre_robustness_verdict"]["gates"], M_numerical_robustness=robust_ok)
    sel = q["choices"]["selected"]["candidate"]
    rows = lambda k: [q["per_world"][w]["P"]["mechanisms"][k] for w in q["world_ids"]]  # noqa: E731
    final = X.disposition(q["validity"], q["mechanics"], q["gates"], rows(sel), rows("M0"), rows("M1"),
                          spec["thresholds"])
    assert (final["disposition"], final["outcome"]) == (q["disposition"], q["outcome"])


def test_qual_disposition_is_recorded_as_found():
    q = QUAL["body"]
    assert q["disposition"] == "PARTIAL_REDUCTION" and q["outcome"] == "OUTCOME_C"
    assert all(q["mechanics"].values()) and all(q["validity"].values())
    failed = sorted(k for k, v in q["gates"].items() if not v)
    assert failed == ["G_state_causality", "L_native_feasibility", "M_numerical_robustness"]
    assert q["choices"] == {"hidden_gain": 2.0, "selected": {"candidate": "K3"}}


_QUAL_REPLAY = """
import ctypes, json, sys
import research.ecs_retention_r2
from elpis.ECS.native import ECSGLibrary
from research.ecs_retention_r2 import experiment as X, protocol as P, run as R
from tests.research._renamed_sources import apply_renamed_lab_binding
apply_renamed_lab_binding(P)
library, world = sys.argv[1], sys.argv[2]
q = P.load(R.QUAL_PATH, "qual")["body"]
stale = P.binding_mismatch(q["implementation"], P.implementation(library))
got = None
if not stale:
    lab = X.Lab(ECSGLibrary(ctypes.CDLL(library)), P.load_spec())
    got = P.plain(X.qual_world(lab, world, q["choices"]))
sys.stdout.write(json.dumps({"stale": stale, "world": got}))
"""


def _strip_timing(value):
    if isinstance(value, dict):
        return {k: _strip_timing(v) for k, v in value.items() if k != "learn_seconds"}
    if isinstance(value, list):
        return [_strip_timing(v) for v in value]
    return value


def test_a_qual_world_reproduces_exactly_under_the_recorded_binding():
    """HISTORICAL REPLAY (bitwise) of one QUAL world; only under the full recorded binding."""
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", _QUAL_REPLAY, str(_library_path()), "qual-0007"], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True)
    result = json.loads(out.stdout)
    if result["stale"]:
        pytest.skip(f"HISTORICAL_REPLAY_ENVIRONMENT_MISMATCH: bitwise QUAL replay is demanded only under the "
                    f"recorded binding: {result['stale']}")
    assert _strip_timing(result["world"]) == _strip_timing(QUAL["body"]["per_world"]["qual-0007"])


# --- RET2F: the interpretation and the authority pointers state the record and claim no more ------------------------

RESULTS = REPO / "docs" / "research" / "ECS_RETENTION_R2_RESULTS.md"
OVERCLAIMS = ("retention is solved", "retention_supported", "outcome_a", "elpis now remembers",
              "continual learning is achieved", "understands language", "promoted to canonical")
FAILED_GATES = ("G state causality", "L native feasibility", "M numerical robustness")


def test_results_report_the_recorded_disposition_and_claim_no_more():
    text = RESULTS.read_text(encoding="utf-8")
    body = QUAL["body"]
    for needle in ("PARTIAL_REDUCTION", "OUTCOME_C", "NO_CANONICAL_PROMOTION", "No candidate is qualified",
                   "SEMANTICS=NONE", "new experiment version", "K3 (selected)", "post hoc",
                   DEV["digest"], P.load(R.FROZEN_PATH, "frozen")["digest"], QUAL["digest"],
                   body["implementation"]["library"]["sha256"], RET2B_COMMIT[:7], body["implementation"]["base_commit"][:7],
                   *FAILED_GATES):
        assert needle in text, needle
    lowered = text.lower()
    for overclaim in OVERCLAIMS:
        assert overclaim not in lowered, overclaim
    counts = body["counts"]
    for name in ("M0", "M1", "C1R", "K1", "K2", "K3"):
        assert f"| {counts[name]['SEQUENCE_HELD']} |" in text, name
    assert f"in **{counts['ablation:consolidation_reset_at_B']['both_retained_at_D']}/24**" in text


def test_authority_pointers_state_the_recorded_disposition():
    system = json.loads((REPO / "ELPIS_SYSTEM.json").read_text(encoding="utf-8"))
    component = next(c for c in system["research"]["components"] if c["path"] == "research/ecs_retention_r2")
    ecsg = next(s for s in system["subsystems"] if s["id"] == "ECS")
    interface = next(i for i in ecsg["incomplete_interfaces"] if "Retention R2" in i).split("Retention R2", 1)[1]
    cognition = (REPO / "docs" / "COGNITION_R0.md").read_text(encoding="utf-8")
    cognition = cognition.split("Retention R2", 1)[1].split("\n\n", 1)[0]
    readme = (R.ROOT / "README.md").read_text(encoding="utf-8")
    for where, text in (("component", component["classification"]), ("ECS interface", interface),
                        ("COGNITION_R0", cognition), ("README", readme)):
        assert "PARTIAL_REDUCTION" in text, where
        assert "OUTCOME_A" not in text and "RETENTION_SUPPORTED" not in text, where
        assert "qualified or canonical" in text or "NO_CANONICAL_PROMOTION" in text, where
    assert component["classification"].startswith("QUALIFICATION, NOT QUALIFIED")
