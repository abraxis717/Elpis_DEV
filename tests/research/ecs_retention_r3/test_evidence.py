"""Retention R3 evidence guards: write-once bytes, binding, recorded verdicts recomputed from recorded rows by the
registered rules, chronology of DEV -> freeze -> QUAL, and bitwise historical replay only under the recorded binding.

The never-skipped current-runtime regression of QUAL is test_runtime_regression.py (added with the QUAL evidence).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

from research.ecs_retention_r3 import experiment as X
from research.ecs_retention_r3 import protocol as P
from research.ecs_retention_r3 import run as R

from ...ECS.test_math_r0 import REPO, _library_path

# SHA-256 of each write-once record as written.
EVIDENCE_SHA256 = {
    "evidence/qual/ecsg-retention-r3.v1.qual.json": "83ce2d61f30bb2927e1288995de5eb00ce41ff48767d54faedde88c8266c20e6",
    "frozen/ecsg-retention-r3.v1.frozen.json": "7be80fee34142a6b226eaf57d9fec2aba21c397d40faa65eb96bde9d2d4b39e2",
    "evidence/dev/ecsg-retention-r3.v1.dev.json": "20a7ff116348da6e1bf8e0ab00a498dcc73bef9fa007ffacb935c94a1d3ed964",
}
DEV = P.load(R.DEV_PATH, "dev")
RET3B_COMMIT = "35408879720a6d697ce07dcb6c19fedbd37845cd"
FROZEN = P.load(R.FROZEN_PATH, "frozen")
QUAL = P.load(R.QUAL_PATH, "qual")


@pytest.mark.parametrize("name", sorted(EVIDENCE_SHA256))
def test_write_once_evidence_bytes_are_unchanged(name):
    data = (R.ROOT / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == EVIDENCE_SHA256[name], f"{name} changed after it was written"


def test_dev_record_is_bound_to_the_clean_ret3b_laboratory():
    body = DEV["body"]
    impl = body["implementation"]
    assert impl["base_commit"] == RET3B_COMMIT and not impl["dirty"]
    assert impl["lab_source_digest"] == P.source_digest_at(RET3B_COMMIT)
    assert impl["numerical_profile"]["blas"]["threads"] == 1
    assert body["world_ids"] == list(P.world_ids(R.SPEC, "DEV")) and body["hidden_gain"] == 2.0
    for key in ("spec", "pass_rule", "candidates_sha256"):
        assert body[key] == P.spec_digests(R.SPEC)[key], key


def test_dev_verdicts_follow_from_the_recorded_rows_and_the_registered_rules():
    body = DEV["body"]
    th = R.SPEC["thresholds"]
    validity = X.task_validity_dev(body["task_check"]["worlds"], th)
    assert validity == body["task_check"]["validity"] and validity["valid"]
    rule = X.k1_dev_rule(body["k1_worlds"], th, R.SPEC)
    assert rule == body["k1_rule"] and rule["proceeds"]
    assert body["disposition"] == "K1_PROCEEDS"
    assert set(body["k1_worlds"]) == set(body["world_ids"])
    for w, row in body["k1_worlds"].items():
        assert set(row["causality"]["reset_challenge"]["stages"]) == {"C"}, w


_DEV_REPLAY = """
import ctypes, json, sys
import research.ecs_retention_r3
from elpis.ECS.native import ECSGLibrary
from research.ecs_retention_r3 import experiment as X, protocol as P, run as R
from tests.research._renamed_sources import apply_renamed_lab_binding
apply_renamed_lab_binding(P)
library, world = sys.argv[1], sys.argv[2]
dev = P.load(R.DEV_PATH, "dev")["body"]
stale = P.binding_mismatch(dev["implementation"], P.implementation(library))
got = None
if not stale:
    lab = X.Lab(ECSGLibrary(ctypes.CDLL(library)), R.SPEC)
    got = {"controls": P.plain(X.dev_controls_world(lab, world)), "k1": P.plain(X.dev_k1_world(lab, world))}
sys.stdout.write(json.dumps({"stale": stale, "world": got}))
"""


def _strip_timing(value):
    if isinstance(value, dict):
        return {k: _strip_timing(v) for k, v in value.items() if k != "learn_seconds"}
    if isinstance(value, list):
        return [_strip_timing(v) for v in value]
    return value


def _replay(script: str, world: str) -> dict:
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", script, str(_library_path()), world], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_a_dev_world_reproduces_exactly_under_the_recorded_binding():
    """HISTORICAL REPLAY (bitwise) of one DEV world; only under the full recorded binding."""
    result = _replay(_DEV_REPLAY, "r3dev-0003")
    if result["stale"]:
        pytest.skip(f"HISTORICAL_REPLAY_ENVIRONMENT_MISMATCH: bitwise DEV replay is demanded only under the "
                    f"recorded binding: {result['stale']}")
    body = DEV["body"]
    assert _strip_timing(result["world"]["controls"]) == _strip_timing(body["task_check"]["worlds"]["r3dev-0003"])
    assert _strip_timing(result["world"]["k1"]) == _strip_timing(body["k1_worlds"]["r3dev-0003"])


def test_frozen_authority_binds_dev_the_laboratory_and_the_qual_worlds():
    f = FROZEN["body"]
    assert f["dev_evidence_digest"] == DEV["digest"] and f["lab_commit"] == RET3B_COMMIT
    assert f["lab_source_digest"] == DEV["body"]["implementation"]["lab_source_digest"]
    assert P.binding_mismatch(DEV["body"]["implementation"], f["implementation"]) == []
    assert not f["implementation"]["dirty"]
    for key in ("spec", "pass_rule", "candidates_sha256"):
        assert f[key] == P.spec_digests(R.SPEC)[key], key
    assert f["qual_world_ids"] == list(P.world_ids(R.SPEC, "QUAL")) and len(f["qual_world_ids"]) == 32
    assert not set(f["qual_world_ids"]) & set(f["dev_world_ids"])
    assert f["task_parameters"] == R.SPEC["task"] and f["thresholds"] == R.SPEC["thresholds"]
    assert f["regime"] == R.SPEC["regime"] and f["pass_rule_text"] == R.SPEC["pass_rule"]
    assert f["k1_state_format"] == "elpis.research.ecs-retention-r3.state.v1"
    assert f["robustness_cores"] == ["Prescott", "Haswell"]


def test_qual_is_bound_to_the_freeze_and_ran_on_disjoint_worlds():
    q = QUAL["body"]
    assert q["frozen_digest"] == FROZEN["digest"]
    assert P.binding_mismatch(FROZEN["body"]["implementation"], q["implementation"]) == []
    assert not q["implementation"]["dirty"] and q["implementation"]["numerical_profile"]["blas"]["threads"] == 1
    assert q["world_ids"] == FROZEN["body"]["qual_world_ids"] and q["world_count"] == 32
    assert not set(q["world_ids"]) & set(DEV["body"]["world_ids"])
    for key in ("spec", "pass_rule", "candidates_sha256"):
        assert q[key] == FROZEN["body"][key], key


def test_recorded_verdict_follows_from_the_recorded_rows_and_the_registered_rule():
    q = QUAL["body"]
    verdict = X.gates(q["per_world"], R.SPEC, q["determinism"], q["transplant_probe"])
    assert P.plain(verdict["decision_record"]) == q["pre_robustness_decision_record"]
    for key in ("validity", "mechanics", "medians", "counts", "native_feasibility_evidence"):
        assert P.plain(verdict[key]) == q[key], key
    robust_ok = all(r["identical"] for r in q["robustness"].values())
    assert q["gates"] == dict(verdict["gates"], L_numerical_robustness=robust_ok)
    worlds = q["world_ids"]
    rows = lambda k: [q["per_world"][w]["P"]["mechanisms"][k] for w in worlds]  # noqa: E731
    final = X.disposition(q["validity"], q["mechanics"], q["gates"], rows("K1"), rows("M0"), rows("M1"),
                          R.SPEC["thresholds"])
    assert (final["disposition"], final["outcome"]) == (q["disposition"], q["outcome"])
    assert P.plain(X.qualifiers(q["per_world"], R.SPEC["thresholds"], robust_ok)) == q["qualifiers"]


def test_gate_l_compared_the_decision_record_under_the_registered_kernels():
    q = QUAL["body"]
    assert set(q["robustness"]) == {"Prescott", "Haswell"}
    for core, child in q["robustness"].items():
        assert child["core_in_effect"] == core and child["threads"] == 1
        assert child["identical"] and child["decision_record"] == q["pre_robustness_decision_record"], core


def test_qual_disposition_is_recorded_as_found():
    q = QUAL["body"]
    assert (q["disposition"], q["outcome"]) == ("RETENTION_SUPPORTED_UNDER_FROZEN_SYNTHETIC_REGIME", "OUTCOME_A")
    assert all(q["gates"].values()) and len(q["gates"]) == 12
    assert all(q["validity"].values()) and all(q["mechanics"].values())
    c = q["counts"]
    assert (c["K1"]["SEQUENCE_HELD"], c["M0"]["SEQUENCE_HELD"], c["M1"]["SEQUENCE_HELD"]) == (32, 0, 32)
    assert c["state_removed"]["SEQUENCE_HELD"] == 0 and c["reset_challenge"]["RESET_DEGRADED"] == 31
    assert c["full_state_transplant_at_B"]["bitwise"] == 32 and c["w_only_negative_control_at_B"]["pass"] == 32
    assert q["qualifiers"]["NUMERICALLY_FRAGILE"] is False
    for w in q["world_ids"]:
        assert set(q["per_world"][w]["P"]["causality"]["reset_challenge"]["stages"]) == {"C"}, w


_QUAL_REPLAY = """
import ctypes, json, sys
import research.ecs_retention_r3
from elpis.ECS.native import ECSGLibrary
from research.ecs_retention_r3 import experiment as X, protocol as P, run as R
from tests.research._renamed_sources import apply_renamed_lab_binding
apply_renamed_lab_binding(P)
library, world = sys.argv[1], sys.argv[2]
q = P.load(R.QUAL_PATH, "qual")["body"]
stale = P.binding_mismatch(q["implementation"], P.implementation(library))
got = None
if not stale:
    got = P.plain(X.qual_world(X.Lab(ECSGLibrary(ctypes.CDLL(library)), R.SPEC), world))
sys.stdout.write(json.dumps({"stale": stale, "world": got}))
"""


def test_a_qual_world_reproduces_exactly_under_the_recorded_binding():
    """HISTORICAL REPLAY (bitwise) of one QUAL world; only under the full recorded binding."""
    result = _replay(_QUAL_REPLAY, "r3qual-0011")
    if result["stale"]:
        pytest.skip(f"HISTORICAL_REPLAY_ENVIRONMENT_MISMATCH: bitwise QUAL replay is demanded only under the "
                    f"recorded binding: {result['stale']}")
    assert _strip_timing(result["world"]) == _strip_timing(QUAL["body"]["per_world"]["r3qual-0011"])


# --- RET3F: the interpretation and the authority pointers state the record and claim no more ------------------------

RESULTS = REPO / "docs" / "research" / "ECS_RETENTION_R3_RESULTS.md"
OVERCLAIMS = ("retention is solved", "elpis now remembers", "continual learning is achieved", "understands language",
              "k1 is canonical", "is now canonical", "general memory")


def test_results_report_the_recorded_disposition_and_claim_no_more():
    text = " ".join(RESULTS.read_text(encoding="utf-8").split())
    q = QUAL["body"]
    for needle in ("RETENTION_SUPPORTED_UNDER_FROZEN_SYNTHETIC_REGIME", "OUTCOME_A", DEV["digest"], FROZEN["digest"],
                   QUAL["digest"], q["implementation"]["library"]["sha256"], RET3B_COMMIT[:7], "SEMANTICS=NONE",
                   "K1 is qualified for the native milestone", "changes no canonical code by itself",
                   "no retention mechanism is canonical", "RESET_DEGRADED 31/32", "does not make K1 canonical",
                   "PARTIAL_REDUCTION"):
        assert needle in text, needle
    lowered = text.lower()
    for overclaim in OVERCLAIMS:
        assert overclaim not in lowered, overclaim
    c = q["counts"]
    for name in ("M0", "M1", "C1R", "K1"):
        assert f"| {c[name]['SEQUENCE_HELD']} |" in text, name


def test_authority_pointers_state_the_recorded_disposition_and_no_canonical_promotion_yet():
    system = json.loads((REPO / "ELPIS_SYSTEM.json").read_text(encoding="utf-8"))
    component = next(c for c in system["research"]["components"] if c["path"] == "research/ecs_retention_r3")
    ecsg = next(s for s in system["subsystems"] if s["id"] == "ECS")
    interface = next(i for i in ecsg["incomplete_interfaces"] if i.startswith("Retention R3"))
    cognition = (REPO / "docs" / "COGNITION_R0.md").read_text(encoding="utf-8")
    cognition = cognition.split("**Retention R3**", 1)[1].split("\n\n", 1)[0]
    readme = (R.ROOT / "README.md").read_text(encoding="utf-8")
    for where, text in (("component", component["classification"]), ("ECS interface", interface),
                        ("COGNITION_R0", cognition), ("README", readme)):
        flat = " ".join(text.split())
        assert "OUTCOME_A" in flat, where
        assert "native" in flat and "milestone" in flat and ("not canonical" in flat or "no retention mechanism is canonical" in flat), where
    assert component["classification"].startswith("QUALIFIED (RESEARCH)")
