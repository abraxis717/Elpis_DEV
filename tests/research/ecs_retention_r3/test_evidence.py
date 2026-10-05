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

from ...ECS_G.test_math_r0 import REPO, _library_path

# SHA-256 of each write-once record as written.
EVIDENCE_SHA256 = {
    "frozen/ecsg-retention-r3.v1.frozen.json": "7be80fee34142a6b226eaf57d9fec2aba21c397d40faa65eb96bde9d2d4b39e2",
    "evidence/dev/ecsg-retention-r3.v1.dev.json": "20a7ff116348da6e1bf8e0ab00a498dcc73bef9fa007ffacb935c94a1d3ed964",
}
DEV = P.load(R.DEV_PATH, "dev")
RET3B_COMMIT = "35408879720a6d697ce07dcb6c19fedbd37845cd"
FROZEN = P.load(R.FROZEN_PATH, "frozen")


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
from elpis.ECS_G.native import ECSGLibrary
from research.ecs_retention_r3 import experiment as X, protocol as P, run as R
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
