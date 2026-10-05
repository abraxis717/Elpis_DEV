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

from ...ECS_G.test_math_r0 import REPO, _library_path

# SHA-256 of each write-once record as written.
EVIDENCE_SHA256 = {
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
from elpis.ECS_G.native import ECSGLibrary
from research.ecs_retention_r2 import experiment as X, protocol as P, run as R
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
