"""Retention R1 recorded DEV disposition on the current runtime. Never skipped for a host or profile difference.

DEV ended TASK_INVALID_ON_DEV under the registered task rule (controls only: M0, M1, O, WSTAR). This module
re-runs that rule on every DEV world at every recorded input scale, in a fresh interpreter whose BLAS/OpenMP
thread environment is fixed before NumPy loads, and requires the recorded verdicts and every boolean class and
count exactly. No float and no W digest is compared (those belong to the historical replay in test_evidence.py).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from research.ecs_retention_r1 import protocol as P
from research.ecs_retention_r1 import run as R

from ...ECS_G.test_math_r0 import REPO, _library_path

DEV = P.load(R.DEV_PATH, "dev")["body"]
VERDICT_KEYS = ("V1_witness", "V2_forgetting", "V3_novelty", "V4_learnability", "valid", "M0_sequence_held",
                "M1_sequence_held")

_RUN = """
import ctypes, json, sys
import research.ecs_retention_r1
from elpis.ECS_G.native import ECSGLibrary
from research.ecs_retention_r1 import experiment as X, numerics, protocol as P
library = sys.argv[1]
spec = P.load_spec()
lab = X.Lab(ECSGLibrary(ctypes.CDLL(library)), spec)
worlds = P.world_ids(spec, "DEV")
out = []
for scale in spec["arms"]["S"]["input_scale_grid"]:
    rows = {w: P.plain(X.dev_controls_world(lab, w, scale)) for w in worlds}
    verdict = X.task_validity_dev(rows, spec["thresholds"])
    out.append({"input_scale": scale, "validity": verdict, "worlds": rows})
    if verdict["valid"]:
        break
sys.stdout.write(json.dumps({"task_rule": out, "threads": numerics.blas_threads(), "core": numerics.blas_core()}))
"""


def _classes(rows: dict) -> dict:
    keep = ("SEQUENCE_HELD", "LEARNED", "HELD", "RETAINED", "CATASTROPHIC")
    return {w: {m: {k: row[m][k] for k in keep} for m in ("M0", "M1")} for w, row in rows.items()}


def test_current_runtime_reaches_the_recorded_dev_disposition():
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", _RUN, str(_library_path())], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True)
    current = json.loads(out.stdout)
    assert current["threads"] == 1, current["threads"]
    recorded = DEV["task_rule"]
    assert [r["input_scale"] for r in current["task_rule"]] == [r["input_scale"] for r in recorded]
    assert not any(r["validity"]["valid"] for r in current["task_rule"])     # still TASK_INVALID_ON_DEV
    for now, then in zip(current["task_rule"], recorded):
        assert {k: now["validity"][k] for k in VERDICT_KEYS} == {k: then["validity"][k] for k in VERDICT_KEYS}, \
            (now["input_scale"], current["core"])
        assert _classes(now["worlds"]) == _classes(then["worlds"]), (now["input_scale"], current["core"])
