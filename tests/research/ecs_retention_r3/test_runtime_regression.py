"""Retention R3 recorded QUAL result on the current runtime. Never skipped for a host or profile difference.

Re-runs every QUAL world's registered computation (K1, controls, C1R, removed engines, the causal branches, the
determinism probe and the clean-process transplant probe) in a fresh interpreter whose BLAS/OpenMP threads are
fixed before NumPy loads, on whatever kernel this host's BLAS selects naturally (nothing is forced). The recorded
decision record (spec pass_rule.decision_record: every decision-bearing R3 quantity) must be reproduced exactly.
Gate L held at QUAL, so no row is exempt (the RR2-CR0 principle has nothing to exempt). Descriptive quantities
(C1R, floats, digests, medians, timings) are not compared; bitwise reproduction is the historical replay in
test_evidence.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from research.ecs_retention_r3 import protocol as P
from research.ecs_retention_r3 import run as R

from ...ECS_G.test_math_r0 import REPO, _library_path

QUAL = P.load(R.QUAL_PATH, "qual")["body"]

_RUN = """
import json, sys
import research.ecs_retention_r3
from research.ecs_retention_r3 import numerics, protocol as P, run as R
_, _, _, verdict = R._verdict(R.Path(sys.argv[1]))
sys.stdout.write(json.dumps({"decision_record": P.plain(verdict["decision_record"]),
                             "threads": numerics.blas_threads(), "core": numerics.blas_core()}))
"""


def test_current_runtime_reaches_the_recorded_decision_record():
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", _RUN, str(_library_path())], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True)
    current = json.loads(out.stdout)
    assert current["threads"] == 1, current["threads"]
    recorded = QUAL["pre_robustness_decision_record"]
    assert all(r["identical"] for r in QUAL["robustness"].values())     # gate L held: nothing is exempt
    for key in ("validity", "mechanics", "gates", "disposition", "outcome", "controls"):
        assert current["decision_record"][key] == recorded[key], (key, current["core"])
    for key in ("K1", "causality"):
        diff = sorted(w for w in recorded[key] if current["decision_record"][key].get(w) != recorded[key][w])
        assert not diff, (key, diff, current["core"])
