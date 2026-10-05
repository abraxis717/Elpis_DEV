"""Retention R2 recorded QUAL verdict on the current runtime. Never skipped for a host or profile difference.

Re-runs every QUAL world's registered computation (all mechanisms, removed engines, ablations, causality,
determinism and the clean-process transplant probe) from the frozen choices, in a fresh interpreter whose
BLAS/OpenMP threads are fixed before NumPy loads, and requires the recorded pre-robustness verdict exactly:
validity, mechanics, gates A-L, disposition, outcome and every per-mechanism count. No float and no W digest is
compared (bitwise reproduction is the historical replay in test_evidence.py). Gate M itself needs forced kernels
and is part of the record, not of this regression.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from research.ecs_retention_r2 import protocol as P
from research.ecs_retention_r2 import run as R

from ...ECS_G.test_math_r0 import REPO, _library_path

QUAL = P.load(R.QUAL_PATH, "qual")["body"]

_RUN = """
import json, sys
import research.ecs_retention_r2
from research.ecs_retention_r2 import numerics, protocol as P, run as R
library = R.Path(sys.argv[1])
qual = P.load(R.QUAL_PATH, "qual")["body"]
_, _, _, verdict = R._verdict(library, qual["choices"])
sys.stdout.write(json.dumps({"comparable": P.plain(R._comparable(verdict)), "threads": numerics.blas_threads(),
                             "core": numerics.blas_core()}))
"""


def test_current_runtime_reaches_the_recorded_qual_verdict():
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", _RUN, str(_library_path())], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True)
    current = json.loads(out.stdout)
    assert current["threads"] == 1, current["threads"]
    recorded = QUAL["pre_robustness_verdict"]
    for key in ("validity", "mechanics", "gates", "disposition", "outcome"):
        assert current["comparable"][key] == recorded[key], (key, current["core"])
    assert current["comparable"]["counts"] == recorded["counts"], current["core"]
