"""Retention R2 recorded QUAL result on the current runtime. Never skipped for a host or profile difference.

Re-runs every QUAL world's registered computation (all mechanisms, removed engines, ablations, causality,
determinism and the clean-process transplant probe) from the frozen choices, in a fresh interpreter whose
BLAS/OpenMP threads are fixed before NumPy loads, on whatever kernel this host's BLAS selects naturally. The
comparison is the corrected contract of _runtime_contract.py (corrective RR2-CR0): validity, mechanics, gates
A-L, disposition, outcome, the selected candidate and every count row equal to the record, except the rows the
QUAL record itself showed to be kernel-sensitive (gate M FAILED, NUMERICALLY_FRAGILE), whose deviations are
reported as that recorded finding. No float and no W digest is compared (bitwise reproduction is the historical
replay in test_evidence.py). Gate M itself needs forced kernels and is part of the record, not of this regression.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings

from . import _runtime_contract as C
from ...ECS_G.test_math_r0 import REPO, _library_path

_RUN = """
import json, sys
import research.ecs_retention_r2
from research.ecs_retention_r2 import numerics, protocol as P, run as R
library = R.Path(sys.argv[1])
qual = P.load(R.QUAL_PATH, "qual")["body"]
_, _, _, verdict = R._verdict(library, qual["choices"])
sys.stdout.write(json.dumps({"comparable": P.plain(R._comparable(verdict)), "threads": numerics.blas_threads(),
                             "core": numerics.blas_core(), "selected": qual["choices"]["selected"]["candidate"]}))
"""


class KnownR2NumericalFragility(UserWarning):
    """KNOWN_R2_NUMERICALLY_FRAGILE_SECONDARY_ROWS deviated on this runtime (the finding RET2E recorded)."""


def test_current_runtime_reaches_the_recorded_qual_verdict():
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", _RUN, str(_library_path())], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True)
    current = json.loads(out.stdout)
    assert current["threads"] == 1, current["threads"]
    result = C.check(current["comparable"], current["selected"], C.qual_body(), C.frozen_body())
    # The kernel name is diagnostic only; nothing above or below depends on it.
    assert not result["violations"], (current["core"], result["violations"], result["fragile_deviations"])
    if result["fragile_deviations"]:
        warnings.warn(KnownR2NumericalFragility(
            f"{C.FRAGILE_LABEL} deviate from the primary QUAL host on this runtime (OpenBLAS {current['core']}), "
            f"as RET2E gate M recorded: {result['fragile_deviations']}"))
