"""Native K1 against the frozen R3 result on the current runtime, with same-host references only. Never skipped
for a host or profile difference; nothing is forced.

Every R3 QUAL world is built on this host (its task data is BLAS-kernel dependent, so recorded trajectory digests are
historical bytes of the recording host and are not compared here; their integrity is pinned in test_evidence.py).
Asserted per world, bitwise: K1-disabled native = same-host canonical Runtime R1 G1; native W_A = same-host
canonical W_A; native query = Runtime R1 forward of the same W; FMS-resident = standalone; envelope round trip;
reset, W-only, transplant and state semantics. Then the native rows, substituted into the recorded R3 rows under the
unchanged R3 gates, must reproduce every decision-bearing R3 quantity and OUTCOME_A.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from ...ECS_G.test_math_r0 import REPO, _library_path


def test_current_runtime_same_host_invariants_and_the_r3_decision_record():
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("OPENBLAS_CORETYPE", None)
    out = subprocess.run([sys.executable, "-m", "research.ecs_k1_native.regression", "--library",
                          str(_library_path())], cwd=REPO, env=env, capture_output=True, text=True, check=True)
    current = json.loads(out.stdout)
    failing = {w: sorted(k for k, v in c.items() if not v) for w, c in current["checks"].items() if not all(c.values())}
    assert not failing, failing
    assert current["determinism"]["identical"] and current["clean_transplant"]["bitwise"]
    assert all(current["decision_record_equal"].values()), (current["decision_record_equal"],
                                                            current["decision_record_diff"])
    assert current["outcome"] == "OUTCOME_A"
