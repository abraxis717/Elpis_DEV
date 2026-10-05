"""Native K1 against the frozen R3 record on the current runtime. Never skipped for a host or profile difference.

Re-runs the differential over every R3 QUAL world in a fresh interpreter (no forced kernel). Asserted: every exact
comparison of the plan (K1-disabled parity with the recorded M0 digests, W_A, resident = standalone, classes,
state-removal classes, W-only control, transplant in process and in a clean process, state semantics,
determinism) and the substituted decision record equal to the recorded R3 one (OUTCOME_A). The float tolerances are
not re-asserted here: the recorded verdict on them (NOT_QUALIFIED, research/ecs_k1_native/evidence) stands, and the
two failing worlds are sensitive to rounding in the R3 laboratory itself (docs/research/ECS_K1_NATIVE_RESULTS.md).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from ...ECS_G.test_math_r0 import REPO, _library_path

EXACT = ("zero_k1_parity", "w_a", "resident_equals_standalone", "classification", "state_removal_classification",
         "w_only_negative_control", "transplant_in_process", "state_semantics")


def test_current_runtime_reproduces_every_exact_comparison_and_the_r3_decision_record():
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-m", "research.ecs_k1_native.run", "check", "--library",
                          str(_library_path())], cwd=REPO, env=env, capture_output=True, text=True, check=True)
    current = json.loads(out.stdout)
    for key in EXACT:
        failing = sorted(w for w, c in current["checks"].items() if not c[key])
        assert not failing, (key, failing)
    assert current["determinism"]["identical"] and current["clean_transplant"]["bitwise"]
    assert all(current["verdict"]["decision_record_equal"].values()), current["verdict"]["decision_record_equal"]
    assert current["verdict"]["outcome"] == "OUTCOME_A"
