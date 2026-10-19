"""Retained path only; the one-ECS gate that lived here is retired.

``native/continuity/qualify.sh`` names this file, and that script cannot change: the K1 Recovery R0 v4 premise P1
binds ``native/continuity`` byte for byte to the ABI v4 commit (``research/qualification/k1_recovery_r0_v4/premise.py``),
and the RUN PASS carries to a later commit only while it holds. The one-ECS assertions were retired with the one-ECS
tree (``docs/NATIVE_SPLIT_CENSUS.md``, section 9); their successor is ``tests/boundary/test_ecs_split.py``. Named
explicitly, as the script names it, this module runs that successor in a fresh interpreter, so the script still runs
the boundary gate; a full collection runs the successor a second time, at a cost of seconds. The path retires with
``native/continuity`` itself (Stage F): once the script no longer names it, the test below fails and this file goes.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUCCESSOR = "tests/boundary/test_ecs_split.py"


def test_the_frozen_continuity_script_runs_the_successor_split_gate():
    assert "tests/boundary/test_one_ecs.py" in (REPO / "native/continuity/qualify.sh").read_text()
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", SUCCESSOR],
                          cwd=REPO, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
