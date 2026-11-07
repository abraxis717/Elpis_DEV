"""Test suite of qualification version v4 (unittest). From the repository root:

    PYTHONPATH=research/cgc_lao/solver_lab/v4/src:research/cgc_lao/solver_lab/v4 \
      python3 -B -m unittest discover -s research/cgc_lao/solver_lab/v4/cgc_lao_v4_tests \
      -t research/cgc_lao/solver_lab/v4 -v
"""
import sys
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
if str(V4 / "src") not in sys.path:
    sys.path.insert(0, str(V4 / "src"))
import cgc_lao_v4  # noqa: E402,F401
