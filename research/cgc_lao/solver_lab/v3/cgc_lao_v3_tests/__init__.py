"""Test suite of qualification version v3 (unittest). From the repository root:

    PYTHONPATH=research/cgc_lao/solver_lab/v3/src:research/cgc_lao/solver_lab/v3 \
      python3 -B -m unittest discover -s research/cgc_lao/solver_lab/v3/cgc_lao_v3_tests \
      -t research/cgc_lao/solver_lab/v3 -v
"""
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
if str(V3 / "src") not in sys.path:
    sys.path.insert(0, str(V3 / "src"))
import cgc_lao_v3  # noqa: E402,F401
