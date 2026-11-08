"""Test suite of the CGC-LAO solver lab study v2 (unittest; numpy; the native tests compile the C11 source into a
temporary directory with the system C compiler). Run from the repository root:

    PYTHONPATH=research/cgc_lao/solver_lab/v2/src:research/cgc_lao/solver_lab/v2 \
      python3 -B -m unittest discover -s research/cgc_lao/solver_lab/v2/cgc_lao_v2_tests \
      -t research/cgc_lao/solver_lab/v2 -v
"""
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2 / "src") not in sys.path:
    sys.path.insert(0, str(V2 / "src"))
import cgc_lao_v2  # noqa: E402,F401  (puts the frozen v1 package on the path)
