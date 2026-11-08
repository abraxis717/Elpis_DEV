"""Mechanical test suite of the CGC-LAO solver lab (unittest; numpy only; no files written, no network).

Run from the repository root:
    PYTHONPATH=research/cgc_lao/solver_lab/src:research/cgc_lao/solver_lab \
      python3 -B -m unittest discover -s research/cgc_lao/solver_lab/cgc_lao_lab_tests \
      -t research/cgc_lao/solver_lab -v
"""
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
if str(LAB / "src") not in sys.path:
    sys.path.insert(0, str(LAB / "src"))
