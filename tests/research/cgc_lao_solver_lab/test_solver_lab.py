"""CGC-LAO solver lab (research/cgc_lao/solver_lab): its self-contained mechanical suite under pytest.

SCIENTIFIC lane (tests/lanes.py). Numpy only, seconds, no files written, no network. The suite lives with
the laboratory (``cgc_lao_lab_tests``) so the laboratory stays self-contained; this module only imports it.
The optional torch parity test skips when torch is absent (it is never a CI dependency).
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[3] / "research" / "cgc_lao" / "solver_lab"
for _path in (str(LAB / "src"), str(LAB)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cgc_lao_lab_tests.test_benchmark import *  # noqa: E402,F401,F403
from cgc_lao_lab_tests.test_boundaries import *  # noqa: E402,F401,F403
from cgc_lao_lab_tests.test_mechanics import *  # noqa: E402,F401,F403
from cgc_lao_lab_tests.test_statistics import *  # noqa: E402,F401,F403
