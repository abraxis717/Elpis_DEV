"""CGC-LAO solver lab qualification version v4 (research/cgc_lao/solver_lab/v4): its suite under pytest.

SCIENTIFIC lane (tests/lanes.py). The suite lives with the study (``cgc_lao_v4_tests``); this module only imports it.
"""
from __future__ import annotations

import sys
from pathlib import Path

V4 = Path(__file__).resolve().parents[3] / "research" / "cgc_lao" / "solver_lab" / "v4"
for _path in (str(V4 / "src"), str(V4)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cgc_lao_v4_tests.test_v4 import *  # noqa: E402,F401,F403
