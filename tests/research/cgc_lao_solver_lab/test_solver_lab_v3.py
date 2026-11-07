"""CGC-LAO solver lab qualification version v3 (research/cgc_lao/solver_lab/v3): its suite under pytest.

SCIENTIFIC lane (tests/lanes.py). The suite lives with the study (``cgc_lao_v3_tests``); this module only imports it.
"""
from __future__ import annotations

import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[3] / "research" / "cgc_lao" / "solver_lab" / "v3"
for _path in (str(V3 / "src"), str(V3)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cgc_lao_v3_tests.test_v3 import *  # noqa: E402,F401,F403
