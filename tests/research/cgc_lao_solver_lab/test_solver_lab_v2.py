"""CGC-LAO solver lab study v2 (research/cgc_lao/solver_lab/v2): its self-contained suite under pytest.

SCIENTIFIC lane (tests/lanes.py). The suite lives with the study (``cgc_lao_v2_tests``); this module only imports it.
Native tests compile the study's C11 source with the system C compiler into a temporary directory; without a
compiler they skip, unless ELPIS_REQUIRE_NATIVE=1 (as in CI), where they fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[3] / "research" / "cgc_lao" / "solver_lab" / "v2"
for _path in (str(V2 / "src"), str(V2)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cgc_lao_v2_tests.test_v2_core import *  # noqa: E402,F401,F403
