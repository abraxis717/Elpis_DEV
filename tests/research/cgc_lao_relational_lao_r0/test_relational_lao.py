"""Relational LAO R0 (research/cgc_lao/relational_lao_r0): its suite under pytest.

SCIENTIFIC lane (tests/lanes.py). The suite lives with the study (``relational_lao_tests``); this module only imports
it. The host tests drive the real ECS_C relational view (native/ECS_C/relational) through research-only ctypes, and
the native relational adapter (native/, compiled into a temporary directory) is checked bitwise against the scalar
reference; with ELPIS_REQUIRE_NATIVE=1 a missing library or compiler fails instead of skipping.
"""
from __future__ import annotations

import sys
from pathlib import Path

R0 = Path(__file__).resolve().parents[3] / "research" / "cgc_lao" / "relational_lao_r0"
for _path in (str(R0 / "src"), str(R0)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from relational_lao_tests.test_mechanics import *  # noqa: E402,F401,F403
from relational_lao_tests.test_r2_gate import *  # noqa: E402,F401,F403
from relational_lao_tests.test_r5_hca import *  # noqa: E402,F401,F403
from relational_lao_tests.test_native import *  # noqa: E402,F401,F403
