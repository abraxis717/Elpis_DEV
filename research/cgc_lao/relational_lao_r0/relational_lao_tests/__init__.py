"""Relational LAO R0 test suite (unittest). From the repository root:

    PYTHONPATH=research/cgc_lao/relational_lao_r0/src:research/cgc_lao/relational_lao_r0 \
      python3 -B -m unittest discover -s research/cgc_lao/relational_lao_r0/relational_lao_tests \
      -t research/cgc_lao/relational_lao_r0 -v

Native tests (the real ECS_C relational view, native HCA) need the built libraries (ELPIS_NATIVE_BUILD or build/);
with ELPIS_REQUIRE_NATIVE=1 their absence fails instead of skipping.
"""
import os
import sys
from pathlib import Path

R0 = Path(__file__).resolve().parents[1]
if str(R0 / "src") not in sys.path:
    sys.path.insert(0, str(R0 / "src"))
import relational_lao  # noqa: E402,F401

REQUIRE_NATIVE = os.environ.get("ELPIS_REQUIRE_NATIVE") == "1"
