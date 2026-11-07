"""CGC-LAO solver lab, qualification version v4 (RESEARCH_ONLY, NO_RUNTIME_AUTHORITY).

The final qualification version: the robust Occam selection rule (smallest mechanism whose DEV 99.9% bounds
pass every gate). Additive: v3, v2 and v1 packages are imported read-only (PREREGISTRATION_V4.md).
"""
import sys
from pathlib import Path

V4_ROOT = Path(__file__).resolve().parents[2]
LAB = V4_ROOT.parent
for _p in (LAB / "v3" / "src", LAB / "v2" / "src", LAB / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import cgc_lao_v3  # noqa: E402,F401
