"""CGC-LAO solver lab, qualification version v3 (RESEARCH_ONLY, NO_RUNTIME_AUTHORITY).

A fresh qualification of the v2 mechanism family after the v2 primary failed G9 on VALIDATION. Additive: the
frozen v2 package (``../../v2/src``) and v1 package (``../../src``) are imported read-only; the SLB2 world,
mechanisms, gates, coefficients, native component, provider contract and host are reused unchanged. New here: the
v3 RNG root and partitions, the v3 candidate set and the CI-bound selection rule (PREREGISTRATION_V3.md).
"""
import sys
from pathlib import Path

V3_ROOT = Path(__file__).resolve().parents[2]
LAB = V3_ROOT.parent
for _p in (LAB / "v2" / "src", LAB / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import cgc_lao_v2  # noqa: E402,F401
