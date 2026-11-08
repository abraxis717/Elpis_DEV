"""CGC-LAO post-freeze admission correction: SCIENTIFIC pytest bridge.

The historical v1-v4 studies and their frozen source digests are unchanged.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "research" / "cgc_lao" / "solver_lab"
for _p in (ROOT, ROOT / "v2" / "src", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from host_closure.test_strict_host import *  # noqa: E402,F401,F403
