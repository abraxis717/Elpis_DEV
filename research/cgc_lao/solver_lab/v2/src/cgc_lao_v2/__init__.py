"""CGC-LAO solver lab, study v2 (SLB2): qualification of the improved reliability-mixture learner, the
measurement-independence contract, equal-information controls, the native C11 component and the research host.

RESEARCH_ONLY, NO_RUNTIME_AUTHORITY. Additive to v1: the frozen v1 package ``cgc_lao_lab`` (``../../src``) is
imported read-only for its counter streams, episode container, selectors and rollout; v2 freezes bind those
files by digest as well. Nothing here is packaged or imported by ``src/elpis``.
"""
import sys
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[2]
V1_SRC = V2_ROOT.parent / "src"
if str(V1_SRC) not in sys.path:
    sys.path.insert(0, str(V1_SRC))
