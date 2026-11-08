"""Relational LAO R0 (RESEARCH ONLY, NOT QUALIFIED, NO RUNTIME AUTHORITY).

Generalises the qualified LAO reliability mechanism (RMF3, solver_lab v4) from independent expert biases to a
bounded relational belief state, consumed from the ECS_C relational view and projected to research-only HCA.
The LAO v1-v4 studies are imported read-only and never modified (SPEC.md).
"""
import sys
from pathlib import Path

R0_ROOT = Path(__file__).resolve().parents[2]
REPO = R0_ROOT.parents[2]
LAB = R0_ROOT.parent / "solver_lab"
for _p in (LAB, LAB / "v2" / "src", LAB / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
