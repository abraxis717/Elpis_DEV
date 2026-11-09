"""Research-only scalar controller algebra; NOT the trained 88-parameter HCA.

No weights, native GPU implementation, global AWR scheduler, disk writes or
persistent state. Exactly one 32-coordinate node per call.
"""
from __future__ import annotations
import math

VALID = {f"{s}/{v}" for s in ("54007", "54031") for v in
         ("STATE8", "STATE8_SIGNED8", "STATE8_TF1")}


def gains(checkpoint: str) -> tuple[float, float]:
    if checkpoint not in VALID:
        raise ValueError("UNKNOWN_CHECKPOINT")
    return (0.70, 0.45) if checkpoint.endswith("/STATE8_SIGNED8") else (0.50, 0.50)


def correct_node(state, hca_prediction, checkpoint: str):
    """Node-local same-gauge-frame radial/transverse correction."""
    a, b = gains(checkpoint)
    if len(state) != 32 or len(hca_prediction) != 32:
        raise ValueError("SHAPE")
    s = tuple(float(x) for x in state)
    p = tuple(float(x) for x in hca_prediction)
    if not all(map(math.isfinite, s+p)):
        raise ValueError("NONFINITE")
    d = tuple(v-u for u, v in zip(s,p))
    norm = math.fsum(x*x for x in s)
    dot = math.fsum(x*y for x, y in zip(s,d))
    radial = tuple(x * dot / max(norm, 1e-24) for x in s)
    result = tuple(v + a*r + b*(delta-r) for v,r,delta in zip(p,radial,d))
    if not all(map(math.isfinite, result)):
        raise ValueError("NONFINITE_RESULT")
    return result
