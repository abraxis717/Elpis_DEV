"""R118-A: the addressable HCA state-instance contract, derived from the frozen source.

Derivation (frozen_snapshot sources, verified by provenance.py):

* ``portable_mechanics.link_builder`` iterates ``dst``/``src`` over the four rows of
  ``graph`` (the *relations*) and, for every participant ``p`` shared by both,
  appends the link operator ``U[dst,p] @ U[src,p].T``. ``r62.graph_step_split``
  aggregates over those links for ``d in range(4)``; the state tensor is
  ``[batch, 4, 4]`` per channel. ``README_R78.txt``: "The HCA's native state is
  (A:R4,B:R4) across four relation nodes."

Hence:

=====================  ==========================================  ==============
concept                historical object                           carries state
=====================  ==========================================  ==============
global participant     member label (0..9) inside relation rows;   no
                       anchors one frame ``U[r,p]`` per incidence
relation               one row of ``graph``; one HCA node with     yes, 8 coords
                       ``(A, B)`` in R^4 x R^4, in its own frame    (A[4], B[4])
local operator role    slot ``d`` in {0,1,2,3} of ``[batch,4,4]``  no
                       (historical source at 0, readout at 3)
HCA state instance     one batch row: 4 slots x 2 channels x 4     yes, 32 coords
                       = 32 coordinates; the R104/R114 controller   (view)
                       acts on these 32 coordinates jointly
link                   (src relation, dst relation, shared p) with derived, never
                       operator ``U[dst,p] U[src,p]^T``             stored globally
=====================  ==========================================  ==============

A participant is never "one HCA state". The frozen operator has a fixed local
arity of four relation slots (``range(4)`` and ``reshape(...,4,...)`` are
hard-coded) and the controller couples the four slots through one radial
scalar, so R118 preserves the 4-slot window as the unit of HCA invocation and
composes larger networks from many overlapping windows (see ARCHITECTURE.md).

The per-slot law is node-local; it only sees its own (A, B), the softmax
salience ``sal`` and the mean collective ``col`` over its in-links. Empty slots
(no bound relation) have zero state and no links and stay exactly zero.

Canonical 32-coordinate layout of an instance (``state32``)::

    index = channel * 16 + slot * 4 + k     channel 0 = A, 1 = B; k = R^4 coordinate

(the controller's dot products are layout independent; the layout only fixes I/O).
"""
from __future__ import annotations

import math

SLOTS = 4
CHANNELS = 2
DIM = 4
SLOT_COORDS = CHANNELS * DIM  # 8 per relation node: A[4] then B[4]
COORDS = SLOTS * SLOT_COORDS  # 32
RADIAL_EPSILON = 1e-24  # research/yang_mills_hca/reference.py

CONTROLLERS = ("NATIVE", "R104", "R107", "R114")
VARIANTS = ("STATE8", "STATE8_SIGNED8", "STATE8_TF1")


def controller_gains(controller: str, checkpoint: str) -> tuple[float, float] | None:
    """(alpha, beta) for a controller configuration, or None for the native HCA.

    R114 is delegated verbatim to the published policy ``reference.gains``
    (STATE8_SIGNED8 -> (0.70, 0.45), other variants -> (0.50, 0.50)). It is
    checkpoint-metadata gating, not a learned router, and reads no teacher state.
    """
    from research.yang_mills_hca import reference

    reference.gains(checkpoint)  # validates the checkpoint name for every controller
    if controller == "NATIVE":
        return None
    if controller == "R104":
        return (0.50, 0.50)
    if controller == "R107":
        return (0.70, 0.45)
    if controller == "R114":
        return reference.gains(checkpoint)
    raise ValueError("UNKNOWN_CONTROLLER:" + str(controller))


def to_state32(a_slots, b_slots) -> tuple[float, ...]:
    """(A[4][4], B[4][4]) nested sequences -> canonical 32-tuple."""
    out = []
    for channel in (a_slots, b_slots):
        if len(channel) != SLOTS:
            raise ValueError("SLOT_COUNT")
        for row in channel:
            if len(row) != DIM:
                raise ValueError("DIM")
            out.extend(float(v) for v in row)
    return tuple(out)


def from_state32(state32):
    if len(state32) != COORDS:
        raise ValueError("SHAPE")
    a = [list(map(float, state32[s * DIM:(s + 1) * DIM])) for s in range(SLOTS)]
    b = [list(map(float, state32[16 + s * DIM:16 + (s + 1) * DIM])) for s in range(SLOTS)]
    return a, b


def correct_instance_exact(state32, prediction32, controller: str, checkpoint: str):
    """The published controller algebra, verbatim (``reference.correct_node``)."""
    gains = controller_gains(controller, checkpoint)
    if gains is None:
        p = tuple(float(v) for v in prediction32)
        if len(p) != COORDS or not all(map(math.isfinite, p)):
            raise ValueError("SHAPE_OR_NONFINITE")
        return p
    from research.yang_mills_hca import reference

    if controller == "R114":
        return reference.correct_node(state32, prediction32, checkpoint)
    # R104 / R107 use the same algebra with fixed gains; reference.correct_node only
    # exposes the R114 policy, so the identical expression is evaluated here.
    a, b = gains
    s = tuple(float(x) for x in state32)
    p = tuple(float(x) for x in prediction32)
    if len(s) != COORDS or len(p) != COORDS:
        raise ValueError("SHAPE")
    if not all(map(math.isfinite, s + p)):
        raise ValueError("NONFINITE")
    d = tuple(v - u for u, v in zip(s, p))
    norm = math.fsum(x * x for x in s)
    dot = math.fsum(x * y for x, y in zip(s, d))
    radial = tuple(x * dot / max(norm, RADIAL_EPSILON) for x in s)
    result = tuple(v + a * r + b * (delta - r) for v, r, delta in zip(p, radial, d))
    if not all(map(math.isfinite, result)):
        raise ValueError("NONFINITE_RESULT")
    return result
