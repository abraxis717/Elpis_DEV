"""R120 preregistered static checkpoint-only correction policy.

RESEARCH_ONLY. Selects an already-qualified frozen operator controller;
does not train, update parameters, or inspect teacher/target states.
"""
from __future__ import annotations

from research.yang_mills_hca.r118_scalable_awr import provenance as P


def controller_for_checkpoint(checkpoint: str) -> str:
    if checkpoint not in P.CHECKPOINTS:
        raise ValueError("UNKNOWN_FROZEN_CHECKPOINT:" + str(checkpoint))
    return "R114" if checkpoint.rsplit("/", 1)[1] == "STATE8_SIGNED8" else "NATIVE"


def create_engine(state, checkpoint, **kwargs):
    """Construct the R119 exact guarded engine with static, teacher-free selection."""
    if "controller" in kwargs or "mode" in kwargs:
        raise ValueError("POLICY_OVERRIDE_FORBIDDEN")
    from research.yang_mills_hca.r119_numerical_hotpath.engine_r119 import R119Engine

    return R119Engine(state, checkpoint, controller=controller_for_checkpoint(checkpoint),
                      mode="EXACT_GUARDED", **kwargs)
