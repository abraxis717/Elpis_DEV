"""Advisory promotion observation over bound phase evidence.

``generate_authority_audit`` binds a source chain of phase-evidence directories,
evaluates the promotion gates, derives the advisory decision and, only when
READY, renders the non-executable plan and checks its closed data surface. The
observation is data: it authorizes nothing (the canonical promotion authority
requires a separate explicit operator approval).
"""

import hashlib
import json

from .decision import make_decision
from .gates import evaluate_gates
from .plan import render_plan, verify_plan_nonexecutable
from .source_binding import build_source_chain


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_authority_audit(config: dict) -> dict:
    """Observe configured plan data; omit all unobserved runtime authority claims."""
    chain = build_source_chain(config)
    decision = make_decision(evaluate_gates(chain), chain)
    plan = render_plan(decision, chain)
    observation = {
        "schema": "elpis.grid81.promotion-plan-observation.v2",
        "source_chain_digest": chain.chain_digest,
        "decision_digest": decision.digest,
        "plan_status": "NOT_RENDERED" if plan is None else "RENDERED",
    }
    if plan is not None:
        observation["plan_check"] = verify_plan_nonexecutable(plan)
    observation["observation_digest"] = hashlib.sha256(json.dumps(
        observation, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    return observation
