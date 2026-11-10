"""Fixed R121 controller schedules. No learned state, teachers or source edits."""
from __future__ import annotations

SIGNED = "STATE8_SIGNED8"
CONTROLLERS = ("NATIVE", "R114", "SIGNED_ONLY", "DELAYED8")


def controller_at(checkpoint: str, policy: str, pass_number: int) -> str:
    if checkpoint.rsplit("/", 1)[-1] not in {"STATE8", SIGNED, "STATE8_TF1"} or "/" not in checkpoint:
        raise ValueError("UNKNOWN_CHECKPOINT")
    if checkpoint not in {f"{seed}/{arm}" for seed in ("54007", "54031") for arm in ("STATE8", SIGNED, "STATE8_TF1")}:
        raise ValueError("UNKNOWN_CHECKPOINT")
    if policy not in CONTROLLERS or type(pass_number) is not int or pass_number < 1 or pass_number > 20:
        raise ValueError("POLICY_OR_PASS")
    if policy in ("NATIVE", "R114"):
        return policy
    if checkpoint.endswith("/" + SIGNED):
        return "R114" if policy == "SIGNED_ONLY" or (policy == "DELAYED8" and pass_number > 8) else "NATIVE"
    return "NATIVE"
