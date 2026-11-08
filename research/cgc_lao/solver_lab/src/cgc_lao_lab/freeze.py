"""Freeze the selected candidates (one per contract) before VALIDATION, binding their configuration,
contract, every source file, the spec and the coefficients by SHA-256, and fixing the run plans."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import evaluate
from .candidates import CANDIDATES
from .env import CONTRACTS, DESIGN_REGIMES, SHIFT_REGIMES

LAB = Path(__file__).resolve().parents[2]
FROZEN = LAB / "FROZEN.json"
REFERENCE_ARMS = ["NOOP", "SGD", "BETA", "CGC_REFERENCE", "STATIC_ATTENUATION", "CLEAN_BAYES", "KNOWN_Q_ORACLE"]
REGIMES = list(DESIGN_REGIMES) + list(SHIFT_REGIMES)
PLANS = {
    "VALIDATION": {"partition": "VALIDATION", "seeds": [0, 32], "episodes": 4096},
    "HOLDOUT": {"partition": "HOLDOUT", "seeds": [0, 64], "episodes": 4096},
}


def commands(phase: str, frozen: dict) -> list:
    plan = PLANS[phase]
    base = (f"python3 -m cgc_lao_lab.report {phase} --partition {plan['partition']} "
            f"--seeds {plan['seeds'][0]}:{plan['seeds'][1]} --episodes {plan['episodes']} --regimes {','.join(REGIMES)}")
    out = []
    for contract, cand in frozen["candidates"].items():
        if contract == "ORIGINAL":
            arms = REFERENCE_ARMS + [cand["name"]]
        else:
            arms = [cand["name"], "KNOWN_Q_ORACLE"]
        out.append(f"{base} --contract {contract} --arms {','.join(arms)} "
                   f"--out evidence/{phase.lower()}/{contract.lower()}.json")
    return out


def build(selection: dict, note: str) -> dict:
    frozen = {
        "schema": "cgc-lao-solver-lab.freeze.v1",
        "note": note,
        "candidates": {contract: {"name": name, "config": CANDIDATES[name], "contract_reads": list(CONTRACTS[contract].reads)}
                       for contract, name in selection.items()},
        "source_sha256": evaluate.source_digest(),
        "source_files_sha256": evaluate.source_file_digests(),
        "spec_sha256": evaluate.file_digest(evaluate.SPEC),
        "coefficients_sha256": evaluate.coefficient_digests(),
        "regimes": REGIMES,
        "reference_arms": REFERENCE_ARMS,
        "plans": PLANS,
    }
    frozen["commands"] = {phase: commands(phase, frozen) for phase in PLANS}
    return frozen


if __name__ == "__main__":
    selection = dict(item.split("=") for item in sys.argv[1:])
    rec = build(selection, "Selected on DEV round 2 by the preregistered rule (most gates passed by point "
                           "estimate, ties broken by ALL terminal regret); frozen before any VALIDATION run.")
    FROZEN.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec["candidates"], indent=1))
    print("\n".join(rec["commands"]["VALIDATION"]))
