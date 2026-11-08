"""Freeze the v2 selection (SELECTION_V2.json) before VALIDATION: configurations, every Python source (v2 and the
v1 files it imports), the native C11 sources, the root CMake wiring, both specs and the v2 coefficients, plus
the exact VALIDATION, HOLDOUT and OOD commands."""
from __future__ import annotations

import json

from cgc_lao_lab import evaluate as ev1

from . import V2_ROOT, candidates2, env2, evaluate2

FROZEN = V2_ROOT / "FROZEN_V2.json"
ROOT_CMAKE = V2_ROOT.parents[3] / "CMakeLists.txt"
PLANS = {
    "VALIDATION": {"partition": "VALIDATION", "seeds": [0, 32], "episodes": 4096,
                   "regimes": list(env2.DESIGN + env2.ROBUSTNESS)},
    "HOLDOUT": {"partition": "HOLDOUT", "seeds": [0, 64], "episodes": 4096,
                "regimes": list(env2.DESIGN + env2.ROBUSTNESS)},
    "OOD": {"partition": "OOD", "seeds": [0, 64], "episodes": 2048,
            "regimes": ["FLIP40"] + list(env2.CORRELATION) + ["EVENT40"] + list(env2.OOD)},
}


def qualification_specs(primary: str, fallback: str) -> dict:
    s = evaluate2.spec
    specs = {a: s(a, "ORIGINAL", "FIRST_ONLY") for a in candidates2.ORIGINAL_BASELINES}
    specs["KNOWN_Q_ORACLE"] = s("KNOWN_Q_ORACLE", "ORIGINAL", "FIRST_ONLY")
    specs[primary] = s(primary, "DUP2_4", "DUPLICATE")
    specs[fallback] = s(fallback, "DUP2_4", "FIRST_ONLY")
    for c in candidates2.CONTROLS:
        if c not in ("NOOP", primary):
            specs[c] = s(c, "DUP2_4", "DUPLICATE")
    specs["KNOWN_Q_DUP"] = s("KNOWN_Q_ORACLE", "DUP2_4", "DUPLICATE")
    return specs


def ood_specs(primary: str, fallback: str) -> dict:
    s = evaluate2.spec
    return {"NOOP": s("NOOP", "ORIGINAL", "FIRST_ONLY"), primary: s(primary, "DUP2_4", "DUPLICATE"),
            f"{primary}@SECOND_EVENT": s(primary, "DUP2_4", "SECOND_EVENT"), fallback: s(fallback, "DUP2_4", "FIRST_ONLY"),
            "RMF_V1": s("RMF_V1", "DUP2_4", "DUPLICATE"), "CLEAN_BAYES": s("CLEAN_BAYES", "ORIGINAL", "FIRST_ONLY"),
            "KNOWN_Q_DUP": s("KNOWN_Q_ORACLE", "DUP2_4", "DUPLICATE")}


def command(phase: str, specs: dict) -> str:
    p = PLANS[phase]
    return (f"PYTHONPATH=src python3 -m cgc_lao_v2.report2 V2-{phase} --partition {p['partition']} "
            f"--seeds {p['seeds'][0]}:{p['seeds'][1]} --episodes {p['episodes']} --regimes {','.join(p['regimes'])} "
            f"--arms '{json.dumps(specs, separators=(',', ':'))}' --out evidence/{phase.lower()}/qualification.json")


def build() -> dict:
    selection = json.loads((V2_ROOT / "SELECTION_V2.json").read_text(encoding="utf-8"))
    primary, fallback = selection["primary"]["chosen"], selection["fallback"]["chosen"]
    cmake_line = "add_subdirectory(research/cgc_lao/solver_lab/v2/native)"
    if cmake_line not in ROOT_CMAKE.read_text(encoding="utf-8"):
        raise SystemExit("root CMakeLists.txt does not wire the native research tests")
    frozen = {
        "schema": "cgc-lao-solver-lab.freeze.v2",
        "note": "Frozen before any v2 VALIDATION, HOLDOUT or OOD run; selection by SELECTION_V2.json (DEV round 1).",
        "candidates": {
            "PRIMARY": {"name": primary, "config": candidates2.BAYES_CONFIGS[primary], "contract": "DUP2_4",
                        "mode": "DUPLICATE", "admission": "Contract I: audited independent re-reads"},
            "FALLBACK": {"name": fallback, "config": candidates2.BAYES_CONFIGS[fallback], "contract": "DUP2_4",
                         "mode": "FIRST_ONLY", "admission": "Contract II: independence not established"},
        },
        "source_sha256": evaluate2.source_digest(),
        "source_files_sha256": evaluate2.source_file_digests(),
        "native_files_sha256": evaluate2.native_file_digests(),
        "root_cmake_wiring": cmake_line,
        "spec_sha256": {"slb1": ev1.file_digest(ev1.SPEC), "slb2": ev1.file_digest(evaluate2.SPEC2)},
        "coefficients_sha256": evaluate2.coefficient_digests(),
        "plans": PLANS,
        "commands": {
            "VALIDATION": command("VALIDATION", qualification_specs(primary, fallback)),
            "HOLDOUT": command("HOLDOUT", qualification_specs(primary, fallback)),
            "OOD": command("OOD", ood_specs(primary, fallback)),
            "VALIDATION_NATIVE_PARITY": "PYTHONPATH=src python3 -m cgc_lao_v2.parity2 VALIDATION evidence/validation/native_parity.json",
            "HOLDOUT_NATIVE_PARITY": "PYTHONPATH=src python3 -m cgc_lao_v2.parity2 HOLDOUT evidence/holdout/native_parity.json",
        },
        "holdout_rule": "open HOLDOUT only if the PRIMARY passes G1-G5, G7, G9, G10 on VALIDATION",
    }
    return frozen


if __name__ == "__main__":
    rec = build()
    FROZEN.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec["candidates"], indent=1))
    print(rec["source_sha256"])
