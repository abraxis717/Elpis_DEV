"""Evaluate the pre-registered Runtime R1 gates (workloads.GATES, fixed at R1A) on the recorded evidence.

    python -m research.ecs_runtime_r1.gates

Measured gates are computed from evidence/baseline.json, evidence/final.json,
evidence/jitter.json and evidence/sanitizers.json exactly as registered.
evidence/overhead.json adds a supplementary split-timing measurement of the
Python control plane, reported next to (never instead of) the registered
estimator. Structural gates are enforced by tests; this names them.
PERFORMANCE_ONLY.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import workloads as WL

EVIDENCE = Path(__file__).resolve().parent / "evidence"
PYTHON_LEARN = ("py_core_learn", "py_core_learn_noreceipt", "py_core_learn_buffer", "py_exec_learn")
STRUCTURAL = {
    "single_ffi": ["tests/ECS_G/test_executor_binding.py::test_learn_is_one_native_call_whatever_k",
                   "tests/ECS_G/test_executor_binding.py::test_query_is_one_native_call_and_read_only",
                   "tests/boundary/test_runtime_r1.py"],
    "no_hot_allocation": ["native/ECS_G/tests/test_ecsg_executor_alloc.c",
                          "tests/ECS_G/test_executor_binding.py::test_native_workspace_allocates_nothing_after_creation"],
    "scalar_parity": ["native/ECS_G/tests/test_ecsg_executor.c (dispatched and baseline-ISA builds)",
                      "native/ECS_G/tests/test_ecsg_executor_txn.c",
                      "tests/ECS_G/test_executor_differential.py",
                      "tests/research/ecs_cognition_r0/test_lab.py::test_qual_measurements_reproduce_exactly (8 worlds)"],
}


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="ascii"))


def _index(evidence: dict) -> dict:
    return {(r["workload"], r["mode"]): r for r in evidence["results"]}


def material_speedup(baseline: dict, final: dict) -> dict:
    base, new = _index(baseline), _index(final)
    rows, ok = {}, True
    for k in WL.R0_STEPS:
        w = f"r0-learn-K{k}"
        before = base[w, "py_core_learn"]["p50_ns"]
        row = {"baseline_py_core_learn_p50_ns": before}
        for mode in PYTHON_LEARN:
            row[f"{mode}_p50_ns"] = new[w, mode]["p50_ns"]
            row[f"speedup_{mode}"] = round(before / new[w, mode]["p50_ns"], 2)
        row["native_speedup_exec_vs_ref"] = round(new[w, "ref_learn"]["p50_ns"] / new[w, "exec_learn"]["p50_ns"], 2)
        if k >= 10:
            ok = ok and row["speedup_py_core_learn"] >= WL.THRESHOLDS["speedup_min"]
        rows[w] = row
    return {"pass": ok, "rule": WL.GATES["material_speedup"], "per_workload": rows}


def python_overhead(final: dict) -> dict:
    new, t = _index(final), WL.THRESHOLDS
    per_mode, ok = {}, True
    for mode in PYTHON_LEARN:
        overhead = {k: new[f"r0-learn-K{k}", mode]["p50_ns"] - new[f"r0-learn-K{k}", "exec_learn"]["p50_ns"]
                    for k in WL.R0_STEPS}
        limit = t["overhead_factor"] * overhead[10] + t["overhead_slack_us"] * 1e3
        passed = overhead[4000] <= limit
        per_mode[mode] = {"overhead_ns": {str(k): v for k, v in overhead.items()}, "limit_ns_at_K4000": limit,
                          "pass": passed}
        ok = ok and passed
    return {"pass": ok, "rule": WL.GATES["python_overhead_k_independent"], "per_mode": per_mode}


def jitter(final: dict, investigation: dict) -> dict:
    flagged = [{"workload": r["workload"], "mode": r["mode"], "p50_ns": r["p50_ns"], "p99_ns": r["p99_ns"],
                "p99_over_p50": round(r["p99_ns"] / r["p50_ns"], 2)}
               for r in final["results"] if r["p99_ns"] > WL.THRESHOLDS["jitter_ratio"] * r["p50_ns"]]
    pairs = []
    for case in investigation["native"]:
        for label in ("unpinned", "pinned"):
            if label in case:
                pairs.append({"case": f"{case['mode']} N={case['width']} R={case['rows']} K={case['steps']}",
                              "placement": label,
                              "operation_p99_over_p50": case[label]["operation"]["p99_over_p50"],
                              "calibration_p99_over_p50": case[label]["calibration"]["p99_over_p50"]})
    gc_rows = [{"operation": r["operation"], "gc_enabled": r["gc_enabled"]["p99_over_p50"],
                "gc_disabled": r["gc_disabled"]["p99_over_p50"]} for r in investigation["python"]]
    return {"pass": True, "rule": WL.GATES["jitter_investigation"], "flagged": flagged,
            "investigation": {"native_vs_calibration": pairs, "python_gc": gc_rows}}


def sanitizers(report: dict) -> dict:
    configs = report["configs"]
    ok = all(c["ctest_returncode"] == 0 and c["tests"] and all(v == "Passed" for v in c["tests"].values())
             and not c["sanitizer_reports"] for c in configs.values())
    asan_py = configs["gcc-asan-ubsan"]["python_under_asan"]
    ok = ok and asan_py["returncode"] == 0 and asan_py["summary"]["failures"] == 0 and \
        asan_py["summary"]["errors"] == 0 and not asan_py["sanitizer_reports"]
    return {"pass": ok, "rule": WL.GATES["sanitizers"],
            "configs": {n: {"tests": c["tests"], "python": c.get("python_under_asan", {}).get("summary")}
                        for n, c in configs.items()}}


def overhead_split_timing(report: dict) -> dict:
    return {"supplementary": True, "method": report["method"],
            "per_mode": report["registered_rule_on_split_timing"], "cold_probe": report["cold_probe"],
            "pass": all(v["pass"] for v in report["registered_rule_on_split_timing"].values())}


def evaluate() -> dict:
    baseline, final = load("baseline.json"), load("final.json")
    out = {"material_speedup": material_speedup(baseline, final),
           "python_overhead_k_independent": python_overhead(final),
           "python_overhead_split_timing": overhead_split_timing(load("overhead.json")),
           "jitter_investigation": jitter(final, load("jitter.json")),
           "sanitizers": sanitizers(load("sanitizers.json"))}
    for gate, tests in STRUCTURAL.items():
        out[gate] = {"enforced_by_tests": tests, "rule": WL.GATES[gate]}
    return out


def main() -> int:
    json.dump(evaluate(), sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
