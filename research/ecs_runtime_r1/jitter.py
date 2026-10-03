"""Runtime R1 tail-latency investigation. PERFORMANCE_ONLY. NO_SCIENTIFIC_CLAIM.

    PYTHONPATH=src python -m research.ecs_runtime_r1.jitter --build <build dir> [--out <evidence.json>]

For each native executor operation, the same harness times a calibration
loop of matched duration (a dependent floating-point chain: no memory
traffic, no calls, no allocation). If the executor's p99/p50 tracks the
calibration's, its tail is the platform's (scheduling, interrupts, frequency
changes on a shared virtual machine), not the executor's. Each pair is also
run pinned to one CPU (taskset). On the Python side, the same operations are
timed with the cyclic garbage collector enabled and disabled.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys

from . import workloads as WL
from .bench import binding, data, driver, environment, library, time_python

CASES = (("exec_forward", 6, 36, 64, 1), ("exec_learn", 6, 36, 64, 1), ("exec_learn", 6, 36, 64, 100),
         ("exec_learn", 6, 36, 64, 4000), ("exec_learn", 6, 1152, 256, 10))


def _native(build, mode, dim, width, rows, steps, pin=None):
    cmd = [str(driver(build)), mode, str(dim), str(width), str(rows), str(steps), str(WL.BUDGET_SECONDS)]
    if pin is not None:
        cmd = ["taskset", "-c", str(pin)] + cmd
    out = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    return {k: out[k] for k in ("samples", "first_call_ns", "min_ns", "p50_ns", "p95_ns", "p99_ns", "max_ns")} | {
        "p99_over_p50": round(out["p99_ns"] / out["p50_ns"], 3)}


def native_pairs(build):
    unit = _native(build, "calibrate", 6, 36, 64, 1)["p50_ns"]  # ns per 1000 chain iterations
    pin = 3 if shutil.which("taskset") else None
    out = []
    for mode, dim, width, rows, steps in CASES:
        row = {"mode": mode, "dim": dim, "width": width, "rows": rows, "steps": steps}
        for label, cpu in (("unpinned", None), ("pinned", pin)):
            if label == "pinned" and cpu is None:
                continue
            op = _native(build, mode, dim, width, rows, steps, cpu)
            k = max(1, round(op["p50_ns"] / unit))
            row[label] = {"operation": op, "calibration": _native(build, "calibrate", dim, width, rows, k, cpu)
                          | {"thousand_iterations": k}}
        out.append(row)
    return out


def python_gc(build):
    from elpis.ECS_G.native import ECSGLibrary, Executor
    from array import array
    api = ECSGLibrary(ctypes.CDLL(str(library(build))))
    out = []
    for kind, steps in (("forward_into", 0), ("learn", 10)):
        w = {"dim": 6, "width": 36, "rows": 64, "steps": steps}
        w0, x, y = data(w, api)
        e = Executor.create(api, 6, 36, w0)
        xb, yb, ob = array("d", [v for r in x for v in r]), array("d", y), array("d", bytes(8 * 64))
        op = (lambda: e.forward_into(xb, ob)) if kind == "forward_into" else (lambda: e.learn(xb, yb, 0.002, 10))
        row = {"operation": f"Executor.{kind}", "steps": steps}
        for label, enabled in (("gc_enabled", True), ("gc_disabled", False)):
            (gc.enable if enabled else gc.disable)()
            try:
                r = time_python(op)
            finally:
                gc.enable()
            row[label] = r | {"p99_over_p50": round(r["p99_ns"] / r["p50_ns"], 3)}
        e.close()
        out.append(row)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research.ecs_runtime_r1.jitter")
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    build = args.build.resolve()
    report = {"label": "PERFORMANCE_ONLY NO_SCIENTIFIC_CLAIM", "binding": binding(build),
              "environment": environment(), "native": native_pairs(build), "python": python_gc(build)}
    if args.out:
        with open(args.out, "x", encoding="ascii") as fh:  # evidence is written once
            json.dump(report, fh, indent=1, sort_keys=True)
            fh.write("\n")
    else:
        json.dump(report, sys.stdout, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
