"""Runtime R1 Python control-plane overhead by split timing (supplementary). PERFORMANCE_ONLY. NO_SCIENTIFIC_CLAIM.

    PYTHONPATH=src python -m research.ecs_runtime_r1.overhead --build <build dir> [--out <evidence.json>]

The registered estimator (workloads.GATES["python_overhead_k_independent"])
subtracts two p50s measured in different processes. At large K one call lasts
tens of milliseconds, and drift and per-call variation on a shared host
(a few percent, i.e. hundreds of microseconds) swamp a control plane that
costs microseconds; adjacent process pairs did not remove it either.

This supplement measures the control plane directly, inside each call: the
native learn entry point is wrapped so that every call records the time
spent in native code, and the overhead of one Python call is its total time
minus that native time. Native variation sits in both terms and cancels.
The wrapper's own cost (two clock reads and a Python call) is measured and
reported. For the audited CognitiveCore.learn the receipt work (experience
digest and two snapshot reads) is overhead by this definition.

A probe separates "more work" from "colder machine": the identical K=1 call
is timed back to back and right after an idle gap (sleep) of 1 ms and 27 ms,
with the Python time split into before and after the native call. Python
code that runs after a long gap, idle or native, runs on cold caches,
predictors and clocks.
"""
from __future__ import annotations

from array import array
import argparse
import ctypes
import json
from pathlib import Path
import statistics
import sys
import time

from . import workloads as WL
from .bench import binding, data, environment, library

MODES = ("py_exec_learn", "py_core_learn_buffer", "py_core_learn_noreceipt", "py_core_learn")
CALLS = {1: 400, 10: 400, 100: 200, 1000: 60, 4000: 30}


class _Split:
    """Times every call of one native entry point."""

    def __init__(self, fn):
        self.fn, self.inner = fn, []

    def __call__(self, *args):
        t = time.perf_counter_ns()
        rc = self.fn(*args)
        self.inner.append(time.perf_counter_ns() - t)
        return rc


def _wrapper_cost(calls=20000):
    noop = _Split(lambda *a: 0)
    t = time.perf_counter_ns()
    for _ in range(calls):
        noop(1, 2)
    total = (time.perf_counter_ns() - t) / calls
    return {"wrapped_noop_call_ns": total, "inner_measured_ns": statistics.median(noop.inner)}


def cold_probe(api, w0, xb, yb, calls=60):
    """Python time before/after one K=1 native learn, back to back and after idle gaps."""
    from elpis.ECS_G.native import Executor
    real, marks = api._x.learn, {}

    def wrapped(*args):
        marks["enter"] = time.perf_counter_ns()
        rc = real(*args)
        marks["exit"] = time.perf_counter_ns()
        return rc
    out = {}
    api._x.learn = wrapped
    try:
        with Executor.create(api, 6, 36, w0) as e:
            for label, gap in (("back_to_back", 0.0), ("after_1ms_idle", 0.001), ("after_27ms_idle", 0.027)):
                pre, post = [], []
                for _ in range(calls):
                    if gap:
                        time.sleep(gap)
                    t0 = time.perf_counter_ns()
                    e.learn(xb, yb, WL.LEARNING_RATE, 1)
                    t1 = time.perf_counter_ns()
                    pre.append(marks["enter"] - t0)
                    post.append(t1 - marks["exit"])
                out[label] = {"before_native_p50_ns": statistics.median(pre),
                              "after_native_p50_ns": statistics.median(post)}
    finally:
        api._x.learn = real
    return out


def measure(build: Path) -> dict:
    from elpis.ECS_G.cognition import CognitiveCore
    from elpis.ECS_G.native import ECSGLibrary, Executor
    api = ECSGLibrary(ctypes.CDLL(str(library(build))))
    real = api._x.learn
    out = {}
    for k in WL.R0_STEPS:
        w = {"dim": 6, "width": 36, "rows": 64, "steps": k}
        w0, x, y = data(w, api)
        xb, yb = array("d", [v for row in x for v in row]), array("d", y)
        rows = {}
        for mode in MODES:
            e = Executor.create(api, 6, 36, w0)
            core = CognitiveCore(e, learning_rate=WL.LEARNING_RATE)
            op = {"py_exec_learn": lambda: e.learn(xb, yb, WL.LEARNING_RATE, k),
                  "py_core_learn_buffer": lambda: core.learn(xb, yb, steps=k, receipt=False),
                  "py_core_learn_noreceipt": lambda: core.learn(x, y, steps=k, receipt=False),
                  "py_core_learn": lambda: core.learn(x, y, steps=k)}[mode]
            op()  # warm
            split = _Split(real)
            api._x.learn = split
            totals = []
            try:
                for _ in range(CALLS[k]):
                    t = time.perf_counter_ns()
                    op()
                    totals.append(time.perf_counter_ns() - t)
            finally:
                api._x.learn = real
            assert len(split.inner) == len(totals), "exactly one native learn call per Python call"
            overhead = [a - b for a, b in zip(totals, split.inner)]
            rows[mode] = {"calls": len(totals), "native_calls_per_call": len(split.inner) / len(totals),
                          "total_p50_ns": statistics.median(totals), "native_p50_ns": statistics.median(split.inner),
                          "overhead_p50_ns": statistics.median(overhead),
                          "overhead_p95_ns": sorted(overhead)[int(0.95 * (len(overhead) - 1))]}
            core.close()
        out[f"r0-learn-K{k}"] = rows
    t = WL.THRESHOLDS
    verdict = {}
    for mode in MODES:
        o10 = out["r0-learn-K10"][mode]["overhead_p50_ns"]
        o4000 = out["r0-learn-K4000"][mode]["overhead_p50_ns"]
        limit = t["overhead_factor"] * o10 + t["overhead_slack_us"] * 1e3
        verdict[mode] = {"overhead_K10_ns": o10, "overhead_K4000_ns": o4000, "limit_ns": limit,
                         "pass": o4000 <= limit}
    w0, x, y = data({"dim": 6, "width": 36, "rows": 64, "steps": 1}, api)
    probe = cold_probe(api, w0, array("d", [v for row in x for v in row]), array("d", y))
    return {"method": "split timing: total Python call minus time inside the native learn call, per call",
            "wrapper": _wrapper_cost(), "per_workload": out, "registered_rule_on_split_timing": verdict,
            "cold_probe": probe}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research.ecs_runtime_r1.overhead")
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    build = args.build.resolve()
    report = {"label": "PERFORMANCE_ONLY NO_SCIENTIFIC_CLAIM SUPPLEMENTARY", "binding": binding(build),
              "environment": environment(), **measure(build)}
    if args.out:
        with open(args.out, "x", encoding="ascii") as fh:  # evidence is written once
            json.dump(report, fh, indent=1, sort_keys=True)
            fh.write("\n")
    else:
        json.dump(report, sys.stdout, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
