"""Runtime R1 benchmark harness. PERFORMANCE_ONLY. NO_SCIENTIFIC_CLAIM.

    PYTHONPATH=src python -m research.ecs_runtime_r1.bench --build <cmake build dir> \
        --phase baseline|final [--out <evidence.json>] [--only <workload id substring>]

Every measurement: explicit warmup, then individually timed samples; min, p50,
p95, p99, max and throughput are reported, plus the first (cold) call. Native
modes run the C driver (``ecsg_runtime_bench``); Python modes time the public
binding. The evidence binds the exact sources, build configuration and
shared-library SHA-256 it measured. Timings are performance evidence only.
"""
from __future__ import annotations

import argparse
from array import array
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

from . import workloads as WL

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent / "evidence"
SOURCES = ("src/elpis/ECS_G/native.py", "src/elpis/ECS_G/cognition.py", "src/elpis/runtime/cognition.py",
           "native/ECS_G/CMakeLists.txt", "native/elpis.cmake", "CMakeLists.txt",
           "research/ecs_runtime_r1/bench.py", "research/ecs_runtime_r1/workloads.py",
           "research/ecs_runtime_r1/native/ecsg_runtime_bench.c")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def binding(build: Path) -> dict:
    native = sorted((REPO / "native" / "ECS_G").glob("src/*.c")) + sorted((REPO / "native" / "ECS_G").glob("include/elpis/*.h"))
    files = {str(p.relative_to(REPO)): sha256(p) for p in native}
    files.update({name: sha256(REPO / name) for name in SOURCES if (REPO / name).is_file()})
    cache = {}
    cache_path = build / "CMakeCache.txt"
    if cache_path.is_file():
        for line in cache_path.read_text(errors="replace").splitlines():
            for key in ("CMAKE_BUILD_TYPE", "CMAKE_C_COMPILER", "CMAKE_C_FLAGS", "CMAKE_C_FLAGS_RELEASE",
                        "ELPIS_ENABLE_ASAN", "ELPIS_ENABLE_UBSAN"):
                if line.startswith(key + ":"):
                    cache[key] = line.split("=", 1)[1]
    compiler = cache.get("CMAKE_C_COMPILER", "cc")
    try:
        version = subprocess.run([compiler, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except (OSError, IndexError):
        version = "unknown"
    return {"head": _git("rev-parse", "HEAD"), "tree": _git("rev-parse", "HEAD^{tree}"),
            "dirty": bool(_git("status", "--porcelain", "--", "src", "native", "research/ecs_runtime_r1",
                               ":(exclude)research/ecs_runtime_r1/evidence")),
            "sources": files, "cmake": cache, "compiler": version,
            "library": {"path": str(library(build).relative_to(build)), "sha256": sha256(library(build))},
            "bench_driver": {"sha256": sha256(driver(build))}}


def environment() -> dict:
    model, flags = "unknown", []
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name") and model == "unknown":
                model = line.split(":", 1)[1].strip()
            if line.startswith("flags") and not flags:
                have = set(line.split(":", 1)[1].split())
                flags = sorted(have & {"sse2", "sse4_2", "avx", "avx2", "fma", "avx512f"})
    except OSError:
        pass
    return {"machine": platform.machine(), "cpu": model, "cpu_flags": flags, "cpus": os.cpu_count(),
            "python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform()}


def library(build: Path) -> Path:
    found = sorted(Path(build).rglob("libelpis_ecsg_math.so"))
    if not found:
        raise SystemExit("libelpis_ecsg_math.so not found in the build")
    return found[0]


def driver(build: Path) -> Path:
    found = sorted(Path(build).rglob("ecsg_runtime_bench"))
    found = [p for p in found if p.is_file() and os.access(p, os.X_OK)]
    if not found:
        raise SystemExit("ecsg_runtime_bench not built in this build tree")
    return found[0]


def summarize(samples_ns, first_ns, warmup) -> dict:
    s = sorted(samples_ns)
    n = len(s)

    def pct(q):
        return s[min(n - 1, max(0, math.ceil(q * n) - 1))]
    p50 = pct(0.50)
    return {"samples": n, "warmup": warmup, "first_call_ns": first_ns, "min_ns": s[0], "p50_ns": p50,
            "p95_ns": pct(0.95), "p99_ns": pct(0.99), "max_ns": s[-1], "mean_ns": sum(s) / n,
            "throughput_per_s": 1e9 / p50 if p50 else None}


def time_python(op) -> dict:
    t0 = time.perf_counter_ns()
    op()
    first = time.perf_counter_ns() - t0
    t0 = time.perf_counter_ns()
    for _ in range(3):
        op()
    estimate = max(1.0, (time.perf_counter_ns() - t0) / 3)
    n = int(min(WL.MAX_SAMPLES, max(WL.MIN_SAMPLES, WL.BUDGET_SECONDS * 1e9 / estimate)))
    warmup = int(min(WL.MAX_WARMUP, max(WL.MIN_WARMUP, 0.2 * WL.BUDGET_SECONDS * 1e9 / estimate)))
    for _ in range(warmup):
        op()
    samples = []
    for _ in range(n):
        t = time.perf_counter_ns()
        op()
        samples.append(time.perf_counter_ns() - t)
    return summarize(samples, first, warmup)


def time_native(build: Path, mode: str, w: dict) -> dict:
    out = subprocess.run([str(driver(build)), mode, str(w["dim"]), str(w["width"]), str(w["rows"]),
                          str(max(1, w["steps"])), str(WL.BUDGET_SECONDS)], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"{mode} failed: {out.stderr}")
    raw = json.loads(out.stdout)
    p50 = raw["p50_ns"]
    return {"samples": raw["samples"], "warmup": raw["warmup"], "first_call_ns": raw["first_call_ns"],
            "min_ns": raw["min_ns"], "p50_ns": p50, "p95_ns": raw["p95_ns"], "p99_ns": raw["p99_ns"],
            "max_ns": raw["max_ns"], "mean_ns": raw["mean_ns"], "throughput_per_s": 1e9 / p50 if p50 else None}


def data(w: dict, api):
    """Deterministic workload data (stdlib RNG); y = 0.9 f_W0(X) by the reference forward map."""
    from elpis.ECS_G.native import WorldState
    rng = random.Random(f"{WL.SEED}-{w['dim']}-{w['width']}-{w['rows']}")
    ws = WL.W_SCALE_AT_36 * math.sqrt(36.0 / w["width"])
    w0 = [rng.gauss(0.0, ws) for _ in range(w["dim"] * w["width"])]
    x = [[rng.gauss(0.0, WL.X_SCALE) for _ in range(w["dim"])] for _ in range(w["rows"])]
    with WorldState.create(api, w["dim"], w["width"], w0) as ref:
        y = [WL.TARGET_FACTOR * v for v in ref.forward(x)]
    return w0, x, y


def python_modes(phase: str) -> dict:
    """Mode name -> (kind, factory(api, w, data) returning a zero-argument op).

    ``baseline`` is the pre-nativeization binding (fork/adopt, per-step Python),
    measured at R1A 3568a77 and only runnable on that binding; ``final`` is
    the executor control plane.
    """
    from elpis.ECS_G.cognition import CognitiveCore
    from elpis.ECS_G import native
    WorldState = native.WorldState
    lr = WL.LEARNING_RATE

    def core(api, w, d):
        return CognitiveCore.create(api, w["dim"], w["width"], d[0], learning_rate=lr)

    def core_query(api, w, d):
        c = core(api, w, d)
        return lambda: c.query(d[1])

    def core_learn(api, w, d):
        c = core(api, w, d)
        return lambda: c.learn(d[1], d[2], steps=w["steps"])

    if phase == "baseline":
        def state_forward(api, w, d):
            s = WorldState.create(api, w["dim"], w["width"], d[0])
            return lambda: s.forward(d[1])

        def state_step(api, w, d):
            s = WorldState.create(api, w["dim"], w["width"], d[0])
            return lambda: s.step(d[1], d[2], lr)

        def snapshot(api, w, d):
            s = WorldState.create(api, w["dim"], w["width"], d[0])
            return lambda: s.snapshot()

        def fork_adopt(api, w, d):
            s = WorldState.create(api, w["dim"], w["width"], d[0])
            return lambda: s.adopt(s.fork())

        def restore(api, w, d):
            with WorldState.create(api, w["dim"], w["width"], d[0]) as s:
                blob = s.snapshot()
            return lambda: WorldState.restore(api, blob).close()
        return {"py_state_forward": ("query", state_forward), "py_core_query": ("query", core_query),
                "py_state_step": ("step", state_step), "py_core_learn": ("learn", core_learn),
                "py_state_snapshot": ("state", snapshot), "py_state_fork_adopt": ("state", fork_adopt),
                "py_state_restore": ("state", restore)}

    Executor = native.Executor

    def buffers(d):
        return array("d", [v for row in d[1] for v in row]), array("d", d[2]), array("d", bytes(8 * len(d[2])))

    def executor(api, w, d):
        return Executor.create(api, w["dim"], w["width"], d[0], max_rows=max(w["rows"], native.DEFAULT_MAX_ROWS))

    def exec_forward(api, w, d):
        e = executor(api, w, d)
        return lambda: e.forward(d[1])

    def exec_forward_into(api, w, d):
        e, (x, _, out) = executor(api, w, d), buffers(d)
        return lambda: e.forward_into(x, out)

    def core_query_into(api, w, d):
        c, (x, _, out) = core(api, w, d), buffers(d)
        return lambda: c.query_into(x, out)

    def exec_learn(api, w, d):
        e, (x, y, _) = executor(api, w, d), buffers(d)
        return lambda: e.learn(x, y, lr, w["steps"])

    def core_learn_noreceipt(api, w, d):
        c = core(api, w, d)
        return lambda: c.learn(d[1], d[2], steps=w["steps"], receipt=False)

    def core_learn_buffer(api, w, d):
        c, (x, y, _) = core(api, w, d), buffers(d)
        return lambda: c.learn(x, y, steps=w["steps"], receipt=False)

    def exec_snapshot(api, w, d):
        e = executor(api, w, d)
        return lambda: e.snapshot()

    def exec_txn(api, w, d):
        e, (x, y, _) = executor(api, w, d), buffers(d)

        def op():
            with e.transaction() as txn:
                txn.learn(x, y, lr)
                txn.commit()
        return op

    def exec_restore(api, w, d):
        with executor(api, w, d) as e:
            blob = e.snapshot()
        return lambda: Executor.restore(api, blob).close()
    return {"py_exec_forward": ("query", exec_forward), "py_exec_forward_into": ("query", exec_forward_into),
            "py_core_query": ("query", core_query), "py_core_query_into": ("query", core_query_into),
            "py_exec_learn": ("learn", exec_learn), "py_core_learn": ("learn", core_learn),
            "py_core_learn_noreceipt": ("learn", core_learn_noreceipt),
            "py_core_learn_buffer": ("learn", core_learn_buffer),
            "py_exec_snapshot": ("state", exec_snapshot), "py_exec_txn": ("state", exec_txn),
            "py_exec_restore": ("state", exec_restore)}


NATIVE_MODES = {"baseline": {"query": ["ref_forward"], "step": ["ref_step"], "learn": ["ref_learn"],
                             "state": ["ref_cold"]},
                "final": {"query": ["ref_forward", "exec_forward"], "step": ["ref_step"],
                          "learn": ["ref_learn", "exec_learn", "exec_txn"], "state": ["ref_cold", "exec_cold"]}}


def memory(api, w: dict) -> dict:
    """Bytes owned per state: the reference (W, plus scratch the old binding allocated per Python step) and the
    executor workspace (allocated once at creation)."""
    d, n, r = w["dim"], w["width"], w["rows"]
    out = {"workload": w["id"], "w_bytes": 8 * d * n, "reference_step_scratch_bytes": 8 * (r * n + r + d * n)}
    if hasattr(api, "workspace_bytes"):
        from elpis.ECS_G.native import DEFAULT_MAX_ROWS
        out["executor_workspace_bytes"] = api.workspace_bytes(d, n, r)
        out["executor_default_workspace_bytes"] = api.workspace_bytes(d, n, max(r, DEFAULT_MAX_ROWS))
    return out


def _progress(result: dict) -> None:
    print(f"{result['workload']:28s} {result['mode']:24s} p50={result['p50_ns'] / 1e3:10.1f}us", file=sys.stderr,
          flush=True)


def run(build: Path, phase: str, only: str | None) -> dict:
    from elpis.ECS_G.native import ECSGLibrary
    api = ECSGLibrary(ctypes.CDLL(str(library(build))))
    results = []
    natives = NATIVE_MODES[phase]
    pymodes = python_modes(phase)
    for w in WL.workloads():
        if only and only not in w["id"]:
            continue
        kinds = [w["kind"]] + (["step"] if w["kind"] == "learn" and w["steps"] == 1 else []) + (
            ["state"] if w["id"] == "r0-query-R64" else [])
        d = data(w, api)
        for kind in kinds:
            for mode in natives.get(kind, []):
                results.append({"workload": w["id"], "scope": w["scope"], "mode": mode, "plane": "native-direct",
                                **time_native(build, mode, w)})
                _progress(results[-1])
            for mode, (mkind, factory) in pymodes.items():
                if mkind != kind:
                    continue
                results.append({"workload": w["id"], "scope": w["scope"], "mode": mode, "plane": "python",
                                **time_python(factory(api, w, d))})
                _progress(results[-1])
    return {"label": "PERFORMANCE_ONLY NO_SCIENTIFIC_CLAIM", "phase": phase, "binding": binding(build),
            "environment": environment(), "workloads": WL.workloads(), "gates": WL.GATES,
            "thresholds": WL.THRESHOLDS, "results": results,
            "memory": [memory(api, w) for w in WL.workloads() if not only or only in w["id"]]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research.ecs_runtime_r1.bench")
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=sorted(NATIVE_MODES))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--only")
    args = parser.parse_args(argv)
    report = run(args.build.resolve(), args.phase, args.only)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "x", encoding="ascii") as fh:  # evidence is written once
            json.dump(report, fh, indent=1, sort_keys=True)
            fh.write("\n")
    else:
        json.dump(report, sys.stdout, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
