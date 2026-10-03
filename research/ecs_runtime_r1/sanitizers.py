"""Runtime R1 sanitizer qualification of the executor. PERFORMANCE_ONLY (engineering evidence).

    PYTHONPATH=src python -m research.ecs_runtime_r1.sanitizers --work <scratch dir> [--out <evidence.json>]

Builds the tree twice with GCC in Debug, once with AddressSanitizer +
UndefinedBehaviorSanitizer (recovery off) and once with ThreadSanitizer, and
runs every ECS_G native test in each (ctest -R '^ECS_G\\.'). Under ASan+UBSan
it also runs the Python executor binding and differential suites against the
instrumented library (libasan preloaded, leak checking off for the
interpreter). ThreadSanitizer covers the one concurrent scenario the
executor admits: overlapping entry refused with BUSY.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from .bench import REPO, _git, sha256

CONFIGS = {"gcc-asan-ubsan": ["-DELPIS_ENABLE_ASAN=ON", "-DELPIS_ENABLE_UBSAN=ON"],
           "gcc-tsan": ["-DELPIS_ENABLE_TSAN=ON"]}
PYTHON_SUITES = ("tests/ECS_G/test_executor_binding.py", "tests/ECS_G/test_executor_differential.py")
# Spawns a fresh interpreter without the preloaded ASan runtime, which an instrumented library cannot load.
PYTHON_DESELECT = ("tests/ECS_G/test_executor_binding.py::test_a_clean_process_runs_the_executor_with_no_model_or_numpy",)


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def qualify(work: Path, name: str, flags: list) -> dict:
    build = work / name
    cfg = _run(["cmake", "-S", str(REPO), "-B", str(build), "-DCMAKE_BUILD_TYPE=Debug", "-DCMAKE_C_COMPILER=gcc",
                "-DCMAKE_CXX_COMPILER=g++", *flags])
    if cfg.returncode:
        raise SystemExit(cfg.stderr)
    targets = ["elpis_ecsg_math", "test_ecsg_math", "test_ecsg_state", "test_ecsg_snapshot", "test_ecsg_executor",
               "test_ecsg_executor_baseline", "test_ecsg_executor_txn", "test_ecsg_executor_alloc"]
    made = _run(["cmake", "--build", str(build), "-j4", "--target", *targets])
    if made.returncode:
        raise SystemExit(made.stdout[-4000:] + made.stderr[-4000:])
    ctest = _run(["ctest", "-R", r"^ECS_G\.", "--output-on-failure"], cwd=build)
    tests = {}
    for line in ctest.stdout.splitlines():
        m = re.match(r"\s*\d+/\d+ Test\s+#\d+: ECS_G\.(\S+) \.*\s*(.*)$", line)
        if m:
            tests[m.group(1)] = "Passed" if m.group(2).startswith("Passed") else "FAILED: " + m.group(2).strip()
    report = {"flags": flags, "compiler": _run(["gcc", "--version"]).stdout.splitlines()[0],
              "ctest_returncode": ctest.returncode, "tests": tests,
              "sanitizer_reports": [ln for ln in (ctest.stdout + ctest.stderr).splitlines()
                                    if "Sanitizer" in ln and ("ERROR" in ln or "WARNING" in ln)]}
    if name == "gcc-asan-ubsan":
        libasan = _run(["gcc", "-print-file-name=libasan.so"]).stdout.strip()
        env = dict(os.environ, LD_PRELOAD=libasan, ASAN_OPTIONS="detect_leaks=0:abort_on_error=1",
                   UBSAN_OPTIONS="halt_on_error=1", ELPIS_NATIVE_BUILD=str(build), ELPIS_REQUIRE_NATIVE="1",
                   PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}")
        deselect = [arg for test in PYTHON_DESELECT for arg in ("--deselect", test)]
        junit = build / "python-under-asan.xml"
        py = _run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}", *deselect,
                   *PYTHON_SUITES], cwd=REPO, env=env)
        suite = ET.parse(junit).getroot()
        suite = suite.find("testsuite") if suite.tag == "testsuites" else suite
        tail = {k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
        report["python_under_asan"] = {"suites": list(PYTHON_SUITES), "deselected": list(PYTHON_DESELECT),
                                       "returncode": py.returncode, "summary": tail,
                                       "sanitizer_reports": [ln for ln in (py.stdout + py.stderr).splitlines()
                                                             if "Sanitizer" in ln and "ERROR" in ln]}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research.ecs_runtime_r1.sanitizers")
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    sources = sorted((REPO / "native" / "ECS_G").glob("src/*.c")) + sorted((REPO / "native" / "ECS_G").glob("tests/*.c"))
    report = {"label": "ENGINEERING_EVIDENCE NO_SCIENTIFIC_CLAIM", "head": _git("rev-parse", "HEAD"),
              "dirty": bool(_git("status", "--porcelain", "--", "native", "src", "tests/ECS_G")),
              "sources": {str(p.relative_to(REPO)): sha256(p) for p in sources},
              "configs": {name: qualify(args.work.resolve(), name, flags) for name, flags in CONFIGS.items()}}
    text = json.dumps(report, indent=1, sort_keys=True) + "\n"
    if args.out:
        with open(args.out, "x", encoding="ascii") as fh:  # evidence is written once
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
