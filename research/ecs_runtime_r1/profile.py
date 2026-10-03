"""Runtime R1 native profile (callgrind instruction counts). PERFORMANCE_ONLY. NO_SCIENTIFIC_CLAIM.

    python -m research.ecs_runtime_r1.profile --build <build dir> --label <name>

Runs the direct-native driver's ``exec_learn`` at the R0 shape (d=6, N=36,
R=64, K=100) under callgrind and summarizes instructions per step and the
hottest source lines of the executor. Instruction counts are deterministic
for a given binary; they complement, and do not replace, wall-clock evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from .bench import driver, library, sha256

SHAPE = {"dim": 6, "width": 36, "rows": 64, "steps": 100}
_LINE = re.compile(r"^\s*([\d,]+) \(\s*([\d.]+)%\)\s{2,}(\S.*)$")


def record(build: Path, out: Path) -> None:
    subprocess.run(["valgrind", "--tool=callgrind", f"--callgrind-out-file={out}", str(driver(build)), "exec_learn",
                    str(SHAPE["dim"]), str(SHAPE["width"]), str(SHAPE["rows"]), str(SHAPE["steps"]), "0.05"],
                   check=True, capture_output=True)


def summarize(cg: Path, top: int = 12) -> dict:
    """Total instructions, executor calls, per-function and hottest-line instruction counts."""
    text = subprocess.run(["callgrind_annotate", "--auto=yes", "--threshold=100", str(cg)], check=True,
                          capture_output=True, text=True).stdout
    total = int(re.search(r"([\d,]+) \(100.0%\)\s+PROGRAM TOTALS", text).group(1).replace(",", ""))
    calls = int(re.search(r"=> \S*ecsg_executor\.c:learn_into\S* \((\d+)x\)", text).group(1))
    functions, lines = {}, []
    for raw in text.splitlines():
        m = _LINE.match(raw)
        if not m or "=>" in raw:
            continue
        count, body = int(m.group(1).replace(",", "")), m.group(3)
        fn = re.match(r"\S*/ecsg_executor\.c:(\S+)", body)
        if fn:
            functions[fn.group(1)] = count
        elif not body.startswith(("/", "???", "PROGRAM", "events")):
            lines.append((count, body.strip()))
    lines.sort(reverse=True)
    steps = calls * SHAPE["steps"]
    return {"total_instructions": total, "learn_calls": calls, "steps": steps,
            "instructions_per_step": round(total / steps), "functions": functions,
            "hottest_lines": [{"instructions": c, "share": round(c / total, 4), "source": s} for c, s in lines[:top]]}


def _cache(build: Path) -> dict:
    keys = ("CMAKE_HOME_DIRECTORY", "CMAKE_BUILD_TYPE", "CMAKE_C_COMPILER", "CMAKE_C_FLAGS_RELWITHDEBINFO",
            "CMAKE_C_FLAGS_RELEASE")
    out = {}
    for line in (build / "CMakeCache.txt").read_text(errors="replace").splitlines():
        for key in keys:
            if line.startswith(key + ":"):
                out[key] = line.split("=", 1)[1]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research.ecs_runtime_r1.profile")
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args(argv)
    build = args.build.resolve()
    cg = Path(tempfile.mkdtemp()) / "exec_learn.cg"
    record(build, cg)
    cache = _cache(build)
    source = Path(cache.pop("CMAKE_HOME_DIRECTORY"))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "native"], cwd=source, capture_output=True,
                                text=True).stdout.strip())
    report = {"label": args.label, "shape": SHAPE, "tool": "callgrind (valgrind 3.22)", "source_head": head,
              "native_tree_dirty": dirty,
              "executor_source_sha256": sha256(source / "native/ECS_G/src/ecsg_executor.c"),
              "library_sha256": sha256(library(build)), "driver_sha256": sha256(driver(build)), "cmake": cache,
              **summarize(cg)}
    json.dump(report, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
