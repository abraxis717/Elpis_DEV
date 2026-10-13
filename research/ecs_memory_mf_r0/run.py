"""Runner of the ECS Memory MF-R0 laboratory (PREREGISTRATION.md). Standard library only; run with `python -I`.

    python -I research/ecs_memory_mf_r0/run.py dev    --build BUILD   DEV seeds 1-4 (repeatable until freeze)
    python -I research/ecs_memory_mf_r0/run.py freeze                 bind the sources by SHA-256 (write once)
    python -I research/ecs_memory_mf_r0/run.py qual   --build BUILD   QUAL seeds 1001-1008 (once, from frozen sources)

Each phase writes one record (evidence/<phase>/RECORD.json): the laboratory output of every seed, the frozen
scoring of it, and the exact source, binary, compiler, build type, host and git identity it came from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]
sys.path.insert(0, str(LAB))
from score import score  # noqa: E402

EVIDENCE = LAB / "evidence"
FREEZE = EVIDENCE / "FREEZE.json"
SEEDS = {"dev": [1, 2, 3, 4], "qual": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]}
SOURCES = sorted(
    [
        "research/ecs_memory_mf_r0/PREREGISTRATION.md",
        "research/ecs_memory_mf_r0/CMakeLists.txt",
        "research/ecs_memory_mf_r0/run.py",
        "research/ecs_memory_mf_r0/score.py",
        "research/ecs_memory_mf_r0/lab/ecs_mem_lab.c",
        "native/substrate/src/sha256.c",
        "native/substrate/include/elpis/sha256.h",
    ]
    + [str(p.relative_to(REPO)) for p in (LAB / "native" / "src").glob("*.c")]
    + [str(p.relative_to(REPO)) for p in (LAB / "native" / "include" / "elpis").glob("*.h")]
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def seal(record: dict) -> dict:
    record = dict(record)
    record.pop("record_sha256", None)
    record["record_sha256"] = hashlib.sha256(canonical(record)).hexdigest()
    return record


def source_digests() -> dict:
    return {s: sha256(REPO / s) for s in SOURCES}


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def build_identity(build: Path) -> dict:
    cache = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if ":" in line and "=" in line and not line.startswith(("//", "#")):
            key, _, value = line.partition("=")
            cache[key.split(":")[0]] = value
    compiler = cache.get("CMAKE_C_COMPILER", "")
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    binary = build / "research" / "ecs_memory_mf_r0" / "ecs_mem_lab"
    return {
        "binary": str(binary.relative_to(build)),
        "binary_sha256": sha256(binary),
        "compiler": version,
        "build_type": cache.get("CMAKE_BUILD_TYPE", ""),
        "sanitizers": {k: cache.get(k, "OFF") for k in ("ELPIS_ENABLE_ASAN", "ELPIS_ENABLE_UBSAN", "ELPIS_ENABLE_TSAN")},
    }


def host_identity() -> dict:
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"machine": platform.machine(), "system": platform.system(), "release": platform.release(), "cpu": cpu,
            "python": platform.python_version()}


def run_seeds(binary: Path, seeds) -> list:
    docs = []
    for seed in seeds:
        out = subprocess.run([str(binary), "--seed", str(seed)], check=True, capture_output=True, text=True).stdout
        docs.append(json.loads(out))
    return docs


def phase(name: str, build: Path) -> None:
    out = EVIDENCE / name / "RECORD.json"
    if name == "dev" and FREEZE.exists():
        sys.exit("refused: the laboratory is frozen; DEV is closed")
    if name == "qual":
        if not FREEZE.exists():
            sys.exit("refused: QUAL needs the freeze manifest")
        frozen = json.loads(FREEZE.read_text())
        if frozen["sources"] != source_digests():
            sys.exit("refused: sources differ from the freeze manifest")
        if out.exists():
            sys.exit("refused: QUAL runs once; its record exists")
    if git("status", "--porcelain", "--", *SOURCES):
        sys.exit("refused: bound sources have uncommitted changes")
    identity = build_identity(build)
    docs = run_seeds(build / identity["binary"], SEEDS[name])
    rows = [row for doc in docs for row in doc["rows"]]
    record = {
        "schema": "elpis.ecs-mem-lab.record.v1",
        "phase": name.upper(),
        "seeds": SEEDS[name],
        "git_head": git("rev-parse", "HEAD"),
        "sources": source_digests(),
        "freeze_sha256": sha256(FREEZE) if name == "qual" else None,
        "build": identity,
        "host": host_identity(),
        "parameters": docs[0]["parameters"],
        "runs": docs,
        "scored": score(rows),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(seal(record), indent=1, sort_keys=True) + "\n")
    for h, v in record["scored"]["verdicts"].items():
        print(h, v["verdict"], json.dumps(v.get("values", {})))


def freeze() -> None:
    if FREEZE.exists():
        sys.exit("refused: the freeze manifest is write-once")
    if git("status", "--porcelain", "--", *SOURCES):
        sys.exit("refused: bound sources have uncommitted changes")
    dev = EVIDENCE / "dev" / "RECORD.json"
    if not dev.exists():
        sys.exit("refused: freeze follows DEV")
    manifest = {
        "schema": "elpis.ecs-mem-lab.freeze.v1",
        "git_head": git("rev-parse", "HEAD"),
        "sources": source_digests(),
        "dev_record_sha256": sha256(dev),
        "qual_seeds": SEEDS["qual"],
    }
    FREEZE.parent.mkdir(parents=True, exist_ok=True)
    FREEZE.write_text(json.dumps(seal(manifest), indent=1, sort_keys=True) + "\n")
    print("frozen", sha256(FREEZE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("dev", "freeze", "qual"))
    parser.add_argument("--build", type=Path, default=REPO / "build")
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    else:
        phase(args.phase, args.build.resolve())


if __name__ == "__main__":
    main()
