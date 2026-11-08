"""Evaluation runs and evidence records.

A run evaluates named arms on (partition, router seeds, regimes, contract) and returns per-seed episode
means. Seeds are evaluated in parallel processes, but results are assembled by seed index, so the record
does not depend on scheduling. A record binds its numbers to the SHA-256 of the lab's source files, the
benchmark spec and the frozen coefficients it used.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from . import arms, harness

LAB = Path(__file__).resolve().parents[2]
SRC = LAB / "src" / "cgc_lao_lab"
SPEC = LAB / "specs" / "slb1.spec.json"


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest() -> str:
    """SHA-256 over the lab's Python sources (sorted relative path, NUL, bytes, NUL)."""
    h = hashlib.sha256()
    for path in sorted(SRC.glob("*.py")):
        h.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def source_file_digests() -> dict:
    """Per-file SHA-256 of the lab's Python sources (a freeze binds every one of them)."""
    return {path.name: file_digest(path) for path in sorted(SRC.glob("*.py"))}


def coefficient_digests() -> dict:
    return {p.name: file_digest(p) for p in sorted((LAB / "coefficients").glob("*.json"))}


def environment() -> dict:
    cpu = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            cpu = next((line.split(":", 1)[1].strip() for line in fh if line.startswith("model name")), "")
    except OSError:
        pass
    return {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(),
            "cpu": cpu, "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "")}


def _worker(task):
    partition, seed, regimes, arm_specs, contract, episodes, chunk = task
    facs = {label: arms.factory(name, **kw) for label, (name, kw) in arm_specs.items()}
    return seed, harness.evaluate_seed(partition, seed, regimes, facs, contract, episodes, chunk)


def run(partition: str, seeds, regimes, arm_specs: dict, contract: str = "ORIGINAL", episodes: int = 2048,
        workers: int = 4, chunk: int = 2048) -> dict:
    """``arm_specs``: label -> (registered arm name, factory overrides)."""
    seeds = list(seeds)
    tasks = [(partition, s, list(regimes), arm_specs, contract, episodes, chunk) for s in seeds]
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=get_context("fork")) as ex:
            by_seed = dict(ex.map(_worker, tasks))
    else:
        by_seed = dict(map(_worker, tasks))
    results: dict = {}
    for rname in regimes:
        for label in arm_specs:
            metrics = by_seed[seeds[0]][rname].get(label)
            if not metrics:
                continue
            results.setdefault(rname, {})[label] = {
                m: [float(f"{by_seed[s][rname][label][m]:.10g}") for s in seeds] for m in metrics}
    return {"partition": partition, "seeds": seeds, "episodes_per_seed": episodes, "contract": contract,
            "regimes": list(regimes), "arms": {k: [v[0], v[1]] for k, v in arm_specs.items()},
            "results": results, "seconds": round(time.time() - t0, 1)}


def seal(record: dict, phase: str, note: str = "") -> dict:
    return {"schema": "cgc-lao-solver-lab.record.v1", "phase": phase, "note": note,
            "source_sha256": source_digest(), "source_files_sha256": source_file_digests(),
            "spec_sha256": file_digest(SPEC),
            "coefficients_sha256": coefficient_digests(), "environment": environment(), **record}


def write(record: dict, path: Path) -> str:
    text = json.dumps(record, indent=1, sort_keys=False) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def specs(names) -> dict:
    return {n: (n, {}) for n in names}
