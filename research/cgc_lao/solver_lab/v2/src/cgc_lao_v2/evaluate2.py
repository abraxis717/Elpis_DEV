"""v2 runs and sealed evidence records. An arm spec is {"arm", "contract", "mode", "overrides"}; every arm of a
record is evaluated on the same episodes. A record binds its numbers to the SHA-256 of every v2 and v1 source
file it ran, both benchmark specs and the v2 coefficients."""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from cgc_lao_lab import evaluate as ev1

from . import V1_SRC, V2_ROOT, arms2, harness2

SRC2 = V2_ROOT / "src" / "cgc_lao_v2"
SPEC2 = V2_ROOT / "specs" / "slb2.spec.json"
NATIVE = V2_ROOT / "native"


def source_file_digests() -> dict:
    out = {f"v2/{p.name}": ev1.file_digest(p) for p in sorted(SRC2.glob("*.py"))}
    out.update({f"v1/{p.name}": ev1.file_digest(p) for p in sorted((V1_SRC / "cgc_lao_lab").glob("*.py"))})
    return out


def native_file_digests() -> dict:
    return {str(p.relative_to(NATIVE)): ev1.file_digest(p) for p in sorted(NATIVE.rglob("*"))
            if p.is_file() and p.suffix in (".c", ".h", ".txt")}


def source_digest() -> str:
    h = hashlib.sha256()
    for name, digest in sorted(source_file_digests().items()):
        h.update(name.encode() + b"\0" + digest.encode() + b"\0")
    return h.hexdigest()


def coefficient_digests() -> dict:
    return {p.name: ev1.file_digest(p) for p in sorted((V2_ROOT / "coefficients").glob("*.json"))}


def spec(arm: str, contract: str = "DUP2_4", mode: str = "DUPLICATE", **overrides) -> dict:
    return {"arm": arm, "contract": contract, "mode": mode, "overrides": overrides}


def _worker(task):
    partition, seed, regimes, specs, episodes, chunk = task
    built = {label: {**s, "factory": arms2.factory(s["arm"], **s["overrides"])} for label, s in specs.items()}
    return seed, harness2.evaluate_seed2(partition, seed, regimes, built, episodes, chunk)


def run(partition: str, seeds, regimes, specs: dict, episodes: int = 2048, workers: int = 4, chunk: int = 2048) -> dict:
    seeds = list(seeds)
    tasks = [(partition, s, list(regimes), specs, episodes, chunk) for s in seeds]
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=get_context("fork")) as ex:
            by_seed = dict(ex.map(_worker, tasks))
    else:
        by_seed = dict(map(_worker, tasks))
    results: dict = {}
    for r in regimes:
        for label in specs:
            metrics = by_seed[seeds[0]][r].get(label)
            if metrics:
                results.setdefault(r, {})[label] = {m: [float(f"{by_seed[s][r][label][m]:.10g}") for s in seeds]
                                                    for m in metrics}
    return {"partition": partition, "seeds": seeds, "episodes_per_seed": episodes, "regimes": list(regimes),
            "arms": {k: {"arm": v["arm"], "contract": v["contract"], "mode": v["mode"], "overrides": v["overrides"]}
                     for k, v in specs.items()},
            "results": results, "seconds": round(time.time() - t0, 1)}


def seal(record: dict, phase: str, note: str = "") -> dict:
    return {"schema": "cgc-lao-solver-lab.record.v2", "phase": phase, "note": note,
            "source_sha256": source_digest(), "source_files_sha256": source_file_digests(),
            "native_files_sha256": native_file_digests(),
            "spec_sha256": {"slb1": ev1.file_digest(ev1.SPEC), "slb2": ev1.file_digest(SPEC2)},
            "coefficients_sha256": coefficient_digests(), "environment": ev1.environment(), **record}


def write(record: dict, path: Path) -> str:
    text = json.dumps(record, indent=1) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
