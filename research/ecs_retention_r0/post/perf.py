"""Retention R0 performance characterization. PERFORMANCE_ONLY. NO_SCIENTIFIC_CLAIM.

    PYTHONPATH=src:. python -m research.ecs_retention_r0.post.perf --library <lib> [--out <evidence.json>]

Learning-B latency on one DEV world at the frozen offset: the canonical
executor for M0 (64 rows) and M1 (128 rows), and the laboratory engine for
the frozen candidates and for the engine without consolidation. The
laboratory engine makes one native call per step plus NumPy work, so its
times characterize the research implementation, not a native one; the
analytic per-step operation counts estimate the native cost. State and
per-step scratch sizes are exact for the frozen regime.
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import ctypes  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from elpis.ECS_G.native import ECSGLibrary  # noqa: E402

from .. import engine as E  # noqa: E402
from .. import task as T  # noqa: E402
from ..protocol import implementation, load, load_spec, spec_digests  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "frozen" / "ecsg-retention-r0.v1.frozen.json"


def _pct(samples, q):
    s = sorted(samples)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]


def _summary(samples):
    return {"samples": len(samples), "p50_s": _pct(samples, 0.5), "p95_s": _pct(samples, 0.95),
            "p99_s": _pct(samples, 0.99), "min_s": min(samples), "max_s": max(samples)}


def measure(library: Path, repeats: int) -> dict:
    spec = load_spec()
    frozen = load(FROZEN, "frozen")["body"]
    choices = frozen["choices"]
    reg = spec["regime"]
    api = ECSGLibrary(ctypes.CDLL(str(library)))
    idx = T.indices(reg["dim"])
    world = T.build_world(spec, "dev-0000", choices["offset"])
    a, b = world.tasks["A"], world.tasks["B"]
    rate, steps = reg["learning_rate"], reg["steps"]
    W_A = E.canonical_learn(api, world.w0, a.x_train, a.y_train, rate, steps)
    XAB, yAB = np.vstack([a.x_train, b.x_train]), np.concatenate([a.y_train, b.y_train])

    def timed(fn, n):
        fn()
        out = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            out.append(time.perf_counter() - t)
        return _summary(out)

    def candidate(family, lam):
        def run():
            c = E.make_candidate(family, lam, idx, W_A.shape)
            c.consolidate(W_A, a.x_train)
            E.engine_learn(api, W_A, b.x_train, b.y_train, rate, steps, c)
        return run
    sel, sec = choices["selected"], choices["secondary"]
    F, d, n, rows = idx.size, reg["dim"], reg["width"], reg["train_rows"]
    return {
        "world": "dev-0000", "offset": choices["offset"], "steps": steps,
        "canonical_executor": {
            "M0_learn_B_64_rows": timed(lambda: E.canonical_learn(api, W_A, b.x_train, b.y_train, rate, steps),
                                        repeats * 4),
            "M1_learn_AuB_128_rows": timed(lambda: E.canonical_learn(api, W_A, XAB, yAB, rate, steps), repeats * 4)},
        "laboratory_engine": {
            "without_consolidation": timed(lambda: E.engine_learn(api, W_A, b.x_train, b.y_train, rate, steps),
                                           repeats),
            f"{sel['family']}_lambda_{sel['lambda']}": timed(candidate(sel["family"], sel["lambda"]), repeats),
            f"{sec['family']}_lambda_{sec['lambda']}": timed(candidate(sec["family"], sec["lambda"]), repeats)},
        "analytic_flops_per_step": E.flops_per_step(d, n, rows, idx),
        "state_bytes": {"W": 8 * d * n, "C1_H_b": E.C1(idx, 1.0).state_bytes(),
                        "C2_Omega_A": E.C2(idx, 1.0, (d, n)).state_bytes(), "M1_stored_experience": 8 * rows * (d + 1)},
        "per_step_scratch_bytes": {"C1": 8 * (F + d * d + 3 * len(idx.triples) * n + d * n),
                                   "C2": 8 * d * n},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research.ecs_retention_r0.post.perf")
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    library = args.library.resolve()
    report = {"label": "PERFORMANCE_ONLY NO_SCIENTIFIC_CLAIM", **spec_digests(load_spec()),
              "implementation": implementation(library), **measure(library, args.repeats)}
    text = json.dumps(report, indent=1, sort_keys=True) + "\n"
    if args.out:
        with open(args.out, "x", encoding="ascii") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
