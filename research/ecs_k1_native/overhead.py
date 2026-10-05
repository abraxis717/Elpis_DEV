"""Python control-plane overhead of the native K1 runtime (docs/performance/ECS_K1_RUNTIME.md). PERFORMANCE_ONLY.

    PYTHONPATH=src:. python -m research.ecs_k1_native.overhead --library <libelpis_ecsg_math.so>

Split timing, as Runtime R1's supplementary measurement: each public call is timed in total and, inside it, the one
native call it makes; overhead = total - native. Also counts native crossings per call (always one). Descriptive:
no threshold was registered for these numbers.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import random
import statistics
import tempfile
import time
from array import array

from elpis.ECS_G.k1 import K1FMSRuntime, K1Library, K1State
from elpis.substrate.residency import Context

D, N, R = 6, 36, 64


class _Timed:
    """Wraps every bound native symbol: counts crossings and accumulates the time spent inside them."""

    def __init__(self, namespace):
        self.calls, self.native_ns = 0, 0
        for name, fn in vars(namespace).copy().items():
            setattr(namespace, name, self._wrap(fn))

    def _wrap(self, fn):
        def call(*args):
            t = time.perf_counter_ns()
            try:
                return fn(*args)
            finally:
                self.native_ns += time.perf_counter_ns() - t
                self.calls += 1
        return call


def _measure(timed, op, samples):
    total, native, calls = [], [], set()
    for _ in range(samples):
        timed.calls, timed.native_ns = 0, 0
        t = time.perf_counter_ns()
        op()
        total.append(time.perf_counter_ns() - t)
        native.append(timed.native_ns)
        calls.add(timed.calls)
    over = [a - b for a, b in zip(total, native)]
    return {"total_p50_us": statistics.median(total) / 1e3, "native_p50_us": statistics.median(native) / 1e3,
            "overhead_p50_us": statistics.median(over) / 1e3,
            "overhead_p95_us": sorted(over)[int(0.95 * len(over))] / 1e3, "native_calls_per_op": sorted(calls)}


def run(library: Path, samples: int = 300) -> dict:
    lib = Path(library).resolve()
    rng = random.Random(7)
    w = array("d", (rng.uniform(-0.18, 0.18) for _ in range(D * N)))
    x = array("d", (rng.uniform(-0.5, 0.5) for _ in range(D * R)))
    y = array("d", (rng.uniform(-0.4, 0.4) for _ in range(R)))
    k1 = K1Library(ctypes.CDLL(str(lib.with_name("libelpis_ecsg_k1.so"))))
    timed = _Timed(k1._k)
    out = {}
    with K1State.create(k1, D, N, w, max_rows=R) as s:
        s.consolidate(x)
        out["K1State.query"] = _measure(timed, lambda: s.query(x), samples)
        buf = array("d", bytes(8 * R))
        out["K1State.query_into"] = _measure(timed, lambda: s.query_into(x, buf), samples)
        for k in (1, 10, 100):
            out[f"K1State.learn K={k}"] = _measure(timed, lambda k=k: s.learn(x, y, 0.002, k), max(20, samples // k))
        out["K1State.consolidate"] = _measure(timed, lambda: s.consolidate(x), samples)
    adapter = ctypes.CDLL(str(lib.with_name("libelpis_ecsg_k1_fms.so")))
    image = k1.envelope_bytes(D, N) - 32
    with tempfile.TemporaryDirectory() as cold, \
            Context(adapter, warm_bytes=8 * image, cold_bytes=10 ** 6, max_objects=4,
                    cold_root=Path(cold) / "cold") as ctx, K1FMSRuntime(k1, ctx, adapter, max_states=2) as r:
        ftimed = _Timed(r._f)
        sid = r.register(b"\x05" * 32, D, N, w, max_rows=R)
        r.consolidate(sid, x)
        out["K1FMSRuntime.query"] = _measure(ftimed, lambda: r.query(sid, x), samples)
        out["K1FMSRuntime.learn K=10"] = _measure(ftimed, lambda: r.learn(sid, x, y, 0.002, 10), samples // 10)
        r.close_state(sid)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research.ecs_k1_native.overhead")
    parser.add_argument("--library", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.library), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
