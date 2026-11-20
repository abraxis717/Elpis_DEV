"""R11 supplement: CPU operator microbenchmark with ROTATED order. PERFORMANCE_ONLY; not part of R11's analysis.

``run_r11 latency`` (frozen) times its three operators in a fixed order (micro bounded, micro E0, legacy), so each
micro measurement follows the legacy operator's large working set. This supplement rotates the order every
repetition and also times the bound's elementwise work alone, to separate the bound from ordering effects.
Records go only where ``--out`` says.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

from ..cgc import CGCTransform, soft_bound


def measure(reps: int = 300, tokens: int = 64) -> dict:
    torch.set_num_threads(1)
    out = {"label": "MEASURED CPU wall-clock, one thread, rotated order; PERFORMANCE_ONLY", "tokens": tokens,
           "torch": torch.__version__, "python": platform.python_version(), "operators": {}}
    for hidden in (64, 768, 1024):
        mods = {"micro_bounded": CGCTransform(hidden, 3, 8, bottleneck=5, preact_bound=3.0).eval(),
                "micro_e0": CGCTransform(hidden, 3, 8, bottleneck=5, preact_bound=None).eval(),
                "legacy_r6_routed": CGCTransform(hidden, hidden, 4, bottleneck=hidden // 4, preact_bound=3.0).eval()}
        x = torch.randn(tokens, hidden)
        u = torch.randn(tokens, 24)
        calls = {**{k: (lambda m=m: m(x)) for k, m in mods.items()}, "bound_only_24_units": lambda: soft_bound(u, 3.0)}
        times = {k: [] for k in calls}
        with torch.no_grad():
            for _ in range(20):
                for f in calls.values():
                    f()
            order = list(calls)
            for rep in range(reps):
                for k in order[rep % len(order):] + order[:rep % len(order)]:
                    t = time.perf_counter()
                    calls[k]()
                    times[k].append(time.perf_counter() - t)
        out["operators"][str(hidden)] = {k: {"median_us": 1e6 * statistics.median(v),
                                             "p90_us": 1e6 * sorted(v)[int(0.9 * len(v))]} for k, v in times.items()}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="the JSON record to write")
    args = parser.parse_args(argv)
    record = measure()
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    else:
        json.dump(record, sys.stdout, indent=1)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
