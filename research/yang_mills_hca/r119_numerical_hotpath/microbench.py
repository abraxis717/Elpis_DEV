"""R119 forensic microbenchmark driver: per-operation cost of subnormal operands/results.

    python -B -m research.yang_mills_hca.r119_numerical_hotpath.microbench OUT.json

Scalar ops (no auto-vectorization), best of 3 timed loops of 4,096 x 1,500 ops, in the
default floating-point environment and with FTZ/DAZ. Diagnostic only.
"""
import json
import sys

import numpy as np

from . import native_r119 as N


def main(path):
    lib, info = N.microbench()
    rng = np.random.default_rng(119)
    n, reps = 4096, 1500
    U = lambda lo, hi: rng.uniform(lo, hi, n)
    cases = [
     ("mul normal*normal->normal", 0, U(.5, 2), U(.5, 2)),
     ("mul tiny*tiny->normal (1e-150^2)", 0, U(.5, 2) * 1e-150, U(.5, 2) * 1e-150),
     ("mul normal*normal->SUBNORMAL", 0, U(.5, 2) * 1.5e-157, U(.5, 2) * 1.5e-157),
     ("mul normal*normal->underflow 0", 0, U(.5, 2) * 1e-200, U(.5, 2) * 1e-200),
     ("mul SUBNORMAL*O(1)->SUBNORMAL", 0, U(.5, 2) * 1e-310, U(.5, 2)),
     ("mul SUBNORMAL*big->normal", 0, U(.5, 2) * 1e-310, U(.5, 2) * 1e20),
     ("mul SUBNORMAL*tiny->underflow 0", 0, U(.5, 2) * 1e-310, U(.5, 2) * 1e-100),
     ("add normal+normal", 1, U(.5, 2), U(.5, 2)),
     ("add SUBNORMAL+SUBNORMAL->SUBNORMAL", 1, U(.5, 2) * 1e-310, U(.5, 2) * 1e-310),
     ("add SUBNORMAL+normal->normal", 1, U(.5, 2) * 1e-310, U(.5, 2)),
     ("sub normal-normal->SUBNORMAL (cancel)", 2, *(lambda x: (x, x * (1 - 2.0 ** -40)))(U(1, 2) * 1e-300)),
     ("add normal+normal->SUBNORMAL (cancel)", 1, *(lambda x: (x, -x * (1 - 2.0 ** -40)))(U(1, 2) * 1e-300)),
     ("sub normal-normal->normal tiny", 2, *(lambda x: (x, x * (1 - 2.0 ** -20)))(U(1, 2) * 1e-280)),
     ("add normal+SUBNORMAL->normal tiny", 1, U(1, 2) * 1e-300, U(.5, 2) * 1e-310),
     ("add normal+SUBNORMAL->SUBNORMAL", 1, U(1.0, 1.4) * 2.2250738585072014e-308, -U(0.6, 0.9) * 2.2250738585072014e-308),
     ("div normal/normal", 3, U(.5, 2), U(.5, 2)),
     ("div SUBNORMAL/1e-24->normal", 3, U(.5, 2) * 1e-310, np.full(n, 1e-24)),
     ("div SUBNORMAL/1->SUBNORMAL", 3, U(.5, 2) * 1e-310, np.full(n, 1.0)),
     ("div normal/normal->SUBNORMAL", 3, U(.5, 2) * 1e-300, U(.5, 2) * 1e10),
     ("div SUBNORMAL/huge->underflow 0", 3, U(.5, 2) * 1e-310, U(.5, 2) * 1e100),
     ("div normal/normal->underflow 0", 3, U(.5, 2) * 1e-300, U(.5, 2) * 1e100),
     ("sqrt normal", 4, U(.5, 2), U(.5, 2)),
     ("sqrt SUBNORMAL", 4, U(.5, 2) * 1e-315, U(.5, 2)),
     ("exp(normal)", 5, U(-1, 1), U(.5, 2)),
     ("exp(-tiny normal)", 5, -U(.5, 2) * 1e-200, U(.5, 2)),
     ("exp(-SUBNORMAL)", 5, -U(.5, 2) * 1e-310, U(.5, 2)),
     ("tanh(normal)", 6, U(-1, 1), U(.5, 2)),
     ("tanh(SUBNORMAL)", 6, U(.5, 2) * 1e-310, U(.5, 2)),
     ("cmp normal", 7, U(.5, 2), U(.5, 2)),
     ("cmp SUBNORMAL", 7, U(.5, 2) * 1e-310, U(.5, 2) * 1e-310),
    ]
    out = np.zeros(n)
    res = []
    for name, op, a, b in cases:
        a = np.ascontiguousarray(a); b = np.ascontiguousarray(b)
        row = {"case": name, "op": op}
        for flush in (0, 1):
            row["ftz_daz_ns" if flush else "default_ns"] = min(
                lib.fpmb_run(op, a.ctypes.data, b.ctypes.data, out.ctypes.data, n, reps, flush) for _ in range(3))
        res.append(row)
    print(f"{'case':38s} {'default ns':>10s} {'FTZ/DAZ ns':>10s}")
    for r in res:
        print(f"{r['case']:38s} {r['default_ns']:10.2f} {r['ftz_daz_ns']:10.2f}")
    json.dump({"info": info, "n": n, "reps": reps, "rows": res}, open(path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
