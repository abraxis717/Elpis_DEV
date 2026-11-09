"""R118-D diagnostic (not an admission measurement): subnormal arithmetic in decaying regimes.

Without external input, repeatedly stepping the same instances is dissipative:
states decay exponentially. Once ||s||^2 < 1e-24 the controller denominator is
pinned at 1e-24 and the product s * (s.d) underflows into the subnormal range,
where x86 arithmetic is microcode-assisted and slow. This diagnostic steps all
SMALL instances from identical initial states with FTZ/DAZ off (default, exact)
and on (opt-in), recording latency per block and the exact deviation it causes.

    python -B -m research.yang_mills_hca.r118_scalable_awr.diag_subnormal [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import bench as B
from . import engine as E
from . import provenance as P

STEPS = 4000
BLOCK = 250


def run(checkpoint="54007/STATE8", controller="R104") -> dict:
    proto, psha = B.protocol()
    cfg = proto["registries"]["SMALL"]
    states, engines = [], []
    for ftz in (False, True):
        _, st, _, _ = B.build(cfg)
        states.append(st)
        engines.append(E.Engine(st, checkpoint, controller=controller, budgets=B.HOT, flush_subnormals=ftz))
    np.testing.assert_array_equal(states[0].pool, states[1].pool)
    allh = np.nonzero(states[0].inst_live)[0]
    rows = np.nonzero(states[0].row_ref > 0)[0]
    tiny = np.finfo(np.float64).tiny
    blocks = []
    lat = ([], [])
    for step in range(1, STEPS + 1):
        for k in (0, 1):
            t0 = time.perf_counter_ns()
            r = engines[k].step(allh)
            lat[k].append((time.perf_counter_ns() - t0) / 1e6)
            assert r.exact
        if step % BLOCK == 0:
            a, b = states[0].pool[rows], states[1].pool[rows]
            scale = float(np.abs(a).max())
            blocks.append({
                "step": step,
                "exact_p50_ms": float(np.percentile(lat[0], 50)), "exact_p95_ms": float(np.percentile(lat[0], 95)),
                "ftz_p50_ms": float(np.percentile(lat[1], 50)), "ftz_p95_ms": float(np.percentile(lat[1], 95)),
                "max_abs_state": scale,
                "exact_subnormal_coords": int(np.sum((np.abs(a) > 0) & (np.abs(a) < tiny))),
                "ftz_exact_zero_coords": int(np.sum(b == 0.0)),
                "ftz_vs_exact_max_abs": float(np.abs(a - b).max()),
                "ftz_vs_exact_max_rel_to_state_scale": float(np.abs(a - b).max() / scale) if scale > 0 else 0.0,
            })
            lat = ([], [])
    return {"phase": "R118-D-diagnostic", "protocol_sha256": psha, "checkpoint": checkpoint,
            "controller": controller, "instances": int(allh.size), "steps": STEPS, "blocks": blocks,
            "environment": P.environment_fingerprint()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    res = run()
    blob = json.dumps(res, indent=1, sort_keys=True, default=P.json_default)
    if args.out:
        with open(args.out, "w") as f:
            f.write(blob + "\n")
    print(blob)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
