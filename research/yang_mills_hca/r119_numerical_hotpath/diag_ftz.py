"""R119 post-hoc diagnostic (not a registered measurement): where FTZ_DAZ deviates.

Rebuilds the registered MIXED controlled regime (STRESS1M, per-row scales 10^U(-320, 0),
active sets from the registered seed for A = 1,024) and, for every step, compares FTZ_DAZ
with EXACT_R118 coordinate by coordinate: magnitude of the exact value, absolute and
relative deviation, and whether the exact value was normal or subnormal.

    python -B -m research.yang_mills_hca.r119_numerical_hotpath.diag_ftz [--steps N] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from research.yang_mills_hca.r118_scalable_awr import bench as B
from research.yang_mills_hca.r118_scalable_awr import provenance as P

from .engine_r119 import R119Engine
from .qualify_r119 import build_states, protocol

TINY = 2.2250738585072014e-308


def run(steps=60) -> dict:
    proto, psha = protocol()
    r118, _ = B.protocol()
    c = proto["controlled_regimes"]
    names = list(c["scales"])
    ri, si, A = names.index("MIXED"), c["active_sizes"].index(1024), 1024
    _, (st,) = build_states(r118["registries"]["STRESS1M"], 1)
    base = st.pool.copy()
    snap = base * 10.0 ** np.random.default_rng(119500).uniform(-320.0, 0.0, size=(base.shape[0], 1))
    st.pool[:] = snap
    eng = {m: R119Engine(st, "54007/STATE8_SIGNED8", controller="R114", budgets=B.HOT, mode=m)
           for m in ("EXACT_R118", "FTZ_DAZ")}
    live = np.nonzero(st.inst_live)[0].astype(np.int32)
    rng = np.random.default_rng(119000 + 10 * ri + si)
    worst = []
    hist = {"deviating_coords": 0, "exact_normal": 0, "exact_subnormal": 0, "exact_zero": 0,
            "rel_gt_1e-12": 0, "rel_gt_1e-6": 0, "rel_ge_1": 0}
    for _ in range(steps):
        act = np.sort(rng.choice(live, size=A, replace=False))
        rows = st.slot_row[act]
        rows = np.unique(rows[rows >= 0])
        outs = {}
        for m, e in eng.items():
            st.pool[rows] = snap[rows]
            assert e.step(act).exact
            outs[m] = st.pool[rows].copy()
        st.pool[rows] = snap[rows]
        ex, fz = outs["EXACT_R118"], outs["FTZ_DAZ"]
        dev = np.abs(fz - ex)
        mask = ex.view(np.uint64) != fz.view(np.uint64)
        hist["deviating_coords"] += int(mask.sum())
        a = np.abs(ex[mask])
        hist["exact_zero"] += int(np.sum(a == 0))
        hist["exact_subnormal"] += int(np.sum((a > 0) & (a < TINY)))
        hist["exact_normal"] += int(np.sum(a >= TINY))
        rel = dev[mask] / np.where(a > 0, a, np.inf)
        hist["rel_gt_1e-12"] += int(np.sum(rel > 1e-12))
        hist["rel_gt_1e-6"] += int(np.sum(rel > 1e-6))
        hist["rel_ge_1"] += int(np.sum(rel >= 1.0))
        idx = np.argwhere(dev > 2.0 ** -1000)
        for r, k in idx[np.argsort(-dev[tuple(idx.T)])][:5]:
            worst.append({"exact": float(ex[r, k]), "ftz": float(fz[r, k]), "abs_dev": float(dev[r, k]),
                          "rel_dev": float(dev[r, k] / abs(ex[r, k])) if ex[r, k] != 0 else None,
                          "row_scale": float(np.abs(snap[rows[r]]).max())})
    worst.sort(key=lambda w: -w["abs_dev"])
    return {"phase": "R119-diagnostic-ftz", "protocol_sha256": psha, "regime": "MIXED", "A": A, "steps": steps,
            "summary": hist, "worst_over_2p-1000": worst[:12], "max_abs_dev": max((w["abs_dev"] for w in worst), default=0.0),
            "environment": P.environment_fingerprint()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    res = run(args.steps)
    blob = json.dumps(res, indent=1, sort_keys=True, default=P.json_default)
    if args.out:
        with open(args.out, "w") as f:
            f.write(blob + "\n")
    print(blob)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
