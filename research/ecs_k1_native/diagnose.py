"""Post-hoc diagnosis of the NOT_QUALIFIED record (descriptive; decides nothing). RESEARCH_ONLY.

For the two worlds outside the planned float tolerances, and the first QUAL world as a control: how far the
unchanged R3 laboratory itself moves over the failing stage when its starting W is perturbed by one ulp in one
entry, against how far native K1 is from the laboratory. Also the native-vs-laboratory divergence along the
stage. Run: ``PYTHONPATH=src:. python -m research.ecs_k1_native.diagnose --library <libelpis_ecsg_math.so>``.
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

from elpis.ECS_G.k1 import K1State  # noqa: E402
from research.ecs_retention_r3 import engine as E  # noqa: E402
from research.ecs_retention_r3 import experiment as X  # noqa: E402
from research.ecs_retention_r3 import run as R3  # noqa: E402

from . import differential as D  # noqa: E402

CASES = (("r3qual-0019", "C"), ("r3qual-0028", "B"), ("r3qual-0000", "B"))
CHECKPOINTS = (1000, 2000, 3000, 4000)


def case(nat, lab, wid: str, stage: str) -> dict:
    world = lab.world(wid)
    states = lab.mechanism_run(world, "K1", keep=True)["_states"]
    prev = X.STAGES[X.STAGES.index(stage) - 1]
    mech, W0, _ = states[prev]
    exp = world.tasks[stage]
    perturbed = W0.copy()
    perturbed[0, 0] = np.nextafter(perturbed[0, 0], np.inf)
    lab_end = E.engine_learn(nat.api, W0, exp.x_train, exp.y_train, lab.rate, lab.steps, mech)[0]
    ulp_end = E.engine_learn(nat.api, perturbed, exp.x_train, exp.y_train, lab.rate, lab.steps, mech)[0]
    native = K1State.create(nat.k1, *W0.shape, D._mv(world.w0), max_rows=len(exp.y_train))
    for t in X.STAGES[:X.STAGES.index(stage)]:
        e = world.tasks[t]
        native.learn(D._mv(e.x_train), D._mv(e.y_train), lab.rate, lab.steps)
        native.consolidate(D._mv(e.x_train))
    start = D._rel(np.asarray(native.w()).reshape(W0.shape), W0)
    envelope = native.snapshot()
    native.close()
    along, W, done = {}, W0.copy(), 0
    for k in CHECKPOINTS:
        W = E.engine_learn(nat.api, W, exp.x_train, exp.y_train, lab.rate, k - done, mech)[0]
        done = k
        with K1State.restore(nat.k1, envelope) as s:
            s.learn(D._mv(exp.x_train), D._mv(exp.y_train), lab.rate, k)
            along[k] = D._rel(np.asarray(s.w()).reshape(W0.shape), W)
    return {"world": wid, "stage": stage, "native_vs_lab_at_stage_start": start,
            "native_vs_lab_along_stage": along, "lab_vs_lab_one_ulp_at_stage_end": D._rel(ulp_end, lab_end)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research.ecs_k1_native.diagnose")
    parser.add_argument("--library", required=True, type=Path)
    args = parser.parse_args(argv)
    nat = D.Native(str(args.library.resolve()))
    lab = X.Lab(nat.api, R3.SPEC)
    out = [case(nat, lab, wid, stage) for wid, stage in CASES]
    sys.stdout.write(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
