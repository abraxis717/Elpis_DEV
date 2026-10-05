"""Same-host current-runtime regression of native K1 against the frozen Retention R3 result. RESEARCH_ONLY.

    python -m research.ecs_k1_native.regression --library <libelpis_ecsg_math.so>

Builds every R3 QUAL world on this host, runs the same-host exact invariants (samehost.py), substitutes the native
rows into the recorded R3 per-world rows and applies the unchanged R3 gates. The decision record must equal the
recorded R3 one (every decision-bearing quantity; R3 gate L showed it kernel invariant) and the outcome must be
OUTCOME_A. No recorded W digest is compared: those are historical bytes of the recording host.
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
from pathlib import Path  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

from research.ecs_retention_r3 import experiment as X  # noqa: E402
from research.ecs_retention_r3 import protocol as P3  # noqa: E402
from research.ecs_retention_r3 import run as R3  # noqa: E402

from . import differential as D  # noqa: E402
from . import samehost as S  # noqa: E402

WORKERS = max(1, min(4, os.cpu_count() or 1))
DECISION_KEYS = ("validity", "mechanics", "gates", "disposition", "outcome", "controls", "K1", "causality")

_STATE = {}


def r3_qual() -> dict:
    return P3.load(R3.QUAL_PATH, "qual")["body"]


def _init(library: str) -> None:
    nat = D.Native(library)
    _STATE.update(nat=nat, lab=X.Lab(nat.api, R3.SPEC), qual=r3_qual())


def _job(wid: str) -> dict:
    out = S.world_samehost(_STATE["nat"], _STATE["lab"], wid, _STATE["qual"]["per_world"][wid])
    out.pop("_main")
    out["_b_envelope"] = out["_b_envelope"].hex()
    return P3.plain(out)


def samehost_worlds(library: Path, worlds) -> dict:
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx, initializer=_init,
                             initargs=(str(library),)) as pool:
        return dict(zip(worlds, pool.map(_job, worlds)))


def determinism(library: Path, wid: str, reference: dict) -> dict:
    """Two fresh native runs of one world: bitwise identical envelopes, and identical to the worker's run."""
    nat = D.Native(str(library))
    lab = X.Lab(nat.api, R3.SPEC)
    world = lab.world(wid)
    runs = [{t: hashlib.sha256(s["envelope"]).hexdigest()
             for t, s in D.standalone_run(nat, world, lab.rate, lab.steps)["stages"].items()} for _ in range(2)]
    return {"world": wid, "identical": all(r == reference for r in runs)}


def clean_transplant(library: Path, per_world: dict) -> dict:
    """Every world's B envelope continued through C and D in one clean child process that inherits this process's
    environment unchanged (nothing is set or removed: the child sees the same host as the parent)."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({w: r["_b_envelope"] for w, r in per_world.items()}, fh)
        path = fh.name
    env = dict(os.environ, PYTHONPATH=f"{P3.REPO / 'src'}{os.pathsep}{P3.REPO}", PYTHONDONTWRITEBYTECODE="1")
    try:
        out = subprocess.run([sys.executable, "-m", "research.ecs_k1_native.run", "transplant", "--library",
                              str(library), "--input", path], capture_output=True, text=True, cwd=P3.REPO, env=env,
                             check=True)
    finally:
        os.unlink(path)
    child = json.loads(out.stdout)
    bitwise = {w: child[w] == {t: per_world[w]["envelopes_sha256"][t] for t in ("C", "D")} for w in per_world}
    return {"worlds": len(bitwise), "bitwise": all(bitwise.values()),
            "failed": sorted(w for w, ok in bitwise.items() if not ok)}


def check(library: Path) -> dict:
    qual = r3_qual()
    worlds = list(qual["world_ids"])
    native = samehost_worlds(library, worlds)
    det = determinism(library, worlds[0], native[worlds[0]]["envelopes_sha256"])
    transplant = clean_transplant(library, native)
    rec = S.decision_record(qual["per_world"], native, R3.SPEC, det["identical"], transplant["bitwise"])
    record = P3.plain(rec["decision_record"])
    recorded = qual["pre_robustness_decision_record"]
    gates = dict(rec["gates"], L_numerical_robustness=qual["gates"]["L_numerical_robustness"])
    sub = rec["substituted"]
    final = X.disposition(rec["validity"], rec["mechanics"], gates,
                          *[[sub[w]["P"]["mechanisms"][k] for w in worlds] for k in ("K1", "M0", "M1")],
                          R3.SPEC["thresholds"])
    return {"checks": {w: r["checks"] for w, r in native.items()},
            "failed_worlds": sorted(w for w, r in native.items() if not r["pass"]),
            "determinism": det, "clean_transplant": transplant,
            "decision_record_equal": {k: record[k] == recorded[k] for k in DECISION_KEYS},
            "decision_record_diff": {k: sorted(w for w in recorded[k] if record[k].get(w) != recorded[k][w])
                                     for k in ("K1", "causality")},
            "outcome": final["outcome"], "disposition": final["disposition"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research.ecs_k1_native.regression")
    parser.add_argument("--library", required=True, type=Path)
    args = parser.parse_args(argv)
    sys.stdout.write(json.dumps(P3.plain(check(args.library.resolve())), sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
