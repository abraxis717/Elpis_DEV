"""Native K1 differential qualification command line. RESEARCH_ONLY.

Run from the repository root with ``PYTHONPATH=src:.`` and a Release native build::

    python -m research.ecs_k1_native.run qualify --library <libelpis_ecsg_math.so>   # once; writes the record
    python -m research.ecs_k1_native.run check   --library <...> [--world ID ...]   # current runtime, no record

``qualify`` refuses a dirty tree, a changed plan or R3 authority, and an existing record. ``transplant`` is the
clean child process both commands launch. Nothing here forces a BLAS kernel; native K1 calls no BLAS.
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
from research.ecs_retention_r3 import numerics  # noqa: E402
from research.ecs_retention_r3 import protocol as P3  # noqa: E402
from research.ecs_retention_r3 import run as R3  # noqa: E402

from . import differential as D  # noqa: E402

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DOMAIN = "elpis.research.ecs-k1-native"
PLAN_FILE = ROOT / "specs" / "ecsg-k1-native.v1.plan.json"
PLAN_MD = ROOT / "PLAN.md"
RECORD = ROOT / "evidence" / "ecsg-k1-native.v1.qualification.json"
WORKERS = max(1, min(4, os.cpu_count() or 1))
DECISION_KEYS = ("validity", "mechanics", "gates", "disposition", "outcome", "controls")


class QualificationError(ValueError):
    """The plan, the R3 authority, the tree or the record refuses the run."""


def plan() -> dict:
    return json.loads(PLAN_FILE.read_text(encoding="utf-8"))


def digest(kind: str, value) -> str:
    return hashlib.sha256(f"{DOMAIN}.{kind}.v1".encode("ascii") + b"\0" + P3.canonical_json(value)).hexdigest()


def r3_qual() -> dict:
    """The R3 QUAL record, verified against the plan's pins."""
    authority = plan()["authority"]
    for key in ("r3_spec", "r3_frozen", "r3_qual"):
        if P3.sha256_file(REPO / authority[key]["path"]) != authority[key]["sha256"]:
            raise QualificationError(f"{authority[key]['path']} differs from the plan's pin")
    record = P3.load(R3.QUAL_PATH, "qual")
    if record["digest"] != authority["r3_qual"]["digest"] or record["body"]["outcome"] != "OUTCOME_A":
        raise QualificationError("the R3 QUAL record is not the planned OUTCOME_A record")
    return record["body"]


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def implementation(library: Path) -> dict:
    library = Path(library).resolve()
    libs = {name: P3.sha256_file(library.with_name(name))
            for name in ("libelpis_ecsg_math.so", "libelpis_ecsg_k1.so", "libelpis_ecsg_k1_fms.so")}
    dirty = _git("status", "--porcelain", "--", "src", "native", "research", ":(exclude)research/ecs_k1_native/evidence")
    return {"base_commit": _git("rev-parse", "HEAD"), "tree": _git("rev-parse", "HEAD^{tree}"), "dirty": bool(dirty),
            "libraries": libs, "build": P3._build(library), "numerical_profile": numerics.profile(),
            "plan_sha256": P3.sha256_file(PLAN_FILE), "plan_md_sha256": P3.sha256_file(PLAN_MD)}


# --- workers ---------------------------------------------------------------------------------------------------

_STATE = {}


def _init(library: str) -> None:
    nat = D.Native(library)
    _STATE.update(nat=nat, lab=X.Lab(nat.api, R3.SPEC), qual=r3_qual(), tol=plan()["tolerances"])


def _job(wid: str) -> dict:
    rec = _STATE["qual"]["per_world"][wid]
    out = D.world_differential(_STATE["nat"], _STATE["lab"], {"world": wid, **rec}, _STATE["tol"])
    out["_b_envelope"] = out["_b_envelope"].hex()
    out["_envelopes"] = {t: hashlib.sha256(e).hexdigest() for t, e in out["_envelopes"].items()}
    return P3.plain(out)


def _map(library: Path, worlds) -> dict:
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx, initializer=_init,
                             initargs=(str(library),)) as pool:
        return dict(zip(worlds, pool.map(_job, worlds)))


def _child_env() -> dict:
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("OPENBLAS_CORETYPE", None)
    return env


def clean_transplant(library: Path, per_world: dict) -> dict:
    """Every world's B envelope continued through C and D in one clean process."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({w: r["_b_envelope"] for w, r in per_world.items()}, fh)
        path = fh.name
    try:
        out = subprocess.run([sys.executable, "-m", "research.ecs_k1_native.run", "transplant", "--library",
                              str(library), "--input", path], capture_output=True, text=True, cwd=REPO,
                             env=_child_env(), check=True)
    finally:
        os.unlink(path)
    child = json.loads(out.stdout)
    bitwise = {w: child[w] == {t: per_world[w]["_envelopes"][t] for t in ("C", "D")} for w in per_world}
    return {"worlds": len(bitwise), "bitwise": all(bitwise.values()),
            "failed": sorted(w for w, ok in bitwise.items() if not ok)}


def determinism(library: Path, wid: str, per_world: dict) -> dict:
    nat = D.Native(str(library))
    lab = X.Lab(nat.api, R3.SPEC)
    world = lab.world(wid)
    runs = [{t: hashlib.sha256(s["envelope"]).hexdigest()
             for t, s in D.standalone_run(nat, world, lab.rate, lab.steps)["stages"].items()} for _ in range(2)]
    return {"world": wid, "runs": runs, "identical": all(r == per_world[wid]["_envelopes"] for r in runs)}


def verdict(per_world: dict, qual: dict, det: dict, transplant: dict) -> dict:
    """The unchanged R3 gates over the recorded rows with the native rows substituted, and the plan's verdict."""
    native = D.substituted(qual["per_world"], per_world)
    gated = X.gates(native, R3.SPEC, {"identical": det["identical"]}, {"bitwise": transplant["bitwise"]})
    record = P3.plain(gated["decision_record"])
    recorded = qual["pre_robustness_decision_record"]
    equal = {key: record[key] == recorded[key] for key in DECISION_KEYS}
    for key in ("K1", "causality"):
        equal[key] = record[key] == recorded[key]
    gates = dict(gated["gates"], L_numerical_robustness=qual["gates"]["L_numerical_robustness"])
    worlds = list(native)
    final = X.disposition(gated["validity"], gated["mechanics"], gates,
                          *[[native[w]["P"]["mechanisms"][k] for w in worlds] for k in ("K1", "M0", "M1")],
                          R3.SPEC["thresholds"])
    world_checks = {w: per_world[w]["pass"] for w in worlds}
    qualified = (all(world_checks.values()) and det["identical"] and transplant["bitwise"] and all(equal.values())
                 and record["outcome"] == "OUTCOME_A" and final["outcome"] == "OUTCOME_A")
    return {"decision_record": record, "decision_record_equal": equal, "gates": gates,
            "disposition": final["disposition"], "outcome": final["outcome"],
            "worlds_passing": sum(world_checks.values()), "failed_worlds": sorted(w for w, ok in world_checks.items()
                                                                                if not ok),
            "verdict": "QUALIFIED" if qualified else "NOT_QUALIFIED"}


def _summary(per_world: dict) -> dict:
    keys = ("w_relative", "h_relative", "a_relative", "query_relative", "nmse_relative")
    bounds = [b for r in per_world.values() for b in r["comparisons"]["boundaries"].values()]
    out = {f"max_{k}": max(b[k] for b in bounds) for k in keys}
    out["max_e_reset_relative"] = max(r["comparisons"]["reset"]["e_reset_relative"] for r in per_world.values())
    out["reference_reproduces_recorded_state_digests"] = sum(
        b["reference_reproduces_recorded_state_digest"] for b in bounds)
    out["boundaries"] = len(bounds)
    out["checks_passing"] = {k: sum(r["checks"][k] for r in per_world.values())
                             for k in next(iter(per_world.values()))["checks"]}
    return out


def evaluate_all(library: Path, worlds=None) -> dict:
    qual = r3_qual()
    worlds = list(worlds or qual["world_ids"])
    per_world = _map(library, worlds)
    det = determinism(library, qual["world_ids"][0], per_world) if qual["world_ids"][0] in per_world else None
    transplant = clean_transplant(library, per_world)
    out = {"per_world": per_world, "determinism": det, "clean_transplant": transplant, "summary": _summary(per_world)}
    if worlds == list(qual["world_ids"]):
        out["verdict"] = verdict(per_world, qual, det, transplant)
    return out


def qualify(library: Path) -> dict:
    if RECORD.exists():
        raise QualificationError("the qualification already ran; a repair is a new plan version")
    impl = implementation(library)
    if impl["dirty"]:
        raise QualificationError("qualification needs a clean tree (commit the harness first)")
    P3.require_single_thread()
    qual = r3_qual()
    result = evaluate_all(library)
    p = plan()
    body = {"experiment": f"{p['name']}.v{p['version']}", "plan_digest": digest("plan", p),
            "r3_qual_digest": plan()["authority"]["r3_qual"]["digest"], "world_ids": qual["world_ids"],
            "tolerances": p["tolerances"], "implementation": impl,
            "per_world": {w: {k: v for k, v in r.items() if not k.startswith("_")}
                          for w, r in result["per_world"].items()},
            "determinism": result["determinism"], "clean_transplant": result["clean_transplant"],
            "summary": result["summary"], **result["verdict"]}
    record = {"body": P3.plain(body), "digest": digest("qualification", body)}
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    with open(RECORD, "xb") as fh:
        fh.write(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n")
    return {"digest": record["digest"], "verdict": body["verdict"], "outcome": body["outcome"],
            "worlds_passing": body["worlds_passing"], "summary": body["summary"]}


def load_record() -> dict:
    record = json.loads(RECORD.read_bytes())
    if digest("qualification", record["body"]) != record["digest"]:
        raise QualificationError("qualification record digest does not match its content")
    return record


def transplant_child(library: Path, inputs: Path) -> dict:
    nat = D.Native(str(library))
    lab = X.Lab(nat.api, R3.SPEC)
    out = {}
    for wid, env in json.loads(inputs.read_text()).items():
        cont = D.continue_from(nat, bytes.fromhex(env), lab.world(wid), lab.rate, lab.steps)
        out[wid] = {t: hashlib.sha256(cont[t]).hexdigest() for t in ("C", "D")}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research.ecs_k1_native.run")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("qualify").add_argument("--library", required=True, type=Path)
    c = sub.add_parser("check")
    c.add_argument("--library", required=True, type=Path)
    c.add_argument("--world", action="append")
    t = sub.add_parser("transplant")
    t.add_argument("--library", required=True, type=Path)
    t.add_argument("--input", required=True, type=Path)
    args = parser.parse_args(argv)
    library = args.library.resolve()
    if args.command == "qualify":
        result = qualify(library)
    elif args.command == "check":
        full = evaluate_all(library, args.world)
        result = {"summary": full["summary"], "clean_transplant": full["clean_transplant"],
                  "determinism": full["determinism"],
                  "checks": {w: r["checks"] for w, r in full["per_world"].items()},
                  "verdict": {k: v for k, v in full.get("verdict", {}).items() if k != "decision_record"},
                  "decision_record": full.get("verdict", {}).get("decision_record")}
    else:
        result = transplant_child(library, args.input)
    sys.stdout.write(json.dumps(P3.plain(result), sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
