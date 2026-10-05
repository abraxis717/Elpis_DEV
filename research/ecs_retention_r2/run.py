"""Retention R2 laboratory command line. RESEARCH_ONLY.

Run from the repository root with ``PYTHONPATH=src:.`` and a Release native build::

    python -m research.ecs_retention_r2.run dev    --library <libelpis_ecsg_math.so>
    python -m research.ecs_retention_r2.run freeze --commit <sha holding the laboratory> --library <...>
    python -m research.ecs_retention_r2.run qual   --library <...>
    python -m research.ecs_retention_r2.run status

Guards: ``dev`` refuses once frozen, needs a clean tree and an effective single BLAS thread. It applies the task
rule on controls only; if no input scale qualifies it records TASK_INVALID_ON_DEV and runs no candidate.
``freeze`` refuses after TASK_INVALID_ON_DEV, needs the DEV evidence to have been produced by the current
laboratory source and implementation, and writes once. ``qual`` refuses unless frozen, the specification,
pass rule, CANDIDATES.md, laboratory source, bound sources, library, build and numerical profile equal the frozen
ones, the tree is clean, the effective BLAS thread count is 1 and the host can force the robustness kernels
(AVX2 and FMA, OpenBLAS). It runs once, including its robustness children, and writes its evidence exclusively.
A failure is recorded as it is.

``probe``, ``transplant`` and ``robust`` are the child processes QUAL launches.
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import ctypes  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
from pathlib import Path  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

from elpis.ECS_G.native import ECSGLibrary  # noqa: E402

from . import engine as E  # noqa: E402
from . import experiment as X  # noqa: E402
from . import numerics  # noqa: E402
from . import task as T  # noqa: E402
from .protocol import (REPO, ROOT, ProtocolError, binding_mismatch, implementation, load, load_spec,  # noqa: E402
                       plain, require_single_thread, source_digest_at, spec_digests, world_ids, write)

SPEC = load_spec()
NAME = f"{SPEC['name']}.v{SPEC['version']}"
DEV_PATH = ROOT / "evidence" / "dev" / f"{NAME}.dev.json"
FROZEN_PATH = ROOT / "frozen" / f"{NAME}.frozen.json"
QUAL_PATH = ROOT / "evidence" / "qual" / f"{NAME}.qual.json"
ROBUST_CORES = ("Prescott", "Haswell")
WORKERS = max(1, min(4, os.cpu_count() or 1))

LIMITATIONS = [
    "Synthetic regression (SEMANTICS=NONE): d = 6, N = 36, the qualified cubic kernel, G1 at rate 0.002, 4000 steps "
    "per experience, 64 experience rows; no claim about language, meaning or general cognition.",
    "Candidate learning runs the canonical native G1 step plus a laboratory-side correction; any promotion needs "
    "its own native implementation and qualification.",
    "The task is jointly realizable by construction (aliased hidden ridges); no capacity-limited regime is tested.",
    "Bitwise reproduction is claimed only under the recorded binding; conclusions are compared across the forced "
    "OpenBLAS kernels of gate M.",
]

# --- worker processes ----------------------------------------------------------------------------------------

_LAB = None


def _init(library: str) -> None:
    global _LAB
    _LAB = X.Lab(ECSGLibrary(ctypes.CDLL(library)), SPEC)


def _job(args):
    kind = args[0]
    if kind == "dev_controls":
        return plain(X.dev_controls_world(_LAB, args[1], args[2]))
    if kind == "dev_candidates":
        return plain(X.dev_candidates_world(_LAB, args[1], args[2]))
    if kind == "qual":
        return plain(X.qual_world(_LAB, args[1], args[2]))
    raise ValueError(kind)


def _map(library: Path, jobs: list) -> list:
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx, initializer=_init,
                             initargs=(str(library),)) as pool:
        return list(pool.map(_job, jobs))


def _lab(library: Path) -> X.Lab:
    return X.Lab(ECSGLibrary(ctypes.CDLL(str(library))), SPEC)


def _child_env(core: str | None = None) -> dict:
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if core:
        env["OPENBLAS_CORETYPE"] = core
    return env


def _child(args: list, core: str | None = None) -> dict:
    out = subprocess.run([sys.executable, "-m", "research.ecs_retention_r2.run", *args], capture_output=True,
                         text=True, cwd=REPO, env=_child_env(core), check=True)
    return json.loads(out.stdout)


# --- DEV ---------------------------------------------------------------------------------------------------------


def dev(library: Path) -> dict:
    if FROZEN_PATH.exists():
        raise ProtocolError("frozen: DEV evidence is closed")
    require_single_thread()
    impl = implementation(library)
    if impl["dirty"]:
        raise ProtocolError("DEV needs a clean tree (commit the laboratory first)")
    worlds = world_ids(SPEC, "DEV")
    th = SPEC["thresholds"]
    task_rule, chosen = [], None
    for gain in SPEC["task"]["hidden_gain_grid"]:
        rows = dict(zip(worlds, _map(library, [("dev_controls", w, gain) for w in worlds])))
        verdict = X.task_validity_dev(rows, th)
        task_rule.append({"hidden_gain": gain, "validity": verdict, "worlds": rows})
        if verdict["valid"]:
            chosen = gain
            break
    body = {"experiment": NAME, "split": "DEV", "world_ids": list(worlds), **spec_digests(SPEC),
            "implementation": impl, "dev_rules": SPEC["dev_rules"], "task_rule": task_rule}
    if chosen is None:
        body.update({"disposition": "TASK_INVALID_ON_DEV", "choices": None})
    else:
        table = dict(zip(worlds, _map(library, [("dev_candidates", w, chosen) for w in worlds])))
        selected, ranking = X.select(table)
        body.update({"disposition": "VALID_TASK_ON_DEV", "candidate_table": table, "ranking": ranking,
                     "choices": {"hidden_gain": chosen, "selected": selected}})
    return {"digest": write(DEV_PATH, "dev", body), "disposition": body["disposition"], "choices": body["choices"]}


# --- freeze ------------------------------------------------------------------------------------------------------


def freeze(commit: str, library: Path) -> dict:
    record = load(DEV_PATH, "dev")
    body = record["body"]
    if body["disposition"] != "VALID_TASK_ON_DEV":
        raise ProtocolError("DEV ended TASK_INVALID_ON_DEV: no freeze, no QUAL")
    require_single_thread()
    impl = implementation(library)
    stale = binding_mismatch(body["implementation"], impl)
    if stale:
        raise ProtocolError(f"DEV evidence was produced by a different implementation: {stale}")
    if any(body[k] != spec_digests(SPEC)[k] for k in ("spec", "pass_rule", "candidates_sha256")):
        raise ProtocolError("specification, pass rule or CANDIDATES.md changed after DEV")
    if source_digest_at(commit) != body["implementation"]["lab_source_digest"]:
        raise ProtocolError(f"commit {commit} does not hold the laboratory source that produced DEV")
    frozen = {"experiment": NAME, **spec_digests(SPEC), "lab_commit": commit,
              "lab_source_digest": body["implementation"]["lab_source_digest"], "dev_evidence_digest": record["digest"],
              "dev_world_ids": body["world_ids"], "choices": body["choices"], "implementation": impl,
              "pass_rule_text": SPEC["pass_rule"], "qual_world_ids": list(world_ids(SPEC, "QUAL")),
              "robustness_cores": list(ROBUST_CORES)}
    return {"digest": write(FROZEN_PATH, "frozen", frozen), "choices": body["choices"]}


# --- QUAL --------------------------------------------------------------------------------------------------------


def _verdict(library: Path, choices: dict) -> tuple[dict, dict, dict, dict]:
    worlds = world_ids(SPEC, "QUAL")
    per_world = dict(zip(worlds, _map(library, [("qual", w, choices) for w in worlds])))
    first = worlds[0]
    lab_factory = lambda: _lab(library)  # noqa: E731
    runs = [X.determinism_digest(lab_factory, first, choices), X.determinism_digest(lab_factory, first, choices),
            _child(["probe", "--library", str(library), "--world", first, "--choices", json.dumps(choices)],
                   os.environ.get("OPENBLAS_CORETYPE"))]
    runs = [plain(r) for r in runs]
    recorded = {s: {k: per_world[first]["P"]["mechanisms"][choices["selected"]["candidate"]]["stages"][s][k]
                    for k in ("W_digest", "state_digest", "nmse")} for s in X.STAGES}
    determinism = {"world": first, "runs": runs, "recorded": recorded,
                   "identical": all(r["stages"] == recorded for r in runs)}
    transplant_probe = _transplant_probe(library, first, choices, per_world)
    return per_world, determinism, transplant_probe, X.gates(per_world, SPEC, choices, determinism, transplant_probe)


def _transplant_probe(library: Path, wid: str, choices: dict, per_world: dict) -> dict:
    """The complete declared state after B, serialized here and continued in a clean process."""
    lab = _lab(library)
    world = lab.world(wid, choices["hidden_gain"])
    kind = choices["selected"]["candidate"]
    run = lab.run_mechanism(world, kind, keep=True)["_run"]
    mech, W, epoch = run["_states"]["B"]
    with tempfile.NamedTemporaryFile(suffix=".state", delete=False) as fh:
        fh.write(E.serialize(mech, W, epoch))
        path = fh.name
    try:
        child = _child(["transplant", "--library", str(library), "--world", wid, "--choices", json.dumps(choices),
                        "--state", path], os.environ.get("OPENBLAS_CORETYPE"))
    finally:
        os.unlink(path)
    recorded = per_world[wid]["P"]["mechanisms"][kind]["stages"]
    bitwise = all(child[s] == {"W_digest": recorded[s]["W_digest"], "state_digest": recorded[s]["state_digest"]}
                  for s in ("C", "D"))
    return {"world": wid, "child": child, "bitwise": bitwise}


def _comparable(verdict: dict) -> dict:
    return {"validity": verdict["validity"], "mechanics": verdict["mechanics"], "gates": verdict["gates"],
            "disposition": verdict["disposition"], "outcome": verdict["outcome"], "counts": verdict["counts"]}


def qual(library: Path) -> dict:
    if QUAL_PATH.exists():
        raise ProtocolError("QUAL already ran; a repair is a new experiment version")
    frozen_record = load(FROZEN_PATH, "frozen")
    frozen = frozen_record["body"]
    digests = spec_digests(SPEC)
    if any(frozen[k] != digests[k] for k in ("spec", "pass_rule", "candidates_sha256")):
        raise ProtocolError("specification, pass rule or CANDIDATES.md changed after the freeze")
    require_single_thread()
    impl = implementation(library)
    stale = binding_mismatch(frozen["implementation"], impl)
    if stale or impl["dirty"]:
        raise ProtocolError(f"QUAL implementation differs from the frozen one: {stale or 'dirty tree'}")
    cpu = numerics.cpu()
    if not {"avx2", "fma"} <= set(cpu["flags"]) or numerics.blas_core() == "unknown":
        raise ProtocolError("QUAL needs an AVX2+FMA host and an OpenBLAS-backed NumPy (robustness kernels)")
    choices = frozen["choices"]
    per_world, determinism, transplant_probe, verdict = _verdict(library, choices)
    primary = plain(_comparable(verdict))   # gate M: each robustness child recomputes exactly this
    robustness = {}
    for core in ROBUST_CORES:
        child = _child(["robust", "--library", str(library), "--choices", json.dumps(choices)], core)
        robustness[core] = {"core_in_effect": child["core"], "threads": child["threads"],
                            "identical": child["core"] == core and child["threads"] == 1
                            and child["comparable"] == primary,
                            "comparable": child["comparable"]}
    robust_ok = all(r["identical"] for r in robustness.values())
    gates = dict(verdict["gates"], M_numerical_robustness=robust_ok)
    worlds = list(per_world)
    sel = choices["selected"]["candidate"]
    final = X.disposition(verdict["validity"], verdict["mechanics"], gates,
                          [per_world[w]["P"]["mechanisms"][sel] for w in worlds],
                          [per_world[w]["P"]["mechanisms"]["M0"] for w in worlds],
                          [per_world[w]["P"]["mechanisms"]["M1"] for w in worlds], SPEC["thresholds"])
    body = {"experiment": NAME, "split": "QUAL", "world_ids": worlds, "world_count": len(worlds),
            "frozen_digest": frozen_record["digest"], **digests, "implementation": impl, "choices": choices,
            "per_world": per_world, "determinism": determinism, "transplant_probe": transplant_probe,
            "robustness": robustness, "validity": verdict["validity"], "mechanics": verdict["mechanics"],
            "gates": gates, "pre_robustness_verdict": primary, "medians": verdict["medians"],
            "counts": verdict["counts"], "native_feasibility_evidence": verdict["native_feasibility_evidence"],
            "consolidation_interface": verdict["consolidation_interface"],
            "disposition": final["disposition"], "outcome": final["outcome"],
            "qualifiers": X.qualifiers(per_world, choices, SPEC["thresholds"], robust_ok),
            "limitations": LIMITATIONS}
    return {"digest": write(QUAL_PATH, "qual", body), "disposition": final["disposition"], "outcome": final["outcome"],
            "gates": gates, "validity": verdict["validity"], "mechanics": verdict["mechanics"]}


# --- children ------------------------------------------------------------------------------------------------------


def probe(library: Path, wid: str, choices: dict) -> dict:
    return plain(X.determinism_digest(lambda: _lab(library), wid, choices))


def transplant(library: Path, wid: str, choices: dict, state: Path) -> dict:
    lab = _lab(library)
    world = lab.world(wid, choices["hidden_gain"])
    mech, W, epoch = E.deserialize(state.read_bytes(), lab.idx, T.uninformed_rng(SPEC, wid, "B"))
    cont = lab.candidate_run(world, choices["selected"]["candidate"], start=("B", mech, W, epoch))
    return {s: {"W_digest": cont["stages"][s]["W_digest"], "state_digest": cont["stages"][s]["state_digest"]}
            for s in ("C", "D")}


def robust(library: Path, choices: dict) -> dict:
    _, _, _, verdict = _verdict(library, choices)
    return {"core": numerics.blas_core(), "threads": numerics.blas_threads(), "comparable": plain(_comparable(verdict))}


def status() -> dict:
    out = {"experiment": NAME, **spec_digests(SPEC)}
    for name, path, kind in (("dev", DEV_PATH, "dev"), ("frozen", FROZEN_PATH, "frozen"), ("qual", QUAL_PATH, "qual")):
        out[name] = load(path, kind)["digest"] if path.exists() else None
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research.ecs_retention_r2.run")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dev", "qual"):
        sub.add_parser(name).add_argument("--library", required=True, type=Path)
    f = sub.add_parser("freeze")
    f.add_argument("--commit", required=True)
    f.add_argument("--library", required=True, type=Path)
    sub.add_parser("status")
    for name in ("probe", "transplant", "robust"):
        p = sub.add_parser(name)
        p.add_argument("--library", required=True, type=Path)
        p.add_argument("--choices", required=True)
        if name != "robust":
            p.add_argument("--world", required=True)
        if name == "transplant":
            p.add_argument("--state", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "dev":
        result = dev(args.library.resolve())
    elif args.command == "freeze":
        result = freeze(args.commit, args.library.resolve())
    elif args.command == "qual":
        result = qual(args.library.resolve())
    elif args.command == "probe":
        result = probe(args.library.resolve(), args.world, json.loads(args.choices))
    elif args.command == "transplant":
        result = transplant(args.library.resolve(), args.world, json.loads(args.choices), args.state)
    elif args.command == "robust":
        result = robust(args.library.resolve(), json.loads(args.choices))
    else:
        result = status()
    sys.stdout.write(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
