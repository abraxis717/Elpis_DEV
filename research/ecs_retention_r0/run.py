"""Retention R0 laboratory command line. RESEARCH_ONLY.

Run from the repository root with ``PYTHONPATH=src:.`` and a native build::

    python -m research.ecs_retention_r0.run dev    --library <libelpis_ecsg_math.so>
    python -m research.ecs_retention_r0.run freeze --commit <sha of the commit holding the laboratory source>
    python -m research.ecs_retention_r0.run qual   --library <libelpis_ecsg_math.so>
    python -m research.ecs_retention_r0.run status

Guards: ``dev`` refuses once frozen and needs a clean tree. ``freeze`` needs
DEV evidence produced by the current laboratory source and implementation,
and writes once. ``qual`` refuses unless frozen, the specification, laboratory
source, bound sources, library, build and numerical profile equal the frozen
ones and the tree is clean; it runs once and writes its evidence
exclusively; a failure is recorded as it is.

``probe`` is the clean-process determinism probe QUAL launches: it gets only
the frozen specification and choices.
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import ctypes  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

from elpis.ECS_G.native import ECSGLibrary  # noqa: E402

from . import experiment as X  # noqa: E402
from .protocol import (REPO, ROOT, ProtocolError, binding_mismatch, implementation, load, load_spec,  # noqa: E402
                       source_digest_at, spec_digests, world_ids, write)

SPEC = load_spec()
NAME = f"{SPEC['name']}.v{SPEC['version']}"
DEV_PATH = ROOT / "evidence" / "dev" / f"{NAME}.dev.json"
FROZEN_PATH = ROOT / "frozen" / f"{NAME}.frozen.json"
QUAL_PATH = ROOT / "evidence" / "qual" / f"{NAME}.qual.json"

LIMITATIONS = [
    "Synthetic regression task (SEMANTICS=NONE): cubic ridge teachers, d=6, N=36, 64 experience rows per "
    "experience; no claim about language, meaning or general cognition.",
    "One ECS_G kernel and its G1 recurrence at the Branch36 reference rate and the Cognitive R0 step budget.",
    "Candidate learning runs the canonical native G1 step plus a laboratory-side consolidation term; a promoted "
    "mechanism would need its own native implementation and qualification.",
    "Bitwise reproduction is claimed for one build and numerical profile, not across platforms.",
]


def _lab(library: Path) -> X.Lab:
    return X.Lab(ECSGLibrary(ctypes.CDLL(str(library))), SPEC)


def dev(library: Path) -> dict:
    if FROZEN_PATH.exists():
        raise ProtocolError("frozen: DEV evidence is closed")
    impl = implementation(library)
    if impl["dirty"]:
        raise ProtocolError("DEV needs a clean tree (commit the laboratory first)")
    lab = _lab(library)
    worlds = world_ids(SPEC, "DEV")
    result = X.dev(lab, worlds)
    body = {"experiment": NAME, "split": "DEV", "worlds": list(worlds), **spec_digests(SPEC),
            "implementation": impl, "dev_rules": SPEC["dev_rules"], **result}
    return {"digest": write(DEV_PATH, "dev", body), "choices": result["choices"],
            "replay_infeasible_on_dev": result["replay_infeasible_on_dev"]}


def freeze(commit: str, library: Path) -> dict:
    record = load(DEV_PATH, "dev")
    body = record["body"]
    impl = implementation(library)
    stale = binding_mismatch(body["implementation"], impl)
    if stale:
        raise ProtocolError(f"DEV evidence was produced by a different implementation: {stale}")
    if body["spec"] != spec_digests(SPEC)["spec"]:
        raise ProtocolError("specification changed after DEV")
    if source_digest_at(commit) != body["implementation"]["lab_source_digest"]:
        raise ProtocolError(f"commit {commit} does not hold the laboratory source that produced DEV")
    frozen = {"experiment": NAME, **spec_digests(SPEC), "lab_commit": commit,
              "lab_source_digest": body["implementation"]["lab_source_digest"],
              "dev_evidence_digest": record["digest"], "dev_worlds": body["worlds"], "choices": body["choices"],
              "implementation": body["implementation"], "pass_rule": SPEC["pass_rule"],
              "qual_worlds": list(world_ids(SPEC, "QUAL"))}
    return {"digest": write(FROZEN_PATH, "frozen", frozen)}


def _probe(library: Path, world: str, choices: dict) -> dict:
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-m", "research.ecs_retention_r0.run", "probe", "--library", str(library),
                          "--world", world, "--choices", json.dumps(choices)],
                         capture_output=True, text=True, cwd=REPO, env=env, check=True)
    return json.loads(out.stdout)


def qual(library: Path) -> dict:
    if QUAL_PATH.exists():
        raise ProtocolError("QUAL already ran; a repair is a new experiment version")
    frozen_record = load(FROZEN_PATH, "frozen")
    frozen = frozen_record["body"]
    digests = spec_digests(SPEC)
    if any(frozen[k] != digests[k] for k in ("spec", "pass_rule", "candidates_sha256")):
        raise ProtocolError("specification, pass rule or candidate definitions changed after the freeze")
    impl = implementation(library)
    stale = binding_mismatch(frozen["implementation"], impl)
    if stale or impl["dirty"]:
        raise ProtocolError(f"QUAL implementation differs from the frozen one: {stale or 'dirty tree'}")
    lab = _lab(library)
    choices = frozen["choices"]
    worlds = world_ids(SPEC, "QUAL")
    per_world = {wid: X.qual_world(lab, wid, choices) for wid in worlds}
    first = worlds[0]
    runs = [X.determinism_digest(lab, first, choices), X.determinism_digest(lab, first, choices),
            _probe(library, first, choices)]
    recorded = {"W_A": per_world[first]["mechanics"]["shared_W_A_digest"],
                "W_AB": per_world[first]["mechanisms"]["selected"]["W_digest"]}
    identical = (all(r == runs[0] for r in runs) and runs[0]["W_A"] == recorded["W_A"]
                 and runs[0]["W_AB"] == recorded["W_AB"])
    determinism = {"world": first, "runs": runs, "recorded": recorded, "identical": identical}
    inputs = X.consolidation_inputs()
    verdict = X.gates(per_world, SPEC, determinism, inputs, choices)
    body = {"experiment": NAME, "split": "QUAL", "worlds": list(worlds), "frozen_digest": frozen_record["digest"],
            **digests, "implementation": impl, "choices": choices, "per_world": per_world,
            "determinism": determinism, "consolidation_inputs": inputs,
            "flops_per_step": X.E.flops_per_step(lab.dim, lab.width, SPEC["regime"]["train_rows"], lab.idx),
            **verdict, "limitations": LIMITATIONS}
    return {"digest": write(QUAL_PATH, "qual", body), "disposition": verdict["disposition"],
            "outcome": verdict["outcome"], "gates": verdict["gates"], "mechanics": verdict["mechanics"]}


def status() -> dict:
    out = {"experiment": NAME, **spec_digests(SPEC)}
    for name, path, kind in (("dev", DEV_PATH, "dev"), ("frozen", FROZEN_PATH, "frozen"), ("qual", QUAL_PATH, "qual")):
        out[name] = load(path, kind)["digest"] if path.exists() else None
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research.ecs_retention_r0.run")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dev", "qual"):
        sub.add_parser(name).add_argument("--library", required=True, type=Path)
    f = sub.add_parser("freeze")
    f.add_argument("--commit", required=True)
    f.add_argument("--library", required=True, type=Path)
    sub.add_parser("status")
    p = sub.add_parser("probe")
    p.add_argument("--library", required=True, type=Path)
    p.add_argument("--world", required=True)
    p.add_argument("--choices", required=True)
    args = parser.parse_args(argv)
    if args.command == "dev":
        result = dev(args.library.resolve())
    elif args.command == "freeze":
        result = freeze(args.commit, args.library.resolve())
    elif args.command == "qual":
        result = qual(args.library.resolve())
    elif args.command == "probe":
        result = X.determinism_digest(_lab(args.library.resolve()), args.world, json.loads(args.choices))
    else:
        result = status()
    json.dump(result, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
