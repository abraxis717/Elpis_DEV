"""Cognitive R0 laboratory command line. RESEARCH_ONLY.

Run from the repository root with ``PYTHONPATH=src`` and a native build::

    python -m research.ecs_cognition_r0.run dev    --library <libelpis_ecsg_math.so>
    python -m research.ecs_cognition_r0.run freeze --commit <sha of the commit holding the lab source>
    python -m research.ecs_cognition_r0.run qual   --library <libelpis_ecsg_math.so>
    python -m research.ecs_cognition_r0.run status

Guards: ``dev`` refuses once the experiment is frozen. ``freeze`` requires DEV
evidence produced by the current laboratory source and writes once. ``qual``
refuses unless the laboratory source equals the frozen source digest, runs
once, and writes its evidence exclusively; a failure is recorded as it is.

``replay`` and ``retrain`` are the clean-process probes QUAL launches: replay
gets only snapshot bytes and the frozen specification (it regenerates
evaluation inputs, never targets); retrain gets only the specification.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from elpis.ECS_G.cognition import CognitiveCore
from elpis.ECS_G.native import ECSGLibrary

from . import experiment as E
from . import task
from .protocol import ROOT, ProtocolError, digest, head, load, numerical_profile, source_digest, world_ids, write

NAME = f"{E.SPEC['name']}.v{E.SPEC['version']}"
SPEC_PATH = ROOT / "specs" / f"{NAME}.spec.json"
DEV_PATH = ROOT / "evidence" / "dev" / f"{NAME}.dev.json"
FROZEN_PATH = ROOT / "frozen" / f"{NAME}.frozen.json"
QUAL_PATH = ROOT / "evidence" / "qual" / f"{NAME}.qual.json"
REPO = ROOT.parents[1]

LIMITATIONS = [
    "Synthetic regression task (SEMANTICS=NONE): a realizable cubic teacher, d=6, N=36, 64 experience rows; "
    "no claim about language, meaning or general cognition.",
    "One ECS_G kernel and its G1 full-batch gradient recurrence at the Branch36 reference rate; neither the "
    "cubic kernel, S3, nor gradient descent is promoted beyond this regime.",
    "Sequential learning of a second experience is measured, not engineered: interference is reported as found.",
    "Bitwise reproduction is claimed for one build and numerical profile, not across platforms.",
]


def library(path: str | None) -> Path:
    if path:
        return Path(path).resolve()
    build = Path(os.environ.get("ELPIS_NATIVE_BUILD", REPO / "build"))
    found = sorted(build.rglob("libelpis_ecsg_math.so"))
    if not found:
        raise ProtocolError("libelpis_ecsg_math.so not found; pass --library or set ELPIS_NATIVE_BUILD")
    return found[0].resolve()


def _api(path: Path) -> ECSGLibrary:
    return ECSGLibrary(ctypes.CDLL(str(path)))


def dev(lib: Path) -> dict:
    if FROZEN_PATH.exists():
        raise ProtocolError("frozen: DEV evidence is closed")
    api, spec = _api(lib), E.SPEC
    worlds = world_ids("DEV", spec["dev_worlds"])
    curves = {wid: E.dev_curve(api, spec, wid) for wid in worlds}
    steps = E.choose_steps(spec, curves)
    informational = {}
    for wid in worlds:
        metrics, checks, _ = E.evaluate_world(api, spec, wid, steps)
        informational[wid] = {"metrics": metrics, "checks": checks, "gates": E.gates_for_world(metrics)}
    body = {"experiment": NAME, "split": "DEV", "spec": spec, "spec_digest": digest("spec", spec),
            "worlds": list(worlds), "curves": curves, "dev_choice_rule": E.DEV_CHOICE, "choices": {"steps": steps},
            "informational_evaluation_at_choice": informational,
            "informational_aggregate": E.aggregate_gates({w: v["metrics"] for w, v in informational.items()}),
            "source_digest": source_digest(), "source_commit": head(), "numerical_profile": numerical_profile()}
    write(SPEC_PATH, "spec", {"spec": spec, "spec_digest": digest("spec", spec)}, exclusive=False)
    out = write(DEV_PATH, "dev", body, exclusive=False)
    print(f"DEV: steps={steps} curves={json.dumps(curves)}")
    return {"digest": out, "steps": steps}


def freeze(commit: str) -> str:
    dev_record = load(DEV_PATH, "dev")
    body = dev_record["body"]
    if body["source_digest"] != source_digest():
        raise ProtocolError("DEV evidence was produced by different laboratory source; rerun dev")
    if body["spec"] != E.SPEC:
        raise ProtocolError("DEV specification differs from the laboratory specification")
    frozen = {"experiment": NAME, "spec": E.SPEC, "spec_digest": digest("spec", E.SPEC),
              "choices": body["choices"], "dev_choice_rule": E.DEV_CHOICE, "pass_rule": E.PASS_RULE,
              "learning_criterion": E.LEARNING_CRITERION, "dev_evidence_digest": dev_record["digest"],
              "source_digest": body["source_digest"], "source_commit": commit,
              "qual_worlds": list(world_ids("QUAL", E.SPEC["qual_worlds"]))}
    out = write(FROZEN_PATH, "frozen", frozen, exclusive=True)
    print(f"frozen {out}")
    return out


def _probe(args: list[str]) -> dict:
    env = {"PYTHONPATH": str(REPO / "src") + os.pathsep + str(REPO), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-m", "research.ecs_cognition_r0.run", *args], cwd=REPO, env=env,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise ProtocolError("clean-process probe failed: " + result.stderr[-2000:])
    return json.loads(result.stdout)


def qual(lib: Path) -> dict:
    frozen_record = load(FROZEN_PATH, "frozen")
    frozen = frozen_record["body"]
    if QUAL_PATH.exists():
        raise ProtocolError("QUAL already ran once; a repair needs a new experiment version")
    if frozen["source_digest"] != source_digest():
        raise ProtocolError("laboratory source changed after freeze; QUAL refused")
    if frozen["pass_rule"] != E.PASS_RULE or frozen["spec"] != E.SPEC:
        raise ProtocolError("pass rule or specification differs from the frozen authority")
    api, spec, steps = _api(lib), frozen["spec"], frozen["choices"]["steps"]
    worlds = frozen["qual_worlds"]
    per_world, gates, checks, persistence = {}, {}, {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for wid in worlds:
            metrics, world_checks, learned = E.evaluate_world(api, spec, wid, steps)
            snap = Path(tmp) / f"{wid}.snapshot"
            snap.write_bytes(learned)
            replay = _probe(["replay", "--world", wid, "--snapshot", str(snap), "--library", str(lib),
                             "--spec", str(FROZEN_PATH)])
            persistence[wid] = {"in_process": metrics["learned_responses_A"], "clean_process": replay["responses"],
                                "bitwise": replay["responses"] == metrics["learned_responses_A"]}
            world_gates = E.gates_for_world(metrics)
            world_gates["C_persistence"] = persistence[wid]["bitwise"]
            per_world[wid], gates[wid], checks[wid] = metrics, world_gates, world_checks
    first = worlds[0]
    again, _, _ = E.evaluate_world(api, spec, first, steps)
    clean = _probe(["retrain", "--world", first, "--steps", str(steps), "--library", str(lib),
                    "--spec", str(FROZEN_PATH)])
    determinism = {
        "in_process_repeat_equal": (again["learned_snapshot"], again["receipt_A"], again["receipt_B"])
        == (per_world[first]["learned_snapshot"], per_world[first]["receipt_A"], per_world[first]["receipt_B"]),
        "clean_process_equal": (clean["snapshot"], clean["receipt"])
        == (per_world[first]["learned_snapshot"], per_world[first]["receipt_A"]),
    }
    aggregate = E.aggregate_gates(per_world)
    leak_guard = {"core_slots_are_state_and_rate": E.slots_are_state_and_rate(),
                  "clean_process_replay_bitwise_all_worlds": all(p["bitwise"] for p in persistence.values())}
    gate_table = {
        "A_learning": all(g["A_learning"] for g in gates.values()) and aggregate["A_median_nmse_learned_le_0.25"],
        "B_state_causality": all(g["B_state_causality"] for g in gates.values()),
        "C_persistence": all(g["C_persistence"] for g in gates.values()),
        "D_continual": all(g["D_continual"] for g in gates.values()),
        "E_controls": (all(g["E_controls"] for g in gates.values()) and aggregate["E_median_shuffled_ge_4x_learned"]
                       and all(leak_guard.values())),
        "F_determinism": all(determinism.values()),
        "G_microstate_authority": all(g["G_microstate_authority"] for g in gates.values()),
    }
    mechanics_ok = all(all(c.values()) for c in checks.values())
    mechanics = "MECHANICS_PASS" if mechanics_ok else "MECHANICS_FAIL"
    if not mechanics_ok:
        scientific = "NOT_TESTED"
    else:
        scientific = "SUPPORTS_UNDER_FROZEN_SYNTHETIC_REGIME" if all(gate_table.values()) else "DID_NOT_SUPPORT"
    retention = {wid: {"retention": m["retention"], "class": m["retention_class"], "catastrophic": m["catastrophic"]}
                 for wid, m in per_world.items()}
    body = {"experiment": NAME, "split": "QUAL", "frozen_digest": frozen_record["digest"],
            "spec_digest": frozen["spec_digest"], "source_digest": source_digest(), "source_commit": head(),
            "numerical_profile": numerical_profile(), "steps": steps, "worlds": worlds, "per_world": per_world,
            "per_world_gates": gates, "aggregate": aggregate, "persistence": persistence,
            "determinism": determinism, "leak_guard": leak_guard, "mechanics_checks": checks,
            "gates": gate_table, "mechanics": mechanics, "scientific": scientific,
            "retention_measured": retention, "pass_rule": frozen["pass_rule"], "limitations": LIMITATIONS}
    out = write(QUAL_PATH, "qual", body, exclusive=True)
    print(f"QUAL {mechanics} {scientific} gates={json.dumps(gate_table)} digest={out}")
    return body


def status() -> None:
    frozen = FROZEN_PATH.exists()
    q = load(QUAL_PATH, "qual")["body"] if QUAL_PATH.exists() else None
    print(f"{NAME}: frozen={frozen} qual={'-' if q is None else q['mechanics'] + ' ' + q['scientific']}")


def _replay(args) -> None:
    """Clean process: snapshot bytes + specification -> responses on evaluation inputs. No targets exist here."""
    spec = load(Path(args.spec), "frozen")["body"]["spec"]
    rows = task.query_inputs(spec, args.world, "A").tolist()
    with CognitiveCore.restore(_api(Path(args.library)), Path(args.snapshot).read_bytes(),
                               learning_rate=spec["learning_rate"]) as core:
        print(json.dumps({"responses": E.responses_digest(core.query(rows))}))


def _retrain(args) -> None:
    """Clean process: specification -> initialization -> experience A -> snapshot digest and receipt."""
    spec = load(Path(args.spec), "frozen")["body"]["spec"]
    core, _, _, receipt = E.learn_a(_api(Path(args.library)), spec, args.world, args.steps)
    with core:
        print(json.dumps({"snapshot": E.snapshot_digest(core.snapshot()), "receipt": receipt.digest}))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research.ecs_cognition_r0.run")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("dev", "qual"):
        sub.add_parser(name).add_argument("--library")
    sub.add_parser("freeze").add_argument("--commit", required=True)
    sub.add_parser("status")
    r = sub.add_parser("replay")
    for flag in ("--world", "--snapshot", "--library", "--spec"):
        r.add_argument(flag, required=True)
    t = sub.add_parser("retrain")
    for flag in ("--world", "--library", "--spec"):
        t.add_argument(flag, required=True)
    t.add_argument("--steps", type=int, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "dev":
        dev(library(args.library))
    elif args.cmd == "freeze":
        freeze(args.commit)
    elif args.cmd == "qual":
        qual(library(args.library))
    elif args.cmd == "status":
        status()
    elif args.cmd == "replay":
        _replay(args)
    else:
        _retrain(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
