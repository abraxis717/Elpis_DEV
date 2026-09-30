"""Laboratory command line: ``dev``, ``freeze``, ``qual``, ``status``. RESEARCH_ONLY.

Run from the repository root with ``PYTHONPATH=src``::

    python -m research.ecs_dynamics.run dev
    python -m research.ecs_dynamics.run freeze --commit <sha of the commit holding the lab source>
    python -m research.ecs_dynamics.run qual
    python -m research.ecs_dynamics.run status

Guards:

* ``dev`` refuses to overwrite DEV evidence for an experiment that is
  already frozen. DEV closes at freeze.
* ``freeze`` requires DEV evidence produced by the current laboratory source
  (same source digest). It writes each frozen specification once; a changed
  experiment needs a new version.
* ``qual`` refuses unless the current laboratory source digest equals the
  frozen one, and it runs each frozen experiment at most once. An existing
  QUAL result file is never overwritten; a failure is recorded as it is, and
  a repair needs a new experiment version.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .experiments import EXPERIMENTS, ORDER
from .results import RESULT_DOMAIN, ExperimentResult
from .spec import (FreezeRegistry, FrozenSpec, SpecError, canonical_json, numerical_profile, research_digest,
                   source_digest, world_ids)

ROOT = Path(__file__).resolve().parent
DEV_DIR = ROOT / "evidence" / "dev"
QUAL_DIR = ROOT / "evidence" / "qual"
FROZEN_DIR = ROOT / "frozen"
SPEC_DIR = ROOT / "specs"


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _head() -> str:
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--", str(ROOT / "*.py"))
    return head + ("+uncommitted-lab-source" if dirty else "")


def _write(path: Path, payload: dict, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(json.loads(canonical_json(payload)), indent=1, sort_keys=True).encode("ascii") + b"\n"
    with open(path, "xb" if exclusive else "wb") as fh:
        fh.write(data)


def _load_result(path: Path) -> dict:
    data = json.loads(path.read_bytes())
    if research_digest(RESULT_DOMAIN, data["result"]) != data["digest"]:
        raise SpecError(f"{path.name}: recorded digest does not match its content")
    return data


def _result(exp, spec, split, metrics, *, frozen_digest=None, findings=(), mechanics="MECHANICS_PASS",
            scientific="NOT_TESTED") -> ExperimentResult:
    return ExperimentResult(
        experiment=exp.name, version=spec.version, split=split, spec_digest=spec.digest, frozen_digest=frozen_digest,
        source_digest=source_digest(), source_commit=_head(), model_family=spec.model_family,
        numerical_profile=numerical_profile(), world_ids=world_ids(spec, split), coarse_representation=spec.coarse_observable,
        target=spec.target, intervention_policy=spec.intervention_protocol, evidence=exp.evidence, metrics=metrics,
        findings=tuple(findings), pass_rule=exp.pass_rule, mechanics=mechanics, scientific=scientific,
        limitations=exp.limitations, sources=exp.sources)


def dev(names=None) -> dict:
    registry = FreezeRegistry(FROZEN_DIR)
    context, digests = {}, {}
    for name in ORDER:
        exp = EXPERIMENTS[name]
        missing = [d for d in exp.depends_on if d not in context]
        if names and name not in names and not any(name in EXPERIMENTS[n].depends_on for n in names):
            continue
        if missing:
            raise SpecError(f"{name} needs DEV choices of {missing}")
        spec = exp.spec
        choices, metrics = exp.dev(spec, context)
        context[name] = choices
        if registry.path(spec).exists():
            print(f"{name}: frozen; DEV evidence is closed, not rewritten", file=sys.stderr)
            continue
        result = _result(exp, spec, "DEV", dict(metrics, choices=choices))
        _write(SPEC_DIR / f"{name}.v{spec.version}.spec.json", {"spec": spec.as_dict(), "digest": spec.digest}, False)
        _write(DEV_DIR / f"{name}.v{spec.version}.dev.json", {"result": result.as_dict(), "digest": result.digest}, False)
        digests[name] = result.digest
        print(f"{name}: DEV done, choices={json.dumps(json.loads(canonical_json(choices)))[:200]}")
    return digests


def freeze(commit: str) -> dict:
    registry = FreezeRegistry(FROZEN_DIR)
    current = source_digest()
    out = {}
    for name in ORDER:
        exp = EXPERIMENTS[name]
        data = _load_result(DEV_DIR / f"{name}.v{exp.spec.version}.dev.json")
        if data["result"]["source_digest"] != current:
            raise SpecError(f"{name}: DEV evidence was produced by different laboratory source; rerun dev")
        choices = data["result"]["metrics"]["choices"]
        spec = exp.spec.with_choices(parameters=dict(exp.spec.parameters, **choices))
        frozen = FrozenSpec(spec, choices, data["digest"], current, commit, exp.pass_rule)
        out[name] = registry.freeze(frozen)
        print(f"{name}: frozen {out[name]}")
    return out


def qual() -> dict:
    registry = FreezeRegistry(FROZEN_DIR)
    current = source_digest()
    summary = {}
    for name in ORDER:
        exp = EXPERIMENTS[name]
        frozen = registry.load(name, exp.spec.version)
        path = QUAL_DIR / f"{name}.v{exp.spec.version}.qual.json"
        if path.exists():
            print(f"{name}: QUAL already ran once; not rerun (a repair needs a new version)", file=sys.stderr)
            summary[name] = _load_result(path)["result"]["scientific"]
            continue
        if frozen.source_digest != current:
            raise SpecError(f"{name}: laboratory source changed after freeze; QUAL refused")
        metrics, findings, checks, passed = exp.qual(frozen.spec, frozen.dev_choices, {})
        mechanics = "MECHANICS_PASS" if all(checks.values()) else "MECHANICS_FAIL"
        if mechanics == "MECHANICS_FAIL":
            scientific = "NOT_TESTED"
        elif passed is None:
            scientific = "DESCRIPTIVE_ONLY"
        else:
            scientific = "SUPPORTS_UNDER_FROZEN_SYNTHETIC_REGIME" if passed else "DID_NOT_SUPPORT"
        result = _result(exp, frozen.spec, "QUAL", dict(metrics, mechanics_checks=checks), frozen_digest=frozen.digest,
                         findings=findings, mechanics=mechanics, scientific=scientific)
        _write(path, {"result": result.as_dict(), "digest": result.digest}, True)
        summary[name] = scientific
        print(f"{name}: QUAL {mechanics} {scientific}")
    return summary


def status() -> None:
    registry = FreezeRegistry(FROZEN_DIR)
    for name in ORDER:
        exp = EXPERIMENTS[name]
        frozen = registry.path(exp.spec).exists()
        q = QUAL_DIR / f"{name}.v{exp.spec.version}.qual.json"
        disp = _load_result(q)["result"] if q.exists() else None
        print(f"{name:26s} frozen={frozen!s:5s} qual={'-' if disp is None else disp['mechanics'] + ' ' + disp['scientific']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research.ecs_dynamics.run")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dev")
    d.add_argument("names", nargs="*")
    f = sub.add_parser("freeze")
    f.add_argument("--commit", required=True)
    sub.add_parser("qual")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    if args.cmd == "dev":
        dev(args.names or None)
    elif args.cmd == "freeze":
        freeze(args.commit)
    elif args.cmd == "qual":
        qual()
    else:
        status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
