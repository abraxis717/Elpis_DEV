"""ECS_C Recovery Q0: what exactly is executing, and whether it is exactly what was frozen.

``observe`` measures the executing object itself:
- the git checkout this instrument runs from (commit, tree, cleanliness);
- every file of the instrument and the specification;
- the subject source files;
- the installed prefix the run deploys from (the testing host, the production host, every library);
- the interpreter (which must run isolated and must not have imported Elpis: this instrument drives the native host
  only as separate processes);
- the toolchain, the platform and the filesystem of the work root;
- the premise.

``compare`` returns every difference from a freeze record. A QUAL run starts only when there is none, and nothing is
created before that (``runner.py run``).

**The premise.** The crash points exist only in the testing builds. That the production build carries the same
protocol is a premise bound to source, not something a run can observe. It is bound three ways, all recorded here and
compared at the run:
- P1, binding: the production view of each subject file (testing-only items removed, as K1 Recovery R0 v4's
  ``premise.production``) is recorded by digest;
- P2, binaries: the production RuntimeCore exports no ``elpis_runtime_testing_*`` symbol and the testing one exports
  them;
- P3, binaries: the production host contains none of the testing host's operation names.
It establishes that the frozen production object is built from the frozen protocol source without its fault hooks.
It does not prove the two builds equivalent.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

FREEZE_SCHEMA = "elpis.ecs-c-recovery-q0.freeze.v1"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SPECIFICATION = "docs/qualification/ECS_C_RECOVERY_Q0.md"
# Excluded from the instrument digest: what the qualification produces or what records the freeze itself.
_NOT_INSTRUMENT = ("evidence", "FREEZE.json", "__pycache__")
# The subject the specification names, bound file by file.
SUBJECT = ("native/ECS_C/authority/src", "native/runtime/src/authority.rs", "native/runtime/src/durable.rs",
           "native/runtime/src/checkpoint.rs", "native/runtime/src/core.rs", "native/runtime/src/ffi.rs",
           "native/runtime/include/elpis/runtime.h", "native/runtime/host/src")
# The files whose production view the premise binds (P1).
PRODUCTION_VIEW = ("native/runtime/src/authority.rs", "native/runtime/src/durable.rs",
                   "native/runtime/src/checkpoint.rs", "native/runtime/src/core.rs", "native/runtime/src/ffi.rs",
                   "native/ECS_C/authority/src/store.rs", "native/ECS_C/authority/src/record.rs")
# What only the testing host carries (P3).
TESTING_HOST_MARKERS = (b"testing_durable_fault", b"testing_control_fault", b"testing_parked", b"--testing-anchor-park",
                        b"testing_write_probe")
LIBRARIES = ("libelpis_ecsg_math.so", "libelpis_ecsg_k1.so", "libelpis_ecs_memory.so", "libelpis_ecs_composite.so",
             "libelpis_hacf_observation.so", "libelpis_ecs_c_authority.so", "libelpis_runtime.so",
             "libelpis_runtime_testing.so")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(repo: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def source(repo: Path) -> dict:
    status = _git(repo, "status", "--porcelain", "--untracked-files=all").splitlines()
    return {"commit": _git(repo, "rev-parse", "HEAD").strip(), "tree": _git(repo, "rev-parse", "HEAD^{tree}").strip(),
            "clean": not status, "status": status[:20]}


def instrument(repo: Path) -> dict:
    root = repo / HERE.relative_to(REPO)
    files = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and not (set(rel.parts) & set(_NOT_INSTRUMENT)):
            files[rel.as_posix()] = sha256(p)
    spec = repo / SPECIFICATION
    return {"files": files, "specification": sha256(spec) if spec.exists() else None}


def subject(repo: Path) -> dict:
    out = {}
    for rel in SUBJECT:
        p = repo / rel
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix in (".rs", ".c", ".h"):
                    out[f.relative_to(repo).as_posix()] = sha256(f)
        else:
            out[rel] = sha256(p) if p.exists() else None
    return out


def _k1r_premise(repo: Path):
    path = repo / "research/qualification/k1_recovery_r0_v4/premise.py"
    spec = importlib.util.spec_from_file_location("k1r4_premise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _symbols(path: Path) -> list:
    out = subprocess.run(["nm", "-D", "--defined-only", str(path)], capture_output=True, text=True)
    return sorted(line.split()[-1] for line in out.stdout.splitlines() if line.split())


def premise(repo: Path, prefix: Path) -> dict:
    production = _k1r_premise(repo).production
    views = {rel: hashlib.sha256(production((repo / rel).read_text()).encode()).hexdigest()
             for rel in PRODUCTION_VIEW}
    lib = prefix / "lib" / "elpis"
    problems = []
    prod = [s for s in _symbols(lib / "libelpis_runtime.so") if s.startswith("elpis_runtime_testing_")]
    test = [s for s in _symbols(lib / "libelpis_runtime_testing.so") if s.startswith("elpis_runtime_testing_")]
    if prod:
        problems.append(f"P2: the production RuntimeCore exports {prod}")
    if not test:
        problems.append("P2: the testing RuntimeCore exports no testing entry point")
    host = (prefix / "bin" / "elpis").read_bytes()
    testing_host = (prefix / "bin" / "elpis_testing").read_bytes()
    carried = [m.decode() for m in TESTING_HOST_MARKERS if m in host]
    if carried:
        problems.append(f"P3: the production host carries {carried}")
    if not all(m in testing_host for m in TESTING_HOST_MARKERS):
        problems.append("P3: the testing host lacks a testing operation")
    return {"production_views": views, "testing_symbols": test, "problems": problems}


def installed(prefix: Path) -> dict:
    lib = prefix / "lib" / "elpis"
    return {"bin": {n: sha256(prefix / "bin" / n) if (prefix / "bin" / n).is_file() else None
                    for n in ("elpis", "elpis_testing")},
            "lib": {n: sha256(lib / n) if (lib / n).is_file() else None for n in LIBRARIES}}


def python() -> dict:
    return {"executable": os.path.realpath(sys.executable), "version": sys.version,
            "implementation": platform.python_implementation(), "isolated": sys.flags.isolated,
            "no_bytecode": sys.flags.dont_write_bytecode,
            "elpis_imported": sorted(n for n in sys.modules if n == "elpis" or n.startswith("elpis."))}


def _tool(argv) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, check=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unavailable"


def machine(work_root: Path) -> dict:
    return {"toolchain": {"cc": _tool(["cc", "--version"]), "c++": _tool(["c++", "--version"]),
                          "cmake": _tool(["cmake", "--version"]), "rustc": _tool(["rustc", "--version"]),
                          "cargo": _tool(["cargo", "--version"])},
            "platform": {"system": platform.system(), "kernel": os.uname().release, "machine": platform.machine(),
                         "libc": "-".join(platform.libc_ver())},
            "filesystem": {"work_root": str(work_root), "type": _tool(["stat", "-f", "-c", "%T", str(work_root)])}}


def observe(*, prefix: Path, work_root: Path, repo: Path = REPO) -> dict:
    return {"source": source(repo), "instrument": instrument(repo), "subject": subject(repo),
            "premise": premise(repo, prefix), "installed": installed(prefix), "python": python(),
            **machine(work_root)}


def _preconditions(observed: dict) -> list:
    problems = []
    if not observed["source"]["clean"]:
        problems.append(f"the checkout is not clean: {observed['source']['status']}")
    py = observed["python"]
    if py["isolated"] != 1:
        problems.append("the interpreter is not in isolated mode (python -I)")
    if py["elpis_imported"]:
        problems.append(f"the instrument imported Elpis modules: {py['elpis_imported']}")
    missing = [n for group in observed["installed"].values() for n, d in group.items() if d is None]
    if missing:
        problems.append(f"the installed prefix lacks {missing}")
    problems += observed["premise"]["problems"]
    return problems


def record(*, prefix: Path, work_root: Path, repo: Path = REPO) -> dict:
    """The freeze record of the executing checkout and prefix (``runner.py freeze``)."""
    observed = observe(prefix=prefix, work_root=work_root, repo=repo)
    problems = _preconditions(observed)
    if problems:
        raise SystemExit("freeze refused: " + "; ".join(problems))
    plan = json.loads((HERE / "plan.json").read_text())
    return {"schema": FREEZE_SCHEMA, "specification": SPECIFICATION, "inputs": "qual", "plan": plan["schema"],
            "gates": ["G1", "G2", "G3", "G4", "G5", "G6"],
            "evidence_schema": {"record": "elpis.ecs-c-recovery-q0.record.v1",
                                "run": "elpis.ecs-c-recovery-q0.run.v1",
                                "verdict": "elpis.ecs-c-recovery-q0.verdict.v1", "freeze": FREEZE_SCHEMA},
            **observed}


_COMPARED = (("source", "commit"), ("source", "tree"), ("instrument",), ("subject",), ("premise",), ("installed",),
             ("python", "executable"), ("python", "version"), ("python", "implementation"), ("toolchain",),
             ("platform",), ("filesystem", "type"))


def _get(d: dict, path: tuple):
    for key in path:
        d = d[key]
    return d


def compare(freeze: dict, observed: dict) -> list:
    """Every way the executing environment differs from the freeze record (empty: it is the frozen one)."""
    if freeze.get("schema") != FREEZE_SCHEMA:
        return [f"not a {FREEZE_SCHEMA} record"]
    problems = _preconditions(observed)
    for path in _COMPARED:
        try:
            want, got = _get(freeze, path), _get(observed, path)
        except (KeyError, TypeError):
            problems.append(f"{'.'.join(path)}: missing from the freeze record")
            continue
        if want != got:
            problems.append(f"{'.'.join(path)} differs from the freeze")
    return problems
