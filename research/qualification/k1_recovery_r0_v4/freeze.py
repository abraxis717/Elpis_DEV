"""K1 Recovery R0 v4: what exactly is executing, and whether it is exactly what was frozen.

A freeze record is not trusted because it is well formed. ``observe`` measures the executing object itself: the git
checkout this instrument runs from (commit, tree, cleanliness), every file of the instrument (this file, the runner,
the checker, the plan) and the specification, the subject source files, the interpreter and the ``elpis`` package
it actually imported (its origin, every loaded ``elpis.*`` module's origin, isolated mode, and the package's content
against the checkout's ``src/elpis``), the build-tree libraries, the deployed ``home/lib`` copies and their catalog
pins, the ABI the admitted libraries report, the toolchain, the platform and the filesystem of the work root.
``compare`` returns every difference from the freeze record; a QUAL run starts only when there is none, and nothing
is created or opened before that (``runner.py run``).

Isolated mode: the runner and every child run as ``python -I -B``. ``-I`` ignores every ``PYTHON*`` environment
variable (``PYTHONPATH`` included) and the script's directory and the user site: ``elpis`` resolves only through the
interpreter's own ``site-packages`` (an installed copy or an editable install's path entry). That is why the imported
package is measured (its root and its content digest), never assumed from an environment variable.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

FREEZE_SCHEMA = "elpis.k1-recovery-r0-v4.freeze.v2"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SPECIFICATION = "docs/qualification/K1_RECOVERY_R0_V4.md"
# Excluded from the instrument digest: what the qualification produces or what records the freeze itself.
_NOT_INSTRUMENT = ("evidence", "FREEZE.json", "__pycache__")
# The subject the specification names (its durability law and the API that drives it), bound file by file.
SUBJECT = ("native/runtime/src", "native/continuity/src", "native/runtime/include/elpis/runtime.h",
           "src/elpis/runtime/core.py", "src/elpis/runtime/composition.py", "src/elpis/runtime/recovery.py",
           "src/elpis/runtime/deployment.py", "src/elpis/runtime/cognition.py", "src/elpis/runtime/fuel.py",
           "src/elpis/ECS/k1.py", "src/elpis/ECS/composite.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def files_digest(root: Path, suffixes=(".py",)) -> dict:
    """``{relative path: sha256}`` of every file under ``root`` with one of ``suffixes`` (no ``__pycache__``)."""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts:
            out[p.relative_to(root).as_posix()] = sha256(p)
    return out


def tree_digest(files: dict) -> str:
    """One digest of a ``{relative path: sha256}`` map (order-independent, content- and name-bound)."""
    h = hashlib.sha256(b"elpis.k1-recovery-r0-v4.tree.v1\0")
    for name in sorted(files):
        h.update(name.encode() + b"\0" + files[name].encode() + b"\n")
    return h.hexdigest()


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
            out.update({f"{rel}/{k}": v for k, v in files_digest(p, (".rs",)).items()})
        elif p.exists():
            out[rel] = sha256(p)
        else:
            out[rel] = None
    return out


def python(repo: Path) -> dict:
    """The interpreter and the ``elpis`` package actually imported (not the one an environment variable names)."""
    import elpis

    root = Path(elpis.__file__).resolve().parent
    outside = sorted(name for name, m in list(sys.modules.items())
                     if (name == "elpis" or name.startswith("elpis.")) and getattr(m, "__file__", None)
                     and not Path(m.__file__).resolve().is_relative_to(root))
    package = files_digest(root)
    source_package = files_digest(repo / "src" / "elpis")
    return {"executable": os.path.realpath(sys.executable), "version": sys.version,
            "implementation": platform.python_implementation(), "isolated": sys.flags.isolated,
            "no_bytecode": sys.flags.dont_write_bytecode, "package_root": str(root),
            "package": tree_digest(package), "source_package": tree_digest(source_package),
            "package_files": len(package), "modules_outside_package": outside}


def native(native_build: Path) -> dict:
    from elpis.runtime.deployment import deployable_libraries

    return {rel: sha256(native_build / rel) if (native_build / rel).is_file() else None
            for _, _, rel in deployable_libraries(testing=True)}


def deployed(home: Path) -> dict:
    """The deployed copies a run admits (``home/lib``), the catalog's pins of them and the ABI the admitted libraries
    report (admission itself verifies sealed bytes against the pins)."""
    import ctypes

    from elpis.runtime.deployment import Deployment

    d = Deployment(home)
    catalog = json.loads((home / d.config["catalog"]).read_text())
    runtime = d.libraries.runtime._lib
    runtime.elpis_runtime_abi_version.restype = ctypes.c_uint32
    runtime.elpis_continuity_abi_version.restype = ctypes.c_uint32
    return {"lib": {p.name: sha256(p) for p in sorted((home / "lib").iterdir())},
            "catalog": {e["library_id"]: e["sha256"] for e in catalog["libraries"]},
            "catalog_sha256": d.config["catalog_sha256"], "deployment_abi": d.config["abi"],
            "reported_abi": {"runtime": int(runtime.elpis_runtime_abi_version()),
                             "continuity": int(runtime.elpis_continuity_abi_version()),
                             "k1": int(d.libraries.k1._k.abi_version())}}


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


def observe(*, native_build: Path, work_root: Path, repo: Path = REPO) -> dict:
    """Everything a freeze binds, measured now (``deployed`` is measured after the deployment exists)."""
    return {"source": source(repo), "instrument": instrument(repo), "subject": subject(repo),
            "python": python(repo), "native": native(native_build), **machine(work_root)}


def trial_deployment(native_build: Path, scratch: Path) -> dict:
    """What a deployment of exactly these libraries is: deployed once into ``scratch`` (which must not exist) and
    measured. The catalog is a canonical function of the libraries, so a run's own deployment must equal it."""
    from elpis.runtime.deployment import init_deployment

    scratch.mkdir(mode=0o700)
    init_deployment(scratch / "home", native_build, testing=True)
    return deployed(scratch / "home")


def record(*, native_build: Path, work_root: Path, plan_runs: dict, scratch: Path, repo: Path = REPO) -> dict:
    """The freeze record of the executing checkout (``runner.py freeze``). Refused unless the checkout is clean, the
    interpreter isolated and the imported package exactly the checkout's ``src/elpis``."""
    observed = observe(native_build=native_build, work_root=work_root, repo=repo)
    problems = _preconditions(observed)
    if problems:
        raise SystemExit("freeze refused: " + "; ".join(problems))
    observed["deployed"] = trial_deployment(native_build, scratch)
    return {"schema": FREEZE_SCHEMA, "specification": SPECIFICATION, "inputs": "qual", "plan_runs": plan_runs,
            "gates": ["G1", "G2", "G3", "G4", "G5", "G6"],
            "evidence_schema": {"record": "elpis.k1-recovery-r0-v4.record.v2",
                                "verdict": "elpis.k1-recovery-r0-v4.verdict.v2",
                                "run": "elpis.k1-recovery-r0-v4.run.v2", "freeze": FREEZE_SCHEMA},
            **observed}


def _preconditions(observed: dict) -> list:
    problems = []
    if not observed["source"]["clean"]:
        problems.append(f"the checkout is not clean: {observed['source']['status']}")
    py = observed["python"]
    if py["isolated"] != 1:
        problems.append("the interpreter is not in isolated mode (python -I)")
    if py["package"] != py["source_package"]:
        problems.append(f"the imported elpis package ({py['package_root']}) is not the checkout's src/elpis")
    if py["modules_outside_package"]:
        problems.append(f"elpis modules loaded from outside the package: {py['modules_outside_package']}")
    if any(v is None for v in observed["native"].values()):
        problems.append("a deployable library is missing from the native build")
    return problems


_COMPARED = (("source", "commit"), ("source", "tree"), ("instrument",), ("subject",), ("python", "executable"),
             ("python", "version"), ("python", "implementation"), ("python", "package"), ("native",),
             ("toolchain",), ("platform",), ("filesystem", "type"))


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


def compare_deployed(freeze: dict, observed: dict) -> list:
    """The run's own deployment must be exactly the frozen one: the ``home/lib`` copies (by name and content) are the
    frozen build's libraries, the catalog pins exactly them, and the admitted libraries report the frozen ABI."""
    problems = []
    want = freeze.get("deployed", {})
    by_name = {Path(rel).name: digest for rel, digest in freeze["native"].items()}
    if observed["lib"] != by_name:
        problems.append("the deployed home/lib copies are not the frozen libraries")
    for key in ("lib", "catalog", "catalog_sha256", "deployment_abi", "reported_abi"):
        if want.get(key) != observed.get(key):
            problems.append(f"deployed.{key} differs from the freeze")
    if observed["deployment_abi"] != {"runtime": 5, "composite": 1} or observed["reported_abi"].get("runtime") != 5:
        problems.append(f"the deployment is not RuntimeCore ABI v5 (composite ABI 1): {observed['deployment_abi']}, "
                        f"reported {observed['reported_abi']}")
    return problems
