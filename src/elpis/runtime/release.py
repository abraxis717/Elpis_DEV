"""A local developer installation bundle: ``python -m elpis.runtime.release --repo REPO --native-build BUILD --out DIR``.

It packages one exact, clean source commit and the libraries an operator built from it, for another developer to
install and exercise without reconstructing the development history. It publishes nothing (the distribution's
release authority is ``NONE_DEFERRED``: ELPIS_SYSTEM.json); it writes one directory::

    elpis-dev-<version>-<commit12>/
        MANIFEST.json      source identity, library digests, toolchain, ABI and format declarations, dependencies,
                           capability and nonclaim manifests, the install / smoke / recovery commands
        source.tar.gz      ``git archive`` of the commit (deterministic: gzip mtime 0)
        native-build/      the deployable libraries in build-tree layout (``init --native-build`` takes it)
        INSTALL.md         the operator guide (docs/OPERATING_ELPIS.md)
        examples/          the example session scripts (docs/examples)
        SHA256SUMS         every file above

A dirty working tree is refused: a bundle names exactly one commit. Native builds are not claimed to be
bit-reproducible across toolchains; the toolchain that produced the libraries is recorded.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from .acceptance import _LIBS
from .cli import CAPABILITIES, NONCLAIMS

MANIFEST_SCHEMA = "elpis.release-bundle.v1"
FORMATS = {"runtime_abi": 6, "composite_abi": 1, "continuity_record": 3, "ELPISEC1": 1, "ELPISGK1": 1,
           "ELPISEM1": 1, "deployment": "elpis.deployment.v1", "lineage": "elpis.lineage.v1",
           "native_catalog": "elpis.inference-authority.v1"}


def _run(args, cwd=None) -> str:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def _tool(args) -> str:
    try:
        return _run(args).splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_bundle(repo: Path, native_build: Path, out: Path) -> Path:
    if _run(["git", "status", "--porcelain"], cwd=repo):
        raise SystemExit("the working tree is not clean: a bundle names exactly one commit")
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    tree = _run(["git", "rev-parse", "HEAD^{tree}"], cwd=repo)
    version = _run([sys.executable, "-c", "import tomllib,sys;print(tomllib.load(open(sys.argv[1],'rb'))"
                    "['project']['version'])", str(repo / "pyproject.toml")])
    root = out / f"elpis-dev-{version}-{commit[:12]}"
    if root.exists():
        raise SystemExit(f"{root} exists: a bundle is written once")
    root.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", "--format=tar", "--prefix=elpis-dev/", commit], cwd=repo,
                             capture_output=True, check=True).stdout
    (root / "source.tar.gz").write_bytes(gzip.compress(archive, mtime=0))
    libraries = []
    for rel in _LIBS:
        target = root / "native-build" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(native_build / rel, target)
        libraries.append({"path": "native-build/" + rel, "sha256": _sha(target), "size": target.stat().st_size})
    shutil.copyfile(repo / "docs" / "OPERATING_ELPIS.md", root / "INSTALL.md")
    shutil.copytree(repo / "docs" / "examples", root / "examples")
    home = "$HOME/elpis-home"
    manifest = {
        "schema": MANIFEST_SCHEMA, "distribution": "elpis-dev", "version": version, "source_commit": commit,
        "source_tree": tree, "source_archive_sha256": _sha(root / "source.tar.gz"), "libraries": libraries,
        "toolchain": {"cc": _tool(["cc", "--version"]), "cmake": _tool(["cmake", "--version"]),
                      "rustc": _tool(["rustc", "--version"]), "python": sys.version.split()[0],
                      "platform": platform.platform()},
        "formats": FORMATS, "python_dependencies": {"base": [], "optional": {"test": "pyproject.toml [test]"}},
        "native_build_dependencies": ["a C11 compiler (gcc or clang)", "cmake >= 3.20", "a Rust toolchain (cargo)",
                                      "pkg-config", "libsqlite3-dev", "libpcre2-dev"],
        "commands": {
            "install": "python3 -m venv .venv && .venv/bin/python -m pip install ./elpis-dev",
            "build": "cmake -S elpis-dev -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel",
            "init": f".venv/bin/python -m elpis.runtime.cli --home {home} init --native-build $PWD/native-build "
                    f"--source-commit {commit}",
            "create": f".venv/bin/python -m elpis.runtime.cli --home {home} create --seed 0",
            "session": f".venv/bin/python -m elpis.runtime.cli --home {home} session --script examples/demo-session.json",
            "smoke_and_recovery": ".venv/bin/python -m elpis.runtime.acceptance --native-build $PWD/native-build "
                                  f"--work $PWD/acceptance --source-commit {commit} --report acceptance.json"},
        "capabilities": CAPABILITIES, "nonclaims": list(NONCLAIMS),
        "known_limitations": [
            "no text interface: no qualified ECS<->DSV4 codec map (ECS_CODEC_UNQUALIFIED)",
            "one lineage per deployment home; volatile work is lost at process exit unless published",
            "crash consistency for process death only; power loss is not claimed",
            "memory LEARN (recurrent FAST) and SLOW consolidation are not available",
            "Linux x86_64 only (sealed memfd admission, openat2)",
        ],
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    sums = [f"{_sha(p)}  {p.relative_to(root)}" for p in sorted(root.rglob("*")) if p.is_file()]
    (root / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return root


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m elpis.runtime.release", description=__doc__.split("\n\n")[0])
    p.add_argument("--repo", required=True, help="the clean source checkout the libraries were built from")
    p.add_argument("--native-build", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    root = build_bundle(Path(args.repo).absolute(), Path(args.native_build).absolute(), Path(args.out).absolute())
    print(json.dumps({"bundle": str(root), "manifest": str(root / "MANIFEST.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
