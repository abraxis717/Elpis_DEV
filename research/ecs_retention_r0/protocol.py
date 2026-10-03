"""Protocol plumbing: specification, canonical JSON, research digests, per-world RNG, write-once files and the
implementation binding. RESEARCH_ONLY.

Digests use the research-only domain ``elpis.research.ecs-retention-r0.*``;
no production identity domain is reused. All randomness derives from
(spec seed, experiment name, world id, stream); there is no global RNG.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DOMAIN = "elpis.research.ecs-retention-r0"
SPEC_FILE = ROOT / "specs" / "ecsg-retention-r0.v1.spec.json"
CANDIDATES_FILE = ROOT / "CANDIDATES.md"

# The implementation every record binds (spec binding_requirements).
BOUND_FILES = ("src/elpis/ECS_G/native.py", "src/elpis/ECS_G/cognition.py",
               "native/ECS_G/include/elpis/ecsg_math.h", "native/ECS_G/include/elpis/ecsg_state.h",
               "native/ECS_G/include/elpis/ecsg_executor.h", "native/ECS_G/src/ecsg_math.c",
               "native/ECS_G/src/ecsg_state.c", "native/ECS_G/src/ecsg_executor.c")
_CMAKE_KEYS = ("CMAKE_BUILD_TYPE", "CMAKE_C_COMPILER", "CMAKE_C_FLAGS", "CMAKE_C_FLAGS_RELEASE",
               "CMAKE_C_FLAGS_RELWITHDEBINFO", "CMAKE_C_FLAGS_DEBUG", "ELPIS_ENABLE_ASAN", "ELPIS_ENABLE_UBSAN",
               "ELPIS_ENABLE_TSAN")


class ProtocolError(ValueError):
    """A specification, freeze or evidence file is malformed, unbound or would be overwritten."""


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.bool_):
        value = bool(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.integer):
        value = int(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ProtocolError("non-finite value in canonical JSON")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ProtocolError(f"{type(value).__name__} is not canonical-JSON representable")


def canonical_json(value) -> bytes:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def digest(kind: str, value) -> str:
    return hashlib.sha256(f"{DOMAIN}.{kind}.v1".encode("ascii") + b"\0" + canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_spec() -> dict:
    return json.loads(SPEC_FILE.read_text(encoding="ascii"))


def spec_digests(spec: dict) -> dict:
    return {"spec": digest("spec", spec), "pass_rule": digest("pass_rule", spec["pass_rule"]),
            "candidates_sha256": sha256_file(CANDIDATES_FILE)}


def world_ids(spec: dict, split: str) -> tuple[str, ...]:
    count = {"DEV": spec["splits"]["dev_worlds"], "QUAL": spec["splits"]["qual_worlds"]}.get(split)
    if count is None:
        raise ProtocolError("split must be DEV or QUAL")
    return tuple(f"{split.lower()}-{i:04d}" for i in range(count))


def _code(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def world_rng(seed: int, name: str, world: str, stream: str) -> np.random.Generator:
    sequence = np.random.SeedSequence(entropy=seed, spawn_key=(_code(name), _code(world), _code(stream)))
    return np.random.Generator(np.random.PCG64(sequence))


def source_digest() -> str:
    entries = [{"file": p.name, "sha256": sha256_file(p)} for p in sorted(ROOT.glob("*.py"))]
    return digest("source", entries)


def source_digest_at(commit: str) -> str:
    """The laboratory source digest of the ``*.py`` files as committed at ``commit``."""
    listing = _git("ls-tree", "--name-only", commit, "research/ecs_retention_r0/")
    names = sorted(Path(n).name for n in listing.splitlines() if n.endswith(".py"))
    entries = []
    for name in names:
        blob = subprocess.run(["git", "show", f"{commit}:research/ecs_retention_r0/{name}"], cwd=REPO,
                              capture_output=True, check=True).stdout
        entries.append({"file": name, "sha256": hashlib.sha256(blob).hexdigest()})
    return digest("source", entries)


def numerical_profile() -> dict:
    return {"numpy": np.__version__, "python": platform.python_version(), "machine": platform.machine(),
            "float": "IEEE-754 binary64",
            "threads": {v: os.environ.get(v, "unset") for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                                                 "MKL_NUM_THREADS")}}


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _build(library: Path) -> dict:
    for parent in library.parents:
        cache = parent / "CMakeCache.txt"
        if cache.is_file():
            out = {}
            for line in cache.read_text(errors="replace").splitlines():
                for key in _CMAKE_KEYS:
                    if line.startswith(key + ":"):
                        out[key] = line.split("=", 1)[1]
            compiler = out.get("CMAKE_C_COMPILER", "cc")
            try:
                version = subprocess.run([compiler, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
            except (OSError, IndexError):
                version = "unknown"
            return {"cmake": out, "compiler": version}
    raise ProtocolError(f"no CMakeCache.txt above {library}; build identity unknown")


def implementation(library: Path) -> dict:
    """Everything the evidence binds about the code and binary that produced it (spec binding_requirements)."""
    library = Path(library).resolve()
    return {"base_commit": _git("rev-parse", "HEAD"), "tree": _git("rev-parse", "HEAD^{tree}"),
            "dirty": bool(_git("status", "--porcelain", "--", "src", "native", "research/ecs_retention_r0",
                               ":(exclude)research/ecs_retention_r0/evidence",
                               ":(exclude)research/ecs_retention_r0/frozen")),
            "lab_source_digest": source_digest(),
            "files": {name: sha256_file(REPO / name) for name in BOUND_FILES},
            "library": {"path": str(library), "sha256": sha256_file(library)},
            "build": _build(library), "numerical_profile": numerical_profile()}


def binding_mismatch(frozen: dict, current: dict) -> list[str]:
    """Bound fields that differ between the frozen implementation and the current one."""
    out = []
    if frozen["lab_source_digest"] != current["lab_source_digest"]:
        out.append("lab_source_digest")
    out += [f"files/{k}" for k in BOUND_FILES if frozen["files"].get(k) != current["files"].get(k)]
    if frozen["library"]["sha256"] != current["library"]["sha256"]:
        out.append("library")
    if frozen["build"] != current["build"]:
        out.append("build")
    if frozen["numerical_profile"] != current["numerical_profile"]:
        out.append("numerical_profile")
    return out


def write(path: Path, kind: str, body: dict, *, exclusive: bool = True) -> str:
    """Write ``{"body", "digest"}``; exclusive files are written once and never replaced."""
    record = {"body": json.loads(canonical_json(body)), "digest": digest(kind, body)}
    data = json.dumps(record, indent=1, sort_keys=True).encode("ascii") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "xb" if exclusive else "wb") as fh:
            fh.write(data)
    except FileExistsError as exc:
        raise ProtocolError(f"{path.name} already exists; a changed experiment needs a new version") from exc
    return record["digest"]


def load(path: Path, kind: str) -> dict:
    record = json.loads(Path(path).read_bytes())
    if digest(kind, record["body"]) != record["digest"]:
        raise ProtocolError(f"{Path(path).name}: recorded digest does not match its content")
    return record
