"""Protocol plumbing: specification, canonical JSON, research digests, per-world RNG, write-once records and the
implementation binding (spec numerical_binding). RESEARCH_ONLY.

Digests use the research-only domain ``elpis.research.ecs-retention-r1.*``; no production identity domain is
reused. All randomness derives from (spec seed, experiment name, world id, stream); there is no global RNG.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from . import numerics

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LAB = "research/ecs_retention_r1"
DOMAIN = "elpis.research.ecs-retention-r1"
SPEC_FILE = ROOT / "specs" / "ecsg-retention-r1.v1.spec.json"
CANDIDATES_FILE = ROOT / "CANDIDATES.md"

# The implementation every record binds: the ECS_G binding, headers and sources, and the build files.
_BOUND_GLOBS = ("src/elpis/ECS_G/*.py", "native/ECS_G/include/elpis/*.h", "native/ECS_G/src/*.c")
_BOUND_EXTRA = ("native/ECS_G/CMakeLists.txt", "CMakeLists.txt")
_CMAKE_KEYS = ("CMAKE_BUILD_TYPE", "CMAKE_C_COMPILER", "CMAKE_C_FLAGS", "CMAKE_C_FLAGS_RELEASE",
               "CMAKE_C_FLAGS_RELWITHDEBINFO", "CMAKE_C_FLAGS_DEBUG", "ELPIS_ENABLE_ASAN", "ELPIS_ENABLE_UBSAN",
               "ELPIS_ENABLE_TSAN", "ELPIS_WARNINGS_AS_ERRORS")


class ProtocolError(ValueError):
    """A specification, freeze or evidence file is malformed, unbound, refused or would be overwritten."""


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


def plain(value):
    """The canonical-JSON round trip of ``value`` (what a record stores)."""
    return json.loads(canonical_json(value))


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
    """Digest of the laboratory's Python sources (every ``*.py`` under the laboratory, by relative path)."""
    entries = [{"file": str(p.relative_to(ROOT)), "sha256": sha256_file(p)} for p in sorted(ROOT.rglob("*.py"))
               if "__pycache__" not in p.parts]
    return digest("source", entries)


def source_digest_at(commit: str) -> str:
    """The laboratory source digest of the ``*.py`` files as committed at ``commit``."""
    listing = _git("ls-tree", "-r", "--name-only", commit, f"{LAB}/")
    names = sorted(n for n in listing.splitlines() if n.endswith(".py"))
    entries = []
    for name in names:
        blob = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=REPO, capture_output=True, check=True).stdout
        entries.append({"file": name[len(LAB) + 1:], "sha256": hashlib.sha256(blob).hexdigest()})
    return digest("source", sorted(entries, key=lambda e: e["file"]))


def bound_files() -> tuple[str, ...]:
    names = {str(p.relative_to(REPO)) for g in _BOUND_GLOBS for p in REPO.glob(g)}
    return tuple(sorted(names | set(_BOUND_EXTRA)))


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
            target_options = "-ffp-contract=off (native/ECS_G/CMakeLists.txt, elpis_ecsg_math)"
            return {"cmake": out, "compiler": version, "ecsg_target_options": target_options}
    raise ProtocolError(f"no CMakeCache.txt above {library}; build identity unknown")


def implementation(library: Path) -> dict:
    """Everything a record binds about the code, binary and numerical environment that produced it."""
    library = Path(library).resolve()
    dirty = _git("status", "--porcelain", "--", "src", "native", LAB, f":(exclude){LAB}/evidence",
                 f":(exclude){LAB}/frozen")
    return {"base_commit": _git("rev-parse", "HEAD"), "tree": _git("rev-parse", "HEAD^{tree}"), "dirty": bool(dirty),
            "lab_source_digest": source_digest(),
            "files": {name: sha256_file(REPO / name) for name in bound_files()},
            "library": {"name": library.name, "sha256": sha256_file(library)},
            "build": _build(library), "numerical_profile": numerics.profile()}


def binding_mismatch(frozen: dict, current: dict) -> list[str]:
    """Bound fields that differ between a recorded implementation and the current one."""
    out = []
    if frozen["lab_source_digest"] != current["lab_source_digest"]:
        out.append("lab_source_digest")
    names = set(frozen["files"]) | set(current["files"])
    out += [f"files/{k}" for k in sorted(names) if frozen["files"].get(k) != current["files"].get(k)]
    for key in ("library", "build", "numerical_profile"):
        if frozen[key] != current[key]:
            out.append(key)
    return out


def require_single_thread() -> None:
    threads = numerics.blas_threads()
    if threads != 1:
        raise ProtocolError(f"effective OpenBLAS thread count is {threads}, not 1; evidence refused")


def write(path: Path, kind: str, body: dict) -> str:
    """Write ``{"body", "digest"}`` exclusively: a record is written once and never replaced."""
    record = {"body": json.loads(canonical_json(body)), "digest": digest(kind, body)}
    data = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"   # compact: size
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "xb") as fh:
            fh.write(data)
    except FileExistsError as exc:
        raise ProtocolError(f"{path.name} already exists; a changed experiment needs a new version") from exc
    return record["digest"]


def load(path: Path, kind: str) -> dict:
    record = json.loads(Path(path).read_bytes())
    if digest(kind, record["body"]) != record["digest"]:
        raise ProtocolError(f"{Path(path).name}: recorded digest does not match its content")
    return record
