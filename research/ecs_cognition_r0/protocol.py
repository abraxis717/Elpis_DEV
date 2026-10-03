"""Protocol plumbing: canonical JSON, research digests, per-world RNG, write-once files. RESEARCH_ONLY.

Digests use the research-only domain ``elpis.research.ecs-cognition-r0.*``;
no production identity domain is reused. All randomness derives from
(spec seed, experiment name, world id, stream); there is no global RNG.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent
DOMAIN = "elpis.research.ecs-cognition-r0"


class ProtocolError(ValueError):
    """A specification, freeze or evidence file is malformed or would be overwritten."""


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


def world_ids(split: str, count: int) -> tuple[str, ...]:
    if split not in ("DEV", "QUAL"):
        raise ProtocolError("split must be DEV or QUAL")
    return tuple(f"{split.lower()}-{i:04d}" for i in range(count))


def _code(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def world_rng(seed: int, name: str, world: str, stream: str) -> np.random.Generator:
    sequence = np.random.SeedSequence(entropy=seed, spawn_key=(_code(name), _code(world), _code(stream)))
    return np.random.Generator(np.random.PCG64(sequence))


def source_digest() -> str:
    entries = [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(ROOT.glob("*.py"))]
    return digest("source", entries)


def numerical_profile() -> dict:
    return {"numpy": np.__version__, "python": platform.python_version(), "machine": platform.machine(),
            "float": "IEEE-754 binary64"}


def head() -> str:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
    dirty = git("status", "--porcelain", "--", str(ROOT / "*.py"))
    return git("rev-parse", "HEAD") + ("+uncommitted-lab-source" if dirty else "")


def write(path: Path, kind: str, body: dict, *, exclusive: bool) -> str:
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
