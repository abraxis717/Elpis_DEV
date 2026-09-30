"""Deterministic experiment specifications, RNG derivation, DEV/QUAL split and freezing.

RESEARCH_ONLY. NO_RUNTIME_AUTHORITY. Digests here use the research-only domain
``elpis.research.ecs-dynamics.*``; no production ECS identity domain is reused.

* :class:`ExperimentSpec` binds every choice an experiment depends on.
* :func:`world_rng` derives all randomness from (spec seed, world id, stream);
  there is no global RNG.
* :func:`split_worlds` is the split authority: world ids are named ``dev-NNNN``
  and ``qual-NNNN`` and the two sets never overlap.
* :class:`FreezeRegistry` writes a frozen QUAL specification once and refuses
  any later write under the same name with different content.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SPEC_SCHEMA = "elpis.research.ecs-dynamics.spec.v1"
SPEC_DOMAIN = "elpis.research.ecs-dynamics.spec-digest.v1"
FROZEN_DOMAIN = "elpis.research.ecs-dynamics.frozen-spec.v1"

MODEL_FAMILIES = ("tanh-rnn-nonreciprocal", "cubic-moment-control", "nonreciprocal-associative", "linear-fixture",
                  "period-map-fixture", "logistic-fixture", "ecs-topology")
SPLITS = ("DEV", "QUAL")


class SpecError(ValueError):
    """An experiment specification is malformed or a frozen spec would change."""


def _plain(value):
    """Convert to JSON-safe plain values; floats must be finite."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.bool_):
        value = bool(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.integer,)):
        value = int(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise SpecError("non-finite value in canonical JSON")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise SpecError(f"value of type {type(value).__name__} is not canonical-JSON representable")


def canonical_json(value) -> bytes:
    """Sorted keys, no whitespace, ASCII, finite floats in repr form."""
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def research_digest(domain: str, value) -> str:
    if not domain.startswith("elpis.research."):
        raise SpecError("research digests must use an elpis.research.* domain")
    return hashlib.sha256(domain.encode("ascii") + b"\0" + canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class ExperimentSpec:
    """Every choice an experiment depends on. Immutable; digest-bound.

    ``parameters`` holds family-specific values (e.g. gain, gamma, width);
    ``protocol`` holds intervention/predictor settings. Both are plain JSON.
    """
    name: str
    version: int
    model_family: str
    dimension: int
    width: int | None
    seed: int
    dev_worlds: int
    qual_worlds: int
    warmup: int
    horizon: int
    perturbation: float
    recurrence_tolerance: float
    fixed_point_tolerance: float
    max_period: int
    lyapunov_horizon: int
    coarse_observable: str
    delay_depth: int
    target: str
    intervention_protocol: str
    parameters: dict = field(default_factory=dict)
    protocol: dict = field(default_factory=dict)
    schema: str = SPEC_SCHEMA

    def __post_init__(self):
        if self.schema != SPEC_SCHEMA:
            raise SpecError("spec schema")
        if self.model_family not in MODEL_FAMILIES:
            raise SpecError(f"unknown model family {self.model_family!r}")
        for name in ("version", "dimension", "seed", "dev_worlds", "qual_worlds", "warmup", "horizon",
                     "max_period", "lyapunov_horizon", "delay_depth"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise SpecError(f"{name} must be a non-negative int")
        if self.width is not None and (type(self.width) is not int or self.width < 1):
            raise SpecError("width must be a positive int or None")
        for name in ("perturbation", "recurrence_tolerance", "fixed_point_tolerance"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0:
                raise SpecError(f"{name} must be a positive finite float")
        from .observables import TARGETS  # late import: observables imports nothing from here
        if self.target not in TARGETS:
            raise SpecError(f"unknown target {self.target!r}")
        # Parameters/protocol must be canonical JSON now, not at digest time.
        canonical_json(self.parameters)
        canonical_json(self.protocol)
        object.__setattr__(self, "parameters", json.loads(canonical_json(self.parameters)))
        object.__setattr__(self, "protocol", json.loads(canonical_json(self.protocol)))

    def as_dict(self) -> dict:
        return json.loads(canonical_json(asdict(self)))

    @property
    def digest(self) -> str:
        return research_digest(SPEC_DOMAIN, self.as_dict())

    def with_choices(self, **changes) -> "ExperimentSpec":
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentSpec":
        return cls(**data)


def world_ids(spec: ExperimentSpec, split: str) -> tuple[str, ...]:
    if split not in SPLITS:
        raise SpecError("split must be DEV or QUAL")
    count = spec.dev_worlds if split == "DEV" else spec.qual_worlds
    prefix = "dev" if split == "DEV" else "qual"
    return tuple(f"{prefix}-{i:04d}" for i in range(count))


def split_worlds(spec: ExperimentSpec) -> dict[str, tuple[str, ...]]:
    """The split authority: disjoint DEV and QUAL world ids."""
    dev, qual = world_ids(spec, "DEV"), world_ids(spec, "QUAL")
    assert not set(dev) & set(qual)
    return {"DEV": dev, "QUAL": qual}


def _stream_code(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def world_rng(spec: ExperimentSpec, world_id: str, stream: str) -> np.random.Generator:
    """Deterministic generator for one (spec, world, stream). No global state."""
    if type(world_id) is not str or type(stream) is not str:
        raise SpecError("world id and stream must be strings")
    sequence = np.random.SeedSequence(entropy=spec.seed,
                                      spawn_key=(_stream_code(spec.name), _stream_code(world_id),
                                                 _stream_code(stream)))
    return np.random.Generator(np.random.PCG64(sequence))


def numerical_profile() -> dict:
    import platform
    return {"numpy": np.__version__, "python": platform.python_version(), "machine": platform.machine(),
            "float": "IEEE-754 binary64"}


def source_digest(root: Path | None = None) -> str:
    """Digest of the laboratory's Python sources (the scientific object's code)."""
    root = Path(__file__).resolve().parent if root is None else Path(root)
    entries = []
    for path in sorted(root.glob("*.py")):
        entries.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return research_digest("elpis.research.ecs-dynamics.source.v1", entries)


@dataclass(frozen=True)
class FrozenSpec:
    """A QUAL specification frozen after DEV: the spec, the DEV choices and the code it binds."""
    spec: ExperimentSpec
    dev_choices: dict
    dev_evidence_digest: str
    source_digest: str
    source_commit: str
    pass_rule: dict

    def as_dict(self) -> dict:
        return {"schema": FROZEN_DOMAIN, "spec": self.spec.as_dict(), "spec_digest": self.spec.digest,
                "dev_choices": _plain(self.dev_choices), "dev_evidence_digest": self.dev_evidence_digest,
                "source_digest": self.source_digest, "source_commit": self.source_commit,
                "pass_rule": _plain(self.pass_rule)}

    @property
    def digest(self) -> str:
        return research_digest(FROZEN_DOMAIN, self.as_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "FrozenSpec":
        if data.get("schema") != FROZEN_DOMAIN:
            raise SpecError("frozen spec schema")
        spec = ExperimentSpec.from_dict(data["spec"])
        if spec.digest != data["spec_digest"]:
            raise SpecError("frozen spec digest does not match its specification")
        return cls(spec, data["dev_choices"], data["dev_evidence_digest"], data["source_digest"],
                   data["source_commit"], data["pass_rule"])


class FreezeRegistry:
    """Write-once store of frozen QUAL specifications (one JSON file per experiment identity)."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def path(self, spec: ExperimentSpec) -> Path:
        return self.directory / f"{spec.name}.v{spec.version}.frozen.json"

    def freeze(self, frozen: FrozenSpec) -> str:
        path = self.path(frozen.spec)
        payload = canonical_json({"frozen": frozen.as_dict(), "digest": frozen.digest})
        if path.exists():
            if path.read_bytes() != payload:
                raise SpecError(f"{path.name} is frozen; a changed experiment needs a new version")
            return frozen.digest
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return frozen.digest

    def load(self, name: str, version: int) -> FrozenSpec:
        path = self.directory / f"{name}.v{version}.frozen.json"
        data = json.loads(path.read_bytes())
        frozen = FrozenSpec.from_dict(data["frozen"])
        if frozen.digest != data["digest"]:
            raise SpecError("frozen spec content does not match its recorded digest")
        return frozen
