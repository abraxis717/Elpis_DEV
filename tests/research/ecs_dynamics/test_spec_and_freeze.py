"""Specification, deterministic RNG, DEV/QUAL split and write-once freezing."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest

from research.ecs_dynamics import spec as S

LAB = Path(S.__file__).resolve().parent


def _spec(**kw):
    base = dict(name="t", version=1, model_family="tanh-rnn-nonreciprocal", dimension=4, width=None, seed=7,
                dev_worlds=2, qual_worlds=3, warmup=1, horizon=2, perturbation=1e-6, recurrence_tolerance=1e-8,
                fixed_point_tolerance=1e-10, max_period=4, lyapunov_horizon=3, coarse_observable="mean",
                delay_depth=1, target="EVENT", intervention_protocol="none", parameters={"b": [1, 2]},
                protocol={})
    base.update(kw)
    return S.ExperimentSpec(**base)


def test_canonical_json_is_sorted_compact_and_rejects_non_finite():
    assert S.canonical_json({"b": 1, "a": [1.5, None]}) == b'{"a":[1.5,null],"b":1}'
    assert S.canonical_json({"x": np.float64(0.5), "y": np.int64(3), "z": np.bool_(True)}) == b'{"x":0.5,"y":3,"z":true}'
    for bad in (float("nan"), float("inf")):
        with pytest.raises(S.SpecError):
            S.canonical_json({"v": bad})


def test_research_digests_use_only_the_research_domain():
    with pytest.raises(S.SpecError):
        S.research_digest("elpis.ecs.state-root.v1", {})
    assert _spec().digest == _spec().digest
    assert _spec().digest != _spec(seed=8).digest
    assert S.SPEC_DOMAIN.startswith("elpis.research.ecs-dynamics.")


def test_spec_binds_every_choice_and_validates():
    d = _spec().as_dict()
    for key in ("model_family", "dimension", "width", "seed", "dev_worlds", "qual_worlds", "warmup", "horizon",
                "perturbation", "recurrence_tolerance", "fixed_point_tolerance", "max_period", "lyapunov_horizon",
                "coarse_observable", "delay_depth", "target", "intervention_protocol"):
        assert key in d
    with pytest.raises(S.SpecError):
        _spec(target="SUFFICIENT")
    with pytest.raises(S.SpecError):
        _spec(model_family="production-ecs")
    with pytest.raises(S.SpecError):
        _spec(perturbation=float("nan"))
    with pytest.raises(S.SpecError):
        _spec(dimension=-1)
    with pytest.raises(Exception):
        _spec().seed = 3  # frozen
    assert S.ExperimentSpec.from_dict(_spec().as_dict()) == _spec()


def test_world_rng_is_deterministic_and_stream_separated():
    s = _spec()
    a = S.world_rng(s, "dev-0000", "x").standard_normal(5)
    assert np.array_equal(a, S.world_rng(s, "dev-0000", "x").standard_normal(5))
    assert not np.array_equal(a, S.world_rng(s, "dev-0001", "x").standard_normal(5))
    assert not np.array_equal(a, S.world_rng(s, "dev-0000", "y").standard_normal(5))
    assert not np.array_equal(a, S.world_rng(_spec(name="u"), "dev-0000", "x").standard_normal(5))


def test_split_authority_is_disjoint():
    worlds = S.split_worlds(_spec())
    assert worlds["DEV"] == ("dev-0000", "dev-0001")
    assert worlds["QUAL"] == ("qual-0000", "qual-0001", "qual-0002")
    assert not set(worlds["DEV"]) & set(worlds["QUAL"])


def test_no_global_rng_in_laboratory_sources():
    pattern = re.compile(r"np\.random\.(seed|rand|randn|randint|normal|uniform|choice|permutation|shuffle|random)\(|"
                         r"^\s*import random\b|default_rng\(\)", re.M)
    offenders = [p.name for p in LAB.glob("*.py") if pattern.search(p.read_text())]
    assert not offenders


def _frozen(spec, choices=None):
    return S.FrozenSpec(spec, choices or {"lam": 0.1}, "d" * 64, S.source_digest(), "c" * 40, {"rule": 1})


def test_freeze_is_write_once_and_verified(tmp_path):
    reg = S.FreezeRegistry(tmp_path)
    digest = reg.freeze(_frozen(_spec()))
    assert reg.freeze(_frozen(_spec())) == digest  # idempotent
    with pytest.raises(S.SpecError, match="new version"):
        reg.freeze(_frozen(_spec(), {"lam": 0.2}))
    loaded = reg.load("t", 1)
    assert loaded.digest == digest and loaded.dev_choices == {"lam": 0.1}
    # A new version is a new identity and may be frozen.
    reg.freeze(_frozen(_spec(version=2), {"lam": 0.2}))
    # Tampering with the stored file is detected on load.
    path = reg.path(_spec())
    path.write_bytes(path.read_bytes().replace(b'"lam":0.1', b'"lam":0.3'))
    with pytest.raises(S.SpecError):
        reg.load("t", 1)


def test_source_digest_changes_with_source(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    first = S.source_digest(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n")
    assert S.source_digest(tmp_path) != first
