"""DEV -> freeze -> QUAL-once guards, and verification of the committed frozen specs and QUAL evidence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.ecs_dynamics import run
from research.ecs_dynamics.experiments import EXPERIMENTS, ORDER, Experiment
from research.ecs_dynamics.results import RESULT_DOMAIN
from research.ecs_dynamics.spec import ExperimentSpec, FreezeRegistry, SpecError, research_digest, source_digest

LAB = Path(run.__file__).resolve().parent


def _fake_experiment(calls):
    spec = ExperimentSpec(name="guard-fixture", version=1, model_family="linear-fixture", dimension=1, width=None,
                          seed=1, dev_worlds=1, qual_worlds=1, warmup=0, horizon=1, perturbation=1e-6,
                          recurrence_tolerance=1e-8, fixed_point_tolerance=1e-10, max_period=1, lyapunov_horizon=0,
                          coarse_observable="mean", delay_depth=0, target="EVENT", intervention_protocol="none")

    def dev(s, ctx):
        calls.append(("dev", s.digest))
        return {"lam": 0.5}, {"dev_metric": 1.0}

    def qual(s, choices, ctx):
        calls.append(("qual", s.parameters["lam"]))
        return {"q": 2.0}, (), {"ok": True}, False  # a failure is recorded as it is

    return Experiment("guard-fixture", spec, "DESCRIPTIVE_RESULT", ("LAB: fixture",), {"rule": "q > 3"}, dev, qual,
                      ("fixture",))


@pytest.fixture
def lab(tmp_path, monkeypatch):
    calls = []
    exp = _fake_experiment(calls)
    monkeypatch.setattr(run, "EXPERIMENTS", {exp.name: exp})
    monkeypatch.setattr(run, "ORDER", (exp.name,))
    for name in ("DEV_DIR", "QUAL_DIR", "FROZEN_DIR", "SPEC_DIR"):
        monkeypatch.setattr(run, name, tmp_path / name.lower())
    return calls, tmp_path


def test_dev_freeze_qual_once(lab, monkeypatch):
    calls, tmp = lab
    run.dev()
    run.freeze("f" * 40)
    frozen = FreezeRegistry(run.FROZEN_DIR).load("guard-fixture", 1)
    assert frozen.spec.parameters["lam"] == 0.5 and frozen.source_commit == "f" * 40
    assert run.qual() == {"guard-fixture": "DID_NOT_SUPPORT"}
    path = run.QUAL_DIR / "guard-fixture.v1.qual.json"
    first = path.read_bytes()
    result = json.loads(first)["result"]
    assert result["frozen_digest"] == frozen.digest and result["split"] == "QUAL"
    assert result["mechanics"] == "MECHANICS_PASS" and result["scientific"] == "DID_NOT_SUPPORT"
    # QUAL runs once: a second invocation does not rerun the experiment or touch the file.
    n_qual = sum(1 for c in calls if c[0] == "qual")
    run.qual()
    assert sum(1 for c in calls if c[0] == "qual") == n_qual and path.read_bytes() == first
    # DEV is closed after freeze: rerunning dev does not rewrite the DEV evidence.
    dev_file = run.DEV_DIR / "guard-fixture.v1.dev.json"
    before = dev_file.read_bytes()
    run.dev()
    assert dev_file.read_bytes() == before


def test_qual_refused_when_source_changed_after_freeze(lab, monkeypatch):
    run.dev()
    run.freeze("f" * 40)
    monkeypatch.setattr(run, "source_digest", lambda: "0" * 64)
    with pytest.raises(SpecError, match="source changed"):
        run.qual()


def test_freeze_refused_for_dev_evidence_from_other_source(lab, monkeypatch):
    run.dev()
    monkeypatch.setattr(run, "source_digest", lambda: "0" * 64)
    with pytest.raises(SpecError, match="different laboratory source"):
        run.freeze("f" * 40)


def test_tampered_dev_evidence_is_rejected(lab):
    run.dev()
    path = run.DEV_DIR / "guard-fixture.v1.dev.json"
    path.write_text(path.read_text().replace('"lam": 0.5', '"lam": 0.7'))
    with pytest.raises(SpecError, match="digest"):
        run.freeze("f" * 40)


# --- committed evidence -------------------------------------------------------------------------


def _committed(kind):
    return sorted((LAB / "evidence" / kind).glob("*.json")) if (LAB / "evidence" / kind).is_dir() else []


@pytest.mark.skipif(not (LAB / "frozen").is_dir(), reason="no frozen specifications committed yet")
def test_committed_frozen_specs_bind_the_current_laboratory_source():
    registry = FreezeRegistry(LAB / "frozen")
    current = source_digest()
    for name in ORDER:
        frozen = registry.load(name, EXPERIMENTS[name].spec.version)
        # Any change to the laboratory's Python source after freezing breaks this: a repair needs a new version.
        assert frozen.source_digest == current, name
        assert frozen.pass_rule == json.loads(json.dumps(EXPERIMENTS[name].pass_rule)), name


@pytest.mark.skipif(not _committed("qual"), reason="no QUAL evidence committed yet")
def test_committed_qual_evidence_is_bound_and_intact():
    registry = FreezeRegistry(LAB / "frozen")
    for name in ORDER:
        path = LAB / "evidence" / "qual" / f"{name}.v{EXPERIMENTS[name].spec.version}.qual.json"
        data = json.loads(path.read_bytes())
        assert research_digest(RESULT_DOMAIN, data["result"]) == data["digest"], name
        result = data["result"]
        frozen = registry.load(name, EXPERIMENTS[name].spec.version)
        assert result["frozen_digest"] == frozen.digest and result["spec_digest"] == frozen.spec.digest
        assert result["source_digest"] == frozen.source_digest
        assert all(w.startswith("qual-") for w in result["world_ids"])
        assert result["flags"]["NO_RUNTIME_AUTHORITY"] is True
        if result["mechanics"] == "MECHANICS_FAIL":
            assert result["scientific"] == "NOT_TESTED"
