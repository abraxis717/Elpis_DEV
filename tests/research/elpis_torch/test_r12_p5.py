"""Elpis Torch R12 P5: the representation-seam diagnostics (mechanics only; DIAGNOSIS ONLY)."""
from __future__ import annotations

import json
import math

import pytest

from .conftest import SKIP_REASON

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch import fast_episodes as fe  # noqa: E402
from research.elpis_torch import specialist_stack as ss  # noqa: E402
from research.elpis_torch.experiments import run_r11 as r11  # noqa: E402
from research.elpis_torch.experiments import run_r12 as r12  # noqa: E402
from research.elpis_torch.experiments import run_r12_p5 as p5  # noqa: E402
from research.elpis_torch.model import parameter_digest  # noqa: E402

NS = "R12_P5_TEST"
PLAN = {"adam_lr": 3e-3, "batch": 2, "dev_episodes": 2, "select_every": 5, "steps": 10}


def _stack(seed=7):
    return fe.freeze(ss.RegressionStack("G2", "BOUNDED_SPECIALIST", seed, namespace=NS))


def test_a_fixed_projection_never_learns_while_the_rest_of_the_stack_does():
    torch.set_num_threads(1)
    learned, info = p5.train_g2(NS, "smooth", 7, PLAN, "BOUNDED_SPECIALIST")
    fixed, finfo = p5.train_g2(NS, "smooth", 7, PLAN, "BOUNDED_SPECIALIST", fixed_projection=True)
    initial = dict(ss.RegressionStack("G2", "BOUNDED_SPECIALIST", 7, namespace=NS).named_parameters())
    assert info["selected_step"] > 0 and finfo["selected_step"] > 0
    name = p5.LAST + p5.FIXED
    assert finfo["projection_unchanged"] and torch.equal(dict(fixed.named_parameters())[name], initial[name])
    assert not info["projection_unchanged"] and not torch.equal(dict(learned.named_parameters())[name], initial[name])
    moved = {n for n, p in fixed.named_parameters() if not torch.equal(p, initial[n])}
    assert {p5.LAST + "cgc.core.weight", p5.LAST + "cgc.proj_out.weight", "readout.weight"} <= moved
    assert not any(p.requires_grad for p in fixed.parameters()) and not fixed.training


def test_substitution_moves_exactly_the_named_tensors_and_touches_neither_source():
    a, b = _stack(7), _stack(8)
    digests = parameter_digest(a), parameter_digest(b)
    names = [p5.LAST + n for n in p5.BRANCH]
    c = p5.substituted(a, b, names, NS, 7, "BOUNDED_SPECIALIST")
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert set(sc) == set(sa)
    for n in sc:
        assert torch.equal(sc[n], sb[n] if n in names else sa[n]), n
    assert (parameter_digest(a), parameter_digest(b)) == digests
    assert not any(p.requires_grad for p in c.parameters()) and not c.training
    assert not any(torch.equal(sa[n], sb[n]) for n in names)            # the substitution is not vacuous


def test_decodability_is_an_out_of_sample_score_per_evaluation_episode():
    d = p5.decodability(_stack(), NS, "smooth", 7, {"split": "DEV", "episodes": 3}, "UNIT", 2)
    assert set(d) == {"probe_code", "probe_stream", "query_code", "query_stream"}
    assert all(len(v) == 2 and all(math.isfinite(x) and x <= 1.0 for x in v) for v in d.values())


def test_the_compact_rows_keep_the_measures_and_every_integrity_gate():
    model = _stack()
    rows = fe.evaluate_fast(model, NS, "smooth", 7, "UNIT", 2)
    small = p5.compact(rows)
    assert [r["fast"] for r in small] == [r["fast"] for r in rows]
    assert all(r["integrity"] == r12._integrity([full]) and r["integrity"]["all"] for r, full in zip(small, rows))


def test_the_diagnosis_runs_only_under_its_trigger_and_its_frozen_inputs(tmp_path, monkeypatch):
    study, amendment = tmp_path / "study.json", tmp_path / "amendment.json"
    study.write_text(json.dumps({"verdict": {"reading": "QUALIFIED_ONLY_WITH_ADAPTATION_TRAINED_INITIALIZATION"}}))
    plan_b = json.loads(r11.PREREGISTRATION.read_text(encoding="utf-8"))["stage_b"]["training"]
    amendment.write_text(json.dumps({"phase": p5.PHASE, "stage_b": {"plan_b": plan_b},
                                     "trigger": {"study_r12_sha256": r12.sha(study), "reading": p5.TRIGGER,
                                                 "preregistration_sha256": r12.sha(r12.PREREGISTRATION)}}))
    monkeypatch.setattr(p5, "STUDY", study)
    monkeypatch.setattr(p5, "AMENDMENT", amendment)
    with pytest.raises(ValueError, match="only after"):
        p5.frozen_p5()
    study.write_text(json.dumps({"verdict": {"reading": p5.TRIGGER}}))          # the study no longer matches
    with pytest.raises(ValueError, match="changed after"):
        p5.frozen_p5()
