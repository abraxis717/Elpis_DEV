"""Elpis Torch R9 (Boardy adversarial qualification): the CGC attribution runner and its fault controls.

The confirmatory runs themselves are records (``results/r9``); these tests check that the runner builds exactly
R7's model, refuses a plan that drifted from R7, removes CGC exactly at evaluation, classifies paired differences as
preregistered, and refuses records whose preregistration or runner digests differ (fault injection).
"""
from __future__ import annotations

import argparse
import json

import pytest

from .conftest import SKIP_REASON

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch.experiments import run_r7, run_r9  # noqa: E402


@pytest.fixture(scope="module")
def plan():
    return run_r9.frozen()


def test_the_runner_builds_exactly_r7s_ple_model_for_arm_a(plan):
    ours, theirs = run_r9.build(plan["arms"]["A"], 1).state_dict(), run_r7.study_model("ple", 1).state_dict()
    assert ours.keys() == theirs.keys()
    assert all(torch.equal(ours[k], theirs[k]) for k in ours)


def test_every_trainable_arm_differs_from_a_only_in_the_routed_channel(plan):
    with torch.device("meta"):
        a = dict(run_r9.build(plan["arms"]["A"], 0).named_parameters())
        for name in ("B", "C", "C_mac"):
            other = dict(run_r9.build(plan["arms"][name], 0).named_parameters())
            shared = {k for k in a if ".experts." not in k}
            assert {k for k in other if ".experts." not in k} == shared
            assert all(a[k].shape == other[k].shape for k in shared)
            assert not any(".cgc" in k for k in other)                     # no CGC parameter in an FFN arm


def test_the_widths_match_the_preregistered_accounting(plan):
    with torch.device("meta"):
        accounts = {n: run_r9.exact_accounting(run_r9.build(plan["arms"][n], 0)) for n in run_r9.trainable(plan)}
    a, c, mac, b = (accounts[n] for n in ("A", "C", "C_mac", "B"))
    assert abs(c["total_parameters"] / a["total_parameters"] - 1) < 0.0003           # C: parameter-matched
    assert abs(mac["expert"]["macs"] / a["expert"]["macs"] - 1) < 0.003              # C_mac: multiply-add-matched
    assert c["expert"]["macs"] < a["expert"]["macs"] < mac["expert"]["macs"] + 100
    assert a["expert"]["cgc_macs_including_cubic"] == 7744 and b["expert"]["cgc_macs_including_cubic"] == 0
    recorded = json.loads((run_r9.RESULTS / "accounting.json").read_text(encoding="utf-8"))["arms"]
    for name, account in accounts.items():
        assert recorded[name]["total_parameters"] == account["total_parameters"]
        assert recorded[name]["expert"] == account["expert"]


def test_a_plan_that_drifted_from_r7_is_refused(plan, tmp_path, monkeypatch):
    drifted = json.loads(run_r9.PREREGISTRATION.read_text(encoding="utf-8"))
    drifted["model"]["cgc_channels"] = 49
    path = tmp_path / "preregistration.json"
    path.write_text(json.dumps(drifted), encoding="utf-8")
    monkeypatch.setattr(run_r9, "PREREGISTRATION", path)
    with pytest.raises(ValueError, match="does not describe R7"):
        run_r9.frozen()


def test_zeroed_gates_remove_cgc_exactly_and_restoring_them_restores_the_model(plan):
    model = run_r9.build(plan["arms"]["A"], 0).double().eval()
    expert = model.blocks[0].sparse.experts[0]
    u = torch.randn(5, model.embed.weight.shape[1], dtype=torch.float64)
    with torch.no_grad():
        full = expert(u)
        previous = run_r9.set_cgc_gates(model, 0.0)
        assert torch.equal(expert(u), expert.ffn(u))                     # exactly the FFN: CGC contributes nothing
        assert len(previous) == 16 and all(v == pytest.approx(0.1) for v in previous)
        run_r9.set_cgc_gates(model, previous)
        assert torch.equal(expert(u), full)


@pytest.mark.parametrize("diffs, expected", [
    ([0.030, 0.025, 0.021], "A_BETTER"),
    ([0.010, 0.012, 0.015], "NOT_DISTINGUISHED"),          # consistent but below delta_min
    ([0.060, -0.001, 0.050], "NOT_DISTINGUISHED"),         # large mean, inconsistent sign
    ([-0.030, -0.020, -0.025], "X_BETTER"),
    ([-0.010, -0.050, 0.001], "NOT_DISTINGUISHED"),
])
def test_paired_differences_are_classified_as_preregistered(plan, diffs, expected):
    assert run_r9._classify(diffs, plan["analysis"]["delta_min_bits_per_byte"]) == expected


def _fake_records(folder, plan, finals, *, tamper=None, missing=()):
    """Copies of the frozen digests with chosen final losses (A also gets a gates-zeroed evaluation)."""
    digests = {"preregistration_sha256": run_r9.sha256_file(run_r9.PREREGISTRATION),
               "runner_sha256": run_r9.sha256_file(run_r9.RUNNER)}
    for arm, per_seed in finals.items():
        for seed, (code, prose) in zip(plan["seeds"], per_seed):
            if (arm, seed) in missing:
                continue
            provenance = dict(digests)
            if tamper == (arm, seed):
                provenance["runner_sha256"] = "0" * 64
            record = {"arm": arm, "seed": seed, "status": "COMPLETED", "provenance": provenance,
                      "corpus": {"digest": plan["corpus"]["digest"]},
                      "final": {"val_code_bits_per_byte": code, "val_prose_bits_per_byte": prose}}
            if arm in ("A", "A0"):
                record["gates_zeroed"] = {"val_code_bits_per_byte": code + 0.5, "val_prose_bits_per_byte": prose + 0.5}
            (folder / f"cgc_{arm}_s{seed}.json").write_text(json.dumps(record), encoding="utf-8")


def _combine(folder):
    return run_r9.command_combine(argparse.Namespace(inputs=str(folder)))


def test_combine_applies_the_frozen_verdicts_and_refuses_tampered_or_missing_records(plan, tmp_path):
    a = [(1.70, 2.20), (1.71, 2.21), (1.72, 2.19)]
    worse = lambda d: [(c + d, p + d) for c, p in a]                    # noqa: E731
    cases = {
        "CGC_BENEFIT_QUALIFIED": {"A": a, "B": worse(0.05), "C": worse(0.03), "C_mac": worse(0.025)},
        "FFN_PREFERRED": {"A": a, "B": worse(0.0), "C": worse(-0.03), "C_mac": worse(-0.03)},
        "CGC_NOT_ESTABLISHED": {"A": a, "B": worse(0.05), "C": worse(0.03), "C_mac": worse(0.005)},
    }
    for verdict, finals in cases.items():
        folder = tmp_path / verdict
        folder.mkdir()
        _fake_records(folder, plan, finals)
        out = _combine(folder)
        assert out["primary_verdict"] == verdict, (verdict, out["comparisons"])
        assert out["comparisons"]["D_vs_A"]["code"]["class"] == "A_BETTER"   # removal raises the loss
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    _fake_records(tampered, plan, cases["CGC_BENEFIT_QUALIFIED"], tamper=("C", 1))
    out = _combine(tampered)
    assert out["primary_verdict"] == "NONQUAL" and out["refused"][0]["file"] == "cgc_C_s1.json"
    gap = tmp_path / "missing"
    gap.mkdir()
    _fake_records(gap, plan, cases["CGC_BENEFIT_QUALIFIED"], missing={("C_mac", 2)})
    out = _combine(gap)
    assert out["primary_verdict"] == "NONQUAL" and out["run_status"]["C_mac_s2"] == "MISSING"


def test_the_committed_study_record_is_what_combine_computes_from_the_committed_runs():
    recorded = json.loads((run_r9.RESULTS / "study_cgc_attribution.json").read_text(encoding="utf-8"))
    fresh = _combine(run_r9.RESULTS)
    assert fresh["refused"] == [] and recorded["refused"] == []
    assert fresh["primary_verdict"] == recorded["primary_verdict"]
    assert fresh["comparisons"] == recorded["comparisons"]
    assert fresh["inputs"] == recorded["inputs"]                           # every run record byte-identical


def test_the_smoke_path_runs():
    out = run_r9.command_smoke(argparse.Namespace())
    assert set(out["final"]) == {"A", "C"}


def test_the_r9_runners_have_exactly_one_write_site():
    source = run_r9.RUNNER.read_text(encoding="utf-8")
    assert source.count(".write_text(") == 1 and "def _write(" in source
    p3 = (run_r9.RUNNER.parent / "run_r9_p3.py").read_text(encoding="utf-8")
    assert p3.count(".write_text(") == 1 and "def _write(" in p3
