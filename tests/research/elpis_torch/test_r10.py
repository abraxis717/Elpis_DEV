"""Elpis Torch R10: the arms, the prequential evaluator and its state-integrity controls, and the preregistered
decision rules (on fabricated records). The confirmatory results themselves are records (``results/r10``).

Controls required by the R10 directive (P5): independent sessions, reset/restart equivalence, no future-token
influence, stable causal chunking, no inference-time parameter update, exact CGC removal.
"""
from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import importlib.util
import json

import pytest

from .conftest import SKIP_REASON

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch import adaptation_tasks as at  # noqa: E402
from research.elpis_torch.experiments import run_r10 as r  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return r.build(r.ARMS["A"], 3).eval()


def test_arms_differ_only_in_the_routed_channel_and_the_attention():
    with torch.device("meta"):
        a = dict(r.build(r.ARMS["A"], 0).named_parameters())
        for name in ("B", "C", "C_mac"):
            other = dict(r.build(r.ARMS[name], 0).named_parameters())
            shared = {k for k in a if ".experts." not in k}
            assert {k for k in other if ".experts." not in k} == shared
            assert all(a[k].shape == other[k].shape for k in shared)
            assert not any(".cgc" in k for k in other)
        d = dict(r.build(r.ARMS["D_A"], 0).named_parameters())
        assert set(a) - set(d) == {k for k in a if ".attn." in k} and set(d) <= set(a)
        assert not any(name.startswith(("encoder.", "lookup.")) for name in a)          # R5 history is off


def test_the_arms_are_matched_as_declared():
    arms = r.command_plan(None)["arms"]
    assert abs(arms["C"]["total_parameters"] / arms["A"]["total_parameters"] - 1) < 3e-4       # parameters
    assert abs(arms["C_mac"]["expert_macs"] / arms["A"]["expert_macs"] - 1) < 3e-3            # multiply-adds
    assert arms["D_A"]["total_parameters"] == 4_635_576 and arms["D_C"]["total_parameters"] == 4_634_536  # R9 A, C
    ple = {name: a["parameters_by_group"]["ple_table"] for name, a in arms.items()}
    assert len(set(ple.values())) == 1                                                        # one PLE treatment


def test_no_future_token_influences_a_prediction(model):
    e = at.eval_set("A", 1, split_seed=11)[0]
    base = r.predict(model, [e])
    p = e.probes[30]
    cut = dataclasses.replace(e, tokens=e.tokens[:p.pos + 1] + [at.tok(".")] * (len(e.tokens) - p.pos - 1),
                              probes=e.probes[:31])
    after = r.predict(model, [cut])
    assert [x[1] for x in base[:31]] == [x[1] for x in after]
    assert max(abs(x[2] - y[2]) for x, y in zip(base[:31], after)) < 1e-4


def test_episodes_are_independent_sessions(model):
    eps = at.eval_set("C_long", 3, split_seed=12)
    together = r.predict(model, eps)
    alone = [x for e in eps for x in r.predict(model, [e])]
    assert [x[1] for x in together] == [x[1] for x in alone]
    assert max(abs(x[2] - y[2]) for x, y in zip(together, alone)) < 1e-4


def test_causal_chunking_is_stable(model):
    eps = at.eval_set("B", 2, split_seed=13)
    base = r.predict(model, eps)
    split = r.predict(model, eps, reset_at=37, reset="none")                 # an extra, misaligned boundary
    assert [x[1] for x in base] == [x[1] for x in split]
    assert max(abs(x[2] - y[2]) for x, y in zip(base, split)) < 1e-4


def test_a_reset_changes_only_what_follows_it_and_a_restart_equals_a_fresh_session(model):
    eps = at.eval_set("C_long", 4, split_seed=14)
    boundary = at.first_probe_index(eps[0])
    base = r.predict(model, eps)
    flat = [p for e in eps for p in e.probes]
    for what in ("ersa", "attention"):
        reset = r.predict(model, eps, reset_at=boundary, reset=what)
        for p, x, y in zip(flat, base, reset):
            if p.pos < boundary:
                assert x[1] == y[1] and abs(x[2] - y[2]) < 1e-5
        assert any(abs(x[2] - y[2]) > 1e-6 for p, x, y in zip(flat, base, reset) if p.pos >= boundary), what
        restart = r.predict(model, eps, reset_at=0, reset=what)              # reset before anything is seen
        assert [x[1] for x in restart] == [x[1] for x in base]


def _byte_digest(model) -> str:
    """SHA-256 over every state-dict entry's name and raw bytes, read through ctypes (no NumPy): the convention of
    ``model.py`` and ``run_r7.py``."""
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        data = p.detach().contiguous()
        h.update(name.encode())
        h.update(ctypes.string_at(data.data_ptr(), data.numel() * data.element_size()))
    return h.hexdigest()


def test_evaluation_and_diagnostics_update_no_parameter(model):
    before = _byte_digest(model)
    diag = {v: at.eval_set(v, 2, split_seed=15) for v in at.TRAIN_VARIANTS}
    r.diagnostics(model, r.ARMS["A"], diag)
    r.evaluate(model, diag)
    assert _byte_digest(model) == before
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.skipif(importlib.util.find_spec("numpy") is None,
                    reason="NOT_RUN: NumPy is not installed here, and the frozen R10 runner hashes parameters through "
                           "NumPy (a documented defect; EVIDENCE_R10.md, limitations). Not a PASS")
def test_the_runner_parameter_digest_is_the_byte_digest(model):
    """The digest behind every record's ``parameters_unchanged_by_evaluation`` hashes the same bytes."""
    assert r._parameters_digest(model) == _byte_digest(model)


def test_zeroed_gates_remove_cgc_exactly_and_restore():
    model = r.build(r.ARMS["A"], 1).double().eval()
    expert = model.blocks[0].sparse.experts[0]
    u = torch.randn(5, 64, dtype=torch.float64)
    with torch.no_grad():
        full = expert(u)
        previous = r.set_cgc_gates(model, 0.0)
        assert torch.equal(expert(u), expert.ffn(u))
        r.set_cgc_gates(model, previous)
        assert torch.equal(expert(u), full)


def test_the_summary_reports_every_preregistered_measure(model):
    sets = {v: at.eval_set(v, 3, split_seed=16) for v in at.TRAIN_VARIANTS}
    summary = r.evaluate(model, sets)
    assert set(summary["A"]["by_k"]) == {str(k) for k in r.K_GRID}
    for key in ("inside_window", "outside_window", "collision_error_rate", "recency_error_rate"):
        assert key in summary["A"]
    for key in ("pre_switch", "post_switch_new_regime", "post_switch_reentry", "recovery_1_3"):
        assert key in summary["B"]
    for v in ("C_short", "C_long", "C_gap"):
        for key in ("pre", "post", "mid", "reentry", "forgetting", "post_recency_error_rate"):
            assert key in summary[v]
    ends = r.endpoints(summary)
    assert set(ends) == {"ADAPT", "RETAIN", "RETAIN_pre", "RETAIN_change"}


ANALYSIS = {"delta_adapt": 0.03, "delta_retain": 0.05,
            "learned_threshold": {"ADAPT": 0.3, "RETAIN": 0.2, "RETAIN_pre": 0.4},
            "screen_min_positive": 2, "t_critical_one_sided": 1.796, "latency_material_ratio": 1.1}


def _fabricate(seeds, a=0.6, c=0.6, cm=0.6, b=0.55, da=0.4, dc=0.4, pre=0.8, held_gap=0.0, noise=0.0):
    out = {}
    for n, s in enumerate(seeds):
        jitter = noise * ((n % 3) - 1)
        for arm, value in (("A", a), ("C", c), ("C_mac", cm), ("B", b), ("D_A", da), ("D_C", dc)):
            v = value + (jitter if arm == "A" else 0.0)
            out[(arm, s)] = {"ADAPT": v, "RETAIN": v, "RETAIN_pre": pre,
                             "heldout_adapt": v + (held_gap if arm == "A" else 0.0),
                             "heldout_retain": v + (held_gap if arm == "A" else 0.0)}
    return out


def test_the_preregistered_rules_on_fabricated_records():
    s3, s12 = [0, 1, 2], list(range(100, 112))
    out = r.analyse(_fabricate(s3, a=0.1, c=0.1, cm=0.1, b=0.1, pre=0.2), s3, ANALYSIS, "screening")
    assert out["endpoints"]["ADAPT"]["verdict"] == "ASSAY_NOT_LEARNED"
    assert out["endpoints"]["RETAIN"]["verdict"] == "ASSAY_NOT_LEARNED"
    out = r.analyse(_fabricate(s3, a=0.1, c=0.1, cm=0.1, b=0.1, pre=0.9), s3, ANALYSIS, "screening")
    assert out["endpoints"]["RETAIN"]["verdict"] == "ASSAY_NOT_LEARNED"      # bindings learned, retention at floor
    out = r.analyse(_fabricate(s3), s3, ANALYSIS, "screening")
    assert out["endpoints"]["ADAPT"]["verdict"] == "SCREEN_NOT_FAVORABLE"
    out = r.analyse(_fabricate(s3, a=0.7), s3, ANALYSIS, "screening")
    assert out["endpoints"]["ADAPT"]["verdict"] == "SCREEN_FAVORABLE"
    assert out["endpoints"]["RETAIN"]["verdict"] == "SCREEN_FAVORABLE"
    out = r.analyse(_fabricate(s3, a=0.7, cm=0.69), s3, ANALYSIS, "screening")              # one matched arm keeps up
    assert out["endpoints"]["ADAPT"]["verdict"] == "SCREEN_NOT_FAVORABLE"
    out = r.analyse(_fabricate(s12, a=0.7, noise=0.01, da=0.5, dc=0.4), s12, ANALYSIS, "confirmation")
    e = out["endpoints"]["ADAPT"]
    assert e["verdict"] == "CONFIRMED" and e["attribution"] == "PERSISTS_WITHOUT_ATTENTION"
    out = r.analyse(_fabricate(s12, a=0.7, noise=0.01, held_gap=-0.2), s12, ANALYSIS, "confirmation")
    assert out["endpoints"]["ADAPT"]["verdict"] == "CONFIRMED_IN_DISTRIBUTION_ONLY"
    assert out["endpoints"]["ADAPT"]["attribution"] == "ATTENTION_DEPENDENT"
    out = r.analyse(_fabricate(s12, a=0.62, noise=0.05), s12, ANALYSIS, "confirmation")     # below delta / noisy
    assert out["endpoints"]["ADAPT"]["verdict"] == "NOT_CONFIRMED"
    partial = _fabricate(s3, a=0.7)
    del partial[("C_mac", 2)]
    out = r.analyse(partial, s3, ANALYSIS, "screening")
    assert out["endpoints"]["ADAPT"]["verdict"] == "NONQUAL"
    # the floor rule never masks a CGC benefit: A alone lifting retention off the floor makes the endpoint learned
    out = r.analyse(_fabricate(s3, a=0.3, c=0.1, cm=0.1, b=0.1, pre=0.9), s3, ANALYSIS, "screening")
    assert out["endpoints"]["RETAIN"]["learned"] and out["endpoints"]["RETAIN"]["verdict"] == "SCREEN_FAVORABLE"


PLAN = {"seeds": {"screening": [0, 1, 2], "confirmation": list(range(100, 112))},
        "confirmation_arms": ["A", "C", "C_mac", "D_A", "D_C"], "analysis": ANALYSIS}


def _stages(screen_records, conf_records=None):
    out = {"screening": r.analyse(screen_records, [0, 1, 2], ANALYSIS, "screening")}
    if conf_records is not None:
        blocks = r.complete_blocks(conf_records, PLAN["seeds"]["confirmation"], PLAN["confirmation_arms"])
        conf = r.analyse(conf_records, blocks, ANALYSIS, "confirmation") if len(blocks) >= 2 else {"endpoints": {}}
        conf.update(status="COMPLETE" if len(blocks) == 12 else "PARTIAL")
        out["confirmation"] = conf
    return out


def test_the_overall_verdicts_follow_the_preregistered_mapping():
    s3, s12 = [0, 1, 2], list(range(100, 112))
    v = r.overall(_stages(_fabricate(s3)), PLAN, None)
    assert v["ADAPT"]["verdict"] == "NO_INCREMENTAL_CGC_ADAPTATION_BENEFIT_ESTABLISHED"
    assert v["RETAIN"]["verdict"] == "NO_INCREMENTAL_CGC_RETENTION_BENEFIT_ESTABLISHED"
    v = r.overall(_stages(_fabricate(s3, a=0.1, c=0.1, cm=0.1, b=0.1, pre=0.2)), PLAN, None)
    assert v["ADAPT"]["verdict"] == v["RETAIN"]["verdict"] == "ASSAY_NOT_LEARNED"
    v = r.overall(_stages(_fabricate(s3, a=0.7)), PLAN, None)                    # favorable, never confirmed
    assert v["ADAPT"]["verdict"] == "PARTIAL"
    short = _fabricate(s12, a=0.7, noise=0.01, da=0.5, dc=0.4)
    for arm in ("A", "C"):
        del short[(arm, 111)]
    v = r.overall(_stages(_fabricate(s3, a=0.7), short), PLAN, None)            # budget ran out: 11 blocks
    assert v["ADAPT"]["verdict"] == "PARTIAL"
    latency = {"arms": {"A": {"decode_ms_p50": 1.2}, "C": {"decode_ms_p50": 1.0}}}
    v = r.overall(_stages(_fabricate(s3, a=0.7), _fabricate(s12, a=0.7, noise=0.01, da=0.5, dc=0.4)), PLAN, latency)
    assert v["ADAPT"]["verdict"] == "CGC_ADAPTATION_BENEFIT_CONFIRMED"
    assert v["ADAPT"]["held_out"] == "SURVIVES" and v["ADAPT"]["attribution"] == "PERSISTS_WITHOUT_ATTENTION"
    assert v["ADAPT"]["latency"] == "QUALITY_LATENCY_TRADEOFF"
    v = r.overall(_stages(_fabricate(s3, a=0.7), _fabricate(s12, a=0.62, noise=0.05)), PLAN, None)
    assert v["ADAPT"]["verdict"] == "NO_INCREMENTAL_CGC_ADAPTATION_BENEFIT_ESTABLISHED"


@pytest.mark.parametrize("edit", ["window", "arms", "k_grid", "task_spec", "confirmation_arms"])
def test_a_plan_that_drifted_from_the_runner_is_refused(tmp_path, monkeypatch, edit):
    plan = json.loads(r.PREREGISTRATION.read_text(encoding="utf-8"))
    assert r.frozen()["phase"] == r.PHASE
    if edit == "window":
        plan["model"]["window"] = 128
    elif edit == "arms":
        plan["arms"]["C"]["ffn_hidden"] = 124
    elif edit == "k_grid":
        plan["evaluation"]["k_grid"] = [0, 1, 2, 4, 8]
    elif edit == "task_spec":
        plan["task_spec"]["c_gap"] = 32
    else:
        plan["confirmation_arms"] = ["C", "C_mac", "X"]
    path = tmp_path / "preregistration.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(r, "PREREGISTRATION", path)
    with pytest.raises(ValueError, match="does not describe this runner"):
        r.frozen()


def test_the_train_command_refuses_unplanned_seeds_and_arms():
    import argparse
    with pytest.raises(SystemExit, match="not a preregistered"):
        r.command_train(argparse.Namespace(arm="A", seed=900))
    with pytest.raises(SystemExit, match="not a confirmation arm"):
        r.command_train(argparse.Namespace(arm="B", seed=100))


def test_the_r10_runner_has_exactly_one_write_site():
    source = r.RUNNER.read_text(encoding="utf-8")
    assert source.count(".write_text(") == 1 and "def _write(" in source
    assert "torch.save" not in source and ".backward(" not in source       # the runner never steps a gradient itself
