"""Elpis Torch R11: the microscopic CGC specialist profile, its backward compatibility, its mathematics, its gate and
routing mechanics, the block-order causal path, and the Stage A/B harness contracts. Results are records."""
from __future__ import annotations

import ctypes
import hashlib
import math

import pytest

from .conftest import SKIP_REASON

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch import cgc_specialist as cs  # noqa: E402
from research.elpis_torch import specialist_tasks as st  # noqa: E402
from research.elpis_torch.cgc import CGCCore, CGCTransform  # noqa: E402
from research.elpis_torch.config import ConfigError, ModelConfig  # noqa: E402
from research.elpis_torch.sparse_channel import ChannelExpert, SparseChannel, SparseConfig  # noqa: E402
from research.elpis_torch.upstream import cgc_ecsg_e0 as upstream  # noqa: E402
from research.elpis_torch.experiments import run_r11 as r  # noqa: E402


def _model(**kw):
    base = dict(vocab_size=2, d_model=64, n_layers=1, n_heads=1, max_seq_len=8, channel="ffn+cgc", ffn_hidden=86)
    base.update(kw)
    return ModelConfig(**base)


# -- backward compatibility ------------------------------------------------------------------------------------------

def test_the_historical_expert_is_unchanged_parameter_for_parameter_and_in_its_arithmetic():
    m = _model()
    s = SparseConfig(experts=1, top_k=1, ffn_hidden=86, cgc_channels=48, cgc_bottleneck=16, cgc_units=4)
    assert (s.cgc_gate_mode, s.cgc_preact_bound, s.mlp_width) == ("learned", "model", None)
    e = ChannelExpert(m, s, torch.float64)
    names = [n for n, _ in e.named_parameters()]
    assert names == ["cgc_gate", "ffn.gate.weight", "ffn.up.weight", "ffn.out.weight", "cgc.proj_in.weight",
                     "cgc.core.weight", "cgc.proj_out.weight"]                 # the historical order (main's)
    assert e.cgc.core.preact_bound == m.cgc_preact_bound == 3.0          # the model's bound, as before
    u = torch.randn(9, 64, dtype=torch.float64)
    with torch.no_grad():
        e.cgc_gate.fill_(0.37)
        assert torch.equal(e(u), e.ffn(u) + e.cgc_gate * e.cgc(u))      # the historical expression, bit for bit


def test_the_r6_scale_layouts_still_count_exactly_what_they_counted():
    from research.elpis_torch.experiments import run_r6 as r6
    from research.elpis_torch.sparse_channel import active_parameters
    expected = {"r6_500m": (507683413, 286499353), "r6_1b": (994692069, 433050471)}
    for name, (total, active) in expected.items():
        counts = active_parameters(r6.scale_model(name))
        assert (counts["total"], counts["active_per_token"]) == (total, active), name


def test_invalid_r11_options_are_refused():
    with pytest.raises(ConfigError):
        SparseConfig(cgc_gate_mode="adaptive")
    with pytest.raises(ConfigError):
        SparseConfig(cgc_preact_bound=-1.0)
    with pytest.raises(ConfigError):
        SparseConfig(kind="ffn+mlp")                     # the control needs an explicit bottleneck


# -- the profile -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("hidden", [64, 768, 1024])
def test_the_micro_adapter_is_8h_plus_121_and_the_core_is_r97w_120(hidden):
    a = cs.adapter_parameters(hidden)
    assert a == {"proj_in": 5 * hidden, "core": 120, "proj_out": 3 * hidden, "gate": 1, "total": 8 * hidden + 121}
    assert CGCTransform.parameter_count(hidden, 3, 8, 5) + 1 == a["total"]


@pytest.mark.parametrize("profile,bound", [("E0_REFERENCE", None), ("BOUNDED_SPECIALIST", 3.0)])
def test_the_profiles_build_bounded_experts_inside_the_historical_class(profile, bound):
    m = _model()
    s = cs.micro_sparse(SparseConfig(experts=2, top_k=1, ffn_hidden=86), profile=profile)
    e = ChannelExpert(m, s, torch.float32)
    assert (e.cgc.core.in_dim, e.cgc.core.channels, e.cgc.core.units) == (5, 3, 8)
    assert e.cgc.core.preact_bound == bound
    acc = cs.check_adapter(e, 64, 86)
    assert acc["counted"]["total"] == 633 and acc["budget"]["within_budget"]
    assert acc["budget"]["adapter_fraction"] < 0.05


def test_every_r6_expert_and_dense_channel_fits_the_5_percent_budget_in_the_micro_profile():
    for hidden, ffn in ((768, 1024), (1024, 683), (768, 2048), (1024, 2731)):
        assert cs.specialist_budget(hidden, ffn, cs.adapter_parameters(hidden)["total"])["within_budget"]


def test_the_micro_core_is_e0_cell_by_cell_against_the_pinned_upstream_source():
    torch.manual_seed(0)
    core = CGCCore(5, 3, 8, preact_bound=None, dtype=torch.float64)
    with torch.no_grad():
        core.weight.copy_(torch.randn(5, 3, 8, dtype=torch.float64) * 0.4)
    x = torch.randn(33, 5, dtype=torch.float64)
    y = core(x)
    for c in range(3):
        ref = upstream.forward(core.weight[:, c, :].detach().contiguous(), x)
        assert float((y[:, c].detach() - ref).abs().max()) <= 1e-12
    target = torch.randn(33, dtype=torch.float64)
    w = core.weight[:, 1, :].detach().clone().requires_grad_(True)
    upstream.mse_loss(w, x, target).backward()
    assert float((w.grad - upstream.analytic_grad(w.detach(), x, target)).abs().max()) <= 1e-12


def test_the_bounded_specialist_is_e0_derived_not_e0():
    core = CGCCore(5, 3, 8, preact_bound=3.0, dtype=torch.float64)
    with torch.no_grad():
        core.weight.copy_(torch.randn(5, 3, 8, dtype=torch.float64))
    x = torch.randn(64, 5, dtype=torch.float64) * 3
    raw = (x @ core.weight.reshape(5, 24)).detach()
    bounded = core.preactivations(x).reshape(-1, 24).detach()
    assert float(bounded.detach().abs().max()) <= 3.0
    assert bool(((bounded - raw).abs() <= raw.abs() ** 3 / 27 + 1e-12).all())          # |u|^3 / (3 b^2)
    assert float((bounded - raw).detach().abs().max()) > 1e-3                           # it is NOT E0 here
    assert float(core(x).detach().abs().max()) <= core.output_ceiling()


# -- gate modes and the matched control ------------------------------------------------------------------------------

def test_the_fixed_gate_has_no_parameter_and_contributes_the_full_branch():
    m = _model()
    s = cs.micro_sparse(SparseConfig(experts=1, top_k=1, ffn_hidden=86), profile="BOUNDED_SPECIALIST",
                        gate_mode="fixed")
    e = ChannelExpert(m, s, torch.float64)
    assert e.cgc_gate is None and "cgc_gate" not in dict(e.named_parameters())
    u = torch.randn(7, 64, dtype=torch.float64)
    with torch.no_grad():
        assert torch.equal(e(u), e.ffn(u) + e.cgc(u))


def test_the_gelu_control_matches_the_cgc_adapter_budget():
    m = _model()
    s = cs.micro_sparse(SparseConfig(experts=1, top_k=1, ffn_hidden=86), profile="BOUNDED_SPECIALIST",
                        gate_mode="fixed", kind="ffn+mlp")
    e = ChannelExpert(m, s, torch.float32)
    assert e.cgc is None and e.mlp is not None
    assert sum(p.numel() for p in e.mlp.core_parameters()) == 120
    assert sum(p.numel() for p in e.mlp.parameters()) == cs.adapter_parameters(64, gate=False)["total"]


def test_a_zero_gate_starves_the_branch_of_every_gradient_but_its_own():
    """Consequence of zero-gate initialization: at g = 0 the CGC's projections and core get exactly zero gradient;
    only the gate itself can move first."""
    m = _model()
    s = cs.micro_sparse(SparseConfig(experts=1, top_k=1, ffn_hidden=86, gate_init=0.0), profile="E0_REFERENCE")
    e = ChannelExpert(m, s, torch.float64)
    with torch.no_grad():
        e.cgc_gate.zero_()
        for p in e.cgc.parameters():
            p.normal_(0, 0.3)
    e(torch.randn(11, 64, dtype=torch.float64)).pow(2).sum().backward()
    assert all(float(p.grad.abs().max()) == 0.0 for p in e.cgc.parameters())
    assert float(e.cgc_gate.grad.abs()) > 0.0


def test_an_expert_receives_gradient_only_from_the_tokens_routed_to_it():
    m = _model()
    s = cs.micro_sparse(SparseConfig(experts=2, top_k=1, ffn_hidden=86, gate_init=1.0), profile="E0_REFERENCE")
    channel = SparseChannel(m, s, torch.float64)
    with torch.no_grad():
        channel.router.weight.zero_()
        channel.router.weight[0, 0] = 10.0                    # positive first coordinate -> expert 0
        for p in channel.parameters():
            if p.ndim >= 2 and p is not channel.router.weight:
                p.normal_(0, 0.2)
    u = torch.randn(16, 64, dtype=torch.float64)
    u[:, 0] = u[:, 0].abs() + 0.1                             # every token routes to expert 0
    channel(u).pow(2).sum().backward()
    assert channel.report["stats"]["counts"] == [16, 0]
    assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in channel.experts[1].parameters())
    assert float(channel.experts[0].cgc.core.weight.grad.abs().max()) > 0.0


# -- the causal path through the block sequence -----------------------------------------------------------------------

def test_a_blocks_cgc_cannot_change_that_blocks_ersa_state_only_later_blocks():
    from research.elpis_torch.ersa_model import ErsaConfig, ErsaLM
    m = ModelConfig(vocab_size=259, d_model=32, n_layers=2, n_heads=2, max_seq_len=64, channel="ffn+cgc",
                    cgc_bottleneck=5, cgc_channels=3, cgc_units=8, dtype="float64")
    model = ErsaLM(ErsaConfig(m, ersa_heads=2, key_dim=8, value_dim=8, chunk=8)).eval()
    tokens = torch.randint(0, 259, (1, 8), generator=torch.Generator().manual_seed(3))

    def run(scale0, scale1):
        with torch.no_grad():
            model.blocks[0].cgc_gate.fill_(scale0)
            model.blocks[1].cgc_gate.fill_(scale1)
        state = model.new_state(1)
        with torch.no_grad():
            logits = model(tokens, state)
        return [c.clone() for c in state.cells], logits

    base_cells, base_logits = run(0.0, 0.0)
    cells0, logits0 = run(1.0, 0.0)                           # block 0's CGC on
    assert torch.equal(cells0[0], base_cells[0])              # its own ERSA state: unchanged
    assert not torch.equal(cells0[1], base_cells[1])          # the next block's state: changed
    cells1, logits1 = run(0.0, 1.0)                           # last block's CGC on
    assert all(torch.equal(a, b) for a, b in zip(cells1, base_cells))     # no ERSA state anywhere changes
    assert not torch.equal(logits1, base_logits)                         # only the readout does


# -- Stage A learners ---------------------------------------------------------------------------------------------

def test_the_stage_a_e0_learner_is_e0_and_the_projected_one_is_its_reparametrization():
    learner = st.Learner("e0")
    core = learner.init_core(5, "TEST")
    x = st.teacher("TEST", "smooth", 1, "TRAIN", 0).support[0]
    y = learner.predict(core, (), {}, x)
    ref = torch.stack([upstream.forward(core[0][:, c, :].contiguous(), x) for c in range(3)], 1)
    assert float((y - ref).abs().max()) <= 1e-12
    proj = st.Learner("e0", hidden=64)
    fixed = proj.fixed(5, "TEST")
    exact = (torch.linalg.pinv(fixed["E"]), torch.linalg.pinv(fixed["R"]))       # P_in = E^+, P_out = R^+
    assert float((proj.predict(core, exact, fixed, x) - y).abs().max()) <= 1e-10


def test_the_inner_loop_moves_only_the_120_core_parameters():
    learner = st.Learner("e0", hidden=64)
    fixed, core, proj = learner.fixed(1, "T"), learner.init_core(1, "T"), learner.init_projections(1, "T")
    e = st.teacher("T", "conditional", 1, "TRAIN", 0)
    before = [p.clone() for p in proj]
    fast, _ = st.adapt(learner, core, proj, fixed, e.support, 8, 0.08, 2.0)
    assert all(torch.equal(a, b) for a, b in zip(before, proj))
    assert sum(p.numel() for p in fast) == 120 and not torch.equal(fast[0], core[0])


def test_r11_episodes_are_its_own_and_deterministic():
    a = st.teacher("R11A", "smooth", 101, "TEST", 0)
    b = st.teacher("R11A", "smooth", 101, "TEST", 0)
    c = st.teacher("R97W", "smooth", 101, "TEST", 0)
    assert torch.equal(a.query[1], b.query[1]) and not torch.equal(a.query[1], c.query[1])
    assert not torch.equal(st.teacher("R11A", "smooth", 101, "DEV", 0).query[1], a.query[1])


# -- Stage B harness ---------------------------------------------------------------------------------------------------

def _byte_digest(module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(module.state_dict().items()):
        data = p.detach().contiguous()
        h.update(name.encode())
        h.update(ctypes.string_at(data.data_ptr(), data.numel() * data.element_size()))
    return h.hexdigest()


@pytest.mark.parametrize("arm", ["G0", "G1", "G2", "G3", "G2_POOLED"])
def test_the_stage_b_arms_have_their_declared_anatomy(arm):
    e, _, s = st.build_arm(arm, "BOUNDED_SPECIALIST", 3)
    names = set(dict(e.named_parameters()))
    if arm == "G0":
        assert e.cgc is None and e.mlp is None
    elif arm == "G1":
        assert "cgc_gate" in names and e.cgc.core.preact_bound == 3.0
    elif arm == "G3":
        assert e.mlp is not None and e.cgc is None and "cgc_gate" not in names
    else:
        assert e.cgc is not None and "cgc_gate" not in names


def test_evaluation_adapts_a_copy_and_leaves_every_weight_byte_identical():
    plan = {"outer_steps": 2, "meta_batch": 1, "adam_lr": 1e-3, "select_every": 1, "dev_episodes": 1}
    out = r.train_stage_b("G2", "BOUNDED_SPECIALIST", "smooth", 3, "TEST", plan)
    other = r.train_stage_b("G2", "BOUNDED_SPECIALIST", "conditional", 4, "TEST", plan)   # a different specialist
    before = _byte_digest(out["expert"])
    rows = st.evaluate_arm(out["expert"], out["context"], "smooth", 3, "TEST", "TEST", 2)
    swap = st.evaluate_arm(out["expert"], out["context"], "smooth", 3, "TEST", "TEST", 2, start_from=other["expert"])
    assert _byte_digest(out["expert"]) == before
    assert rows[0]["curve"]["8"][0] != rows[0]["curve"]["0"][0]                      # the copy did adapt
    assert swap[0]["curve"]["0"] != rows[0]["curve"]["0"]                            # another specialist was used


def test_the_branch_initialization_is_the_stage_a_adapter_rule_for_both_specialists():
    for arm in ("G2", "G3"):
        e, _, _ = st.build_arm(arm, "BOUNDED_SPECIALIST", 4)
        info = st.init_branch_like_stage_a(e, 4, "T")
        assert info["proj_in_std"] == 64 ** -0.5 and info["proj_out_std"] == 3 ** -0.5
    e, _, _ = st.build_arm("G1", "BOUNDED_SPECIALIST", 4)
    st.init_branch_like_stage_a(e, 4, "T")
    assert float(e.cgc_gate) == pytest.approx(0.1)                                   # the historical gate is kept


# -- decision rules and runner discipline ---------------------------------------------------------------------------------

def test_the_sign_test_and_the_better_rule():
    assert r.sign_test_p([-1.0] * 10) == pytest.approx(1 / 1024)
    assert r.sign_test_p([1.0] * 10) == pytest.approx(1.0)
    rule = {"max_ratio": 0.95, "alpha": 0.05, "min_group_fraction": 2 / 3}
    assert r.better([[math.log(0.8)] * 12] * 3, rule)["better"]
    assert not r.better([[math.log(0.97)] * 12] * 3, rule)["better"]                 # below the useful effect
    mixed = [[math.log(0.5)] * 12, [math.log(1.2)] * 12, [math.log(1.2)] * 12]
    assert not r.better(mixed, rule)["better"]                                       # one group carries it


def test_the_r11_runner_writes_in_one_place_and_never_saves_weights():
    source = r.RUNNER.read_text(encoding="utf-8")
    assert source.count(".write_text(") == 1 and "def _write(" in source
    assert "torch.save" not in source


@pytest.mark.parametrize("edit", ["geometry", "bounds", "stage_a_arms", "stage_b_arms", "protocol"])
def test_a_plan_that_drifted_from_the_runner_is_refused(tmp_path, monkeypatch, edit):
    import json
    plan = json.loads(r.PREREGISTRATION.read_text(encoding="utf-8"))
    assert r.frozen()["phase"] == r.PHASE
    if edit == "geometry":
        plan["profile"]["geometry"] = [16, 48, 4]
    elif edit == "bounds":
        plan["profile"]["bounds"]["BOUNDED_SPECIALIST"] = 5.0
    elif edit == "stage_a_arms":
        plan["stage_a"]["arms"]["E0_META"][1] = 64
    elif edit == "stage_b_arms":
        plan["stage_b"]["arms"]["G2"]["gate_mode"] = "learned"
    else:
        plan["stage_a"]["protocol"]["inner_steps"] = 16
    path = tmp_path / "preregistration.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(r, "PREREGISTRATION", path)
    with pytest.raises(ValueError, match="does not describe this runner"):
        r.frozen()


def test_confirmation_commands_refuse_unplanned_seeds():
    import argparse
    with pytest.raises(SystemExit, match="not a preregistered"):
        r.command_stage_a(argparse.Namespace(dev=False, family="smooth", seed=11, lr_proj=None))


# -- Stage C (amendment) --------------------------------------------------------------------------------------------------

def test_the_stage_c_stack_is_the_repository_blocks_with_the_micro_specialist():
    from research.elpis_torch import specialist_stack as ss
    from research.elpis_torch.ersa_model import ErsaBlock
    g2 = ss.RegressionStack("G2", "BOUNDED_SPECIALIST", 7)
    assert all(isinstance(b, ErsaBlock) and b.sparse is not None for b in g2.blocks)
    expert = g2.blocks[0].sparse.experts[0]
    assert expert.cgc_gate is None and expert.cgc.core.weight.shape == (5, 3, 8)
    assert cs.check_adapter(expert, 64, 86)["counted"]["total"] == 632
    g0 = ss.RegressionStack("G0", "BOUNDED_SPECIALIST", 7)
    assert g0.blocks[0].sparse.experts[0].cgc is None


def test_the_stage_c_stack_is_causal_and_its_ablation_removes_only_the_specialist():
    from research.elpis_torch import specialist_stack as ss
    model = ss.RegressionStack("G2", "BOUNDED_SPECIALIST", 7).eval()
    tokens, _ = ss.batch_tokens("TEST", ["smooth"], 7, "TEST", 0, 1, 8)
    with torch.no_grad():
        full = model(tokens, model.new_state(1))
        prefix = model(tokens[:, :20], model.new_state(1))
        assert float((full[:, :20] - prefix).abs().max()) < 1e-5            # no future-token influence
        ablated = model(tokens, model.new_state(1), ablate_blocks=(0, 1))
        assert float((ablated - full).abs().max()) > 1e-6
    g0 = ss.RegressionStack("G0", "BOUNDED_SPECIALIST", 7).eval()
    with torch.no_grad():
        assert torch.allclose(g0(tokens, g0.new_state(1)), g0(tokens, g0.new_state(1), ablate_blocks=(0, 1)))


def test_stage_c_evaluation_leaves_the_weights_byte_identical():
    from research.elpis_torch import specialist_stack as ss
    model = ss.RegressionStack("G2", "BOUNDED_SPECIALIST", 7)
    before = _byte_digest(model)
    ss.evaluate_stack(model, "TEST", "smooth", 7, 1)
    assert _byte_digest(model) == before


def test_a_stage_c_amendment_that_drifted_from_its_runner_is_refused(tmp_path, monkeypatch):
    import json
    from research.elpis_torch.experiments import run_r11_stage_c as c
    plan = json.loads(c.AMENDMENT.read_text(encoding="utf-8"))
    assert c.frozen()["phase"] == c.PHASE
    plan["k_grid"] = [0, 32]
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(c, "AMENDMENT", path)
    with pytest.raises(ValueError, match="does not describe this runner"):
        c.frozen()
