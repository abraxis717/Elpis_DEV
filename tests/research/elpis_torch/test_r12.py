"""Elpis Torch R12: the volatile fast state of the last block's specialist core, and the Stage C seam harness."""
from __future__ import annotations

import copy
import ctypes
import hashlib
import math
import pickle

import pytest

from .conftest import SKIP_REASON

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from torch.func import functional_call  # noqa: E402

from research.elpis_torch import fast_episodes as fe  # noqa: E402
from research.elpis_torch import specialist_stack as ss  # noqa: E402
from research.elpis_torch.fast_state import (FastStateConfig, FastStateError, ObservedSupport,  # noqa: E402
                                             SpecialistFastState, frozen_pass, last_channel)
from research.elpis_torch.model import parameter_digest  # noqa: E402
from research.elpis_torch.session import Status  # noqa: E402

NS = "R12_TEST"


def _stack(arm="G2", dtype=torch.float32):
    torch.manual_seed(0)
    return fe.freeze(ss.RegressionStack(arm, "BOUNDED_SPECIALIST", 7, namespace=NS).to(dtype))


def _support(model, index=0, k=32, dtype=torch.float32):
    probes, y = fe.probe_inputs(NS, "smooth", 7, "UNIT", index, k)
    probes, y = probes.to(dtype), y.to(dtype)
    _, pre = frozen_pass(model, probes)
    return ObservedSupport.observed(probes, y, source="unit teacher"), pre


def _state_bytes(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        data = tensor.detach().contiguous()
        digest.update(name.encode())
        digest.update(ctypes.string_at(data.data_ptr(), data.numel() * data.element_size()))
    return digest.hexdigest()


@pytest.mark.parametrize("arm", ["G1", "G2", "G3"])
def test_the_fast_state_changes_no_parameter_and_reset_restores_the_base_exactly(arm):
    model = _stack(arm)
    before, state = parameter_digest(model), _state_bytes(model)
    support, pre = _support(model)
    tokens, _ = ss.episode_tokens(NS, "smooth", 7, "UNIT", 0, 32)
    frozen, query_pre = frozen_pass(model, tokens[None])
    fast = SpecialistFastState(model, session="unit")
    results = fast.adapt(fast.support_loss(support, pre), steps=8)
    assert [r.status for r in results] == [Status.APPLIED] * 8 and fast.delta_norm() > 0
    with torch.no_grad():
        assert not torch.equal(fast.output(query_pre), frozen)          # the core does change the output
        fast.reset()
        assert torch.equal(fast.output(query_pre), frozen)              # and reset is exact
    fast.close()
    assert parameter_digest(model) == before and _state_bytes(model) == state


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_cached_last_block_path_equals_full_recomputation_in_output_and_gradient(dtype):
    model = _stack("G2", dtype)
    support, pre = _support(model, dtype=dtype)
    fast = SpecialistFastState(model, session="parity")
    fast.adapt(fast.support_loss(support, pre), steps=3)
    delta = {n: d.clone().requires_grad_(True) for n, d in fast.delta().items()}
    cached = (fast.output(pre, delta) - support.targets).pow(2).mean()
    full = (functional_call(model, fast.overrides(delta), (support.inputs, model.new_state(1))) - support.targets).pow(2).mean()
    (g_cached,), (g_full,) = torch.autograd.grad(cached, list(delta.values())), torch.autograd.grad(full, list(delta.values()))
    assert float(cached.detach()) == float(full.detach())
    assert torch.equal(g_cached, g_full)


def test_the_update_is_the_r11_clipped_sgd_step_and_the_pdf_gradient():
    """Eight bounded transactions equal eight plain clipped SGD steps (lr 0.08, global clip 2.0; R11 Stage B), and
    the autograd gradient equals the analytic local update sum_s delta phi'(r~) kappa(r) q (float64)."""
    from research.elpis_torch.experiments.run_r12 import _analytic_core_gradient
    model = _stack("G2", torch.float64)
    support, pre = _support(model, dtype=torch.float64)
    fast = SpecialistFastState(model, session="sgd")
    fast.adapt(fast.support_loss(support, pre), steps=8)
    name = "cgc.core.weight"
    reference = SpecialistFastState(model, session="reference")
    w = torch.zeros_like(fast.delta()[name])
    for _ in range(8):
        d = w.clone().requires_grad_(True)
        loss = (reference.output(pre, {name: d}) - support.targets).pow(2).mean()
        (g,) = torch.autograd.grad(loss, [d])
        analytic = _analytic_core_gradient(reference, pre, support.targets, {name: w})
        assert float((analytic - g).abs().max()) < 1e-12
        w = w - 0.08 * min(1.0, 2.0 / float(g.norm())) * g
    assert float((w - fast.delta()[name]).abs().max()) < 1e-14


def test_the_bound_cannot_bind_within_the_protocol():
    model = _stack("G2")
    probes, y = fe.probe_inputs(NS, "smooth", 7, "UNIT", 0, 32)
    huge = ObservedSupport.observed(probes, y * 1e4, source="unit: large targets clip every step")
    _, pre = frozen_pass(model, probes)
    fast = SpecialistFastState(model, session="bound")
    results = fast.adapt(fast.support_loss(huge, pre), steps=8)
    assert all(r.clipped and not r.projected for r in results)
    assert fast.delta_norm() <= FastStateConfig().bound() + 1e-6 and FastStateConfig().bound() == pytest.approx(1.28)


def test_disabled_learning_and_the_step_cap_refuse_atomically():
    model = _stack("G2")
    support, pre = _support(model)
    off = SpecialistFastState(model, session="off", config=FastStateConfig(learning_enabled=False))
    assert [r.status for r in off.adapt(off.support_loss(support, pre), steps=8)] == [Status.REFUSED_LEARNING_DISABLED]
    assert off.delta_norm() == 0.0 and off.steps_used == 0
    capped = SpecialistFastState(model, session="cap")
    results = capped.adapt(capped.support_loss(support, pre), steps=9)
    assert [r.status for r in results] == [Status.APPLIED] * 8 + [Status.REFUSED_STEP_BUDGET] and capped.steps_used == 8
    frozen_delta = capped.delta()
    nan = SpecialistFastState(model, session="nan")
    assert nan.adapt(lambda delta: torch.tensor(math.nan), steps=1)[0].status is Status.REFUSED_NONFINITE_LOSS
    assert all(torch.equal(a, b) for a, b in zip(capped.delta().values(), frozen_delta.values()))


def test_sessions_over_one_model_share_nothing_mutable():
    model = _stack("G2")
    support, pre = _support(model)
    a, b = SpecialistFastState(model, session="a"), SpecialistFastState(model, session="b")
    a.adapt(a.support_loss(support, pre), steps=4)
    assert b.delta_norm() == 0.0 and a.delta_norm() > 0
    with torch.no_grad():
        assert torch.equal(b.output(pre), frozen_pass(model, support.inputs)[0])
    assert a.identity()["session"] != b.identity()["session"] and a.identity()["model_digest"] == b.identity()["model_digest"]


def test_a_fast_state_is_never_serialized_or_copied_and_a_closed_one_refuses():
    model = _stack("G2")
    fast = SpecialistFastState(model, session="volatile")
    for attempt in (lambda: pickle.dumps(fast), lambda: copy.deepcopy(fast)):
        with pytest.raises(FastStateError):
            attempt()
    fast.close()
    with pytest.raises(FastStateError):
        fast.reset()


def test_only_the_last_block_of_a_fixed_single_expert_frozen_model_can_be_adapted():
    model = _stack("G2")
    with pytest.raises(FastStateError):
        SpecialistFastState(model, session="x", block=0)                   # earlier block: refused (replay)
    with pytest.raises(FastStateError):
        SpecialistFastState(_stack("G0"), session="x")                       # FFN only: nothing to adapt
    two = ss.RegressionStack("G2", "BOUNDED_SPECIALIST", 7, experts=2, namespace=NS)
    with pytest.raises(FastStateError):
        SpecialistFastState(fe.freeze(two), session="x")                    # learned routing: out of scope
    training = ss.RegressionStack("G2", "BOUNDED_SPECIALIST", 7, namespace=NS)
    with pytest.raises(FastStateError):
        SpecialistFastState(training, session="x")                          # not a frozen checkpoint
    with pytest.raises(FastStateError):
        SpecialistFastState(model, session=" ")                             # no session identity


def test_a_last_block_core_changes_no_recurrent_or_window_state_of_an_ersa_lm():
    from research.elpis_torch.experiments.run_r12 import command_parity
    import argparse
    checks = command_parity(argparse.Namespace())["checks"]
    assert checks["ersa_lm_window"] == {"states_unchanged_by_the_core": True, "logits_changed": True}
    assert checks["earlier_block_refused"] is True
    for name, check in checks.items():
        if isinstance(check, dict) and "output_full_vs_cached_max_abs" in check:
            assert check["output_full_vs_cached_max_abs"] == 0.0 and check["gradient_full_vs_cached_max_abs"] == 0.0, name
            assert check["states_unchanged_by_the_core"] and not check["projected"], name


@pytest.mark.parametrize("arm", ["G1", "G2", "G3"])
def test_the_batched_training_update_is_the_transactional_update(arm):
    model = _stack(arm, torch.float64)
    probes, y = fe.batch_probes(NS, ["smooth", "conditional"], 7, "UNIT", 0, 3, 16)
    probes, y = probes.double(), y.double()
    _, pre = frozen_pass(model, probes)
    batched = fe.batched_fast_update(model, pre, y, FastStateConfig())
    for b in range(3):
        fast = SpecialistFastState(model, session=f"b{b}")
        support = ObservedSupport.observed(probes[b:b + 1], y[b:b + 1], source="unit")
        fast.adapt(fast.support_loss(support, pre[b:b + 1]), steps=8)
        for n, d in fast.delta().items():
            assert float((fast._base[n] + d - batched[n][b]).abs().max()) < 1e-12


@pytest.mark.parametrize("dtype, tolerance", [(torch.float32, 1e-5), (torch.float64, 1e-12)])
def test_the_training_override_with_base_cores_is_the_routed_channel(dtype, tolerance):
    """Under base cores, the vectorized override computes the routed channel's own function (one expert, weight 1).

    Equal up to rounding, not bit for bit. The override multiplies the same matrices batched per episode (``vmap``),
    while the grouped dispatch multiplies them flattened, so a kernel path may sum them in a different order. The
    AVX2 path does: 2.4e-7 in float32 and 4.4e-16 in float64, at outputs near 2.8. The default path matches bit for
    bit. A changed core must change the output, so the hook is live."""
    model = _stack("G2", dtype)
    tokens, _ = ss.batch_tokens(NS, ["smooth"], 7, "UNIT", 0, 3, 8)
    tokens = tokens.to(dtype)
    live = dict(last_channel(model)[2].named_parameters())
    cores = {"cgc.core.weight": live["cgc.core.weight"].unsqueeze(0).expand(3, 5, 3, 8)}
    with torch.no_grad():
        with fe.override_last_core(model, cores):
            hooked = model(tokens, model.new_state(3))
        plain = model(tokens, model.new_state(3))
        with fe.override_last_core(model, {"cgc.core.weight": cores["cgc.core.weight"] + 0.1}):
            changed = model(tokens, model.new_state(3))
    assert float((hooked - plain).abs().max()) <= tolerance
    assert float((changed - plain).abs().max()) > 1e3 * tolerance


def test_probes_carry_no_target_and_a_probe_after_the_support_context_would_read_its_target():
    """The leakage contract: a probe token holds no target, and a context-free probe pass cannot see any target. The
    rejected alternative (probes after a context holding the pairs) moves with the probe's own target."""
    model = _stack("G2")
    probes, y = fe.probe_inputs(NS, "smooth", 7, "UNIT", 0, 8)
    assert not bool(probes[..., fe.TARGET_SLOT].any())
    context, _ = ss.episode_tokens(NS, "smooth", 7, "UNIT", 0, 8)
    context = context[None, :8]                                             # the 8 support tokens [x, y, 0]
    altered = context.clone()
    altered[..., fe.TARGET_SLOT] += 1.0                                      # the same inputs, other targets
    with torch.no_grad():
        after = model(torch.cat([context, probes], 1), model.new_state(1))[:, 8:]
        after_altered = model(torch.cat([altered, probes], 1), model.new_state(1))[:, 8:]
    assert not torch.equal(after, after_altered)                             # the rejected design leaks
    e = fe.st.teacher(NS, "smooth", 7, "UNIT", 0)
    assert not bool((e.support[0][:, None, :] == e.query[0][None, :, :]).all(-1).any())   # queries are new inputs


def test_evaluation_records_every_control_and_its_integrity_holds():
    model = _stack("G2")
    rows = fe.evaluate_fast(model, NS, "conditional", 7, "UNIT", 2)
    for row in rows:
        assert set(row["fast"]) == {"0", "1", "2", "4", "8"} and row["fast"]["0"] == row["frozen"]["query"]
        assert row["reset"]["exact"] and row["disabled"]["exact"]
        assert row["disabled"]["status"] == ["REFUSED_LEARNING_DISABLED"]
        assert row["leakage"] == {"query_support_disjoint": True, "probe_tokens_carry_no_target": True}
        assert row["parity"]["states_unchanged_by_the_core"] and row["parity"]["full_vs_cached_max_abs"] == 0.0
        assert all(t["status"] == "APPLIED" and not t["projected"] for t in row["trace"])
        assert {"shuffled", "previous_episode_core", "fast_ood", "fast_context_free", "alignment_r2"} <= set(row)
    assert set(fe.evaluate_fast(_stack("G0"), NS, "conditional", 7, "UNIT", 1)[0]) == {
        "episode", "frozen", "null", "leakage"}


def test_the_episodic_trainer_without_a_specialist_is_r11s_trainer():
    plan = {"adam_lr": 3e-3, "batch": 4, "dev_episodes": 2, "select_every": 3, "steps": 6}
    models = []
    for trainer in (ss.train_stack, fe.train_fast):
        torch.manual_seed(0)
        model = ss.RegressionStack("G0", "BOUNDED_SPECIALIST", 7, namespace=NS)
        info = trainer(model, NS, ["smooth"], 7, plan)
        models.append((model, info["selected_step"]))
    (a, sa), (b, sb) = models
    assert sa == sb and all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_the_episodic_trainer_adapts_only_through_the_last_core_and_reaches_every_parameter():
    plan = {"adam_lr": 3e-3, "batch": 4, "dev_episodes": 2, "select_every": 2, "steps": 2}
    torch.manual_seed(0)
    model = ss.RegressionStack("G2", "BOUNDED_SPECIALIST", 7, namespace=NS)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    fe.train_fast(model, NS, ["smooth"], 7, plan)
    moved = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert "blocks.1.sparse.experts.0.cgc.core.weight" in moved and "readout.weight" in moved
