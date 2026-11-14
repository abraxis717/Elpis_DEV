"""The batched CGC operators against the E0 reference: what is faithful, what is an extension, what is bounded."""
from __future__ import annotations

import math

import pytest

from .conftest import SKIP_REASON

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch import e0  # noqa: E402
from research.elpis_torch.cgc import CGCCore, CGCTransform, phi, phi_prime, soft_bound  # noqa: E402

F64 = torch.float64


def _core(d, c, n, *, bound=None, scale=0.3, seed=0, dtype=F64):
    core = CGCCore(d, c, n, preact_bound=bound, dtype=dtype)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        core.weight.copy_((scale * torch.randn(d, c, n, generator=g, dtype=F64)).to(dtype))
    return core


def _inputs(*shape, seed=1):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=F64)


@pytest.mark.parametrize("d,c,n", [(6, 3, 36), (5, 4, 6), (5, 3, 8), (16, 16, 4), (1, 1, 1)])
def test_the_unbounded_core_is_e0_channel_by_channel(d, c, n):
    core, x = _core(d, c, n), _inputs(11, d)
    got = core(x)
    assert got.shape == (11, c)
    for channel in range(c):
        reference = e0.forward(core.weight[:, channel, :].detach(), x)
        assert torch.allclose(got[:, channel], reference, rtol=1e-12, atol=1e-12)


def test_the_core_maps_hidden_states_of_any_leading_shape():
    core, x = _core(8, 5, 3), _inputs(2, 7, 8)
    got = core(x)
    assert got.shape == (2, 7, 5)
    flat = core(x.reshape(14, 8)).reshape(2, 7, 5)
    assert torch.allclose(got, flat, rtol=1e-13, atol=1e-13)
    assert core.preactivations(x).shape == (2, 7, 5, 3)
    with pytest.raises(ValueError):
        core(_inputs(2, 7, 9))


def test_the_horner_form_is_the_reference_primitive():
    u = torch.linspace(-5, 5, 2001, dtype=F64)
    assert torch.allclose(phi(u), e0.phi(u), rtol=1e-13, atol=1e-13)
    assert torch.allclose(phi_prime(u), e0.phi_prime(u), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("bottleneck", [None, 4])
def test_each_cell_of_the_transform_is_e0_on_the_full_hidden_input(bottleneck):
    """Dense: W_c is the core weight. Bottleneck: W_c = P_in^T core_c, a rank-constrained E0 weight."""
    hidden, channels, units = 12, 7, 5
    transform = CGCTransform(hidden, channels, units, bottleneck=bottleneck, dtype=F64)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in transform.parameters():
            p.copy_(0.3 * torch.randn(p.shape, generator=g, dtype=F64))
    x = _inputs(9, hidden)
    cells = transform.cells(x)
    for channel in range(channels):
        w = transform.effective_e0_weight(channel).detach()
        assert w.shape == (hidden, units)
        if bottleneck is not None:
            assert int(torch.linalg.matrix_rank(w)) <= bottleneck
        assert torch.allclose(cells[:, channel], e0.forward(w, x), rtol=1e-11, atol=1e-11)
    # The readout is the only thing between the cells and the output.
    assert torch.allclose(transform(x), cells @ transform.proj_out.weight.T, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("hidden,channels,units,bottleneck", [(64, 86, 5, None), (64, 121, 6, 32), (32, 16, 4, 8)])
def test_parameter_and_compute_formulas_match_the_module(hidden, channels, units, bottleneck):
    transform = CGCTransform(hidden, channels, units, bottleneck=bottleneck)
    actual = sum(p.numel() for p in transform.parameters())
    assert actual == CGCTransform.parameter_count(hidden, channels, units, bottleneck)
    assert CGCTransform.multiply_accumulates(hidden, channels, units, bottleneck) == actual + 3 * channels * units


def test_the_soft_bound_is_a_bounded_deviation_from_e0_and_vanishes_in_the_limit():
    u = torch.linspace(-2, 2, 801, dtype=F64)
    for b in (1.0, 3.0, 10.0):
        bounded = soft_bound(u, b)
        assert float(bounded.abs().max()) < b
        assert bool(((bounded - u).abs() <= u.abs() ** 3 / (3 * b * b) + 1e-15).all())
    assert soft_bound(u, None) is u
    x = _inputs(6, 8)
    faithful = _core(8, 4, 5)(x)
    errors = []
    for b in (3.0, 30.0, 300.0, 1e6):
        bounded = _core(8, 4, 5, bound=b)(x)
        errors.append(float((bounded - faithful).detach().abs().max()))
    assert errors == sorted(errors, reverse=True) and errors[-1] < 1e-9


def test_the_bound_caps_every_cell_whatever_the_weights_and_inputs_are():
    core = _core(8, 4, 5, bound=3.0, scale=1e6)
    x = 1e6 * _inputs(3, 8)
    y = core(x)
    assert bool(torch.isfinite(y).all())
    assert float(y.detach().abs().max()) <= core.output_ceiling() == 5 * 0.5 * (3 + 9 + 27)
    assert core.diagnostics(x)["saturated_fraction"] > 0.99
    assert _core(8, 4, 5).output_ceiling() == math.inf


def test_the_unbounded_core_grows_cubically_which_is_why_the_bound_exists():
    core, x = _core(8, 4, 5, scale=1.0), _inputs(64, 8)
    small, large = core(x).detach().abs().mean(), core(10 * x).detach().abs().mean()
    assert 300 < float(large / small) < 1000            # between quadratic (100x) and cubic (1000x) growth
    huge = _core(8, 4, 5, scale=1.0, dtype=torch.float32)(1e13 * x.float())
    assert not bool(torch.isfinite(huge).all())          # float32 overflows: the documented hazard
    bounded = _core(8, 4, 5, bound=3.0, scale=1.0, dtype=torch.float32)(1e13 * x.float())
    assert bool(torch.isfinite(bounded).all())


@pytest.mark.parametrize("bound", [None, 3.0])
def test_gradients_are_exact_finite_and_nonzero(bound):
    core, x = _core(4, 3, 2, bound=bound), _inputs(5, 4).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda w, v: phi(soft_bound(
        v @ w.reshape(4, 6), bound)).unflatten(-1, (3, 2)).sum(-1), (core.weight, x), eps=1e-6, atol=1e-6)
    (grad,) = torch.autograd.grad(core(x).sum(), core.weight)
    assert bool(torch.isfinite(grad).all()) and float(grad.abs().min()) > 0
    if bound is None:   # d y_c / d w_{c,i} = sum_r x_r phi'(x_r . w_{c,i})
        u = x.detach() @ core.weight.detach().reshape(4, 6)
        expected = (x.detach().T @ e0.phi_prime(u)).reshape(4, 3, 2)
        assert torch.allclose(grad, expected, rtol=1e-11, atol=1e-11)


def test_float32_execution_agrees_with_float64():
    reference, x = _core(16, 8, 4), _inputs(4, 9, 16)
    single = _core(16, 8, 4, dtype=torch.float32)
    assert torch.allclose(single(x.float()).double(), reference(x), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("kwargs", [dict(in_dim=0, channels=1, units=1), dict(in_dim=1, channels=1.5, units=1),
                                    dict(in_dim=1, channels=1, units=1, preact_bound=0.0),
                                    dict(in_dim=1, channels=1, units=1, preact_bound=float("inf"))])
def test_invalid_operator_shapes_are_refused(kwargs):
    with pytest.raises(ValueError):
        CGCCore(**kwargs)
