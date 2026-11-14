"""The E0 reference is the upstream mathematics: checked four independent ways before anything is built on it.

1. bitwise against the pinned upstream source (digest-verified, loaded from its file);
2. against exact rational arithmetic on the same float64 inputs;
3. against the S3 moment identity (E0 is the cubic polynomial fixed by mu, M, T3);
4. its gradient against autograd and central finite differences;
5. against the qualified native ECS_G kernel (libelpis_ecsg_math) in its qualified regime, when it is built.
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
from fractions import Fraction

import pytest

from .conftest import REPO, SKIP_REASON

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch import e0  # noqa: E402
from research.elpis_torch import upstream as pin  # noqa: E402

UPSTREAM_FILE = REPO / "research" / "elpis_torch" / "upstream" / "cgc_ecsg_e0.py"
# (d, N, R): the ECS_G qualified shape, the CGC120 wrapper cell, the R97W benchmark cell, and two others.
SHAPES = [(6, 36, 9), (5, 6, 12), (5, 8, 7), (1, 1, 3), (17, 3, 5)]


def _upstream():
    spec = importlib.util.spec_from_file_location("elpis_torch_pinned_cgc_e0", UPSTREAM_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case(d, n, r, seed=0, scale=0.3):
    g = torch.Generator().manual_seed(1000 * seed + 31 * d + n)
    w = scale * torch.randn(d, n, generator=g, dtype=torch.float64)
    x = torch.randn(r, d, generator=g, dtype=torch.float64)
    y = torch.randn(r, generator=g, dtype=torch.float64)
    return w, x, y


def test_the_vendored_upstream_file_is_the_pinned_bytes():
    assert hashlib.sha256(UPSTREAM_FILE.read_bytes()).hexdigest() == pin.UPSTREAM_SHA256
    assert len(pin.UPSTREAM_COMMIT) == 40 and pin.UPSTREAM_LICENSE == "Apache-2.0"


def test_provenance_document_states_the_same_pin():
    text = (REPO / "research" / "elpis_torch" / "PROVENANCE.md").read_text(encoding="utf-8")
    for needle in (pin.UPSTREAM_COMMIT, pin.UPSTREAM_SHA256, pin.UPSTREAM_PATH, pin.UPSTREAM_REPOSITORY,
                   pin.UPSTREAM_LICENSE):
        assert needle in text, needle


@pytest.mark.parametrize("d,n,r", SHAPES)
def test_forward_gradient_and_step_equal_the_upstream_bitwise(d, n, r):
    up = _upstream()
    w, x, y = _case(d, n, r)
    assert torch.equal(e0.forward(w, x), up.forward(w, x))
    assert torch.equal(e0.mse_loss(w, x, y), up.mse_loss(w, x, y))
    assert torch.equal(e0.analytic_grad(w, x, y), up.analytic_grad(w, x, y))
    assert torch.equal(e0.gd_step(w, x, y, 0.002), up.gd_step(w, x, y, 0.002))
    u = torch.linspace(-4, 4, 257, dtype=torch.float64)
    assert torch.equal(e0.phi(u), up.phi(u)) and torch.equal(e0.phi_prime(u), up.phip(u))


@pytest.mark.parametrize("d,n,r", SHAPES)
def test_forward_equals_exact_rational_arithmetic(d, n, r):
    """sum_i (u + u^2 + u^3)/2 with u = x.w_i, evaluated exactly on the float64 inputs (no rounding at all)."""
    w, x, _ = _case(d, n, r)
    got = e0.forward(w, x).tolist()
    wf = [[Fraction(v) for v in row] for row in w.tolist()]
    for row, value in zip(x.tolist(), got):
        xf = [Fraction(v) for v in row]
        exact = Fraction(0)
        for i in range(n):
            u = sum(xf[a] * wf[a][i] for a in range(d))
            exact += (u + u * u + u * u * u) / 2
        scale = max(1.0, abs(float(exact)))
        assert abs(Fraction(value) - exact) <= Fraction(1, 10 ** 12) * Fraction(scale)


@pytest.mark.parametrize("d,n,r", SHAPES)
def test_forward_equals_the_s3_moment_polynomial(d, n, r):
    up = _upstream()
    w, x, _ = _case(d, n, r)
    mu, m, t3 = e0.moments(w)
    assert torch.allclose(e0.forward(w, x), e0.forward_from_moments(mu, m, t3, x), rtol=1e-11, atol=1e-11)
    # The upstream packed S3 holds exactly these moments' upper-triangular entries.
    up_mu, up_m, up_t = up.project_s3(w)
    ii, jj = torch.triu_indices(d, d)
    assert torch.equal(up_mu, mu) and torch.allclose(up_m, m[ii, jj], rtol=0, atol=1e-14)
    assert torch.allclose(up.forward_s3(up_mu, up_m, up_t, x), e0.forward(w, x), rtol=1e-11, atol=1e-11)
    assert up_t.numel() == d * (d + 1) * (d + 2) // 6


@pytest.mark.parametrize("d,n,r", SHAPES)
def test_analytic_gradient_equals_autograd_and_finite_differences(d, n, r):
    w, x, y = _case(d, n, r)
    leaf = w.clone().requires_grad_(True)
    auto = torch.autograd.grad(e0.mse_loss(leaf, x, y), leaf)[0]
    analytic = e0.analytic_grad(w, x, y)
    assert torch.allclose(analytic, auto, rtol=1e-11, atol=1e-11)
    eps = 1e-6
    for a, i in {(0, 0), (d - 1, n - 1), (d // 2, n // 2)}:
        step = torch.zeros_like(w)
        step[a, i] = eps
        numeric = (e0.mse_loss(w + step, x, y) - e0.mse_loss(w - step, x, y)) / (2 * eps)
        assert abs(float(numeric - analytic[a, i])) <= 1e-6 * max(1.0, abs(float(analytic[a, i])))


@pytest.mark.parametrize("width", [36, 48, 72])
def test_the_reference_and_the_batched_core_agree_with_the_qualified_native_kernel(width):
    """native/ECS_G in its qualified regime (d = 6, float64): the C kernel, the reference and the tensor form.

    A parity check of the forward map on these inputs. It reads the native library and qualifies nothing.
    """
    from tests.conftest import require_native_library
    from research.elpis_torch.cgc import CGCCore

    library = ctypes.CDLL(str(require_native_library("elpis_ecsg_math")))
    native = library.elpis_ecsg_forward_f64
    native.restype = ctypes.c_int
    native.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_size_t,
                       ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.POINTER(ctypes.c_double)]
    dim, rows = 6, 64
    w, x, _ = _case(dim, width, rows, scale=0.2)
    # The kernel's W is row-major [dim, width]: exactly the contiguous layout of the reference's tensor.
    w_buffer = (ctypes.c_double * (dim * width))(*w.reshape(-1).tolist())
    x_buffer = (ctypes.c_double * (rows * dim))(*x.reshape(-1).tolist())
    out = (ctypes.c_double * rows)()
    assert native(w_buffer, dim, width, x_buffer, rows, out) == 0
    expected = torch.tensor(list(out), dtype=torch.float64)
    assert torch.allclose(e0.forward(w, x), expected, rtol=1e-13, atol=1e-13)
    core = CGCCore(dim, 1, width, dtype=torch.float64)
    with torch.no_grad():
        core.weight.copy_(w[:, None, :])
    assert torch.allclose(core(x)[:, 0], expected, rtol=1e-12, atol=1e-12)


def test_phi_prime_is_the_derivative_and_never_below_one_third():
    u = torch.linspace(-6, 6, 4801, dtype=torch.float64, requires_grad=True)
    (auto,) = torch.autograd.grad(e0.phi(u).sum(), u)
    assert torch.allclose(auto, e0.phi_prime(u.detach()), rtol=1e-12, atol=1e-12)
    assert float(e0.phi_prime(u.detach()).min()) >= 1 / 3 - 1e-12
    assert float(e0.phi_prime(torch.tensor(-1 / 3, dtype=torch.float64))) == pytest.approx(1 / 3, abs=1e-15)
    assert bool((e0.phi(u.detach())[1:] > e0.phi(u.detach())[:-1]).all())     # strictly increasing


def test_repeated_updates_stay_differentiable():
    """R97O's property: a gradient flows through K successive G1 steps back to the initial weights."""
    up = _upstream()
    w, x, y = _case(5, 6, 12, scale=0.03)
    for steps in (2, 4, 8):
        w0 = w.clone().requires_grad_(True)
        mine, theirs = w0, w0
        for _ in range(steps):
            mine, theirs = e0.gd_step(mine, x, y, 0.01), up.gd_step(theirs, x, y, 0.01)
        grad = torch.autograd.grad(e0.mse_loss(mine, x, y), w0, retain_graph=True)[0]
        assert torch.equal(grad, torch.autograd.grad(up.mse_loss(theirs, x, y), w0)[0])
        assert bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0


def test_inputs_outside_the_reference_contract_are_refused():
    w, x, y = _case(5, 6, 4)
    with pytest.raises(e0.E0Error):
        e0.forward(w.float(), x.float())
    with pytest.raises(e0.E0Error):
        e0.forward(w, x[:, :4])
    with pytest.raises(e0.E0Error):
        e0.forward(w.reshape(-1), x)
    bad = w.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(e0.E0Error):
        e0.forward(bad, x)
    with pytest.raises(e0.E0Error):
        e0.mse_loss(w, x, y.float())
