"""GEOMETRIC CONTRACT — signed dual-view transport and covariance measurement.

The expectations here are analytic: they are computed from the fixture's own
matrices with plain float64 arithmetic, independently of whichever transport
adapter is under test.  A transport adapter therefore cannot qualify by
reporting its own residuals as correct.

Scope limits (restated in the report): this measures transport/channel
covariance only.  It does not certify a learned HCA, native semantic
admission, or any language-model capability.
"""
from __future__ import annotations
import math

from . import contract_r0 as C

SIGNED_VIEW_T = 8.0          # frozen default from the master contract
COVARIANCE_TOLERANCE = 1e-9  # previously qualified transport-seam threshold
NEGATIVE_CONTROL_FLOOR = 1e-4


def _norm(v):
    return math.sqrt(math.fsum(x * x for x in v))


def aggregate_signed(transported, temperature=SIGNED_VIEW_T):
    """Contract-level signed dual view: salient T-softmax mean + uniform mean."""
    norms = [_norm(v) for v in transported]
    ex = [temperature * n for n in norms]
    total = math.fsum(math.exp(t) for t in ex)
    w = [math.exp(t) / total for t in ex]
    salient = tuple(math.fsum(w[i] * transported[i][j]
                              for i in range(len(transported))) for j in range(4))
    collective = tuple(math.fsum(v[j] for v in transported) / len(transported)
                       for j in range(4))
    return salient, collective


def aggregate_squared(transported):
    """Deliberately WRONG control: componentwise squared mean (not a vector)."""
    return tuple(math.fsum(v[j] * v[j] for v in transported) / len(transported)
                 for j in range(4))


def _mat_apply(m, v):
    return tuple(math.fsum(m[i][j] * v[j] for j in range(4)) for i in range(4))


def _max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def _expected_canonical(fixture, rid, frame):
    """Analytic expectation: U_{relation<-participant} applied to x."""
    return _mat_apply(fixture["base"][(rid, frame)], fixture["inputs"][frame])


def measure_transport(adapter_base, adapter_changed, fixture):
    """Residuals for one transport adapter under independent local gauges.

    adapter_base    : bound to fixture["base"]
    adapter_changed : bound to fixture["changed"]  (U' = G_r U G_p^-1)
    """
    cov = roundtrip = norm_err = order_err = 0.0
    agg_cov = 0.0
    for rid in fixture["relations"]:
        transported, transported_gauge = [], []
        for frame in fixture["participants"]:
            x = fixture["inputs"][frame]
            y = tuple(adapter_base.to_canonical(rid, frame, x))
            expected = _expected_canonical(fixture, rid, frame)
            cov = max(cov, _max_abs_diff(y, expected))
            xp = _mat_apply(fixture["gauge_participant"][frame], x)
            yp = tuple(adapter_changed.to_canonical(rid, frame, xp))
            gr = fixture["gauge_relation"][rid]
            yp_expected = _mat_apply(gr, expected)
            cov = max(cov, _max_abs_diff(yp, yp_expected))
            back = tuple(adapter_base.to_view(rid, frame, y))
            roundtrip = max(roundtrip, _max_abs_diff(back, x))
            norm_err = max(norm_err, abs(_norm(y) - _norm(x)))
            transported.append(y)
            transported_gauge.append(yp)
        a, b = aggregate_signed(transported)
        ap, bp = aggregate_signed(transported_gauge)
        gr = fixture["gauge_relation"][rid]
        agg_cov = max(agg_cov, _max_abs_diff(ap, _mat_apply(gr, a)),
                      _max_abs_diff(bp, _mat_apply(gr, b)))
        rev = list(reversed(transported))
        ra, rb = aggregate_signed(rev)
        order_err = max(order_err, _max_abs_diff(ra, a), _max_abs_diff(rb, b))
    return {"transport_covariance": cov, "roundtrip": roundtrip,
            "orthogonal_norm": norm_err, "incidence_order": order_err,
            "signed_view_aggregate": agg_cov}


def negative_controls(adapter_base, adapter_changed, fixture):
    """Deliberately incorrect controls that MUST be detected as non-covariant."""
    raw = squared = 0.0
    for rid in fixture["relations"]:
        local = [fixture["inputs"][f] for f in fixture["participants"]]
        local_gauge = [_mat_apply(fixture["gauge_participant"][f], fixture["inputs"][f])
                       for f in fixture["participants"]]
        gr = fixture["gauge_relation"][rid]
        naive = tuple(math.fsum(v[j] for v in local) / len(local) for j in range(4))
        naive_p = tuple(math.fsum(v[j] for v in local_gauge) / len(local_gauge)
                        for j in range(4))
        raw = max(raw, _max_abs_diff(naive_p, _mat_apply(gr, naive)))
        sq = aggregate_squared([_mat_apply(fixture["base"][(rid, f)], fixture["inputs"][f])
                                for f in fixture["participants"]])
        sqp = aggregate_squared([_mat_apply(fixture["changed"][(rid, f)],
                                            _mat_apply(fixture["gauge_participant"][f],
                                                       fixture["inputs"][f]))
                                 for f in fixture["participants"]])
        squared = max(squared, _max_abs_diff(sqp, _mat_apply(gr, sq)))
    return {"raw_local_frame": raw, "squared_view": squared}


def disconnected_isolation(adapter, fixture, zero_input=(0.0, 0.0, 0.0, 0.0)):
    """Zero-input relation must aggregate to exact zero while others stay live.

    The non-vacuity check (other relations nonzero) prevents a trivially
    zeroing adapter from passing.
    """
    target = fixture["relations"][0]
    transported = [tuple(adapter.to_canonical(target, f, zero_input))
                   for f in fixture["participants"]]
    a, b = aggregate_signed(transported)
    isolation = max(max(abs(x) for x in a), max(abs(x) for x in b))
    others = []
    for rid in fixture["relations"][1:]:
        ta = [tuple(adapter.to_canonical(rid, f, fixture["inputs"][f]))
              for f in fixture["participants"]]
        oa, ob = aggregate_signed(ta)
        others.append(max(max(abs(x) for x in oa), max(abs(x) for x in ob)))
    return isolation, (min(others) if others else 0.0)


# --------------------------------------------------- learned recurrence audit
def equivariant_radial(v):
    """Odd radial nonlinearity: rho(Gv) = G rho(v) for orthogonal G."""
    n = _norm(v)
    s = math.tanh(n) / max(n, 1e-14)
    return tuple(s * x for x in v)


def equivariant_reference_step(state, a, b):
    return equivariant_radial(tuple(0.55 * state[j] + 0.25 * a[j] + 0.20 * b[j]
                                    for j in range(4)))


def non_equivariant_mock_step(state, a, b):
    """Stand-in for the historical GRU_302 shape: coordinatewise tanh."""
    return tuple(math.tanh(0.55 * state[j] + 0.25 * a[j] + 0.20 * b[j])
                 for j in range(4))


def recurrence_covariance(step_fn, fixture, adapter_base, adapter_changed,
                          steps=4):
    """Multi-step covariance of a recurrent step function over signed views.

    Distinguishes LEARNED RECURRENCE covariance from TRANSPORT covariance:
    a transport adapter may pass measure_transport while this still fails.
    """
    worst = 0.0
    for rid in fixture["relations"]:
        gr = fixture["gauge_relation"][rid]
        base_views = aggregate_signed(
            [_mat_apply(fixture["base"][(rid, f)], fixture["inputs"][f])
             for f in fixture["participants"]])
        gauge_views = aggregate_signed(
            [_mat_apply(fixture["changed"][(rid, f)],
                        _mat_apply(fixture["gauge_participant"][f], fixture["inputs"][f]))
             for f in fixture["participants"]])
        s = sp = (0.0, 0.0, 0.0, 0.0)
        for _ in range(steps):
            s = step_fn(s, base_views[0], base_views[1])
            sp = step_fn(sp, gauge_views[0], gauge_views[1])
            worst = max(worst, _max_abs_diff(sp, _mat_apply(gr, s)))
    return worst
