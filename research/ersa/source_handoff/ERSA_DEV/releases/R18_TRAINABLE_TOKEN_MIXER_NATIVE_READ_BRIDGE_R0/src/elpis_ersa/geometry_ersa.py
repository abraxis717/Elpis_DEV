"""ERSA GEOMETRY R0 — signed SO(2)xSO(2) transport, aggregation, covariance.

Gauge group: rho(a1,a2) = diag(R(a1), R(a2)) acting on R^4 as two 2D blocks.
This is a REFERENCE transport: REFERENCE_NOT_YM_AUTHORITY.  The Yang-Mills
interface in Elpis_DEV is a reserved boundary (WAITING_FOR_QUALIFIED_SOURCE);
no identity about the eventual YM operator is claimed here.

Expectations in this module are analytic (computed from the fixture's own
matrices with plain float64 arithmetic), so an operator under test cannot
qualify by reporting its own residuals as correct.
"""
from __future__ import annotations
import math

TWO_BLOCK = (0, 1, 2, 3)


def rot(a: float) -> tuple:
    c, s = math.cos(a), math.sin(a)
    return ((c, -s), (s, c))


def rho_apply(angles, v) -> tuple:
    """Apply diag(R(a1),R(a2)) to a 4-vector without building the matrix."""
    (a1, a2) = angles
    x1, x2, x3, x4 = v
    c1, s1 = math.cos(a1), math.sin(a1)
    c2, s2 = math.cos(a2), math.sin(a2)
    return (c1 * x1 - s1 * x2, s1 * x1 + c1 * x2,
            c2 * x3 - s2 * x4, s2 * x3 + c2 * x4)


def mat_apply(m, v) -> tuple:
    return tuple(math.fsum(m[i][j] * v[j] for j in range(4)) for i in range(4))


def rho_matrix(angles) -> tuple:
    a1, a2 = angles
    R1, R2 = rot(a1), rot(a2)
    return ((R1[0][0], R1[0][1], 0.0, 0.0),
            (R1[1][0], R1[1][1], 0.0, 0.0),
            (0.0, 0.0, R2[0][0], R2[0][1]),
            (0.0, 0.0, R2[1][0], R2[1][1]))


def rho_inverse(angles) -> tuple:
    return (-angles[0], -angles[1])


def rho_compose(a, b) -> tuple:
    return (a[0] + b[0], a[1] + b[1])


def norm4(v) -> float:
    return math.sqrt(math.fsum(x * x for x in v))


def cos4(a, b) -> float:
    """SIGNED cosine: sign of the alignment is preserved, magnitude is not."""
    na, nb = norm4(a), norm4(b)
    if na < 1e-14 or nb < 1e-14:
        return 0.0
    return math.fsum(a[i] * b[i] for i in range(4)) / (na * nb)


def max_abs_diff(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


# ------------------------------------------------------------------- transport
class ReferenceTransport:
    """U_{relation<-participant} = rho(alpha_r - beta_e) on both 2D blocks.

    Gauge law (matches the predecessor's `changed = G_r U G_p^-1`):
      participant frames gain gamma_p, relation canonical frames gain GAMMA,
      so U' = rho(alpha_r + GAMMA - beta_e - gamma_p) = G_r U G_p^-1.
    """
    def __init__(self, relation_angles=None, participant_angles=None,
                  gauge_relation=(0.0, 0.0), gauge_participant=None):
        self.relation_angles = dict(relation_angles or {})
        self.participant_angles = dict(participant_angles or {})
        self.gauge_relation = tuple(gauge_relation)
        self.gauge_participant = dict(gauge_participant or {})

    def bind_frame(self, relation_id, frame_id, operator=None):
        if operator is not None:
            raise ValueError("OPERATOR_OUTSIDE_RHO_SO2xSO2")
        self.relation_angles[relation_id] = tuple(
            self.relation_angles.get(relation_id, (0.0, 0.0)))
        self.participant_angles[frame_id] = tuple(
            self.participant_angles.get(frame_id, (0.0, 0.0)))

    def set_relation_frame(self, relation_id, angles):
        self.relation_angles[relation_id] = (float(angles[0]), float(angles[1]))

    def set_participant_frame(self, frame_id, angles):
        self.participant_angles[frame_id] = (float(angles[0]), float(angles[1]))

    def list_frames(self):
        return tuple(sorted(self.participant_angles))

    def to_canonical(self, relation_id, frame_id, vec):
        if frame_id not in self.participant_angles:
            raise PermissionError("UNBOUND_FRAME")
        a = self.relation_angles.get(relation_id, (0.0, 0.0))
        b = self.participant_angles[frame_id]
        g = self.gauge_participant.get(frame_id, (0.0, 0.0))
        gam = self.gauge_relation
        u = (a[0] + gam[0] - b[0] - g[0], a[1] + gam[1] - b[1] - g[1])
        return rho_apply(u, vec)

    def to_view(self, relation_id, frame_id, vec):
        if frame_id not in self.participant_angles:
            raise PermissionError("UNBOUND_FRAME")
        a = self.relation_angles.get(relation_id, (0.0, 0.0))
        b = self.participant_angles[frame_id]
        g = self.gauge_participant.get(frame_id, (0.0, 0.0))
        gam = self.gauge_relation
        u = (a[0] + gam[0] - b[0] - g[0], a[1] + gam[1] - b[1] - g[1])
        return rho_apply(rho_inverse(u), vec)


# ------------------------------------------------------------------ aggregation
def aggregate_signed(transported, weights):
    """SIGNED relevance aggregation.  L1 normalisation, never squared.

    Returns (salient, collective).  `salient` keeps weak sources proportional to
    their signed relevance and lets anti-aligned contributions subtract.
    If the signed mass is degenerate the documented fallback is the uniform
    collective mean (never a silent zeroing).
    """
    n = len(transported)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)
    if weights is None:
        weights = [1.0] * n
    z = math.fsum(abs(w) for w in weights)
    if z < 1e-12:
        salient = tuple(math.fsum(v[j] for v in transported) / n
                        for j in range(4))
    else:
        salient = tuple(math.fsum((weights[i] / z) * transported[i][j]
                                  for i in range(n)) for j in range(4))
    collective = tuple(math.fsum(v[j] for v in transported) / n
                       for j in range(4))
    return salient, collective


def aggregate_magnitude_only(transported, weights):
    """DELIBERATELY WRONG control: relevance signs discarded."""
    return aggregate_signed(transported, [abs(w) for w in weights])[0]


def aggregate_squared(transported, weights=None):
    """DELIBERATELY WRONG control: componentwise squared mean (not a vector)."""
    n = len(transported)
    if not n:
        return (0.0, 0.0, 0.0, 0.0)
    w = [1.0] * n if weights is None else weights
    z = math.fsum(abs(x) for x in w) or 1.0
    return tuple(math.fsum(w[i] * transported[i][j] ** 2 for i in range(n)) / z
                 for j in range(4))


def directional_fidelity(observed, expected) -> float:
    """1 - cos(aggregate, independent expectation).  Magnitude-only fails this."""
    return 1.0 - cos4(observed, expected)


# ------------------------------------------------------- full-operator gauges
def measure_full_operator_equivariance(operator, fixture):
    """Covariance of the WHOLE attention operator, not the transport routine.

    `operator(transport, relations, query) -> aggregate4` is called twice:
    once in the base gauge and once with every participant frame rotated by
    gamma_p and the relation canonical frames rotated by GAMMA.  A transport
    adapter can pass measure_transport while this still fails, which is exactly
    the trap: scoring computed in a non-invariant frame is not equivariant.
    """
    worst = 0.0
    for q in fixture["queries"]:
        a = operator(fixture["base_transport"], q)
        ap = operator(fixture["gauge_transport"], q)
        g = fixture["gauge_relation"]
        worst = max(worst, max_abs_diff(ap, rho_apply(g, a)))
    return worst


def equivariant_radial(v):
    """Odd radial nonlinearity: rho(G v) = G rho(v) for orthogonal G."""
    n = norm4(v)
    s = math.tanh(n) / max(n, 1e-14)
    return tuple(s * x for x in v)


def non_equivariant_mock_step(v):
    """Stand-in for the historical coordinatewise-tanh shape (GRU_302 family)."""
    return tuple(math.tanh(x) for x in v)
